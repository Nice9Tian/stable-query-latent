"""Validation probe: can a frozen VICReg encoder's code predict a game's tags?

This is a diagnostic, NOT part of VICReg training. The encoder is loaded frozen;
gradients never flow into it. We:

  1. Load a VICReg checkpoint and freeze the encoder.
  2. For each game, sample a few review views, encode them, and average the
     (num_latents, output_dim) codes into one cached per-game feature.
  3. Train a TagRegressionHead (its own optimizer) on those frozen features to
     predict the multi-hot tag labels from tag_build.py.
  4. Stop on its own when learning plateaus (val-loss patience, and optionally a
     gradient-norm floor).

A rising tag mAP across VICReg checkpoints means the encoder is learning a more
robust game representation. Run tag_build.py first.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VICReg_review.model import LatentArrayMLP, TagRegressionHead  # noqa: E402
from VICReg_review.train_vicreg_review_h5 import load_game_views  # noqa: E402

DEFAULT_H5 = SCRIPT_DIR / "h5" / "game_review_cleaned_3_sentences.h5"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "heads" / "vicreg_review_h5_best.pt"
DEFAULT_TAGS_DIR = SCRIPT_DIR / "tags"
DEFAULT_OUT_DIR = SCRIPT_DIR / "heads"


def decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def atomic_text_write(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_frozen_encoder(checkpoint_path, input_dim, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = checkpoint.get("args", {})
    defaults = dict(latent_dim=256, num_latents=256, num_heads=8,
                    dropout=0.1, output_dim=18, reduce_hidden=(128, 64, 32))
    cfg = {key: saved.get(key, defaults[key]) for key in defaults}
    model = LatentArrayMLP(
        input_dim=input_dim,
        latent_dim=cfg["latent_dim"],
        num_latents=cfg["num_latents"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        output_dim=cfg["output_dim"],
        reduce_hidden=tuple(cfg["reduce_hidden"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, cfg, checkpoint.get("epoch"), checkpoint.get("global_step")


def sample_game_views(h5, game_index, sample_fraction, num_views, rng, cache_dtype):
    views = []
    while len(views) < num_views:
        view_a, view_b = load_game_views(
            h5, game_index, sample_fraction, rng, cache_dtype, pin_cache=False
        )
        views.append(view_a)
        if len(views) < num_views:
            views.append(view_b)
    return views[:num_views]


@torch.no_grad()
def extract_features(encoder, h5_path, sample_fraction, feature_views, seed, cache_dtype, device, amp):
    rng = np.random.default_rng(seed)
    cache_np = np.dtype(cache_dtype)
    with h5py.File(h5_path, "r") as h5:
        game_names = [decode_name(name) for name in h5["game_names"][:]]
        num_games = len(game_names)
        feats = None
        for game_index in range(num_games):
            views = sample_game_views(h5, game_index, sample_fraction, feature_views, rng, cache_np)
            stacked = []
            for view in views:
                tensor = view.unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                    code = encoder(tensor, key_padding_mask=None)
                stacked.append(code.squeeze(0).float())
            mean_code = torch.stack(stacked, dim=0).mean(dim=0)  # (num_latents, output_dim)
            if feats is None:
                feats = torch.empty((num_games, *mean_code.shape), dtype=torch.float32)
            feats[game_index] = mean_code.cpu()
            if (game_index + 1) % 50 == 0 or game_index + 1 == num_games:
                print(f"features {game_index + 1}/{num_games}", flush=True)
    return feats, game_names


def load_labels(tags_dir):
    vocab = json.loads((Path(tags_dir) / "tag_vocab.json").read_text(encoding="utf-8"))
    npz = np.load(Path(tags_dir) / "tag_labels.npz", allow_pickle=False)
    return vocab, [str(name) for name in npz["game_names"]], npz["labels"].astype(np.float32)


def align_labels(feature_names, label_names, labels):
    index = {name: row for row, name in enumerate(label_names)}
    aligned = np.zeros((len(feature_names), labels.shape[1]), dtype=np.float32)
    for row, name in enumerate(feature_names):
        if name in index:
            aligned[row] = labels[index[name]]
    return aligned


def split_indices(keep, test_ratio, seed):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(keep)
    test_count = max(1, int(round(len(keep) * test_ratio)))
    return shuffled[test_count:], shuffled[:test_count]


def average_precision(scores, labels):
    if labels.sum() == 0:
        return np.nan
    order = np.argsort(-scores)
    ordered = labels[order]
    cum_tp = np.cumsum(ordered)
    precision = cum_tp / (np.arange(len(ordered)) + 1)
    return float((precision * ordered).sum() / ordered.sum())


def evaluate(logits, labels):
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    preds = (probs >= 0.5).astype(np.float32)
    tp = float((preds * labels).sum())
    fp = float((preds * (1 - labels)).sum())
    fn = float(((1 - preds) * labels).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    aps = [average_precision(probs[:, tag], labels[:, tag]) for tag in range(labels.shape[1])]
    aps = np.asarray(aps, dtype=np.float64)
    mean_ap = float(np.nanmean(aps)) if np.any(~np.isnan(aps)) else 0.0
    return {"micro_f1": micro_f1, "precision": precision, "recall": recall, "mAP": mean_ap}, aps


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    with h5py.File(args.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])

    encoder, cfg, enc_epoch, enc_step = load_frozen_encoder(args.checkpoint, input_dim, device)
    print(f"encoder loaded from {args.checkpoint} (epoch={enc_epoch} step={enc_step}) cfg={cfg}", flush=True)

    feats, feature_names = extract_features(
        encoder, args.h5, args.sample_fraction, args.feature_views,
        args.seed, args.cache_dtype, device, args.amp,
    )
    vocab, label_names, raw_labels = load_labels(args.tags_dir)
    labels = align_labels(feature_names, label_names, raw_labels)
    num_tags = labels.shape[1]
    print(f"features={tuple(feats.shape)} tags={num_tags} target_mode={vocab['target_mode']}", flush=True)

    keep = np.flatnonzero((labels > 0).any(axis=1))
    train_idx, val_idx = split_indices(keep, args.test_ratio, args.seed)
    print(f"games_with_labels={len(keep)} train={len(train_idx)} val={len(val_idx)}", flush=True)

    feats_flat = feats.to(device)
    labels_t = torch.from_numpy(labels).to(device)
    train_x = feats_flat[train_idx]
    train_y = labels_t[train_idx]
    val_x = feats_flat[val_idx]
    val_y = labels[val_idx]

    head = TagRegressionHead(
        num_tags=num_tags,
        num_latents=feats.shape[1],
        latent_out_dim=feats.shape[2],
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        pool=args.pool,
    ).to(device)
    print(f"head params={sum(p.numel() for p in head.parameters())} pool={args.pool}", flush=True)

    # pos_weight counters tag sparsity in the multi-label BCE.
    pos = train_y.clamp(0, 1).sum(dim=0)
    neg = train_y.shape[0] - pos
    pos_weight = (neg / pos.clamp(min=1.0)).clamp(max=args.max_pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    best_score = -float("inf")
    best_state = None
    best_metrics = None
    plateau_count = 0
    grad_ema = None

    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_x)
        loss = criterion(logits, train_y.clamp(0, 1) if vocab["target_mode"] == "binary" else train_y)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip))
        optimizer.step()
        grad_ema = grad_norm if grad_ema is None else 0.9 * grad_ema + 0.1 * grad_norm

        head.eval()
        with torch.no_grad():
            val_logits = head(val_x).cpu().numpy()
            val_loss = float(
                nn.functional.binary_cross_entropy_with_logits(
                    torch.from_numpy(val_logits), torch.from_numpy(val_y)
                )
            )
        metrics, _ = evaluate(val_logits, val_y)
        score = metrics["mAP"]
        row = {"epoch": epoch, "train_loss": float(loss.detach().cpu()), "val_loss": val_loss,
               "grad_norm": grad_norm, "grad_ema": grad_ema, **metrics}
        history.append(row)

        improved = score > best_score + args.min_delta
        if improved:
            best_score = score
            best_metrics = row
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            plateau_count = 0
        else:
            plateau_count += 1

        if epoch == 1 or epoch % args.log_every == 0 or improved:
            print(
                f"epoch={epoch:04d} train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
                f"mAP={metrics['mAP']:.4f} micro_f1={metrics['micro_f1']:.4f} "
                f"grad_ema={grad_ema:.4f} plateau={plateau_count}/{args.patience}",
                flush=True,
            )

        # Self-stop: val-mAP plateau, or the gradient flattened out.
        if plateau_count >= args.patience:
            print(f"early stop: no val mAP gain for {args.patience} epochs", flush=True)
            break
        if args.grad_plateau > 0 and grad_ema < args.grad_plateau:
            print(f"early stop: smoothed grad norm {grad_ema:.5f} < {args.grad_plateau}", flush=True)
            break

    if best_state is not None:
        head.load_state_dict(best_state)

    # Per-tag AP report on the best head.
    head.eval()
    with torch.no_grad():
        final_logits = head(val_x).cpu().numpy()
    final_metrics, per_tag_ap = evaluate(final_logits, val_y)
    tags = vocab["tags"]
    ranked = sorted(
        [(tags[i], per_tag_ap[i]) for i in range(num_tags) if not np.isnan(per_tag_ap[i])],
        key=lambda item: -item[1],
    )

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "encoder_epoch": enc_epoch,
        "encoder_global_step": enc_step,
        "encoder_cfg": cfg,
        "num_tags": num_tags,
        "target_mode": vocab["target_mode"],
        "games_with_labels": int(len(keep)),
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "best_metrics": best_metrics,
        "final_val_metrics": final_metrics,
        "top_tags": ranked[:15],
        "bottom_tags": ranked[-15:],
        "stopped_epoch": history[-1]["epoch"] if history else 0,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_text_write(json.dumps(report, ensure_ascii=False, indent=2), args.report_json)
    if history:
        columns = list(history[0].keys())
        lines = ["\t".join(columns)]
        for row in history:
            lines.append("\t".join(
                f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c]) for c in columns
            ))
        atomic_text_write("\n".join(lines) + "\n", args.history_tsv)
    if not args.no_save:
        atomic_torch_save(
            {"head_state_dict": head.state_dict(), "num_tags": num_tags, "tags": tags,
             "pool": args.pool, "hidden_dims": list(args.hidden_dims), "report": report},
            args.head_out,
        )

    print(
        f"DONE best_mAP={best_score:.4f} final_mAP={final_metrics['mAP']:.4f} "
        f"micro_f1={final_metrics['micro_f1']:.4f}",
        flush=True,
    )
    print(f"wrote {args.report_json}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default=str(DEFAULT_H5))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--tags-dir", default=str(DEFAULT_TAGS_DIR))
    parser.add_argument("--head-out", default=str(DEFAULT_OUT_DIR / "tag_probe_head.pt"))
    parser.add_argument("--report-json", default=str(DEFAULT_OUT_DIR / "tag_probe_report.json"))
    parser.add_argument("--history-tsv", default=str(DEFAULT_OUT_DIR / "tag_probe_history.tsv"))
    parser.add_argument("--no-save", action="store_true")

    parser.add_argument("--feature-views", type=int, default=4,
                        help="Views sampled per game; their codes are averaged into one feature.")
    parser.add_argument("--sample-fraction", type=float, default=0.6)
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--pool", choices=["flatten", "mean"], default="flatten")

    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=50,
                        help="Stop after this many epochs without val mAP improvement.")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-plateau", type=float, default=0.0,
                        help="Stop when the smoothed grad norm falls below this (0 disables).")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
