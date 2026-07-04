"""TAG recall on the 100-game wiki description set (real_doc.h5).

Mirrors the text_variant_eval protocol exactly, swapping the 2-game variant
texts for the 100 verified wiki descriptions:

* anchor features = per-combo eval_features_full_fv4.npz (mean over latents),
  probe trained on the sweep's own tag_text_eval_split.json (train), alpha +
  threshold selected on val — identical to the reported anchor numbers.
* wiki features   = real_doc.h5 sentence embeddings -> frozen best checkpoint
  via encode_text_centroid (feature_views=4, sample_fraction=0.6), the same
  path the variant texts took.
* raw baseline    = the same ridge protocol on raw_identity_cache_ms4000.npz
  game vectors, tested on the per-game mean of the wiki sentence embeddings.

Reports micro-F1 on the wiki games (overall and split by probe train/val/test
membership) against the matched anchor subset, for raw vs each combo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from VICReg_review.text_variant_eval import (  # noqa: E402
    evaluate_scores,
    train_anchor_ridge,
)
from VICReg_review.identity_diagnostic import encode_text_centroid  # noqa: E402
from VICReg_review.train_tag_probe import load_frozen_encoder  # noqa: E402

DEFAULT_SWEEP_DIR = Path(r"C:\runpod_data\stable-query-latent\VICReg_review\heads\cloud_full_sweep_a100")
DEFAULT_REAL_DOC = SCRIPT_DIR / "wiki_descriptions" / "real_doc.h5"
DEFAULT_LABEL_H5 = ROOT / "game_review_data" / "text_h5.h5"
DEFAULT_COMBOS = (
    "dim036_grl_n2000_view20_lat1024x4",
    "dim036_nogrl_n2000_view20_lat1024x4",
)


def align_labels(h5_path: Path, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """text_variant_eval.align_labels, but reading tag datasets directly so the
    vector-less local text_h5.h5 works as the label source."""
    with h5py.File(h5_path, "r") as h5:
        tag_key = "tag_names" if "tag_names" in h5 else "tap_names"
        label_key = "tag_labels" if "tag_labels" in h5 else "tap_labels"
        tags = [t.decode() if isinstance(t, bytes) else str(t) for t in h5[tag_key][:]]
        label_names = [n.decode() if isinstance(n, bytes) else str(n) for n in h5["game_names"][:]]
        labels = (h5[label_key][:] > 0).astype(np.int8)
    index = {name: i for i, name in enumerate(label_names)}
    y = np.zeros((len(names), labels.shape[1]), dtype=np.int8)
    for row, name in enumerate(names):
        if name in index:
            y[row] = labels[index[name]]
    return y, tags


def load_real_doc(path: Path):
    with h5py.File(path, "r") as h5:
        vectors = h5["vectors"][:]
        offsets = h5["game_sentence_offsets"][:]
        appids = [a.decode() for a in h5["appids"][:]]
        names = [n.decode() for n in h5["game_names"][:]]
    blocks = {appids[i]: vectors[offsets[i]:offsets[i + 1]] for i in range(len(appids))}
    return blocks, appids, dict(zip(appids, names))


def probe_and_score(args, X_anchor, y, name_to_index, split, X_wiki, wiki_rows):
    """Train the anchor ridge, return anchor-test metrics + wiki metrics."""
    scaler, ridge, alpha, threshold, _ = train_anchor_ridge(args, X_anchor, y, name_to_index, split)
    out = {"alpha": alpha, "threshold": threshold}
    test_idx = np.asarray([name_to_index[n] for n in split["test"] if n in name_to_index], dtype=np.int64)
    out["anchor_test"] = evaluate_scores(
        y[test_idx], ridge.predict(scaler.transform(X_anchor[test_idx])), threshold)

    wiki_scores = ridge.predict(scaler.transform(X_wiki))
    y_wiki = y[wiki_rows]
    out["wiki_all"] = evaluate_scores(y_wiki, wiki_scores, threshold)
    out["anchor_same_games"] = evaluate_scores(
        y_wiki, ridge.predict(scaler.transform(X_anchor[wiki_rows])), threshold)

    membership = {}
    for split_name in ("train", "val", "test"):
        members = set(split[split_name])
        mask = np.asarray([n in members for n in wiki_rows_names], dtype=bool)
        if mask.any():
            membership[f"wiki_{split_name}_games"] = evaluate_scores(
                y_wiki[mask], wiki_scores[mask], threshold)
            membership[f"wiki_{split_name}_games"]["n_games"] = int(mask.sum())
    out.update(membership)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--real-doc", type=Path, default=DEFAULT_REAL_DOC)
    parser.add_argument("--label-h5", type=Path, default=DEFAULT_LABEL_H5)
    parser.add_argument("--combos", nargs="+", default=list(DEFAULT_COMBOS))
    parser.add_argument("--feature-views", type=int, default=4)
    parser.add_argument("--sample-fraction", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path,
                        default=SCRIPT_DIR / "wiki_descriptions" / "real_doc_tag_eval.json")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    probe_args = SimpleNamespace(tag_text_threshold_steps=33)
    encode_args = SimpleNamespace(
        feature_views=args.feature_views, sample_fraction=args.sample_fraction,
        seed=args.seed, amp=True)

    split = {k: [str(x) for x in v] for k, v in
             json.loads((args.sweep_dir / "tag_text_eval_split.json").read_text(encoding="utf-8")).items()
             if k in ("train", "val", "test")}
    blocks, wiki_appids, wiki_names = load_real_doc(args.real_doc)

    raw = np.load(args.sweep_dir / "raw_identity_cache_ms4000.npz", allow_pickle=True)
    raw_names = [str(n) for n in raw["names"]]
    appid_to_name = {str(a): str(n) for a, n in zip(raw["appids"], raw["names"])}

    in_gallery = [a for a in wiki_appids if a in appid_to_name]
    missing = [wiki_names[a] for a in wiki_appids if a not in appid_to_name]
    print(f"wiki games in gallery: {len(in_gallery)}/{len(wiki_appids)}"
          + (f" (missing: {', '.join(missing[:5])}...)" if missing else ""), flush=True)

    global wiki_rows_names
    wiki_rows_names = [appid_to_name[a] for a in in_gallery]

    report = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "wiki_games_in_gallery": len(in_gallery), "systems": {}}

    # ---------------- raw baseline ----------------
    X_raw = raw["X"].astype(np.float32)
    name_to_index = {n: i for i, n in enumerate(raw_names)}
    y, _tags = align_labels(args.label_h5, raw_names)
    wiki_rows = np.asarray([name_to_index[n] for n in wiki_rows_names], dtype=np.int64)
    X_wiki_raw = np.stack([blocks[a].mean(axis=0) for a in in_gallery]).astype(np.float32)
    report["systems"]["raw_embedding"] = probe_and_score(
        probe_args, X_raw, y, name_to_index, split, X_wiki_raw, wiki_rows)
    print(f"raw: anchor_test={report['systems']['raw_embedding']['anchor_test']['micro_f1']:.4f} "
          f"wiki={report['systems']['raw_embedding']['wiki_all']['micro_f1']:.4f}", flush=True)

    # ---------------- combos ----------------
    for combo in args.combos:
        combo_dir = args.sweep_dir / combo
        feats_npz = np.load(combo_dir / "eval_features_full_fv4.npz", allow_pickle=True)
        feats = feats_npz["feats"].astype(np.float32)
        names = [str(n) for n in feats_npz["names"]]
        X_anchor = feats.mean(axis=1)
        name_to_index = {n: i for i, n in enumerate(names)}
        y, _tags = align_labels(args.label_h5, names)
        wiki_rows = np.asarray([name_to_index[n] for n in wiki_rows_names], dtype=np.int64)

        encoder, cfg, epoch, _ = load_frozen_encoder(
            combo_dir / "vicreg_review_h5_best.pt", X_raw.shape[1], device)
        X_wiki = np.stack([
            encode_text_centroid(encoder, blocks[a], encode_args, device)
            for a in in_gallery
        ]).astype(np.float32)
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

        result = probe_and_score(probe_args, X_anchor, y, name_to_index, split, X_wiki, wiki_rows)
        result["checkpoint_epoch"] = epoch
        report["systems"][combo] = result
        print(f"{combo}: anchor_test={result['anchor_test']['micro_f1']:.4f} "
              f"wiki={result['wiki_all']['micro_f1']:.4f} "
              f"(anchor on same games={result['anchor_same_games']['micro_f1']:.4f})", flush=True)

    tmp = args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(args.out)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
