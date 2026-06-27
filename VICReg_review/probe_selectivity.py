"""Selectivity probe: does the encoder KEEP content while DROPPING subjectivity?

This is the fair evaluation for the actual hypothesis (not "tag-F1 >= 0.85").
The siamese VICReg encoder + sentiment adversary is supposed to retain a game's
story/mechanics signal and strip subjective opinion. A bare F1 number can't show
that; a *relative* one can.

For each representation we measure three decodabilities with the same fair
5-fold CV used by train_tag_probe.py:

  content_f1     micro-F1 over CONTENT tags (mechanics + story)      -> keep high
  subjective_f1  micro-F1 over SUBJECTIVE tags (affect/mood/quality) -> drop
  sentiment_r2   R^2 of regressing each game's mean SST sentiment    -> drop hard
                 (the exact axis the GRL adversary attacks)

We run this on the RAW Qwen embedding (the information ceiling) and on the frozen
VICReg code, then report RETENTION = vicreg / raw for each axis. The hypothesis
predicts:

  content retention   >>  subjective retention  >=  sentiment retention

i.e. content survives the encoder, subjectivity/sentiment does not. The headline
number is the SELECTIVITY GAP = content_retention - sentiment_retention; a
positive gap is evidence the encoder does what it claims. Absolute F1 is NOT the
point here -- both reps are scored against the same tiny dataset, so the gap is
what's diagnostic.

Inputs are the caches built by ceiling_diagnostic.py (raw) and train_tag_probe.py
(vicreg). Sentiment targets are built here from the frozen SST head and cached.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VICReg_review.model import load_mlp4_a_sentiment_head  # noqa: E402

DEFAULT_H5 = PROJECT_ROOT / "game_review_data" / "embedding_h5.h5"
DEFAULT_TAGS_DIR = SCRIPT_DIR / "tags"
DEFAULT_RAW_CACHE = SCRIPT_DIR / "tags" / "raw_mean_features.npz"
DEFAULT_VIC_CACHE = SCRIPT_DIR / "tags" / "probe_feat_vicreg_review_h5_latest_3_fv4_sf0.6.npz"
DEFAULT_SST = PROJECT_ROOT / "sst" / "heads" / "mlp4_1024_128_32_8_1_best.pt"
DEFAULT_SENT_CACHE = SCRIPT_DIR / "tags" / "game_sentiment.npz"


def decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def build_sentiment_targets(h5_path, sst_ckpt, max_sentences, seed, device):
    """Per-game mean SST sentiment from a random sentence sample (the subjective axis)."""
    head = load_mlp4_a_sentiment_head(sst_ckpt, map_location=device).to(device).eval()
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as h5:
        names = [decode_name(n) for n in h5["game_names"][:]]
        game_offsets = h5["game_review_offsets"][:]
        review_offsets = h5["review_offsets"]
        vectors = h5["vectors"]
        sent = np.zeros(len(names), dtype=np.float32)
        for gi in range(len(names)):
            s_start = int(review_offsets[int(game_offsets[gi])])
            s_end = int(review_offsets[int(game_offsets[gi + 1])])
            n = s_end - s_start
            if n <= max_sentences:
                idx = np.arange(s_start, s_end)
            else:
                idx = np.sort(rng.choice(n, size=max_sentences, replace=False)) + s_start
            block = torch.from_numpy(vectors[idx].astype(np.float32)).to(device)
            with torch.no_grad():
                scores = head(block).squeeze(-1).float().cpu().numpy()
            sent[gi] = float(scores.mean())
            if (gi + 1) % 50 == 0 or gi + 1 == len(names):
                print(f"sentiment {gi + 1}/{len(names)}", flush=True)
    return np.asarray(names), sent


def load_npz_features(path):
    data = np.load(path, allow_pickle=True)
    feats = data["feats"]
    if feats.ndim == 3:  # (games, num_latents, output_dim) -> flatten
        feats = feats.reshape(feats.shape[0], -1)
    return [str(n) for n in data["names"]], feats.astype(np.float32)


def load_labels(tags_dir, h5_path):
    from VICReg_review.train_tag_probe import load_labels as load_tap_labels
    return load_tap_labels(tags_dir, h5_path)


def kfold(n, k, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        yield tr, va


def micro_prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def probe_tags(X, y, tag_cols, folds, min_train_pos, C, seed):
    """Fair CV micro-F1 over a chosen set of tag columns."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tp = fp = fn = 0.0
    scored = 0
    for tr, va in kfold(X.shape[0], folds, seed):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
        for t in tag_cols:
            if int(y[tr, t].sum()) < min_train_pos or int(y[va, t].sum()) < 1:
                continue
            scored += 1
            clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
            clf.fit(Xtr, y[tr, t])
            trp = clf.predict_proba(Xtr)[:, 1]
            vap = clf.predict_proba(Xva)[:, 1]
            best_thr, best = 0.5, -1.0
            for thr in np.linspace(0.1, 0.9, 33):
                f1, _, _ = micro_prf(
                    float(((trp >= thr) & (y[tr, t] > 0)).sum()),
                    float(((trp >= thr) & (y[tr, t] == 0)).sum()),
                    float(((trp < thr) & (y[tr, t] > 0)).sum()),
                )
                if f1 > best:
                    best, best_thr = f1, thr
            pred = vap >= best_thr
            tru = y[va, t] > 0
            tp += float((pred & tru).sum()); fp += float((pred & ~tru).sum()); fn += float((~pred & tru).sum())
    f1, p, r = micro_prf(tp, fp, fn)
    return {"micro_f1": f1, "precision": p, "recall": r, "scored_cells_tags": scored}


def probe_sentiment(X, s, folds, seed):
    """Fair CV R^2 + Pearson r of regressing per-game mean sentiment."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    preds = np.zeros_like(s)
    for tr, va in kfold(X.shape[0], folds, seed):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
        reg = Ridge(alpha=10.0).fit(Xtr, s[tr])
        preds[va] = reg.predict(Xva)
    ss_res = float(((s - preds) ** 2).sum())
    ss_tot = float(((s - s.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    pear = float(np.corrcoef(s, preds)[0, 1]) if s.std() > 0 else 0.0
    return {"r2": r2, "pearson": pear}


def align(names, label_names, y, sent_names, sent):
    li = {n: i for i, n in enumerate(label_names)}
    si = {n: i for i, n in enumerate(sent_names)}
    rows = [r for r, n in enumerate(names) if n in li and n in si]
    keep = np.array(rows, dtype=int)
    Y = np.stack([y[li[names[r]]] for r in rows])
    S = np.array([sent[si[names[r]]] for r in rows], dtype=np.float32)
    return keep, Y, S


def evaluate(rep_name, X, Y, S, cols, args, results):
    content = probe_tags(X, Y, cols["content"], args.folds, args.min_train_pos, args.C, args.seed)
    subjective = probe_tags(X, Y, cols["subjective"], args.folds, args.min_train_pos, args.C, args.seed)
    sentiment = probe_sentiment(X, S, args.folds, args.seed)
    print(f"[{rep_name}] content_f1={content['micro_f1']:.4f}  "
          f"subjective_f1={subjective['micro_f1']:.4f}  "
          f"sentiment_r2={sentiment['r2']:.4f} (r={sentiment['pearson']:.3f})", flush=True)
    results[rep_name] = {"content": content, "subjective": subjective, "sentiment": sentiment}


def run(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Sentiment targets (cache).
    if Path(args.sent_cache).exists() and not args.rebuild_sentiment:
        d = np.load(args.sent_cache, allow_pickle=True)
        sent_names, sent = [str(n) for n in d["names"]], d["sent"]
        print(f"loaded sentiment targets from {args.sent_cache}", flush=True)
    else:
        sent_names, sent = build_sentiment_targets(args.h5, args.sst_checkpoint, args.max_sentences, args.seed, device)
        np.savez(args.sent_cache, names=np.asarray(sent_names), sent=sent)
        print(f"cached sentiment targets -> {args.sent_cache}", flush=True)
    print(f"game sentiment: mean={sent.mean():.3f} std={sent.std():.3f} "
          f"min={sent.min():.3f} max={sent.max():.3f}", flush=True)

    tags, label_names, y = load_labels(args.tags_dir, args.h5)
    tag_id = {t: i for i, t in enumerate(tags)}
    groups_path = Path(args.tags_dir) / "tag_groups.json"
    if groups_path.exists():
        groups = json.loads(groups_path.read_text(encoding="utf-8"))
        cols = {g: [tag_id[t] for t in groups[g] if t in tag_id] for g in ("content", "subjective", "mechanics", "story")}
    else:
        cols = {
            "content": list(range(len(tags))),
            "subjective": [],
            "mechanics": list(range(len(tags))),
            "story": [],
        }
    print(f"group sizes: content={len(cols['content'])} subjective={len(cols['subjective'])} "
          f"(mechanics={len(cols['mechanics'])} story={len(cols['story'])})", flush=True)

    raw_names, raw_X = load_npz_features(args.raw_cache)
    vic_names, vic_X = load_npz_features(args.vic_cache)

    results = {}
    # Raw ceiling.
    keep, Y, S = align(raw_names, label_names, y, sent_names, sent)
    evaluate("raw", raw_X[keep], Y, S, cols, args, results)
    # VICReg code.
    keep, Y, S = align(vic_names, label_names, y, sent_names, sent)
    evaluate("vicreg", vic_X[keep], Y, S, cols, args, results)

    # Retention vs the raw ceiling.
    def retention(axis, key):
        rv = results["raw"][axis][key]
        vv = results["vicreg"][axis][key]
        return (vv / rv) if rv > 0 else float("nan")

    content_ret = retention("content", "micro_f1")
    subj_ret = retention("subjective", "micro_f1")
    sent_ret = retention("sentiment", "r2")
    gap = content_ret - sent_ret

    print("=" * 70, flush=True)
    print("RETENTION (vicreg / raw-ceiling):", flush=True)
    print(f"  content     {content_ret:.3f}", flush=True)
    print(f"  subjective  {subj_ret:.3f}", flush=True)
    print(f"  sentiment   {sent_ret:.3f}", flush=True)
    print(f"SELECTIVITY GAP (content - sentiment) = {gap:+.3f}", flush=True)
    verdict = ("supports hypothesis: content kept, sentiment dropped"
               if gap > 0.15 and sent_ret < 0.85 else
               "weak/no selectivity: encoder does not clearly separate the axes")
    print(f"VERDICT: {verdict}", flush=True)

    report = {
        "raw_cache": str(Path(args.raw_cache).resolve()),
        "vic_cache": str(Path(args.vic_cache).resolve()),
        "folds": args.folds,
        "min_train_pos": args.min_train_pos,
        "results": results,
        "retention": {"content": content_ret, "subjective": subj_ret, "sentiment": sent_ret},
        "selectivity_gap": gap,
        "verdict": verdict,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.report_json}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", default=str(DEFAULT_H5))
    p.add_argument("--tags-dir", default=str(DEFAULT_TAGS_DIR))
    p.add_argument("--raw-cache", default=str(DEFAULT_RAW_CACHE))
    p.add_argument("--vic-cache", default=str(DEFAULT_VIC_CACHE))
    p.add_argument("--sst-checkpoint", default=str(DEFAULT_SST))
    p.add_argument("--sent-cache", default=str(DEFAULT_SENT_CACHE))
    p.add_argument("--report-json", default=str(SCRIPT_DIR / "heads" / "selectivity_report.json"))
    p.add_argument("--rebuild-sentiment", action="store_true")
    p.add_argument("--max-sentences", type=int, default=2000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--min-train-pos", type=int, default=2)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
