"""Ceiling diagnostic: what micro-F1 is even achievable on these 293 games?

This bypasses the VICReg encoder entirely. For each game we mean-pool a random
sample of its raw Qwen sentence embeddings into one 1024-d feature, then run
k-fold cross-validation with a per-tag linear classifier. The resulting micro-F1
is an *upper bound* on what any frozen-encoder probe can reach: if the raw
embedding can't predict tags, no compression of it (the 18-d VICReg code) can.

Fairness rule (per the task): a tag is only scored on a fold when it has >= 2
positive games in that fold's TRAIN split (and >= 1 in val). Tags too rare to
learn are excluded from the metric instead of silently tanking it.
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_H5 = PROJECT_ROOT / "game_review_data" / "embedding_h5.h5"
DEFAULT_TAGS_DIR = SCRIPT_DIR / "tags"
DEFAULT_CACHE = SCRIPT_DIR / "tags" / "raw_mean_features.npz"


def decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def build_mean_features(h5_path, max_sentences, seed):
    """One mean-pooled 1024-d vector per game from a random sentence sample."""
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as h5:
        names = [decode_name(n) for n in h5["game_names"][:]]
        game_offsets = h5["game_review_offsets"][:]
        review_offsets = h5["review_offsets"]
        vectors = h5["vectors"]
        dim = vectors.shape[1]
        feats = np.zeros((len(names), dim), dtype=np.float32)
        for gi in range(len(names)):
            r_start = int(game_offsets[gi])
            r_end = int(game_offsets[gi + 1])
            s_start = int(review_offsets[r_start])
            s_end = int(review_offsets[r_end])
            n_sent = s_end - s_start
            if n_sent <= max_sentences:
                block = vectors[s_start:s_end].astype(np.float32)
            else:
                # Random contiguous-friendly sample: sort indices to keep the
                # h5 read mostly sequential.
                idx = np.sort(rng.choice(n_sent, size=max_sentences, replace=False)) + s_start
                block = vectors[idx].astype(np.float32)
            feats[gi] = block.mean(axis=0)
            if (gi + 1) % 50 == 0 or gi + 1 == len(names):
                print(f"mean-pool {gi + 1}/{len(names)}", flush=True)
    return np.asarray(names), feats


def load_labels(tags_dir, h5_path):
    from VICReg_review.train_tag_probe import load_labels as load_tap_labels
    return load_tap_labels(tags_dir, h5_path)


def align(feature_names, label_names, labels):
    index = {n: i for i, n in enumerate(label_names)}
    out = np.zeros((len(feature_names), labels.shape[1]), dtype=np.int8)
    keep = np.zeros(len(feature_names), dtype=bool)
    for row, name in enumerate(feature_names):
        if name in index:
            out[row] = labels[index[name]]
            keep[row] = True
    return out, keep


def kfold_indices(n, k, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, val


def micro_f1(pred, true):
    tp = float((pred & true).sum())
    fp = float((pred & ~true).sum())
    fn = float((~pred & true).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def run(args):
    cache = Path(args.cache)
    if cache.exists() and not args.rebuild:
        data = np.load(cache, allow_pickle=True)
        feat_names, feats = list(data["names"]), data["feats"]
        print(f"loaded cached features {feats.shape} from {cache}", flush=True)
    else:
        feat_names, feats = build_mean_features(args.h5, args.max_sentences, args.seed)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, names=np.asarray(feat_names), feats=feats)
        print(f"cached features {feats.shape} -> {cache}", flush=True)

    tags, label_names, labels = load_labels(args.tags_dir, args.h5)
    y, keep = align(np.asarray(feat_names), label_names, labels)
    feats = feats[keep]
    y = y[keep]
    n, num_tags = y.shape
    print(f"games_with_labels={n} tags={num_tags}", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Accumulate predictions across folds, but only on (fold, tag) pairs that
    # satisfy the fairness rule, so the micro-F1 is computed on learnable cells.
    all_pred, all_true = [], []
    per_tag_tp = np.zeros(num_tags)
    per_tag_fp = np.zeros(num_tags)
    per_tag_fn = np.zeros(num_tags)
    scored_tags = np.zeros(num_tags, dtype=bool)

    for fold, (tr, va) in enumerate(kfold_indices(n, args.folds, args.seed)):
        scaler = StandardScaler().fit(feats[tr])
        Xtr = scaler.transform(feats[tr])
        Xva = scaler.transform(feats[va])
        fold_pred = np.zeros((len(va), num_tags), dtype=bool)
        fold_true = y[va].astype(bool)
        learnable = np.zeros(num_tags, dtype=bool)
        for t in range(num_tags):
            tr_pos = int(y[tr, t].sum())
            va_pos = int(y[va, t].sum())
            if tr_pos < args.min_train_pos or va_pos < 1:
                continue
            learnable[t] = True
            scored_tags[t] = True
            clf = LogisticRegression(
                C=args.C, max_iter=2000, class_weight="balanced",
            )
            clf.fit(Xtr, y[tr, t])
            prob = clf.predict_proba(Xva)[:, 1]
            # Tune threshold on TRAIN to avoid val leakage.
            tr_prob = clf.predict_proba(Xtr)[:, 1]
            best_thr, best_f1 = 0.5, -1.0
            for thr in np.linspace(0.1, 0.9, 33):
                pr = tr_prob >= thr
                f1, _, _ = micro_f1(pr, y[tr, t].astype(bool))
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr
            pred = prob >= best_thr
            fold_pred[:, t] = pred
            per_tag_tp[t] += float((pred & fold_true[:, t]).sum())
            per_tag_fp[t] += float((pred & ~fold_true[:, t]).sum())
            per_tag_fn[t] += float((~pred & fold_true[:, t]).sum())
        # Micro over learnable cells only.
        mask = learnable
        all_pred.append(fold_pred[:, mask])
        all_true.append(fold_true[:, mask])
        f1, p, r = micro_f1(fold_pred[:, mask], fold_true[:, mask])
        print(f"fold {fold}: learnable_tags={int(mask.sum())} micro_f1={f1:.4f} P={p:.3f} R={r:.3f}", flush=True)

    # Overall micro-F1 across all folds (learnable cells).
    tp = per_tag_tp.sum(); fp = per_tag_fp.sum(); fn = per_tag_fn.sum()
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    # Macro-F1 over scored tags.
    per_f1 = []
    for t in np.flatnonzero(scored_tags):
        p = per_tag_tp[t] / (per_tag_tp[t] + per_tag_fp[t]) if per_tag_tp[t] + per_tag_fp[t] else 0.0
        r = per_tag_tp[t] / (per_tag_tp[t] + per_tag_fn[t]) if per_tag_tp[t] + per_tag_fn[t] else 0.0
        per_f1.append((tags[t], 2 * p * r / (p + r) if p + r else 0.0, int((y[:, t]).sum())))
    macro = float(np.mean([f for _, f, _ in per_f1])) if per_f1 else 0.0

    print("=" * 60, flush=True)
    print(f"OVERALL micro_f1={F1:.4f} P={P:.3f} R={R:.3f}", flush=True)
    print(f"macro_f1={macro:.4f} over {int(scored_tags.sum())} scored tags", flush=True)

    # How does micro-F1 climb as we drop rare / un-learnable tags?
    doc_freq = y.sum(axis=0)
    print("-" * 60, flush=True)
    print("micro-F1 restricted to tags with doc-freq >= floor:", flush=True)
    for floor in [5, 10, 20, 30, 40, 60, 80]:
        sel = scored_tags & (doc_freq >= floor)
        if not sel.any():
            continue
        tp = per_tag_tp[sel].sum(); fp = per_tag_fp[sel].sum(); fn = per_tag_fn[sel].sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"  freq>={floor:3d}: tags={int(sel.sum()):3d} micro_f1={f1:.4f} P={p:.3f} R={r:.3f}", flush=True)
    # Also: keep only tags whose own CV F1 clears a quality bar.
    print("micro-F1 restricted to tags with per-tag F1 >= bar:", flush=True)
    per_tag_f1_full = np.zeros(num_tags)
    for t in np.flatnonzero(scored_tags):
        pp = per_tag_tp[t] / (per_tag_tp[t] + per_tag_fp[t]) if per_tag_tp[t] + per_tag_fp[t] else 0.0
        rr = per_tag_tp[t] / (per_tag_tp[t] + per_tag_fn[t]) if per_tag_tp[t] + per_tag_fn[t] else 0.0
        per_tag_f1_full[t] = 2 * pp * rr / (pp + rr) if pp + rr else 0.0
    for bar in [0.5, 0.6, 0.7]:
        sel = scored_tags & (per_tag_f1_full >= bar)
        if not sel.any():
            continue
        tp = per_tag_tp[sel].sum(); fp = per_tag_fp[sel].sum(); fn = per_tag_fn[sel].sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"  f1>={bar}: tags={int(sel.sum()):3d} micro_f1={f1:.4f}", flush=True)
    per_f1.sort(key=lambda x: -x[1])
    print("top tags:", [(t, round(f, 2)) for t, f, _ in per_f1[:12]], flush=True)
    print("bottom tags:", [(t, round(f, 2)) for t, f, _ in per_f1[-12:]], flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", default=str(DEFAULT_H5))
    p.add_argument("--tags-dir", default=str(DEFAULT_TAGS_DIR))
    p.add_argument("--cache", default=str(DEFAULT_CACHE))
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--max-sentences", type=int, default=4000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--min-train-pos", type=int, default=2)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
