"""Dual-probe validation: tag probe + PXI probe on a (frozen) VICReg encoder.

Used two ways:
  * standalone, on a checkpoint:  python dual_probe.py --checkpoint ...
  * in-training, every N epochs:   train_vicreg_review_h5.py --probe-every N

It reports, all from the encoder code (no raw embeddings needed, so it is cheap
enough to run during training):

  tag axis (5-fold CV, fair, stats pool for speed):
    tag_content_f1      micro-F1 over content tags (mechanics+story)  -> want high
    tag_subjective_f1   micro-F1 over subjective tags                 -> want lower
    code_sentiment_r2   R^2 regressing cached SST sentiment from code -> want low
  PXI axis (leave-one-out on the games shared with the PXI benchmark):
    pxi_func_f1         median-split F1 over functional dims
    pxi_psych_f1        median-split F1 over psychological dims

Inputs are the TAP labels stored in the H5 plus probe_selectivity.py /
pxi_vicreg_overlap.py artifacts; any missing input is skipped gracefully (NaN) so training never
dies because of the probe.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VICReg_review.train_tag_probe import (  # noqa: E402
    extract_features, pool_features, load_labels, kfold_indices, micro_prf,
)

DEFAULT_TAGS_DIR = SCRIPT_DIR / "tags"
DEFAULT_SENTIMENT = SCRIPT_DIR / "tags" / "game_sentiment.npz"
DEFAULT_PXI_OVERLAP = PROJECT_ROOT / "PXIbench_test" / "pxi_vicreg_overlap.json"

PXI_FUNCTIONAL = ["functional_progress_feedback", "functional_ease_of_control",
                  "functional_audiovisual_appeal", "functional_goals_and_rules", "functional_challenge"]
PXI_PSYCHOLOGICAL = ["psychological_meaning", "psychological_mastery", "psychological_curiosity",
                     "psychological_autonomy", "psychological_immersion"]

METRIC_KEYS = ["tag_content_f1", "tag_subjective_f1", "tag_selectivity",
               "code_sentiment_r2", "pxi_func_f1", "pxi_psych_f1", "pxi_delta", "n_pxi"]


def _align_labels(feature_names, label_names, labels):
    index = {n: i for i, n in enumerate(label_names)}
    out = np.zeros((len(feature_names), labels.shape[1]), dtype=np.int8)
    keep = np.zeros(len(feature_names), dtype=bool)
    for row, name in enumerate(feature_names):
        if name in index:
            out[row] = labels[index[name]]
            keep[row] = True
    return out, keep


def _group_f1(X, y, tag_cols, folds, min_train_pos, C, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tp = fp = fn = 0.0
    for tr, va in kfold_indices(X.shape[0], folds, seed):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
        for t in tag_cols:
            if int(y[tr, t].sum()) < min_train_pos or int(y[va, t].sum()) < 1:
                continue
            clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, y[tr, t])
            trp, vap = clf.predict_proba(Xtr)[:, 1], clf.predict_proba(Xva)[:, 1]
            best_thr, best = 0.5, -1.0
            for thr in np.linspace(0.1, 0.9, 17):
                f1, _, _ = micro_prf(float(((trp >= thr) & (y[tr, t] > 0)).sum()),
                                     float(((trp >= thr) & (y[tr, t] == 0)).sum()),
                                     float(((trp < thr) & (y[tr, t] > 0)).sum()))
                if f1 > best:
                    best, best_thr = f1, thr
            pred = vap >= best_thr
            tru = y[va, t] > 0
            tp += float((pred & tru).sum()); fp += float((pred & ~tru).sum()); fn += float((~pred & tru).sum())
    return micro_prf(tp, fp, fn)[0]


def _sentiment_r2(X, names, sentiment_cache, folds, seed):
    path = Path(sentiment_cache)
    if not path.exists():
        return float("nan")
    d = np.load(path, allow_pickle=True)
    sent = {str(n): float(s) for n, s in zip(d["names"], d["sent"])}
    rows = [i for i, n in enumerate(names) if n in sent]
    if len(rows) < folds + 1:
        return float("nan")
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    Xs = X[rows]
    y = np.array([sent[names[i]] for i in rows], dtype=float)
    pred = np.zeros(len(y))
    for tr, va in kfold_indices(len(y), folds, seed):
        sc = StandardScaler().fit(Xs[tr])
        pred[va] = Ridge(alpha=10.0).fit(sc.transform(Xs[tr]), y[tr]).predict(sc.transform(Xs[va]))
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _pxi_group_f1(Xmap, overlap_path, groups_cols, pca=8, C=1.0):
    path = Path(overlap_path)
    if not path.exists():
        return float("nan"), float("nan"), 0
    ov = json.loads(path.read_text(encoding="utf-8"))
    dims = ov["dims"]; di = {d: k for k, d in enumerate(dims)}
    appid2pxi = {m["appid"]: [m["pxi"][d] for d in dims] for m in ov["matches"]}
    appids = [a for a in appid2pxi if a in Xmap]
    if len(appids) < 6:
        return float("nan"), float("nan"), len(appids)
    X = np.array([Xmap[a] for a in appids]); Y = np.array([appid2pxi[a] for a in appids], dtype=float)
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def loo_f1(cols):
        tp = fp = fn = 0.0
        for k in cols:
            y = Y[:, di[k]]
            yb = (y >= np.median(y))
            pred = np.zeros(len(yb), dtype=bool)
            for i in range(len(yb)):
                tr = [j for j in range(len(yb)) if j != i]
                if len(set(yb[tr])) < 2:
                    pred[i] = bool(round(float(np.mean(yb[tr])))); continue
                sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[i:i + 1])
                if pca and pca < Xtr.shape[1]:
                    p = PCA(n_components=min(pca, Xtr.shape[0] - 1)).fit(Xtr)
                    Xtr, Xte = p.transform(Xtr), p.transform(Xte)
                pred[i] = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, yb[tr]).predict(Xte)[0]
            tp += float((pred & yb).sum()); fp += float((pred & ~yb).sum()); fn += float((~pred & yb).sum())
        return micro_prf(tp, fp, fn)[0]

    return loo_f1(groups_cols[0]), loo_f1(groups_cols[1]), len(appids)


def evaluate(feats, names, tags_dir=DEFAULT_TAGS_DIR, h5_path=None, sentiment_cache=DEFAULT_SENTIMENT,
             pxi_overlap=DEFAULT_PXI_OVERLAP, folds=5, min_train_pos=2, C=1.0, seed=42, pool="stats"):
    """feats: (G, num_latents, output_dim) numpy. Returns a flat metric dict."""
    tags, label_names, labels = load_labels(tags_dir, h5_path)
    tag_id = {t: i for i, t in enumerate(tags)}
    groups_path = Path(tags_dir) / "tag_groups.json"
    if groups_path.exists() and h5_path is None:
        groups = json.loads(groups_path.read_text(encoding="utf-8"))
        content_cols = [tag_id[t] for t in groups.get("content", []) if t in tag_id]
        subj_cols = [tag_id[t] for t in groups.get("subjective", []) if t in tag_id]
    else:
        content_cols = list(range(len(tags)))
        subj_cols = []

    y, keep = _align_labels(names, label_names, labels)
    X = pool_features(feats, pool)
    Xk, yk, names_k = X[keep], y[keep], [n for n, k in zip(names, keep) if k]

    content_f1 = _group_f1(Xk, yk, content_cols, folds, min_train_pos, C, seed)
    subj_f1 = _group_f1(Xk, yk, subj_cols, folds, min_train_pos, C, seed)
    sent_r2 = _sentiment_r2(Xk, names_k, sentiment_cache, folds, seed)

    Xmap = {n.split("_")[0]: X[i] for i, n in enumerate(names)}
    func_f1, psych_f1, n_pxi = _pxi_group_f1(Xmap, pxi_overlap, (PXI_FUNCTIONAL, PXI_PSYCHOLOGICAL))

    return {
        "tag_content_f1": content_f1,
        "tag_subjective_f1": subj_f1,
        "tag_selectivity": content_f1 - subj_f1,
        "code_sentiment_r2": sent_r2,
        "pxi_func_f1": func_f1,
        "pxi_psych_f1": psych_f1,
        "pxi_delta": (func_f1 - psych_f1) if not (np.isnan(func_f1) or np.isnan(psych_f1)) else float("nan"),
        "n_pxi": n_pxi,
    }


def probe_encoder(encoder, h5_path, device, amp=False, feature_views=2, sample_fraction=0.6,
                  cache_dtype="float16", seed=42, **eval_kwargs):
    """Extract features from a (frozen-for-the-call) encoder and run both probes.
    Saves/restores the encoder's train/eval mode so it is safe to call mid-training.
    """
    was_training = encoder.training
    encoder.eval()
    try:
        with torch.no_grad():
            feats, names = extract_features(
                encoder, str(h5_path), sample_fraction, feature_views, seed, cache_dtype, device, amp
            )
        return evaluate(feats, names, h5_path=h5_path, seed=seed, **eval_kwargs)
    finally:
        if was_training:
            encoder.train()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(SCRIPT_DIR / "heads" / "sweep_adv" / "vicreg_adv10_best.pt"))
    p.add_argument("--h5", default=str(PROJECT_ROOT / "game_review_data" / "embedding_h5.h5"))
    p.add_argument("--tags-dir", default=str(DEFAULT_TAGS_DIR))
    p.add_argument("--feature-views", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    import h5py
    from VICReg_review.train_tag_probe import load_frozen_encoder
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with h5py.File(args.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])
    encoder, _, epoch, step = load_frozen_encoder(args.checkpoint, input_dim, device)
    metrics = probe_encoder(encoder, args.h5, device, amp=args.amp,
                            feature_views=args.feature_views, tags_dir=args.tags_dir, seed=args.seed)
    print(f"checkpoint epoch={epoch} step={step}")
    for k in METRIC_KEYS:
        print(f"  {k:20s} {metrics[k]}")
    print(time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
