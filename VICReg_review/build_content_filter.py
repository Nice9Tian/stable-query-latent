"""Content keep-classifier from BERTopic categories + filtered game vectors.

Maps the 127 tuned BERTopic topics (100k-sentence sample of the 2020-game
corpus) into the paper's categories, keeps THREE content classes --
gameplay & mechanics, story & narrative, aesthetics & audio -- trains a
logistic sentence classifier (keep vs drop) on the sample embeddings, then
streams the H5 to build per-game FILTERED mean vectors (same first-4000-
sentence cap as the raw baseline). Outputs:

    content_keep_model.npz      W (1024,), b, meta
    raw_filtered_game_vectors.npz   X (2020,1024), names, keep_rate

    python VICReg_review/build_content_filter.py --fit-dir <bertopic_fit dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# topic -> category mapping (tuned run, seed 42, 127 topics; see topic_words.json)
GAMEPLAY = {11, 16, 17, 21, 24, 25, 26, 28, 30, 31, 34, 38, 39, 43, 44, 53, 54,
            55, 59, 60, 63, 64, 68, 78, 80, 81, 87, 90, 92, 99, 101, 103, 104,
            109, 113, 117, 121}
NARRATIVE = {29, 37, 57, 69, 86, 120}
AESTHETIC = {6, 13, 72, 74, 106, 125}
KEEP = GAMEPLAY | NARRATIVE | AESTHETIC
# impure mega-topic and spam/garbled clusters: excluded from TRAINING entirely
EXCLUDE_TRAIN = {0, 8, 15, 23, 32, 47, 73, 75, 76, 79, 82, 84, 89, 94, 95, 96,
                 97, 105, 107, 108, 112, 114, 115, 116, 118, 119, 122, 126}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit-dir", required=True, help="dir with topics.npy")
    p.add_argument("--h5", default=str(ROOT / "game_review_data/embedding_h5.h5"))
    p.add_argument("--cache-stem", default=str(
        ROOT / "game_review_data/bertopic_cache/balanced_prefix_h5_4990fee61c_n100000_minchars20_skipmeta1"))
    p.add_argument("--out-dir", default=str(ROOT / "VICReg_review/heads/cloud_full_sweep_a100"))
    p.add_argument("--max-game-sentences", type=int, default=4000)
    args = p.parse_args()

    import h5py
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    topics = np.load(Path(args.fit_dir) / "topics.npy")
    emb = np.load(f"{args.cache_stem}_embeddings.npy").astype(np.float32)
    ok = (topics != -1) & ~np.isin(topics, list(EXCLUDE_TRAIN))
    y = np.isin(topics, list(KEEP)).astype(np.int64)
    Xs, ys = emb[ok], y[ok]
    print(f"training sentences: {len(ys)} (keep={int(ys.sum())}, "
          f"drop={int((1 - ys).sum())})")
    Xtr, Xva, ytr, yva = train_test_split(Xs, ys, test_size=0.15,
                                          random_state=42, stratify=ys)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xva, yva)
    from sklearn.metrics import f1_score
    f1 = f1_score(yva, clf.predict(Xva))
    print(f"keep-classifier val acc={acc:.3f} F1={f1:.3f}")

    out_root = Path(args.out_dir)
    np.savez(out_root / "content_keep_model.npz",
             W=clf.coef_[0].astype(np.float32),
             b=np.float32(clf.intercept_[0]),
             val_acc=acc, val_f1=f1,
             keep_topics=np.asarray(sorted(KEEP)))
    W, b = clf.coef_[0].astype(np.float32), np.float32(clf.intercept_[0])

    # ---- filtered per-game means from the H5 (first-4000-sentence cap,
    # matching the raw baseline's ms4000 protocol)
    with h5py.File(args.h5, "r") as h5:
        names = [x.decode() if isinstance(x, bytes) else str(x)
                 for x in h5["game_names"][:]]
        gro = h5["game_review_offsets"][:]
        ro = h5["review_offsets"][:]
        vec = h5["vectors"]
        X = np.zeros((len(names), vec.shape[1]), dtype=np.float32)
        keep_rate = np.zeros(len(names), dtype=np.float32)
        for i in range(len(names)):
            s = int(ro[int(gro[i])])
            e = int(ro[int(gro[i + 1])]) if i + 1 < len(gro) else int(vec.shape[0])
            e = min(e, s + args.max_game_sentences)
            block = vec[s:e].astype(np.float32)
            m = (block @ W + b) > 0
            keep_rate[i] = float(m.mean()) if len(m) else 0.0
            X[i] = block[m].mean(axis=0) if m.any() else block.mean(axis=0)
            if i % 200 == 0:
                print(f"  game {i}/{len(names)} keep_rate so far "
                      f"{keep_rate[: i + 1].mean():.3f}", flush=True)
    np.savez(out_root / "raw_filtered_game_vectors.npz",
             X=X, names=np.asarray(names, dtype=object), keep_rate=keep_rate)
    print(f"filtered game vectors -> {out_root / 'raw_filtered_game_vectors.npz'}; "
          f"mean keep rate {keep_rate.mean():.3f}")


if __name__ == "__main__":
    main()
