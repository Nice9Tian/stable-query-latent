"""Raw-embedding baselines for the real-text tag regression (no learned encoder).

The paper needs "how much does our representation beat the naked embedding".
Game side = mean of raw Qwen sentence embeddings (raw_identity_cache X,
2020x1024). Ridge is trained on the SAME split/labels as the main eval and
tested on the 258 generated articles, with four article featurisations:

  sent_mean   : mean of the article's raw sentence embeddings (from the pilot
                sentence cache -- same sentences the main eval used)
  whole_text  : the WHOLE article embedded as one string (fresh Qwen pass)
  filt_mean   : sentence mean after the BERTopic content filter (only
                gameplay/story/aesthetics sentences kept)   [--keep-model]
  filt_whole  : whole-text embedding of the KEPT sentences concatenated

The filtered game side uses per-game filtered means computed by
build_filtered_game_vectors.py (--filtered-x). Without those flags only the
raw modes run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANTS = ("neutral", "positive", "negative", "noname")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", default=str(ROOT / "game_review_data/embedding_h5.h5"))
    p.add_argument("--out-dir", default=str(ROOT / "VICReg_review/heads/cloud_full_sweep_a100"))
    p.add_argument("--raw-cache", default=None,
                   help="default <out-dir>/raw_identity_cache_ms4000.npz")
    p.add_argument("--cache", default=None,
                   help="article sentence-embedding cache (pilot cache)")
    p.add_argument("--variant-dir", default=str(ROOT / "VICReg_review/text_variants_generated"))
    p.add_argument("--keep-model", default=None,
                   help="npz with W,b of the sentence keep-classifier (enables filt_*)")
    p.add_argument("--filtered-x", default=None,
                   help="npz with X (2020xD) filtered game means (enables filt_*)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import numpy as np

    from VICReg_review import disturbtion_embed
    from VICReg_review import eval_battery_worker as worker
    from VICReg_review import text_variant_eval as tve

    out_root = Path(args.out_dir)
    raw = np.load(Path(args.raw_cache or out_root / "raw_identity_cache_ms4000.npz"),
                  allow_pickle=True)
    X_raw = raw["X"].astype(np.float32)
    names = [str(n) for n in raw["names"]]
    n2i = {n: i for i, n in enumerate(names)}
    ev = worker.build_eval_args(argparse.Namespace(h5=args.h5, out_dir=str(out_root)))
    ev.out_dir = out_root
    y, _tags = tve.align_labels(Path(args.h5), names)
    split = tve.make_or_load_split(out_root / "tag_text_eval_split.json", names, ev)

    cache = disturbtion_embed.load_npz_payload(
        Path(args.cache or out_root / "text_variant_pilot_cache.npz"))
    c_names = [str(x) for x in cache["names"]]
    c_variants = [str(x) for x in cache["variants"]]
    c_vec = np.asarray(cache["vectors"], dtype=np.float32)   # (total_sent, 1024)
    c_off = np.asarray(cache["offsets"], dtype=np.int64)
    c_texts = [str(t) for t in cache["texts"]]

    rec, rec_text = {}, {}
    for i, (n, v) in enumerate(zip(c_names, c_variants)):
        rec[(n, v)] = c_vec[c_off[i]:c_off[i + 1]]
        rec_text[(n, v)] = c_texts[i]
    games = sorted({n for (n, v) in rec if n in n2i and
                    all((n, w) in rec for w in VARIANTS)})
    print(f"{len(games)} article games; raw game matrix {X_raw.shape}")

    keep_fn = None
    if args.keep_model:
        km = np.load(args.keep_model)
        W, b = km["W"].astype(np.float32), km["b"].astype(np.float32)
        keep_fn = lambda S: (S @ W + b) > 0          # (n_sent,) bool
        print(f"keep-classifier loaded: {args.keep_model}")

    embedder = None
    def embed_whole(texts):
        nonlocal embedder
        if embedder is None:
            embedder = disturbtion_embed.LocalEmbedder(
                disturbtion_embed.DEFAULT_LOCAL_MODEL, device=args.device, batch_size=8)
        return np.asarray(embedder.embed(texts), dtype=np.float32)

    def article_feats(mode):
        out = {}
        if mode in ("sent_mean", "filt_mean"):
            for g in games:
                for v in VARIANTS:
                    S = rec[(g, v)]
                    if mode == "filt_mean" and keep_fn is not None:
                        m = keep_fn(S)
                        S = S[m] if m.any() else S
                    out[(g, v)] = S.mean(axis=0)
        else:                                          # whole-text modes
            todo_texts, keys = [], []
            for g in games:
                for v in VARIANTS:
                    t = rec_text[(g, v)]
                    if mode == "filt_whole" and keep_fn is not None:
                        S = rec[(g, v)]
                        m = keep_fn(S)
                        sents = disturbtion_embed.split_text(t, 4096)
                        if m.any() and len(sents) == len(m):
                            t = " ".join(s for s, k in zip(sents, m) if k)
                    todo_texts.append(t)
                    keys.append((g, v))
            E = embed_whole(todo_texts)
            out = {k: E[i] for i, k in enumerate(keys)}
        return out

    def run_mode(mode, X_game):
        a_scaler, a_ridge, _al, a_th, _vm = tve.train_anchor_ridge(
            ev, X_game, y, n2i, split)
        ya = np.stack([y[n2i[g]] for g in games])
        Xa = np.stack([X_game[n2i[g]] for g in games])
        sub = tve.micro_prf(ya, a_ridge.predict(a_scaler.transform(Xa)), a_th)
        af = article_feats(mode)
        res = {"anchor_subset_f1": sub["micro_f1"]}
        drops = []
        for v in VARIANTS:
            Xv = np.stack([af[(g, v)] for g in games])
            m = tve.micro_prf(ya, a_ridge.predict(a_scaler.transform(Xv)), a_th)
            res[v] = m["micro_f1"]
            drops.append(sub["micro_f1"] - m["micro_f1"])
        res["mean_drop"] = float(np.mean(drops))
        return res

    X_filt = None
    if args.filtered_x:
        X_filt = np.load(args.filtered_x)["X"].astype(np.float32)

    results = {}
    results["raw_sent_mean"] = run_mode("sent_mean", X_raw)
    results["raw_whole_text"] = run_mode("whole_text", X_raw)
    if keep_fn is not None and X_filt is not None:
        results["filt_sent_mean"] = run_mode("filt_mean", X_filt)
        results["filt_whole_text"] = run_mode("filt_whole", X_filt)

    tve.atomic_json_write({"n_games": len(games), "results": results},
                          out_root / "realtext_raw_baselines.json")
    print(f"\nRAW-EMBEDDING BASELINES  (tag ridge, {len(games)} article games)")
    print(f"{'mode':18} {'anchor_sub':>10} {'neutral':>8} {'positive':>8} "
          f"{'negative':>8} {'noname':>8} {'drop':>7}")
    for k, r in results.items():
        print(f"{k:18} {r['anchor_subset_f1']:10.3f} "
              + " ".join(f"{r[v]:8.3f}" for v in VARIANTS) + f" {r['mean_drop']:7.3f}")


if __name__ == "__main__":
    main()
