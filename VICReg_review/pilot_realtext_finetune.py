"""Frozen-backbone linear fine-tune on the generated real-text articles.

Splits the article GAMES (~258) into train/test (70/30, fixed seed; split by
game so no game leaks across). Backbone stays frozen; two linear heads are
fit on the train games' article features and evaluated on the held-out games:

  zero-shot : the existing anchor-ridge probe (trained on review features)
              applied to article features unchanged -- the pilot baseline.
  head      : a fresh ridge tag head trained directly ON article features
              (all four variants of the train games).
  adapter   : ridge linear map article-feature -> the game's anchor feature,
              then the UNTOUCHED anchor probe scores the adapted feature.
              The adapter also plugs into retrieval (adapted query vs the
              mean-anchor gallery).

Per combo writes real_text_finetune_pilot.json; a summary table compares
zero-shot / head / adapter F1 (test games, mean over variants) plus retrieval
median rank & hit@1 before/after adaptation (neutral variant).

    python VICReg_review/pilot_realtext_finetune.py --out-dir <local out_dir> \
        [--combos "extra1,extra2"] [--train-frac 0.7] [--seed 20260705]
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
OUT_NAME = "real_text_finetune_pilot.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", default=str(ROOT / "game_review_data/embedding_h5.h5"))
    p.add_argument("--out-dir", default=str(ROOT / "VICReg_review/heads/cloud_full_sweep_a100"))
    p.add_argument("--cache", default=None)
    p.add_argument("--combos", default="")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=20260705)
    p.add_argument("--train-sizes", default="",
                   help="label-efficiency curve: comma-separated train-set sizes "
                        "(games), e.g. '20,40,80,181'. Test set stays the same "
                        "30%% holdout; leftover train-pool games act as the val "
                        "fold for alpha selection. Adapter only.")
    p.add_argument("--curve-seeds", type=int, default=3,
                   help="resamples per size for the curve (mean/std reported)")
    args = p.parse_args()

    import h5py
    import numpy as np
    import torch
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from VICReg_review import disturbtion_embed
    from VICReg_review import eval_battery_worker as worker
    from VICReg_review import run_data_view_sweep as sweep
    from VICReg_review import text_variant_eval as tve
    from VICReg_review.identity_diagnostic import l2_normalize

    out_root = Path(args.out_dir)
    cache_path = Path(args.cache or out_root / "text_variant_pilot_cache.npz")
    if not cache_path.exists():
        raise SystemExit(f"pilot cache missing: {cache_path} -- run pilot_realtext_tag.py first")
    combos = [row["combo_id"] for row in json.loads(
        (out_root / "champions.json").read_text(encoding="utf-8"))["champions"]]
    combos += [c.strip() for c in args.combos.split(",")
               if c.strip() and c.strip() not in combos]

    ev = worker.build_eval_args(argparse.Namespace(h5=args.h5, out_dir=str(out_root)))
    ev.out_dir = out_root
    ev.text_variant_cache = cache_path
    cache = disturbtion_embed.load_npz_payload(cache_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(ev.h5, "r") as h5:
        input_dim = int(h5.attrs["input_dim"])

    def fit_ridge(Xtr, ytr, Xva, yva):
        """alpha + threshold picked on the val fold, mirroring train_anchor_ridge."""
        scaler = StandardScaler().fit(Xtr)
        best = None
        for alpha in tve.DEFAULT_ALPHAS:
            m = Ridge(alpha=float(alpha)).fit(scaler.transform(Xtr), ytr.astype(np.float32))
            sv = m.predict(scaler.transform(Xva))
            for th in tve.threshold_grid(sv, 33):
                f1 = tve.micro_prf(yva, sv, float(th))["micro_f1"]
                if best is None or f1 > best[0]:
                    best = (f1, m, scaler, float(alpha), float(th))
        return best[1], best[2], best[3], best[4]

    rows = []
    for cid in combos:
        cdir = out_root / cid
        ready, ckpt = worker.combo_ready(cdir)
        if not ready:
            print(f"  {cid}: NOT ready -- skipped")
            continue
        feats, names = sweep.build_vicreg_feature_cache(ev, ckpt, cdir)
        anchor = np.asarray(feats).mean(axis=1).astype(np.float32)   # (games, dim)
        name_to_index = {n: i for i, n in enumerate(names)}
        y, _tags = tve.align_labels(Path(ev.h5), names)
        split = tve.make_or_load_split(
            Path(getattr(ev, "tag_text_split_json", "") or out_root / "tag_text_eval_split.json"),
            names, ev)
        a_scaler, a_ridge, _a, a_th, _m = tve.train_anchor_ridge(ev, anchor, y, name_to_index, split)

        encoder, _, _, _ = sweep.load_frozen_encoder(ckpt, input_dim, device)
        vfeat = tve.encode_variant_features(ev, encoder, cache, device)
        art_games = sorted({n for (n, v) in vfeat
                            if n in name_to_index and all((n, w) in vfeat for w in VARIANTS)})
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(art_games))
        n_tr = int(round(len(art_games) * args.train_frac))
        tr_games = [art_games[i] for i in order[:n_tr]]
        te_games = [art_games[i] for i in order[n_tr:]]
        n_fit = int(round(len(tr_games) * 0.8))          # inner fit/val split
        fit_g, val_g = tr_games[:n_fit], tr_games[n_fit:]

        def stack(games):
            X = np.stack([vfeat[(g, v)] for g in games for v in VARIANTS]).astype(np.float32)
            yy = np.stack([y[name_to_index[g]] for g in games for _ in VARIANTS])
            return X, yy

        if args.train_sizes:
            # ---- label-efficiency curve: adapter only, fixed test set
            gal = l2_normalize(anchor)
            def ret_median(Xq):
                ranks = []
                for g, q in zip(te_games, l2_normalize(np.asarray(Xq, dtype=np.float32))):
                    sims = gal @ q
                    ranks.append(int((sims > sims[name_to_index[g]]).sum()) + 1)
                return float(np.median(ranks))

            Xte_by_v = {v: np.stack([vfeat[(g, v)] for g in te_games]).astype(np.float32)
                        for v in VARIANTS}
            yte = np.stack([y[name_to_index[g]] for g in te_games])
            zs_f1 = float(np.mean([tve.micro_prf(
                yte, a_ridge.predict(a_scaler.transform(Xte_by_v[v])), a_th)["micro_f1"]
                for v in VARIANTS]))
            zs_med = ret_median(Xte_by_v["neutral"])

            CURVE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)
            sizes = [int(s) for s in args.train_sizes.split(",") if s.strip()]
            curve = []
            for size in sizes:
                size = min(size, len(tr_games))
                f1s, meds = [], []
                for s in range(args.curve_seeds):
                    srng = np.random.default_rng(args.seed + 1000 * s + size)
                    sub = list(srng.permutation(tr_games)[:size])
                    val = [g for g in tr_games if g not in sub][:60] or sub
                    Xs, _ = stack(sub)
                    As = np.stack([anchor[name_to_index[g]] for g in sub for _ in VARIANTS])
                    Xv, yv = stack(val)
                    best = None
                    for alpha in CURVE_ALPHAS:
                        sc = StandardScaler().fit(Xs)
                        ad = Ridge(alpha=float(alpha)).fit(sc.transform(Xs), As)
                        sv = a_ridge.predict(a_scaler.transform(ad.predict(sc.transform(Xv))))
                        f1 = tve.micro_prf(yv, sv, a_th)["micro_f1"]
                        if best is None or f1 > best[0]:
                            best = (f1, ad, sc)
                    _bf, ad, sc = best
                    f1s.append(float(np.mean([tve.micro_prf(
                        yte, a_ridge.predict(a_scaler.transform(
                            ad.predict(sc.transform(Xte_by_v[v])))), a_th)["micro_f1"]
                        for v in VARIANTS])))
                    meds.append(ret_median(ad.predict(sc.transform(Xte_by_v["neutral"]))))
                curve.append({"size": size,
                              "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                              "ret_median_mean": float(np.mean(meds))})
            report = {"combo_id": cid, "seed": args.seed, "n_test_games": len(te_games),
                      "zero_shot": {"f1": zs_f1, "ret_median": zs_med}, "curve": curve}
            tve.atomic_json_write(report, cdir / "real_text_labelcurve.json")
            rows.append(report)
            del feats, encoder, vfeat
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  {cid}: curve done (test={len(te_games)}) "
                  + " ".join(f"n{c['size']}={c['f1_mean']:.3f}" for c in curve))
            continue

        X_fit, y_fit = stack(fit_g)
        X_val, y_val = stack(val_g)

        # ---- head A: direct article tag head
        h_ridge, h_scaler, h_alpha, h_th = fit_ridge(X_fit, y_fit, X_val, y_val)

        # ---- head B: linear adapter article-feat -> anchor-feat (probe untouched)
        A_fit = np.stack([anchor[name_to_index[g]] for g in fit_g for _ in VARIANTS])
        ad_best = None
        for alpha in tve.DEFAULT_ALPHAS:
            ad_s = StandardScaler().fit(X_fit)
            ad = Ridge(alpha=float(alpha)).fit(ad_s.transform(X_fit), A_fit)
            sv = a_ridge.predict(a_scaler.transform(ad.predict(ad_s.transform(X_val))))
            f1 = tve.micro_prf(y_val, sv, a_th)["micro_f1"]
            if ad_best is None or f1 > ad_best[0]:
                ad_best = (f1, ad, ad_s, float(alpha))
        _f1, adapter, ad_scaler, ad_alpha = ad_best
        adapt = lambda X: adapter.predict(ad_scaler.transform(X))

        # ---- evaluate on held-out games
        per_variant = {}
        for v in VARIANTS:
            Xte = np.stack([vfeat[(g, v)] for g in te_games]).astype(np.float32)
            yte = np.stack([y[name_to_index[g]] for g in te_games])
            per_variant[v] = {
                "zero_shot": tve.micro_prf(yte, a_ridge.predict(a_scaler.transform(Xte)), a_th),
                "head": tve.micro_prf(yte, h_ridge.predict(h_scaler.transform(Xte)), h_th),
                "adapter": tve.micro_prf(yte, a_ridge.predict(a_scaler.transform(adapt(Xte))), a_th),
            }
        Xa_te = np.stack([anchor[name_to_index[g]] for g in te_games])
        y_te = np.stack([y[name_to_index[g]] for g in te_games])
        anchor_ceiling = tve.micro_prf(y_te, a_ridge.predict(a_scaler.transform(Xa_te)), a_th)

        # ---- retrieval on test games (neutral): mean-anchor gallery
        gal = l2_normalize(anchor)
        def ret(Xq):
            ranks = []
            for g, q in zip(te_games, l2_normalize(np.asarray(Xq, dtype=np.float32))):
                sims = gal @ q
                ranks.append(int((sims > sims[name_to_index[g]]).sum()) + 1)
            r = np.asarray(ranks)
            return {"hit_at_1": float((r == 1).mean()), "median_rank": float(np.median(r))}
        Xn_te = np.stack([vfeat[(g, "neutral")] for g in te_games]).astype(np.float32)
        retrieval = {"zero_shot": ret(Xn_te), "adapter": ret(adapt(Xn_te))}

        mean_over = lambda k: float(np.mean([per_variant[v][k]["micro_f1"] for v in VARIANTS]))
        report = {
            "combo_id": cid, "seed": args.seed,
            "n_train_games": len(tr_games), "n_test_games": len(te_games),
            "head_alpha": h_alpha, "adapter_alpha": ad_alpha,
            "per_variant": per_variant, "anchor_ceiling_test": anchor_ceiling,
            "retrieval_neutral_test": retrieval,
            "mean_f1": {k: mean_over(k) for k in ("zero_shot", "head", "adapter")},
        }
        tve.atomic_json_write(report, cdir / OUT_NAME)
        rows.append(report)
        del feats, encoder, vfeat
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {cid}: done (train={len(tr_games)} test={len(te_games)})")

    if args.train_sizes:
        tve.atomic_json_write({"rows": rows}, out_root / "real_text_labelcurve_summary.json")
        sizes = [c["size"] for c in rows[0]["curve"]] if rows else []
        print(f"\nLABEL-EFFICIENCY CURVE  (adapter F1 mean over 4 variants, "
              f"{args.curve_seeds} seeds; ret = neutral median rank)")
        head = f"{'combo_id':40} {'zs':>6} " + " ".join(f"{'n' + str(s):>6}" for s in sizes) \
            + f" {'zs_ret':>7} " + " ".join(f"{'r' + str(s):>5}" for s in sizes)
        print(head)
        print("-" * len(head))
        for r in rows:
            print(f"{r['combo_id']:40} {r['zero_shot']['f1']:6.3f} "
                  + " ".join(f"{c['f1_mean']:6.3f}" for c in r["curve"])
                  + f" {r['zero_shot']['ret_median']:7.0f} "
                  + " ".join(f"{c['ret_median_mean']:5.0f}" for c in r["curve"]))
        return

    tve.atomic_json_write({"rows": rows}, out_root / "real_text_finetune_summary.json")
    print(f"\nLINEAR FINE-TUNE ON REAL TEXT  "
          f"(frozen backbone; F1 = mean over 4 variants on held-out games)")
    head = (f"{'combo_id':40} {'zeroshot':>8} {'head':>8} {'adapter':>8} {'ceiling':>8}"
            f" {'ret_med':>7} {'ret_ad':>7} {'hit1':>6} {'hit1_ad':>7}")
    print(head)
    print("-" * len(head))
    for r in rows:
        rv = r["retrieval_neutral_test"]
        print(f"{r['combo_id']:40} {r['mean_f1']['zero_shot']:8.3f} "
              f"{r['mean_f1']['head']:8.3f} {r['mean_f1']['adapter']:8.3f} "
              f"{r['anchor_ceiling_test']['micro_f1']:8.3f}"
              f" {rv['zero_shot']['median_rank']:7.0f} {rv['adapter']['median_rank']:7.0f}"
              f" {rv['zero_shot']['hit_at_1']:6.3f} {rv['adapter']['hit_at_1']:7.3f}")


if __name__ == "__main__":
    main()
