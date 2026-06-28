"""PXI probe: can the frozen VICReg code predict player-experience dimensions?

A second evaluation axis beyond the Steam-tag probe ("not just TAG"). It reuses
the SAME standard as the tag-F1 probe:

  1. baseline first  -> a chance baseline AND the raw 1024-d Qwen
     embedding (the information ceiling, like ceiling_diagnostic.py);
  2. then the frozen VICReg code, reported as retention = vicreg / raw;
  3. the task matches the pseudo_text setup: text -> averaged PXI score value.
     We report regression metrics (Pearson r, R^2, MAE).

It compares two dimension GROUPS:
  functional    progress_feedback, ease_of_control, audiovisual_appeal,
                goals_and_rules, challenge          (closer to design / mechanics)
  psychological meaning, mastery, curiosity, autonomy, immersion
                                                     (subjective felt experience)

Hypothesis tie-in: if the encoder keeps mechanics/content and drops subjective
signal, functional dims should retain better than psychological ones.

** Hard caveat: N = 21 overlapping games. This is a FRAMEWORK + feasibility test,
not a powered result. Treat every number as indicative only. Build the game pool
up (game_review_data/build.py over more PXI games) for a real measurement. **

Run PXIbench_test/pxi_vicreg_overlap.py first.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OVERLAP = SCRIPT_DIR / "pxi_vicreg_overlap.json"
RAW_CACHE = ROOT / "VICReg_review" / "tags" / "raw_mean_features.npz"
DEFAULT_VIC = ROOT / "VICReg_review" / "tags" / "probe_feat_vicreg_adv10_best_fv4_sf0.6.npz"

FUNCTIONAL = ["functional_progress_feedback", "functional_ease_of_control",
              "functional_audiovisual_appeal", "functional_goals_and_rules", "functional_challenge"]
PSYCHOLOGICAL = ["psychological_meaning", "psychological_mastery", "psychological_curiosity",
                 "psychological_autonomy", "psychological_immersion"]


def load_features(path, pool):
    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d["names"]]
    f = d["feats"]
    if f.ndim == 3:
        if pool == "flatten":
            f = f.reshape(f.shape[0], -1)
        elif pool == "stats":
            f = np.concatenate([f.mean(1), f.std(1)], axis=1)
        else:
            f = f.mean(1)
    return {names[i].split("_")[0]: f[i] for i in range(len(names))}


def _reduce(Xtr, Xte, pca):
    """Standardize, then optional PCA fit on TRAIN only (proper inside-CV reduction).
    With features >> samples (raw 1024-d, N=21) PCA is what makes the ceiling a
    real ceiling instead of pure overfit noise."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if pca and pca < Xtr.shape[1]:
        p = PCA(n_components=min(pca, Xtr.shape[0] - 1)).fit(Xtr)
        Xtr, Xte = p.transform(Xtr), p.transform(Xte)
    return Xtr, Xte


def loo_regress(X, y, alpha, pca):
    from sklearn.linear_model import Ridge
    n = len(y)
    pred = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Xtr, Xte = _reduce(X[tr], X[i:i + 1], pca)
        pred[i] = Ridge(alpha=alpha).fit(Xtr, y[tr]).predict(Xte)[0]
    r = 0.0 if np.std(pred) < 1e-9 else float(np.corrcoef(y, pred)[0, 1])
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = float(np.abs(y - pred).mean())
    return r, r2, mae


def eval_rep(name, feat_map, appids, Y, dims, alpha, pca, rng=None):
    """Returns per-dim regression metrics for text -> averaged PXI score."""
    X = np.array([feat_map[a] for a in appids])
    per = {}
    for k, d in enumerate(dims):
        y = Y[:, k].copy()
        if rng is not None:           # chance: shuffle labels
            y = rng.permutation(y)
        r, r2, mae = loo_regress(X, y, alpha, pca)
        per[d] = {"pearson": r, "r2": r2, "mae": mae}
    return per


def group_mean(per, group, key):
    vals = [per[d][key] for d in group]
    return float(np.mean(vals))


def fit_export_head(feat_map, appids, Y, dims, args, out_path):
    """Fit the final deployable PXI head on all overlap games and save arrays.

    Validation uses the same stats-pooled VICReg feature as pxi_probe. The head
    stores a Ridge regressor for continuous averaged PXI scores.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X = np.array([feat_map[a] for a in appids], dtype=np.float32)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    pca = None
    Xp = Xs
    if args.pca and args.pca < Xs.shape[1]:
        pca = PCA(n_components=min(args.pca, Xs.shape[0] - 1)).fit(Xs)
        Xp = pca.transform(Xs)

    ridge_coef = np.zeros((len(dims), Xp.shape[1]), dtype=np.float32)
    ridge_intercept = np.zeros(len(dims), dtype=np.float32)
    target_mean = Y.mean(axis=0).astype(np.float32)
    target_std = Y.std(axis=0).astype(np.float32)
    target_min = Y.min(axis=0).astype(np.float32)
    target_max = Y.max(axis=0).astype(np.float32)

    for k, dim in enumerate(dims):
        y = Y[:, k].astype(np.float32)
        ridge = Ridge(alpha=args.vic_alpha).fit(Xp, y)
        ridge_coef[k] = ridge.coef_.astype(np.float32)
        ridge_intercept[k] = float(ridge.intercept_)

    payload = {
        "kind": "linear_pxi_probe",
        "target_kind": "mean_regression",
        "dims": list(dims),
        "functional_dims": list(FUNCTIONAL),
        "psychological_dims": list(PSYCHOLOGICAL),
        "appid_order": list(appids),
        "pool": args.vic_pool,
        "feature_dim": int(X.shape[1]),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "pca_components": None if pca is None else pca.components_.astype(np.float32),
        "pca_mean": None if pca is None else pca.mean_.astype(np.float32),
        "ridge_coef": ridge_coef,
        "ridge_intercept": ridge_intercept,
        "target_mean": target_mean,
        "target_std": target_std,
        "target_min": target_min,
        "target_max": target_max,
        "alpha": float(args.vic_alpha),
        "pca": int(args.pca),
        "n_games": int(len(appids)),
        "overlap_json": str(Path(OVERLAP).resolve()),
        "caveat": "N is tiny; use as a feasibility probe, not a powered PXI result.",
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"exported deployable PXI probe -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vic-cache", default=str(DEFAULT_VIC))
    ap.add_argument("--raw-cache", default=str(RAW_CACHE))
    ap.add_argument("--vic-pool", choices=["stats", "flatten", "mean"], default="stats",
                    help="stats(36-d) is safest at N=21; flatten matches the F1 probe but overfits here.")
    ap.add_argument("--raw-alpha", type=float, default=100.0)
    ap.add_argument("--vic-alpha", type=float, default=10.0)
    ap.add_argument("--C", type=float, default=1.0, help=argparse.SUPPRESS)
    ap.add_argument("--pca", type=int, default=8,
                    help="PCA components (fit inside each LOO fold) so features>>samples "
                         "doesn't make the baseline pure overfit. 0 disables.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-json", default=str(SCRIPT_DIR / "pxi_probe_report.json"))
    ap.add_argument("--export-head", default=None,
                    help="Save a deployable .pt PXI probe fit on all overlapping games.")
    args = ap.parse_args()

    ov = json.loads(Path(OVERLAP).read_text(encoding="utf-8"))
    dims = ov["dims"]
    matches = ov["matches"]
    appid2pxi = {m["appid"]: [m["pxi"][d] for d in dims] for m in matches}

    raw = load_features(args.raw_cache, "flatten")
    vic = load_features(args.vic_cache, args.vic_pool)
    appids = [a for a in appid2pxi if a in raw and a in vic]
    Y = np.array([appid2pxi[a] for a in appids], dtype=float)
    di = {d: k for k, d in enumerate(dims)}
    func = [di[d] for d in FUNCTIONAL]
    psyc = [di[d] for d in PSYCHOLOGICAL]
    print(f"PXI probe: N={len(appids)} overlapping games, {len(dims)} dims "
          f"(functional={len(FUNCTIONAL)}, psychological={len(PSYCHOLOGICAL)})")
    print(f"vic_pool={args.vic_pool} (dim {vic[appids[0]].shape[0]})  raw dim {raw[appids[0]].shape[0]}")
    print("=" * 72)

    rng = np.random.default_rng(args.seed)
    chance = eval_rep("chance", raw, appids, Y, dims, args.raw_alpha, args.pca, rng=rng)
    raw_per = eval_rep("raw", raw, appids, Y, dims, args.raw_alpha, args.pca)
    vic_per = eval_rep("vicreg", vic, appids, Y, dims, args.vic_alpha, args.pca)

    def block(per):
        return {
            "functional": {"pearson": group_mean(per, FUNCTIONAL, "pearson"),
                           "r2": group_mean(per, FUNCTIONAL, "r2"),
                           "mae": group_mean(per, FUNCTIONAL, "mae")},
            "psychological": {"pearson": group_mean(per, PSYCHOLOGICAL, "pearson"),
                              "r2": group_mean(per, PSYCHOLOGICAL, "r2"),
                              "mae": group_mean(per, PSYCHOLOGICAL, "mae")},
        }

    cb, rb, vb = block(chance), block(raw_per), block(vic_per)

    print(f"{'GROUP / metric':28s} {'chance':>9s} {'raw(ceil)':>10s} {'vicreg':>9s} {'retention':>10s}")
    for g in ("functional", "psychological"):
        for metric in ("pearson", "r2", "mae"):
            ret = (vb[g][metric] / rb[g][metric]) if abs(rb[g][metric]) > 1e-6 else float("nan")
            print(f"{g+'  '+metric:28s} {cb[g][metric]:9.3f} {rb[g][metric]:10.3f} "
                  f"{vb[g][metric]:9.3f} {ret:10.3f}")
        print("-" * 72)

    print("\nper-dimension (vicreg): pearson / mae")
    for grp, cols in (("functional", FUNCTIONAL), ("psychological", PSYCHOLOGICAL)):
        for d in cols:
            print(f"  [{grp[:4]}] {d:34s} r={vic_per[d]['pearson']:+.3f}  mae={vic_per[d]['mae']:.3f}")

    verdict_r = vb["functional"]["pearson"] - vb["psychological"]["pearson"]
    verdict_mae = vb["functional"]["mae"] - vb["psychological"]["mae"]
    print("\n" + "=" * 72)
    print(f"functional − psychological:  Δpearson={verdict_r:+.3f}  ΔMAE={verdict_mae:+.3f}")
    print("NOTE: N=21 — framework/feasibility only, not a powered result.")

    report = {
        "n_games": len(appids), "vic_cache": str(Path(args.vic_cache).resolve()),
        "vic_pool": args.vic_pool, "dims": dims,
        "chance": cb, "raw_ceiling": rb, "vicreg": vb,
        "per_dim_vicreg": vic_per, "per_dim_raw": raw_per,
        "functional_minus_psychological": {"pearson": verdict_r, "mae": verdict_mae},
        "caveat": "N=21 overlapping games; feasibility framework only.",
    }
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.report_json}")
    if args.export_head:
        fit_export_head(vic, appids, Y, dims, args, args.export_head)


if __name__ == "__main__":
    main()
