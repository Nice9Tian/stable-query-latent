"""Run v1 vs v2 (and a param-matched v2) across 10 random seeds and aggregate.

For each seed all three configs share the SAME train/test split (the split depends
only on seed + split_by, not the model), so v1 vs v2 is a paired comparison. We
report per-config mean +/- std and the paired (v2 - v1) differences.

Writes heads/seed_sweep.json with every run plus the aggregates.
"""

import contextlib
import io
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))  # so `import test_latent_query_model` works

import test_latent_query_model as T  # noqa: E402

H5 = str(SCRIPT_DIR / "pesudo_data" / "benchmark_sentence_latent_query_multi.h5")
META_SEED = 20240622
NUM_SEEDS = 10
EPOCHS = 800

# (model_name, hidden_dim)
CONFIGS = [("v1", 64), ("v2", 64), ("v2", 56)]


def run_one(model_name, hidden_dim, seed):
    """Train one config/seed silently; return best-checkpoint metrics."""
    with contextlib.redirect_stdout(io.StringIO()):
        metrics = T.train_and_test(
            h5_path=H5,
            epochs=EPOCHS,
            batch_size=128,
            learning_rate=3e-4,
            min_learning_rate=1e-5,
            test_ratio=0.2,
            seed=seed,
            hidden_dim=hidden_dim,
            flat_dim=128,
            query_sizes=(32, 16, 8),
            num_heads=8,
            dropout=0.0,
            device_name=None,
            model_out=None,
            history_txt=None,
            per_dim_txt=None,
            preload_data=True,
            split_by="score_combo",
            model_name=model_name,
        )
    return {
        "acc": metrics["test_accuracy"],
        "mae": metrics["test_mae"],
        "ce": metrics["test_ce"],
    }


def key_of(model_name, hidden_dim):
    return f"{model_name}_h{hidden_dim}"


def main():
    seeds = [int(s) for s in np.random.default_rng(META_SEED).integers(1, 1_000_000, NUM_SEEDS)]
    print(f"seeds ({NUM_SEEDS}, meta={META_SEED}): {seeds}", flush=True)

    results = {key_of(m, h): [] for m, h in CONFIGS}
    started = time.time()
    for seed in seeds:
        for model_name, hidden_dim in CONFIGS:
            metrics = run_one(model_name, hidden_dim, seed)
            results[key_of(model_name, hidden_dim)].append({"seed": seed, **metrics})
            print(
                f"[{time.time() - started:6.0f}s] {key_of(model_name, hidden_dim):7s} "
                f"seed={seed:6d}  acc={metrics['acc']:.4f}  mae={metrics['mae']:.4f}  ce={metrics['ce']:.4f}",
                flush=True,
            )

    # Per-config aggregates.
    summary = {}
    for key, runs in results.items():
        summary[key] = {}
        for metric in ("acc", "mae", "ce"):
            vals = [r[metric] for r in runs]
            summary[key][metric] = {
                "mean": st.mean(vals),
                "std": st.stdev(vals),
                "min": min(vals),
                "max": max(vals),
            }

    # Paired (v2 - v1) differences, seed by seed (same split per seed).
    paired = {}
    v1 = {r["seed"]: r for r in results["v1_h64"]}
    for v2_key in ("v2_h64", "v2_h56"):
        v2 = {r["seed"]: r for r in results[v2_key]}
        diffs = {metric: [] for metric in ("acc", "mae", "ce")}
        wins = 0
        for seed in v1:
            for metric in ("acc", "mae", "ce"):
                diffs[metric].append(v2[seed][metric] - v1[seed][metric])
            if v2[seed]["acc"] > v1[seed]["acc"]:
                wins += 1
        paired[f"{v2_key}_minus_v1_h64"] = {
            "acc_diff_mean": st.mean(diffs["acc"]),
            "acc_diff_std": st.stdev(diffs["acc"]),
            "mae_diff_mean": st.mean(diffs["mae"]),
            "ce_diff_mean": st.mean(diffs["ce"]),
            "acc_win_count": wins,
            "n": len(v1),
        }

    out = {
        "meta_seed": META_SEED,
        "seeds": seeds,
        "epochs": EPOCHS,
        "configs": [key_of(m, h) for m, h in CONFIGS],
        "results": results,
        "summary": summary,
        "paired": paired,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    out_path = SCRIPT_DIR / "heads" / "seed_sweep.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n===== SUMMARY (mean +/- std over %d seeds) =====" % NUM_SEEDS)
    for key in results:
        s = summary[key]
        print(f"{key:7s}  acc={s['acc']['mean']:.4f}+/-{s['acc']['std']:.4f}  "
              f"mae={s['mae']['mean']:.4f}+/-{s['mae']['std']:.4f}  "
              f"ce={s['ce']['mean']:.4f}+/-{s['ce']['std']:.4f}")
    print("\n===== PAIRED (v2 - v1_h64), per-seed =====")
    for k, p in paired.items():
        print(f"{k}: acc +{p['acc_diff_mean']:.4f}+/-{p['acc_diff_std']:.4f}  "
              f"mae {p['mae_diff_mean']:+.4f}  ce {p['ce_diff_mean']:+.4f}  "
              f"acc_wins={p['acc_win_count']}/{p['n']}")
    print(f"\nwrote {out_path}  (elapsed {out['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
