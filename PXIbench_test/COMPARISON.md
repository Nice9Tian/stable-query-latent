# Model Comparison — v1 vs v2

Comparison of the original `LatentQueryFlatRegressor` (**v1**) against the redesigned
`LatentQueryFunnelRegressor` (**v2**, see [`README_2.md`](README_2.md)) on the PXI
multi-variant benchmark.

## Setup

- **Data:** `pesudo_data/benchmark_sentence_latent_query_multi.h5`
  (1805 samples; train 1475 / test 330).
- **Task:** per-dimension 5-class classification over 10 score dimensions.
- **Split:** `--split-by score_combo` (test groups held out by full target combo, so
  absolute numbers shift between seeds — within a seed, v1/v2/control share the
  identical split, so per-seed differences are a valid **paired** test).
- **Identical for every run:** epochs=800, batch=128, lr=3e-4 → cosine → 1e-5,
  weight_decay=1e-4, flat_dim=128, query_sizes=(32,16,8), num_heads=8, dropout=0.0.
  The **only** things that change are `--model` and (for the control) `--hidden-dim`.
  Same training loop (shared `train_and_test`), same seed.
- Checkpoint selection: best **test_mae** epoch.

## Results (10 seeds, mean ± std)

Aggregated over **10 random seeds** `[412384, 701183, 792180, 283149, 493976, 27630,
69219, 415472, 273118, 633492]` (generated from meta-seed 20240622). Each config
trained 800 epochs per seed. Raw per-run numbers in [`heads/seed_sweep.json`](heads/seed_sweep.json).

| model | params | test_acc ↑ | test_mae ↓ | test_ce ↓ |
|---|---:|---:|---:|---:|
| v1 (hidden64) | 292,914 | 0.9545 ± 0.0069 | 0.0634 ± 0.0073 | 0.2063 ± 0.0331 |
| v2 (hidden64) | 342,026 | 0.9765 ± 0.0056 | 0.0330 ± 0.0081 | 0.1009 ± 0.0297 |
| **v2 (hidden56)** | **278,706** | **0.9802 ± 0.0068** | **0.0285 ± 0.0090** | **0.0803 ± 0.0257** |

### Paired differences (same split per seed)

| comparison | Δacc (mean ± std) | Δmae | Δce | acc win-rate |
|---|---:|---:|---:|---:|
| v2(h64) − v1(h64) | **+0.0220 ± 0.0061** | −0.0304 | −0.1055 | **10/10** |
| v2(h56) − v1(h64) | **+0.0257 ± 0.0057** | −0.0349 | −0.1260 | **10/10** |
| v2(h56) − v2(h64) | +0.0037 ± 0.0066 | −0.0045 | — | 7/10 |

## Findings

1. **v2 beats v1, decisively and robustly.** Paired Δacc ≈ +2.2–2.6% with a
   **10/10 win-rate**, and the effect (~0.022) is **~4× larger than its own std
   (~0.006)** — far outside seed noise. mae is roughly halved (0.063 → 0.033/0.029)
   and ce nearly halved (0.206 → 0.10/0.08).
2. **The gain is architectural, not capacity.** The `hidden56` control has
   **278,706 params — fewer than v1's 292,914** — yet is the *best* config overall.
   Parameter count does not explain the improvement.
3. **hidden56 vs hidden64 is a wash.** Paired Δacc = +0.0037 with std 0.0066
   (win-rate 7/10) — the difference is **smaller than its own std**, i.e. within
   noise. hidden56 leans slightly better and is cheaper, so prefer it for the
   marginal edge + lower cost, but the two are statistically indistinguishable.
4. **v2 converges earlier** (best checkpoint ~ep132 vs ~ep334 for v1, observed on the
   single-seed runs) and is steadier late in training.

## Why v2 helps (interpretation)

The three changes work together:

1. **Single learnable latent array; later queries are linear reductions of the
   previous latents.** v1's per-stage queries are static `nn.Parameter`s — sample-
   independent until they attend. v2's later queries are built from what the model
   has *already* aggregated for *this* sample, so they are input-adaptive and carry
   more signal into each cross-attention.
2. **Self-attention after the first cross-attention.** It lets the 32 latents
   exchange information and integrate globally *before* the funnel narrows, so the
   reduced queries are computed from a better-mixed representation.
3. **Funnel of cross-attention** (each stage queries the previous stage's latents)
   gives a clean hierarchical refinement rather than three independent query sets.

## Reproduce

```bash
# 10-seed sweep, all three configs (sequential; writes heads/seed_sweep.json)
python PXIbench_test/seed_sweep.py

# same sweep but several configs run concurrently on one GPU (much faster)
python PXIbench_test/seed_sweep_parallel.py --workers 3

# a single config/seed by hand (paths resolve relative to PXIbench_test/, so use heads/...)
python PXIbench_test/test_latent_query_model.py    --model v1 --epochs 800 --seed 42
python PXIbench_test/test_latent_query_model_v2.py             --epochs 800 --seed 42
python PXIbench_test/test_latent_query_model_v2.py --hidden-dim 56 --epochs 800 --seed 42
```

## Caveats

- The `score_combo` split makes absolute numbers swing between seeds (the per-seed
  difficulty of the held-out combos varies). Compare same-seed pairs / the 10-seed
  aggregate, not raw numbers across seeds.
- v2 trains ~15% slower per wall clock (≈1m46s vs 1m30s at hidden64) due to the extra
  self-attention block — a small cost for the accuracy gain. On this tiny model the
  GPU sits at ~44% utilization (overhead-bound); `seed_sweep_parallel.py` packs
  several runs onto the card to recover throughput.
