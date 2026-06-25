# PXI probe — a second evaluation axis beyond tag-F1

**Goal.** Test whether the frozen VICReg code predicts *player-experience*
dimensions (PXI), not just Steam tags. Compare two dimension groups —
**functional** (≈ design/mechanics) vs **psychological** (≈ subjective felt
experience) — on the *same standard as the tag-F1 probe*.

`pxi_probe.py` (run `pxi_vicreg_overlap.py` first).

## 2026-06-25 final-checkpoint refresh

The current final encoder is
`VICReg_review/heads/hierarchical64_align_reco/vicreg_review_h5_latest.pt`.
A live dual probe on the 21 PXI-overlap games reports:

- `pxi_func_f1 = 0.486`
- `pxi_psych_f1 = 0.569`
- `pxi_delta = -0.082`
- `tag_content_f1 = 0.614`
- `code_sentiment_r2 = -0.009`

So PXI is still underpowered at `N = 21`. The historical raw-vs-vicreg table
below is kept for the earlier `centroid64_grl` checkpoint.

## Same discipline as the tag probe

1. **Baseline first**, two of them:
   - **chance**: shuffled labels (calibrates Pearson ≈ 0, R² ≈ 0).
   - **raw 1024-d Qwen embedding**: the information ceiling (as in
     `ceiling_diagnostic.py`).
2. **then VICReg code**, compared against the raw ceiling.
3. Regression metrics are Pearson r, R², and MAE for each PXI dimension group.
4. Leave-one-out CV (N is tiny), PCA fit inside each fold so a 1024-d ceiling at
   N=21 isn't pure overfit.

## Historical PXI-only result (N = 21, centroid64_grl)

| group / metric | chance | raw (ceiling) | vicreg |
|---|---:|---:|---:|
| functional Pearson | -0.049 | -0.231 | -0.269 |
| functional R² | -0.193 | -0.328 | -0.501 |
| functional MAE | 0.681 | 0.716 | 0.776 |
| psychological Pearson | -0.103 | 0.003 | -0.093 |
| psychological R² | -0.184 | -0.220 | -0.366 |
| psychological MAE | 0.752 | 0.744 | 0.797 |

functional - psychological on the final VICReg code:
**ΔPearson = -0.176**, **ΔMAE = -0.021**.

## Verdict: framework works, but N=21 is underpowered

**The headline is the caveat.** The raw-embedding **ceiling itself is not
learnable** at N=21: raw Pearson is negative for functional dimensions and near
zero for psychological dimensions, with negative R² for both groups. When the
ceiling cannot fit, VICReg-vs-raw retention is not meaningful.

So: **no reliable PXI signal at N=21.** The final centroid64_grl checkpoint does
not produce a powered PXI result; this page should be reported as a framework
check, not as evidence that PXI prediction works from the current Steam-review
set.

## What this delivers, and how to make it real

- A **complete, tested framework** that drops straight onto a bigger game set:
  same script, no changes, just more rows in `pxi_vicreg_overlap.json`.
- **The fix is data, not code.** The PXI benchmark has 361 games, most of them
  famous with abundant Steam reviews; only 21 are in the current 293-game training
  set. Pull + embed reviews for more PXI games (`game_review_data/build.py`) to get
  N up to ~300, then this probe gives a powered functional-vs-psychological answer.

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe PXIbench_test/pxi_vicreg_overlap.py
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe PXIbench_test/pxi_probe.py `
  --vic-cache VICReg_review/heads/centroid64_grl/probe_feat_vicreg_review_h5_latest_fv4_sf0.6.npz `
  --vic-pool stats `
  --report-json PXIbench_test/pxi_probe_centroid64_grl_report.json
```
