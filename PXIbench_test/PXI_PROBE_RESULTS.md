# PXI probe — a second evaluation axis beyond tag-F1

**Goal.** Test whether the frozen VICReg code predicts *player-experience*
dimensions (PXI), not just Steam tags. Compare two dimension groups —
**functional** (≈ design/mechanics) vs **psychological** (≈ subjective felt
experience) — on the *same standard as the tag-F1 probe*.

`pxi_probe.py` (run `pxi_vicreg_overlap.py` first).

## Same standard as F1

1. **Baseline first**, two of them:
   - **chance**: shuffled labels (calibrates F1 ≈ base rate, R² ≈ 0).
   - **raw 1024-d Qwen embedding**: the information ceiling (as in
     `ceiling_diagnostic.py`).
2. **then VICReg code**, reported as **retention = vicreg / raw**.
3. **F1 view**: each PXI dim is split at its median (high/low) → micro-F1, exactly
   like the tag probe. Plus regression metrics (Pearson r, R²).
4. Leave-one-out CV (N is tiny), PCA fit inside each fold so a 1024-d ceiling at
   N=21 isn't pure overfit.

## Result (N = 21 overlapping games)

| group / metric | chance | raw (ceiling) | vicreg | retention |
|---|---|---|---|---|
| functional F1 | 0.56 | 0.48 | 0.60 | 1.26 |
| functional Pearson | −0.05 | −0.23 | −0.20 | — |
| psychological F1 | 0.47 | 0.53 | 0.55 | 1.04 |
| psychological Pearson | −0.10 | 0.00 | 0.06 | — |

functional − psychological: **ΔF1 = +0.05**, Δpearson = −0.26.

## Verdict: framework works, but N=21 is underpowered

**The headline is the caveat.** The raw-embedding **ceiling itself sits at chance**
(functional raw F1 0.48 < chance 0.56; every R² negative). When the ceiling can't
fit, nothing downstream is measurable — the VICReg-vs-raw retention numbers are
noise (one Pearson retention blew up to ×19 because the denominator ≈ 0).

So: **no reliable PXI signal at N=21.** There is a *faint* hint that functional
(mechanics-ish) dims are retained slightly better than psychological ones
(ΔF1 +0.05), which is the direction the "keep mechanics, drop subjective"
hypothesis predicts — but it is well within noise and must not be reported as a
result.

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
  --vic-cache VICReg_review/tags/probe_feat_vicreg_adv10_best_fv4_sf0.6.npz
```
