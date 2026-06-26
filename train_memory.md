# Train Memory

Updated: 2026-06-27 00:00:00 +09:00

## Current Sweep

The full VICReg review sweep is intentionally paused.
The ablation definition was corrected: the no-GRL arm must also disable the
recommendation loss. Older no-GRL results in the existing sweep directory should
be treated as stale for the corrected ablation. The GRL arm can remain valid if
its checkpoint matches the saved manifest/report pair.

Output directory:
`VICReg_review/heads/final_full_sweep_20260626_052814`

The parent sweep scheduler was stopped so no new combinations start. The already-started child training run was allowed to finish before stopping.

## Progress

- `data_view_sweep_summary.csv` currently reflects the old sweep definition and
  should not be treated as the corrected ablation summary.
- The latest completed training-only run is `dim018_grl_n200_view80`.
- `dim018_grl_n200_view80` training manifest:
  `VICReg_review/heads/final_full_sweep_20260626_052814/dim018_grl_n200_view80/vicreg_review_h5_manifest.json`
- `dim018_grl_n200_view80` status: `done`
- `dim018_grl_n200_view80` epoch: `30`
- `dim018_grl_n200_view80` step: `120`
- `dim018_grl_n200_view80` backward mode: `split_recompute`
- `dim018_grl_n200_view80` max view sentences: `0`
- `dim018_grl_n200_view80` used the old recommendation-loss setting; it should
  be re-evaluated together with the corrected no-GRL arm before trusting any
  ablation conclusion.
- `dim018_nogrl_n200_view80` has not been trained yet under the corrected setup.
- No active VICReg sweep/training process was found after pausing.

Important caveat: `sweep_manifest.json` is stale because the scheduler was manually stopped. It still reports `status=running` and `done_combinations=24`. For the corrected ablation, trust per-combination manifests and the updated sweep code first.

## Resume Command

Run from the repository root:

```powershell
& 'C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe' 'VICReg_review/run_data_view_sweep.py' --out-dir 'VICReg_review/heads/final_full_sweep_20260626_052814'
```

Expected behavior on resume:

1. Detect that `dim018_grl_n200_view80` is already trained and continue only if its
   manifest/reported settings match the corrected sweep definition.
2. Train `dim018_nogrl_n200_view80` with recommendation loss disabled.
3. Evaluate both corrected arms.
4. Continue the remaining full grid.

## Notes For Continuing

- Keep using full-window training: `max_view_sentences=0`.
- Keep using `split_recompute`; it is the current OOM-avoidance path.
- Do not start a new output directory unless explicitly requested. Reusing the existing output directory preserves resume behavior and prevents losing the already completed combinations.
- Do not rely only on `data_view_sweep_summary.csv` until the resumed scheduler refreshes it.
