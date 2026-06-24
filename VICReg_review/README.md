# VICReg Review

Self-supervised game-review encoder:

- Pick one game JSON from `game_review_data/game_review_cleaned_3_sentences`.
- Build two views by independently sampling 60 percent of reviews.
- Encode both views with one shared `LatentArrayMLP` (`Latent_Array_MLP` alias):
  input `1024 -> latent_dim 256`, 256 learnable query slots, a single
  cross-attention layer (no residuals, no extra blocks), then a per-latent funnel
  `256 -> 128 -> 64 -> 32 -> 18`. Output is `(B, 256, 18)`.
- Train with VICReg consistency on the final 18-d latent codes.
- Apply a frozen SST MLP4-A sentiment head through GRL so the latent codes become
  sentiment-confusing (driven toward output 0.5 = maximum entropy). Because the
  head needs 1024-d inputs, the adversary holds a learnable up-projection probe
  (`18 -> 256 -> 1024`) placed *after* the GRL, so the encoder is always the
  adversarial party while the probe tries to recover sentiment confidence.

Default run:

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_vicreg_review.py --device cuda --amp
```

Useful smoke test:

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_vicreg_review.py --device cpu --epochs 1 --steps-per-epoch 1 --limit-games 1 --max-sentences 8 --no-save
```

Outputs are written under `VICReg_review/heads/` by default. Checkpoints and JSON
manifests are ignored by the project `.gitignore`.

HDF5 path for faster training:

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/build_review_h5.py --workers 2 --shards 8
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_vicreg_review_h5.py --device cuda --amp --epochs 100 --batch-size 16
```

`build_review_h5.py` also writes TAP labels into the H5:
`tap_names`, `tap_labels`, `tap_raw_counts`, `appids`, and `game_titles`.
The only tag mapping source is `VICReg_review/tags/tap_mapping.json`, where each
fine Steam tag maps to one coarse TAP class or `"del"`.

Use `--cache-mode full` only when RAM can hold the next prepared epoch. The
default `queue` mode overlaps H5 loading with GPU training using a bounded
prefetch queue.

## Tag validation probe (diagnostic only)

A separate, **validation-only** head checks whether the frozen encoder code can
predict a game's Steam tags. It never touches the VICReg loss — the encoder is
loaded frozen and the probe has its own optimizer and self-stops when learning
plateaus. Rising tag mAP across VICReg checkpoints = a more robust representation.
The probe flattens the `(256, 18)` code (4608 dims) before its MLP.

```powershell
# Probe a checkpoint: review text -> frozen encoder -> TAP labels.
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_tag_probe.py `
  --device cuda --amp --checkpoint VICReg_review/heads/vicreg_review_h5_best.pt
```

`train_tag_probe.py` (redesigned) caches one averaged feature per game
(`--feature-views`), pools it (`--pool flatten|stats|mean`), and runs **k-fold
cross-validation** with a **fairness rule** (a tag is scored only where it has
`--min-train-pos` train positives) per-tag logistic regression. It reports
micro-F1 mean±std, a frequency-floor breakdown, and (with `--export-head PATH`)
saves a portable linear probe for `validation.py`.

See `TAG_PROBE_RESULTS.md` for the full analysis: the raw-embedding F1 ceiling
(~0.50 on 293 games), why 0.85 is unreachable, and the selectivity result.

## Dual-probe validation during training

`train_vicreg_review_h5.py` runs a **tag + PXI dual probe** on the live encoder
every `--probe-every` epochs (**default 1** — on; set 0 for smoke tests) and on the
last epoch, logging a validation curve to `heads/dual_probe_history.tsv`. Per probe it reports, all from the frozen code:
`tag_content_f1`, `tag_subjective_f1`, `tag_selectivity` (= content − subjective),
`code_sentiment_r2` (sentiment suppression), and `pxi_func_f1` / `pxi_psych_f1`
(functional vs psychological, on the 21 PXI-overlap games). The probe uses the
`stats` pool for speed, restores the encoder's train mode, and never aborts
training on failure. Standalone: `python dual_probe.py --checkpoint ...`.

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_vicreg_review_h5.py `
  --device cuda --amp --probe-every 5
```

## Hypothesis test: keeps mechanics+story, drops sentiment

The real claim is that the encoder + sentiment adversary retains content and
filters opinion. `probe_selectivity.py` measures content-tag F1, subjective-tag
F1, and SST-sentiment R² on the VICReg code vs the raw embedding (ceiling) and
reports retention = vicreg/raw. TAP labels are already coarse/non-subjective, so
they are treated as content labels. `ceiling_diagnostic.py` is the raw-embedding
upper bound.

```powershell
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/probe_selectivity.py --device cuda `
  --vic-cache VICReg_review/tags/probe_feat_<checkpoint>_fv4_sf0.6.npz
```

## Validation UI (`validation.py`)

Aligned with the current probe. Three steps: export the deployable linear probe,
build the in-domain game pool, then launch the UI (it auto-loads both):

```powershell
# 1. export the linear probe from the best encoder.
#    Use --pool stats for deployment: flatten has ~3.7k near-zero-variance dims that
#    make StandardScaler explode on out-of-distribution text (saturated prob=1 tags).
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_tag_probe.py `
  --checkpoint VICReg_review/heads/sweep_adv/vicreg_adv10_best.pt --pool stats `
  --device cpu --export-head VICReg_review/heads/tag_probe_linear.pt

# 2. launch
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe validation.py
```

Pipeline: text → local Qwen embedding → frozen encoder code → pool → L2 normalize →
per-TAP logistic probe → TAP probabilities. Game candidates come from the same
H5 `tap_raw_counts` matrix used to train/evaluate the probe, so matching cannot
drift from the label mapping.
