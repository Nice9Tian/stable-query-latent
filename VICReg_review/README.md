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
# 1. Build the tag vocabulary + per-game multi-hot labels from games.json.
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/tag_build.py

# 2. Probe a checkpoint: review text -> frozen encoder -> (16, 18) code -> tags.
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe VICReg_review/train_tag_probe.py `
  --device cuda --amp --checkpoint VICReg_review/heads/vicreg_review_h5_best.pt
```

`tag_build.py` writes `VICReg_review/tags/{tag_vocab.json,tag_labels.npz}`
(`--min-count`, `--top-k`, `--target-mode binary|weight`, `--source tags|genres|categories`).
`train_tag_probe.py` caches one averaged feature per game (`--feature-views`),
trains a `TagRegressionHead` with BCE, and stops on val-mAP patience
(`--patience`) or a gradient-norm floor (`--grad-plateau`). It writes a report
with overall mAP/micro-F1 and the best/worst per-tag average precision.
