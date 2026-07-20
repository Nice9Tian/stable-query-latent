# w9 — the full contrastive-experiment suite (release build)

Self-contained reproduction package for the paper's experiment campaign:
the I-CE recipe and every baseline/ablation arm (CE, MoCo-queue, BYOL,
VICReg-epd, slot/readout grids, twin-pack `pk` family, sliding-window `swin`
family, temperature sweep), on both the fixed split and 5-fold
cross-validation.

This build ships **code only** and performs **no repository
synchronisation** and **no cloud-provider API calls** (both were research
operations, removed on purpose). Every notebook assumes the code is already
present in this folder.

## Layout

```
w9/
├── Pod/                     # workers + one notebook per experiment
│   ├── w9_a100_worker.py    # fixed-split worker: every arm/loss family
│   ├── w9_cv_worker.py      # 5-fold worker (fold-inductive splits)
│   ├── w9_jobs.py           # claim files, labels, GPU detection, monitors
│   ├── h5_staging.py        # parallel copy of large assets to local disk
│   └── w9_*.ipynb           # experiments (see table below)
└── VICReg_review/           # minimal model/eval package the workers import
    ├── model.py             # tower, expander, VICReg loss
    └── text_variant_eval.py # ridge tag probe, split helpers, micro-F1
```

## Data prerequisites

The notebooks expect a data directory (default `/workspace/fusion_cache_w9`,
edit the constant in each notebook's first cell) holding the corpus assets:
`games.npz`, `wiki_eval.npz`, `wscan_gal_rev.npz`, `wscan_pool_rev.*`,
`ss_queries_rev.npz`, `ss_queries_rev_S.npy`, `wiki_clean_views.npz`,
`sp_raw_views.npz`, `tag_labels.npz`, `wiki_eval_split.json`,
`_tag_splitM.json`, and (for full-pool training) `full_pool_fp16.npy` +
`full_pool_meta.npz` with the `full_pool_READY` marker. These are produced
by the data pipeline in `release/dataset_builder`.

## Running

1. Put this `w9/` folder and the data directory on the machine (any CUDA
   box; 24 GB VRAM suffices for anchor budgets ≤ 2,048 sentences, 80 GB for
   4,096).
2. Open a notebook under `Pod/` and run its cells top to bottom. Each
   notebook is one experiment: it stages data, launches its workers, and
   prints the readout table at the end.
3. Multiple machines may run the same notebook against a shared volume:
   atomic claim files make them split the job queue safely; a tower's done
   marker is its final projection `.npz`.

| Notebook | Experiment |
|---|---|
| `w9_a100.ipynb` | wave-1 fixed-split arms |
| `w9_final_experiment.ipynb` | CE vs I-CE × {512…4096} × 5 folds (the paper's scaling table) |
| `w9_cv.ipynb` | 5-fold base pair |
| `w9_experiment_5fold_2.ipynb` | slot8 / MoCo-queue / BYOL / VICReg-epd × 5 folds @4096 |
| `w9_i2ce_t.ipynb` | temperature sweep × 5 folds × {512, 2048} |
| `w9_flash*.ipynb` | loss-ladder and structure grids |
| `w9_scale.ipynb`, `w9_mq.ipynb`, `w9_save_4096_tag.ipynb` | anchor-scale & anchor-supply arms |
| `w9_packageview.ipynb` | twin-pack `pk{N}` fractal family |
| `w9_scale_query.ipynb` | slot-count / readout capacity grid |
| `w9_swin.ipynb`, `w9_swin_5fold.ipynb` | sliding fresh-window CE field |
| `w9_i2ce_continue.ipynb` | budget-extension run |

Selection is deployment-faithful throughout: checkpoints are picked on
validation-fold review pseudo-queries; LLM rewrites are evaluation-only.
