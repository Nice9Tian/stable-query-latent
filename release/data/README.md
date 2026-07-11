# data/ — the data root

Code and data are strictly separated: every heavy artefact lives here, and
**everything except these READMEs is gitignored**. The four
sub-directories are split by producer / consumer:

| Directory | Contents | Written by | Read by |
|---|---|---|---|
| `corpora/` | text corpora (wiki / sp) | `steam_reviews_framework/run.py` auto-unpacks the bundled zips | `dataset_builder/build_assets.py` |
| `reviews/` | heavy review files (h5) | URL download, or rebuilt via `dataset_builder/reviews/` | build_assets + the pod full pool |
| `assets/` | training/eval tensors (npz/npy) | `dataset_builder/build_assets.py` | the trainers (framework / experiment) |
| `results/` | checkpoints / projections / result jsons | the trainers | `contrast_experiment/report.py` |

Every location can be redirected by environment variable (zero-copy links
to an existing layout; see `dataset_builder/paths.py`):

```
LARICE_DATA_ROOT  data root (default: this directory)   LARICE_ASSETS   assets/
LARICE_CORPORA    corpora/                               LARICE_RESULTS  results/
LARICE_EMBED_H5   reviews/embedding_h5.h5                LARICE_TEXT_H5  reviews/text_h5.h5
```

One-click preparation: `python steam_reviews_framework/run.py --data-only`.
