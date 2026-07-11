# larice — Latent Represent I-CE

Cross-domain-robust set representation learning: the **larice tower**
(I-CE with CE gating; N latent queries cross-attending over frozen
sentence embeddings — SetPoolN lineage). Reference task: train on Steam
reviews, evaluate on wiki-derived rewritten queries under the
fully-inductive R60 protocol.

## Three layers (strictly one-way dependencies: experiment → framework → main_model)

```
main_model/       THE model — larice (LariceTower + LariceConfig).
                  Champion tower only, no task heads; standalone-publishable.
                  Tensor protocol [data, view]:
                  x[B,V,S,D] + mask  →  z[B,V,N×DM] (concat readout)
steam_reviews_framework/   Steam task binding — calls main_model: protocol /
                  sampler / anchors / unified trainer / backhead_name
                  (two-phase name-recall head) / backhead_tag (23 tags) /
                  run.py + train_champion.py (path 1)
contrast_experiment/   Full contrast suite — contrast_models (CE / BYOL /
                  ArcFace / gate & I-dose variants) + contrast_heads +
                  run.py (one-click, --cv for 5-fold) + report
                  + pod/ (multi-machine parallel route, see its README)
dataset_builder/  Dataset reconstruction — reviews (Kaggle → clean → split
                  → embed-all → h5), corpora (wiki scrape → clean →
                  rewrites; six sp corpora), build_assets, API templates
data/             (gitignored) every heavy artefact: h5 / npz / texts / results
```

## Two one-click reproduction paths

```bash
pip install -r requirements.txt

# Path 1 — train the champion (cegate2: CE-gated + I×2, vsel selection)
python steam_reviews_framework/run.py

# Path 2 — train ALL contrast arms (18; --cv adds 6 recipes × 5 folds;
#          finishes by writing the comparison table)
python contrast_experiment/run.py [--cv]

# Pod path (multi-machine parallel contrast suite):
#   open contrast_experiment/pod/w9_all.ipynb on RunPod
```

The two run.py entries divide cleanly: **the framework run trains only the
champion; the experiment run trains everything else**. They share one
data-preparation pipeline (automatic, resume-safe):

1. **Corpora (bundled, reproducibility first)** — `wikipage.zip`
   (wiki_clean / variants / llm, 814 games) and `storepage.zip` (six sp
   corpora, 1,811 games) in `steam_reviews_framework/corpora_bundles/`
   auto-unpack into `data/corpora/`. **Bundled texts always win: Wikipedia
   is never re-scraped and the LLM is never re-run** — results cannot
   drift with live wiki edits or non-deterministic rewrites.
2. **Heavy review files** — downloaded via `LARICE_EMBED_H5_URL` /
   `LARICE_TEXT_H5_URL` when missing, or rebuilt once from the Kaggle dump
   via `dataset_builder/reviews/`.
3. **Tensor assets** — `dataset_builder/build_assets.py` fills in whatever
   is missing.

## Data / code separation

All heavy artefacts live under `data/` (or link an existing layout with
environment variables — zero copying):

```
LARICE_DATA_ROOT   data root (default release/data)
LARICE_ASSETS      training/eval tensors     LARICE_RESULTS   ckpts + result jsons
LARICE_CORPORA     text corpora              LARICE_EMBED_H5 / LARICE_TEXT_H5   review h5
```

`dataset_builder/rebuild_data.py --check` reports layer by layer what is
missing and which command produces it. Credentials (both gitignored, copy
from the `*.template.txt` next to them in `dataset_builder/`):
`llmAPI.txt` for LLM corpus rewriting, `embeddingAPI.txt` for the cloud
embedding endpoint.

## Protocol highlights (R60, finalized 2026-07-11)

- Evaluation universe = 814 cleaned wiki pages; fixed split seed 20260711 =
  204 test / 203 val / 407 train (`dataset_builder/wiki_eval_split.json` —
  the authoritative file, ships with the code);
- **Fully inductive**: no text of a held-out game — reviews, documents,
  pseudo-queries, or gallery-negative gradients — enters any training
  stage; the full 2,020-game gallery is used only with a frozen tower at
  eval time;
- Queries = the four wiki variants (neutral / noname / positive /
  negative), complete text, no truncation;
- Anchor = store-page text prefix + whole reviews @ 512 sentences;
  review-level rejection sampling a(L) = 0.2 → 0.9 keeps long "gold-mine"
  reviews reachable;
- Towers train the full 1,000-epoch budget with NO early stopping,
  checkpoint every 50 ep; the best checkpoint is picked post-hoc by vsel
  (best of the two noname axes + neutral as an additive sanity gate) and
  re-verified with 10 seeds.
