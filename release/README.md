# latenarray-I2CE — Invariant Game Representation Learning from Steam Reviews

Code and data pipeline for the I-CE representation described in the
submitted manuscript. A small attention-pooling tower reads cached
sentence embeddings of Steam reviews and produces one 128-d vector per
game. It is trained by a compound objective: a cross-entropy term (CE)
that classifies each fresh review view against a gallery of every game's
anchor pack, and an invariance term (I) that pulls the views of one game
together. The result is a single index that answers both name-intact and
name-stripped queries, and carries enough semantics for a tag probe.

The tower is about 0.36 M trainable parameters over a frozen
Qwen3-Embedding-0.6B (1024-d, last-token pooling, no instruction prefix).
Every review is sentence-split and embedded once; training thereafter
reads only the cached vectors.

## Where the paper's numbers come from

**`contrast_experiment/w9/` is the campaign behind the manuscript's
tables**, and it is the right entry point if you want to reproduce a
published figure. It is self-contained (workers, notebooks, and the
minimal model package they import) and its arm names are the keys in the
released result JSONs: `wcle_i2ce_icetf`, `wcle_ce_cetf`,
`wcle_bce_cetf` (SimCLR-style), `wcle_byol_bytf`, `wcle_vic_cetf`
(VICReg), `wcle_swin*` (the sliding fresh-window teacher),
`wcle_vfai2ce_icetf` (the views-first-anchor control). See
[`contrast_experiment/w9/README.md`](contrast_experiment/w9/README.md)
for the notebook-to-experiment map.

The two `run.py` entry points below are the packaged reproduction paths.
They train the same tower under an older default protocol, and the table
in [Protocol](#protocol-what-the-code-actually-runs) states exactly where
those defaults differ from the manuscript's headline setting.

All numbers the manuscript prints are **zero-shot cosine retrieval**: no
retrieval head is trained anywhere, a query is the tower's encoding of
text it has never seen, and the candidates are all 2,020 games ranked by
cosine against the anchor gallery. In the result files that is the `zs_*`
/ `zsbest_*` family. The `ft4var_*` files come from the two-phase
name-recall head under `steam_reviews_framework/backhead_name.py`, a
separate line of work the manuscript does not report.

## Layers

Dependencies run strictly one way: experiment → framework → main_model.

```
main_model/                the tower alone (LariceTower + LariceConfig),
                           task-agnostic, standalone-publishable.
                           Tensor protocol [data, view]:
                           x[B,V,S,D] + mask  ->  z[B,V,out_dim]
steam_reviews_framework/   the Steam binding: protocol (splits, the
                           inductive exclusion rule, the vsel selection
                           score) / sampler / anchors / trainer /
                           backhead_name / backhead_tag / run.py
contrast_experiment/       the baseline and ablation suite: contrast_models
                           (CE, BYOL, ArcFace, gate and I-dose variants),
                           contrast_heads, run.py, report.py,
                           w9/ (the manuscript's campaign)
dataset_builder/           corpus reconstruction: reviews (Kaggle -> clean
                           -> split -> embed -> h5), corpora (wiki scrape
                           -> clean -> LLM rewrites), build_assets,
                           API templates
data/                      every heavy artefact (gitignored except READMEs)
```

## Quick start

```bash
pip install -r requirements.txt

# Path 1 - train one champion tower on the fixed split
python steam_reviews_framework/run.py

# Path 1 on a cross-validation fold
python steam_reviews_framework/train_champion.py --cv-fold 0

# Path 2 - the whole contrast roster, then the comparison table
python contrast_experiment/run.py [--cv]

# Data only, no training
python steam_reviews_framework/run.py --data-only
```

Both entries share one data-preparation pipeline, and every step is
resume-safe: rerunning skips whatever already exists, so an interrupted
run continues where it stopped.

1. **Corpora (bundled, reproducibility first).** `wikipage.zip`
   (wiki_clean / variants / llm, 814 games) and `storepage.zip` (six
   store-page corpora, 1,811 games) ship in
   `steam_reviews_framework/corpora_bundles/` and unpack into
   `data/corpora/`. The bundled texts always win: Wikipedia is never
   re-scraped and the LLM is never re-run, so results cannot drift with
   live wiki edits or non-deterministic rewrites.
2. **Review files.** The 73-million-sentence embedding h5 and the
   text/tag h5 are downloaded when `LARICE_EMBED_H5_URL` /
   `LARICE_TEXT_H5_URL` are set, or rebuilt once from the Kaggle dump via
   `dataset_builder/reviews/`.
3. **Tensor assets.** `dataset_builder/build_assets.py` fills in whatever
   is missing. `dataset_builder/rebuild_data.py --check` reports layer by
   layer what is absent and which command produces it.

Credentials live in `dataset_builder/llmAPI.txt` (corpus rewriting) and
`dataset_builder/embeddingAPI.txt` (cloud embedding endpoint), both
gitignored, both with a `*.template.txt` next to them. They can also be
typed into the settings block at the top of
`steam_reviews_framework/run.py`.

## Protocol (what the code actually runs)

Fixed throughout, and identical to the manuscript:

- **Corpus.** 2,020 games released 2020-2024, 6.6 M reviews, 73 M
  sentences. A game enters only if at least 500 of its reviews survive a
  300-character floor.
- **Fully inductive.** No text of a held-out game reaches any training
  stage: not its reviews, its documents, its pseudo-queries, nor a
  gallery-negative gradient. The training-time gallery covers train games
  only; the full 2,020-game gallery is used with a frozen tower at
  evaluation time.
- **Strong views.** Whole reviews, never truncated, drawn by rejection
  sampling with acceptance `a(L) = 0.2 + 0.7*(L-Lmin)/(Lmax-Lmin)`
  recomputed per game, until the view holds at least 16 sentences. Three
  review views plus one document view (LLM-rewritten wiki where one
  exists, else the store page, else a fourth review view).
- **Anchor pack (the weak view).** Store-page prefix, then whole reviews
  to a sentence budget, re-encoded with gradient at every step.
- **Tower.** 4 latent queries, 128-d, 4 heads, one cross-attention layer
  over the raw 1024-d sentence embeddings, mean-pooled over slots
  (`readout="pool"`), then a two-layer MLP to an L2-normalized vector.
- **Optimizer.** AdamW, lr 5e-4, weight decay 1e-4, batch 192 games,
  16 steps per epoch, gradient clipping at 5.0, AMP. Frozen `tau = 0.02`,
  invariance weight 2.0.
- **Selection.** Checkpoints every 50 epochs, no online early stopping.
  The deployed checkpoint is picked post-hoc on validation-fold queries
  and never on test.
- **Splits.** `dataset_builder/wiki_eval_split.json` (seed 20260711) is
  the authoritative file and ships with the code. The fixed split is
  204 test / 203 val / 407 excluded of the 814-game wiki universe, which
  leaves a 1,613-game training gallery. `--cv-fold k` and `--cv` permute
  the same 814 games into five folds (fold k = test, fold k+1 = val, the
  other three = train), which leaves 1,694 per fold.

Where the packaged defaults differ from the manuscript's headline
configuration:

| | `run.py` default | the manuscript's headline |
|---|---|---|
| anchor budget | 512 sentences (`GCAP` in `build_assets.py`) | 4,096, rebuilt on the fly by the w9 workers (`--anchor-cap`) |
| epochs | 1,000 (600 for the `--cv` recipes) | 2,000 |
| split | the fixed 204/203/407 partition | five-fold over the same 814 games |
| CE scope | gated: CE fires only on games carrying a document view (`champion_cegate2`) | ungated, every batch game (`wcle_i2ce_icetf`) |
| readout | zero-shot cosine **and** the two-phase name head | zero-shot cosine only |

None of these are hard-coded: `--epochs`, `--cv-fold`, the arm roster in
`contrast_experiment/contrast_models/roster.py`, and `GCAP` each move one
of them. The manuscript's exact combinations are already wired up as the
w9 notebooks.

## Hardware

The anchor budget sets the requirement, because the gallery is
re-encoded with gradient at every step. A 24 GB desktop GPU covers
budgets up to 2,048 sentences, and retrieval has already saturated by
1,024 (a 22.3 GiB peak, within 0.02 of the full configuration on every
reading). The 4,096-sentence budget occupies a single 80 GB A100, about
61 GiB peak and roughly 6.7 wall-clock hours for a 2,000-epoch run.

At inference the cost is ordinary: one forward pass through the frozen
0.6B embedder (under 3 GiB) plus a millisecond-level inner-product search
against the pre-computed anchors.

## Data and code separation

Every heavy artefact lives under `data/`, outside the code tree, and each
location is overridable so an existing layout can be linked in without
copying:

```
LARICE_DATA_ROOT   data root (default: release/data)
LARICE_ASSETS      training/eval tensors     LARICE_RESULTS  checkpoints + result jsons
LARICE_CORPORA     text corpora              LARICE_EMBED_H5 / LARICE_TEXT_H5  review h5
```

See [`data/README.md`](data/README.md) for what each sub-directory holds
and which stage writes it.

## Requirements

Python 3.11+, `numpy`, `torch`, `h5py`, `scikit-learn`, `scipy`,
`requests`. The data pipeline additionally needs `wtpsplit` (the SaT
sentence splitter), `transformers` and `sentence-transformers` (the local
Qwen3-Embedding backend), and `pandas` (Kaggle review preparation). A run
that only trains from prepared assets needs none of that second group.
