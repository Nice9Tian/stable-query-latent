# data/assets/ — training / evaluation tensor assets

All produced by `dataset_builder/build_assets.py` (fixed SEED=20260711;
resume-safe: existing files are skipped). `gidx` always indexes rows of
`games.npz`.

| File | Structure | Meaning |
|---|---|---|
| `games.npz` | `names` (2020,) object | **the gidx basis** — order = `game_names` of embedding_h5 |
| `wiki_clean_views.npz` `wiki_llm_views.npz` `sp_raw_views.npz` | `S` (N, MAXLEN, 1024) fp16 zero-padded; `S_len` (N,) i32; `gidx` (N,) i64; `names` (N,) object | full-document views: SaT split (no cap) + per-sentence embedding. Row-standardized (rown) at load time before entering the tower |
| `wscan_pool_rev.npy` | (2020, 2048, 1024) fp16 | the review pool: 2,048-sentence budget per game; **gold guarantee** = the 3 longest reviews enter unconditionally, the rest fill in whole at random; vectors already per-row mean0/std1 |
| `wscan_pool_rev_rid.npy` | (2020, 2048) i32 | per-row review id in the pool (−1 = padding) — the basis of review-level rejection sampling |
| `wscan_pool_rev_len.npy` | (2020,) i32 | used rows per game |
| `wscan_gal_rev.npz` | `gal` (2020, 512, 1024) fp16; `gal_len` i32; `gal_doc_len` i32 | **the anchors**: sp_raw document sentences FIRST, whole reviews fill the 512 budget; `gal_doc_len` = doc-prefix length (0 = no doc) — the head's doc-gating masks the prefix through it |
| `ss_queries_rev.npz` | `off` (8081,) i64; `gidx` (8080,) i64 | pseudo-query index: 4 per game, anchor-shaped (doc prefix + whole reviews @512), flat-stored |
| `ss_queries_rev_S.npy` | (Σ sentences, 1024) fp16, ~9.5 GB | pseudo-query vectors; query j = `S[off[j]:off[j+1]]`. **Loaded as mmap** |
| `wiki_eval.npz` | `S` (3256, MAXLEN, 1024) fp16; `S_len`; `gidx`; `names`; `variants` object | evaluation queries = 814 games × 4 variants, **complete text, no truncation** |
| `_tag_split.json` | train/val/test game-name lists | the tag-probe split (seed 42, 0.7/0.15) |

The protocol split is NOT in this directory: the authoritative
`wiki_eval_split.json` (seed 20260711, 204/203/407) ships with the code
under `dataset_builder/`.
