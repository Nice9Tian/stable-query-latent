# data/corpora/ — text corpora

**Source: the zips bundled in the repository**
(`steam_reviews_framework/corpora_bundles/`: `wikipage.zip` +
`storepage.zip`), auto-unpacked here by `steam_reviews_framework/run.py`.
**Reproducibility rule: the bundled corpora are ALWAYS used — Wikipedia is
never re-scraped and the LLM is never re-run** (wiki pages get edited and
LLM rewrites are non-deterministic; re-collecting would make results
drift).

All files are UTF-8. `<appid>` is the Steam appid, matching the game-name
prefix in `assets/games.npz` (`<appid>_<GameName>`).

## From wikipage.zip (814 games per set)

| Path | Structure | Role |
|---|---|---|
| `wiki_clean/<appid>_<GameName>.txt` | cleaned wiki page: game-content sections only, match-validated | tower doc views (clean family) |
| `wiki_variants/<appid>/{neutral,noname,positive,negative}.txt` | four LLM rewriting styles, ≥300 chars each | **evaluation queries** (the text source of wiki_eval.npz) |
| `wiki_llm/<appid>_….txt` | faithful sentence-by-sentence rewrite (same filenames as wiki_clean) | pretraining-leak ablation (*_wllm arms) |
| `wiki_clean_manifest.json` | per page: kept sections / chars / match rank / verdict | provenance |

## From storepage.zip (1,811 games per set, `<appid>.txt`)

| Directory | Contents |
|---|---|
| `sp_raw/` | store-page text (cleaned). **The anchor's document prefix** + the tower's sp doc view |
| `sp_neutral/` | neutral rewrite |
| `sp_llm/` | faithful sentence-by-sentence rewrite |
| `sp_positive/` / `sp_negative/` | sentiment rewrites |
| `sp_noname/` | neutral with every name removed (proper nouns → invented words) |
| `sp_manifest.json` | generation provenance |

Consumption: `dataset_builder/build_assets.py` splits with SaT (drops
<10-char fragments, **no sentence cap**), embeds every sentence with
Qwen3-Embedding-0.6B, and writes `assets/*_views.npz`.
