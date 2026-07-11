# data/reviews/ — the heavy review files

Obtain either way:
- set `LARICE_EMBED_H5_URL` / `LARICE_TEXT_H5_URL` and let
  `steam_reviews_framework/run.py` download them;
- rebuild once from the Kaggle dump: `dataset_builder/reviews/`
  (prepare_kaggle_steam_reviews → build.py clean + sentence split →
  Build_new.py embed everything → h5_corpus.py assemble).

## embedding_h5.h5 (~164 GB)

2,020 games, 6,649,325 reviews, 73,289,517 sentences. Three-level offset
indexing:

```
review-id range of game g :  game_review_offsets[g] : game_review_offsets[g+1]
sentence-row range of review r :  review_offsets[r] : review_offsets[r+1]
vector of sentence row i :  vectors[i]
```

| Field | Shape / dtype | Meaning |
|---|---|---|
| `vectors` | (73289517, 1024) float16 | Qwen3-Embedding-0.6B sentence vectors (last-token pool) |
| `texts` / `sentence_ids` | (73289517,) object | sentence text / id |
| `review_ids` | (6649325,) object | review id |
| `review_offsets` | (6649326,) int64 | review → sentence rows |
| `game_review_offsets` | (2021,) int64 | game → review ids |
| `game_names` / `appids` / `game_titles` | (2020,) object | `<appid>_<GameName>` etc. |
| `tag_labels` / `tag_names` / `tag_raw_counts` | (2020,23) uint8 / (23,) / (2020,23) f32 | the 23-tag taxonomy |
| `positive` `negative` `positive_rate` `release_date` … | (2020,) | store metadata |

## text_h5.h5 (the light index)

**Same schema and game order as embedding_h5 but WITHOUT `vectors`** —
the framework reads only `game_names` + `tag_labels` (23-tag supervision)
from it, so tag evaluation loads without mounting the 164 GB file.

## games.json (optional)

Store metadata for the 2,020 games (raw descriptions etc.). **Needed only
to REGENERATE the sp corpora**
(`dataset_builder/corpora/build_sp_corpus.py`); normal reproduction uses
the bundled zips and never touches this file.
