# Game Review BERTopic

- Generated: 2026-06-23T18:24:22
- Input: `C:\Users\admin\Documents\studable query latent\game_review_data\game_review_cleaned_3_sentences`
- Input files: 293
- Input directory size: 99.97 GB
- Sample method: `balanced_prefix_per_game`
- Sampled documents: 100000
- Embedding dimension: 1024
- UMAP `n_neighbors`: 100
- HDBSCAN `min_cluster_size`: 100
- CountVectorizer `min_df`: 1
- Random state: 42
- Fit time: 254.8 seconds

Note: the source sentence/vector directory is about 100 GB. This run uses a deterministic per-game prefix sample and skips metadata records with `review_id < 3` before fitting BERTopic. It is a practical topic snapshot, not a full all-sentence HDBSCAN fit.

## Environment

| Package | Version |
|---|---:|
| `bertopic` | `0.17.4` |
| `hdbscan` | `0.8.44` |
| `umap-learn` | `0.5.12` |
| `ijson` | `3.5.0` |
| `numpy` | `2.4.3` |
| `scikit-learn` | `1.8.0` |

## Summary

- Topics excluding outliers: 4
- Outlier documents: 2884
- Outlier rate: 2.88%

## Topic Table

| Topic | Count | Top Words |
|---:|---:|---|
| -1 | 2884 | outliers |
| 0 | 96691 | game, like, que, just, time, games, play, really, dont, story |
| 1 | 166 | bad, good, good decent, decent bad, beautiful good, beautiful, decent, bad beautiful, bad good, bad bad |
| 2 | 157 | offer shrine, shrine gain, gain offer, shrine, gain, offer, fame, attached, great, ihr |
| 3 | 102 | simulator, simulator fusion, simulator simulator, janken, fusion rise, prodigy wings, wings silicon, rise cute, roommate undercover, lifeunknown |

## Top Topic Examples

### Topic 0 (96691 docs)

Top words: game, like, que, just, time, games, play, really, dont, story

- `1009290_6672.json review=3 sentence=sentence_1`: 艾玛，今天呢，游戏发行不到三个月就开始打折咯~
- `1009290_6672.json review=3 sentence=sentence_2`: 游戏近期更新，这下可以彻底放弃这款粪作了~
- `1009290_6672.json review=3 sentence=sentence_3`: 弃掉之前，做做好事，还是说说原因避免别人被坑吧~

### Topic 1 (166 docs)

Top words: bad, good, good decent, decent bad, beautiful good, beautiful, decent, bad beautiful, bad good, bad bad

- `1038250_6432.json review=4 sentence=sentence_25`: Максимум, что тут годного есть - это Microphone от American Authors, Mother от Amazons и Friends от The Hara.
- `1049410_23830.json review=7 sentence=sentence_2`: Beautiful☑ Good☐ Decent☐ Bad☐
- `1049410_23830.json review=7 sentence=sentence_9`: Very good☐ Good☐ Not too bad☐ Bad☐

### Topic 2 (157 docs)

Top words: offer shrine, shrine gain, gain offer, shrine, gain, offer, fame, attached, great, ihr

- `1057750_6138.json review=7 sentence=sentence_51`: TSORF has no great revelation, no great resolution, no great message, no challenge whatsoever.
- `1124300_25961.json review=10 sentence=sentence_19`: There's no fame attached to it and no, fame attached to a wonder that just so happens to give faith doesn't count.
- `1124300_25961.json review=10 sentence=sentence_27`: No more holy sites, no more grievances neither for nor against.

### Topic 3 (102 docs)

Top words: simulator, simulator fusion, simulator simulator, janken, fusion rise, prodigy wings, wings silicon, rise cute, roommate undercover, lifeunknown

- `1049590_61216.json review=6 sentence=sentence_2`: 异世界猜拳勇者：Isekai Janken Hero
- `1049590_61216.json review=6 sentence=sentence_3`: 命运模拟:2024：Orgasm Simulator
- `1049590_61216.json review=6 sentence=sentence_4`: 2024命运模拟器2023：ORGASM SIMULATOR 2023召唤与合体：Summon and fusion
