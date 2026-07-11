# data/reviews/ — 评论重型文件

获取方式二选一:
- 设 `LARICE_EMBED_H5_URL` / `LARICE_TEXT_H5_URL`,由
  `steam_reviews_framework/run.py` 自动下载;
- 从 Kaggle 原始数据重建一次:`data_pipeline/reviews/`
  (prepare_kaggle_steam_reviews → build.py 清洗分句 → Build_new.py
  全量嵌入 → h5_corpus.py 组装)。

## embedding_h5.h5(~164 GB)

2,020 游戏、6,649,325 条评论、73,289,517 句。三级 offset 索引:

```
游戏 g 的评论 id 区间:  game_review_offsets[g] : game_review_offsets[g+1]
评论 r 的句行区间:      review_offsets[r]      : review_offsets[r+1]
句行 i 的向量:          vectors[i]
```

| 字段 | 形状 / dtype | 说明 |
|---|---|---|
| `vectors` | (73289517, 1024) float16 | Qwen3-Embedding-0.6B 句向量(last-token pool) |
| `texts` / `sentence_ids` | (73289517,) object | 句原文 / 句 id |
| `review_ids` | (6649325,) object | 评论 id |
| `review_offsets` | (6649326,) int64 | 评论 → 句行 |
| `game_review_offsets` | (2021,) int64 | 游戏 → 评论 id |
| `game_names` / `appids` / `game_titles` | (2020,) object | `<appid>_<GameName>` 等 |
| `tag_labels` / `tag_names` / `tag_raw_counts` | (2020,23) uint8 / (23,) / (2020,23) f32 | 23 标签体系 |
| `positive` `negative` `positive_rate` `release_date` 等 | (2020,) | 商店元数据 |

## text_h5.h5(轻索引)

与 embedding_h5 **同 schema、同游戏顺序,但没有 `vectors`**——
框架只从它读 `game_names` + `tag_labels`(23 标签监督),
无需挂载 164 GB 大文件即可完成头训练之外的标签评测装载。

## games.json(可选)

2,020 游戏商店元数据(描述原文等)。**仅在重新生成 sp 语料时需要**
(`data_pipeline/corpora/build_sp_corpus.py`);正常复现走内置压缩包,
不需要此文件。
