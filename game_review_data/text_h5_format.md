# text_h5.h5 格式说明

`text_h5.h5` 是游戏评论管线里的中间文本语料文件。它保存“句子文本 + 评论/游戏层级偏移 + 游戏元数据”，供 `embedding_h5.h5` 直接继承并补上向量。

对应实现：
- 写入：`game_review_data/h5_corpus.py::build_text_h5()`
- 读取并补向量：`game_review_data/h5_corpus.py::embed_text_h5()`
- 训练侧最终消费：`VICReg_review/train_vicreg_review_h5.py`

## 1. 文件级约定

- HDF5 schema：`game_review_text_h5.v1`
- 字符串数据集使用 UTF-8 变长字符串
- 所有索引、偏移都是 0-based
- `texts` 的行顺序就是 embedding 的输入顺序，不能打乱
- `embedding_h5.h5` 是 `text_h5.h5` 的超集：除 `vectors` 外，其余数据集和属性应保持一致

## 2. 必要数据集

### 句子级

| 名称 | dtype | shape | 含义 |
|---|---:|---:|---|
| `texts` | UTF-8 string | `(sentences,)` | 每一行是一条待 embedding 的句子文本 |
| `sentence_ids` | UTF-8 string | `(sentences,)` | 句子在原评论中的编号，如 `sentence_1` |

### 评论级

| 名称 | dtype | shape | 含义 |
|---|---:|---:|---|
| `review_ids` | UTF-8 string | `(reviews,)` | 原评论编号，通常是评论 JSON 内的 0-based 索引 |
| `review_offsets` | `int64` | `(reviews + 1,)` | 评论到句子的偏移，指向 `texts` / `sentence_ids` |

### 游戏级

| 名称 | dtype | shape | 含义 |
|---|---:|---:|---|
| `game_review_offsets` | `int64` | `(games + 1,)` | 游戏到评论的偏移，指向 `review_ids` / `review_offsets` |
| `game_names` | UTF-8 string | `(games,)` | 游戏条目名，通常是句子 JSON 文件名去掉扩展名 |
| `appids` | UTF-8 string | `(games,)` | Steam appid |
| `game_titles` | UTF-8 string | `(games,)` | 游戏标题 |
| `release_date` | UTF-8 string | `(games,)` | Steam 商店发行日期；没有来源时为空字符串 |
| `tags_json` | UTF-8 string | `(games,)` | `tags` 字典的 JSON 序列化 |
| `positive` | `int64` | `(games,)` | 正向评论数 |
| `negative` | `int64` | `(games,)` | 负向评论数 |
| `positive_rate` | `float32` | `(games,)` | 正向比例 |
| `recommendation_label_source` | UTF-8 string | `(games,)` | 标签来源说明，如 `review_csv` / `games_json` |
| `source_sentence_files` | UTF-8 string | `(games,)` | 每个游戏对应的句子 JSON 源文件绝对路径 |

### 可选 TAG 数据

如果未启用 `--no-tag-labels`，还会写入：

| 名称 | dtype | shape | 含义 |
|---|---:|---:|---|
| `tag_names` | UTF-8 string | `(tag_count,)` | TAG 维度名 |
| `tag_labels` | `uint8` | `(games, tag_count)` | 二值标签 |
| `tag_raw_counts` | `float32` | `(games, tag_count)` | 原始计数 |

同时会写入这些属性：
- `tag_mapping_json`
- `tag_mapping_path`
- `tag_missing_appids`
- `tag_count`

## 3. 必要属性

`text_h5.h5` 至少应包含这些属性：

- `schema = "game_review_text_h5.v1"`
- `source`
- `created_at`
- `games`
- `reviews`
- `sentences`
- `games_json`
- `sentences_dir`
- `recommendation_label_min_length`
- `recommendation_label_review_dirs`

这些属性主要用于校验、追溯和恢复，不参与 embedding 数学计算。

## 4. 偏移关系

这是最重要的部分。

### 句子 -> 评论

对任意评论 `r`：

```text
review_offsets[r] : review_offsets[r + 1]
```

表示该评论对应的句子行区间，切片结果直接作用于 `texts` 和 `sentence_ids`。

### 评论 -> 游戏

对任意游戏 `g`：

```text
game_review_offsets[g] : game_review_offsets[g + 1]
```

表示该游戏包含的评论行区间，切片结果作用于 `review_ids` 和 `review_offsets`。

### 约束

- `review_offsets[0] == 0`
- `review_offsets[-1] == sentences`
- `game_review_offsets[0] == 0`
- `game_review_offsets[-1] == reviews`
- `len(texts) == len(sentence_ids) == sentences`
- `len(review_ids) == reviews`
- `len(game_names) == len(appids) == len(game_titles) == len(release_date) == games`

## 5. 排序规则

写入顺序必须稳定，后续 embedding 依赖这个顺序原样继承。

当前实现的顺序是：

1. 游戏文件按文件名排序
2. 每个游戏内的评论按 `review_id` 数值顺序
3. 每条评论内的句子按 `sentence_id` 数值顺序

因此，`texts[i]` 的位置一旦写入，就不应再重排。

## 6. 和 embedding_h5.h5 的关系

`embed_text_h5()` 的行为很简单：

1. 读取 `text_h5.h5`
2. 原样复制除 `vectors` 以外的所有数据集和属性
3. 按 `texts` 的顺序逐批调用 embedder
4. 写入 `vectors` 数据集
5. 把输出文件标记为 `game_review_embedding_h5.v1`

所以，`embedding_h5.h5` 与 `text_h5.h5` 的差异只有：

- 多了 `vectors`
- 属性里的 `schema`、`embedding_*` 字段更新

训练脚本 `VICReg_review/train_vicreg_review_h5.py` 只要求：

- `vectors`
- `review_offsets`
- `game_review_offsets`
- `game_names`

如果你能正确生成 `text_h5.h5`，并且 `embed_text_h5()` 成功跑完，那么生成的 `embedding_h5.h5` 就可以直接喂给训练侧。

## 7. 最小对接建议

如果你在别的项目里要生成兼容文件，请确保：

1. 先把所有句子文本放进 `texts`
2. 用 `review_offsets` 和 `game_review_offsets` 串起层级
3. 保持顺序不变
4. 不要改动 `embed_text_h5()` 需要的 `texts` 行数
5. 最终由 embedding 阶段补上 `vectors`

## 8. 快速验收

可以用这些规则人工检查：

```text
review_offsets[0] == 0
review_offsets[-1] == len(texts)
game_review_offsets[0] == 0
game_review_offsets[-1] == len(review_ids)
len(game_review_offsets) == len(game_names) + 1
```

如果这些成立，`text_h5.h5` 的层级关系基本就是对的。
