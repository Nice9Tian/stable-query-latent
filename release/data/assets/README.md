# data/assets/ — 训练/评测张量资产

全部由 `data_pipeline/build_assets.py` 生成(固定 SEED=20260711,断点续建:
已存在的文件自动跳过)。`gidx` 一律指 `games.npz` 里的行号。

| 文件 | 结构 | 说明 |
|---|---|---|
| `games.npz` | `names` (2020,) object | **gidx 基准**,顺序 = embedding_h5 的 `game_names` |
| `wiki_clean_views.npz` `wiki_llm_views.npz` `sp_raw_views.npz` | `S` (N, MAXLEN, 1024) fp16 零填充;`S_len` (N,) i32;`gidx` (N,) i64;`names` (N,) object | 全文档视图:SaT 分句(不设上限)+ 逐句嵌入。装载时做逐行标准化(rown)后送塔 |
| `wscan_pool_rev.npy` | (2020, 2048, 1024) fp16 | 评论池:每游戏 2048 句预算;**金矿保证** = 3 条最长评论无条件进入,其余整条随机填充;向量已 per-row mean0/std1 |
| `wscan_pool_rev_rid.npy` | (2020, 2048) i32 | 池内每句所属评论 id(-1 = 填充)——评论级拒绝采样的依据 |
| `wscan_pool_rev_len.npy` | (2020,) i32 | 每游戏实际占用句数 |
| `wscan_gal_rev.npz` | `gal` (2020, 512, 1024) fp16;`gal_len` i32;`gal_doc_len` i32 | **锚**:sp_raw 文档句在前、整条评论填满 512 预算;`gal_doc_len`=文档前缀长(0=无文档),头的文档门控据此屏蔽前缀 |
| `ss_queries_rev.npz` | `off` (8081,) i64;`gidx` (8080,) i64 | 伪查询索引:每游戏 4 条、锚形(文档前缀+整评论@512),扁平存储 |
| `ss_queries_rev_S.npy` | (Σ句, 1024) fp16,~9.5 GB | 伪查询句向量本体;查询 j = `S[off[j]:off[j+1]]`。**mmap 装载** |
| `wiki_eval.npz` | `S` (3256, MAXLEN, 1024) fp16;`S_len`;`gidx`;`names`;`variants` object | 评测查询 = 814 游戏 × 4 变体,**完整全文无截断** |
| `_tag_split.json` | train/val/test 游戏名列表 | 标签探针分割(seed 42,0.7/0.15) |

协议分割不在此目录:权威 `wiki_eval_split.json`(seed 20260711,
204/203/407)随代码入库在 `data_pipeline/` 下。
