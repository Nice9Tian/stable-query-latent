# data/ — 数据根目录

代码与数据严格分离:所有重资产都在这里,**除各 README 外全部 gitignored**。
四个子目录按"谁生产、谁消费"划分:

| 目录 | 放什么 | 谁写入 | 谁读取 |
|---|---|---|---|
| `corpora/` | 文本语料(wiki/sp) | `steam_reviews_framework/run.py` 自动解压内置压缩包 | `data_pipeline/build_assets.py` |
| `reviews/` | 评论重型文件(h5) | 按 URL 下载,或 `data_pipeline/reviews/` 重建 | build_assets + pod 全量池 |
| `assets/` | 训练/评测张量(npz/npy) | `data_pipeline/build_assets.py` | 训练器(framework/experiment) |
| `results/` | 检查点/投影/结果 json | 训练器 | `contrast_experiment/report.py` |

每个位置都可用环境变量重定向(零拷贝链接已有布局,见
`data_pipeline/paths.py`):

```
LARICE_DATA_ROOT  数据根(默认本目录)      LARICE_ASSETS    assets/
LARICE_CORPORA    corpora/                  LARICE_RESULTS   results/
LARICE_EMBED_H5   reviews/embedding_h5.h5   LARICE_TEXT_H5   reviews/text_h5.h5
```

一键准备:`python steam_reviews_framework/run.py --data-only`。
