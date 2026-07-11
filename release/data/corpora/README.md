# data/corpora/ — 文本语料

**来源:仓库内置压缩包**(`steam_reviews_framework/corpora_bundles/` 的
`wikipage.zip` + `storepage.zip`),由 `steam_reviews_framework/run.py`
自动解压到这里。**复现铁律:永远使用内置语料——不重新抓取 wiki,
不重新过 LLM**(wiki 会被编辑、LLM 改写非确定,重采会导致结果漂移)。

所有文件 UTF-8;`<appid>` = Steam appid,与 `assets/games.npz` 的
游戏名前缀对应(`<appid>_<GameName>`)。

## wikipage.zip 解出(每套 814 游戏)

| 目录/文件 | 结构 | 用途 |
|---|---|---|
| `wiki_clean/<appid>_<GameName>.txt` | 净化维基页:仅保留游戏内容章节,匹配校验通过 | 塔训练文档视图(clean 家族) |
| `wiki_variants/<appid>/{neutral,noname,positive,negative}.txt` | 四风格 LLM 改写,每篇 ≥300 字符 | **评测查询**(wiki_eval.npz 的文本源) |
| `wiki_llm/<appid>_….txt` | 逐句忠实改写(与 wiki_clean 同名) | 预训练泄露消融(*_wllm 臂) |
| `wiki_clean_manifest.json` | 每页保留章节/字数/匹配名次/判定 | 溯源 |

## storepage.zip 解出(每套 1811 游戏,`<appid>.txt`)

| 目录 | 内容 |
|---|---|
| `sp_raw/` | 商店页原文(清洗后)。**锚的文档前缀** + 塔的 sp 文档视图 |
| `sp_neutral/` | 中立改写 |
| `sp_llm/` | 逐句忠实改写 |
| `sp_positive/` / `sp_negative/` | 情感改写 |
| `sp_noname/` | 中立且抹除全部名字(专名换虚构词) |
| `sp_manifest.json` | 生成溯源 |

消费方式:`dataset_builder/build_assets.py` 用 SaT 分句(丢 <10 字符碎片,
**不设句数上限**)→ Qwen3-Embedding-0.6B 嵌入 → `assets/*_views.npz`。
