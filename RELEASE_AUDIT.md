# RELEASE_AUDIT.md — 全项目文件审计与可发布版重构蓝图

> 审计日期 2026-07-12。目标:把仓库整理为可发布版本 —— **模型框架与数据管线分离**,
> 提供两条复现路径(①本机重建数据→训练冠军模型;②本机重建数据→训练全部 R60 对照组合),
> 本机与 pod(ipynb)双端可训。模型文件夹将来可独立上 GitHub 作为通用框架。
>
> 术语:**当前设计 = R60 wcle 协议**(wiki 评测宇宙 814、分割 204/203/407、全归纳、
> 评论级采样、sp_clean+评论混合锚、CE 门控/I 剂量塔家族、vsel 三轴选择、检查点事后寻优)。
> 在此之前的一切(VICReg 36 维时代、PXI、SST、推荐头、旧融合/旧七臂)均为旧设计。

---

## 第一部分:逐文件/逐目录审计

图例:✅ 保留(当前设计) | ♻️ 迁移/改造后保留 | 📦 归档(旧设计,移入 archive/) | 🗑️ 可删 | ⚠️ 发布阻断项

### 1. 仓库根目录

| 文件 | 作用 | 判定 |
|---|---|---|
| `CLAUDE.md` / `AGENTS.md` | 助手工作约定 | ✅(发布版可精简) |
| `README.md` | 项目说明(内容停留在旧设计) | ♻️ 按新结构重写 |
| `requirements.txt` | 依赖 | ♻️ 需补齐(wtpsplit/h5py/sklearn/scipy 等) |
| `cloud_embedding.py` | HF TEI 云端嵌入客户端(数据管线用) | ✅ → data-pipeline |
| `latent_query_model.py` / `_v2.py` | 早期模型定义(PXI 时代) | 📦 |
| `colab_game_review_pipeline.py` | Colab 版评论管线(被 game_review_data 取代) | 📦 |
| `prepare_pods.ipynb` | 旧 pod 引导 | 📦 |
| `2077_*.txt` / `AO_*.txt`(8 个) | 手工试验文本(赛博朋克/AO 四变体,早期演示) | 📦 |
| `_tmp_cs_split.json` / `_tmp_ctrl_split.json` / `_tmp_fig_split.json` | 旧 tag 分割缓存副本 | 🗑️ |
| `model_review.md` / `review.md` / `presentation.md` / `speaker_note.md` / `game_review_bertopic_tuned.md` | 历史分析/讲稿 | 📦 → archive/docs |
| `tokenAPI.txt`(gitignored)/ `tokenAPI.template.txt` | 凭证模板 | ✅ |
| `.gitignore` / `.gitattributes` | — | ♻️ 需为新结构更新 |

### 2. `game_review_data/` —— 评论数据管线(数据侧地基,基本保留)

| 文件 | 作用 | 判定 |
|---|---|---|
| `prepare_kaggle_steam_reviews.py` | Kaggle 原始评论 → 清洗 | ✅ 路径①②的第 1 步 |
| `build_metadata.py` / `split_data.py` / `build.py` | CSV→JSON→分句→(嵌入) 编排 | ✅ |
| `embedding_data.py` | LocalEmbedder / CloudEmbedder(当前构建脚本仍在 import) | ✅ 核心 |
| `Build_new.py` / `build_1.py` / `build_2.py` / `combine.py` / `combine_shard.py` / `h5_corpus.py` | 全量嵌入 → embedding_h5 的分片构建/合并(pod 时代产物,仍是重建 h5 的唯一路径) | ✅ 需整并为单一入口 |
| `embedding_incloud.py` | 云端嵌入变体 | ♻️ 并入上者 |
| `enrich_steam_store_metadata.py` | 商店元数据补全(games.json 来源) | ✅ |
| `games.json` | 2020 游戏元数据(sp 语料源) | ✅ 数据 |
| `embedding_h5.h5`(164 GB)+ `.s3meta` | 全量 73.3M 句向量 + 评论边界 | ✅ 数据(重建产物,桶上有) |
| `text_h5.h5` + manifest / `text_h5_format.md` | 句索引 + 23-tag 标签 | ✅ 数据 |
| `run_game_review_bertopic.py` / `run_topic0_bertopic.py` / `bertopic_cache/` | BERTopic 主题分析(一次性探索) | 📦 |
| `note.txt` / `kaggle_storepage_data/` | 备忘/外部数据缓存 | 📦 |

### 3. `VICReg_review/` —— 混合区:当前语料管线 + 大量旧 VICReg 时代脚本

**✅ 当前设计(语料管线,→ data-pipeline/corpora):**

| 文件 | 作用 |
|---|---|
| `collect_wiki_descriptions.py` | 抓取维基页(924 篇源头) |
| `build_wiki_clean.py` | 章节过滤+名字校验+消歧义排除 → wiki_clean 814 |
| `wiki_variants_rewrite.py` | wiki 四变体(评测查询文本) |
| `wiki_llm_rewrite.py` | wiki_llm 逐句改写(泄露消融语料) |
| `build_sp_corpus.py` / `sp_neutral_gen.py` / `sp_llm_rewrite.py` / `sp_variants_gen.py` / `sp_rescue.py` | sp 六语料(raw/neutral/llm/pos/neg/noname ×1811) |
| `generate_text_variants.py` | LLM 改写共用客户端(`chat`/`SYSTEM_PROMPTS`)。**⚠️ 第 33 行硬编码 API key —— 发布阻断项,必须改为读 tokenAPI.txt** |
| `gen_store_neutral.py` | 858 店面中立改写(已被 sp_neutral 吸收,留作 provenance) ♻️ |
| `text_variant_eval.py` | anchor-ridge 标签评测 + tag 分割(当前协议 import 中) ✅ → 框架 eval |
| `pod_selfstop.py` | pod 自停硬化梯子 ✅ → 框架 pod |
| `tag_mapping.py` | 23-tag 映射 ✅ |
| 语料数据目录:`wiki_clean/ wiki_variants/ wiki_llm/ sp_raw/ sp_neutral/ sp_llm/ sp_positive/ sp_negative/ sp_noname/` + 各 manifest | ✅ 数据(重建产物) |
| `_deprecated/`(旧语料+无效页隔离区) | 📦 保留隔离,发布包不含 |
| 各 `*_run*.log` / `*_rewrite*.log` 等日志 | 🗑️ |

**📦 旧设计(VICReg 36 维时代全家,→ archive/vicreg_era):**
`model.py`(旧 VICReg 编码器)、`train_vicreg_review*.py`(×3)、`train_tag_probe.py`†、`sweep/` 与 `sweep_cloud.py`、`dual_probe.py`、`probe_*.py`、`pilot_realtext_*.py`(×6)、`realtext_grid_eval.py`、`eval_battery_worker.py`、`eval_gpu_pool.py`、`eval_stage_gate.py`、`get_champions_namerank.py`、`capability_vs_data.py`、`ceiling_diagnostic.py`、`identity_diagnostic.py`、`coarse_tag_test.py`、`disturbtion_embed.py`、`predict_text.py`、`mem_budget.py`、`oom_proxy.py`、`shard_cache.py`、`test_*.py`(×5,测的都是旧代码)、`build_content_filter.py`、`build_real_doc_h5.py`、`build_review_h5.py`、`real_doc_tag_eval.py`、`run_adv_sweep.py`、`run_data_view_sweep.py`、`h5/ heads/ tags*/` 旧产物目录。
† `train_tag_probe.load_frozen_encoder` 仍被个别旧脚本 import,当前协议不依赖。

### 4. `Pod/` —— 双时代并存

**✅ 当前(R60 战役):** `w9_all.ipynb`(唯一入口)、`w9_a100_worker.py`、`w9_cv_worker.py`、`h5_staging.py`;`w9_a100.ipynb` / `w9_cv.ipynb`(分体版,♻️ 可留作参考或并档)。

**📦 旧(VICReg 云扫荡时代):** `training.ipynb`、`setup.ipynb`、`run.ipynb`、`prepare_training.ipynb`、`auto_champions.ipynb`、`auto_stop_fulln.ipynb`、`fivefold_cv.ipynb` + `fivefold_worker.py`(旧协议 CV)、`eval_*.ipynb`(×3)、`clean_*.ipynb`(×3)、`check_paralle.ipynb`、`realtext_grid.ipynb`、`realtime_reader.ipynb`、`reconstruct_timings.ipynb`、`reload.ipynb`、`start_old.ipynb`、`verify_checkpoints.ipynb`、`prune_small_n.py`。

### 5. `PXIbench_test/`、`sst/`、`backheads/` —— 三个完整的旧实验

全目录 📦 归档:PXI 基准探针(早期)、SST 情感回归(嵌入质量侧证,RESULTS.md 有结论价值)、
推荐头时代(backheads)。它们对论文叙事无贡献,对框架发布是噪声;各自 README/RESULTS 随档保留。

### 6. `tools/`

`sync_runpod_artifacts.ps1` / `sync_results.ps1` / `find_runpod_h5.ps1` ✅(取回通路);
`h5Read_Tools.py` / `npy_reader.py` / `CSV_Reader.py` / `logging_tee.py` ♻️ 并入框架 utils;`download.txt` 🗑️。

### 7. ⚠️ 关键缺口:当前协议的核心代码在仓库之外(scratchpad)

以下文件是 R60 的**本机管线本体**,目前只存在于临时目录,**发布前必须迁入仓库**:

| scratchpad 文件 | 作用 | 迁移目标 |
|---|---|---|
| `sp_wiki_scan.py` | 协议 setup(分割/排除/采样器/画廊/模型) | 框架 `protocol.py` + `model.py` |
| `w9_cell.py` | 本机训练细胞(13 臂 + 头网格 + 寻优) | 框架 `train_tower.py`(与 pod worker 合一) |
| `build_pool_rev.py` | 池/锚/伪查询构建 | data-pipeline `build_assets.py` |
| `build_wiki_eval.py` | 评测查询嵌入 + 分割 | data-pipeline |
| `build_views_generic.py` | 语料 → 训练视图 npz | data-pipeline |
| `ef_local.py` | 冻结基线 + 锚成分评测 | 框架 `eval.py` |
| `fusion_cache/` 关键 npz(池/锚/视图/评测/伪查询/tag)+ `wiki_eval_split.json` | 协议资产 | `data/assets/`(gitignored,可由管线重建/从桶下载) |

同样注意:`w9_a100_worker.py` 与 `w9_cv_worker.py` 是 `w9_cell` 的两份带漂移拷贝——发布版应**合一为单一训练器**,本机脚本与 pod notebook 都只是薄壳。

---

## 第二部分:可发布版目标结构

```
larice/                                 (发布仓库,Latent Represent I-CE)
│
├── main_model/             ★★ 主模型 larice 本体(LariceTower)—— 将来独立上 GitHub 的就是这一层
│   │                          只有冠军塔,不带任何塔后头;任务无关
│   ├── model.py            LariceTower(I-CE + CE门控配方;参数化:q 数量 N、
│   │                       视图数 NV、tau(frozen 值/learnable 初值)、
│   │                       I 权重与门控函数、DM、heads、kdim)
│   ├── config.py           LariceConfig —— 一个 dataclass 生成一座冠军塔
│   └── README.md           张量协议 + 读出说明(见下)
│
│   ◆ 标准张量协议:一切输入前两轴固定为 [data, view] ——
│     x[B, V, S, D_in] + mask[B, V, S](S=集合元素数,D_in=上游嵌入维,
│     任何任务只要把样本嵌入摆成这个形状即可用);单视图任务 V=1;
│     输出 z[B, V, N×DM](N 槽直接 concat)。
│     I 不变性损失沿 view 轴收,CE 沿 data 轴收。
│     README 注明:名称召回类任务改用 pool(均值池化)读出效果更好。
│
├── steam_reviews_framework/          ★ Steam-reviews 任务绑定 —— 调用 model 完成任务
│   ├── backhead_name.py    名称召回 BackHead(两阶段:phase1 伪查询名 +
│   │                       phase2 wiki-neutral,ICEtf 微调;vsel 选优)
│   ├── backhead_tag.py     标签 BackHead(23-tag anchor-ridge 读出,
│   │                       m4/tag 指标)
│   ├── protocol.py         814 宇宙分割、归纳排除集、vsel 三轴选择公式
│   ├── sampler.py          评论级拒绝采样 a(L)、全量池/2048池、pad+mask
│   ├── anchors.py          sp_clean+评论混合锚、画廊(train/full/nodoc)
│   ├── data.py             语料/资产装载(h5、npz、mmap、GPU/RAM 布局)
│   ├── train.py            统一训练器(塔+检查点+断点续跑+头网格+事后寻优)
│   ├── eval.py             metrics4 / 裸塔ZS / 冻结基线 / 锚成分(SPg_nd)
│   └── train_champion.py   路径①入口:冠军配方(cegate2)一键复现
│
├── contrast_experiment/         ★ 全量实验启动器 —— 训练各种塔、完成对比
│   ├── contrast_models/    全部对照塔(纯CE / BYOL / ArcFace / 各门控与
│   │                       I 剂量变体…)—— 复用 model 的塔骨架,替换损失
│   ├── contrast_heads/     对照塔的头(CEtf / BYtf / ARCtf / 纯CE 两阶段等)
│   ├── arms.yaml           全部臂的声明式定义(臂 = 塔配方 + 门控 + 头配)
│   ├── run_all.py          路径②入口:13 臂固定分割 roster(本机或 pod 队列)
│   ├── run_cv.py           6 配方 × 5 折 CV
│   ├── report.py           横评总表 / 学习曲线 / 剂量曲线聚合
│   └── pod/                多机并行通路:w9_all.ipynb(暂存+认领+队列+
│                           审计+自停)、workers、h5_staging、pod_selfstop
│
├── dataset_builder/          数据重建 —— 与模型完全解耦
│   ├── reviews/            Kaggle评论 → 清洗 → 分句 → 全量嵌入 h5
│   ├── corpora/            wiki 抓取→清洗→四变体/llm;sp 六语料(读 tokenAPI)
│   └── build_assets.py     池/锚/伪查询/视图npz/评测npz/分割json
│
├── data/                   (gitignored)h5、npz、语料文本 —— 桶下载或管线重建
├── archive/                全部 📦 项(PXIbench_test/ sst/ backheads/ vicreg_era/
│                           old_pods/ docs/)
├── README.md               重写:两条路径 + 本机/pod 双端说明
└── requirements.txt / tokenAPI.template.txt / CLAUDE.md
```

**依赖方向(严格单向)**:`contrast_experiment → steam_reviews_framework → model`;
model 不 import 任何上层;steam_reviews_framework 不 import experiment。
职责边界:**model 只含冠军塔与配置(干净、无头)**;冠军塔在 Steam 任务
上的 BackHeads 归 framework;全部对照塔与对照头收在 experiment 的
contrast_models/ 与 contrast_heads/,不污染 model。将来独立发布时:
`model/` 原样成仓(通用框架);`steam_reviews_framework` 是它的第一个应用示例;
`contrast_experiment` 是论文复现包。
(命名注意:根目录旧 `backheads/` 目录是推荐头时代产物,归档进
archive/ 后与 `steam_reviews_framework/backhead_name.py`/`backhead_tag.py` 不冲突。)

**两条用户路径**(README 主线):

```
路径① 冠军复现:
  python dataset_builder/rebuild_data.py      # 或从发布桶直接下载 data/
  python steam_reviews_framework/train_champion.py    # cegate2: CE门控+I×2, vsel选优
路径② 全对照复现(R60 设计,不含任何旧设计):
  python dataset_builder/rebuild_data.py
  python contrast_experiment/run_all.py [--cv]   # 13 臂 + 可选 6配方×5折
pod 路径:开 pod → 跑 steam_reviews_framework/pod/w9_all.ipynb(同一套 train.py 内核,
  experiment 的臂清单通过 arms.yaml 注入)
```

---

## 第三部分:执行清单(建议顺序)

1. **⚠️ 安全第一**:`generate_text_variants.py` 硬编码 key 改为读 `tokenAPI.txt`;全库扫一遍确认无其它密钥;
2. **迁移 scratchpad 七件**入仓(это当前协议的单点故障——临时目录一旦清理,本机管线即失传);
3. **worker 合一**:w9_cell / w9_a100_worker / w9_cv_worker → `steam_reviews_framework/train.py` + 三个薄壳(本机 CLI、pod 固定分割、pod CV),消除三份拷贝漂移;
4. **建 `archive/`** 并移入全部 📦 项(git mv,历史保留);顺带把工作区里悬置的 ~1000 个 `text_variants_generated` 删除记录一并提交;
5. **数据/模型分离**:`data/` 统一收纳 + .gitignore 更新;`dataset_builder/` 收纳全部构建脚本;
6. **重写 README** + 补 `requirements.txt` + 两个入口脚本;
7. 待 R60 战役出全数据后,把冠军配置固化进 `configs/arms.yaml` 默认值,`steam_reviews_framework/` 拆分独立仓。

**风险提示**:第 2、3 步做完前不要清理 scratchpad;第 4 步 git mv 会让 pod 的旧 notebook 路径失效——先等当前战役跑完再动 `Pod/` 旧文件。
