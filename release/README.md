# larice — Latent Represent I-CE

跨域鲁棒的集合表征学习:larice 塔(I-CE + CE 门控;N 潜查询交叉注意力,
源自 SetPoolN 血统)构建于冻结句嵌入之上。
参考任务:在 Steam 评论域训练、在 wiki 改写查询域评测(R60 全归纳协议)。

## 三层结构(依赖严格单向:experiment → framework → main_model)

```
main_model/       主模型 larice 本体(LariceTower + LariceConfig)——
                  只有冠军塔,
                  无任何塔后头,可独立成仓;张量协议 [data, view]:
                  x[B,V,S,D]+mask → z[B,V,N×DM](concat)
steam_reviews_framework/    Steam 任务绑定 —— 调用 main_model:协议/采样/锚/
                  统一训练器/backhead_name(名称召回两阶段头)/
                  backhead_tag(23标签)/run.py+train_champion.py(路径①)
contrast_experiment/   全量对照 —— contrast_models(CE/BYOL/ArcFace/门控与剂量
                  变体)+ contrast_heads + run.py(全对照一键,--cv 五折)+ report
                  + pod/(RunPod 多机并行通路,见其 README)
dataset_builder/    数据重建 —— reviews(Kaggle→清洗→分句→全量嵌入 h5)、
                  corpora(wiki 抓取→净化→改写;sp 六语料)、build_assets
data/             (gitignored)全部重数据:h5 / npz / 语料文本 / 结果
```

## 两条一键复现路径

```bash
pip install -r requirements.txt

# 路径① 一键训练冠军模型(cegate2:CE门控 + I×2,vsel 选优)
python steam_reviews_framework/run.py

# 路径② 一键训练全部对照组合(18 臂;--cv 加跑 6 配方 × 5 折;末尾自动出对照表)
python contrast_experiment/run.py [--cv]

# pod 路径(多机并行全对照):RunPod 打开 contrast_experiment/pod/w9_all.ipynb
```

两个 run.py 分工:**框架的 run = 只训冠军;实验的 run = 训全部对照**。
它们共享同一套数据准备(自动执行、断点续跑):

1. **语料(内置压缩包,复现性优先)** —— `steam_reviews_framework/corpora_bundles/`
   里的 `wikipage.zip`(wiki_clean/variants/llm,814 游戏)与 `storepage.zip`
   (sp 六语料,1811 游戏)自动解压到 `data/corpora/`。**永远优先使用内置
   语料:不重新抓取 wiki、不重新过 LLM**——避免 wiki 后续被编辑、或清洗
   口径不一致导致结果漂移;
2. **评论重型文件** —— 缺失时按 `LARICE_EMBED_H5_URL`/`LARICE_TEXT_H5_URL`
   自动下载,或用 `dataset_builder/reviews/` 从 Kaggle 原始数据重建一次;
3. **张量资产** —— `dataset_builder/build_assets.py` 自动补齐缺失项。

## 数据与代码分离

所有重资产在 `data/` 之下(或用环境变量把现有布局链接进来,零拷贝):

```
LARICE_DATA_ROOT   数据根(默认 release/data)
LARICE_ASSETS      训练/评测张量npz     LARICE_RESULTS   检查点与结果json
LARICE_CORPORA     语料文本             LARICE_EMBED_H5  / LARICE_TEXT_H5  评论h5
```

`dataset_builder/rebuild_data.py --check` 会逐层报告缺什么、用哪条命令补。
LLM 改写语料需要 `llmAPI.txt`(url=/token=/model=,gitignored)。

## 协议要点(R60,2026-07-11 定稿)

- 评测宇宙 = 814 篇净化 wiki 页;固定分割 seed 20260711 =
  204 test / 203 val / 407 train(`steam_reviews_framework/wiki_eval_split.json`,
  权威文件,随代码入库);
- **全归纳**:留出游戏的任何文本(评论/文档/伪查询/画廊负梯度)不进任何
  训练阶段;完整 2020 画廊只在冻结塔评测时使用;
- 查询 = wiki 四变体(neutral/noname/positive/negative)完整文本;
- 锚 = sp 商店文本前缀 + 整条评论 @512 句;评论级拒绝采样
  a(L)=0.2→0.9(金矿长评可达);
- 塔 1000ep 不早停,每 50ep 检查点,事后按 vsel(noname 双轴取优 +
  neutral 加性门)选最优检查点,冠军配置 10 seed 复验。
