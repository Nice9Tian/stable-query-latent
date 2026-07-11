# studable-query-latent — release

跨域鲁棒的游戏表征学习:冻结句嵌入之上的 SetPool 塔(I-CE + CE 门控),
在 Steam 评论域训练、在 wiki 改写查询域评测(R60 全归纳协议)。

## 三层结构(依赖严格单向:experiment → framework → model)

```
model/            通用模型包 —— 只有冠军塔,无任何塔后头,可独立成仓
                  张量协议 [data, view]:x[B,V,S,D]+mask → z[B,V,N×DM](concat)
sql_framework/    Steam 任务绑定 —— 调用 model:协议/采样/锚/统一训练器/
                  backhead_name(名称召回两阶段头)/backhead_tag(23标签)/
                  train_champion.py(路径①)/pod/(RunPod 通路)
sql_experiment/   全量对照 —— contrast_models(CE/BYOL/ArcFace/门控与剂量
                  变体)+ contrast_heads + run_all/run_cv/report
data_pipeline/    数据重建 —— reviews(Kaggle→清洗→分句→全量嵌入 h5)、
                  corpora(wiki 抓取→净化→改写;sp 六语料)、build_assets
data/             (gitignored)全部重数据:h5 / npz / 语料文本 / 结果
```

## 两条复现路径

```bash
pip install -r requirements.txt

# 路径① 本机重建数据 → 训练冠军模型(cegate2:CE门控 + I×2,vsel 选优)
python data_pipeline/rebuild_data.py
python sql_framework/train_champion.py

# 路径② 本机重建数据 → 训练全部对照组合(本次 R60 设计,不含旧设计)
python data_pipeline/rebuild_data.py
python sql_experiment/run_all.py            # 18 对照臂,断点续跑(冠军走路径①)
python sql_experiment/run_cv.py             # 可选:6 配方 × 5 折
python sql_experiment/report.py             # 汇总对照表

# pod 路径:在 RunPod 打开 sql_framework/pod/w9_all.ipynb(见其 README)
```

## 数据与代码分离

所有重资产在 `data/` 之下(或用环境变量把现有布局链接进来,零拷贝):

```
SQL_DATA_ROOT   数据根(默认 release/data)
SQL_ASSETS      训练/评测张量npz     SQL_RESULTS   检查点与结果json
SQL_CORPORA     语料文本             SQL_EMBED_H5  / SQL_TEXT_H5  评论h5
```

`data_pipeline/rebuild_data.py --check` 会逐层报告缺什么、用哪条命令补。
LLM 改写语料需要 `llmAPI.txt`(url=/token=/model=,gitignored)。

## 协议要点(R60,2026-07-11 定稿)

- 评测宇宙 = 814 篇净化 wiki 页;固定分割 seed 20260711 =
  204 test / 203 val / 407 train(`sql_framework/wiki_eval_split.json`,
  权威文件,随代码入库);
- **全归纳**:留出游戏的任何文本(评论/文档/伪查询/画廊负梯度)不进任何
  训练阶段;完整 2020 画廊只在冻结塔评测时使用;
- 查询 = wiki 四变体(neutral/noname/positive/negative)完整文本;
- 锚 = sp 商店文本前缀 + 整条评论 @512 句;评论级拒绝采样
  a(L)=0.2→0.9(金矿长评可达);
- 塔 1000ep 不早停,每 50ep 检查点,事后按 vsel(noname 双轴取优 +
  neutral 加性门)选最优检查点,冠军配置 10 seed 复验。
