# contrast_experiment — 全量对照实验层

训练**除冠军外的一切塔**并完成对比(冠军走
`steam_reviews_framework/run.py`)。臂的声明式清单在
`contrast_models/roster.py`(18 臂:I-CE 家族 / 纯CE / ArcFace / BYOL /
CE 门控 I 剂量阶梯 / I 门控镜像 / 随机门与 nodoc 对照 / wiki_llm 泄露消融),
对应头配在 `contrast_heads/configs.py`。

```bash
python contrast_experiment/run.py                 # 一键:数据准备 + 18 臂 + 对照表
python contrast_experiment/run.py --arms ce byol  # 子集
python contrast_experiment/run.py --cv            # 追加 6 配方 × 5 折 CV
python contrast_experiment/report.py              # 单独重出汇总表
```

一切断点续跑:跑完的塔/头/折自动跳过,可随时中断重启。

## pod/ —— 多机并行通路(本层的加速器)

**全量对照 = 大量互相独立的作业(18 臂 + 30 个 CV 折次),天然适合
多机并行**。`pod/` 提供在 RunPod(或任何带共享网络卷的 GPU 云)上把
整个 roster 撒到 N 台机器同时跑的完整方案:

- **分布式存储共享数据**:所有机器挂同一个网络卷(等价于 S3 桶),
  资产只需准备一次(原子改名 + READY 标记,断点续传),结果也汇聚
  在卷上,任何一台机器都能出完整对照表;
- **多机原子认领**:每个作业一个 claim 文件(排它创建),一臂只会被
  一台机器认领;机器中途死掉,claim 超过 12h 自动过期被他机接管——
  **开 N 台机器 ≈ N 倍吞吐,互不踩踏,无需任何中心调度**;
- **本地暂存**:`h5_staging.parallel_copy` 把全量池多线程拷到本地盘/
  共享内存再 mmap(网络卷随机读极慢,这一步是 10 倍级提速);
- **断点续跑**:worker 每检查点落一份 rolling resume bundle,机器重启
  后从断点继续;
- **自动关机**:队列空且审计通过后 `pod_selfstop` 关机,杜绝空耗。

用法:每台机器各开一份 `pod/w9_all.ipynb` 顺序执行即可;最省钱的
姿势是先让第一台完成"ONE-TIME 资产准备"再开其余机器。作业清单在
notebook 第一格(与本层 roster 同一套臂定义)。

## 依赖方向

本层 → `steam_reviews_framework` → `larice`;对照塔复用 larice 的塔
骨架、替换损失(`contrast_models/byol.py`、`arcface.py`),对照头复用
框架的两阶段头机制、只换损失哲学(`contrast_heads/configs.py`)。
框架对本层不可见——删掉整个 contrast_experiment 不影响冠军复现。
