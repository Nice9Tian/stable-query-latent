# Task List

## 0\. 整体视野

* **论文主线 / champion 线**：`i2ce@2048`（全 gallery、2000ep、ZS-primary 读数）。scale grid（`w9\_scale.ipynb`）= {ce, i2ce, ai2ce} × {512/1024/2048/4096} @2000ep，验证"I-CE 随 cap 缩放"论点。
* **@512 的角色**：**筛选 cap**。30-cell 结构消融、loss ladder、gating 消融全部在 @512 跑——便宜、能多塔并行。@512 出效应的赢家再**晋升到 @2048** 做 champion-cap 确认行。
* **本 view-composition grid 遵循同一惯例**：4 个新 arm 排在 **@512**（不是 @2048）。所以它**确实是相对 @2048 主线的简化版/筛选版**——有意为之（2026-07-16 已确认）。若 @512 显出效应，下一步加 @2048 晋升行（届时归 big 类、上 A100）。
* **基线对齐**：scale grid 已把 `i2ce@512` 续到 **2000ep**——d1r3 基线在 2000ep 存在，新行用 2000ep 与它同预算可比。注意 `nodoc@512` 只有 **1000ep**（gating 时代），与它比较有预算错位，必要时 extend。

## 1\. View-composition 扫描（v\_review 剂量阶梯 + 双 doc 槽）— 代码完成 + smoke 通过，待 push/排队

**状态**：代码完成；静态检查通过；**本地 GPU smoke 通过**（2026-07-16，RTX 3060/cuda\_py 环境：导入真实 worker 模块，SetPoolN + 完整 CE/I 损失，NV=5/6/7 各 3 步 backward，峰值 548/632/715 MiB，损失有限）。语料在 pod 网络卷上，本地跑不了全保真 `--measure-vram`——它等价于真实运行的前 3 步，排队后自动覆盖。**尚未 push、尚未排队训练。**

**代号**：显式语法 `\[d<k>]\[w<k>]\[sp<k>]r<n>\_i2ce`（旧名 r4i2ce/wspi2ce 已废弃，映射与模型参数见 [model\_history.md](model_history.md)）。

**预算**：4 行全部 **2000ep**（job 表新增 per-job epochs 覆盖字段，见下）；其余 103 个待跑行不受影响（仍 FS\_EPOCHS=1000）。

### 要做什么

回答两个此前从未测过的问题（NV=4 一直是写死的协议常量）：

1. **v\_review 剂量**：review view 从 3 加到 4/5/6（doc 槽保留），noname/tag 怎么变？

   * 新 arm：`d1r4\_i2ce`（4R+1D, NV=5）、`d1r5\_i2ce`（5R+1D, NV=6）、`d1r6\_i2ce`（6R+1D, NV=7）
   * 对照：`i2ce@512`＝d1r3（**2000ep 已有**，scale grid）；`nodoc`（4R+0D，1000ep，预算错位注意）
2. **wiki 与 store page 同时在场**：tier 协议里两者互斥（`g2store` 剔除有 wiki 的游戏）。新 arm `w1sp1r3\_i2ce`（3R+1W+1SP, NV=5），sp 槽用**不排除** wiki 游戏的 `g2store\_all`；每个 doc 槽缺 doc 回退 review view。

   * 关键 A/B：`w1sp1r3\_i2ce` vs `d1r4\_i2ce`（同 NV=5——第 5 个 view 是 sp doc 还是 review）

已知混淆（代码注释已记录）：per-view-sum 约定下 CE 项数随 NV 涨（4→5/6/7），I 边数 6→10/15/21——测的是"整个目标随 view 数缩放"，不是单独的 I 剂量。

### 改了哪些文件

|文件|改动|
|-|-|
|`Pod/w9\_a100\_worker.py`|① `ARMS` 注册 4 个新 arm；② `\_IW` 加 4 项（2.0）；③ `vp\_m` 显式语法解析 → N\_DOC/N\_WIKI/N\_SP/N\_REV/NV\_ARM（其余 87 arm 隐式 d1r3 不受影响）；④ `g2store\_all` 映射；⑤ `assemble\_doc\_view` 加 `tiers\_` 参数；⑥ `pairs` 用 `NV\_ARM`；⑦ 训练循环按 N\_REV/N\_DOC/N\_WIKI/N\_SP 组装 view；⑧ 启动打印 `nv=X(nR+kD+kW+kSP)`，`sp<k>`+`--no-sp-view` 互斥断言|
|`Pod/w9\_jobs.py`|① `FS\_JOBS` 加 4 行 @512 clean 2000ep（small 类 → `w9\_l40.ipynb` 认领）；② job 元组扩到 8 字段：**第 8 位 = per-job epochs 覆盖**（None＝FS\_EPOCHS；label 不含 epochs——一个 label 只拥有一个预算，改预算走 extend）；`\_pad`/`fs\_label`/`run\_job` 相应适配|
|`Pod/w9\_viewgrid.ipynb`|**新建**（2026-07-16）：view grid 的专属 pod notebook（w9\_mq 模式：sync → RAM staging（含 wiki\_clean + sp\_raw 两个 view 包）→ full-pool staging → VRAM 调度 → readout → auto-stop）。两处适配：warmup 按 **arm** 计费（同 @512 但 NV=5/6/7 峰值不同）；readout 参照行 = i2ce@512（d1r3, 2000ep）与 nodoc（1000ep，标注预算错位）。label/claims 与 FS\_JOBS 同名——与 w9\_l40 双路径靠 claim 互斥，不会重跑|

**未动**：`release/contrast\_experiment/pod/` 镜像（release 快照；pod 会 force-sync origin/main）。BCE/BYOL/epdb 路径的 `NV` 原样（那些 arm 永远 NV=4）。

### 下一步（按顺序）

1. **提交并 push**（pod notebook 从 origin/main 同步）。
2. 开专属 pod 跑 `Pod/w9\_viewgrid.ipynb`（推荐：per-arm VRAM 打包 + 专属 readout + auto-stop）；或让 `w9\_l40.ipynb` 的正常 drain 认领（epochs=2000 由 job 表第 8 字段下发）。两路 claim 互斥。开跑后第一分钟看启动行：`nv=7(6R+1D)` / `nv=5(3R+1W+1SP)`。
3. 读数对照表：剂量曲线 `d1r3(=i2ce@512@2000ep)` → `d1r4` → `d1r5` → `d1r6`；结构对 `w1sp1r3` vs `d1r4`（同 NV）+ vs `i2ce`/`nodoc`。重点：noname\_h1/h5 与 noname\_tagF1 是否随 v\_review 同向。
4. @512 出效应 → 加 @2048 晋升行（big 类，A100；届时再进 job 表）。

\---

## 2\. 遗留观察（讨论产生，未开工）

* **EMA 影子塔 × 全 gallery i2ce@2048**（"shadow-gallery" arm）：等 `mq3072i2ce` 4-cap 读数 + `bkq192i2cce` 读数出来再裁决——bkq（在线快照、无梯度、全覆盖）vs mq（EMA、queue 子集）可拆开"tag 保住是因为 EMA 还是因为锚不带梯度"。先行零成本测试：对现有 i2ce@2048 checkpoint 做事后权重平均（SWA 式）重跑 tag readout。
* **论文方法段勘误**（贴文 vs 代码）：two views→4 views（3R+1D）；learnable τ→训练期固定 τ=0.02（可学习 logt 只在 FT readout head）；2,020-way→训练期 1613-way（train pool）；300 epochs→1000/2000；"raw sentence embeddings"→逐行标准化后入注意力。

