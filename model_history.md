# Model History — 模型总体视角

**本文件的定位（user decree 2026-07-16）**：模型的**总体视角**。塔结构、
训练协议（view 来源 / 锚构造）、损失、代号沿革——凡是"模型是什么、怎么训"
的事实都记在这里，且此后所有变更都要更新此文件。job 排布 / campaign 进度
的视角在 `task_list.md`；两者互补，不重复。

**铁律**：已经在 volume 上有结果文件的 arm 永远不改名（label 是
claim/result 的兼容契约）；只有从未跑过的 arm 才允许重命名。

---

## 1. 训练协议沿革（view 来源：两代协议）

### 第一代 · 本地协议（history；"两级采样"）

先从全量评论池给每个游戏**预采样 2048 个句子**（`wscan_pool_rev.npy`，
带 review id 的 `wscan_pool_rev_rid.npy`），训练时的 review view 再**在这
2048 句以内**抽整条 review 累积到 ≥W。即：全量池 → 采样池 → view 两级。
当年在本地机器上跑的就是这条路。代码仍保留：`FULL_POOL=False`
（w9_jobs.py 注释 "local-protocol parity"），worker 里走 `RIDp`/`POOL`
分支。

### 第二代 · w9 fp 协议（now；全量在线采样）

`--full-pool`（`FULL_POOL=True`，w9 campaign 默认）：view 直接从**全量
评论语料**在线采样——`full_pool_fp16.npy`（~150GB flat npy，驻留主机
RAM），每 view 整条 review 累积到 ≥W=16 句，句向量读取后逐行标准化。
结果文件带 **`_fp` 后缀**（如 `ft4var_w9_..._fp_best.json`）区分协议。

**两代协议的结果不可混比**；w9 结果族全部是 `_fp`。

### review view 采样器：**拒绝采样**（W 系列协议核心，2026-07-16 深查确认）

训练期 review view 用**长度偏置的拒绝采样**抽取**整条 review**（永不截断）：

```
提议:   i ~ Uniform(该游戏的全部 review)          # rng.integers(n)
拒绝1:  已取过的 review → 重抽（无放回）
拒绝2:  以概率 a(L_i) 接受，否则重抽
接受率: a(L) = 0.2 + 0.7 · (L − Lmin)/(Lmax − Lmin)   # 逐游戏归一
终止:   累计句数 ≥ W(=16) 或该游戏 review 取尽
退化:   全部 review 等长 → a ≡ 0.9
```

- **设计意图**（release/steam_reviews_framework/sampler.py 文档原文）：
  游戏内最长 review 90% 接受、最短 20%——"long gold-mine reviews stay
  reachable while short spam does not dominate"。
- **性质**：长度偏置是**游戏内相对的**（逐游戏用自己的 Lmin/Lmax 归一）；
  0.2 下限保证短 review 仍可达、0.9 上限保持随机性；W 是**下界**——最后
  一条接受的 review 会越过 W（整条进入，不截断），view 长度有方差；
  review 稀少的游戏可能取尽仍 <W（`taken.all()` 保证终止）。
- **三处同构实现**：`Pod/w9_a100_worker.py` `sample_views`、
  `Pod/w9_cv_worker.py`、`release/steam_reviews_framework/sampler.py`
  （canonical 文档版）。**两代协议都用它**（fp 全量分支和 2048 池分支
  构建同一张 (starts, lens, a) 表）；doc 槽的 review **回退**也走它。
- **不用拒绝采样的采样点**（对照，勿混淆）：
  - **锚 gallery 构建**：`rng.permutation` + 贪心装箱到 cap（sp doc 前缀
    先入，整条 review 随机序能装则装）——无长度偏置；
  - **pseudo-query（ss_queries）**：同 permutation+贪心（锚形状的独立
    第二视图）；
  - **旧协议 2048 句池的构建**：TOPK 最长 review **强制先入**（"gold
    guarantee"）+ permutation 贪心填充——所以旧协议下拒绝采样器面对的
    池子本身已偏向长 review；fp 协议下采样器面对全量 review；
  - **doc view**：整篇原样，无采样。
- **沿革**：VICReg 时代（第 0 代）**无拒绝采样**——按比例均匀无放回抽
  review（`rng.choice(n, size=ceil(fraction·n))`），且句子级截断
  （`limit_view_sentences` 均匀丢句，**会**打碎 review）。拒绝采样 +
  整条 review 不截断是 W 系列 wcle 协议引入的，并写进了 worker 头注
  （"rejection view sampler a(L)=... over WHOLE reviews (no truncation)"）。

## 2. 锚（anchor）侧：anchor_cap 的含义与 cap 阶梯

- **定义**：每个游戏的 gallery 锚行 = 用**最多 `anchor_cap` 个锚句**
  （store page doc 前缀先入，其后随机整条 review 填满）过同一塔 set-pool
  成的 128-d 向量。`@512` 指的就是这个 cap——**锚句预算，不是训练采样池**。
- **构建**：@512 的锚包本地预构建后上传 volume（`wscan_gal_rev.npz`，
  2020×512×1024，含 `gal_doc_len` doc 前缀长度）；@1024/2048/4096 由 pod
  现建（有预构建包则直接用）；**cap 超过 2048 池上限时锚源必须是全量语料**
  （worker 里有 assert，要求 `--full-pool`）。
- **训练期 vs eval 期 gallery**：训练 CE 的 gallery 只含 train pool
  （2020−407 val/test = **1613** 行，同塔**带梯度**每步重编码）；eval 用
  全部 2020 行。
- **cap 阶梯的角色**：`@512` = **筛选档**（结构消融 / loss ladder /
  view-composition grid 都在这跑，赢家晋升）；`@2048` = **champion /
  论文主线 cap**；scale grid = {512, 1024, 2048, 4096} @2000ep 单预算
  缩放曲线。

## 3. 共享塔结构（W 系列所有 CE/I 家族 arm 相同）

- **塔** `SetPoolN(4)`：4 个可学习 latent query（`q0`: 4×128）对 view 的
  句子集合做单层 cross-attention（`nn.MultiheadAttention`，embed 128-d、
  4 heads、kdim=vdim=1024），4 个 slot 输出取平均 → MLP head
  （128→256→GELU→256→128，无 BN）→ L2 归一化 → 128-d 部署嵌入。
  总参数 ≈0.36M。head **不可丢弃**（输出即部署空间，非 SimCLR 式
  projector；真正的可丢弃 projector 是 E-space arm：exp/cmp/pj）。
- **输入**：1024-d 句向量，进注意力前逐行标准化（减均值除标准差）。
- **损失**（plain i2ce 族；全部作用在同一个部署空间 z 上）：
  - CE：每个 view 独立 `cross_entropy(z @ gallery.T / τ)`，对所有 view
    求和；τ=0.02 **固定不学**（可学习温度只存在于 eval 侧 FT readout
    head，init 1/0.07）。
  - I：所有 view 两两配对 `1 − cos(z_i, z_j)` 取平均，权重 IW（i2ce=2）。
- **优化**：AdamW lr 5e-4 / wd 1e-4，batch 192，per_epoch 3072 样本，
  grad-clip 5.0，AMP。
- **变体开关**（举例）：`c` 后缀 = 训练期 centering（head 后、L2 前减
  μ-EMA，如 cegate2c/i2ccec）；`cegate<k>` 数字 = I 权重剂量；`mq<N>` =
  MoCo 队列 + 影子塔（权重 EMA m=0.99）；`bkq/bkb` = memory bank。

## 4. View 组成：显式命名语法（2026-07-16 起）

```
[d<k>][w<k>][sp<k>]r<n>_i2ce
```

| token | 含义 |
|---|---|
| `d<k>` | k 个 **tier 化 doc 槽**（协议槽：wiki → store page → review 三级回退） |
| `w<k>` | k 个 **wiki 专属槽**（只查 g2wiki；缺 doc 回退 review view） |
| `sp<k>` | k 个 **store page 专属槽**（全 sp 覆盖 `g2store_all`，**不排除**有 wiki 的游戏；缺 doc 回退 review view） |
| `r<n>` | n 个 review 随机采样 view（每 view 句子预算 W=16，fp 协议下从全量语料采） |

协议标准 `i2ce`（NV=4）== `d1r3`（所有非 grid arm 隐式如此）。
解析：`Pod/w9_a100_worker.py` 的 `vp_m` 正则
`(?:d(\d+))?(?:w(\d+))?(?:sp(\d+))?r(\d+)_i2ce$`。

**协议约定**：per-view-sum——CE 项数 = NV、I 边数 = C(NV,2)，view 数增加
时整个目标同步放大（这是 view grid 的已知混淆，非 bug）。

### doc 槽的数据门槛（构建期，全部 arm 共享）

- wiki：正文 ≥300 字符（`build_wiki_clean.py` MIN_CHARS；不足 = "thin"，
  该游戏视为无 wiki）；
- wiki/sp view 打包：丢 <10 字符碎句后整篇 ≥2 句才进 npz
  （`build_assets.py`）；
- 未过门槛 → 不进 g2wiki/g2store/g2store_all → 该槽训练时回退 review
  view。训练期对过槛 doc **无长度下限**（整篇原样用，mask 盖 padding）；
  唯一训练期长度开关是 `--doc-lead`（截断，上限方向）。

## 5. 代号沿革（now_name ↔ history_name）

| now_name | history_name | label（now） | 状态 |
|---|---|---|---|
| `d1r4_i2ce` | `r4i2ce` | `wcle_d1r4_i2ce_icetf` | 2026-07-16 改名；从未运行，零成本 |
| `d1r5_i2ce` | `r5i2ce` | `wcle_d1r5_i2ce_icetf` | 同上 |
| `d1r6_i2ce` | `r6i2ce` | `wcle_d1r6_i2ce_icetf` | 同上 |
| `w1sp1r3_i2ce` | `wspi2ce` | `wcle_w1sp1r3_i2ce_icetf` | 同上；旧名把 wiki+sp 压缩成 "wsp"，不显式 |

改名原因（user decree）：名字必须能读出 view 组成，每个 token 显式携带数量。

## 6. 当前 view-composition grid（@512 筛选档，fp 协议，2000ep）

| arm | view 组成 | NV | CE 项 | I 边 | 测什么 |
|---|---|---|---|---|---|
| `i2ce`（基线，=d1r3） | 3 review + 1 tier 化 doc | 4 | 4 | 6 | 协议标准（@512@2000ep 基线在 scale grid） |
| `nodoc`（基线） | 4 review + 0 doc | 4 | 4 | 6 | 恒定 NV 下 doc 槽的价值（注意：1000ep） |
| `d1r4_i2ce` | 4 review + 1 tier 化 doc | 5 | 5 | 10 | v_review 剂量 +1 |
| `d1r5_i2ce` | 5 review + 1 tier 化 doc | 6 | 6 | 15 | v_review 剂量 +2 |
| `d1r6_i2ce` | 6 review + 1 tier 化 doc | 7 | 7 | 21 | v_review 剂量 +3 |
| `w1sp1r3_i2ce` | 3 review + 1 wiki 槽 + 1 sp 槽 | 5 | 5 | 10 | wiki 与 sp **同场**（tier 协议里互斥）；sp 槽用 `g2store_all` |

对照关系：`w1sp1r3_i2ce` vs `d1r4_i2ce` 同 NV=5——第 5 个 view 是 sp doc
还是 review，隔离 sp 视图的边际价值；vs `i2ce`(d1r3) 隔离"+1 view"总效应。
@512 出效应 → 加 @2048 晋升行。

**Owning notebook**：`Pod/w9_viewgrid.ipynb`（w9_mq 模式；VRAM warmup 按
arm 计费——同 @512 但 NV 不同峰值不同；readout 参照行 = i2ce@512(d1r3) 与
nodoc）。同名行也在 `w9_jobs.FS_JOBS`（w9_l40 可认领），双路径靠 claim
文件互斥。

## 7. 读出协议：ZS-only（2026-07-16 起，user decree "不再 fine tune"）

- **主口径 = 裸塔 zero-shot**。FT readout head 整段退役为 `--head` 显式
  选通（默认关）；两个 worker（fs / cv）同步。
- **zs_traj**（每 50ep 一条）：test 侧全 4 变体 h1+h5、tag(neu/non)、
  **val 侧** neu/non 检索 + `zvsel`；旧 traj 缺 val 键时从投影 npz
  免 GPU 补算（`tower_{name}_ep{k}.npz` 存 SPg/SPa）。
- **选点**：`zsbest_{name}.json` = val 选点（平分取更早 ep）。
  - fs：`zvsel = max(S(v_non@1;0.45), S(v_non5@5;0.65)) + S(v_neu@1;0.85)`
    （头时代 vsel 分段分原公式移植；S_fn 低于目标指数罚、高于线性奖）。
  - cv：`cvsel = v_non@1 + v_non@5 + 2·v_non_tagF1`（user 公式，noname
    聚焦；val noname tag 用同一 anchor-ridge 在 val 查询上预测）。
- **完成标志 = `tower_{name}_fp_ep{N}.npz` 存在**（不再是 ft4var
  best.json——ZS-only 下后者根本不写）；notebook drain 全按此跳过，
  worker 对低预算完成的塔自动从最新 ckpt 续训（EXTEND 回退）。

## 8. 结构裁决与逐视图 CE 铁律（2026-07-17，30 格 @512 网格完赛）

- **结构最优 = 素面 i2ce**（部署态、逐视图 CE、无投影）：ZS
  .926/.657，tag .721/.745（@2000ep）。E 空间（exp/cmp/pj）、共享 E、
  双 E、池化——所有结构修饰零增益到灾难。缩放理论建在 i2ce 上。
- **铁律（user law，16/16 无例外）**：**池化 CE 是视图压制捷径**——池化
  只约束均值方向，编码器学会条件策略（信号进好认的视图、其余输出互相
  抵消的噪声），逐视图评测崩塌。pool 格均值 neu .572/non .261 vs 逐视图
  .877/.575；pool→E→CE 最惨（neu .31-.42，zvsel 至 −223）；tag 同步掉。
  **此后一切 arm 的 CE 必须逐视图**；`_CE_POOL` 处有法条注释；16 个
  池化格保留为论文反面证据。i2poolce@2048（唯一未跑池化行）已除名。

## 9. 锚供给变体：tag 保卫战（2026-07-17，`Pod/w9_save_4096_tag.ipynb`）

- **诊断**（数据定罪）：tag 随 cap 下降跟**锚供给机制**走、不跟 I 走——
  512 下 i2ce tag（.728）> ce（.702）；C 项无效（i2cce@2048 .706 ≈
  i2ce .707）；mq 队列同 cap/同 I/同 C 恢复 tag（.736）。凶手 = **锚共
  适应**：CE 梯度穿过 Zg（1613 锚每步带梯度重编），把锚嵌入朝"可分但
  离内容流形"方向游戏化，cap 越大锚包越平滑越可游戏（i2ce@4096 的 ZS
  tag 训练内单调衰减 .726→.709）。
- **三臂**（@4096/2000ep，ZS-only）：
  - `i2q2ce`：软带 I，`IW·(1−cos)²`——梯度比 = 2(1−cos)（cos=.5 等强、
    .9 时 5× 软、.98 时 25× 软），近对齐死区保个性；n 可推广为 q{n}。
  - `i2sgce`：**stop-grad gallery**，锚编码包 `torch.no_grad()`（**非
    .detach()**——detach 先建图再丢，前向峰值不降）。实测显存：4096
    45G→18G（一张 A100 可双塔）、**8192 89G→36G ⇒ 全库 8192 锚解锁**，
    `i2sgce@8192` 已入列（史上首个 full-gallery 8192 格）。
  - `i2esce`：**EMA 影子 gallery**（m=0.99，滞后 1/(1−m)=100 步≈6ep）。
    `USE_SHADOW` 旗标把 mq 的影子机械泛化（创建/EMA/ckpt/resume/extend
    七处，mq 行为不变）。动力学：不动点与 sgce 相同；CE 不管的子空间
    线性化特征值 {0, −(η+1−m)}——差模衰减、共模冻结（个性被保存）。
- **调度**：该 notebook 的 VRAM warmup 按 **(arm, cap)** 分测（同 cap
  下 grad 与 no-grad 臂峰值差 ~2.5×，按 cap 一刀切会误排）。
- **token 速查**（新增代号语法）：`q{n}` = I 的 (1−cos)^n 软带；`sg` =
  stop-grad gallery；`es` = EMA-shadow gallery。


## 10. N 包分形家族 pk{N}（2026-07-17~18，`Pod/w9_packageview.ipynb`）

- **设计**（用户）：锚预算切成 PKN 个连续包（@g1024 的 2×512 与
  i2ce@1024 锚**逐句相同** ⇒ pk-vs-joint 纯隔离聚合方式）。包层 i2ce =
  每包对均值画廊独立 CE（锚侧自带判别目标 = 反游戏化杠杆）+ 包间 I ×2；
  部署/评测画廊 = normalize(alive-mean(e₁…e_PKN))。视图层三态：`vce`
  纯逐视图 CE / `vi2ce` 完整 i2ce（视图 I ×2）/ `sgvce` 视图追 detach
  画廊（包层保梯度）。
- **裁决**（@1024/2000ep，rvsel）：`pk2i2cevi2ce` **non .716 = noname
  史上最高**（同句同预算完胜 i2ce@1024 .657 +.059）；vce .647、sgvce
  .574（qtag .694 最高）——视图 I 第三次也是最强一次被判不可摘除；
  视图↔锚双向共适应正是 noname 之源。
- **消融**（用户）：`pk2i2vce` = 包层只留 I×2（无包 CE，锚只靠视图 CE
  反传训练）——与 vce 成包 CE 单因子对。
- **容量阶梯**（用户 2026-07-18，全用冠军 vi2ce 配方）：pk2@2048
  （2×1024）/ pk2@4096（2×2048）/ **新臂 `pk4i2cevi2ce`@2048**（4×512）。
  pk2@2048-vs-pk4@2048 定预算只动包数；pk2@2048-vs-pk2@1024 只动包尺寸。
- **实现要点**：worker 硬编码 A/B 泛化为 PKN 列表（`pk(\d+)` 解析）；
  大 cap 下小游戏后段包可能**全空**（池 < k·包长）——防 NaN 双保险：
  空包注意力解锁 0 位（读零填充行，保有限）+ aliveW 把空包从均值/包
  CE/包间 I 全部加权剔除（pk2 全存活时与旧 eA/eB 代码**逐位相等**，
  冒烟已证）。
- **token 速查**：`pk{N}` = N 包分形；`v...` 段 = 视图层配方。

- **循环回填修正**（用户 2026-07-18）：欠填包不关闭——评论区间
  `[gal_doc, gal_len)` **正序绕回复制**直至整行填满 cap，每包永远满员
  （文档前缀不入循环：复制品会逃出 pack-0 的 nodoc 掩码泄进 noname 评
  测）。aliveW 机械保留为全存活保险丝（仅护极端"纯文档行"）。掩码在
  回填后重建。
