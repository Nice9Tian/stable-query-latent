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

- **随机锚包修正**（用户 2026-07-18，取代循环回填）：包 = 对该游戏锚句
  集的**独立种子化随机 h 抽样**（`PK_SEED=20260718`+游戏号 → 逐位复
  现；**包间允许重合**——用户否决互斥连续切片）。抽样 = 评论区间
  `[gal_doc, gal_len)` 上平铺 randperm：池够无放回、池短平衡绕回 ⇒
  包永远满员，欠填/回填问题整体消失。文档前缀仍钉死 pack-0 头部不入
  抽样。包按索引现场 gather（`pk_sg`），零 cap 级张量复制。纯文档退
  化行照旧 aliveW 剔除。**注意**：已出分的三座 @1024 塔（.716/.647/
  .574）训练于旧的互斥连续切片——与随机包阶梯对比时带构造差。

- **ranAn 撤销 + 缓存借用澄清**（用户 2026-07-18）：@1024 包布局 = 文档
  前缀 + 整评论按种子随机接纳序排列 ⇒ 旧连续对半**本就是随机互斥划分**，
  新抽样仅增"可重合 + 句粒度"两点——构造差可忽略，`_ranAn` 重训格撤销，
  已出分三塔继续可比。锚集借用无需任何改动：pk@cap 与 i2ce@cap 走同一
  确定性构建路径（同种子/同语料/同代码，日志证 i2ce@4096 为现场构建，
  桶上 g4096 预构建 npz 从未被 stage 过）⇒ 每个 cap 上 pk 包抽样自
  i2ce 同款锚句集，逐位同源。

## 11. SWIN 滑窗新鲜锚场（用户 2026-07-19，`Pod/w9_swin.ipynb`）

- **动机链**（三具尸体+一个活口定出的设计空间）：bkq/bkb证明**缓存快学
  生的行 = 场不连贯必死**（刷新多快都没用，bkq192 周期 8.4 步照死）；
  i2sgce 证明零滞后免梯度自我目标死；bce 证明"全新鲜小场"存活（tag
  .738）但**无锚**混杂了覆盖税与失锚税。SWIN = 第一台能干净测覆盖率
  价格的仪器：锚保留、全新鲜、全对称、noname 梯度边全开、零缓存。
- **语义** `swin{W}step{S}loop{L}i2ce`：锚目录为环序列；每优化步跑 L 个
  微通道，第 l 通道新鲜编码 ring[p+lW, p+(l+1)W) 的 W 个游戏（带梯度），
  与**永远在场**的当前 batch 锚拼成本通道独立 CE 分区（softmax 各自分
  区；窗口列与 batch 重复者掩掉）；**逐通道立即 backward**（retain 共享
  视图图，窗口激活随手释放 = 计算量换显存）；I×2 走通用视图链一次；步
  末 p += S·W（ckpt/resume/extend 携带指针）。每步覆盖 (bs+LW)/N。
- **首格** `swin168step1loop2i2ce@4096`：覆盖 33%/步，全环 9.6 步一扫
  （相邻步 2× 重叠），预估 ~25-30G vs 全价 45G；将来 N 上万时窗口成本
  不随 N 涨。冒烟：分块 backward 梯度与单图**逐位相等**、去重掩码、绕
  环全覆盖、8-cell notebook。参照系 = i2ce@4096(.750)/i2q2ce(.716)/
  i2esce(tag .741)。
- **token 速查**：`swin{W}` 窗口游戏数；`step{S}` 指针步进（单位 W）；
  `loop{L}` 每步微通道数。

- **swin 语义终案**（用户 2026-07-19，ce90ea7）：`step{S}` = **每微通道
  滑动的游戏数**（丢 S 旧引 S 新；通道 l 窗口起点 = p+l·S，相邻通道重叠
  W−S；步末 p += L·S）。原始设想 step=1 逐游戏滑动、loop=绕环圈数
  （1613×2=3260 次反传/步）被用户否决为计算天价。首格改立
  `swin168step84loop2i2ce@4096`：半窗滑动链，每环位恰被 2 通道覆盖，
  全环 10 步一扫，覆盖 27.5%/步，每步窗口计算量与旧解读相同。

## 12. 容量/读出网格裁决（2026-07-19 收分，`Pod/w9_scale_query.ipynb`）

- **slot 轴平坦**：slot{4,8,16}×mean 的 non = .657/.662/.652（噪声带），
  m4z .862/.857/.857，tag 以 slot4 最高（.721/.745）。容量不是瓶颈，
  SST"换嵌入别加头"先例在 w9 复现；单层交叉注意力 + 4 槽 + mean 池
  被确证为正确复杂度点，加槽/加层议案归档。
- **learned linear pool 判死**：三 line 格全败于 mean 对照——slot4line
  .745/.451、slot8line **.382/.162（近全灭）**、slot16line .912/.588。
  初始化=精确均值仍挡不住 SGD 漂移；槽间 mean 对称先验是承重墙。
  "可学灵活性=捷径"家族律第三案（前两案：pooled CE、锚共适应）。
- **选点视界观察**：slot8line 的 rvsel（3.503→3.518）全场最高而 test
  已死——val 评论伪查询（域内）看不见改写域崩塌；rvsel 只用于塔内选
  点无恙，但跨域鲁棒性必须看 test 改写列（report-only）才能审出。

- **第二波收分**（2026-07-19，rvsel 同尺;i2ce@4096 rvsel=.706 非旧 .750）：
  ①包 CE 单因子对：`pk2i2vce`.603 vs `pk2i2cevce`.647 ⇒ 包 CE = 真器官
  （+.044 non），锚侧自持分离压力不可省。②包数轴证伪：`pk4i2cevi2ce`
  4×512（g2048）non .676 —— 输冠军 pk2 2×512（.716，半预算）.040，与
  i2ce@2048 联合池恰平（.676）⇒ 结合 slot 轴平坦，pk2@1024 的 +.059
  是"双包结构+512 健康包长"红利，非聚合容量。③头条：pk2 2×512 ≈
  i2ce@4096（.716 vs .706，1/4 锚预算）。④pk2 2×2048（g4096）从未跑
  成：单卡三塔叠 78.8G OOM 启动即死（claim 已清，重开 packageview 会
  独跑它）；它是"双包结构是否在 4096 成立"的缺席法官。

## 13. 第二波 5 折 @4096（用户 2026-07-19，`Pod/w9_experiment_5fold_2.ipynb`）

- **动机**：pk2 每包仍是 4 槽——聚合容量在 4096 的问题由
  `slot8i2cemean@4096` 直接裁决（slot 名义臂映射为普通 i2ce 损失 + 8 槽
  塔，NSLOT 从臂名解析）；同场补齐四基线的 5 折：`mq3072i2ce`（I2CE 的
  MoCo 队列版）/`mq3072ce`/`byol`/`epdb_v20i10c20`（VICReg epd，
  batch=all；byol/VICReg 训练侧不碰锚，4096 只作用于评测画廊）。
- **规模**：5 臂 × 5 折 = 25 塔 @2000ep，seed=fold 与第一波 ce/i2ce
  配对；readout 引第一波 ce/i2ce@4096 五折为参照行。
- **CV worker 移植**：MoCo 影子塔+环（prefill/供给/CE/EMA/ckpt/resume/
  EXTEND 七处，fs 语义逐字）；train_vicreg（epd 接线，VICReg_review.model
  依赖）；epd 分派；train_byol 补 measure-vram（否则 warmup 测不出成本
  永不调度）；warmup 改按【臂】计量（slot8≈45G，mq/byol/epd 远轻）。

## 14. SWIN 首格裁决（2026-07-19 收分）：覆盖律与剂量律双成立

- `swin168step84loop2i2ce@4096` zsbest(rvsel ep1150)：neu .941 / **non
  .691**（轨迹峰 .711@1250）/ m4z .882 / **tag .732/.733** —— 对全价
  i2ce@4096（.706 / .696/.706）：noname 噪声级之差，tag 反超 +.03。
- **覆盖律**：27.5%/步的新鲜对称带梯度覆盖 ≈ 全目录分离压力（对照同
  省成本的 es .613 / mq .578 —— 省法不同命不同：缓存/慢场丢检索，轮转
  新鲜窗不丢）。i2ce 全价负样本场浪费 ~2/3 算力。
- **剂量律**：每锚 ~1/3 步数暴露于视图 CE 反传 → 锚共适应稀释，tag 逼
  近 EMA 双王（es .735/.741）而不付其 noname 税。noname-tag 同旋钮的
  "中间点"首次实测，两头皆近优 ⇒ 4096 档新两全王。
- 工程：~25-30G（vs 45G），成本与目录规模解耦（万级路径已验证）。

## 15. τ 扫描（用户 2026-07-19，`Pod/w9_i2ce_t.ipynb`）

- **动机**：τ=0.02 开局冻结至今从未调过（臂后缀 `tf` = tau-frozen）；
  它是剂量旋钮第三轴（单步 CE 锐度，×50 的极尖 softmax 把雕刻集中在
  最难负样本上）——tag 案未审共犯。判决问题：软 τ 是沿 non-tag 前沿
  滑动（τ 无罪），还是存在 τ* 同时 non≥.70 tag≥.73（τ = 比 swin 更
  便宜的剂量阀）。
- **网格**：{`i2cet05`(τ=.05), `i2cet10`(τ=.10), `i2ce_icetl`(可学 τ)}
  × 5 折 × {512, 4096} = 30 塔@2000ep；τ=.02 参照 = 第一波 ce/i2ce 折。
  可学 τ：log 参数化、init .02、无 wd（wd 会把 τ 拉向 1）、inv_t 夹
  [5,200]、进优化器与梯度裁剪、ckpt 时打印当前值——顺带回答"模型自己
  想要多尖"。
- **实现**：cv worker τ 从臂名解析（`i2cet{NN}` → NN/100；`_icetl` 后
  缀 → 可学）；v4doc 全部 5 处 CE logit 位换 `_invt()`（固定臂返回常
  数，tl 臂返回 exp(log_invt) 张量）；bundle 携带 log_invt。

## 16. 覆盖扫描第一果（2026-07-19 收分）：膝盖 ≤19.7%，swin336 共驻 OOM

- `swin84step42loop2i2ce@4096`：non **.691**（与 swin168 的 .691 逐位
  相等），tag .727/.726（略低于 swin168 的 .732/.733）。**19.7% 与
  27.5% 对 noname 零边际差**——覆盖律膝盖 ≤19.7%，比先前估计更靠左；
  tag 仍随覆盖降低小幅上升，方向与剂量律一致。
- `swin336step168loop2i2ce@4096`：**OOM 未出分**——非训练中途崩溃，是
  启动第一步 `eNow` 编码即撞墙（锚建好、加载后 18.6G，共驻另两进程已
  占 38.59G+39.30G≈78G，几乎无余量）。与 pk2@4096 那次同病：**多塔
  共卡估算未按实测峰值排布**——教训再记一遍，调度前必须用已测峰值
  （swin168≈？待补测）而非猜测值分配显存。

## 17. swin 五折翻案（2026-07-20 收分，`w9_swin_5fold`）：两律降级

- 逐折配对（@4096，rvsel）：noname swin .702±.034 vs i2ce **.730±.036**，
  配对差 **−.028±.029（0 胜 2 平 3 负）**；tag 完全打平（.712/.712 vs
  .712/.718）；neutral/m4z 亦平或微逊。
- **修正一：覆盖律"27.5%≈全价"被推翻**——全价负样本场在折级稳定多付
  +.028 noname；固定分割上的 .691≈.706 与 swin84=swin168 逐位相等均属
  单分割噪声假饱和（后者待折验降级）。
- **修正二：剂量律 tag 红利未复现**——固定分割 +.03 在五折归零；es/mq
  的 tag 拯救链（多塔多 cap）仍立，但"1/3 剂量白拿 tag"是单分割幻影。
- **swin 新身份**：省 ~40% 显存、tag 打平、付 −.028 noname 税的 i2ce
  ——工程折中格，非两全王。i2ce@4096 折级 non=.730（固定分割 .706 低
  估了它）。
- **方法论**：折间 std ±.034 ⇒ 固定分割上 <.03 的差异不可读；此前
  ±.01~.02 级排名一律待折验。CV 产物名带 `_fp` 后缀
  （zsbest_w9cv_..._g4096_fp.json），探查勿再漏。

- **覆盖扫描收官**（2026-07-20，swin336 重跑成功 ep700）：19.7%/27.5%/
  43.1% 三点 non=.691/.691/.686、tag=.727/.733/.715——全段平台，固定分
  割分辨率（±.03）下覆盖率无可读效应；swin336 双倍窗口算力零收益。
  swin 家族终局画像：vs 全价 = 折级 −.028 non、tag 平（五折裁决）；
  家族内部 = 窗口大小无关紧要 ⇒ **要用就用最小窗 swin84**（~22G）。
  "膝盖"定位需逐点上折，成本不值，此研究线收官。
