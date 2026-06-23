# 整体模型 Pipeline 设计说明


## 1. 一句话概览

这个项目训练的是一个 **game review set encoder**：

```text
一个游戏的很多条评论句向量
  -> 随机采样两个评论子集
  -> 同一个 LatentArrayMLP encoder 编码
  -> VICReg 让两个子集表示一致且不坍缩
  -> GRL adversary 去除可被 SST 情绪头读出的情绪信息
  -> 得到一个固定大小的 game representation: (256, 18)
```

训练完后，再用一个单独的 **Tag 回归 probe** 检查这个 representation 是否能预测 Steam tags。

## 2. 总体结构图

```text
H5 dataset
  game -> reviews -> sentences -> 1024-d Qwen embeddings
        |
        | 每次训练，对同一个 game 独立采样两份 60% reviews
        v
  view A: selected review sentences, shape (S_a, 1024)
  view B: selected review sentences, shape (S_b, 1024)
        |
        | shared encoder: LatentArrayMLP
        v
  z_a: (256, 18)
  z_b: (256, 18)
        |
        +------------------------------+
        |                              |
        v                              v
  VICReg loss                    Sentiment GRL adversary
  - invariance                   - 18 -> 256 -> 1024 probe
  - variance                     - frozen SST MLP4 head
  - covariance                   - maximize sentiment uncertainty for encoder
        |                              |
        +--------------+---------------+
                       v
                 training loss
```

诊断/验证阶段：

```text
frozen encoder output: (256, 18)
        |
        v
flatten: 4608
        |
        v
TagRegressionHead
  4608 -> 256 -> 128
        |
        +--> presence logits: 219 tags
        |
        +--> count logits: 219 tags
```

## 3. 输入数据

训练不直接读原始文本，而是读取已经构建好的 H5：

```text
VICReg_review/h5/game_review_cleaned_3_sentences.h5
```

当前本地数据规模：

```text
games:      293
reviews:    407,421
sentences:  8,992,181
input_dim:  1024
dtype:      float16
```

数据组织逻辑：

```text
game
  -> many reviews
      -> many sentences
          -> each sentence has a 1024-d embedding
```

所以模型看到的是 sentence embedding 序列，而不是原始文本。

## 4. 双 view 采样机制

每个训练 step 会从一个 game 构造两个不同 view：

```text
view A = 随机抽取该 game 的 60% reviews
view B = 再独立随机抽取该 game 的 60% reviews
```

这里的 “60% random mask” 更准确地说是 **review-level random subset sampling**：

- 随机单位是 review。
- 被选中的 review 会保留其中所有 sentence embeddings。
- 不是 token mask。
- 不是 embedding 维度 mask。
- A/B 两个 view 独立采样，所以它们部分重叠、部分不同。
- 目标是让模型学到“同一个 game 的不同评论子集应该有稳定表示”。

这种机制相当于 self-supervised augmentation：

```text
same game, different review subsets -> should map to similar representation
```

## 5. Encoder: LatentArrayMLP

核心 encoder 是：

```text
LatentArrayMLP
```

输入一个 game view：

```text
(S, 1024)
```

其中 `S` 是这个 view 里所有被选中 reviews 的 sentence 总数。

输出固定大小表示：

```text
(256, 18)
```

也就是：

```text
256 个 latent slots
每个 slot 18 维
```

### 5.1 Encoder 内部结构

```text
sentence embeddings: (S, 1024)
        |
        v
LayerNorm(1024)
        |
        v
Linear(1024 -> 256)
        |
        v
context vectors: (S, 256)

learnable latent array: (256, 256)
        |
        v
CrossAttention
  query = learnable latent array
  key   = context vectors
  value = context vectors
        |
        v
latent vectors: (256, 256)
        |
        v
LayerNorm(256)
        |
        v
shared per-latent MLP
  256 -> 128 -> 64 -> 32 -> 18
        |
        v
final code: (256, 18)
```

默认参数：

```text
input_dim    = 1024
latent_dim   = 256
num_latents  = 256
num_heads    = 8
dropout      = 0.1
output_dim   = 18
reduce MLP   = 256 -> 128 -> 64 -> 32 -> 18
```

encoder 参数量约：

```text
638,514
```

### 5.2 为什么用 Latent Array

每个 game 的评论数量和句子数量都不同。如果直接把所有句子 flatten 或平均，会丢失很多结构；如果对所有句子做 full self-attention，长评论集合又会太贵。

Latent array 的作用是把任意长度输入压缩成固定长度表示：

```text
variable-length sentence set
        |
        v
fixed-size latent representation
```

复杂度大致是：

```text
O(sentence_count * num_latents)
```

而不是：

```text
O(sentence_count^2)
```

### 5.3 和 Perceiver 的区别

这个 encoder 借鉴了 Perceiver 的 latent-query cross-attention 思路，但不是完整 Perceiver。

相同点：

- 都使用 learnable latent queries。
- 都让 latent queries cross-attend 到输入特征。
- 都能处理可变长度输入并输出固定长度 latent set。

不同点：

```text
Perceiver:
  input -> cross-attention -> latent self-attention blocks -> possibly repeat -> decoder/query output

当前模型:
  input -> single cross-attention -> per-latent MLP -> output
```

当前模型没有：

- latent self-attention stack
- 多层 cross-attention
- Transformer residual block
- Transformer FFN block
- decoder query
- positional encoding

所以它更像一个轻量版的 **latent-query pooling encoder**。

## 6. VICReg 自监督目标

对同一个 game 的两个 view：

```text
view A -> encoder -> z_a: (256, 18)
view B -> encoder -> z_b: (256, 18)
```

VICReg 希望：

1. 两个表示相似。
2. 表示不能坍缩成常数。
3. 表示的不同维度不要高度冗余。

loss 形式：

```text
VICReg = 25 * invariance
       + 25 * variance
       +  1 * covariance
```

### 6.1 Invariance

```text
MSE(z_a, z_b)
```

让同一个 game 的两个评论子集编码一致。

### 6.2 Variance

对 `(batch, 256, 18)` 展平成：

```text
(batch * 256, 18)
```

要求每个 18 维 channel 有足够标准差，避免 collapse。

### 6.3 Covariance

同样在 `(batch * 256, 18)` 上计算 18x18 covariance matrix，惩罚非对角项。

这表示：VICReg 去相关的是最后 18 个 latent channels，而不是 4608 维 flatten 后的全部特征。

## 7. Sentiment GRL Adversary

除了 VICReg，训练还加入一个 adversarial objective：让 encoder 学到的 game representation 尽量不携带简单 sentiment 信息。

原因是 Steam review 里情绪强烈，如果 representation 过度依赖“好评/差评情绪”，可能不能学到更稳定的 game-level 语义。

### 7.1 Adversary 结构

encoder 输出：

```text
(batch, 256, 18)
```

把每个 latent vector 当成一个样本：

```text
(batch * 256, 18)
```

然后走 adversary：

```text
18-d latent vector
  -> Gradient Reversal Layer
  -> Linear(18 -> 256, bias=False)
  -> GELU
  -> Linear(256 -> 1024, bias=False)
  -> L2 normalize
  -> frozen SST sentiment head
  -> sentiment probability
  -> Bernoulli entropy
```

frozen SST head 是：

```text
1024 -> 128 -> 32 -> 8 -> 1 -> sigmoid
```

### 7.2 GRL 的含义

probe 的目标：

```text
尽量从 latent code 中恢复确定的 sentiment
```

encoder 的目标由于 GRL 被反转：

```text
让 probe + frozen SST head 读不出确定 sentiment
```

最终效果是希望 sentiment probability 接近不确定状态。

### 7.3 GRL schedule

默认不是一开始就强行对抗：

```text
epoch 0-5:   GRL lambda = 0
epoch 5-15:  linearly ramp to 1
epoch 15+:   GRL lambda = 1
```

这样 encoder 先学 VICReg 稳定表示，再逐渐加入 sentiment 去除压力。

## 8. 总训练 loss

训练阶段的总目标：

```text
total_loss = VICReg loss + adversary_weight * adversary_entropy_loss
```

默认：

```text
adversary_weight = 1.0
```

优化器：

```text
AdamW
lr = 3e-4
weight_decay = 1e-4
grad_clip = 1.0
```

被优化的参数：

```text
LatentArrayMLP encoder
adversary up-projection probe
```

不被优化的参数：

```text
frozen SST sentiment head
```

默认可训练参数约：

```text
encoder:                 638,514
adversary probe:          266,752
total optimized params:   905,266
```

## 9. 训练时的显存设计

默认使用：

```text
backward_mode = recompute
```

这是为了处理长 review set。

普通训练会保留所有 view 的 attention 激活，显存压力很大。当前实现先缓存 latent output，再在反向传播时逐 view 重新 forward，把梯度 replay 回 encoder。

简化理解：

```text
first pass:
  long view -> encoder -> latent code
  不保留完整激活

loss backward:
  得到 latent code 的梯度

replay:
  重新跑 long view -> encoder
  用 latent 梯度反传 encoder
```

这样牺牲一些计算时间，换更低显存占用。

## 10. Tag 回归 Probe

Tag probe 是一个诊断模块，不参与 VICReg 训练。

它回答的问题是：

```text
冻结 encoder 后，game representation 里是否包含可解码的 Steam tag 语义？
```

### 10.1 Probe 数据流

```text
game
  -> sample 8 random 60% review views
  -> frozen encoder encodes each view
  -> average 8 codes
  -> feature: (256, 18)
  -> TagRegressionHead
  -> tag predictions
```

`training.py` 默认每 5 个 encoder epoch 运行一次 probe。

### 10.2 当前 tag label

当前 tag vocabulary：

```text
num_tags = 219
target_mode = binary
source = Steam tags
min_count = 5 games
```

所以每个 game 的目标是一个 219 维 multi-hot tag vector。

### 10.3 TagRegressionHead 结构

输入：

```text
(256, 18)
```

flatten：

```text
256 * 18 = 4608
```

head：

```text
LayerNorm(4608)
Linear(4608 -> 256)
GELU
Dropout(0.1)

LayerNorm(256)
Linear(256 -> 128)
GELU
Dropout(0.1)

LayerNorm(128)
```

两个输出分支：

```text
presence branch:
  Linear(128 -> 219)
  -> sigmoid
  -> tag exists probability

count branch:
  Linear(128 -> 219)
  -> softplus
  -> expm1
  -> predicted raw tag count
```

当前 219 tags 时，TagRegressionHead 参数量约：

```text
1,279,286
```

### 10.4 Probe 训练目标

presence 分支：

```text
BCEWithLogitsLoss
```

使用 softened positive weight，避免稀有 tag 过度影响训练。

count 分支：

```text
SmoothL1 loss in log1p count space
```

总 probe loss：

```text
presence_loss + 0.1 * count_loss
```

评估指标：

```text
mAP
micro-F1
precision
recall
predicted tags per game
count MAE
count RMSE
```

Probe 只更新 TagRegressionHead，不更新 encoder。

## 11. 推理 / validation.py

`validation.py` 用训练好的 encoder 和 tag probe head 做文本到 tag 的预测：

```text
用户输入文本
  -> split into sentences
  -> local Qwen embedding
  -> sentence embeddings: (S, 1024)
  -> frozen LatentArrayMLP encoder
  -> code: (256, 18)
  -> TagRegressionHead
  -> tag probabilities + predicted counts
```

然后 UI 会：

1. 按 tag probability 排序展示标签。
2. 用预测 tag vector 和 games.json 中的 game tag vector 做 cosine similarity，展示最可能匹配的游戏。

## 12. 模块职责总结

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| H5 dataset | game review sentence embeddings | variable-length sentence set | 提供训练数据 |
| 60% review sampler | one game | two random review subsets | 构造 self-supervised views |
| LatentArrayMLP | `(S,1024)` | `(256,18)` | 把可变长度评论集合压缩成固定 game representation |
| VICReg | `z_a,z_b` | self-supervised loss | 保持同 game views 一致、防 collapse、去冗余 |
| Sentiment GRL adversary | latent vectors | adversarial entropy loss | 抑制 sentiment shortcut |
| TagRegressionHead | frozen `(256,18)` | tag presence/count | 诊断 representation 的 tag 可解码性 |
| validation.py | raw text | predicted tags / games | 使用训练结果做交互式验证 |

## 13. 最终理解

这个设计的核心思想是：

```text
不要直接监督模型预测某个标签；
先让模型通过同一 game 的不同 review 子集学习稳定 game-level 表示；
再用 VICReg 防止表示坍缩；
再用 GRL 减少情绪捷径；
最后用 tag probe 检查这个表示是否自然包含游戏类型/玩法/风格语义。
```

最终得到的 encoder 可以把任意数量的评论句向量压缩成一个固定结构：

```text
game code = 256 latent slots x 18 dims
```

这个 code 才是整个 pipeline 的核心产物。
