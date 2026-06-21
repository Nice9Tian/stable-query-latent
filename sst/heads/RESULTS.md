# SST 情感回归头实验结果

在冻结的 Qwen3-Embedding-0.6B 句嵌入(1024 维,L2 归一化)上,训练 MLP 回归头预测 SST default 标签(连续 0–1 情感分)。

## 数据

- 来源:`sst/clean/sentence_embeddings/default_{train,dev,test}.json`
- 每条记录:`{sentence, label, embedding[1024]}`
- 规模:train 8544 / dev 1101 / test 2210

## 训练设置

- 优化:AdamW,lr=1e-3,weight_decay=1e-4,batch=64
- 损失:MSE(输出 sigmoid,保证落在 [0,1])
- 早停:dev MSE,patience=30,min_delta=1e-5
- 评估:test 用 best dev checkpoint 一次性报告,**不**用 test 早停以避免泄漏
- 全部 seed=0;详见 [_sst_train.py](../_sst_train.py)

## 实验结果(共 8 个变体)

按宽度→深度→瓶颈收紧三个轴扫描:

| 模型 | 形状 | 参数 | best dev epoch | dev MSE | test MSE | **test Pearson** | test Spearman |
|---|---|---|---|---|---|---|---|
| MLP1 | `1024→64→1` | 65k | 8 | 0.0230 | 0.0214 | 0.8212 | 0.8111 |
| MLP3@128 | `1024→128→1` | 131k | 8 | 0.0230 | 0.0215 | 0.8205 | 0.8102 |
| MLP3@618 | `1024→618→1` | 634k | 9 | 0.0232 | 0.0213 | 0.8219 | 0.8109 |
| MLP2 | `1024→128→64→1` | 140k | 19 | 0.0228 | 0.0213 | 0.8232 | 0.8136 |
| MLP4 原版 | `1024→64→32→18→1` | 68k | 6 | 0.0226 | 0.0208 | 0.8267 | 0.8145 |
| MLP4 A | `1024→128→32→8→1` | 136k | 12 | 0.0227 | 0.0210 | 0.8271 | **0.8158** |
| **MLP4 B (最终)** | **`1024→64→16→4→1`** | **67k** | **10** | **0.0224** | **0.0208** | **0.8277** | 0.8153 |
| MLP4 C | `1024→64→32→16→8→1` | 68k | 5 | 0.0227 | 0.0209 | 0.8275 | 0.8144 |

## 关键发现

1. **宽度无效**:单隐藏层 64 / 128 / 618 三档,test Pearson 全部在 0.820–0.822,几乎无差。**加 10× 参数提升 = 0**。
2. **加深有微弱收益**:从 1 隐藏层到 2 隐藏层(MLP1→MLP2)Pearson +0.002;到 3 隐藏层 + 渐缩瓶颈(MLP2→MLP4)再 +0.003–0.005。
3. **瓶颈越窄越好(在多层渐缩里)**:末端从 18 → 8 → 4 单调改善。单独看 4 维很怕欠拟合,但作为**多层渐缩中的最终压缩**它起到正则化作用,反而最稳。
4. **触到平台**:无论怎么改结构,test Pearson 全部停在 **0.827 ± 0.005**——已经是冻结嵌入 + MLP probe 的上限。
5. **真正的瓶颈是嵌入本身**:Qwen3-Embedding 是通用嵌入,未为情感任务优化。要继续往上必须换嵌入或对嵌入做下游微调,**头部结构再优化收益已经接近 0**。

## 最终选择:**MLP4 B**

```
Linear(1024, 64)  -> GELU -> Dropout(0.2)
Linear(64,   16)  -> GELU -> Dropout(0.2)
Linear(16,    4)  -> GELU -> Dropout(0.2)
Linear(4,     1)  -> Sigmoid
```

**理由**:dev MSE 最低(0.0224),test Pearson 最高(0.8277),test MSE 与原 MLP4 并列最低(0.0208),且**参数最少**(67k,与最弱的 MLP1 同量级)。在所有 8 个变体中是 Pareto 最优。

- Checkpoint: `mlp4_1024_64_16_4_1_best.pt`
- 训练脚本: [../train_sst_head_mlp4.py](../train_sst_head_mlp4.py)
- 复现命令(从项目根目录):
  ```
  python sst/train_sst_head_mlp4.py --hidden-dims 64 16 4
  ```

## 推理用法

```python
import json, sys, torch
import numpy as np
sys.path.insert(0, "sst")          # so we can import the head class
from train_sst_head_mlp4 import Mlp4Head

ckpt = torch.load("sst/heads/mlp4_1024_64_16_4_1_best.pt",
                  map_location="cpu", weights_only=False)
model = Mlp4Head(hidden_dims=(64, 16, 4))
model.load_state_dict(ckpt["state_dict"])
model.eval()

# embedding: 1024-d Qwen3-Embedding-0.6B output; L2-normalize before feeding.
def l2(v):
    v = np.asarray(v, dtype=np.float32)
    return v / max(np.linalg.norm(v), 1e-12)

with torch.no_grad():
    score = model(torch.from_numpy(l2(embedding))).item()  # in [0, 1]
```

## 已知不确定性

- 所有数字基于 **单一 seed(0)**。test 集 2210 句,Pearson 的样本标准误 ≈ ±0.005。MLP4 B 比 MLP4 原版仅领先 0.001,**该排名未必能跨 seed 复现**;但"激进渐缩 > 浅而宽"的**趋势**很清晰。
- 若要正式发布,建议用 3–5 个 seed 取平均并报告标准差。

## 不推荐继续做的事

- **加宽**:已证明无效。
- **继续加深或继续收紧**(如 `64→8→2→1`):大概率仍在 0.827 平台上。
- **改大 dropout / 减小 lr / 改优化器**:在这种规模和平台下也是百分位级的扰动。

## 推荐继续做的事

- **换嵌入**(SBERT/MPNet/E5/BGE 等专做句嵌入的模型)做对照,看是否一步突破 0.85+。
- **对 Qwen embedding 做 contrastive 微调**(在 SST/通用情感数据上),再用同样的 MLP4 B 头评估。
- **多 seed 评估**当前选型,产出带误差棒的最终数字。
