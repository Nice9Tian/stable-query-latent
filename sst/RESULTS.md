# SST 情感回归头实验结果

在冻结的 Qwen3-Embedding-0.6B 句嵌入(1024 维,L2 归一化)上,训练 MLP 回归头预测 SST default 标签(连续 0-1 情感分)。

本文件记录最终 10-seed 平均结果。早期 single-seed 结果曾显示 MLP4 B 最好,但扩展到 seeds `0..9` 后,MLP4 B 的稳定性不足;最终推荐改为 MLP4 A 或更轻量的 MLP4 original。

## 数据

- 来源:`sst/clean/sentence_embeddings/default_{train,dev,test}.json`
- 每条记录:`{sentence, label, embedding[1024]}`
- 规模:train 8544 / dev 1101 / test 2210
- 输入预处理:每条 embedding 做 L2 normalization

## 训练设置

- 优化:AdamW,lr=1e-3,weight_decay=1e-4,batch=64
- 损失:MSE(输出 sigmoid,保证落在 [0,1])
- 早停:dev MSE,patience=30,min_delta=1e-5
- 评估:test 用 best dev checkpoint 一次性报告,不使用 test early stopping
- dropout:0.2
- seeds:`0,1,2,3,4,5,6,7,8,9`
- 总训练次数:8 models x 10 seeds = 80 runs
- 训练脚本:参见 `train_sst_head_mlp{1..4}.py` 和 `_sst_train.py`

## 10-Seed 平均结果

按 test Pearson 均值从高到低排序。数值为 mean +/- sample std。

| 排名 | 模型 | 形状 | 参数 | dev MSE | best epoch | test MSE | test Pearson | test Spearman |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | MLP4 A | `1024->128->32->8->1` | 136k | 0.0227 +/- 0.0002 | 6.5 +/- 3.4 | 0.0210 +/- 0.0001 | **0.8264 +/- 0.0012** | **0.8144 +/- 0.0012** |
| 2 | MLP4 original | `1024->64->32->18->1` | 68k | 0.0227 +/- 0.0001 | 6.8 +/- 1.8 | 0.0210 +/- 0.0001 | 0.8263 +/- 0.0012 | 0.8139 +/- 0.0015 |
| 3 | MLP4 C | `1024->64->32->16->8->1` | 68k | 0.0228 +/- 0.0001 | 5.6 +/- 2.1 | 0.0211 +/- 0.0002 | 0.8255 +/- 0.0014 | 0.8129 +/- 0.0018 |
| 4 | MLP4 B | `1024->64->16->4->1` | 67k | 0.0229 +/- 0.0006 | 11.4 +/- 4.5 | 0.0212 +/- 0.0006 | 0.8248 +/- 0.0050 | 0.8129 +/- 0.0035 |
| 5 | MLP2 | `1024->128->64->1` | 140k | 0.0229 +/- 0.0001 | 9.5 +/- 4.5 | 0.0213 +/- 0.0003 | 0.8234 +/- 0.0016 | 0.8140 +/- 0.0013 |
| 6 | MLP1 | `1024->64->1` | 65k | 0.0230 +/- 0.0000 | 8.9 +/- 2.0 | 0.0214 +/- 0.0002 | 0.8220 +/- 0.0013 | 0.8115 +/- 0.0012 |
| 7 | MLP3@128 | `1024->128->1` | 131k | 0.0231 +/- 0.0001 | 9.7 +/- 4.0 | 0.0214 +/- 0.0002 | 0.8219 +/- 0.0017 | 0.8113 +/- 0.0016 |
| 8 | MLP3@618 | `1024->618->1` | 634k | 0.0233 +/- 0.0001 | 11.1 +/- 4.8 | 0.0214 +/- 0.0003 | 0.8218 +/- 0.0022 | 0.8112 +/- 0.0021 |

## 关键发现

1. **最终推荐:MLP4 A 或 MLP4 original。** MLP4 A 的 10-seed 平均 test Pearson 最高(`0.8264 +/- 0.0012`),test Spearman 也最高(`0.8144 +/- 0.0012`)。MLP4 original 几乎并列(`0.8263 +/- 0.0012`),参数约为 MLP4 A 的一半,是更轻量的稳健选择。
2. **single-seed 最优不稳定。** MLP4 B 在旧 single-seed 结果中 Pearson 最高(`0.8277`),但 10-seed 平均降到 `0.8248 +/- 0.0050`;其中 seed 2 掉到 Pearson `0.8110`,导致方差明显偏大。
3. **单层加宽无收益。** MLP1、MLP3@128、MLP3@618 的 10-seed 平均 Pearson 都在 `0.8218-0.8220`;把 hidden 从 64 加到 618 没有带来有效提升。
4. **适度加深有效。** MLP2 相比单隐藏层提升到 Pearson `0.8234`,MLP4 系列进一步提升到 `0.8255-0.8264`。
5. **总体平台仍在 0.826 左右。** 10-seed 后最佳均值没有突破 `0.827`,支持“冻结 Qwen embedding + MLP probe 已接近上限”的判断。进一步提升更可能来自更适合情感任务的 embedding 或下游微调,而不是继续堆回归头。

## 最终选择

如果优先追求平均指标:

```text
MLP4 A: 1024 -> 128 -> 32 -> 8 -> 1
Linear(1024, 128) -> GELU -> Dropout(0.2)
Linear(128, 32)   -> GELU -> Dropout(0.2)
Linear(32, 8)     -> GELU -> Dropout(0.2)
Linear(8, 1)      -> Sigmoid
```

复现命令:

```powershell
python sst/train_sst_head_mlp4.py --hidden-dims 128 32 8
```

如果优先追求参数效率和稳定性:

```text
MLP4 original: 1024 -> 64 -> 32 -> 18 -> 1
Linear(1024, 64) -> GELU -> Dropout(0.2)
Linear(64, 32)   -> GELU -> Dropout(0.2)
Linear(32, 18)   -> GELU -> Dropout(0.2)
Linear(18, 1)    -> Sigmoid
```

复现命令:

```powershell
python sst/train_sst_head_mlp4.py --hidden-dims 64 32 18
```

## 产物

- 10-seed 汇总 JSON:`sst/heads/seed10_results.json`
- 10-seed checkpoints:`sst/heads/seed10_checkpoints/`
- 每次训练日志:`sst/heads/seed10_logs/`

以上产物中的 `.json`、`.pt`、`.log` 按 `.gitignore` 规则不跟踪。
