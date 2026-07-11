# data/results/ — 训练产物

训练器(`steam_reviews_framework/run.py` 冠军 / `contrast_experiment/run.py`
全对照)写入;`contrast_experiment/report.py` 聚合。命名约定
(`{arm}` = 臂名,如 `champion_cegate2`;CV 臂 = `cv_{recipe}_fold{k}`):

| 文件 | 结构 | 说明 |
|---|---|---|
| `ckpt_{arm}_ep{K}.pt` | torch state_dict | 塔检查点,每 50 ep 一个(1000 ep 不早停) |
| `tower_{arm}_ep{K}.npz` | `SPg` (2020,128) 锚投影;`SPg_nd` 无文档锚投影;`SPa` 评测查询投影;`SPq` 伪查询投影;`SPd` + `SPd_gidx` 家族文档投影 | 冻结塔在该检查点的全部投影——头训练只吃这个,不再碰塔 |
| `zs_traj_{arm}.json` | `{ep50: {nm_neutral, nm_noname, tag_neutral, tag_noname}, …}` | 各检查点裸塔零样本轨迹 |
| `ft4var_{arm}_ep{K}{head}.json` | `{per_seed: [{neutral:{h1,h5,med,tag}, noname:…, positive:…, negative:…, vscore, v_neu, v_non, v_non5}, …]}` | 该检查点头结果(pass1 = 3 seed);`{head}` 空 = 主头,`_p1ce`/`_ce`/`_cetf` = 消融头 |
| `ft4var_{arm}_best{head}.json` | 同上 + `best_ep` + `vsel_by_ep` | **事后寻优结果**:按平均 vsel(只看验证,从不看测试)选出最优检查点并补满 10 seed |
| `summary.md` | markdown 表 | report.py 汇总:固定分割排行 + CV 均值±方差 |

指标口径:`h1/h5` = hit@1/hit@5(2020 全画廊检索),`med` = 中位名次,
`tag` = 23 标签 micro-F1,`m4` = 四变体 hit@1 均值;`vscore` = vsel
选择分(noname 双轴取优 + neutral 加性门,见
`steam_reviews_framework/protocol.py`)。
