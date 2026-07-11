# Pod 训练通路(RunPod / 任意带网络卷的 GPU 云)

在 pod 上打开 **`w9_all.ipynb`** 按序执行即可 —— 单 notebook 完成:
依赖安装 → 资产准备(断点续传,原子改名 + READY 标记)→ 本地暂存
(`h5_staging.parallel_copy` 多线程拷到本地盘/共享内存)→ 作业队列
(多机安全:排它创建 claim 文件,过期 12h 自动接管)→ 完成审计 →
`pod_selfstop` 自动关机(杜绝空耗)。

- `w9_a100_worker.py` — 固定分割作业(一臂 = 塔 + 每检查点头 + vsel 选优),
  滚动 resume bundle,重启后从断点继续;
- `w9_cv_worker.py` — 五折 CV 作业(fold 参数化);
- 两个 worker 与本机 `steam_reviews_framework/train.py` 是同一协议(R60)的实现,
  产物 json/npz 命名兼容,`contrast_experiment/report.py` 可直接聚合。

多机并行:每台机器各开一份 notebook,claim 机制保证一个作业只被一台
认领;先让第一台完成"ONE-TIME 资产准备"再开其余机器最省钱。
