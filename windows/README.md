# windows/ — Larice 锚点工作台（Qt + 内嵌 Python）

Qt Widgets 桌面应用,内嵌 Python(pybind11,结构参考 `CodeX2Thirdpart`),
用**暂定冠军塔 + 头(2ice_cegate = CE-gated I-CE, I×2)**把一段游戏描述
变成嵌入锚点,并给出游戏名与标签的预测。

## 界面

- **左上**:多行编辑框输入游戏描述 + 「嵌入 → 锚点」按钮。
- **右上**:本次会话的锚点列表(两列:名字 | 锚点前三维 `x, x, x, …`)。
  名字单元格可双击改名;点击某行可回看该锚点的预测。
  下方「批量导出 CSV」把全部锚点(名字 + 完整向量)存成 CSV。
- **下半**:两张预测表,均按可能性从大到小排列 ——
  预测游戏(2020 类 softmax 概率)/ 预测标签(ridge 探针分数,
  粗体 = 超过决策阈值)。

首次成功加载模型时,自动把**全部游戏的锚 ID** 写到
`assets/games_anchors.csv`(anchor_id, name, 锚前三维)。

## 首次构建(一次性)

1. `setup_python_pybind11.bat` —— 在本目录安装 `tools/python312`
   与 `external/pybind11`(与 CodeX2Thirdpart 相同的脚本)。
2. 用 Qt Creator(Qt 6.x)打开 `CMakeLists.txt` 构建运行(目标
   `LariceAnchorStudioApp`),或命令行 `cmake -B build && cmake --build build`。

## 打分发包

Release 配置下构建目标 **`LariceWindows_release`**(Qt Creator 里在
构建步骤选择该目标,或命令行
`cmake --build build --config Release --target LariceWindows_release`),
产出自包含的 `windows/dist/`:

```
dist/
├── LariceAnchorStudio.exe      # 根启动器(静态,双击这个)
└── resources/
    ├── LariceAnchorStudioApp.exe + Qt 运行时(windeployqt)
    ├── python3.dll / python312.dll / vcruntime140*.dll
    ├── larice_bridge.py / runtime_requirements.txt
    ├── python312/              # 完整内置 Python(exe/DLLs/Lib/tcl)
    └── assets/                 # tower.pt + champion_assets.npz + meta.json
```

把 `dist/` 整个拷走即可分发;目标机首次运行会自动 pip 装 torch 等
(需联网;Qwen 嵌入模型也在首次嵌入时下载)。构建前确认
`windows/assets/` 已由 export_assets.py 生成,否则包里没有模型
(CMake 会给 WARNING)。

注意:首启 pip 安装写入包内 `resources/python312/Lib/site-packages`,
所以 `dist/` 要放在**用户可写**的位置(桌面/文档均可),不要装进
`Program Files`。

首次**运行**时程序自动 `pip install` 运行依赖
(`runtime_requirements.txt`:torch、transformers 等)到内置 Python,
然后加载冠军塔和头。**检测到 NVIDIA GPU 时自动装 CUDA 版 torch**
(cu128 源;可用 `LARICE_TORCH_INDEX` 覆盖),无 GPU 机器装 CPU 版。
若曾误装 CPU 版(嵌入慢、GPU 空闲、状态栏显示"设备 cpu"),用内置
python 执行:`python -m pip uninstall -y torch && python -m pip install
torch --index-url https://download.pytorch.org/whl/cu128`。
Qwen3-Embedding-0.6B 会在第一次嵌入时从 HuggingFace 下载(走 HF 缓存)。

## 模型资产(先在训练环境导出一次)

App 从 `windows/assets/` 读取(可用 `LARICE_ASSETS_DIR` 覆盖):

| 文件 | 内容 |
|---|---|
| `tower.pt` | 冠军塔 state_dict(champion_cegate2) |
| `champion_assets.npz` | 头 W/b/logt、gallery 特征(塔空间 + 头空间)、列标准化 mu/sd、tag ridge 探针(coef/intercept/scaler/threshold)、2020 游戏名、23 标签名 |
| `meta.json` | 导出溯源 + vsel/测试指标 |

生成方式(需要 release 数据资产 + 训练好的 champion checkpoint):

```
C:/Users/admin/anaconda3/envs/cuda_Vit/python.exe windows/export_assets.py
```

默认取 `data/results` 里 vsel 选中的 epoch(否则最新 ckpt),
训 `--seeds 3` 个头取 vsel 最优,连同 tag 探针一起打包。

## 推理管线(与 release/steam_reviews_framework 逐步对齐)

```
描述文本 → 分句(正则近似;训练期为 SaT)
        → Qwen3-Embedding-0.6B(last-token pool,LocalEmbedder 配方)
        → rown 行标准化 → 冠军塔(pool 读出,L2)
        → 按 gallery 列统计 (mu, sd) 标准化 → 线性头 + L2 = 锚点
游戏预测:softmax(锚点 · gallery头特征ᵀ × e^logt)
标签预测:ridge 探针分数,显示 clip(score, 0, 1),阈值以上加粗
```

## 排障

- 每一步都写进 `windows/larice_app.log`(GUI 无控制台,卡住先看它)。
- 设 `LARICE_AUTOTEST=1` 启动:模型就绪后自动嵌入一段测试文本(冒烟)。
- 卡死时用 `py-spy dump --native --pid <pid>` 抓栈。已修过一个必踩坑:
  worker 线程若没有**终身 CPython 线程状态**,torch 的 `device_lazy_init`
  会在第一次嵌入时 GIL 自死锁(`PyBridge::initPython` 即此修复,勿删)。

## 文件

- `launcher.cpp` —— dist 根启动器(静态小 exe,转发 argv 启动
  `resources/LariceAnchorStudioApp.exe`,与参考项目同款)。
- `cmake/deploy_qt_runtime.cmake` —— windeployqt 部署脚本(原样复制)。
- `main.cpp` / `mainwindow.{h,cpp}` —— Qt 界面;Python 全部跑在
  worker 线程(每次调用自取 GIL,主线程常驻释放 GIL,
  与 CodeX2Thirdpart 相同的解释器生命周期模式)。
- `larice_bridge.py` —— 内嵌 Python 桥:bootstrap(pip)、加载模型、
  嵌入 + 预测、CSV 导出;所有函数返回 JSON 字符串,从不跨边界抛异常。
- `export_assets.py` —— 训练环境侧,把 checkpoint 变成资产包。
