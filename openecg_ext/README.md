# OpenECG 扩展模块 —— ECG 滤波与心搏分类

本目录是对上游开源项目 [jwlee95/OpenECG](https://github.com/jwlee95/OpenECG)
的一个小扩展，由课程项目新增，**不是上游仓库本身**。

## 上游现状（截至 2026-10）

上游仓库当前只有一个 README（原文 "ECG Monitoring system design with Open Source
H/W and Pythons / 추가적인 작업중……"，即采集硬件部分仍在开发）和两个 Jupyter
notebook：

| 文件 | 作用 |
|---|---|
| `NeuroKit2를_이용하여_4~5초_정도의_심전도_파형을_그리는_코드.ipynb` | 调 `neurokit2.ecg_simulate` 生成 4 s / 1000 Hz / 70 bpm 模拟心电并绘图 |
| `심전도_신호를_그리는_코드는.ipynb` | 用 numpy 高斯模型拼出 P/QRS/T 的 10 s / 1000 Hz / 60 bpm 模拟心电并绘图 |

上游**没有**滤波、R 峰检测、心搏分类，也没有可运行的采集端代码。本扩展补的正是
这一层：读取它的模拟信号，接上本项目的预处理与分类，并把结果画回它的信号上。

## 新增内容

```
openecg_ext/
  README.md                 本文件
  analysis/
    __init__.py
    simulate.py             复刻上游 notebook 的模拟模型（高斯 P/QRS/T）
    pipeline.py             重采样 1000→360 Hz、0.5–40 Hz 滤波、R 峰检测、256 点切分
    run_demo.py             命令行入口，输出图与 JSON
```

上游 notebook 未复制进本目录（上游未声明许可证），使用时按上面的表格对照；
`simulate.py` 顶部注明了每个参数来自哪个 notebook。

## 运行

```bash
D:\openecg\.venv\Scripts\python.exe openecg_ext\analysis\run_demo.py
```

输出到 `results/openecg_demo/`：

- `openecg_sim.png`：模拟信号、滤波与 R 峰、切分心搏与分类结果三联图
- `openecg_demo.json`：心搏数、心率、每拍异常概率、冻结阈值与判读

## 说明

- 模拟信号是合成波形，不是真实患者数据；本演示用于**验证流程可接入**，不作为
  分类性能评价。性能评价见 DS2 的 49,705 个真实心搏。
- 分类模型权重来自 `results/workpoint_cv_v2/final_binary.pt`，冻结阈值 0.5879。
- 重采样用 `scipy.signal.resample_poly(1000→360)`，与训练时 360 Hz 的采样率一致。
