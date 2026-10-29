# 参考 OpenECG 思路的自定义合成信号流程验证

本模块参考 [jwlee95/OpenECG](https://github.com/jwlee95/OpenECG) 的高斯波形构造思路，自行生成合成 ECG，并接入本课程项目的离线预处理和冻结二分类器。它没有读取上游 notebook 实际输出的数组，也不是对上游原始波形的等价复现。

## 上游现状（截至 2026-10）

上游仓库当前只有一个 README（原文 "ECG Monitoring system design with Open Source
H/W and Pythons / 추가적인 작업중……"，即采集硬件部分仍在开发）和两个 Jupyter
notebook：

| 文件 | 作用 |
|---|---|
| `NeuroKit2를_이용하여_4~5초_정도의_심전도_파형을_그리는_코드.ipynb` | 调 `ecg_simulate` 生成 4 s / 1000 Hz / 70 bpm 模拟心电，并调用 `ecg_process` 做预处理与 R 峰检测 |
| `심전도_신호를_그리는_코드는.ipynb` | 用 numpy 高斯模型拼出 P/QRS/T 的 10 s / 1000 Hz / 60 bpm 模拟心电并绘图 |

上游尚未包含本项目的训练分类器及可运行的硬件采集实现。本模块生成自己的合成输入，验证预处理、切分、模型加载和绘图流程。

本模块的采样率、时长和心率与上游高斯示例一致，但波形参数不同：

| 参数 | 上游高斯示例 | 本模块 |
|---|---:|---:|
| R 波宽度 sigma / 幅值 | 0.03 / 1.5 | 0.008 / 1.0 |
| Q 波宽度 sigma / 幅值 | 0.015 / -0.5 | 0.008 / -0.1 |
| S 波宽度 sigma / 幅值 | 0.015 / -0.5 | 0.010 / -0.2 |
| T 波中心（相对拍起点） | 0.60 s | 0.55 s |

现有 10 拍全部判正常的结果仅说明本组自定义合成输入下流程可以运行，不作为上游接入成功或独立分类性能的证据。

## 新增内容

```
openecg_ext/
  README.md                 本文件
  analysis/
    __init__.py
    simulate.py             自定义高斯 P/QRS/T 合成波形
    pipeline.py             重采样 1000→360 Hz、0.5–40 Hz 滤波、R 峰检测、256 点切分
    run_demo.py             命令行入口，输出图与 JSON
```

上游 notebook 未复制进本目录。本模块采用高斯波叠加思路，参数差异见上表。

## 运行

仅维护现有图示可运行以下命令。它读取保存的每拍概率，重建同一确定性合成输入后绘图，不加载分类器、不执行模型推理，也不改变判读结果：

```bash
python openecg_ext/analysis/run_demo.py --plot-existing
```

若另行开展新演示，应在独立工作副本运行：

```bash
python openecg_ext/analysis/run_demo.py
```

输出到 `results/openecg_demo/`：

- `openecg_sim.png`：模拟信号、滤波与 R 峰、切分心搏与分类结果三联图
- `openecg_demo.json`：心搏数、心率、每拍异常概率、冻结阈值与判读

## 说明

- 模拟信号是自定义合成波形，幅值采用任意单位，未作真实采集链路的 mV 标定。本演示用于验证软件流程，不作为分类性能评价。性能评价见 DS2 的 49,705 个真实心搏。
- 分类模型权重来自 `results/workpoint_cv_v2/final_binary.pt`，冻结阈值 0.5879。
- 重采样用 `scipy.signal.resample_poly(1000→360)`，与训练时 360 Hz 的采样率一致。
- 分类内部仍按训练规则逐拍做 z-score，绘图单位与实际绘制数据保持一致。
- 上述命令会执行模型推理并覆盖演示输出，仅供另建工作副本时使用。报告修订读取已有结果；本次文稿修订未新增实验、未改变已有判读结果。
