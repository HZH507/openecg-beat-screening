# ECG 心搏筛查与去噪课程项目

本项目实现离线 ECG 预处理、R 峰对齐心搏切分、轻量二分类筛查、四分类探索、一维 DnCNN 去噪及硬件链路的软件在环（SIL）仿真。课程报告包含个人信息，未随本仓库公开；本仓库只提供代码、模型权重、图表与指标 JSON，报告中的全部数字均可由本仓库脚本与结果文件复现。

## 本轮结果与固定口径

二分类主指标为 AUC：DS2 为 **0.8558**，合并五折 OOF 为 **0.7829**。OOF 每折只使用 16 或 17 条外折训练记录，最终重拟合使用全部 21 条；两者对应不同训练规模和数据分布。未做训练规模消融，不能把 AUC 差异全部归因于训练量。

主工作点由 DS1 合并 OOF 预测选择：在 **Sp≥0.90 时 Se 最大**；并列时优先更高 Sp，再优先更高阈值。完整精度阈值为 **0.5878590941429138**，报告显示为 0.5879。DS2 实测 Se **72.27%**、Sp **80.16%**、Acc **79.29%**、Macro-F1 **0.6533**。OOF 的 Sp≥90% 约束在 DS2 只达到 80.16%，这是阈值和概率校准跨记录迁移的发现；没有据此回头补选阈值。

DS2 阴性占 89.04%，全判正常即可得到 Acc 89.04%、Macro-F1 0.4710、Se 0、Sp 100%。因此准确率作为工作点附属指标，不能代替 AUC、Se/Sp 和 F1。

| 样本部分 | 记录数 | 心搏数 | 用途 |
|---|---:|---:|---|
| DS1 排除 201 后 | 21 | 49,050 | 训练、记录级验证、OOF |
| DS2 | 22 | 49,705 | 固定协议下测试 |
| 201 | 1 | 1,963 | 与 DS2 的 202 属于同一受试者，整条排除 |
| 102/104/107/217 | 4 | 733 | 协议未纳入 |
| 合计 | 48 | 101,451 | 统一样本口径 |

DS2 的 N/S/V/F 分别为 44,259/1,837/3,221/388。49,050+49,705+1,963+733=101,451。

## 文件结构

```text
src/                              训练、评价、画图、报告与自动核验代码
sil/                              AFE、ADC、传输与四通路SIL评价（四个py及README）
results/sil/
  sil_metrics.json                同一冻结模型与阈值下的四通路指标
  sil_afe.png                     AFE响应与波形对照
  sil_roc.png                     理想、仅量化+链路、设备、整数时延对齐后的ROC对照
  sil_predictions.npz             四通路逐拍概率、标签、记录与样本位置缓存
results/workpoint_cv_v2/
  protocol.json                   二分类五折与内层选轮次的预声明规则
  fold_1/ ... fold_5/              外折模型、内层选择记录和 OOF 预测
  oof_predictions.npz             每条记录仅在未参加训练的外折模型下预测
  frozen_workpoints.json          OOF 工作点、最终模型和来源哈希
  final_binary.pt                 最终二分类模型，9,058 参数
  ds2_predictions.npz             唯一一次干净二分类 DS2 推理缓存
  ds2_summary.json                AUC、冻结工作点、全判正常基线
  ds2_threshold_curve_exploratory.csv
                                  从同一缓存推导的完整探索性阈值曲线
  ds2_evaluation.json             干净二分类推理次数及冻结顺序审计
results/report_final/
  dataset_accounting.json          四段样本口径
  aux_split.json                  排除 201 后辅助模型的记录级划分
  experiment_protocol.json        辅助模型与噪声评价预声明规则
  models/cnn_light_best.pt         本轮四分类 CNN，9,188 参数
  models/mlp_best.pt               本轮四分类 MLP 对照
  models/denoiser_best.pt          本轮 DnCNN，46,049 参数
  classification_metrics.json      本轮四分类结果
  binary_noise_results.json        本轮 9 个含噪条件 × 3 种处理
  binary_noise_metadata.json       噪声模型哈希、阈值、抽样和分段来源
  noise_predictions.npz            噪声条件下的缓存预测
  demo_info.json                  新四分类模型 demo 真值与预测对照
  model_cost.json                 本轮各模型参数量和 CPU 时延测量
  report_data.json                 原主评价、噪声与辅助实验的汇总数据入口
  figures/                        本轮报告图表
  report_table_manifest.json       每张表、图、关键段落及SIL独立来源哈希
  report_verification.json         独立逐表与来源核验结果
README.md
requirements.txt
```

代码包保留以上相对路径。旧阶段的根目录指标和模型不作为本轮报告来源，历史备份和 `report/archive/` 不作为提交稿。原始 PhysioNet 数据和预处理大数组不放入代码包。

## 模型与评价来源

- 二分类唯一使用 `results/workpoint_cv_v2/final_binary.pt` 和 `frozen_workpoints.json` 中的完整精度阈值。最终模型在排除 201 的全部 21 条记录上训练 3 轮；3 是五折内层所选轮次 2/7/1/3/21 的中位数。
- 四分类 CNN、MLP 和 DnCNN 均重新排除 201。辅助模型使用 17 条训练记录、4 条验证记录；验证记录为 **108、124、205、223**，训练/验证分别 **40,407/8,643 拍**。训练中的多数类上限或 DnCNN 心搏抽样，另见各自训练配置 JSON。
- 四分类新指标以 `classification_metrics.json` 为准，不沿用上一轮数值。demo 加载本轮四分类 CNN 和 DnCNN，输出 N/S/V/F；CNN 只使用心搏形态，不使用 RR 特征。
- 本轮噪声评价固定使用最终二分类模型、上述完整精度阈值和重新排除 201 的 DnCNN。bw/em/ma × 5/10/15 dB 共 9 个含噪条件，每个条件比较未处理、传统滤波、DnCNN，共 **27 组评价**。子集为全部 5,446 个异常心搏和固定种子抽取的 4,000 个正常心搏，共 9,446 拍。结论由新指标产生，不预设 DnCNN 有效。
- NSTDB 在**每个通道内先按时间切分前 65% 与后 35%，再拼接通道**。前段用于去噪器训练/验证，后段用于测试；避免先拼接通道再切分导致相关通道的相同时段落在两侧。旧噪声结果不复用。
- 原干净二分类 DS2 主评价只推理一次，工作点表、逐记录异常检出和探索性全阈值曲线均从该缓存计算，审计和缓存保持不变。四分类、固定噪声条件及新增 SIL 各自执行评价；SIL 包含四通路推理和理想通路复算，先前经授权修正移位方向并仅重跑 SIL，随后采用用户重跑的组件消融输出。原主评价的“一次”不表示整个项目从未再次向模型输入 DS2；全部新增实验均使用原冻结权重与阈值，不据其结果补选工作点。
- 模型资源以 `model_cost.json` 的参数量、线程数、批大小、预热、重复次数和实测时间为准。INT8 权重体积是存储估算，未完成 INT8 量化推理、MCU 实测或硬件实时部署。当前 `sosfiltfilt` 预处理使用未来样本，仅适用于离线分析。

## 硬件链路仿真（软件在环）

`sil/afe.py`、`sil/adc.py`、`sil/link.py` 和 `sil/run_sil.py` 将 DS2 全部 49,705 拍送入理想通路、仅量化+链路、设备链路及设备链路的整数时延对齐分支，使用同一个冻结二分类模型和完整精度阈值 0.5878590941429138。报告从本轮 `results/sil/sil_metrics.json`、四通路逐拍缓存与两张图读取结果，不借此选择新阈值，也不改动原干净 DS2 主评价的缓存与一次推理审计。

设备链路依次为因果 AFE（0.5 Hz 二阶高通与 40 Hz 四阶低通）、11 bit / ±5 mV 量化（±0.5 LSB 均匀抖动，LSB 约 4.89 µV）、20 采样/包与设定 1% 丢包、20 ms 缓冲记录、重组及现有数字预处理。缺包采用上一包末样本保持；缓冲时间仅记录，重组依据时间戳，不额外平移波形。现有数字预处理仍为离线处理，因此这项仿真不证明端到端实时性。

时延由每条记录的理想波形与设备波形互相关估计。正 lag 表示设备信号滞后，故左移设备信号；负 lag 表示超前，故右移，边界用端值填充。先前发现并修正了旧脚本相反的移位方向，获得授权后仅重跑 SIL；理想与未对齐设备通路复现原指标，原“对齐后排序能力几乎无影响”的结论不再采用。正确的整数对齐仍未恢复理想通路 AUC，且冻结阈值下 Se/Sp 发生迁移；未来设备应用需要在独立开发数据上重新校准工作点。本轮缓存的平均概率差不支持“概率整体下移”。所有数值以本轮 JSON 为准，修订清单保留备份旧结果与当前结果的对照。

新增的仅量化+链路消融绕过 AFE，保留 11 bit 量化与 1% 丢包及现有数字预处理。该通路 AUC 为 0.8530（相对理想为 −0.0027）、Se 为 72.75%，接近理想通路的 0.8558 / 72.27%；加入 AFE 后 AUC 为 0.8456、Se 为 62.14%。组件消融表明该组设置下的主要损失来自 AFE 与数字带通串联的高频衰减，量化和传输不是主因。整数对齐后 AUC 仍为 0.8435，剩余差异还包含频率相关相位与形态变化。

源记录本身已是 360 Hz 数字信号，**未模拟采样前混叠**。丢包率、包大小和缓冲为设定值，不是硬件实测；SIL 不等于实物联调。上述对齐属于离线参考辅助分析，尚未验证真实设备时间戳或 R 峰检测器的延迟补偿效果。报告正文、表 9、图 12/13 与摘要和结论的 SIL 表述均须绑定同一版 JSON、逐拍缓存和源码。

仅在另建的工作副本、具备原始记录和冻结模型后重现 SIL：

```powershell
python sil/run_sil.py
```

该命令会重新执行四通路推理并覆盖工作副本的 `results/sil/`，应先备份。报告重建本身不需要运行它。代码包同时包含 `sil/` 与 `results/sil/`，`report_table_manifest.json` 和 `revision_changes.json` 记录 SIL JSON 及四个源码文件的 SHA-256；原主评价与原图表汇总 JSON 保持各自来源链。

## 环境与数据

已运行环境为 Windows 11、Python 3.13.7、PyTorch 2.15.0+cpu、NumPy 2.5.4、SciPy 1.18.1、WFDB 4.3.1、scikit-learn 1.9.1、python-docx 1.2.0。具体环境记录见 `results/report_final/runtime_environment.json`。训练每进程 2 个 Torch CPU 线程，时延测试的设置单独记录在 `model_cost.json`。

在项目根目录建立独立环境：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

数据来自 [MIT-BIH 1.0.0](https://physionet.org/content/mitdb/1.0.0/) 和 [NSTDB 1.0.0](https://physionet.org/content/nstdb/1.0.0/)。原始目录应为：

```text
data/mitdb/mit-bih-arrhythmia-database-1.0.0/100.dat 等记录文件
data/nstdb/bw.dat、bw.hea、em.dat、em.hea、ma.dat、ma.hea
data/processed/beats.npz
data/processed/dataset_stats.json
```

`prepare_dataset.py` 可由原始 WFDB 文件重建 `beats.npz` 和统计 JSON。已有预处理文件应先复制备份，再在新的实验工作副本中重建；不要在冻结结果所在的原目录覆盖输入数据。

## 从已交付结果重建报告

重建报告不重新训练、不重新推理干净 DS2，也不重新选择阈值。保留现有实验 JSON、缓存、模型及需要的本地数据，在项目根目录运行：

```powershell
python src/assemble_final_report_data.py
python src/make_final_figures.py
python src/build_medium_report.py
python src/verify_report.py
```

如果只需要重建 DOCX，且已保留交付的本轮图表，可以跳过 `make_final_figures.py`。`build_medium_report.py` 从 `report_data.json` 读取原实验指标，并单独读取 `results/sil/sil_metrics.json` 与 SIL 两张图；不要直接手改生成的表格数字。`verify_report.py` 将 DOCX 的每张表、关键文字和嵌入图片与实验 JSON/哈希逐项核对，发现不一致即报错。SIL 来源单独绑定，不改写原 `report_data.json` 或原 `figure_manifest.json`。

原生 Word 页图与分页诊断可用 `render_docx_word.ps1`，只读源 DOCX，通过 Word 页 EMF 生成 PNG，不导 PDF。最终交付验收还应逐页检查文字、页眉页码、表格是否跨页、空白页及孤行。QA 页图仅用于检查，不作为报告附件。

本轮最终 PDF 由 `export_report_pdf.ps1` 生成，`verify_exported_pdf.py` 核对 Word/PDF 页数、末页及当前指标；`verify_report.py --require-pdf` 进一步独立核对 DOCX/PDF 哈希与内容。同版本凭证为 `results/report_final/pdf_export_verification.json`。SNR 使用逐拍对数比值的算术均值，RMSE 使用全部心搏/采样点的全局均方根，二者不能互相反推。附录结尾说明位于表 A2 之前，避免单独形成尾页。

## 在新工作副本中完整复现实验

现有冻结结果不删除、不覆盖。完整重训只在**另建的工作副本**进行：复制 `src/`、`sil/`、`README.md`、`requirements.txt` 和自备数据到新目录，保留交付结果在原目录；新副本的 `results/workpoint_cv_v2/`、`results/report_final/` 与 `results/sil/` 应尚不存在。主评价脚本会拒绝覆盖已经冻结或已经开始的评价；SIL 脚本会覆盖自身输出，因此必须在新副本运行。

从新副本根目录按以下顺序运行。路径由 `config.py` 的项目根目录动态解析；下列 Python 命令可迁移，不使用硬编码本机路径的旧 PowerShell 调度脚本。

```powershell
# 需要重建预处理数组时，先确认输入在新副本并完成备份
python src/prepare_dataset.py

# 二分类：21条记录，五折OOF，内层AUC选轮次
python src/prepare_workpoint_cv.py
python src/train_workpoint_fold.py --fold 1
python src/train_workpoint_fold.py --fold 2
python src/train_workpoint_fold.py --fold 3
python src/train_workpoint_fold.py --fold 4
python src/train_workpoint_fold.py --fold 5
python src/freeze_workpoint.py
python src/evaluate_workpoint_once.py

# 辅助模型：统一排除201；冻结后执行固定噪声和四分类评价
python src/prepare_final_experiment.py
python src/train_final_aux.py --model cnn
python src/train_final_aux.py --model mlp
python src/train_final_aux.py --model denoiser
python src/evaluate_final_consistency.py

# 软件在环：固定上述模型与阈值，不由设备通路结果回调工作点
python sil/run_sil.py

# 图文与自动核验
python src/assemble_final_report_data.py
python src/make_final_figures.py
python src/build_medium_report.py
python src/verify_report.py
```

新副本得到的是该次重训结果，应读取新生成的 JSON 和校验输出，不把本 README 的已交付指标强行写回重训结果。历史上 DS2 已用于早期实验；本轮明确的是当前冻结流程下干净二分类的一次推理，不把数据集描述成从未被查看过的测试集。
