# 数据目录（不随仓库提交）

本目录存放原始数据与预处理结果，**体积约 200 MB，不纳入版本控制**。按课程要求，数据集只提供来源链接，不随报告或仓库分发。

## 需要的三个来源

| 数据 | 用途 | 获取地址 | 放置位置 |
|---|---|---|---|
| MIT-BIH Arrhythmia Database 1.0.0 | 训练/测试心搏（48 条记录、约 11 万心搏、360 Hz） | https://physionet.org/content/mitdb/1.0.0/ | `data/mitdb/mit-bih-arrhythmia-database-1.0.0/` |
| MIT-BIH Noise Stress Test Database 1.0.0 | 三种真实噪声 bw / em / ma | https://physionet.org/content/nstdb/1.0.0/ | `data/nstdb/` |
| OpenECG（jwlee95） | 模拟信号流程与接入对象 | https://github.com/jwlee95/OpenECG | `data/OpenECG-master/` |

## 恢复步骤

```powershell
# 1. 下载并解压 MIT-BIH，确保存在
#    data/mitdb/mit-bih-arrhythmia-database-1.0.0/100.dat 等文件
# 2. 下载 NSTDB 的 bw.dat/.hea、em.dat/.hea、ma.dat/.hea 到 data/nstdb/
# 3. 生成预处理心搏（约 2 分钟）
python src/prepare_dataset.py
```

`prepare_dataset.py` 会输出：

- `data/processed/beats.npz`：101,451 个心搏 × 256 点
- `data/processed/dataset_stats.json`：逐记录类别统计

这两份文件同样不提交，但可由上面的命令重建。

## 许可

MIT-BIH 与 NSTDB 按 Open Data Commons Attribution License v1.0（ODC-By 1.0）使用；OpenECG 的使用遵循其仓库声明。本仓库不复制、不再分发上述数据。
