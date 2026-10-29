"""OpenECG 模拟信号模型（复刻上游 notebook，非上游代码原样拷贝）。

参数与波形构成取自 jwlee95/OpenECG 仓库中的
``심전도_신호를_그리는_코드는.ipynb``：
    采样率 1000 Hz、时长 10 s、心率 60 bpm，
    用高斯函数分别模拟 P、QRS(Q/R/S)、T 波并叠加。
上游 notebook 未声明许可证，因此这里只复刻其数学描述，不复制其文件。
"""
from __future__ import annotations

import numpy as np


def gaussian(x: np.ndarray, mu: float, sigma: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def simulate_openecg(duration: float = 10.0,
                     fs: int = 1000,
                     heart_rate_bpm: float = 60.0,
                     noise: float = 0.0,
                     seed: int = 20261029):
    """生成 OpenECG notebook 同款的合成 ECG。

    返回 ``(t, signal)``，signal 单位为任意幅值（mV 量级）。
    """
    t = np.linspace(0.0, duration, int(fs * duration), endpoint=False)
    sig = np.zeros_like(t)

    beat = 60.0 / heart_rate_bpm
    for i in range(int(duration / beat)):
        b = i * beat
        sig += gaussian(t, b + 0.15, 0.020, 0.20)    # P 波
        sig += gaussian(t, b + 0.28, 0.008, -0.10)   # Q 波
        sig += gaussian(t, b + 0.30, 0.008, 1.00)    # R 波
        sig += gaussian(t, b + 0.32, 0.010, -0.20)   # S 波
        sig += gaussian(t, b + 0.55, 0.040, 0.30)    # T 波

    if noise > 0:
        rng = np.random.default_rng(seed)
        sig = sig + rng.normal(0.0, noise, size=sig.shape)
    return t, sig.astype(np.float64)
