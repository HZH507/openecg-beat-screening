"""11 bit / 360 Hz ADC 量化模型。"""
from __future__ import annotations

import numpy as np

BITS = 11
V_MIN = -5.0      # mV，MIT-BIH 记录仪的量程
V_MAX = 5.0


def quantize(signal: np.ndarray, bits: int = BITS, vmin: float = V_MIN,
             vmax: float = V_MAX, dither_lsb: float = 0.5, seed: int = 42):
    """量化到 bits 位 ADC，并在量化前叠加 ±dither_lsb LSB 的均匀抖动。

    返回 (量化后信号, LSB 对应的电压)。
    """
    rng = np.random.default_rng(seed)
    lsb = (vmax - vmin) / (2 ** bits - 1)
    x = np.clip(np.asarray(signal, dtype=np.float64), vmin, vmax)
    x = x + rng.uniform(-dither_lsb, dither_lsb, size=x.shape) * lsb
    code = np.round((np.clip(x, vmin, vmax) - vmin) / lsb)
    return vmin + code * lsb, lsb
