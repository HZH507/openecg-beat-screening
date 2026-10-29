"""模拟前端（AFE）：0.5 Hz 高通 + 40 Hz 四阶 Butterworth 抗混叠低通，因果实现。"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz

FS = 360
HP_HZ = 0.5
LP_HZ = 40.0
ORDER = 4


def design(fs: int = FS):
    """返回 (高通 SOS, 低通 SOS)。按双线性离散化近似模拟前端。"""
    hp = butter(2, HP_HZ, btype="high", fs=fs, output="sos")
    lp = butter(ORDER, LP_HZ, btype="low", fs=fs, output="sos")
    return hp, lp


def apply(signal: np.ndarray, hp=None, lp=None) -> np.ndarray:
    """因果滤波：模拟硬件里滤波器只会用到当前与过去的样本。"""
    if hp is None or lp is None:
        hp, lp = design()
    return sosfilt(lp, sosfilt(hp, np.asarray(signal, dtype=np.float64)))


def response(sos, fs: int = FS, n: int = 4096):
    """幅频响应，供作图。"""
    w, h = sosfreqz(sos, worN=n, fs=fs)
    return w, 20.0 * np.log10(np.abs(h) + 1e-12)
