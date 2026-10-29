"""处理自定义合成信号的数组输入，执行预处理与二分类。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.signal import (butter, filtfilt, find_peaks, iirnotch,
                          resample_poly, sosfiltfilt)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import ECGCNNLight  # noqa: E402

FS = 360            # 与 MIT-BIH 训练一致
WIN = 256           # R 峰前 96 点、后 160 点
WIN_BEFORE = 96
DEFAULT_CKPT = PROJECT_ROOT / "results" / "workpoint_cv_v2" / "final_binary.pt"
DEFAULT_THRESHOLD = 0.5878590941429138


def to_model_rate(signal: np.ndarray, fs_in: int = 1000) -> np.ndarray:
    """把输入模拟信号重采样到模型要求的 360 Hz。"""
    return resample_poly(np.asarray(signal, dtype=np.float64), FS, fs_in)


def preprocess(signal: np.ndarray) -> np.ndarray:
    """三阶 Butterworth 0.5–40 Hz 零相位带通 + 60 Hz 陷波。"""
    sos = butter(3, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")
    y = sosfiltfilt(sos, signal)
    b, a = iirnotch(60.0, 30.0, fs=FS)
    return filtfilt(b, a, y)


def detect_rpeaks(signal: np.ndarray) -> np.ndarray:
    """Pan–Tompkins 风格的 R 峰检测（与报告 3.2 节一致）。"""
    sos = butter(3, [5.0, 15.0], btype="bandpass", fs=FS, output="sos")
    y = sosfiltfilt(sos, signal)
    d = np.diff(y, prepend=y[0])
    energy = uniform_filter1d(d * d, size=max(3, int(0.12 * FS)))
    thr = max(np.percentile(energy, 96) * 0.25, energy.mean() * 0.5)
    peaks, _ = find_peaks(energy, distance=int(0.25 * FS), height=thr)

    refined = []
    for p in peaks:
        a = max(0, p - int(0.08 * FS))
        b = min(len(y), p + int(0.08 * FS))
        refined.append(a + int(np.argmax(np.abs(y[a:b]))))
    return np.asarray(sorted(set(refined)), dtype=np.int64)


def segment_beats(signal: np.ndarray, peaks: np.ndarray):
    """按 R 峰切 256 点心搏窗口，丢弃越界片段。"""
    beats, used = [], []
    for p in peaks:
        lo = int(p) - WIN_BEFORE
        if lo >= 0 and lo + WIN <= len(signal):
            beats.append(signal[lo:lo + WIN])
            used.append(int(p))
    if not beats:
        raise ValueError("没有完整的心搏窗口，检查信号长度或 R 峰检测结果")
    return np.asarray(beats, dtype=np.float32), np.asarray(used, dtype=np.int64)


def load_classifier(ckpt_path: Path = DEFAULT_CKPT):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model = ECGCNNLight(num_classes=2)
    model.load_state_dict(state)
    model.eval()
    threshold = (payload.get("threshold", DEFAULT_THRESHOLD)
                 if isinstance(payload, dict) else DEFAULT_THRESHOLD)
    return model, float(threshold)


@torch.no_grad()
def classify(model, beats: np.ndarray) -> np.ndarray:
    """逐拍 z-score 后送入模型，返回异常类概率。"""
    xn = (beats - beats.mean(axis=1, keepdims=True)) / (beats.std(axis=1, keepdims=True) + 1e-6)
    logits = model(torch.from_numpy(xn).float().unsqueeze(1))
    return torch.softmax(logits, dim=1).numpy()[:, 1]
