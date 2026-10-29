"""Shared data, noise and evaluation utilities."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, medfilt, sosfiltfilt
from sklearn.model_selection import train_test_split

from config import (CLASS_NAMES, FS, NSTDB, PROCESSED, SEED, WIN)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_beats(with_rr: bool = False):
    d = np.load(PROCESSED / "beats.npz", allow_pickle=False)
    if with_rr:
        return d["X"], d["y"], d["rr"], d["record"], d["sample"]
    return d["X"], d["y"], d["record"], d["sample"]


def split_train_val(y: np.ndarray, val_fraction: float = 0.1,
                    seed: int = SEED):
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_fraction, random_state=seed, stratify=y)
    return train_idx, val_idx


def standardize_per_beat(x: np.ndarray, eps: float = 1e-6):
    """x: (..., L). Returns z-scored signal and the per-beat scale."""
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True) + eps
    return (x - mu) / sd, sd.squeeze(-1)


def load_nstdb_noise() -> dict[str, dict[str, np.ndarray]]:
    """Load the three MIT-BIH noise-stress records.

    The first 65 % of each 30-min record is used for training/validation and the
    last 35 % for testing.  Keeping the segments disjoint prevents the denoiser
    from memorising the exact noise waveforms used at test time.
    """
    bank: dict[str, dict[str, np.ndarray]] = {}
    for kind in ("bw", "em", "ma"):
        rec = wfdb.rdrecord(str(NSTDB / kind))
        sig = np.asarray(rec.p_signal, dtype=np.float32)
        # Use both channels, concatenated, so that the bank is 2 x 30 min long.
        sig = np.concatenate([sig[:, 0], sig[:, 1]])
        sig = sig - sig.mean()
        cut = int(0.65 * len(sig))
        bank[kind] = {"train": sig[:cut], "test": sig[cut:]}
    return bank


def _crop(noise: np.ndarray, length: int, rng: np.random.Generator):
    if len(noise) <= length:
        reps = int(np.ceil(length / len(noise)))
        noise = np.tile(noise, reps)
    start = int(rng.integers(0, len(noise) - length + 1))
    seg = noise[start:start + length].astype(np.float32)
    seg = seg - seg.mean()
    return seg


def _scale_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float):
    pc = float(np.mean(clean ** 2)) + 1e-12
    pn = float(np.mean(noise ** 2)) + 1e-12
    factor = np.sqrt(pc / pn / (10.0 ** (snr_db / 10.0)))
    return noise * factor


def add_noise(clean: np.ndarray, bank, rng: np.random.Generator,
              split: str = "train", snr_db: float | None = None,
              kind: str | None = None) -> np.ndarray:
    """Add one of five noise types to each row of ``clean`` (B, L)."""
    clean = np.asarray(clean, dtype=np.float32)
    out = clean.copy()
    for i in range(len(clean)):
        snr_i = float(rng.uniform(0.0, 15.0)) if snr_db is None else float(snr_db)
        k = kind if kind is not None else str(
            rng.choice(["bw", "em", "ma", "white", "mains"]))
        if k == "white":
            n = rng.normal(0.0, 1.0, size=clean.shape[1]).astype(np.float32)
        elif k == "mains":
            t = np.arange(clean.shape[1]) / FS
            f0 = float(rng.choice([50.0, 60.0]))
            n = (np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
                 + 0.35 * np.sin(2 * np.pi * 2 * f0 * t + rng.uniform(0, 2 * np.pi))
                 + 0.15 * np.sin(2 * np.pi * 3 * f0 * t + rng.uniform(0, 2 * np.pi)))
            n = n.astype(np.float32)
        else:
            n = _crop(bank[k][split], clean.shape[1], rng)
        n = _scale_noise(clean[i], n, snr_i)
        out[i] = clean[i] + n
    return out


def snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    clean = np.asarray(clean, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    noise = clean - estimate
    return float(10.0 * np.log10((np.sum(clean ** 2) + 1e-12) /
                                 (np.sum(noise ** 2) + 1e-12)))


def rmse(clean: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(clean) - np.asarray(estimate)) ** 2)))


def classical_filter(noisy: np.ndarray) -> np.ndarray:
    """Traditional 0.5--40 Hz band-pass + 60 Hz notch + 3-point median."""
    noisy = np.asarray(noisy, dtype=np.float64)
    sos = butter(3, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")
    y = sosfiltfilt(sos, noisy)
    b, a = iirnotch(60.0, 30.0, fs=FS)
    y = filtfilt(b, a, y)
    y = medfilt(y, kernel_size=3)
    return y.astype(np.float32)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, X: np.ndarray,
                   extra: np.ndarray | None = None, batch_size: int = 512,
                   device: str = "cpu") -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        # Classifier training uses BeatDataset z-score scaling. Apply exactly
        # the same scaling for validation, clean testing and noisy testing.
        batch = X[i:i + batch_size]
        batch = (batch - batch.mean(axis=1, keepdims=True)) / (batch.std(axis=1, keepdims=True) + 1e-6)
        xb = torch.from_numpy(batch).float().unsqueeze(1).to(device)
        if extra is None:
            outs.append(model(xb).cpu().numpy())
        else:
            eb = torch.from_numpy(extra[i:i + batch_size]).float().to(device)
            outs.append(model(xb, eb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def class_counts(y: np.ndarray) -> dict[str, int]:
    return {c: int(np.sum(y == i)) for i, c in enumerate(CLASS_NAMES)}
