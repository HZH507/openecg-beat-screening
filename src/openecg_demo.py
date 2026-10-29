"""End-to-end OpenECG-style demo: filter -> R-peak detection -> denoise -> classify.

The purpose is to show how the two trained networks can be inserted into an
OpenECG acquisition/visualisation pipeline.  It is not a real-time optimised
implementation; the report discusses the edge-deployment path separately.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import wfdb
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt, find_peaks

from config import (CLASS_NAMES, FIGURES, FS, MITDB, MODELS, RESULTS,
                    WIN, WIN_BEFORE)
from models import DnCNN1d, ECGCNNLight
from utils import save_json

torch.set_num_threads(4)


def bandpass(sig: np.ndarray, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    sos = butter(3, [low, high], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, sig).astype(np.float32)


def detect_rpeaks(raw: np.ndarray) -> np.ndarray:
    """A compact Pan--Tompkins-style detector used only for the demo."""
    y = bandpass(raw, 5.0, 15.0)
    d = np.diff(y, prepend=y[0])
    energy = uniform_filter1d(d * d, size=max(3, int(0.12 * FS)))
    thr = max(np.percentile(energy, 96) * 0.25, energy.mean() * 0.5)
    peaks, _ = find_peaks(energy, distance=int(0.25 * FS), height=thr)
    refined = []
    for p in peaks:
        a, b = max(0, p - int(0.08 * FS)), min(len(y), p + int(0.08 * FS))
        refined.append(a + int(np.argmax(np.abs(y[a:b]))))
    return np.asarray(sorted(set(refined)), dtype=np.int64)


def match_labels(peaks: np.ndarray, ann_samples: np.ndarray,
                 ann_symbols: np.ndarray, tol: int = 36):
    labels = []
    for p in peaks:
        j = int(np.argmin(np.abs(ann_samples - p)))
        labels.append(ann_symbols[j] if abs(ann_samples[j] - p) <= tol else "?")
    return labels


def load_models():
    cnn = ECGCNNLight()
    ckpt = torch.load(MODELS / "cnn_light_best.pt", map_location="cpu",
                      weights_only=False)
    cnn.load_state_dict(ckpt["model"])
    cnn.eval()
    dn = DnCNN1d(channels=32, depth=6, kernel=11)
    dckpt = torch.load(MODELS / "denoiser_best.pt", map_location="cpu",
                       weights_only=False)
    dn.load_state_dict(dckpt["model"])
    dn.eval()
    return cnn, dn


def rr_features(peaks: np.ndarray) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=np.float64)
    out = np.ones((len(peaks), 3), dtype=np.float32)
    if len(peaks) < 3:
        return out
    rr = np.diff(peaks)
    med = float(np.median(rr)) + 1e-6
    for i in range(len(peaks)):
        prev = rr[i - 1] if i > 0 else med
        nxt = rr[i] if i < len(peaks) - 1 else med
        out[i] = [prev / med, nxt / med, prev / (prev + nxt + 1e-6)]
    out[:, :2] = np.clip(out[:, :2], 0.2, 3.0)
    return out


@torch.no_grad()
def classify_beats(cnn, dn, beats: np.ndarray, rr: np.ndarray):
    mu = beats.mean(axis=1, keepdims=True)
    sd = beats.std(axis=1, keepdims=True) + 1e-6
    xn = (beats - mu) / sd
    xb = torch.from_numpy(xn).float().unsqueeze(1)
    den = dn(xb).squeeze(1).numpy() * sd + mu
    den_n = np.stack([(b - b.mean()) / (b.std() + 1e-6) for b in den])
    # ECGCNNLight uses waveform only; RR features are not inputs to this model.
    logits = cnn(torch.from_numpy(den_n).float().unsqueeze(1)).numpy()
    prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return logits.argmax(axis=1), prob, den


def main() -> None:
    cnn, dn = load_models()

    # Detection quality on a clean, mostly normal record.
    rec100 = wfdb.rdrecord(str(MITDB / "100"))
    ann100 = wfdb.rdann(str(MITDB / "100"), "atr")
    sig100 = np.asarray(rec100.p_signal[:, 0], dtype=np.float32)
    peaks100 = detect_rpeaks(sig100)
    gt = ann100.sample[np.isin(ann100.symbol, list("NLRBAaJSVEFj"))]
    matched = 0
    for p in peaks100:
        if len(gt) and np.min(np.abs(gt - p)) <= int(0.1 * FS):
            matched += 1
    se = matched / max(len(gt), 1)
    ppv = matched / max(len(peaks100), 1)
    print(f"record 100 detection: Se={se:.3f}, PPV={ppv:.3f}, "
          f"GT={len(gt)}, det={len(peaks100)}")

    # A ten-second excerpt of record 208 contains frequent PVCs and fusion beats.
    rec = wfdb.rdrecord(str(MITDB / "208"))
    ann = wfdb.rdann(str(MITDB / "208"), "atr")
    sig = np.asarray(rec.p_signal[:, 0], dtype=np.float32)
    centre = ann.sample[np.isin(ann.symbol, ["F", "V"])]
    centre = int(centre[len(centre) // 2])
    start = max(0, centre - 4 * FS)
    end = min(len(sig), start + 10 * FS)
    seg = sig[start:end]
    seg_f = bandpass(seg)
    peaks = detect_rpeaks(seg)
    beats = []
    valid = []
    for i, p in enumerate(peaks):
        lo = p - WIN_BEFORE
        hi = lo + WIN
        if lo >= 0 and hi <= len(seg_f):
            beats.append(seg_f[lo:hi])
            valid.append(i)
    beats = np.asarray(beats, dtype=np.float32)
    rr_all = rr_features(peaks)
    rr_used = rr_all[valid]
    pred, prob, den = classify_beats(cnn, dn, beats, rr_used)
    true_symbols = match_labels(peaks, ann.sample - start, ann.symbol)
    peaks_used = peaks[valid]
    true_symbols = [true_symbols[i] for i in valid]
    # Map the demo's nearest annotation to the four display classes.
    from config import AAMI_MAP
    true_cls = [AAMI_MAP.get(s, "?") for s in true_symbols]

    out = {
        "model_task":"four-class exploration",
        "classifier_checkpoint":"cnn_light_best.pt",
        "classifier_parameters":9188,
        "uses_rr_features":False,
        "record": "208",
        "start_sample": int(start),
        "fs": FS,
        "signal": seg,
        "peaks": peaks_used.tolist(),
        "pred": pred.tolist(),
        "prob": prob.tolist(),
        "true_symbols": true_symbols,
        "true_cls": true_cls,
        "truth_prediction_pairs":[{'true':t,'pred':CLASS_NAMES[int(p)]} for t,p in zip(true_cls,pred)],
        "mismatches":int(sum(t!=CLASS_NAMES[int(p)] for t,p in zip(true_cls,pred))),
        "detection": {"se": se, "ppv": ppv, "n_gt": int(len(gt)),
                      "n_det": int(len(peaks100))},
    }
    np.savez_compressed(RESULTS / "demo_record208.npz",
                        signal=seg, peaks=peaks_used, pred=pred, prob=prob,
                        den=den, true_cls=np.asarray(true_cls, dtype=object))
    save_json(RESULTS / "demo_info.json", {k: v for k, v in out.items()
                                          if k != "signal"})
    print("demo beats:", len(beats), "pred counts:",
          {c: int(np.sum(pred == i)) for i, c in enumerate(CLASS_NAMES)})


if __name__ == "__main__":
    main()
