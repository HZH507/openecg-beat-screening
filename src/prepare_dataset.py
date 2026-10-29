"""Extract R-peak-centred beats from the MIT-BIH Arrhythmia Database.

The script follows the AAMI EC57 grouping used in most inter-patient ECG beat
classification papers.  Q beats (paced, unclassifiable and unknown) are excluded,
leaving four supervised classes: N, S, V and F.

Run from the project root:
    python src/prepare_dataset.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from config import (AAMI_MAP, CLASS_NAMES, FS, LABEL_TO_INDEX, MITDB,
                    PROCESSED, WIN, WIN_AFTER, WIN_BEFORE)


BEAT_SYMBOLS = set(AAMI_MAP) | {"/", "f", "Q"}


def rr_features(samples: np.ndarray) -> np.ndarray:
    """Previous/next RR intervals and their balance, normalised by the record median."""
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    out = np.ones((n, 3), dtype=np.float32)
    if n < 3:
        return out
    rr = np.diff(samples)
    med = float(np.median(rr)) + 1e-6
    for i in range(n):
        prev = rr[i - 1] if i > 0 else med
        nxt = rr[i] if i < n - 1 else med
        out[i] = [prev / med, nxt / med, prev / (prev + nxt + 1e-6)]
    out[:, :2] = np.clip(out[:, :2], 0.2, 3.0)
    return out


def bandpass_filter(sig: np.ndarray) -> np.ndarray:
    """0.5--40 Hz zero-phase band-pass plus a 60 Hz mains notch."""
    sos = butter(3, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")
    out = sosfiltfilt(sos, sig)
    b, a = iirnotch(60.0, 30.0, fs=FS)
    out = filtfilt(b, a, out)
    return out.astype(np.float32)


def main() -> None:
    records = sorted([p.stem for p in MITDB.glob("*.hea")])
    print(f"records: {len(records)}")

    beats: list[np.ndarray] = []
    labels: list[int] = []
    rec_ids: list[str] = []
    samples: list[int] = []
    symbols: list[str] = []
    rr_rows: list[np.ndarray] = []
    per_record: dict[str, Counter] = defaultdict(Counter)
    skipped = Counter()

    for rec in records:
        path = str(MITDB / rec)
        rec_obj = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, "atr")
        # Lead 0 is MLII for most records and a valid limb/chest lead otherwise.
        sig = np.asarray(rec_obj.p_signal[:, 0], dtype=np.float64)
        sig = bandpass_filter(sig)
        padded = np.pad(sig, (WIN_BEFORE, WIN_AFTER), mode="edge")

        beat_samples = np.asarray(
            [s for s, sym in zip(ann.sample, ann.symbol)
             if sym in BEAT_SYMBOLS], dtype=np.int64)
        rr_all = rr_features(beat_samples)
        rr_lookup = {int(s): rr_all[i] for i, s in enumerate(beat_samples)}

        kept = 0
        for sample, symbol in zip(ann.sample, ann.symbol):
            cls = AAMI_MAP.get(symbol)
            if cls is None:
                skipped[symbol] += 1
                continue
            start = int(sample)
            window = padded[start:start + WIN]
            if window.shape[0] != WIN:
                skipped["edge"] += 1
                continue
            # Remove residual offset and store physical units (mV).
            window = window - np.median(window)
            beats.append(window.astype(np.float32))
            labels.append(LABEL_TO_INDEX[cls])
            rec_ids.append(rec)
            samples.append(int(sample))
            symbols.append(symbol)
            rr_rows.append(rr_lookup.get(int(sample), np.array([1.0, 1.0, 0.5],
                                                               dtype=np.float32)))
            per_record[rec][cls] += 1
            kept += 1
        print(f"  {rec}: {kept:5d} beats")

    X = np.stack(beats).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    rr = np.stack(rr_rows).astype(np.float32)
    rec_arr = np.asarray(rec_ids)
    sample_arr = np.asarray(samples, dtype=np.int64)
    symbol_arr = np.asarray(symbols)

    out = PROCESSED / "beats.npz"
    np.savez_compressed(out, X=X, y=y, rr=rr, record=rec_arr,
                        sample=sample_arr, symbol=symbol_arr)
    print(f"saved {X.shape} -> {out}")

    counts = Counter(CLASS_NAMES[i] for i in y)
    stats = {
        "n_beats": int(len(y)),
        "n_records": len(records),
        "class_counts": {c: int(counts.get(c, 0)) for c in CLASS_NAMES},
        "per_record": {r: {c: int(per_record[r].get(c, 0)) for c in CLASS_NAMES}
                       for r in records},
        "skipped_symbols": dict(skipped),
        "window": {"before": WIN_BEFORE, "after": WIN_AFTER, "length": WIN,
                   "fs": FS},
    }
    (PROCESSED / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats["class_counts"], ensure_ascii=False))
    print("skipped:", dict(skipped.most_common(12)))


if __name__ == "__main__":
    main()
