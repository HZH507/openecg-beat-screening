"""在 DS2 上比较理想、设备链路及整数时延对齐三通路，固定模型与阈值。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import wfdb  # noqa: E402
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "sil"))

from config import AAMI_MAP, CLASS_NAMES, DS2, MITDB, WIN, WIN_AFTER, WIN_BEFORE  # noqa: E402
from models import ECGCNNLight  # noqa: E402

import adc  # noqa: E402
import afe  # noqa: E402
import link  # noqa: E402

FS = 360
THRESHOLD = 0.5878590941429138
CKPT = PROJECT_ROOT / "results" / "workpoint_cv_v2" / "final_binary.pt"
OUT = PROJECT_ROOT / "results" / "sil"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def bandpass_filter(sig: np.ndarray) -> np.ndarray:
    """与 prepare_dataset.py 完全一致的 0.5–40 Hz 零相位带通 + 60 Hz 陷波。"""
    sos = butter(3, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")
    out = sosfiltfilt(sos, sig)
    b, a = iirnotch(60.0, 30.0, fs=FS)
    return filtfilt(b, a, out).astype(np.float32)


def build_beats(filtered: np.ndarray, samples: np.ndarray):
    padded = np.pad(filtered, (WIN_BEFORE, WIN_AFTER), mode="edge")
    beats, kept = [], []
    for s in samples:
        w = padded[s:s + WIN]
        if w.shape[0] != WIN:
            continue
        beats.append(w - np.median(w))
        kept.append(int(s))
    return np.asarray(beats, dtype=np.float32), np.asarray(kept, dtype=np.int64)


def classify(model, beats: np.ndarray) -> np.ndarray:
    xn = (beats - beats.mean(1, keepdims=True)) / (beats.std(1, keepdims=True) + 1e-6)
    with torch.no_grad():
        logits = model(torch.from_numpy(xn).float().unsqueeze(1))
        return torch.softmax(logits, 1).numpy()[:, 1]


def metrics(y, prob):
    pred = (prob >= THRESHOLD).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    f1n = 2 * tn / max(2 * tn + fp + fn, 1)
    f1a = 2 * tp / max(2 * tp + fp + fn, 1)
    return {
        "auc": float(roc_auc_score(y, prob)),
        "accuracy": float((tp + tn) / len(y)),
        "macro_f1": float((f1n + f1a) / 2),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def estimate_lag(ref: np.ndarray, test: np.ndarray, max_lag: int = 40,
                 start: int = 10000, span: int = 20000) -> int:
    """用互相关估计 test 相对 ref 的整数样本延迟（正数表示 test 滞后）。"""
    a = ref[start:start + span]
    best, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        b = test[start + lag:start + lag + span]
        if len(b) != span:
            continue
        score = float(np.dot(a - a.mean(), b - b.mean()))
        if score > best:
            best, best_lag = score, lag
    return best_lag


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = ECGCNNLight(num_classes=2)
    model.load_state_dict(ck["model"])
    model.eval()

    hp, lp = afe.design()
    y_all, p_ideal, p_sil, p_align, p_ql = [], [], [], [], []
    records_all, samples_all = [], []
    lags = []
    link_stats_total = {"n_packets": 0, "lost_packets": 0}
    example = None

    for rec in DS2:
        path = str(MITDB / rec)
        sig = np.asarray(wfdb.rdrecord(path).p_signal[:, 0], dtype=np.float64)
        ann = wfdb.rdann(path, "atr")
        symbols = np.asarray(ann.symbol)
        keep = np.array([AAMI_MAP.get(s) is not None for s in symbols])
        samples = np.asarray(ann.sample)[keep]
        cls = np.array([CLASS_NAMES.index(AAMI_MAP[s]) for s in symbols[keep]])
        sample_label = {int(s): int(c) for s, c in zip(samples, cls)}

        # 理想通路
        ideal_sig = bandpass_filter(sig)
        beats_i, kept_i = build_beats(ideal_sig, samples)
        pi = classify(model, beats_i)

        # 设备链路
        analog = afe.apply(sig, hp, lp)
        quant, lsb = adc.quantize(analog, seed=int(rec) % 1000)
        stream, stats = link.transmit(quant, fs=FS, seed=int(rec) % 1000)
        sil_sig = bandpass_filter(stream)
        beats_s, kept_s = build_beats(sil_sig, samples)
        ps = classify(model, beats_s)

        # 设备链路 + 时延对齐（真实系统用时间戳/检测器消除固定相位延迟）
        lag = estimate_lag(ideal_sig, sil_sig)
        lags.append(lag)
        if lag > 0:
            # 正 lag 表示设备信号滞后：向左移，末端用边值补齐。
            aligned = np.concatenate([sil_sig[lag:], np.full(lag, sil_sig[-1])])
        elif lag < 0:
            # 负 lag 表示设备信号提前：向右移，开头用边值补齐。
            aligned = np.concatenate([np.full(-lag, sil_sig[0]), sil_sig[:lag]])
        else:
            aligned = sil_sig
        beats_a, kept_a = build_beats(aligned, samples)
        pa = classify(model, beats_a)

        # 消融：跳过 AFE，仅量化 + 分包/丢包（用于归因）
        quant_q, _ = adc.quantize(sig, seed=int(rec) % 1000)
        stream_q, _ = link.transmit(quant_q, fs=FS, seed=int(rec) % 1000)
        beats_q, kept_q = build_beats(bandpass_filter(stream_q), samples)
        pq = classify(model, beats_q)

        assert np.array_equal(kept_i, kept_s) and np.array_equal(kept_i, kept_a)
        assert np.array_equal(kept_i, kept_q)
        y_rec = np.asarray([sample_label[int(s)] != 0 for s in kept_i], dtype=np.int64)
        y_all.append(y_rec)
        records_all.append(np.full(len(kept_i), rec, dtype="U3"))
        samples_all.append(kept_i)
        p_ideal.append(pi); p_sil.append(ps); p_align.append(pa); p_ql.append(pq)
        link_stats_total["n_packets"] += stats["n_packets"]
        link_stats_total["lost_packets"] += stats["lost_packets"]
        if example is None:
            example = (rec, sig, analog, stream, lsb)

    y = np.concatenate(y_all)
    p_ideal = np.concatenate(p_ideal); p_sil = np.concatenate(p_sil)
    p_align = np.concatenate(p_align); p_ql = np.concatenate(p_ql)

    # 与冻结缓存核对（应为逐拍一致）
    cache = np.load(PROJECT_ROOT / "results" / "workpoint_cv_v2" / "ds2_predictions.npz",
                    allow_pickle=True)
    max_dev = float(np.max(np.abs(p_ideal - cache["prob"]))) if len(p_ideal) == len(cache["prob"]) else None

    res = {
        "n_beats": int(len(y)), "n_abnormal": int(y.sum()),
        "threshold": THRESHOLD,
        "ideal": metrics(y, p_ideal),
        "quant_link": metrics(y, p_ql),
        "sil": metrics(y, p_sil),
        "sil_aligned": metrics(y, p_align),
        "delta_auc_quant_link": float(roc_auc_score(y, p_ql) - roc_auc_score(y, p_ideal)),
        "delta_auc": float(roc_auc_score(y, p_sil) - roc_auc_score(y, p_ideal)),
        "delta_auc_aligned": float(roc_auc_score(y, p_align) - roc_auc_score(y, p_ideal)),
        "mean_abs_prob_diff": float(np.mean(np.abs(p_sil - p_ideal))),
        "max_abs_prob_diff": float(np.max(np.abs(p_sil - p_ideal))),
        "n_prediction_flip": int((((p_ideal >= THRESHOLD) != (p_sil >= THRESHOLD))).sum()),
        "n_prediction_flip_aligned": int((((p_ideal >= THRESHOLD) != (p_align >= THRESHOLD))).sum()),
        "mean_abs_prob_diff_aligned": float(np.mean(np.abs(p_align - p_ideal))),
        "mean_lag_samples": float(np.mean(lags)),
        "alignment_method": "per-record waveform cross-correlation; positive lag shifted left, negative lag shifted right; edge padding",
        "alignment_scope": "offline reference-assisted SIL analysis; no threshold or model adjustment",
        "lag_ms": round(1000.0 * float(np.mean(lags)) / FS, 2),
        "lags": lags,
        "ideal_matches_frozen_cache_max_dev": max_dev,
        "link": {**link_stats_total,
                 "loss_rate": round(link_stats_total["lost_packets"] / max(link_stats_total["n_packets"], 1), 4)},
        "adc": {"bits": adc.BITS, "vmin": adc.V_MIN, "vmax": adc.V_MAX,
                "lsb_mV": (adc.V_MAX - adc.V_MIN) / (2 ** adc.BITS - 1)},
        "afe": {"hp_hz": afe.HP_HZ, "lp_hz": afe.LP_HZ, "order": afe.ORDER, "fs": FS,
                "note": "源数据已是 360 Hz 数字记录，未模拟采样前混叠"},
    }
    predictions_path = OUT / "sil_predictions.npz"
    res["probability_summary"] = {
        name: {"mean": float(np.mean(prob, dtype=np.float64)), "median": float(np.median(prob)),
               "predicted_positive_rate": float(np.mean(prob >= THRESHOLD))}
        for name, prob in (("ideal", p_ideal), ("quant_link", p_ql),
                           ("sil", p_sil), ("sil_aligned", p_align))
    }
    np.savez_compressed(predictions_path, y=y, prob_ideal=p_ideal, prob_sil=p_sil,
                        prob_aligned=p_align, prob_quant_link=p_ql,
                        record=np.concatenate(records_all),
                        sample=np.concatenate(samples_all))
    res["predictions_path"] = "results/sil/sil_predictions.npz"
    res["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    (OUT / "sil_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # 图 1：AFE 响应 + 波形
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.4), gridspec_kw={"height_ratios": [1, 1.15]})
    w_hp, h_hp = afe.response(hp); w_lp, h_lp = afe.response(lp)
    ax = axes[0]
    ax.semilogx(w_lp, h_hp + h_lp, color="#C0392B", lw=3.2, alpha=0.35, label="级联")
    ax.semilogx(w_hp, h_hp, color="#7f8c8d", lw=1.3, label="高通 0.5 Hz")
    ax.semilogx(w_lp, h_lp, color="#2E6F9E", lw=1.3, label="低通 40 Hz（4 阶）")
    ax.axvline(40, ls="--", lw=1, color="#999"); ax.axvline(0.5, ls="--", lw=1, color="#999")
    ax.set_xlim(0.1, 180); ax.set_ylim(-60, 3)
    ax.set_xlabel("频率 / Hz"); ax.set_ylabel("幅度 / dB")
    ax.set_title("模拟前端（AFE）幅频响应（级联曲线与低通在 0.5 Hz 以上重合）")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3, which="both")

    rec, raw, analog, stream, lsb = example
    seg = slice(int(30 * FS), int(38 * FS))
    t = np.arange(int(30 * FS), int(38 * FS)) / FS
    ax = axes[1]
    ax.plot(t, raw[seg], color="#bdc3c7", lw=1.0, label="原始（含直流偏置）")
    ax.plot(t, afe.apply(raw, hp, lp)[seg], color="#2E6F9E", lw=1.0, label="AFE 输出（已去直流）")
    ax.plot(t, stream[seg], color="#C0392B", lw=0.9, alpha=0.8, label="ADC+分包+重组")
    ax.set_xlabel("时间 / s"); ax.set_ylabel("幅值 / mV")
    ax.set_title("记录 %s 的 8 s 片段：原始 / AFE / 设备链路（LSB=%.2f µV）" % (rec, lsb * 1000))
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "sil_afe.png", dpi=200)

    # 图 2：ROC
    f_i, t_i, _ = roc_curve(y, p_ideal); f_s, t_s, _ = roc_curve(y, p_sil)
    f_a, t_a, _ = roc_curve(y, p_align); f_q, t_q, _ = roc_curve(y, p_ql)
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot(f_i, t_i, color="#2E6F9E", lw=1.8, label="理想通路 AUC=%.4f" % res["ideal"]["auc"])
    ax.plot(f_q, t_q, color="#8E44AD", lw=1.3, ls=":", label="仅量化+链路 AUC=%.4f" % res["quant_link"]["auc"])
    ax.plot(f_s, t_s, color="#C0392B", lw=1.5, ls="--", label="设备链路 AUC=%.4f" % res["sil"]["auc"])
    ax.plot(f_a, t_a, color="#27AE60", lw=1.5, ls="-.", label="设备链路+时延对齐 AUC=%.4f" % res["sil_aligned"]["auc"])
    ax.plot([0, 1], [0, 1], ls=":", color="#999", lw=1)
    ax.set_xlabel("1 − 特异性"); ax.set_ylabel("敏感度")
    ax.set_title("DS2 49,705 拍：四条通路的 ROC")
    ax.legend(frameon=False, loc="lower right", fontsize=8.2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "sil_roc.png", dpi=200)

    print(json.dumps({k: res[k] for k in ("n_beats", "delta_auc_quant_link", "delta_auc",
                                          "delta_auc_aligned", "n_prediction_flip",
                                          "n_prediction_flip_aligned", "mean_abs_prob_diff",
                                          "mean_abs_prob_diff_aligned", "lag_ms",
                                          "ideal_matches_frozen_cache_max_dev")},
                     ensure_ascii=False))
    print("ideal", {k: round(v, 4) for k, v in res["ideal"].items() if isinstance(v, float)})
    print("ql   ", {k: round(v, 4) for k, v in res["quant_link"].items() if isinstance(v, float)})
    print("sil  ", {k: round(v, 4) for k, v in res["sil"].items() if isinstance(v, float)})
    print("align", {k: round(v, 4) for k, v in res["sil_aligned"].items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
