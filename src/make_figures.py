"""Generate all report figures from the saved data and results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (AAMI_MAP, CLASS_NAMES, DS1, DS2, FIGURES, FS, MITDB,
                    MODELS, RESULTS, WIN_BEFORE)
from models import DnCNN1d, ECGCNN, MLPBaseline
from utils import add_noise, classical_filter, load_beats, load_nstdb_noise

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
FONT = r"C:\Windows\Fonts\msyh.ttc"
if Path(FONT).exists():
    font_manager.fontManager.addfont(FONT)
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})
COLORS = {"N": "#2E6F9E", "S": "#E08A1E", "V": "#C0392B", "F": "#6C5B7B"}
METHOD_COLORS = {"raw": "#9E9E9E", "classical": "#E08A1E",
                 "denoised": "#2E6F9E"}


def save(fig, name: str):
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path.name)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Dataset distribution
# ---------------------------------------------------------------------------
def fig_dataset():
    stats = load_json(Path("data/processed/dataset_stats.json"))
    per = stats["per_record"]
    counts = {"DS1": {c: 0 for c in CLASS_NAMES},
              "DS2": {c: 0 for c in CLASS_NAMES}}
    for rec, row in per.items():
        if rec not in DS1 and rec not in DS2:
            continue
        key = "DS1" if rec in DS1 else "DS2"
        for c in CLASS_NAMES:
            counts[key][c] += int(row.get(c, 0))

    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    x = np.arange(len(CLASS_NAMES))
    w = 0.36
    for j, (split, color) in enumerate([("DS1", "#2E6F9E"), ("DS2", "#E08A1E")]):
        vals = [counts[split][c] for c in CLASS_NAMES]
        bars = ax.bar(x + (j - 0.5) * w, vals, width=w, color=color,
                      label=f"{split}（{'训练+验证' if split == 'DS1' else '测试'}）")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7,
                        rotation=0 if v < 10000 else 0)
    ax.set_yscale("log")
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylabel("心搏数（对数坐标）")
    ax.set_xlabel("AAMI 心搏类别")
    ax.set_title("MIT-BIH 心搏类别分布（按记录划分）")
    fig.subplots_adjust(bottom=.24)
    fig.text(.5, .025, "102/104/107/217 共 733 个心搏不参与划分", ha="center", fontsize=7)
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(100, 120000)
    save(fig, "fig01_dataset_distribution.png")


# ---------------------------------------------------------------------------
# 2. Representative beats
# ---------------------------------------------------------------------------
def fig_beats():
    X, y, rec, _ = load_beats()
    mask = np.isin(rec, DS2)
    X, y = X[mask], y[mask]
    t = (np.arange(X.shape[1]) - WIN_BEFORE) / FS * 1000
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), sharex=True)
    for i, c in enumerate(CLASS_NAMES):
        idx = np.flatnonzero(y == i)
        amp = np.ptp(X[idx], axis=1)
        pick = idx[np.argmin(np.abs(amp - np.median(amp)))]
        ax = axes.flat[i]
        ax.plot(t, X[pick], color=COLORS[c], lw=1.2)
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.axvspan(-80, 80, color=COLORS[c], alpha=0.08, lw=0)
        ax.set_title(f"{c} 类（测试记录 {rec[mask][pick]}）")
        ax.set_ylabel("幅值 / mV")
        ax.set_xlim(-250, 400)
    for ax in axes[-1]:
        ax.set_xlabel("相对 R 峰时间 / ms")
    fig.suptitle("四类心搏的典型波形（0.5–40 Hz 带通后）", y=1.00)
    fig.tight_layout()
    save(fig, "fig02_beat_examples.png")


# ---------------------------------------------------------------------------
# 3. Filtering example
# ---------------------------------------------------------------------------
def fig_preprocess():
    import wfdb
    rec = wfdb.rdrecord(str(MITDB / "100"))
    ann = wfdb.rdann(str(MITDB / "100"), "atr")
    sig = np.asarray(rec.p_signal[: 12 * FS, 0], dtype=np.float64)
    sos = butter(3, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")
    fil = sosfiltfilt(sos, sig)
    t = np.arange(len(sig)) / FS

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6))
    axes[0].plot(t, sig, color="#B0B0B0", lw=0.8, label="原始信号")
    axes[0].plot(t, fil, color="#2E6F9E", lw=1.0, label="0.5–40 Hz 带通")
    axes[0].set_xlim(0, 12)
    axes[0].set_ylabel("幅值 / mV")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[0].set_title("基线漂移与高频噪声抑制")

    keep = (ann.sample > 2 * FS) & (ann.sample < 5 * FS)
    axes[1].plot(t, fil, color="#2E6F9E", lw=1.1)
    axes[1].scatter(ann.sample[keep] / FS, fil[ann.sample[keep]],
                    color="#C0392B", s=16, zorder=3, label="R 峰")
    axes[1].set_xlim(2, 5)
    axes[1].set_xlabel("时间 / s")
    axes[1].set_ylabel("幅值 / mV")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].set_title("局部放大：R 峰位置")
    fig.tight_layout()
    save(fig, "fig03_filtering.png")


# ---------------------------------------------------------------------------
# 4. Denoising example
# ---------------------------------------------------------------------------
@torch.no_grad()
def _denoise(dn, noisy):
    mu = noisy.mean(axis=1, keepdims=True)
    sd = noisy.std(axis=1, keepdims=True) + 1e-6
    x = (noisy - mu) / sd
    out = dn(torch.from_numpy(x).float().unsqueeze(1)).squeeze(1).numpy()
    return out * sd + mu


def fig_denoise():
    X, y, rec, _ = load_beats()
    mask = np.isin(rec, DS2)
    X = X[mask]
    amp = np.ptp(X, axis=1)
    idx = int(np.argmin(np.abs(amp - np.percentile(amp, 70))))
    clean = X[idx:idx + 1]
    bank = load_nstdb_noise()
    dn = DnCNN1d(channels=32, depth=6, kernel=11)
    ckpt = torch.load(MODELS / "denoiser_best.pt", map_location="cpu",
                      weights_only=False)
    dn.load_state_dict(ckpt["model"])
    dn.eval()

    t = (np.arange(clean.shape[1]) - WIN_BEFORE) / FS * 1000
    fig, axes = plt.subplots(4, 2, figsize=(8.2, 7.3), sharex=True)
    for col, kind in enumerate(["em", "bw"]):
        rng = np.random.default_rng(2026 + col)
        noisy = add_noise(clean, bank, rng, split="test", snr_db=5.0,
                          kind=kind)
        den = _denoise(dn, noisy)
        cla = np.stack([classical_filter(x) for x in noisy])
        rows = [("干净心搏", clean[0], "#333333"),
                ("含噪（5 dB）", noisy[0], "#9E9E9E"),
                ("传统滤波", cla[0], "#E08A1E"),
                ("DnCNN 去噪", den[0], "#2E6F9E")]
        for row, (label, s, color) in enumerate(rows):
            ax = axes[row, col]
            ax.plot(t, s, color=color, lw=1.15)
            ax.axvline(0, color="k", ls=":", lw=0.7)
            ax.set_xlim(-250, 400)
            if row == 0:
                ax.set_title({"em": "电极运动伪迹（EM）",
                              "bw": "基线漂移（BW）"}[kind])
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == 3:
                ax.set_xlabel("相对 R 峰时间 / ms")
            # annotate SNR for the noisy / output traces
            if row in (1, 2, 3):
                from utils import snr_db
                val = snr_db(clean[0], s)
                ax.text(0.985, 0.10, f"SNR {val:.1f} dB", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="none", alpha=0.75))
    fig.suptitle("同一条心搏在 5 dB 噪声下的去噪效果", y=0.995)
    fig.tight_layout()
    save(fig, "fig04_denoising_example.png")


# ---------------------------------------------------------------------------
# 5. Training curves
# ---------------------------------------------------------------------------
def fig_training():
    cnn_path = RESULTS / "cnn_light_history.json"
    if not cnn_path.exists():
        cnn_path = RESULTS / "cnn_history.json"
    cnn = load_json(cnn_path)
    den = load_json(RESULTS / "denoiser_history.json")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    ep = [r["epoch"] for r in cnn]
    axes[0].plot(ep, [r["train_loss"] for r in cnn], color="#2E6F9E",
                 marker="o", ms=2.5, label="训练损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("交叉熵损失", color="#2E6F9E")
    axes[0].tick_params(axis="y", colors="#2E6F9E")
    ax2 = axes[0].twinx()
    ax2.plot(ep, [r["val_macro_f1"] for r in cnn], color="#C0392B",
             marker="s", ms=2.5, label="验证 Macro-F1")
    ax2.set_ylabel("验证 Macro-F1", color="#C0392B")
    ax2.tick_params(axis="y", colors="#C0392B")
    ax2.grid(False)
    axes[0].set_title("(a) 轻量 1D CNN 分类器")
    axes[0].set_xticks(ep[::max(1, len(ep) // 8)])

    ep = [r["epoch"] for r in den]
    axes[1].plot(ep, [r["train_loss"] for r in den], color="#2E6F9E",
                 marker="o", ms=2.5, label="训练 MSE")
    axes[1].plot(ep, [r["val_mse"] for r in den], color="#E08A1E",
                 marker="s", ms=2.5, label="验证 MSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("(b) DnCNN 去噪器")
    ax3 = axes[1].twinx()
    ax3.plot(ep, [r["val_delta_snr"] for r in den], color="#6C5B7B",
             marker="^", ms=2.5, label="ΔSNR")
    ax3.set_ylabel("验证 ΔSNR / dB", color="#6C5B7B")
    ax3.tick_params(axis="y", colors="#6C5B7B")
    ax3.grid(False)
    lines = axes[1].get_lines() + ax3.get_lines()
    fig.legend(lines, [ln.get_label() for ln in lines],
               frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig05_training_curves.png")


# ---------------------------------------------------------------------------
# 6. Confusion matrices
# ---------------------------------------------------------------------------
def fig_confusion():
    d = np.load(RESULTS / "test_predictions.npz", allow_pickle=True)
    y = d["y"]
    metrics = load_json(RESULTS / "classification_metrics.json")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, key, title in [(axes[0], "cnn", "1D CNN"),
                           (axes[1], "mlp", "MLP 基线")]:
        cm = confusion_matrix(y, d[key], labels=list(range(4)))
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j] * 100:.1f}%",
                        ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.55 else "#222222")
        ax.set_xticks(range(4), CLASS_NAMES)
        ax.set_yticks(range(4), CLASS_NAMES)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        m = metrics[key]
        ax.set_title(f"{title}\nAcc={m['accuracy']:.3f}, Macro-F1={m['macro_f1']:.3f}")
        ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.026, pad=0.03,
                 label="行归一化比例")
    save(fig, "fig06_confusion_matrices.png")


# ---------------------------------------------------------------------------
# 7. Per-class F1
# ---------------------------------------------------------------------------
def fig_per_class():
    metrics = load_json(RESULTS / "classification_metrics.json")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(4)
    w = 0.36
    for j, (key, label, color) in enumerate(
            [("cnn", "1D CNN", "#2E6F9E"), ("mlp", "MLP 基线", "#E08A1E")]):
        vals = [metrics[key]["report"][c]["f1-score"] for c in CLASS_NAMES]
        ax.bar(x + (j - 0.5) * w, vals, width=w, color=color, label=label)
        for xi, v in zip(x + (j - 0.5) * w, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
    supports = [int(metrics["cnn"]["report"][c]["support"]) for c in CLASS_NAMES]
    ax.set_xticks(x, [f"{c}\n(n={s:,})" for c, s in zip(CLASS_NAMES, supports)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("独立测试集上各类别 F1")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig07_per_class_f1.png")


# ---------------------------------------------------------------------------
# 8. Noise robustness
# ---------------------------------------------------------------------------
def fig_noise():
    rows = load_json(RESULTS / "binary_noise_results.json")
    clean = next(r for r in rows if r["condition"] == "clean")
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.6))
    kinds = [("bw", "基线漂移"), ("em", "电极运动"), ("ma", "肌电干扰")]
    for ax, (kind, title) in zip(axes.flat[:3], kinds):
        sub = [r for r in rows if r["kind"] == kind and r["snr_db"] in (5, 10, 15)]
        snrs = [r["snr_db"] for r in sub]
        for method, label in [("raw", "未处理"), ("classical", "传统滤波"),
                              ("denoised", "DnCNN 去噪")]:
            ax.plot(snrs, [r[method]["macro_f1"] for r in sub],
                    marker="o" if method == "raw" else "s",
                    color=METHOD_COLORS[method], label=label)
        ax.axhline(clean["denoised"]["macro_f1"], color="#2E6F9E",
                   ls="--", lw=0.8, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("输入 SNR / dB")
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0, 0.9)
        ax.set_xticks([5, 10, 15])
    ax = axes.flat[3]
    for method, label in [("raw", "未处理"), ("classical", "传统滤波"),
                          ("denoised", "DnCNN 去噪")]:
        vals = []
        for snr in (5, 10, 15):
            sub = [r for r in rows if r["kind"] in ("bw", "em", "ma")
                   and r["snr_db"] == snr]
            vals.append(np.mean([r[method]["macro_f1"] for r in sub]))
        ax.plot([5, 10, 15], vals, marker="o",
                color=METHOD_COLORS[method], label=label)
    ax.axhline(clean["denoised"]["macro_f1"], color="#2E6F9E", ls="--", lw=0.8)
    ax.set_title("三种噪声平均")
    ax.set_xlabel("输入 SNR / dB")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 0.9)
    ax.set_xticks([5, 10, 15])
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("二分类筛查的噪声鲁棒性（Macro-F1）", y=1.00)
    fig.tight_layout()
    save(fig, "fig08_noise_robustness.png")


# ---------------------------------------------------------------------------
# 8b. Binary screening: confusion matrix and ROC
# ---------------------------------------------------------------------------
def fig_binary():
    d = np.load(RESULTS / "binary_predictions.npz")
    y, pred, prob = d["y"], d["pred"], d["prob"]
    m = load_json(RESULTS / "binary_metrics.json")
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6))
    ax = axes[0]
    cmn = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{cmn[i, j] * 100:.1f}%",
                    ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.55 else "#222222")
    ax.set_xticks([0, 1], ["预测正常", "预测异常"])
    ax.set_yticks([0, 1], ["真实正常", "真实异常"])
    ax.set_title(f"测试集混淆矩阵（阈值 {m['threshold']:.2f}）\nAcc={m['accuracy']:.3f}, "
                 f"Macro-F1={m['macro_f1']:.3f}")
    ax.grid(False)

    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, prob)
    ax = axes[1]
    ax.plot(fpr, tpr, color="#2E6F9E", lw=1.6,
            label=f"AUC = {m['auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=0.8)
    points = load_json(RESULTS / "binary_threshold_metrics.json")
    palette = ["#C0392B", "#E08A1E", "#6C5B7B"]
    for i, (point, color) in enumerate(zip(points, palette)):
        t = point['threshold']
        x, yy = 1-point['specificity'], point['sensitivity']
        ax.scatter(x, yy, color=color, zorder=3, s=28)
        ax.annotate(f"阈值 {t:.2f}\nSe={yy:.2f}, Sp={1-x:.2f}",
                    (x, yy), xytext=(.48, .20 + .17*i), textcoords="axes fraction",
                    fontsize=7, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=.7))
    ax.set_xlabel("假阳性率（1 − 特异性）")
    ax.set_ylabel("真阳性率（敏感度）")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("受试者工作特征曲线")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig06b_binary_roc.png")


# ---------------------------------------------------------------------------
# 8c. Denoising SNR improvement
# ---------------------------------------------------------------------------
def fig_denoise_snr():
    rows = load_json(RESULTS / "binary_noise_results.json")
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.1), sharey=True)
    for ax, (kind, title) in zip(axes, [("bw", "基线漂移"),
                                        ("em", "电极运动"),
                                        ("ma", "肌电干扰")]):
        sub = [r for r in rows if r["kind"] == kind and r["snr_db"] in (5, 10, 15)]
        x = [r["snr_db"] for r in sub]
        ax.plot(x, [r["snr_raw"] for r in sub], color="#9E9E9E",
                marker="o", label="输入")
        ax.plot(x, [r["snr_classical"] for r in sub], color="#E08A1E",
                marker="s", label="传统滤波")
        ax.plot(x, [r["snr_denoised"] for r in sub], color="#2E6F9E",
                marker="^", label="DnCNN")
        for xi, r in zip(x, sub):
            ax.annotate(f"{r['snr_denoised'] - r['snr_raw']:+.1f}",
                        (xi, r["snr_denoised"]), xytext=(0, 5),
                        textcoords="offset points", ha="center",
                        fontsize=6.5, color="#2E6F9E")
        ax.plot([5, 15], [5, 15], color="#BBBBBB", ls=":", lw=0.7)
        ax.set_title(title)
        ax.set_xlabel("输入 SNR / dB")
        ax.set_xticks([5, 10, 15])
    axes[0].set_ylabel("输出 SNR / dB")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("NSTDB 噪声下的输出信噪比（标注为 ΔSNR/dB）", y=1.02)
    fig.tight_layout()
    save(fig, "fig05b_denoising_snr.png")


# ---------------------------------------------------------------------------
# 9. Architecture
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, color="#EAF0F6", edge="#2E6F9E"):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.03,rounding_size=0.08",
                       linewidth=0.9, edgecolor=edge, facecolor=color)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=7.5)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.2, 8.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    stages = [
        ("输入\n1 × 256", "单导联心搏，R 峰对齐"),
        ("Conv1d 16@7 + ReLU\nMaxPool/2", "16 × 128"),
        ("Conv1d 32@5 + ReLU\nMaxPool/2", "32 × 64"),
        ("Conv1d 64@3 + ReLU\nMaxPool/2", "64 × 32"),
        ("全局平均池化 + Dropout 0.2", "64 × 1"),
        ("全连接 2", "正常 / 异常"),
    ]
    ys = np.linspace(10.9, 1.2, len(stages))
    for i, (title, shape) in enumerate(stages):
        color = "#F4F7FA" if i not in (0, len(stages) - 1) else "#E4EEF8"
        _box(ax, 4.4, ys[i], 6.2, 0.78, title, color=color)
        ax.text(7.75, ys[i], shape, ha="left", va="center", fontsize=7,
                color="#555555")
        if i < len(stages) - 1:
            ax.annotate("", xy=(4.4, ys[i + 1] + 0.45),
                        xytext=(4.4, ys[i] - 0.45),
                        arrowprops=dict(arrowstyle="-|>", color="#2E6F9E",
                                        lw=1.0))
    ax.text(1.0, 11.8, "ECGCNN-Light", fontsize=13, fontweight="bold",
            color="#2E6F9E")
    ax.text(1.0, 11.52, "参数量 9,058；FP32 约 36 KB；INT8 约 9 KB", fontsize=7.5,
            color="#555555")
    save(fig, "fig09_cnn_architecture.png")


# ---------------------------------------------------------------------------
# 10. OpenECG system integration
# ---------------------------------------------------------------------------
def fig_system():
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def row(y, boxes, color, label):
        ax.text(0.2, y + 0.72, label, fontsize=9, fontweight="bold",
                color="#333333")
        n = len(boxes)
        w = 1.75
        gap = (12 - 0.8 - n * w) / max(n - 1, 1)
        for i, text in enumerate(boxes):
            x = 0.4 + w / 2 + i * (w + gap)
            _box(ax, x, y, w, 0.95, text, color=color)
            if i < n - 1:
                ax.annotate("", xy=(x + w / 2 + gap - 0.02, y),
                            xytext=(x + w / 2 + 0.02, y),
                            arrowprops=dict(arrowstyle="-|>", color="#777777",
                                            lw=0.9))

    row(4.5, ["电极 / 导联", "模拟前端\nAD8232 / ADS1292", "MCU 采样\n360 Hz / 11 bit",
              "BLE 传输", "OpenECG 上位机"], "#F2F2F2", "硬件采集链路（OpenECG）")
    row(2.2, ["0.5–40 Hz\n带通 + 60 Hz 陷波", "R 峰检测\n与心搏切分", "DnCNN\n去噪",
              "1D CNN\n二分类", "显示 / 预警"], "#E4EEF8", "软件处理链路（本报告实现）")
    # dashed highlight
    xs = [0.4, 12 - 0.4]
    rect = FancyBboxPatch((2.9, 1.45), 6.4, 1.55,
                          boxstyle="round,pad=0.05,rounding_size=0.1",
                          fill=False, ec="#C0392B", lw=1.0, ls="--")
    ax.add_patch(rect)
    ax.text(6.1, 3.18, "本报告新增的深度学习模块", ha="center", fontsize=8,
            color="#C0392B")
    save(fig, "fig10_system_integration.png")


# ---------------------------------------------------------------------------
# 11. End-to-end demo
# ---------------------------------------------------------------------------
def fig_demo():
    path = RESULTS / "demo_record208.npz"
    if not path.exists():
        print("demo file missing, skip")
        return
    d = np.load(path, allow_pickle=True)
    sig = d["signal"]
    peaks = d["peaks"]
    pred = d["pred"]
    true_cls = [str(x) for x in d["true_cls"]]
    info = load_json(RESULTS / "demo_info.json")
    t = np.arange(len(sig)) / FS
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.plot(t, sig, color="#333333", lw=0.9)
    for c in CLASS_NAMES:
        idx = [i for i, p in enumerate(pred) if CLASS_NAMES[int(p)] == c]
        if not idx:
            continue
        ax.scatter(peaks[idx] / FS, sig[peaks[idx]], s=42,
                   color=COLORS[c], edgecolor="white", linewidth=0.6,
                   zorder=3)
    # Mark only mismatches, otherwise the labels overlap heavily.
    for i, p in enumerate(peaks):
        if true_cls[i] != CLASS_NAMES[pred[i]]:
            ax.annotate(f"T:{true_cls[i]}\nP:{CLASS_NAMES[pred[i]]}",
                        (p / FS, sig[p]), xytext=(0, 10),
                        textcoords="offset points", ha="center",
                        fontsize=6, color="#C0392B")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("幅值 / mV")
    ax.set_title("OpenECG 四分类模型演示：记录 208 的 10 s 片段（圆点=预测类别）")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                      markerfacecolor=COLORS[c], markeredgecolor="white",
                      label=f"{c}（预测）") for c in CLASS_NAMES]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="upper right",
              fontsize=7)
    ax.text(0.01, 0.03,
            f"R 峰检测：Se={info['detection']['se']:.3f}, "
            f"PPV={info['detection']['ppv']:.3f}（记录 100）",
            transform=ax.transAxes, fontsize=7, color="#555555")
    fig.tight_layout()
    save(fig, "fig11_openecg_demo.png")


def main():
    fig_dataset()
    fig_beats()
    fig_preprocess()
    fig_denoise()
    fig_training()
    fig_binary()
    fig_confusion()
    fig_per_class()
    fig_noise()
    fig_denoise_snr()
    fig_architecture()
    fig_system()
    fig_demo()


if __name__ == "__main__":
    main()
