"""在自定义合成信号上演示处理流程，或只用已有分类结果重绘图。

用法（在项目根目录执行）：
    D:\\openecg\\.venv\\Scripts\\python.exe openecg_ext\\analysis\\run_demo.py
    ... --noise 0.01 --out results/openecg_demo
    python openecg_ext/analysis/run_demo.py --plot-existing

--plot-existing 只复原确定性的绘图输入并读取保存的概率，不加载模型或推理。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pipeline import (FS, WIN, classify, detect_rpeaks, load_classifier,  # noqa: E402
                      preprocess, segment_beats, to_model_rate)
from simulate import simulate_openecg  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--heart-rate", type=float, default=60.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "openecg_demo")
    ap.add_argument("--plot-existing", action="store_true",
                    help="Redraw from saved demo probabilities without model loading/inference")
    args = ap.parse_args()

    saved = None
    if args.plot_existing:
        saved = json.loads((args.out / "openecg_demo.json").read_text(encoding="utf-8-sig"))
        args.duration = float(saved["duration_s"])
        args.heart_rate = float(saved["heart_rate_bpm_target"])
        args.noise = float(saved["noise"])

    t, sim = simulate_openecg(args.duration, 1000, args.heart_rate, args.noise)
    sig = to_model_rate(sim, fs_in=1000)
    filtered = preprocess(sig)
    peaks = detect_rpeaks(filtered)
    beats, used = segment_beats(filtered, peaks)

    if saved is None:
        model, threshold = load_classifier()
        prob = classify(model, beats)
        pred = (prob >= threshold).astype(int)
    else:
        threshold = float(saved["threshold"])
        prob = np.asarray(saved["prob_abnormal"], dtype=float)
        pred = np.asarray(saved["pred"], dtype=int)
        if len(peaks) != saved["n_peaks"] or len(beats) != saved["n_beats"]:
            raise ValueError("Saved demo and deterministic drawing inputs have different beat counts")
        if len(prob) != len(beats) or not np.array_equal(pred, prob >= threshold):
            raise ValueError("Saved probabilities and predictions do not match")
        rr_mean = float(np.mean(np.diff(used) / FS))
        if not np.isclose(rr_mean, saved["mean_rr_s"], atol=1e-12):
            raise ValueError("Saved demo timing does not match the reconstructed plot inputs")
    labels = ["异常" if p else "正常" for p in pred]

    rr = np.diff(used) / FS
    result = {
        "source": "Custom Gaussian ECG inspired by OpenECG; modified waveform parameters, arbitrary amplitude",
        "fs_sim_hz": 1000,
        "fs_model_hz": FS,
        "duration_s": args.duration,
        "heart_rate_bpm_target": args.heart_rate,
        "noise": args.noise,
        "n_peaks": int(len(peaks)),
        "n_beats": int(len(beats)),
        "heart_rate_bpm_measured": float(60.0 / np.median(rr)) if len(rr) else None,
        "mean_rr_s": float(np.mean(rr)) if len(rr) else None,
        "threshold": threshold,
        "prob_abnormal": [round(float(p), 4) for p in prob],
        "pred": [int(p) for p in pred],
        "n_abnormal": int(pred.sum()),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    if saved is None:
        (args.out / "openecg_demo.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        result = saved

    t_model = np.arange(len(sig)) / FS
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.2),
                             gridspec_kw={"height_ratios": [1.0, 1.0, 1.25]})

    ax = axes[0]
    ax.plot(t, sim, color="#1f4e79", lw=1.0)
    ax.set_title("自定义合成心电（1000 Hz，%d s，%.0f bpm）" % (args.duration, args.heart_rate))
    ax.set_xlabel("时间 / s"); ax.set_ylabel("幅值 / a.u.")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t_model, filtered, color="#2E6F9E", lw=1.0)
    ax.plot(used / FS, filtered[used], "o", ms=4, mfc="none", mec="#C0392B", mew=1.2,
            label="R 峰（检出 %d 个）" % len(used))
    ax.set_title("重采样至 360 Hz，0.5–40 Hz 带通 + 60 Hz 陷波")
    ax.set_xlabel("时间 / s"); ax.set_ylabel("幅值 / a.u.")
    ax.legend(frameon=False, loc="upper right"); ax.grid(alpha=0.3)

    ax = axes[2]
    for i, (p, label) in enumerate(zip(prob, labels)):
        ax.plot(np.arange(WIN), beats[i], lw=0.9,
                color="#C0392B" if label == "异常" else "#2E6F9E", alpha=0.85)
    ax.set_title("切分心搏（256 点）与分类结果：异常 %d / %d，阈值 %.4f"
                 % (int(pred.sum()), len(pred), threshold))
    ax.set_xlabel("拍内采样点"); ax.set_ylabel("幅值 / a.u.")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out / "openecg_sim.png", dpi=200)
    print(json.dumps({k: result[k] for k in
                      ("n_peaks", "n_beats", "heart_rate_bpm_measured",
                       "threshold", "n_abnormal")}, ensure_ascii=False))
    print("saved:", args.out / "openecg_sim.png")


if __name__ == "__main__":
    main()
