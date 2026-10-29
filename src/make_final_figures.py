"""Draw the final report figures from current caches only.

No network is instantiated and no model forward is called. Two safe raw-data
plotting functions are reused from make_figures: representative annotated beats
and the filtering illustration. Their output directory is redirected in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve


CLASSES = ["N", "S", "V", "F"]
KINDS = [("bw", "基线漂移"), ("em", "电极运动"), ("ma", "肌电")]
CLASS_COLOR = {"N": "#2E6F9E", "S": "#DB8E29", "V": "#BE493C", "F": "#79618E"}
METHODS = [("raw", "未处理", "#888888", "o"),
           ("classical", "传统滤波", "#DC8C22", "s"),
           ("denoised", "DnCNN", "#2E6F9E", "^")]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw(root):
    final = root / "results" / "report_final"
    cv = root / "results" / "workpoint_cv_v2"
    required = [final / name for name in (
        "report_data.json", "dataset_accounting.json", "classification_metrics.json",
        "classification_predictions.npz", "cnn_light_history.json", "denoiser_history.json",
        "binary_noise_results.json", "denoising_example.npz", "demo_info.json", "demo_record208.npz")]
    required += [cv / name for name in ("ds2_summary.json", "oof_selection.json", "frozen_workpoints.json", "ds2_predictions.npz")]
    absent = [str(p) for p in required if not p.exists()]
    if absent:
        raise FileNotFoundError("Current results incomplete; no old-result fallback:\n" + "\n".join(absent))
    data = read(final / "report_data.json")
    accounting = read(final / "dataset_accounting.json")
    metrics = read(final / "classification_metrics.json")
    noise = read(final / "binary_noise_results.json")
    if isinstance(noise, dict):
        noise = noise["rows"]
    noise = [r for r in noise if r["kind"] != "clean"]
    by_noise = {(r["kind"], int(r["snr_db"])): r for r in noise}
    if set(by_noise) != {(k, s) for k, _ in KINDS for s in (5, 10, 15)}:
        raise ValueError("Exactly nine current noise conditions are required")
    binary = read(cv / "ds2_summary.json")
    oof = read(cv / "oof_selection.json")
    frozen = read(cv / "frozen_workpoints.json")
    protected = {str(cv / name): digest(cv / name) for name in ("frozen_workpoints.json", "final_binary.pt", "ds2_predictions.npz", "ds2_summary.json")}
    assert protected[str(cv / "final_binary.pt")] == frozen["final_model_sha256"]
    assert protected[str(cv / "ds2_predictions.npz")] == binary["cached_predictions_sha256"]
    figure_dir = final / "figures"
    figure_dir.mkdir(exist_ok=True)
    if Path(r"C:\Windows\Fonts\msyh.ttc").exists():
        font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "figure.dpi": 120, "savefig.dpi": 220,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .22, "grid.linestyle": "--",
    })
    manifest = {"policy": "cached predictions and existing raw signals only; no model inference",
                "sources": {str(p): digest(p) for p in required}, "figures": []}

    def save(fig, name, sources, annotations=None):
        path = figure_dir / name
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        manifest["figures"].append({"filename": name, "path": str(path), "sha256": digest(path),
                                    "sources": sources, "annotation_values": annotations or {}})
        print("saved", name, flush=True)

    # Raw-data illustrations. Neither function instantiates or calls a model.
    sys.path.insert(0, str(root / "src"))
    import make_figures as legacy
    legacy.FIGURES = figure_dir
    legacy.fig_beats()
    legacy.fig_preprocess()
    for name in ("fig02_beat_examples.png", "fig03_filtering.png"):
        manifest["figures"].append({"filename": name, "path": str(figure_dir / name),
                                    "sha256": digest(figure_dir / name), "sources": ["processed/raw WFDB signals"],
                                    "annotation_values": {"model_inference": False}})
    # Restore the slightly larger figure typography after the safe import.
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})

    # Four-part accounting. Two panels retain visibility of the small exclusions
    # without using a logarithmic axis for stacked quantities.
    groups = [g for g in accounting["groups"] if g.get("key") != "total"]
    assert len(groups) == 4
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7), gridspec_kw={"width_ratios": [1.3, 1]})
    labels = [f"{g['label']}\n{g['n_records']} 条记录" for g in groups]
    for ax, indices, title in ((axes[0], [0, 1], "训练验证与 DS2"), (axes[1], [2, 3], "单列排除样本")):
        bottom = np.zeros(len(indices))
        for c in CLASSES:
            values = [groups[i]["counts"][c] for i in indices]
            ax.bar(np.arange(len(indices)), values, bottom=bottom, color=CLASS_COLOR[c], width=.58, label=c)
            if indices == [0, 1] and c in ("N", "V"):
                for j, value in enumerate(values):
                    ax.text(j, bottom[j] + value/2, f"{value:,}", ha="center", va="center",
                            fontsize=8.5, color="white")
            bottom += values
        for j, i in enumerate(indices):
            ax.text(j, bottom[j], f"{groups[i]['n']:,}", ha="center", va="bottom", fontsize=11)
        ax.set_xticks(np.arange(len(indices)), [labels[i] for i in indices])
        ax.set_ylim(0, float(bottom.max()) * 1.2)
        ax.set_title(title); ax.set_ylabel("心搏数")
    handles = [Line2D([0], [0], color=CLASS_COLOR[c], lw=7, label=c) for c in CLASSES]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(.5, -.035))
    fig.suptitle("四段式样本口径  合计 101,451 拍", y=1.01)
    fig.tight_layout(rect=(0, .05, 1, .95))
    save(fig, "fig01_dataset_distribution.png", ["dataset_accounting.json"],
         {"group_n": [g["n"] for g in groups], "ds2_counts": groups[1]["counts"], "total": accounting["total"]["n"]})

    # Hardware/software distinction. Hardware words describe a design only.
    fig, ax = plt.subplots(figsize=(9.0, 4.1)); ax.set(xlim=(0, 12), ylim=(0, 5.7)); ax.axis("off")

    def boxes(y, strings, color, heading, dashed=False):
        ax.text(.25, y+.65, heading, fontsize=10, fontweight="bold", va="bottom")
        width = 2.03; gap = .31
        for i, text in enumerate(strings):
            x = .2 + i*(width+gap)
            p = FancyBboxPatch((x, y-.45), width, .9, boxstyle="round,pad=.04,rounding_size=.08",
                               fc=color, ec="#778693", lw=.9, linestyle="--" if dashed else "-")
            ax.add_patch(p); ax.text(x+width/2, y, text, ha="center", va="center", fontsize=9)
            if i < len(strings)-1:
                ax.annotate("", (x+width+gap-.04, y), (x+width+.04, y),
                            arrowprops={"arrowstyle":"->", "color":"#777777", "lw":.9})

    boxes(4.35, ["电极与导联\n候选设计", "模拟前端\n待适配 AFE", "MCU 采样\n待实接验证", "通信接口\n待联调", "OpenECG\n接入设计"],
          "#F3F3F3", "硬件与通信  候选接入设计 尚未实接测试", dashed=True)
    boxes(2.1, ["WFDB 读取\n离线零相位滤波", "R 峰检测/对齐\n心搏切分", "DnCNN\n可选去噪", "CNN 二分类筛查\n或四分类演示", "离线显示\n与结果评价"],
          "#E7EFF6", "已运行软件  离线处理与评价")
    ax.text(6, .58, "零相位处理使用未来样本；当前流程不代表在线实时部署", ha="center", fontsize=9)
    ax.text(6, .19, "演示加载四分类权重，二分类工作点来自独立冻结记录", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "fig10_system_integration.png", ["experiment_protocol.json", "demo_info.json"],
         {"hardware_tested": False, "offline": True, "demo_task": "four_class"})

    # Training traces: unlike the old plot, normalized training MSE and physical
    # validation MSE are explicitly plotted on different axes with their units.
    cnn_history = read(final / "cnn_light_history.json")
    dn_history = read(final / "denoiser_history.json")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.1))
    ep = [r["epoch"] for r in cnn_history]
    left = axes[0]
    l1, = left.plot(ep, [r["train_loss"] for r in cnn_history], color="#2E6F9E", label="训练交叉熵")
    right = left.twinx()
    l2, = right.plot(ep, [r["val_macro_f1"] for r in cnn_history], color="#BE493C", label="验证 Macro-F1")
    left.set(xlabel="Epoch", ylabel="训练交叉熵", title="(a) 排除 201 后的四分类 CNN")
    right.set_ylabel("验证 Macro-F1"); right.grid(False)
    left.legend(handles=[l1,l2], frameon=False, loc="best")
    ep = [r["epoch"] for r in dn_history]
    left = axes[1]
    l1, = left.plot(ep, [r["train_loss"] for r in dn_history], color="#2E6F9E", label="训练 MSE（标准化域）")
    right = left.twinx()
    l2, = right.plot(ep, [r["val_mse"] for r in dn_history], color="#DB8E29", label="验证 MSE（mV²）")
    left.set(xlabel="Epoch", ylabel="训练 MSE（标准化域）", title="(b) DnCNN  两个纵轴的单位与尺度不同")
    right.set_ylabel("验证 MSE / mV²"); right.grid(False)
    left.legend(handles=[l1,l2], frameon=False, loc="best")
    fig.tight_layout(h_pad=2)
    save(fig, "fig05_training_curves.png", ["cnn_light_history.json", "denoiser_history.json"],
         {"denoiser_train_mse_unit": "standardized", "denoiser_val_mse_unit": "mV^2"})

    # Frozen binary results: cached inference only.
    with np.load(cv / "ds2_predictions.npz", allow_pickle=False) as p:
        y, prob = p["y"], p["prob"]
    primary = binary["working_points"]["sp90_max_se"]
    pred = np.asarray(prob, dtype=float) >= primary["threshold"]
    cm = confusion_matrix(y, pred, labels=[0, 1])
    cmn = cm / cm.sum(1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7))
    axes[0].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{cm[i,j]:,}\n{cmn[i,j]*100:.1f}%", ha="center", va="center", fontsize=11,
                         color="white" if cmn[i,j]>.55 else "#222222")
    axes[0].set_xticks([0,1], ["预测正常", "预测异常"]); axes[0].set_yticks([0,1], ["真值正常", "真值异常"])
    axes[0].set_title(f"主阈值 {primary['threshold']:.4f}\nAcc={primary['accuracy']:.4f}  Macro-F1={primary['macro_f1']:.4f}")
    axes[0].grid(False)
    fpr, tpr, _ = roc_curve(y, prob)
    axes[1].plot(fpr, tpr, color="#2E6F9E", lw=1.7, label=f"DS2 AUC={binary['auc']:.4f}")
    axes[1].plot([0,1], [0,1], ls=":", color="#aaaaaa", lw=.9)
    for key, label, color, marker in [("sp90_max_se", "OOF Sp≥90% 主点", "#357744", "*"),
                                      ("youden_max", "OOF Youden 点", "#79618E", "o"),
                                      ("default_050", "默认 0.50", "#888888", "s")]:
        r = binary["working_points"][key]
        axes[1].scatter(1-r["specificity"], r["sensitivity"], s=64 if marker=="*" else 25,
                        marker=marker, color=color, label=f"{label}  t={r['threshold']:.4f}", zorder=3)
    axes[1].set(xlim=(0,1), ylim=(0,1.02), xlabel="1 − Sp", ylabel="Se", title="DS2 ROC 与 OOF 选定工作点")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout(w_pad=1.8)
    save(fig, "fig06b_binary_roc.png", ["workpoint_cv_v2/ds2_predictions.npz", "workpoint_cv_v2/ds2_summary.json", "workpoint_cv_v2/oof_selection.json"],
         {"ds2_auc":binary["auc"], "oof_auc":oof["oof_auc"], "primary_threshold":primary["threshold"], "confusion_matrix":cm.tolist()})

    # Cached reconstruction example. No denoiser is loaded for visualization.
    with np.load(final / "denoising_example.npz", allow_pickle=False) as ex:
        cache = {k:ex[k] for k in ex.files}
    clean = np.asarray(cache["clean"]).reshape(-1, np.asarray(cache["clean"]).shape[-1])[0]
    fs = 360.; before = 96
    time_ms = (np.arange(len(clean))-before)/fs*1000
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 6.2), sharex=True)
    for column, (kind, title) in enumerate((("em", "电极运动"), ("bw", "基线漂移"))):
        signals = [("干净心搏", clean, "#333333"),
                   ("含噪输入", np.asarray(cache[kind+"_raw"]).reshape(-1,len(clean))[0], "#888888"),
                   ("传统滤波", np.asarray(cache[kind+"_classical"]).reshape(-1,len(clean))[0], "#DC8C22"),
                   ("DnCNN", np.asarray(cache[kind+"_denoised"]).reshape(-1,len(clean))[0], "#2E6F9E")]
        for row, (label, arr, color) in enumerate(signals):
            ax = axes[row,column]; ax.plot(time_ms, arr, lw=1.1, color=color)
            ax.axvline(0, ls=":", color="#bbbbbb", lw=.7)
            if row==0: ax.set_title(title+"  5 dB")
            if column==0: ax.set_ylabel(label+" / mV")
            if row==3: ax.set_xlabel("相对 R 峰时间 / ms")
            ax.set_xlim(time_ms.min(), time_ms.max())
            if row:
                snr = 10*np.log10((np.sum(clean.astype(float)**2)+1e-12)/(np.sum((clean.astype(float)-arr)**2)+1e-12))
                ax.text(.98,.06,f"SNR {snr:.1f} dB", transform=ax.transAxes, ha="right", fontsize=8)
    fig.suptitle("同一固定心搏的缓存重建结果", y=1.005)
    fig.tight_layout(h_pad=.5,w_pad=1.0)
    save(fig, "fig04_denoising_example.png", ["denoising_example.npz"], {"model_inference":False})

    # Current SNR reconstruction results, including negative gains.
    fig, axes = plt.subplots(1,3,figsize=(8.0,3.15),sharey=True)
    for ax,(kind,title) in zip(axes,KINDS):
        rows=[by_noise[kind,s] for s in (5,10,15)]
        for method,label,color,marker in METHODS:
            ax.plot([5,10,15], [r["snr_"+method] for r in rows], color=color, marker=marker, label=label)
        for s,r in zip((5,10,15),rows):
            delta=r["snr_denoised"]-r["snr_raw"]
            ax.annotate(f"{delta:+.1f}",(s,r["snr_denoised"]),xytext=(0,6),textcoords="offset points",ha="center",fontsize=9,color="#2E6F9E")
        ax.set_title(title);ax.set_xlabel("输入 SNR / dB");ax.set_xticks([5,10,15])
    axes[0].set_ylabel("输出 SNR / dB")
    handles, labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,frameon=False,loc="lower center",bbox_to_anchor=(.5,-.025))
    fig.suptitle("本轮九条件重建 SNR  标注为 ΔSNR / dB",y=1.03)
    fig.tight_layout(rect=(0,.06,1,.96))
    save(fig,"fig05b_denoising_snr.png",["binary_noise_results.json"],
         {f"{k}_{s}":by_noise[k,s]["snr_denoised"]-by_noise[k,s]["snr_raw"] for k,_ in KINDS for s in (5,10,15)})

    # Noise classification: no clean row and no carry-over old-model baseline.
    fig,axes=plt.subplots(2,2,figsize=(7.6,5.2),sharex=True,sharey=True)
    for ax,(kind,title) in zip(axes.flat,KINDS):
        for method,label,color,marker in METHODS:
            ax.plot([5,10,15],[by_noise[kind,s][method]["macro_f1"] for s in (5,10,15)],
                    marker=marker,color=color,label=label)
        ax.set_title(title)
    for method,label,color,marker in METHODS:
        axes[1,1].plot([5,10,15],[np.mean([by_noise[k,s][method]["macro_f1"] for k,_ in KINDS]) for s in (5,10,15)],
                       marker=marker,color=color,label=label)
    axes[1,1].set_title("三种噪声的算术均值")
    for ax in axes.flat:
        ax.set_xticks([5,10,15]);ax.set_ylim(0,1)
    for ax in axes[:,0]:ax.set_ylabel("Macro-F1")
    for ax in axes[1]:ax.set_xlabel("输入 SNR / dB")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,frameon=False,loc="lower center",bbox_to_anchor=(.5,-.025))
    fig.suptitle(f"当前模型与冻结阈值 {primary['threshold']:.4f} 下的噪声分类",y=1.02)
    fig.tight_layout(rect=(0,.06,1,.97))
    save(fig,"fig08_noise_robustness.png",["binary_noise_results.json","workpoint_cv_v2/frozen_workpoints.json"],
         {f"{k}_{s}":{m:by_noise[k,s][m]["macro_f1"] for m,_,_,_ in METHODS} for k,_ in KINDS for s in (5,10,15)})

    # New four-class predictions only.
    with np.load(final/"classification_predictions.npz",allow_pickle=False) as p:
        yy=p["y"];class_pred={k:p[k] for k in ("cnn","mlp")}
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.6))
    for ax,(key,title) in zip(axes,[("cnn","四分类 CNN"),("mlp","MLP 对照")]):
        cm=confusion_matrix(yy,class_pred[key],labels=range(4));cmn=cm/np.maximum(cm.sum(1,keepdims=True),1)
        im=ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j,i,f"{cm[i,j]}\n{cmn[i,j]*100:.1f}%",ha="center",va="center",fontsize=8,
                        color="white" if cmn[i,j]>.55 else "#222222")
        r=metrics[key]
        ax.set_title(f"{title}\nAcc={r['accuracy']:.4f}  Macro-F1={r['macro_f1']:.4f}")
        ax.set_xticks(range(4),CLASSES);ax.set_yticks(range(4),CLASSES)
        ax.set_xlabel("预测类别");ax.set_ylabel("真值类别");ax.grid(False)
    fig.subplots_adjust(left=.07,right=.91,bottom=.17,top=.80,wspace=.27)
    cax=fig.add_axes([.93,.18,.018,.60]);fig.colorbar(im,cax=cax,label="行归一化比例")
    save(fig,"fig06_confusion_matrices.png",["classification_predictions.npz","classification_metrics.json"],
         {k:{"accuracy":metrics[k]["accuracy"],"macro_f1":metrics[k]["macro_f1"]} for k in ("cnn","mlp")})

    # New demo, clearly four-class with prediction markers and truth mismatch.
    info=read(final/"demo_info.json")
    with np.load(final/"demo_record208.npz",allow_pickle=True) as p:
        signal=p["signal"];peaks=p["peaks"].astype(int);pred=p["pred"];truth=[str(v) for v in p["true_cls"]]
    pred_labels=[CLASSES[int(v)] if not isinstance(v,str) else v for v in pred]
    fs=float(info.get("fs",360));time=np.arange(len(signal))/fs
    fig,ax=plt.subplots(figsize=(8.6,3.5));ax.plot(time,signal,color="#333333",lw=.9)
    for c in CLASSES:
        pick=[i for i,label in enumerate(pred_labels) if label==c]
        if pick:
            ax.scatter(peaks[pick]/fs,signal[peaks[pick]],s=36,color=CLASS_COLOR[c],edgecolors="white",linewidths=.5,zorder=3)
    mismatches=[]
    for i,(t,p) in enumerate(zip(truth,pred_labels)):
        if t!=p:
            mismatches.append({"index":i,"true":t,"pred":p})
            ax.annotate(f"T:{t}\nP:{p}",(peaks[i]/fs,signal[peaks[i]]),xytext=(0,13 if signal[peaks[i]]>=0 else -28),
                        textcoords="offset points",ha="center",fontsize=8,color="#BE493C")
    handles=[Line2D([0],[0],marker="o",linestyle="",color=CLASS_COLOR[c],label=f"{c}（预测）") for c in CLASSES]
    ax.legend(handles=handles,ncol=4,frameon=False,loc="upper right")
    ax.set(xlabel="时间 / s",ylabel="幅值 / mV",title=f"四分类模型演示  记录 {info['record']}  真值与预测错分 {len(mismatches)} 拍")
    ax.margins(y=.22);ax.set_xlim(0,time.max())
    fig.tight_layout()
    save(fig,"fig11_openecg_demo.png",["demo_record208.npz","demo_info.json"],
         {"task":"four_class","predictions":pred_labels,"truth":truth,"mismatches":mismatches,"uses_rr":False})

    # Two optional source-aligned supporting figures retained in the code zip.
    fig,ax=plt.subplots(figsize=(6.7,3.5));x=np.arange(4);w=.35
    for j,(key,label,color) in enumerate((("cnn","四分类 CNN","#2E6F9E"),("mlp","MLP","#DB8E29"))):
        values=[metrics[key]["report"][c]["f1-score"] for c in CLASSES]
        ax.bar(x+(j-.5)*w,values,width=w,label=label,color=color)
        for xx,v in zip(x+(j-.5)*w,values):ax.text(xx,v+.015,f"{v:.3f}",ha="center",fontsize=9)
    supports=[int(metrics["cnn"]["report"][c]["support"]) for c in CLASSES]
    ax.set_xticks(x,[f"{c}\n(n={n:,})" for c,n in zip(CLASSES,supports)])
    ax.set(ylim=(0,1.06),ylabel="F1",title="本轮四分类各类 F1");ax.legend(frameon=False)
    fig.tight_layout();save(fig,"fig07_per_class_f1.png",["classification_metrics.json"])

    models=data["cost"]["models"]
    bin_cost=next(m for m in models if m["key"] in ("binary","binary_cnn","cnn_binary"))
    rows=data["architecture"]["binary"]
    fig,ax=plt.subplots(figsize=(6.4,6.2));ax.axis("off");ax.set(xlim=(0,10),ylim=(0,len(rows)+1.4))
    for i,row in enumerate(rows):
        y=len(rows)-i
        rect=FancyBboxPatch((.5,y-.28),6.8,.58,boxstyle="round,pad=.05",fc="#E7EFF6",ec="#748C9F",lw=.9)
        ax.add_patch(rect);ax.text(3.9,y,str(row[1]),ha="center",va="center",fontsize=9)
        ax.text(7.6,y,str(row[2]),ha="left",va="center",fontsize=8.5)
        if i<len(rows)-1:ax.annotate("",(3.9,y-.68),(3.9,y-.34),arrowprops={"arrowstyle":"->","lw":.9,"color":"#748C9F"})
    ax.text(.5,len(rows)+.85,"冻结二分类 CNN",fontsize=14,fontweight="bold")
    ax.text(.5,len(rows)+.4,f"{bin_cost['params']:,} 参数；FP32 {bin_cost['fp32_kib']:.1f} KiB；INT8 权重估算 {bin_cost['int8_kib']:.1f} KiB",fontsize=8.5)
    fig.tight_layout();save(fig,"fig09_cnn_architecture.png",["report_data.json"],{"params":bin_cost["params"]})

    for path,before_hash in protected.items():
        if digest(path)!=before_hash:
            raise RuntimeError("Frozen binary artifact changed during plotting: "+path)
    manifest["frozen_binary_unchanged"] = True
    (final/"figure_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Final figure set complete; frozen binary cache unchanged.",flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=ap.parse_args();draw(args.root.resolve())


if __name__=="__main__":
    main()
