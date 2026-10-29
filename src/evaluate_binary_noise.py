"""Binary screening performance under NSTDB noise, with and without denoising."""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)

from config import DS2, MODELS, RESULTS, SEED
from models import DnCNN1d, ECGCNNLight
from utils import (add_noise, classical_filter, load_beats, load_nstdb_noise,
                   predict_logits, save_json, snr_db)

torch.set_num_threads(4)


@torch.no_grad()
def denoise_batch(model, noisy, batch=512):
    mu = noisy.mean(axis=1, keepdims=True)
    sd = noisy.std(axis=1, keepdims=True) + 1e-6
    xn = (noisy - mu) / sd
    outs = []
    for i in range(0, len(xn), batch):
        xb = torch.from_numpy(xn[i:i + batch]).float().unsqueeze(1)
        outs.append(model(xb).squeeze(1).cpu().numpy())
    return np.concatenate(outs, axis=0) * sd + mu


def bin_metrics(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(np.int64)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "sensitivity": round(float(tp / max(tp + fn, 1)), 4),
        "specificity": round(float(tn / max(tn + fp, 1)), 4),
        "auc": round(float(roc_auc_score(y, prob)), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def main():
    from split_record_level import sha256
    protocol = json.loads((RESULTS / 'val_threshold.json').read_text(encoding='utf-8'))
    assert protocol['frozen']
    for name, digest in protocol['model_sha256'].items():
        assert sha256(MODELS/name) == digest, 'Frozen model changed'
    marker = RESULTS / 'binary_noise_evaluation.json'
    if marker.exists():
        raise RuntimeError('Noise evaluation already started; refusing a second pass')
    save_json(marker, {'status':'started','threshold':protocol['threshold'],
                       'protocol_sha256':sha256(RESULTS/'val_threshold.json')})
    X, y4, rr, rec, _ = load_beats(with_rr=True)
    y = (y4 > 0).astype(np.int64)
    mask = np.isin(rec, DS2)
    Xte, yte, rrte = X[mask], y[mask], rr[mask]

    # Keep all abnormal beats and a random 4,000 normal beats.
    rng = np.random.default_rng(SEED)
    pos = np.flatnonzero(yte == 1)
    neg = rng.choice(np.flatnonzero(yte == 0), size=min(4000, int((yte == 0).sum())),
                     replace=False)
    sel = np.sort(np.concatenate([pos, neg]))
    Xs, ys, rrs = Xte[sel], yte[sel], rrte[sel]
    print("subset", len(ys), "abnormal", int(ys.sum()))

    clf = ECGCNNLight(num_classes=2)
    ck = torch.load(MODELS / "binary_best.pt", map_location="cpu",
                    weights_only=False)
    clf.load_state_dict(ck["model"])
    clf.eval()

    dn = DnCNN1d(channels=32, depth=6, kernel=11)
    dk = torch.load(MODELS / "denoiser_best.pt", map_location="cpu",
                    weights_only=False)
    dn.load_state_dict(dk["model"])
    dn.eval()

    bank = load_nstdb_noise()
    rows = []
    for kind in ("clean", "bw", "em", "ma"):
        snrs = [None] if kind == "clean" else [5, 10, 15]
        for snr in snrs:
            rng = np.random.default_rng(SEED + (0 if snr is None else snr)
                                        + {"clean": 0, "bw": 1, "em": 2,
                                           "ma": 3}[kind] * 100)
            if kind == "clean":
                noisy = Xs.copy()
                label = "clean"
            else:
                noisy = add_noise(Xs, bank, rng, split="test", snr_db=snr,
                                  kind=kind)
                label = f"{kind}_{snr}dB"
            den = denoise_batch(dn, noisy)
            cla = np.stack([classical_filter(x) for x in noisy])
            row = {"condition": label, "kind": kind, "snr_db": snr,
                   "threshold":protocol["threshold"], "n":len(ys)}
            if kind != "clean":
                for name, arr in (("raw", noisy), ("classical", cla),
                                  ("denoised", den)):
                    row[f"snr_{name}"] = round(float(np.mean(
                        [snr_db(c, a) for c, a in zip(Xs, arr)])), 3)
                    row[f"rmse_{name}"] = round(float(np.sqrt(
                        np.mean((Xs - arr) ** 2))), 5)
            for name, arr in (("raw", noisy), ("classical", cla),
                              ("denoised", den)):
                arr_n = np.stack([(x - x.mean()) / (x.std() + 1e-6)
                                  for x in arr])
                logits = predict_logits(clf, arr_n)
                prob = torch.softmax(torch.from_numpy(logits), 1).numpy()[:, 1]
                row[name] = bin_metrics(ys, prob, protocol["threshold"])
            rows.append(row)
            print(label, {k: row[k] for k in
                          ("snr_raw", "snr_denoised", "raw", "denoised")
                          if k in row})
    save_json(RESULTS / "binary_noise_results.json", rows)
    save_json(marker, {'status':'complete','threshold':protocol['threshold'],
                       'n':len(ys), 'n_abnormal':int(ys.sum()),
                       'subset_global_indices':np.flatnonzero(mask)[sel].tolist(),
                       'noise_seed_rule':'SEED + SNR + kind_index*100; unchanged from pre-experiment',
                       'protocol_sha256':sha256(RESULTS/'val_threshold.json')})


if __name__ == "__main__":
    main()
