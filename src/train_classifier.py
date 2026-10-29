"""Train a 1-D CNN (or MLP baseline) for four-class ECG beat classification."""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import CLASS_NAMES, DS1, MODELS, RESULTS, SEED
from models import ECGCNN, ECGCNNLight, MLPBaseline, count_parameters
from utils import (class_counts, load_beats, predict_logits, save_json,
                   set_seed)
from split_record_level import load_record_split, sha256

torch.set_num_threads(4)
DEVICE = "cpu"


class BeatDataset(Dataset):
    def __init__(self, X: np.ndarray, rr: np.ndarray, y: np.ndarray,
                 augment: bool = False):
        self.X = X.astype(np.float32)
        self.rr = rr.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.augment:
            # Small timing jitter, amplitude scaling and residual baseline wander.
            shift = int(np.random.randint(-5, 6))
            if shift:
                x = np.roll(x, shift)
            x *= np.random.uniform(0.9, 1.1)
            t = np.arange(len(x)) / 360.0
            x += np.random.uniform(0.005, 0.03) * np.sin(
                2 * np.pi * np.random.uniform(0.5, 1.5) * t
                + np.random.uniform(0, 2 * np.pi))
            x += np.random.normal(0, 0.008, size=x.shape)
        x = (x - x.mean()) / (x.std() + 1e-6)
        return (torch.from_numpy(x[None, :]).float(),
                torch.from_numpy(self.rr[i]).float(), int(self.y[i]))


def evaluate(model, X, y):
    logits = predict_logits(model, X, device=DEVICE)
    pred = logits.argmax(axis=1)
    return {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "pred": pred,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "mlp"], default="cnn")
    ap.add_argument("--arch", choices=["standard", "light"], default="standard")
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--max-n", type=int, default=18000)
    ap.add_argument("--max-v", type=int, default=3000)
    ap.add_argument("--sampler", action="store_true",
                    help="use mild class-balanced sampling with unweighted loss")
    args = ap.parse_args()

    if (RESULTS / 'val_threshold.json').exists():
        raise RuntimeError('Protocol frozen; training is closed')
    set_seed(SEED)
    X, y, rr, rec, _ = load_beats(with_rr=True)
    tr, va, record_split = load_record_split(rec)
    Xtr, ytr, rrtr = X[tr], y[tr], rr[tr]
    Xva, yva, rrva = X[va], y[va], rr[va]

    # Keep every S/F beat but cap the two majority classes.  This makes each
    # epoch much cheaper while preserving the clinically rare events.
    rng = np.random.default_rng(SEED)
    keep = []
    for c, cap in ((0, args.max_n), (1, None), (2, args.max_v), (3, None)):
        idx = np.flatnonzero(ytr == c)
        if cap is not None and len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.append(idx)
    sel = np.sort(np.concatenate(keep))
    Xtr, ytr, rrtr = Xtr[sel], ytr[sel], rrtr[sel]
    print(f"model={args.model} train={len(Xtr)} val={len(Xva)}")
    print("class counts train:", class_counts(ytr))

    save_json(RESULTS / f'{args.model}_{args.arch}_training_config.json', {
        **vars(args), 'seed':SEED, 'train_records':record_split['train_records'],
        'val_records':record_split['val_records'], 'train_n_after_cap':len(ytr),
        'val_n':len(yva), 'train_class_counts':class_counts(ytr)})
    counts = np.bincount(ytr, minlength=len(CLASS_NAMES)).astype(np.float64)
    # Mild inverse-frequency weights.  Full inverse frequency and square-root
    # inverse frequency both caused validation collapse in pilot runs; a power
    # of 0.35 keeps S/F visible while preserving stable optimisation.
    weights = np.where(counts > 0, counts ** -0.35, 0.0)
    weights = weights / weights.mean()
    print("class weights:", np.round(weights, 3))
    class_weights = torch.tensor(weights, dtype=torch.float32)

    train_ds = BeatDataset(Xtr, rrtr, ytr, augment=True)
    if args.sampler:
        sample_w = (counts ** -0.35)[ytr]
        sample_w = sample_w / sample_w.sum()
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(ytr), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch,
                                  sampler=sampler, num_workers=0,
                                  drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch,
                                  shuffle=True, num_workers=0,
                                  drop_last=True)

    use_extra = args.model == "cnn" and args.arch == "standard"
    if args.model == "cnn":
        model = (ECGCNN(width=args.width) if args.arch == "standard"
                 else ECGCNNLight())
        ckpt_name = "cnn_best.pt" if args.arch == "standard" else "cnn_light_best.pt"
    else:
        model = MLPBaseline()
        ckpt_name = "mlp_best.pt"
    print("parameters:", count_parameters(model))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if args.sampler:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights,
                                      label_smoothing=0.05)

    history = []
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for step, (xb, rb, yb) in enumerate(train_loader, 1):
            opt.zero_grad(set_to_none=True)
            logits = model(xb, rb) if use_extra else model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.detach()) * len(xb)
            n_seen += len(xb)
            if step % 50 == 0:
                print(f"  epoch {epoch:02d} step {step:03d} "
                      f"loss {running / n_seen:.4f}", flush=True)
        sched.step()
        train_loss = running / max(n_seen, 1)

        model.eval()
        val_logits = predict_logits(
            model, Xva, rrva if use_extra else None, device=DEVICE)
        val_pred = val_logits.argmax(axis=1)
        val_acc = accuracy_score(yva, val_pred)
        val_f1 = f1_score(yva, val_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                        "val_acc": round(float(val_acc), 5),
                        "val_macro_f1": round(float(val_f1), 5)})
        print(f"epoch {epoch:02d}: train {train_loss:.4f} "
              f"val acc {val_acc:.4f} macro-F1 {val_f1:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

        if val_f1 > best_f1 + 1e-4:
            best_f1 = float(val_f1)
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
            torch.save({"model": best_state, "epoch": epoch,
                        "val_macro_f1": best_f1,
                        "class_weights": class_weights.tolist(),
                        "record_split_sha256":sha256(RESULTS / "record_split.json"),
                        "train_records":record_split["train_records"],
                        "val_records":record_split["val_records"]},
                       MODELS / ckpt_name)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    torch.save({"model": model.state_dict(), "epoch": len(history),
                "val_macro_f1": best_f1,
                "class_weights": class_weights.tolist()},
               MODELS / ckpt_name.replace("_best", "_last"))
    hist_name = ("cnn_light" if args.arch == "light" and args.model == "cnn"
                 else args.model)
    save_json(RESULTS / f"{hist_name}_history.json", history)
    print(f"best val macro-F1 {best_f1:.4f}; elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
