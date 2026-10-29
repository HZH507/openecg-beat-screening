"""Train the 1-D DnCNN ECG denoiser on DS1 and save the best checkpoint."""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import DS1, MODELS, RESULTS, SEED
from models import DnCNN1d, count_parameters
from utils import (add_noise, load_beats, load_nstdb_noise, save_json,
                   set_seed, snr_db)
from split_record_level import load_record_split, sha256

torch.set_num_threads(4)
EPOCHS = 12
BATCH = 128
LR = 1e-3
DEVICE = "cpu"
MAX_TRAIN = 12000
PATIENCE = 4


def prepare_batch(clean: np.ndarray, bank, rng, split: str = "train"):
    noisy = add_noise(clean, bank, rng, split=split)
    scale = noisy.std(axis=1, keepdims=True) + 1e-6
    xn = (noisy - noisy.mean(axis=1, keepdims=True)) / scale
    cn = (clean - noisy.mean(axis=1, keepdims=True)) / scale
    return (torch.from_numpy(xn).float().unsqueeze(1),
            torch.from_numpy(cn).float(),
            torch.from_numpy(noisy).float(),
            torch.from_numpy(clean).float())


@torch.no_grad()
def validate(model, clean, bank, rng):
    model.eval()
    noisy = add_noise(clean, bank, rng, split="train")
    scale = noisy.std(axis=1, keepdims=True) + 1e-6
    xn = (noisy - noisy.mean(axis=1, keepdims=True)) / scale
    cn = (clean - noisy.mean(axis=1, keepdims=True)) / scale
    xb = torch.from_numpy(xn).float().unsqueeze(1)
    out = model(xb).numpy() * scale.squeeze(1) + noisy.mean(axis=1)
    loss = float(np.mean((out - clean) ** 2))
    snr_in = float(np.mean([snr_db(c, n) for c, n in zip(clean, noisy)]))
    snr_out = float(np.mean([snr_db(c, o) for c, o in zip(clean, out)]))
    return loss, snr_in, snr_out


def main() -> None:
    if (RESULTS / 'val_threshold.json').exists():
        raise RuntimeError('Protocol frozen; training is closed')
    set_seed(SEED)
    X, y, rec, _ = load_beats()
    tr, va, record_split = load_record_split(rec)
    Xtr, Xva = X[tr], X[va]
    if len(Xtr) > MAX_TRAIN:
        rng = np.random.default_rng(SEED)
        pick = rng.choice(len(Xtr), size=MAX_TRAIN, replace=False)
        Xtr = Xtr[np.sort(pick)]
    save_json(RESULTS / 'denoiser_training_config.json', {
        'epochs':EPOCHS,'batch':BATCH,'lr':LR,'patience':PATIENCE,'max_train':MAX_TRAIN,
        'seed':SEED,'train_n_after_cap':len(Xtr),'val_n':len(Xva),
        'train_records':record_split['train_records'],'val_records':record_split['val_records']})
    bank = load_nstdb_noise()
    print(f"train {len(Xtr)}, val {len(Xva)}")

    train_ds = TensorDataset(torch.from_numpy(Xtr).float())
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, drop_last=True)

    model = DnCNN1d(channels=32, depth=6, kernel=11).to(DEVICE)
    print("parameters:", count_parameters(model))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    rng = np.random.default_rng(SEED)
    val_rng = np.random.default_rng(SEED + 1)
    # Fixed validation noisy set: regenerate once and reuse across epochs.
    val_noisy = add_noise(Xva, bank, val_rng, split="train")
    val_scale = val_noisy.std(axis=1, keepdims=True) + 1e-6
    val_mu = val_noisy.mean(axis=1, keepdims=True)
    val_x = torch.from_numpy((val_noisy - val_mu) / val_scale).float().unsqueeze(1)
    val_c = (Xva - val_mu) / val_scale
    del val_noisy

    history = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for step, (clean,) in enumerate(train_loader, 1):
            clean_np = clean.squeeze(1).numpy()
            xb, cb, _, _ = prepare_batch(clean_np, bank, rng, split="train")
            xb, cb = xb.to(DEVICE), cb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb).squeeze(1)
            loss = loss_fn(out, cb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.detach()) * len(xb)
            n_seen += len(xb)
            if step % 80 == 0:
                print(f"  epoch {epoch:02d} step {step:03d} "
                      f"loss {running / n_seen:.5f}", flush=True)
        sched.step()
        train_loss = running / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            out = np.concatenate([model(val_x[i:i+128]).squeeze(1).numpy()
                                  for i in range(0,len(val_x),128)]) * val_scale + val_mu
        val_loss = float(np.mean((out - Xva) ** 2))
        snr_in = float(np.mean([snr_db(c, n) for c, n in
                                zip(Xva, val_x.squeeze(1).numpy() * val_scale + val_mu)]))
        snr_out = float(np.mean([snr_db(c, o) for c, o in zip(Xva, out)]))
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                        "val_mse": round(val_loss, 6),
                        "val_snr_in": round(snr_in, 3),
                        "val_snr_out": round(snr_out, 3),
                        "val_delta_snr": round(snr_out - snr_in, 3)})
        print(f"epoch {epoch:02d}: train {train_loss:.5f} val {val_loss:.5f} "
              f"SNR {snr_in:.2f}->{snr_out:.2f} dB "
              f"({time.time() - t0:.0f}s)", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
            torch.save({"model": best_state, "epoch": epoch,
                        "val_mse": best_loss,
                        "record_split_sha256":sha256(RESULTS / "record_split.json"),
                        "train_records":record_split["train_records"],
                        "val_records":record_split["val_records"]}, MODELS / "denoiser_best.pt")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stop at epoch {epoch}")
                break

    torch.save({"model": model.state_dict(), "epoch": len(history),
                "val_mse": best_loss}, MODELS / "denoiser_last.pt")
    save_json(RESULTS / "denoiser_history.json", history)
    print(f"best val MSE {best_loss:.6f}; elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
