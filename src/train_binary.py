"""Train a binary class-weight candidate using DS1 training/validation only."""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from config import MODELS, RESULTS, SEED
from models import ECGCNNLight, count_parameters
from split_record_level import load_record_split, sha256
from train_classifier import BeatDataset
from utils import load_beats, predict_logits, save_json, set_seed

torch.set_num_threads(4)


def metrics(y, pred, proba):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=[0,1], average='macro', zero_division=0)),
            'sensitivity': float(tp / max(tp + fn, 1)),
            'specificity': float(tn / max(tn + fp, 1)),
            'auc': float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else None,
            'confusion_matrix': cm.tolist(), 'n': int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abnormal-weight', type=int, choices=[3,5,9], required=True)
    ap.add_argument('--epochs', type=int, default=24)
    args = ap.parse_args()
    if (RESULTS / 'val_threshold.json').exists():
        raise RuntimeError('Protocol frozen; training is closed')
    set_seed(SEED)
    X, y4, rr, rec, _ = load_beats(with_rr=True)
    tr, va, split = load_record_split(rec)
    y = (y4 > 0).astype(np.int64)
    Xtr, ytr, rrtr = X[tr], y[tr], rr[tr]
    Xva, yva = X[va], y[va]
    candidate_dir = MODELS / 'candidates'
    candidate_dir.mkdir(exist_ok=True)
    prefix = f'binary_w{args.abnormal_weight}'
    checkpoint_path = candidate_dir / f'{prefix}_best.pt'
    if checkpoint_path.exists():
        raise RuntimeError('Candidate already exists; refusing to overwrite')
    loader = DataLoader(BeatDataset(Xtr, rrtr, ytr, augment=True),
                        batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    model = ECGCNNLight(num_classes=2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1., float(args.abnormal_weight)]), label_smoothing=0.03)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(prefix, 'train', len(tr), 'val', len(va), 'parameters', count_parameters(model), flush=True)
    history, best_f1, best_state, best_epoch, bad = [], -1., None, None, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0., 0
        for xb, _, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            running += float(loss.detach()) * len(xb)
            n += len(xb)
        sched.step()
        prob = torch.softmax(torch.from_numpy(predict_logits(model, Xva)), 1).numpy()[:,1]
        m = metrics(yva, (prob >= .5).astype(np.int64), prob)
        history.append({'epoch':epoch, 'train_loss':running/n, 'val_acc':m['accuracy'],
                        'val_macro_f1':m['macro_f1'], 'val_auc':m['auc']})
        save_json(RESULTS / f'{prefix}_history.json', history)
        print(f"{prefix} epoch {epoch:02d}: loss {running/n:.4f} val F1 {m['macro_f1']:.4f} AUC {m['auc']:.4f} ({time.time()-start:.0f}s)", flush=True)
        if m['macro_f1'] > best_f1 + 1e-4:
            best_f1, best_epoch, bad = m['macro_f1'], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({'model':best_state, 'epoch':epoch, 'val_macro_f1':best_f1,
                        'abnormal_weight':args.abnormal_weight, 'seed':SEED,
                        'record_split_sha256':sha256(RESULTS/'record_split.json'),
                        'train_records':split['train_records'], 'val_records':split['val_records'],
                        'checkpoint_rule':'maximum validation Macro-F1 at threshold 0.5; patience 6',
                        'parameters':count_parameters(model)}, checkpoint_path)
        else:
            bad += 1
            if bad >= 6:
                print('early stop', flush=True)
                break
    model.load_state_dict(best_state)
    prob = torch.softmax(torch.from_numpy(predict_logits(model, Xva)),1).numpy()[:,1]
    np.savez_compressed(RESULTS/f'{prefix}_val_predictions.npz', y=yva, prob=prob, global_index=va)
    print('best epoch', best_epoch, 'validation F1 at 0.5', best_f1, flush=True)


if __name__ == '__main__':
    main()
