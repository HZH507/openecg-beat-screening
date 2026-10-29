"""Select binary class weight and threshold on validation, then freeze all models."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import shutil

import numpy as np

from config import MODELS, RESULTS
from split_record_level import sha256
from train_binary import metrics
from utils import save_json


def main():
    target = RESULTS/'val_threshold.json'
    if target.exists():
        raise RuntimeError('Already frozen; refusing to rescan')
    grid = np.round(np.arange(.01, 1., .01), 2)
    rows = []
    for weight in (3,5,9):
        with np.load(RESULTS/f'binary_w{weight}_val_predictions.npz') as d:
            y, prob = d['y'], d['prob']
        for threshold in grid:
            m = metrics(y, (prob >= threshold).astype(np.int64), prob)
            rows.append({'weight':weight, 'threshold':float(threshold), **m})
    # Deterministic ties: F1, specificity, sensitivity, lower weight,
    # threshold nearest 0.5, then lower threshold.
    best = max(rows, key=lambda r:(r['macro_f1'],r['specificity'],r['sensitivity'],
                                  -r['weight'],-abs(r['threshold']-.5),-r['threshold']))
    recall_rows = [r for r in rows if r['weight']==best['weight'] and r['sensitivity']>=.8]
    recall = max(recall_rows, key=lambda r:(r['specificity'],r['macro_f1'],r['threshold'])) if recall_rows else None
    candidate = MODELS/'candidates'/f"binary_w{best['weight']}_best.pt"
    shutil.copy2(candidate, MODELS/'binary_best.pt')
    shutil.copy2(RESULTS/f"binary_w{best['weight']}_history.json", RESULTS/'binary_history.json')
    model_files = ('binary_best.pt','cnn_light_best.pt','denoiser_best.pt','mlp_best.pt')
    hashes = {name:sha256(MODELS/name) for name in model_files}
    selected = []
    for t in (.5,best['threshold'],recall['threshold'] if recall else .5):
        if t not in selected:
            selected.append(t)
    result = {'frozen':True, 'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
              'selection_dataset':'DS1 four held-out validation records only',
              'objective':'maximum validation Macro-F1 jointly over weights 3/5/9 and threshold grid 0.01..0.99 step 0.01',
              'epoch_rule':'maximum validation Macro-F1 at threshold 0.5 per candidate; patience 6',
              'tie_rule':'Macro-F1, specificity, sensitivity, lower weight, nearer 0.5, lower threshold',
              'selected_weight':best['weight'], 'threshold':best['threshold'],
              'validation_metrics':best, 'recall_constrained_validation_point':recall,
              'predeclared_test_thresholds':selected,
              'record_split_sha256':sha256(RESULTS/'record_split.json'),
              'model_sha256':hashes, 'threshold_grid':grid.tolist(), 'validation_sweep':rows,
              'test_policy':'one final locked evaluation after freeze; no refit or threshold change after observing DS2'}
    save_json(target,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('validation_sweep','threshold_grid')}, indent=2))


if __name__=='__main__':
    main()
