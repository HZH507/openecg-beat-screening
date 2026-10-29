"""Freeze an 18/4 DS1 record split without consulting DS2 outcomes."""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np

from config import CLASS_NAMES, DS1, DS2, PROCESSED, RESULTS, SEED


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_record_split(records):
    path = RESULTS / 'record_split.json'
    obj = json.loads(path.read_text(encoding='utf-8'))
    assert obj['dataset_sha256'] == sha256(PROCESSED / 'beats.npz'), 'Dataset changed'
    tr = np.asarray(obj['train_indices'], dtype=np.int64)
    va = np.asarray(obj['val_indices'], dtype=np.int64)
    assert len(set(obj['train_records'])) == 18 and len(set(obj['val_records'])) == 4
    assert not set(obj['train_records']) & set(obj['val_records'])
    assert set(obj['train_records'] + obj['val_records']) == set(DS1)
    assert not set(obj['train_records'] + obj['val_records']) & set(DS2)
    assert not np.intersect1d(tr, va).size
    assert set(records[tr]) == set(obj['train_records'])
    assert set(records[va]) == set(obj['val_records'])
    assert np.array_equal(np.sort(np.r_[tr, va]), np.flatnonzero(np.isin(records, DS1)))
    return tr, va, obj


def main():
    out = RESULTS / 'record_split.json'
    if out.exists():
        raise RuntimeError('Split already frozen; refusing to overwrite')
    with np.load(PROCESSED / 'beats.npz', allow_pickle=False) as d:
        y, rec = d['y'], d['record']
    counts = {r: np.bincount(y[rec == r], minlength=4) for r in DS1}
    totals = np.sum(list(counts.values()), axis=0)
    rates = {r: float(counts[r][1:].sum() / counts[r].sum()) for r in DS1}
    ordered = sorted(DS1, key=lambda r: (rates[r], r))
    strata = [list(g) for g in np.array_split(ordered, 4)]
    # One record per abnormal-fraction quartile. Among those combinations,
    # preserve each class's DS1 validation fraction near 4/22; no model score
    # or DS2 label enters this decision. Seed 42 resolves numerical ties.
    rng = np.random.default_rng(SEED)
    candidates = list(itertools.product(*strata))
    rng.shuffle(candidates)
    best = None
    for candidate in candidates:
        c = sum((counts[r] for r in candidate), np.zeros(4, dtype=np.int64))
        if np.any(c == 0) or np.any(totals - c == 0):
            continue
        score = float(np.mean((c / totals - 4 / 22) ** 2))
        if best is None or score < best[0] - 1e-15:
            best = (score, sorted(candidate), c)
    assert best is not None
    val_records = best[1]
    train_records = sorted(set(DS1) - set(val_records))
    tr = np.flatnonzero(np.isin(rec, train_records))
    va = np.flatnonzero(np.isin(rec, val_records))
    obj = {
        'protocol': 'DS1 record-level 18 train / 4 validation; DS2 held out',
        'seed': SEED, 'dataset_sha256': sha256(PROCESSED / 'beats.npz'),
        'selection_rule': 'one record per DS1 abnormal-fraction quartile; minimize mean squared difference of per-class validation fractions from 4/22; seeded tie order',
        'selection_score': best[0], 'strata': strata,
        'train_records': train_records, 'val_records': val_records,
        'train_n': int(len(tr)), 'val_n': int(len(va)),
        'train_counts': {c: int(np.sum(y[tr] == i)) for i, c in enumerate(CLASS_NAMES)},
        'val_counts': {c: int(np.sum(y[va] == i)) for i, c in enumerate(CLASS_NAMES)},
        'ds1_record_abnormal_fraction': rates,
        'train_indices': tr.tolist(), 'val_indices': va.tolist(),
        'excluded_records': ['102', '104', '107', '217'],
        'known_subject_overlap': '201 in DS1 and 202 in DS2 are the same subject; record independence is not complete subject independence',
    }
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    load_record_split(rec)
    print(json.dumps({k: obj[k] for k in ('train_records','val_records','train_n','val_n','train_counts','val_counts')}, indent=2))


if __name__ == '__main__':
    main()
