"""Predeclare a DS1-only five-fold protocol, excluding subject record 201."""
from __future__ import annotations

from datetime import datetime,timezone
import itertools

import numpy as np

from config import DS1,DS2,PROCESSED,ROOT,SEED
from workpoint_common import OUT,read,save,sha256,threshold_curve,metrics_at,select_points


def inner_split(records,y,rec,seed):
    rates={r:float((y[rec==r]>0).mean()) for r in records}
    ordered=sorted(records,key=lambda r:(rates[r],r))
    strata=[list(g) for g in np.array_split(ordered,4)]
    counts={r:np.bincount((y[rec==r]>0).astype(int),minlength=2) for r in records}
    total=sum(counts.values()); target=4/len(records)
    choices=list(itertools.product(*strata))
    np.random.default_rng(seed).shuffle(choices)
    best=None
    for choice in choices:
        c=sum((counts[r] for r in choice),np.zeros(2,dtype=int))
        if np.any(c==0) or np.any(total-c==0):
            continue
        score=float(np.mean((c/total-target)**2))
        if best is None or score<best[0]-1e-15:
            best=(score,sorted(choice))
    assert best is not None
    return sorted(set(records)-set(best[1])),best[1]


def self_test():
    # Tied scores, endpoints and class-imbalance cases exercise cumulative
    # threshold counts against independently computed predictions.
    y=np.array([0,1,0,1,0,1,0])
    p=np.array([0.,.2,.2,.6,.9,1.,1.])
    curve=threshold_curve(y,p,extra=(.56,.900001))
    for i,t in enumerate(curve['threshold']):
        expected=metrics_at(y,p,t)
        for k,v in expected.items():
            assert abs(float(curve[k][i])-v)<1e-12,(k,t)
    points,_=select_points(y,p)
    assert points['sp90_max_se']['specificity']>=.9


def main():
    self_test()
    if OUT.exists():
        raise RuntimeError('Versioned experiment directory already exists; refusing overwrite')
    OUT.mkdir(parents=True)
    with np.load(PROCESSED/'beats.npz',allow_pickle=False) as d:
        y,rec=d['y'],d['record']
    eligible=sorted(set(DS1)-{'201'})
    assert len(eligible)==21 and not set(eligible)&set(DS2)
    rates={r:float((y[rec==r]>0).mean()) for r in eligible}
    ordered=sorted(eligible,key=lambda r:(rates[r],r))
    rng=np.random.default_rng(SEED)
    # Stratify by record abnormal fraction: each consecutive block of five
    # contributes one record to each fold. Last block has the 21st record.
    groups=[[] for _ in range(5)]
    for offset in range(0,len(ordered),5):
        block=ordered[offset:offset+5]
        for r,f in zip(block,rng.permutation(5)):
            groups[int(f)].append(r)
    folds=[]
    for f,outer_val in enumerate(groups,1):
        outer_val=sorted(outer_val); outer_train=sorted(set(eligible)-set(outer_val))
        inner_train,inner_val=inner_split(outer_train,y,rec,SEED+f*100)
        assert not set(outer_val)&set(inner_train+inner_val)
        assert '201' not in outer_train+outer_val
        folds.append({'fold':f,'outer_train_records':outer_train,'outer_val_records':outer_val,
                      'inner_train_records':inner_train,'inner_val_records':inner_val,
                      'outer_train_n':int(np.isin(rec,outer_train).sum()),
                      'outer_val_n':int(np.isin(rec,outer_val).sum()),
                      'outer_val_abnormal':int(np.sum((y>0)&np.isin(rec,outer_val)))})
    assert sorted(r for f in folds for r in f['outer_val_records'])==eligible
    source_names=['workpoint_common.py','prepare_workpoint_cv.py','train_workpoint_fold.py',
                  'freeze_workpoint.py','evaluate_workpoint_once.py','verify_workpoint.py',
                  'models.py','utils.py','train_classifier.py','config.py','run_workpoint_cv.ps1']
    protocol={
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,
        'purpose':'DS1-only five-fold OOF operating-point selection; single final DS2 inference',
        'eligible_records':eligible,'excluded_subject_record':'201',
        'exclusion_reason':'201 and DS2 record 202 belong to the same subject; 201 excluded from all binary training, early stopping, OOF threshold selection and final refit',
        'eligible_n':int(np.isin(rec,eligible).sum()),'excluded_201_n':int(np.sum(rec=='201')),
        'fold_rule':'sort DS1 eligible records by abnormal fraction, then seeded random assignment within consecutive blocks of five; each outer validation fold has 4 or 5 records',
        'folds':folds,
        'primary_rule':'maximize pooled OOF sensitivity subject to pooled OOF specificity >= 0.90; ties higher specificity, then higher threshold',
        'secondary_rule':'maximize pooled OOF Youden J = Se+Sp-1; ties higher specificity, then higher threshold',
        'reference_threshold':.5,
        'epoch_selection':'within each outer-training partition, four inner validation records choose best AUC checkpoint; patience 6, min improvement 0.0001, max 24 epochs. Refit fresh on entire outer-training partition for that fixed epoch count; outer validation labels never select epochs',
        'final_refit_rule':'fresh single model on all 21 eligible records for median of the five inner-selected epoch counts; fixed abnormal weight 3:1',
        'class_weight_reason':'pre-specified retention of the original 3:1 training setting; no class-weight reselection in this revision',
        'training':{'abnormal_weight':3,'max_epochs':24,'patience':6,'min_auc_improvement':1e-4,
                    'batch':128,'learning_rate':1e-3,'weight_decay':1e-4,'label_smoothing':.03,
                    'scheduler':'cosine T_max=24 for both inner training and fixed-epoch refits',
                    'torch_threads_per_process':2,'max_concurrent_folds':2,
                    'inner_seed_rule':'42 + 100*fold','outer_refit_seed_rule':'42 + 100*fold + 1',
                    'final_refit_seed':1041},
        'primary_metric':'AUC; accuracy is subordinate to the selected operating point',
        'test_policy':'freeze OOF thresholds and final model hash before DS2. Infer one clean probability vector once; derive all predeclared workpoints, all-normal baseline, and complete exploratory threshold curves from that cached vector. No DS2-based changes.',
        'test_exploration_label':'探索性分析，仅展示冻结后模型的阈值敏感性，不参与选择工作点',
        'prior_test_exposure':'Previous binary experiments already used DS2; this is a new, prespecified revision evaluated once. It must not be described as a never-before-seen pristine test set.',
        'existing_experiments_preserved':'Prior four-class, DnCNN and noise results are unchanged and remain attributed to their own model/threshold; do not transfer their noise metrics to this refitted binary model.',
        'dataset_sha256':sha256(PROCESSED/'beats.npz'),
        'source_sha256':{n:sha256(ROOT/'src'/n) for n in source_names},
    }
    save(OUT/'protocol.json',protocol)
    print({k:protocol[k] for k in ('eligible_n','excluded_201_n','primary_rule')},flush=True)
    for f in folds:
        print('fold',f['fold'],'outer validation',f['outer_val_records'],
              'n',f['outer_val_n'],'abnormal',f['outer_val_abnormal'],flush=True)
    print('Threshold-count self-test: PASS',flush=True)


if __name__=='__main__':
    main()
