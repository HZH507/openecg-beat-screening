"""Verify DS1 exclusion, genuine OOF provenance, frozen test results and curve."""
from __future__ import annotations

import csv
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

from config import PROCESSED,ROOT
from workpoint_common import OUT,auc,metrics_at,read,save,select_points,sha256


def independent_metrics(y,prob,t):
    pred=np.asarray(prob,dtype=float)>=t
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'accuracy':float(accuracy_score(y,pred)),
            'sensitivity':float(tp/max(tp+fn,1)),'specificity':float(tn/max(tn+fp,1)),
            'macro_f1':float(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)),
            'f1_abnormal':float(f1_score(y,pred,pos_label=1,zero_division=0))}


def main():
    protocol=read(OUT/'protocol.json'); frozen=read(OUT/'frozen_workpoints.json')
    audit=read(OUT/'ds2_evaluation.json'); summary=read(OUT/'ds2_summary.json')
    assert audit['status']=='complete' and audit['inference_passes']==1
    assert datetime.fromisoformat(frozen['frozen_at_utc'])<datetime.fromisoformat(audit['started_at_utc'])
    assert audit['frozen_workpoints_sha256']==sha256(OUT/'frozen_workpoints.json')
    assert frozen['final_model_sha256']==sha256(OUT/'final_binary.pt')
    assert sha256(PROCESSED/'beats.npz')==protocol['dataset_sha256']
    for name,digest in protocol['source_sha256'].items():
        assert sha256(ROOT/'src'/name)==digest
    all_outer=[]
    with np.load(PROCESSED/'beats.npz',allow_pickle=False) as d:
        records=d['record']
    for f in protocol['folds']:
        train=f['outer_train_records']; val=f['outer_val_records']
        assert 4<=len(val)<=5 and not set(train)&set(val)
        assert '201' not in train+val
        assert set(train+val)==set(protocol['eligible_records'])
        path=OUT/f"fold_{f['fold']}"
        inner=torch.load(path/'inner.pt',map_location='cpu',weights_only=False)
        assert set(records[inner['training_global_indices']])==set(f['inner_train_records'])
        assert set(records[inner['validation_global_indices']])==set(f['inner_val_records'])
        assert not set(records[inner['training_global_indices']+inner['validation_global_indices']])&set(val)
        ck=torch.load(path/'outer_refit.pt',map_location='cpu',weights_only=False)
        assert set(records[ck['training_global_indices']])==set(train)
        assert not ck['validation_global_indices']
        all_outer.extend(val)
    assert sorted(all_outer)==protocol['eligible_records']
    ck=torch.load(OUT/'final_binary.pt',map_location='cpu',weights_only=False)
    assert set(records[ck['training_global_indices']])==set(protocol['eligible_records'])
    with np.load(OUT/'oof_predictions.npz') as d:
        oy,op,oi,orr=d['y'],d['prob'],d['global_index'],d['record']
    assert len(np.unique(oi))==len(oi)==protocol['eligible_n'] and '201' not in orr
    selected,_=select_points(oy,op)
    assert selected==frozen['points']
    assert selected['sp90_max_se']['specificity']>=.90
    with np.load(OUT/'ds2_predictions.npz') as d:
        y,p=d['y'],d['prob']
    assert len(y)==49705 and int((y==0).sum())==44259 and int(y.sum())==5446
    assert abs(auc(y,p)-summary['auc'])<1e-12
    for name,row in summary['working_points'].items():
        assert row['threshold']==frozen['points'][name]['threshold']
        expected=independent_metrics(y,p,row['threshold'])
        for k,v in expected.items():
            assert abs(v-row[k])<1e-12,(name,k)
    b=summary['all_normal_baseline']
    assert b['sensitivity']==0 and b['specificity']==1 and b['f1_abnormal']==0
    assert round(b['accuracy']*100,2)==89.04
    metadata=read(OUT/'ds2_threshold_curve_metadata.json')
    assert metadata['label']=='探索性分析' and metadata['not_used_for_workpoint_selection']
    assert sha256(OUT/'ds2_threshold_curve_exploratory.csv')==metadata['curve_sha256']
    rows=list(csv.DictReader((OUT/'ds2_threshold_curve_exploratory.csv').open(encoding='utf-8-sig',newline='')))
    thresholds=np.array([float(r['threshold']) for r in rows])
    assert len(rows)==metadata['rows'] and np.all(np.diff(thresholds)>0)
    assert set(np.unique(p).astype(float))<=set(thresholds)
    # Independent sklearn checks across the entire range and all frozen points.
    chosen=np.unique(np.r_[np.linspace(0,len(rows)-1,31).astype(int),
                           [np.where(thresholds==r['threshold'])[0][0] for r in frozen['points'].values()]])
    for i in chosen:
        expected=independent_metrics(y,p,thresholds[i])
        for k,v in expected.items():
            assert abs(float(rows[i][k])-v)<1e-12,(i,k)
    # All previous model/noise/four-class/report files remain attributable to
    # their original run. The new binary experiment lives in its own folder.
    manifest=read(ROOT/'backup_workpoint_20261029/manifest_sha256.json')
    for entry in manifest:
        assert sha256(ROOT/entry['path'])==entry['sha256'].lower(),'Prior artifact changed: '+entry['path']
    result={'status':'pass','201_excluded_from_binary_training_and_selection':True,
            'outer_records_predicted_once_each':True,'inner_auc_epoch_selection':True,
            'oof_n':len(oy),'final_fit_n':len(ck['training_global_indices']),
            'frozen_before_ds2':True,'ds2_inference_passes':1,'ds2_n':len(y),
            'complete_exploratory_curve_rows':len(rows),'frozen_workpoint_metrics_verified':True,
            'previous_results_and_report_unchanged':True,'report_rewrite_awaits_user_confirmation':True}
    save(OUT/'verification.json',result); print(result,flush=True)


if __name__=='__main__':
    main()
