"""Pool genuine held-record OOF predictions, refit on DS1, then freeze."""
from __future__ import annotations

from datetime import datetime,timezone

import numpy as np

from config import ROOT
from train_workpoint_fold import check_protocol,fit_model
from utils import load_beats
from workpoint_common import OUT,auc,baseline,read,save,select_points,sha256,write_curve


def main():
    if (OUT/'frozen_workpoints.json').exists():
        raise RuntimeError('Working points already frozen; refusing reselection')
    protocol=read(OUT/'protocol.json'); check_protocol(protocol)
    items=[]; summaries=[]
    for f in range(1,6):
        path=OUT/f'fold_{f}'; info=read(path/'complete.json')
        assert sha256(path/'outer_refit.pt')==info['checkpoint_sha256']
        assert sha256(path/'oof_predictions.npz')==info['predictions_sha256']
        with np.load(path/'oof_predictions.npz',allow_pickle=False) as d:
            items.append({k:d[k].copy() for k in ('global_index','y','prob','record')})
        summaries.append(info)
    joined={k:np.concatenate([d[k] for d in items]) for k in items[0]}
    order=np.argsort(joined['global_index']); joined={k:v[order] for k,v in joined.items()}
    X,y4,rr,rec,_=load_beats(with_rr=True); y=(y4>0).astype(np.int64)
    eligible=np.flatnonzero(np.isin(rec,protocol['eligible_records']))
    np.testing.assert_array_equal(joined['global_index'],eligible)
    np.testing.assert_array_equal(joined['y'],y[eligible])
    assert '201' not in joined['record'] and len(np.unique(joined['global_index']))==len(eligible)
    np.savez_compressed(OUT/'oof_predictions.npz',**joined)
    points,curve=select_points(joined['y'],joined['prob'])
    write_curve(OUT/'oof_threshold_curve.csv',curve)
    final_epochs=int(np.median([s['selected_epochs'] for s in summaries]))
    selection={'selected_at_utc':datetime.now(timezone.utc).isoformat(),
               'selected_using':'pooled five-fold DS1 OOF only, record 201 excluded',
               'primary_rule':protocol['primary_rule'],'secondary_rule':protocol['secondary_rule'],
               'oof_auc':auc(joined['y'],joined['prob']),'oof_n':len(eligible),
               'oof_all_normal_baseline':baseline(joined['y']),
               'points':points,'inner_selected_epochs':[s['selected_epochs'] for s in summaries],
               'final_epochs':final_epochs,'class_weight':3,
               'protocol_sha256':sha256(OUT/'protocol.json'),
               'oof_predictions_sha256':sha256(OUT/'oof_predictions.npz')}
    save(OUT/'oof_selection.json',selection)
    print('OOF SELECTION',selection,flush=True)
    fit_model(X,y,rr,eligible,epochs=final_epochs,seed=protocol['training']['final_refit_seed'],
              path=OUT,tag='final_binary',protocol=protocol)
    frozen={**selection,'frozen':True,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
            'final_model_sha256':sha256(OUT/'final_binary.pt'),
            'final_train_records':protocol['eligible_records'],
            'source_sha256':protocol['source_sha256'],
            'test_policy':protocol['test_policy'],
            'test_exploration_label':protocol['test_exploration_label']}
    save(OUT/'frozen_workpoints.json',frozen)
    print('FROZEN: final model and OOF working points; DS2 inference is now allowed.',flush=True)


if __name__=='__main__':
    main()
