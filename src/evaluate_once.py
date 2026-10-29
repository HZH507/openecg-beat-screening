"""One final DS2 evaluation of the frozen models and predeclared thresholds."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import platform
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

from config import CLASS_NAMES, DS2, MODELS, RESULTS, ROOT, SEED
from models import DnCNN1d, ECGCNNLight, MLPBaseline, count_parameters
from split_record_level import sha256
from train_binary import metrics
from utils import load_beats, predict_logits, save_json, set_seed

torch.set_num_threads(4)


def four_metrics(y, pred, prob):
    return {'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=[0,1,2,3],average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(y,pred,average='weighted',zero_division=0)),
            'macro_auc_ovr':float(roc_auc_score(y,prob,multi_class='ovr',average='macro',labels=[0,1,2,3])),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2,3]).tolist(),
            'report':classification_report(y,pred,labels=[0,1,2,3],target_names=CLASS_NAMES,output_dict=True,zero_division=0),
            'n':int(len(y))}


def measure_latency(model):
    # Timing uses synthetic inputs, so it does not reread or rescore DS2.
    x = torch.from_numpy(np.random.default_rng(SEED).normal(size=(128,1,256)).astype(np.float32))
    with torch.inference_mode():
        for _ in range(3):
            model(x)
        start = time.perf_counter()
        for _ in range(10):
            model(x)
        return (time.perf_counter()-start)/10/len(x)*1000


def main():
    set_seed(SEED)
    frozen_path = RESULTS/'val_threshold.json'
    protocol = json.loads(frozen_path.read_text(encoding='utf-8'))
    assert protocol['frozen']
    assert sha256(RESULTS/'record_split.json') == protocol['record_split_sha256']
    for name,digest in protocol['model_sha256'].items():
        assert sha256(MODELS/name)==digest, 'Frozen model changed: '+name
    marker = RESULTS/'final_test_evaluation.json'
    if marker.exists():
        raise RuntimeError('DS2 evaluation has already started; refusing a second pass')
    audit = {'status':'started','started_at_utc':datetime.now(timezone.utc).isoformat(),
             'protocol_sha256':sha256(frozen_path),'model_sha256':protocol['model_sha256'],
             'test_policy':'one frozen suite; clean full DS2 plus predefined 5/10/15dB noise subset; no test-driven tuning'}
    save_json(marker,audit)
    X,y4,rr,rec,_ = load_beats(with_rr=True)
    mask = np.isin(rec,DS2)
    Xte,yte4,recte = X[mask],y4[mask],rec[mask]
    yte=(yte4>0).astype(np.int64)
    assert len(yte)==49705, 'Unexpected dataset version'
    models={}
    for key,model,file in [('binary',ECGCNNLight(num_classes=2),'binary_best.pt'),
                           ('cnn',ECGCNNLight(),'cnn_light_best.pt'),
                           ('mlp',MLPBaseline(),'mlp_best.pt'),
                           ('denoiser',DnCNN1d(channels=32,depth=6,kernel=11),'denoiser_best.pt')]:
        ck=torch.load(MODELS/file,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']); model.eval(); models[key]=model
    logits=predict_logits(models['binary'],Xte)
    prob=torch.softmax(torch.from_numpy(logits),1).numpy()[:,1]
    threshold=protocol['threshold']
    pred=(prob>=threshold).astype(np.int64)
    m={**metrics(yte,pred,prob),'threshold':threshold,'abnormal_weight':protocol['selected_weight'],
       'protocol_sha256':audit['protocol_sha256']}
    save_json(RESULTS/'binary_metrics.json',m)
    np.savez_compressed(RESULTS/'binary_predictions.npz',y=yte,pred=pred,prob=prob,record=recte,global_index=np.flatnonzero(mask))
    threshold_rows=[{'threshold':t,**metrics(yte,(prob>=t).astype(np.int64),prob)}
                    for t in protocol['predeclared_test_thresholds']]
    save_json(RESULTS/'binary_threshold_metrics.json',threshold_rows)
    record_rows=[]
    for r in DS2:
        pick=recte==r
        rm=metrics(yte[pick],pred[pick],prob[pick])
        tn,fp,fn,tp=np.asarray(rm['confusion_matrix']).ravel()
        record_rows.append({'record':r,'n':int(pick.sum()),'abnormal':int(yte[pick].sum()),
                            'false_negative':int(fn),'true_positive':int(tp),'false_positive':int(fp),
                            'sensitivity':rm['sensitivity'],'specificity':rm['specificity'],
                            'class_counts':{c:int(np.sum(yte4[pick]==i)) for i,c in enumerate(CLASS_NAMES)}})
    save_json(RESULTS/'binary_per_record.json',record_rows)
    four_results,predictions={},{}
    for key in ('cnn','mlp'):
        lg=predict_logits(models[key],Xte)
        pp=torch.softmax(torch.from_numpy(lg),1).numpy()
        yp=pp.argmax(axis=1)
        four_results[key]=four_metrics(yte4,yp,pp)
        predictions[key]=yp
        print(key,'Acc',four_results[key]['accuracy'],'Macro-F1',four_results[key]['macro_f1'],flush=True)
    save_json(RESULTS/'classification_metrics.json',four_results)
    np.savez_compressed(RESULTS/'test_predictions.npz',y=yte4,record=recte,**predictions)
    cost={}
    for key in ('binary','denoiser'):
        label='cnn' if key=='binary' else key
        n=count_parameters(models[key])
        cost['params_'+label]=n
        cost[label+'_fp32_kb']=n*4/1024
        cost[label+'_int8_kb']=n/1024
        cost[label+'_ms_per_beat']=measure_latency(models[key])
    cost.update({'params_four_class_cnn':count_parameters(models['cnn']),
                 'params_mlp':count_parameters(models['mlp']), 'threads':4,
                 'timing_batch':128,'timing_repetitions':10,'timing_input':'seeded synthetic input',
                 'int8_note':'weight-size arithmetic only; no quantized model execution'})
    save_json(RESULTS/'model_cost.json',cost)
    import evaluate_binary_noise
    evaluate_binary_noise.main()
    env={'python':platform.python_version(),'platform':platform.platform(),
         'cpu_threads':torch.get_num_threads(),'cuda_available':torch.cuda.is_available(),
         'packages':{p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','wfdb','matplotlib','scikit-learn','python-docx')}}
    save_json(RESULTS/'runtime_environment.json',env)
    old_root=ROOT/'backup_20261029/results'
    old=json.loads((old_root/'binary_metrics.json').read_text(encoding='utf-8'))
    old_noise=json.loads((old_root/'binary_noise_results.json').read_text(encoding='utf-8'))
    new_noise=json.loads((RESULTS/'binary_noise_results.json').read_text(encoding='utf-8'))
    keys=('accuracy','macro_f1','sensitivity','specificity','auc')
    comparison={'old_primary_threshold':.5,'new_primary_threshold':threshold,
                'new_weight':protocol['selected_weight'],
                'clean':{k:{'old':old[k],'new':m[k],'delta':m[k]-old[k]} for k in keys},
                'new_at_predeclared_thresholds':threshold_rows,
                'noise':[],
                'four_class_old':json.loads((old_root/'classification_metrics.json').read_text(encoding='utf-8')),
                'four_class_new':four_results,
                'interpretation_note':'Comparison changes the record validation protocol, selected class weight, selected epoch, selected threshold, and consistent train/inference z-score scaling together; do not attribute every difference solely to removal of record leakage. Noise uses the original subset/seed rule and the newly frozen threshold.'}
    for row in new_noise:
        if row['kind']=='clean':
            continue
        oldrow=next(r for r in old_noise if r['condition']==row['condition'])
        comparison['noise'].append({'condition':row['condition'],
            **{method:{'old':oldrow[method]['macro_f1'],'new':row[method]['macro_f1'],
                       'delta':row[method]['macro_f1']-oldrow[method]['macro_f1']}
               for method in ('raw','classical','denoised')}})
    save_json(RESULTS/'new_old_comparison.json',comparison)
    audit.update(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 n=len(yte),noise_conditions=9,threshold=threshold,report_rewrite_authorized=False)
    save_json(marker,audit)
    print('FINAL BINARY METRICS',json.dumps(m,indent=2),flush=True)


if __name__=='__main__':
    main()
