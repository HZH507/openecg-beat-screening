"""Frozen-model auxiliary/27-condition noise suite; never re-infer clean binary DS2."""
from __future__ import annotations

from datetime import datetime,timezone
import importlib.metadata,platform,time

import numpy as np
import torch
import wfdb
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,roc_auc_score

from config import DS2,CLASS_NAMES,MITDB,FS,WIN,WIN_BEFORE,AAMI_MAP,ROOT
from models import ECGCNNLight,MLPBaseline,DnCNN1d,count_parameters
from utils import load_beats,predict_logits,add_noise,classical_filter,snr_db
from final_common import OUT,MODELS,CV,read,save,assert_binary_frozen,load_noise_bank,binary_metrics
from workpoint_common import sha256
from openecg_demo import bandpass,detect_rpeaks,match_labels,rr_features,classify_beats

torch.set_num_threads(4)


@torch.inference_mode()
def denoise(model,noisy,batch=128):
    mu=noisy.mean(1,keepdims=True);sd=noisy.std(1,keepdims=True)+1e-6
    x=(noisy-mu)/sd
    return np.concatenate([model(torch.from_numpy(x[i:i+batch]).unsqueeze(1)).squeeze(1).numpy()
                           for i in range(0,len(x),batch)])*sd+mu


def four_metrics(y,pred,prob):
    return {'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=[0,1,2,3],average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(y,pred,average='weighted',zero_division=0)),
            'macro_auc_ovr':float(roc_auc_score(y,prob,multi_class='ovr',average='macro')),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2,3]).tolist(),
            'report':classification_report(y,pred,labels=[0,1,2,3],target_names=CLASS_NAMES,output_dict=True,zero_division=0),
            'n':len(y)}


def demo(cnn,dn,hashes):
    record=wfdb.rdrecord(str(MITDB/'208'));ann=wfdb.rdann(str(MITDB/'208'),'atr')
    sig=np.asarray(record.p_signal[:,0],dtype=np.float32)
    centres=ann.sample[np.isin(ann.symbol,['F','V'])];centre=int(centres[len(centres)//2])
    start=max(0,centre-4*FS);segment=sig[start:start+10*FS]
    filtered=bandpass(segment);peaks=detect_rpeaks(segment)
    valid=[i for i,p in enumerate(peaks) if p-WIN_BEFORE>=0 and p-WIN_BEFORE+WIN<=len(filtered)]
    beats=np.asarray([filtered[peaks[i]-WIN_BEFORE:peaks[i]-WIN_BEFORE+WIN] for i in valid])
    pred,prob,den=classify_beats(cnn,dn,beats,rr_features(peaks)[valid])
    symbols=match_labels(peaks,ann.sample-start,ann.symbol);symbols=[symbols[i] for i in valid]
    truth=[AAMI_MAP.get(s,'?') for s in symbols];used_peaks=peaks[valid]
    r100=wfdb.rdrecord(str(MITDB/'100'));a100=wfdb.rdann(str(MITDB/'100'),'atr')
    p100=detect_rpeaks(np.asarray(r100.p_signal[:,0],dtype=np.float32))
    gt=a100.sample[np.isin(a100.symbol,list('NLRBAaJSVEFj'))]
    matched=sum(bool(np.min(np.abs(gt-p))<=int(.1*FS)) for p in p100)
    info={'model_task':'four-class exploration','classifier_checkpoint':'results/report_final/models/cnn_light_best.pt',
          'classifier_parameters':count_parameters(cnn),'classifier_sha256':hashes['cnn_light_best.pt'],
          'denoiser_sha256':hashes['denoiser_best.pt'],'uses_rr_features':False,'record':'208','fs':FS,
          'start_sample':int(start),'peaks':used_peaks.tolist(),'pred':pred.tolist(),'prob':prob.tolist(),
          'true_symbols':symbols,'true_cls':truth,'mismatches':sum(t!=CLASS_NAMES[int(p)] for t,p in zip(truth,pred)),
          'pred_counts':{c:int(np.sum(pred==i)) for i,c in enumerate(CLASS_NAMES)},
          'truth_prediction_pairs':[{'true':t,'pred':CLASS_NAMES[int(p)]} for t,p in zip(truth,pred)],
          'detection':{'se':matched/max(len(gt),1),'ppv':matched/max(len(p100),1),'n_gt':len(gt),'n_det':len(p100)},
          'interpretation':'record 208 is DS1 training/validation material; this is a functional demonstration, not independent classification validation'}
    save(OUT/'demo_info.json',info)
    np.savez_compressed(OUT/'demo_record208.npz',signal=segment,peaks=used_peaks,pred=pred,prob=prob,den=den,true_cls=np.asarray(truth))
    print('NEW DEMO',info['pred_counts'],'mismatches',info['mismatches'],flush=True)


def costs(models,paths):
    x=torch.from_numpy(np.random.default_rng(42).normal(size=(128,1,256)).astype(np.float32))
    rows=[]
    for key,name in [('binary','轻量 CNN 二分类'),('cnn','轻量 CNN 四分类'),('mlp','MLP 四分类'),('denoiser','DnCNN')]:
        model=models[key];samples=[]
        with torch.inference_mode():
            for _ in range(5):model(x)
            for _ in range(20):
                start=time.perf_counter();model(x);samples.append((time.perf_counter()-start)*1000/len(x))
        n=count_parameters(model)
        rows.append({'key':key,'name':name,'params':n,'fp32_kib':n*4/1024,'int8_kib':n/1024,
                     'latency_ms':float(np.median(samples)),'latency_ms_samples':samples,
                     'checkpoint':str(paths[key].relative_to(ROOT)).replace('\\','/'),'checkpoint_sha256':sha256(paths[key])})
    save(OUT/'model_cost.json',{'models':rows,'threads':4,'batch':128,'repetitions':20,'warmups':5,
          'statistic':'median batch latency divided by 128','input':'seeded synthetic FP32 tensor',
          'scope':'network forward only; excludes filtering/R-peak detection/normalization/I-O',
          'int8_note':'weight byte-count estimate only; no INT8 execution or MCU measurement'})


def main():
    frozen=assert_binary_frozen();marker=OUT/'consistency_evaluation.json'
    if marker.exists():raise RuntimeError('Consistency suite already started; refusing duplicate evaluation')
    for name in ('cnn_light_best.pt','mlp_best.pt','denoiser_best.pt'):
        ck=torch.load(MODELS/name,map_location='cpu',weights_only=False)
        assert '201' not in ck['train_records']+ck['val_records']
        assert ck['aux_split_sha256']==sha256(OUT/'aux_split.json')
    hashes={name:sha256(MODELS/name) for name in ('cnn_light_best.pt','mlp_best.pt','denoiser_best.pt')}
    save(OUT/'aux_models_frozen.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
          'model_sha256':hashes,'aux_split_sha256':sha256(OUT/'aux_split.json'),
          'binary_sha256':frozen['final_model_sha256'],'threshold':frozen['points']['sp90_max_se']['threshold']})
    audit={'status':'started','started_at_utc':datetime.now(timezone.utc).isoformat(),
           'binary_clean_ds2_inference_passes_added':0,'auxiliary_clean_ds2_passes_each':1,
           'noise_conditions':9,'noise_methods_per_condition':3,'binary_sha256':frozen['final_model_sha256'],
           'threshold':frozen['points']['sp90_max_se']['threshold']}
    save(marker,audit)
    paths={'binary':CV/'final_binary.pt','cnn':MODELS/'cnn_light_best.pt','mlp':MODELS/'mlp_best.pt','denoiser':MODELS/'denoiser_best.pt'}
    models={'binary':ECGCNNLight(num_classes=2),'cnn':ECGCNNLight(),'mlp':MLPBaseline(),'denoiser':DnCNN1d(channels=32,depth=6,kernel=11)}
    for key,model in models.items():
        model.load_state_dict(torch.load(paths[key],map_location='cpu',weights_only=False)['model']);model.eval()
    X,y4,rr,rec,_=load_beats(with_rr=True)
    with np.load(CV/'ds2_predictions.npz') as cache:
        test=cache['global_index'];yb=cache['y'];binary_prob=cache['prob']
    Xte=X[test];yte=y4[test]
    metrics={};predictions={'y':yte,'record':rec[test],'global_index':test}
    for key in ('cnn','mlp'):
        logits=predict_logits(models[key],Xte);prob=torch.softmax(torch.from_numpy(logits),1).numpy()
        pred=prob.argmax(1);metrics[key]=four_metrics(yte,pred,prob)
        predictions[key]=pred;predictions[key+'_prob']=prob
        print('NEW FOUR CLASS',key,metrics[key]['accuracy'],metrics[key]['macro_f1'],flush=True)
    save(OUT/'classification_metrics.json',metrics)
    np.savez_compressed(OUT/'classification_predictions.npz',**predictions)
    # All noise inputs are fixed before any per-condition classification.
    rng=np.random.default_rng(42);pos=np.flatnonzero(yb==1)
    neg=rng.choice(np.flatnonzero(yb==0),size=4000,replace=False);sel=np.sort(np.r_[pos,neg])
    xs=Xte[sel];ys=yb[sel];indices=test[sel]
    bank=load_noise_bank();rows=[];noise_preds={'y':ys,'global_index':indices}
    amp=np.ptp(xs,axis=1);example_i=int(np.argmin(abs(amp-np.percentile(amp,70))))
    example={'clean':xs[example_i]};threshold=audit['threshold']
    for kind in ('bw','em','ma'):
        for snr in (5,10,15):
            rng=np.random.default_rng(42+snr+{'bw':1,'em':2,'ma':3}[kind]*100)
            raw=add_noise(xs,bank,rng,split='test',snr_db=snr,kind=kind)
            classical=np.stack([classical_filter(x) for x in raw])
            den=denoise(models['denoiser'],raw)
            row={'condition':f'{kind}_{snr}dB','kind':kind,'snr_db':snr,'threshold':threshold,'n':len(ys)}
            for method,array in [('raw',raw),('classical',classical),('denoised',den)]:
                row['snr_'+method]=float(np.mean([snr_db(c,a) for c,a in zip(xs,array)]))
                row['rmse_'+method]=float(np.sqrt(np.mean((xs-array)**2)))
                logits=predict_logits(models['binary'],array)
                prob=torch.softmax(torch.from_numpy(logits),1).numpy()[:,1]
                noise_preds[f'{kind}_{snr}_{method}_prob']=prob
                row[method]=binary_metrics(ys,prob,threshold)
                if snr==5 and kind in ('bw','em'):example[f'{kind}_{method}']=array[example_i]
            rows.append(row);save(OUT/'binary_noise_results.json',rows)
            print('NEW NOISE',row['condition'],{m:row[m]['macro_f1'] for m in ('raw','classical','denoised')},flush=True)
    np.savez_compressed(OUT/'noise_predictions.npz',**noise_preds)
    np.savez_compressed(OUT/'denoising_example.npz',**example)
    save(OUT/'binary_noise_metadata.json',{'n':len(ys),'n_abnormal':int(ys.sum()),'n_normal':int((ys==0).sum()),
          'threshold':threshold,'classifier_sha256':frozen['final_model_sha256'],
          'denoiser_sha256':hashes['denoiser_best.pt'],'conditions':9,'methods':['raw','classical','denoised'],
          'subset_global_indices':indices.tolist(),'noise_source_split':read(OUT/'experiment_protocol.json')['noise_source_split'],
          'noise_seed_rule':read(OUT/'experiment_protocol.json')['noise_seed_rule'],
          'classification_preprocessing':'shared per-beat z-score for all three methods; previous noise evaluation already standardized beats',
          'old_noise_metrics_reused':False,'predictions_sha256':sha256(OUT/'noise_predictions.npz')})
    demo(models['cnn'],models['denoiser'],hashes)
    costs(models,paths)
    save(OUT/'runtime_environment.json',{'python':platform.python_version(),'platform':platform.platform(),
          'cpu_threads':4,'training_threads_per_process':2,
          'packages':{p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','wfdb','matplotlib','scikit-learn','python-docx')}})
    # Cache-derived fields are exported for plotting without another binary inference.
    binary=read(CV/'ds2_summary.json');primary=binary['working_points']['sp90_max_se']
    save(OUT/'binary_metrics.json',{**primary,'auc':binary['auc']})
    save(OUT/'binary_threshold_metrics.json',list(binary['working_points'].values()))
    np.savez_compressed(OUT/'binary_predictions.npz',y=yb,prob=binary_prob,pred=(binary_prob.astype(float)>=threshold).astype(int))
    np.savez_compressed(OUT/'test_predictions.npz',**predictions)
    audit.update(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 aux_model_sha256=hashes,noise_predictions_sha256=sha256(OUT/'noise_predictions.npz'))
    save(marker,audit);assert_binary_frozen()
    print('CONSISTENCY SUITE COMPLETE; clean binary cache/model/threshold unchanged.',flush=True)


if __name__=='__main__':main()
