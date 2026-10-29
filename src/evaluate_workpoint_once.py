"""Infer DS2 once after freeze, then export a fully exploratory cached curve."""
from __future__ import annotations

from datetime import datetime,timezone
import importlib.metadata
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve

from config import DS2,ROOT
from models import ECGCNNLight
from utils import load_beats,predict_logits
from workpoint_common import OUT,auc,baseline,metrics_at,read,save,sha256,threshold_curve,write_curve

torch.set_num_threads(2)


def plots(y,prob,summary,curve):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':10,'savefig.dpi':200})
    fig,axes=plt.subplots(3,1,figsize=(10.2,9),sharex=True)
    t=curve['threshold']; mask=(t>=0)&(t<=1)
    for key,label,color in [('sensitivity','Se','#b84235'),('specificity','Sp','#2877a5')]:
        axes[0].plot(t[mask],curve[key][mask],label=label,color=color,lw=1.5)
    axes[1].plot(t[mask],curve['accuracy'][mask],label='Acc',color='#555555',lw=1.5)
    axes[1].axhline(summary['all_normal_baseline']['accuracy'],ls=':',color='#999999',
                   label=f"全判正常 Acc={summary['all_normal_baseline']['accuracy']:.4f}")
    for key,label,color in [('macro_f1','Macro-F1','#705b94'),('f1_abnormal','F1_abnormal','#d48a24')]:
        axes[2].plot(t[mask],curve[key][mask],label=label,color=color,lw=1.5)
    for ax in axes:
        for key,color in [('sp90_max_se','#326c3d'),('youden_max','#8f6b9d'),('default_050','#999999')]:
            point=summary['working_points'][key]
            ax.axvline(point['threshold'],color=color,ls='--',lw=.9,alpha=.8)
        ax.set_ylim(-.02,1.03); ax.set_xlim(0,1); ax.grid(alpha=.2)
        ax.set_ylabel('指标值'); ax.legend(loc='best',frameon=False)
    axes[2].set_xlabel('异常概率阈值')
    fig.suptitle('DS2 完整阈值曲线  探索性分析',fontsize=15)
    fig.text(.5,.025,'工作点已由 DS1 OOF 冻结；本图从同一次缓存预测推导，不参与阈值选择',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95)); fig.savefig(OUT/'ds2_exploratory_threshold_curve.png'); plt.close(fig)
    fpr,tpr,_=roc_curve(y,prob)
    fig,ax=plt.subplots(figsize=(6.6,5.4))
    ax.plot(fpr,tpr,color='#2877a5',label=f"DS2 AUC = {summary['auc']:.4f}")
    ax.plot([0,1],[0,1],ls=':',color='#aaaaaa')
    descriptions=[('sp90_max_se','OOF Sp≥0.90 主点','#326c3d'),
                  ('youden_max','OOF Youden 点','#8f6b9d'),('default_050','默认 0.50','#777777')]
    for i,(key,label,color) in enumerate(descriptions):
        point=summary['working_points'][key]; x=1-point['specificity']; yy=point['sensitivity']
        ax.scatter(x,yy,color=color,zorder=3)
        ax.annotate(f"{label}\nt={point['threshold']:.4f}; Se={yy:.3f}, Sp={1-x:.3f}",
                    (x,yy),xytext=(.43,.12+.16*i),textcoords='axes fraction',fontsize=8,color=color,
                    arrowprops={'arrowstyle':'-','color':color,'lw':.7})
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='1 − Sp',ylabel='Se',title='冻结后模型的 DS2 ROC 与预先选定工作点')
    ax.grid(alpha=.2); ax.legend(loc='lower right',frameon=False)
    fig.tight_layout(); fig.savefig(OUT/'ds2_roc_frozen_points.png'); plt.close(fig)


def main():
    marker=OUT/'ds2_evaluation.json'
    if marker.exists():
        raise RuntimeError('DS2 evaluation already started; refusing second inference')
    frozen=read(OUT/'frozen_workpoints.json'); protocol=read(OUT/'protocol.json')
    assert frozen['frozen']
    assert sha256(OUT/'protocol.json')==frozen['protocol_sha256']
    assert sha256(OUT/'final_binary.pt')==frozen['final_model_sha256']
    for name,digest in frozen['source_sha256'].items():
        assert sha256(ROOT/'src'/name)==digest,'Frozen source changed: '+name
    audit={'status':'started','started_at_utc':datetime.now(timezone.utc).isoformat(),
           'frozen_workpoints_sha256':sha256(OUT/'frozen_workpoints.json'),
           'model_sha256':frozen['final_model_sha256'],'inference_passes':1,
           'exploratory_curve_policy':protocol['test_exploration_label']}
    save(marker,audit)
    X,y4,rr,rec,_=load_beats(with_rr=True)
    idx=np.flatnonzero(np.isin(rec,DS2)); Xte=X[idx]; y=(y4[idx]>0).astype(np.int64)
    assert len(idx)==49705
    model=ECGCNNLight(num_classes=2)
    ck=torch.load(OUT/'final_binary.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model']); model.eval()
    logits=predict_logits(model,Xte)
    prob=torch.softmax(torch.from_numpy(logits),1).numpy()[:,1]
    np.savez_compressed(OUT/'ds2_predictions.npz',global_index=idx,y=y,prob=prob,record=rec[idx])
    points={key:metrics_at(y,prob,p['threshold']) for key,p in frozen['points'].items()}
    summary={'dataset':'DS2','n':len(y),'n_normal':int((y==0).sum()),'n_abnormal':int(y.sum()),
             'primary_metric':'AUC','auc':auc(y,prob),'primary_workpoint':'sp90_max_se',
             'working_points':points,'all_normal_baseline':baseline(y),
             'primary_selection_rule':protocol['primary_rule'],
             'selection_dataset':'DS1 pooled OOF, excluding record 201',
             'note':'Sp>=0.90 is an OOF selection constraint; DS2 specificity is measured, not guaranteed.',
             'accuracy_note':'阴性占约89%，准确率不适合作为本任务的主指标。',
             'prior_test_exposure':protocol['prior_test_exposure'],
             'model_sha256':frozen['final_model_sha256'],'frozen_workpoints_sha256':audit['frozen_workpoints_sha256'],
             'cached_predictions_sha256':sha256(OUT/'ds2_predictions.npz')}
    save(OUT/'ds2_summary.json',summary)
    curve=threshold_curve(y,prob,extra=[p['threshold'] for p in frozen['points'].values()])
    write_curve(OUT/'ds2_threshold_curve_exploratory.csv',curve)
    save(OUT/'ds2_threshold_curve_metadata.json',{
        'label':'探索性分析','purpose':protocol['test_exploration_label'],
        'definition':'every distinct predicted-probability breakpoint plus endpoints and frozen workpoints; positive if probability >= threshold',
        'rows':len(curve['threshold']),'columns':list(curve),
        'predictions_sha256':sha256(OUT/'ds2_predictions.npz'),
        'frozen_before_test':True,'not_used_for_workpoint_selection':True,
        'curve_sha256':sha256(OUT/'ds2_threshold_curve_exploratory.csv')})
    rows=[]; primary_threshold=frozen['points']['sp90_max_se']['threshold']
    for r in DS2:
        mask=rec[idx]==r; m=metrics_at(y[mask],prob[mask],primary_threshold)
        rows.append({'record':r,**m,'abnormal':int(y[mask].sum())})
    save(OUT/'ds2_per_record_primary.json',rows)
    save(OUT/'runtime_environment.json',{'python':platform.python_version(),'torch_threads':torch.get_num_threads(),
          'packages':{p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','scikit-learn')}})
    plots(y,prob,summary,curve)
    audit.update(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 n=len(y),auc=summary['auc'],cached_predictions_sha256=summary['cached_predictions_sha256'])
    save(marker,audit)
    print('FINAL DS2 SUMMARY',summary,flush=True)


if __name__=='__main__':
    main()
