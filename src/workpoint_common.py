"""Shared tools for DS1-only operating-point selection and cached test analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from config import ROOT

OUT = ROOT / 'results' / 'workpoint_cv_v2'
FIELDS = ['threshold','sensitivity','specificity','accuracy','macro_f1','f1_abnormal',
          'f1_normal','youden_j','tp','fp','tn','fn','n']


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def save(path,obj):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ratio(a,b):
    return np.divide(a,b,out=np.zeros_like(np.asarray(a),dtype=np.float64),where=np.asarray(b)!=0)


def metrics_at(y,prob,threshold):
    y=np.asarray(y,dtype=np.int64)
    pred=np.asarray(prob,dtype=np.float64)>=threshold
    tp=int(np.sum(pred & (y==1))); fp=int(np.sum(pred & (y==0)))
    tn=int(np.sum(~pred & (y==0))); fn=int(np.sum(~pred & (y==1)))
    se=tp/max(tp+fn,1); sp=tn/max(tn+fp,1)
    f1a=2*tp/max(2*tp+fp+fn,1); f1n=2*tn/max(2*tn+fp+fn,1)
    return dict(threshold=float(threshold),sensitivity=se,specificity=sp,
                accuracy=(tp+tn)/len(y),macro_f1=(f1a+f1n)/2,
                f1_abnormal=f1a,f1_normal=f1n,youden_j=se+sp-1,
                tp=tp,fp=fp,tn=tn,fn=fn,n=len(y))


def threshold_curve(y,prob,extra=()):
    """Every distinct probability breakpoint, plus endpoints/requested points.

    Prediction is abnormal iff probability >= threshold. The value above one
    explicitly represents all-negative predictions even for saturated scores.
    All rows are derived from one cached probability vector, never reinference.
    """
    y=np.asarray(y,dtype=np.int64); prob=np.asarray(prob,dtype=np.float64)
    assert len(y)==len(prob) and np.isfinite(prob).all()
    order=np.argsort(prob,kind='stable')
    p=prob[order]; yy=y[order]
    thresholds=np.unique(np.r_[0.,p,.5,1.,np.nextafter(1.,2.),np.asarray(extra,dtype=float)])
    left=np.searchsorted(p,thresholds,side='left')
    positives=int(y.sum()); negatives=len(y)-positives
    cum_pos=np.r_[0,np.cumsum(yy)]
    fn=cum_pos[left]; tn=left-fn; tp=positives-fn; fp=negatives-tn
    se=ratio(tp,positives); sp=ratio(tn,negatives)
    fa=ratio(2*tp,2*tp+fp+fn); fno=ratio(2*tn,2*tn+fp+fn)
    return {'threshold':thresholds,'sensitivity':se,'specificity':sp,
            'accuracy':ratio(tp+tn,len(y)),'macro_f1':(fa+fno)/2,
            'f1_abnormal':fa,'f1_normal':fno,'youden_j':se+sp-1,
            'tp':tp,'fp':fp,'tn':tn,'fn':fn,'n':np.full(len(thresholds),len(y))}


def row_at(curve,index):
    return {key:(int(curve[key][index]) if key in ('tp','fp','tn','fn','n')
                 else float(curve[key][index])) for key in FIELDS}


def select_points(y,prob):
    curve=threshold_curve(y,prob)
    feasible=np.flatnonzero(curve['specificity']>=.90)
    primary=max(feasible,key=lambda i:(curve['sensitivity'][i],curve['specificity'][i],curve['threshold'][i]))
    youden=max(range(len(curve['threshold'])),key=lambda i:(curve['youden_j'][i],curve['specificity'][i],curve['threshold'][i]))
    return {'sp90_max_se':row_at(curve,primary),'youden_max':row_at(curve,youden),
            'default_050':metrics_at(y,prob,.5)},curve


def write_curve(path,curve):
    import csv
    with open(path,'w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream)
        writer.writerow(FIELDS)
        for i in range(len(curve['threshold'])):
            row=row_at(curve,i)
            writer.writerow([row[k] for k in FIELDS])


def baseline(y):
    return metrics_at(y,np.zeros(len(y)),np.nextafter(1.,2.))


def auc(y,prob):
    return float(roc_auc_score(y,prob))
