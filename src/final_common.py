"""Report-revision experiment paths and immutable binary provenance checks."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import wfdb

from config import ROOT,NSTDB
from workpoint_common import sha256,metrics_at,auc

OUT=ROOT/'results'/'report_final'
MODELS=OUT/'models'
FIGURES=OUT/'figures'
CV=ROOT/'results'/'workpoint_cv_v2'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def assert_binary_frozen():
    frozen=read(CV/'frozen_workpoints.json'); audit=read(CV/'ds2_evaluation.json')
    assert sha256(CV/'final_binary.pt')==frozen['final_model_sha256']
    assert sha256(CV/'frozen_workpoints.json')==audit['frozen_workpoints_sha256']
    assert sha256(CV/'ds2_predictions.npz')==audit['cached_predictions_sha256']
    assert audit['inference_passes']==1 and audit['status']=='complete'
    for name,digest in frozen['source_sha256'].items():
        assert sha256(ROOT/'src'/name)==digest,'Frozen binary source changed: '+name
    return frozen


def load_noise_bank():
    """Separate both NSTDB channels at the same chronological 65% boundary.

    Split before channel concatenation, so the two correlated channels at a
    physical time cannot lie on opposite sides of the train/test boundary.
    This helper leaves the binary experiment's frozen utils.py untouched.
    """
    bank={}
    for kind in ('bw','em','ma'):
        rec=wfdb.rdrecord(str(NSTDB/kind))
        sig=np.asarray(rec.p_signal,dtype=np.float32)
        cut=int(.65*len(sig))
        bank[kind]={part:np.concatenate([sig[a:b,c] for c in range(sig.shape[1])])
                    for part,a,b in [('train',0,cut),('test',cut,len(sig))]}
        for part in bank[kind]:
            bank[kind][part]-=bank[kind][part].mean()
    return bank


def binary_metrics(y,prob,threshold):
    return {**metrics_at(y,prob,threshold),'auc':auc(y,prob)}
