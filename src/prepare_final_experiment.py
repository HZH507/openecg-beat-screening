"""Declare auxiliary model and noise evaluation protocol without record 201."""
from __future__ import annotations

from datetime import datetime,timezone
import itertools

import numpy as np

from config import DS1,DS2,PROCESSED,CLASS_NAMES,SEED
from final_common import OUT,MODELS,FIGURES,CV,read,save,assert_binary_frozen
from workpoint_common import sha256


def main():
    assert_binary_frozen()
    if OUT.exists():
        raise RuntimeError('Final report experiment directory already exists; refusing overwrite')
    for p in (OUT,MODELS,FIGURES):p.mkdir(exist_ok=True)
    with np.load(PROCESSED/'beats.npz',allow_pickle=False) as d:y,rec=d['y'],d['record']
    eligible=sorted(set(DS1)-{'201'})
    counts={r:np.bincount(y[rec==r],minlength=4) for r in eligible}
    total=sum(counts.values()); rates={r:float(counts[r][1:].sum()/counts[r].sum()) for r in eligible}
    ordered=sorted(eligible,key=lambda r:(rates[r],r))
    strata=[list(g) for g in np.array_split(ordered,4)]
    candidates=list(itertools.product(*strata)); np.random.default_rng(SEED).shuffle(candidates)
    best=None
    for choice in candidates:
        c=sum((counts[r] for r in choice),np.zeros(4,dtype=int))
        if np.any(c==0) or np.any(total-c==0):continue
        score=float(np.mean((c/total-4/21)**2))
        if best is None or score<best[0]-1e-15:best=(score,sorted(choice))
    assert best is not None
    val_records=best[1]; train_records=sorted(set(eligible)-set(val_records))
    tr=np.flatnonzero(np.isin(rec,train_records)); va=np.flatnonzero(np.isin(rec,val_records))
    split={'seed':SEED,'train_records':train_records,'val_records':val_records,
           'train_n':len(tr),'val_n':len(va),'train_indices':tr.tolist(),'val_indices':va.tolist(),
           'train_counts':{c:int(np.sum(y[tr]==i)) for i,c in enumerate(CLASS_NAMES)},
           'val_counts':{c:int(np.sum(y[va]==i)) for i,c in enumerate(CLASS_NAMES)},
           'eligible_records':eligible,'excluded_record':'201','excluded_201_n':1963,
           'rule':'one validation record per DS1 eligible abnormal-fraction quartile; minimize per-class validation fraction discrepancy from 4/21; every class present on both sides; seeded ties',
           'selection_used_ds2':False,'dataset_sha256':sha256(PROCESSED/'beats.npz')}
    save(OUT/'aux_split.json',split)
    groups=[]
    for key,label,records in [('trainval','训练与验证',eligible),('ds2','测试 DS2',DS2),
                               ('excluded201','排除 201',['201']),('protocol_excluded','协议未纳入',['102','104','107','217'])]:
        mask=np.isin(rec,records)
        groups.append({'key':key,'label':label,'records':records,'n_records':len(records),'n':int(mask.sum()),
                       'counts':{c:int(np.sum(y[mask]==i)) for i,c in enumerate(CLASS_NAMES)}})
    assert [g['n'] for g in groups]==[49050,49705,1963,733]
    assert groups[1]['counts']==dict(N=44259,S=1837,V=3221,F=388)
    accounting={'groups':groups,'total':{'n_records':48,'n':101451,
                'counts':{c:sum(g['counts'][c] for g in groups) for c in CLASS_NAMES}},
                'source':'processed dataset record membership; four groups fixed by user-confirmed protocol'}
    assert sum(g['n'] for g in groups)==101451
    save(OUT/'dataset_accounting.json',accounting)
    frozen=read(CV/'frozen_workpoints.json')
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,
              'excluded_201_from_all_auxiliary_models':True,'aux_split_sha256':sha256(OUT/'aux_split.json'),
              'binary_checkpoint':'results/workpoint_cv_v2/final_binary.pt',
              'binary_sha256':frozen['final_model_sha256'],
              'binary_threshold':frozen['points']['sp90_max_se']['threshold'],
              'clean_binary_ds2_policy':'reuse frozen prediction cache; no clean DS2 inference',
              'auxiliary_policy':'train and select on DS1 minus 201; freeze checkpoints before their single clean DS2 evaluation',
              'noise_policy':'nine predeclared bw/em/ma x 5/10/15 dB conditions; raw/classical/DnCNN; all use the frozen binary classifier and full-precision primary threshold; no tuning',
              'noise_subset':'all 5446 DS2 abnormal plus seeded 4000 normal beats; n=9446',
              'noise_seed_rule':'42 + SNR + {bw:1,em:2,ma:3}[kind]*100',
              'noise_source_split':'per channel chronological first 65% training/validation; last 35% testing; split before concatenating channels',
              'noise_source_change':'This revision makes the chronological boundary shared by both channels; prior runs split a concatenated channel bank. No old noise metrics are reused.',
              'latency_protocol':'FP32 CPU, 4 threads, synthetic batch 128, 5 warmups and 20 timed repetitions; median batch time /128; INT8 numbers are weight-size estimates only'}
    save(OUT/'experiment_protocol.json',protocol)
    print({k:split[k] for k in ('train_records','val_records','train_n','val_n','train_counts','val_counts')},flush=True)


if __name__=='__main__':main()
