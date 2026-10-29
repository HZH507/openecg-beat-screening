"""Nested epoch selection, then record-held-out prediction for one OOF fold."""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import ROOT
from models import ECGCNNLight,count_parameters
from train_classifier import BeatDataset
from utils import load_beats,predict_logits,set_seed
from workpoint_common import OUT,auc,read,save,sha256

torch.set_num_threads(2)


def check_protocol(protocol):
    assert not (OUT/'frozen_workpoints.json').exists(),'Experiment frozen; training closed'
    for name,digest in protocol['source_sha256'].items():
        assert sha256(ROOT/'src'/name)==digest,'Protocol source changed: '+name


def fit_model(X,y,rr,train_idx,*,epochs,seed,path,tag,protocol,val_idx=None):
    cfg=protocol['training']
    set_seed(seed)
    loader=DataLoader(BeatDataset(X[train_idx],rr[train_idx],y[train_idx],augment=True),
                      batch_size=cfg['batch'],shuffle=True,num_workers=0,drop_last=True)
    model=ECGCNNLight(num_classes=2)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor([1.,float(cfg['abnormal_weight'])]),
                                  label_smoothing=cfg['label_smoothing'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['max_epochs'])
    best_auc=-1.; best_epoch=None; best_state=None; bad=0; history=[]; start=time.time()
    print(tag,'training',len(train_idx),'validation',0 if val_idx is None else len(val_idx),flush=True)
    for epoch in range(1,epochs+1):
        model.train(); running=0.; n=0
        for xb,_,yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(xb),yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            running+=float(loss.detach())*len(xb); n+=len(xb)
        scheduler.step()
        row={'epoch':epoch,'train_loss':running/n,'elapsed_seconds':time.time()-start}
        if val_idx is not None:
            logits=predict_logits(model,X[val_idx])
            prob=torch.softmax(torch.from_numpy(logits),1).numpy()[:,1]
            score=auc(y[val_idx],prob); row['inner_val_auc']=score
            if score>best_auc+cfg['min_auc_improvement']:
                best_auc=score; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); bad=0
            else:
                bad+=1
        history.append(row); save(path/f'{tag}_history.json',history)
        print(tag,f'epoch {epoch:02d} loss {running/n:.5f}',
              f'inner AUC {row["inner_val_auc"]:.5f}' if val_idx is not None else 'fixed epoch refit',
              f'({time.time()-start:.0f}s)',flush=True)
        if val_idx is not None and bad>=cfg['patience']:
            print(tag,'inner early stop',flush=True)
            break
    if val_idx is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch=epochs
    model.eval()
    checkpoint={'model':model.state_dict(),'epoch':best_epoch,'inner_val_auc':best_auc if val_idx is not None else None,
                'seed':seed,'abnormal_weight':cfg['abnormal_weight'],'parameters':count_parameters(model),
                'protocol_sha256':sha256(OUT/'protocol.json'),'training_global_indices':train_idx.tolist(),
                'validation_global_indices':[] if val_idx is None else val_idx.tolist(),
                'selection_rule':'inner validation AUC' if val_idx is not None else 'fixed epoch count selected without outer validation labels'}
    torch.save(checkpoint,path/f'{tag}.pt')
    return model,best_epoch,best_auc


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,choices=range(1,6),required=True)
    args=ap.parse_args(); protocol=read(OUT/'protocol.json'); check_protocol(protocol)
    fold=protocol['folds'][args.fold-1]
    path=OUT/f'fold_{args.fold}'; path.mkdir(exist_ok=True)
    if (path/'complete.json').exists():
        print('Fold already complete; no retraining',args.fold,flush=True); return
    X,y4,rr,rec,_=load_beats(with_rr=True); y=(y4>0).astype(np.int64)
    inner_train=np.flatnonzero(np.isin(rec,fold['inner_train_records']))
    inner_val=np.flatnonzero(np.isin(rec,fold['inner_val_records']))
    outer_train=np.flatnonzero(np.isin(rec,fold['outer_train_records']))
    outer_val=np.flatnonzero(np.isin(rec,fold['outer_val_records']))
    assert not np.intersect1d(outer_val,np.r_[inner_train,inner_val]).size
    assert not np.any(rec[np.r_[outer_train,outer_val]]=='201')
    save(path/'status.json',{'stage':'inner epoch selection','fold':args.fold})
    _,epochs,inner_auc=fit_model(X,y,rr,inner_train,epochs=protocol['training']['max_epochs'],
                               seed=42+args.fold*100,path=path,tag='inner',protocol=protocol,val_idx=inner_val)
    save(path/'status.json',{'stage':'outer refit','fold':args.fold,'fixed_epochs':epochs,'inner_auc':inner_auc})
    model,_,_=fit_model(X,y,rr,outer_train,epochs=epochs,seed=42+args.fold*100+1,
                        path=path,tag='outer_refit',protocol=protocol)
    prob=torch.softmax(torch.from_numpy(predict_logits(model,X[outer_val])),1).numpy()[:,1]
    np.savez_compressed(path/'oof_predictions.npz',global_index=outer_val,y=y[outer_val],prob=prob,record=rec[outer_val])
    summary={**fold,'selected_epochs':epochs,'inner_best_auc':inner_auc,'outer_oof_auc':auc(y[outer_val],prob),
             'checkpoint_sha256':sha256(path/'outer_refit.pt'),
             'predictions_sha256':sha256(path/'oof_predictions.npz'),'protocol_sha256':sha256(OUT/'protocol.json')}
    save(path/'complete.json',summary); save(path/'status.json',{'stage':'complete',**summary})
    print('FOLD COMPLETE',summary,flush=True)


if __name__=='__main__':
    main()
