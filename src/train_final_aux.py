"""Train four-class CNN/MLP or DnCNN using the final 201-free record split."""
from __future__ import annotations

import argparse,copy,time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,WeightedRandomSampler,TensorDataset
from sklearn.metrics import f1_score,accuracy_score

from config import CLASS_NAMES,SEED
from models import ECGCNNLight,MLPBaseline,DnCNN1d,count_parameters
from train_classifier import BeatDataset
from utils import load_beats,set_seed,predict_logits,add_noise,snr_db
from final_common import OUT,MODELS,read,save,assert_binary_frozen,load_noise_bank
from workpoint_common import sha256

torch.set_num_threads(2)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['cnn','mlp','denoiser'],required=True);args=ap.parse_args()
    assert_binary_frozen()
    if (OUT/'aux_models_frozen.json').exists():raise RuntimeError('Auxiliary models frozen; training closed')
    set_seed(SEED);split=read(OUT/'aux_split.json')
    X,y,rr,rec,_=load_beats(with_rr=True)
    tr=np.asarray(split['train_indices']);va=np.asarray(split['val_indices'])
    assert '201' not in rec[np.r_[tr,va]]
    rng=np.random.default_rng(SEED)
    if args.model=='denoiser':
        used=tr[np.sort(rng.choice(len(tr),size=min(12000,len(tr)),replace=False))]
        epochs,patience,lr=12,4,1e-3
        model=DnCNN1d(channels=32,depth=6,kernel=11)
        loader=DataLoader(TensorDataset(torch.from_numpy(X[used]).float()),batch_size=128,shuffle=True,drop_last=True)
        bank=load_noise_bank();val_noise=add_noise(X[va],bank,np.random.default_rng(SEED+1),split='train')
        vmu=val_noise.mean(1,keepdims=True);vsd=val_noise.std(1,keepdims=True)+1e-6
        vx=torch.from_numpy((val_noise-vmu)/vsd).unsqueeze(1)
        snr_in=float(np.mean([snr_db(c,n) for c,n in zip(X[va],val_noise)]))
        criterion=nn.MSELoss();ckpt='denoiser_best.pt';config_name='denoiser_training.json'
    else:
        kept=[]
        for c,cap in [(0,18000),(1,None),(2,3000),(3,None)]:
            idx=tr[y[tr]==c]
            if cap is not None and len(idx)>cap:idx=rng.choice(idx,size=cap,replace=False)
            kept.append(idx)
        used=np.sort(np.concatenate(kept)); counts=np.bincount(y[used],minlength=4).astype(float)
        weights=counts**-.35;weights/=weights.mean()
        ds=BeatDataset(X[used],rr[used],y[used],augment=True)
        if args.model=='cnn':
            sampler=WeightedRandomSampler(torch.as_tensor((counts**-.35)[y[used]],dtype=torch.double),len(used),replacement=True)
            loader=DataLoader(ds,batch_size=128,sampler=sampler,drop_last=True)
            criterion=nn.CrossEntropyLoss(label_smoothing=.05);model=ECGCNNLight();lr=1e-3;ckpt='cnn_light_best.pt'
        else:
            loader=DataLoader(ds,batch_size=128,shuffle=True,drop_last=True)
            criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32),label_smoothing=.05)
            model=MLPBaseline();lr=5e-4;ckpt='mlp_best.pt'
        epochs,patience=22,8;config_name='classification_config.json'
    if (MODELS/ckpt).exists():raise RuntimeError('Checkpoint already exists; refusing overwrite')
    config={'model':args.model,'seed':SEED,'max_epochs':epochs,'patience':patience,'batch':128,'lr':lr,
            'threads':2,'train_records':sorted(set(rec[used])),'val_records':split['val_records'],
            'train_n_after_cap':len(used),'train_n_before_cap':len(tr),'val_n':len(va),
            'train_global_indices':used.tolist(),'val_global_indices':va.tolist(),
            'train_counts':{c:int(np.sum(y[used]==i)) for i,c in enumerate(CLASS_NAMES)},
            'val_counts':split['val_counts'],'excluded_201':True,
            'aux_split_sha256':sha256(OUT/'aux_split.json'),'params':count_parameters(model)}
    if args.model=='denoiser':
        config['noise_split']=read(OUT/'experiment_protocol.json')['noise_source_split'];save(OUT/config_name,config)
    else:
        configs=read(OUT/config_name) if (OUT/config_name).exists() else {}
        configs[args.model]=config;save(OUT/config_name,configs)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    history=[];best=float('inf') if args.model=='denoiser' else -1.;best_epoch=None;bad=0;start=time.time()
    noise_rng=np.random.default_rng(SEED)
    for epoch in range(1,epochs+1):
        model.train();running=0.;n=0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            if args.model=='denoiser':
                clean=batch[0].numpy();noisy=add_noise(clean,bank,noise_rng,split='train')
                mu=noisy.mean(1,keepdims=True);sd=noisy.std(1,keepdims=True)+1e-6
                xb=torch.from_numpy((noisy-mu)/sd).unsqueeze(1)
                cb=torch.from_numpy((clean-mu)/sd)
                loss=criterion(model(xb).squeeze(1),cb)
            else:
                xb,_,yb=batch;loss=criterion(model(xb),yb)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            running+=float(loss.detach())*len(xb);n+=len(xb)
        sched.step();model.eval()
        if args.model=='denoiser':
            with torch.inference_mode():out=np.concatenate([model(vx[i:i+128]).squeeze(1).numpy() for i in range(0,len(vx),128)])*vsd+vmu
            score=float(np.mean((out-X[va])**2));snr_out=float(np.mean([snr_db(c,o) for c,o in zip(X[va],out)]))
            row={'epoch':epoch,'train_loss':running/n,'val_mse':score,'val_snr_in':snr_in,'val_snr_out':snr_out,'val_delta_snr':snr_out-snr_in}
            improved=score<best
        else:
            pred=predict_logits(model,X[va]).argmax(1)
            score=float(f1_score(y[va],pred,labels=[0,1,2,3],average='macro',zero_division=0))
            row={'epoch':epoch,'train_loss':running/n,'val_acc':float(accuracy_score(y[va],pred)),'val_macro_f1':score}
            improved=score>best+1e-4
        history.append(row);save(OUT/(args.model+'_history.json' if args.model!='cnn' else 'cnn_light_history.json'),history)
        print(args.model,f'epoch {epoch:02d}',row,f'({time.time()-start:.0f}s)',flush=True)
        if improved:
            best=score;best_epoch=epoch;bad=0
            torch.save({**config,'model':copy.deepcopy(model.state_dict()),'epoch':epoch,
                        'validation_metric':score,'selection_rule':'minimum validation MSE' if args.model=='denoiser' else 'maximum validation four-class Macro-F1',
                        },MODELS/ckpt)
        else:
            bad+=1
            if bad>=patience:break
    config['best_epoch']=best_epoch;config['best_validation_metric']=best
    if args.model=='denoiser':save(OUT/config_name,config)
    else:
        configs=read(OUT/config_name);configs[args.model]=config;save(OUT/config_name,configs)
    print('COMPLETE',args.model,'epoch',best_epoch,'validation',best,flush=True)


if __name__=='__main__':main()
