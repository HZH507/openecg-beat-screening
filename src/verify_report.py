"""Independently verify the final ECG DOCX against frozen and new experiment data.

No model inference is performed. Binary metrics are recomputed from the single
frozen DS2 cache; every table cell is compared with the manifest and numeric
cells are additionally checked against the experiment JSON/cache.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

NS = {
    'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'pr':'http://schemas.openxmlformats.org/package/2006/relationships',
    'dc':'http://purl.org/dc/elements/1.1/',
    'cp':'http://schemas.openxmlformats.org/package/2006/metadata/core-properties',
    'dcterms':'http://purl.org/dc/terms/',
}
CLASS_NAMES = ('N','S','V','F')
REPOSITORY_URL = 'https://github.com/HZH507/openecg-beat-screening'
KNOWN_DS2 = {'N':44259,'S':1837,'V':3221,'F':388}
EXPECTED_TOTALS = [(21,49050),(22,49705),(1,1963),(4,733),(48,101451)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def blob_digest(data):
    return hashlib.sha256(data).hexdigest()


def normal(text):
    return re.sub(r'\s+',' ',str(text).replace('\u00a0',' ').replace('\u3000',' ')).strip()


def compact(text):
    return re.sub(r'\s+','',str(text)).replace(',','').replace('，','')


def element_text(element):
    # Native Word equations store their editable text in m:t, not w:t.
    tags={f"{{{NS['w']}}}t",f"{{{NS['m']}}}t"}
    return ''.join(node.text or '' for node in element.iter() if node.tag in tags)


class Audit:
    def __init__(self):
        self.checks=[]

    def check(self, condition, message):
        if not condition:
            raise AssertionError(message)
        self.checks.append(message)

    def equal(self,actual,expected,message):
        self.check(actual==expected,f'{message}: {actual!r} != {expected!r}')

    def close(self,actual,expected,message,tolerance=1e-11):
        self.check(abs(float(actual)-float(expected))<=tolerance,
                   f'{message}: {actual!r} != {expected!r}')


def cached_binary_metrics(y,prob,threshold):
    y=np.asarray(y,dtype=np.int64); prob=np.asarray(prob,dtype=np.float64)
    pred=prob>=threshold
    tp=int(np.sum((y==1)&pred)); fp=int(np.sum((y==0)&pred))
    tn=int(np.sum((y==0)&~pred)); fn=int(np.sum((y==1)&~pred))
    f1a=2*tp/max(2*tp+fp+fn,1); f1n=2*tn/max(2*tn+fp+fn,1)
    return {'tp':tp,'fp':fp,'tn':tn,'fn':fn,'n':len(y),
            'sensitivity':tp/max(tp+fn,1),'specificity':tn/max(tn+fp,1),
            'accuracy':(tp+tn)/len(y),'macro_f1':(f1a+f1n)/2,
            'f1_abnormal':f1a,'f1_normal':f1n}


def binary_auc(y,prob):
    """Mann--Whitney AUC with average ranks for tied cached probabilities."""
    y=np.asarray(y,dtype=np.int64); prob=np.asarray(prob,dtype=np.float64)
    order=np.argsort(prob,kind='stable'); sorted_prob=prob[order]
    starts=np.r_[0,np.flatnonzero(sorted_prob[1:]!=sorted_prob[:-1])+1]
    stops=np.r_[starts[1:],len(y)]
    ranks=np.empty(len(y),dtype=float)
    for begin,end in zip(starts,stops):
        ranks[order[begin:end]]=(begin+1+end)/2
    n1=int(y.sum()); n0=len(y)-n1
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def source_path(root,data,key,default):
    paths=data.get('source_files',{})
    candidate=Path(paths.get(key,default))
    return candidate if candidate.is_absolute() else root/candidate


def project_file(root,path):
    """Resolve saved absolute provenance paths under the current package root."""
    text=str(path).replace('\\','/')
    found=re.search(r'(?:^|/)(results|src|report|data)/',text)
    if found:
        return root/text[found.start(1):]
    candidate=Path(path)
    return candidate if candidate.is_absolute() else root/candidate


def verify_frozen_binary(root,data,audit):
    folder=root/'results/workpoint_cv_v2'
    frozen=read(folder/'frozen_workpoints.json'); summary=read(folder/'ds2_summary.json')
    protocol=read(folder/'protocol.json'); test=read(folder/'ds2_evaluation.json')
    audit.check(frozen['frozen'],'binary working points are frozen')
    audit.equal(test['status'],'complete','clean DS2 evaluation completed')
    audit.equal(test['inference_passes'],1,'clean DS2 inferred once')
    audit.check(datetime.fromisoformat(frozen['frozen_at_utc'])<datetime.fromisoformat(test['started_at_utc']),
                'working points frozen before clean DS2 evaluation')
    audit.equal(digest(folder/'frozen_workpoints.json'),test['frozen_workpoints_sha256'],'frozen JSON hash')
    audit.equal(digest(folder/'final_binary.pt'),frozen['final_model_sha256'],'frozen final binary model hash')
    audit.equal(digest(folder/'protocol.json'),frozen['protocol_sha256'],'binary protocol hash')
    audit.equal(digest(folder/'ds2_predictions.npz'),summary['cached_predictions_sha256'],'clean DS2 cache hash')
    for name,sha in frozen['source_sha256'].items():
        audit.equal(digest(root/'src'/name),sha,'frozen source '+name)
    audit.equal(len(frozen['final_train_records']),21,'binary final train record count')
    audit.check('201' not in frozen['final_train_records'],'201 absent from final binary training')
    with np.load(folder/'ds2_predictions.npz',allow_pickle=False) as d:
        y,prob,record=d['y'],d['prob'],d['record']
    audit.equal(len(y),49705,'cached clean DS2 size')
    audit.equal(int((y==0).sum()),44259,'cached clean DS2 negatives')
    audit.equal(int(y.sum()),5446,'cached clean DS2 positives')
    auc=binary_auc(y,prob)
    audit.close(auc,summary['auc'],'independently recomputed clean AUC')
    for name,point in summary['working_points'].items():
        audit.close(point['threshold'],frozen['points'][name]['threshold'],'frozen threshold '+name)
        measured=cached_binary_metrics(y,prob,point['threshold'])
        for key,value in measured.items():
            audit.close(point[key],value,'cached clean metric '+name+'/'+key)
    baseline=cached_binary_metrics(y,np.zeros(len(y)),2.)
    for key,value in baseline.items():
        audit.close(summary['all_normal_baseline'][key],value,'all-normal baseline '+key)
    with np.load(folder/'oof_predictions.npz',allow_pickle=False) as d:
        oy,op,oi,orec=d['y'],d['prob'],d['global_index'],d['record']
    audit.equal(len(oy),49050,'OOF total')
    audit.equal(len(np.unique(oi)),49050,'OOF predictions exactly once per beat')
    audit.check('201' not in orec,'201 absent from OOF predictions')
    audit.close(binary_auc(oy,op),frozen['oof_auc'],'independently recomputed OOF AUC')
    audit.check(frozen['points']['sp90_max_se']['specificity']>=.9,'OOF specificity constraint')
    audit.close(frozen['points']['sp90_max_se']['threshold'],.5878590941429138,'adopted primary threshold')
    audit.equal(data['binary'],summary,'report binary data matches frozen source')
    audit.close(data['oof']['oof_auc'],frozen['oof_auc'],'report OOF AUC matches source')
    return frozen,summary,protocol,(y,prob,record)


def verify_dataset(data,audit):
    groups=data['dataset']['groups']
    audit.equal(len(groups),5,'table 1 has four groups plus total')
    for group,(n_records,n) in zip(groups,EXPECTED_TOTALS):
        audit.equal(int(group['n_records']),n_records,'dataset records '+group['key'])
        audit.equal(int(group['n']),n,'dataset beats '+group['key'])
        audit.equal(sum(int(group['counts'][c]) for c in CLASS_NAMES),n,'dataset class sum '+group['key'])
    audit.equal(sum(int(g['n']) for g in groups[:-1]),groups[-1]['n'],'four-group total')
    audit.equal(groups[1]['counts'],KNOWN_DS2,'known DS2 four-class distribution')
    for c in CLASS_NAMES:
        audit.equal(sum(int(g['counts'][c]) for g in groups[:-1]),int(groups[-1]['counts'][c]),'class total '+c)
    audit.equal(data['aux_split']['seed'],42,'auxiliary split seed')
    tr=data['aux_split']['train_records']; va=data['aux_split']['val_records']
    audit.check(not set(tr)&set(va),'auxiliary train/validation records disjoint')
    audit.check('201' not in tr+va,'201 absent from auxiliary train/validation')
    audit.equal(len(set(tr+va)),21,'auxiliary eligible record coverage')
    audit.equal(data['aux_split']['train_n']+data['aux_split']['val_n'],49050,'auxiliary eligible beat coverage')


def verify_raw_experiments(root,data,frozen,audit):
    folder=root/'results/report_final'
    raw_dataset=read(folder/'dataset_accounting.json')
    for report_group,raw_group in zip(data['dataset']['groups'][:-1],raw_dataset['groups']):
        for key in ('key','label','n_records','n','counts'):
            audit.equal(report_group[key],raw_group[key],'dataset accounting source '+report_group['key']+'/'+key)
    for key in ('n_records','n','counts'):
        audit.equal(data['dataset']['groups'][-1][key],raw_dataset['total'][key],'dataset total source '+key)
    split=read(folder/'aux_split.json')
    audit.equal(data['aux_split'],split,'auxiliary split uses current source JSON')
    classification=read(folder/'classification_metrics.json')
    audit.equal(data['classification'],classification,'four-class metrics use current source JSON')
    noise=read(folder/'binary_noise_results.json')
    audit.equal(data['noise']['rows'],noise,'noise metrics use current source JSON')
    metadata=read(folder/'binary_noise_metadata.json')
    audit.equal(metadata['classifier_sha256'],frozen['final_model_sha256'],'noise classifier is adopted refitted binary')
    audit.close(metadata['threshold'],frozen['points']['sp90_max_se']['threshold'],'noise uses exact frozen threshold')
    audit.close(data['noise']['threshold'],metadata['threshold'],'report noise threshold source')
    audit.equal(digest(folder/'models/denoiser_best.pt'),metadata['denoiser_sha256'],'noise uses new DnCNN')
    audit.equal(digest(folder/'noise_predictions.npz'),metadata['predictions_sha256'],'noise probability cache hash')
    auxiliary_audit=read(folder/'consistency_evaluation.json')
    auxiliary_frozen=read(folder/'aux_models_frozen.json')
    audit.equal(auxiliary_audit['status'],'complete','current auxiliary/noise suite completed')
    audit.equal(auxiliary_audit['binary_clean_ds2_inference_passes_added'],0,'current suite adds no clean binary inference')
    audit.equal(auxiliary_audit['noise_conditions'],9,'current suite has nine fixed noise conditions')
    audit.equal(auxiliary_audit['noise_methods_per_condition'],3,'current suite has three fixed methods')
    audit.check(datetime.fromisoformat(auxiliary_frozen['frozen_at_utc'])<=datetime.fromisoformat(auxiliary_audit['started_at_utc']),
                'auxiliary weights frozen before their evaluation')
    for name,sha in auxiliary_frozen['model_sha256'].items():
        audit.equal(digest(folder/'models'/name),sha,'new auxiliary frozen hash '+name)
    audit.equal(len(noise),9,'exactly nine noise conditions')
    with np.load(folder/'noise_predictions.npz',allow_pickle=False) as cached:
        y=np.asarray(cached['y']); indices=np.asarray(cached['global_index'])
        audit.equal(len(y),data['noise']['n'],'noise subset size matches report')
        audit.equal(len(np.unique(indices)),len(y),'noise subset indices unique')
        audit.equal(int(y.sum()),5446,'noise subset retains all DS2 abnormal beats')
        audit.equal(int((y==0).sum()),4000,'noise subset has 4000 normal beats')
        with np.load(root/'results/workpoint_cv_v2/ds2_predictions.npz',allow_pickle=False) as clean:
            mapped=np.searchsorted(clean['global_index'],indices)
            audit.equal(clean['global_index'][mapped].tolist(),indices.tolist(),'noise indices are frozen DS2 indices')
            audit.equal(clean['y'][mapped].tolist(),y.tolist(),'noise labels match frozen DS2 labels')
        conditions=set()
        for row in noise:
            kind=row['kind']; snr=int(row['snr_db']); conditions.add((kind,snr))
            audit.close(row['snr_raw'],snr,'noise input SNR '+kind+'/'+str(snr),tolerance=.005)
            for method in ('raw','classical','denoised'):
                key=f'{kind}_{snr}_{method}_prob'
                audit.check(key in cached,'noise cached probabilities '+key)
                p=cached[key]
                audit.equal(len(p),len(y),'noise probability length '+key)
                measured=cached_binary_metrics(y,p,metadata['threshold'])
                for name in ('accuracy','macro_f1','sensitivity','specificity'):
                    audit.close(row[method][name],measured[name],'noise independently recomputed '+key+'/'+name,tolerance=5.1e-5)
                audit.close(row[method]['auc'],binary_auc(y,p),'noise independently recomputed AUC '+key,tolerance=5.1e-5)
                for name in ('tp','fp','tn','fn'):
                    if name in row[method]:
                        audit.equal(row[method][name],measured[name],'noise confusion count '+key+'/'+name)
        audit.equal(conditions,{(k,s) for k in ('bw','em','ma') for s in (5,10,15)},'all nine predefined noise conditions')
    with np.load(folder/'classification_predictions.npz',allow_pickle=False) as cached:
        y=np.asarray(cached['y'],dtype=int)
        audit.equal(len(y),49705,'four-class clean test sample count')
        with np.load(root/'results/workpoint_cv_v2/ds2_predictions.npz',allow_pickle=False) as clean:
            audit.equal(cached['global_index'].tolist(),clean['global_index'].tolist(),'four-class uses same frozen DS2 indices')
            audit.equal((y>0).astype(int).tolist(),clean['y'].tolist(),'four-class label grouping matches frozen binary cache')
        for i,c in enumerate(CLASS_NAMES):
            audit.equal(int((y==i).sum()),KNOWN_DS2[c],'four-class test support '+c)
        for name in ('cnn','mlp'):
            pred=np.asarray(cached[name],dtype=int)
            audit.equal(len(pred),len(y),'four-class prediction count '+name)
            cm=np.bincount(4*y+pred,minlength=16).reshape(4,4)
            m=classification[name]
            audit.equal(cm.tolist(),m['confusion_matrix'],'four-class confusion matrix '+name)
            audit.close(float(np.trace(cm)/len(y)),m['accuracy'],'four-class accuracy '+name)
            f1=[]
            for i,c in enumerate(CLASS_NAMES):
                score=2*cm[i,i]/max(cm[i].sum()+cm[:,i].sum(),1)
                f1.append(score)
                audit.close(score,m['report'][c]['f1-score'],'four-class F1 '+name+'/'+c)
                audit.equal(int(m['report'][c]['support']),KNOWN_DS2[c],'four-class support report '+name+'/'+c)
            audit.close(float(np.mean(f1)),m['macro_f1'],'four-class Macro-F1 '+name)
    demo=read(folder/'demo_info.json')
    audit.equal(data['demo'],demo,'demo uses new source JSON')
    audit.equal(demo['classifier_parameters'],9188,'demo uses four-class architecture')
    audit.equal(demo['classifier_sha256'],digest(folder/'models/cnn_light_best.pt'),'demo uses new four-class checkpoint')
    audit.equal(demo['denoiser_sha256'],digest(folder/'models/denoiser_best.pt'),'demo uses new DnCNN checkpoint')
    audit.check('four' in demo['model_task'].lower() or '四分类' in demo['model_task'],'demo task explicitly four-class')
    audit.equal(demo['uses_rr_features'],False,'demo does not silently claim RR input')
    predicted=[CLASS_NAMES[int(p)] for p in demo['pred']]
    truth=demo['true_cls']
    audit.equal(len(predicted),len(truth),'demo true/predicted labels align')
    audit.equal(demo['mismatches'],sum(a!=b for a,b in zip(truth,predicted)),'demo mismatch count')
    audit.equal(demo['truth_prediction_pairs'],[{'true':t,'pred':p} for t,p in zip(truth,predicted)],'demo exact truth/prediction pairs')
    with np.load(folder/'demo_record208.npz',allow_pickle=True) as cached:
        audit.equal(cached['pred'].tolist(),demo['pred'],'demo image cache predictions')
        audit.equal(cached['true_cls'].tolist(),truth,'demo image cache truth')
    cost=read(folder/'model_cost.json')
    audit.equal(data['cost'],cost,'resource table uses newly measured JSON')
    audit.check(cost['threads']>0 and cost['batch']>0 and cost['repetitions']>0,'cost measurement configuration recorded')
    environment=read(folder/'runtime_environment.json')
    audit.equal(data['environment']['python'],environment['python'],'report Python environment source')
    audit.equal(data['environment']['packages'],environment['packages'],'report package environment source')
    den_training=read(folder/'denoiser_training.json')
    den_history=read(folder/'denoiser_history.json')
    audit.equal(data['denoiser']['training'],den_training,'denoiser training source')
    audit.equal(data['denoiser']['history'],den_history,'denoiser history source')
    # Loading own generated local checkpoints reads metadata and tensors only;
    # this function never calls a model's forward method.
    import torch
    sys.path.insert(0,str(root/'src'))
    from models import ECGCNNLight,DnCNN1d,MLPBaseline
    model_specs={
        9058:(root/'results/workpoint_cv_v2/final_binary.pt',ECGCNNLight(num_classes=2)),
        9188:(folder/'models/cnn_light_best.pt',ECGCNNLight()),
        41412:(folder/'models/mlp_best.pt',MLPBaseline()),
        46049:(folder/'models/denoiser_best.pt',DnCNN1d(channels=32,depth=6,kernel=11)),
    }
    for expected,(checkpoint,model) in model_specs.items():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model'])
        audit.equal(sum(p.numel() for p in model.parameters()),expected,'checkpoint parameter count '+str(expected))
        if expected!=9058:
            audit.equal(ck['train_records'],split['train_records'],'current model train records '+checkpoint.name)
            audit.equal(ck['val_records'],split['val_records'],'current model validation records '+checkpoint.name)
            audit.check('201' not in ck['train_records']+ck['val_records'],'201 excluded from model '+checkpoint.name)
            audit.equal(ck['aux_split_sha256'],digest(folder/'aux_split.json'),'current split hash '+checkpoint.name)
        if expected==46049:
            audit.equal(data['denoiser']['best_epoch'],ck['epoch'],'denoiser chosen epoch')
    for item in cost['models']:
        n=int(item['params'])
        audit.check(n in model_specs,'resource row identifies known architecture '+str(n))
        checkpoint,_=model_specs[n]
        audit.close(item['fp32_kib'],n*4/1024,'FP32 size arithmetic '+item['key'])
        audit.close(item['int8_kib'],n/1024,'INT8 size arithmetic '+item['key'])
        audit.check(float(item['latency_ms'])>0,'new measured latency positive '+item['key'])
        samples=item['latency_ms_samples']
        audit.equal(len(samples),cost['repetitions'],'latency repeat count '+item['key'])
        audit.check(all(x>0 for x in samples),'all latency samples positive '+item['key'])
        audit.close(float(np.median(samples)),item['latency_ms'],'latency median arithmetic '+item['key'])
        if 'checkpoint_sha256' in item:
            audit.equal(item['checkpoint_sha256'],digest(checkpoint),'timed checkpoint hash '+item['key'])
        elif 'model_sha256' in cost:
            audit.equal(cost['model_sha256'][item['key']],digest(checkpoint),'timed checkpoint hash '+item['key'])
        else:
            raise AssertionError('Cost provenance needs checkpoint_sha256 for '+item['key'])
    return classification,noise,demo,cost


def document_parts(path):
    with zipfile.ZipFile(path) as archive:
        blobs={n:archive.read(n) for n in archive.namelist()}
    document=ET.fromstring(blobs['word/document.xml'])
    body=document.find('w:body',NS)
    elements=list(body)
    tables=[]; captions=[]; previous=''
    for child in elements:
        if child.tag.endswith('}p'):
            if element_text(child).strip():
                previous=element_text(child)
        elif child.tag.endswith('}tbl'):
            matrix=[]
            for row in child.findall('w:tr',NS):
                matrix.append([element_text(cell) for cell in row.findall('w:tc',NS)])
            tables.append((child,matrix)); captions.append(previous)
    paragraphs=[element_text(p) for p in body.findall('w:p',NS)]
    return blobs,document,tables,captions,paragraphs


def verify_manifest(manifest,tables,captions,paragraphs,audit):
    declared=manifest['tables']
    audit.equal(len(tables),len(declared),'all DOCX tables declared')
    ids=[t['id'] for t in declared]
    audit.equal(len(ids),len(set(ids)),'table IDs unique')
    expected_ids={*[f'table{i}' for i in range(1,11)],'tableA1'}
    audit.equal(set(ids),expected_ids,'complete table set')
    by_id={}
    for index,((element,matrix),spec) in enumerate(zip(tables,declared)):
        expected=([spec['headers']] if spec['headers'] else [])+spec['rows']
        audit.equal(len(matrix),len(expected),'table row count '+spec['id'])
        for ri,(actual_row,expected_row) in enumerate(zip(matrix,expected)):
            audit.equal(len(actual_row),len(expected_row),f"table columns {spec['id']}/{ri}")
            for ci,(actual,exp) in enumerate(zip(actual_row,expected_row)):
                audit.equal(normal(actual),normal(exp),f"DOCX cell {spec['id']}/{ri}/{ci}")
        if spec['id']!='cover':
            audit.equal(normal(captions[index]),normal(spec['caption']),'table caption '+spec['id'])
        by_id[spec['id']]={'element':element,'matrix':matrix,'spec':spec}
    text='\n'.join(paragraphs)
    for item in manifest.get('key_text',[]):
        audit.check(normal(item['text']) in normal(text),'key paragraph rendered '+item['id'])
    return by_id


def numeric_cell(actual,expected,message,audit):
    audit.equal(compact(actual),compact(expected),message)


def verify_table_numbers(data,by_id,summary,clean_cache,audit):
    # Detailed table checks are independent of the builder's manifest strings.
    # They bind each numeric cell to the source experiment data.
    def check_matrix(table_id,expected):
        actual=by_id[table_id]['matrix'][1:]
        audit.equal(len(actual),len(expected),'independent source rows '+table_id)
        for ri,(row,expected_row) in enumerate(zip(actual,expected)):
            audit.equal(len(row),len(expected_row),'independent source columns '+table_id+'/'+str(ri))
            for ci,(value,source_value) in enumerate(zip(row,expected_row)):
                numeric_cell(value,source_value,f'independent source cell {table_id}/{ri}/{ci}',audit)

    def integer(n):
        return f'{int(n):,}'

    def percent(n):
        return f'{n*100:.2f}%'

    def number(n,digits=4):
        return f'{float(n):.{digits}f}'

    expected1=[]
    for g in data['dataset']['groups']:
        expected1.append([g['label'],str(g['n_records']),*[integer(g['counts'][c]) for c in CLASS_NAMES],integer(g['n'])])
    check_matrix('table1',expected1)
    # Architecture labels are presentation choices, but all operation/output
    # cells must equal the declared current model architecture. Its parameter
    # count and weight shapes were independently loaded above without forward.
    for table_id,model_key in [('table2','binary'),('table3','denoiser')]:
        check_matrix(table_id,data['architecture'][model_key])
    binary_arch=' '.join(str(v) for row in data['architecture']['binary'] for v in row)
    den_arch=' '.join(str(v) for row in data['architecture']['denoiser'] for v in row)
    for signature in ('1×256','16×128','32×64','64×32'):
        audit.check(signature in compact(binary_arch),'binary architecture output '+signature)
    for number_value in ('11','32','256'):
        audit.check(number_value in den_arch,'denoiser architecture dimension '+number_value)
    expected4=[]
    for key,label in [('sp90_max_se','OOF Sp≥90% 主点'),('youden_max','OOF Youden 点'),('default_050','默认点')]:
        m=summary['working_points'][key]
        expected4.append([label,number(m['threshold']),percent(m['accuracy']),number(m['macro_f1']),
                          percent(m['sensitivity']),percent(m['specificity']),number(m['f1_abnormal'])])
    m=summary['all_normal_baseline']
    expected4.append(['全判正常','—',percent(m['accuracy']),number(m['macro_f1']),percent(m['sensitivity']),percent(m['specificity']),number(m['f1_abnormal'])])
    check_matrix('table4',expected4)
    expected5=[]; expected_a2=[]
    noise={(r['kind'],int(r['snr_db'])):r for r in data['noise']['rows']}
    for kind,label in [('bw','基线漂移'),('em','电极运动'),('ma','肌电')]:
        for snr in (5,10,15):
            r=noise[kind,snr]
            expected5.append([f'{label}/{snr}',number(r['snr_raw'],1),number(r['snr_denoised'],1),
                              f"{r['snr_denoised']-r['snr_raw']:+.1f}",
                              number(r['raw']['macro_f1']),number(r['classical']['macro_f1']),number(r['denoised']['macro_f1'])])
            expected_a2.append([f'{label}/{snr}',number(r['rmse_raw']),number(r['rmse_denoised']),
                                number(r['raw']['auc']),number(r['denoised']['auc'])])
    check_matrix('table5',expected5); check_matrix('tableA1',expected_a2)
    expected6=[]
    for key,label in [('cnn','轻量 CNN'),('mlp','MLP')]:
        m=data['classification'][key]
        expected6.append([label,percent(m['accuracy']),number(m['macro_f1']),
                          *[number(m['report'][c]['f1-score']) for c in CLASS_NAMES]])
    check_matrix('table6',expected6)
    y,prob,record=clean_cache
    record_results={str(r['record']):r for r in data['per_record']}
    for r,m in record_results.items():
        mask=record==r; expected=cached_binary_metrics(y[mask],prob[mask],summary['working_points']['sp90_max_se']['threshold'])
        audit.equal(int(m['n']),expected['n'],'per-record total '+r)
        audit.equal(int(m['abnormal']),int(y[mask].sum()),'per-record abnormal '+r)
        for key in ('sensitivity','specificity'):
            audit.close(m[key],expected[key],'per-record recomputed '+r+'/'+key)
        audit.equal(int(m.get('fn',m.get('false_negative',-1))),expected['fn'],'per-record recomputed FN '+r)
    expected7=[]
    for r in ('232','200','213','233','210','222'):
        mask=record==r; m=cached_binary_metrics(y[mask],prob[mask],summary['working_points']['sp90_max_se']['threshold'])
        expected7.append([r,integer(m['n']),integer(int(y[mask].sum())),integer(m['fn']),percent(m['sensitivity']),percent(m['specificity'])])
    check_matrix('table7',expected7)
    expected8=[]
    for m in data['cost']['models']:
        expected8.append([m['name'],integer(m['params']),f"{m['fp32_kib']:.1f} KiB",
                          f"{m['int8_kib']:.1f} KiB",f"{m['latency_ms']:.3f} ms"])
    check_matrix('table8',expected8)
    resources=[
        ['MIT-BIH','https://physionet.org/content/mitdb/1.0.0/'],
        ['NSTDB','https://physionet.org/content/nstdb/1.0.0/'],
        ['OpenECG','https://github.com/jwlee95/OpenECG'],
        ['PyTorch','https://pytorch.org/'],
    ]
    check_matrix('table10',resources)


def verify_figure_values(root,data,summary,cache,audit):
    folder=root/'results/report_final'
    manifest=read(folder/'figure_manifest.json')
    audit.check(manifest['frozen_binary_unchanged'],'plotting preserves frozen binary artifacts')
    for path,sha in manifest['sources'].items():
        audit.equal(digest(project_file(root,path)),sha,'figure source hash '+Path(path).name)
    figures={f['filename']:f for f in manifest['figures']}
    audit.equal(len(figures),13,'thirteen current report/package figures')
    for name,f in figures.items():
        audit.equal(digest(folder/'figures'/name),f['sha256'],'current figure file hash '+name)
    accounting=figures['fig01_dataset_distribution.png']['annotation_values']
    audit.equal(accounting['group_n'],[49050,49705,1963,733],'figure 2 four-part counts')
    audit.equal(accounting['ds2_counts'],KNOWN_DS2,'figure 2 corrected DS2 class counts')
    audit.equal(accounting['total'],101451,'figure 2 total')
    binary=figures['fig06b_binary_roc.png']['annotation_values']
    audit.close(binary['ds2_auc'],summary['auc'],'figure 6 clean AUC')
    audit.close(binary['oof_auc'],data['oof']['oof_auc'],'figure 6 OOF AUC')
    audit.close(binary['primary_threshold'],summary['working_points']['sp90_max_se']['threshold'],'figure 6 frozen primary threshold')
    y,p,_=cache
    m=cached_binary_metrics(y,p,binary['primary_threshold'])
    audit.equal(binary['confusion_matrix'],[[m['tn'],m['fp']],[m['fn'],m['tp']]],'figure 6 cached confusion matrix')
    noise={(r['kind'],int(r['snr_db'])):r for r in data['noise']['rows']}
    deltas=figures['fig05b_denoising_snr.png']['annotation_values']
    noise_f1=figures['fig08_noise_robustness.png']['annotation_values']
    for (kind,snr),r in noise.items():
        key=f'{kind}_{snr}'
        audit.close(deltas[key],r['snr_denoised']-r['snr_raw'],'figure 8 reconstruction delta '+key)
        for method in ('raw','classical','denoised'):
            audit.close(noise_f1[key][method],r[method]['macro_f1'],'figure 9 current F1 '+key+'/'+method)
    four=figures['fig06_confusion_matrices.png']['annotation_values']
    for name in ('cnn','mlp'):
        for key in ('accuracy','macro_f1'):
            audit.close(four[name][key],data['classification'][name][key],'figure 10 '+name+'/'+key)
    demo=figures['fig11_openecg_demo.png']['annotation_values']
    audit.equal(demo['task'],'four_class','figure 11 task')
    audit.equal(demo['truth'],data['demo']['true_cls'],'figure 11 truth labels')
    predicted=[CLASS_NAMES[int(i)] for i in data['demo']['pred']]
    audit.equal(demo['predictions'],predicted,'figure 11 predicted labels')
    audit.equal(len(demo['mismatches']),data['demo']['mismatches'],'figure 11 mismatch count')
    audit.equal(demo['uses_rr'],False,'figure 11 does not claim RR input')


def cover_text(document):
    """Extract only cover text, without logging any personal field values."""
    result=[]
    for child in document.find('w:body',NS):
        text=element_text(child)
        if compact(text)=='摘要':
            return [normal(value) for value in result if normal(value)]
        result.append(text)
    raise AssertionError('Abstract boundary required to separate the cover')


def verify_cover(document,manifest,audit,cover_source=None):
    current=cover_text(document)
    joined='\n'.join(current)
    provenance=manifest.get('cover_provenance',{})
    if cover_source is not None:
        _,source,_,_,_=document_parts(cover_source)
        # A boolean assertion is deliberate: audit output must never expose
        # names, student numbers or supervisor values from either document.
        audit.check(current==cover_text(source),'cover text preserved from user-provided source')
        audit.equal(provenance.get('mode'),'preserved','cover preservation mode recorded')
        audit.equal(provenance.get('source_sha256'),digest(cover_source),'cover source file hash')
        return {'mode':'preserved','source_sha256':digest(cover_source)}
    audit.equal(provenance.get('mode'),'blank','public report uses blank cover mode')
    for label in ('姓名','学号','指导教师'):
        audit.equal(compact(joined).count(label),1,'one blank cover personal field '+label)
        matching=[text for text in current if label in compact(text)]
        audit.equal(len(matching),1,'blank cover field located '+label)
        value=re.sub(r'^.*?'+label+r'\s*[:：]?','',compact(matching[0]),count=1)
        audit.check(not value or re.fullmatch(r'[_＿]+',value) is not None,
                    'public personal field remains blank '+label)
    return {'mode':'blank'}


def verify_text_and_ooxml(root,data,manifest,blobs,document,by_id,paragraphs,audit):
    full_text='\n'.join(paragraphs+[' '.join(cell for row in v['matrix'] for cell in row) for v in by_id.values()])
    normalized=normal(full_text)
    for old in (r'(?<![\d.])93\.17(?!\d)',r'(?<![\d.])0\.763(?!\d)',r'(?<![\d.])0\.829(?!\d)'):
        audit.check(re.search(old,full_text) is None,'obsolete numeric token absent '+old)
    audit.check(re.search(r'0\.329\s*(?:→|->|到|提升到|提高到).*?0\.591',full_text) is None,'obsolete strong noise claim absent')
    for value in ('0.8558','0.7829','0.5879','72.27%','80.16%','79.29%','0.6533','89.04%','0.4710'):
        audit.check(value in normalized,'required adopted metric in report '+value)
    audit.check('准确率不是主指标' in compact(normalized) or '准确率不适合作为本任务的主指标' in compact(normalized),
                'accuracy is explicitly subordinate')
    audit.check('阴性' in normalized and '89%' in compact(normalized),'negative prevalence disclosed')
    audit.check('OOF' in normalized and ('Sp≥90%' in compact(normalized) or 'Sp≥0.90' in compact(normalized) or '特异性≥90%' in compact(normalized)),
                'OOF specificity selection constraint disclosed')
    audit.check('Youden' in normalized,'Youden reference rule disclosed')
    audit.check('探索性分析' in normalized,'DS2 full threshold curve explicitly exploratory')
    audit.check('工作点' in normalized and '分数分布' in normalized and '重拟合' in normalized,
                'operating-point transfer and model-refit score distribution are distinguished')
    audit.check('概率校准' in normalized and ('专门评价' in normalized or '专门检验' in normalized or '专门评估' in normalized),
                'probability calibration is identified as requiring a separate evaluation')
    for claim in ('概率校准不能直接跨记录迁移','概率校准无法直接跨记录迁移','概率校准失效'):
        audit.check(claim not in compact(normalized),'unsupported probability-calibration conclusion absent '+claim)
    audit.check('少' in normalized and 'OOF' in normalized,'OOF smaller training partitions discussed')
    audit.check('四分类' in normalized and '演示' in normalized,'four-class demo task disclosed')
    audit.check('未来样本' in normalized and '离线' in normalized,'zero-phase filtering offline limitation disclosed')
    audit.check('未做' in normalized or '未完成' in normalized,'experiment limitations present')
    audit.check('参考文献' in normalized,'reference section present')
    for i in range(1,7):
        audit.check(f'[{i}]' in normalized,'reference entry '+str(i))
    # Personal fields appear only on the cover. Values are independently checked
    # by verify_cover, either against the user-provided source or blank fields.
    if manifest.get('cover_provenance',{}).get('mode')=='blank':
        for word in ('姓名','学号','指导教师'):
            audit.equal(compact(full_text).count(word),1,'one cover personal field '+word)
    core=ET.fromstring(blobs['docProps/core.xml'])
    for tag in ('dc:creator','cp:lastModifiedBy'):
        text=core.findtext(tag,default='',namespaces=NS)
        audit.check('python-docx' not in text.lower(),'clean author property '+tag)
    for tag in ('dcterms:created','dcterms:modified'):
        date=core.findtext(tag,default='',namespaces=NS)
        audit.check(date and int(date[:4])>=2026,'current document date '+tag)
    audit.check(not any('comments' in name.lower() for name in blobs),'no reviewer comments shipped')
    audit.check(not document.findall('.//w:ins',NS) and not document.findall('.//w:del',NS),'no tracked changes shipped')
    styles=ET.fromstring(blobs['word/styles.xml'])
    title_style_ids=set()
    for style_name in ('Title','Subtitle'):
        style=next((node for node in styles.findall('w:style',NS)
                    if node.get('{'+NS['w']+'}styleId')==style_name or
                    (node.find('w:name',NS) is not None and
                     node.find('w:name',NS).get('{'+NS['w']+'}val','').lower()==style_name.lower())),None)
        audit.check(style is not None,'title style exists '+style_name)
        title_style_ids.add(style.get('{'+NS['w']+'}styleId'))
        audit.check(style.find('w:pPr/w:pBdr',NS) is None,'title style has no paragraph border '+style_name)
    for paragraph in document.findall('.//w:p',NS):
        style=paragraph.find('w:pPr/w:pStyle',NS)
        if style is not None and style.get('{'+NS['w']+'}val') in title_style_ids:
            audit.check(paragraph.find('w:pPr/w:pBdr',NS) is None,
                        'title paragraph has no direct border '+style.get('{'+NS['w']+'}val'))
    for name,spec in by_id.items():
        rows=spec['element'].findall('w:tr',NS)
        for i,row in enumerate(rows):
            audit.check(row.find('w:trPr/w:cantSplit',NS) is not None,f'row cannot split {name}/{i}')
    sections=document.findall('.//w:sectPr',NS)
    audit.check(bool(sections),'section properties exist')
    for i,section in enumerate(sections):
        size=section.find('w:pgSz',NS)
        audit.check(size is not None,'page size section '+str(i))
        audit.check(abs(int(size.get('{'+NS['w']+'}w'))-11906)<=2 and abs(int(size.get('{'+NS['w']+'}h'))-16838)<=2,
                    'A4 portrait section '+str(i))
    headers='\n'.join(element_text(ET.fromstring(b)) for n,b in blobs.items() if re.fullmatch(r'word/header\d+\.xml',n))
    footers='\n'.join(b.decode('utf-8') for n,b in blobs.items() if re.fullmatch(r'word/footer\d+\.xml',n))
    audit.check('海南大学' in headers and '生物医学传感检测与仪器' in headers,'header preserved')
    audit.check('PAGE' in footers and '第' in footers and '页' in footers,'page number field present')
    relationships=ET.fromstring(blobs['word/_rels/document.xml.rels'])
    targets={r.get('Id'):r.get('Target') for r in relationships}
    image_nodes=document.findall('.//a:blip',NS)
    audit.equal(len(image_nodes),len(manifest['figures']),'all inline figures declared')
    for node,figure in zip(image_nodes,manifest['figures']):
        rid=node.get('{'+NS['r']+'}embed'); target=targets[rid]
        package_name='word/'+target.lstrip('/') if not target.startswith('/') else target.lstrip('/')
        if package_name not in blobs:
            package_name=str(Path('word')/target).replace('\\','/')
        actual=blob_digest(blobs[package_name])
        supplement_images={'sil_afe.png':'results/sil/sil_afe.png','sil_roc.png':'results/sil/sil_roc.png',
                           'openecg_sim.png':'results/openecg_demo/openecg_sim.png'}
        image_path=supplement_images.get(figure['filename'],data['figures'].get(figure['filename']))
        audit.check(image_path is not None,'embedded image has an independent source mapping '+figure['filename'])
        file=project_file(root,image_path)
        audit.equal(actual,digest(file),'embedded image matches final file '+figure['filename'])
        audit.equal(actual,figure['sha256'],'embedded image matches manifest '+figure['filename'])


def verify_measurement_definitions_and_tail(document,by_id,paragraphs,audit):
    """Check the four editorial corrections independently of manifest prose."""
    start=next((i for i,t in enumerate(paragraphs) if compact(t).startswith('3.5评价指标')),None)
    stop=next((i for i,t in enumerate(paragraphs) if compact(t).startswith('3.6')),None)
    audit.check(start is not None and stop is not None and start<stop,'measurement definition section located')
    scope='\n'.join(paragraphs[start:stop]); reduced=compact(scope)
    body_children=list(document.find('w:body',NS))
    begin=next(i for i,node in enumerate(body_children) if compact(element_text(node)).startswith('3.5评价指标'))
    end=next(i for i,node in enumerate(body_children) if compact(element_text(node)).startswith('3.6'))
    formula_nodes=[math for node in body_children[begin:end] for math in node.findall('.//m:oMath',NS)]
    def has_snr_i(math):
        return any(element_text(node.find('m:e',NS))=='SNR' and element_text(node.find('m:sub',NS)) in ('i','ᵢ')
                   for node in math.findall('.//m:sSub',NS)
                   if node.find('m:e',NS) is not None and node.find('m:sub',NS) is not None)
    def has_time_sum(math):
        return any(element_text(node.find('m:sub',NS)) in ('t','t=1')
                   for node in math.findall('.//m:nary',NS) if node.find('m:sub',NS) is not None)
    native_definition=any(has_snr_i(math) and has_time_sum(math) for math in formula_nodes)
    plain_definition=any(re.search(r'SNR(?:ᵢ|_i)\s*=\s*10',p) and
                         any(token in p for token in ('Σₜ','Σ_t','∑ₜ','∑_t')) for p in paragraphs[start:stop])
    audit.check(native_definition or plain_definition,
                'SNR formula itself is indexed per beat and sums over sample t')
    audit.check('逐拍' in scope or '各拍' in scope,'per-beat SNR interpretation disclosed')
    audit.check('均值' in scope or '算术平均' in scope or '平均值' in scope,
                'reported SNR is the mean over beats')
    audit.check('RMSE' in scope and '全局' in scope and '均方根' in scope,
                'RMSE uses global root mean square rather than mean beat SNR')
    audit.check(('口径不同' in reduced or '口径并不相同' in reduced) and
                ('不能互相反推' in reduced or '不能相互反推' in reduced or '不可互相反推' in reduced),
                'SNR and RMSE cannot be inferred from one another under different aggregations')
    # Caption checks use actual DOCX paragraphs, not the builder's manifest.
    for number,caption in [
        (2,'图 2  训练/验证、测试、201 排除与协议未纳入的四段样本分布'),
        (6,'图 6  混淆矩阵、ROC 与 OOF 冻结工作点'),
    ]:
        matches=[p for p in paragraphs if re.match(r'^图\s*'+str(number)+r'(?!\d)',p.strip())]
        audit.equal(len(matches),1,'one actual figure caption '+str(number))
        audit.equal(compact(matches[0]),compact(caption),'exact separated figure caption '+str(number))
    children=list(document.find('w:body',NS));table=by_id['tableA1']['element']
    index=children.index(table)
    after=[element_text(node).strip() for node in children[index+1:] if node.tag==f"{{{NS['w']}}}p"]
    after=[text for text in after if text]
    audit.check(bool(after) and compact(after[0]).startswith('附录B代码与数据仓库'),
                'supplementary table A1 is followed by the repository appendix')
    earlier=[element_text(node) for node in children[:index] if node.tag==f"{{{NS['w']}}}p"]
    audit.check(any(compact(p).startswith('附录A补充结果') for p in earlier),
                'supplementary results appendix A exists')
    audit.check(any('表5' in compact(p) and 'SNR' in p and 'Macro-F1' in p and 'RMSE' in p for p in earlier),
                'appendix introduction describes both SNR and Macro-F1 in table 5')
    appendix='\n'.join(after)
    for token in ('https://github.com/HZH507/openecg-beat-screening','openecg_ext/','sil/','data/README.md'):
        audit.check(token in appendix,'repository appendix names source '+token)
    audit.check('部分' in by_id['table7']['spec']['caption'] or '代表性' in by_id['table7']['spec']['caption'],
                'table 7 is explicitly a partial per-record presentation')
    audit.check(any('22' in p and ('完整' in p or '全部' in p) and 'json' in p.lower() for p in paragraphs),
                'full per-record results are available beyond the six displayed examples')


def sil_alignment_pulse_checks(script):
    """Exercise only two extracted NumPy snippets, never import the SIL runner.

    The synthetic pulse proves the estimator's lag sign and the compensating
    shift direction without loading torch, a model, ECG data or a cache.
    """
    tree=ast.parse(Path(script).read_text(encoding='utf-8-sig'))
    estimator=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='estimate_lag')
    main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    branches=[node for node in ast.walk(main) if isinstance(node,ast.If) and
              isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and
              node.test.left.id=='lag' and isinstance(node.test.ops[0],ast.Gt)]
    if len(branches)!=1:
        raise AssertionError('SIL runner must expose one signed-lag compensation branch')
    namespace={'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[estimator],type_ignores=[])),
                 str(script),'exec'),namespace)
    branch=compile(ast.fix_missing_locations(ast.Module(body=[branches[0]],type_ignores=[])),
                   str(script),'exec')
    reference=np.zeros(4096,dtype=np.float64); reference[1000]=1.
    results=[]
    for physical_lag in (4,-3):
        test=np.zeros_like(reference); test[1000+physical_lag]=1.
        estimated=namespace['estimate_lag'](reference,test,max_lag=20,start=500,span=2000)
        namespace.update(lag=estimated,sil_sig=test)
        exec(branch,namespace)
        results.append({'physical_lag':physical_lag,'estimated_lag':estimated,
                        'reference_peak':1000,'compensated_peak':int(np.argmax(namespace['aligned']))})
    return results


def verify_sil(root,manifest,by_id,summary,paragraphs,audit):
    """Validate the side-loaded SIL supplement against its independent inputs."""
    metrics_path=root/'results/sil/sil_metrics.json'
    sil=read(metrics_path)
    source=manifest['sil_source']
    audit.equal(project_file(root,source['path']).resolve(),metrics_path.resolve(),
                'SIL manifest binds the canonical raw metrics JSON')
    audit.equal(source['sha256'],digest(metrics_path),'SIL metrics JSON revision hash')
    files=manifest['sil_source_files']
    expected_files={f'sil/{name}' for name in ('afe.py','adc.py','link.py','run_sil.py')}
    audit.equal({str(f['path']).replace('\\','/') for f in files},expected_files,
                'SIL manifest binds all four source files')
    for spec in files:
        audit.equal(spec['sha256'],digest(project_file(root,spec['path'])),
                    'SIL source revision hash '+spec['path'])
    predictions_path=project_file(root,sil['predictions_path'])
    audit.equal(predictions_path.resolve(),(root/'results/sil/sil_predictions.npz').resolve(),
                'SIL evaluation binds the canonical per-beat predictions')
    audit.equal(digest(predictions_path),sil['predictions_sha256'],'SIL per-beat prediction cache hash')
    audit.check(bool(sil.get('alignment_method')),'SIL compensation method recorded in raw results')
    with np.load(predictions_path,allow_pickle=False) as source_cache:
        cached={k:source_cache[k] for k in source_cache.files}
    for name in ('y','prob_ideal','prob_quant_link','prob_sil','prob_aligned','record','sample'):
        audit.check(name in cached,'SIL prediction cache contains '+name)
        audit.equal(cached[name].shape,(49705,),'SIL cache column size '+name)
    with np.load(root/'results/workpoint_cv_v2/ds2_predictions.npz',allow_pickle=False) as frozen_cache:
        audit.equal(cached['y'].tolist(),frozen_cache['y'].tolist(),'SIL labels equal frozen cache in beat order')
        audit.equal(cached['record'].astype(str).tolist(),frozen_cache['record'].astype(str).tolist(),
                    'SIL record identities equal frozen cache in beat order')
        ideal_deviation=float(np.max(np.abs(cached['prob_ideal']-frozen_cache['prob'])))
    audit.close(ideal_deviation,sil['ideal_matches_frozen_cache_max_dev'],
                'SIL ideal cache deviation independently recomputed per beat')
    for record in np.unique(cached['record']):
        samples=cached['sample'][cached['record']==record]
        audit.check(np.all(np.diff(samples)>0),'SIL source samples increase within record '+str(record))
    audit.equal(sil['n_beats'],49705,'SIL covers every frozen DS2 beat')
    audit.equal(sil['n_abnormal'],5446,'SIL abnormal total')
    primary=summary['working_points']['sp90_max_se']
    audit.close(sil['threshold'],primary['threshold'],'SIL uses unchanged full-precision frozen threshold')
    for key in ('auc','accuracy','macro_f1','sensitivity','specificity','tp','fp','tn','fn'):
        expected=summary['auc'] if key=='auc' else primary[key]
        audit.close(sil['ideal'][key],expected,'SIL ideal equals frozen clean result '+key)
    max_dev=sil['ideal_matches_frozen_cache_max_dev']
    audit.check(max_dev is not None and 0<=max_dev<=1.3e-7,'SIL ideal per-beat cache match within float32 rounding')
    rows=by_id['table9']['matrix'][1:]
    audit.equal(len(rows),4,'four SIL routes in table 9')
    for row,(key,label) in zip(rows,[('ideal','理想'),('quant_link','仅量化+链路'),
                                    ('sil','设备链路'),('sil_aligned','设备链路+时延对齐')]):
        route=sil[key]
        probabilities=cached[{'ideal':'prob_ideal','quant_link':'prob_quant_link',
                              'sil':'prob_sil','sil_aligned':'prob_aligned'}[key]]
        audit.check(np.all(np.isfinite(probabilities)) and np.all((probabilities>=0)&(probabilities<=1)),
                    'SIL cached probabilities are finite and bounded '+key)
        distribution={'mean':float(np.mean(probabilities,dtype=np.float64)),
                      'median':float(np.median(probabilities)),
                      'predicted_positive_rate':float(np.mean(probabilities>=sil['threshold']))}
        for statistic,value in distribution.items():
            audit.close(sil['probability_summary'][key][statistic],value,
                        'SIL probability distribution independently recomputed '+key+'/'+statistic)
        measured=cached_binary_metrics(cached['y'],probabilities,sil['threshold'])
        audit.close(route['auc'],binary_auc(cached['y'],probabilities),'SIL AUC independently recomputed '+key)
        for metric in ('accuracy','macro_f1','sensitivity','specificity','tp','fp','tn','fn'):
            audit.close(route[metric],measured[metric],'SIL cached prediction metric '+key+'/'+metric)
        audit.check(label in compact(row[0]),'SIL route label '+key)
        audit.equal(len(row),6,'SIL table columns '+key)
        expected=[f"{route['auc']:.4f}",f"{route['accuracy']*100:.2f}%",f"{route['macro_f1']:.4f}",
                  f"{route['sensitivity']*100:.2f}%",f"{route['specificity']*100:.2f}%"]
        for i,(value,source_value) in enumerate(zip(row[1:],expected)):
            numeric_cell(value,source_value,f'independent SIL table cell {key}/{i}',audit)
        tp,fp,tn,fn=(route[k] for k in ('tp','fp','tn','fn'))
        audit.equal(tp+fp+tn+fn,sil['n_beats'],'SIL confusion total '+key)
        audit.equal(tp+fn,sil['n_abnormal'],'SIL positive denominator '+key)
        calculated={'accuracy':(tp+tn)/sil['n_beats'],'sensitivity':tp/(tp+fn),
                    'specificity':tn/(tn+fp),'macro_f1':(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))/2}
        for metric,value in calculated.items():
            audit.close(route[metric],value,'SIL confusion-derived metric '+key+'/'+metric)
    audit.close(sil['delta_auc_quant_link'],sil['quant_link']['auc']-sil['ideal']['auc'],
                'SIL quantization/transport ablation AUC delta')
    audit.close(sil['delta_auc'],sil['sil']['auc']-sil['ideal']['auc'],'SIL raw AUC delta')
    audit.close(sil['delta_auc_aligned'],sil['sil_aligned']['auc']-sil['ideal']['auc'],'SIL aligned AUC delta')
    ideal_prob=cached['prob_ideal']
    for route,suffix in [('sil',''),('sil_aligned','_aligned')]:
        prob=cached['prob_sil' if route=='sil' else 'prob_aligned']
        audit.close(sil['mean_abs_prob_diff'+suffix],float(np.mean(np.abs(prob-ideal_prob))),
                    'SIL independently recomputed mean absolute probability shift '+route)
        flip=int(np.sum((prob>=sil['threshold'])!=(ideal_prob>=sil['threshold'])))
        audit.equal(sil['n_prediction_flip'+suffix],flip,'SIL independently recomputed prediction flips '+route)
    audit.close(sil['max_abs_prob_diff'],float(np.max(np.abs(cached['prob_sil']-ideal_prob))),
                'SIL independently recomputed maximum raw probability shift')
    audit.equal(len(sil['lags']),22,'SIL delay estimate for every DS2 record')
    audit.close(sil['mean_lag_samples'],sum(sil['lags'])/22,'SIL mean lag from record lags')
    audit.equal(sil['lag_ms'],round(1000*sil['mean_lag_samples']/sil['afe']['fs'],2),'SIL delay milliseconds')
    for pulse in sil_alignment_pulse_checks(root/'sil/run_sil.py'):
        audit.equal(pulse['estimated_lag'],pulse['physical_lag'],'SIL synthetic lag sign '+str(pulse['physical_lag']))
        audit.equal(pulse['compensated_peak'],pulse['reference_peak'],
                    'SIL compensation removes rather than doubles synthetic lag '+str(pulse['physical_lag']))
    audit.equal(sil['afe']['hp_hz'],.5,'SIL AFE high-pass frequency')
    audit.equal(sil['afe']['lp_hz'],40.,'SIL AFE low-pass frequency')
    audit.equal(sil['afe']['order'],4,'SIL AFE low-pass order')
    audit.equal(sil['afe']['fs'],360,'SIL sample frequency')
    audit.equal(sil['adc']['bits'],11,'SIL ADC bits')
    audit.equal((sil['adc']['vmin'],sil['adc']['vmax']),(-5.,5.),'SIL ADC range in mV')
    audit.close(sil['adc']['lsb_mV'],10/(2**11-1),'SIL ADC LSB step')
    audit.close(sil['link']['loss_rate'],.01,'SIL configured packet loss')
    # Read literals and call syntax only; never import any experiment module.
    def module_constants(path):
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        values={}
        for node in tree.body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                try:values[node.targets[0].id]=ast.literal_eval(node.value)
                except (ValueError,TypeError):pass
        return tree,values
    afe_tree,afe_constants=module_constants(root/'sil/afe.py')
    for name,value in [('FS',360),('HP_HZ',.5),('LP_HZ',40.),('ORDER',4)]:
        audit.equal(afe_constants[name],value,'SIL AFE source constant '+name)
    apply=next(n for n in afe_tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply')
    calls=[n.func.id for n in ast.walk(apply) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    audit.equal(calls.count('sosfilt'),2,'SIL AFE applies two causal SOS stages')
    audit.check('sosfiltfilt' not in calls and 'filtfilt' not in calls,'SIL AFE does not use zero-phase filtering')
    adc_tree,adc_constants=module_constants(root/'sil/adc.py')
    for name,value in [('BITS',11),('V_MIN',-5.),('V_MAX',5.)]:
        audit.equal(adc_constants[name],value,'SIL ADC source constant '+name)
    def defaults(tree,function):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
        values={}
        for arg,value in zip(fn.args.args[-len(fn.args.defaults):],fn.args.defaults):
            try:values[arg.arg]=ast.literal_eval(value)
            except (ValueError,TypeError):pass
        return values
    audit.equal(defaults(adc_tree,'quantize')['dither_lsb'],.5,'SIL source uses half-LSB dither')
    link_tree,_=module_constants(root/'sil/link.py')
    link_defaults=defaults(link_tree,'transmit')
    for name,value in [('samples_per_packet',20),('loss_rate',.01),('latency_s',.02)]:
        audit.equal(link_defaults[name],value,'SIL transport source default '+name)
    runner_tree=ast.parse((root/'sil/run_sil.py').read_text(encoding='utf-8-sig'))
    runner_main=next(n for n in runner_tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    assignments={}
    for node in ast.walk(runner_main):
        if isinstance(node,ast.Assign):
            targets=[]
            for target in node.targets:
                targets.extend(target.elts if isinstance(target,ast.Tuple) else [target])
            for target in targets:
                if isinstance(target,ast.Name):assignments[target.id]=node.value
    q_input=assignments['quant_q']
    audit.check(isinstance(q_input,ast.Call) and isinstance(q_input.func,ast.Attribute)
                and isinstance(q_input.func.value,ast.Name) and q_input.func.value.id=='adc'
                and q_input.func.attr=='quantize' and isinstance(q_input.args[0],ast.Name)
                and q_input.args[0].id=='sig','SIL component ablation quantizes the original signal without AFE')
    q_transport=assignments['stream_q']
    audit.check(isinstance(q_transport,ast.Call) and isinstance(q_transport.func,ast.Attribute)
                and isinstance(q_transport.func.value,ast.Name) and q_transport.func.value.id=='link'
                and q_transport.func.attr=='transmit' and isinstance(q_transport.args[0],ast.Name)
                and q_transport.args[0].id=='quant_q','SIL component ablation transmits the quantized original signal')
    q_beats=assignments['beats_q']
    audit.check(isinstance(q_beats,ast.Call) and isinstance(q_beats.func,ast.Name)
                and q_beats.func.id=='build_beats' and isinstance(q_beats.args[0],ast.Call)
                and isinstance(q_beats.args[0].func,ast.Name) and q_beats.args[0].func.id=='bandpass_filter'
                and isinstance(q_beats.args[0].args[0],ast.Name) and q_beats.args[0].args[0].id=='stream_q',
                'SIL component ablation retains the existing digital preprocessing and beat builder')
    start=next((i for i,p in enumerate(paragraphs) if compact(p).startswith('4.6硬件链路仿真')),None)
    stop=next((i for i,p in enumerate(paragraphs) if start is not None and i>start and compact(p).startswith('4.7')),None)
    audit.check(start is not None and stop is not None,'SIL subsection 4.6 located before conclusions')
    scope='\n'.join(paragraphs[start:stop]); reduced=compact(scope)
    for token in ('软件在环','AFE','0.5','40','四阶','因果','11bit','±5mV','±0.5LSB',
                  '4.89','20采样/包','1%','20ms','重组','数字预处理','冻结','0.5879','49705'):
        audit.check(compact(token) in reduced,'SIL method disclosed '+token)
    audit.check(f'{max_dev:.1e}' in reduced or '1.2×10⁻⁷' in reduced or '1.2×10−7' in reduced,
                'SIL ideal per-beat cache deviation disclosed')
    for token in ('360Hz','数字记录','采样前混叠','设定值','实物联调','工作点','校准'):
        audit.check(compact(token) in reduced,'SIL boundary/calibration disclosed '+token)
    audit.check('未模拟' in reduced,'SIL no pre-sampling aliasing disclosed')
    audit.check('不是实物联调' in reduced or '不等同于实物联调' in reduced or '不能替代实物联调' in reduced,
                'SIL is explicitly distinguished from physical integration')
    # Conclusions must follow the JSON rather than the previous runner's values.
    for key,delta in [('quant_link',sil['delta_auc_quant_link']),
                      ('sil',sil['delta_auc']),('sil_aligned',sil['delta_auc_aligned'])]:
        audit.check(f'{delta:+.4f}'.replace('-','−') in reduced.replace('-','−'),
                    'SIL AUC delta is stated from current metrics '+key)
    se_delta=100*(sil['sil']['sensitivity']-sil['ideal']['sensitivity'])
    audit.check(f'{se_delta:+.1f}'.replace('-','−') in reduced.replace('-','−'),
                'SIL unaligned sensitivity delta is stated from current metrics')
    audit.check(f"{sil['lag_ms']:.1f}" in reduced,'SIL phase delay magnitude disclosed')
    audit.check('时间戳' in reduced or '检测器' in reduced,'SIL phase-delay compensation mechanism disclosed')
    audit.check('重新校准' in reduced or ('重新选择' in reduced and '独立验证' in reduced),
                'SIL workpoint selection and independent validation are disclosed future needs')
    audit.check('四通路' in compact(by_id['table9']['spec']['caption']),'SIL table 9 caption names four routes')
    for route in ('ideal','quant_link','sil','sil_aligned'):
        audit.check(f"{sil[route]['auc']:.4f}" in reduced,
                    'SIL component ablation paragraph states AUC '+route)
    for route in ('ideal','quant_link','sil'):
        audit.check(f"{sil[route]['sensitivity']*100:.2f}%" in reduced,
                    'SIL component ablation paragraph states sensitivity '+route)
    for token in ('组件消融','11bit','1%','AFE','可能','幅频','相频','波形'):
        audit.check(compact(token) in reduced,'SIL component ablation interpretation disclosed '+token)
    audit.check(any(token in reduced for token in ('尚未分别','未分别','不能分别确定','未单独隔离','尚未单独隔离')),
                'SIL interpretation acknowledges that individual AFE mechanisms were not isolated')
    for claim in ('双重高频衰减是主要来源','损失主要来自AFE的串联低通','串联低通是主要损失来源'):
        audit.check(claim not in reduced,'unisolated low-pass causal claim absent '+claim)
    audit.check('尚未进行AFE、量化与丢包的组件消融' not in reduced,
                'obsolete claim of no SIL component ablation removed')
    for route in ('ideal','sil_aligned'):
        audit.check(f"{sil['probability_summary'][route]['predicted_positive_rate']*100:.2f}%" in reduced,
                    'SIL positive-prediction-rate transition is disclosed '+route)
    for metric in ('sensitivity','specificity'):
        for route in ('ideal','sil_aligned'):
            audit.check(f"{sil[route][metric]*100:.2f}%" in reduced,
                        'SIL aligned workpoint transition is disclosed '+route+'/'+metric)
    if sil['delta_auc_aligned']<-.005:
        audit.check(any(word in reduced for word in ('未恢复','未回到','没有恢复','没有回到')),
                    'SIL conclusion states that time shifting did not restore ranking')
        for phrase in ('几乎无影响','未使整体排序能力出现明显下降','对排序能力几乎无影响'):
            audit.check(phrase not in reduced,'unsupported previous SIL ranking conclusion absent '+phrase)
    aligned_mean=float(np.mean(cached['prob_aligned'],dtype=np.float64))
    ideal_mean=float(np.mean(cached['prob_ideal'],dtype=np.float64))
    if aligned_mean>=ideal_mean:
        audit.check('概率整体下移' not in reduced,'SIL conclusion does not claim downward shift when mean increased')
    abstract_end=next((i for i,p in enumerate(paragraphs) if compact(p).startswith('关键词')),None)
    abstract='\n'.join(paragraphs[:abstract_end]) if abstract_end is not None else ''
    audit.check(abstract_end is not None and ('SIL' in abstract or '软件在环' in abstract),
                'abstract includes SIL result sentence')
    audit.check('组件消融' in compact(abstract) and 'AFE' in abstract,
                'abstract includes the whole-AFE component-ablation observation')
    audit.check('损失主要来自AFE的串联低通' not in compact(abstract),
                'abstract avoids an unisolated low-pass causal conclusion')
    conclusion_end=next((i for i,p in enumerate(paragraphs) if i>stop and compact(p).startswith('参考文献')),len(paragraphs))
    conclusion='\n'.join(paragraphs[stop:conclusion_end])
    audit.check('SIL' in conclusion or '软件在环' in conclusion,'conclusion includes SIL result sentence')
    images={f['id']:f for f in manifest['figures']}
    for fid,name in [('figure12','sil_afe.png'),('figure13','sil_roc.png')]:
        audit.equal(images[fid]['filename'],name,'SIL embedded figure numbering '+fid)
        audit.equal(images[fid]['sha256'],digest(root/'results/sil'/name),'SIL current source image hash '+fid)
    audit.check('四通路' in compact(images['figure13']['caption']),'SIL figure 13 caption names four routes')
    audit.equal(len(manifest['figures']),14,'eleven original plus two SIL and one synthetic-input figure')
    return {'metrics_sha256':digest(metrics_path),'threshold':sil['threshold'],
            'n_beats':sil['n_beats'],'ideal_cache_max_deviation':max_dev,
            'synthetic_lag_compensation_verified':True}


def poppler_executable(name,explicit=None):
    if explicit:
        result=Path(explicit)
    else:
        bundled=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'/f'{name}.exe'
        found=shutil.which(name)
        result=bundled if bundled.is_file() else (Path(found) if found else None)
    if result is None or not result.is_file():
        raise RuntimeError(f'PDF validation requires Poppler {name}; pass its executable path explicitly')
    return result


def verify_synthetic_demo(root,manifest,summary,paragraphs,audit):
    """Check the existing synthetic-input result, without generating predictions."""
    path=root/'results/openecg_demo/openecg_demo.json'
    demo=read(path)
    source=manifest['openecg_source']
    audit.equal(project_file(root,source['path']).resolve(),path.resolve(),'synthetic demo source mapping')
    audit.equal(source['sha256'],digest(path),'synthetic demo JSON hash')
    audit.equal(demo['n_peaks'],10,'synthetic demo detected peaks')
    audit.equal(demo['n_beats'],10,'synthetic demo segmented beats')
    audit.equal(len(demo['prob_abnormal']),demo['n_beats'],'synthetic demo cached probability count')
    audit.equal(len(demo['pred']),demo['n_beats'],'synthetic demo cached prediction count')
    audit.close(demo['threshold'],summary['working_points']['sp90_max_se']['threshold'],
                'synthetic demo retains full frozen threshold')
    p=np.asarray(demo['prob_abnormal'],dtype=float)
    audit.check(np.all(np.isfinite(p)) and np.all((p>=0)&(p<=1)),'synthetic demo cached scores bounded')
    audit.check(np.all(np.abs(p-demo['threshold'])>.00005),
                'rounded synthetic cached probabilities are unambiguous at the frozen threshold')
    predicted=(p>=demo['threshold']).astype(int).tolist()
    audit.equal(demo['pred'],predicted,'synthetic cached classes agree with frozen threshold')
    audit.equal(sum(predicted),demo['n_abnormal'],'synthetic cached abnormal count')
    audit.equal(demo['n_abnormal'],0,'synthetic demo reports zero abnormal detections')
    audit.close(demo['heart_rate_bpm_measured'],60/demo['mean_rr_s'],'synthetic demo RR-derived heart rate')
    audit.equal(demo['fs_sim_hz'],1000,'synthetic input sampling rate')
    audit.equal(demo['fs_model_hz'],360,'synthetic resampled rate')
    start=next((i for i,p in enumerate(paragraphs) if compact(p).startswith('4.7')),None)
    stop=next((i for i,p in enumerate(paragraphs) if start is not None and i>start and compact(p).startswith('5讨论')),None)
    audit.check(start is not None and stop is not None,'synthetic-input section is present')
    scope=compact('\n'.join(paragraphs[start:stop]))
    for token in ('自定义','合成','NeuroKit2','ecg_process','滤波','R峰','参数','性能'):
        audit.check(token in scope,'synthetic-input scope disclosed '+token)
    audit.check('修改' in scope or '调整' in scope,'synthetic waveform parameter changes disclosed')
    audit.check('不是' in scope or '不构成' in scope or '未直接' in scope,
                'synthetic demo is explicitly limited to its actual integration scope')
    for claim in ('没有滤波、R峰检测','可直接消费上游的模拟信号与时序','复刻其高斯'):
        audit.check(claim not in scope,'obsolete upstream-equivalence claim absent '+claim)
    audit.check(f"{min(p):.2f}" in scope and f"{max(p):.2f}" in scope,
                'synthetic reported score range agrees with cached values')
    images={f['id']:f for f in manifest['figures']}
    figure=images['figure14']
    audit.equal(figure['filename'],'openecg_sim.png','synthetic-input figure 14 mapping')
    audit.equal(figure['sha256'],digest(root/'results/openecg_demo/openecg_sim.png'),
                'synthetic-input figure 14 current image hash')
    # The plotted beats are pre-standardization waveform slices. The classifier
    # internally standardizes a separate array, so the plot cannot say z-score.
    plotting=ast.parse((root/'openecg_ext/analysis/run_demo.py').read_text(encoding='utf-8-sig'))
    for node in ast.walk(plotting):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('set_ylabel','set'):
            labels=[arg.value for arg in node.args if isinstance(arg,ast.Constant) and isinstance(arg.value,str)]
            labels += [arg.value.value for arg in node.keywords if arg.arg=='ylabel'
                       and isinstance(arg.value,ast.Constant) and isinstance(arg.value.value,str)]
            audit.check(not any(value.strip().lower()=='z-score' for value in labels),
                        'unstandardized synthetic waveform is not labelled z-score')
    return {'metrics_sha256':digest(path),'n_beats':demo['n_beats'],'n_abnormal':demo['n_abnormal'],
            'prediction_inference_performed':False}


def verify_repository_hyperlink(blobs,document,audit):
    """Require a real external OOXML link in appendix B, not URL-looking text."""
    relationships=ET.fromstring(blobs['word/_rels/document.xml.rels'])
    by_id={node.attrib['Id']:node.attrib for node in relationships}
    appendix=False
    found=[]
    for node in document.find('w:body',NS):
        if compact(element_text(node)).startswith('附录B'):
            appendix=True
        if not appendix:
            continue
        for link in node.findall('.//w:hyperlink',NS):
            rel=by_id.get(link.attrib.get(f"{{{NS['r']}}}id"),{})
            if rel.get('Target')==REPOSITORY_URL:
                audit.equal(rel.get('TargetMode'),'External','repository hyperlink relationship is external')
                audit.equal(rel.get('Type'),NS['r']+'/hyperlink','repository hyperlink relationship type')
                audit.equal(element_text(link),REPOSITORY_URL,'repository hyperlink display text matches destination')
                found.append(link)
    audit.equal(len(found),1,'appendix B contains one genuine repository hyperlink')
    return {'target':REPOSITORY_URL,'external_hyperlinks_in_appendix_B':len(found)}


def pdf_uri_links(page):
    links=[]
    for ref in page.get('/Annots',[]):
        annotation=ref.get_object()
        action=annotation.get('/A')
        action=action.get_object() if action is not None else {}
        if annotation.get('/Subtype')=='/Link' and action.get('/S')=='/URI':
            links.append({'uri':str(action.get('/URI','')),
                          'rect':[float(v) for v in annotation.get('/Rect',[])]})
    return links


def extract_pdf_text(pdf,pdf_python=None):
    """Read actual PDF content via pypdf, using the bundled runtime if needed."""
    if pdf_python is None:
        try:
            from pypdf import PdfReader
        except ImportError:
            pass
        else:
            reader=PdfReader(pdf)
            texts=[page.extract_text() or '' for page in reader.pages]
            return {'pages':len(texts),'last':texts[-1],'text':'\n'.join(texts),
                    'last_page_uri_links':pdf_uri_links(reader.pages[-1])}
    executable=Path(pdf_python) if pdf_python else Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    if not executable.is_file():
        raise RuntimeError('PDF validation requires pypdf or a --pdf-python runtime containing pypdf')
    code=("import json,sys\nfrom pypdf import PdfReader\n"
          "r=PdfReader(sys.argv[1]);t=[p.extract_text() or '' for p in r.pages];links=[]\n"
          "for ref in r.pages[-1].get('/Annots',[]):\n"
          " a=ref.get_object();action=a.get('/A');action=action.get_object() if action is not None else {}\n"
          " if a.get('/Subtype')=='/Link' and action.get('/S')=='/URI':\n"
          "  links.append({'uri':str(action.get('/URI','')),'rect':[float(v) for v in a.get('/Rect',[])]})\n"
          "print(json.dumps({'pages':len(t),'last':t[-1],'text':'\\n'.join(t),'last_page_uri_links':links},ensure_ascii=False))")
    output=subprocess.check_output([str(executable),'-c',code,str(pdf)],env={**os.environ,'PYTHONUTF8':'1'})
    return json.loads(output.decode('utf-8'))


def verify_pdf_pair(root,docx,audit,*,required=False,pdf=None,record_path=None,pdfinfo=None,pdf_python=None):
    """Bind the exported PDF to the current DOCX, then inspect actual PDF text."""
    folder=root/'results/report_final'
    record_path=record_path or folder/'pdf_export_verification.json'
    pdf=pdf or root/'report/ECG_课程报告_标准版.pdf'
    if not record_path.exists():
        audit.check(not required,'PDF export evidence exists when required')
        return None
    record=read(record_path)
    current=(record.get('status')=='pass' and record.get('source_docx_sha256')==digest(docx) and
             pdf.is_file() and record.get('pdf_sha256')==digest(pdf))
    if not current and not required:
        return None  # DOCX-only pre-export validation; never call an old PDF current.
    audit.check(current,'PDF evidence and PDF both match the current DOCX revision')
    audit.equal(record['renderer'],'Microsoft Word','final PDF exporter is Word')
    layout=read(folder/'word_layout_diagnostics.json')
    audit.equal(layout['source_sha256'].lower(),digest(docx),'Word layout evidence belongs to current DOCX')
    info_exe=poppler_executable('pdfinfo',pdfinfo)
    info=subprocess.check_output([str(info_exe),str(pdf)]).decode('utf-8',errors='replace')
    page_match=re.search(r'^Pages:\s*(\d+)',info,re.M)
    audit.check(page_match is not None,'actual PDF page count readable by independent Poppler')
    pages=int(page_match.group(1))
    audit.check(pages>=2,'final PDF has a cover and substantive report pages')
    audit.equal(pages,record['pdf_page_count'],'actual PDF pages equal export record')
    audit.equal(pages,record['word_page_count'],'PDF and Word page counts match')
    audit.equal(pages,layout['page_count'],'PDF pages equal read-only Word layout diagnostics')
    actual=extract_pdf_text(pdf,pdf_python)
    audit.equal(actual['pages'],pages,'pypdf and Poppler agree on actual PDF page count')
    repository_links=[link for link in actual['last_page_uri_links']
                      if link['uri']==REPOSITORY_URL]
    audit.check(bool(repository_links),'last PDF page has a real URI link to the repository')
    for link in repository_links:
        rect=link['rect']
        audit.check(len(rect)==4 and rect[2]>rect[0] and rect[3]>rect[1],
                    'repository PDF link has a nonempty clickable rectangle')
    last=actual['last'];all_text=actual['text']
    audit.equal(normal(last),normal(record['pdf_last_page_text']),'actual PDF last page matches recorded content')
    audit.equal(normal(record['word_last_page_text']),normal(layout['pages'][-1]['text']),
                'recorded Word last page matches Word diagnostic content')
    audit.check('表A1' in compact(last),'last PDF page contains supplementary table A1 rather than only a closing note')
    audit.check('肌电/15' in compact(last),'last PDF page contains the last table A1 condition')
    for condition in ('基线漂移/5','基线漂移/10','基线漂移/15','电极运动/5','电极运动/10',
                      '电极运动/15','肌电/5','肌电/10','肌电/15'):
        audit.check(condition in compact(last),'complete table A1 remains together on final PDF page '+condition)
    audit.check(len(compact(last))>=150,'last PDF page has substantive table content')
    audit.check('表A1' in compact(record['word_last_page_text']),'last Word page contains table A1')
    audit.check('附录B' in compact(last) and '代码与数据仓库' in compact(last),
                'last PDF page contains the substantive repository appendix')
    audit.check(not any(p.get('possibly_blank',False) for p in layout['pages']),
                'Word diagnostics show no blank SIL-expanded pages')
    for old in (r'(?<![\d.])93\.17(?!\d)',r'(?<![\d.])0\.763(?!\d)',r'(?<![\d.])0\.829(?!\d)'):
        audit.check(re.search(old,all_text) is None,'obsolete metric absent from actual final PDF '+old)
    for value in ('0.8558','0.7829','0.5879','72.27%','80.16%','79.29%','0.6533'):
        audit.check(value in compact(all_text),'adopted metric present in actual final PDF '+value)
    audit.check('4.6硬件链路仿真' in compact(all_text),'SIL subsection is present in actual final PDF')
    sil=read(root/'results/sil/sil_metrics.json')
    for route in ('ideal','quant_link','sil','sil_aligned'):
        for metric in ('auc','macro_f1'):
            audit.check(f"{sil[route][metric]:.4f}" in compact(all_text),'SIL metric present in actual final PDF '+route+'/'+metric)
    report_pdfs=sorted(p.resolve() for p in (root/'report').glob('*.pdf'))
    audit.equal(report_pdfs,[pdf.resolve()],'report submission directory contains only the current PDF')
    return {'pdf':str(pdf),'pdf_sha256':digest(pdf),'pages':pages,
            'source_docx_sha256':digest(docx),'export_record_sha256':digest(record_path),
            'last_page_contains_table_A1':True,'repository_hyperlink':REPOSITORY_URL,
            'repository_link_annotation_count':len(repository_links),
            'independent_pdf_reader':'Poppler page count plus pypdf text and link annotations'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--docx',type=Path)
    parser.add_argument('--cover-source',type=Path,help='User-provided original DOCX whose cover must be preserved; omitted for blank public covers')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--require-pdf',action='store_true',help='Require a current Word-exported PDF and matching page/content evidence')
    parser.add_argument('--pdf',type=Path)
    parser.add_argument('--pdf-record',type=Path)
    parser.add_argument('--pdfinfo',type=Path)
    parser.add_argument('--pdf-python',type=Path,help='Python runtime containing pypdf, if the verifier runtime lacks it')
    args=parser.parse_args(); root=args.root.resolve()
    docx=args.docx or root/'report/ECG_课程报告_标准版.docx'
    output=args.output or root/'results/report_final/report_verification.json'
    folder=root/'results/report_final'; data=read(folder/'report_data.json')
    manifest=read(folder/'report_table_manifest.json'); audit=Audit()
    audit.equal(digest(folder/'report_data.json'),manifest['source_data_sha256'],'manifest source data hash')
    frozen,summary,protocol,cache=verify_frozen_binary(root,data,audit)
    verify_dataset(data,audit)
    verify_raw_experiments(root,data,frozen,audit)
    blobs,document,tables,captions,paragraphs=document_parts(docx)
    hyperlink_evidence=verify_repository_hyperlink(blobs,document,audit)
    by_id=verify_manifest(manifest,tables,captions,paragraphs,audit)
    cover_policy=verify_cover(document,manifest,audit,args.cover_source)
    verify_table_numbers(data,by_id,summary,cache,audit)
    verify_figure_values(root,data,summary,cache,audit)
    verify_text_and_ooxml(root,data,manifest,blobs,document,by_id,paragraphs,audit)
    verify_measurement_definitions_and_tail(document,by_id,paragraphs,audit)
    sil_evidence=verify_sil(root,manifest,by_id,summary,paragraphs,audit)
    synthetic_evidence=verify_synthetic_demo(root,manifest,summary,paragraphs,audit)
    pdf_evidence=verify_pdf_pair(root,docx,audit,required=args.require_pdf or args.pdf is not None,
                               pdf=args.pdf,record_path=args.pdf_record,
                               pdfinfo=args.pdfinfo,pdf_python=args.pdf_python)
    result={'status':'pass','verified_at_utc':datetime.now(timezone.utc).isoformat(),
            'docx':str(docx),'docx_sha256':digest(docx),'checks':len(audit.checks),
            'tables':len(tables),'figures':len(manifest['figures']),
            'frozen_binary_preserved':True,'clean_ds2_inference_passes':1,
            'cover_policy':cover_policy,
            'sil_verification':sil_evidence,
            'synthetic_demo_verification':synthetic_evidence,
            'repository_hyperlink_verification':hyperlink_evidence,
            'pdf_exported':pdf_evidence is not None,'details':audit.checks}
    if pdf_evidence is not None:result['pdf_verification']=pdf_evidence
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='details'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
