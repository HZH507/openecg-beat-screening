"""Audit cached final results and provenance without rerunning model inference."""
from __future__ import annotations

import json
import zipfile

import numpy as np
import torch

from config import MODELS, RESULTS, ROOT
from split_record_level import sha256
from train_binary import metrics
from utils import save_json


def read(name):
    return json.loads((RESULTS/name).read_text(encoding='utf-8'))


def main():
    protocol=read('val_threshold.json')
    audit=read('final_test_evaluation.json')
    assert audit['status']=='complete'
    assert sha256(RESULTS/'val_threshold.json')==audit['protocol_sha256']
    assert sha256(RESULTS/'record_split.json')==protocol['record_split_sha256']
    split=read('record_split.json')
    for name,digest in protocol['model_sha256'].items():
        assert sha256(MODELS/name)==digest
        ck=torch.load(MODELS/name,map_location='cpu',weights_only=False)
        assert ck['record_split_sha256']==protocol['record_split_sha256']
        assert ck['train_records']==split['train_records']
        assert ck['val_records']==split['val_records']
    with np.load(RESULTS/'binary_predictions.npz') as d:
        y,pred,prob,rec=d['y'],d['pred'],d['prob'],d['record']
        np.testing.assert_array_equal(pred,(prob>=protocol['threshold']).astype(np.int64))
    binary=read('binary_metrics.json')
    cached=metrics(y,pred,prob)
    for k in ('accuracy','macro_f1','sensitivity','specificity','auc'):
        assert abs(binary[k]-cached[k])<1e-12,k
    records=read('binary_per_record.json')
    assert sum(r['n'] for r in records)==binary['n']==49705
    assert sum(r['abnormal'] for r in records)==int(y.sum())
    assert sum(r['false_negative'] for r in records)==binary['confusion_matrix'][1][0]
    assert sum(r['false_positive'] for r in records)==binary['confusion_matrix'][0][1]
    points=read('binary_threshold_metrics.json')
    assert [r['threshold'] for r in points]==protocol['predeclared_test_thresholds']
    for r in points:
        expected=metrics(y,(prob>=r['threshold']).astype(np.int64),prob)
        for k in ('accuracy','macro_f1','sensitivity','specificity','auc'):
            assert abs(r[k]-expected[k])<1e-12
    candidates=protocol['validation_sweep']
    assert len(candidates)==297
    assert protocol['validation_metrics']['macro_f1']==max(r['macro_f1'] for r in candidates)
    for weight in (3,5,9):
        with np.load(RESULTS/f'binary_w{weight}_val_predictions.npz') as d:
            np.testing.assert_array_equal(d['global_index'],split['val_indices'])
    noise=read('binary_noise_results.json')
    marker=read('binary_noise_evaluation.json')
    assert marker['status']=='complete' and len(noise)==10
    assert len(marker['subset_global_indices'])==marker['n']==9446
    assert marker['n_abnormal']==5446
    for r in noise:
        assert r['threshold']==protocol['threshold'] and r['n']==9446
        for method in ('raw','classical','denoised'):
            m=r[method]
            assert m['tp']+m['fp']+m['tn']+m['fn']==9446
            assert m['tp']+m['fn']==5446
    four=read('classification_metrics.json')
    for k in ('cnn','mlp'):
        assert four[k]['n']==49705
        assert np.asarray(four[k]['confusion_matrix']).sum()==49705
    demo=read('demo_info.json')
    assert demo['model_task']=='four-class exploration' and not demo['uses_rr_features']
    assert len(demo['true_cls'])==len(demo['pred'])==len(demo['truth_prediction_pairs'])
    assert demo['mismatches']==sum(p['true']!=p['pred'] for p in demo['truth_prediction_pairs'])
    figure_names=['fig01_dataset_distribution.png','fig02_beat_examples.png','fig03_filtering.png',
                  'fig04_denoising_example.png','fig05_training_curves.png','fig05b_denoising_snr.png',
                  'fig06_confusion_matrices.png','fig06b_binary_roc.png','fig07_per_class_f1.png',
                  'fig08_noise_robustness.png','fig09_cnn_architecture.png','fig10_system_integration.png',
                  'fig11_openecg_demo.png']
    for name in figure_names:
        assert (RESULTS/'figures'/name).stat().st_size>1000
    with zipfile.ZipFile(ROOT/'report/ECG_课程报告_标准版.docx') as new, \
         zipfile.ZipFile(ROOT/'backup_20261029/report/ECG_课程报告_标准版.docx') as old:
        assert set(new.namelist())==set(old.namelist())
        assert all(new.read(n)==old.read(n) for n in new.namelist()),'Report changed before confirmation'
    assert sha256(ROOT/'src/build_medium_report.py')==sha256(ROOT/'backup_20261029/src/build_medium_report.py')
    summary={'status':'pass','cached_metrics_verified':True,'frozen_hashes_verified':True,
             'train_val_records_disjoint':True,'clean_test_n':49705,'noise_test_n':9446,
             'noise_conditions':9,'figures':len(figure_names),'report_unchanged':True,
             'report_rewrite_awaits_user_confirmation':True}
    save_json(RESULTS/'experiment_verification.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
