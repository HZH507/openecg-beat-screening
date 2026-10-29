"""Consolidate only authoritative current results for report construction."""
from __future__ import annotations

from datetime import datetime,timezone
import shutil
import numpy as np

from config import ROOT,DS2
from final_common import OUT,CV,read,save,assert_binary_frozen
from workpoint_common import metrics_at,sha256


def main():
    frozen=assert_binary_frozen()
    assert read(OUT/'consistency_evaluation.json')['status']=='complete'
    if not (OUT/'references.json').exists():
        shutil.copy2(ROOT/'src/references.json',OUT/'references.json')
    accounting=read(OUT/'dataset_accounting.json')
    dataset={'groups':accounting['groups']+[{**accounting['total'],'key':'total','label':'合计'}]}
    binary=read(CV/'ds2_summary.json');oof=read(CV/'oof_selection.json');cv=read(CV/'protocol.json')
    metadata=read(OUT/'binary_noise_metadata.json')
    assert metadata['classifier_sha256']==frozen['final_model_sha256']
    assert metadata['threshold']==binary['working_points']['sp90_max_se']['threshold']
    with np.load(CV/'ds2_predictions.npz') as d:y,p,rec=d['y'],d['prob'],d['record']
    rows=[]
    for r in DS2:
        mask=rec==r;m=metrics_at(y[mask],p[mask],metadata['threshold'])
        rows.append({'record':r,**m,'abnormal':int(y[mask].sum())})
    save(OUT/'ds2_per_record_primary.json',rows)
    assert sum(r['fn'] for r in rows)==binary['working_points']['sp90_max_se']['fn']
    training=read(OUT/'denoiser_training.json')
    files=['dataset_accounting.json','aux_split.json','classification_metrics.json','classification_config.json',
           'denoiser_training.json','denoiser_history.json','binary_noise_results.json','binary_noise_metadata.json',
           'demo_info.json','model_cost.json','runtime_environment.json','references.json','ds2_per_record_primary.json']
    environment=read(OUT/'runtime_environment.json')
    if (OUT/'hardware.json').exists():
        environment['hardware']=read(OUT/'hardware.json')
        environment['cpu_model']=environment['hardware']['Name']
    data={'dataset':dataset,'binary':binary,'oof':oof,'cv':cv,
          'aux_split':read(OUT/'aux_split.json'),'classification':read(OUT/'classification_metrics.json'),
          'classification_config':read(OUT/'classification_config.json'),
          'classification_training':read(OUT/'classification_config.json'),
          'denoiser':{'best_epoch':training['best_epoch'],'training':training,'history':read(OUT/'denoiser_history.json')},
          'noise':{'n':metadata['n'],'threshold':metadata['threshold'],'metadata':metadata,'rows':read(OUT/'binary_noise_results.json')},
          'demo':read(OUT/'demo_info.json'),'cost':read(OUT/'model_cost.json'),
          'environment':environment,'per_record':rows,
          'references':read(OUT/'references.json'),
          'architecture':{
              'binary':[['输入','R 峰对齐标准化心搏','1×256'],['卷积 1','Conv1d(1,16,7)+ReLU+Pool/2','16×128'],
                        ['卷积 2','Conv1d(16,32,5)+ReLU+Pool/2','32×64'],['卷积 3','Conv1d(32,64,3)+ReLU+Pool/2','64×32'],
                        ['分类','GAP+Dropout 0.2+Linear(64,2)','2']],
              'denoiser':[['输入','含噪标准化心搏','1×256'],['首层','Conv1d(1,32,11)+ReLU','32×256'],
                          ['隐藏层×4','Conv1d(32,32,11)+BN+ReLU','32×256'],['输出层','Conv1d(32,1,11)','1×256'],
                          ['残差','输入−预测噪声','1×256']],
              'source':'src/models.py','source_sha256':sha256(ROOT/'src/models.py')},
          'reproduction':{
              'commands':['python src/assemble_final_report_data.py','python src/make_final_figures.py',
                          'python src/build_medium_report.py','python src/verify_report.py'],
              'files':[['src/prepare_workpoint_cv.py 等','记录级 OOF 选点与冻结协议'],
                       ['src/train_final_aux.py','排除 201 后四分类与 DnCNN 重训'],
                       ['src/evaluate_final_consistency.py','冻结模型的九条件三处理评价与演示'],
                       ['results/workpoint_cv_v2/','二分类模型 工作点与一次干净测试缓存'],
                       ['results/report_final/','本轮辅助模型 图表及原始指标'],
                       ['src/build_medium_report.py 与 verify_report.py','读取缓存生成与逐表验收']]},
          'resource_links':[['MIT-BIH','https://physionet.org/content/mitdb/1.0.0/'],
                            ['NSTDB','https://physionet.org/content/nstdb/1.0.0/'],
                            ['OpenECG','https://github.com/jwlee95/OpenECG'],['PyTorch','https://pytorch.org/']],
          'provenance':{'assembled_at_utc':datetime.now(timezone.utc).isoformat(),
                        'raw_result_sha256':{f:sha256(OUT/f) for f in files},
                        'frozen_binary_workpoints_sha256':sha256(CV/'frozen_workpoints.json'),
                        'binary_clean_cache_sha256':sha256(CV/'ds2_predictions.npz'),
                        'clean_binary_ds2_added_inference':0}}
    names=['fig01_dataset_distribution.png','fig02_beat_examples.png','fig03_filtering.png',
           'fig04_denoising_example.png','fig05_training_curves.png','fig05b_denoising_snr.png',
           'fig06_confusion_matrices.png','fig06b_binary_roc.png','fig07_per_class_f1.png',
           'fig08_noise_robustness.png','fig09_cnn_architecture.png','fig10_system_integration.png',
           'fig11_openecg_demo.png']
    data['figures']={name:'results/report_final/figures/'+name for name in names}
    save(OUT/'report_data.json',data)
    print('Assembled current report data; primary threshold',metadata['threshold'],flush=True)


if __name__=='__main__':main()
