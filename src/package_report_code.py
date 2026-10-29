"""Package only current report sources/results, with byte-for-byte validation."""
from __future__ import annotations

from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT/'results/report_final'
CV=ROOT/'results/workpoint_cv_v2'
SIL=ROOT/'results/sil'
SIL_SOURCE_NAMES=('afe.py','adc.py','link.py','run_sil.py')


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    docx=ROOT/'report/ECG_课程报告_标准版.docx'
    verification=read(FINAL/'report_verification.json');layout=read(FINAL/'word_layout_diagnostics.json')
    assert verification['status']=='pass' and verification['docx_sha256']==digest(docx)
    pdf=read(FINAL/'pdf_export_verification.json')
    assert pdf['status']=='pass' and pdf['source_docx_sha256']==digest(docx)
    assert digest(ROOT/pdf['pdf_path'])==pdf['pdf_sha256']
    assert pdf['word_page_count']==pdf['pdf_page_count']==layout['page_count']
    assert layout['source_sha256'].lower()==digest(docx) and not layout['exported_pdf']
    assert not layout['blank_pages'] and not layout['possible_widows_or_orphans'] and not layout['orphan_headings']
    assert not layout.get('pages_with_missing_header',[])
    assert all(not t['spans_pages'] for t in layout['tables'])
    data=read(FINAL/'report_data.json');frozen=read(CV/'frozen_workpoints.json')
    assert digest(CV/'final_binary.pt')==frozen['final_model_sha256']
    for name,h in frozen['source_sha256'].items():assert digest(ROOT/'src'/name)==h
    primary=data['binary']['working_points']['sp90_max_se']
    table_manifest=read(FINAL/'report_table_manifest.json')
    sil=read(SIL/'sil_metrics.json')
    assert sil['threshold']==primary['threshold']
    assert sil['n_beats']==sum(sil['ideal'][k] for k in ('tp','fp','tn','fn'))
    assert all(sil['ideal'][k]==primary[k] for k in ('accuracy','macro_f1','sensitivity','specificity','tp','fp','tn','fn'))
    assert sil['ideal']['auc']==data['binary']['auc']
    assert sil['n_beats']==sum(sil['quant_link'][k] for k in ('tp','fp','tn','fn'))
    assert abs(sil['delta_auc_quant_link']-(sil['quant_link']['auc']-sil['ideal']['auc']))<1e-12
    sil_source={'path':'results/sil/sil_metrics.json','sha256':digest(SIL/'sil_metrics.json')}
    sil_source_files=[{'path':'sil/'+name,'sha256':digest(ROOT/'sil'/name)} for name in SIL_SOURCE_NAMES]
    assert table_manifest['sil_source']==sil_source
    assert table_manifest['sil_source_files']==sil_source_files
    assert sil['predictions_path']=='results/sil/sil_predictions.npz'
    sil_predictions={'path':sil['predictions_path'],'sha256':digest(ROOT/sil['predictions_path'])}
    assert sil_predictions['sha256']==sil['predictions_sha256']
    probability_summary=sil['probability_summary']
    correction=None
    prior_path=ROOT/'backup_sil_report_20261029/results/sil/sil_metrics.json'
    if prior_path.is_file():
        prior_sil=read(prior_path)
        assert all(sil[k]==prior_sil[k] for k in ('ideal','sil'))
        correction={
            'change':'corrected positive-lag shift from right to left; negative-lag shift from left to right; reran SIL only with user authorization',
            'previous_source':{'path':prior_path.relative_to(ROOT).as_posix(),'sha256':digest(prior_path)},
            'previous_aligned_metrics':prior_sil['sil_aligned'],
            'previous_delta_auc_aligned':prior_sil['delta_auc_aligned'],
            'ideal_and_unaligned_device_metrics_reproduced':True}
    elif (FINAL/'revision_changes.json').is_file():
        correction=read(FINAL/'revision_changes.json').get('sil',{}).get('alignment_correction')
    if correction is not None:
        correction.update(current_aligned_metrics=sil['sil_aligned'],
                          current_delta_auc_aligned=sil['delta_auc_aligned'],
                          withdrawn_conclusions=['aligned AUC almost unchanged','probabilities shifted down overall'])
    changes={
        'docx_sha256':digest(docx),'pages':layout['page_count'],'pdf_exported':True,
        'pdf_sha256':pdf['pdf_sha256'],'pdf_pages':pdf['pdf_page_count'],
        'frozen_binary':{'selection_rule':frozen['primary_rule'],'threshold':primary['threshold'],
                         'oof_auc':data['oof']['oof_auc'],'ds2_auc':data['binary']['auc'],
                         'ds2_primary_metrics':primary,'clean_ds2_inference_passes':1},
        'dataset':data['dataset'],'auxiliary_split':{k:data['aux_split'][k] for k in ('train_records','val_records','train_n','val_n')},
        'new_classification':{k:{m:data['classification'][k][m] for m in ('accuracy','macro_f1')} for k in ('cnn','mlp')},
        'denoiser_best_epoch':data['denoiser']['best_epoch'],
        'noise':{'threshold':data['noise']['threshold'],'conditions':[
            {'condition':r['condition'],**{m:r[m]['macro_f1'] for m in ('raw','classical','denoised')}} for r in data['noise']['rows']],
            'old_threshold_056_results_used':False,'per_channel_chronological_split':'65% training/validation, 35% test'},
        'demo':{'record':'208','task':'four-class functional demo','n_beats':len(data['demo']['pred']),
                'mismatches':data['demo']['mismatches'],'pred_counts':data['demo']['pred_counts'],
                'not_independent_validation':True},
        'cost':data['cost'],
        'sil':{
            'source':sil_source,'source_files':sil_source_files,'predictions':sil_predictions,
            'n_beats':sil['n_beats'],'n_abnormal':sil['n_abnormal'],'threshold':sil['threshold'],
            'pathway_metrics':{k:sil[k] for k in ('ideal','quant_link','sil','sil_aligned')},
            'delta_auc_quant_link':sil['delta_auc_quant_link'],
            'delta_auc':sil['delta_auc'],'delta_auc_aligned':sil['delta_auc_aligned'],
            'mean_lag_samples':sil['mean_lag_samples'],'lag_ms':sil['lag_ms'],
            'alignment_method':sil['alignment_method'],'alignment_scope':sil['alignment_scope'],
            'alignment_correction':correction,
            'probability_summary':probability_summary,
            'mean_probability_delta_aligned_vs_ideal':probability_summary['sil_aligned']['mean']-probability_summary['ideal']['mean'],
            'ideal_matches_frozen_cache_max_dev':sil['ideal_matches_frozen_cache_max_dev'],
            'afe':sil['afe'],'adc':sil['adc'],'link':sil['link'],
            'scope':'software in the loop; not physical hardware integration',
            'source_is_digitized_360_hz':True,'pre_sampling_aliasing_simulated':False,
            'link_parameters_are_configured_not_measured':True,
            'frozen_model_and_threshold_unchanged':True,
            'inference_scope':'original clean DS2 main evaluation cache/audit unchanged; SIL separately runs four pathways including ideal reconstruction and quantization/link ablation; authorized correction reran SIL; component ablation outputs subsequently supplied by user'},
        'paragraph_changes':[
            {'section':'摘要/4.1/5/6','change':'OOF与DS2 AUC同报；主点规则、全正常基线与校准迁移发现一致；去掉旧主指标'},
            {'section':'2.1与表1','change':'四段式样本总账，全部模型训练/验证排除201，733心搏单列'},
            {'section':'2.2/3.2/3.3','change':'逐拍归一化、五折与内层选轮次、冻结后一次干净推理；辅助17/4及实际抽样量披露'},
            {'section':'3.4/4.2/5','change':'新模型新阈值27组噪声评价，逐通道时间切分修正；分别解释重建与分类，按实测收益/退化写结论'},
            {'section':'4.3/4.5','change':'采用新四分类权重与demo；真值和预测分开；208仅DS1功能演示'},
            {'section':'3.1/3.6/4.5/5','change':'离线软件、硬件接口设计和资源估算分开；INT8仅估算；新桌面CPU摊销时延'},
            {'section':'参考文献/附录/封面属性','change':'补六条文献与新复现命令；清属性和封面蓝线；个人信息只留一处占位'}],
        'table_changes':['表1','表4','表5','表6','表7','表8','表9（SIL四通路，增加仅量化+链路）','表10（原表9资源）','表A1（增加SIL）','表A2'],
        'figure_changes':table_manifest['figures'],
        'verification':{'checks':verification['checks'],'tables':verification['tables'],'embedded_figures':verification['figures']},
    }
    changes['paragraph_changes'].extend([
        {'section':'3.5','change':'SNR采用逐拍定义及各拍dB均值；RMSE为全局均方根，聚合口径不同不能互相反推'},
        {'section':'图2/图6/附录尾部','change':'图注补分隔符；尾部说明并入表A2之前，缩小尾部空段，避免两行孤页'},
        {'section':'PDF交付','change':'Word从当前DOCX重新导出同名PDF覆盖旧版，核对页数和完整末页'}])
    changes['paragraph_changes'].extend([
        {'section':'SIL源码/结果','change':'经授权修正正lag左移、负lag右移并仅重跑SIL；主评价缓存及模型阈值保持；记录备份旧对齐结果与当前结果的对照，撤回原排序几乎无影响和概率整体下移结论'},
        {'section':'4.6/表9/图12/图13','change':'SIL链路方法、同一冻结模型与阈值下的四通路结果、相位和工作点迁移分析；表9在理想通路后增加仅量化+链路，图13采用四条ROC；明确数字源、未模拟采样前混叠及非实物联调边界'},
        {'section':'摘要/6','change':'补充SIL结论，并保留原OOF/DS2主评价与阈值冻结口径'},
        {'section':'表10/表A1/代码包','change':'资源表由表9顺延为表10；表A1增加sil四个源文件及results/sil输出，代码包纳入完整sil模块与SIL结果；绑定JSON与源码哈希'}])
    changes['paragraph_changes'].append(
        {'section':'摘要/4.6','change':'采用用户重跑的组件消融结果，替换尚未消融的旧段落；仅量化与链路接近理想，主要损失归于AFE串联低通；保留整数时延对齐后的频率相关相位与形态差异说明'})
    save(FINAL/'revision_changes.json',changes)
    scripts=['config.py','models.py','utils.py','prepare_dataset.py','split_record_level.py','train_classifier.py',
             'openecg_demo.py','make_figures.py','workpoint_common.py','prepare_workpoint_cv.py','train_workpoint_fold.py',
             'freeze_workpoint.py','evaluate_workpoint_once.py','verify_workpoint.py','run_workpoint_cv.ps1',
             'final_common.py','prepare_final_experiment.py','train_final_aux.py','evaluate_final_consistency.py',
             'assemble_final_report_data.py','make_final_figures.py','build_medium_report.py','verify_report.py',
             'render_docx_word.ps1','export_report_pdf.ps1','verify_exported_pdf.py','package_report_code.py','references.json']
    paths=[ROOT/'README.md',ROOT/'requirements.txt']+[ROOT/'src'/n for n in scripts]
    for folder in (CV,FINAL,ROOT/'sil',SIL):
        paths.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix not in ('.pyc','.tmp') and p.name!='package_verification.json')
    entries=[]
    for p in sorted(set(paths),key=lambda p:str(p.relative_to(ROOT))):
        assert p.is_file(),p
        relative=p.relative_to(ROOT).as_posix()
        assert p.suffix.lower() not in ('.pdf','.docx','.emf'),relative
        entries.append({'path':relative,'bytes':p.stat().st_size,'sha256':digest(p)})
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'report_docx_sha256':digest(docx),
              'report_pdf_sha256':pdf['pdf_sha256'],'report_pdf_pages':pdf['pdf_page_count'],
              'binary_checkpoint_sha256':frozen['final_model_sha256'],'entries':entries,
              'sil_source':sil_source,'sil_source_files':sil_source_files,
              'sil_predictions':sil_predictions,
              'note':'Current authoritative results only. No raw PhysioNet data, old experimental results, archived reports, PDFs or QA page images.'}
    target=ROOT/'report/ECG_project_code.zip';temp=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for e in entries:z.write(ROOT/e['path'],e['path'])
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        for e in entries:assert hashlib.sha256(z.read(e['path'])).hexdigest()==e['sha256'],e['path']
        assert not any(n.lower().endswith(('.pdf','.docx')) for n in z.namelist())
    temp.replace(target)
    result={'status':'pass','zip':str(target),'bytes':target.stat().st_size,'sha256':digest(target),
            'entry_count':len(entries)+1,'docx_sha256':digest(docx),'all_entries_verified':True,'pdfs_in_zip':0}
    result.update(pdf_exported=True,pdf_sha256=pdf['pdf_sha256'],pdf_pages=pdf['pdf_page_count'])
    save(FINAL/'package_verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
