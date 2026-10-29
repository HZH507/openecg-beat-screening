"""Verify a Word-exported PDF, then atomically install the current submission."""
from __future__ import annotations

import argparse,hashlib,json,re,subprocess
from datetime import datetime,timezone
from pathlib import Path
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT/'results/report_final'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def norm(text):return re.sub(r'\s+','',text)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pdf',type=Path,default=ROOT/'report/ECG_课程报告_标准版.pdf')
    ap.add_argument('--word-summary',type=Path,default=FINAL/'word_export_summary.json')
    ap.add_argument('--install',action='store_true')
    args=ap.parse_args()
    docx=ROOT/'report/ECG_课程报告_标准版.docx'
    word=read(args.word_summary);layout=read(FINAL/'word_layout_diagnostics.json')
    h=digest(docx)
    assert word['source_docx_sha256']==h and layout['source_sha256'].lower()==h
    reader=PdfReader(args.pdf);texts=[p.extract_text() or '' for p in reader.pages]
    assert len(texts)==word['word_page_count']==layout['page_count']
    assert all(len(norm(t))>10 for t in texts),'Blank PDF page'
    last=norm(texts[-1]);full=norm('\n'.join(texts))
    assert '表A2' in last and '肌电/15' in last and len(last)>400,'Incomplete or isolated last page'
    assert '完整精度阈值' in last and last.index('完整精度阈值')<last.index('表A2')
    for token in ('0.8558','0.7829','0.5879','72.27%','80.16%','79.29%','0.6533',
                  '算术均值','全局均方根','不能互相反推',
                  '训练/验证、测试、201排除与协议未纳入的四段样本分布',
                  '混淆矩阵、ROC与OOF冻结工作点'):
        assert token in full,'Missing PDF content: '+token
    assert '93.17%' not in full,'Old PDF content detected'
    assert not re.search(r'(?<!\d)0\.(?:763|829)(?!\d)',full),'Old metrics detected'
    pdfinfo=Path(r'C:\Users\hzh\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdfinfo.exe')
    info=subprocess.check_output([str(pdfinfo),str(args.pdf)],encoding='utf-8',errors='replace')
    match=re.search(r'^Pages:\s*(\d+)',info,re.M)
    assert match and int(match.group(1))==len(texts)
    canonical=ROOT/'report/ECG_课程报告_标准版.pdf'
    if args.install and args.pdf.resolve()!=canonical.resolve():
        temporary=canonical.with_suffix('.pdf.new.tmp')
        temporary.write_bytes(args.pdf.read_bytes());temporary.replace(canonical)
    target=canonical if args.install else args.pdf
    assert digest(target)==digest(args.pdf)
    record={'status':'pass','source_docx_sha256':h,'pdf_sha256':digest(target),
            'pdf_path':target.relative_to(ROOT).as_posix() if target.is_relative_to(ROOT) else str(target),
            'word_page_count':word['word_page_count'],'pdf_page_count':len(texts),
            'word_last_adjusted_page':word['word_last_adjusted_page'],
            'word_last_page_text':layout['pages'][-1]['text'],'pdf_last_page_text':texts[-1],
            'renderer':'Microsoft Word','exported_at_utc':word['exported_at_utc'],
            'verified_at_utc':datetime.now(timezone.utc).isoformat(),
            'old_pdf_replaced':bool(args.install),'pdfinfo_page_count':int(match.group(1)),
            'last_page_contains_complete_table_A2':True,'isolated_trailing_note':False}
    (FINAL/'pdf_export_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if not k.endswith('_text')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
