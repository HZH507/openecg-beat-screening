"""Package current report materials after verifying the exact final document.

The archive includes frozen predictions and their original source bytes. It
excludes personal documents, page images and private Word/PDF diagnostic text.
No training, inference, threshold selection or experimental metric changes occur.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from verify_delivery import digest, read, verify


SCRIPT_NAMES = (
    "config.py", "models.py", "utils.py", "prepare_dataset.py", "split_record_level.py",
    "train_classifier.py", "openecg_demo.py", "make_figures.py", "workpoint_common.py",
    "prepare_workpoint_cv.py", "train_workpoint_fold.py", "freeze_workpoint.py",
    "evaluate_workpoint_once.py", "verify_workpoint.py", "run_workpoint_cv.ps1",
    "final_common.py", "prepare_final_experiment.py", "train_final_aux.py",
    "evaluate_final_consistency.py", "assemble_final_report_data.py", "make_final_figures.py",
    "build_medium_report.py", "verify_report.py", "render_docx_word.ps1", "export_report_pdf.ps1",
    "verify_exported_pdf.py", "package_report_code.py", "verify_delivery.py", "references.json",
)
PRIVATE_NAMES = {
    "word_layout_diagnostics.json", "word_export_summary.json", "pdf_export_verification.json",
    "package_verification.json",
}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def current_layout(path, docx_hash):
    layout = read(path)
    assert layout["source_sha256"].lower() == docx_hash, "Stale Word layout evidence"
    for key in ("blank_pages", "possible_widows_or_orphans", "orphan_headings"):
        assert not layout.get(key, []), "Unresolved Word layout finding: " + key
    assert not layout.get("pages_with_missing_header", []), "Missing running header"
    assert all(not table["spans_pages"] for table in layout["tables"]), "Split table"
    # Do not copy layout['pages'] or cover text into public metadata.
    return {"page_count": layout["page_count"], "blank_page_count": 0,
            "split_table_count": 0, "orphan_heading_count": 0,
            "private_layout_evidence_sha256": digest(path)}


def package_paths(root):
    paths = [root / name for name in ("README.md", "requirements.txt", ".gitignore", ".gitattributes")]
    paths += [root / "data/README.md"]
    paths += [root / "src" / name for name in SCRIPT_NAMES]
    for folder in ("results/workpoint_cv_v2", "results/report_final", "sil",
                   "results/sil", "openecg_ext", "results/openecg_demo"):
        for path in (root / folder).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.name in PRIVATE_NAMES or path.suffix.lower() in (".pyc", ".tmp", ".log"):
                continue
            assert path.suffix.lower() not in (".docx", ".pdf", ".emf"), path
            paths.append(path)
    return sorted(set(paths), key=lambda p: p.relative_to(root).as_posix())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--docx", type=Path)
    ap.add_argument("--verification", type=Path)
    ap.add_argument("--layout", type=Path)
    ap.add_argument("--pdf", type=Path, help="Optional matching final PDF; never included in the archive")
    ap.add_argument("--previous-docx", type=Path, help="Optional backed-up input for change-log hash")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    final = root / "results/report_final"
    cv = root / "results/workpoint_cv_v2"
    docx = args.docx or root / "report/ECG_课程报告_标准版.docx"
    docx_hash = digest(docx)
    audit = read(args.verification or final / "report_verification.json")
    assert audit["status"] == "pass" and audit["docx_sha256"] == docx_hash, "Stale document audit"
    assert (audit["tables"], audit["figures"]) == (11, 14), "Unexpected submission structure"
    assert audit["frozen_binary_preserved"] and audit["clean_ds2_inference_passes"] == 1
    layout = current_layout(args.layout or final / "word_layout_diagnostics.json", docx_hash)
    pdf_summary = None
    if args.pdf:
        evidence = audit.get("pdf_verification", {})
        assert audit.get("pdf_exported") and evidence.get("source_docx_sha256") == docx_hash
        assert evidence.get("pdf_sha256") == digest(args.pdf), "Stale PDF audit"
        assert evidence["pages"] == layout["page_count"], "Word/PDF page count mismatch"
        pdf_summary = {"sha256": digest(args.pdf), "pages": evidence["pages"]}
    frozen = read(cv / "frozen_workpoints.json")
    summary = read(cv / "ds2_summary.json")
    changes = {
        "revision_scope": "report wording, figure labels, reproducible distribution and verification only",
        "report_docx_sha256": docx_hash,
        "previous_docx_sha256": digest(args.previous_docx) if args.previous_docx else None,
        "new_experiments": False, "model_inference_performed": False,
        "frozen_binary": {"model_sha256": frozen["final_model_sha256"],
                          "selection_rule": frozen["primary_rule"],
                          "threshold": summary["working_points"]["sp90_max_se"]["threshold"],
                          "oof_auc": frozen["oof_auc"], "ds2_auc": summary["auc"],
                          "primary_metrics": summary["working_points"]["sp90_max_se"],
                          "clean_ds2_inference_passes": 1},
        "numeric_changes": [],
        "paragraph_changes": [
            {"section": "摘要/关键词/4.1/5/6", "change": "将目标特异性未保持解释为工作点与分数分布迁移，概率校准问题留待专门评价"},
            {"section": "摘要/4.6/README", "change": "将整套AFE消融与单独低通因果归因区分，保留幅频、相频与形态改变为可能解释"},
            {"section": "1/3.6/4.7/图14", "change": "说明上游NeuroKit2已有预处理及R峰检测；将演示准确命名为参考上游思路的自定义合成信号流程验证"},
            {"section": "表7", "change": "注明展示部分记录及选择目的，完整22条结果仍保存在JSON"},
            {"section": "附录A/B", "change": "补充结果采用表A1；修正表5内容说明；附录B列出仓库；与生成器同步"},
        ],
        "figure_changes": [
            {"figure": 6, "change": "标题明确DS2 ROC与OOF选定工作点；曲线和指标不变"},
            {"figure": 14, "change": "明确自定义合成输入；纵轴匹配实际未标准化波形；读取既有概率，不推理模型"},
        ],
        "repository_changes": ["发布原冻结NPZ与CSV缓存和本轮图表", "保留原冻结源码及JSON的准确换行字节",
                               "补充标准库文件完整性检查", "更新11表14图验收规则", "排除含个人封面全文的本地QA材料"],
        "verification": {"checks": audit["checks"], "tables": audit["tables"], "figures": audit["figures"]},
        "word_layout": layout, "pdf": pdf_summary,
    }
    save(final / "revision_changes.json", changes)
    entries = []
    for path in package_paths(root):
        assert path.is_file(), path
        entries.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                        "sha256": digest(path)})
    manifest = {"schema_version": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "report_docx_sha256": docx_hash, "report_pdf": pdf_summary,
                "report_pages": layout["page_count"], "entries": entries,
                "scope": "Authoritative frozen results, model weights, figure sources and code; no raw databases or personal documents",
                "line_endings": "Exact original bytes; Git attributes disable automatic newline conversion"}
    # Validate required cache coverage and original hashes before publishing any manifest.
    delivery_check = verify(root, manifest)
    save(root / "PACKAGE_MANIFEST.json", manifest)
    target = args.output or root / "report/ECG_project_code.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in entries:
            archive.write(root / entry["path"], entry["path"])
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        for entry in entries:
            assert hashlib.sha256(archive.read(entry["path"])).hexdigest() == entry["sha256"], entry["path"]
        assert not any(name.lower().endswith((".pdf", ".docx", ".emf")) for name in archive.namelist())
    temporary.replace(target)
    result = {"status": "pass", "zip_filename": target.name, "bytes": target.stat().st_size,
              "sha256": digest(target), "entry_count": len(entries) + 1,
              "docx_sha256": docx_hash, "pdf": pdf_summary,
              "all_entries_verified": True, "personal_documents_in_zip": 0,
              "delivery_check": delivery_check}
    save(final / "package_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
