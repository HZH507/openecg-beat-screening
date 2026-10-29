"""Build the submission DOCX from the frozen and consistency-checked results.

No training, model inference or PDF export is performed by this script.
The table manifest is a rendering record; independent source verification is
implemented separately in verify_report.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Cm, Pt, RGBColor


BODY = 10.6
LINE = 1.38
CN = "宋体"
HEAD = "黑体"
EN = "Times New Roman"
CLASSES = ["N", "S", "V", "F"]
KINDS = [("bw", "基线漂移"), ("em", "电极运动"), ("ma", "肌电")]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(value):
    return f"{int(value):,}"


def percent(value, digits=2):
    return f"{float(value) * 100:.{digits}f}%"


def number(value, digits=4):
    return f"{float(value):.{digits}f}"


def must(condition, message):
    if not condition:
        raise ValueError(message)


def get_record(row, *names):
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"None of {names!r} found in result row")


def math_run(text):
    run = OxmlElement("m:r")
    props = OxmlElement("w:rPr")
    size = OxmlElement("w:sz"); size.set(qn("w:val"), "21"); props.append(size)
    run.append(props)
    value = OxmlElement("m:t"); value.text = text; run.append(value)
    return run


def math_sub(base, subscript):
    node = OxmlElement("m:sSub")
    expression = OxmlElement("m:e"); expression.append(base); node.append(expression)
    sub = OxmlElement("m:sub"); sub.append(math_run(subscript)); node.append(sub)
    return node


def beat_sample(estimate=False):
    base = math_run("x")
    if estimate:
        accent = OxmlElement("m:acc"); props = OxmlElement("m:accPr")
        char = OxmlElement("m:chr"); char.set(qn("m:val"), "̂"); props.append(char); accent.append(props)
        expression = OxmlElement("m:e"); expression.append(base); accent.append(expression); base = accent
    return math_sub(base, "i,t")


def math_square(nodes):
    node = OxmlElement("m:sSup"); expression = OxmlElement("m:e")
    for child in nodes: expression.append(child)
    node.append(expression)
    power = OxmlElement("m:sup"); power.append(math_run("2")); node.append(power)
    return node


def sum_over_time(expression):
    node = OxmlElement("m:nary"); props = OxmlElement("m:naryPr")
    for name, value in (("chr", "∑"), ("limLoc", "subSup"), ("supHide", "1")):
        setting = OxmlElement("m:" + name); setting.set(qn("m:val"), value); props.append(setting)
    node.append(props)
    sub = OxmlElement("m:sub"); sub.append(math_run("t")); node.append(sub)
    node.append(OxmlElement("m:sup"))
    body = OxmlElement("m:e"); body.append(expression); node.append(body)
    return node


def document_with_cover(cover_from: Path | None):
    """Keep a local submission cover without putting identity in public sources.

    The existing document package also preserves the cover's styles, section
    boundary, headers and footers. Only content from the abstract onwards is
    replaced. The public/default build uses blank identity fields instead.
    """
    if cover_from is None:
        return Document()
    doc = Document(cover_from)
    body = doc.element.body
    children = list(body)
    start = next((i for i, element in enumerate(children)
                  if "".join(element.xpath(".//w:t/text()")).replace(" ", "").strip() == "摘要"), None)
    must(start is not None, "Cover source has no abstract heading")
    must(any(element.xpath(".//w:sectPr") for element in children[:start]),
         "Cover source must have a section break before the abstract")
    must(not any(element.tag == qn("w:tbl") for element in children[:start]),
         "The submission cover must use paragraphs, not an untracked table")
    for element in children[start:]:
        if element.tag != qn("w:sectPr"):
            body.remove(element)
    return doc


def build(root: Path, data_path: Path, out: Path, cover_from: Path | None = None):
    data = read_json(data_path)
    final = root / "results" / "report_final"
    binary = data["binary"]
    oof = data["oof"]
    cv = data["cv"]
    auxiliary = data["aux_split"]
    noise = data["noise"]
    classification = data["classification"]
    denoiser = data["denoiser"]
    demo = data["demo"]
    cost = data["cost"]
    environment = data["environment"]
    groups = data["dataset"]["groups"]
    architecture = data["architecture"]
    sil_path = root / "results" / "sil" / "sil_metrics.json"
    sil = read_json(sil_path)
    openecg_path = root / "results" / "openecg_demo" / "openecg_demo.json"
    openecg = read_json(openecg_path)
    data["figures"] = dict(data["figures"])
    data["figures"].update({name: f"results/sil/{name}" for name in ("sil_afe.png", "sil_roc.png")})
    data["figures"]["openecg_sim.png"] = "results/openecg_demo/openecg_sim.png"
    bpoint = binary["working_points"]["sp90_max_se"]
    baseline = binary["all_normal_baseline"]
    oof_point = oof["points"]["sp90_max_se"]
    cost_models = cost["models"]
    by_model = {m["key"]: m for m in cost_models}
    binary_cost = next(m for m in cost_models if m["key"] in ("binary", "binary_cnn", "cnn_binary"))
    denoiser_cost = next(m for m in cost_models if m["key"] in ("denoiser", "dncnn"))
    four_cost = next(m for m in cost_models if m["key"] in ("cnn", "four_class", "cnn_light", "four_class_cnn"))
    noise_rows = [r for r in noise["rows"] if r["kind"] != "clean"]
    noise_by_key = {(r["kind"], int(r["snr_db"])): r for r in noise_rows}
    must(set(noise_by_key) == {(k, s) for k, _ in KINDS for s in (5, 10, 15)}, "Exactly nine current noise conditions are required")
    must(float(noise["threshold"]) == float(bpoint["threshold"]), "Noise threshold differs from the frozen primary threshold")
    must("201" not in auxiliary["train_records"] + auxiliary["val_records"], "Record 201 is present in the auxiliary split")
    must("201" not in cv["eligible_records"], "Record 201 is present in binary CV")
    must(len(cv["eligible_records"]) == 21, "Binary protocol must have 21 eligible records")
    must(sum(int(g["n"]) for g in groups[:-1]) == int(groups[-1]["n"]) == 101451, "Four-part accounting does not close")
    must(int(binary["n"]) == 49705, "Unexpected DS2 size")
    must(int(sil["n_beats"]) == int(binary["n"]), "SIL must include all DS2 beats")
    must(float(sil["threshold"]) == float(bpoint["threshold"]), "SIL threshold differs from the frozen primary threshold")
    must(float(sil["ideal_matches_frozen_cache_max_dev"]) <= 1.3e-7, "SIL ideal path differs from the frozen cache")
    for metric in ("auc", "accuracy", "macro_f1", "sensitivity", "specificity"):
        reference = binary["auc"] if metric == "auc" else bpoint[metric]
        must(abs(float(sil["ideal"][metric]) - float(reference)) < 1e-10, f"SIL ideal {metric} mismatch")
    must(int(openecg["n_beats"]) == 10, "OpenECG demo must contain ten beats")
    must(int(openecg["n_abnormal"]) == 0, "OpenECG demo must not flag any simulated beat")
    must(abs(float(openecg["threshold"]) - float(bpoint["threshold"])) < 1e-12,
         "OpenECG demo threshold differs from the frozen primary threshold")
    must(abs(float(openecg["heart_rate_bpm_measured"]) - float(openecg["heart_rate_bpm_target"])) < 0.5,
         "OpenECG demo heart rate is implausible")

    references_path = final / "references.json"
    references = data.get("references")
    if references is None:
        references = read_json(references_path)
    if isinstance(references, dict):
        references = references["references"]
    must(len(references) == 6, "Six verified references are required")

    manifest = {
        "schema_version": 1,
        "source_data": str(data_path),
        "source_data_sha256": sha256(data_path),
        "sil_source": {"path": "results/sil/sil_metrics.json", "sha256": sha256(sil_path)},
        "sil_source_files": [{"path": f"sil/{name}", "sha256": sha256(root / "sil" / name)}
                             for name in ("afe.py", "adc.py", "link.py", "run_sil.py")],
        "openecg_source": {"path": "results/openecg_demo/openecg_demo.json", "sha256": sha256(openecg_path)},
        "cover_provenance": ({"mode": "preserved", "source_sha256": sha256(cover_from)}
                             if cover_from else {"mode": "blank"}),
        "tables": [], "key_text": [], "figures": [],
    }
    doc = document_with_cover(cover_from)
    cp = doc.core_properties
    cp.title = "基于深度学习的 ECG 心搏异常筛查与去噪"
    cp.subject = "生物医学传感检测与仪器课程项目报告"
    cp.author = ""
    cp.last_modified_by = ""
    cp.comments = ""
    cp.keywords = "ECG, AUC, OOF, MIT-BIH, DnCNN"
    cp.category = "课程项目报告"
    cp.identifier = ""
    cp.language = "zh-CN"
    cp.revision = 1
    cp.created = datetime.now(timezone.utc)
    cp.modified = cp.created

    def configure_section(section, cover=False):
        section.page_width, section.page_height = Cm(21.0), Cm(29.7)
        section.top_margin, section.bottom_margin = Cm(2.1), Cm(2.0)
        section.left_margin, section.right_margin = Cm(2.2), Cm(2.2)
        section.header_distance, section.footer_distance = Cm(1.0), Cm(.9)
        section.different_first_page_header_footer = cover

    if cover_from is None:
        configure_section(doc.sections[0], cover=True)
        normal = doc.styles["Normal"]
        normal.font.name = EN
        normal._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), CN)
        normal.font.size = Pt(BODY)
        normal.paragraph_format.line_spacing = LINE
        normal.paragraph_format.space_after = Pt(0)
        normal.paragraph_format.widow_control = True
        for style_name, size in (("Title", 20), ("Subtitle", 10), ("Heading 1", 13), ("Heading 2", 11.5)):
            style = doc.styles[style_name]
            style.font.name = HEAD if style_name != "Subtitle" else EN
            style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), HEAD if style_name != "Subtitle" else CN)
            style.font.size = Pt(size)
            style.font.color.rgb = RGBColor(0, 0, 0)
            style.font.bold = style_name != "Subtitle"
            for border in list(style.element.xpath("./w:pPr/w:pBdr")):
                border.getparent().remove(border)

    def font(run, size=BODY, bold=False, cn=CN, italic=False):
        run.font.name = EN
        run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), cn)
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = RGBColor(0, 0, 0)

    def para(text, first=True, size=BODY, bold=False,
             align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=0, key=None, source=None):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.alignment = align
        pf.line_spacing = LINE
        pf.space_before, pf.space_after = Pt(before), Pt(after)
        pf.widow_control = True
        if key == "cost_protocol":
            pf.keep_together = True
        if first:
            pf.first_line_indent = Cm(.7)
        font(p.add_run(text), size=size, bold=bold)
        if key:
            manifest["key_text"].append({"id": key, "text": text, "source": source})
        return p

    def heading(text, level=1):
        p = doc.add_paragraph(style=f"Heading {level}")
        p.paragraph_format.space_before = Pt(7 if level == 1 else 5)
        p.paragraph_format.space_after = Pt(4 if level == 1 else 2)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.keep_together = True
        font(p.add_run(text), 13 if level == 1 else 11.5, bold=True, cn=HEAD)

    def caption(text, keep=True):
        p = doc.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.keep_with_next = keep
        p.paragraph_format.keep_together = True
        font(p.add_run(text), 8.5)

    def table(table_id, headers, rows, widths, title, fs=8.0, left_columns=()):
        headers = [str(v) for v in headers]
        rows = [[str(v) for v in row] for row in rows]
        must(all(len(row) == len(headers) for row in rows), f"Invalid row width in {table_id}")
        caption(title)
        t = doc.add_table(rows=1, cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for col, width in zip(t.columns, widths):
            col.width = Cm(width)
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "bottom", "left", "right", "insideH", "insideV"):
            element = OxmlElement(f"w:{edge}")
            element.set(qn("w:val"), "single")
            element.set(qn("w:sz"), "4")
            element.set(qn("w:color"), "D9D9D9")
            borders.append(element)
        t._tbl.tblPr.append(borders)
        margins = OxmlElement("w:tblCellMar")
        for side, value in (("top", 48), ("bottom", 48), ("left", 65), ("right", 65)):
            el = OxmlElement(f"w:{side}")
            el.set(qn("w:w"), str(value)); el.set(qn("w:type"), "dxa")
            margins.append(el)
        t._tbl.tblPr.append(margins)
        for values in rows:
            t.add_row()
        for ri, values in enumerate([headers] + rows):
            row = t.rows[ri]
            trpr = row._tr.get_or_add_trPr()
            trpr.append(OxmlElement("w:cantSplit"))
            if ri == 0:
                trpr.append(OxmlElement("w:tblHeader"))
            for ci, (cell, value, width) in enumerate(zip(row.cells, values, widths)):
                cell.width = Cm(width)
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                if ri == 0:
                    fill = OxmlElement("w:shd"); fill.set(qn("w:fill"), "DCE6F1")
                    cell._tc.get_or_add_tcPr().append(fill)
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ri and ci in left_columns else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.02
                p.paragraph_format.keep_together = True
                p.paragraph_format.keep_with_next = ri < len(rows)
                font(p.add_run(value), fs, bold=ri == 0)
        spacer = doc.add_paragraph()
        spacer.paragraph_format.space_after = Pt(0)
        if table_id in ("tableA1", "tableA2"):
            spacer.paragraph_format.space_before = Pt(0)
            spacer.paragraph_format.line_spacing = Pt(2)
        manifest["tables"].append({"id": table_id, "caption": title, "headers": headers, "rows": rows})

    def figure(name, width, title, figure_id):
        path = Path(data.get("figures", {}).get(name, str(final / "figures" / name)))
        if not path.is_absolute():
            path = root / path
        must(path.exists(), f"Current figure missing: {path}")
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Cm(width))
        caption(title, keep=False)
        manifest["figures"].append({"id": figure_id, "filename": name, "path": str(path),
                                    "caption": title, "sha256": sha256(path)})

    def page_number(p):
        font(p.add_run("第 "), 8.5)
        run = p.add_run()
        for typ, text in (("begin", None), (None, "PAGE"), ("separate", None)):
            el = OxmlElement("w:fldChar" if typ else "w:instrText")
            if typ:
                el.set(qn("w:fldCharType"), typ)
            else:
                el.set(qn("xml:space"), "preserve"); el.text = text
            run._r.append(el)
        txt = OxmlElement("w:t"); txt.text = "1"; run._r.append(txt)
        end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end"); run._r.append(end)
        font(run, 8.5)
        font(p.add_run(" 页"), 8.5)

    # Public reproduction has blank fields; local submissions may keep a cover.
    if cover_from is None:
        for _ in range(2):
            doc.add_paragraph()
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        font(p.add_run("海 南 大 学"), 22, bold=True, cn=HEAD)
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        font(p.add_run("《生物医学传感检测与仪器》课程项目报告"), 11.5)
        p = doc.add_paragraph(style="Title"); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(50)
        p.paragraph_format.space_after = Pt(0)
        font(p.add_run("基于深度学习的 ECG"), 20, bold=True, cn=HEAD)
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        font(p.add_run("心搏异常筛查与去噪"), 20, bold=True, cn=HEAD)
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(8)
        font(p.add_run("ECG Beat Abnormality Screening and Denoising"), 9.5, italic=True)
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(22)
        font(p.add_run("面向 OpenECG 的离线分析与接入设计"), 10.5)
        cover_rows = [["姓    名", "________________"], ["学    号", "________________"],
                      ["专    业", "生物医学工程"], ["指导教师", "________________"],
                      ["日    期", "2026 年 10 月"]]
        for label, value in cover_rows:
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            font(p.add_run(label + "："), 10.5, cn=HEAD)
            font(p.add_run(value), 10.5)
        section = doc.add_section(WD_SECTION.NEW_PAGE)
        configure_section(section)
        pg = OxmlElement("w:pgNumType"); pg.set(qn("w:start"), "1"); section._sectPr.append(pg)
        hp = section.header.paragraphs[0]; hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        font(hp.add_run("海南大学 · 生物医学传感检测与仪器"), 8)
        fp = section.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER; page_number(fp)

    heading("摘  要")
    abstract = (
        "单导联 ECG 的异常心搏比例低，噪声和跨记录差异会同时影响筛查结果。本文构建离线预处理、"
        f"R 峰对齐心搏切分、{integer(binary_cost['params'])} 参数二分类 CNN 与 {integer(denoiser_cost['params'])} 参数 DnCNN，"
        "并给出面向 OpenECG 的接入设计。参考其思路构建自定义合成信号并完成流程验证，"
        f"{int(openecg['n_beats'])} 拍全部判为正常。排除与测试记录 202 属于同一受试者的记录 201 后，"
        f"在 {integer(oof['oof_n'])} 个心搏上开展五折记录级交叉验证，合并 OOF AUC 为 {number(oof['oof_auc'])}；"
        f"最终模型在 DS2 的 {integer(binary['n'])} 个心搏上 AUC 为 {number(binary['auc'])}。"
        f"OOF 规则选出的阈值 {number(bpoint['threshold'])} 在 DS2 上取得 Se {percent(bpoint['sensitivity'])}、"
        f"Sp {percent(bpoint['specificity'])}、Acc {percent(bpoint['accuracy'])}、Macro-F1 {number(bpoint['macro_f1'])}。"
        "OOF 的 Sp≥90% 约束未能在 DS2 保持，提示跨记录及模型重拟合后的分数分布变化。去噪、四分类与演示均在排除 201 后重跑，"
        "其收益与误差按当前实测结果分别报告。"
        f"在 DS2 全量软件在环仿真中，设备链路整数时延对齐后的 AUC 为 {number(sil['sil_aligned']['auc'])}，"
        "未恢复理想通路的排序能力，且固定工作点发生迁移，说明设备域需另行校准，尚不能替代实物联调。"
        "组件消融显示性能差异主要随 AFE 的引入出现，尚未分别确定幅频、相频响应和波形改变的贡献。"
    )
    para(abstract, key="abstract", source="binary/oof/cost/sil")
    p = doc.add_paragraph(); font(p.add_run("关键词："), bold=True, cn=HEAD)
    font(p.add_run("ECG；AUC；记录级交叉验证；工作点迁移；DnCNN；OpenECG"))

    heading("1  项目背景与目标")
    para("可穿戴心电与 Holter 数据需要自动筛查辅助复核。单看总体准确率容易掩盖少数异常心搏的漏检；"
         "同一记录同时进入训练和验证，也会使模型选择过于乐观。本项目围绕两项可复核的问题展开："
         "形态网络在未见记录上的排序能力如何，以及去噪后重建质量的变化是否能转化为同一阈值下的分类收益。")
    para("OpenECG 项目提供心电信号模拟与绘图示例，其中 NeuroKit2 示例还包含预处理和 R 峰检测，采集与硬件部分仍在开发[6]。"
         "本项目已经运行的功能为 WFDB 文件读取、离线滤波、心搏切分、神经网络评价与演示，"
         "并参考上游思路完成自定义合成信号流程验证；硬件采集、BLE 通信和上位机实时接口属于接入设计，"
         "尚未与实体设备联调。研究目标是提供可重复的软件实验与误差分析，不以硬件部署或临床验证为结论。")
    para("主要工作包括：统一排除记录 201 的数据口径；以记录级 OOF 预测选择并冻结二分类工作点；"
         "在冻结模型上比较原始含噪、传统滤波和 DnCNN 三种处理；重训四分类探索模型并更新演示；"
         "测量桌面 CPU 开销，并区分实测结果与部署估算。")
    para("另以参考 OpenECG 思路的自定义合成信号检验处理流程（4.7），并以软件在环方式检验候选 AFE、ADC 与分包重组"
         "对冻结模型的影响（4.6）；所用链路参数为仿真设定，未进行实物联调。")

    heading("2  数据与评价方案")
    heading("2.1  数据集与记录级划分", 2)
    para("使用 MIT-BIH 心律失常数据库[1]和 MIT-BIH 噪声压力测试数据库 NSTDB[2]。MIT-BIH 包含 48 条记录、"
         "47 名受试者，采样率为 360 Hz。依据 AAMI 分组合并为 N、S、V、F 四类，起搏或无法归类的 Q 类不进入监督任务。"
         "以专家 R 峰为中心截取前 96 点、后 160 点，共 256 点；处理后样本总数为 101,451。")
    para("以 de Chazal 的记录级 DS1/DS2 划分[3]为基础，额外排除 DS1 中的 201，因为它与 DS2 的 202 来自同一受试者。"
         "本轮二分类、四分类和去噪器的训练与模型选择均不使用 201。表 1 将全部样本分成四段，避免把排除样本误计入 DS2。",
         key="dataset_scope", source="dataset/aux_split/cv")
    accounting_rows = [[g["label"], str(g["n_records"]), *[integer(g["counts"][c]) for c in CLASSES], integer(g["n"])] for g in groups]
    table("table1", ["用途", "记录数", "N", "S", "V", "F", "心搏数"], accounting_rows,
          [4.4, 1.2, 2.3, 1.7, 1.7, 1.5, 2.5], "表 1  四段式样本口径及 AAMI 类别数量", fs=7.9, left_columns=(0,))
    para("训练/验证 49,050 + 测试 DS2 49,705 + 201 排除 1,963 + 协议未纳入 733 = 101,451。"
         "协议未纳入的 102/104/107/217 共 733 个心搏；201 的排除原因与这四条记录不同。"
         "DS2 中 N 44,259、S 1,837、V 3,221、F 388，合计 49,705。", first=False, size=8.8,
         key="dataset_accounting", source="dataset")
    para("NSTDB 噪声按各通道共同的物理时间边界划分：每通道前 65% 用于训练/验证，后 35% 用于测试，再分别拼接两通道。"
         "先沿时间划分再拼接，避免同一时刻的一路通道进入训练、另一路进入测试。该实现修正和模型重训均属于本轮噪声实验条件。",
         key="noise_time_split", source="denoiser.training/noise")
    figure("fig02_beat_examples.png", 13.6, "图 1  四类心搏的典型波形与标注类别", "figure1")
    figure("fig01_dataset_distribution.png", 13.6, "图 2  训练/验证、测试、201 排除与协议未纳入的四段样本分布", "figure2")

    heading("2.2  预处理", 2)
    para("对每条记录做三阶 Butterworth 0.5–40 Hz 带通与 60 Hz 陷波，再按专家 R 峰切分心搏并减去中位数。"
         "分类训练、验证与推理均逐拍做 z-score；去噪输入与目标按含噪心搏的均值和标准差缩放，输出再还原为 mV。"
         "原噪声评价路径本来就逐拍标准化；新旧噪声结果的变化不能归因为修复干净测试路径的归一化问题。")
    para("预处理使用 sosfiltfilt 等前后向零相位运算，会用到未来样本，因此当前流程只能用于离线分析。"
         "端到端演示采用 Pan–Tompkins 风格的 R 峰检测器[5]；真实在线采集还需要因果滤波、流式切窗与延迟评价。")
    figure("fig03_filtering.png", 13.4, "图 3  离线带通滤波前后的 ECG 与 R 峰位置", "figure3")

    heading("3  方法")
    heading("3.1  系统总体结构", 2)
    para("已运行的软件链路为：读取记录、离线预处理、R 峰检测或标注对齐、心搏切分、DnCNN 去噪及 CNN 分类。"
         "OpenECG 硬件、模拟前端、MCU 与通信链路只作为候选接入架构；本实验没有完成硬件实接、实时部署或 MCU 测试。"
         "二分类用于正常/异常筛查，四分类模型单独用于亚类探索和可视化演示，二者权重与结果不混用。")
    figure("fig10_system_integration.png", 15.2, "图 4  已运行离线软件与 OpenECG 候选接入设计", "figure4")

    heading("3.2  轻量分类器与 OOF 工作点选择", 2)
    para(f"二分类器输入为 1×256 的标准化心搏，采用三阶段卷积、ReLU 和最大池化，全局平均池化后经 Dropout 与线性层输出正常/异常 logits，"
         f"不使用 BatchNorm，共 {integer(binary_cost['params'])} 个参数。模型仅使用单拍波形，不输入 RR 特征。")
    table("table2", ["阶段", "操作", "输出"], architecture["binary"], [2.3, 9.5, 3.6],
          f"表 2  二分类 CNN 的结构（{integer(binary_cost['params'])} 参数）", fs=8.0)
    train_cfg = cv["training"]
    para(f"二分类使用异常:正常={train_cfg['abnormal_weight']}:1 的类别权重、{train_cfg['label_smoothing']:.2f} 标签平滑、AdamW、"
         f"学习率 {train_cfg['learning_rate']:.0e}、批大小 {train_cfg['batch']} 和余弦调度。"
         "类别权重沿用预先规定的训练设置，本轮未按测试结果重新选择。增强包括小幅时间抖动、幅度缩放、基线扰动和高斯噪声。")
    fold_text = "；".join(f"第 {f['fold']} 折 " + "/".join(f["outer_val_records"]) for f in cv["folds"])
    para(f"将排除 201 后的 21 条记录按异常比例分层并以固定种子分为五折，每折留出 4–5 条记录：{fold_text}。"
         "每一拍的 OOF 概率均由未训练过其所属记录的模型产生。各外折训练分区再留出 4 条内部验证记录，"
         f"以内部验证 AUC 选择轮次，最多 {train_cfg['max_epochs']} 轮、耐心值 {train_cfg['patience']}；外折留出标签不用于选轮次。",
         key="cv_disclosure", source="cv")
    selected_epochs = "/".join(str(v) for v in oof["inner_selected_epochs"])
    para(f"五折内层选出的轮次为 {selected_epochs}，最终模型在全部 21 条合格记录上按中位数训练 {oof['final_epochs']} 轮。"
         f"合并 {integer(oof['oof_n'])} 拍 OOF 概率后，主规则为“Sp≥0.90 时 Se 最大”；并列时优先更高 Sp，再取更高阈值。"
         f"选定完整精度阈值为 {bpoint['threshold']:.16f}，正文显示为 {number(bpoint['threshold'])}。"
         "同时预先保留 OOF Youden J 最大点和默认 0.50 点作对照。评价使用 JSON 中的完整精度值，不用排版舍入值重新计算。",
         key="workpoint_rule", source="cv/oof/binary")
    para("工作点和最终模型哈希在本轮干净 DS2 主评价前冻结，该流程仅生成一次二分类概率缓存，随后主评价的所有工作点、ROC 与完整阈值曲线均从该缓存推导。"
         "完整 DS2 阈值曲线明确标为探索性分析，不参与工作点选择；评价后没有回头补选阈值。此前探索轮次曾查看 DS2，"
         "因此本报告不把整个研究过程描述为从未接触过测试数据。",
         key="frozen_policy", source="cv/binary")

    heading("3.3  去噪器与辅助模型的记录级验证", 2)
    aux_val = "/".join(auxiliary["val_records"])
    classification_training = data.get("classification_training", data.get("classification_config"))
    must(classification_training is not None, "Current classification training metadata is required")
    cnn_training = classification_training["cnn"]
    mlp_training = classification_training["mlp"]
    dn_training = denoiser["training"]
    para(f"四分类 CNN、MLP 对照与 DnCNN 使用同一份排除 201 的辅助划分：训练 {len(auxiliary['train_records'])} 条记录、"
         f"{integer(auxiliary['train_n'])} 拍，验证 {len(auxiliary['val_records'])} 条记录、{integer(auxiliary['val_n'])} 拍；"
         f"验证记录为 {aux_val}。训练与验证记录不重叠，均不含 DS2。各模型的样本上限、最佳轮次和实际使用索引保存在训练元数据中。",
         key="auxiliary_split", source="aux_split")
    para(f"上述训练拍数为候选全集，实际训练使用了数量上限：四分类 CNN 取 {integer(cnn_training['train_n_after_cap'])} 拍、"
         f"MLP 取 {integer(mlp_training['train_n_after_cap'])} 拍，DnCNN 固定抽取 {integer(dn_training['train_n_after_cap'])} 拍；"
         f"三者验证均使用全部 {integer(auxiliary['val_n'])} 拍。四分类的 N 类上限为 18,000、V 类上限为 3,000，S/F 类保留；"
         "这些抽样只在训练记录内部进行，不把留出记录的心搏补回训练。",
         key="auxiliary_actual_training_n", source="classification_training/denoiser.training")
    para(f"去噪器借鉴图像 DnCNN 的残差学习思想[4]并改为一维卷积：输入含噪心搏，预测噪声分量，再以输入减去噪声得到重建。网络含首层卷积、四个带 BatchNorm 的隐藏卷积层和输出卷积，"
         f"共 {integer(denoiser_cost['params'])} 个参数。本轮去噪器训练和验证均排除 201，按验证 MSE 选择第 {denoiser['best_epoch']} 轮权重。"
         "训练噪声包括 NSTDB 的 bw/em/ma 以及合成白噪声和工频成分，输入 SNR 随机变化；不根据 DS2 去噪指标选择权重。",
         key="denoiser_protocol", source="denoiser/aux_split/cost")
    para("去噪器仅使用各噪声通道前 65% 物理时间段；其余后 35% 留给噪声测试。训练与验证的噪声输入固定规则保存在元数据中。"
         "训练损失在标准化域计算，验证 MSE 按还原后的物理幅值计算，二者数值尺度不同，训练曲线不能直接作等量比较。")
    table("table3", ["阶段", "操作", "输出"], architecture["denoiser"], [2.3, 9.5, 3.6],
          f"表 3  一维 DnCNN 的结构（{integer(denoiser_cost['params'])} 参数）", fs=8.0)
    figure("fig05_training_curves.png", 14.0, "图 5  排除 201 后四分类 CNN 与 DnCNN 的训练和验证曲线", "figure5")

    heading("3.4  噪声评价与传统滤波对照", 2)
    noise_metadata = noise["metadata"]
    must(int(noise_metadata['n_abnormal']) + int(noise_metadata['n_normal']) == int(noise['n']), "Noise subset counts do not close")
    para(f"噪声评价固定使用 {integer(noise['n'])} 个 DS2 心搏：保留全部 {integer(noise_metadata['n_abnormal'])} 个异常心搏，"
         f"并固定抽取 {integer(noise_metadata['n_normal'])} 个正常心搏。"
         "分别叠加 bw 基线漂移、em 电极运动和 ma 肌电噪声的 5/10/15 dB 三档干扰，"
         "同一条件的同一输入同时进入未处理、传统滤波与 DnCNN 三条支路，共九条件、三种处理。"
         f"三条支路均逐拍标准化后送入本轮最终二分类模型，统一使用冻结阈值 {number(bpoint['threshold'])}，没有按噪声档重新选阈值。",
         key="noise_protocol", source="noise/binary")
    para("该子集的异常/正常比例不同于全量 DS2，因此噪声 Macro-F1 仅用于同一条件下三种处理的配对比较，"
         "不直接与全量干净 DS2 的 Macro-F1 横向比较。")
    para("传统支路采用三阶 0.5–40 Hz 带通、60 Hz 陷波及 3 点中值滤波。噪声随机种子、抽样索引、模型哈希及各条件预测缓存随代码包保存。"
         "噪声是对冻结模型的追加固定条件评价，不用于重新训练、选择阈值或调整去噪结论。")

    heading("3.5  评价指标", 2)
    para("主指标为二分类 ROC AUC，同时报告 OOF 与 DS2。Se、Sp、Acc、Macro-F1 和异常类 F1 用于描述冻结工作点，"
         "并与全判正常基线比较。本任务阴性占约 89%，准确率不是主指标。重建指标为输入/输出 SNR、ΔSNR 和 RMSE；"
         "重建改善不等价于分类改善，二者分别解释。")
    para("Se = TP/(TP+FN)，Sp = TN/(TN+FP)，F1异常 = 2TP/(2TP+FP+FN)，Macro-F1 = (F1正常+F1异常)/2。",
         first=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=9.2)
    equation = doc.add_paragraph()
    equation.alignment = WD_ALIGN_PARAGRAPH.CENTER
    equation.paragraph_format.keep_together = True
    equation.paragraph_format.keep_with_next = True
    equation.paragraph_format.space_before = Pt(2)
    equation.paragraph_format.space_after = Pt(2)
    math = OxmlElement("m:oMath")
    math.append(math_sub(math_run("SNR"), "i"))
    math.append(math_run(" = 10 "))
    math.append(math_sub(math_run("log"), "10")); math.append(math_run("("))
    fraction = OxmlElement("m:f")
    numerator = OxmlElement("m:num")
    numerator.append(sum_over_time(math_square([beat_sample()])))
    denominator = OxmlElement("m:den")
    denominator.append(sum_over_time(math_square([math_run("("), beat_sample(), math_run("−"), beat_sample(True), math_run(")")])))
    fraction.append(numerator); fraction.append(denominator); math.append(fraction); math.append(math_run(")"))
    equation._p.append(math)
    para("其中 i 为心搏索引，t 为拍内采样点。报告的 SNR 是各拍 SNRᵢ 的算术均值，ΔSNR 为输出与输入均值之差；"
         "RMSE 则按全部心搏、全部采样点的平方误差求全局均方根。两者聚合口径不同，不能互相反推；"
         "平均 ΔSNR 为正与全局 RMSE 增大可以同时出现。", first=False, size=9.2,
         key="snr_rmse_aggregation", source="src/evaluate_final_consistency.py/snr_db/np.mean")

    heading("3.6  OpenECG 接入与部署边界", 2)
    para("软件已验证的入口是离线 WFDB 记录与端到端演示。面向 OpenECG 的实时接口、AFE/MCU 采样与通信流程仍属设计，"
         "不能据此声称完成硬件采集、实时报警或临床验证。当前零相位滤波依赖未来样本，上机需替换为因果实现并重新评价延迟与检测误差。")
    para("参数量和权重大小用于初步资源预算。INT8 体积按每参数一字节估算，尚未实际量化运行；"
         "理想每周期一次乘加的计算只能提供算术耗时下界，不能覆盖访存、算子支持、采集和通信调度。桌面 CPU 时延也不能直接换算成 MCU 实测性能。")

    heading("4  结果与分析")
    packages = environment["packages"]
    para(f"实验使用 {environment.get('platform', 'Windows')}，CPU 为 {environment['cpu_model']}；"
         f"Python {environment['python']}、PyTorch {packages['torch']}、"
         f"NumPy {packages['numpy']}、SciPy {packages['scipy']}。训练种子、线程数和复现参数记录在各实验 JSON 中，"
         "本文只采用本轮当前模型与缓存结果，不把早期探索数字并入提交稿。",
         key="environment", source="environment")

    heading("4.1  AUC 与冻结工作点", 2)
    para(f"合并 OOF AUC 为 {number(oof['oof_auc'])}，最终模型在 DS2 上 AUC 为 {number(binary['auc'])}。"
         "OOF 每折仅用 16 或 17 条记录训练，最终模型使用 21 条；较少的训练记录是 OOF 相对保守的一个合理解释。"
         "但未做只改变训练量的消融，折间记录难度、模型随机性和概率尺度也可能影响二者差异，不能将差距全部归因于训练量。",
         key="auc_results", source="binary/oof/cv")
    para(f"主阈值 {number(bpoint['threshold'])} 在 OOF 上满足 Sp {percent(oof_point['specificity'])}、Se {percent(oof_point['sensitivity'])}；"
         f"迁移至 DS2 后，Se 为 {percent(bpoint['sensitivity'])}、Sp 为 {percent(bpoint['specificity'])}、Acc 为 {percent(bpoint['accuracy'])}、"
         f"Macro-F1 为 {number(bpoint['macro_f1'])}。OOF 的 Sp≥90% 约束在 DS2 只兑现到 {percent(bpoint['specificity'])}，"
         "说明 OOF 选定阈值未保持目标特异性，提示跨记录及模型重拟合后的分数分布变化。"
         "概率校准还需检验预测概率与实际发生频率的对应关系，本实验未作专门评价；本轮没有回看 DS2 去补选阈值。",
         key="primary_results", source="binary/oof")
    labels = [("sp90_max_se", "OOF Sp≥90% 主点"), ("youden_max", "OOF Youden 点"), ("default_050", "默认点")]
    workpoint_rows = []
    for key, label in labels:
        r = binary["working_points"][key]
        workpoint_rows.append([label, number(r["threshold"]), percent(r["accuracy"]), number(r["macro_f1"]),
                               percent(r["sensitivity"]), percent(r["specificity"]), number(r["f1_abnormal"])])
    workpoint_rows.append(["全判正常", "—", percent(baseline["accuracy"]), number(baseline["macro_f1"]),
                           percent(baseline["sensitivity"]), percent(baseline["specificity"]), number(baseline["f1_abnormal"])])
    table("table4", ["工作点", "阈值", "Acc", "Macro-F1", "Se", "Sp", "F1异常"], workpoint_rows,
          [3.5, 1.6, 2.0, 2.3, 2.0, 2.0, 2.2], f"表 4  DS2 冻结工作点与全判正常基线（{integer(binary['n'])} 拍）", fs=7.8)
    para(f"DS2 阴性占 {percent(baseline['accuracy'])}，全判正常即有 {percent(baseline['accuracy'])} 的 Acc，但 Se 和异常类 F1 均为零。"
         "因此表 4 的准确率必须与 Se、Sp 和 F1 一起阅读，不能单凭 Acc 判断筛查价值。完整阈值曲线作为探索性分析保存，"
         "其中任何后验较优点都没有替换表中的 OOF 冻结工作点。", key="baseline_interpretation", source="binary")
    figure("fig06b_binary_roc.png", 13.4, "图 6  混淆矩阵、ROC 与 OOF 冻结工作点", "figure6")

    heading("4.2  当前模型的去噪与噪声鲁棒性", 2)
    para("表 5 使用排除 201 后重训的去噪器、本轮最终二分类模型及同一冻结阈值，替代上一轮全部噪声数字。"
         "本轮还修正了噪声源按通道共同物理时间先分段后拼接的实现。原噪声路径已经逐拍标准化，跨轮变化同时包含模型和噪声源划分的变化，"
         "不能归因为修复干净测试路径的归一化。以下结论直接来自九条件实测，不预设 DnCNN 必然改善分类。",
         key="noise_scope", source="noise/denoiser")
    table5_rows = []
    for kind, name in KINDS:
        for snr in (5, 10, 15):
            r = noise_by_key[kind, snr]
            table5_rows.append([f"{name}/{snr}", number(r["snr_raw"], 1), number(r["snr_denoised"], 1),
                                f"{r['snr_denoised'] - r['snr_raw']:+.1f}", number(r["raw"]["macro_f1"]),
                                number(r["classical"]["macro_f1"]), number(r["denoised"]["macro_f1"])])
    table("table5", ["噪声/SNR", "输入SNR", "输出SNR", "ΔSNR", "F1未处理", "F1传统", "F1DnCNN"],
          table5_rows, [2.7, 1.8, 1.8, 1.5, 2.6, 2.6, 2.6],
          f"表 5  冻结阈值 {number(bpoint['threshold'])} 下九条件重建 SNR 与分类 Macro-F1", fs=7.6)
    low_parts = []
    for kind, name in KINDS:
        r = noise_by_key[kind, 5]
        delta = r["denoised"]["macro_f1"] - r["raw"]["macro_f1"]
        best_method = max(("raw", "classical", "denoised"), key=lambda k: r[k]["macro_f1"])
        best_name = {"raw": "未处理", "classical": "传统滤波", "denoised": "DnCNN"}[best_method]
        low_parts.append(f"{name} F1 由 {number(r['raw']['macro_f1'])} 变为 {number(r['denoised']['macro_f1'])}"
                         f"（ΔF1 {delta:+.4f}，传统滤波 {number(r['classical']['macro_f1'])}，本条件最佳为{best_name}）")
    para("5 dB 下，" + "；".join(low_parts) + "。这些结果表明去噪收益依赖噪声类型和固定工作点，不能将个别条件的改善概括为所有低 SNR 条件均有效。",
         key="noise_low_snr", source="noise")
    reconstruction_parts, classification_parts = [], []
    for kind, name in KINDS:
        r = noise_by_key[kind, 15]
        reconstruction_parts.append(f"{name} RMSE {number(r['rmse_raw'])}→{number(r['rmse_denoised'])}，"
                                    f"ΔSNR {r['snr_denoised'] - r['snr_raw']:+.1f} dB")
        classification_parts.append(f"{name} AUC {number(r['raw']['auc'])}→{number(r['denoised']['auc'])}，"
                                    f"F1 {number(r['raw']['macro_f1'])}→{number(r['denoised']['macro_f1'])}")
    para("15 dB 的重建结果单独观察：" + "；".join(reconstruction_parts) + "。负 ΔSNR 或 RMSE 增大意味着该重建指标变差，可能包含过平滑；不能只依据输出曲线更平滑判断质量。",
         key="noise_15db_reconstruction", source="noise")
    para("同一 15 dB 条件的分类结果为：" + "；".join(classification_parts) + "。分类改善与重建退化可以同时发生，二者不能写成同一结论。完整 RMSE 与 AUC 见表 A1。",
         key="noise_15db_classification", source="noise")
    figure("fig04_denoising_example.png", 11.5, "图 7  本轮 DnCNN 在固定含噪输入上的波形重建示例", "figure7")
    figure("fig05b_denoising_snr.png", 13.0, "图 8  本轮九条件输入 输出 SNR 与 ΔSNR", "figure8")
    figure("fig08_noise_robustness.png", 12.2, "图 9  当前二分类模型冻结阈值下三种处理的 Macro-F1", "figure9")

    heading("4.3  排除 201 后的四分类探索", 2)
    cnn, mlp = classification["cnn"], classification["mlp"]
    para(f"四分类 CNN 共 {integer(four_cost['params'])} 个参数，在同一 DS2 上 Acc 为 {percent(cnn['accuracy'])}、Macro-F1 为 {number(cnn['macro_f1'])}；"
         f"MLP 对照分别为 {percent(mlp['accuracy'])} 和 {number(mlp['macro_f1'])}。两者均在排除 201 的辅助划分上重新训练，表 6 和图 10 均使用本轮预测。",
         key="four_class_results", source="classification/cost/aux_split")
    table6_rows = []
    for key, label in (("cnn", "轻量 CNN"), ("mlp", "MLP")):
        r = classification[key]
        table6_rows.append([label, percent(r["accuracy"]), number(r["macro_f1"]),
                            *[number(r["report"][c]["f1-score"]) for c in CLASSES]])
    table("table6", ["模型", "Acc", "Macro-F1", "N F1", "S F1", "V F1", "F F1"], table6_rows,
          [2.7, 2.4, 2.7, 2.0, 2.0, 2.0, 2.0], f"表 6  排除 201 后四分类探索结果（DS2 {integer(binary['n'])} 拍）", fs=7.8)
    para("单拍形态对 S 与 N 的区分有限，F 类样本也较少。缺少 RR/节律信息可能是误分类原因之一，但本实验没有做 RR 特征消融，"
         "因此不能将某类错误完全归因于未使用 RR。四分类结果用于探索亚类差异，不作为诊断或二分类筛查指标。")
    figure("fig06_confusion_matrices.png", 13.7, "图 10  本轮四分类 CNN 与 MLP 在 DS2 上的混淆矩阵", "figure10")

    heading("4.4  冻结主点的逐记录误差", 2)
    records = {str(r["record"]): r for r in data["per_record"]}
    record_order = ["232", "200", "213", "233", "210", "222"]
    table7_rows = []
    for record in record_order:
        r = records[record]
        table7_rows.append([record, integer(r["n"]), integer(r["abnormal"]), integer(get_record(r, "fn", "false_negative")),
                            percent(r["sensitivity"]), percent(r["specificity"])])
    table("table7", ["记录", "心搏数", "异常数", "漏检数", "Se", "Sp"], table7_rows,
          [1.8, 2.6, 2.6, 2.6, 2.9, 2.9], f"表 7  主阈值 {number(bpoint['threshold'])} 下部分代表性记录的异常检出", fs=8.0)
    para("表 7 固定展示六条记录，便于对照高检出、较多漏检与低特异性等不同情况；该列表并非随机抽样，也未按异常数排序。"
         "全部 22 条 DS2 记录的结果见仓库 results/workpoint_cv_v2/ds2_per_record_primary.json。", first=False, size=8.8)
    para(f"主点在 DS2 共检出异常 {integer(bpoint['tp'])} 拍、漏检 {integer(bpoint['fn'])} 拍，同时产生 {integer(bpoint['fp'])} 拍正常心搏误报。"
         "表 7 的差异说明总体 AUC 或平均敏感度不能替代逐记录评价。单拍形态、节律信息缺失、噪声以及记录间分布差异均可能参与形成误差，"
         "本实验尚未通过消融将其影响分开。",
         key="error_counts", source="binary/per_record")

    heading("4.5  四分类演示与模型开销", 2)
    pairs = demo.get("truth_prediction_pairs")
    if pairs is None:
        pairs = [{"true": t, "pred": CLASSES[int(p)] if not isinstance(p, str) else p}
                 for t, p in zip(demo["true_cls"], demo["pred"])]
    truth_counts = Counter(str(p["true"]) for p in pairs)
    pred_counts = Counter(str(p["pred"]) for p in pairs)
    mismatches = [p for p in pairs if p["true"] != p["pred"]]
    truth_text = "/".join(f"{c} {truth_counts[c]}" for c in CLASSES)
    pred_text = "/".join(f"{c} {pred_counts[c]}" for c in CLASSES)
    errors = Counter(f"{p['true']}→{p['pred']}" for p in mismatches)
    error_text = "、".join(f"{k} {v} 拍" for k, v in errors.items()) if errors else "无匹配标签错分"
    detection = demo["detection"]
    para(f"演示加载本轮四分类探索权重（{integer(four_cost['params'])} 参数），不是冻结二分类权重；仅输入波形，不使用 RR。"
         f"记录 {demo['record']} 的片段共切出 {len(pairs)} 拍，真值为 {truth_text}，预测为 {pred_text}；"
         f"错分 {len(mismatches)} 拍（{error_text}）。图中圆点表示预测类别，真值与预测不一致处单独标注。"
         f"记录 100 的 R 峰检测 Se={number(detection['se'])}、PPV={number(detection['ppv'])}，仅说明该记录在所用容差下的结果，不能外推至所有节律或噪声。",
         key="demo_results", source="demo/cost")
    para(f"记录 {demo['record']} 属于 DS1 训练/验证材料，该片段仅用于检查离线处理、模型加载与可视化功能，"
         "不作为独立分类性能或泛化验证。片段预测全部符合真值也不能代替 DS2 的独立记录评价。")
    figure("fig11_openecg_demo.png", 15.0, "图 11  本轮四分类权重的端到端演示与真值预测对照", "figure11")
    table8_rows = [[m["name"], integer(m["params"]), f"{m['fp32_kib']:.1f} KiB", f"{m['int8_kib']:.1f} KiB",
                    f"{m['latency_ms']:.3f} ms"] for m in cost_models]
    table("table8", ["模型", "参数量", "FP32", "INT8估算", "CPU时延/拍"], table8_rows,
          [3.8, 2.6, 3.0, 3.0, 3.4], "表 8  当前权重的参数量与桌面 CPU 推理开销", fs=8.0)
    para(f"时延以 {cost['threads']} 个 CPU 线程、批大小 {cost['batch']}、重复 {cost['repetitions']} 次的设置重新测量；"
         "表中为重复批次耗时中位数除以批大小得到的每拍 FP32 摊销时延，不代表单拍在线延迟。FP32 和 INT8 列只计算权重载荷，"
         "不包含激活、框架及通信内存；INT8 尚未实际量化运行。上述结果用于资源预算，不能声称已在 Cortex-M 或其他 MCU 上完成实时部署。",
         key="cost_protocol", source="cost")
    para("表 8 的计时范围只含网络前向，不含滤波、R 峰检测、逐拍标准化和 I/O，因此也不代表完整端到端处理时延。")

    heading("4.6  硬件链路仿真（软件在环）", 2)
    para(f"使用 DS2 全部 {integer(sil['n_beats'])} 拍进行软件在环（SIL）比较。设备通路依次为："
         "AFE（0.5 Hz 二阶高通与 40 Hz 四阶低通，均为因果滤波）→11 bit、±5 mV 量化"
         f"（均匀抖动 ±0.5 LSB，LSB={sil['adc']['lsb_mV'] * 1000:.2f} µV）→20 采样/包、1% 设定丢包率、20 ms 缓冲"
         "→接收重组→现有数字预处理与逐拍标准化→同一冻结二分类模型。"
         "仅量化+链路分支跳过 AFE，其余处理相同。"
         f"四条通路均沿用阈值 {number(sil['threshold'])}，未在设备通路上重新选点。",
         key="sil_method", source="sil/afe.py,adc.py,link.py,run_sil.py; results/sil/sil_metrics.json")
    figure("sil_afe.png", 14.1, "图 12  软件在环 AFE 幅频响应与设备链路波形", "figure12")
    para("理想通路直接使用现有数字预处理；缺包以接收端上一包末样本保持，20 ms 缓冲仅作为传输设定记录，"
         "重组按时间戳定位，不额外平移信号。时延对齐分支以理想与设备波形的互相关估计每条记录的整数样本偏移，"
         "用于分析前端相位影响；实际设备需由时间戳或检测器完成补偿。"
         f"理想通路与原冻结缓存逐拍一致，最大概率偏差为 {sil['ideal_matches_frozen_cache_max_dev']:.1e}（浮点精度量级）。",
         key="sil_cache_and_alignment", source="sil/run_sil.py; results/sil/sil_metrics.json")
    para("本节为额外的四通路推理与理想通路复核。时延对齐按“正 lag 表示设备信号滞后”实现："
         "对设备信号左移相应样本数，边界用端值填充；该分支只用于分析前端相位影响，不参与阈值选择。"
         "原干净 DS2 主评价的一次推理审计、模型、阈值和缓存均保持不变，未据仿真结果重新选点。")
    sil_rows = []
    for label, key in [("理想", "ideal"), ("仅量化+链路", "quant_link"),
                       ("设备链路", "sil"), ("设备链路+时延对齐", "sil_aligned")]:
        row = sil[key]
        sil_rows.append([label, number(row["auc"]), percent(row["accuracy"]), number(row["macro_f1"]),
                         percent(row["sensitivity"]), percent(row["specificity"])])
    table("table9", ["通路", "AUC", "Acc", "Macro-F1", "Se", "Sp"], sil_rows,
          [4.8, 2.1, 2.2, 2.4, 2.3, 2.3], "表 9  DS2 全量软件在环四通路评价（同一冻结模型与阈值）", fs=7.8)
    figure("sil_roc.png", 10.5, "图 13  软件在环四通路 ROC 对照", "figure13")
    ideal, quant_link, device, aligned = sil["ideal"], sil["quant_link"], sil["sil"], sil["sil_aligned"]
    para(f"正确整数对齐后，AUC 相对理想通路变化为 {aligned['auc'] - ideal['auc']:+.4f}，"
         f"由 {number(ideal['auc'])} 降至 {number(aligned['auc'])}，未恢复理想通路的排序能力。"
         "组件消融中，仅做 11 bit 量化与 1% 丢包时，"
         f"AUC 为 {number(quant_link['auc'])}（{quant_link['auc'] - ideal['auc']:+.4f}），"
         f"Se 为 {percent(quant_link['sensitivity'])}，与理想通路的 {number(ideal['auc'])} / {percent(ideal['sensitivity'])} 基本持平；"
         f"加入 AFE 后 AUC 降至 {number(device['auc'])}（{device['auc'] - ideal['auc']:+.4f}），"
         f"Se 降至 {percent(device['sensitivity'])}。性能差异主要随 AFE 的引入出现，可能与额外的幅频、相频响应及波形改变有关。"
         "该对照同时改变了高通、低通及因果相位响应，尚未分别确定各因素的贡献，不能将损失单独归因于串联低通。"
         f"整数时延对齐后 AUC 仍为 {number(aligned['auc'])}，说明本次固定整数平移不足以消除剩余差异。",
         key="sil_auc_conclusion", source="results/sil/sil_metrics.json")
    para(f"不补偿时，AUC 变化为 {device['auc'] - ideal['auc']:+.4f}，"
         f"Se 变化为 {(device['sensitivity'] - ideal['sensitivity']) * 100:+.1f} 个百分点。"
         f"互相关测得平均偏移 {sil['mean_lag_samples']:.2f} 个采样，约 {sil['lag_ms']:.1f} ms；"
         "设备侧需由时间戳或检测器处理相位偏移。但本实验的整数平移未恢复 AUC，"
         "说明该通路的影响不能仅用固定延迟解释，还需考察因果滤波带来的波形变化。",
         key="sil_delay_conclusion", source="results/sil/sil_metrics.json")
    para(f"即使对齐，冻结阈值下 Se 仍由 {percent(ideal['sensitivity'])} 变为 {percent(aligned['sensitivity'])}，"
         f"Sp 由 {percent(ideal['specificity'])} 变为 {percent(aligned['specificity'])}，"
         f"阳性判定比例由 {percent(sil['probability_summary']['ideal']['predicted_positive_rate'])} 变为 "
         f"{percent(sil['probability_summary']['sil_aligned']['predicted_positive_rate'])}，"
         "显示分数分布和固定工作点发生迁移。设备链路上的工作点需在另设的训练侧开发数据上重新选择并独立验证，"
         "与 4.1 的目标特异性未保持现象相呼应；概率校准仍需专门评价。本轮继续保留冻结阈值，不使用 DS2 补选工作点。",
         key="sil_calibration_conclusion", source="results/sil/sil_metrics.json")
    para("仿真边界：源数据已是 360 Hz 数字记录，未模拟采样前混叠；丢包率、包长与缓冲时间等均为设定值，"
         "并非设备实测。该实验属于软件在环仿真，数字预处理仍按离线方式运行，不是实物联调，"
         "也不构成端到端硬件实时性证明。",
         key="sil_limits", source="sil/README.md,afe.py,link.py")

    heading("4.7  参考 OpenECG 思路的自定义合成信号流程验证", 2)
    prob = openecg["prob_abnormal"]
    para("上游 OpenECG 仓库提供两个 Jupyter notebook：分别用 NeuroKit2 与 numpy 高斯模型生成模拟心电。"
         "其中 NeuroKit2 示例调用 nk.ecg_process 完成预处理和 R 峰检测；上游尚未包含本项目的训练分类器和硬件采集实现[6]。"
         "本项目的 openecg_ext/analysis 模块参考其高斯 P/QRS/T 建模思路，使用修改后的幅值、宽度和时序参数生成自定义信号，"
         "并未直接读取上游运行输出。例如 R 波高斯宽度由 0.03 s 改为 0.008 s，幅值由 1.5 改为 1.0；合成幅值未作物理标定。"
         "pipeline.py 完成 1000→360 Hz 重采样、0.5–40 Hz 零相位带通与 60 Hz 陷波、R 峰检测及 256 点切分，"
         "run_demo.py 加载与 4.6 相同的冻结二分类模型与阈值。",
         key="openecg_integration", source="openecg_ext/analysis/*.py; results/openecg_demo/openecg_demo.json")
    figure("openecg_sim.png", 14.0,
           "图 14  自定义合成信号的三段处理：模拟波形、滤波与 R 峰、切分心搏与分类", "figure14")
    para(f"在 {openecg['duration_s']:.0f} s、{openecg['heart_rate_bpm_target']:.0f} bpm 的模拟片段上，"
         f"R 峰全部检出（{openecg['n_peaks']} 个，RR = {openecg['mean_rr_s']:.3f} s，"
         f"实测心率 {openecg['heart_rate_bpm_measured']:.1f} bpm）；{openecg['n_beats']} 拍的异常概率为 "
         f"{min(prob):.2f}–{max(prob):.2f}，均低于阈值 {number(openecg['threshold'])}，"
         f"判为异常 {openecg['n_abnormal']} 拍。该结果说明自定义合成输入下的数据转换、检测、切分和模型调用流程能够运行，"
         "不能作为上游原始波形的直接接入验证。合成波形不用于分类性能评价，性能仍以 DS2 的 49,705 个真实心搏为准。",
         key="openecg_result", source="results/openecg_demo/openecg_demo.json")

    heading("5  讨论与局限")
    para(f"本轮主要结果是 OOF AUC {number(oof['oof_auc'])} 与 DS2 AUC {number(binary['auc'])}，同时必须保留 OOF 与 DS2 的差异。"
         "较少的每折训练记录使 OOF 结果可能更保守，记录难度和概率尺度也会变化；没有训练量消融时，不能把最终模型较高的 AUC 解释为单一因素的因果收益。"
         "准确率受阴性比例影响，冻结工作点仍存在大量误报和漏检。")
    para(f"OOF 选定阈值在 DS2 上仅达到 Sp {percent(bpoint['specificity'])}，未保持 Sp≥90% 的目标，"
         "提示跨记录及模型重拟合后的分数分布变化。该工作点现象与概率校准评价应予区分；"
         "后续如研究概率校准，需专门评价预测概率与实际发生频率的对应关系，并另设训练侧校准方案与独立评价数据。"
         "本轮保留工作点冻结、一次干净 DS2 推理缓存及未回看补选阈值的完整记录。")
    improvements = [r["denoised"]["macro_f1"] - r["raw"]["macro_f1"] for r in noise_rows]
    wins = sum(v > 0 for v in improvements); losses = sum(v < 0 for v in improvements)
    para(f"去噪分类 Macro-F1 相对未处理在九条件中改善 {wins} 个、下降 {losses} 个，ΔF1 范围为 {min(improvements):+.4f} 至 {max(improvements):+.4f}。"
         "这描述了当前冻结模型与阈值的观测，不证明去噪对所有记录或噪声均有效。重建 SNR、RMSE 与分类 AUC、F1 必须分别解释；"
         "传统滤波也可能在部分条件胜出。",
         key="noise_overall", source="noise")
    para("局限包括单库、单导联和受试者规模有限；专家 R 峰切分弱化了真实检测误差；OOF 各折较小且存在记录间异质性；"
         "四分类缺少系统的节律特征消融；模型尚无跨数据库、硬件实时性或临床评价。记录 201 已排除，但这不能替代外部数据验证。"
         "报告中的 MCU/BLE 均为候选设计，AFE 与传输链路仅做软件在环仿真，零相位处理仍受离线条件限制。")

    heading("6  结论")
    para(f"在统一排除记录 201 后，完成记录级 OOF 选点、二分类冻结评价、四分类重训、DnCNN 重训和固定条件噪声对照。"
         f"二分类 OOF AUC 为 {number(oof['oof_auc'])}，DS2 AUC 为 {number(binary['auc'])}；主阈值 {number(bpoint['threshold'])} 的 DS2 "
         f"Se/Sp/Acc/Macro-F1 分别为 {percent(bpoint['sensitivity'])}/{percent(bpoint['specificity'])}/{percent(bpoint['accuracy'])}/{number(bpoint['macro_f1'])}。"
         f"软件在环整数时延对齐后的 AUC 为 {number(sil['sil_aligned']['auc'])}，仍低于理想通路且冻结工作点发生迁移，"
         "提示单一时延补偿不足，设备链路上的工作点需要独立验证，并进一步开展实物联调。"
         "当前证据支持离线软件实验与工作点迁移现象的分析，不支持实时硬件部署或临床诊断结论。",
         key="conclusion", source="binary/oof/sil")

    heading("参考文献")
    for index, reference in enumerate(references, 1):
        if isinstance(reference, str):
            text = reference
        else:
            text = reference.get("text") or reference.get("citation")
            if text is None:
                text = ". ".join(str(reference[k]) for k in ("authors", "title", "year") if k in reference)
                locator = reference.get("doi") or reference.get("url")
                if locator:
                    text += ". " + str(locator)
        text = str(text)
        if text.startswith(f"[{index}]"):
            text = text[len(f"[{index}]"):].strip()
        para(f"[{index}] {text}", first=False, size=8.7, before=2, after=2)

    heading("数据与实现资源")
    resource_rows = [
        ["MIT-BIH", "https://physionet.org/content/mitdb/1.0.0/"],
        ["NSTDB", "https://physionet.org/content/nstdb/1.0.0/"],
        ["OpenECG", "https://github.com/jwlee95/OpenECG"],
        ["PyTorch", "https://pytorch.org/"],
    ]
    table("table10", ["资源", "链接"], resource_rows, [3.2, 13.2], "表 10  数据集与实现资源链接", fs=7.7, left_columns=(1,))

    heading("附录 A  补充结果")
    para("正文表 5 汇总重建 SNR 与三种处理的 Macro-F1；九条件的重建 RMSE 与分类 AUC 汇总如下。"
         "所有结果仅用于课程学习和研究分析，不构成临床诊断建议。")
    appendix_rows = []
    for kind, name in KINDS:
        for snr in (5, 10, 15):
            r = noise_by_key[kind, snr]
            appendix_rows.append([f"{name}/{snr}", number(r["rmse_raw"]), number(r["rmse_denoised"]),
                                  number(r["raw"]["auc"]), number(r["denoised"]["auc"])])
    table("tableA1", ["噪声/SNR", "RMSE未处理", "RMSE DnCNN", "AUC未处理", "AUC DnCNN"], appendix_rows,
          [3.2, 3.3, 3.3, 3.3, 3.3], "表 A1  九条件的重建 RMSE 与分类 AUC", fs=7.8)

    heading("附录 B  代码与数据仓库")
    repo_url = "https://github.com/HZH507/openecg-beat-screening"
    repo_paragraph = para("代码、模型权重、冻结预测缓存与本报告所用图表见 GitHub 仓库：",
                          align=WD_ALIGN_PARAGRAPH.LEFT)
    relationship = repo_paragraph.part.relate_to(repo_url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship)
    hyperlink.set(qn("w:history"), "1")
    link_run = repo_paragraph.add_run(repo_url)
    font(link_run)
    link_run.font.color.rgb = RGBColor(0x05, 0x63, 0xC1)
    link_run.font.underline = True
    hyperlink.append(link_run._r)
    repo_paragraph._p.append(hyperlink)
    font(repo_paragraph.add_run("。"))
    para("其中 openecg_ext/ 为 4.7 节参考 OpenECG 思路的自定义合成信号流程模块，sil/ 为 4.6 节的软件在环链路仿真；"
         "数据集 MIT-BIH 与 NSTDB 不随仓库分发，获取方式见仓库内 data/README.md。"
         "已有缓存可用于核验结果与重建报告，无需重新训练或推理；具体步骤与文件哈希见仓库说明。")

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    manifest_path = final / "report_table_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    print("table manifest", manifest_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--data", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--cover-from", "--cover-source", dest="cover_from", type=Path,
                    help="Optional local DOCX whose cover is preserved; omitted means blank identity fields")
    args = ap.parse_args()
    root = args.root.resolve()
    data_path = args.data or root / "results" / "report_final" / "report_data.json"
    out = args.output or root / "report" / "ECG_课程报告_标准版.docx"
    build(root, data_path, out, cover_from=args.cover_from)


if __name__ == "__main__":
    main()
