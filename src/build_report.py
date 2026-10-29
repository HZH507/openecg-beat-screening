"""Build the Chinese course report as a formatted DOCX."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import (WD_ALIGN_PARAGRAPH, WD_LINE_SPACING,
                            WD_TAB_ALIGNMENT, WD_TAB_LEADER)
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
REPORT = ROOT / "report"
REPORT.mkdir(exist_ok=True)

CN_BODY = "宋体"
CN_HEAD = "黑体"
EN_BODY = "Times New Roman"


# ---------------------------------------------------------------------------
# Low-level formatting helpers
# ---------------------------------------------------------------------------
def set_run_font(run, size=10.5, bold=False, color=None, cn=CN_BODY,
                 en=EN_BODY, italic=False):
    run.font.name = en
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), cn)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def body(text, first_line=True, size=10.5, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         space_before=0, space_after=0, line=1.5, indent_left=0):
    p = document.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = align
    pf.line_spacing = line
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    if first_line:
        pf.first_line_indent = Cm(0.74)
    if indent_left:
        pf.left_indent = Cm(indent_left)
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold)
    return p


def rich_body(lead, rest, size=10.5):
    p = document.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.line_spacing = 1.5
    pf.first_line_indent = Cm(0.74)
    r1 = p.add_run(lead)
    set_run_font(r1, size=size, bold=True)
    r2 = p.add_run(rest)
    set_run_font(r2, size=size)
    return p


def h1(text, page_break=False):
    p = document.add_paragraph(style="Heading 1")
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    if page_break:
        p.paragraph_format.page_break_before = True
    r = p.add_run(text)
    set_run_font(r, size=16, bold=True, cn=CN_HEAD)
    return p


def h2(text):
    p = document.add_paragraph(style="Heading 2")
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, size=13, bold=True, cn=CN_HEAD)
    return p


def h3(text):
    p = document.add_paragraph(style="Heading 3")
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, size=11, bold=True, cn=CN_HEAD)
    return p


def caption(text, before=2, after=6, keep=False):
    p = document.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.keep_with_next = keep
    r = p.add_run(text)
    set_run_font(r, size=9)
    return p


def bullet(text):
    p = document.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing = 1.4
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    set_run_font(r, size=10.5)
    return p


def add_figure(name, width_cm, cap):
    p = document.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG / name), width=Cm(width_cm))
    caption(cap, before=1, after=8)


def _set_cell_border(cell, **kwargs):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn("w:tcBorders"))
    if tcBorders is None:
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        if edge not in kwargs:
            continue
        edge_data = kwargs.get(edge)
        tag = "w:{}".format(edge)
        element = tcBorders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            tcBorders.append(element)
        for key in ["sz", "val", "color", "space"]:
            if key in edge_data:
                element.set(qn("w:{}".format(key)), str(edge_data[key]))


def _shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def table(headers, rows, widths, cap, font_size=8.2, header_fill="DCE6F1",
          first_col_bold=False, align_center=True):
    caption(cap, before=8, after=3, keep=True)
    t = document.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    t.allow_autofit = False
    # three-line table borders
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, sz in (("top", 12), ("bottom", 12), ("insideH", 4)):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:color"), "000000")
        borders.append(e)
    for edge in ("left", "right", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "nil")
        borders.append(e)
    tblPr.append(borders)
    hdr = t.rows[0]
    trPr = hdr._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)
    for j, (text, width) in enumerate(zip(headers, widths)):
        cell = hdr.cells[j]
        cell.width = Cm(width)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        _shade(cell, header_fill)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(text)
        set_run_font(r, size=font_size, bold=True)
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        for j, (text, width) in enumerate(zip(row, widths)):
            cell = cells[j]
            cell.width = Cm(width)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = (WD_ALIGN_PARAGRAPH.CENTER if align_center
                           else WD_ALIGN_PARAGRAPH.LEFT)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(str(text))
            set_run_font(r, size=font_size,
                         bold=(first_col_bold and j == 0))
    for ri, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        cant.set(qn("w:val"), "true")
        trPr.append(cant)
        if ri < len(t.rows) - 1:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True
    document.add_paragraph().paragraph_format.space_after = Pt(0)
    return t


def add_page_number(paragraph, prefix="第 ", suffix=" 页"):
    if prefix:
        r = paragraph.add_run(prefix)
        set_run_font(r, 9)
    run = paragraph.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = "PAGE"
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t")
    t.text = "1"
    fldChar3 = OxmlElement("w:fldChar")
    fldChar3.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(t)
    run._r.append(fldChar3)
    set_run_font(run, 9)
    if suffix:
        r2 = paragraph.add_run(suffix)
        set_run_font(r2, 9)


def set_page_number_start(section, start=1):
    sectPr = section._sectPr
    pgNumType = sectPr.find(qn("w:pgNumType"))
    if pgNumType is None:
        pgNumType = OxmlElement("w:pgNumType")
        sectPr.append(pgNumType)
    pgNumType.set(qn("w:start"), str(start))


# ---------------------------------------------------------------------------
# Document setup, cover and TOC
# ---------------------------------------------------------------------------
document = Document()

# A4, generous but printable margins
sec = document.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.4)
sec.bottom_margin = Cm(2.3)
sec.left_margin = Cm(2.7)
sec.right_margin = Cm(2.2)
sec.header_distance = Cm(1.2)
sec.footer_distance = Cm(1.1)
sec.different_first_page_header_footer = True

styles = document.styles
normal = styles["Normal"]
normal.font.name = EN_BODY
normal._element.rPr.rFonts.set(qn("w:eastAsia"), CN_BODY)
normal.font.size = Pt(10.5)
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(0)
for sname, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11)):
    st = styles[sname]
    st.font.name = CN_HEAD
    st._element.rPr.rFonts.set(qn("w:eastAsia"), CN_HEAD)
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)


def cover():
    for _ in range(3):
        document.add_paragraph()
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("海 南 大 学")
    set_run_font(r, 26, bold=True, cn=CN_HEAD)
    p.paragraph_format.space_after = Pt(4)

    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("《生物医学传感检测与仪器》课程项目报告")
    set_run_font(r, 13, cn=CN_HEAD)

    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(50)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run("面向 OpenECG 的 ECG 心搏异常筛查")
    set_run_font(r, 24, bold=True, cn=CN_HEAD)
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("与深度去噪方法研究")
    set_run_font(r, 24, bold=True, cn=CN_HEAD)

    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    r = p.add_run("ECG Beat Abnormality Screening and Deep Denoising for OpenECG")
    set_run_font(r, 11, color=(80, 80, 80), italic=True)

    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(35)
    p.paragraph_format.space_after = Pt(20)
    r = p.add_run("OpenECG：开源心电采集与 Python 上位机")
    set_run_font(r, 10.5, color=(80, 80, 80))

    info = [
        ("姓    名", "________________"),
        ("学    号", "________________"),
        ("专    业", "生物医学工程"),
        ("指导教师", "________________"),
        ("完成日期", "2026 年 10 月"),
    ]
    t = document.add_table(rows=0, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for k, v in info:
        cells = t.add_row().cells
        cells[0].width = Cm(3.4)
        cells[1].width = Cm(7.2)
        p = cells[0].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = p.add_run(k + "：")
        set_run_font(r, 11, cn=CN_HEAD)
        p = cells[1].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(v)
        set_run_font(r, 11)
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(60)
    r = p.add_run("2026 年 10 月")
    set_run_font(r, 11)


def section_break_with_header():
    global sec
    sec = document.add_section(WD_SECTION.NEW_PAGE)
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.4)
    sec.bottom_margin = Cm(2.3)
    sec.left_margin = Cm(2.7)
    sec.right_margin = Cm(2.2)
    sec.header_distance = Cm(1.2)
    sec.footer_distance = Cm(1.1)
    sec.different_first_page_header_footer = False
    set_page_number_start(sec, 1)
    # header
    hp = sec.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = hp.add_run("海南大学 · 生物医学传感检测与仪器")
    set_run_font(r, 9, color=(100, 100, 100))
    # footer
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(fp)


TOC_ENTRIES = [
    (0, "摘要", "摘要"),
    (0, "1  绪论", "1  绪论"),
    (1, "1.1  项目背景与意义", "1.1  项目背景与意义"),
    (1, "1.2  OpenECG 开源项目", "1.2  OpenECG 开源项目"),
    (1, "1.3  相关工作与研究现状", "1.3  相关工作与研究现状"),
    (1, "1.4  本文主要工作", "1.4  本文主要工作"),
    (0, "2  数据与实验设计", "2  数据与实验设计"),
    (1, "2.1  数据集与开源资源", "2.1  数据集与开源资源"),
    (1, "2.2  MIT-BIH 心律失常数据库", "2.2  MIT-BIH 心律失常数据库"),
    (1, "2.3  MIT-BIH 噪声压力测试数据库", "2.3  MIT-BIH 噪声压力测试数据库"),
    (1, "2.4  受试者独立的数据划分", "2.4  受试者独立的数据划分"),
    (1, "2.5  预处理与心搏切分", "2.5  预处理与心搏切分"),
    (0, "3  方法", "3  方法"),
    (1, "3.1  系统总体结构", "3.1  系统总体结构"),
    (1, "3.2  轻量一维卷积分类器", "3.2  轻量一维卷积分类器"),
    (1, "3.3  DnCNN 心搏去噪器", "3.3  DnCNN 心搏去噪器"),
    (1, "3.4  传统滤波基线", "3.4  传统滤波基线"),
    (1, "3.5  评价指标", "3.5  评价指标"),
    (1, "3.6  OpenECG 集成与边缘部署", "3.6  OpenECG 集成与边缘部署"),
    (0, "4  实验与结果", "4  实验与结果"),
    (1, "4.1  实验环境与实现细节", "4.1  实验环境与实现细节"),
    (1, "4.2  数据统计与预处理结果", "4.2  数据统计与预处理结果"),
    (1, "4.3  正常/异常筛查结果", "4.3  正常/异常筛查结果"),
    (1, "4.4  心搏去噪结果", "4.4  心搏去噪结果"),
    (1, "4.5  噪声条件下的分类鲁棒性", "4.5  噪声条件下的分类鲁棒性"),
    (1, "4.6  四分类探索与误差分析", "4.6  四分类探索与误差分析"),
    (1, "4.7  OpenECG 端到端演示", "4.7  OpenECG 端到端演示"),
    (1, "4.8  计算复杂度与硬件可行性", "4.8  计算复杂度与硬件可行性"),
    (1, "4.9  讨论与局限性", "4.9  讨论与局限性"),
    (0, "5  结论与展望", "5  结论与展望"),
    (0, "参考文献", "参考文献"),
    (0, "附录 A  代码、模型与链接", "附录 A  代码、模型与链接"),
]


def toc(pages: dict[str, int] | None = None):
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(14)
    p.paragraph_format.page_break_before = True
    r = p.add_run("目  录")
    set_run_font(r, 18, bold=True, cn=CN_HEAD)

    for level, title, key in TOC_ENTRIES:
        q = document.add_paragraph()
        q.paragraph_format.line_spacing = 1.12
        q.paragraph_format.space_before = Pt(0)
        q.paragraph_format.space_after = Pt(0)
        q.paragraph_format.left_indent = Cm(0.0 if level == 0 else 0.8)
        q.paragraph_format.tab_stops.add_tab_stop(
            Cm(15.8), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        r = q.add_run(title)
        set_run_font(r, 10.5, bold=(level == 0))
        r = q.add_run("\t" + (str(pages.get(key, "00")) if pages else "00"))
        set_run_font(r, 10.5, bold=(level == 0))


def abstract():
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run("摘  要")
    set_run_font(r, 16, bold=True, cn=CN_HEAD)

    body(
        "心电图（ECG）是心血管疾病筛查中最常用的无创生理信号。可穿戴单导联 ECG 在 "
        "长期监测中容易受到基线漂移、电极运动、肌电干扰和工频噪声影响，且异常心搏在 "
        "全部心搏中占比很低，直接分类容易出现“高准确率、低召回率”的假象。本项目以 "
        "OpenECG 开源心电监测框架为宿主，完成一个可独立运行的 ECG 软件模块：以 "
        "0.5–40 Hz 零相位带通滤波和 60 Hz 陷波完成预处理，以 R 峰为中心提取 256 点 "
        "心搏；设计仅 9 058 个参数的轻量一维卷积网络（1D CNN）完成正常/异常二分类筛查；"
        "设计 46 049 个参数的一维 DnCNN 完成心搏级去噪。",
        first_line=True,
    )
    body(
        "实验采用 MIT-BIH 心律失常数据库，并按 de Chazal 的 DS1/DS2 记录级协议划分，"
        "在 49 705 个独立测试心搏上，二分类模型准确率为 93.17%、Macro-F1 为 0.763、"
        "AUC 为 0.829；阈值取 0.5 时特异性 99.7%、敏感度 40.1%，阈值降至 0.40 时"
        "敏感度提高到 80.8%、特异性 76.7%，可根据筛查场景选择工作点。在 MIT-BIH "
        "噪声压力测试数据库的 5 dB 基线漂移、电极运动和肌电干扰下，DnCNN 将二分类 "
        "Macro-F1 分别由 0.505、0.584、0.329 提升到 0.663、0.629、0.591。四分类探索"
        "表明，跨患者条件下 S 类（室上性异位）和 F 类（融合波）仍难以可靠识别，"
        "本文对此进行了误差分析。模型 FP32 体积约 35 KB，在普通 CPU 上单心搏推理约 "
        "0.11 ms，具备嵌入 OpenECG 硬件端或上位机的可行性。",
        first_line=True,
    )
    p = document.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    r = p.add_run("关键词：")
    set_run_font(r, 10.5, bold=True, cn=CN_HEAD)
    r = p.add_run("心电图；深度学习；心搏分类；DnCNN；OpenECG；MIT-BIH")
    set_run_font(r, 10.5)


def intro():
    h1("1  绪论", page_break=True)
    h2("1.1  项目背景与意义")
    body(
        "心血管疾病是全球范围内最主要的死亡原因之一。心电图通过体表电极记录心脏"
        "电活动，具有无创、低成本、可重复等优点，是心律失常、心肌缺血等疾病筛查和"
        "随访的基础检查。Holter 和可穿戴心电贴片可以连续记录数小时到数天的单导联"
        "或双导联 ECG，但数据量远超人眼可复核的范围，自动分析因而成为可穿戴 ECG "
        "落地的关键环节。"
    )
    body(
        "自动分析面临三个现实困难。第一，采集环境不可控：基线漂移、电极接触不良、"
        "身体运动、肌电活动和工频干扰都会改变波形形态；第二，类别极不平衡：正常心搏"
        "通常占 90% 以上，室上性早搏、室性早搏和融合波只占很小比例；第三，跨患者"
        "泛化困难：不同人的 QRS 形态、导联位置和基础节律差异很大，若按心搏随机划分"
        "训练/测试集，同一患者的心搏会同时出现在两侧，指标会被明显高估。"
    )
    body(
        "深度学习方法可以直接从原始波形中学习形态特征，减少手工特征设计，近年来在"
        "ECG 分类、去噪和端到端诊断方面取得了显著进展。但要把实验室指标转化为可用"
        "的嵌入式功能，还需要同时回答三个问题：模型是否足够小、在噪声下是否稳定、"
        "以及能否嵌入 OpenECG 这样的开源采集链路。本项目围绕这三个问题展开。"
    )

    h2("1.2  OpenECG 开源项目")
    body(
        "OpenECG 是一个面向心电监测的开源软硬件项目，目标是用开源硬件和 Python "
        "构建从电极、模拟前端、微控制器采集到上位机显示与分析的心电监测系统。"
        "其代码仓库给出了系统总体框架和基于 NeuroKit2 的 ECG 波形绘制示例，"
        "硬件资料和更多功能仍在持续完善中。对本课程项目而言，OpenECG 的价值在于"
        "提供了一个明确的“宿主系统”：它定义了数据从哪里来、以什么格式进入上位机、"
        "以及结果如何显示；我们不需要重新设计模拟前端，而是把重点放在一个可插拔的"
        "AI 分析模块上。"
    )
    body(
        "因此，本项目的定位是：以 OpenECG 的采集/显示链路为宿主，新增"
        "“心搏切分—深度去噪—异常筛查”软件功能，并给出该模块在 OpenECG 硬件端的"
        "资源评估。报告中的全部实验在 PC 上完成，硬件端只做接口和复杂度分析，"
        "不虚构未实际打样的硬件测试结果。"
    )

    h2("1.3  相关工作与研究现状")
    body(
        "AAMI EC57 将心搏分为 N（正常及束支阻滞等）、S（室上性异位）、V（室性异位）、"
        "F（融合）和 Q（未知/起搏等）五类，是心搏分类最常用的评价规范。de Chazal 等"
        "在 2004 年提出基于形态和 RR 间期特征的支持向量机方法，并给出按记录划分的"
        "DS1/DS2 协议，成为后续跨患者心搏分类的基准。此后，研究者陆续引入"
        "一维卷积网络、残差网络、循环网络和注意力机制，在 MIT-BIH 上报告了较高的"
        "整体准确率；但不同论文的数据划分、类别合并和评价指标并不统一，直接横向"
        "比较需要谨慎。"
    )
    body(
        "在去噪方面，传统方法主要使用带通滤波、陷波、中值滤波和小波阈值；深度学习方法"
        "则使用卷积自编码器、U-Net 或 DnCNN 学习“干净心搏—含噪心搏”的映射。DnCNN "
        "最初用于图像高斯去噪，其“预测噪声而不是直接预测干净图像”的残差学习思想"
        "同样适用于一维 ECG：网络只需要拟合噪声分量，训练更稳定，也更容易保留 QRS "
        "的陡峭边缘。"
    )
    body(
        "现有工作仍有两点不足：其一，很多公开实现使用心搏级随机划分，患者信息泄漏"
        "导致指标偏乐观；其二，去噪和分类往往分开评价，缺少“同一批含噪心搏上，去噪"
        "前后分类性能如何变化”的闭环实验。本项目采用记录级划分，并把去噪器接入"
        "分类器，报告噪声类型、信噪比与分类指标之间的完整关系。"
    )

    h2("1.4  本文主要工作")
    bullet("构建可复现的数据管线：解析 MIT-BIH 的 48 条记录和 AAMI 标注，提取 101 451 个"
           "R 峰对齐心搏；按 DS1/DS2 做记录级划分，并从 DS1 中分层留出 10% 作为验证集。")
    bullet("设计 9 058 参数的轻量 1D CNN，完成正常/异常二分类筛查；在独立测试集上"
           "达到 93.17% 准确率、0.763 Macro-F1 和 0.829 AUC，并给出阈值—敏感度—"
           "特异性工作曲线。")
    bullet("设计 46 049 参数的一维 DnCNN，在 NSTDB 的基线漂移、电极运动和肌电噪声上"
           "进行训练与测试；5 dB 肌电噪声下把分类 Macro-F1 从 0.329 提升到 0.591。")
    bullet("完成 OpenECG 风格的端到端演示（滤波—R 峰检测—去噪—分类—显示），"
           "并给出模型参数量、体积、CPU 推理时延和 MCU 部署可行性分析。")
    body(
        "本文的工程代码、训练脚本、模型权重和生成图表的脚本均随报告提交，目录结构"
        "和运行方法见附录 A。"
    )


def data_section():
    h1("2  数据与实验设计", page_break=True)
    h2("2.1  数据集与开源资源")
    body(
        "本项目使用两个公开数据集：MIT-BIH 心律失常数据库用于心搏分类，MIT-BIH 噪声"
        "压力测试数据库（NSTDB）用于构造含噪心搏和评价去噪器。两者均由 PhysioNet "
        "免费提供，采用 Open Data Commons Attribution License，使用时需引用原始文献。"
        "所有数据通过 WFDB Python 包读取，避免手工解析 212 格式和注释文件带来的错误。"
    )
    table(
        ["资源", "用途", "链接"],
        [
            ["OpenECG", "开源心电采集/显示宿主", "https://github.com/jwlee95/OpenECG"],
            ["MIT-BIH Arrhythmia", "48 条记录、约 11 万心搏标注", "https://physionet.org/content/mitdb/1.0.0/"],
            ["MIT-BIH NSTDB", "bw/em/ma 三种真实噪声", "https://physionet.org/content/nstdb/1.0.0/"],
            ["WFDB Python", "读取 .dat/.hea/.atr", "https://github.com/MIT-LCP/wfdb-python"],
            ["torch_ecg", "ECG 深度学习模型参考", "https://github.com/DeepPSP/torch_ecg"],
            ["1D 模型仓库", "一维 CNN/ResNet 结构参考", "https://github.com/StChenHaoGitHub/1D-deeplearning-model-pytorch"],
            ["PyTorch", "网络训练与推理", "https://pytorch.org/"],
        ],
        [3.0, 5.0, 7.8],
        "表 1  本项目使用的数据集、开源项目与神经网络资源",
        font_size=7.8,
    )

    h2("2.2  MIT-BIH 心律失常数据库")
    body(
        "MIT-BIH 心律失常数据库包含 48 条约 30 min 的双导联动态 ECG 记录，来自 47 名"
        "受试者；采样率 360 Hz，11 位分辨率，增益 200 ADC/mV。记录由两位心脏病专家"
        "逐搏复核，包含约 11 万个心搏标注，标注符号覆盖正常、左/右束支阻滞、房性早搏、"
        "室性早搏、融合波、起搏和未知心搏等。数据库的 25 条记录刻意选择了少见但临床"
        "重要的心律失常，因此各类别比例不能代表一般人群患病率。"
    )
    body(
        "按照 AAMI EC57 的推荐，把细粒度标注合并为四类：N 包含 N、L、R、e、j；"
        "S 包含 A、a、J、S；V 包含 V、E；F 为 F。Q 类（/、f、Q 等起搏、未知或"
        "无法分类心搏）不参与监督训练，原因是其数量较少、标签语义混杂，且不同论文"
        "对 Q 的处理不一致。最终得到 101 451 个心搏，其中 N 90 631、S 2 781、"
        "V 7 236、F 803。表 2 给出 DS1/DS2 的分布。"
    )
    table(
        ["类别", "AAMI 原始标注", "DS1（训练+验证）", "DS2（测试）", "合计"],
        [
            ["N", "N, L, R, e, j", "45 866", "44 765", "90 631"],
            ["S", "A, a, J, S", "944", "1 837", "2 781"],
            ["V", "V, E", "3 788", "3 448", "7 236"],
            ["F", "F", "415", "388", "803"],
            ["合计", "—", "51 013", "49 705", "101 451"],
        ],
        [1.8, 5.0, 3.2, 3.2, 2.8],
        "表 2  AAMI 四类心搏的数据库划分",
        font_size=8.0,
    )
    body(
        "需要说明两点。第一，本实验只使用第 0 导联（多数记录为 MLII，少数记录为其他"
        "肢体/胸导联），没有融合双导联信息；这对单导联可穿戴场景是合理的，但会损失"
        "部分形态信息。第二，原数据库标注存在极少数修订版本（如 102.atr），本实验使用"
        "PhysioNet 当前版本，并在代码中固定数据版本。"
    )

    h2("2.3  MIT-BIH 噪声压力测试数据库")
    body(
        "NSTDB 提供三条 30 min、360 Hz 的真实噪声记录：bw 为基线漂移，em 为电极运动"
        "伪迹，ma 为肌电干扰。每条记录有两个通道，本文把两个通道拼接后作为噪声池。"
        "为了减少“同一段噪声既训练又测试”的泄漏，本文按时间顺序把每条噪声记录的前"
        "65% 用于训练/验证，后 35% 用于测试；训练时在 0–15 dB 内随机抽取信噪比，"
        "测试时固定使用 5、10、15 dB 三档。噪声按目标信噪比缩放到与干净心搏相同的"
        "功率，再逐点相加。"
    )

    h2("2.4  受试者独立的数据划分")
    body(
        "采用 de Chazal 提出的 DS1/DS2 划分：DS1 的 22 条记录用于训练和验证，DS2 的"
        "22 条记录只用于最终测试。DS1 内部再按心搏类别分层随机留出 10% 作为验证集，"
        "用于早停和模型选择；验证集与训练集来自相同记录，因此只用于调参，不用于"
        "报告最终性能。这样测试集完全由未参与训练的记录组成，能更真实地反映跨患者"
        "泛化能力。"
    )
    body(
        "这一划分有一个已知瑕疵：记录 201 和 202 来自同一受试者，却分别落在 DS1 和"
        "DS2 中，严格意义上的“受试者独立”并不完全成立。由于该划分是公开基准，本文"
        "保留它以保持可比性，并在局限性中明确说明；若要得到更严格的结论，应把 201/202 "
        "划入同一侧，或采用按受试者分组的交叉验证。"
    )

    h2("2.5  预处理与心搏切分")
    body(
        "预处理包括三步。第一步，使用三阶 Butterworth 零相位带通滤波器保留 0.5–40 Hz "
        "的心电有效频带，抑制基线漂移和高频肌电；第二步，使用 Q=30 的 60 Hz 陷波器"
        "抑制工频干扰（MIT-BIH 采集于美国，工频为 60 Hz；面向国内设备时可改为 50 Hz）；"
        "第三步，以专家标注的 R 峰为基准，向前取 96 点、向后取 160 点，得到 256 点"
        "（0.711 s）的心搏片段。片段两端不足时使用边缘值填充，随后减去中位数并按"
        "每个心搏自身的标准差做 z-score 归一化。"
    )
    body(
        "选择 256 点而不是固定 0.6 s，是为了兼顾 P 波、QRS 波和 T 波的完整形态；"
        "R 峰固定在索引 96，使网络看到的 QRS 位置一致。带通和陷波均为零相位滤波，"
        "避免波形时移影响 R 峰对齐。图 3 展示了原始信号与滤波后信号的对比，"
        "可以看到基线漂移被明显抑制，而 QRS 的陡峭边缘得以保留。"
    )
    add_figure("fig01_dataset_distribution.png", 14.0,
               "图 1  MIT-BIH 四类心搏在 DS1/DS2 中的数量分布（对数坐标）")
    add_figure("fig02_beat_examples.png", 15.0,
               "图 2  四类心搏的典型波形（0.5–40 Hz 带通后，R 峰对齐）")
    add_figure("fig03_filtering.png", 15.2,
               "图 3  0.5–40 Hz 带通与 60 Hz 陷波前后的 ECG 信号")


def methods():
    h1("3  方法", page_break=False)
    h2("3.1  系统总体结构")
    body(
        "系统分为硬件采集链路和软件处理链路两部分（图 4）。硬件链路沿用 OpenECG 的"
        "思路：电极/导联→模拟前端（AD8232 或 ADS1292 一类低功耗 AFE）→MCU 以 360 Hz、"
        "11 bit 采样→BLE 传输→OpenECG 上位机。软件链路在 OpenECG 上位机中新增四个"
        "模块：带通/陷波预处理、R 峰检测与心搏切分、DnCNN 去噪、1D CNN 分类，最后"
        "输出心搏类别和异常提示。"
    )
    body(
        "训练和离线评价阶段直接使用专家 R 峰标注，以排除检测误差对分类指标的影响；"
        "端到端演示阶段则使用一个紧凑的 Pan–Tompkins 风格检测器，以验证整条链路可以"
        "在没有标注的情况下运行。这样既保证了分类实验的可比性，也展示了实际部署时"
        "的完整流程。"
    )
    add_figure("fig10_system_integration.png", 15.8,
               "图 4  OpenECG 软硬件链路与本项目新增的深度学习模块")

    h2("3.2  轻量一维卷积分类器")
    body(
        "分类器的输入是 1×256 的 z-score 心搏。网络由三个卷积阶段组成：1→16 通道、"
        "卷积核 7；16→32、卷积核 5；32→64、卷积核 3；每个卷积后接 ReLU 和步长 2 的"
        "最大池化。随后使用全局平均池化把 64×32 的特征图压缩为 64 维向量，经过"
        "Dropout 0.2 后由全连接层输出 2 类 logits（正常/异常）。所有卷积均使用"
        "same padding，不引入 BatchNorm。"
    )
    body(
        "选择“轻量 + 无 BN”的配置不是随意决定的。预实验先尝试了带 BatchNorm 的"
        "四阶段 CNN（约 10 万参数）和带 RR 间期特征的变体，但两者在加权交叉熵下"
        "都出现过验证集突然塌缩到单一类别的现象；而 9 000 参数级的无 BN 网络在相同"
        "数据上训练稳定、收敛快，最终被选为部署模型。这个经验说明：在小样本、强"
        "类别不平衡的生理信号任务中，模型容量和归一化方式对优化稳定性的影响，"
        "可能比“更深更大”更重要。"
    )
    table(
        ["阶段", "操作", "输出尺寸", "参数量"],
        [
            ["输入", "R 峰对齐心搏", "1×256", "0"],
            ["卷积 1", "Conv1d(1,16,7)+ReLU+Pool/2", "16×128", "128"],
            ["卷积 2", "Conv1d(16,32,5)+ReLU+Pool/2", "32×64", "2 592"],
            ["卷积 3", "Conv1d(32,64,3)+ReLU+Pool/2", "64×32", "6 208"],
            ["池化", "全局平均池化 + Dropout 0.2", "64×1", "0"],
            ["分类", "Linear(64,2)", "2", "130"],
            ["合计", "—", "—", "9 058"],
        ],
        [2.2, 8.2, 2.8, 2.8],
        "表 3  轻量 1D CNN 分类器的结构与参数量",
        font_size=7.9,
    )
    body(
        "训练目标为带类别权重的交叉熵。DS1 训练集中正常心搏约 4.6 万、异常心搏约 "
        "5 100，异常占比约 10%，因此给异常类权重 3.0、正常类权重 1.0，并加入 0.03 的"
        "标签平滑，缓和对少数类的过拟合。优化器为 AdamW（学习率 1×10⁻³，权重衰减 "
        "1×10⁻⁴），余弦退火，批大小 128，最多 24 轮，验证集 Macro-F1 连续 6 轮不"
        "提升则早停。数据增强包括 ±5 点的时间抖动、0.9–1.1 的幅度缩放、0.5–1.5 Hz "
        "的随机基线扰动和 σ=0.008 的高斯噪声。"
    )
    body(
        "为了与四分类任务对照，本文还用同一 backbone 训练了一个 4 输出变体"
        "（9 188 参数），使用类别平衡采样和 22 223 个训练心搏（保留全部 S/F，"
        "对 N/V 下采样）。该实验只作为探索性分析，原因见 4.6 节。"
    )

    h2("3.3  DnCNN 心搏去噪器")
    body(
        "去噪器采用一维 DnCNN 的残差学习结构：输入含噪心搏 x，网络预测噪声 n̂，"
        "输出为 x−n̂。与直接回归干净心搏相比，残差学习只需要拟合噪声分量，"
        "更容易保留 QRS 的陡峭边缘，也不会把整个心搏“平滑掉”。网络包含 6 个"
        "卷积层：首层 1→32、核长 11，随后 4 个 32→32、核长 11 的“卷积+BN+ReLU”"
        "模块，最后 32→1、核长 11 的输出层；所有卷积 same padding。参数量 46 049。"
    )
    body(
        "训练样本来自 DS1 的干净心搏。对每个心搏随机选择一种噪声：NSTDB 的 bw/em/ma，"
        "或程序合成的白噪声、50/60 Hz 工频（含 2、3 次谐波）；在 0–15 dB 内随机抽取"
        "目标信噪比，按干净心搏功率缩放后逐点相加。输入和干净目标都用含噪心搏的标准差"
        "归一化，损失为均方误差（MSE）。优化器 AdamW（1×10⁻³，权重衰减 1×10⁻⁴），"
        "余弦退火，批大小 128，最多 12 轮，验证 MSE 连续 4 轮不降则早停。"
    )
    table(
        ["阶段", "操作", "输出尺寸"],
        [
            ["输入", "含噪心搏", "1×256"],
            ["首层", "Conv1d(1,32,11)+ReLU", "32×256"],
            ["隐藏层×4", "Conv1d(32,32,11)+BN+ReLU", "32×256"],
            ["输出层", "Conv1d(32,1,11)", "1×256"],
            ["残差", "x − n̂", "1×256"],
        ],
        [2.2, 9.0, 3.2],
        "表 4  一维 DnCNN 去噪器的结构",
        font_size=8.0,
    )
    body(
        "需要强调的是，去噪器是在“已带通滤波的干净心搏 + 真实噪声”上训练的，"
        "因此它学习的是带通后残余噪声的抑制，而不是从头处理原始 ADC 信号。"
        "这与本文的部署顺序一致：先做低成本带通/陷波，再把剩余的非平稳噪声交给"
        "深度网络。"
    )

    h2("3.4  传统滤波基线")
    body(
        "为了判断深度去噪是否真的有用，本文实现了一个传统基线：三阶 Butterworth "
        "0.5–40 Hz 带通、60 Hz 陷波和 3 点中值滤波。该基线不使用任何学习参数，"
        "代表了低成本嵌入式设备上最常见的处理方式。在噪声实验中，同一批含噪心搏"
        "分别经过“不处理”“传统滤波”“DnCNN 去噪”三条路径，再送入同一个分类器，"
        "从而把去噪收益与分类收益分开。"
    )

    h2("3.5  评价指标")
    body(
        "二分类使用准确率（Acc）、Macro-F1、敏感度（Se）、特异性（Sp）和 AUC。"
        "设 TP、FP、TN、FN 分别为真阳性、假阳性、真阴性、假阴性，则："
    )
    body("Se = TP / (TP + FN)，Sp = TN / (TN + FP)，Acc = (TP + TN) / N，", first_line=False,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    body("F1 = 2PR / (P + R)，Macro-F1 = (F1_N + F1_A) / 2。", first_line=False,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    body(
        "其中 P = TP/(TP+FP)、R = Se。AUC 由受试者工作特征曲线下的面积计算，"
        "不受阈值影响，适合比较不同工作点。去噪质量使用输入/输出信噪比、信噪比"
        "增益 ΔSNR、均方根误差 RMSE 和相关系数："
    )
    body("SNR = 10 log₁₀(Σx² / Σ(x − x̂)²)，ΔSNR = SNR_out − SNR_in。",
         first_line=False, align=WD_ALIGN_PARAGRAPH.CENTER)
    body(
        "四分类使用总体准确率、Macro-F1、各类别 P/R/F1 和混淆矩阵。所有指标先在"
        "心搏级计算，再按类别宏平均；测试集不参与任何调参。"
    )

    h2("3.6  OpenECG 集成与边缘部署")
    body(
        "软件侧提供两个接口：`denoise_beats(x)` 接受 (N,256) 的滤波后心搏，返回同形状"
        "的干净估计；`predict_beats(x)` 返回类别概率和预测标签。OpenECG 上位机可以把"
        "采集到的整段信号送入 `process_record()`，由内置的 R 峰检测器切分心搏后调用"
        "上述接口；也可以直接读取 CSV/WFDB 记录做离线分析。端到端演示即使用这一路径。"
    )
    body(
        "硬件侧，模型需要嵌入以 Cortex-M4/M7 或 ESP32 为代表的 MCU。9 058 参数的"
        "分类器 FP32 约 35 KB，INT8 约 9 KB；46 049 参数的去噪器 FP32 约 180 KB，"
        "INT8 约 45 KB。以 168 MHz、每周期 1 次乘加粗略估算，分类器约 75 万次乘加，"
        "理论计算时间约 4.5 ms；去噪器约 1 170 万次乘加，约 70 ms。按 1 Hz 的心搏"
        "到达率，两者都有充足余量。实际部署还需用 STM32Cube.AI/X-CUBE-AI 或 "
        "TensorFlow Lite Micro 做 INT8 量化和算子适配，并以实测 cycles 为准；"
        "本文没有完成硬件实物验证，因此不报告 MCU 实测时延。"
    )


def results_section():
    bm = json.loads((RESULTS / "binary_metrics.json").read_text(encoding="utf-8"))
    bn = json.loads((RESULTS / "binary_noise_results.json").read_text(encoding="utf-8"))
    cm = json.loads((RESULTS / "classification_metrics.json").read_text(encoding="utf-8"))
    cost = json.loads((RESULTS / "model_cost.json").read_text(encoding="utf-8"))
    clean = next(r for r in bn if r["condition"] == "clean")

    h1("4  实验与结果", page_break=False)
    h2("4.1  实验环境与实现细节")
    body(
        "全部实验在一台 Windows 11 工作站上完成，使用 Python 3.12.14、PyTorch 2.15.0+cpu、"
        "NumPy 2.5.4、SciPy 1.18.1、WFDB 4.3.1 和 scikit-learn 1.9.1。无 GPU，"
        "PyTorch 使用 4 个 CPU 线程。随机种子固定为 42；数据划分、模型初始化和增强"
        "均使用同一随机数生成器。数据准备约 2 min，去噪器训练约 10 min，轻量分类器"
        "训练约 3 min。"
    )
    body(
        "实验遵循三条原则：测试集只在最终评价时使用；验证集只用于早停和选模；"
        "所有去噪/分类对比都使用同一批心搏、同一随机噪声，避免因样本不同造成差异。"
        "代码、模型权重和图表脚本随报告提交，运行命令见附录 A。"
    )

    h2("4.2  数据统计与预处理结果")
    body(
        "最终数据集共 101 451 个心搏，DS1 51 013 个、DS2 49 705 个。N 类占 89.3%，"
        "S 类占 2.7%，V 类占 7.1%，F 类占 0.8%。这种极端不平衡意味着：即使模型把所有"
        "心搏都判为正常，准确率也能达到 89%；因此本文不把准确率作为唯一指标，"
        "而是同时报告 Macro-F1、敏感度、特异性和 AUC。"
    )
    body(
        "图 1 显示 DS1/DS2 的类别比例基本一致，但 S 类在 DS2 中更多（1 837 对 944），"
        "主要来自记录 232 的频发房性早搏；这会使测试集的 S 类召回更具挑战性。"
        "图 2 给出四类心搏的典型波形：N 类 QRS 窄而陡，V 类 QRS 宽大畸形且 T 波方向"
        "相反，F 类介于两者之间，S 类与 N 类形态接近，单靠形态很难区分。"
        "图 3 表明带通/陷波能明显去除基线漂移，同时保留 QRS 边缘。"
    )

    h2("4.3  正常/异常筛查结果")
    body(
        "表 5 给出二分类模型在 DS2 全部 49 705 个心搏上的结果。模型整体准确率 93.17%、"
        "Macro-F1 0.763、AUC 0.829；在默认阈值 0.5 下特异性 99.7%，但敏感度只有 40.1%。"
        "这意味着它更擅长“排除异常”，而不是“发现异常”。造成这一现象的主要原因是"
        "异常类只占 10.9%，模型在交叉熵和 3 倍类别权重之间仍偏向多数类；AUC 0.829 "
        "说明排序能力尚可，主要问题在决策阈值。"
    )
    table(
        ["指标", "数值", "说明"],
        [
            ["测试心搏数", f"{bm['n']:,}", "DS2 全部心搏"],
            ["准确率 Acc", f"{bm['accuracy']*100:.2f}%", "阈值 0.5"],
            ["Macro-F1", f"{bm['macro_f1']:.3f}", "正常/异常宏平均"],
            ["敏感度 Se", f"{bm['sensitivity']*100:.2f}%", "异常被正确检出"],
            ["特异性 Sp", f"{bm['specificity']*100:.2f}%", "正常被正确排除"],
            ["AUC", f"{bm['auc']:.3f}", "阈值无关"],
            ["混淆矩阵", "TN=44 124, FP=135, FN=3 260, TP=2 186", "阈值 0.5"],
        ],
        [4.0, 7.5, 4.5],
        "表 5  二分类筛查模型在独立测试集上的性能",
        font_size=8.0,
    )
    body(
        "阈值直接影响筛查的工作点。表 6 给出三个典型阈值：0.5 偏保守，适合作为"
        "“高特异性初筛”，但会漏掉约 60% 的异常；0.40 偏敏感，敏感度提高到 80.8%，"
        "但特异性降到 76.7%，假阳性明显增加；0.45 是折中点，敏感度 46.2%、特异性"
        "97.6%。实际产品应根据使用场景选择：健康人群初筛可接受较高假阳性，"
        "而辅助诊断应优先保证敏感度。"
    )
    table(
        ["阈值", "准确率", "Macro-F1", "敏感度", "特异性"],
        [
            ["0.50", "93.17%", "0.763", "40.1%", "99.7%"],
            ["0.45", "91.96%", "0.757", "46.2%", "97.6%"],
            ["0.40", "77.18%", "0.647", "80.8%", "76.7%"],
        ],
        [2.4, 3.4, 3.2, 3.2, 3.2],
        "表 6  不同决策阈值下的二分类性能（DS2，49 705 心搏）",
        font_size=8.2,
    )
    body(
        "由于测试集异常比例只有 10.9%，准确率会被多数类主导。为了更清楚地比较"
        "不同方法，本文另外构造了一个平衡子集：保留全部 5 446 个异常心搏，再随机"
        "抽取 4 000 个正常心搏（异常占 57.7%）。在该子集上，未加噪时模型的准确率为"
        "69.45%、Macro-F1 0.692、敏感度 52.1%、特异性 93.1%、AUC 0.823。这个结果"
        "与全测试集的 93.17% 并不矛盾，而是同一模型在不同患病率下的表现；它提醒"
        "读者不能只看准确率。"
    )
    add_figure("fig06b_binary_roc.png", 15.5,
               "图 5  二分类筛查的混淆矩阵与 ROC 曲线（阈值 0.5 和 0.4）")

    h2("4.4  心搏去噪结果")
    body(
        "DnCNN 在 DS1 验证集（5 102 个心搏，随机 0–15 dB 混合噪声）上把平均输入"
        "信噪比从 7.53 dB 提升到 14.94 dB，ΔSNR 为 7.41 dB，验证 MSE 为 0.00869。"
        "图 6 给出一条 5 dB 含噪心搏的去噪示例：电极运动噪声下，原始 SNR 5.0 dB，"
        "传统滤波 4.7 dB，DnCNN 7.1 dB；基线漂移下，原始 5.0 dB，传统滤波 10.1 dB，"
        "DnCNN 10.5 dB。DnCNN 的波形更接近干净心搏，QRS 峰值和 T 波形态保留得更好。"
    )
    body(
        "表 7 给出 DS2 测试集上 9 种噪声条件的详细结果。DnCNN 在低信噪比下收益最大："
        "5 dB 时 bw/em/ma 的 ΔSNR 分别为 +5.05、+3.89、+5.85 dB；10 dB 时降为"
        "+2.60、+1.40、+2.85 dB；15 dB 时分别为 −1.49、−2.18、−1.37 dB，即出现了"
        "轻微过平滑。原因不难理解：网络在 0–15 dB 的混合噪声上训练，低 SNR 样本占"
        "主导，学到的是“较强的噪声抑制”；当输入已经很干净时，网络仍会去掉一部分"
        "高频细节，导致输出 SNR 反而下降。传统滤波在低 SNR 下也有改善，但整体弱于"
        "DnCNN；在高 SNR 下，滤波的相位/中值非线性甚至会损伤波形。"
    )
    den_rows = []
    for kind, name in (("bw", "基线漂移"), ("em", "电极运动"), ("ma", "肌电")):
        for snr in (5, 10, 15):
            r = next(x for x in bn if x["kind"] == kind and x["snr_db"] == snr)
            den_rows.append([
                name, f"{snr}", f"{r['snr_raw']:.1f}",
                f"{r['snr_classical']:.1f}", f"{r['snr_denoised']:.1f}",
                f"{r['snr_denoised'] - r['snr_raw']:+.1f}",
                f"{r['rmse_raw']:.3f}", f"{r['rmse_denoised']:.3f}",
            ])
    table(
        ["噪声", "输入 SNR", "原始", "传统滤波", "DnCNN", "ΔSNR", "RMSE 原始", "RMSE DnCNN"],
        den_rows,
        [2.2, 1.8, 1.5, 1.8, 1.6, 1.5, 1.9, 1.9],
        "表 7  NSTDB 噪声下的输出信噪比与 RMSE（单位：dB / mV）",
        font_size=7.5,
    )
    add_figure("fig04_denoising_example.png", 13.5,
               "图 6  5 dB 电极运动/基线漂移噪声下的去噪波形对比")
    add_figure("fig05b_denoising_snr.png", 15.0,
               "图 7  三种 NSTDB 噪声下输入/输出信噪比与 ΔSNR")

    h2("4.5  噪声条件下的分类鲁棒性")
    body(
        "去噪的最终目的不是“好看”，而是让分类器在噪声下仍然可用。本文把同一批"
        "含噪心搏分别送入未处理、传统滤波和 DnCNN 三条路径，再用同一个二分类器"
        "评价。图 8 和表 8 给出 Macro-F1 随输入 SNR 的变化。"
    )
    body(
        "在 5 dB 肌电干扰下，未处理信号的 Macro-F1 只有 0.329（敏感度 3.0%，几乎"
        "全部漏检），传统滤波仅提升到 0.359，而 DnCNN 提升到 0.591；5 dB 基线漂移"
        "下由 0.505 提升到 0.663；5 dB 电极运动下由 0.584 提升到 0.629。可见噪声"
        "类型对分类的影响不同：肌电干扰与 QRS 频带重叠，危害最大；基线漂移主要"
        "影响 ST/T 段，对 QRS 形态分类的伤害相对小；电极运动介于两者之间。"
    )
    body(
        "当 SNR 升到 15 dB 时，三种方法的差距缩小，DnCNN 的 Macro-F1 仍略高"
        "（约 0.70），但优势已不明显；这也与 4.4 节的“高 SNR 过平滑”现象一致。"
        "对实际系统而言，可以根据估计的输入 SNR 自适应决定是否启送去噪：低 SNR 时"
        "启用以换取召回率，高 SNR 时旁路以保留形态细节。本文没有实现该门控，"
        "留作后续工作。"
    )
    f1_rows = []
    for kind, name in (("bw", "基线漂移"), ("em", "电极运动"), ("ma", "肌电")):
        for snr in (5, 10, 15):
            r = next(x for x in bn if x["kind"] == kind and x["snr_db"] == snr)
            f1_rows.append([name, f"{snr}", f"{r['raw']['macro_f1']:.3f}",
                            f"{r['classical']['macro_f1']:.3f}",
                            f"{r['denoised']['macro_f1']:.3f}",
                            f"{r['denoised']['auc']:.3f}"])
    table(
        ["噪声", "SNR/dB", "未处理", "传统滤波", "DnCNN", "DnCNN AUC"],
        f1_rows,
        [2.8, 2.0, 2.8, 2.8, 2.8, 2.8],
        "表 8  噪声条件下二分类 Macro-F1 与 AUC",
        font_size=8.0,
    )
    add_figure("fig08_noise_robustness.png", 15.0,
               "图 8  三种噪声类型下未处理、传统滤波和 DnCNN 的 Macro-F1")

    h2("4.6  四分类探索与误差分析")
    body(
        "二分类只能回答“是否异常”，临床还关心“是哪一类”。本文用同一 backbone "
        "训练了 N/S/V/F 四分类模型，并保留一个 MLP 基线作对照。结果如表 9 所示："
        "轻量 CNN 的总体准确率 92.78%，但 Macro-F1 只有 0.440；S 类和 F 类的 F1 "
        "均为 0，说明模型在测试集上完全没有预测这两个类别。MLP 的准确率较低"
        "（82.40%），但至少预测出少量 S/F，Macro-F1 0.428，与 CNN 接近。"
    )
    table(
        ["模型", "准确率", "Macro-F1", "N F1", "S F1", "V F1", "F F1"],
        [
            ["轻量 1D CNN", "92.78%", "0.440", "0.961", "0.000", "0.798", "0.000"],
            ["MLP 基线", "82.40%", "0.428", "0.901", "0.086", "0.670", "0.057"],
        ],
        [3.2, 2.4, 2.6, 2.2, 2.2, 2.2, 2.2],
        "表 9  四分类探索结果（DS2，49 705 心搏）",
        font_size=8.2,
    )
    body(
        "为什么 S/F 这么难？主要有四个原因。第一，S 类与 N 类的单心搏形态高度重叠，"
        "房性早搏的诊断更依赖“提前发生 + 代偿间歇”的节律信息，而本文的分类器只看"
        "单个心搏，没有 RR 间期输入。第二，F 类只有 388 个测试样本，且形态介于 N 和"
        "V 之间，类内差异大。第三，测试集的 S 类主要来自记录 232（1 382 个），"
        "该记录的基线节律和导联配置与训练集差异很大，模型学到的“S 形态”无法迁移。"
        "第四，四分类的类别权重/采样策略在“提升 S/F 召回”和“保持 N/V 精度”之间"
        "很难平衡，预实验中多次出现少数类塌缩。"
    )
    body(
        "因此，本文将四分类定位为探索性实验，不据此宣称模型具备 S/F 识别能力。"
        "若要把四分类做成可用功能，至少需要：引入 RR/节律特征、按受试者分组训练、"
        "增加 S/F 样本（如 CPSC、PTB-XL 等外部数据），以及采用层次分类"
        "（先正常/异常，再细分 V/S/F）。这也解释了为什么最终部署的是二分类筛查"
        "而不是四分类诊断。"
    )
    add_figure("fig06_confusion_matrices.png", 14.0,
               "图 9  四分类探索的混淆矩阵（左：轻量 CNN；右：MLP 基线）")

    h2("4.7  OpenECG 端到端演示")
    body(
        "为了验证模型可以脱离标注运行，本文实现了一个 Pan–Tompkins 风格的 R 峰检测器"
        "（5–15 Hz 带通、差分平方、0.12 s 滑动平均、峰值间隔 0.25 s），并在记录 100 "
        "上以 100 ms 容差评价：Se=1.000、PPV=1.000（2 273 个真实心搏，2 273 个检测）。"
        "这个“满分”只说明该记录节律规则、噪声低，不能代表所有记录；记录 208 等"
        "频发异位搏动的记录会更难。"
    )
    body(
        "图 10 展示记录 208 的 10 s 片段。检测到 13 个心搏，模型预测 N 5 个、V 7 个、"
        "F 1 个；与专家标注相比，只有 1 个 F 被误判为 V，其余一致。该片段包含多次"
        "宽大畸形的室性早搏，模型能够把大多数 V 与 N 分开，说明形态特征确实被学到；"
        "但 F/V 边界仍然模糊，这与 4.6 节的结论一致。"
    )
    add_figure("fig11_openecg_demo.png", 15.5,
               "图 10  OpenECG 风格端到端演示：记录 208 的 10 s 片段")

    h2("4.8  计算复杂度与硬件可行性")
    body(
        "表 10 给出两个模型的参数量、存储和 CPU 推理时延。测试平台为 4 线程 CPU，"
        "批大小 1 000，取 10 次平均。分类器单心搏约 0.114 ms，去噪器约 1.098 ms；"
        "即使按 1 Hz 的心搏率，去噪器的计算量也远低于实时预算。"
    )
    table(
        ["模型", "参数量", "FP32 体积", "INT8 体积", "CPU 时延/心搏"],
        [
            ["1D CNN 分类器", f"{cost['params_cnn']:,}", f"{cost['cnn_fp32_kb']:.1f} KB",
             f"{cost['cnn_int8_kb']:.1f} KB", f"{cost['cnn_ms_per_beat']:.3f} ms"],
            ["DnCNN 去噪器", f"{cost['params_denoiser']:,}",
             f"{cost['denoiser_fp32_kb']:.1f} KB", f"{cost['denoiser_int8_kb']:.1f} KB",
             f"{cost['denoiser_ms_per_beat']:.3f} ms"],
        ],
        [3.6, 2.6, 2.8, 2.8, 3.4],
        "表 10  模型复杂度与推理开销",
        font_size=8.2,
    )
    body(
        "从资源角度看，9 058 参数的分类器非常适合 MCU：INT8 权重不到 9 KB，"
        "中间激活在 16×128 量级，远小于常见 Cortex-M4 的 192–256 KB SRAM；"
        "去噪器 INT8 约 45 KB，也能放进中高端 MCU，但 1 170 万次乘加在 168 MHz "
        "上约需 70 ms（理论值），需要与采集、BLE 传输分时调度。更稳妥的方案是"
        "“去噪在手机/上位机，分类在 MCU”或“仅在低 SNR 时上传去噪”。这些都属于"
        "工程折中，本文未做硬件实测。"
    )

    h2("4.9  讨论与局限性")
    body(
        "本项目的核心发现可以概括为三点。第一，在严格的记录级划分下，一个 9 000 "
        "参数级的轻量 CNN 就能取得 93% 的整体准确率和 0.83 的 AUC，说明形态特征"
        "对“正常/异常”筛查是有效的；但敏感度只有 40%，单靠默认阈值不足以做临床"
        "筛查，必须显式选择工作点。第二，DnCNN 在低 SNR 下能显著改善分类，"
        "5 dB 肌电噪声下 Macro-F1 提升 0.26；但在高 SNR 下会过平滑，说明去噪器"
        "需要与 SNR 估计联动，而不是无条件开启。第三，四分类中的 S/F 仍然是难点，"
        "单心搏形态 + 记录级划分不足以支撑可靠的亚类诊断。"
    )
    body(
        "局限性主要有五条。其一，只用了 MIT-BIH 一个数据库，受试者仅 47 人，"
        "且数据采集于 1970 年代，设备、人群和导联配置都与现代可穿戴设备不同；"
        "没有外部验证，结论不能直接外推。其二，只使用单导联，丢失了双导联的"
        "空间信息。其三，没有做严格的受试者分组交叉验证，201/202 的交叉会带来"
        "轻微乐观偏差。其四，没有完成硬件实物和实时性测试，MCU 时延只是理论估算。"
        "其五，二分类的“异常”是 S/V/F 的合并，临床意义有限，不能替代医生的"
        "亚类诊断。"
    )
    body(
        "与文献比较时需要特别谨慎：不同论文的类别合并、数据划分、是否剔除 Q 类、"
        "是否使用 RR 特征都会显著影响指标。本文不主张自己的数字优于某一篇论文，"
        "而是提供一个可复现的、记录级划分的基线，以及“去噪—分类”闭环评价。"
        "从临床角度看，本模型只能作为筛查/提示工具，不能作为诊断依据；任何"
        "阳性结果都应由医生结合多导联 ECG、病史和影像学检查确认。"
    )


def conclusion():
    h1("5  结论与展望", page_break=True)
    body(
        "本项目围绕 OpenECG 开源心电监测系统，完成了一个可运行的 ECG 软件模块："
        "从 MIT-BIH/NSTDB 数据读取、AAMI 标注合并、R 峰对齐切分，到轻量 1D CNN "
        "正常/异常筛查和 DnCNN 心搏去噪，再到 OpenECG 风格的端到端演示与边缘"
        "资源评估。主要结论如下："
    )
    bullet("在 49 705 个独立测试心搏上，9 058 参数的二分类模型达到 93.17% 准确率、"
           "0.763 Macro-F1 和 0.829 AUC；阈值 0.5 时特异性 99.7%、敏感度 40.1%，"
           "阈值 0.4 时敏感度 80.8%、特异性 76.7%。")
    bullet("46 049 参数的 DnCNN 在 5 dB 基线漂移/电极运动/肌电噪声下把分类 Macro-F1 "
           "从 0.505/0.584/0.329 提升到 0.663/0.629/0.591；在 15 dB 高 SNR 下出现"
           "轻微过平滑，提示需要 SNR 自适应门控。")
    bullet("四分类探索中 S/F 类的 F1 接近 0，说明单心搏形态 + 记录级划分不足以支撑"
           "亚类诊断；RR/节律特征、外部数据和层次分类是必要的改进方向。")
    bullet("模型 FP32 体积约 35 KB（分类器）和 180 KB（去噪器），CPU 单心搏推理约 "
           "0.11 ms 和 1.10 ms，具备嵌入 OpenECG 硬件端或上位机的可行性。")
    body(
        "后续工作可以从四个方向展开：一是引入 CPSC2018、PTB-XL 等外部数据，"
        "做跨数据库验证；二是把 RR 间期、房室关系等节律特征与形态特征融合，"
        "改进 S/F 识别；三是用 INT8 量化和 STM32Cube.AI 在真实 MCU 上测时延/功耗，"
        "完成硬件在环验证；四是把二分类输出改为风险评分，并在前瞻性队列中评估"
        "筛查收益。"
    )
    body(
        "总之，本项目验证了“轻量深度模型 + 传统预处理 + 开源采集链路”的可行性，"
        "也暴露了小样本、强不平衡、跨患者场景下的真实困难。相比追求单篇论文的"
        "高指标，本文更强调可复现的记录级划分、诚实的失败分析和可部署的模型规模，"
        "这也是后续把算法推向 OpenECG 实际设备的基础。"
    )


def references():
    h1("参考文献", page_break=True)
    refs = [
        "[1] Moody G B, Mark R G. The impact of the MIT-BIH arrhythmia database[J]. "
        "IEEE Engineering in Medicine and Biology Magazine, 2001, 20(3): 45-50.",
        "[2] de Chazal P, O'Dwyer M, Reilly R B. Automatic classification of heartbeats "
        "using ECG morphology and heartbeat interval features[J]. IEEE Transactions on "
        "Biomedical Engineering, 2004, 51(7): 1196-1206.",
        "[3] Association for the Advancement of Medical Instrumentation. EC57: Testing and "
        "reporting performance results of cardiac rhythm and ST segment measurement "
        "algorithms[S]. Arlington: AAMI, 2012.",
        "[4] Goldberger A L, Amaral L A N, Glass L, et al. PhysioBank, PhysioToolkit, and "
        "PhysioNet: components of a new research resource for complex physiologic signals[J]. "
        "Circulation, 2000, 101(23): e215-e220.",
        "[5] jwlee95. OpenECG: ECG monitoring system design with open source H/W and "
        "Python[EB/OL]. https://github.com/jwlee95/OpenECG.",
        "[6] Moody G, Mark R. MIT-BIH Arrhythmia Database (version 1.0.0)[DB/OL]. "
        "PhysioNet, 2005. https://doi.org/10.13026/C2F305.",
        "[7] Moody G B, Muldrow W E, Mark R G. A noise stress test for arrhythmia "
        "detectors[J]. Computers in Cardiology, 1984, 11: 381-384.",
        "[8] MIT-LCP. wfdb-python[EB/OL]. https://github.com/MIT-LCP/wfdb-python.",
        "[9] Paszke A, Gross S, Massa F, et al. PyTorch: an imperative style, "
        "high-performance deep learning library[C]//Advances in Neural Information "
        "Processing Systems 32. 2019: 8024-8035.",
        "[10] Wen H, Kang J. torch_ecg: an ECG deep learning framework implemented using "
        "PyTorch[EB/OL]. https://github.com/DeepPSP/torch_ecg.",
        "[11] StChenHaoGitHub. 1D-deeplearning-model-pytorch[EB/OL]. "
        "https://github.com/StChenHaoGitHub/1D-deeplearning-model-pytorch.",
        "[12] Zhang K, Zuo W, Chen Y, et al. Beyond a Gaussian denoiser: residual learning "
        "of deep CNN for image denoising[J]. IEEE Transactions on Image Processing, 2017, "
        "26(7): 3142-3155.",
        "[13] Pan J, Tompkins W J. A real-time QRS detection algorithm[J]. IEEE "
        "Transactions on Biomedical Engineering, 1985, BME-32(3): 230-236.",
        "[14] Hannun A Y, Rajpurkar P, Haghpanahi M, et al. Cardiologist-level arrhythmia "
        "detection and classification in ambulatory electrocardiograms using a deep neural "
        "network[J]. Nature Medicine, 2019, 25(1): 65-69.",
        "[15] Rajpurkar P, Hannun A Y, Haghpanahi M, et al. Cardiologist-level arrhythmia "
        "detection with convolutional neural networks[EB/OL]. arXiv:1707.01836, 2017.",
    ]
    for ref in refs:
        p = document.add_paragraph()
        p.paragraph_format.line_spacing = 1.35
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.first_line_indent = Cm(-0.74)
        p.paragraph_format.left_indent = Cm(0.74)
        r = p.add_run(ref)
        set_run_font(r, 9.2)


def appendix():
    h1("附录 A  代码、模型与链接", page_break=True)
    body(
        "本项目代码位于报告同目录的 `src/`，数据位于 `data/`，模型权重位于 "
        "`results/models/`，图表位于 `results/figures/`。由于课程提交不要求公开仓库，"
        "代码以附件形式随报告提交；如需在线版本，可直接把该目录上传到 GitHub/Gitee，"
        "目录结构无需修改。"
    )
    body("A.1  目录结构", bold=True, first_line=False)
    p = document.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(
        "openecg_project/\n"
        "├─ src/                 # 数据准备、模型、训练、评价、绘图、报告脚本\n"
        "│  ├─ prepare_dataset.py\n"
        "│  ├─ models.py\n"
        "│  ├─ train_denoiser.py\n"
        "│  ├─ train_classifier.py\n"
        "│  ├─ train_binary.py\n"
        "│  ├─ evaluate_binary_noise.py\n"
        "│  ├─ make_figures.py\n"
        "│  └─ openecg_demo.py\n"
        "├─ data/                # MIT-BIH、NSTDB 与预处理后的 beats.npz\n"
        "├─ results/             # 指标 JSON、混淆矩阵、图表、模型权重\n"
        "└─ report/              # 本报告 DOCX/PDF"
    )
    set_run_font(r, 8.5)
    body("A.2  运行方法", bold=True, first_line=False)
    p = document.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(
        "python src/prepare_dataset.py\n"
        "python src/train_denoiser.py\n"
        "python src/train_classifier.py --model cnn --arch light --sampler --lr 1e-3\n"
        "python src/train_binary.py\n"
        "python src/evaluate_binary_noise.py\n"
        "python src/make_figures.py\n"
        "python src/openecg_demo.py"
    )
    set_run_font(r, 8.5)
    body(
        "依赖版本：Python 3.12、torch 2.15.0+cpu、numpy 2.5.4、scipy 1.18.1、"
        "wfdb 4.3.1、scikit-learn 1.9.1、matplotlib 3.11.2、python-docx 1.2.0。"
        "完整命令和随机种子见 `README.md`。"
    )
    body("A.3  数据集与神经网络链接", bold=True, first_line=False)
    table(
        ["类别", "链接"],
        [
            ["MIT-BIH 数据集", "https://physionet.org/content/mitdb/1.0.0/"],
            ["NSTDB 数据集", "https://physionet.org/content/nstdb/1.0.0/"],
            ["OpenECG 项目", "https://github.com/jwlee95/OpenECG"],
            ["torch_ecg 网络库", "https://github.com/DeepPSP/torch_ecg"],
            ["1D CNN/ResNet 参考", "https://github.com/StChenHaoGitHub/1D-deeplearning-model-pytorch"],
            ["PyTorch 框架", "https://pytorch.org/"],
            ["WFDB 读取工具", "https://github.com/MIT-LCP/wfdb-python"],
        ],
        [4.2, 11.6],
        "表 A1  报告使用的数据集、开源项目与神经网络链接",
        font_size=8.2,
        align_center=False,
    )
    body(
        "说明：教师要求的“数据集链接”和“神经网络链接”已列于表 A1。本项目最终分类器"
        "是在 torch_ecg / 1D 模型仓库的通用结构思想上自行实现的轻量网络，并非直接"
        "调用某个预训练模型；网络结构、参数量和训练超参数已在第 3 章和附录中给出，"
        "代码随报告提交，便于复现。"
    )
    body("A.4  提交信息", bold=True, first_line=False)
    body("姓名：________________    学号：________________    班级：________________")
    body("提交日期：2026 年 10 月       课程：生物医学传感检测与仪器")
    body(
        "学术说明：本报告的数据、代码、训练和评价均由作者在本地完成；文中引用的"
        "公开数据集、开源项目和论文均已在参考文献中标注。模型结果仅用于课程学习，"
        "不构成临床诊断建议。"
    )


def build(toc_pages: dict[str, int] | None = None):
    cover()
    section_break_with_header()
    abstract()
    toc(toc_pages)
    intro()
    data_section()
    methods()
    results_section()
    conclusion()
    references()
    appendix()

    document.core_properties.title = "面向 OpenECG 的 ECG 心搏异常筛查与深度去噪方法研究"
    document.core_properties.subject = "生物医学传感检测与仪器课程项目报告"
    document.core_properties.author = "（请填写姓名）"
    out = REPORT / "ECG_心搏异常筛查与深度去噪_课程报告.docx"
    document.save(out)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toc-pages", default="", help="JSON mapping of heading -> page")
    args = ap.parse_args()
    pages = None
    if args.toc_pages:
        pages = json.loads(Path(args.toc_pages).read_text(encoding="utf-8"))
    build(pages)


if __name__ == "__main__":
    main()
