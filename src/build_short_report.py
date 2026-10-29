"""Build the 5-page (plus cover) concise course report."""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
RESULTS = ROOT / "results"
DATA = ROOT / "data"

BODY = 10.8
LINE = 1.38
CN = "宋体"
HEAD = "黑体"
EN = "Times New Roman"

doc = Document()

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin = Cm(1.9)
sec.bottom_margin = Cm(1.8)
sec.left_margin = Cm(2.0)
sec.right_margin = Cm(2.0)
sec.header_distance = Cm(0.9)
sec.footer_distance = Cm(0.8)
sec.different_first_page_header_footer = True

normal = doc.styles["Normal"]
normal.font.name = EN
normal._element.rPr.rFonts.set(qn("w:eastAsia"), CN)
normal.font.size = Pt(BODY)
normal.paragraph_format.line_spacing = LINE
normal.paragraph_format.space_after = Pt(0)
for name, size in (("Heading 1", 12.5), ("Heading 2", 11)):
    st = doc.styles[name]
    st.font.name = HEAD
    st._element.rPr.rFonts.set(qn("w:eastAsia"), HEAD)
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)


def font(run, size=BODY, bold=False, cn=CN, color=None, italic=False):
    run.font.name = EN
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), cn)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def para(text, first=True, size=BODY, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         before=0, after=0, line=LINE):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = align
    pf.line_spacing = line
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if first:
        pf.first_line_indent = Cm(0.66)
    r = p.add_run(text)
    font(r, size=size, bold=bold)
    return p


def h1(text):
    p = doc.add_paragraph(style="Heading 1")
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    font(r, 12.5, bold=True, cn=HEAD)


def h2(text):
    p = doc.add_paragraph(style="Heading 2")
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    font(r, 11, bold=True, cn=HEAD)


def caption(text, before=1, after=2):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    font(r, 8.2)


def table(headers, rows, widths, cap, fs=7.6):
    caption(cap)
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, sz in (("top", 10), ("bottom", 10), ("insideH", 3)):
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
    for j, (text, w) in enumerate(zip(headers, widths)):
        c = hdr.cells[j]
        c.width = Cm(w)
        c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        tcPr = c._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "DCE6F1")
        tcPr.append(shd)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.0
        font(p.add_run(text), fs, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for j, (text, w) in enumerate(zip(row, widths)):
            c = cells[j]
            c.width = Cm(w)
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            font(p.add_run(str(text)), fs)
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def figure(name, width_cm, cap):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(RESULTS / "figures" / name), width=Cm(width_cm))
    caption(cap)


def page_number(p):
    r = p.add_run("第 ")
    font(r, 8)
    run = p.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "separate")
    tt = OxmlElement("w:t"); tt.text = "1"
    f3 = OxmlElement("w:fldChar"); f3.set(qn("w:fldCharType"), "end")
    run._r.append(f1); run._r.append(it); run._r.append(f2)
    run._r.append(tt); run._r.append(f3)
    font(run, 8)
    r2 = p.add_run(" 页")
    font(r2, 8)


# ---------------------------------------------------------------------------
# Cover
# ---------------------------------------------------------------------------
for _ in range(2):
    doc.add_paragraph()
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
font(p.add_run("海 南 大 学"), 22, bold=True, cn=HEAD)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
font(p.add_run("《生物医学传感检测与仪器》课程项目报告"), 11)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(48)
font(p.add_run("基于深度学习的 ECG"), 20, bold=True, cn=HEAD)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
font(p.add_run("心搏异常筛查与去噪"), 20, bold=True, cn=HEAD)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(6)
font(p.add_run("ECG Beat Abnormality Screening and Denoising"), 9, color=(90, 90, 90), italic=True)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(26)
font(p.add_run("基于 OpenECG 开源心电采集链路"), 10, color=(80, 80, 80))

rows = [("姓    名", "________________"), ("学    号", "________________"),
        ("专    业", "生物医学工程"), ("指导教师", "________________"),
        ("日    期", "2026 年 10 月")]
t = doc.add_table(rows=0, cols=2)
t.alignment = WD_TABLE_ALIGNMENT.CENTER
for k, v in rows:
    cells = t.add_row().cells
    cells[0].width = Cm(3.2); cells[1].width = Cm(7.0)
    p = cells[0].paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    font(p.add_run(k + "："), 10.5, cn=HEAD)
    p = cells[1].paragraphs[0]
    font(p.add_run(v), 10.5)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(40)
font(p.add_run("2026 年 10 月"), 10)

# ---------------------------------------------------------------------------
# Body section
# ---------------------------------------------------------------------------
sec = doc.add_section(WD_SECTION.NEW_PAGE)
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin = Cm(1.9); sec.bottom_margin = Cm(1.8)
sec.left_margin = Cm(2.0); sec.right_margin = Cm(2.0)
sec.header_distance = Cm(0.9); sec.footer_distance = Cm(0.8)
sec.different_first_page_header_footer = False
sectPr = sec._sectPr
pgNum = OxmlElement("w:pgNumType"); pgNum.set(qn("w:start"), "1"); sectPr.append(pgNum)
hp = sec.header.paragraphs[0]; hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
font(hp.add_run("海南大学 · 生物医学传感检测与仪器"), 7.5, color=(120, 120, 120))
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
page_number(fp)

# read metrics
bm = json.loads((RESULTS / "binary_metrics.json").read_text(encoding="utf-8"))
bn = json.loads((RESULTS / "binary_noise_results.json").read_text(encoding="utf-8"))
cm = json.loads((RESULTS / "classification_metrics.json").read_text(encoding="utf-8"))
cost = json.loads((RESULTS / "model_cost.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# 1. Abstract and background
# ---------------------------------------------------------------------------
h1("摘  要")
para(
    "可穿戴 ECG 易受基线漂移、电极运动和肌电干扰，且异常心搏占比很低。本文以 "
    "OpenECG 采集链路为宿主，实现 0.5–40 Hz 预处理、R 峰对齐心搏切分、9 058 参数"
    "轻量 1D CNN 正常/异常筛查和 46 049 参数 DnCNN 去噪。按 MIT-BIH 的 DS1/DS2 "
    "记录级划分，在 49 705 个独立测试心搏上取得 93.17% 准确率、0.763 Macro-F1 和 "
    "0.829 AUC；5 dB 肌电噪声下 DnCNN 将 Macro-F1 从 0.329 提升到 0.591。",
    first=True,
)
p = doc.add_paragraph()
font(p.add_run("关键词："), 9.5, bold=True, cn=HEAD)
font(p.add_run("ECG；心搏分类；DnCNN；OpenECG；MIT-BIH"), 9.5)

h1("1  项目背景与目标")
para(
    "Holter 和可穿戴心电设备的数据量远超人工复核能力，自动筛查是落地的前提。"
    "现有公开实现常用心搏级随机划分，患者信息泄漏会高估指标；同时很多工作把去噪"
    "和分类分开评价，缺少“同一批含噪心搏上去噪前后分类如何变化”的闭环。"
    "本项目不追求重新设计模拟前端，而是把 OpenECG 作为采集/显示宿主，重点实现"
    "一个可插拔的软件模块：先做低成本带通/陷波，再用轻量 CNN 做正常/异常筛查，"
    "并用 DnCNN 改善低信噪比下的分类；最后给出模型在 MCU/上位机上的资源评估。"
)

# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------
h1("2  数据与评价方案")
h2("2.1  数据集与记录级划分")
para(
    "使用 MIT-BIH 心律失常数据库（48 条记录、47 名受试者、360 Hz、约 11 万心搏）"
    "和 MIT-BIH 噪声压力测试数据库（bw/em/ma 三种真实噪声）。按 AAMI EC57 合并为 "
    "N/S/V/F 四类，剔除起搏/未知的 Q 类，得到 101 451 个心搏；以 R 峰为中心取 "
    "−96～+160 点（256 点，0.711 s）。采用 de Chazal 的 DS1/DS2 划分：DS1 的 22 条"
    "记录用于训练/验证（90%/10%），DS2 的 22 条记录只用于测试。记录 201/202 来自"
    "同一受试者，严格受试者独立条件下存在 1 例交叉，结果可能略偏乐观。",
)
table(
    ["类别", "DS1（训练+验证）", "DS2（测试）", "合计"],
    [["N", "45 866", "44 765", "90 631"],
     ["S", "944", "1 837", "2 781"],
     ["V", "3 788", "3 448", "7 236"],
     ["F", "415", "388", "803"],
     ["合计", "51 013", "49 705", "101 451"]],
    [1.8, 4.4, 4.4, 4.4],
    "表 1  AAMI 四类心搏的数据库划分",
)

h2("2.2  预处理")
para(
    "对每条记录先做三阶 Butterworth 0.5–40 Hz 零相位带通和 60 Hz 陷波，再按专家 "
    "R 峰切分心搏；每个心搏减去中位数并做 z-score 归一化。训练/测试阶段使用专家 "
    "R 峰标注以排除检测误差；端到端演示时改用 Pan–Tompkins 风格检测器。",
)

# ---------------------------------------------------------------------------
# 3. Method
# ---------------------------------------------------------------------------
h1("3  方法")
h2("3.1  轻量 1D CNN 分类器")
para(
    "输入为 1×256 心搏。网络采用三个卷积阶段：1→16（核 7）、16→32（核 5）、"
    "32→64（核 3），每个卷积后接 ReLU 和步长 2 最大池化；全局平均池化后经 "
    "Dropout 0.2 和全连接层输出 2 类 logits（正常/异常）。模型不使用 BatchNorm，"
    "共 9 058 个参数。训练用带类别权重的交叉熵（异常:正常=3:1）、0.03 标签平滑、"
    "AdamW（1×10⁻³）、余弦退火、批大小 128、最多 24 轮，验证 Macro-F1 连续 6 轮"
    "不升则早停；增强包括 ±5 点抖动、0.9–1.1 幅度缩放、0.5–1.5 Hz 基线扰动和高斯噪声。",
)
table(
    ["阶段", "操作", "输出"],
    [["输入", "R 峰对齐心搏", "1×256"],
     ["卷积 1", "Conv1d(1,16,7)+ReLU+Pool/2", "16×128"],
     ["卷积 2", "Conv1d(16,32,5)+ReLU+Pool/2", "32×64"],
     ["卷积 3", "Conv1d(32,64,3)+ReLU+Pool/2", "64×32"],
     ["分类", "GAP+Dropout 0.2+Linear(64,2)", "2"]],
    [2.2, 9.4, 3.4],
    "表 2  轻量 1D CNN 的结构（9 058 参数）",
)

h2("3.2  DnCNN 去噪器")
para(
    "去噪器采用一维 DnCNN 残差学习：输入含噪心搏 x，网络预测噪声 n̂，输出 x−n̂。"
    "网络含 6 个卷积层（首层 1→32、核 11；4 个 32→32 的卷积+BN+ReLU；输出层 "
    "32→1、核 11），参数量 46 049。训练样本来自 DS1 干净心搏，随机叠加 NSTDB 的 "
    "bw/em/ma 或合成白噪声/工频，信噪比在 0–15 dB 内随机；输入和目标按含噪心搏"
    "标准差归一化，损失为 MSE，AdamW（1×10⁻³）、批大小 128、最多 12 轮，验证 MSE "
    "连续 4 轮不降则早停。",
)

h2("3.3  评价指标")
para(
    "二分类报告准确率、Macro-F1、敏感度 Se、特异性 Sp 和 AUC；去噪报告输入/输出"
    "信噪比、ΔSNR 和 RMSE。所有测试集心搏不参与训练或调参。",
)

# ---------------------------------------------------------------------------
# 4. Results
# ---------------------------------------------------------------------------
h1("4  结果与分析")
h2("4.1  正常/异常筛查")
para(
    f"表 3 给出 DS2 全部 {bm['n']:,} 个心搏的结果。默认阈值 0.5 时准确率 "
    f"{bm['accuracy']*100:.2f}%、Macro-F1 {bm['macro_f1']:.3f}、AUC {bm['auc']:.3f}，"
    f"特异性 {bm['specificity']*100:.1f}%，但敏感度只有 {bm['sensitivity']*100:.1f}%。"
    "原因是异常类仅占 10.9%，模型偏向多数类；AUC 0.829 说明排序能力尚可，主要问题"
    "在阈值。把阈值降到 0.40 可将敏感度提高到 80.8%，但特异性降到 76.7%；0.45 是"
    "折中点（敏感度 46.2%、特异性 97.6%）。在“全部异常+4 000 正常”的平衡子集上，"
    "Macro-F1 为 0.692、AUC 0.823，说明全测试集 93.17% 的准确率受患病率影响很大。",
)
table(
    ["阈值", "准确率", "Macro-F1", "敏感度", "特异性"],
    [["0.50", "93.17%", "0.763", "40.1%", "99.7%"],
     ["0.45", "91.96%", "0.757", "46.2%", "97.6%"],
     ["0.40", "77.18%", "0.647", "80.8%", "76.7%"]],
    [2.4, 3.6, 3.4, 3.4, 3.4],
    "表 3  不同阈值下的二分类性能（49 705 心搏）",
)
figure("fig06b_binary_roc.png", 10.5,
       "图 1  二分类筛查的混淆矩阵与 ROC 曲线（阈值 0.5 和 0.4）")

h2("4.2  去噪与噪声鲁棒性")
para(
    "DnCNN 在 DS1 验证集上把平均输入 SNR 从 7.53 dB 提升到 14.94 dB（ΔSNR 7.41 dB）。"
    "表 4 在 DS2 上给出 5/10/15 dB 三种 NSTDB 噪声的结果：低 SNR 下收益最大，"
    "5 dB 时 bw/em/ma 的 ΔSNR 分别为 +5.1/+3.9/+5.9 dB；15 dB 时分别为 "
    "−1.5/−2.2/−1.4 dB，出现轻微过平滑。分类侧，5 dB 肌电干扰下未处理 Macro-F1 "
    "仅 0.329（敏感度 3.0%），传统滤波 0.359，DnCNN 提升到 0.591；5 dB 基线漂移"
    "由 0.505 提升到 0.663，5 dB 电极运动由 0.584 提升到 0.629。高 SNR 下三种方法"
    "差距缩小，提示可根据估计 SNR 决定是否启送去噪。",
)
rows = []
for kind, name in (("bw", "基线漂移"), ("em", "电极运动"), ("ma", "肌电")):
    for snr in (5, 10, 15):
        r = next(x for x in bn if x["kind"] == kind and x["snr_db"] == snr)
        rows.append([f"{name}/{snr}", f"{r['snr_raw']:.1f}",
                     f"{r['snr_denoised']:.1f}",
                     f"{r['snr_denoised']-r['snr_raw']:+.1f}",
                     f"{r['raw']['macro_f1']:.3f}",
                     f"{r['classical']['macro_f1']:.3f}",
                     f"{r['denoised']['macro_f1']:.3f}"])
table(
    ["噪声/SNR", "输入", "DnCNN 输出", "ΔSNR", "F1 未处理", "F1 传统", "F1 DnCNN"],
    rows,
    [3.0, 1.5, 2.2, 1.5, 2.4, 2.2, 2.2],
    "表 4  NSTDB 噪声下的去噪与二分类 Macro-F1（dB）",
    fs=7.2,
)
figure("fig08_noise_robustness.png", 8.6,
       "图 2  三种噪声类型下未处理、传统滤波和 DnCNN 的 Macro-F1")

h2("4.3  四分类探索、端到端演示与复杂度")
para(
    f"同一 backbone 的四分类变体（9 188 参数）在 DS2 上准确率 "
    f"{cm['cnn']['accuracy']*100:.2f}%、Macro-F1 {cm['cnn']['macro_f1']:.3f}，"
    "但 S/F 类的 F1 为 0；MLP 基线准确率 82.40%、Macro-F1 0.428（表 5）。这说明"
    "单心搏形态 + 记录级划分不足以支撑 S/F 亚类诊断，最终部署二分类筛查而不是"
    "四分类诊断。"
)
table(
    ["模型", "Acc", "Macro-F1", "N F1", "S F1", "V F1", "F F1"],
    [["轻量 CNN", "92.78%", "0.440", "0.961", "0.000", "0.798", "0.000"],
     ["MLP", "82.40%", "0.428", "0.901", "0.086", "0.670", "0.057"]],
    [3.0, 2.4, 2.6, 2.2, 2.2, 2.2, 2.2],
    "表 5  四分类探索结果（DS2，49 705 心搏）",
    fs=7.4,
)
para(
    "S 类与 N 类形态高度重叠，房性早搏更依赖“提前发生+代偿间歇”的节律信息；"
    "F 类测试样本仅 388 个且形态介于 N/V 之间；测试集 S 类又主要来自训练集未见的"
    "记录 232。这些因素共同导致 S/F 跨患者识别失败，也说明后续应引入 RR/节律特征"
    "和外部数据。"
)
para(
    "端到端演示在记录 208 的 10 s 片段检测 13 个心搏，预测 N 5、V 7、F 1，仅 1 个 F "
    "误判为 V；记录 100 上 R 峰检测 Se=PPV=1.000（100 ms 容差）。"
)
para(
    f"分类器 9 058 参数，FP32 约 {cost['cnn_fp32_kb']:.0f} KB、INT8 约 "
    f"{cost['cnn_int8_kb']:.0f} KB，CPU 单心搏约 {cost['cnn_ms_per_beat']:.3f} ms；"
    f"去噪器 46 049 参数，FP32 约 {cost['denoiser_fp32_kb']:.0f} KB、INT8 约 "
    f"{cost['denoiser_int8_kb']:.0f} KB，CPU 约 {cost['denoiser_ms_per_beat']:.2f} ms。"
    "按 168 MHz、每周期 1 次乘加估算，分类器约 4.5 ms、去噪器约 70 ms，1 Hz 心搏率"
    "下余量充足；但本文未做 MCU 实测。",
)

# ---------------------------------------------------------------------------
# 5. Discussion and conclusion
# ---------------------------------------------------------------------------
h1("5  讨论与局限")
para(
    "本项目的三点结论是：① 9 000 参数级 CNN 即可取得 93% 整体准确率和 0.83 AUC，"
    "但默认阈值敏感度仅 40%，必须显式选择工作点；② DnCNN 在低 SNR 下显著改善"
    "分类，但高 SNR 下会过平滑，应与 SNR 估计联动；③ S/F 四分类仍困难，原因是"
    "单心搏形态重叠、F 样本少、S 类跨患者差异大。局限包括：仅用 MIT-BIH 单库、"
    "单导联、47 名受试者，数据采集于 1970 年代；201/202 存在 1 例受试者交叉；"
    "未完成硬件实物和实时性测试；二分类的“异常”是 S/V/F 合并，不能替代亚类诊断。"
)
h1("6  结论")
para(
    "以 OpenECG 为宿主，完成了“预处理—心搏切分—DnCNN 去噪—轻量 CNN 筛查”的"
    "可运行软件模块。独立测试集 93.17% 准确率、0.763 Macro-F1、0.829 AUC；"
    "5 dB 肌电噪声下 DnCNN 把 Macro-F1 从 0.329 提升到 0.591。模型仅 35 KB（FP32），"
    "具备嵌入 OpenECG 硬件端或上位机的可行性。后续可引入 RR/节律特征、外部数据、"
    "INT8 量化和真实 MCU 测试。"
)

# ---------------------------------------------------------------------------
# Links, refs
# ---------------------------------------------------------------------------
h1("数据与神经网络链接")
table(
    ["资源", "链接"],
    [["MIT-BIH", "https://physionet.org/content/mitdb/1.0.0/"],
     ["NSTDB", "https://physionet.org/content/nstdb/1.0.0/"],
     ["OpenECG", "https://github.com/jwlee95/OpenECG"],
     ["torch_ecg", "https://github.com/DeepPSP/torch_ecg"],
     ["1D 模型参考", "https://github.com/StChenHaoGitHub/1D-deeplearning-model-pytorch"],
     ["PyTorch", "https://pytorch.org/"]],
    [3.2, 12.6],
    "表 6  数据集与神经网络链接",
    fs=7.0,
)
para(
    "代码、模型权重和图表随报告提交（src/、results/models/、results/figures/）；"
    "完整训练/评价脚本见 README.md。"
)

h1("附录 A  代码与运行")
para(
    "完整代码、模型权重和图表见随报告提交的 ECG_project_code.zip。主要脚本："
    "prepare_dataset.py（数据准备）、train_denoiser.py（DnCNN）、train_binary.py"
    "（二分类）、evaluate_binary_noise.py（噪声评价）、make_figures.py（图表）。"
)
p = doc.add_paragraph()
p.paragraph_format.left_indent = Cm(0.5)
p.paragraph_format.line_spacing = 1.05
font(p.add_run(
    "python src/prepare_dataset.py\n"
    "python src/train_denoiser.py\n"
    "python src/train_binary.py\n"
    "python src/evaluate_binary_noise.py\n"
    "python src/make_figures.py"
), 8.0)
para(
    "环境：Python 3.12、PyTorch 2.15.0+cpu、NumPy 2.5、SciPy 1.18、WFDB 4.3；"
    "数据从 PhysioNet 下载后放入 data/ 目录，运行说明见 README.md。"
)
table(
    ["文件/目录", "作用"],
    [["src/prepare_dataset.py", "解析 WFDB、AAMI 合并、心搏切分"],
     ["src/models.py、train_*.py", "网络定义与训练"],
     ["results/models/*.pt", "二分类、四分类、DnCNN 权重"],
     ["results/figures/", "报告全部图表"],
     ["data/processed/beats.npz", "101 451 个预处理心搏"]],
    [6.0, 10.0],
    "表 A1  主要文件说明",
    fs=7.2,
)
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(3)
font(p.add_run("姓名：____________  学号：____________  班级：____________"), 8.5)
font(p.add_run("    提交日期：2026 年 10 月"), 8.5)

out = REPORT / "ECG_课程报告_精简版.docx"
doc.save(out)
print("saved", out)
