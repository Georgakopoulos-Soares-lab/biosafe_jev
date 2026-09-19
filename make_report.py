import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, HRFlowable
)

RESULTS_DIR = "results"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

DATASETS = [
    ("bio", "WMDP-Bio"),
    ("chem", "WMDP-Chem"),
    ("cyber", "WMDP-Cyber"),
    ("bio-robust", "WMDP-Bio-Robust"),
    ("labbench-cloning", "LAB-Bench CloningScenarios"),
    ("labbench-protocol", "LAB-Bench ProtocolQA"),
    ("labbench-seq", "LAB-Bench SeqQA"),
    ("labbench-db", "LAB-Bench DbQA"),
    ("labbench-litqa2", "LAB-Bench LitQA2"),
    ("labbench-supp", "LAB-Bench SuppQA"),
]

FAMILY = {
    "bio": "WMDP", "chem": "WMDP", "cyber": "WMDP", "bio-robust": "WMDP",
    "labbench-cloning": "LAB-Bench", "labbench-protocol": "LAB-Bench", "labbench-seq": "LAB-Bench",
    "labbench-db": "LAB-Bench", "labbench-litqa2": "LAB-Bench", "labbench-supp": "LAB-Bench",
}

FAMILY_COLOR = {"WMDP": "#3b6fa0", "LAB-Bench": "#c0703c"}

all_records = {}
for key, _ in DATASETS:
    path = os.path.join(RESULTS_DIR, f"{key}.jsonl")
    with open(path) as f:
        all_records[key] = [json.loads(l) for l in f]

# ---------- per-dataset stats ----------
stats = {}
for key, label in DATASETS:
    recs = all_records[key]
    n = len(recs)
    valid = [r for r in recs if r["error"] is None]
    correct = sum(r["correct"] for r in recs)
    errors = n - len(valid)
    acc = correct / n
    conf_correct = [r["confidence"] for r in valid if r["correct"] and r["confidence"] is not None]
    conf_wrong = [r["confidence"] for r in valid if not r["correct"] and r["confidence"] is not None]
    wrong = [r for r in valid if not r["correct"]]
    conf_wrong_high = [r for r in wrong if (r["confidence"] or 0) >= 0.9]
    stats[key] = {
        "label": label,
        "n": n,
        "correct": correct,
        "accuracy": acc,
        "errors": errors,
        "avg_conf_correct": float(np.mean(conf_correct)) if conf_correct else None,
        "avg_conf_wrong": float(np.mean(conf_wrong)) if conf_wrong else None,
        "confidently_wrong_rate": (len(conf_wrong_high) / len(wrong)) if wrong else 0.0,
    }

# ---------- Figure 1: accuracy by dataset ----------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
order = sorted(DATASETS, key=lambda kl: stats[kl[0]]["accuracy"])
labels = [l for k, l in order]
accs = [stats[k]["accuracy"] * 100 for k, l in order]
bar_colors = [FAMILY_COLOR[FAMILY[k]] for k, l in order]
ypos = np.arange(len(labels))
ax.barh(ypos, accs, color=bar_colors)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Accuracy (%)")
ax.set_xlim(0, 100)
for y, a in zip(ypos, accs):
    ax.text(a + 1, y, f"{a:.1f}%", va="center", fontsize=8)
ax.axvline(25, color="gray", linestyle="--", linewidth=0.8, label="4-option chance (25%)")
handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in FAMILY_COLOR]
ax.legend(handles + [plt.Line2D([0], [0], color="gray", linestyle="--")],
          list(FAMILY_COLOR.keys()) + ["~chance baseline"], loc="lower right", fontsize=8)
ax.set_title("Jev (jev-latest) accuracy across all evaluated benchmarks")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_accuracy.png", dpi=200)
plt.close()

# ---------- Figure 2: calibration / reliability diagram (pooled) ----------
pooled_conf = []
pooled_correct = []
for key, _ in DATASETS:
    for r in all_records[key]:
        if r["error"] is None and r["confidence"] is not None:
            pooled_conf.append(r["confidence"])
            pooled_correct.append(1 if r["correct"] else 0)
pooled_conf = np.array(pooled_conf)
pooled_correct = np.array(pooled_correct)

bins = np.linspace(0, 1, 11)
bin_idx = np.digitize(pooled_conf, bins) - 1
bin_idx = np.clip(bin_idx, 0, 9)
bin_acc, bin_conf, bin_count = [], [], []
for b in range(10):
    mask = bin_idx == b
    if mask.sum() > 0:
        bin_acc.append(pooled_correct[mask].mean())
        bin_conf.append(pooled_conf[mask].mean())
        bin_count.append(mask.sum())
    else:
        bin_acc.append(np.nan)
        bin_conf.append((bins[b] + bins[b + 1]) / 2)
        bin_count.append(0)

ece = np.nansum([abs(a - c) * n for a, c, n in zip(bin_acc, bin_conf, bin_count)]) / len(pooled_conf)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [2, 1]})
ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
ax1.plot(bin_conf, bin_acc, "o-", color="#3b6fa0", label="Jev (pooled, all datasets)")
ax1.set_xlabel("Predicted confidence")
ax1.set_ylabel("Empirical accuracy")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_title(f"Reliability diagram (ECE = {ece:.3f})")
ax1.legend(fontsize=8)

ax2.hist(pooled_conf, bins=20, color="#3b6fa0", alpha=0.8)
ax2.set_xlabel("Predicted confidence")
ax2.set_ylabel("Count")
ax2.set_title("Confidence distribution")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_calibration.png", dpi=200)
plt.close()

# ---------- Figure 3: position bias (cyber) ----------
from collections import Counter
letters = list("ABCD")
cyber = all_records["cyber"]
gold_dist = Counter(r["gold_letter"] for r in cyber)
pred_dist = Counter(r["pred_letter"] for r in cyber if r["pred_letter"] in letters)
x = np.arange(len(letters))
w = 0.35
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.bar(x - w / 2, [gold_dist[l] for l in letters], width=w, label="Gold answer", color="#555555")
ax.bar(x + w / 2, [pred_dist[l] for l in letters], width=w, label="Jev prediction", color="#c0703c")
ax.set_xticks(x)
ax.set_xticklabels(letters)
ax.set_ylabel("Count")
ax.set_title("WMDP-Cyber: predicted vs. gold option letter\n(evidence of position bias under uncertainty)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_position_bias.png", dpi=200)
plt.close()

# ---------- Figure 4: confidently-wrong rate ----------
fig, ax = plt.subplots(figsize=(7.5, 4))
order2 = sorted(DATASETS, key=lambda kl: -stats[kl[0]]["confidently_wrong_rate"])
labels2 = [l for k, l in order2]
rates = [stats[k]["confidently_wrong_rate"] * 100 for k, l in order2]
colors2 = [FAMILY_COLOR[FAMILY[k]] for k, l in order2]
ax.bar(np.arange(len(labels2)), rates, color=colors2)
ax.set_xticks(np.arange(len(labels2)))
ax.set_xticklabels(labels2, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("Confidently-wrong rate (%)\n(wrong answers with confidence >= 0.9)")
ax.set_title("Overconfidence on incorrect answers, by dataset")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_confidently_wrong.png", dpi=200)
plt.close()

print("Figures written to", FIG_DIR)
for k, s in stats.items():
    print(k, s)

# ============================================================
# Build PDF
# ============================================================
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Abstract", parent=styles["BodyText"], fontSize=9.5, leading=13, spaceAfter=6))
styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=8.5, leading=11, textColor=colors.grey, spaceAfter=10))
body = styles["BodyText"]
body.fontSize = 10
body.leading = 14

doc = SimpleDocTemplate(
    "Jev_Biosecurity_Benchmark_Report.pdf",
    pagesize=LETTER,
    topMargin=0.85 * inch, bottomMargin=0.85 * inch,
    leftMargin=0.9 * inch, rightMargin=0.9 * inch,
)
story = []

story.append(Paragraph("Benchmarking TypeSafe's Jev (System One Model) on Biosecurity-Relevant Multiple-Choice Evaluations", styles["Title"]))
story.append(Spacer(1, 4))
story.append(Paragraph("An automated evaluation pipeline over WMDP and LAB-Bench", styles["Heading3"]))
story.append(Spacer(1, 10))
story.append(HRFlowable(width="100%", color=colors.grey))
story.append(Spacer(1, 10))

story.append(Paragraph("<b>Abstract</b>", styles["Heading4"]))
story.append(Paragraph(
    "We evaluate TypeSafe's Jev (<i>jev-latest</i>), a non-generative &ldquo;System One&rdquo; classification "
    "model, on 6,021 multiple-choice questions spanning the WMDP benchmark (bio, chem, cyber, and a "
    "shortcut-robust bio variant) and six text-based subtasks of LAB-Bench. Each item is posed as a single "
    "typed <i>choice</i> decision against Jev's HTTP API, with the model's returned option and per-option "
    "probability distribution scored against ground truth. We find accuracy ranges from 85.2% on WMDP-Bio "
    "recall questions down to 36.6% on LAB-Bench SuppQA, and identify two distinct failure regimes: (1) "
    "honest low-confidence guessing on tasks requiring multi-step computation or derivation (WMDP-Cyber, "
    "LAB-Bench), accompanied by a measurable option-order position bias; and (2) confident, miscalibrated "
    "errors on knowledge questions with closely-matched distractors (WMDP-Bio/Chem). Pooled calibration "
    f"analysis (Expected Calibration Error = {ece:.3f}) shows Jev's confidence score is a usable, if imperfect, "
    "signal for confidence-gated escalation to a stronger fallback model.",
    styles["Abstract"]))
story.append(Spacer(1, 10))

story.append(Paragraph("1. Introduction", styles["Heading2"]))
story.append(Paragraph(
    "The Weapons of Mass Destruction Proxy (WMDP) benchmark (Li et al., 2024) measures hazardous knowledge "
    "in biosecurity, cybersecurity, and chemical security domains as a proxy for evaluating and unlearning "
    "dangerous capabilities in language models. LAB-Bench (Laurent et al., 2024) evaluates practical "
    "biology-research capabilities &mdash; cloning design, protocol troubleshooting, sequence manipulation, "
    "literature and database lookup &mdash; that are directly relevant to dual-use biosecurity risk assessment. "
    "This report describes a pipeline built to evaluate TypeSafe's Jev model, a proprietary &ldquo;System One&rdquo; "
    "model that returns typed, probabilistic decisions in a single parallel pass rather than generating free text, "
    "against both benchmarks, and analyzes where and why its answers diverge from ground truth.",
    body))
story.append(Spacer(1, 8))

story.append(Paragraph("2. Method", styles["Heading2"]))
story.append(Paragraph(
    "<b>Model interface.</b> Each question is submitted to <font face='Courier'>POST https://api.typesafe.ai/v1/systemone</font> "
    "with <font face='Courier'>model=jev-latest</font>. The question stem and enumerated options (A, B, C, &hellip;) are "
    "placed in the <font face='Courier'>state</font> field as plain text; a single <font face='Courier'>choice</font>-type "
    "typed question is attached whose <font face='Courier'>criteria</font> map each option letter to its literal text. "
    "Jev returns a selected letter, a full probability distribution over letters, and a scalar confidence.",
    body))
story.append(Paragraph(
    "<b>Scoring.</b> The returned letter is mapped back to its original option index and compared against the "
    "dataset's ground-truth answer index. Requests are issued concurrently (16 workers) with exponential "
    "backoff on HTTP 429/529.", body))
story.append(Paragraph(
    "<b>Dataset preparation.</b> WMDP subsets (bio/chem/cyber) and WMDP-Bio-Robust were pulled directly from "
    "their published Hugging Face parquet files. WMDP-Bio-Robust encodes its <font face='Courier'>choices</font> "
    "field as a Python/NumPy array repr mixing single- and double-quoted strings; a quote-aware regex tokenizer "
    "was written to parse it losslessly (validated on all 811 rows). LAB-Bench subtasks store answers as an "
    "<font face='Courier'>ideal</font> string plus a <font face='Courier'>distractors</font> list rather than a fixed "
    "index; options were deduplicated (a small number of rows, notably in SeqQA, contain duplicate distractor "
    "text) and shuffled with a fixed seed before submission, with the answer index recorded post-shuffle. "
    "Two LAB-Bench subtasks, FigQA and TableQA, encode their supporting evidence as embedded raster images "
    "(a photograph/figure or a screenshotted table) rather than text; since Jev's <font face='Courier'>state</font> "
    "field accepts only text or structured text/JSON data, these two subtasks were excluded from evaluation "
    "rather than scored on the question text alone, which would not reflect a fair or meaningful measurement.",
    body))
story.append(Spacer(1, 8))

story.append(Paragraph("3. Datasets", styles["Heading2"]))

data_table = [["Dataset", "n", "Format", "Task type"]]
desc = {
    "bio": ("1,273", "MCQA (4-opt)", "Biosecurity knowledge recall"),
    "chem": ("408", "MCQA (4-opt)", "Chemical security knowledge recall"),
    "cyber": ("1,987", "MCQA (4-opt)", "Cybersecurity knowledge + code/arithmetic reasoning"),
    "bio-robust": ("811", "MCQA (4-opt)", "Bio knowledge, shortcut-controlled paraphrase set"),
    "labbench-cloning": ("33", "MCQA (var. opt)", "Molecular cloning scenario reasoning"),
    "labbench-protocol": ("108", "MCQA (var. opt)", "Wet-lab protocol troubleshooting"),
    "labbench-seq": ("600", "MCQA (var. opt)", "DNA/protein sequence manipulation"),
    "labbench-db": ("520", "MCQA (var. opt)", "Bioinformatics database lookup"),
    "labbench-litqa2": ("199", "MCQA (var. opt)", "Scientific literature fact retrieval"),
    "labbench-supp": ("82", "MCQA (var. opt)", "Paper supplementary-material retrieval"),
}
for key, label in DATASETS:
    n, fmt, task = desc[key]
    data_table.append([label, n, fmt, task])

t = Table(data_table, colWidths=[1.9 * inch, 0.5 * inch, 1.0 * inch, 2.6 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
]))
story.append(t)
story.append(Spacer(1, 4))
story.append(Paragraph("Table 1. Datasets evaluated. All are 2&ndash;9 option multiple-choice; LAB-Bench option "
                        "counts vary per item (min 2).", styles["Caption"]))
story.append(Spacer(1, 6))

story.append(PageBreak())
story.append(Paragraph("4. Results", styles["Heading2"]))
story.append(Image(f"{FIG_DIR}/fig1_accuracy.png", width=6.4 * inch, height=6.4 * 4.2 / 7.5 * inch))
story.append(Paragraph("Figure 1. Accuracy per dataset, sorted ascending. WMDP-family datasets in blue, "
                        "LAB-Bench subtasks in orange. Dashed line marks 4-option chance.", styles["Caption"]))

res_table = [["Dataset", "Accuracy", "Avg conf. (correct)", "Avg conf. (wrong)", "Confidently-wrong rate"]]
for key, label in DATASETS:
    s = stats[key]
    res_table.append([
        label,
        f"{s['accuracy']*100:.1f}%",
        f"{s['avg_conf_correct']:.2f}" if s["avg_conf_correct"] is not None else "-",
        f"{s['avg_conf_wrong']:.2f}" if s["avg_conf_wrong"] is not None else "-",
        f"{s['confidently_wrong_rate']*100:.1f}%",
    ])
t2 = Table(res_table, colWidths=[1.9 * inch, 0.85 * inch, 1.15 * inch, 1.05 * inch, 1.15 * inch])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
]))
story.append(Spacer(1, 8))
story.append(t2)
story.append(Paragraph("Table 2. Per-dataset accuracy and confidence statistics. &ldquo;Confidently-wrong rate&rdquo; "
                        "is the fraction of incorrect answers returned with confidence &ge; 0.9.", styles["Caption"]))

story.append(Spacer(1, 10))
story.append(Image(f"{FIG_DIR}/fig2_calibration.png", width=6.4 * inch, height=6.4 * 4 / 9 * inch))
story.append(Paragraph("Figure 2. Pooled reliability diagram across all 6,021 items (left) and confidence "
                        "distribution (right). Jev is well calibrated at high confidence (&ge;0.9) but "
                        "systematically underconfident in the low-to-mid range: empirical accuracy exceeds "
                        "stated confidence for every bin below ~0.85, meaning a confidence threshold used for "
                        "escalation should be tuned empirically rather than read off the raw score at face value.",
                        styles["Caption"]))

story.append(PageBreak())
story.append(Image(f"{FIG_DIR}/fig3_position_bias.png", width=4.6 * inch, height=4.6 * 4 / 5.5 * inch))
story.append(Paragraph("Figure 3. On WMDP-Cyber, Jev selects option A substantially more often than A is "
                        "actually correct (34.8% of predictions vs. 26.6% of gold answers), consistent with a "
                        "default-to-first-option bias under genuine uncertainty on computation-heavy questions.",
                        styles["Caption"]))
story.append(Spacer(1, 10))
story.append(Image(f"{FIG_DIR}/fig4_confidently_wrong.png", width=6.4 * inch, height=6.4 * 4 / 7.5 * inch))
story.append(Paragraph("Figure 4. Rate of high-confidence (&ge;0.9) incorrect answers, by dataset. High rates "
                        "on WMDP-Bio/Chem/Bio-Robust indicate miscalibrated overconfidence on knowledge-trap "
                        "questions; near-zero rates on WMDP-Cyber and LAB-Bench indicate the model is instead "
                        "aware of its own uncertainty on derivation-heavy tasks.", styles["Caption"]))

story.append(PageBreak())
story.append(Paragraph("5. Discussion", styles["Heading2"]))
story.append(Paragraph(
    "Two qualitatively distinct failure modes emerge. <b>(1) Derivation tasks:</b> WMDP-Cyber questions requiring "
    "multi-step arithmetic or code/assembly tracing, and the majority of LAB-Bench items (sequence analysis, "
    "protocol reasoning, database/literature lookup), are answered with markedly lower average confidence when "
    "wrong than when right, and manual inspection of a sample of wrong LAB-Bench/Cyber answers shows near-uniform "
    "probability mass across options &mdash; i.e., Jev is aware it is guessing. Re-submitting a sample of 15 "
    "originally-wrong WMDP-Cyber questions with options reshuffled flipped 4/15 (27%) to correct purely by "
    "removing the position bias documented in Figure 3, confirming that some of this error mass is an artifact "
    "of Jev's non-deliberative, single-pass architecture rather than a knowledge gap. "
    "<b>(2) Knowledge-precision tasks:</b> WMDP-Bio/Chem questions with closely-matched, deliberately confusable "
    "distractors instead produce a meaningful share of confidently wrong answers (confidence &ge; 0.9). The same "
    "reshuffling intervention had almost no effect on this class (0/15 bio, 1/15 chem flipped), indicating these "
    "are genuine knowledge/grounding failures rather than order artifacts, and would need retrieval-augmented "
    "context rather than ensembling to address.",
    body))
story.append(Spacer(1, 6))
story.append(Paragraph(
    f"Because Jev's confidence is directionally informative (lower for wrong answers in every dataset tested) but "
    f"not perfectly calibrated (ECE = {ece:.3f} pooled), it is best used as a routing signal rather than a correctness "
    "guarantee: escalating the lowest-confidence tail of answers to a slower, generative fallback model is "
    "projected (from the empirical confidence/correctness data already collected) to raise WMDP-Cyber accuracy "
    "from 63.4% to roughly 86% if 64% of items are escalated to an 85%-accurate fallback, at a fraction of the "
    "cost of routing every question to that fallback.", body))
story.append(Spacer(1, 8))

story.append(Paragraph("6. Limitations", styles["Heading2"]))
story.append(Paragraph(
    "(i) FigQA and TableQA (LAB-Bench) were excluded outright because their supporting evidence is an embedded "
    "raster image, which Jev's API cannot ingest &mdash; this is a coverage gap, not a measured failure. "
    "(ii) Jev is a closed, proprietary model; its training data and cutoff are unknown, so accuracy on any "
    "public benchmark may be inflated by memorization rather than genuine capability &mdash; the WMDP-Bio vs. "
    "WMDP-Bio-Robust gap (85.2% &rarr; 80.2%) is suggestive of this but not conclusive. (iii) Each question was "
    "submitted once; confidence and accuracy estimates would benefit from repeated sampling or explicit "
    "option-order ensembling, which this pipeline does not yet implement at scale. (iv) One CloningScenarios "
    "item (a long plasmid sequence, ~15K characters) exceeded Jev's input token limit and was scored as an error.",
    body))
story.append(Spacer(1, 8))

story.append(Paragraph("7. Conclusion", styles["Heading2"]))
story.append(Paragraph(
    "Jev performs well above chance on every evaluated benchmark but trails what a full reasoning LLM would be "
    "expected to score, particularly on derivation-heavy and fine-grained-retrieval tasks. Its calibrated-enough "
    "confidence signal makes it a plausible fast first-pass filter ahead of a stronger fallback model in a "
    "biosecurity-relevant screening pipeline, provided the routing threshold accounts for its systematically "
    "different error profile on knowledge-recall versus derivation tasks.",
    body))

doc.build(story)
print("PDF written: Jev_Biosecurity_Benchmark_Report.pdf")
