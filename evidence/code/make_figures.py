"""Figures for the manuscript. Vector PDF for LaTeX, PNG for inspection.

fig1_uncertainty   (a) reliability of the two uncertainty quantities the API exposes;
                   (b) risk-coverage under p_max deferral.
fig2_permutation   (a) instability under four identical calls versus four cyclic
                       rotations, isolating run-to-run variation from option order;
                   (b) accuracy against mean inference cost for selective permutation
                       averaging, with random and oracle selection for reference.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import DERIVED, FIGURES, LABEL, load_all, wilson

plt.rcParams.update({
    # Type 42 (TrueType) rather than matplotlib's default Type 3: IEEE PDF eXpress
    # rejects Type 3 fonts outright.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # The IEEE template specifies 8 pt Times New Roman for figure labels.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.15,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
})

BLUE, GREEN, RED, GREY, PLUM, SAND = (
    "#2f6690", "#3a7d44", "#b3402f", "#9aa0a6", "#8c4a6e", "#c0703c")
with open(os.path.join(DERIVED, "stats.json")) as _fh:
    S = json.load(_fh)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"{name}.{ext}"), dpi=320)
    plt.close(fig)
    print(f"  {name}.pdf")


# --------------------------------------------------------------- Figure 1
def fig1_uncertainty():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.16, 2.35))

    # (a) Reliability. Bins with <25 items are dropped: p_max >= 1/n leaves the lowest
    # bins nearly empty, and their Wilson intervals would span the axis.
    MINBIN = 25
    rel_p = [b for b in S["a2_calibration"]["reliability_pooled"]["on_pmax"]
             if b["n"] >= MINBIN]
    rel_c = [b for b in S["a2_calibration"]["reliability_pooled"]["on_confidence_field"]
             if b["n"] >= MINBIN]
    ece_p = S["a2_calibration"]["pooled"]["on_pmax"]["ece_fixed"]
    ece_c = S["a2_calibration"]["pooled"]["on_confidence_field"]["ece_fixed"]

    axA.plot([0, 1], [0, 1], color="0.45", ls=(0, (3, 2.5)), lw=0.85, zorder=1,
             label="perfect calibration")
    for rel, col, mk, lab in ((rel_c, PLUM, "s", f"vendor field (ECE {ece_c:.3f})"),
                              (rel_p, GREEN, "o", f"$p_{{\\max}}$ (ECE {ece_p:.3f})")):
        x = [b["mean_conf"] for b in rel]
        y = [b["accuracy"] for b in rel]
        err = [[b["accuracy"] - b["ci_lo"] for b in rel],
               [b["ci_hi"] - b["accuracy"] for b in rel]]
        axA.errorbar(x, y, yerr=err, marker=mk, ms=3.0, color=col, capsize=1.4,
                     lw=1.1, elinewidth=0.8, zorder=3, label=lab)
    axA.set_xlabel("reported score")
    axA.set_ylabel("empirical accuracy")
    axA.set_xlim(0, 1.02)
    axA.set_ylim(0, 1.02)
    axA.set_xticks(np.arange(0, 1.01, 0.25))
    axA.set_yticks(np.arange(0, 1.01, 0.25))
    axA.set_title("(a) reliability of the two reported quantities", loc="left")
    axA.legend(loc="upper left", frameon=False, borderaxespad=0.25, handlelength=1.6)

    # (b) Risk-coverage.
    rc = S["a4_selective_prediction"]["risk_coverage"]
    sel = S["a4_selective_prediction"]["per_dataset"]
    series = [("bio", GREEN, "-"), ("bio-robust", BLUE, "-"),
              ("cyber", RED, "-"), ("labbench-seq", SAND, (0, (3, 1.6)))]
    for key, col, ls in series:
        cur = np.array(rc[key]["curve"])
        axB.plot(cur[:, 0] * 100, cur[:, 1] * 100, color=col, ls=ls, lw=1.15,
                 label=f"{LABEL[key]} (AUROC {sel[key]['p_max']['auroc']:.2f})")
    axB.axvline(60, color="0.55", lw=0.75, ls=(0, (2.5, 2)), zorder=1)
    axB.annotate("defer 40%", xy=(61.5, 102), fontsize=6.0, color="0.4",
                 ha="left", va="top")
    axB.set_xlabel("coverage: items answered (%)")
    axB.set_ylabel("accuracy on retained items (%)")
    axB.set_xlim(10, 100)
    axB.set_ylim(58, 103)
    axB.set_title("(b) accuracy against coverage", loc="left")
    axB.legend(loc="lower left", frameon=False, borderaxespad=0.25, handlelength=1.6)

    fig.tight_layout(w_pad=1.8)
    save(fig, "fig1_uncertainty")


# --------------------------------------------------------------- Figure 2
def fig2_permutation():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.16, 2.35))
    ks = ["bio", "cyber"]

    # (a) Where the instability comes from. Four identical calls isolate run-to-run
    # variation; four rotations add option order on top of it.
    rep = S["a11_repeatability"]
    x = np.arange(len(ks))
    w = 0.34
    ident, rot, e_i, e_r = [], [], [], []
    for k in ks:
        n = rep[k]["n_items"]
        for rate, vals, errs in ((rep[k]["inconsistency_identical"], ident, e_i),
                                 (rep[k]["inconsistency_rotations"], rot, e_r)):
            p, lo, hi = wilson(int(round(rate * n)), n)
            vals.append(p * 100)
            errs.append([(p - lo) * 100, (hi - p) * 100])
    axA.bar(x - w / 2, ident, w, yerr=np.array(e_i).T, capsize=2, color=GREY,
            error_kw={"lw": 0.75}, label="4 identical calls")
    axA.bar(x + w / 2, rot, w, yerr=np.array(e_r).T, capsize=2, color=RED,
            error_kw={"lw": 0.75}, label="4 cyclic rotations")
    for i in range(len(ks)):
        top = rot[i] + e_r[i][1]
        axA.annotate(f"+{rot[i] - ident[i]:.0f} pp", (i + w / 2, top + 2.2),
                     ha="center", fontsize=6.3, color="0.3")
    axA.set_xticks(x)
    axA.set_xticklabels([LABEL[k] for k in ks])
    axA.set_ylabel("items not answered\nidentically every time (%)")
    axA.set_ylim(0, 50)
    axA.set_title("(a) run-to-run variation vs. option order", loc="left")
    axA.legend(loc="upper left", frameon=False, borderaxespad=0.25, handlelength=1.4)

    # (b) Accuracy against mean inference cost.
    casc = S["a6_cascade"]
    col_of = {"cyber": RED, "bio": GREEN}
    for k in ks:
        col = col_of[k]
        base = casc[k]["tier1_accuracy"] * 100
        cx = [p["calls_per_item"] for p in casc[k]["confidence_router"]]
        for arm, ls, lw, alpha, mk in (
                ("confidence_router", "-", 1.25, 1.0, "o"),
                ("random_router", (0, (2.2, 2)), 0.85, 0.85, None),
                ("oracle_router", (0, (1, 1.7)), 0.85, 0.6, None)):
            axB.plot(cx, [p["accuracy"] * 100 - base for p in casc[k][arm]],
                     ls=ls, color=col, lw=lw, alpha=alpha, marker=mk, ms=2.4)
    handles = [plt.Line2D([], [], color=col_of[k], lw=1.25, label=LABEL[k]) for k in ks]
    handles += [
        plt.Line2D([], [], color="0.3", lw=1.25, marker="o", ms=2.4,
                   label="least-confident first"),
        plt.Line2D([], [], color="0.3", lw=0.85, ls=(0, (2.2, 2)), label="random"),
        plt.Line2D([], [], color="0.3", lw=0.85, ls=(0, (1, 1.7)), label="oracle"),
    ]
    axB.axhline(0, color="0.72", lw=0.65, zorder=1)
    axB.set_xlabel("mean inference calls per item")
    axB.set_ylabel("accuracy gain over\na single evaluation (pp)")
    axB.set_xlim(0.85, 4.15)
    axB.set_ylim(-0.9, 12.2)
    axB.set_xticks([1, 1.5, 2, 2.5, 3, 3.5, 4])
    axB.set_title("(b) selective permutation averaging", loc="left")
    axB.legend(handles=handles, loc="upper left", frameon=False, ncol=2,
               columnspacing=0.8, handlelength=1.6, borderaxespad=0.25)

    fig.tight_layout(w_pad=1.8)
    save(fig, "fig2_permutation")


def main():
    os.makedirs(FIGURES, exist_ok=True)
    load_all()  # fail fast if the raw records are missing
    print("figures:")
    fig1_uncertainty()
    fig2_permutation()


if __name__ == "__main__":
    main()
