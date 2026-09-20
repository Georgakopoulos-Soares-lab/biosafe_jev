"""LaTeX tables for the manuscript, generated from derived/stats.json.

Table I is the manuscript's only table: per-dataset size, option range, chance-adjusted
lift, calibration, error detection, high-confidence error rate, and retained accuracy.
"""
import json
import os

from common import DERIVED, FAMILY, KEYS, LABEL

with open(os.path.join(DERIVED, "stats.json")) as _fh:
    S = json.load(_fh)
OUT = os.path.join(DERIVED, "tables")


def fmt(x, nd=3):
    return "--" if x is None or x != x else f"{x:.{nd}f}"


def table1():
    hl = S["headline"]["per_dataset"]
    cb = S["a8_position_bias"]["chance_baselines"]
    cal = S["a2_calibration"]["per_dataset"]
    sel = S["a4_selective_prediction"]["per_dataset"]
    ce = S["a3_conditional_error"]["per_dataset"]

    be = S["a7_breakeven"]
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Per-dataset results. \emph{Lift} is chance-adjusted accuracy, "
         r"$(\mathrm{acc}-c)/(1-c)$, using each item's own option count $c=1/n$ rather than a "
         r"single $25\%$ line, since option counts range from 2 to 10. ECE and AUROC are "
         r"computed on $p_{\max}$ rather than on the vendor \texttt{confidence} field "
         r"(\S\ref{sec:uncertainty}). $P(\text{err}\mid p_{\max}\!\geq\!0.9)$ is the error "
         r"rate given a confident answer, with Wilson intervals. \emph{Retained} is accuracy "
         r"on the items kept when the least confident 40\% are withheld "
         r"(\S\ref{sec:selective}).}",
         r"\label{tab:main}",
         r"\footnotesize",
         r"\setlength{\tabcolsep}{3.3pt}",
         r"\begin{tabular}{@{}llrccccccc@{}}", r"\toprule",
         r"Suite & Dataset & $N$ & opts. & accuracy [95\% CI] & lift & ECE & AUROC & "
         r"$P(\text{err}\mid p_{\max}\!\geq\!0.9)$ & retained \\", r"\midrule"]
    prev_fam = None
    for k in KEYS:
        h, c = hl[k], cb[k]
        fam = FAMILY[k]
        if prev_fam is not None and fam != prev_fam:
            L.append(r"\midrule")
        famcell = fam if fam != prev_fam else ""
        prev_fam = fam
        opts = (f"{c['min_options']}" if c["min_options"] == c["max_options"]
                else f"{c['min_options']}--{c['max_options']}")
        acc = f"{h['accuracy']:.3f} [{h['ci'][0]:.2f}, {h['ci'][1]:.2f}]"
        e = ce[k]
        # n for each high-confidence subset is quoted in the text where it matters;
        # omitted here to keep the table inside \textwidth.
        peh = (f"{e['p_error_given_high_conf']:.3f} "
               f"[{e['p_error_given_high_conf_ci'][0]:.2f}, "
               f"{e['p_error_given_high_conf_ci'][1]:.2f}]")
        d = be[k]["by_deferral"]["0.4"]
        L.append(f"{famcell} & {LABEL[k]} & {h['n']} & {opts} & {acc} & "
                 f"{fmt(c['chance_adjusted_lift'])} & {fmt(cal[k]['on_pmax']['ece_fixed'])} & "
                 f"{fmt(sel[k]['p_max']['auroc'])} & {peh} & "
                 f"{fmt(d['retained_accuracy'])} \\\\")
    tot = S["headline"]["totals"]
    pooled_ece = S["a2_calibration"]["pooled"]["on_pmax"]["ece_fixed"]
    pooled_auroc = S["a4_selective_prediction"]["pooled"]["p_max"]["auroc"]
    L += [r"\midrule",
          f"\\multicolumn{{2}}{{@{{}}l}}{{\\textbf{{pooled}}}} & {tot['n']} & 2--10 & "
          f"{tot['accuracy']:.3f} [{tot['ci'][0]:.3f}, {tot['ci'][1]:.3f}] & -- & "
          f"{pooled_ece:.3f} & {pooled_auroc:.3f} & -- & -- \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "table1_main.tex")
    with open(path, "w") as fh:
        fh.write(table1() + "\n")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
