"""Turn derived/stats.json into manuscript/numbers.tex as a set of \\stat* macros.

Every statistic quoted in the manuscript prose comes from here. The Makefile fails the
build if main.tex references a macro this file does not define, or if this file defines a
macro main.tex never uses -- so a stale number cannot survive a rebuild.
"""
import json
import math
import os

from common import DERIVED

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGET = os.path.join(ROOT, "manuscript", "numbers.tex")
with open(os.path.join(DERIVED, "stats.json")) as _fh:
    S = json.load(_fh)


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}\\%"


def pp(x, nd=1):
    return f"{100 * x:+.{nd}f}"


def ppu(x, nd=1):
    """Unsigned percentage points, for magnitudes (spreads, gaps) that cannot be negative."""
    return f"{100 * x:.{nd}f}"


def num(x, nd=3):
    return f"{x:.{nd}f}"


def comma(n):
    return f"{n:,}".replace(",", "{,}")


def sci(p):
    """Render a p-value for prose: plain decimal above 0.001, mantissa/exponent below.

    Uses log10 rather than repeated multiplication, which loses precision and can
    fail to terminate sensibly on denormals such as the 1e-76 seen in the stability test.
    """
    if p <= 0:
        return "<10^{-300}"
    if p >= 1e-3:
        return f"{p:.3g}"
    exp = math.floor(math.log10(p))
    mant = p / (10.0 ** exp)
    if mant >= 9.95:  # rounding to 1 s.f. would carry into the next decade
        mant, exp = 1.0, exp + 1
    return f"{mant:.1f}\\!\\times\\!10^{{{exp}}}"


def build():
    hl, a1, a2 = S["headline"], S["a1_confidence_semantics"], S["a2_calibration"]
    a3, a4, a5 = S["a3_conditional_error"], S["a4_selective_prediction"], S["a5_circular"]
    a6, a7, a8 = S["a6_cascade"], S["a7_breakeven"], S["a8_position_bias"]
    a9, a10 = S["a9_sensitivity"], S["a10_cost"]
    cy, bi = a5["cyber"], a5["bio"]
    M = {}

    # ---- scale of the study
    M["statNItems"] = comma(hl["totals"]["n"])
    M["statNDatasets"] = str(hl["totals"]["n_datasets"])
    M["statNCallsTotal"] = comma(a10["n_calls_total"])
    M["statNCallsSingle"] = comma(a10["n_calls_single_pass"])
    M["statNCallsCircular"] = comma(a10["n_calls_circular"])
    M["statUsdTotal"] = f"{a10['est_usd_total']:.2f}"
    M["statUsdPerM"] = f"{a10['usd_per_m_input_tokens']:.3f}"
    M["statNErrors"] = str(a9["error_handling"]["total_errors"])

    # ---- trap one: the confidence field
    M["statConfMatch"] = pct(a1["candidates"]["rescaled_pmax"]["frac_within_tol"])
    M["statConfTol"] = num(a1["tolerance"], 3)
    M["statConfAltPmax"] = pct(a1["candidates"]["raw_pmax"]["frac_within_tol"])
    M["statConfAltMargin"] = pct(a1["candidates"]["margin"]["frac_within_tol"])
    M["statConfAltEntropy"] = pct(a1["candidates"]["normalised_certainty"]["frac_within_tol"])
    M["statConfExample"] = num(a1["worked_example"]["reported_confidence"], 2)
    M["statEcePmax"] = num(a2["pooled"]["on_pmax"]["ece_fixed"])
    M["statEceConf"] = num(a2["pooled"]["on_confidence_field"]["ece_fixed"])
    M["statEcePmaxLo"] = num(a2["pooled"]["ece_pmax_ci"][0])
    M["statEcePmaxHi"] = num(a2["pooled"]["ece_pmax_ci"][1])
    M["statEceRatio"] = num(a2["pooled"]["on_confidence_field"]["ece_fixed"] /
                            a2["pooled"]["on_pmax"]["ece_fixed"], 1)
    M["statBrierPmax"] = num(a2["pooled"]["on_pmax"]["brier"])

    # ---- trap two: the conditional
    for key, tag in (("bio", "Bio"), ("cyber", "Cyber"), ("labbench-litqa2", "LitQA"),
                     ("labbench-supp", "Supp")):
        e = a3["per_dataset"][key]
        M[f"statErrGivenHi{tag}"] = pct(e["p_error_given_high_conf"])
        if tag != "Supp":   # the manuscript quotes SuppQA only via the reverse conditional
            M[f"statHiGivenErr{tag}"] = pct(e["p_high_conf_given_error"])
        M[f"statNHi{tag}"] = comma(e["n_high_conf"])
    e = a3["per_dataset"]["labbench-litqa2"]
    M["statErrGivenHiLitQACI"] = (f"[{100*e['p_error_given_high_conf_ci'][0]:.0f},\\,"
                                  f"{100*e['p_error_given_high_conf_ci'][1]:.0f}]")
    t = next(t for t in a3["tests"] if t["a"] == "bio" and t["b"] == "cyber")
    M["statFisherBioCyberP"] = sci(t["p"])
    M["statFisherBioCyberQ"] = sci(t["q_bh"])

    # ---- trap three: permutation instability
    M["statConsistCyber"] = pct(cy["consistency"])
    M["statConsistBio"] = pct(bi["consistency"])
    M["statInconsistCyber"] = pct(1 - cy["consistency"])
    M["statRotSpreadCyber"] = ppu(cy["acc_rotation_spread"])
    M["statRotSpreadBio"] = ppu(bi["acc_rotation_spread"])
    M["statFlipWCCyber"] = pct(cy["flip_wrong_to_correct"])
    M["statFlipCWCyber"] = pct(cy["flip_correct_to_wrong"])
    M["statFlipWCBio"] = pct(bi["flip_wrong_to_correct"])
    M["statFlipCWBio"] = pct(bi["flip_correct_to_wrong"])
    M["statErrStableCyber"] = pct(cy["error_rate_stable"])
    M["statErrUnstableCyber"] = pct(cy["error_rate_unstable"])
    M["statErrRatioCyber"] = num(cy["observed_error_ratio"], 2)
    M["statNullRatioCyber"] = num(cy["null"]["null_error_ratio_mean"], 2)
    M["statNullRatioCyberLo"] = num(cy["null"]["null_error_ratio_ci"][0], 2)
    M["statNullRatioCyberHi"] = num(cy["null"]["null_error_ratio_ci"][1], 2)
    M["statNullRatioBio"] = num(bi["null"]["null_error_ratio_mean"], 2)
    # --- nondeterminism: the free control arm
    ndc, ndb = cy["nondeterminism"], bi["nondeterminism"]
    M["statFlipIdenticalCyber"] = pct(ndc["flip_rate_identical_input"], 2)
    M["statFlipIdenticalBio"] = pct(ndb["flip_rate_identical_input"], 2)
    M["statFlipIdenticalCyberCI"] = (f"[{100*ndc['flip_rate_ci'][0]:.1f},\\,"
                                     f"{100*ndc['flip_rate_ci'][1]:.1f}]")
    M["statNCompared"] = comma(ndc["n_compared"])
    M["statNDisagreeCyber"] = str(ndc["n_disagree"])
    M["statNoisePredCyber"] = pct(ndc["inconsistency_predicted_by_noise"])
    M["statNoisePredBio"] = pct(ndb["inconsistency_predicted_by_noise"])
    M["statExcessCyber"] = ppu(ndc["excess_over_noise"])
    M["statExcessBio"] = ppu(ndb["excess_over_noise"])
    M["statNoiseShareCyber"] = pct(ndc["share_of_instability_from_noise"], 0)
    M["statNoiseShareBio"] = pct(ndb["share_of_instability_from_noise"], 0)
    M["statNCircItemsCyber"] = comma(cy["n_items"])
    M["statNCircItemsBio"] = comma(bi["n_items"])

    # ---- what survives
    M["statAurocPooled"] = num(a4["pooled"]["p_max"]["auroc"])
    M["statAurocPooledLo"] = num(a4["pooled"]["p_max"]["auroc_ci"][0])
    M["statAurocPooledHi"] = num(a4["pooled"]["p_max"]["auroc_ci"][1])
    M["statAurocSpread"] = num(a4["signal_comparison"]["max_minus_min_auroc"], 3)
    M["statAurocBio"] = num(a4["per_dataset"]["bio"]["p_max"]["auroc"])
    M["statAurocCyber"] = num(a4["per_dataset"]["cyber"]["p_max"]["auroc"])
    for key, tag in (("bio", "Bio"), ("cyber", "Cyber")):
        rc = a4["risk_coverage"][key]
        M[f"statRC{tag}Base"] = num(rc["base_accuracy"])
        M[f"statRC{tag}AtSixty"] = num(rc["acc_at_coverage"]["0.6"])

    # ---- PACT
    M["statEnsGainCyber"] = pp(cy["ensemble_gain"])
    M["statEnsGainCyberLo"] = pp(cy["ensemble_gain_ci"][0])
    M["statEnsGainCyberHi"] = pp(cy["ensemble_gain_ci"][1])
    M["statEnsGainBio"] = pp(bi["ensemble_gain"])
    M["statEnsGainBioLo"] = pp(bi["ensemble_gain_ci"][0])
    M["statEnsGainBioHi"] = pp(bi["ensemble_gain_ci"][1])
    M["statEnsAccCyber"] = num(cy["ensemble_acc"])
    M["statSingleAccCyber"] = num(cy["single_pass_acc"])
    M["statMajAccCyber"] = num(cy["majority_acc"])
    c20 = a6["cyber"]["confidence_router"][2]
    c40 = a6["cyber"]["confidence_router"][4]
    r40 = a6["cyber"]["random_router"][4]
    M["statCascadeCyberAtTwenty"] = num(c20["accuracy"])
    M["statCascadeCyberAtForty"] = num(c40["accuracy"])
    M["statCascadeCyberCallsForty"] = num(c40["calls_per_item"], 1)
    M["statCascadeAdvantageForty"] = pp(a6["cyber"]["advantage_over_random_at_0.4"])
    M["statCascadeRandBeatCyber"] = pct(
        a6["cyber"]["frac_random_draws_beating_router_at_0.4"], 1)
    M["statCascadeAdvantageBio"] = pp(a6["bio"]["advantage_over_random_at_0.4"])
    M["statCascadeRandBeatBio"] = pct(
        a6["bio"]["frac_random_draws_beating_router_at_0.4"], 1)
    frac = (c40["accuracy"] - a6["cyber"]["tier1_accuracy"]) / (
        a6["cyber"]["tier2_accuracy"] - a6["cyber"]["tier1_accuracy"])
    M["statCascadeFracOfGain"] = pct(frac, 0)
    M["statCascadeCostSaving"] = pct(1 - c40["calls_per_item"] / 4.0, 0)
    for key, tag in (("bio", "Bio"), ("cyber", "Cyber"), ("labbench-litqa2", "LitQA")):
        M[f"statBreakeven{tag}"] = num(a7[key]["by_deferral"]["0.4"]["breakeven_fallback_accuracy"])

    # ---- position bias
    pb = a8["per_dataset"]
    M["statPosChiCyber"] = num(pb["cyber"]["chi2"], 1)
    M["statPosPCyber"] = "<10^{-4}" if pb["cyber"]["p_calibrated"] == 0 else sci(pb["cyber"]["p_calibrated"])
    M["statPosPBio"] = num(pb["bio"]["p_calibrated"], 2)
    M["statPosNullCrit"] = num(pb["cyber"]["null_chi2_95th"], 1)
    M["statPosPowerBio"] = num(pb["bio"]["power_vs_worst_case"], 2)
    M["statPosPredACyber"] = pct(pb["cyber"]["pred_first_share"])
    M["statPosGoldACyber"] = pct(pb["cyber"]["gold_first_share"])
    st = a8["stratified"]["cyber"]
    M["statPosLowPredA"] = pct(st["low"]["pred_first_share"])
    M["statPosLowGoldA"] = pct(st["low"]["gold_first_share"])
    M["statPosLowP"] = "<10^{-4}" if st["low"]["p_calibrated"] == 0 else sci(st["low"]["p_calibrated"])
    M["statPosHighPredA"] = pct(st["high"]["pred_first_share"])
    M["statPosHighGoldA"] = pct(st["high"]["gold_first_share"])
    M["statPosHighP"] = num(st["high"]["p_calibrated"], 2)
    M["statPosHighPower"] = num(st["high"]["power_vs_worst_case"], 2)
    M["statNSigPosDatasets"] = str(sum(1 for v in pb.values() if v["q_bh"] < 0.05))
    M["statNTestedPosDatasets"] = str(len(pb))

    # ---- measurement apparatus
    q = a9["quantisation"]
    M["statTieFrac"] = pct(q["frac_with_tie"], 1)
    M["statTieObservedFirst"] = str(q["ties_resolved_to_first_option"])
    M["statTieExpectedFirst"] = num(q["ties_expected_first_if_random"], 1)
    M["statTieContainingFirst"] = str(q["n_ties_containing_first_option"])
    M["statNTied"] = str(q["n_with_tie_at_max"])
    M["statDistinctProbs"] = str(q["distinct_probability_values"])

    # ---- chance baselines
    cb = a8["chance_baselines"]
    M["statChanceSupp"] = num(cb["labbench-supp"]["mean_chance"])
    M["statLiftBio"] = num(cb["bio"]["chance_adjusted_lift"])
    M["statLiftCyber"] = num(cb["cyber"]["chance_adjusted_lift"])
    M["statAccBio"] = pct(cb["bio"]["accuracy"])
    M["statAccSupp"] = pct(cb["labbench-supp"]["accuracy"])
    M["statAccBioRobust"] = pct(cb["bio-robust"]["accuracy"])
    M["statAccCyber"] = pct(cb["cyber"]["accuracy"])
    M["statAccProtocol"] = pct(cb["labbench-protocol"]["accuracy"])
    # per-dataset calibration quality, to show it is not uniform
    cal = a2["per_dataset"]
    M["statEceWorstWmdp"] = num(max(cal[k]["on_pmax"]["ece_fixed"]
                                    for k in ("bio", "bio-robust", "chem", "cyber")))
    M["statEceLitQA"] = num(cal["labbench-litqa2"]["on_pmax"]["ece_fixed"])
    M["statEceSupp"] = num(cal["labbench-supp"]["on_pmax"]["ece_fixed"])
    M["statCascadeCyberRandForty"] = num(r40["accuracy"])
    return M


def main():
    M = build()
    # TeX control sequences are letters only; a digit in a name is a hard compile error
    # ("Missing \begin{document}"), so reject it here rather than in the LaTeX log.
    bad = sorted(k for k in M if not k.isalpha())
    if bad:
        raise SystemExit(f"illegal macro name(s), letters only: {', '.join(bad)}")
    lines = ["% AUTO-GENERATED by evidence/code/export_numbers.py -- DO NOT EDIT BY HAND.",
             "% Regenerate with: make numbers", ""]
    for k in sorted(M):
        lines.append(f"\\newcommand{{\\{k}}}{{{M[k]}}}")
    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    with open(TARGET, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {TARGET} ({len(M)} macros)")


if __name__ == "__main__":
    main()
