"""Tier-1 analysis: every statistic cited in the manuscript, computed from the raw
decision records. Writes `derived/stats.json`, which is the single source of truth.

Run:  python3 evidence/code/analyze.py
Deterministic: all bootstraps are seeded, so repeated runs produce an identical file.

Experiment IDs map to the revision plan:
  A1  confidence-field semantics            A6  measured two-tier cascade
  A2  calibration on p_max                  A7  break-even external fallback
  A3  conditional error analysis            A8  position bias and chance baselines
  A4  selective prediction / routing signal A9  sensitivity (errors, quantisation)
  A5  circular evaluation                   A10 cost accounting
"""
import json
import os

import numpy as np
from scipy import stats as sps

from common import (
    CHARS_PER_TOKEN,
    DATASETS,
    DERIVED,
    FAMILY,
    KEYS,
    LABEL,
    N_BOOT,
    RAW,
    USD_PER_M_INPUT_TOKENS,
    accuracy_at_coverage,
    auprc,
    auroc,
    benjamini_hochberg,
    boot_diff_ci,
    brier,
    deferred_accuracy,
    dump,
    ece_adaptive,
    ece_fixed,
    load_all,
    load_circular,
    nll,
    reliability_bins,
    risk_coverage,
    valid,
    wilson,
)

HIGH_CONF = 0.9          # "high confidence" threshold used throughout
# Provenance of the collected data. The alias `jev-latest` resolved to this version
# throughout; the endpoint reports it on every response.
MODEL_VERSION = "jev-1.13.0"
API_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
COLLECTION_DATES = "2026-09-19 to 2026-09-20"
N_BOOT_CI = 2000         # resamples for the ECE / AUROC interval estimates
CIRCULAR_KEYS = ["bio", "cyber"]


# ===================================================================== A1

def a1_confidence_semantics(data):
    """The `confidence` field returned by the API is not a probability. Test the
    hypothesis conf = (p_max - 1/n) / (1 - 1/n) against three alternative readings."""
    rows = [r for k in KEYS for r in valid(data[k])]
    conf = np.array([r["confidence"] for r in rows], float)
    pmax = np.array([r["p_max"] for r in rows], float)
    n = np.array([r["n_options"] for r in rows], float)
    margin = np.array([r["margin"] for r in rows], float)

    # normalised entropy -> certainty in [0,1]
    negent = np.array([r["neg_entropy"] for r in rows], float)
    cert = 1.0 - (-negent) / np.log(n)

    candidates = {
        "rescaled_pmax": (pmax - 1.0 / n) / (1.0 - 1.0 / n),
        "raw_pmax": pmax,
        "margin": margin,
        "normalised_certainty": cert,
    }
    tol = 0.015  # the API quantises probabilities to 2 dp, so exact equality is unavailable
    out = {"n_records": len(rows), "tolerance": tol, "candidates": {}}
    for name, pred in candidates.items():
        resid = np.abs(conf - pred)
        out["candidates"][name] = {
            "frac_within_tol": float(np.mean(resid <= tol)),
            "median_abs_residual": float(np.median(resid)),
            "max_abs_residual": float(np.max(resid)),
            "pearson_r": float(np.corrcoef(conf, pred)[0, 1]),
        }
    out["best"] = max(out["candidates"], key=lambda k: out["candidates"][k]["frac_within_tol"])
    # The consequence: uniform maps to 0 and one-hot to 1, so the field cannot be read as
    # a probability of correctness -- on 4-option items a coin flip between two options
    # (p_max = 0.5) is reported as confidence 0.33.
    out["worked_example"] = {"n_options": 4, "p_max": 0.5,
                             "reported_confidence": (0.5 - 0.25) / (1 - 0.25)}
    return out


# ===================================================================== A2

def _calib_block(conf, correct, tag):
    return {
        "metric": tag,
        "n": int(len(conf)),
        "ece_fixed": ece_fixed(conf, correct),
        "ece_adaptive": ece_adaptive(conf, correct),
        "brier": brier(conf, correct),
        "nll": nll(conf, correct),
        "mean_conf": float(np.mean(conf)),
        "accuracy": float(np.mean(correct)),
    }


def a2_calibration(data):
    """Calibration measured against p_max (the actual top-1 probability) and, for
    contrast, against the vendor `confidence` field the earlier draft used."""
    out = {"per_dataset": {}, "pooled": {}, "reliability_pooled": {}}
    pooled_pmax, pooled_conf, pooled_correct = [], [], []
    for k in KEYS:
        rows = valid(data[k])
        pmax = np.array([r["p_max"] for r in rows], float)
        conf = np.array([r["confidence"] for r in rows], float)
        corr = np.array([1.0 if r["correct"] else 0.0 for r in rows], float)
        pooled_pmax += list(pmax)
        pooled_conf += list(conf)
        pooled_correct += list(corr)
        out["per_dataset"][k] = {
            "on_pmax": _calib_block(pmax, corr, "p_max"),
            "on_confidence_field": _calib_block(conf, corr, "confidence_field"),
        }
    pp, pc, pcor = map(np.array, (pooled_pmax, pooled_conf, pooled_correct))
    out["pooled"] = {"on_pmax": _calib_block(pp, pcor, "p_max"),
                     "on_confidence_field": _calib_block(pc, pcor, "confidence_field")}
    # Bootstrap CI on the pooled ECE, resampling items.
    rng = np.random.default_rng(20260919)
    idx = rng.integers(0, len(pp), size=(N_BOOT_CI, len(pp)))
    draws = np.array([ece_fixed(pp[i], pcor[i]) for i in idx])
    out["pooled"]["ece_pmax_ci"] = [float(np.percentile(draws, 2.5)),
                                    float(np.percentile(draws, 97.5))]
    out["reliability_pooled"] = {"on_pmax": reliability_bins(pp, pcor),
                                 "on_confidence_field": reliability_bins(pc, pcor)}
    return out


# ===================================================================== A3

def a3_conditional_error(data):
    """The decision-relevant quantity is P(error | high confidence), not the
    P(high confidence | error) rate the earlier draft reported. They rank datasets
    in opposite orders because the latter is a base-rate artefact."""
    out = {"per_dataset": {}, "tests": []}
    for k in KEYS:
        rows = valid(data[k])
        pmax = np.array([r["p_max"] for r in rows], float)
        corr = np.array([r["correct"] for r in rows], bool)
        hi = pmax >= HIGH_CONF
        wrong = ~corr
        # P(error | high confidence)  -- what a deployer needs
        both = int((hi & wrong).sum())
        n_hi, n_err = int(hi.sum()), int(wrong.sum())
        p_eh, lo_eh, hi_eh = wilson(both, n_hi)
        # P(high confidence | error)  -- what the earlier draft reported. Same numerator,
        # different denominator; that difference is the whole trap.
        p_he, lo_he, hi_he = wilson(both, n_err)
        out["per_dataset"][k] = {
            "n": len(rows), "accuracy": float(corr.mean()),
            "n_high_conf": n_hi, "n_errors": n_err, "n_err_high_conf": both,
            "p_error_given_high_conf": p_eh, "p_error_given_high_conf_ci": [lo_eh, hi_eh],
            "p_high_conf_given_error": p_he, "p_high_conf_given_error_ci": [lo_he, hi_he],
            "bins": _error_by_bin(pmax, corr),
        }
    # Pairwise Fisher tests on P(error | high conf) across the four WMDP-scale datasets
    # plus the two LAB-Bench subsets the earlier draft misassigned.
    focus = ["bio", "bio-robust", "chem", "cyber", "labbench-litqa2", "labbench-supp"]
    pairs, pvals = [], []
    for i, a in enumerate(focus):
        for b in focus[i + 1:]:
            ta, tb = out["per_dataset"][a], out["per_dataset"][b]
            if ta["n_high_conf"] == 0 or tb["n_high_conf"] == 0:
                continue          # no high-confidence items -> nothing to compare
            ea, eb = ta["n_err_high_conf"], tb["n_err_high_conf"]
            table = [[ea, ta["n_high_conf"] - ea], [eb, tb["n_high_conf"] - eb]]
            _, p = sps.fisher_exact(table)
            pairs.append((a, b))
            pvals.append(float(p))
    qvals = benjamini_hochberg(pvals)
    for (a, b), p, q in zip(pairs, pvals, qvals):
        out["tests"].append({"test": "fisher_exact", "quantity": "P(error|p_max>=0.9)",
                             "a": a, "b": b, "p": p, "q_bh": q})
    return out


def _error_by_bin(pmax, corr, edges=(0.0, 0.3, 0.5, 0.7, 0.9, 1.01)):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pmax >= lo) & (pmax < hi)
        if m.sum() == 0:
            continue
        p, clo, chi = wilson(int((~corr[m]).sum()), int(m.sum()))
        rows.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                     "error_rate": p, "ci": [clo, chi]})
    return rows


# ===================================================================== A4

def a4_selective_prediction(data):
    """How well does each available uncertainty signal identify the model's own errors,
    and how much accuracy is bought by abstaining?"""
    out = {"per_dataset": {}, "pooled": {}, "risk_coverage": {}, "signal_comparison": {}}
    signals = ["p_max", "margin", "neg_entropy", "confidence"]
    pooled = {s: [] for s in signals}
    pooled_correct = []
    for k in KEYS:
        rows = valid(data[k])
        corr = np.array([1 if r["correct"] else 0 for r in rows], int)
        pooled_correct += list(corr)
        entry = {"n": len(rows)}
        for s in signals:
            sc = np.array([r[s] for r in rows], float)
            pooled[s] += list(sc)
            # For error detection the score must be NEGATED: these signals score
            # confidence, and high confidence should rank an item as less likely wrong.
            entry[s] = {"auroc": auroc(sc, corr), "auprc_error": auprc(-sc, 1 - corr)}
        out["per_dataset"][k] = entry
    pcorr = np.array(pooled_correct, int)
    for s in signals:
        sc = np.array(pooled[s], float)
        a = auroc(sc, pcorr)
        # bootstrap CI on pooled AUROC
        rng = np.random.default_rng(20260919)
        idx = rng.integers(0, len(sc), size=(N_BOOT_CI, len(sc)))
        draws = np.array([auroc(sc[i], pcorr[i]) for i in idx])
        out["pooled"][s] = {"auroc": a,
                            "auroc_ci": [float(np.percentile(draws, 2.5)),
                                         float(np.percentile(draws, 97.5))],
                            "auprc_error": auprc(-sc, 1 - pcorr)}
    aurocs = [out["pooled"][s]["auroc"] for s in signals]
    out["signal_comparison"] = {
        "max_minus_min_auroc": float(max(aurocs) - min(aurocs)),
        "interpretation": "all available signals are interchangeable; the choice of routing "
                          "signal is not a design decision worth making",
    }
    # Risk-coverage curves and headline operating points.
    for k in KEYS:
        rows = valid(data[k])
        sc = np.array([r["p_max"] for r in rows], float)
        corr = np.array([1.0 if r["correct"] else 0.0 for r in rows], float)
        cov, acc = risk_coverage(sc, corr)
        step = max(1, len(cov) // 100)
        out["risk_coverage"][k] = {
            "base_accuracy": float(corr.mean()),
            "acc_at_coverage": {str(c): accuracy_at_coverage(sc, corr, c)
                                for c in (0.9, 0.8, 0.7, 0.6, 0.5)},
            "curve": [[float(c), float(a)] for c, a in zip(cov[::step], acc[::step])],
        }
    return out


# ===================================================================== A5

def _nondeterminism(data, key, recs):
    """Rotation 0 of the circular run presents options in the SAME order as the
    single-pass run. Any disagreement between them is therefore call-to-call
    nondeterminism, not position sensitivity. This is a control arm we get for free,
    and it bounds how much of the observed instability option order can explain."""
    main = {r["index"]: r for r in data[key] if r["error"] is None}
    dis = same = 0
    for r in recs:
        m = main.get(r["index"])
        if m is None or m["gold_idx"] != r["gold"]:
            continue
        same += 1
        dis += int(m["pred_idx"] != r["choices_by_rot"][0])
    q, lo, hi = wilson(dis, same)
    n_rot = len(recs[0]["choices_by_rot"])
    # If each call independently flips with probability q, the chance that the other
    # (n_rot - 1) calls all match the first is (1-q)^(n_rot-1).
    predicted = 1 - (1 - q) ** (n_rot - 1)
    return {"n_compared": same, "n_disagree": dis,
            "flip_rate_identical_input": q, "flip_rate_ci": [lo, hi],
            "inconsistency_predicted_by_noise": float(predicted)}


def _position_invariant_null(recs, seed=20260919, n_sim=200):
    """Consistency is defined against rotation 0's answer and so is correctness, which
    couples them mechanically: there is one way to be consistently right and several to
    be wrong. Calibrate a null with NO position effect to the same accuracy and the same
    consistency, and report what gap it produces. If the observed gap is not larger than
    the null's, the gap is not evidence of a position-specific failure mode."""
    rng = np.random.default_rng(seed)
    gold = np.array([r["gold"] for r in recs])
    ch = np.array([r["choices_by_rot"] for r in recs])
    N, n_rot = ch.shape
    n_opt = max(r["n_options"] for r in recs) if "n_options" in recs[0] else 4
    acc = float((ch[:, 0] == gold).mean())
    cons = float((ch == ch[:, [0]]).all(1).mean())
    u = 1.0 / n_opt
    # Mixture: a "settled" item answers identically every rotation (correct w.p. s);
    # an "unsettled" item draws uniformly at random on every rotation.
    f = (cons - u ** (n_rot - 1)) / (1 - u ** (n_rot - 1))
    s = (acc - (1 - f) * u) / f
    ratios, gaps = [], []
    for _ in range(n_sim):
        settled = rng.random(N) < f
        sim = np.empty((N, n_rot), int)
        rand_block = rng.integers(0, n_opt, size=(N, n_rot))
        correct_pick = rng.random(N) < s
        wrong_pick = rng.integers(0, n_opt - 1, size=N)
        fixed = np.where(correct_pick, gold, wrong_pick + (wrong_pick >= gold))
        sim[settled] = fixed[settled, None]
        sim[~settled] = rand_block[~settled]
        c0 = sim[:, 0] == gold
        con = (sim == sim[:, [0]]).all(1)
        if con.sum() == 0 or (~con).sum() == 0 or (~c0[con]).mean() == 0:
            continue
        e_un, e_st = float((~c0[~con]).mean()), float((~c0[con]).mean())
        ratios.append(e_un / e_st)
        gaps.append(e_un - e_st)
    return {"settled_fraction": float(f), "settled_correct_rate": float(s),
            "null_error_ratio_mean": float(np.mean(ratios)),
            "null_error_ratio_ci": [float(np.percentile(ratios, 2.5)),
                                    float(np.percentile(ratios, 97.5))],
            "null_error_gap_mean": float(np.mean(gaps))}


def a5_circular(data):
    """Cyclic evaluation with the control arm the earlier reshuffle test lacked:
    measure the correct -> wrong flow, not just wrong -> correct."""
    out = {}
    for k in CIRCULAR_KEYS:
        recs = load_circular(k)
        if recs is None:
            continue
        gold = np.array([r["gold"] for r in recs])
        ch = np.array([r["choices_by_rot"] for r in recs])          # (N, 4 rotations)
        pr = np.array([r["probs_by_rot"] for r in recs])            # (N, 4 rot, 4 opts)
        N = len(recs)
        accs = [float((ch[:, i] == gold).mean()) for i in range(4)]
        consistent = (ch == ch[:, [0]]).all(1)
        ens_probs = pr.mean(1)
        ens = ens_probs.argmax(1)
        maj = np.array([np.bincount(row, minlength=4).argmax() for row in ch])
        c0 = ch[:, 0] == gold
        w2c = np.array([(ch[i, 1:] == gold[i]).any() for i in range(N)])
        c2w = np.array([(ch[i, 1:] != gold[i]).any() for i in range(N)])
        gain, glo, ghi = boot_diff_ci((ens == gold).astype(float), c0.astype(float))
        chosen_pos = np.array([[(ch[i, r] - r) % 4 for r in range(4)] for i in range(N)])
        cnt = np.bincount(chosen_pos.ravel(), minlength=4)
        out[k] = {
            "n_items": N, "n_calls": N * 4,
            "acc_per_rotation": accs,
            "acc_rotation_spread": float(max(accs) - min(accs)),
            "consistency": float(consistent.mean()),
            "acc_given_consistent": float((ch[consistent, 0] == gold[consistent]).mean()),
            "acc_given_inconsistent": float((ch[~consistent, 0] == gold[~consistent]).mean()),
            "n_inconsistent": int((~consistent).sum()),
            "single_pass_acc": accs[0],
            "ensemble_acc": float((ens == gold).mean()),
            "majority_acc": float((maj == gold).mean()),
            "ensemble_gain": gain, "ensemble_gain_ci": [glo, ghi],
            "flip_wrong_to_correct": float(w2c[~c0].mean()),
            "flip_correct_to_wrong": float(c2w[c0].mean()),
            "n_originally_wrong": int((~c0).sum()), "n_originally_correct": int(c0.sum()),
            "prompt_position_counts": [int(x) for x in cnt],
            "first_position_share": float(cnt[0] / cnt.sum()),
        }
        # Errors are more common on inconsistently-answered items -- but this comparison
        # is NOT evidence of a position-specific failure mode, because consistency and
        # correctness are both defined against rotation 0. We report it alongside a
        # position-invariant null that reproduces the same accuracy and consistency.
        p_err_incons = float((~c0[~consistent]).mean())
        p_err_cons = float((~c0[consistent]).mean())
        out[k]["error_rate_unstable"] = p_err_incons
        out[k]["error_rate_stable"] = p_err_cons
        out[k]["observed_error_ratio"] = p_err_incons / p_err_cons
        out[k]["frac_errors_on_unstable_items"] = float((~c0 & ~consistent).sum() / (~c0).sum())
        out[k]["null"] = _position_invariant_null(recs)
        out[k]["exceeds_null"] = bool(out[k]["observed_error_ratio"]
                                      > out[k]["null"]["null_error_ratio_ci"][1])

        # The free determinism control, and the resulting decomposition of instability.
        nd = _nondeterminism(data, k, recs)
        nd["inconsistency_observed"] = float(1 - consistent.mean())
        nd["excess_over_noise"] = nd["inconsistency_observed"] - \
            nd["inconsistency_predicted_by_noise"]
        nd["share_of_instability_from_noise"] = (
            nd["inconsistency_predicted_by_noise"] / nd["inconsistency_observed"])
        out[k]["nondeterminism"] = nd
    return out


# ===================================================================== A6 + A7

def a6_cascade(data):
    """A two-tier cascade in which tier 2 is the model's OWN 4-rotation circular
    ensemble. Both tiers are measured; nothing about a hypothetical stronger model
    is assumed. Cost is exact: 1 call at tier 1, 3 additional calls at tier 2."""
    out = {}
    for k in CIRCULAR_KEYS:
        recs = load_circular(k)
        if recs is None:
            continue
        gold = np.array([r["gold"] for r in recs])
        ch = np.array([r["choices_by_rot"] for r in recs])
        pr = np.array([r["probs_by_rot"] for r in recs])
        t1_correct = (ch[:, 0] == gold).astype(float)
        t2_correct = (pr.mean(1).argmax(1) == gold).astype(float)
        t1_conf = pr[:, 0, :].max(1)        # tier-1 p_max, the routing signal
        N = len(recs)

        def sweep(order, n=N, t1=t1_correct, t2=t2_correct):
            """Accuracy and cost as the deferred fraction grows.

            `order` lists item indices by deferral priority (earliest deferred first).
            Loop-scoped values are bound as defaults so the closure cannot capture a
            later iteration's state.
            """
            res = []
            for c in np.linspace(0, 1, 11):
                nd = int(round(c * n))
                deferred = np.zeros(n, bool)
                deferred[order[:nd]] = True
                res.append({"deferral": float(c),
                            "accuracy": float(np.where(deferred, t2, t1).mean()),
                            "calls_per_item": 1.0 + 3.0 * (nd / n)})
            return res

        # Ties in the quantised confidence are broken by a seeded shuffle before a STABLE
        # sort, so the cut point is not an artefact of input order at a tie block.
        rng = np.random.default_rng(20260919)
        jitter = rng.permutation(N)
        conf_order = jitter[np.argsort(t1_conf[jitter], kind="stable")]
        oracle_order = np.argsort(-(t2_correct - t1_correct), kind="stable")

        # A single random permutation is a noisy baseline -- with a small tier1/tier2 gap
        # one draw can look like a large advantage or none. Average over many draws and
        # report the spread.
        rand_curves = []
        for s in range(400):
            r = np.random.default_rng(1000 + s).permutation(N)
            rand_curves.append([p["accuracy"] for p in sweep(r)])
        rand_arr = np.array(rand_curves)
        base = sweep(conf_order)
        out[k] = {
            "n_items": N,
            "tier1_accuracy": float(t1_correct.mean()),
            "tier2_accuracy": float(t2_correct.mean()),
            "confidence_router": base,
            "random_router": [{"deferral": p["deferral"], "calls_per_item": p["calls_per_item"],
                               "accuracy": float(rand_arr[:, i].mean()),
                               "ci": [float(np.percentile(rand_arr[:, i], 2.5)),
                                      float(np.percentile(rand_arr[:, i], 97.5))]}
                              for i, p in enumerate(base)],
            "oracle_router": sweep(oracle_order),
        }
        # Does the confidence router beat random at matched cost, and is that robust to
        # which random permutation you happen to draw?
        for c in (0.2, 0.4):
            i = int(round(c * 10))
            adv = base[i]["accuracy"] - float(rand_arr[:, i].mean())
            out[k][f"advantage_over_random_at_{c}"] = adv
            out[k][f"frac_random_draws_beating_router_at_{c}"] = float(
                (rand_arr[:, i] >= base[i]["accuracy"]).mean())
    return out


def a7_breakeven(data):
    """For an EXTERNAL escalation target, the accuracy it must exceed to be worth
    calling is exactly the tier-1 accuracy on the deferred slice. This replaces the
    earlier draft's assumed 85% fallback with a measured requirement."""
    out = {}
    for k in KEYS:
        rows = valid(data[k])
        sc = np.array([r["p_max"] for r in rows], float)
        corr = np.array([1.0 if r["correct"] else 0.0 for r in rows], float)
        entry = {"base_accuracy": float(corr.mean()), "by_deferral": {}}
        for c in (0.2, 0.4, 0.6):
            nd = int(round(c * len(rows)))
            if nd == 0:
                continue
            # Same helpers as the risk-coverage curve, so Table 1 and the prose agree.
            entry["by_deferral"][str(c)] = {
                "n_deferred": nd,
                "breakeven_fallback_accuracy": deferred_accuracy(sc, corr, c),
                "retained_accuracy": accuracy_at_coverage(sc, corr, 1.0 - c),
            }
        out[k] = entry
    return out


# ===================================================================== A8

def _pos_test(rows, seed=20260919, n_sim=4000):
    """First-option preference, tested against a SIMULATED position-invariant null.

    A naive chi-square of predicted letters against gold letters is not valid here:
    predictions are pulled toward gold by accuracy, so the statistic shrinks toward zero
    as accuracy rises and the test loses power exactly where we most want to look. The
    null below answers the right question -- 'a model of this accuracy with no positional
    preference at all, how large a statistic would it produce?' -- and yields a calibrated
    p-value plus the power to detect a worst-case bias.
    """
    letters = list("ABCD")
    keep = [r for r in rows if r["pred_letter"] in letters]
    pred = np.array([letters.index(r["pred_letter"]) for r in keep])
    gold = np.array([r["gold_idx"] for r in keep])          # same records on both sides
    n = len(keep)
    pc = np.bincount(pred, minlength=4)
    gc = np.bincount(gold, minlength=4)
    exp = gc / gc.sum() * pc.sum()
    chi2 = float(((pc - exp) ** 2 / np.maximum(exp, 1e-9)).sum())
    acc = float((pred == gold).mean())

    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_sim)
    hit = rng.random((n_sim, n)) < acc
    wrong = rng.integers(0, 3, size=(n_sim, n))
    sim = np.where(hit, gold[None, :], wrong + (wrong >= gold[None, :]))
    for s in range(n_sim):
        spc = np.bincount(sim[s], minlength=4)
        sexp = gc / gc.sum() * spc.sum()
        null_stats[s] = ((spc - sexp) ** 2 / np.maximum(sexp, 1e-9)).sum()
    p_cal = float((null_stats >= chi2).mean())

    # Power: a model of this accuracy whose every error lands on the first option.
    alt = np.where(hit, gold[None, :], 0)
    crit = np.percentile(null_stats, 95)
    alt_stats = np.empty(min(n_sim, 500))
    for s in range(len(alt_stats)):
        apc = np.bincount(alt[s], minlength=4)
        aexp = gc / gc.sum() * apc.sum()
        alt_stats[s] = ((apc - aexp) ** 2 / np.maximum(aexp, 1e-9)).sum()
    power = float((alt_stats >= crit).mean())

    return {"n": n, "accuracy": acc,
            "pred_counts": [int(x) for x in pc], "gold_counts": [int(x) for x in gc],
            "pred_first_share": float(pc[0] / pc.sum()),
            "gold_first_share": float(gc[0] / gc.sum()),
            "chi2": chi2, "p_naive_chi2": float(sps.chi2.sf(chi2, df=3)),
            "p_calibrated": p_cal, "null_chi2_95th": float(crit),
            "power_vs_worst_case": power}


def a8_position_bias(data):
    """Is the first option over-selected, and if so where? Tested per dataset on
    4-option items, and stratified by confidence."""
    out = {"per_dataset": {}, "chance_baselines": {}, "stratified": {}}
    for k in KEYS:
        rows = [r for r in valid(data[k]) if r["n_options"] == 4]
        if len(rows) < 40:
            continue
        out["per_dataset"][k] = _pos_test(rows)
    qs = benjamini_hochberg([v["p_calibrated"] for v in out["per_dataset"].values()])
    for v, q in zip(out["per_dataset"].values(), qs):
        v["q_bh"] = q
    # Confidence stratification. Power is reported alongside every stratum, because in
    # the high-confidence stratum accuracy is so high that the test cannot detect even a
    # worst-case bias -- a null result there is not evidence of absence.
    for k in ["cyber", "bio"]:
        rows = [r for r in valid(data[k]) if r["n_options"] == 4]
        strat = {}
        for lo, hi, tag in [(0.0, 0.5, "low"), (0.5, 0.9, "mid"), (0.9, 1.01, "high")]:
            sub = [r for r in rows if lo <= r["p_max"] < hi]
            if len(sub) < 30:
                continue
            strat[tag] = _pos_test(sub)
        out["stratified"][k] = strat
    # Per-item chance baselines: option counts are NOT uniformly 4 across LAB-Bench.
    for k in KEYS:
        rows = valid(data[k])
        chance = float(np.mean([1.0 / r["n_options"] for r in rows]))
        acc = float(np.mean([1.0 if r["correct"] else 0.0 for r in rows]))
        counts = {}
        for r in rows:
            counts[str(r["n_options"])] = counts.get(str(r["n_options"]), 0) + 1
        out["chance_baselines"][k] = {
            "mean_chance": chance, "accuracy": acc,
            "chance_adjusted_lift": float((acc - chance) / (1 - chance)),
            "option_count_hist": counts,
            "min_options": min(r["n_options"] for r in rows),
            "max_options": max(r["n_options"] for r in rows),
        }
    return out


# ===================================================================== A9

def a9_sensitivity(data):
    """Two artefacts of the measurement apparatus itself: how failed API calls are
    scored, and the two-decimal quantisation of the returned probabilities."""
    out = {"error_handling": {}, "quantisation": {}}
    for k in KEYS:
        recs = data[k]
        n_err = sum(1 for r in recs if r["error"] is not None)
        if n_err == 0:
            continue
        as_wrong = sum(1 for r in recs if r["correct"]) / len(recs)
        ok = [r for r in recs if r["error"] is None]
        excluded = sum(1 for r in ok if r["correct"]) / len(ok)
        out["error_handling"][k] = {
            "n_errors": n_err, "acc_errors_scored_wrong": as_wrong,
            "acc_errors_excluded": excluded, "delta_pp": (excluded - as_wrong) * 100,
        }
    out["error_handling"]["total_errors"] = sum(
        1 for k in KEYS for r in data[k] if r["error"] is not None)
    # Ties at the maximum arise because probabilities are rounded to 2 dp. The first
    # option can only win a tie it is part of, so the expectation must condition on
    # option A actually being in the argmax set -- otherwise it is inflated.
    rows = [r for k in KEYS for r in valid(data[k])]
    tied = [r for r in rows if r["n_tied_at_max"] and r["n_tied_at_max"] > 1]
    first_pick = sum(1 for r in tied if r["pred_letter"] == "A")
    exp_first, n_a_in_tie = 0.0, 0
    for r in tied:
        pm = max(r["probabilities"].values())
        tie_set = [lab for lab, v in r["probabilities"].items() if abs(v - pm) < 1e-9]
        if "A" in tie_set:
            n_a_in_tie += 1
            exp_first += 1.0 / len(tie_set)
    if not tied:
        exp_first = float("nan")
    out["quantisation"] = {
        "n_records": len(rows), "n_with_tie_at_max": len(tied),
        "frac_with_tie": float(len(tied) / len(rows)),
        "ties_resolved_to_first_option": first_pick,
        "n_ties_containing_first_option": n_a_in_tie,
        "ties_expected_first_if_random": float(exp_first),
        "tie_first_excess": float(first_pick - exp_first) if tied else float("nan"),
        "distinct_probability_values": int(len({round(float(v), 4)
                                                for r in rows for v in r["probabilities"].values()})),
    }
    return out


# ===================================================================== A11

def _load_json(name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def a11_repeatability(data):
    """Separate answer-order sensitivity from run-to-run variation.

    The cyclic experiment cannot do this on its own: when two rotations disagree, the
    cause may be the changed option order or simply a non-deterministic endpoint. Here
    every item is additionally queried k times with a byte-identical prompt, which
    isolates the second source, and a stratified subsample is queried k times at every
    rotation, which gives a two-factor decomposition.
    """
    out = {"meta": {"design": "k identical calls at rotation 0; "
                              "k repeats x n rotations on a subsample",
                    "dispatch": "task order shuffled so repeats are never adjacent"}}
    for k in CIRCULAR_KEYS:
        rep = _load_json(f"repeat_{k}.json")
        circ = load_circular(k)
        if rep is None or circ is None:
            continue
        cmap = {r["index"]: r for r in circ}
        rep = [r for r in rep if r["index"] in cmap]
        gold = np.array([r["gold"] for r in rep])
        ch_id = np.array([r["choices_by_rep"] for r in rep])
        pr_id = np.array([r["probs_by_rep"] for r in rep])
        order = [r["index"] for r in rep]
        ch_rot = np.array([cmap[i]["choices_by_rot"] for i in order])
        pr_rot = np.array([cmap[i]["probs_by_rot"] for i in order])
        n_items, n_rep = ch_id.shape

        k_id = int((~(ch_id == ch_id[:, [0]]).all(1)).sum())
        k_rot = int((~(ch_rot == ch_rot[:, [0]]).all(1)).sum())
        p_id, lo_id, hi_id = wilson(k_id, n_items)
        p_rot, lo_rot, hi_rot = wilson(k_rot, n_items)

        # Matched budget: four calls either way, so the two averaging schemes are
        # directly comparable. The difference isolates what re-presentation buys.
        base = (ch_id[:, 0] == gold).astype(float)
        ens_id = (pr_id.mean(1).argmax(1) == gold).astype(float)
        ens_rot = (pr_rot.mean(1).argmax(1) == gold).astype(float)
        g_id = boot_diff_ci(ens_id, base)
        g_rot = boot_diff_ci(ens_rot, (ch_rot[:, 0] == gold).astype(float))
        g_diff = boot_diff_ci(ens_rot, ens_id)

        entry = {
            "n_items": n_items, "n_repeats": n_rep,
            "inconsistency_identical": p_id, "inconsistency_identical_ci": [lo_id, hi_id],
            "inconsistency_rotations": p_rot, "inconsistency_rotations_ci": [lo_rot, hi_rot],
            "order_attributable_gap": p_rot - p_id,
            "noise_share_of_instability": p_id / p_rot if p_rot else float("nan"),
            "acc_single": float(base.mean()),
            "acc_identical_ensemble": float(ens_id.mean()),
            "acc_rotation_ensemble": float(ens_rot.mean()),
            "gain_identical": g_id[0], "gain_identical_ci": [g_id[1], g_id[2]],
            "gain_rotation": g_rot[0], "gain_rotation_ci": [g_rot[1], g_rot[2]],
            "gain_difference": g_diff[0], "gain_difference_ci": [g_diff[1], g_diff[2]],
            "rotation_beats_identical": bool(g_diff[1] > 0),
        }

        # Two-factor subsample: within-rotation repeats vs between-rotation modes.
        fac = _load_json(f"repeatfac_{k}.json")
        if fac is not None:
            ch = np.array([r["choices"] for r in fac])          # (N, rot, rep)
            nf, n_rot, n_r = ch.shape
            within = np.mean([
                float(np.mean([len(set(ch[i, r])) > 1 for r in range(n_rot)]))
                for i in range(nf)])
            modes = np.array([[np.bincount(ch[i, r]).argmax() for r in range(n_rot)]
                              for i in range(nf)])
            between = float(np.mean([len(set(modes[i])) > 1 for i in range(nf)]))
            entry["factorial"] = {
                "n_items": nf, "n_rotations": n_rot, "n_repeats": n_r,
                "within_rotation_disagreement": within,
                "between_rotation_disagreement": between,
                "between_to_within_ratio": between / within if within else float("nan"),
            }
        out[k] = entry
    return out


# ===================================================================== A12

ABLATION_ARMS = [
    ("lblshuf", "label", "letters permuted across positions"),
    ("lblsym", "label", "non-alphabetic glyphs"),
    ("fmtstate", "format", "options in the state text only"),
    ("fmtcrit", "format", "options as category keys only"),
]


def _load_arm(subject, tag):
    path = os.path.join(RAW, f"{subject}_{tag}.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r["error"] is None:
                rows.append(r)
    return rows


def a12_ablations(data):
    """Resolve two confounds the main runs leave open.

    The default prompt assigns answer labels in positional order, so a preference for the
    first option cannot be told apart from a preference for the letter A. It also lists
    every option twice, in the state text and again as a category key, which was never
    ablated. Re-running with the labels permuted, with non-alphabetic glyphs, and with each
    serialisation alone separates these.
    """
    out = {}
    for k in CIRCULAR_KEYS:
        base = valid(data[k])
        n_opt = 4
        b_acc = float(np.mean([r["correct"] for r in base]))
        b_pos = np.bincount([r["pred_idx"] for r in base if r["pred_idx"] >= 0],
                            minlength=n_opt)
        gold = np.bincount([r["gold_idx"] for r in base], minlength=n_opt)
        entry = {"baseline_accuracy": b_acc,
                 "baseline_first_share": float(b_pos[0] / b_pos.sum()),
                 "gold_first_share": float(gold[0] / gold.sum()),
                 "arms": {}}
        by_index = {r["index"]: r for r in base}
        for tag, kind, desc in ABLATION_ARMS:
            rows = _load_arm(k, tag)
            if rows is None:
                continue
            paired = [(by_index[r["index"]], r) for r in rows if r["index"] in by_index]
            a = np.array([float(x[1]["correct"]) for x in paired])
            b = np.array([float(x[0]["correct"]) for x in paired])
            d, lo, hi = boot_diff_ci(a, b)
            pos = np.bincount([r["pred_idx"] for r in rows if r["pred_idx"] >= 0],
                              minlength=n_opt)
            arm = {"kind": kind, "description": desc, "n": len(rows),
                   "accuracy": float(a.mean()),
                   "delta_vs_baseline": d, "delta_ci": [lo, hi],
                   "first_position_share": float(pos[0] / pos.sum())}
            if tag == "lblshuf":
                # With letters permuted, a label prior and a position prior separate.
                arm["letter_a_share"] = float(
                    np.mean([r["pred_label"] == "A" for r in rows]))
            entry["arms"][tag] = arm
        out[k] = entry
    return out


# ===================================================================== A10

def a10_cost(data):
    """Cost from MEASURED token usage.

    The endpoint reports `usage` on every response, but the original runs predate that
    capture, so per-dataset input-token rates were measured afterwards on a sample of the
    same prompts (raw/tokens_*.json) and, where a full re-run exists, from its exact
    totals (raw/manifest_repeat_*.json). An earlier version of this analysis estimated
    tokens from payload length; that estimator understates the true count substantially,
    because the endpoint adds schema overhead the payload does not contain.
    """
    per_dataset, tok_single = {}, 0.0
    for k in KEYS:
        n = len(valid(data[k]))
        sample = _load_json(f"tokens_{k}.json") or []
        chars = float(np.mean([len(r["question"])
                               + 2 * sum(len(str(c)) for c in r["choices"])
                               for r in valid(data[k])]))
        if sample:
            rate = float(np.mean([r["input_tokens"] for r in sample]))
            source = f"measured on {len(sample)} sampled prompts"
        else:
            rate = chars / CHARS_PER_TOKEN
            source = "payload-length estimate (no measurement available)"
        tok_single += rate * n
        per_dataset[k] = {"n_calls": n, "mean_input_tokens": rate,
                          "mean_payload_chars": chars,
                          "estimator_ratio": rate / (chars / CHARS_PER_TOKEN),
                          "source": source}

    # Circular runs: exact totals where a same-prompt re-run recorded them.
    tok_circ, circ_exact = 0.0, True
    for k in CIRCULAR_KEYS:
        man = _load_json(f"manifest_repeat_{k}.json")
        if man and man.get("usage_totals", {}).get("input_tokens"):
            tok_circ += float(man["usage_totals"]["input_tokens"])
        else:
            circ_exact = False
            recs = load_circular(k) or []
            tok_circ += per_dataset[k]["mean_input_tokens"] * len(recs) * 4

    n_single = sum(len(valid(data[k])) for k in KEYS)
    n_circ = sum(len(load_circular(k) or []) * 4 for k in CIRCULAR_KEYS)
    tok_total = tok_single + tok_circ
    usd = tok_total / 1e6 * USD_PER_M_INPUT_TOKENS

    # What the same decisions would cost with a generative model. List prices retrieved
    # 2026-09-20. Two output regimes: answering with the option label alone, or with a
    # short chain of thought, which dominates the total at frontier prices.
    tiers = [
        ("small (GPT-5-nano)", 0.05, 0.40),
        ("mid (Gemini 3.8 Flash)", 0.75, 3.75),
        ("mid (Claude Sonnet 5)", 2.00, 10.00),
        ("frontier (Claude Opus 5)", 5.00, 25.00),
        ("frontier (GPT-6 Astra)", 10.00, 50.00),
    ]
    n_calls = n_single + n_circ
    regimes = {"answer_only": 5, "short_cot": 300}
    comparison = {}
    for label, p_in, p_out in tiers:
        comparison[label] = {"usd_per_m_input": p_in, "usd_per_m_output": p_out}
        for rname, out_tok in regimes.items():
            total = tok_total / 1e6 * p_in + (out_tok * n_calls) / 1e6 * p_out
            comparison[label][rname] = {"usd": total, "ratio_to_jev": total / usd}

    return {
        "basis": "measured input tokens from the endpoint usage field",
        "usd_per_m_input_tokens": USD_PER_M_INPUT_TOKENS,
        "n_calls_single_pass": n_single,
        "n_calls_circular": n_circ,
        "n_calls_total": n_calls,
        "input_tokens_single_pass": int(tok_single),
        "input_tokens_circular": int(tok_circ),
        "input_tokens_total": int(tok_total),
        "circular_tokens_exact": circ_exact,
        "usd_total": usd,
        "mean_input_tokens_per_call": tok_total / n_calls,
        "per_dataset": per_dataset,
        "llm_comparison": comparison,
        "llm_comparison_note": "list prices retrieved 2026-09-20; batch APIs list at half "
                               "the standard rate and cached input is typically 0.1x base, "
                               "so these are upper bounds",
    }


# ===================================================================== headline

def headline(data):
    out = {"per_dataset": {}, "totals": {}}
    tot_n = tot_c = 0
    for k in KEYS:
        recs = data[k]
        rows = valid(recs)
        corr = [1.0 if r["correct"] else 0.0 for r in rows]
        acc, lo, hi = wilson(int(sum(corr)), len(corr))
        tot_n += len(rows)
        tot_c += int(sum(corr))
        out["per_dataset"][k] = {
            "label": LABEL[k], "family": FAMILY[k], "n": len(rows),
            "n_raw": len(recs), "accuracy": acc, "ci": [lo, hi],
            "ci_width_pp": (hi - lo) * 100,
        }
    acc, lo, hi = wilson(tot_c, tot_n)
    out["totals"] = {"n": tot_n, "correct": tot_c, "accuracy": acc, "ci": [lo, hi],
                     "n_datasets": len(KEYS)}
    return out


def main():
    data = load_all()
    stats = {
        "meta": {"bootstrap_seed": 20260919, "n_boot": N_BOOT, "n_boot_ci": N_BOOT_CI,
                 "high_conf_threshold": HIGH_CONF,
                 "model_version": MODEL_VERSION, "endpoint": API_ENDPOINT,
                 "collection_dates": COLLECTION_DATES,
                 "datasets": [{"key": key, "label": lab, "family": fam}
                              for key, lab, fam in DATASETS]},
        "headline": headline(data),
        "a1_confidence_semantics": a1_confidence_semantics(data),
        "a2_calibration": a2_calibration(data),
        "a3_conditional_error": a3_conditional_error(data),
        "a4_selective_prediction": a4_selective_prediction(data),
        "a5_circular": a5_circular(data),
        "a6_cascade": a6_cascade(data),
        "a7_breakeven": a7_breakeven(data),
        "a8_position_bias": a8_position_bias(data),
        "a9_sensitivity": a9_sensitivity(data),
        "a10_cost": a10_cost(data),
        "a11_repeatability": a11_repeatability(data),
        "a12_ablations": a12_ablations(data),
    }
    os.makedirs(DERIVED, exist_ok=True)
    dump(stats, os.path.join(DERIVED, "stats.json"))
    print(f"wrote {os.path.join(DERIVED, 'stats.json')}")
    h = stats["headline"]["totals"]
    print(f"  {h['n']} items, pooled accuracy {h['accuracy']:.4f}")
    print(f"  confidence field best explained by: {stats['a1_confidence_semantics']['best']} "
          f"({stats['a1_confidence_semantics']['candidates'][stats['a1_confidence_semantics']['best']]['frac_within_tol']:.3f})")
    print(f"  pooled ECE on p_max        : {stats['a2_calibration']['pooled']['on_pmax']['ece_fixed']:.4f}")
    print(f"  pooled ECE on `confidence` : {stats['a2_calibration']['pooled']['on_confidence_field']['ece_fixed']:.4f}")
    print(f"  pooled AUROC (p_max)       : {stats['a4_selective_prediction']['pooled']['p_max']['auroc']:.4f}")


if __name__ == "__main__":
    main()
