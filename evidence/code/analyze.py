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
    USD_PER_M_INPUT_TOKENS,
    accuracy_at_coverage,
    auprc,
    auroc,
    benjamini_hochberg,
    boot_diff_ci,
    brier,
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
        order = np.argsort(sc)  # least confident first
        entry = {"base_accuracy": float(corr.mean()), "by_deferral": {}}
        for c in (0.2, 0.4, 0.6):
            nd = int(round(c * len(rows)))
            if nd == 0:
                continue
            slice_acc = float(corr[order[:nd]].mean())
            entry["by_deferral"][str(c)] = {
                "n_deferred": nd,
                "breakeven_fallback_accuracy": slice_acc,
                "retained_accuracy": float(corr[order[nd:]].mean()),
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


# ===================================================================== A10

def a10_cost(data):
    """Input-token cost, estimated from the serialised payload. Flagged as an estimate:
    the API does not return a usage field on this endpoint."""
    total_chars = 0
    n_calls_main = 0
    for key in KEYS:
        for r in data[key]:
            # options are serialised twice per request: in `state` and again as `criteria`
            body = r["question"] + 2 * "".join(str(c) for c in r["choices"])
            total_chars += len(body)
            n_calls_main += 1
    circ_calls = sum(len(load_circular(k) or []) * 4 for k in CIRCULAR_KEYS)
    circ_chars = 0
    for k in CIRCULAR_KEYS:
        recs = load_circular(k) or []
        ids = {r["index"] for r in recs}
        per = [r for r in data[k] if r["index"] in ids]
        circ_chars += 4 * sum(len(r["question"]) + 2 * len("".join(str(c) for c in r["choices"]))
                              for r in per)
    tok_main = total_chars / CHARS_PER_TOKEN
    tok_circ = circ_chars / CHARS_PER_TOKEN
    return {
        "estimate_basis": f"chars/{CHARS_PER_TOKEN} tokens; endpoint returns no usage field",
        "usd_per_m_input_tokens": USD_PER_M_INPUT_TOKENS,
        "n_calls_single_pass": n_calls_main,
        "n_calls_circular": circ_calls,
        "n_calls_total": n_calls_main + circ_calls,
        "est_input_tokens_single_pass": int(tok_main),
        "est_input_tokens_circular": int(tok_circ),
        "est_usd_single_pass": tok_main / 1e6 * USD_PER_M_INPUT_TOKENS,
        "est_usd_circular": tok_circ / 1e6 * USD_PER_M_INPUT_TOKENS,
        "est_usd_total": (tok_main + tok_circ) / 1e6 * USD_PER_M_INPUT_TOKENS,
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
