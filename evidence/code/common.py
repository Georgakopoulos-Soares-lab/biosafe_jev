"""Shared loading and statistics helpers for the Jev evaluation evidence package.

Every number that appears in the manuscript is produced by `analyze.py` using these helpers
and written to `derived/stats.json`. Nothing is typed by hand into the paper.
"""
import json
import os

import numpy as np
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
DERIVED = os.path.join(ROOT, "derived")
FIGURES = os.path.join(ROOT, "figures")

BOOTSTRAP_SEED = 20260919
N_BOOT = 10000

# Display order and labels used across figures, tables and the manuscript.
DATASETS = [
    ("bio", "WMDP-Bio", "WMDP"),
    ("bio-robust", "WMDP-Bio-Robust", "WMDP"),
    ("chem", "WMDP-Chem", "WMDP"),
    ("cyber", "WMDP-Cyber", "WMDP"),
    ("labbench-cloning", "LB-CloningScenarios", "LAB-Bench"),
    ("labbench-protocol", "LB-ProtocolQA", "LAB-Bench"),
    ("labbench-seq", "LB-SeqQA", "LAB-Bench"),
    ("labbench-db", "LB-DbQA", "LAB-Bench"),
    ("labbench-litqa2", "LB-LitQA2", "LAB-Bench"),
    ("labbench-supp", "LB-SuppQA", "LAB-Bench"),
]
KEYS = [k for k, _, _ in DATASETS]
LABEL = {k: lab for k, lab, _ in DATASETS}
FAMILY = {k: fam for k, _, fam in DATASETS}

# TypeSafe published input pricing at the time of the runs (USD per million input tokens).
# Output tokens are not billed for the System-One endpoint.
USD_PER_M_INPUT_TOKENS = 0.042
CHARS_PER_TOKEN = 4.0  # standard rough estimate; flagged as an estimate wherever it is used


# --------------------------------------------------------------------------- loading

def load_records(key):
    """Load one dataset's decision records, annotated with derived uncertainty signals."""
    path = os.path.join(RAW, f"{key}.jsonl")
    out = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            r["dataset"] = key
            r["n_options"] = len(r["choices"])
            if r["error"] is None and r["probabilities"]:
                p = np.array(sorted(r["probabilities"].values(), reverse=True), dtype=float)
                r["p_max"] = float(p[0])
                r["margin"] = float(p[0] - p[1]) if len(p) > 1 else float(p[0])
                r["neg_entropy"] = float(-_entropy(p))
                r["n_tied_at_max"] = int(np.sum(np.isclose(p, p[0])))
            else:
                r["p_max"] = r["margin"] = r["neg_entropy"] = None
                r["n_tied_at_max"] = None
            out.append(r)
    return out


def load_all():
    return {k: load_records(k) for k in KEYS}


def load_circular(key):
    """Load a cyclic-permutation run: per item, the chosen option and full probability
    vector for each rotation, already mapped back to ORIGINAL option indices."""
    path = os.path.join(RAW, f"perm_{key}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def valid(recs):
    """Records with a successful API response. See `error_sensitivity` for the alternative."""
    return [r for r in recs if r["error"] is None and r["p_max"] is not None]


def _entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


# --------------------------------------------------------------------------- intervals

def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion. Correct at the extremes, unlike
    the normal approximation, which matters for the small LAB-Bench subsets."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(phat), float(max(0.0, centre - half)), float(min(1.0, centre + half))


def boot_ci(values, statistic=np.mean, n_boot=N_BOOT, seed=BOOTSTRAP_SEED, alpha=0.05):
    """Percentile bootstrap CI. Seeded, so `analyze.py` is reproducible bit for bit."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    draws = statistic(values[idx], axis=1)
    return (float(statistic(values)),
            float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def boot_diff_ci(a, b, n_boot=N_BOOT, seed=BOOTSTRAP_SEED, alpha=0.05):
    """Bootstrap CI for a paired difference in means (same items, two conditions)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    draws = a[idx].mean(1) - b[idx].mean(1)
    return (float(a.mean() - b.mean()),
            float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def benjamini_hochberg(pvals):
    """Return BH-adjusted q-values, preserving input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.clip(ranked, 0, 1)
    return [float(x) for x in q]


# --------------------------------------------------------------------------- calibration

def ece_fixed(conf, correct, n_bins=10):
    """Expected calibration error with equal-width bins (Guo et al. 2017)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            total += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(total / len(conf))


def ece_adaptive(conf, correct, n_bins=10):
    """Equal-mass (adaptive) binning. Fixed-width ECE is unstable when the confidence
    distribution is concentrated, which it is here."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(conf)
    conf, correct = conf[order], correct[order]
    splits = np.array_split(np.arange(len(conf)), n_bins)
    total = 0.0
    for s in splits:
        if len(s):
            total += len(s) * abs(correct[s].mean() - conf[s].mean())
    return float(total / len(conf))


def brier(conf, correct):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return float(np.mean((conf - correct) ** 2))


def nll(conf, correct, eps=1e-6):
    conf, correct = np.clip(np.asarray(conf, float), eps, 1 - eps), np.asarray(correct, float)
    return float(-np.mean(correct * np.log(conf) + (1 - correct) * np.log(1 - conf)))


def reliability_bins(conf, correct, n_bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        acc, lo, hi = wilson(int(correct[m].sum()), int(m.sum()))
        rows.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                     "mean_conf": float(conf[m].mean()), "accuracy": acc,
                     "ci_lo": lo, "ci_hi": hi, "n": int(m.sum())})
    return rows


# --------------------------------------------------------------------------- discrimination

def auroc(scores, labels):
    """AUROC via the rank (Mann-Whitney) identity; handles ties correctly, which matters
    because the API quantises probabilities to two decimals."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = sps.rankdata(scores)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def auprc(scores, labels):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(np.asarray(labels, int), np.asarray(scores, float)))


def risk_coverage(scores, correct):
    """Accuracy on the retained set as a function of coverage, deferring lowest score first."""
    scores, correct = np.asarray(scores, float), np.asarray(correct, float)
    order = np.argsort(-scores)  # most confident first
    kept = correct[order]
    cum = np.cumsum(kept) / np.arange(1, len(kept) + 1)
    cov = np.arange(1, len(kept) + 1) / len(kept)
    return cov, cum


def accuracy_at_coverage(scores, correct, coverage):
    cov, acc = risk_coverage(scores, correct)
    i = int(np.clip(round(coverage * len(cov)) - 1, 0, len(cov) - 1))
    return float(acc[i])


def dump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
