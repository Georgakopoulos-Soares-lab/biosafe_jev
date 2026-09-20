# Evidence package

Data and analysis code behind *Auditing System-1 Models on Biosecurity-Relevant
Benchmarks*. See the repository [README](../README.md) for how to build the paper.

## Layout

```
code/      analysis and data-collection scripts
raw/       decision records: one JSONL per dataset, plus perm_*.json for the cyclic runs
derived/   stats.json (the single source of every reported number) and tables/table1_main.tex
figures/   the paper's two figures, as PDF (for LaTeX) and PNG (for inspection)
```

## Scripts

| Script | Purpose |
|---|---|
| `common.py` | loading, interval, calibration, and discrimination helpers |
| `analyze.py` | computes every reported statistic into `derived/stats.json` |
| `make_figures.py` | `figures/fig1_uncertainty`, `figures/fig2_permutation` |
| `make_tables.py` | `derived/tables/table1_main.tex` |
| `export_numbers.py` | `stats.json` → `manuscript/numbers.tex` LaTeX macros |
| `make_manifest.py` | `MANIFEST.md`, tracing each macro to its source |
| `jev_client.py` | shared API client: request construction, retry, response parsing |
| `run_eval.py` | single-pass data collection (needs `TYPESAFE_API_KEY`) |
| `run_circular.py` | cyclic-rotation data collection (needs `TYPESAFE_API_KEY`) |
| `run_repeat.py` | repeatability study: identical-prompt repeats, rotation x repeat factorial, and per-dataset token measurement (needs `TYPESAFE_API_KEY`) |

Only the first six are needed to reproduce the analysis; the last four contact the API.

## What each analysis block establishes

| `stats.json` block | Establishes |
|---|---|
| `a1_confidence_semantics` | the vendor `confidence` field is `(p_max − 1/n)/(1 − 1/n)`, not a probability of correctness |
| `a2_calibration` | calibration on `p_max` (pooled ECE 0.034) against the vendor field (0.091) |
| `a3_conditional_error` | `P(error \| p_max ≥ 0.9)` per dataset, with pairwise Fisher tests |
| `a4_selective_prediction` | AUROC for four candidate signals; risk–coverage curves |
| `a5_circular` | answer stability across rotations; run-to-run variation; the null model below |
| `a6_cascade` | accuracy against inference cost, with random (400 draws) and oracle baselines |
| `a7_breakeven` | accuracy an external second-stage model would have to exceed (not used in the paper) |
| `a8_position_bias` | first-option preference against a simulated null, with power reported |
| `a9_sensitivity` | error-scoring convention; whether quantisation ties explain the bias (they do not) |
| `a10_cost` | call counts and measured spend, with a generative-LLM price comparison |
| `a11_repeatability` | identical-prompt vs rotation instability, the two-factor decomposition, and the matched-budget averaging control |
| `a12_ablations` | label scheme (permuted letters, non-alphabetic glyphs) and prompt format (state-only, criteria-only), separating a position prior from a label prior |

`numbers.tex` is a superset of what the paper cites: every statistic is exported so that
`MANIFEST.md` can trace all of them, whether or not the prose quotes each one.

## Provenance

All calls used the alias `jev-latest`, which resolved to `jev-1.13.0` throughout, against
`https://api.typesafe.ai/v1/systemone`, between 2026-09-19 and 2026-09-20. Runs made after
usage capture was added write a `raw/manifest_*.json` recording endpoint, model version,
start and finish times, call count, and token totals.

## Corrections made during internal review

An earlier version of this analysis was audited adversarially, and three results did not
survive. They are corrected here and, where the corrected result is negative, reported as
such in the paper:

1. **Calibration was measured against the wrong quantity.** The `confidence` field is an
   affine rescaling of the top-1 probability; corrected pooled ECE is 0.034, not 0.091.
2. **"Errors concentrate on unstable items" is descriptive, not causal.** Consistency and
   correctness are both defined relative to the first rotation, which couples them. A
   position-invariant null matched to the same accuracy and consistency produces a
   *larger* error ratio (5.43) than observed (2.93).
3. **The positional χ² test was invalid.** Comparing predicted against correct labels loses
   power as accuracy rises, exactly where the question mattered. Replaced by a simulated
   null with an explicit power calculation.

Two further defects were fixed in the pipeline: AUPRC was computed with inverted score
polarity, and the tie-breaking expectation did not condition on the first option being in
the tied set (correct expectation 23.0, not 35.5, which removes the apparent effect).

## Known limitations of this package

- The model is proprietary and closed-weight; responses are not guaranteed reproducible.
- `perm_*.json` and `repeat_*.json` cover the four-option WMDP suites only, so instability
  is a lower bound.
- Token usage was not captured during the original runs; per-dataset rates were measured
  afterwards on the same prompts (`raw/tokens_*.json`). The earlier payload-length
  estimator understated true input tokens by 2-8x, so the cost figure was revised upward.
- The label and format ablations were run on WMDP-Bio and WMDP-Cyber only
  (`raw/*_lblshuf|lblsym|fmtstate|fmtcrit.jsonl`), so their conclusions do not
  automatically extend to the LAB-Bench subtasks.
