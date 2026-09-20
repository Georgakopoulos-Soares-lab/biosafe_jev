# Auditing System-1 Models on Biosecurity-Relevant Benchmarks

Reproduction material for *Auditing System-1 Models on Biosecurity-Relevant Benchmarks:
Calibration, Selective Prediction, and Permutation Instability in a Non-Generative Model*
(BioAISS 2026, ACSAC workshop; short technical paper).

The paper is a reliability audit of one commercial non-generative "System-1" model,
TypeSafe **Jev**, on 6,020 multiple-choice items drawn from WMDP, a paraphrase-robust
WMDP-Bio variant, and six LAB-Bench subtasks. It measures accuracy, calibration, error
detection, selective prediction, and sensitivity to the order in which answer options are
presented. It does **not** evaluate a biosafety classifier, an information-hazard detector,
or any deployed screening system.

**Paper:** [`manuscript/main.pdf`](manuscript/main.pdf) — 6 pages, US Letter, IEEE
conference format.

## Layout

```
manuscript/   LaTeX source, figures, bibliography, and the build guards
evidence/     raw decision records, analysis code, derived statistics
data/         benchmark source parquet (gitignored; only needed to re-collect responses)
```

## Reproducing the analysis

No API key and no network access are required. The raw decision records are included.

```sh
cd manuscript
make evidence   # re-run the analysis chain, regenerate figures, tables, and numbers.tex
make            # build main.pdf and run the build guards
```

`make evidence` requires `numpy`, `scipy`, `pandas`, `matplotlib`, and `scikit-learn`.
`make` requires [`tectonic`](https://tectonic-typesetting.github.io/) (`pdflatex` is not
used). All bootstraps are seeded, so `analyze.py` is deterministic: two runs produce a
byte-identical `stats.json`.

## How the numbers get into the paper

Every statistic in the manuscript is computed by `evidence/code/analyze.py`, written to
`evidence/derived/stats.json`, and exported as a LaTeX macro into
`manuscript/numbers.tex`. The prose contains no hand-typed statistics, and the build fails
if the text cites a macro the pipeline did not emit. `evidence/MANIFEST.md` maps each of
the 128 macros back to the producing function and the raw file it derives from.

The build also fails on an unresolved `\ref` or `\cite`, a page size other than US Letter,
or a page count above the 6-page limit — see `manuscript/check_build.py`.

## Re-collecting model responses

Requires `TYPESAFE_API_KEY` and the benchmark parquet files in `data/`.

```sh
cd evidence/code
python3 run_eval.py bio chem cyber bio-robust labbench-seq ...   # single pass per item
python3 run_circular.py cyber bio                                # all cyclic rotations
```

`run_eval.py` also provides two ablation arms for confounds the study leaves open. Neither
was run, and no reported result depends on them:

- `--label-mode {positional,shuffled,symbolic}` — the default assigns answer labels in
  positional order, so label identity and position are collinear and a first-option
  preference cannot be separated from a first-label preference.
- `--format-mode {both,state,criteria}` — the default serialises each option twice, in the
  state text and as a category key.

## Scope and limitations

WMDP measures knowledge in hazardous domains; it does not test whether a model can judge
that a request is unsafe. Accuracy on it is not screening performance. One proprietary
model is evaluated, so the results do not generalise to System-1 models as a class. Latency
was not measured. The full limitations are in Section V of the paper.

## Citation

Provatas, K. A. and Georgakopoulos-Soares, I. *Auditing System-1 Models on
Biosecurity-Relevant Benchmarks: Calibration, Selective Prediction, and Permutation
Instability in a Non-Generative Model.* BioAISS 2026 (ACSAC workshop).
