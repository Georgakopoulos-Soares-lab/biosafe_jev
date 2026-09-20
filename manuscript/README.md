# Manuscript

*Auditing System-1 Models on Biosecurity-Relevant Benchmarks: Calibration, Selective
Prediction, and Permutation Instability in a Non-Generative Model* — BioAISS 2026 (ACSAC
workshop), **short technical paper**.

## Build

```sh
make            # tectonic -> main.pdf, then runs the build guards
make evidence   # re-run the full analysis chain, then rebuild everything
```

`tectonic` is required (`pdflatex` is not used). `make evidence` additionally needs
`numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`.

## Build guards (`check_build.py`)

The build fails, rather than silently producing a bad PDF, if:

1. `main.tex` uses a `\stat*` macro that `numbers.tex` does not define;
2. any `\ref` or `\cite` is unresolved (checked in both the log and the rendered text);
3. the page size is not US Letter (612×792 pt) — BioAISS enforces this strictly;
4. more than 6 pages count toward the limit. The LLM Usage Statement is excluded, but
   only if it starts its own page; if it shares a page with references or body text, that
   page counts. The current build is 6 pages in total, so the exclusion is not relied on.

`numbers.tex` is generated — never edit it. Change the analysis and re-run
`make evidence`.

## Submission checklist

- [x] `\documentclass[conference,compsoc]{IEEEtran}`, IEEEtran.cls v1.8b
- [x] US Letter, double column
- [x] Non-blinded, with affiliations and a corresponding author
- [x] 6 pages in total, including references and the LLM Usage Statement
- [x] LLM Usage Statement present and clearly marked
- [x] All 18 references verified against arXiv / proceedings
- [ ] Submit via the ACSAC HotCRP instance

`IEEEtran.cls` is committed at v1.8b because BioAISS mandates that exact
version; it is not taken from the tectonic bundle.
