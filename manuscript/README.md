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

## Build guards

Two scripts run after every build; either can fail it.

`check_build.py` fails if:

1. `main.tex` uses a `\stat*` macro that `numbers.tex` does not define;
2. any `\ref` or `\cite` is unresolved (checked in both the log and the rendered text);
3. the page size is not US Letter (612×792 pt) — BioAISS enforces this strictly;
4. more than 6 pages count toward the limit. The LLM Usage Statement is excluded, but
   only if it starts its own page; if it shares a page with references or body text, that
   page counts. The current build is 6 pages in total, so the exclusion is not relied on.

`check_parity.py` goes further, and fails if:

1. `numbers.tex` has drifted from what `analyze.py` currently produces;
2. any macro the text cites does not appear, with that value, in the **rendered** PDF text;
3. a cross-check between prose claims and `stats.json` fails (for example, the paper
   asserting a significant cascade advantage on WMDP-Bio, where the data does not support
   one);
4. a banned phrase appears — claims the study explicitly does not make, such as describing
   WMDP accuracy as screening performance.

`numbers.tex` is generated — never edit it. Change the analysis and re-run
`make evidence`.

## Submission checklist

- [x] `\documentclass[conference,compsoc]{IEEEtran}`, IEEEtran.cls v1.8b
- [x] US Letter, double column
- [x] Non-blinded, with affiliations and a corresponding author
- [x] 6 pages in total, including references and the LLM Usage Statement
- [x] LLM Usage Statement present and clearly marked
- [x] All 19 references verified against arXiv / proceedings
- [ ] Submit via the ACSAC HotCRP instance

`IEEEtran.cls` is committed at v1.8b because BioAISS mandates that exact
version; it is not taken from the tectonic bundle.

## Conformance to the IEEE template

Checked against `IEEE-conference-template-062824.tex` and `IEEEtran_HOWTO.pdf`.

`IEEEtran.cls` here is **byte-identical** to the one shipped with the template (v1.8b),
as the CFP requires.

Matches the template:

- no math, symbols, or footnotes in the title or abstract (the template flags this
  as CRITICAL);
- figure labels set in 8 pt Times New Roman;
- axis labels written as words with units in parentheses, never units alone
  ("Magnetization (kA/m)", not "kA/m");
- `Fig.` used for figure references, including at the start of a sentence;
- figure captions below the figure, table caption above the table;
- "Acknowledgment" spelled without the "e";
- references give every author for fewer than six, and `et al.` from six upward;
- no template guidance text remains.

Deliberate deviations, each with a reason:

| Deviation | Reason |
|---|---|
| `[conference,compsoc]` rather than the template's `[conference]` | The CFP states this option explicitly; it overrides the generic template. Changes section numbering to arabic and table captions to "TABLE 1". |
| `\usepackage[T1]{fontenc}` added | Not in the template, which assumes pdfLaTeX. Under XeTeX (tectonic) the default Unicode encoding leaves `ptm` with no shapes, so the body silently falls back to Latin Modern and `\textbf` degrades to regular weight. T1 restores the template's intended Times with working bold. Harmless under pdfLaTeX. |
| `booktabs`, `url` added | Table rules and the code-availability link. |
| `algorithmic`, `xcolor` dropped | Unused. |
| "Index Terms:" rather than "Index Terms---" | Requested; the separator is hardcoded in the class, so the keywords environment is redefined in the preamble. |
