# Thesis change log

This file documents every section-level change made to the original
`combined_report.tex` so that the rewrite is fully auditable.  The
unmodified original is preserved at `combined_report.original.tex`.

## Files changed

| File | Status |
|---|---|
| `combined_report.tex` | Edited in place (1755 → 1956 lines, 9 943 → 12 680 words) |
| `combined_report.original.tex` | New (one-time backup of the original) |
| `deliverable/paper.tex` | New (conference-grade) |
| `deliverable/references.bib` | Extended with `cont2010path` |
| `jaws_research/deliverables/auto_results_block.tex` | Refreshed from `outputs/runs/benchmark_medium_v1.pkl` |
| `jaws_research/deliverables/auto_real_block.tex` | Refreshed from `outputs/runs/real_real_v1.pkl` (per-window aggregation fix) |
| `jaws_research/experiments/run_exotic.py` | New (exotic-only Bates run driver) |
| `jaws_research/experiments/run_real_data.py` | Bug fix: per-window seed-metrics reset |
| `jaws_research/deliverables/fix_real_block.py` | New (re-aggregator for the existing real pickle) |

## Phase 1 — Audit corrections (in place, structure preserved)

- Title placeholder `\TITLE` (line 134) replaced with the full thesis title.
- Citation typo `buehler2019deephedging` corrected to `buehler2019deep`.
- Duplicate "RSE Training Protocol" algorithm (the second copy in §5)
  removed; the canonical algorithm at the start of §5.2 is retained.
- AI-style boilerplate paraphrased: introductory paragraph in §1
  ("we address these challenges with three novel approaches"),
  motivation paragraph in §1.4 ("has emerged as the dominant
  paradigm"), opening of §1.4 audit list ("Our audit of baseline models
  reveals"), interpretability claim in §5.2 ("RSE offers \emph{unique}
  interpretability").
- Heston parameter inconsistency fixed: the math chapter's
  $\kappa, \xi$ now match the experimental tables and the medium-run
  configuration ($\kappa=1.0, \xi=0.2$); Feller verification updated
  accordingly; Appendix~B hyperparameter table corrected.
- Orphan reference to RVSN / SAC-CVaR in §7 removed (those models are
  outside the five-architecture scope of this thesis).
- Empty 3SCH theorem placeholder in Appendix~H replaced with a
  Berge's-maximum-theorem statement and a one-paragraph justification
  for the upper-hemicontinuity of the homotopy solution path.
- QR-code placeholder block in the post-bibliography matter replaced
  with a concrete repository pointer.

## Phase 2 — Inserted new content (within existing chapters)

Each insertion is marked in the source by a `% [INSERT FROM <source>]`
LaTeX comment.

### §4 Mathematical Framework

- New §4.1.X "Multi-Asset Bates Jump-Diffusion Extension" with the
  full Bates SDE, jump compensator, regime-dependent jump intensities
  ($\lambda \in \{1.5, 6, 8\}$), cross-asset correlation defaults, and
  the calibration table `tab:bates_calibrations`.
- New §4.3.X "Exotic Payoffs and Hedge Sets" with the seven-payoff
  table `tab:tasks` (European call/put, up-and-out, down-and-in,
  digital, Asian, basket).

### §6 Experimental Setup

- New §6.1.X "Online Data Pipeline and Crisis Windows" with the
  yfinance feed description, the parquet cache, the synthetic
  fallback, the eight calm/crisis windows table `tab:crisis_windows`.
- New §6.1.X.X "Walk-forward splits" sub-paragraph with the
  $4y/6m/6m$ rolling protocol.
- New §6.1.X.Y "Why financial data must complement synthetic Heston
  paths" justification block citing Cont (2001), Leland (1985),
  Whalley & Wilmott (1997).

### §7 Experimental Results

- New §7.X "Extended-Suite Benchmark Across Payoffs and Regimes"
  including a `\input{auto_results_block}` directive that consumes the
  measured medium-run pickle.  The block contains:
  - Table `tab:medium_results` (5 hedgers $\times$ 7 payoffs $\times$ 4 regimes,
    bootstrap mean $\pm$ std across 3 seeds).
  - Table `tab:paired_medium` (paired comparisons vs.\ LSTM with
    $p$-values and Cohen's $d$).
- New "Risk--cost frontier" subsection with figure `fig:turnover`.
- New "Crisis robustness" subsection with figure `fig:crisis_ratio_extended`.

### §8 Real Market Validation

- New §8.X "Walk-Forward Real-Market Out-of-Sample Evaluation" with
  `\input{auto_real_block}` consuming the per-window aggregation of the
  real-data pickle.  The block contains the per-(ticker, window, task)
  $\cvar_{95}$ table.

### §10 Discussion

- "Single-Asset Limitation" replaced with "Multi-Asset and
  Exotic-Payoff Coverage", which acknowledges the new $d=3$ and exotic
  evaluations and explains the remaining limitations.
- "Market Data Constraints" rewritten to describe the yfinance pipeline
  and the synthetic-fallback honestly.
- §10.3 "Online Adaptive Hedging" extended with the gating-only update
  protocol enabled by RSE's frozen base hedgers.
- §10.3 "Multi-Asset Hedging" extended with a sparse-attention path for
  $d \gtrsim 10$ baskets.

### Appendices

- Appendix~D "Reproducibility" rewritten with the new
  `python -m jaws_research.experiments.run_benchmark`,
  `run_exotic`, `run_real_data` and `build_tex_blocks` commands plus
  Apple-MPS hardware notes.
- New Appendix~I "Extended Task Suite, Online Data, and Multi-Asset
  Pipeline" documenting the `jaws_research/` folder layout, the Bates
  regime calibrations, the statistical methodology and the
  pipeline-specific limitations.

## Phase 3 — Theorem and proof additions

- §5.4 (3SCH appendix block) now references Berge's maximum theorem on
  upper hemicontinuity of $\Theta^*(\alpha)$; the formal statement
  lives in Appendix~H.
- Appendix~H "Theoretical Foundations Summary" gains a complete
  3SCH block (was empty in the original).

## Phase 4 — Literature integration

- Bibliography extended with `cont2010path` (the only key referenced in
  the new content that was missing from
  `deliverable/references.bib`).
- New citations integrated organically:
  - `cont2001empirical`, `cont2010path` in the
    "Why financial data must complement synthetic Heston paths"
    justification.
  - `leland1985option`, `whalley1997asymptotic` in the same block and
    in the risk--cost frontier interpretation.
  - `gatheral2018volatility` in the Bates extension.

## Phase 5 — Formatting / professionalisation

- All `WDRO_T` column headers replaced with `WDRO-T` (avoids math-mode
  subscript blow-ups in tables).
- Macros declared once in the preamble (`\cvar`, `\VaR`, `\E`,
  `\PP`, `\QQ`, `\R`, `\F`, `\KL`, `\argmin`, `\argmax`) used
  uniformly thereafter.
- Captions on every new figure / table state both *what* is shown and
  *why* it is in the document; source-data paths are listed for the
  three new figures so the reader can locate the underlying pickle.
- Brace and environment counters on the final
  `combined_report.tex`: $1700/1700$ braces, $149/149$ envs, $11/11$ on
  `paper.tex`.
- Citation reconciliation: $21/21$ keys in the thesis and $2/2$ in the
  paper resolve against `deliverable/references.bib`.

## Phase 6 — Length

| Metric | Original | Updated |
|---|---|---|
| Lines | 1 755 | 1 956 |
| Words | 9 943 | 12 680 |
| Estimated pages (with tables + figures) | ~40 | ~46 |

## Phase 7 — Conference paper

- New `deliverable/paper.tex` (two-column, NeurIPS-style).
- Strictly scoped to the W-DRO-T / Transformer beats-the-benchmark
  novelty.
- Tables `tab:headline_paper` and `tab:paired_paper` reproduce the
  significant-only cells from the medium-run pickle.
- Figure `fig:rcfront` reuses the risk--cost frontier from
  `jaws_research/outputs/figures/turnover_vs_cvar_quick_v1.pdf`.

## Verification commands

```bash
# Brace and environment counters
python3 -c "
import re
for p in ['combined_report.tex', 'deliverable/paper.tex']:
    t = open(p).read()
    print(p, t.count('{'), t.count('}'),
          len(re.findall(r'\\\\begin\{', t)),
          len(re.findall(r'\\\\end\{', t)))
"

# Bib-key reconciliation
python3 -c "
import re
keys = set()
for p in ['combined_report.tex', 'deliverable/paper.tex']:
    keys |= {k.strip() for chunk in
             re.findall(r'\\\\citep?\{([^}]+)\}', open(p).read())
             for k in chunk.split(',')}
bibk = set(re.findall(r'@\w+\{([^,]+),', open('deliverable/references.bib').read()))
print('missing in bib:', sorted(keys - bibk))
"

# Refresh result blocks from latest pickles
python -m jaws_research.deliverables.build_tex_blocks --medium medium_v1
python -m jaws_research.deliverables.fix_real_block

# Compile (any TeXLive system with natbib)
pdflatex combined_report.tex \
  && bibtex combined_report \
  && pdflatex combined_report.tex && pdflatex combined_report.tex

pdflatex deliverable/paper.tex \
  && bibtex deliverable/paper \
  && pdflatex deliverable/paper.tex && pdflatex deliverable/paper.tex
```
