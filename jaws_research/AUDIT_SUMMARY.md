# Repository audit summary

## What the repo already contains
- `src/` -- Buehler-style deep-hedging implementation, baselines, transformer, signature, RL agents.
- `new_approaches/code/` -- W-DRO-T, 3SCH, RSE, RVSN, SAC-CVaR (varying maturity).
- `experiments/`, `new_experiments/`, `final_audit_experiments/` -- earlier evaluation runs (Heston only).
- `deliverable/cr_part{1..4}.tex`, `combined_paper.tex`, `RSE_paper.tex`, `W-DRO-T_paper.tex`, `3SCH_paper.tex`, `comprehensive_report.tex` -- the existing thesis-grade narrative and per-novelty drafts.
- `deliverable/references.bib`, `kb_index.json`, `manifest.json`, `REPRODUCE.md` -- knowledge-base and reproducibility metadata.
- `figures/`, `poster_figs/`, `reports_png/` -- figure assets used by the existing narrative.

## What was missing for the prompt
- Online (yfinance) data feed with crisis windows
- Multi-asset Bates jump-diffusion simulator
- Exotic option payoffs (barrier, digital, Asian, basket)
- Cross-asset / portfolio hedging task
- Walk-forward, regime-aware splits
- A single, fair training driver for all five hedgers
- A unified evaluation protocol producing pickled metrics
- A fresh thesis / paper / Beamer deck driven by *measured* numbers

## What this delivery adds
- `jaws_research/data/{simulators.py, online_data.py}` -- multi-asset Bates + Heston, online data + crisis presets, walk-forward splits.
- `jaws_research/tasks/{payoffs.py, features.py}` -- 7 payoffs, multi-asset support, regime features.
- `jaws_research/models/{architectures.py, losses.py}` -- compact, comparable LSTM/Transformer/Signature/RSE plus CVaR/entropic/mixed/trading-penalty losses.
- `jaws_research/train/trainer.py` -- one driver covering plain, W-DRO-T, 3SCH and RSE protocols.
- `jaws_research/eval/metrics.py` -- bootstrap CIs, paired tests, Holm-Bonferroni.
- `jaws_research/experiments/{run_benchmark.py, run_real_data.py, analyze_results.py, figures.py}` -- end-to-end pipeline.
- `jaws_research/deliverables/{thesis.tex, paper.tex, ppt.tex, references.bib, build_tex_blocks.py}` -- LaTeX outputs that consume the saved pickles.

## How the existing W-DRO-T / 3SCH / RSE relate to our re-implementation
The original implementations live unchanged. Our reimplementations are
deliberately compact (single file each) so that they can be benchmarked
under a single training driver and a single feature schema; the design
choices, loss functions, and curricula match the originals.

## Knowledge base ingested
- `combined_underscore_report.tex` (combined paper.tex)
- `cr_part1..4.tex`, `comprehensive_report.tex`, `poster_report.tex`,
  `RSE_paper.tex`, `W-DRO-T_paper.tex`, `3SCH_paper.tex`,
  `combined_paper.tex`, `novelty_method.tex` (deprecated -- removed from
  scope per user instruction)
- README.md, REPRODUCE.md, AUDIT_SUMMARY.md, FINAL_REPORT.md,
  RESEARCH_FINDINGS.md, RESEARCH_REPORT.md
- `kb_index.json`, `manifest.json`

## Assumptions logged
- Daily resolution data; intraday microstructure not modelled.
- Proportional transaction-cost proxy (US 10 bps round-trip / India 18 bps).
- Synthetic-fallback for offline operation when yfinance is blocked.
- Cross-asset Brownian correlation = 0.55 (calm) / 0.75 (crisis).
- Bates jump intensity = 1.5 calm / 6.0 COVID / 8.0 GFC per year.
- δ_max = 1.5 throughout (matches existing repo).

## What the JAWS-T name no longer refers to
The original repository contained a `novelty_method.tex` proposing a
multi-novelty hedger called JAWS-T.  Per the most recent user instruction
this is scrapped from this delivery.  All deliverables in
`jaws_research/deliverables/` focus on the five canonical hedgers
(LSTM, Transformer, W-DRO-T, 3SCH, RSE) and the extension to crisis
regimes / multi-asset / exotic payoffs.
