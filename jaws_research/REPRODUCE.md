# Reproducibility Checklist - JAWS Research Pipeline

This pipeline (`jaws_research/`) extends the existing repository's deep-hedging
work with: **online market data** + **crisis regimes**, **multi-asset Bates
jump-diffusion**, **exotic option payoffs**, and a **fair, reproducible
benchmark** of all five canonical hedgers.

The source `combined_underscore_report.tex` and the existing models live
unchanged in the repo; this directory adds a self-contained pipeline that
can be re-run end-to-end.

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# core deps: torch>=2.0, numpy, scipy, pandas, matplotlib, yfinance
```

GPU optional (CUDA / Apple MPS / CPU all supported; MPS is auto-detected on
macOS Apple Silicon).  The pipeline uses 0.5-3 GB RAM per run.

## 2. Layout

```
jaws_research/
├── data/
│   ├── simulators.py     # Heston (1-asset) + Bates (d-asset) + crisis presets
│   └── online_data.py    # yfinance fetch with cache + crisis windows
├── tasks/
│   ├── payoffs.py        # vanilla, barrier, digital, Asian, basket
│   └── features.py       # 5-channel features + 6-channel regime features
├── models/
│   ├── architectures.py  # LSTM, Transformer, Signature, RSE
│   └── losses.py         # CVaR, entropic, mixed, trading penalty
├── train/
│   └── trainer.py        # BaseTrainer / WDROTrainer / ThreeStageTrainer / RSETrainer
├── eval/
│   └── metrics.py        # bootstrap CI, paired test, Holm-Bonferroni
├── experiments/
│   ├── run_benchmark.py  # synthetic benchmark across regimes/tasks
│   ├── run_real_data.py  # real-market crisis-window evaluation
│   ├── analyze_results.py
│   └── figures.py
├── deliverables/
│   ├── thesis.tex        # 40-50 pp thesis
│   ├── paper.tex         # conference-grade paper (W-DRO-T as headline)
│   ├── ppt.tex           # Beamer slides
│   ├── references.bib
│   ├── auto_results_block.tex
│   ├── auto_real_block.tex
│   └── build_tex_blocks.py
└── outputs/
    ├── cache/            # yfinance Parquet cache
    ├── runs/             # benchmark_*.pkl + real_*.pkl
    ├── logs/             # human-readable run logs
    ├── tables/           # CSV summaries + LaTeX tables
    └── figures/          # PDF + PNG
```

## 3. Reproduction Commands

```bash
# (a) quick smoke test (~5 min on MPS)
python -m jaws_research.experiments.run_benchmark --mode quick --label quick_v1

# (b) medium-scale benchmark (3-5 seeds, full task suite, ~1.5 h on MPS)
python -m jaws_research.experiments.run_benchmark --mode medium --label medium_v1

# (c) full benchmark (5+ seeds, 6+ regimes, ~3-5 h)
python -m jaws_research.experiments.run_benchmark --mode full --label full_v1

# (d) real-market crisis-window OOS (~30 min)
python -m jaws_research.experiments.run_real_data --label real_v1

# (e) tables + figures
python -m jaws_research.experiments.analyze_results --label medium_v1
python -m jaws_research.experiments.figures        --label medium_v1

# (f) refresh the LaTeX auto-blocks
python -m jaws_research.deliverables.build_tex_blocks
```

## 4. Run-tracking artefacts

Every run produces:

- `outputs/runs/benchmark_<label>.pkl`: dict with `config`, `results` (per
  scenario+task: per-model per-seed metrics, raw P&L vectors, deltas),
  `elapsed_sec`, `device`.
- `outputs/logs/benchmark_<label>.log`: human-readable progress log.
- After analysis: `outputs/tables/raw_metrics_<label>.csv`,
  `outputs/tables/agg_*.csv`, `outputs/tables/paired_LSTM_*.csv`,
  `outputs/tables/winners_*.csv`, plus a LaTeX table for the European-call
  pivot.
- `outputs/figures/*.pdf` + `*.png` for each saved plot.

## 5. Statistical protocol

- 5 seeds in {42, 142, 242, 342, 442}
- 2,000 bootstrap resamples for 95 % CIs
- Paired t-test against the LSTM baseline (per scenario × task cell)
- Holm-Bonferroni correction at α = 0.05 across models per cell
- Cohen's d effect sizes
- Mean P&L parity check (all models within ≤ 1 % of each other on the same
  task) confirms that comparisons are not biased by drift differences.

## 6. Assumptions logged

| Decision | Default | Justification |
|---|---|---|
| Time grid | 30 daily steps over 30/365 yr | matches Buehler & Kozyra |
| Heston feller | 2κθ ≥ ξ² (warning if violated) | preserves variance positivity |
| Bates jump intensity in crisis | λ = 6 (COVID) / 8 (GFC) | matches empirical jump frequency |
| Cross-asset corr (basket) | 0.55 normal / 0.75 crisis | matches realised SPY-QQQ-IWM correlation |
| Transaction cost US / India | 0.001 / 0.0018 | bid-ask + STT + slippage proxy |
| δ_max | 1.5 | identical to existing repo |
| ε_max for W-DRO-T | 0.1 | matches W-DRO-T paper |
| 3SCH α schedule | 0.8 → 0.2 | matches existing repo |
| Number of regimes K | 4 | matches existing repo (also best in ablation) |
| Synthetic-fallback feature set | EWMA realised variance | when no `v` available from market data |
| Real-data resolution | daily OHLCV (yfinance) | free, reproducible, no licence requirement |

## 7. Honesty and limitations

- Daily-resolution real data masks intraday microstructure.
- We do not model order-book impact; transaction costs are proportional
  proxies.
- The synthetic-fallback path generator is calibrated to historical
  volatility levels but does not reproduce the full empirical correlation
  structure.
- Numbers reported in the LaTeX deliverables are from `quick_v1` until
  `medium_v1` lands; after `build_tex_blocks` runs, the tables refresh
  automatically.

## 8. Re-using the existing repository

The pipeline is additive. The existing
`/new_approaches/code/{w_dro_t.py, three_stage.py, rse.py, ...}` and `/src/`
packages remain unchanged.  Our `jaws_research/models/architectures.py`
re-implements the five canonical hedgers as compact, comparable classes so
that the benchmark can use a single training driver and a single feature
schema across all of them; the original implementations remain available
for those who want to use them in production.
