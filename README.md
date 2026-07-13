# Distributionally Robust and Regime-Aware Deep Hedging Under Market Frictions

Reproducible code for the pass-5 study of deep hedging under transaction
costs, evaluated in-distribution (Heston), across a regime-calibrated
SPY/NIFTY stress battery, and — most importantly — on a **causal
walk-forward backtest of raw OHLCV** with strictly trailing features.

> The manuscript (`paper/`, LaTeX) is compiled separately and is **not** part
> of this code artefact by design (see `.gitignore` / `PACKAGING.md`).

## What's here

| Model | Idea |
|---|---|
| **LSTM** | recurrent hedger (baseline, Kozyra two-stage) |
| **3SCH** | three-stage CVaR→mixed→entropic curriculum |
| **RSE** | regime-switching ensemble (LSTM+Transformer+Signature, gated by a 4-regime classifier) |
| **W-DRO-T** | Wasserstein-DRO transformer (gradient-norm dual penalty) |
| **research variants** | adaptive (vol-conditioned) Wasserstein radius; friction-aware / rich-feature RSE; **Whalley–Wilmott no-trade-band overlay** |

### Headline findings (pilot-validated; full multi-seed on Colab)
- **RSE** attains the lowest in-distribution CVaR₉₅ (−3.29% vs LSTM, d=−7.41).
- The causal walk-forward is **friction-conditional**: RSE wins the SPY (3 bps)
  crisis cells; the low-turnover baseline wins the NIFTY (18 bps) book.
- **W-DRO-T's DRO penalty is numerically inert** (~0.016% of the loss at ε=0.1),
  so it behaves like an entropic transformer — a mechanistic characterization,
  not a win.
- The **no-trade-band overlay** cuts turnover ~30–50% at iso-CVaR on low-friction
  (US) books; it does not help high-friction crisis cells (the CVaR tail needs
  the trades a mean-variance-optimal band suppresses).

## Install
```bash
pip install -r requirements.txt          # CORE block is sufficient
```
Python 3.10+; PyTorch 2.1+. Device auto-detected (CUDA → MPS → CPU).

## Reproduce
```bash
python sanity_check.py           # fast smoke test (model forwards + causality test)
bash run_all.sh core             # causal walk-forward battery (5 seeds) + bootstrap CIs
bash run_all.sh improved         # no-trade band, adaptive-DRO, RSE frontier, features
bash run_all.sh transfer         # cross-market transfer US<->India
bash run_all.sh indist           # in-distribution 10-seed + TC + regime stress
bash run_all.sh full             # everything
```
Single experiment for debugging: `bash run_single_experiment.sh {walkforward|band|transfer|wdrot|rse}`.
On a GPU use **Google Colab** — see **`COLAB_RUN.md`** (the W-DRO-T
double-backward is ~6× faster on CUDA than Apple MPS).

## Repository structure
```
walk_forward_backtest.py     causal OHLCV walk-forward battery (LSTM/3SCH/RSE/WDROT), per-window PnL
walk_forward_bootstrap.py    paired moving-block-bootstrap CVaR95 CIs (configurable --ref/--models)
wf_band_pilot.py             Whalley–Wilmott no-trade-band sweep;  analyze_band.py summarises it
wf_cross_market.py           cross-market transfer (train US<->India, deploy on crisis)
wf_variants_pilot.py         adaptive-DRO vs vanilla; friction-aware / rich-feature RSE
sanity_check.py              fast smoke test         run_all.sh / run_single_experiment.sh  orchestration
push_results.sh              persist result JSONs to GitHub (PAT via env, no secrets on disk)
experiment_manifest.json     every experiment (script, command, seeds, outputs)   seeds.json  seed registry
src/                         env (Heston/Bates), models (LSTM/transformer), losses (entropic/CVaR)
new_approaches/code/         three_stage, rse, w_dro_t, + pass-5: w_dro_t_adaptive, rse_improved, no_trade_band
new_approaches/experiments/  run_full_experiments.py (in-distribution 10-seed; --r for Table-1 reproduction)
tests/                       test_walkforward_causality.py (feature causality regression, 2 passed)
```

## Outputs
Each script writes a small JSON to the repo root (schema
`state[market][cell][model][seed] = metrics`, with a per-window `pnl`
list for bootstrapping). Logs go to `results/logs/`. These JSONs are the
provenance for the paper's tables and are committed; large pickles,
checkpoints, datasets, figures and the LaTeX are gitignored.

## Hardware
Small: batch 64–256, `d_model` 64. < 2 GB GPU RAM; a free Colab **T4**
suffices. CPU works but W-DRO-T is slow. Runtime budget: see `COLAB_RUN.md`.

## Reproducibility
- Seeds in `seeds.json`; every experiment enumerated in `experiment_manifest.json`.
- Report the cross-seed spread **and** the block-bootstrap CIs — MPS is not
  bit-deterministic at a fixed seed (~15% drift on RSE crisis cells); CUDA is
  tighter. For maximal determinism on CUDA set
  `torch.use_deterministic_algorithms(True, warn_only=True)` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
```
git clone -b research/pass5-improvements https://github.com/prathamkailashya/deep-hedging.git
cd deep-hedging && pip install -r requirements.txt && bash run_all.sh core
```
