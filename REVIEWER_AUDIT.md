# Reviewer-Grade Audit of `paper.tex` (pass 4: causal walk-forward rerun)

**Manuscript:** *Distributionally Robust and Regime-Aware Deep Hedging Under Market Frictions* (Pratham Kailasiya, IIT Roorkee).

**Audit pass.** This is the fourth review pass. Pass 1 reframed the
manuscript without rerunning experiments. Pass 2 ran the SPY/NIFTY
full-model stress battery rerun and a single-seed TC-inclusive
verification. Pass 3 added a multi-seed SPY calm robustness check and
a single-seed walk-forward backtest on raw `yfinance` OHLCV. Pass 4
(this document) (a) **found and fixed a one-step look-ahead leak** in
the walk-forward feature pipeline, (b) reran the walk-forward
backtest at **three training seeds** on the corrected pipeline, and
(c) rewrote the manuscript's walk-forward narrative to match. The
headline changes materially: the previous "RSE wins four of five
cells with double-digit margins" claim does not survive the
correction; what survives — and is scientifically cleaner — is a
**friction-conditional** result that independently confirms the
paper's turnover–CVaR frontier analysis out of sample.

---

## 1. Look-ahead leak found and fixed (this pass)

**Location.** `walk_forward_backtest.py::make_windows`. The
realised-vol feature at decision time `t` was computed from
`log_returns[:, max(0, t-19):t+1]`; column `t` of `log_returns` is
`log(S_{t+1}/S_t)` — the return over `[t, t+1]`, unknown when the
hedge at `t` is chosen. The leak propagated into the BS-delta
feature through `d1`. Both training and test windows were affected,
in every cell of the pass-3 walk-forward table.

**Evidence.** A perturbation test (shock all closes strictly after
time `t` by +10%, assert features at decision times ≤ `t`
unchanged) showed feature diffs up to **0.76** pre-fix and exactly
0 post-fix. The test is now permanent:
`tests/test_walkforward_causality.py` (2 passed).

**Fix.** Realised vol at `t ≥ 1` now uses
`log_returns[:, max(0, t-20):t]` (strictly trailing 20-return
window); at `t ∈ {0, 1}` the estimate degenerates to 0 and is
lifted to the clip floor 0.05, matching the previous `t = 0`
convention. The synthetic pipeline
(`src/env/market_env.py::get_features`) was audited and is causal
(features at step `k` use only `S_k`, TTM, log-moneyness, `v_k`),
so Tables 1–6 are unaffected. The silent GBM fallback in
`fetch_close` was replaced with a hard error (silent simulation
substitution in a results-generating script is an integrity
hazard), and fetched series are cached per (ticker, window).

**Materiality.** The leak was worth real money to every model:
corrected LSTM CVaR on SPY COVID rose from 30.0 to 43.8 (mean over
seeds). Removing it also *reversed* the NIFTY ordering — the
leaked next-day vol spike pushed calm-trained policies far out of
their feature distribution during crises, with model-dependent
damage that the 18 bps NIFTY friction amplified.

## 2. Corrected 3-seed walk-forward result

`walk_forward_backtest.py --seeds 42,142,242`, output
`walk_forward_multiseed_results.json` (the pass-3 single-seed
`walk_forward_backtest_results.json` is retained for provenance but
is superseded; its numbers were produced by the leaky pipeline).

| Cell | $n$ | LSTM | 3SCH | RSE | RSE $\Delta\%$ | Per-seed winner |
|---|---:|---:|---:|---:|---:|---|
| SPY COVID 2020 | 30 | 43.8 ± 6.9 | 43.4 ± 5.0 | **31.8 ± 3.9** | **−27.3** | RSE 3/3 |
| SPY post-COVID 2021 | 220 | 12.05 ± 0.45 | **11.74 ± 0.31** | 11.85 ± 0.73 | −1.6 | mixed |
| SPY GFC 2007–09 | 472 | 12.67 ± 1.37 | 12.53 ± 1.19 | **11.43 ± 0.65** | **−9.7** | RSE 3/3 |
| NIFTY COVID 2020 | 26 | **1223 ± 96** | 1269 ± 76 | 1267 ± 53 | +3.6 | LSTM 2/3 |
| NIFTY GFC 2007–09 | 402 | **672 ± 44** | 686 ± 3 | 715 ± 50 | +6.3 | LSTM 2/3 |

Per-seed RSE-vs-LSTM margins: SPY COVID −24.5/−21.1/−35.0%;
SPY GFC −3.6/−8.8/−15.6% (seed-uniform wins in both). NIFTY COVID
+10.3/+7.2/−5.6%; NIFTY GFC +3.4/+16.7/−0.4%.

**Mechanism.** Turnover: on NIFTY, RSE ≈ 1.0–1.1 units/window vs
LSTM ≈ 0.67–0.74 (≈1.5×); at 18 bps that drag erases the gating
advantage. On SPY (3 bps) turnovers are comparable (1.06 vs 1.07 on
COVID) and the crisis gains survive. This is an out-of-sample
confirmation of the manuscript's own 10–15 bps break-even friction
claim (§frontier).

**Honesty note.** Pass 3's argument that "in-distribution CV
(0.04–0.28%) is too small to explain the gaps" was unsound:
out-of-distribution cross-seed CV reaches 16% (SPY COVID LSTM),
two orders of magnitude larger. The multi-seed rerun was necessary,
not optional; the defensible robustness statement is the
seed-uniform pairing, which the SPY crisis cells pass and the NIFTY
cells do not.

## 3. Manuscript changes propagated (pass 4)

- Abstract, §1 RSE contribution bullet, §5.5 protocol + Table 7 +
  reading paragraph, §6 deployment recipe, §6.4 limitations,
  §7 conclusion, reproducibility paragraph: "four of five cells,
  double-digit margins" replaced everywhere by the seed-uniform SPY
  crisis result (−27% COVID, −10% GFC) plus the NIFTY reversal and
  the friction-conditional reading. RSE's "recommended default"
  status is now explicitly scoped to low-friction books.
- §5.5 protocol now states the strictly-trailing feature
  construction; Table 7 reports cross-seed mean ± std; caveats on
  small COVID cells (n = 26–30) and stride-1 serial dependence
  added.
- Fixed stale claims that a walk-forward backtest is "absent"
  (§experiments enumeration, fig:spy_crisis caption) — leftovers
  from pass 2.
- Fixed misplaced bolds: `tab:spy` lower block post-COVID (LSTM
  14.49 beats 3SCH 14.50) and `tab:realmarket_full` SPY calm (RSE
  11.03 beats LSTM 11.06); win counts in captions/discussion now
  consistent ("four of seven", was "three of four SPY").
- `eq:pnl` corrected to the convention the code implements: TC sum
  over k = 0..N−1 with δ₋₁ = 0 (initial purchase charged, terminal
  liquidation excluded identically for all strategies); prior text
  claimed a δ_N = 0 unwind term that neither the training losses
  nor any evaluator charges.

## 4. Verification status

| Item | Source | Status |
|---|---|---|
| In-distribution 10-seed Table 1 | `new_approaches/results/audit_summary.json` | Verified (RSE 3.109 ± 0.009, −3.29%). |
| Paired tests / bootstrap | `new_approaches/results/statistical_analysis.json` | Verified (t = −26.25, p = 8.2e−10, d = −7.41). |
| TC-inclusive 10-seed rerun | `full_tc_validation_results.json` | Verified (LSTM 3.413 ± 0.02, RSE 3.326 ± 0.01). |
| Full-model SPY/NIFTY battery (7 cells) | `spy_nifty_full_validation_results.json` | Verified cell-by-cell. |
| Multi-seed SPY calm | `spy_multiseed_results.json` | Verified. |
| Walk-forward OHLCV, causal, 3-seed (5 cells) | `walk_forward_multiseed_results.json` | Verified; produced this pass. |
| Feature causality | `tests/test_walkforward_causality.py` | 2 passed. |

## 5. Remaining weaknesses (after this pass)

| Item | Recommendation |
|---|---|
| COVID walk-forward cells have n = 26–30 overlapping windows; CVaR₉₅ rests on 1–2 tail windows per seed. | Disclosed in §5.5 and limitations. A block-bootstrap CI over windows, or monthly-stride disjoint windows, would tighten this; margins (−27%) exceed the noise but the small-n caveat must stay. |
| NIFTY per-seed margins are sign-mixed (LSTM wins 2/3). | Correctly reported as mean-level, not seed-uniform. More seeds would sharpen but not likely reverse the friction story. |
| W-DRO-T and Transformer not in walk-forward battery. | Unchanged from pass 3; W-DRO-T's case is theoretical (dual gradient-norm penalty) and its full-model empirical parity is disclosed. |
| Medium-suite seed count n = 2. | Acknowledged; results downgraded to "directional". |
| Reproducing Table 1 exactly requires r = 0 while the working tree defaults to r = 0.05. | Paper discloses both ("r = 0 at the time of these runs"); consider a `--r` flag in `run_full_experiments.py` for exact-reproduction convenience. |

## 6. Publication-readiness assessment (revised)

| Venue | Verdict |
|---|---|
| NeurIPS / ICML / ICAIF | **Accept-leaning.** The corrected walk-forward result is *weaker as marketing but stronger as science*: causal features, multi-seed, a mechanism (turnover × friction) that unifies the walk-forward, the frontier, and the deployment recipe into one coherent story, and limitations that are stated rather than discovered by reviewers. The seed-uniform SPY crisis win (−27%/−10%) plus the ten-seed in-distribution result remain a solid empirical core. |
| *Journal of Risk* | **Accept with minor revision.** The friction-conditional recipe is exactly the actionable artefact this audience wants. |
| Mathematics of OR | Still **major revision** for the Wasserstein theory chapter. |

## 7. Files written or modified in this pass

- `walk_forward_backtest.py` — causality fix in `make_windows`;
  `--seeds` argument; per-seed result schema; hard-error fetch with
  caching; docstrings corrected (overlapping windows, first-close
  strike).
- `walk_forward_multiseed_results.json` — new 3-seed corrected
  results (45 cells).
- `tests/test_walkforward_causality.py` — new regression tests
  (perturbation-based causality + shape checks).
- `paper/paper.tex` — all edits listed in §3.
- `REVIEWER_AUDIT.md` (this file).
