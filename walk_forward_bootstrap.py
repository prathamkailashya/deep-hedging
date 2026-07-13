#!/usr/bin/env python3
"""
Block-bootstrap confidence intervals for the causal walk-forward backtest.

Motivation (audit weakness #1). The crisis test cells are short and the
windows overlap at stride 1: SPY/NIFTY COVID have only n = 26-30 windows,
and CVaR_95 there rests on 1-2 tail windows per seed. A single point CVaR
is therefore not enough to judge whether a method's advantage is real.
This script attaches uncertainty to every walk-forward cell.

Method.
  * Input: walk_forward_ci_results.json, which stores the per-window PnL
    vector for every (market, test, model, seed). Because the test loader
    is built with shuffle=False, all models in a cell share the same
    window ordering, so PnL vectors are aligned window-for-window and can
    be paired.
  * Circular moving-block bootstrap (Politis & Romano 1992) over the
    window index. A block length L = max(2, round(n**(1/3))) preserves the
    serial dependence induced by the stride-1 overlap; circular wrapping
    gives every window equal resampling weight. B = 5000 resamples, fixed
    RNG seed for reproducibility.
  * Paired: within one resample the SAME block indices are applied to the
    candidate and to the LSTM reference, so
        dCVaR%% = 100 * (CVaR_cand - CVaR_LSTM) / CVaR_LSTM
    is a genuine paired statistic (identical windows for both models).
  * Draws are pooled across the training seeds (B per seed -> n_seeds*B
    draws) so the reported 95%% interval folds in both window-sampling and
    seed uncertainty. p_improve = P(dCVaR < 0) is the bootstrap
    probability that the candidate lowers CVaR_95 versus LSTM.

Output: walk_forward_bootstrap.json plus a Markdown summary on stdout.
This script does no training and can be re-run at will.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "walk_forward_ci_results.json"
OUT = ROOT / "walk_forward_bootstrap.json"

B = 5000
RNG_SEED = 12345
REF = "LSTM"          # reference model for paired deltas
MODEL_ORDER = ["LSTM", "3SCH", "RSE", "WDROT"]


def cvar95(x: np.ndarray) -> float:
    """CVaR_95 of PnL: negative mean of the worst 5% of outcomes."""
    thr = np.percentile(x, 5)
    tail = x[x <= thr]
    return float(-tail.mean())


def block_length(n: int) -> int:
    return max(2, int(round(n ** (1.0 / 3.0))))


def cbb_index_draws(n: int, L: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap index matrix, shape [B, n].

    Each resample is built from ceil(n / L) blocks of length L, each
    starting at a uniform random position and wrapping around mod n, then
    truncated to length n."""
    n_blocks = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, n_blocks))          # [B, n_blocks]
    offsets = np.arange(L)                                    # [L]
    # [B, n_blocks, L] -> [B, n_blocks*L] -> [B, n]
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    idx = idx.reshape(B, n_blocks * L)[:, :n]
    return idx


def summarize(draws: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(draws)),
        "ci95": [float(np.percentile(draws, 2.5)),
                 float(np.percentile(draws, 97.5))],
    }


def main() -> int:
    global RESULTS, OUT, REF, MODEL_ORDER
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(RESULTS),
                    help="input results JSON (must contain per-window 'pnl')")
    ap.add_argument("--out", default=None, help="output bootstrap JSON")
    ap.add_argument("--ref", default=REF, help="reference model for paired deltas")
    ap.add_argument("--models", default=",".join(MODEL_ORDER),
                    help="comma-separated model order to report")
    args = ap.parse_args()
    RESULTS = Path(args.results)
    OUT = Path(args.out) if args.out else RESULTS.with_name(
        RESULTS.stem.replace("_results", "") + "_bootstrap.json")
    REF = args.ref
    MODEL_ORDER = [m.strip() for m in args.models.split(",") if m.strip()]

    if not RESULTS.exists():
        print(f"missing {RESULTS}; run the backtest first", file=sys.stderr)
        return 1
    state = json.loads(RESULTS.read_text())
    rng = np.random.default_rng(RNG_SEED)

    out = {
        "meta": {
            "source": RESULTS.name,
            "B": B,
            "rng_seed": RNG_SEED,
            "reference_model": REF,
            "block_length_rule": "L = max(2, round(n**(1/3)))",
            "method": "paired circular moving-block bootstrap over windows; "
                      "draws pooled across seeds",
        }
    }
    md = ["# Walk-forward block-bootstrap CVaR_95 (paired vs LSTM)\n"]

    for market, tests in state.items():
        if market == "meta":
            continue
        out[market] = {}
        for test, models in tests.items():
            # per-model per-seed PnL arrays
            pnl = {m: {s: np.asarray(v["pnl"], dtype=float)
                       for s, v in seeds.items() if "pnl" in v}
                   for m, seeds in models.items()}
            present = [m for m in MODEL_ORDER if m in pnl and pnl[m]]
            if REF not in present:
                continue
            seeds_common = sorted(
                set.intersection(*[set(pnl[m].keys()) for m in present]),
                key=int)
            if not seeds_common:
                continue
            n = len(next(iter(pnl[REF].values())))
            L = block_length(n)

            # accumulate pooled bootstrap draws across seeds
            cvar_draws = {m: [] for m in present}
            delta_draws = {m: [] for m in present if m != REF}
            point_cvar = {m: [] for m in present}
            for s in seeds_common:
                # guard: all present models must share this seed & length
                if any(s not in pnl[m] or len(pnl[m][s]) != n
                       for m in present):
                    continue
                idx = cbb_index_draws(n, L, B, rng)           # [B, n]
                ref_boot = np.array([cvar95(pnl[REF][s][row]) for row in idx])
                for m in present:
                    xm = pnl[m][s]
                    m_boot = np.array([cvar95(xm[row]) for row in idx])
                    cvar_draws[m].append(m_boot)
                    point_cvar[m].append(cvar95(xm))
                    if m != REF:
                        # paired: same idx rows for candidate and reference
                        delta_draws[m].append(
                            100.0 * (m_boot - ref_boot) / ref_boot)

            cell = {"n": n, "block_length": L,
                    "seeds": seeds_common, "models": {}}
            for m in present:
                allc = np.concatenate(cvar_draws[m])
                entry = {
                    "cvar_point": float(np.mean(point_cvar[m])),
                    "cvar_ci95": summarize(allc)["ci95"],
                }
                if m != REF:
                    alld = np.concatenate(delta_draws[m])
                    ds = summarize(alld)
                    entry["delta_pct_point"] = float(
                        100.0 * (np.mean(point_cvar[m]) - np.mean(point_cvar[REF]))
                        / np.mean(point_cvar[REF]))
                    entry["delta_pct_ci95"] = ds["ci95"]
                    entry["p_improve"] = float(np.mean(alld < 0.0))
                cell["models"][m] = entry
            out[market][test] = cell

            # markdown
            md.append(f"## {market} / {test}  (n={n}, L={L}, "
                      f"seeds={seeds_common})\n")
            md.append("| model | CVaR95 | CVaR95 95% CI | ΔCVaR% vs LSTM | "
                      "ΔCVaR% 95% CI | P(improve) |")
            md.append("|---|---:|---|---:|---|---:|")
            for m in present:
                e = cell["models"][m]
                lo, hi = e["cvar_ci95"]
                if m == REF:
                    md.append(f"| {m} | {e['cvar_point']:.2f} | "
                              f"[{lo:.2f}, {hi:.2f}] | — | — | — |")
                else:
                    dlo, dhi = e["delta_pct_ci95"]
                    md.append(
                        f"| {m} | {e['cvar_point']:.2f} | [{lo:.2f}, {hi:.2f}] | "
                        f"{e['delta_pct_point']:+.1f}% | "
                        f"[{dlo:+.1f}%, {dhi:+.1f}%] | {e['p_improve']:.2f} |")
            md.append("")

    OUT.write_text(json.dumps(out, indent=2))
    print("\n".join(md))
    print(f"\n[bootstrap] wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
