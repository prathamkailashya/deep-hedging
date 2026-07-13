#!/usr/bin/env python3
"""
Cross-market transfer (reviewer-flagged generalisation experiment).

Train a hedging policy on one market's calm 2017-2019 window and deploy it
- with frozen weights - on ANOTHER market's crisis cells. This is a strict
out-of-distribution test that no in-market backtest provides.

It is well-defined because the five features are scale-invariant
(normalised price S/S0, log-moneyness, realised vol, time-to-maturity,
BS delta), so a SPY-trained policy consumes NIFTY features directly. The
TC-inclusive evaluator uses the *destination* market's friction.

Directions: SPY->NIFTY and NIFTY->SPY, for LSTM / 3SCH / RSE / WDROT, each
also with the Whalley-Wilmott no-trade band (K=0.6) evaluated at the
destination friction. Per-window PnL is stored for bootstrap CIs.

Output: wf_cross_market_results.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import walk_forward_backtest as wf
from no_trade_band import BandedPolicy

ROOT = Path(__file__).resolve().parent
DIRECTIONS = [("SPY", "NIFTY"), ("NIFTY", "SPY")]
BAND_K = 0.6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,142,242")
    p.add_argument("--band-k", type=float, default=BAND_K)
    p.add_argument("--out", default=str(ROOT / "wf_cross_market_results.json"))
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)

    dev = wf.device()
    print(f"[xmarket] dev={dev} seeds={seeds} band_k={args.band_k}", flush=True)
    state = json.loads(out.read_text()) if out.exists() else {}

    for src, dst in DIRECTIONS:
        key = f"{src}__to__{dst}"
        state.setdefault(key, {})
        scfg, dcfg = wf.WINDOWS[src], wf.WINDOWS[dst]
        dst_tc = dcfg["c_tc"]
        train_loader = wf.make_loader(
            wf.make_windows(wf.fetch_close(scfg["ticker"], *scfg["train"]), stride=1))
        print(f"=== train {src} -> deploy {dst} (eval tc={dst_tc}) ===", flush=True)
        for seed in seeds:
            models = {}
            for name, trainer_fn in wf.MODEL_TRAINERS:
                t0 = time.time()
                models[name] = trainer_fn(train_loader, dev, seed)
                print(f"  seed {seed}: trained {name} in {time.time()-t0:.0f}s", flush=True)
            for tname, (s0, s1) in dcfg["tests"].items():
                state[key].setdefault(tname, {})
                closes = wf.fetch_close(dcfg["ticker"], s0, s1)
                if len(closes) < wf.N_STEPS + 2:
                    continue
                test_loader = wf.make_loader(wf.make_windows(closes, stride=1), shuffle=False)
                for name, model in models.items():
                    raw = wf.evaluate(model, test_loader, dst_tc, dev)
                    state[key][tname].setdefault(name, {})[str(seed)] = raw
                    banded = wf.evaluate(BandedPolicy(model, c=dst_tc, K=args.band_k),
                                         test_loader, dst_tc, dev)
                    state[key][tname].setdefault(f"{name}_band", {})[str(seed)] = banded
                    out.write_text(json.dumps(state, indent=2))
                    print(f"    {tname}/{name}[{seed}]: raw CVaR={raw['cvar_95']:.2f}"
                          f"/t{raw['turnover']:.2f}  band CVaR={banded['cvar_95']:.2f}"
                          f"/t{banded['turnover']:.2f}", flush=True)
    print("[xmarket] done", flush=True)


if __name__ == "__main__":
    main()
