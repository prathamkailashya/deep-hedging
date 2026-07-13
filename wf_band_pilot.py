#!/usr/bin/env python3
"""
No-trade-band overlay pilot (Whalley-Wilmott / Zakamouline).

Trains each base policy once on the calm 2017-2019 window, then evaluates
it on the crisis cells with the band OFF (K=0, identity) and ON at a sweep
of strengths K. The band is inference-only, so the sweep is cheap. Reuses
the exact walk-forward evaluator (TC-inclusive CVaR95, turnover, per-window
PnL) via the BandedPolicy wrapper.

Scientific question: does a gamma/vol-scaled no-trade band improve the
turnover-CVaR frontier -- especially on the high-friction NIFTY book -- and
can banded-RSE beat raw-LSTM where raw-RSE could not?
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import torch

import walk_forward_backtest as wf
from src.models.kozyra_models import HedgingLSTM
from three_stage import ThreeStageTrainer
from rse import RegimeSwitchingEnsemble, RSETrainer
from no_trade_band import BandedPolicy

ROOT = Path(__file__).resolve().parent
K_SWEEP = [0.0, 0.3, 0.6, 1.0, 1.5, 2.5]


def train_lstm(loader, dev, seed):
    wf.set_seed(seed)
    m = HedgingLSTM(state_dim=5, hidden_size=50, num_layers=2, delta_scale=1.5).to(dev)
    ThreeStageTrainer(m, lr_stage1=1e-3, lr_stage3=1e-4, weight_decay=1e-4,
                      epochs_stage1=30, epochs_stage3=20, patience_stage1=10,
                      patience_stage3=8, grad_clip=5.0, device=dev).train_full(loader, loader)
    return m


def train_rse(loader, dev, seed):
    wf.set_seed(seed)
    m = RegimeSwitchingEnsemble(input_dim=5, n_regimes=4, delta_max=1.5).to(dev)
    RSETrainer(m, device=dev).train(loader, loader, epochs=50)
    return m


BASES = [("LSTM", train_lstm), ("RSE", train_rse)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42")
    p.add_argument("--markets", default="SPY,NIFTY")
    p.add_argument("--out", default=str(ROOT / "wf_band_results.json"))
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    out = Path(args.out)

    dev = wf.device()
    print(f"[band] dev={dev} seeds={seeds} markets={markets} K={K_SWEEP}", flush=True)
    state = json.loads(out.read_text()) if out.exists() else {}

    for market in markets:
        cfg = wf.WINDOWS[market]; c = cfg["c_tc"]
        state.setdefault(market, {})
        tr = cfg["train"]
        train_loader = wf.make_loader(wf.make_windows(wf.fetch_close(cfg["ticker"], *tr), stride=1))
        print(f"=== {market} (c={c}) ===", flush=True)
        for seed in seeds:
            models = {}
            for name, fn in BASES:
                t0 = time.time(); models[name] = fn(train_loader, dev, seed)
                print(f"  seed {seed}: trained {name} in {time.time()-t0:.0f}s", flush=True)
            for tname, (s0, s1) in cfg["tests"].items():
                state[market].setdefault(tname, {})
                tc_closes = wf.fetch_close(cfg["ticker"], s0, s1)
                if len(tc_closes) < wf.N_STEPS + 2:
                    continue
                test_loader = wf.make_loader(wf.make_windows(tc_closes, stride=1), shuffle=False)
                for name, model in models.items():
                    for K in K_SWEEP:
                        tag = f"{name}_K{K}"
                        m = wf.evaluate(BandedPolicy(model, c=c, K=K), test_loader, c, dev)
                        state[market][tname].setdefault(tag, {})[str(seed)] = m
                        out.write_text(json.dumps(state, indent=2))
                    # compact per-model line
                    row = state[market][tname]
                    cs = "  ".join(f"K{K}:CVaR={row[f'{name}_K{K}'][str(seed)]['cvar_95']:.2f}"
                                   f"/t{row[f'{name}_K{K}'][str(seed)]['turnover']:.2f}"
                                   for K in K_SWEEP)
                    print(f"    {tname}/{name}[{seed}]: {cs}", flush=True)
    print("[band] done", flush=True)


if __name__ == "__main__":
    main()
