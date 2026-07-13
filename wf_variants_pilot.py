#!/usr/bin/env python3
"""
Pilot harness for the pass-5 W-DRO-T / RSE improvements.

Reuses the *exact* causal walk-forward data pipeline of
walk_forward_backtest.py (same trailing features, same overlapping
30-day windows, same TC-inclusive evaluator) so the improved variants
are directly comparable to the pass-5 battery.

Variants
  --which rse   : RSE_base, RSE_fric (tc + turnover penalty),
                  RSE_rich (9-d regime feats), RSE_richfric
  --which wdrot : WDROT_van (constant radius), WDROT_adapt (vol-conditioned)

Each variant is trained on the calm 2017-2019 window and evaluated,
frozen, on the market's crisis cells. Output schema matches the battery
(state[market][test][variant][seed] = metrics incl. per-window pnl), so
walk_forward_bootstrap-style CIs can be computed afterwards.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import torch

import walk_forward_backtest as wf   # data pipeline + evaluate + WINDOWS
from rse import RegimeSwitchingEnsemble, RSETrainer
from rse_improved import (RSETrainerFrictionAware, RegimeSwitchingEnsembleRich)
from w_dro_t import WDROTransformerHedger, WDROTrainer
from w_dro_t_adaptive import WDROTransformerHedgerAdaptive

ROOT = Path(__file__).resolve().parent


def train_rse_base(loader, dev, seed, tc, lam):
    wf.set_seed(seed)
    m = RegimeSwitchingEnsemble(input_dim=5, n_regimes=4, delta_max=1.5).to(dev)
    RSETrainer(m, device=dev).train(loader, loader, epochs=50)
    return m

def train_rse_fric(loader, dev, seed, tc, lam):
    wf.set_seed(seed)
    m = RegimeSwitchingEnsemble(input_dim=5, n_regimes=4, delta_max=1.5).to(dev)
    RSETrainerFrictionAware(m, device=dev, tc=tc, turnover_lambda=lam).train(
        loader, loader, epochs=50)
    return m

def train_rse_rich(loader, dev, seed, tc, lam):
    wf.set_seed(seed)
    m = RegimeSwitchingEnsembleRich(input_dim=5, n_regimes=4, delta_max=1.5).to(dev)
    RSETrainer(m, device=dev).train(loader, loader, epochs=50)
    return m

def train_rse_richfric(loader, dev, seed, tc, lam):
    wf.set_seed(seed)
    m = RegimeSwitchingEnsembleRich(input_dim=5, n_regimes=4, delta_max=1.5).to(dev)
    RSETrainerFrictionAware(m, device=dev, tc=tc, turnover_lambda=lam).train(
        loader, loader, epochs=50)
    return m

def train_wdrot_van(loader, dev, seed, tc, lam):
    wf.set_seed(seed)
    m = WDROTransformerHedger(input_dim=5, d_model=64, n_heads=4, n_layers=3,
                              epsilon=0.1, delta_max=1.5)
    WDROTrainer(m, lr=1e-3, weight_decay=1e-4, device=dev).train(loader, loader, epochs=80)
    return m

def train_wdrot_adapt(loader, dev, seed, tc, lam):
    wf.set_seed(seed)
    m = WDROTransformerHedgerAdaptive(input_dim=5, d_model=64, n_heads=4, n_layers=3,
                                      epsilon=0.1, delta_max=1.5, vol_ref=0.2, beta=3.0)
    WDROTrainer(m, lr=1e-3, weight_decay=1e-4, device=dev).train(loader, loader, epochs=80)
    return m

GROUPS = {
    "rse":   [("RSE_base", train_rse_base), ("RSE_fric", train_rse_fric),
              ("RSE_rich", train_rse_rich), ("RSE_richfric", train_rse_richfric)],
    "wdrot": [("WDROT_van", train_wdrot_van), ("WDROT_adapt", train_wdrot_adapt)],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=["rse", "wdrot"], required=True)
    p.add_argument("--seeds", default="42")
    p.add_argument("--markets", default="SPY,NIFTY")
    p.add_argument("--turnover-lambda", type=float, default=1e-3)
    p.add_argument("--fric-mult", type=float, default=1.0,
                   help="train the *_fric variants as if friction were this "
                        "multiple of the deployment tc (scale-free turnover "
                        "pressure); evaluation always uses the true tc")
    p.add_argument("--only", default="",
                   help="comma-separated variant names to run (subset of the group)")
    p.add_argument("--out", default=str(ROOT / "wf_variants_results.json"))
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    variants = GROUPS[args.which]
    only = [x.strip() for x in args.only.split(",") if x.strip()]
    if only:
        variants = [(n, f) for (n, f) in variants if n in only]
    out = Path(args.out)

    dev = wf.device()
    print(f"[variants] which={args.which} dev={dev} seeds={seeds} "
          f"markets={markets} lambda={args.turnover_lambda}", flush=True)
    state = json.loads(out.read_text()) if out.exists() else {}

    for market in markets:
        cfg = wf.WINDOWS[market]
        tc = cfg["c_tc"]
        state.setdefault(market, {})
        tr = cfg["train"]
        train_closes = wf.fetch_close(cfg["ticker"], tr[0], tr[1])
        train_data = wf.make_windows(train_closes, stride=1)
        train_loader = wf.make_loader(train_data)
        print(f"=== {market} (tc={tc}) {train_data['features'].shape[0]} train windows ===", flush=True)
        train_tc = tc * args.fric_mult   # friction seen by *_fric during training
        for seed in seeds:
            models = {}
            for name, fn in variants:
                t0 = time.time()
                models[name] = fn(train_loader, dev, seed, train_tc, args.turnover_lambda)
                print(f"  seed {seed}: trained {name} in {time.time()-t0:.0f}s "
                      f"(train_tc={train_tc:.4f})", flush=True)
            for tname, (s0, s1) in cfg["tests"].items():
                state[market].setdefault(tname, {})
                test_closes = wf.fetch_close(cfg["ticker"], s0, s1)
                if len(test_closes) < wf.N_STEPS + 2:
                    continue
                test_loader = wf.make_loader(wf.make_windows(test_closes, stride=1),
                                             shuffle=False)
                for name, model in models.items():
                    m = wf.evaluate(model, test_loader, tc, dev)  # eval at TRUE tc
                    state[market][tname].setdefault(name, {})[str(seed)] = m
                    out.write_text(json.dumps(state, indent=2))
                    print(f"    {tname}/{name}[{seed}]: CVaR={m['cvar_95']:.3f} "
                          f"turn={m['turnover']:.3f} n={m['n_windows']}", flush=True)
    print("[variants] done", flush=True)


if __name__ == "__main__":
    main()
