#!/usr/bin/env python3
"""
Walk-forward backtest on raw OHLCV data for SPY and NIFTY-50.

Training window: 2017-01-01 -- 2019-12-31 (calm).
Test windows:
  - SPY: COVID-acute 2020-02-01 -- 2020-04-30
  - SPY: GFC 2007-07-01 -- 2009-06-30
  - SPY: Post-COVID 2021-01-01 -- 2021-12-31
  - NIFTY: COVID-acute 2020-02-01 -- 2020-04-30
  - NIFTY: GFC 2007-07-01 -- 2009-06-30

Protocol:
  - Train each model on overlapping (stride-1) 30-day windows from the
    training period (last day = option maturity; spot at the window
    start = strike).
  - Evaluate on the same kind of overlapping 30-day windows from each
    test period, using the *fixed* trained policy.
  - The realised-vol feature at decision time t uses only returns
    realised strictly before t (trailing window of up to 20 returns),
    so every feature is measurable at the time the hedge is chosen.
  - Report TC-inclusive CVaR_95 per (ticker, test_window, model, seed),
    and persist the per-window PnL vector so that block-bootstrap
    confidence intervals can be computed post-hoc (walk_forward_bootstrap.py)
    without retraining.

Models: LSTM, 3SCH, RSE, and WDROT (Wasserstein-DRO Transformer, the
manuscript's flagship robust method, added to the causal walk-forward
in pass 5).

Output: walk_forward_ci_results.json
(schema state[market][test][model][seed] = metrics, where metrics
includes a per-window "pnl" list). The pass-4 file
walk_forward_multiseed_results.json (LSTM/3SCH/RSE only, no per-window
PnL) is left untouched as provenance; the pass-3 single-seed file
walk_forward_backtest_results.json was produced by a pre-fix (leaky)
feature pipeline and is retained only for the record.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "new_approaches" / "code"))

from src.models.kozyra_models import HedgingLSTM
from three_stage import ThreeStageTrainer
from rse import RegimeSwitchingEnsemble, RSETrainer
from w_dro_t import WDROTransformerHedger, WDROTrainer

OUT = ROOT / "walk_forward_ci_results.json"

WINDOWS = {
    "SPY": {
        "ticker": "SPY",
        "c_tc": 0.0003,
        "r": 0.05,
        "train": ("2017-01-01", "2019-12-31"),
        "tests": {
            "covid_2020": ("2020-02-01", "2020-04-30"),
            "post_covid_2021": ("2021-01-01", "2021-12-31"),
            "gfc_2008": ("2007-07-01", "2009-06-30"),
        },
    },
    "NIFTY": {
        "ticker": "^NSEI",
        "c_tc": 0.0018,
        "r": 0.065,
        "train": ("2017-01-01", "2019-12-31"),
        "tests": {
            "covid_2020": ("2020-02-01", "2020-04-30"),
            "gfc_2008": ("2007-07-01", "2009-06-30"),
        },
    },
}

N_STEPS = 30
T_FRAC = N_STEPS / 252.0
BATCH = 64


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_CLOSE_CACHE: Dict[Tuple[str, str, str], np.ndarray] = {}


def fetch_close(ticker: str, start: str, end: str) -> np.ndarray:
    """Fetch daily close prices. Raises on failure: results in this
    script feed the manuscript, so silently substituting simulated
    prices is never acceptable."""
    key = (ticker, start, end)
    if key not in _CLOSE_CACHE:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False,
                         auto_adjust=True)
        if df.empty or "Close" not in df.columns:
            raise RuntimeError(f"empty yfinance frame for {ticker} "
                               f"{start}--{end}")
        _CLOSE_CACHE[key] = df["Close"].to_numpy().flatten()
    return _CLOSE_CACHE[key]


def make_windows(closes: np.ndarray, n_steps: int = N_STEPS,
                 stride: int = 1) -> Dict[str, np.ndarray]:
    """Slide overlapping 30-day windows (stride 1). Strike = first close
    of the window. Payoff = (S_T - K)+.

    Causality: the realised-vol feature at decision time t is the std of
    log-returns realised strictly before t, i.e. log_returns[:, max(0, t-20):t],
    where log_returns[:, k] = log(S_{k+1}/S_k) is known only at time k+1.
    At t in {0, 1} no in-window history exists, so the estimate degenerates
    to 0 and is lifted to the clip floor of 0.05.
    """
    starts = list(range(0, len(closes) - n_steps - 1, stride))
    if not starts:
        raise RuntimeError("not enough data for window")
    paths = np.stack([closes[i:i + n_steps + 1] for i in starts]).astype(np.float32)
    S0 = paths[:, :1]
    strike = paths[:, 0]
    payoff = np.maximum(paths[:, -1] - strike, 0)
    norm = paths / S0
    log_mny = np.log(np.maximum(paths[:, :-1], 1e-8) /
                     strike[:, None])
    log_returns = np.diff(np.log(np.maximum(paths, 1e-8)), axis=1)
    realised_vol = np.zeros_like(log_mny)
    for t in range(1, log_mny.shape[1]):
        seg = log_returns[:, max(0, t - 20):t]
        realised_vol[:, t] = seg.std(axis=1) * math.sqrt(252)
    realised_vol = np.clip(realised_vol, 0.05, 1.5)
    tau = np.linspace(T_FRAC, 0, n_steps)
    tau_b = np.broadcast_to(tau[None, :], log_mny.shape)
    from scipy.stats import norm as _norm
    d1 = (log_mny + (0.0 + 0.5 * realised_vol ** 2) * tau_b) / (
        realised_vol * np.sqrt(np.maximum(tau_b, 1e-8))
    )
    bs_delta = _norm.cdf(d1)
    features = np.stack(
        [norm[:, :-1], log_mny, realised_vol, tau_b, bs_delta],
        axis=-1,
    ).astype(np.float32)
    return {
        "features": torch.from_numpy(features),
        "prices":   torch.from_numpy(paths),
        "payoff":   torch.from_numpy(payoff),
    }


def split_windows(d: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict]:
    n = d["features"].shape[0]
    idx = torch.randperm(n)
    cut = max(1, int(0.9 * n))
    sel = lambda ii: {k: v[ii] for k, v in d.items()}
    return sel(idx[:cut]), sel(idx[cut:])


def make_loader(d: Dict[str, torch.Tensor], batch_size: int = BATCH,
                shuffle: bool = True):
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(d["features"], d["prices"], d["payoff"])
    def _coll(b):
        f, p, y = zip(*b)
        return {"features": torch.stack(f), "prices": torch.stack(p),
                "stock_paths": torch.stack(p), "payoff": torch.stack(y)}
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                       collate_fn=_coll)


def evaluate(model, loader, c_tc: float, dev: str) -> Dict[str, float]:
    model.eval()
    # Use the model's own device so that a model which fell back to CPU
    # (e.g. WDROT when MPS lacks double-backward) still evaluates correctly.
    mdev = next(model.parameters()).device
    rows_d, rows_p, rows_y = [], [], []
    with torch.no_grad():
        for b in loader:
            d = model(b["features"].to(mdev)).cpu()
            rows_d.append(d); rows_p.append(b["prices"]); rows_y.append(b["payoff"])
    deltas = torch.cat(rows_d).numpy()
    prices = torch.cat(rows_p).numpy()
    payoffs = torch.cat(rows_y).numpy()
    dS = prices[:, 1:] - prices[:, :-1]
    n = min(deltas.shape[1], dS.shape[1])
    hedge = (deltas[:, :n] * dS[:, :n]).sum(axis=1)
    dch = np.zeros_like(deltas); dch[:, 0] = deltas[:, 0]
    dch[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
    s_t = np.concatenate([prices[:, :1], prices[:, 1:-1]], axis=1)
    tc = (np.abs(dch) * s_t * c_tc).sum(axis=1)
    pnl = -payoffs + hedge - tc
    return {
        "cvar_95": float(-pnl[pnl <= np.percentile(pnl, 5)].mean()),
        "mean_pnl": float(pnl.mean()),
        "std_pnl":  float(pnl.std()),
        "turnover": float(np.mean(np.sum(np.abs(np.diff(deltas, axis=1)), axis=1))),
        "n_windows": int(len(pnl)),
        # Per-window PnL, in test-window order (identical order across models
        # because the test loader is built with shuffle=False). Persisted so
        # that paired moving-block-bootstrap CVaR_95 CIs can be computed
        # post-hoc without retraining. Rounded to keep the JSON compact.
        "pnl": [round(float(x), 5) for x in pnl.tolist()],
    }


def train_lstm_on(loader, dev, seed):
    set_seed(seed)
    model = HedgingLSTM(state_dim=5, hidden_size=50, num_layers=2,
                        delta_scale=1.5).to(dev)
    trainer = ThreeStageTrainer(
        model, lr_stage1=1e-3, lr_stage3=1e-4, weight_decay=1e-4,
        epochs_stage1=30, epochs_stage3=20, patience_stage1=10,
        patience_stage3=8, grad_clip=5.0, device=dev,
    )
    trainer.train_full(loader, loader)
    return model


def train_3sch_on(loader, dev, seed):
    set_seed(seed)
    model = HedgingLSTM(state_dim=5, hidden_size=50, num_layers=2,
                        delta_scale=1.5).to(dev)
    trainer = ThreeStageTrainer(
        model, lr_stage1=1e-3, lr_stage2=5e-4, lr_stage3=1e-4,
        weight_decay=1e-4, epochs_stage1=20, epochs_stage2=10,
        epochs_stage3=15, patience_stage1=10, patience_stage3=8,
        grad_clip=5.0, device=dev,
    )
    trainer.train_full(loader, loader)
    return model


def train_rse_on(loader, dev, seed):
    set_seed(seed)
    model = RegimeSwitchingEnsemble(input_dim=5, n_regimes=4,
                                    delta_max=1.5).to(dev)
    trainer = RSETrainer(model, device=dev)
    trainer.train(loader, loader, epochs=50)
    return model


def train_wdrot_on(loader, dev, seed):
    """Wasserstein-DRO Transformer (manuscript flagship robust method).

    Same 80-epoch protocol used in the in-distribution full-model suite
    (50 CVaR-pretraining + 30 entropic/DRO with epsilon annealed 0->0.1).
    The DRO gradient penalty needs a second-order (double-backward)
    gradient through attention; if the accelerator backend does not
    support that (some MPS builds do not), fall back to CPU rather than
    silently skipping the model."""
    set_seed(seed)
    model = WDROTransformerHedger(input_dim=5, d_model=64, n_heads=4,
                                  n_layers=3, epsilon=0.1, delta_max=1.5)
    try:
        trainer = WDROTrainer(model, lr=1e-3, weight_decay=1e-4, device=dev)
        trainer.train(loader, loader, epochs=80)
        model._eval_device = dev
    except (RuntimeError, NotImplementedError) as e:
        print(f"    [WDROT] {dev} training failed ({type(e).__name__}: "
              f"{str(e)[:80]}...); retrying on cpu", flush=True)
        set_seed(seed)
        model = WDROTransformerHedger(input_dim=5, d_model=64, n_heads=4,
                                      n_layers=3, epsilon=0.1, delta_max=1.5)
        trainer = WDROTrainer(model, lr=1e-3, weight_decay=1e-4, device="cpu")
        trainer.train(loader, loader, epochs=80)
        model._eval_device = "cpu"
    return model


MODEL_TRAINERS = [
    ("LSTM",  train_lstm_on),
    ("3SCH",  train_3sch_on),
    ("RSE",   train_rse_on),
    ("WDROT", train_wdrot_on),
]


def seed_complete(state, market: str, tests, seed: int) -> bool:
    return all(
        str(seed) in state.get(market, {}).get(t, {}).get(m, {})
        for t in tests
        for m, _ in MODEL_TRAINERS
    )


def main():
    import argparse
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,142,242")
    p.add_argument("--markets", default="",
                   help="comma-separated subset of {SPY,NIFTY}; empty = all "
                        "(used for pilot runs)")
    p.add_argument("--tests", default="",
                   help="comma-separated subset of test-window names; empty = "
                        "all (used for pilot runs)")
    p.add_argument("--out", default=str(OUT),
                   help="output JSON path (pilot runs should use a scratch file)")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    OUT = Path(args.out)
    markets_sel = [m.strip() for m in args.markets.split(",") if m.strip()]
    tests_sel = [t.strip() for t in args.tests.split(",") if t.strip()]

    dev = device()
    print(f"[walk_forward] device={dev}  seeds={seeds}  out={OUT.name}"
          f"  markets={markets_sel or 'all'}  tests={tests_sel or 'all'}",
          flush=True)
    state = {}
    if OUT.exists():
        state = json.loads(OUT.read_text())

    for market, cfg in WINDOWS.items():
        if markets_sel and market not in markets_sel:
            continue
        if tests_sel:
            cfg = dict(cfg)
            cfg["tests"] = {t: w for t, w in cfg["tests"].items()
                            if t in tests_sel}
            if not cfg["tests"]:
                continue
        print(f"\n=== {market} ===", flush=True)
        state.setdefault(market, {})
        t_start, t_end = cfg["train"]
        print(f"  training closes {cfg['ticker']} {t_start}..{t_end}", flush=True)
        train_closes = fetch_close(cfg["ticker"], t_start, t_end)
        if len(train_closes) < N_STEPS + 2:
            print(f"  insufficient training data ({len(train_closes)} days); skipping {market}", flush=True)
            continue
        train_data = make_windows(train_closes, stride=1)
        n_train_windows = train_data["features"].shape[0]
        print(f"  {n_train_windows} training windows", flush=True)

        for seed in seeds:
            if seed_complete(state, market, cfg["tests"], seed):
                print(f"  seed {seed}: all cells cached", flush=True)
                continue

            train_loader = make_loader(train_data)
            models = {}
            for name, trainer_fn in MODEL_TRAINERS:
                t0 = time.time()
                models[name] = trainer_fn(train_loader, dev, seed)
                print(f"  seed {seed}: trained {name} in {time.time()-t0:.0f}s", flush=True)

            for tname, (s_test, e_test) in cfg["tests"].items():
                state[market].setdefault(tname, {})
                print(f"  seed {seed}: test window {tname} {s_test}..{e_test}", flush=True)
                test_closes = fetch_close(cfg["ticker"], s_test, e_test)
                if len(test_closes) < N_STEPS + 2:
                    print(f"    insufficient test data ({len(test_closes)} days); skipping", flush=True)
                    continue
                test_data = make_windows(test_closes, stride=1)
                test_loader = make_loader(test_data, shuffle=False)
                for name, model in models.items():
                    cell = state[market][tname].setdefault(name, {})
                    if str(seed) in cell:
                        continue
                    m = evaluate(model, test_loader, cfg["c_tc"], dev)
                    cell[str(seed)] = m
                    OUT.write_text(json.dumps(state, indent=2))
                    print(f"    {name}[{seed}]: CVaR={m['cvar_95']:.3f}  "
                          f"turn={m['turnover']:.3f}  "
                          f"n={m['n_windows']}", flush=True)
    print("[walk_forward] done", flush=True)


if __name__ == "__main__":
    main()
