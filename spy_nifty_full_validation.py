#!/usr/bin/env python3
"""
SPY / NIFTY regime-calibrated stress battery with the *full* model
implementations (`new_approaches/code/*`), replacing the simplified
validation-grade models in
``new_approaches/experiments/extended_real_market_validation.py``.

Output: ``spy_nifty_full_validation_results.json`` (written incrementally).

Models trained per cell: LSTM, Transformer, 3SCH, RSE. W-DRO-T is
skipped here because its per-seed wall time (~2h on MPS) is
prohibitive across the 4-7 scenarios; we report the closed-form
relationship between W-DRO-T's in-distribution turnover and its
expected CVaR shift in the manuscript instead.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "new_approaches" / "code"))

from src.env.heston import HestonParams                  # noqa: E402
from src.env.data_generator import DataGenerator         # noqa: E402
from src.models.kozyra_models import HedgingLSTM         # noqa: E402
from src.models.transformer import TransformerHedge     # noqa: E402
from three_stage import ThreeStageTrainer                # noqa: E402
from rse import RegimeSwitchingEnsemble, RSETrainer      # noqa: E402

SPY = {
    "S0": 450.0, "r": 0.05, "c_tc": 0.0003,
    "scenarios": {
        "calm_15":      0.15,
        "post_covid_20": 0.20,
        "covid_65":      0.65,
        "gfc_80":        0.80,
    },
}
NIFTY = {
    "S0": 18000.0, "r": 0.065, "c_tc": 0.0018,
    "scenarios": {
        "calm_14": 0.14,
        "covid_55": 0.55,
        "gfc_70":   0.70,
    },
}

OUT = ROOT / "spy_nifty_full_validation_results.json"


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_data(vol: float, S0: float, r: float, seed: int,
               n_train: int = 8000, n_test: int = 5000):
    set_seed(seed)
    gen = DataGenerator(
        n_steps=30,
        T=30 / 365,
        S0=S0,
        K=S0,
        r=r,
        model_type="heston",
        heston_params=HestonParams(
            S0=S0, v0=vol * vol, r=r,
            kappa=1.0, theta=vol * vol, sigma=0.2, rho=-0.7,
        ),
    )
    train_data, val_data, test_data = gen.generate_train_val_test(
        n_train=n_train, n_val=2000, n_test=n_test,
        base_seed=seed, compute_bs_delta=True,
    )
    for ds in (train_data, val_data, test_data):
        if ds.bs_deltas is not None:
            bs = ds.bs_deltas.unsqueeze(-1)
            ds.features = torch.cat([ds.features, bs], dim=-1)
            ds.n_features = ds.features.shape[2]
    train_loader, val_loader, test_loader = gen.get_dataloaders(
        train_data, val_data, test_data, batch_size=256,
    )
    return train_loader, val_loader, test_loader, test_data


def evaluate(model, test_loader, test_data, c_tc: float, dev: str) -> Dict[str, float]:
    model.eval()
    parts = []
    with torch.no_grad():
        for batch in test_loader:
            parts.append(model(batch["features"].to(dev)).cpu())
    deltas = torch.cat(parts).numpy()
    prices = test_data.stock_paths.numpy()
    payoffs = test_data.payoffs.numpy()

    dS = prices[:, 1:] - prices[:, :-1]
    n = min(deltas.shape[1], dS.shape[1])
    hedge = (deltas[:, :n] * dS[:, :n]).sum(axis=1)
    dch = np.zeros_like(deltas)
    dch[:, 0] = deltas[:, 0]
    dch[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
    s_trade = np.concatenate([prices[:, :1], prices[:, 1:-1]], axis=1)
    tc = (np.abs(dch) * s_trade * c_tc).sum(axis=1)
    pnl = -payoffs + hedge - tc
    cv95 = float(-pnl[pnl <= np.percentile(pnl, 5)].mean())
    turnover = float(np.mean(np.sum(np.abs(np.diff(deltas, axis=1)), axis=1)))
    return {
        "cvar_95": cv95,
        "std_pnl": float(pnl.std()),
        "mean_pnl": float(pnl.mean()),
        "turnover": turnover,
        "tc_mean": float(tc.mean()),
    }


def make_lstm(dev: str):
    return HedgingLSTM(state_dim=5, hidden_size=50, num_layers=2,
                       delta_scale=1.5).to(dev)


def make_transformer(dev: str):
    return TransformerHedge(input_dim=5, d_model=64, n_heads=4,
                            n_layers=3, dropout=0.1).to(dev)


def make_rse(dev: str):
    return RegimeSwitchingEnsemble(input_dim=5, n_regimes=4,
                                   delta_max=1.5).to(dev)


def train_lstm_like(model, train, val, dev: str, epochs=(30, 20)):
    trainer = ThreeStageTrainer(
        model, lr_stage1=1e-3, lr_stage3=1e-4,
        weight_decay=1e-4,
        epochs_stage1=epochs[0], epochs_stage3=epochs[1],
        patience_stage1=10, patience_stage3=8,
        grad_clip=5.0, device=dev,
    )
    trainer.train_full(train, val)


def train_3sch(model, train, val, dev: str, epochs=(20, 10, 15)):
    trainer = ThreeStageTrainer(
        model, lr_stage1=1e-3, lr_stage2=5e-4, lr_stage3=1e-4,
        weight_decay=1e-4,
        epochs_stage1=epochs[0],
        epochs_stage2=epochs[1],
        epochs_stage3=epochs[2],
        patience_stage1=10, patience_stage3=8,
        grad_clip=5.0, device=dev,
    )
    trainer.train_full(train, val)


def train_rse_proper(model, train, val, dev: str, epochs=50):
    trainer = RSETrainer(model, device=dev)
    trainer.train(train, val, epochs=epochs)


def load_state() -> Dict:
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {}


def save_state(state: Dict) -> None:
    OUT.write_text(json.dumps(state, indent=2))


def run_cell(market: str, sname: str, vol: float, cfg: Dict, dev: str,
             state: Dict, models_to_run, seed: int = 42) -> None:
    cell = state.setdefault(market, {}).setdefault(sname, {})
    train, val, test, test_data = build_data(
        vol, cfg["S0"], cfg["r"], seed,
    )
    for mname, builder, trainer in models_to_run:
        if mname in cell:
            print(f"[{market} {sname}] {mname}: cached", flush=True)
            continue
        t0 = time.time()
        set_seed(seed)
        model = builder(dev)
        trainer(model, train, val, dev)
        m = evaluate(model, test, test_data, cfg["c_tc"], dev)
        m["train_time_s"] = time.time() - t0
        cell[mname] = m
        save_state(state)
        print(f"[{market} {sname}] {mname}: "
              f"CVaR95={m['cvar_95']:.3f}  turnover={m['turnover']:.3f}  "
              f"time={m['train_time_s']:.0f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--markets", default="SPY,NIFTY")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dev = device()
    print(f"[spy_nifty_full] device={dev}", flush=True)
    state = load_state()

    # Transformer is intentionally omitted from the full-model SPY/NIFTY
    # rerun: at $8$K paths the wall time per cell on shared MPS is
    # incompatible with completing the full battery in a reasonable
    # window. The closed-form analysis of Table~\ref{tab:tc_check}
    # provides the Transformer-side comparison.
    models = [
        ("LSTM", make_lstm,  train_lstm_like),
        ("3SCH", make_lstm,  train_3sch),
        ("RSE",  make_rse,   train_rse_proper),
    ]

    if "SPY" in args.markets.split(","):
        for sname, vol in SPY["scenarios"].items():
            print(f"\n=== SPY {sname} (sigma={vol}) ===", flush=True)
            run_cell("SPY", sname, vol, SPY, dev, state, models, args.seed)
    if "NIFTY" in args.markets.split(","):
        for sname, vol in NIFTY["scenarios"].items():
            print(f"\n=== NIFTY {sname} (sigma={vol}) ===", flush=True)
            run_cell("NIFTY", sname, vol, NIFTY, dev, state, models, args.seed)
    print("[spy_nifty_full] done", flush=True)


if __name__ == "__main__":
    main()
