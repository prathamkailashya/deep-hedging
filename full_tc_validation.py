#!/usr/bin/env python3
"""
Full TC-inclusive validation for the in-distribution headline ranking.

Retrains LSTM and RSE under r=0.05 with the same 10-seed grid as the
original benchmark and evaluates each model on the *complete* P\\&L
formula of eq.~(3) (transaction-cost term included). Results are
written incrementally to ``full_tc_validation_results.json`` so the
manuscript can be updated as soon as enough seeds are available.

Designed to be cheap enough to finish within a few hours on Apple
Silicon: 25k training paths, 80 epochs split 50/30, batch size 256.

Run:
    python full_tc_validation.py [--models LSTM,RSE] [--seeds 10]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "new_approaches" / "code"))

from src.env.heston import HestonParams                  # noqa: E402
from src.env.data_generator import DataGenerator         # noqa: E402
from src.models.kozyra_models import HedgingLSTM         # noqa: E402
from three_stage import ThreeStageTrainer                # noqa: E402
from rse import RegimeSwitchingEnsemble, RSETrainer      # noqa: E402

C_TC = 1e-3
R = 0.05
SEED_GRID = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]
RESULTS_FILE = ROOT / "full_tc_validation_results.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_data(seed: int, n_train: int = 25000, n_test: int = 10000):
    set_seed(seed)
    gen = DataGenerator(
        n_steps=30,
        T=30 / 365,
        S0=100.0,
        K=100.0,
        r=R,
        model_type="heston",
        heston_params=HestonParams(
            S0=100.0, v0=0.04, r=R,
            kappa=1.0, theta=0.04, sigma=0.2, rho=-0.7,
        ),
    )
    train_data, val_data, test_data = gen.generate_train_val_test(
        n_train=n_train, n_val=5000, n_test=n_test,
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


def cvar95(x: np.ndarray) -> float:
    return float(-x[x <= np.percentile(x, 5)].mean())


def evaluate(model, test_loader, test_data, dev: str) -> Dict[str, float]:
    model.eval()
    parts: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in test_loader:
            parts.append(model(batch["features"].to(dev)).cpu())
    deltas = torch.cat(parts).numpy()
    prices = test_data.stock_paths.numpy()
    payoffs = test_data.payoffs.numpy()

    price_changes = prices[:, 1:] - prices[:, :-1]
    n = min(deltas.shape[1], price_changes.shape[1])
    hedge = (deltas[:, :n] * price_changes[:, :n]).sum(axis=1)
    pnl_notc = -payoffs + hedge

    delta_changes = np.zeros_like(deltas)
    delta_changes[:, 0] = deltas[:, 0]
    delta_changes[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
    s_at_trade = np.concatenate([prices[:, :1], prices[:, 1:-1]], axis=1)
    tc = (np.abs(delta_changes) * s_at_trade * C_TC).sum(axis=1)
    pnl_tc = pnl_notc - tc

    turnover = float(np.mean(np.sum(np.abs(np.diff(deltas, axis=1)), axis=1)))

    return {
        "cvar_95_notc": cvar95(pnl_notc),
        "cvar_95_tc":   cvar95(pnl_tc),
        "mean_pnl_notc": float(pnl_notc.mean()),
        "mean_pnl_tc":   float(pnl_tc.mean()),
        "std_pnl_notc": float(pnl_notc.std()),
        "std_pnl_tc":   float(pnl_tc.std()),
        "tc_mean":      float(tc.mean()),
        "turnover":     turnover,
    }


def train_lstm(train, val, dev: str, seed: int):
    set_seed(seed)
    model = HedgingLSTM(state_dim=5, hidden_size=50, num_layers=2,
                        delta_scale=1.5).to(dev)
    trainer = ThreeStageTrainer(
        model,
        lr_stage1=1e-3, lr_stage3=1e-4,
        weight_decay=1e-4,
        epochs_stage1=50, epochs_stage3=30,
        patience_stage1=15, patience_stage3=10,
        grad_clip=5.0, device=dev,
    )
    trainer.train_full(train, val)
    return model


def train_rse(train, val, dev: str, seed: int):
    set_seed(seed)
    model = RegimeSwitchingEnsemble(input_dim=5, n_regimes=4,
                                    delta_max=1.5).to(dev)
    trainer = RSETrainer(model, device=dev)
    trainer.train(train, val, epochs=80)
    return model


def load_existing() -> Dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def save(state: Dict) -> None:
    RESULTS_FILE.write_text(json.dumps(state, indent=2))


def run(models: List[str], seeds: List[int]) -> Dict:
    dev = device()
    print(f"[full_tc_validation] device={dev}  r={R}  c_tc={C_TC}", flush=True)
    state = load_existing()
    for seed in seeds:
        key = str(seed)
        state.setdefault(key, {})
        train, val, test, test_data = build_data(seed)
        for m in models:
            if m in state[key]:
                print(f"[seed {seed}] {m}: already done, skipping", flush=True)
                continue
            t0 = time.time()
            print(f"[seed {seed}] training {m}...", flush=True)
            if m == "LSTM":
                model = train_lstm(train, val, dev, seed)
            elif m == "RSE":
                model = train_rse(train, val, dev, seed)
            else:
                continue
            metrics = evaluate(model, test, test_data, dev)
            metrics["train_time_s"] = time.time() - t0
            state[key][m] = metrics
            save(state)
            print(f"[seed {seed}] {m}: "
                  f"CVaR95_tc={metrics['cvar_95_tc']:.4f}  "
                  f"CVaR95_notc={metrics['cvar_95_notc']:.4f}  "
                  f"turnover={metrics['turnover']:.3f}  "
                  f"time={metrics['train_time_s']:.0f}s", flush=True)
    return state


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="LSTM,RSE")
    p.add_argument("--seeds", type=int, default=10)
    args = p.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    seeds = SEED_GRID[: args.seeds]
    run(models, seeds)
    print("[full_tc_validation] done", flush=True)


if __name__ == "__main__":
    main()
