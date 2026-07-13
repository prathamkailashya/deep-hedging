#!/usr/bin/env python3
"""
Multi-seed robustness check for the SPY calm-15 regime, using the
full-fidelity model implementations. Reports mean +/- std across
seeds for LSTM and 3SCH (RSE is skipped because its per-seed wall
time on shared MPS is too costly for a quick robustness panel).
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "new_approaches" / "code"))

from src.env.heston import HestonParams
from src.env.data_generator import DataGenerator
from src.models.kozyra_models import HedgingLSTM
from three_stage import ThreeStageTrainer

SEEDS = [42, 142, 242]
OUT = ROOT / "spy_multiseed_results.json"
C_TC = 0.0003
R = 0.05
S0 = 450.0
SIGMA = 0.15


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_data(seed):
    set_seed(seed)
    gen = DataGenerator(
        n_steps=30, T=30/365, S0=S0, K=S0, r=R,
        model_type='heston',
        heston_params=HestonParams(
            S0=S0, v0=SIGMA**2, r=R,
            kappa=1.0, theta=SIGMA**2, sigma=0.2, rho=-0.7,
        ),
    )
    train, val, test = gen.generate_train_val_test(
        n_train=8000, n_val=2000, n_test=5000,
        base_seed=seed, compute_bs_delta=True,
    )
    for ds in (train, val, test):
        if ds.bs_deltas is not None:
            ds.features = torch.cat([ds.features, ds.bs_deltas.unsqueeze(-1)], dim=-1)
            ds.n_features = ds.features.shape[2]
    return gen.get_dataloaders(train, val, test, batch_size=256) + (test,)


def evaluate(model, test_loader, test_data, dev):
    model.eval()
    deltas = []
    with torch.no_grad():
        for b in test_loader:
            deltas.append(model(b['features'].to(dev)).cpu())
    deltas = torch.cat(deltas).numpy()
    prices = test_data.stock_paths.numpy()
    payoffs = test_data.payoffs.numpy()
    dS = prices[:, 1:] - prices[:, :-1]
    n = min(deltas.shape[1], dS.shape[1])
    hedge = (deltas[:, :n] * dS[:, :n]).sum(axis=1)
    dch = np.zeros_like(deltas); dch[:, 0] = deltas[:, 0]
    dch[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
    s_t = np.concatenate([prices[:, :1], prices[:, 1:-1]], axis=1)
    tc = (np.abs(dch) * s_t * C_TC).sum(axis=1)
    pnl = -payoffs + hedge - tc
    return {
        'cvar_95': float(-pnl[pnl <= np.percentile(pnl, 5)].mean()),
        'turnover': float(np.mean(np.sum(np.abs(np.diff(deltas, axis=1)), axis=1))),
    }


def train_model(maker, train, val, dev, seed, kw):
    set_seed(seed)
    model = maker(dev)
    trainer = ThreeStageTrainer(model, **kw, device=dev)
    trainer.train_full(train, val)
    return model


def main():
    dev = device()
    print(f"[multiseed] device={dev}", flush=True)
    state = {}
    if OUT.exists():
        state = json.loads(OUT.read_text())
    for seed in SEEDS:
        k = str(seed)
        state.setdefault(k, {})
        train, val, test, test_data = build_data(seed)
        for name, maker, kw in [
            ('LSTM', lambda d: HedgingLSTM(state_dim=5, hidden_size=50, num_layers=2,
                                          delta_scale=1.5).to(d),
             dict(lr_stage1=1e-3, lr_stage3=1e-4, weight_decay=1e-4,
                  epochs_stage1=30, epochs_stage3=20,
                  patience_stage1=10, patience_stage3=8, grad_clip=5.0)),
            ('3SCH', lambda d: HedgingLSTM(state_dim=5, hidden_size=50, num_layers=2,
                                           delta_scale=1.5).to(d),
             dict(lr_stage1=1e-3, lr_stage2=5e-4, lr_stage3=1e-4, weight_decay=1e-4,
                  epochs_stage1=20, epochs_stage2=10, epochs_stage3=15,
                  patience_stage1=10, patience_stage3=8, grad_clip=5.0)),
        ]:
            if name in state[k]:
                print(f"[seed {seed}] {name}: cached", flush=True)
                continue
            t0 = time.time()
            model = train_model(maker, train, val, dev, seed, kw)
            m = evaluate(model, test, test_data, dev)
            m['train_time_s'] = time.time() - t0
            state[k][name] = m
            OUT.write_text(json.dumps(state, indent=2))
            print(f"[seed {seed}] {name}: CVaR={m['cvar_95']:.3f}  "
                  f"turn={m['turnover']:.3f}  t={m['train_time_s']:.0f}s",
                  flush=True)
    print("[multiseed] done", flush=True)


if __name__ == "__main__":
    main()
