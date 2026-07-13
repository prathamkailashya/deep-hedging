#!/usr/bin/env python3
"""
Re-evaluate the in-distribution headline ranking with the *full* P&L formula
that includes the proportional transaction cost (TC) term. The published
``compute_pnl`` in ``new_approaches/experiments/run_full_experiments.py``
applies TC during training (via ``ThreeStageTrainer._compute_pnl``) but
omits the same TC term in the reported evaluation P&L. This script:

1. retrains LSTM and RSE for two seeds (LSTM is cheap: ~5 min on Apple Silicon),
2. evaluates each on the *same* test set using two metrics
   - ``cvar_95_notc`` (the legacy evaluation formula),
   - ``cvar_95_tc``  (the formula in eq.~(3) of the manuscript),
3. reports the relative gap so the headline ranking can be checked.

The script is deliberately small. The goal is a sanity check, not a new
benchmark.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'new_approaches' / 'code'))

from src.env.heston import HestonParams  # noqa: E402
from src.env.data_generator import DataGenerator  # noqa: E402
from src.models.kozyra_models import HedgingLSTM  # noqa: E402
from three_stage import ThreeStageTrainer  # noqa: E402
from rse import RegimeSwitchingEnsemble, RSETrainer  # noqa: E402


C_TC = 1e-3


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_data(seed: int, n_train: int = 20000, n_test: int = 10000,
               r: float = 0.05):
    set_seed(seed)
    generator = DataGenerator(
        n_steps=30,
        T=30 / 365,
        S0=100.0,
        K=100.0,
        r=r,
        model_type='heston',
        heston_params=HestonParams(
            S0=100.0, v0=0.04, r=r,
            kappa=1.0, theta=0.04, sigma=0.2, rho=-0.7,
        ),
    )
    train_data, val_data, test_data = generator.generate_train_val_test(
        n_train=n_train, n_val=5000, n_test=n_test,
        base_seed=seed, compute_bs_delta=True,
    )
    for ds in (train_data, val_data, test_data):
        if ds.bs_deltas is not None:
            bs = ds.bs_deltas.unsqueeze(-1)
            ds.features = torch.cat([ds.features, bs], dim=-1)
            ds.n_features = ds.features.shape[2]
    train_loader, val_loader, test_loader = generator.get_dataloaders(
        train_data, val_data, test_data, batch_size=256,
    )
    return train_loader, val_loader, test_loader, test_data


def compute_metrics(deltas: np.ndarray, prices: np.ndarray, payoffs: np.ndarray):
    price_changes = prices[:, 1:] - prices[:, :-1]
    n = min(deltas.shape[1], price_changes.shape[1])
    hedge = (deltas[:, :n] * price_changes[:, :n]).sum(axis=1)
    pnl_notc = -payoffs + hedge

    delta_changes = np.zeros_like(deltas)
    delta_changes[:, 0] = deltas[:, 0]
    delta_changes[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
    # Use S_k at each rebalancing step (the manuscript's eq. 3)
    s_at_trade = np.concatenate([prices[:, :1], prices[:, 1:-1]], axis=1)
    tc = (np.abs(delta_changes) * s_at_trade * C_TC).sum(axis=1)
    pnl_tc = pnl_notc - tc

    cvar95 = lambda x: float(-x[x <= np.percentile(x, 5)].mean())
    turnover = float(np.mean(np.sum(np.abs(np.diff(deltas, axis=1)), axis=1)))
    return {
        'cvar_95_notc': cvar95(pnl_notc),
        'cvar_95_tc':   cvar95(pnl_tc),
        'mean_pnl_notc': float(pnl_notc.mean()),
        'mean_pnl_tc':   float(pnl_tc.mean()),
        'tc_mean':       float(tc.mean()),
        'turnover':      turnover,
    }


def evaluate(model, test_loader, test_data, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for batch in test_loader:
            parts.append(model(batch['features'].to(device)).cpu())
    deltas = torch.cat(parts).numpy()
    return compute_metrics(deltas, test_data.stock_paths.numpy(), test_data.payoffs.numpy())


def train_lstm(loaders, device, epochs1=30, epochs3=20):
    train, val, _, _ = loaders
    model = HedgingLSTM(state_dim=5, hidden_size=50, num_layers=2,
                        delta_scale=1.5).to(device)
    trainer = ThreeStageTrainer(
        model,
        lr_stage1=1e-3, lr_stage3=1e-4,
        weight_decay=1e-4,
        epochs_stage1=epochs1, epochs_stage3=epochs3,
        patience_stage1=10, patience_stage3=8,
        grad_clip=5.0, device=device,
    )
    trainer.train_full(train, val)
    return model


def train_rse(loaders, device, epochs=50):
    train, val, _, _ = loaders
    model = RegimeSwitchingEnsemble(input_dim=5, n_regimes=4,
                                    delta_max=1.5).to(device)
    trainer = RSETrainer(model, device=device)
    trainer.train(train, val, epochs=epochs)
    return model


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Device: {device}')
    seeds = [42, 142]
    out = {}
    for seed in seeds:
        print(f'\n=== seed {seed} ===')
        train_loader, val_loader, test_loader, test_data = build_data(seed)
        out.setdefault(str(seed), {})

        t0 = time.time()
        set_seed(seed)
        lstm = train_lstm((train_loader, val_loader, test_loader, test_data), device)
        out[str(seed)]['LSTM'] = evaluate(lstm, test_loader, test_data, device)
        out[str(seed)]['LSTM']['train_time_s'] = time.time() - t0

        t0 = time.time()
        set_seed(seed)
        rse = train_rse((train_loader, val_loader, test_loader, test_data), device)
        out[str(seed)]['RSE'] = evaluate(rse, test_loader, test_data, device)
        out[str(seed)]['RSE']['train_time_s'] = time.time() - t0

        print(json.dumps(out[str(seed)], indent=2))

    with open(ROOT / 'tc_recompute_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('Saved tc_recompute_results.json')


if __name__ == '__main__':
    main()
