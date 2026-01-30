#!/usr/bin/env python3
"""
Quick test of simplified Hybrid Sig-LSTM model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from utils.config import set_seed
from env.data_generator import DataGenerator
from env.heston import HestonParams
from models.kozyra_models import HedgingLSTM
from models.hybrid_simple import HybridSigLSTM, HybridTrainer
from train.trainer import DeepHedgingTrainer

set_seed(42)
device = 'cpu'

N_TRAIN, N_VAL, N_TEST = 15000, 3000, 30000
BATCH_SIZE = 256
DELTA_MAX = 1.5

print("="*60)
print("TESTING SIMPLIFIED HYBRID MODEL")
print("="*60)

# Generate data
print("\nGenerating data...")
heston_params = HestonParams(S0=100.0, v0=0.04, r=0.0, kappa=1.0, theta=0.04, sigma=0.2, rho=-0.7)
generator = DataGenerator(n_steps=30, T=30/365, S0=100.0, K=100.0, r=0.0,
                          cost_multiplier=0.0, model_type='heston', heston_params=heston_params)

train_data, val_data, test_data = generator.generate_train_val_test(
    n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, base_seed=42
)
train_loader, val_loader, test_loader = generator.get_dataloaders(
    train_data, val_data, test_data, batch_size=BATCH_SIZE
)

input_dim = test_data.n_features
print(f"Input dim: {input_dim}")


def compute_metrics(pnl, deltas):
    losses = -pnl
    delta_changes = np.abs(np.diff(
        np.concatenate([np.zeros((len(deltas), 1)), deltas, np.zeros((len(deltas), 1))], axis=1), axis=1
    ))
    volume = np.mean(np.sum(delta_changes, axis=1))
    
    scaled = -1.0 * pnl
    max_val = np.max(scaled)
    entropic = (max_val + np.log(np.mean(np.exp(scaled - max_val))))
    
    return {
        'std_pnl': np.std(pnl),
        'cvar_95': np.mean(losses[losses >= np.percentile(losses, 95)]),
        'entropic': entropic,
        'volume': volume,
        'smooth': np.mean(np.abs(np.diff(deltas, axis=1)))
    }


# 1. LSTM Baseline
print("\n[1] LSTM Baseline...")
lstm = HedgingLSTM(state_dim=input_dim, hidden_size=50, num_layers=2, delta_scale=DELTA_MAX).to(device)
trainer = DeepHedgingTrainer(model=lstm, lambda_risk=1.0, learning_rate=0.0005, device=device)
trainer.fit(train_loader, val_loader, n_epochs=40, patience=15, verbose=True)
_, lstm_pnl, lstm_deltas = trainer.evaluate(test_loader)
lstm_metrics = compute_metrics(lstm_pnl, lstm_deltas)
print(f"  LSTM: std={lstm_metrics['std_pnl']:.4f}, CVaR95={lstm_metrics['cvar_95']:.4f}, vol={lstm_metrics['volume']:.4f}")


# 2. Simplified Hybrid
print("\n[2] Simplified Hybrid Sig-LSTM...")
hybrid = HybridSigLSTM(
    input_dim=input_dim,
    hidden_size=50,
    num_layers=2,
    sig_window=5,
    delta_max=DELTA_MAX,
    max_increment=0.2,
    dropout=0.1
).to(device)

print(f"  Parameters: {sum(p.numel() for p in hybrid.parameters()):,}")

hybrid_trainer = HybridTrainer(
    model=hybrid,
    lambda_risk=1.0,
    increment_penalty=0.01,
    device=device
)

hybrid_trainer.train_stage1(train_loader, val_loader, n_epochs=25, lr=0.001, patience=10)
hybrid_trainer.train_stage2(train_loader, val_loader, n_epochs=15, lr=0.0001, patience=10)

hybrid_metrics, hybrid_pnl, hybrid_deltas = hybrid_trainer.evaluate(test_loader)
print(f"  Hybrid: std={hybrid_metrics['std_pnl']:.4f}, CVaR95={hybrid_metrics['cvar_95']:.4f}, vol={hybrid_metrics['trading_volume']:.4f}")


# Comparison
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"{'Metric':<15} {'LSTM':>12} {'Hybrid':>12} {'Improvement':>15}")
print("-"*60)

cvar_imp = (lstm_metrics['cvar_95'] - hybrid_metrics['cvar_95']) / lstm_metrics['cvar_95'] * 100
std_ratio = hybrid_metrics['std_pnl'] / lstm_metrics['std_pnl']
vol_ratio = hybrid_metrics['trading_volume'] / lstm_metrics['volume']

print(f"{'Std P&L':<15} {lstm_metrics['std_pnl']:>12.4f} {hybrid_metrics['std_pnl']:>12.4f} {std_ratio:>14.2f}x")
print(f"{'CVaR 95%':<15} {lstm_metrics['cvar_95']:>12.4f} {hybrid_metrics['cvar_95']:>12.4f} {cvar_imp:>14.2f}%")
print(f"{'Entropic':<15} {lstm_metrics['entropic']:>12.4f} {hybrid_metrics['entropic_risk']:>12.4f}")
print(f"{'Volume':<15} {lstm_metrics['volume']:>12.4f} {hybrid_metrics['trading_volume']:>12.4f} {vol_ratio:>14.2f}x")
print(f"{'Smoothness':<15} {lstm_metrics['smooth']:>12.4f} {hybrid_metrics['delta_smoothness']:>12.4f}")

print("\n" + "="*60)
if cvar_imp > 5 and std_ratio < 1.1 and vol_ratio < 1.1:
    print("✓ SUCCESS: Hybrid beats LSTM on tail risk!")
elif cvar_imp > 0:
    print("~ PARTIAL: Hybrid improves CVaR but not significantly")
else:
    print("✗ FAIL: Hybrid does not improve tail risk")
print("="*60)
