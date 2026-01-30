#!/usr/bin/env python3
"""
Quick validation for Kozyra Two-Stage Training only.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import torch
from datetime import datetime

from utils.config import set_seed
from env.data_generator import DataGenerator
from env.heston import HestonParams
from models.kozyra_models import HedgingLSTM
from train.kozyra_trainer import KozyraTwoStageTrainer

# Settings
N_TRAIN = 10000
N_VAL = 2000
N_TEST = 20000
BATCH_SIZE = 256
DELTA_SCALE = 1.5

set_seed(42)
device = 'cpu'

print("="*60)
print("KOZYRA TWO-STAGE TRAINING VALIDATION")
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print("="*60)

# Generate data
print("\nGenerating data...")
heston_params = HestonParams(S0=100.0, v0=0.04, r=0.0, kappa=1.0, theta=0.04, sigma=0.2, rho=-0.7)
generator = DataGenerator(n_steps=30, T=30/365, S0=100.0, K=100.0, r=0.0, cost_multiplier=0.0,
                          model_type='heston', heston_params=heston_params)

train_data, val_data, test_data = generator.generate_train_val_test(
    n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, base_seed=42
)
train_loader, val_loader, test_loader = generator.get_dataloaders(
    train_data, val_data, test_data, batch_size=BATCH_SIZE
)
input_dim = test_data.n_features
print(f"Data ready: {N_TRAIN} train, {N_VAL} val, {N_TEST} test")

# Create model
print("\nCreating model...")
model = HedgingLSTM(
    state_dim=input_dim,
    hidden_size=50,
    num_layers=2,
    dropout=0.1,
    delta_scale=DELTA_SCALE
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create two-stage trainer
trainer = KozyraTwoStageTrainer(
    model=model,
    alpha_cvar=0.95,
    lambda_risk=1.0,
    gamma=1e-3,
    nu=1e8,
    band_width=0.15,
    cost_multiplier=0.0,
    lr_stage1=0.0005,
    lr_stage2=0.0001,
    device=device
)

# Stage 1
print("\n" + "="*60)
print("STAGE 1: CVaR Pretraining")
print("="*60)
trainer.train_stage1(train_loader, val_loader, n_epochs=30, patience=10, verbose=True)

# Stage 2
print("\n" + "="*60)
print("STAGE 2: Transaction Cost Fine-tuning")
print("="*60)
trainer.train_stage2(train_loader, val_loader, n_epochs=20, patience=10, verbose=True)

# Evaluate
print("\n" + "="*60)
print("EVALUATION")
print("="*60)
metrics, pnl, deltas = trainer.evaluate(test_loader)

print(f"\nResults:")
print(f"  Mean P&L:  {metrics['mean_pnl']:.4f}")
print(f"  Std P&L:   {metrics['std_pnl']:.4f}")
print(f"  VaR95:     {metrics['var_95']:.4f}")
print(f"  CVaR95:    {metrics['cvar_95']:.4f}")
print(f"  Entropic:  {metrics['entropic_risk']:.4f}")
print(f"  Volume:    {metrics['total_trading_volume']:.4f}")
print(f"  Max |δ|:   {np.max(np.abs(deltas)):.4f}")

# Check acceptance
passed = True
issues = []
if metrics['std_pnl'] > 2.0:
    issues.append(f"P&L std {metrics['std_pnl']:.2f} > 2.0")
    passed = False
if metrics['total_trading_volume'] > 10.0:
    issues.append(f"Volume {metrics['total_trading_volume']:.2f} > 10.0")
    passed = False
if np.max(np.abs(pnl)) > 15.0:
    issues.append(f"Extreme P&L: {np.max(np.abs(pnl)):.2f}")
    passed = False

print("\n" + "="*60)
if passed:
    print("✓ KOZYRA TWO-STAGE PASSES ACCEPTANCE CRITERIA")
else:
    print("✗ ACCEPTANCE ISSUES:")
    for issue in issues:
        print(f"  - {issue}")
print("="*60)
print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")
