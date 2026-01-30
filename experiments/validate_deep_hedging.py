#!/usr/bin/env python3
"""
Quick validation for Deep Hedging model with CVaR loss (more stable than entropic).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm

from utils.config import set_seed
from env.data_generator import DataGenerator
from env.heston import HestonParams
from models.deep_hedging import DeepHedgingModel
from train.losses import CVaRLoss

# Settings
N_TRAIN = 20000
N_VAL = 5000
N_TEST = 50000
BATCH_SIZE = 256
N_EPOCHS = 50
LEARNING_RATE = 0.0005
DELTA_SCALE = 1.5

set_seed(42)
device = 'cpu'

print("="*60)
print("DEEP HEDGING VALIDATION (CVaR Loss)")
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

# Create model with bounded deltas
print("\nCreating model...")
model = DeepHedgingModel(
    input_dim=input_dim,
    n_steps=30,
    lambda_risk=1.0,
    share_weights=False,  # Separate networks per timestep (Buehler et al.)
    delta_scale=DELTA_SCALE
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Delta scale: {DELTA_SCALE}")

# Stage 1: CVaR loss for stable initial training
cvar_loss = CVaRLoss(alpha=0.95)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)


def compute_pnl(deltas, stock_paths, payoffs):
    """Compute P&L."""
    price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]
    hedging_gains = torch.sum(deltas * price_changes, dim=1)
    return -payoffs + hedging_gains


def train_epoch(model, dataloader, loss_fn, optimizer):
    """Train one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in dataloader:
        features = batch['features'].to(device)
        stock_paths = batch['stock_paths'].to(device)
        payoffs = batch['payoff'].to(device)
        
        optimizer.zero_grad()
        deltas = model(features)
        pnl = compute_pnl(deltas, stock_paths, payoffs)
        loss = loss_fn(pnl)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


@torch.no_grad()
def validate(model, dataloader, loss_fn):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_pnl = []
    all_deltas = []
    n_batches = 0
    
    for batch in dataloader:
        features = batch['features'].to(device)
        stock_paths = batch['stock_paths'].to(device)
        payoffs = batch['payoff'].to(device)
        
        deltas = model(features)
        pnl = compute_pnl(deltas, stock_paths, payoffs)
        loss = loss_fn(pnl)
        
        total_loss += loss.item()
        all_pnl.append(pnl.cpu().numpy())
        all_deltas.append(deltas.cpu().numpy())
        n_batches += 1
    
    pnl = np.concatenate(all_pnl)
    deltas = np.concatenate(all_deltas)
    
    return total_loss / n_batches, np.mean(pnl), np.std(pnl), pnl, deltas


# Two-Stage Training (like Kozyra)
print("\n" + "="*60)
print("STAGE 1: CVaR Pretraining (stable initialization)")
print("="*60)

best_val_loss = float('inf')
patience_counter = 0
patience = 10

pbar = tqdm(range(30), desc="Stage 1")
for epoch in pbar:
    train_loss = train_epoch(model, train_loader, cvar_loss, optimizer)
    val_loss, val_mean, val_std, _, _ = validate(model, val_loader, cvar_loss)
    
    pbar.set_postfix({
        'train': f"{train_loss:.4f}",
        'val': f"{val_loss:.4f}",
        'std': f"{val_std:.4f}"
    })
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Stage 1 early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)

# Stage 2: Fine-tune with entropic loss
print("\n" + "="*60)
print("STAGE 2: Entropic Risk Fine-tuning")
print("="*60)

from train.losses import EntropicLoss
entropic_loss = EntropicLoss(lambda_risk=1.0)
optimizer2 = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=1e-4)

best_val_loss = float('inf')
patience_counter = 0

pbar = tqdm(range(20), desc="Stage 2")
for epoch in pbar:
    train_loss = train_epoch(model, train_loader, entropic_loss, optimizer2)
    val_loss, val_mean, val_std, _, _ = validate(model, val_loader, entropic_loss)
    
    pbar.set_postfix({
        'train': f"{train_loss:.4f}",
        'val': f"{val_loss:.4f}",
        'std': f"{val_std:.4f}"
    })
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Stage 2 early stopping at epoch {epoch}")
            break

model.load_state_dict(best_state)

# Final evaluation
print("\n" + "="*60)
print("EVALUATION")
print("="*60)

_, _, _, pnl, deltas = validate(model, test_loader, entropic_loss)
losses = -pnl

# Entropic risk
lambda_risk = 1.0
scaled = -lambda_risk * pnl
max_val = np.max(scaled)
entropic = (max_val + np.log(np.mean(np.exp(scaled - max_val)))) / lambda_risk

# Trading volume
delta_changes = np.abs(np.diff(
    np.concatenate([np.zeros((len(deltas), 1)), deltas, np.zeros((len(deltas), 1))], axis=1),
    axis=1
))
volume = np.mean(np.sum(delta_changes, axis=1))

print(f"\nResults:")
print(f"  Mean P&L:  {np.mean(pnl):.4f}")
print(f"  Std P&L:   {np.std(pnl):.4f}")
print(f"  VaR95:     {np.percentile(losses, 95):.4f}")
print(f"  CVaR95:    {np.mean(losses[losses >= np.percentile(losses, 95)]):.4f}")
print(f"  Entropic:  {entropic:.4f}")
print(f"  Volume:    {volume:.4f}")
print(f"  Max |δ|:   {np.max(np.abs(deltas)):.4f}")

# Check acceptance
passed = True
issues = []
if np.std(pnl) > 2.0:
    issues.append(f"P&L std {np.std(pnl):.2f} > 2.0")
    passed = False
if volume > 10.0:
    issues.append(f"Volume {volume:.2f} > 10.0")
    passed = False
if np.max(np.abs(pnl)) > 15.0:
    issues.append(f"Extreme P&L: {np.max(np.abs(pnl)):.2f}")
    passed = False

print("\n" + "="*60)
if passed:
    print("✓ DEEP HEDGING PASSES ACCEPTANCE CRITERIA")
else:
    print("✗ ACCEPTANCE ISSUES:")
    for issue in issues:
        print(f"  - {issue}")
print("="*60)
print(f"Finished: {datetime.now().strftime('%H:%M:%S')}")
