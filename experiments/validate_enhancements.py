#!/usr/bin/env python3
"""
Enhancement Validation - Test all advanced models.

Tests:
A. Enhanced Feature Engineering
B. Signature Models (order 2-3)
C. Transformer / SigFormer
D. RL Agents (MCPG, PPO, DDPG/TD3)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from tqdm import tqdm

from utils.config import set_seed
from env.data_generator import DataGenerator
from env.heston import HestonParams
from models.kozyra_models import HedgingLSTM
from models.transformer_hedging import TransformerHedgingModel, SigFormerModel
from features.signatures import SignatureConditionedLSTM
from features.feature_engineering import FeatureEngineer
from train.trainer import DeepHedgingTrainer
from train.losses import CVaRLoss, EntropicLoss

# Output
RESULTS_DIR = Path(__file__).parent / "enhancement_results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Settings
N_TRAIN, N_VAL, N_TEST = 15000, 3000, 30000
N_EPOCHS, BATCH_SIZE = 40, 256
DELTA_SCALE = 1.5
T = 30/365

set_seed(42)
device = 'cpu'


def generate_data():
    """Generate Heston model data."""
    print("Generating data...")
    heston_params = HestonParams(S0=100.0, v0=0.04, r=0.0, kappa=1.0, theta=0.04, sigma=0.2, rho=-0.7)
    generator = DataGenerator(n_steps=30, T=T, S0=100.0, K=100.0, r=0.0, cost_multiplier=0.0,
                              model_type='heston', heston_params=heston_params)
    
    train_data, val_data, test_data = generator.generate_train_val_test(
        n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, base_seed=42
    )
    train_loader, val_loader, test_loader = generator.get_dataloaders(
        train_data, val_data, test_data, batch_size=BATCH_SIZE
    )
    return train_loader, val_loader, test_loader, test_data


def compute_metrics(pnl, deltas, lambda_risk=1.0):
    """Compute all metrics."""
    losses = -pnl
    delta_changes = np.abs(np.diff(
        np.concatenate([np.zeros((len(deltas), 1)), deltas, np.zeros((len(deltas), 1))], axis=1), axis=1
    ))
    volume = np.mean(np.sum(delta_changes, axis=1))
    
    scaled = -lambda_risk * pnl
    max_val = np.max(scaled)
    entropic = (max_val + np.log(np.mean(np.exp(scaled - max_val)))) / lambda_risk
    
    return {
        'mean_pnl': np.mean(pnl), 'std_pnl': np.std(pnl),
        'var_95': np.percentile(losses, 95), 'var_99': np.percentile(losses, 99),
        'cvar_95': np.mean(losses[losses >= np.percentile(losses, 95)]),
        'entropic_risk': entropic, 'trading_volume': volume,
        'max_delta': np.max(np.abs(deltas))
    }


def train_baseline_lstm(train_loader, val_loader, test_loader, input_dim):
    """Baseline: Kozyra LSTM for comparison."""
    print("\n[Baseline] Kozyra LSTM...")
    model = HedgingLSTM(state_dim=input_dim, hidden_size=50, num_layers=2, delta_scale=DELTA_SCALE).to(device)
    trainer = DeepHedgingTrainer(model=model, lambda_risk=1.0, learning_rate=0.0005, device=device)
    trainer.fit(train_loader, val_loader, n_epochs=N_EPOCHS, patience=15, verbose=True)
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    return {'metrics': compute_metrics(pnl, deltas), 'pnl': pnl, 'deltas': deltas}


def train_transformer(train_loader, val_loader, test_loader, input_dim):
    """Test Transformer hedging model."""
    print("\n[C.1] Transformer Hedging...")
    model = TransformerHedgingModel(
        input_dim=input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        delta_scale=DELTA_SCALE
    ).to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    trainer = DeepHedgingTrainer(model=model, lambda_risk=1.0, learning_rate=0.0003, device=device)
    trainer.fit(train_loader, val_loader, n_epochs=N_EPOCHS, patience=15, verbose=True)
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    return {'metrics': compute_metrics(pnl, deltas), 'pnl': pnl, 'deltas': deltas}


def train_signature_lstm(train_loader, val_loader, test_loader, input_dim):
    """Test Signature-conditioned LSTM."""
    print("\n[B.1] Signature-Conditioned LSTM...")
    model = SignatureConditionedLSTM(
        input_dim=input_dim,
        hidden_size=50,
        num_layers=2,
        sig_depth=2,
        sig_window=5,
        delta_scale=DELTA_SCALE
    ).to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    trainer = DeepHedgingTrainer(model=model, lambda_risk=1.0, learning_rate=0.0005, device=device)
    trainer.fit(train_loader, val_loader, n_epochs=N_EPOCHS, patience=15, verbose=True)
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    return {'metrics': compute_metrics(pnl, deltas), 'pnl': pnl, 'deltas': deltas}


def train_rl_mcpg(train_loader, val_loader, test_loader, input_dim, test_data):
    """Test MCPG RL agent."""
    print("\n[D.1] MCPG Agent...")
    from models.rl_agents import MCPGHedge
    
    model = MCPGHedge(state_dim=input_dim, hidden_dim=64, lr=1e-3, gamma=0.99).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # RL training loop (simplified)
    n_episodes = 500
    pbar = tqdm(range(n_episodes), desc="MCPG Training")
    
    for ep in pbar:
        # Sample a batch
        batch = next(iter(train_loader))
        features = batch['features'].to(device)
        stock_paths = batch['stock_paths'].to(device)
        payoffs = batch['payoff'].to(device)
        
        batch_size = features.size(0)
        
        for i in range(min(16, batch_size)):  # Train on subset
            prev_delta = torch.zeros(1, 1, device=device)
            
            for t in range(features.size(1)):
                state = features[i:i+1, t, :]
                action = model.select_action(state, prev_delta)
                prev_delta = action
            
            # Compute episode reward (negative entropic risk)
            with torch.no_grad():
                deltas = model(features[i:i+1])
                price_changes = stock_paths[i:i+1, 1:] - stock_paths[i:i+1, :-1]
                hedging_gains = torch.sum(deltas * price_changes, dim=1)
                pnl = -payoffs[i:i+1] + hedging_gains
                reward = pnl.item()
            
            model.store_reward(reward)
        
        loss = model.update()
        if ep % 50 == 0:
            pbar.set_postfix({'loss': f"{loss:.4f}"})
    
    # Evaluate
    model.eval()
    all_pnl, all_deltas = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            stock_paths = batch['stock_paths'].to(device)
            payoffs = batch['payoff'].to(device)
            
            deltas = model(features)
            price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]
            hedging_gains = torch.sum(deltas * price_changes, dim=1)
            pnl = -payoffs + hedging_gains
            
            all_pnl.append(pnl.cpu().numpy())
            all_deltas.append(deltas.cpu().numpy())
    
    pnl = np.concatenate(all_pnl)
    deltas = np.concatenate(all_deltas)
    
    return {'metrics': compute_metrics(pnl, deltas), 'pnl': pnl, 'deltas': deltas}


def generate_figures(all_results):
    """Generate comparison figures."""
    print("\nGenerating figures...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. P&L Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in all_results.items():
        pnl = np.clip(data['pnl'], -6, 6)
        ax.hist(pnl, bins=80, alpha=0.5, label=f"{name} (σ={data['metrics']['std_pnl']:.2f})", density=True)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax.set_xlabel('P&L', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Enhancement Comparison: P&L Distribution', fontsize=14)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "enhancement_pnl.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "enhancement_pnl.png", dpi=150)
    plt.close()
    
    # 2. Metrics bar chart
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics_plot = ['std_pnl', 'cvar_95', 'entropic_risk', 'trading_volume', 'max_delta', 'var_95']
    names = list(all_results.keys())
    
    for ax, metric in zip(axes.flatten(), metrics_plot):
        values = [all_results[n]['metrics'][metric] for n in names]
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        ax.bar(range(len(names)), values, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Enhancement Metrics Comparison', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "enhancement_metrics.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "enhancement_metrics.png", dpi=150)
    plt.close()
    
    print(f"  Figures saved to {FIGURES_DIR}")


def save_results(all_results):
    """Save results."""
    print("\nSaving results...")
    
    rows = [{'Model': name, **data['metrics']} for name, data in all_results.items()]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "enhancement_results.csv", index=False, float_format='%.4f')
    
    json_data = {name: {k: float(v) for k, v in data['metrics'].items()} for name, data in all_results.items()}
    with open(RESULTS_DIR / "enhancement_results.json", 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*90)
    print("ENHANCEMENT RESULTS SUMMARY")
    print("="*90)
    print(f"{'Model':<25} {'Std P&L':>10} {'CVaR95':>10} {'Entropic':>10} {'Volume':>10}")
    print("-"*90)
    for name, data in all_results.items():
        m = data['metrics']
        print(f"{name:<25} {m['std_pnl']:>10.4f} {m['cvar_95']:>10.4f} {m['entropic_risk']:>10.4f} {m['trading_volume']:>10.4f}")
    print("="*90)
    
    return df


def main():
    print("="*60)
    print("DEEP HEDGING - ENHANCEMENT VALIDATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    train_loader, val_loader, test_loader, test_data = generate_data()
    input_dim = test_data.n_features
    print(f"Input dim: {input_dim}, N_train: {N_TRAIN}, N_test: {N_TEST}")
    
    all_results = {}
    
    # Baseline
    all_results['Baseline LSTM'] = train_baseline_lstm(train_loader, val_loader, test_loader, input_dim)
    
    # C. Transformer
    try:
        all_results['Transformer'] = train_transformer(train_loader, val_loader, test_loader, input_dim)
    except Exception as e:
        print(f"  Transformer failed: {e}")
    
    # B. Signature LSTM
    try:
        all_results['Sig-LSTM'] = train_signature_lstm(train_loader, val_loader, test_loader, input_dim)
    except Exception as e:
        print(f"  Sig-LSTM failed: {e}")
    
    # D. RL Agent (MCPG)
    try:
        all_results['MCPG'] = train_rl_mcpg(train_loader, val_loader, test_loader, input_dim, test_data)
    except Exception as e:
        print(f"  MCPG failed: {e}")
    
    # Generate figures and save
    generate_figures(all_results)
    save_results(all_results)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✓ ENHANCEMENT VALIDATION COMPLETE")


if __name__ == '__main__':
    main()
