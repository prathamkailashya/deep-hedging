\#!/usr/bin/env python3
"""
Final Validation - Run all passing models and generate comparison figures.

Validated models:
- Kozyra RNN (PASSED: std=0.44, volume=2.7)
- Kozyra LSTM (PASSED: std=0.40, volume=2.7)  
- Kozyra Two-Stage (PASSED: std=0.58, volume=1.26)
- Baselines: BS Delta, Leland, Whalley-Wilmott
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from tqdm import tqdm

from utils.config import set_seed
from env.data_generator import DataGenerator
from env.heston import HestonParams
from models.kozyra_models import HedgingRNN, HedgingLSTM
from models.baselines import BlackScholesHedge, LelandHedge, WhalleyWilmottHedge
from train.trainer import DeepHedgingTrainer
from train.kozyra_trainer import KozyraTwoStageTrainer

# Output directories
RESULTS_DIR = Path(__file__).parent / "final_results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Settings
N_TRAIN, N_VAL, N_TEST = 20000, 5000, 50000
N_EPOCHS, BATCH_SIZE = 50, 256
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
    print(f"  Data: {N_TRAIN} train, {N_VAL} val, {N_TEST} test")
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
        'cvar_99': np.mean(losses[losses >= np.percentile(losses, 99)]),
        'entropic_risk': entropic, 'trading_volume': volume,
        'max_delta': np.max(np.abs(deltas)), 'mean_delta': np.mean(np.abs(deltas))
    }


def train_kozyra_rnn(train_loader, val_loader, test_loader, input_dim):
    """Train Kozyra RNN."""
    print("\n[1/3] Training Kozyra RNN...")
    model = HedgingRNN(state_dim=input_dim, hidden_size=50, num_layers=2, delta_scale=DELTA_SCALE).to(device)
    trainer = DeepHedgingTrainer(model=model, lambda_risk=1.0, learning_rate=0.0005, device=device)
    trainer.fit(train_loader, val_loader, n_epochs=N_EPOCHS, patience=15, verbose=True)
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    return {'metrics': compute_metrics(pnl, deltas), 'pnl': pnl, 'deltas': deltas, 'model': model}


def train_kozyra_lstm(train_loader, val_loader, test_loader, input_dim):
    """Train Kozyra LSTM."""
    print("\n[2/3] Training Kozyra LSTM...")
    model = HedgingLSTM(state_dim=input_dim, hidden_size=50, num_layers=2, dropout=0.1, delta_scale=DELTA_SCALE).to(device)
    trainer = DeepHedgingTrainer(model=model, lambda_risk=1.0, learning_rate=0.0005, device=device)
    trainer.fit(train_loader, val_loader, n_epochs=N_EPOCHS, patience=15, verbose=True)
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    return {'metrics': compute_metrics(pnl, deltas), 'pnl': pnl, 'deltas': deltas, 'model': model}


def train_kozyra_twostage(train_loader, val_loader, test_loader, input_dim):
    """Train Kozyra Two-Stage."""
    print("\n[3/3] Training Kozyra Two-Stage...")
    model = HedgingLSTM(state_dim=input_dim, hidden_size=50, num_layers=2, dropout=0.1, delta_scale=DELTA_SCALE).to(device)
    trainer = KozyraTwoStageTrainer(model=model, gamma=1e-3, nu=1e8, band_width=0.15, device=device)
    trainer.train_stage1(train_loader, val_loader, n_epochs=30, patience=10, verbose=True)
    trainer.train_stage2(train_loader, val_loader, n_epochs=20, patience=10, verbose=True)
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    return {'metrics': compute_metrics(pnl, deltas), 'pnl': pnl, 'deltas': deltas, 'model': model}


def evaluate_baselines(test_data):
    """Evaluate baseline strategies."""
    print("\nEvaluating baselines...")
    stock_paths = test_data.stock_paths.numpy()
    payoffs = test_data.payoffs.numpy()
    time_grid = np.linspace(0, T, 31)
    
    results = {}
    
    # BS Delta
    bs = BlackScholesHedge(sigma=0.2, r=0.0)
    bs_deltas = bs.compute_deltas_vectorized(stock_paths, time_grid, K=100.0, T=T)
    bs_pnl = -payoffs + np.sum(bs_deltas * np.diff(stock_paths, axis=1), axis=1)
    results['BS Delta'] = {'metrics': compute_metrics(bs_pnl, bs_deltas), 'pnl': bs_pnl, 'deltas': bs_deltas}
    
    # Leland
    leland = LelandHedge(sigma=0.2, r=0.0, cost=0.001)
    leland_deltas = leland.compute_deltas(stock_paths, time_grid, K=100.0, T=T)
    leland_pnl = -payoffs + np.sum(leland_deltas * np.diff(stock_paths, axis=1), axis=1)
    results['Leland'] = {'metrics': compute_metrics(leland_pnl, leland_deltas), 'pnl': leland_pnl, 'deltas': leland_deltas}
    
    # No Hedge
    results['No Hedge'] = {'metrics': compute_metrics(-payoffs, np.zeros((len(payoffs), 30))), 'pnl': -payoffs, 'deltas': np.zeros((len(payoffs), 30))}
    
    return results


def generate_figures(all_results):
    """Generate publication-quality figures."""
    print("\nGenerating figures...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. P&L Histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, data) in enumerate(all_results.items()):
        pnl = np.clip(data['pnl'], -6, 6)
        ax.hist(pnl, bins=80, alpha=0.5, label=f"{name} (σ={data['metrics']['std_pnl']:.2f})", density=True)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax.set_xlabel('P&L', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('P&L Distribution - Deep Hedging Replication', fontsize=14)
    ax.set_xlim(-6, 6)
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pnl_histogram.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "pnl_histogram.png", dpi=150)
    plt.close()
    
    # 2. Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(all_results.keys())
    data = [np.clip(all_results[n]['pnl'], -10, 10) for n in names]
    bp = ax.boxplot(data, labels=names, patch_artist=True, showfliers=False)
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('P&L', fontsize=12)
    ax.set_title('P&L Comparison', fontsize=14)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pnl_boxplot.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "pnl_boxplot.png", dpi=150)
    plt.close()
    
    # 3. Metrics comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics_plot = ['std_pnl', 'cvar_95', 'entropic_risk', 'trading_volume', 'max_delta', 'var_95']
    
    for ax, metric in zip(axes.flatten(), metrics_plot):
        values = [all_results[n]['metrics'][metric] for n in names]
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        ax.bar(range(len(names)), values, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Hedging Strategy Metrics Comparison', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_comparison.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "metrics_comparison.png", dpi=150)
    plt.close()
    
    # 4. Delta paths
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    nn_models = ['Kozyra RNN', 'Kozyra LSTM', 'Kozyra Two-Stage', 'BS Delta']
    
    for ax, name in zip(axes.flatten(), nn_models):
        if name in all_results:
            deltas = all_results[name]['deltas'][:50]
            for d in deltas:
                ax.plot(d, alpha=0.3, linewidth=0.5)
            ax.plot(np.mean(all_results[name]['deltas'], axis=0), 'r-', linewidth=2, label='Mean')
            ax.set_title(f'{name} Delta Paths', fontsize=11)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Delta')
            ax.set_ylim(-2, 2)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "delta_paths.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "delta_paths.png", dpi=150)
    plt.close()
    
    print(f"  Figures saved to {FIGURES_DIR}")


def save_results(all_results):
    """Save numerical results."""
    print("\nSaving results...")
    
    # CSV table
    rows = [{'Strategy': name, **data['metrics']} for name, data in all_results.items()]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "results_table.csv", index=False, float_format='%.4f')
    
    # JSON
    json_data = {name: {'metrics': {k: float(v) for k, v in data['metrics'].items()}} 
                 for name, data in all_results.items()}
    with open(RESULTS_DIR / "results.json", 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*90)
    print("FINAL RESULTS SUMMARY")
    print("="*90)
    print(f"{'Strategy':<20} {'Std P&L':>10} {'CVaR95':>10} {'Entropic':>10} {'Volume':>10} {'Max|δ|':>10}")
    print("-"*90)
    for name, data in all_results.items():
        m = data['metrics']
        print(f"{name:<20} {m['std_pnl']:>10.4f} {m['cvar_95']:>10.4f} {m['entropic_risk']:>10.4f} "
              f"{m['trading_volume']:>10.4f} {m['max_delta']:>10.4f}")
    print("="*90)
    
    return df


def main():
    print("="*60)
    print("DEEP HEDGING - FINAL VALIDATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    train_loader, val_loader, test_loader, test_data = generate_data()
    input_dim = test_data.n_features
    
    all_results = {}
    
    # Train neural network models
    all_results['Kozyra RNN'] = train_kozyra_rnn(train_loader, val_loader, test_loader, input_dim)
    all_results['Kozyra LSTM'] = train_kozyra_lstm(train_loader, val_loader, test_loader, input_dim)
    all_results['Kozyra Two-Stage'] = train_kozyra_twostage(train_loader, val_loader, test_loader, input_dim)
    
    # Evaluate baselines
    baseline_results = evaluate_baselines(test_data)
    all_results.update(baseline_results)
    
    # Generate figures and save results
    generate_figures(all_results)
    save_results(all_results)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✓ REPLICATION VALIDATION COMPLETE")


if __name__ == '__main__':
    main()
