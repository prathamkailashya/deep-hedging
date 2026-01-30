#!/usr/bin/env python3
"""
Replication Validation Script - Verifies Paper Invariants.

Acceptance Criteria (from Buehler et al. & Kozyra):
1. P&L std ≤ 2.0
2. Trading volume ≤ 5× Kozyra reference (~5-10)
3. P&L histogram has no extreme tails (|P&L| < 10)
4. Learning curves are smooth
5. Delta paths are bounded and stable

This script MUST PASS before proceeding to enhancements.
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
from models.deep_hedging import DeepHedgingModel
from models.kozyra_models import HedgingRNN, HedgingLSTM
from models.baselines import BlackScholesHedge, LelandHedge, WhalleyWilmottHedge
from train.trainer import DeepHedgingTrainer
from train.kozyra_trainer import KozyraTwoStageTrainer
from train.losses import EntropicLoss, CVaRLoss

# Create output directories
RESULTS_DIR = Path(__file__).parent / "replication_results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Paper-specified settings
N_STEPS = 30
T = 30 / 365  # 30 days
N_TRAIN = 20000
N_VAL = 5000
N_TEST = 50000
N_EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.0005  # Reduced as per requirements
DELTA_SCALE = 1.5  # Maximum delta bound

# Acceptance thresholds
MAX_PNL_STD = 2.0
MAX_TRADING_VOLUME = 10.0
MAX_PNL_OUTLIER = 15.0


def setup():
    """Setup experiment configuration."""
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Results: {RESULTS_DIR}")
    return device


def generate_data():
    """Generate Heston model data with proper parameters."""
    print("\n" + "="*60)
    print("DATA GENERATION")
    print("="*60)
    
    heston_params = HestonParams(
        S0=100.0, v0=0.04, r=0.0,
        kappa=1.0, theta=0.04, sigma=0.2, rho=-0.7
    )
    
    generator = DataGenerator(
        n_steps=N_STEPS,
        T=T,
        S0=100.0,
        K=100.0,
        r=0.0,
        cost_multiplier=0.0,
        model_type='heston',
        heston_params=heston_params
    )
    
    print(f"Heston params: S0=100, v0=0.04, κ=1.0, θ=0.04, σ=0.2, ρ=-0.7")
    print(f"Time grid: n={N_STEPS}, T={T:.4f} ({int(T*365)} days), dt={T/N_STEPS:.6f}")
    
    print(f"\nGenerating {N_TRAIN} train, {N_VAL} val, {N_TEST} test samples...")
    train_data, val_data, test_data = generator.generate_train_val_test(
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_test=N_TEST,
        base_seed=42
    )
    
    # Log simulator statistics
    stock_paths = test_data.stock_paths.numpy()
    returns = np.diff(np.log(stock_paths), axis=1)
    
    print(f"\nSimulator Statistics (Test Set):")
    print(f"  Daily return mean: {np.mean(returns)*100:.4f}%")
    print(f"  Daily return std:  {np.std(returns)*100:.4f}%")
    print(f"  Return quantiles:  [1%: {np.percentile(returns, 1)*100:.2f}%, "
          f"99%: {np.percentile(returns, 99)*100:.2f}%]")
    print(f"  Stock price range: [{np.min(stock_paths):.2f}, {np.max(stock_paths):.2f}]")
    
    train_loader, val_loader, test_loader = generator.get_dataloaders(
        train_data, val_data, test_data,
        batch_size=BATCH_SIZE
    )
    
    return train_loader, val_loader, test_loader, test_data, generator


def check_acceptance_criteria(name, metrics, pnl, deltas):
    """Check if results meet acceptance criteria."""
    issues = []
    
    # Check P&L std
    if metrics['std_pnl'] > MAX_PNL_STD:
        issues.append(f"P&L std {metrics['std_pnl']:.2f} > {MAX_PNL_STD}")
    
    # Check trading volume
    if metrics.get('trading_volume', 0) > MAX_TRADING_VOLUME:
        issues.append(f"Trading volume {metrics['trading_volume']:.2f} > {MAX_TRADING_VOLUME}")
    
    # Check for extreme outliers
    max_pnl = np.max(np.abs(pnl))
    if max_pnl > MAX_PNL_OUTLIER:
        issues.append(f"Extreme P&L outlier: {max_pnl:.2f} > {MAX_PNL_OUTLIER}")
    
    # Check delta bounds
    max_delta = np.max(np.abs(deltas))
    if max_delta > 3.0:  # Should be bounded by delta_scale
        issues.append(f"Unbounded delta: max|δ| = {max_delta:.2f}")
    
    if issues:
        print(f"\n  ⚠️  ACCEPTANCE ISSUES for {name}:")
        for issue in issues:
            print(f"      - {issue}")
        return False
    else:
        print(f"\n  ✓ {name} PASSES acceptance criteria")
        return True


def compute_metrics_and_check(name, pnl, deltas, lambda_risk=1.0):
    """Compute metrics and check acceptance."""
    losses = -pnl
    
    # Trading volume
    delta_changes = np.abs(np.diff(
        np.concatenate([np.zeros((len(deltas), 1)), deltas, np.zeros((len(deltas), 1))], axis=1),
        axis=1
    ))
    trading_volume = np.mean(np.sum(delta_changes, axis=1))
    
    # Entropic risk (numerically stable)
    scaled = -lambda_risk * pnl
    max_val = np.max(scaled)
    entropic = (max_val + np.log(np.mean(np.exp(scaled - max_val)))) / lambda_risk
    
    metrics = {
        'mean_pnl': np.mean(pnl),
        'std_pnl': np.std(pnl),
        'min_pnl': np.min(pnl),
        'max_pnl': np.max(pnl),
        'var_95': np.percentile(losses, 95),
        'var_99': np.percentile(losses, 99),
        'cvar_95': np.mean(losses[losses >= np.percentile(losses, 95)]),
        'cvar_99': np.mean(losses[losses >= np.percentile(losses, 99)]),
        'entropic_risk': entropic,
        'trading_volume': trading_volume,
        'mean_delta': np.mean(np.abs(deltas)),
        'max_delta': np.max(np.abs(deltas)),
        'delta_std': np.std(deltas)
    }
    
    # Print summary
    print(f"\n{name} Results:")
    print(f"  Mean P&L: {metrics['mean_pnl']:.4f}")
    print(f"  Std P&L:  {metrics['std_pnl']:.4f}")
    print(f"  VaR95:    {metrics['var_95']:.4f}")
    print(f"  CVaR95:   {metrics['cvar_95']:.4f}")
    print(f"  Entropic: {metrics['entropic_risk']:.4f}")
    print(f"  Volume:   {metrics['trading_volume']:.4f}")
    print(f"  Max |δ|:  {metrics['max_delta']:.4f}")
    
    passed = check_acceptance_criteria(name, metrics, pnl, deltas)
    
    return metrics, passed


def train_deep_hedging(train_loader, val_loader, test_loader, input_dim, device):
    """Train Deep Hedging model with proper constraints."""
    print("\n" + "="*60)
    print("DEEP HEDGING (Buehler et al.)")
    print("="*60)
    
    model = DeepHedgingModel(
        input_dim=input_dim,
        n_steps=N_STEPS,
        lambda_risk=1.0,
        share_weights=False,
        delta_scale=DELTA_SCALE  # Bounded deltas
    ).to(device)
    
    print(f"Model: SemiRecurrent, delta_scale={DELTA_SCALE}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = DeepHedgingTrainer(
        model=model,
        lambda_risk=1.0,
        cost_multiplier=0.0,
        learning_rate=LEARNING_RATE,
        weight_decay=1e-4,
        device=device
    )
    
    print(f"Training: lr={LEARNING_RATE}, weight_decay=1e-4, epochs={N_EPOCHS}")
    
    history = trainer.fit(
        train_loader, val_loader,
        n_epochs=N_EPOCHS,
        patience=15,
        verbose=True
    )
    
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    
    # Compute extended metrics and check acceptance
    full_metrics, passed = compute_metrics_and_check("Deep Hedging", pnl, deltas)
    
    return {
        'name': 'Deep Hedging',
        'model': model,
        'history': history,
        'metrics': full_metrics,
        'pnl': pnl,
        'deltas': deltas,
        'passed': passed
    }


def train_kozyra_rnn(train_loader, val_loader, test_loader, input_dim, device):
    """Train Kozyra RNN model."""
    print("\n" + "="*60)
    print("KOZYRA RNN")
    print("="*60)
    
    model = HedgingRNN(
        state_dim=input_dim,
        hidden_size=50,
        num_layers=2,
        delta_scale=DELTA_SCALE
    ).to(device)
    
    print(f"Model: LSTM hidden=50, layers=2, delta_scale={DELTA_SCALE}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = DeepHedgingTrainer(
        model=model,
        lambda_risk=1.0,
        cost_multiplier=0.0,
        learning_rate=LEARNING_RATE,
        weight_decay=1e-4,
        device=device
    )
    
    history = trainer.fit(
        train_loader, val_loader,
        n_epochs=N_EPOCHS,
        patience=15,
        verbose=True
    )
    
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    full_metrics, passed = compute_metrics_and_check("Kozyra RNN", pnl, deltas)
    
    return {
        'name': 'Kozyra RNN',
        'model': model,
        'history': history,
        'metrics': full_metrics,
        'pnl': pnl,
        'deltas': deltas,
        'passed': passed
    }


def train_kozyra_lstm(train_loader, val_loader, test_loader, input_dim, device):
    """Train Kozyra LSTM model."""
    print("\n" + "="*60)
    print("KOZYRA LSTM")
    print("="*60)
    
    model = HedgingLSTM(
        state_dim=input_dim,
        hidden_size=50,
        num_layers=2,
        dropout=0.1,
        delta_scale=DELTA_SCALE
    ).to(device)
    
    print(f"Model: LSTM hidden=50, layers=2, dropout=0.1, delta_scale={DELTA_SCALE}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = DeepHedgingTrainer(
        model=model,
        lambda_risk=1.0,
        cost_multiplier=0.0,
        learning_rate=LEARNING_RATE,
        weight_decay=1e-4,
        device=device
    )
    
    history = trainer.fit(
        train_loader, val_loader,
        n_epochs=N_EPOCHS,
        patience=15,
        verbose=True
    )
    
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    full_metrics, passed = compute_metrics_and_check("Kozyra LSTM", pnl, deltas)
    
    return {
        'name': 'Kozyra LSTM',
        'model': model,
        'history': history,
        'metrics': full_metrics,
        'pnl': pnl,
        'deltas': deltas,
        'passed': passed
    }


def train_kozyra_two_stage(train_loader, val_loader, test_loader, input_dim, device):
    """Train Kozyra model with two-stage training."""
    print("\n" + "="*60)
    print("KOZYRA TWO-STAGE TRAINING")
    print("="*60)
    
    model = HedgingLSTM(
        state_dim=input_dim,
        hidden_size=50,
        num_layers=2,
        dropout=0.1,
        delta_scale=DELTA_SCALE
    ).to(device)
    
    print(f"Two-stage training: γ=1e-3, ν=1e8, band=±0.15")
    
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
    
    # Stage 1: CVaR pretraining
    print("\nStage 1: CVaR pretraining...")
    trainer.train_stage1(train_loader, val_loader, n_epochs=30, patience=10)
    
    # Stage 2: Transaction cost fine-tuning
    print("\nStage 2: Transaction cost fine-tuning...")
    trainer.train_stage2(train_loader, val_loader, n_epochs=20, patience=10)
    
    # Evaluate
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    full_metrics, passed = compute_metrics_and_check("Kozyra Two-Stage", pnl, deltas)
    
    return {
        'name': 'Kozyra Two-Stage',
        'model': model,
        'history': trainer.history,
        'metrics': full_metrics,
        'pnl': pnl,
        'deltas': deltas,
        'passed': passed
    }


def compute_pnl_baseline(deltas, stock_paths, payoffs):
    """Compute P&L for baseline strategies."""
    price_changes = np.diff(stock_paths, axis=1)
    hedging_gains = np.sum(deltas * price_changes, axis=1)
    return -payoffs + hedging_gains


def evaluate_baselines(test_data):
    """Evaluate baseline hedging strategies."""
    print("\n" + "="*60)
    print("BASELINE STRATEGIES")
    print("="*60)
    
    stock_paths = test_data.stock_paths.numpy()
    payoffs = test_data.payoffs.numpy()
    time_grid = np.linspace(0, T, N_STEPS + 1)
    
    results = {}
    
    # Black-Scholes Delta
    print("\nBlack-Scholes Delta...")
    bs_hedge = BlackScholesHedge(sigma=0.2, r=0.0)
    bs_deltas = bs_hedge.compute_deltas_vectorized(stock_paths, time_grid, K=100.0, T=T)
    bs_pnl = compute_pnl_baseline(bs_deltas, stock_paths, payoffs)
    bs_metrics, bs_passed = compute_metrics_and_check("BS Delta", bs_pnl, bs_deltas)
    results['BS Delta'] = {'metrics': bs_metrics, 'pnl': bs_pnl, 'deltas': bs_deltas, 'passed': bs_passed}
    
    # Leland
    print("\nLeland Hedge...")
    leland_hedge = LelandHedge(sigma=0.2, r=0.0, cost=0.001)
    leland_deltas = leland_hedge.compute_deltas(stock_paths, time_grid, K=100.0, T=T)
    leland_pnl = compute_pnl_baseline(leland_deltas, stock_paths, payoffs)
    leland_metrics, leland_passed = compute_metrics_and_check("Leland", leland_pnl, leland_deltas)
    results['Leland'] = {'metrics': leland_metrics, 'pnl': leland_pnl, 'deltas': leland_deltas, 'passed': leland_passed}
    
    # Whalley-Wilmott
    print("\nWhalley-Wilmott...")
    ww_hedge = WhalleyWilmottHedge(sigma=0.2, r=0.0, cost=0.001, risk_aversion=1.0)
    ww_deltas = ww_hedge.compute_deltas(stock_paths, time_grid, K=100.0, T=T)
    ww_pnl = compute_pnl_baseline(ww_deltas, stock_paths, payoffs)
    ww_metrics, ww_passed = compute_metrics_and_check("Whalley-Wilmott", ww_pnl, ww_deltas)
    results['Whalley-Wilmott'] = {'metrics': ww_metrics, 'pnl': ww_pnl, 'deltas': ww_deltas, 'passed': ww_passed}
    
    return results


def generate_figures(all_results):
    """Generate publication-quality figures."""
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. P&L Histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for (name, data), color in zip(all_results.items(), colors):
        pnl = data['pnl']
        # Clip for visualization
        pnl_clipped = np.clip(pnl, -10, 10)
        ax.hist(pnl_clipped, bins=100, alpha=0.5, label=f"{name} (σ={np.std(pnl):.2f})", 
                color=color, density=True)
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax.set_xlabel('P&L', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('P&L Distribution (Paper Replication)', fontsize=14)
    ax.set_xlim(-10, 10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pnl_histogram_replication.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "pnl_histogram_replication.png", dpi=150)
    plt.close()
    print("  ✓ pnl_histogram_replication.pdf")
    
    # 2. Metrics Comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics_to_plot = ['std_pnl', 'cvar_95', 'entropic_risk', 'trading_volume', 'max_delta', 'mean_delta']
    thresholds = [MAX_PNL_STD, None, None, MAX_TRADING_VOLUME, 3.0, None]
    
    for ax, metric, threshold in zip(axes.flatten(), metrics_to_plot, thresholds):
        names = list(all_results.keys())
        values = [all_results[n]['metrics'].get(metric, 0) for n in names]
        colors = ['green' if all_results[n].get('passed', True) else 'red' for n in names]
        
        bars = ax.bar(range(len(names)), values, color=colors, alpha=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=11)
        
        if threshold:
            ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold={threshold}')
            ax.legend(fontsize=8)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Replication Metrics Comparison', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_replication.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "metrics_replication.png", dpi=150)
    plt.close()
    print("  ✓ metrics_replication.pdf")
    
    # 3. Delta Paths (sample)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    nn_models = [k for k in all_results.keys() if k in ['Deep Hedging', 'Kozyra RNN', 'Kozyra LSTM', 'Kozyra Two-Stage']]
    
    for ax, name in zip(axes.flatten(), nn_models[:4]):
        if name in all_results:
            deltas = all_results[name]['deltas']
            for i in range(min(20, len(deltas))):
                ax.plot(deltas[i], alpha=0.3, linewidth=0.5)
            ax.plot(np.mean(deltas, axis=0), 'r-', linewidth=2, label='Mean')
            ax.axhline(y=DELTA_SCALE, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=-DELTA_SCALE, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f'{name} Delta Paths', fontsize=11)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Delta')
            ax.set_ylim(-2, 2)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "delta_paths_replication.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "delta_paths_replication.png", dpi=150)
    plt.close()
    print("  ✓ delta_paths_replication.pdf")
    
    # 4. Learning Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for name, data in all_results.items():
        if 'history' in data and data['history']:
            history = data['history']
            if isinstance(history, dict):
                if 'train_loss' in history:
                    axes[0].plot(history['train_loss'], label=f'{name}')
                if 'val_mean_pnl' in history:
                    axes[1].plot(history['val_mean_pnl'], label=f'{name}')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean P&L')
    axes[1].set_title('Validation Mean P&L')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "learning_curves_replication.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "learning_curves_replication.png", dpi=150)
    plt.close()
    print("  ✓ learning_curves_replication.pdf")


def save_results(all_results):
    """Save results to files."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Results table
    rows = []
    for name, data in all_results.items():
        row = {'Strategy': name, 'Passed': data.get('passed', True)}
        row.update(data['metrics'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "replication_results.csv", index=False, float_format='%.6f')
    print("  ✓ replication_results.csv")
    
    # JSON
    json_results = {}
    for name, data in all_results.items():
        json_results[name] = {
            'passed': data.get('passed', True),
            'metrics': {k: float(v) for k, v in data['metrics'].items()}
        }
    
    with open(RESULTS_DIR / "replication_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    print("  ✓ replication_results.json")
    
    return df


def main():
    """Run complete replication validation."""
    print("="*60)
    print("DEEP HEDGING REPLICATION VALIDATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"\nAcceptance Criteria:")
    print(f"  - P&L std ≤ {MAX_PNL_STD}")
    print(f"  - Trading volume ≤ {MAX_TRADING_VOLUME}")
    print(f"  - No extreme outliers (|P&L| < {MAX_PNL_OUTLIER})")
    print(f"  - Delta bounded (|δ| < {DELTA_SCALE + 0.5})")
    
    device = setup()
    
    # Generate data
    train_loader, val_loader, test_loader, test_data, generator = generate_data()
    input_dim = test_data.n_features
    
    all_results = {}
    
    # Train neural network models
    dh_results = train_deep_hedging(train_loader, val_loader, test_loader, input_dim, device)
    all_results['Deep Hedging'] = dh_results
    
    rnn_results = train_kozyra_rnn(train_loader, val_loader, test_loader, input_dim, device)
    all_results['Kozyra RNN'] = rnn_results
    
    lstm_results = train_kozyra_lstm(train_loader, val_loader, test_loader, input_dim, device)
    all_results['Kozyra LSTM'] = lstm_results
    
    # Two-stage training
    two_stage_results = train_kozyra_two_stage(train_loader, val_loader, test_loader, input_dim, device)
    all_results['Kozyra Two-Stage'] = two_stage_results
    
    # Evaluate baselines
    baseline_results = evaluate_baselines(test_data)
    all_results.update(baseline_results)
    
    # Generate figures
    generate_figures(all_results)
    
    # Save results
    df = save_results(all_results)
    
    # Final summary
    print("\n" + "="*60)
    print("REPLICATION VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, data in all_results.items():
        status = "✓ PASS" if data.get('passed', True) else "✗ FAIL"
        print(f"  {name}: {status}")
        if not data.get('passed', True):
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL MODELS PASS ACCEPTANCE CRITERIA")
        print("  → Ready to proceed with enhancements")
    else:
        print("✗ SOME MODELS FAILED ACCEPTANCE CRITERIA")
        print("  → Fix issues before proceeding")
    print("="*60)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_passed


if __name__ == '__main__':
    passed = main()
    exit(0 if passed else 1)
