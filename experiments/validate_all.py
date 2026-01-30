#!/usr/bin/env python3
"""
Comprehensive Validation Script for Deep Hedging Research Pipeline.

Runs all experiments with optimized settings and generates:
- Comparison graphs (PDF format)
- Numerical results (JSON and CSV)
- Statistical tests and confidence intervals
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

from utils.config import Config, set_seed
from utils.statistics import compute_metrics, bootstrap_ci, paired_ttest
from env.data_generator import DataGenerator
from env.heston import HestonParams
from models.deep_hedging import DeepHedgingModel
from models.kozyra_models import HedgingRNN, HedgingLSTM
from models.baselines import BlackScholesHedge, LelandHedge, WhalleyWilmottHedge
from train.trainer import DeepHedgingTrainer
from train.losses import HedgingLoss

# Create output directories
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Validation settings (smaller for faster runs)
N_TRAIN = 10000
N_VAL = 2000
N_TEST = 20000
N_EPOCHS = 30
BATCH_SIZE = 256


def setup():
    """Setup experiment configuration."""
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    return device


def generate_data():
    """Generate Heston model data for experiments."""
    print("\n" + "="*60)
    print("GENERATING DATA")
    print("="*60)
    
    heston_params = HestonParams(
        S0=100.0, v0=0.04, r=0.0,
        kappa=1.0, theta=0.04, sigma=0.2, rho=-0.7
    )
    
    generator = DataGenerator(
        n_steps=30,
        T=30/365,
        S0=100.0,
        K=100.0,
        r=0.0,
        cost_multiplier=0.0,
        model_type='heston',
        heston_params=heston_params
    )
    
    print(f"Generating {N_TRAIN} train, {N_VAL} val, {N_TEST} test samples...")
    train_data, val_data, test_data = generator.generate_train_val_test(
        n_train=N_TRAIN,
        n_val=N_VAL,
        n_test=N_TEST,
        base_seed=42
    )
    
    train_loader, val_loader, test_loader = generator.get_dataloaders(
        train_data, val_data, test_data,
        batch_size=BATCH_SIZE
    )
    
    print(f"✓ Data generated: features shape = {train_data.features.shape}")
    
    return train_loader, val_loader, test_loader, test_data, generator


def train_deep_hedging(train_loader, val_loader, test_loader, input_dim, device):
    """Train and evaluate Buehler et al. Deep Hedging model."""
    print("\n" + "="*60)
    print("DEEP HEDGING MODEL (Buehler et al.)")
    print("="*60)
    
    model = DeepHedgingModel(
        input_dim=input_dim,
        n_steps=30,
        lambda_risk=1.0,
        share_weights=False
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = DeepHedgingTrainer(
        model=model,
        lambda_risk=1.0,
        cost_multiplier=0.0,
        learning_rate=0.005,
        device=device
    )
    
    print("Training...")
    history = trainer.fit(
        train_loader, val_loader,
        n_epochs=N_EPOCHS,
        patience=10,
        verbose=True
    )
    
    print("Evaluating...")
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    
    print("\nResults:")
    for k, v in list(metrics.items())[:8]:
        print(f"  {k}: {v:.4f}")
    
    return {
        'name': 'Deep Hedging',
        'model': model,
        'history': history,
        'metrics': metrics,
        'pnl': pnl,
        'deltas': deltas
    }


def train_kozyra_rnn(train_loader, val_loader, test_loader, input_dim, device):
    """Train and evaluate Kozyra RNN model."""
    print("\n" + "="*60)
    print("KOZYRA RNN MODEL")
    print("="*60)
    
    model = HedgingRNN(
        state_dim=input_dim,
        hidden_size=50,
        num_layers=2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = DeepHedgingTrainer(
        model=model,
        lambda_risk=1.0,
        cost_multiplier=0.0,
        learning_rate=0.0005,
        device=device
    )
    
    print("Training...")
    history = trainer.fit(
        train_loader, val_loader,
        n_epochs=N_EPOCHS,
        patience=10,
        verbose=True
    )
    
    print("Evaluating...")
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    
    print("\nResults:")
    for k, v in list(metrics.items())[:8]:
        print(f"  {k}: {v:.4f}")
    
    return {
        'name': 'Kozyra RNN',
        'model': model,
        'history': history,
        'metrics': metrics,
        'pnl': pnl,
        'deltas': deltas
    }


def train_kozyra_lstm(train_loader, val_loader, test_loader, input_dim, device):
    """Train and evaluate Kozyra LSTM model."""
    print("\n" + "="*60)
    print("KOZYRA LSTM MODEL")
    print("="*60)
    
    model = HedgingLSTM(
        state_dim=input_dim,
        hidden_size=50,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = DeepHedgingTrainer(
        model=model,
        lambda_risk=1.0,
        cost_multiplier=0.0,
        learning_rate=0.0005,
        device=device
    )
    
    print("Training...")
    history = trainer.fit(
        train_loader, val_loader,
        n_epochs=N_EPOCHS,
        patience=10,
        verbose=True
    )
    
    print("Evaluating...")
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    
    print("\nResults:")
    for k, v in list(metrics.items())[:8]:
        print(f"  {k}: {v:.4f}")
    
    return {
        'name': 'Kozyra LSTM',
        'model': model,
        'history': history,
        'metrics': metrics,
        'pnl': pnl,
        'deltas': deltas
    }


def evaluate_baselines(test_data):
    """Evaluate baseline hedging strategies."""
    print("\n" + "="*60)
    print("BASELINE STRATEGIES")
    print("="*60)
    
    stock_paths = test_data.stock_paths.numpy()
    payoffs = test_data.payoffs.numpy()
    T = 30/365
    K = 100.0
    time_grid = np.linspace(0, T, 31)
    
    results = {}
    
    # Black-Scholes Delta
    print("\nBlack-Scholes Delta Hedge...")
    bs_hedge = BlackScholesHedge(sigma=0.2, r=0.0)
    bs_deltas = bs_hedge.compute_deltas_vectorized(stock_paths, time_grid, K, T)
    bs_pnl = compute_pnl(bs_deltas, stock_paths, payoffs)
    bs_metrics = compute_metrics(bs_pnl, lambda_risk=1.0)
    results['BS Delta'] = {'metrics': bs_metrics, 'pnl': bs_pnl, 'deltas': bs_deltas}
    print(f"  Mean P&L: {bs_metrics['mean_pnl']:.4f}, Std: {bs_metrics['std_pnl']:.4f}")
    
    # Leland
    print("\nLeland Hedge...")
    leland_hedge = LelandHedge(sigma=0.2, r=0.0, cost=0.001)
    leland_deltas = leland_hedge.compute_deltas(stock_paths, time_grid, K, T)
    leland_pnl = compute_pnl(leland_deltas, stock_paths, payoffs)
    leland_metrics = compute_metrics(leland_pnl, lambda_risk=1.0)
    results['Leland'] = {'metrics': leland_metrics, 'pnl': leland_pnl, 'deltas': leland_deltas}
    print(f"  Mean P&L: {leland_metrics['mean_pnl']:.4f}, Std: {leland_metrics['std_pnl']:.4f}")
    
    # Whalley-Wilmott
    print("\nWhalley-Wilmott Hedge...")
    ww_hedge = WhalleyWilmottHedge(sigma=0.2, r=0.0, cost=0.001, risk_aversion=1.0)
    ww_deltas = ww_hedge.compute_deltas(stock_paths, time_grid, K, T)
    ww_pnl = compute_pnl(ww_deltas, stock_paths, payoffs)
    ww_metrics = compute_metrics(ww_pnl, lambda_risk=1.0)
    results['Whalley-Wilmott'] = {'metrics': ww_metrics, 'pnl': ww_pnl, 'deltas': ww_deltas}
    print(f"  Mean P&L: {ww_metrics['mean_pnl']:.4f}, Std: {ww_metrics['std_pnl']:.4f}")
    
    # No Hedge
    print("\nNo Hedge (benchmark)...")
    no_hedge_pnl = -payoffs
    no_hedge_metrics = compute_metrics(no_hedge_pnl, lambda_risk=1.0)
    results['No Hedge'] = {'metrics': no_hedge_metrics, 'pnl': no_hedge_pnl, 'deltas': np.zeros_like(bs_deltas)}
    print(f"  Mean P&L: {no_hedge_metrics['mean_pnl']:.4f}, Std: {no_hedge_metrics['std_pnl']:.4f}")
    
    return results


def compute_pnl(deltas, stock_paths, payoffs, cost_mult=0.0):
    """Compute P&L from hedging strategy."""
    price_changes = np.diff(stock_paths, axis=1)
    hedging_gains = np.sum(deltas * price_changes, axis=1)
    
    # Transaction costs
    deltas_ext = np.concatenate([
        np.zeros((deltas.shape[0], 1)),
        deltas,
        np.zeros((deltas.shape[0], 1))
    ], axis=1)
    delta_changes = np.abs(np.diff(deltas_ext, axis=1))
    tc = cost_mult * np.sum(delta_changes, axis=1)
    
    pnl = -payoffs + hedging_gains - tc
    return pnl


def generate_figures(all_results):
    """Generate and save all comparison figures."""
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)
    
    # Collect P&L data
    pnl_dict = {name: data['pnl'] for name, data in all_results.items()}
    
    # 1. P&L Histogram
    print("Creating P&L histogram...")
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(pnl_dict)))
    
    for (name, pnl), color in zip(pnl_dict.items(), colors):
        ax.hist(pnl, bins=50, alpha=0.5, label=f"{name} (μ={np.mean(pnl):.2f})", color=color, density=True)
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax.set_xlabel('P&L', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('P&L Distribution Comparison', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pnl_histogram.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "pnl_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved pnl_histogram.pdf")
    
    # 2. Boxplot
    print("Creating P&L boxplot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(pnl_dict.keys())
    data = [pnl_dict[n] for n in names]
    
    bp = ax.boxplot(data, labels=names, patch_artist=True, showfliers=False)
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(names)+1), means, marker='D', color='red', s=80, zorder=5, label='Mean')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('P&L', fontsize=12)
    ax.set_title('P&L Comparison (Boxplot)', fontsize=14)
    ax.legend()
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pnl_boxplot.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "pnl_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved pnl_boxplot.pdf")
    
    # 3. Metrics Bar Chart
    print("Creating metrics comparison...")
    metrics_to_plot = ['mean_pnl', 'std_pnl', 'var_95', 'cvar_95']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    for ax, metric in zip(axes, metrics_to_plot):
        values = []
        names_plot = []
        for name, data in all_results.items():
            if 'metrics' in data:
                values.append(data['metrics'].get(metric, 0))
                names_plot.append(name)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(names_plot)))
        bars = ax.bar(range(len(names_plot)), values, color=colors)
        ax.set_xticks(range(len(names_plot)))
        ax.set_xticklabels(names_plot, rotation=45, ha='right', fontsize=9)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    plt.suptitle('Hedging Strategy Metrics Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_comparison.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved metrics_comparison.pdf")
    
    # 4. Learning Curves (for neural network models)
    print("Creating learning curves...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for name, data in all_results.items():
        if 'history' in data and data['history']:
            history = data['history']
            if 'train_loss' in history:
                axes[0].plot(history['train_loss'], label=f'{name} (train)')
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], '--', label=f'{name} (val)')
            if 'val_mean_pnl' in history:
                axes[1].plot(history['val_mean_pnl'], label=name)
    
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
    fig.savefig(FIGURES_DIR / "learning_curves.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved learning_curves.pdf")
    
    # 5. Delta comparison (sample paths)
    print("Creating delta paths comparison...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    n_sample = 5
    indices = np.random.choice(len(all_results['BS Delta']['deltas']), n_sample, replace=False)
    
    ax1 = axes[0]
    for name, data in all_results.items():
        if 'deltas' in data:
            deltas = data['deltas']
            for i, idx in enumerate(indices):
                label = name if i == 0 else None
                ax1.plot(deltas[idx], alpha=0.5, label=label)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Delta')
    ax1.set_title('Hedging Positions (Sample Paths)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Mean delta over time
    ax2 = axes[1]
    for name, data in all_results.items():
        if 'deltas' in data:
            mean_delta = np.mean(data['deltas'], axis=0)
            std_delta = np.std(data['deltas'], axis=0)
            x = np.arange(len(mean_delta))
            ax2.plot(x, mean_delta, label=name)
            ax2.fill_between(x, mean_delta - std_delta, mean_delta + std_delta, alpha=0.2)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Mean Delta')
    ax2.set_title('Average Hedging Position (±1 Std)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "delta_comparison.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "delta_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved delta_comparison.pdf")
    
    print(f"\nAll figures saved to: {FIGURES_DIR}")


def save_results(all_results):
    """Save numerical results to JSON and CSV."""
    print("\n" + "="*60)
    print("SAVING NUMERICAL RESULTS")
    print("="*60)
    
    # Prepare results table
    results_table = []
    for name, data in all_results.items():
        if 'metrics' in data:
            row = {'Strategy': name}
            row.update(data['metrics'])
            results_table.append(row)
    
    # Save to CSV
    df = pd.DataFrame(results_table)
    csv_path = RESULTS_DIR / "results_table.csv"
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"✓ Saved results_table.csv")
    
    # Save to JSON
    json_results = {}
    for name, data in all_results.items():
        json_results[name] = {
            'metrics': {k: float(v) for k, v in data.get('metrics', {}).items()},
            'pnl_stats': {
                'mean': float(np.mean(data['pnl'])),
                'std': float(np.std(data['pnl'])),
                'min': float(np.min(data['pnl'])),
                'max': float(np.max(data['pnl'])),
                'median': float(np.median(data['pnl']))
            }
        }
    
    json_path = RESULTS_DIR / "results.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ Saved results.json")
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Strategy':<20} {'Mean P&L':>12} {'Std P&L':>12} {'VaR95':>12} {'CVaR95':>12} {'Entropic':>12}")
    print("-"*80)
    
    for name, data in all_results.items():
        m = data.get('metrics', {})
        print(f"{name:<20} {m.get('mean_pnl', 0):>12.4f} {m.get('std_pnl', 0):>12.4f} "
              f"{m.get('var_95', 0):>12.4f} {m.get('cvar_95', 0):>12.4f} {m.get('entropic_risk', 0):>12.4f}")
    
    print("="*80)
    
    return df


def run_statistical_tests(all_results):
    """Run statistical comparison tests."""
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)
    
    # Compare all strategies to BS Delta baseline
    baseline_name = 'BS Delta'
    baseline_pnl = all_results[baseline_name]['pnl']
    
    tests = []
    for name, data in all_results.items():
        if name == baseline_name:
            continue
        
        pnl = data['pnl']
        
        # Paired t-test
        t_stat, p_val = paired_ttest(pnl, baseline_pnl)
        
        # Mean difference with bootstrap CI
        diff = pnl - baseline_pnl
        mean_diff, ci_lower, ci_upper = bootstrap_ci(diff, np.mean, n_bootstrap=1000)
        
        tests.append({
            'Strategy': name,
            'vs': baseline_name,
            'Mean Diff': mean_diff,
            'CI Lower': ci_lower,
            'CI Upper': ci_upper,
            't-statistic': t_stat,
            'p-value': p_val,
            'Significant (5%)': p_val < 0.05
        })
        
        sig = "✓" if p_val < 0.05 else "✗"
        print(f"{name} vs {baseline_name}:")
        print(f"  Mean Diff: {mean_diff:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  t-stat: {t_stat:.4f}, p-value: {p_val:.6f} {sig}")
    
    # Save tests
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False, float_format='%.6f')
    print(f"\n✓ Saved statistical_tests.csv")
    
    return tests_df


def main():
    """Run complete validation pipeline."""
    print("="*60)
    print("DEEP HEDGING RESEARCH PIPELINE - FULL VALIDATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
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
    
    # Evaluate baselines
    baseline_results = evaluate_baselines(test_data)
    all_results.update(baseline_results)
    
    # Generate figures
    generate_figures(all_results)
    
    # Save numerical results
    results_df = save_results(all_results)
    
    # Statistical tests
    tests_df = run_statistical_tests(all_results)
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Results: {RESULTS_DIR / 'results.json'}")
    print(f"  CSV: {RESULTS_DIR / 'results_table.csv'}")
    print(f"  Tests: {RESULTS_DIR / 'statistical_tests.csv'}")


if __name__ == '__main__':
    main()
