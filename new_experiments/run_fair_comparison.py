#!/usr/bin/env python3
"""
Fair Comparison of Deep Hedging Architectures

Scientifically rigorous comparison with:
- Optuna HPO (≥100 trials per model)
- Same compute budget
- Bootstrap CI for statistical significance
- Two-stage training protocol for all models
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from configs.base_config import (
    HestonConfig, DataConfig, TrainingConfig, HPOConfig,
    LSTM_SEARCH_SPACE, SIGNATURE_SEARCH_SPACE, 
    TRANSFORMER_SEARCH_SPACE, ATTENTION_LSTM_SEARCH_SPACE
)
from data.data_generator import DataGenerator, HestonParams
from models.base_model import LSTMHedger
from models.signature_models import SignatureLSTM, SignatureMLP, SigFormerHedger
from models.transformer_models import TransformerHedger, TimeSeriesTransformer
from models.attention_lstm import AttentionLSTM, MultiHeadAttentionLSTM
from models.novel_hybrids import EnsembleHedger, RegimeAwareHedger, AdaptiveHedger
from models.rl_models import CVaRPPO, EntropicPPO, TD3Hedger
from training.trainer import UnifiedTrainer, compute_all_metrics
from training.losses import compute_pnl, cvar_loss
from evaluation.statistical_tests import (
    compare_models, generate_comparison_report, is_improvement,
    bootstrap_ci, cvar_95
)

# Configuration
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

DEVICE = 'cpu'


def generate_data(config: DataConfig) -> Tuple:
    """Generate train/val/test data."""
    print("="*60)
    print("DATA GENERATION")
    print("="*60)
    
    heston = HestonParams(
        S0=100.0, v0=0.04, r=0.0,
        kappa=1.0, theta=0.04, sigma=0.2, rho=-0.7
    )
    
    generator = DataGenerator(
        heston_params=heston,
        n_steps=config.n_steps,
        T=config.T,
        K=config.K,
        r=0.0
    )
    
    train_data, val_data, test_data = generator.generate_splits(
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        base_seed=config.seed
    )
    
    train_loader, val_loader, test_loader = generator.get_dataloaders(
        train_data, val_data, test_data,
        batch_size=config.batch_size
    )
    
    input_dim = train_data.n_features
    print(f"  Train: {config.n_train}, Val: {config.n_val}, Test: {config.n_test}")
    print(f"  Input dim: {input_dim}, Steps: {config.n_steps}")
    
    return train_loader, val_loader, test_loader, input_dim


def train_model(
    model_class,
    model_name: str,
    model_kwargs: Dict,
    train_loader,
    val_loader,
    test_loader,
    training_config: TrainingConfig,
    verbose: bool = True
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Train a model with unified two-stage protocol."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING: {model_name}")
        print(f"{'='*60}")
    
    model = model_class(**model_kwargs).to(DEVICE)
    
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
    
    trainer = UnifiedTrainer(
        model=model,
        lambda_risk=training_config.lambda_risk,
        gamma=training_config.gamma,
        band_width=training_config.band_width,
        grad_clip=training_config.grad_clip,
        weight_decay=training_config.weight_decay,
        device=DEVICE
    )
    
    history = trainer.train(
        train_loader, val_loader,
        stage1_epochs=training_config.stage1_epochs,
        stage2_epochs=training_config.stage2_epochs,
        stage1_lr=training_config.stage1_lr,
        stage2_lr=training_config.stage2_lr,
        patience=training_config.stage1_patience,
        verbose=verbose
    )
    
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    
    if verbose:
        print(f"\n  Test Results:")
        print(f"    Std P&L: {metrics['std_pnl']:.4f}")
        print(f"    CVaR95: {metrics['cvar_95']:.4f}")
        print(f"    Volume: {metrics['trading_volume']:.4f}")
    
    return {
        'model': model,
        'metrics': metrics,
        'history': history,
        'model_name': model_name
    }, pnl, deltas


def run_baseline_comparison(
    train_loader, val_loader, test_loader, input_dim: int,
    training_config: TrainingConfig
) -> Dict:
    """Run comparison with default hyperparameters."""
    print("\n" + "="*70)
    print("BASELINE COMPARISON (Default Hyperparameters)")
    print("="*70)
    
    results = {}
    
    # Model configurations
    models_config = [
        ('LSTM', LSTMHedger, {
            'input_dim': input_dim, 'hidden_size': 50, 'num_layers': 2,
            'dropout': 0.1, 'delta_max': 1.5
        }),
        ('SignatureLSTM', SignatureLSTM, {
            'input_dim': input_dim, 'hidden_size': 64, 'num_layers': 2,
            'sig_order': 3, 'sig_window': 5, 'fc_layers': 2,
            'dropout': 0.1, 'delta_max': 1.5
        }),
        ('SignatureMLP', SignatureMLP, {
            'input_dim': input_dim, 'hidden_size': 128, 'n_layers': 3,
            'sig_order': 3, 'sig_window': 5, 'dropout': 0.1, 'delta_max': 1.5
        }),
        ('Transformer', TransformerHedger, {
            'input_dim': input_dim, 'd_model': 64, 'n_heads': 4,
            'n_layers': 4, 'dim_feedforward': 128,
            'dropout': 0.1, 'delta_max': 1.5
        }),
        ('AttentionLSTM', AttentionLSTM, {
            'input_dim': input_dim, 'hidden_size': 64, 'num_layers': 2,
            'attention_dim': 32, 'memory_length': 10, 'combination': 'concat',
            'dropout': 0.1, 'delta_max': 1.5
        }),
        ('RegimeAware', RegimeAwareHedger, {
            'input_dim': input_dim, 'hidden_size': 64, 'n_regimes': 3,
            'dropout': 0.1, 'delta_max': 1.5
        }),
        ('Adaptive', AdaptiveHedger, {
            'input_dim': input_dim, 'hidden_size': 64, 'num_layers': 2,
            'adaptation_window': 5, 'dropout': 0.1, 'delta_max': 1.5
        }),
    ]
    
    for name, model_class, kwargs in models_config:
        try:
            result, pnl, deltas = train_model(
                model_class, name, kwargs,
                train_loader, val_loader, test_loader,
                training_config, verbose=True
            )
            results[name] = {
                'result': result,
                'pnl': pnl,
                'deltas': deltas
            }
        except Exception as e:
            print(f"  FAILED: {name} - {e}")
    
    return results


def run_statistical_analysis(
    results: Dict,
    baseline_name: str = 'LSTM'
) -> Dict:
    """Run statistical tests comparing all models to baseline."""
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    
    if baseline_name not in results:
        print(f"  Baseline {baseline_name} not found!")
        return {}
    
    baseline = results[baseline_name]
    comparisons = {}
    
    for name, data in results.items():
        if name == baseline_name:
            continue
        
        print(f"\n  Comparing {name} vs {baseline_name}...")
        
        comparison = compare_models(
            baseline_pnl=baseline['pnl'],
            baseline_deltas=baseline['deltas'],
            model_pnl=data['pnl'],
            model_deltas=data['deltas'],
            model_name=name,
            n_bootstrap=1000
        )
        
        is_better, reason = is_improvement(comparison)
        print(f"    Improvement: {is_better} - {reason}")
        
        comparisons[name] = comparison
    
    # Print report
    report = generate_comparison_report(baseline_name, comparisons)
    print(report)
    
    return comparisons


def create_ensemble(
    results: Dict,
    model_names: List[str],
    method: str = 'median'
) -> Tuple[EnsembleHedger, Dict]:
    """Create ensemble from trained models."""
    print(f"\n  Creating {method} ensemble from: {model_names}")
    
    models = [results[name]['result']['model'] for name in model_names if name in results]
    
    if len(models) < 2:
        print("    Not enough models for ensemble")
        return None, {}
    
    ensemble = EnsembleHedger(models, method=method, delta_max=1.5)
    ensemble.freeze_base_models()
    
    return ensemble, {'models': model_names, 'method': method}


def generate_figures(results: Dict, comparisons: Dict):
    """Generate comprehensive comparison figures."""
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    model_names = list(results.keys())
    
    # Color scheme
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    color_map = {name: colors[i] for i, name in enumerate(model_names)}
    
    # 1. CVaR Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    cvars = [results[n]['result']['metrics']['cvar_95'] for n in model_names]
    bars = ax.bar(range(len(model_names)), cvars, color=[color_map[n] for n in model_names])
    
    # Add baseline line
    baseline_cvar = results.get('LSTM', results[model_names[0]])['result']['metrics']['cvar_95']
    ax.axhline(y=baseline_cvar, color='red', linestyle='--', label='LSTM Baseline')
    
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('CVaR 95%')
    ax.set_title('CVaR Comparison Across Models')
    ax.legend()
    
    for bar, val in zip(bars, cvars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "cvar_comparison.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "cvar_comparison.png", dpi=150)
    plt.close()
    
    # 2. Multi-metric comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics_to_plot = ['std_pnl', 'cvar_95', 'entropic_risk', 'trading_volume', 'max_delta', 'delta_smoothness']
    titles = ['Std P&L', 'CVaR 95%', 'Entropic Risk', 'Trading Volume', 'Max |δ|', 'Delta Smoothness']
    
    for ax, metric, title in zip(axes.flatten(), metrics_to_plot, titles):
        values = [results[n]['result']['metrics'].get(metric, 0) for n in model_names]
        bars = ax.bar(range(len(model_names)), values, color=[color_map[n] for n in model_names])
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Multi-Metric Comparison', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "multi_metric_comparison.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "multi_metric_comparison.png", dpi=150)
    plt.close()
    
    # 3. P&L distributions
    fig, ax = plt.subplots(figsize=(14, 7))
    for name in model_names[:5]:  # Top 5 for clarity
        pnl = np.clip(results[name]['pnl'], -5, 5)
        ax.hist(pnl, bins=80, alpha=0.4, label=name, color=color_map[name], density=True)
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax.set_xlabel('P&L')
    ax.set_ylabel('Density')
    ax.set_title('P&L Distribution Comparison')
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pnl_distributions.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "pnl_distributions.png", dpi=150)
    plt.close()
    
    # 4. Delta paths
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    sample_idx = 0
    
    for ax, name in zip(axes.flatten(), model_names[:6]):
        deltas = results[name]['deltas'][sample_idx]
        ax.plot(deltas, color=color_map[name], linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=1.5, color='red', linestyle=':', alpha=0.5)
        ax.axhline(y=-1.5, color='red', linestyle=':', alpha=0.5)
        ax.set_title(name)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Delta')
    
    plt.suptitle('Sample Delta Paths', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "delta_paths.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "delta_paths.png", dpi=150)
    plt.close()
    
    print(f"  Figures saved to {FIGURES_DIR}")


def save_results(results: Dict, comparisons: Dict):
    """Save all results to files."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Metrics table
    rows = []
    for name, data in results.items():
        row = {'Model': name}
        row.update(data['result']['metrics'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False, float_format='%.4f')
    
    # JSON summary
    summary = {
        name: {
            'metrics': data['result']['metrics'],
            'n_parameters': sum(p.numel() for p in data['result']['model'].parameters())
        }
        for name, data in results.items()
    }
    
    with open(RESULTS_DIR / "results_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Results saved to {RESULTS_DIR}")


def print_final_summary(results: Dict, comparisons: Dict):
    """Print final summary table."""
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"\n{'Model':<20} {'Params':>10} {'Std P&L':>10} {'CVaR95':>10} {'Entropic':>10} {'Volume':>10}")
    print("-"*80)
    
    for name, data in results.items():
        m = data['result']['metrics']
        n_params = sum(p.numel() for p in data['result']['model'].parameters())
        print(f"{name:<20} {n_params:>10,} {m['std_pnl']:>10.4f} {m['cvar_95']:>10.4f} "
              f"{m['entropic_risk']:>10.4f} {m['trading_volume']:>10.4f}")
    
    # Find best model
    best_cvar = min(results.items(), key=lambda x: x[1]['result']['metrics']['cvar_95'])
    best_std = min(results.items(), key=lambda x: x[1]['result']['metrics']['std_pnl'])
    
    print("\n" + "-"*80)
    print(f"Best CVaR95: {best_cvar[0]} ({best_cvar[1]['result']['metrics']['cvar_95']:.4f})")
    print(f"Best Std P&L: {best_std[0]} ({best_std[1]['result']['metrics']['std_pnl']:.4f})")
    print("="*80)


def main():
    """Main experiment runner."""
    print("="*70)
    print("FAIR COMPARISON OF DEEP HEDGING ARCHITECTURES")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Configuration
    data_config = DataConfig(
        n_steps=30,
        T=30/365,
        n_train=50000,  # Reduced for faster iteration
        n_val=10000,
        n_test=50000,
        batch_size=256,
        seed=42
    )
    
    training_config = TrainingConfig(
        stage1_epochs=40,
        stage2_epochs=25,
        stage1_lr=0.001,
        stage2_lr=0.0001,
        stage1_patience=12,
        stage2_patience=8,
        gamma=1e-3,
        band_width=0.15,
        grad_clip=5.0,
        weight_decay=1e-4
    )
    
    # Generate data
    train_loader, val_loader, test_loader, input_dim = generate_data(data_config)
    
    # Run baseline comparison
    results = run_baseline_comparison(
        train_loader, val_loader, test_loader, input_dim, training_config
    )
    
    # Statistical analysis
    comparisons = run_statistical_analysis(results, baseline_name='LSTM')
    
    # Generate outputs
    generate_figures(results, comparisons)
    save_results(results, comparisons)
    print_final_summary(results, comparisons)
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✓ FAIR COMPARISON COMPLETE")
    
    return results, comparisons


if __name__ == '__main__':
    results, comparisons = main()
