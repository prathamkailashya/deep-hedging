#!/usr/bin/env python3
"""
Ablation Study: Novel Hybrid Signature-Attention Hedger (H-SAH)

Compares:
1. LSTM (baseline)
2. Sig-LSTM (existing)
3. H-SAH (novel hybrid - no RL)
4. H-SAH + CVaR-PPO (risk-aware RL fine-tuning)

Success criteria:
- CVaR95 ↓ by ≥ 5-10% vs LSTM
- Entropic risk ↓
- Std P&L ≤ LSTM × 1.1
- Trading volume ≤ LSTM × 1.1
- Delta paths bounded and smooth
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from tqdm import tqdm
from scipy import stats

from utils.config import set_seed
from env.data_generator import DataGenerator
from env.heston import HestonParams
from models.kozyra_models import HedgingLSTM
from models.hybrid_sah import HybridSignatureAttentionHedger, HSAHTrainer
from models.cvar_ppo import CVaRPPO, CVaRPPOTrainer
from features.signatures import SignatureConditionedLSTM
from train.trainer import DeepHedgingTrainer

# Output directories
RESULTS_DIR = Path(__file__).parent / "ablation_results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Experiment settings
N_TRAIN, N_VAL, N_TEST = 20000, 5000, 50000
N_STEPS = 30
BATCH_SIZE = 256
DELTA_MAX = 1.5
T = 30/365

set_seed(42)
device = 'cpu'


def generate_data():
    """Generate Heston model data."""
    print("="*60)
    print("DATA GENERATION")
    print("="*60)
    
    heston_params = HestonParams(
        S0=100.0, v0=0.04, r=0.0, 
        kappa=1.0, theta=0.04, sigma=0.2, rho=-0.7
    )
    generator = DataGenerator(
        n_steps=N_STEPS, T=T, S0=100.0, K=100.0, r=0.0,
        cost_multiplier=0.0, model_type='heston', heston_params=heston_params
    )
    
    train_data, val_data, test_data = generator.generate_train_val_test(
        n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, base_seed=42
    )
    train_loader, val_loader, test_loader = generator.get_dataloaders(
        train_data, val_data, test_data, batch_size=BATCH_SIZE
    )
    
    print(f"  Train: {N_TRAIN}, Val: {N_VAL}, Test: {N_TEST}")
    print(f"  Input dim: {test_data.n_features}")
    
    return train_loader, val_loader, test_loader, test_data.n_features


def compute_metrics(pnl: np.ndarray, deltas: np.ndarray, lambda_risk: float = 1.0) -> dict:
    """Compute all hedging metrics."""
    losses = -pnl
    
    # Trading volume
    delta_changes = np.abs(np.diff(
        np.concatenate([np.zeros((len(deltas), 1)), deltas, np.zeros((len(deltas), 1))], axis=1),
        axis=1
    ))
    volume = np.mean(np.sum(delta_changes, axis=1))
    
    # Entropic risk
    scaled = -lambda_risk * pnl
    max_val = np.max(scaled)
    entropic = (max_val + np.log(np.mean(np.exp(scaled - max_val)))) / lambda_risk
    
    # Delta smoothness
    smoothness = np.mean(np.abs(np.diff(deltas, axis=1)))
    
    return {
        'mean_pnl': np.mean(pnl),
        'std_pnl': np.std(pnl),
        'var_95': np.percentile(losses, 95),
        'var_99': np.percentile(losses, 99),
        'cvar_95': np.mean(losses[losses >= np.percentile(losses, 95)]),
        'cvar_99': np.mean(losses[losses >= np.percentile(losses, 99)]),
        'entropic_risk': entropic,
        'trading_volume': volume,
        'max_delta': np.max(np.abs(deltas)),
        'delta_smoothness': smoothness
    }


def bootstrap_ci(data: np.ndarray, metric_fn, n_bootstrap: int = 1000, alpha: float = 0.05):
    """Compute bootstrap confidence interval."""
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        bootstrap_stats.append(metric_fn(data[idx]))
    
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return lower, upper


def train_lstm_baseline(train_loader, val_loader, test_loader, input_dim):
    """Train LSTM baseline (Kozyra-style)."""
    print("\n" + "="*60)
    print("[1/4] LSTM BASELINE")
    print("="*60)
    
    model = HedgingLSTM(
        state_dim=input_dim,
        hidden_size=50,
        num_layers=2,
        delta_scale=DELTA_MAX
    ).to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = DeepHedgingTrainer(
        model=model,
        lambda_risk=1.0,
        learning_rate=0.0005,
        device=device
    )
    
    trainer.fit(train_loader, val_loader, n_epochs=50, patience=15, verbose=True)
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    
    return {
        'model': model,
        'metrics': compute_metrics(pnl, deltas),
        'pnl': pnl,
        'deltas': deltas
    }


def train_sig_lstm(train_loader, val_loader, test_loader, input_dim):
    """Train Signature-conditioned LSTM."""
    print("\n" + "="*60)
    print("[2/4] SIGNATURE-LSTM")
    print("="*60)
    
    model = SignatureConditionedLSTM(
        input_dim=input_dim,
        hidden_size=50,
        num_layers=2,
        sig_depth=2,
        sig_window=5,
        delta_scale=DELTA_MAX
    ).to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = DeepHedgingTrainer(
        model=model,
        lambda_risk=1.0,
        learning_rate=0.0005,
        device=device
    )
    
    trainer.fit(train_loader, val_loader, n_epochs=50, patience=15, verbose=True)
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    
    return {
        'model': model,
        'metrics': compute_metrics(pnl, deltas),
        'pnl': pnl,
        'deltas': deltas
    }


def train_hsah(train_loader, val_loader, test_loader, input_dim):
    """Train Hybrid Signature-Attention Hedger."""
    print("\n" + "="*60)
    print("[3/4] HYBRID SIGNATURE-ATTENTION HEDGER (H-SAH)")
    print("="*60)
    
    model = HybridSignatureAttentionHedger(
        input_dim=input_dim,
        n_steps=N_STEPS,
        embed_dim=64,
        n_heads=4,  # Must divide embed_dim evenly
        sig_window=5,
        sig_order=3,
        delta_max=DELTA_MAX,
        max_increment=0.3,
        dropout=0.1
    ).to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = HSAHTrainer(
        model=model,
        lambda_risk=1.0,
        gamma=1e-3,
        band_width=0.15,
        device=device
    )
    
    # Two-stage training
    trainer.train_stage1(train_loader, val_loader, n_epochs=30, lr=0.001, patience=10)
    trainer.train_stage2(train_loader, val_loader, n_epochs=20, lr=0.0001, patience=10, nu=1.0)  # Reduced nu
    
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    
    return {
        'model': model,
        'trainer': trainer,
        'metrics': compute_metrics(pnl, deltas),
        'pnl': pnl,
        'deltas': deltas
    }


def train_hsah_rl(train_loader, val_loader, test_loader, input_dim, hsah_model):
    """Fine-tune H-SAH with CVaR-PPO."""
    print("\n" + "="*60)
    print("[4/4] H-SAH + CVaR-PPO FINE-TUNING")
    print("="*60)
    
    rl_model = CVaRPPO(
        state_dim=input_dim,
        hidden_dim=64,
        delta_max=DELTA_MAX,
        max_increment=0.2,
        lr_policy=1e-4,
        lr_value=3e-4,
        clip_epsilon=0.1,
        action_penalty=1e-3,
        n_epochs=5,
        device=device
    )
    
    print(f"  Parameters: {sum(p.numel() for p in rl_model.parameters()):,}")
    
    # Warm-start from H-SAH
    rl_model.warm_start_from_model(hsah_model, train_loader)
    
    trainer = CVaRPPOTrainer(
        model=rl_model,
        lambda_risk=1.0,
        device=device
    )
    
    # Train with risk-aware RL
    trainer.train(train_loader, n_episodes=100, verbose=True)
    
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    
    return {
        'model': rl_model,
        'metrics': compute_metrics(pnl, deltas),
        'pnl': pnl,
        'deltas': deltas
    }


def statistical_comparison(baseline_pnl: np.ndarray, model_pnl: np.ndarray, model_name: str):
    """Perform statistical tests comparing model to baseline."""
    print(f"\n  Statistical comparison: {model_name} vs LSTM")
    
    # CVaR comparison
    def cvar95(x):
        losses = -x
        return np.mean(losses[losses >= np.percentile(losses, 95)])
    
    baseline_cvar = cvar95(baseline_pnl)
    model_cvar = cvar95(model_pnl)
    
    # Bootstrap CIs
    baseline_ci = bootstrap_ci(baseline_pnl, cvar95)
    model_ci = bootstrap_ci(model_pnl, cvar95)
    
    improvement = (baseline_cvar - model_cvar) / baseline_cvar * 100
    
    print(f"    LSTM CVaR95: {baseline_cvar:.4f} [{baseline_ci[0]:.4f}, {baseline_ci[1]:.4f}]")
    print(f"    {model_name} CVaR95: {model_cvar:.4f} [{model_ci[0]:.4f}, {model_ci[1]:.4f}]")
    print(f"    Improvement: {improvement:.2f}%")
    
    # Check if CIs don't overlap (significant at 95%)
    significant = model_ci[1] < baseline_ci[0]
    print(f"    Statistically significant: {'YES' if significant else 'NO'}")
    
    return {
        'baseline_cvar': baseline_cvar,
        'model_cvar': model_cvar,
        'improvement_pct': improvement,
        'significant': significant,
        'baseline_ci': baseline_ci,
        'model_ci': model_ci
    }


def check_acceptance(results: dict, baseline_metrics: dict) -> dict:
    """Check if model meets acceptance criteria."""
    metrics = results['metrics']
    
    checks = {
        'cvar_improved': metrics['cvar_95'] < baseline_metrics['cvar_95'],
        'entropic_improved': metrics['entropic_risk'] < baseline_metrics['entropic_risk'],
        'std_acceptable': metrics['std_pnl'] <= baseline_metrics['std_pnl'] * 1.1,
        'volume_acceptable': metrics['trading_volume'] <= baseline_metrics['trading_volume'] * 1.1,
        'delta_bounded': metrics['max_delta'] <= DELTA_MAX * 1.05,
        'delta_smooth': metrics['delta_smoothness'] < 0.3
    }
    
    checks['all_passed'] = all(checks.values())
    
    return checks


def generate_figures(all_results: dict):
    """Generate comprehensive comparison figures."""
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    models = list(all_results.keys())
    colors = {'LSTM': '#1f77b4', 'Sig-LSTM': '#ff7f0e', 'H-SAH': '#2ca02c', 'H-SAH+RL': '#d62728'}
    
    # 1. P&L Histograms
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in models:
        pnl = np.clip(all_results[name]['pnl'], -6, 6)
        ax.hist(pnl, bins=80, alpha=0.5, label=f"{name} (σ={all_results[name]['metrics']['std_pnl']:.3f})",
                color=colors.get(name, 'gray'), density=True)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax.set_xlabel('P&L', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('P&L Distribution Comparison', fontsize=14)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pnl_histogram.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "pnl_histogram.png", dpi=150)
    plt.close()
    
    # 2. Metrics Bar Chart
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics_plot = ['std_pnl', 'cvar_95', 'entropic_risk', 'trading_volume', 'max_delta', 'delta_smoothness']
    titles = ['Std P&L', 'CVaR 95%', 'Entropic Risk', 'Trading Volume', 'Max |δ|', 'Delta Smoothness']
    
    for ax, metric, title in zip(axes.flatten(), metrics_plot, titles):
        values = [all_results[m]['metrics'][metric] for m in models]
        bars = ax.bar(range(len(models)), values, color=[colors.get(m, 'gray') for m in models])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Ablation Study: Metrics Comparison', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_comparison.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "metrics_comparison.png", dpi=150)
    plt.close()
    
    # 3. Sample Delta Paths
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sample_idx = 0
    
    for ax, name in zip(axes.flatten(), models):
        deltas = all_results[name]['deltas'][sample_idx]
        ax.plot(deltas, color=colors.get(name, 'gray'), linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=DELTA_MAX, color='red', linestyle=':', alpha=0.5)
        ax.axhline(y=-DELTA_MAX, color='red', linestyle=':', alpha=0.5)
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Delta', fontsize=10)
        ax.set_title(f'{name} Delta Path', fontsize=12)
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Sample Delta Paths (Path #0)', fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "delta_paths.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "delta_paths.png", dpi=150)
    plt.close()
    
    # 4. Trading Volume Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name in models:
        deltas = all_results[name]['deltas']
        changes = np.abs(np.diff(
            np.concatenate([np.zeros((len(deltas), 1)), deltas, np.zeros((len(deltas), 1))], axis=1),
            axis=1
        ))
        volumes = np.sum(changes, axis=1)
        ax.hist(volumes, bins=50, alpha=0.5, label=f"{name} (μ={np.mean(volumes):.2f})",
               color=colors.get(name, 'gray'), density=True)
    
    ax.set_xlabel('Trading Volume', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Trading Volume Distribution', fontsize=14)
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "volume_distribution.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "volume_distribution.png", dpi=150)
    plt.close()
    
    # 5. CVaR Improvement Bar Chart
    baseline_cvar = all_results['LSTM']['metrics']['cvar_95']
    fig, ax = plt.subplots(figsize=(8, 5))
    
    improvements = []
    for name in models:
        cvar = all_results[name]['metrics']['cvar_95']
        imp = (baseline_cvar - cvar) / baseline_cvar * 100
        improvements.append(imp)
    
    bars = ax.bar(range(len(models)), improvements, color=[colors.get(m, 'gray') for m in models])
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.7)
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='5% target')
    ax.axhline(y=10, color='green', linestyle=':', alpha=0.5, label='10% target')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel('CVaR95 Improvement (%)', fontsize=12)
    ax.set_title('CVaR95 Improvement vs LSTM Baseline', fontsize=14)
    ax.legend()
    
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%',
               ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "cvar_improvement.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "cvar_improvement.png", dpi=150)
    plt.close()
    
    print(f"  Figures saved to {FIGURES_DIR}")


def save_results(all_results: dict, stats: dict):
    """Save results to files."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Metrics table
    rows = []
    for name, data in all_results.items():
        row = {'Model': name}
        row.update(data['metrics'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "ablation_metrics.csv", index=False, float_format='%.4f')
    
    # JSON results
    json_data = {
        name: {k: float(v) if isinstance(v, (np.floating, float)) else v 
               for k, v in data['metrics'].items()}
        for name, data in all_results.items()
    }
    json_data['statistical_tests'] = {
        k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else (bool(vv) if isinstance(vv, (np.bool_, bool)) else vv)
            for kk, vv in v.items() if not isinstance(vv, tuple)}
        for k, v in stats.items()
    }
    
    with open(RESULTS_DIR / "ablation_results.json", 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"  Results saved to {RESULTS_DIR}")


def print_summary(all_results: dict, acceptance: dict, stats: dict):
    """Print final summary."""
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    
    print(f"\n{'Model':<15} {'Std P&L':>10} {'CVaR95':>10} {'Entropic':>10} {'Volume':>10} {'Smooth':>10}")
    print("-"*80)
    
    for name, data in all_results.items():
        m = data['metrics']
        print(f"{name:<15} {m['std_pnl']:>10.4f} {m['cvar_95']:>10.4f} "
              f"{m['entropic_risk']:>10.4f} {m['trading_volume']:>10.4f} {m['delta_smoothness']:>10.4f}")
    
    print("\n" + "-"*80)
    print("ACCEPTANCE CRITERIA")
    print("-"*80)
    
    baseline = all_results['LSTM']['metrics']
    
    for name in ['Sig-LSTM', 'H-SAH', 'H-SAH+RL']:
        if name not in acceptance:
            continue
        checks = acceptance[name]
        status = "✓ PASS" if checks['all_passed'] else "✗ FAIL"
        print(f"\n{name}: {status}")
        print(f"  CVaR improved: {'✓' if checks['cvar_improved'] else '✗'}")
        print(f"  Entropic improved: {'✓' if checks['entropic_improved'] else '✗'}")
        print(f"  Std acceptable (≤{baseline['std_pnl']*1.1:.3f}): {'✓' if checks['std_acceptable'] else '✗'}")
        print(f"  Volume acceptable (≤{baseline['trading_volume']*1.1:.2f}): {'✓' if checks['volume_acceptable'] else '✗'}")
    
    print("\n" + "-"*80)
    print("STATISTICAL SIGNIFICANCE (Bootstrap 95% CI)")
    print("-"*80)
    
    for name, s in stats.items():
        sig = "SIGNIFICANT" if s['significant'] else "not significant"
        print(f"\n{name} vs LSTM:")
        print(f"  CVaR improvement: {s['improvement_pct']:.2f}% ({sig})")
    
    print("\n" + "="*80)


def main():
    print("="*60)
    print("ABLATION STUDY: HYBRID SIGNATURE-ATTENTION HEDGER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Generate data
    train_loader, val_loader, test_loader, input_dim = generate_data()
    
    all_results = {}
    acceptance = {}
    stats = {}
    
    # 1. LSTM Baseline
    all_results['LSTM'] = train_lstm_baseline(train_loader, val_loader, test_loader, input_dim)
    baseline_metrics = all_results['LSTM']['metrics']
    
    # 2. Signature-LSTM
    all_results['Sig-LSTM'] = train_sig_lstm(train_loader, val_loader, test_loader, input_dim)
    acceptance['Sig-LSTM'] = check_acceptance(all_results['Sig-LSTM'], baseline_metrics)
    stats['Sig-LSTM'] = statistical_comparison(
        all_results['LSTM']['pnl'], all_results['Sig-LSTM']['pnl'], 'Sig-LSTM'
    )
    
    # 3. H-SAH (novel hybrid)
    all_results['H-SAH'] = train_hsah(train_loader, val_loader, test_loader, input_dim)
    acceptance['H-SAH'] = check_acceptance(all_results['H-SAH'], baseline_metrics)
    stats['H-SAH'] = statistical_comparison(
        all_results['LSTM']['pnl'], all_results['H-SAH']['pnl'], 'H-SAH'
    )
    
    # 4. H-SAH + CVaR-PPO (only if H-SAH shows promise)
    if acceptance['H-SAH']['cvar_improved'] or acceptance['H-SAH']['entropic_improved']:
        all_results['H-SAH+RL'] = train_hsah_rl(
            train_loader, val_loader, test_loader, input_dim, 
            all_results['H-SAH']['model']
        )
        acceptance['H-SAH+RL'] = check_acceptance(all_results['H-SAH+RL'], baseline_metrics)
        stats['H-SAH+RL'] = statistical_comparison(
            all_results['LSTM']['pnl'], all_results['H-SAH+RL']['pnl'], 'H-SAH+RL'
        )
    else:
        print("\n  Skipping RL fine-tuning: H-SAH did not improve tail risk")
    
    # Generate outputs
    generate_figures(all_results)
    save_results(all_results, stats)
    print_summary(all_results, acceptance, stats)
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✓ ABLATION STUDY COMPLETE")
    
    return all_results, acceptance, stats


if __name__ == '__main__':
    main()
