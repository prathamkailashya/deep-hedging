#!/usr/bin/env python3
"""
COMPLETE AUDIT EXPERIMENT RUNNER

This script executes all parts of the scientific audit:
1. Scientific validation of codebase
2. Fair comparison with 10 seeds
3. Statistical analysis with bootstrap CI
4. Generate all figures and tables

Run: python run_complete_audit.py
"""

import sys
import os

# Add paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import modules
from new_experiments.data.data_generator import DataGenerator, HestonParams
from new_experiments.models.base_model import LSTMHedger
from new_experiments.models.signature_models import SignatureLSTM, SignatureMLP
from new_experiments.models.transformer_models import TransformerHedger
from new_experiments.models.attention_lstm import AttentionLSTM

from final_audit_experiments.audit.scientific_audit import ScientificAuditor
from final_audit_experiments.training.fair_trainer import FairTrainer, FairTrainingConfig, TrainingResult
from final_audit_experiments.evaluation.statistical_analysis import (
    ComprehensiveComparison, SeedRobustnessAnalyzer, generate_comparison_table,
    bootstrap_ci, cvar_statistic
)


# ============================================================
# CONFIGURATION
# ============================================================

class ExperimentConfig:
    """Complete experiment configuration."""
    
    # Data
    n_train: int = 50000
    n_val: int = 10000
    n_test: int = 50000
    batch_size: int = 256
    n_steps: int = 30
    T: float = 30 / 365
    K: float = 100.0
    
    # Seeds for robustness
    n_seeds: int = 10
    base_seed: int = 42
    
    # Training (identical for all models)
    stage1_epochs: int = 40
    stage2_epochs: int = 25
    stage1_lr: float = 0.001
    stage2_lr: float = 0.0001
    patience: int = 12
    
    # Model constraints
    delta_max: float = 1.5
    grad_clip: float = 5.0
    weight_decay: float = 1e-4
    
    # Statistical analysis
    n_bootstrap: int = 10000
    confidence: float = 0.95
    alpha: float = 0.05
    
    # Output
    results_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results'
    )


def create_model(model_name: str, input_dim: int, config: ExperimentConfig):
    """Create model with fair hyperparameters."""
    
    if model_name == 'LSTM':
        return LSTMHedger(
            input_dim=input_dim,
            hidden_size=50,
            num_layers=2,
            dropout=0.0,
            delta_max=config.delta_max
        )
    
    elif model_name == 'SignatureLSTM':
        return SignatureLSTM(
            input_dim=input_dim,
            hidden_size=64,
            num_layers=2,
            sig_order=3,
            sig_window=5,
            dropout=0.1,
            delta_max=config.delta_max
        )
    
    elif model_name == 'SignatureMLP':
        return SignatureMLP(
            input_dim=input_dim,
            hidden_size=64,
            n_layers=3,
            sig_order=3,
            sig_window=5,
            dropout=0.1,
            delta_max=config.delta_max
        )
    
    elif model_name == 'Transformer':
        return TransformerHedger(
            input_dim=input_dim,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            delta_max=config.delta_max
        )
    
    elif model_name == 'AttentionLSTM':
        return AttentionLSTM(
            input_dim=input_dim,
            hidden_size=64,
            num_layers=2,
            attention_dim=32,
            memory_length=10,
            combination='concat',
            dropout=0.1,
            delta_max=config.delta_max
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_single_seed_experiment(
    model_name: str,
    seed: int,
    train_loader,
    val_loader,
    test_loader,
    input_dim: int,
    config: ExperimentConfig,
    device: str,
    verbose: bool = True
) -> TrainingResult:
    """Run single seed experiment for one model."""
    
    # Create model
    model = create_model(model_name, input_dim, config)
    
    # Create trainer with fair config
    train_config = FairTrainingConfig(
        stage1_epochs=config.stage1_epochs,
        stage2_epochs=config.stage2_epochs,
        stage1_lr=config.stage1_lr,
        stage2_lr=config.stage2_lr,
        stage1_patience=config.patience,
        stage2_patience=config.patience // 2,
        grad_clip=config.grad_clip,
        weight_decay=config.weight_decay
    )
    
    trainer = FairTrainer(
        model=model,
        config=train_config,
        device=device,
        model_name=model_name
    )
    
    # Train
    result = trainer.train(train_loader, val_loader, seed=seed, verbose=verbose)
    
    # Evaluate
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    result.test_metrics = metrics
    result.test_pnl = pnl
    result.test_deltas = deltas
    
    return result


def run_fair_comparison(
    config: ExperimentConfig,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete fair comparison with multiple seeds.
    
    Returns comprehensive results for all models across all seeds.
    """
    
    print("=" * 70)
    print("FAIR COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"Seeds: {config.n_seeds}")
    print(f"Models: LSTM, SignatureLSTM, SignatureMLP, Transformer, AttentionLSTM")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Generate data (fixed seed for reproducibility)
    print("\n[1/4] Generating data...")
    heston_params = HestonParams()
    data_gen = DataGenerator(
        heston_params=heston_params,
        n_steps=config.n_steps,
        T=config.T,
        K=config.K
    )
    
    train_data, val_data, test_data = data_gen.generate_splits(
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        base_seed=config.base_seed
    )
    
    train_loader, val_loader, test_loader = data_gen.get_dataloaders(
        train_data, val_data, test_data,
        batch_size=config.batch_size
    )
    
    input_dim = train_data.n_features
    print(f"  Input dim: {input_dim}")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Models to compare
    models = ['LSTM', 'SignatureLSTM', 'SignatureMLP', 'Transformer', 'AttentionLSTM']
    seeds = [config.base_seed + i * 100 for i in range(config.n_seeds)]
    
    # Store all results
    all_results: Dict[str, Dict[int, TrainingResult]] = {m: {} for m in models}
    
    # Run experiments
    print(f"\n[2/4] Training all models ({config.n_seeds} seeds each)...")
    total_runs = len(models) * len(seeds)
    run_count = 0
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        for seed in seeds:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] {model_name} - Seed {seed}")
            
            result = run_single_seed_experiment(
                model_name=model_name,
                seed=seed,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                input_dim=input_dim,
                config=config,
                device=device,
                verbose=verbose
            )
            
            all_results[model_name][seed] = result
            
            # Print test metrics
            print(f"  CVaR95: {result.test_metrics['cvar_95']:.4f}")
            print(f"  Std P&L: {result.test_metrics['std_pnl']:.4f}")
    
    return {
        'results': all_results,
        'models': models,
        'seeds': seeds,
        'config': config.__dict__
    }


def run_statistical_analysis(
    experiment_results: Dict[str, Any],
    config: ExperimentConfig
) -> Dict[str, Any]:
    """Run comprehensive statistical analysis."""
    
    print("\n[3/4] Running statistical analysis...")
    
    all_results = experiment_results['results']
    models = experiment_results['models']
    seeds = experiment_results['seeds']
    
    # Convert to format for comparison
    # {model_name: {seed: {metric: value}}}
    comparison_data = {}
    for model_name in models:
        comparison_data[model_name] = {}
        for seed in seeds:
            if seed in all_results[model_name]:
                comparison_data[model_name][seed] = all_results[model_name][seed].test_metrics
    
    # Run comparison
    comparator = ComprehensiveComparison(
        baseline_name='LSTM',
        n_bootstrap=config.n_bootstrap,
        confidence=config.confidence,
        alpha=config.alpha
    )
    
    comparison_results = comparator.compare_models(
        comparison_data,
        metrics=['cvar_95', 'cvar_99', 'std_pnl', 'entropic_risk', 'trading_volume']
    )
    
    # Seed robustness analysis
    seed_analyzer = SeedRobustnessAnalyzer(n_seeds=config.n_seeds, confidence=config.confidence)
    
    seed_robustness = {}
    for model_name in models:
        model_seeds = {s: r.test_metrics['cvar_95'] for s, r in all_results[model_name].items()}
        seed_robustness[model_name] = seed_analyzer.analyze_metric_across_seeds(model_seeds)
    
    return {
        'comparison': comparison_results,
        'seed_robustness': seed_robustness,
        'comparison_table': generate_comparison_table(comparison_results)
    }


def generate_figures(
    experiment_results: Dict[str, Any],
    analysis_results: Dict[str, Any],
    output_dir: str
):
    """Generate all figures for the report."""
    
    print("\n[4/4] Generating figures...")
    
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    all_results = experiment_results['results']
    models = experiment_results['models']
    seeds = experiment_results['seeds']
    
    # 1. CVaR comparison boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    cvar_data = []
    labels = []
    for model_name in models:
        cvar_values = [all_results[model_name][s].test_metrics['cvar_95'] for s in seeds]
        cvar_data.append(cvar_values)
        labels.append(model_name)
    
    bp = ax.boxplot(cvar_data, labels=labels, patch_artist=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('CVaR95')
    ax.set_title('CVaR95 Distribution Across Seeds (Lower is Better)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'cvar_comparison_boxplot.pdf'), dpi=300)
    plt.savefig(os.path.join(figures_dir, 'cvar_comparison_boxplot.png'), dpi=300)
    plt.close()
    
    # 2. P&L histogram comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Use first seed for histogram comparison
    first_seed = seeds[0]
    for i, model_name in enumerate(models):
        ax = axes[i]
        pnl = all_results[model_name][first_seed].test_pnl
        ax.hist(pnl, bins=50, alpha=0.7, color=colors[i], edgecolor='black')
        ax.axvline(np.mean(pnl), color='red', linestyle='--', label=f'Mean: {np.mean(pnl):.3f}')
        ax.axvline(np.percentile(-pnl, 95), color='orange', linestyle=':', label=f'VaR95: {np.percentile(-pnl, 95):.3f}')
        ax.set_xlabel('P&L')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{model_name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[-1].axis('off')  # Hide unused subplot
    plt.suptitle('P&L Distributions (Seed={})'.format(first_seed))
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pnl_histograms.pdf'), dpi=300)
    plt.savefig(os.path.join(figures_dir, 'pnl_histograms.png'), dpi=300)
    plt.close()
    
    # 3. Delta paths comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, model_name in enumerate(models):
        ax = axes[i]
        deltas = all_results[model_name][first_seed].test_deltas
        # Plot first 10 paths
        for j in range(min(10, len(deltas))):
            ax.plot(deltas[j], alpha=0.5, linewidth=0.8)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Delta')
        ax.set_title(f'{model_name}')
        ax.set_ylim(-1.6, 1.6)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
    
    axes[-1].axis('off')
    plt.suptitle('Sample Delta Paths')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'delta_paths.pdf'), dpi=300)
    plt.savefig(os.path.join(figures_dir, 'delta_paths.png'), dpi=300)
    plt.close()
    
    # 4. Seed robustness plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.6
    
    means = []
    errors = []
    for model_name in models:
        robustness = analysis_results['seed_robustness'][model_name]
        means.append(robustness['mean'])
        errors.append(robustness['std'])
    
    bars = ax.bar(x, means, width, yerr=errors, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Model')
    ax.set_ylabel('CVaR95 (mean ± std across seeds)')
    ax.set_title('Seed Robustness Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'seed_robustness.pdf'), dpi=300)
    plt.savefig(os.path.join(figures_dir, 'seed_robustness.png'), dpi=300)
    plt.close()
    
    # 5. Learning curves (first seed)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, model_name in enumerate(models):
        ax = axes[i]
        result = all_results[model_name][first_seed]
        
        # Stage 1
        s1_epochs = len(result.stage1_train_loss)
        ax.plot(range(s1_epochs), result.stage1_train_loss, 'b-', label='Stage1 Train', alpha=0.7)
        ax.plot(range(s1_epochs), result.stage1_val_loss, 'b--', label='Stage1 Val', alpha=0.7)
        
        # Stage 2
        s2_epochs = len(result.stage2_train_loss)
        offset = s1_epochs
        ax.plot(range(offset, offset + s2_epochs), result.stage2_train_loss, 'r-', label='Stage2 Train', alpha=0.7)
        ax.plot(range(offset, offset + s2_epochs), result.stage2_val_loss, 'r--', label='Stage2 Val', alpha=0.7)
        
        ax.axvline(s1_epochs, color='green', linestyle=':', alpha=0.5, label='Stage transition')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{model_name}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    axes[-1].axis('off')
    plt.suptitle('Learning Curves')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'learning_curves.pdf'), dpi=300)
    plt.savefig(os.path.join(figures_dir, 'learning_curves.png'), dpi=300)
    plt.close()
    
    print(f"  Figures saved to: {figures_dir}")


def save_results(
    experiment_results: Dict[str, Any],
    analysis_results: Dict[str, Any],
    audit_results: Dict[str, Any],
    output_dir: str
):
    """Save all results to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to serializable format
    serializable_results = {}
    for model_name, seeds_dict in experiment_results['results'].items():
        serializable_results[model_name] = {}
        for seed, result in seeds_dict.items():
            serializable_results[model_name][seed] = {
                'test_metrics': result.test_metrics,
                'n_parameters': result.n_parameters,
                'training_time': result.training_time,
                'stage1_epochs': result.stage1_epochs_run,
                'stage2_epochs': result.stage2_epochs_run
            }
    
    # Save experiment results
    with open(os.path.join(output_dir, 'experiment_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save analysis results (convert dataclasses to dicts)
    analysis_serializable = {
        'comparison_table': analysis_results['comparison_table'],
        'seed_robustness': analysis_results['seed_robustness']
    }
    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        json.dump(analysis_serializable, f, indent=2)
    
    # Save audit results
    with open(os.path.join(output_dir, 'audit_results.json'), 'w') as f:
        json.dump(audit_results, f, indent=2)
    
    # Generate summary table
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("FAIR COMPARISON RESULTS SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append("Model Performance (Mean ± Std across {} seeds):".format(experiment_results['config']['n_seeds']))
    summary_lines.append("-" * 80)
    summary_lines.append(f"{'Model':<20} {'CVaR95':>12} {'Std P&L':>12} {'Entropic':>12} {'Volume':>12}")
    summary_lines.append("-" * 80)
    
    for model_name in experiment_results['models']:
        robustness = analysis_results['seed_robustness'][model_name]
        
        # Get mean metrics across seeds
        metrics_list = [experiment_results['results'][model_name][s].test_metrics for s in experiment_results['seeds']]
        
        cvar = np.mean([m['cvar_95'] for m in metrics_list])
        cvar_std = np.std([m['cvar_95'] for m in metrics_list])
        std_pnl = np.mean([m['std_pnl'] for m in metrics_list])
        entropic = np.mean([m['entropic_risk'] for m in metrics_list])
        volume = np.mean([m['trading_volume'] for m in metrics_list])
        
        summary_lines.append(
            f"{model_name:<20} {cvar:>6.4f}±{cvar_std:.3f} {std_pnl:>12.4f} {entropic:>12.4f} {volume:>12.4f}"
        )
    
    summary_lines.append("-" * 80)
    summary_lines.append("")
    summary_lines.append("Statistical Comparison (vs LSTM baseline):")
    summary_lines.append(analysis_results['comparison_table'])
    
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"  Results saved to: {output_dir}")


def main():
    """Main entry point for complete audit."""
    
    print("=" * 70)
    print("COMPLETE SCIENTIFIC AUDIT OF DEEP HEDGING")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    
    # Configuration
    config = ExperimentConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Ensure output directory exists
    os.makedirs(config.results_dir, exist_ok=True)
    
    # PART 1: Scientific Audit
    print("\n" + "=" * 70)
    print("PART 1: SCIENTIFIC VALIDATION")
    print("=" * 70)
    
    auditor = ScientificAuditor()
    audit_results = auditor.run_full_audit()
    
    if not audit_results['passed_all']:
        print("\n⚠️  WARNING: Some audits failed. Review before proceeding.")
    
    # PART 2: Fair Comparison
    print("\n" + "=" * 70)
    print("PART 2: FAIR COMPARISON ({} SEEDS)".format(config.n_seeds))
    print("=" * 70)
    
    experiment_results = run_fair_comparison(config, device=device, verbose=True)
    
    # PART 3: Statistical Analysis
    print("\n" + "=" * 70)
    print("PART 3: STATISTICAL ANALYSIS")
    print("=" * 70)
    
    analysis_results = run_statistical_analysis(experiment_results, config)
    
    # Print comparison table
    print("\n" + analysis_results['comparison_table'])
    
    # Generate figures
    generate_figures(experiment_results, analysis_results, config.results_dir)
    
    # Save all results
    save_results(experiment_results, analysis_results, audit_results, config.results_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"\nResults saved to: {config.results_dir}")
    
    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    # Find best model
    best_cvar = float('inf')
    best_model = None
    for model_name in experiment_results['models']:
        mean_cvar = analysis_results['seed_robustness'][model_name]['mean']
        if mean_cvar < best_cvar:
            best_cvar = mean_cvar
            best_model = model_name
    
    print(f"\nBest CVaR95: {best_model} ({best_cvar:.4f})")
    
    # Check if AttentionLSTM claim holds
    lstm_cvar = analysis_results['seed_robustness']['LSTM']['mean']
    attn_cvar = analysis_results['seed_robustness']['AttentionLSTM']['mean']
    improvement = (lstm_cvar - attn_cvar) / lstm_cvar * 100
    
    print(f"\nAttentionLSTM vs LSTM:")
    print(f"  LSTM CVaR95: {lstm_cvar:.4f}")
    print(f"  AttentionLSTM CVaR95: {attn_cvar:.4f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    # Check statistical significance
    comparison = analysis_results['comparison']
    if 'AttentionLSTM' in comparison['comparisons']:
        attn_result = comparison['comparisons']['AttentionLSTM']['cvar_95']
        print(f"  Significant: {'YES' if attn_result.significant else 'NO'}")
        print(f"  p-value: {attn_result.p_value:.4f}")
        print(f"  Effect size (Cohen's d): {attn_result.effect_size:.3f}")
    
    return {
        'audit': audit_results,
        'experiment': experiment_results,
        'analysis': analysis_results
    }


if __name__ == '__main__':
    results = main()
