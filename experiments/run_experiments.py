#!/usr/bin/env python3
"""
Main Experiment Runner for Deep Hedging Research.

Runs all experiments for replicating and extending:
- Buehler et al. (Deep Hedging)
- Kozyra (Oxford MSc thesis)

Usage:
    python run_experiments.py --experiment all
    python run_experiments.py --experiment deep_hedging
    python run_experiments.py --experiment kozyra
    python run_experiments.py --experiment baselines
    python run_experiments.py --experiment enhancements
    python run_experiments.py --experiment real_data
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np

from utils.config import Config, set_seed
from utils.logging_utils import ExperimentLogger
from env.data_generator import DataGenerator
from env.heston import HestonParams
from models.deep_hedging import DeepHedgingModel
from models.kozyra_models import HedgingRNN, HedgingLSTM, KozyraTwoStageModel
from models.baselines import BlackScholesHedge, LelandHedge, WhalleyWilmottHedge, evaluate_baseline
from train.trainer import DeepHedgingTrainer
from train.kozyra_trainer import KozyraTwoStageTrainer
from eval.evaluator import HedgingEvaluator, compare_strategies
from eval.plotting import save_all_figures, plot_pnl_histogram, plot_pnl_boxplot


def run_deep_hedging_replication(config: Config, logger: ExperimentLogger):
    """
    Replicate Buehler et al. Deep Hedging experiments.
    
    Section 5: Numerical Experiments
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Deep Hedging Replication (Buehler et al.)")
    logger.info("=" * 60)
    
    device = config.training.device
    
    # Generate data
    logger.info("Generating data...")
    heston_params = HestonParams(
        S0=config.market.S0,
        v0=config.market.v0,
        r=config.market.r,
        kappa=config.market.kappa,
        theta=config.market.theta,
        sigma=config.market.sigma,
        rho=config.market.rho
    )
    
    generator = DataGenerator(
        n_steps=config.market.n_steps,
        T=config.market.T,
        S0=config.market.S0,
        K=config.market.K,
        r=config.market.r,
        cost_multiplier=config.market.cost_multiplier,
        model_type='heston',
        heston_params=heston_params
    )
    
    train_data, val_data, test_data = generator.generate_train_val_test(
        n_train=config.training.n_train,
        n_val=config.training.n_val,
        n_test=config.training.n_test,
        base_seed=config.training.seed
    )
    
    train_loader, val_loader, test_loader = generator.get_dataloaders(
        train_data, val_data, test_data,
        batch_size=config.training.batch_size
    )
    
    # Create model
    input_dim = train_data.n_features
    n_steps = train_data.n_steps
    
    logger.info(f"Input dim: {input_dim}, N steps: {n_steps}")
    
    model = DeepHedgingModel(
        input_dim=input_dim,
        n_steps=n_steps,
        cost_multiplier=config.market.cost_multiplier,
        lambda_risk=config.training.lambda_risk,
        share_weights=False
    )
    
    # Train
    trainer = DeepHedgingTrainer(
        model=model,
        lambda_risk=config.training.lambda_risk,
        cost_multiplier=config.market.cost_multiplier,
        learning_rate=config.training.learning_rate,
        device=device,
        logger=logger
    )
    
    logger.info("Training Deep Hedging model...")
    history = trainer.fit(
        train_loader, val_loader,
        n_epochs=config.training.n_epochs,
        patience=config.training.patience,
        verbose=True
    )
    
    # Evaluate
    logger.info("Evaluating...")
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    
    logger.info("\nTest Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    return {
        'model': model,
        'history': history,
        'metrics': metrics,
        'pnl': pnl,
        'deltas': deltas,
        'test_data': test_data
    }


def run_kozyra_experiments(config: Config, logger: ExperimentLogger):
    """
    Replicate Kozyra RNN/LSTM and two-stage training experiments.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Kozyra RNN/LSTM Replication")
    logger.info("=" * 60)
    
    device = config.training.device
    
    # Generate data
    generator = DataGenerator(
        n_steps=config.market.n_steps,
        T=config.market.T,
        S0=config.market.S0,
        K=config.market.K,
        cost_multiplier=config.market.cost_multiplier
    )
    
    train_data, val_data, test_data = generator.generate_train_val_test(
        n_train=config.training.n_train,
        n_val=config.training.n_val,
        n_test=config.training.n_test,
        base_seed=config.training.seed
    )
    
    train_loader, val_loader, test_loader = generator.get_dataloaders(
        train_data, val_data, test_data,
        batch_size=config.kozyra.batch_size
    )
    
    results = {}
    
    # 1. Basic RNN
    logger.info("\n--- Training Kozyra RNN ---")
    rnn_model = HedgingRNN(
        state_dim=train_data.n_features,
        hidden_size=config.kozyra.hidden_size,
        num_layers=config.kozyra.num_layers
    )
    
    rnn_trainer = DeepHedgingTrainer(
        model=rnn_model,
        lambda_risk=config.training.lambda_risk,
        cost_multiplier=config.market.cost_multiplier,
        learning_rate=config.kozyra.learning_rate,
        device=device,
        logger=logger
    )
    
    rnn_history = rnn_trainer.fit(train_loader, val_loader, n_epochs=50)
    rnn_metrics, rnn_pnl, rnn_deltas = rnn_trainer.evaluate(test_loader)
    
    results['rnn'] = {
        'model': rnn_model,
        'metrics': rnn_metrics,
        'pnl': rnn_pnl,
        'deltas': rnn_deltas
    }
    
    # 2. LSTM
    logger.info("\n--- Training Kozyra LSTM ---")
    lstm_model = HedgingLSTM(
        state_dim=train_data.n_features,
        hidden_size=config.kozyra.hidden_size,
        num_layers=config.kozyra.num_layers,
        dropout=0.1
    )
    
    lstm_trainer = DeepHedgingTrainer(
        model=lstm_model,
        lambda_risk=config.training.lambda_risk,
        cost_multiplier=config.market.cost_multiplier,
        learning_rate=config.kozyra.learning_rate,
        device=device,
        logger=logger
    )
    
    lstm_history = lstm_trainer.fit(train_loader, val_loader, n_epochs=50)
    lstm_metrics, lstm_pnl, lstm_deltas = lstm_trainer.evaluate(test_loader)
    
    results['lstm'] = {
        'model': lstm_model,
        'metrics': lstm_metrics,
        'pnl': lstm_pnl,
        'deltas': lstm_deltas
    }
    
    # 3. Two-Stage Training
    logger.info("\n--- Training Kozyra Two-Stage Model ---")
    two_stage_model = KozyraTwoStageModel(
        state_dim=train_data.n_features,
        n_steps=train_data.n_steps,
        hidden_size=config.kozyra.hidden_size,
        num_layers=config.kozyra.num_layers,
        gamma=config.kozyra.gamma,
        nu=config.kozyra.nu,
        band_width=config.kozyra.band_width,
        lambda_risk=config.training.lambda_risk
    )
    
    two_stage_trainer = KozyraTwoStageTrainer(
        model=two_stage_model,
        alpha_cvar=0.95,
        lambda_risk=config.training.lambda_risk,
        gamma=config.kozyra.gamma,
        nu=config.kozyra.nu,
        band_width=config.kozyra.band_width,
        cost_multiplier=config.market.cost_multiplier,
        lr_stage1=config.kozyra.learning_rate,
        lr_stage2=config.kozyra.learning_rate / 5,
        device=device,
        logger=logger
    )
    
    two_stage_history = two_stage_trainer.fit(
        train_loader, val_loader,
        n_epochs_stage1=30,
        n_epochs_stage2=20
    )
    
    two_stage_metrics = two_stage_trainer.evaluate(test_loader)
    
    results['two_stage'] = {
        'model': two_stage_model,
        'metrics': two_stage_metrics,
        'history': two_stage_history
    }
    
    return results


def run_baselines(config: Config, logger: ExperimentLogger, test_data):
    """
    Run baseline hedging strategies for comparison.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT: Baseline Strategies")
    logger.info("=" * 60)
    
    stock_paths = test_data.stock_paths.numpy()
    payoffs = test_data.payoffs.numpy()
    time_grid = np.linspace(0, config.market.T, config.market.n_steps + 1)
    
    results = {}
    
    # Black-Scholes Delta
    logger.info("Running Black-Scholes Delta Hedge...")
    bs_hedge = BlackScholesHedge(sigma=0.2, r=config.market.r)
    bs_pnl, bs_info = evaluate_baseline(
        bs_hedge, stock_paths, time_grid,
        config.market.K, config.market.T, config.market.cost_multiplier
    )
    results['bs_delta'] = {
        'pnl': bs_pnl,
        'deltas': bs_hedge.compute_deltas_vectorized(stock_paths, time_grid, config.market.K, config.market.T),
        'info': bs_info
    }
    
    # Leland
    logger.info("Running Leland Hedge...")
    leland_hedge = LelandHedge(sigma=0.2, r=config.market.r, cost=config.market.cost_multiplier)
    leland_pnl, leland_info = evaluate_baseline(
        leland_hedge, stock_paths, time_grid,
        config.market.K, config.market.T, config.market.cost_multiplier
    )
    results['leland'] = {
        'pnl': leland_pnl,
        'deltas': leland_hedge.compute_deltas(stock_paths, time_grid, config.market.K, config.market.T),
        'info': leland_info
    }
    
    # Whalley-Wilmott
    logger.info("Running Whalley-Wilmott Hedge...")
    ww_hedge = WhalleyWilmottHedge(
        sigma=0.2, r=config.market.r,
        cost=config.market.cost_multiplier,
        risk_aversion=config.training.lambda_risk
    )
    ww_pnl, ww_info = evaluate_baseline(
        ww_hedge, stock_paths, time_grid,
        config.market.K, config.market.T, config.market.cost_multiplier
    )
    results['whalley_wilmott'] = {
        'pnl': ww_pnl,
        'deltas': ww_hedge.compute_deltas(stock_paths, time_grid, config.market.K, config.market.T),
        'info': ww_info
    }
    
    # Print comparison
    logger.info("\nBaseline Results:")
    for name, data in results.items():
        pnl = data['pnl']
        logger.info(f"  {name}: mean={np.mean(pnl):.4f}, std={np.std(pnl):.4f}, "
                   f"VaR95={np.percentile(-pnl, 95):.4f}")
    
    return results


def run_all_experiments(config: Config, save_dir: str):
    """Run all experiments and generate results."""
    
    set_seed(config.training.seed)
    
    # Create experiment logger
    logger = ExperimentLogger("deep_hedging_full", base_dir=save_dir)
    logger.log_config(config.to_dict())
    
    all_results = {}
    
    # 1. Deep Hedging Replication
    dh_results = run_deep_hedging_replication(config, logger)
    all_results['deep_hedging'] = dh_results
    
    # 2. Kozyra Experiments
    kozyra_results = run_kozyra_experiments(config, logger)
    all_results['kozyra'] = kozyra_results
    
    # 3. Baselines
    baseline_results = run_baselines(config, logger, dh_results['test_data'])
    all_results['baselines'] = baseline_results
    
    # 4. Generate comparison plots
    logger.info("\nGenerating comparison plots...")
    
    pnl_dict = {
        'Deep Hedging': dh_results['pnl'],
        'Kozyra RNN': kozyra_results['rnn']['pnl'],
        'Kozyra LSTM': kozyra_results['lstm']['pnl'],
        'BS Delta': baseline_results['bs_delta']['pnl'],
        'Leland': baseline_results['leland']['pnl'],
        'Whalley-Wilmott': baseline_results['whalley_wilmott']['pnl']
    }
    
    delta_dict = {
        'Deep Hedging': dh_results['deltas'],
        'BS Delta': baseline_results['bs_delta']['deltas']
    }
    
    figures_dir = logger.figures_dir
    
    plot_pnl_histogram(pnl_dict, save_path=str(figures_dir / "pnl_histogram.pdf"))
    plot_pnl_boxplot(pnl_dict, save_path=str(figures_dir / "pnl_boxplot.pdf"))
    
    # Save final results
    final_metrics = {}
    for name, pnl in pnl_dict.items():
        losses = -pnl
        final_metrics[name] = {
            'mean_pnl': float(np.mean(pnl)),
            'std_pnl': float(np.std(pnl)),
            'var_95': float(np.percentile(losses, 95)),
            'cvar_95': float(np.mean(losses[losses >= np.percentile(losses, 95)]))
        }
    
    logger.save_results(final_metrics)
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 60)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Deep Hedging Experiments")
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'deep_hedging', 'kozyra', 'baselines', 
                               'enhancements', 'real_data'],
                       help='Experiment to run')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--save_dir', type=str, default='../experiments',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cpu/cuda)')
    parser.add_argument('--cost', type=float, default=0.0,
                       help='Transaction cost multiplier')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Override with command line args
    config.training.seed = args.seed
    if args.device:
        config.training.device = args.device
    config.market.cost_multiplier = args.cost
    
    # Run experiments
    if args.experiment == 'all':
        run_all_experiments(config, args.save_dir)
    elif args.experiment == 'deep_hedging':
        logger = ExperimentLogger("deep_hedging", base_dir=args.save_dir)
        run_deep_hedging_replication(config, logger)
    elif args.experiment == 'kozyra':
        logger = ExperimentLogger("kozyra", base_dir=args.save_dir)
        run_kozyra_experiments(config, logger)
    else:
        print(f"Experiment {args.experiment} not yet implemented")


if __name__ == '__main__':
    main()
