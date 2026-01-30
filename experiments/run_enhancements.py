#!/usr/bin/env python3
"""
Enhanced Model Experiments for Deep Hedging.

Tests advanced models beyond the baseline replication:
- Signature features (order 3-5, time-augmented)
- Transformer / SigFormer
- RL agents: MCPG, PPO, DDPG/TD3
- Optuna hyperparameter optimization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import numpy as np
import torch
from tqdm import tqdm

from utils.config import Config, set_seed
from utils.logging_utils import ExperimentLogger
from env.data_generator import DataGenerator
from models.deep_hedging import DeepHedgingModel
from models.transformer import TransformerHedge, SigFormer
from models.signature_models import SignatureHedge, WindowedSignatureHedge
from models.rl_agents import PPOHedge, DDPGHedge, MCPGHedge
from train.trainer import DeepHedgingTrainer
from train.optuna_tuning import OptunaHyperparameterTuner
from eval.evaluator import HedgingEvaluator, compare_strategies
from eval.plotting import plot_pnl_histogram, plot_metrics_comparison


def run_signature_experiments(config: Config, logger: ExperimentLogger, 
                              train_loader, val_loader, test_loader, 
                              input_dim: int, n_steps: int):
    """Run signature-based model experiments."""
    logger.info("\n" + "=" * 60)
    logger.info("SIGNATURE MODELS")
    logger.info("=" * 60)
    
    device = config.training.device
    results = {}
    
    # Test different signature depths
    for sig_depth in [3, 4, 5]:
        logger.info(f"\n--- Signature Depth {sig_depth} ---")
        
        try:
            model = SignatureHedge(
                input_dim=input_dim,
                sig_depth=sig_depth,
                hidden_dim=64,
                n_layers=2,
                use_time_augmentation=True
            ).to(device)
            
            trainer = DeepHedgingTrainer(
                model=model,
                lambda_risk=config.training.lambda_risk,
                learning_rate=0.001,
                device=device
            )
            
            history = trainer.fit(train_loader, val_loader, n_epochs=30, patience=5, verbose=True)
            metrics, pnl, deltas = trainer.evaluate(test_loader)
            
            results[f'sig_depth_{sig_depth}'] = {
                'metrics': metrics,
                'pnl': pnl
            }
            
            logger.info(f"  Mean P&L: {metrics['mean_pnl']:.4f}")
            logger.info(f"  CVaR 95%: {metrics['cvar_95']:.4f}")
            
        except Exception as e:
            logger.warning(f"  Failed: {e}")
    
    return results


def run_transformer_experiments(config: Config, logger: ExperimentLogger,
                                train_loader, val_loader, test_loader,
                                input_dim: int, n_steps: int):
    """Run transformer-based model experiments."""
    logger.info("\n" + "=" * 60)
    logger.info("TRANSFORMER MODELS")
    logger.info("=" * 60)
    
    device = config.training.device
    results = {}
    
    # Standard Transformer
    logger.info("\n--- Standard Transformer ---")
    try:
        model = TransformerHedge(
            input_dim=input_dim,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        ).to(device)
        
        trainer = DeepHedgingTrainer(
            model=model,
            lambda_risk=config.training.lambda_risk,
            learning_rate=0.0005,
            device=device
        )
        
        history = trainer.fit(train_loader, val_loader, n_epochs=30, patience=5, verbose=True)
        metrics, pnl, deltas = trainer.evaluate(test_loader)
        
        results['transformer'] = {'metrics': metrics, 'pnl': pnl}
        logger.info(f"  Mean P&L: {metrics['mean_pnl']:.4f}")
        
    except Exception as e:
        logger.warning(f"  Failed: {e}")
    
    # SigFormer
    logger.info("\n--- SigFormer ---")
    try:
        model = SigFormer(
            input_dim=input_dim,
            sig_depth=3,
            d_model=64,
            n_heads=4,
            n_layers=2
        ).to(device)
        
        trainer = DeepHedgingTrainer(
            model=model,
            lambda_risk=config.training.lambda_risk,
            learning_rate=0.0005,
            device=device
        )
        
        history = trainer.fit(train_loader, val_loader, n_epochs=30, patience=5, verbose=True)
        metrics, pnl, deltas = trainer.evaluate(test_loader)
        
        results['sigformer'] = {'metrics': metrics, 'pnl': pnl}
        logger.info(f"  Mean P&L: {metrics['mean_pnl']:.4f}")
        
    except Exception as e:
        logger.warning(f"  Failed: {e}")
    
    return results


def run_rl_experiments(config: Config, logger: ExperimentLogger,
                       train_loader, val_loader, test_loader,
                       input_dim: int, n_steps: int):
    """Run reinforcement learning agent experiments."""
    logger.info("\n" + "=" * 60)
    logger.info("REINFORCEMENT LEARNING AGENTS")
    logger.info("=" * 60)
    
    device = config.training.device
    results = {}
    
    # PPO
    logger.info("\n--- PPO Agent ---")
    try:
        model = PPOHedge(
            state_dim=input_dim,
            hidden_dim=64,
            lr_policy=3e-4,
            lr_value=1e-3
        ).to(device)
        
        # Simple evaluation (RL agents need custom training)
        model.eval()
        test_features = next(iter(test_loader))['features'].to(device)
        with torch.no_grad():
            deltas = model(test_features).cpu().numpy()
        
        logger.info(f"  PPO model created, delta range: [{deltas.min():.3f}, {deltas.max():.3f}]")
        results['ppo'] = {'model': model}
        
    except Exception as e:
        logger.warning(f"  Failed: {e}")
    
    # DDPG
    logger.info("\n--- DDPG Agent ---")
    try:
        model = DDPGHedge(
            state_dim=input_dim,
            hidden_dim=64
        ).to(device)
        
        logger.info(f"  DDPG model created")
        results['ddpg'] = {'model': model}
        
    except Exception as e:
        logger.warning(f"  Failed: {e}")
    
    return results


def run_optuna_tuning(config: Config, logger: ExperimentLogger,
                      train_loader, val_loader,
                      input_dim: int, n_steps: int):
    """Run Optuna hyperparameter optimization."""
    logger.info("\n" + "=" * 60)
    logger.info("OPTUNA HYPERPARAMETER TUNING")
    logger.info("=" * 60)
    
    tuner = OptunaHyperparameterTuner(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        n_steps=n_steps,
        device=config.training.device,
        n_epochs=20,
        patience=5
    )
    
    # Tune Deep Hedging
    logger.info("\n--- Tuning Deep Hedging ---")
    try:
        dh_results = tuner.optimize(model_type='deep_hedging', n_trials=20)
        logger.info(f"  Best params: {dh_results['best_params']}")
        logger.info(f"  Best value: {dh_results['best_value']:.4f}")
    except Exception as e:
        logger.warning(f"  Failed: {e}")
        dh_results = None
    
    # Tune LSTM
    logger.info("\n--- Tuning LSTM ---")
    try:
        lstm_results = tuner.optimize(model_type='lstm', n_trials=20)
        logger.info(f"  Best params: {lstm_results['best_params']}")
        logger.info(f"  Best value: {lstm_results['best_value']:.4f}")
    except Exception as e:
        logger.warning(f"  Failed: {e}")
        lstm_results = None
    
    return {'deep_hedging': dh_results, 'lstm': lstm_results}


def main():
    parser = argparse.ArgumentParser(description="Enhanced Model Experiments")
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'signature', 'transformer', 'rl', 'optuna'],
                       help='Experiment to run')
    parser.add_argument('--save_dir', type=str, default='../experiments')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_train', type=int, default=50000)
    parser.add_argument('--n_epochs', type=int, default=30)
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    config = Config()
    config.training.n_train = args.n_train
    config.training.n_epochs = args.n_epochs
    
    logger = ExperimentLogger("enhancements", base_dir=args.save_dir)
    logger.log_config(config.to_dict())
    
    # Generate data
    logger.info("Generating data...")
    generator = DataGenerator(
        n_steps=config.market.n_steps,
        T=config.market.T,
        S0=config.market.S0,
        K=config.market.K
    )
    
    train_data, val_data, test_data = generator.generate_train_val_test(
        n_train=args.n_train,
        n_val=10000,
        n_test=50000,
        base_seed=args.seed
    )
    
    train_loader, val_loader, test_loader = generator.get_dataloaders(
        train_data, val_data, test_data,
        batch_size=config.training.batch_size
    )
    
    input_dim = train_data.n_features
    n_steps = train_data.n_steps
    
    all_results = {}
    
    # Run experiments
    if args.experiment in ['all', 'signature']:
        all_results['signature'] = run_signature_experiments(
            config, logger, train_loader, val_loader, test_loader, input_dim, n_steps
        )
    
    if args.experiment in ['all', 'transformer']:
        all_results['transformer'] = run_transformer_experiments(
            config, logger, train_loader, val_loader, test_loader, input_dim, n_steps
        )
    
    if args.experiment in ['all', 'rl']:
        all_results['rl'] = run_rl_experiments(
            config, logger, train_loader, val_loader, test_loader, input_dim, n_steps
        )
    
    if args.experiment in ['all', 'optuna']:
        all_results['optuna'] = run_optuna_tuning(
            config, logger, train_loader, val_loader, input_dim, n_steps
        )
    
    # Save results
    logger.save_results({k: str(v) for k, v in all_results.items()})
    
    logger.info("\n" + "=" * 60)
    logger.info("ENHANCEMENT EXPERIMENTS COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
