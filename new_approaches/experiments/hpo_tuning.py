#!/usr/bin/env python3
"""
Hyperparameter Optimization for Novel Deep Hedging Algorithms.

Uses Optuna for Bayesian optimization with pruning.
Focuses on promising candidates identified from initial experiments.

Usage:
    python hpo_tuning.py --model W-DRO-T --n-trials 50
    python hpo_tuning.py --all --n-trials 30
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add paths
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'new_approaches' / 'code'))

from src.utils.seed import set_seed
from src.env.heston import HestonParams
from src.env.data_generator import DataGenerator
from src.train.losses import EntropicLoss, CVaRLoss
from src.models.kozyra_models import HedgingLSTM
from src.models.transformer import TransformerHedge

# Novel algorithms
from w_dro_t import WDROTransformerHedger, WDROTrainer
from rvsn import AdaptiveSignatureHedger, RSVNTrainer
from sac_cvar import CVaRConstrainedSAC, HedgingEnvironment
from three_stage import ThreeStageTrainer
from rse import RegimeSwitchingEnsemble, RSETrainer


class HPOObjective:
    """Optuna objective for hyperparameter optimization."""
    
    def __init__(
        self,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str = 'cpu'
    ):
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.cvar_loss = CVaRLoss(alpha=0.95)
    
    def __call__(self, trial: Trial) -> float:
        """Evaluate hyperparameters."""
        set_seed(42)
        
        # Get model and trainer based on model_name
        if self.model_name == 'LSTM':
            model, trainer = self._create_lstm(trial)
        elif self.model_name == 'Transformer':
            model, trainer = self._create_transformer(trial)
        elif self.model_name == 'W-DRO-T':
            model, trainer = self._create_wdrot(trial)
        elif self.model_name == 'RVSN':
            model, trainer = self._create_rvsn(trial)
        elif self.model_name == '3SCH':
            model, trainer = self._create_3sch(trial)
        elif self.model_name == 'RSE':
            model, trainer = self._create_rse(trial)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Train with pruning
        try:
            trainer.train_full(self.train_loader, self.val_loader)
        except Exception as e:
            print(f"Training failed: {e}")
            return float('inf')
        
        # Evaluate on validation set
        val_cvar = self._evaluate(model)
        
        # Report for pruning
        trial.report(val_cvar, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return val_cvar
    
    def _evaluate(self, model: nn.Module) -> float:
        """Evaluate model on validation set."""
        model.eval()
        all_pnl = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch.get('features', batch.get('stock_paths')).to(self.device)
                prices = batch.get('stock_paths', batch.get('prices')).to(self.device)
                payoff = batch.get('payoff', batch.get('payoffs')).to(self.device)
                
                deltas = model(features)
                
                # Compute P&L
                price_changes = prices[:, 1:] - prices[:, :-1]
                n_steps = min(deltas.shape[1], price_changes.shape[1])
                hedge_gains = (deltas[:, :n_steps] * price_changes[:, :n_steps]).sum(dim=1)
                pnl = -payoff + hedge_gains
                all_pnl.append(pnl.cpu())
        
        pnl = torch.cat(all_pnl)
        cvar = self.cvar_loss(pnl).item()
        return cvar
    
    def _create_lstm(self, trial: Trial):
        """Create LSTM with hyperparameters from trial."""
        hidden_size = trial.suggest_int('hidden_size', 32, 128, step=16)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dropout = trial.suggest_float('dropout', 0.0, 0.3)
        lr = trial.suggest_float('lr_stage1', 1e-4, 1e-2, log=True)
        
        model = HedgingLSTM(
            state_dim=5,
            hidden_size=hidden_size,
            num_layers=num_layers,
            delta_scale=1.5
        ).to(self.device)
        
        trainer = ThreeStageTrainer(
            model,
            lr_stage1=lr,
            lr_stage3=lr / 10,
            epochs_stage1=20,
            epochs_stage2=10,
            epochs_stage3=15,
            device=str(self.device)
        )
        
        return model, trainer
    
    def _create_transformer(self, trial: Trial):
        """Create Transformer with hyperparameters from trial."""
        d_model = trial.suggest_int('d_model', 32, 128, step=16)
        n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
        n_layers = trial.suggest_int('n_layers', 2, 6)
        dropout = trial.suggest_float('dropout', 0.0, 0.3)
        lr = trial.suggest_float('lr_stage1', 1e-4, 1e-2, log=True)
        
        model = TransformerHedge(
            input_dim=5,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        ).to(self.device)
        
        trainer = ThreeStageTrainer(
            model,
            lr_stage1=lr,
            lr_stage3=lr / 10,
            epochs_stage1=20,
            epochs_stage2=10,
            epochs_stage3=15,
            device=str(self.device)
        )
        
        return model, trainer
    
    def _create_wdrot(self, trial: Trial):
        """Create W-DRO-T with hyperparameters from trial."""
        d_model = trial.suggest_int('d_model', 32, 128, step=16)
        n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
        n_layers = trial.suggest_int('n_layers', 2, 5)
        epsilon = trial.suggest_float('epsilon', 0.01, 0.5, log=True)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        model = WDROTransformerHedger(
            input_dim=5,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            epsilon=epsilon,
            delta_max=1.5
        ).to(self.device)
        
        trainer = WDROTrainer(model, lr=lr, device=str(self.device))
        
        return model, trainer
    
    def _create_rvsn(self, trial: Trial):
        """Create RVSN with hyperparameters from trial."""
        max_depth = trial.suggest_int('max_depth', 2, 5)
        hidden_dim = trial.suggest_int('hidden_dim', 32, 128, step=16)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        model = AdaptiveSignatureHedger(
            input_dim=5,
            max_depth=max_depth,
            hidden_dim=hidden_dim,
            delta_max=1.5
        ).to(self.device)
        
        trainer = RSVNTrainer(model, lr=lr, device=str(self.device))
        
        return model, trainer
    
    def _create_3sch(self, trial: Trial):
        """Create 3SCH with hyperparameters from trial."""
        hidden_size = trial.suggest_int('hidden_size', 32, 128, step=16)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        lr_stage1 = trial.suggest_float('lr_stage1', 1e-4, 1e-2, log=True)
        lr_ratio = trial.suggest_float('lr_ratio', 0.05, 0.2)
        epochs_stage2 = trial.suggest_int('epochs_stage2', 10, 30, step=5)
        
        model = HedgingLSTM(
            state_dim=5,
            hidden_size=hidden_size,
            num_layers=num_layers,
            delta_scale=1.5
        ).to(self.device)
        
        trainer = ThreeStageTrainer(
            model,
            lr_stage1=lr_stage1,
            lr_stage2=lr_stage1 * lr_ratio * 5,
            lr_stage3=lr_stage1 * lr_ratio,
            epochs_stage1=30,
            epochs_stage2=epochs_stage2,
            epochs_stage3=20,
            device=str(self.device)
        )
        
        return model, trainer
    
    def _create_rse(self, trial: Trial):
        """Create RSE with hyperparameters from trial."""
        n_regimes = trial.suggest_int('n_regimes', 2, 6)
        gating_hidden = trial.suggest_int('gating_hidden', 16, 64, step=8)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        model = RegimeSwitchingEnsemble(
            input_dim=5,
            n_regimes=n_regimes,
            delta_max=1.5
        ).to(self.device)
        
        trainer = RSETrainer(model, device=str(self.device))
        
        return model, trainer


def run_hpo(
    model_name: str,
    n_trials: int = 50,
    n_train: int = 20000,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Run HPO for a single model."""
    print(f"\n{'='*60}")
    print(f"HPO for {model_name}")
    print(f"{'='*60}")
    
    set_seed(42)
    
    # Generate data
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
        model_type='heston',
        heston_params=heston_params
    )
    
    train_data, val_data, test_data = generator.generate_train_val_test(
        n_train=n_train,
        n_val=5000,
        n_test=10000,
        base_seed=42,
        compute_bs_delta=True
    )
    
    # Augment with BS delta
    for dataset in [train_data, val_data, test_data]:
        if dataset.bs_deltas is not None:
            bs_delta = dataset.bs_deltas.unsqueeze(-1)
            dataset.features = torch.cat([dataset.features, bs_delta], dim=-1)
    
    train_loader, val_loader, test_loader = generator.get_dataloaders(
        train_data, val_data, test_data, batch_size=256
    )
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    )
    
    objective = HPOObjective(
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Best results
    print(f"\nBest trial:")
    print(f"  Value (CVaR95): {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")
    
    return {
        'model': model_name,
        'best_value': study.best_trial.value,
        'best_params': study.best_trial.params,
        'n_trials': n_trials
    }


def main():
    parser = argparse.ArgumentParser(description='HPO for Novel Deep Hedging')
    parser.add_argument('--model', type=str, help='Model to tune')
    parser.add_argument('--all', action='store_true', help='Tune all models')
    parser.add_argument('--n-trials', type=int, default=30, help='Number of trials')
    parser.add_argument('--n-train', type=int, default=20000, help='Training samples')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    args = parser.parse_args()
    
    results_dir = ROOT / 'new_approaches' / 'results' / 'hpo'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        models = ['LSTM', 'Transformer', 'W-DRO-T', 'RVSN', '3SCH', 'RSE']
    elif args.model:
        models = [args.model]
    else:
        models = ['W-DRO-T', '3SCH']  # Default: most promising
    
    all_results = {}
    for model in models:
        try:
            result = run_hpo(model, args.n_trials, args.n_train, args.device)
            all_results[model] = result
        except Exception as e:
            print(f"HPO failed for {model}: {e}")
            all_results[model] = {'error': str(e)}
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(results_dir / f'hpo_results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nHPO results saved to {results_dir}")
    
    # Summary
    print("\n" + "="*60)
    print("HPO SUMMARY")
    print("="*60)
    for model, result in all_results.items():
        if 'error' in result:
            print(f"{model}: FAILED - {result['error']}")
        else:
            print(f"{model}: CVaR95 = {result['best_value']:.4f}")


if __name__ == '__main__':
    main()
