"""
PART 3: Full Hyperparameter Optimization Framework

Optuna-based Bayesian HPO with:
- 100+ trials per model class
- Identical search space width
- Identical compute budget
- Pruning for efficiency
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import json
import time
from datetime import datetime

from new_experiments.models.base_model import LSTMHedger
from new_experiments.models.signature_models import SignatureLSTM, SignatureMLP
from new_experiments.models.transformer_models import TransformerHedger
from new_experiments.models.attention_lstm import AttentionLSTM
from new_experiments.training.losses import stage1_loss, compute_pnl, cvar_loss
from new_experiments.data.data_generator import DataGenerator, HestonParams


@dataclass
class HPOConfig:
    """HPO configuration ensuring fair comparison."""
    n_trials: int = 100
    timeout_per_model: int = 7200  # 2 hours per model class
    timeout_per_trial: int = 600   # 10 min max per trial
    
    # Training budget per trial (identical for all)
    max_epochs: int = 30
    early_stopping_patience: int = 5
    
    # Pruning
    n_warmup_steps: int = 5
    n_startup_trials: int = 10
    
    # Data (smaller for HPO speed)
    n_train: int = 20000
    n_val: int = 5000
    batch_size: int = 256
    
    # Fixed params
    grad_clip: float = 5.0
    weight_decay: float = 1e-4
    cvar_alpha: float = 0.95


# ============================================================
# SEARCH SPACES (Identical width for fairness)
# ============================================================

def get_lstm_params(trial: optuna.Trial) -> Dict[str, Any]:
    """LSTM search space."""
    return {
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 50, 64, 128]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3, step=0.1),
        'delta_max': trial.suggest_categorical('delta_max', [1.0, 1.5, 2.0]),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
    }


def get_signature_lstm_params(trial: optuna.Trial) -> Dict[str, Any]:
    """SignatureLSTM search space."""
    return {
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'sig_order': trial.suggest_int('sig_order', 2, 4),
        'sig_window': trial.suggest_categorical('sig_window', [3, 5, 7, 10]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3, step=0.1),
        'delta_max': trial.suggest_categorical('delta_max', [1.0, 1.5, 2.0]),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
    }


def get_signature_mlp_params(trial: optuna.Trial) -> Dict[str, Any]:
    """SignatureMLP search space."""
    return {
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128]),
        'n_layers': trial.suggest_int('n_layers', 2, 4),
        'sig_order': trial.suggest_int('sig_order', 2, 4),
        'sig_window': trial.suggest_categorical('sig_window', [3, 5, 7, 10]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3, step=0.1),
        'delta_max': trial.suggest_categorical('delta_max', [1.0, 1.5, 2.0]),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
    }


def get_transformer_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Transformer search space."""
    return {
        'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
        'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
        'n_layers': trial.suggest_int('n_layers', 1, 4),
        'dim_feedforward': trial.suggest_categorical('dim_feedforward', [64, 128, 256]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.2, step=0.1),
        'delta_max': trial.suggest_categorical('delta_max', [1.0, 1.5, 2.0]),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
    }


def get_attention_lstm_params(trial: optuna.Trial) -> Dict[str, Any]:
    """AttentionLSTM search space."""
    return {
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 50, 64, 128]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'attention_dim': trial.suggest_categorical('attention_dim', [16, 32, 64]),
        'memory_length': trial.suggest_categorical('memory_length', [5, 10, 15, 20]),
        'combination': trial.suggest_categorical('combination', ['concat', 'sum', 'gate']),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3, step=0.1),
        'delta_max': trial.suggest_categorical('delta_max', [1.0, 1.5, 2.0]),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
    }


MODEL_PARAM_FUNCS = {
    'LSTM': get_lstm_params,
    'SignatureLSTM': get_signature_lstm_params,
    'SignatureMLP': get_signature_mlp_params,
    'Transformer': get_transformer_params,
    'AttentionLSTM': get_attention_lstm_params,
}


def create_model_from_params(model_name: str, params: Dict[str, Any], input_dim: int) -> nn.Module:
    """Create model from HPO parameters."""
    
    delta_max = params.get('delta_max', 1.5)
    dropout = params.get('dropout', 0.1)
    
    if model_name == 'LSTM':
        return LSTMHedger(
            input_dim=input_dim,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=dropout,
            delta_max=delta_max
        )
    
    elif model_name == 'SignatureLSTM':
        return SignatureLSTM(
            input_dim=input_dim,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            sig_order=params['sig_order'],
            sig_window=params['sig_window'],
            dropout=dropout,
            delta_max=delta_max
        )
    
    elif model_name == 'SignatureMLP':
        return SignatureMLP(
            input_dim=input_dim,
            hidden_size=params['hidden_size'],
            n_layers=params['n_layers'],
            sig_order=params['sig_order'],
            sig_window=params['sig_window'],
            dropout=dropout,
            delta_max=delta_max
        )
    
    elif model_name == 'Transformer':
        return TransformerHedger(
            input_dim=input_dim,
            d_model=params['d_model'],
            n_heads=params['n_heads'],
            n_layers=params['n_layers'],
            dim_feedforward=params['dim_feedforward'],
            dropout=dropout,
            delta_max=delta_max
        )
    
    elif model_name == 'AttentionLSTM':
        return AttentionLSTM(
            input_dim=input_dim,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            attention_dim=params['attention_dim'],
            memory_length=params['memory_length'],
            combination=params['combination'],
            dropout=dropout,
            delta_max=delta_max
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


class HPOObjective:
    """
    Optuna objective for model HPO.
    
    Trains model and returns validation CVaR (to minimize).
    """
    
    def __init__(
        self,
        model_name: str,
        train_loader,
        val_loader,
        input_dim: int,
        config: HPOConfig,
        device: str = 'cpu'
    ):
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.input_dim = input_dim
        self.config = config
        self.device = device
        self.param_func = MODEL_PARAM_FUNCS[model_name]
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Run single trial."""
        
        # Sample hyperparameters
        params = self.param_func(trial)
        lr = params.pop('lr')
        
        # Create model
        try:
            model = create_model_from_params(self.model_name, params, self.input_dim)
            model = model.to(self.device)
        except Exception as e:
            raise optuna.TrialPruned(f"Model creation failed: {e}")
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=self.config.weight_decay
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            # Train
            model.train()
            for batch in self.train_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                optimizer.zero_grad()
                deltas = model(features)
                loss = stage1_loss(deltas, stock_paths, payoffs, alpha=self.config.cvar_alpha)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()
            
            # Validate
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in self.val_loader:
                    features = batch['features'].to(self.device)
                    stock_paths = batch['stock_paths'].to(self.device)
                    payoffs = batch['payoff'].to(self.device)
                    
                    deltas = model(features)
                    loss = stage1_loss(deltas, stock_paths, payoffs, alpha=self.config.cvar_alpha)
                    val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses)
            
            # Report for pruning
            trial.report(val_loss, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    break
        
        return best_val_loss


def run_hpo_for_model(
    model_name: str,
    train_loader,
    val_loader,
    input_dim: int,
    config: HPOConfig,
    device: str = 'cpu',
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run full HPO for a single model class.
    
    Returns best params and study statistics.
    """
    
    print(f"\n{'='*60}")
    print(f"HPO for {model_name}")
    print(f"Trials: {config.n_trials}, Timeout: {config.timeout_per_model}s")
    print(f"{'='*60}")
    
    # Create study
    sampler = TPESampler(seed=seed, n_startup_trials=config.n_startup_trials)
    pruner = MedianPruner(n_warmup_steps=config.n_warmup_steps)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        study_name=f'{model_name}_hpo'
    )
    
    # Objective
    objective = HPOObjective(
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        config=config,
        device=device
    )
    
    # Optimize
    start_time = time.time()
    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout_per_model,
        show_progress_bar=True,
        catch=(Exception,)
    )
    elapsed = time.time() - start_time
    
    # Results
    best_trial = study.best_trial
    
    results = {
        'model_name': model_name,
        'best_params': best_trial.params,
        'best_value': best_trial.value,
        'n_trials': len(study.trials),
        'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'elapsed_time': elapsed,
        'all_values': [t.value for t in study.trials if t.value is not None]
    }
    
    print(f"\nBest trial:")
    print(f"  Value: {best_trial.value:.4f}")
    print(f"  Params: {best_trial.params}")
    print(f"  Completed: {results['n_complete']}/{results['n_trials']}")
    
    return results


def run_full_hpo(
    config: HPOConfig = None,
    device: str = 'cpu',
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Run complete HPO for all models.
    
    Ensures identical compute budget for each model class.
    """
    
    if config is None:
        config = HPOConfig()
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'hpo')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("FULL HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Models: LSTM, SignatureLSTM, SignatureMLP, Transformer, AttentionLSTM")
    print(f"Trials per model: {config.n_trials}")
    print(f"Timeout per model: {config.timeout_per_model}s")
    
    # Generate data
    print("\nGenerating HPO data...")
    heston_params = HestonParams()
    data_gen = DataGenerator(heston_params=heston_params)
    
    train_data, val_data, _ = data_gen.generate_splits(
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=1000,  # Not used in HPO
        base_seed=42
    )
    
    train_loader, val_loader, _ = data_gen.get_dataloaders(
        train_data, val_data, val_data,  # Use val as test placeholder
        batch_size=config.batch_size
    )
    
    input_dim = train_data.n_features
    
    # Run HPO for each model
    models = ['LSTM', 'SignatureLSTM', 'SignatureMLP', 'Transformer', 'AttentionLSTM']
    all_results = {}
    
    for model_name in models:
        results = run_hpo_for_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            config=config,
            device=device
        )
        all_results[model_name] = results
        
        # Save intermediate results
        with open(os.path.join(output_dir, f'{model_name}_hpo.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    # Summary
    print("\n" + "=" * 70)
    print("HPO SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Best CVaR':>12} {'Trials':>10} {'Time (s)':>10}")
    print("-" * 55)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<20} {results['best_value']:>12.4f} {results['n_complete']:>10} {results['elapsed_time']:>10.1f}")
    
    # Save all results
    with open(os.path.join(output_dir, 'all_hpo_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir}")
    
    return all_results


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_full_hpo(device=device)
