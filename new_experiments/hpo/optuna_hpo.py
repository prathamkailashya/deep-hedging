"""
Optuna Hyperparameter Optimization Framework

Fair comparison requires:
- ≥100 trials per model class
- Same compute budget
- Best validation CVaR selection
- Full logging of all trials
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Type
from dataclasses import dataclass
import logging

# Suppress optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class HPOResult:
    """Results from hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    study_name: str
    all_trials: list
    total_time: float


class ModelObjective:
    """
    Optuna objective for model hyperparameter optimization.
    
    Evaluates model on validation CVaR after two-stage training.
    """
    
    def __init__(
        self,
        model_class: Type[nn.Module],
        search_space: Dict[str, list],
        train_loader,
        val_loader,
        input_dim: int,
        n_steps: int = 30,
        max_epochs: int = 30,
        patience: int = 5,
        device: str = 'cpu'
    ):
        self.model_class = model_class
        self.search_space = search_space
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters from search space."""
        params = {}
        for name, values in self.search_space.items():
            if isinstance(values[0], bool):
                params[name] = trial.suggest_categorical(name, values)
            elif isinstance(values[0], int):
                params[name] = trial.suggest_categorical(name, values)
            elif isinstance(values[0], float):
                params[name] = trial.suggest_categorical(name, values)
            elif isinstance(values[0], str):
                params[name] = trial.suggest_categorical(name, values)
            else:
                params[name] = trial.suggest_categorical(name, values)
        return params
    
    def _create_model(self, params: Dict[str, Any]) -> nn.Module:
        """Create model with suggested parameters."""
        # Extract model-specific params
        model_params = {k: v for k, v in params.items() 
                       if k not in ['learning_rate', 'batch_size']}
        model_params['input_dim'] = self.input_dim
        
        return self.model_class(**model_params).to(self.device)
    
    def _train_and_evaluate(
        self,
        model: nn.Module,
        params: Dict[str, Any],
        trial: optuna.Trial
    ) -> float:
        """Two-stage training with early stopping and pruning."""
        from ..training.losses import stage1_loss, stage2_loss, compute_pnl, cvar_loss
        
        lr = params.get('learning_rate', 0.001)
        
        # Stage 1: CVaR pretraining
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        best_val_cvar = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            model.train()
            for batch in self.train_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                optimizer.zero_grad()
                deltas = model(features)
                loss = stage1_loss(deltas, stock_paths, payoffs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            
            # Validation
            model.eval()
            val_cvars = []
            with torch.no_grad():
                for batch in self.val_loader:
                    features = batch['features'].to(self.device)
                    stock_paths = batch['stock_paths'].to(self.device)
                    payoffs = batch['payoff'].to(self.device)
                    
                    deltas = model(features)
                    pnl, _ = compute_pnl(deltas, stock_paths, payoffs)
                    val_cvars.append(cvar_loss(pnl).item())
            
            val_cvar = np.mean(val_cvars)
            
            # Report for pruning
            trial.report(val_cvar, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            if val_cvar < best_val_cvar:
                best_val_cvar = val_cvar
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        # Stage 2: Entropic fine-tuning (shorter)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
        
        for epoch in range(self.max_epochs // 2):
            model.train()
            for batch in self.train_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                optimizer.zero_grad()
                deltas = model(features)
                loss = stage2_loss(deltas, stock_paths, payoffs, gamma=1e-3)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            
            # Validation
            model.eval()
            val_cvars = []
            with torch.no_grad():
                for batch in self.val_loader:
                    features = batch['features'].to(self.device)
                    stock_paths = batch['stock_paths'].to(self.device)
                    payoffs = batch['payoff'].to(self.device)
                    
                    deltas = model(features)
                    pnl, _ = compute_pnl(deltas, stock_paths, payoffs)
                    val_cvars.append(cvar_loss(pnl).item())
            
            val_cvar = np.mean(val_cvars)
            
            if val_cvar < best_val_cvar:
                best_val_cvar = val_cvar
        
        return best_val_cvar
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        params = self._suggest_params(trial)
        model = self._create_model(params)
        
        try:
            val_cvar = self._train_and_evaluate(model, params, trial)
            return val_cvar
        except Exception as e:
            print(f"  Trial failed: {e}")
            return float('inf')


def run_hpo(
    model_class: Type[nn.Module],
    model_name: str,
    search_space: Dict[str, list],
    train_loader,
    val_loader,
    input_dim: int,
    n_trials: int = 100,
    timeout: int = 7200,  # 2 hours
    n_jobs: int = 1,
    results_dir: str = 'results',
    device: str = 'cpu'
) -> HPOResult:
    """
    Run Optuna hyperparameter optimization.
    
    Args:
        model_class: Model class to optimize
        model_name: Name for logging
        search_space: Dict of param_name -> list of values
        train_loader: Training data
        val_loader: Validation data
        input_dim: Input feature dimension
        n_trials: Number of trials (≥100 for fair comparison)
        timeout: Max time in seconds
        n_jobs: Parallel jobs
        results_dir: Directory for results
        device: Device to use
        
    Returns:
        HPOResult with best params and all trial info
    """
    print(f"\n{'='*60}")
    print(f"HPO: {model_name}")
    print(f"Trials: {n_trials}, Timeout: {timeout}s")
    print(f"{'='*60}")
    
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Create study
    study = optuna.create_study(
        study_name=f"hpo_{model_name}",
        direction='minimize',  # Minimize CVaR
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    
    # Create objective
    objective = ModelObjective(
        model_class=model_class,
        search_space=search_space,
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        device=device
    )
    
    # Run optimization
    start_time = time.time()
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    total_time = time.time() - start_time
    
    # Collect results
    all_trials = []
    for trial in study.trials:
        all_trials.append({
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': str(trial.state)
        })
    
    result = HPOResult(
        best_params=study.best_params,
        best_value=study.best_value,
        n_trials=len(study.trials),
        study_name=model_name,
        all_trials=all_trials,
        total_time=total_time
    )
    
    # Save results
    with open(results_path / f"hpo_{model_name}.json", 'w') as f:
        json.dump({
            'best_params': result.best_params,
            'best_value': result.best_value,
            'n_trials': result.n_trials,
            'total_time': result.total_time,
            'all_trials': result.all_trials
        }, f, indent=2)
    
    print(f"\n  Best CVaR: {result.best_value:.4f}")
    print(f"  Best params: {result.best_params}")
    print(f"  Total time: {total_time/60:.1f} min")
    
    return result


def run_all_hpo(
    train_loader,
    val_loader,
    input_dim: int,
    n_trials: int = 100,
    results_dir: str = 'results',
    device: str = 'cpu'
) -> Dict[str, HPOResult]:
    """
    Run HPO for all model classes.
    
    Ensures same compute budget for fair comparison.
    """
    from ..models.base_model import LSTMHedger
    from ..models.signature_models import SignatureLSTM, SignatureMLP, SigFormerHedger
    from ..models.transformer_models import TransformerHedger, TimeSeriesTransformer
    from ..models.attention_lstm import AttentionLSTM, MultiHeadAttentionLSTM
    from ..configs.base_config import (
        LSTM_SEARCH_SPACE, SIGNATURE_SEARCH_SPACE,
        TRANSFORMER_SEARCH_SPACE, ATTENTION_LSTM_SEARCH_SPACE
    )
    
    results = {}
    
    # Define model configurations
    models_config = [
        ('LSTM', LSTMHedger, LSTM_SEARCH_SPACE),
        ('SignatureLSTM', SignatureLSTM, SIGNATURE_SEARCH_SPACE),
        ('SignatureMLP', SignatureMLP, SIGNATURE_SEARCH_SPACE),
        ('SigFormer', SigFormerHedger, SIGNATURE_SEARCH_SPACE),
        ('Transformer', TransformerHedger, TRANSFORMER_SEARCH_SPACE),
        ('TimeSeriesTransformer', TimeSeriesTransformer, TRANSFORMER_SEARCH_SPACE),
        ('AttentionLSTM', AttentionLSTM, ATTENTION_LSTM_SEARCH_SPACE),
        ('MultiHeadAttentionLSTM', MultiHeadAttentionLSTM, ATTENTION_LSTM_SEARCH_SPACE),
    ]
    
    # Same timeout per model for fair comparison
    timeout_per_model = 7200  # 2 hours
    
    for name, model_class, search_space in models_config:
        try:
            result = run_hpo(
                model_class=model_class,
                model_name=name,
                search_space=search_space,
                train_loader=train_loader,
                val_loader=val_loader,
                input_dim=input_dim,
                n_trials=n_trials,
                timeout=timeout_per_model,
                results_dir=results_dir,
                device=device
            )
            results[name] = result
        except Exception as e:
            print(f"  HPO failed for {name}: {e}")
    
    return results
