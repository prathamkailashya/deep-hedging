"""
Hyperparameter Optimization using Optuna.

Provides automated hyperparameter tuning for deep hedging models.
"""

import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Any
import numpy as np

try:
    from ..models.deep_hedging import DeepHedgingModel
    from ..models.kozyra_models import HedgingRNN, HedgingLSTM
    from .trainer import DeepHedgingTrainer
    from .losses import HedgingLoss
except ImportError:
    from models.deep_hedging import DeepHedgingModel
    from models.kozyra_models import HedgingRNN, HedgingLSTM
    from train.trainer import DeepHedgingTrainer
    from train.losses import HedgingLoss


class OptunaHyperparameterTuner:
    """
    Hyperparameter tuner using Optuna.
    
    Optimizes model architecture and training hyperparameters
    to minimize validation loss.
    """
    
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        input_dim: int,
        n_steps: int,
        cost_multiplier: float = 0.0,
        lambda_risk: float = 1.0,
        device: str = 'cpu',
        n_epochs: int = 30,
        patience: int = 5
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.cost_multiplier = cost_multiplier
        self.lambda_risk = lambda_risk
        self.device = device
        self.n_epochs = n_epochs
        self.patience = patience
        
        self.best_params = None
        self.best_value = float('inf')
    
    def _create_model(self, trial: Trial, model_type: str) -> nn.Module:
        """Create model with trial-suggested hyperparameters."""
        
        if model_type == 'deep_hedging':
            share_weights = trial.suggest_categorical('share_weights', [True, False])
            
            model = DeepHedgingModel(
                input_dim=self.input_dim,
                n_steps=self.n_steps,
                cost_multiplier=self.cost_multiplier,
                lambda_risk=self.lambda_risk,
                share_weights=share_weights
            )
        
        elif model_type == 'rnn':
            hidden_size = trial.suggest_int('hidden_size', 32, 128, step=16)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            
            model = HedgingRNN(
                state_dim=self.input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers
            )
        
        elif model_type == 'lstm':
            hidden_size = trial.suggest_int('hidden_size', 32, 128, step=16)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout = trial.suggest_float('dropout', 0.0, 0.3, step=0.1)
            
            model = HedgingLSTM(
                state_dim=self.input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def _objective(self, trial: Trial, model_type: str) -> float:
        """Optuna objective function."""
        
        # Hyperparameters
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        # Create model
        model = self._create_model(trial, model_type)
        
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = HedgingLoss(
            cost_multiplier=self.cost_multiplier,
            lambda_risk=self.lambda_risk,
            risk_measure='entropic'
        )
        
        model = model.to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            # Training
            model.train()
            for batch in self.train_loader:
                features = batch['features'].to(self.device)
                stock_paths = batch['stock_paths'].to(self.device)
                payoffs = batch['payoff'].to(self.device)
                
                optimizer.zero_grad()
                deltas = model(features)
                loss = loss_fn(deltas, stock_paths, payoffs)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0.0
            n_batches = 0
            
            with torch.no_grad():
                for batch in self.val_loader:
                    features = batch['features'].to(self.device)
                    stock_paths = batch['stock_paths'].to(self.device)
                    payoffs = batch['payoff'].to(self.device)
                    
                    deltas = model(features)
                    loss = loss_fn(deltas, stock_paths, payoffs)
                    val_loss += loss.item()
                    n_batches += 1
            
            val_loss /= n_batches
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
            
            # Pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_loss
    
    def optimize(
        self,
        model_type: str = 'deep_hedging',
        n_trials: int = 50,
        timeout: Optional[int] = None,
        n_jobs: int = 1
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            model_type: Type of model ('deep_hedging', 'rnn', 'lstm')
            n_trials: Number of trials
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs
        
        Returns:
            Dictionary with best parameters and study results
        """
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        
        study.optimize(
            lambda trial: self._objective(trial, model_type),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }
    
    def get_best_model(self, model_type: str) -> nn.Module:
        """Create model with best hyperparameters."""
        if self.best_params is None:
            raise RuntimeError("Must run optimize() first")
        
        class MockTrial:
            def __init__(self, params):
                self.params = params
            
            def suggest_categorical(self, name, choices):
                return self.params.get(name, choices[0])
            
            def suggest_int(self, name, low, high, step=1):
                return self.params.get(name, low)
            
            def suggest_float(self, name, low, high, step=None, log=False):
                return self.params.get(name, low)
        
        mock_trial = MockTrial(self.best_params)
        return self._create_model(mock_trial, model_type)


def tune_deep_hedging(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    n_steps: int,
    n_trials: int = 50,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Convenience function for tuning Deep Hedging model.
    """
    tuner = OptunaHyperparameterTuner(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        n_steps=n_steps,
        device=device
    )
    
    return tuner.optimize(model_type='deep_hedging', n_trials=n_trials)


def tune_kozyra_rnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    n_steps: int,
    n_trials: int = 50,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Convenience function for tuning Kozyra RNN model.
    """
    tuner = OptunaHyperparameterTuner(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        n_steps=n_steps,
        device=device
    )
    
    return tuner.optimize(model_type='lstm', n_trials=n_trials)
