"""
Base Configuration for Fair Deep Hedging Comparison

All models use identical:
- Data generation
- Loss functions  
- Training protocol (two-stage)
- Delta bounding
- Gradient clipping
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class HestonConfig:
    """Heston model parameters."""
    S0: float = 100.0
    v0: float = 0.04
    r: float = 0.0
    kappa: float = 1.0
    theta: float = 0.04
    sigma: float = 0.2
    rho: float = -0.7


@dataclass
class DataConfig:
    """Data generation configuration."""
    n_steps: int = 30
    T: float = 30 / 365  # 30 days
    K: float = 100.0  # ATM strike
    
    n_train: int = 90000
    n_val: int = 10000
    n_test: int = 100000
    
    batch_size: int = 256
    seed: int = 42


@dataclass
class TrainingConfig:
    """Training configuration - same for all models."""
    # Stage 1: CVaR pretraining
    stage1_epochs: int = 50
    stage1_lr: float = 0.001
    stage1_patience: int = 15
    
    # Stage 2: Entropic fine-tuning
    stage2_epochs: int = 30
    stage2_lr: float = 0.0001
    stage2_patience: int = 10
    
    # Trading penalty
    gamma: float = 1e-3
    nu: float = 1e8
    band_width: float = 0.15
    
    # Regularization
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    
    # Risk measure
    lambda_risk: float = 1.0
    cvar_alpha: float = 0.95


@dataclass
class ModelConfig:
    """Base model configuration."""
    delta_max: float = 1.5
    dropout: float = 0.1


@dataclass
class HPOConfig:
    """Hyperparameter optimization configuration."""
    n_trials: int = 100
    timeout_per_trial: int = 600  # 10 min max per trial
    sampler: str = "TPE"  # Tree-structured Parzen Estimator
    pruner: str = "MedianPruner"
    
    # Same compute budget for all models
    max_epochs_per_trial: int = 30
    early_stopping_patience: int = 5


@dataclass  
class EvaluationConfig:
    """Evaluation configuration."""
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    
    metrics: List[str] = field(default_factory=lambda: [
        'mean_pnl', 'std_pnl', 'var_95', 'var_99', 
        'cvar_95', 'cvar_99', 'entropic_risk',
        'trading_volume', 'max_delta', 'delta_smoothness'
    ])


# HPO Search Spaces for each model class
LSTM_SEARCH_SPACE = {
    'hidden_size': [32, 50, 64, 128],
    'num_layers': [1, 2, 3],
    'dropout': [0.0, 0.1, 0.2, 0.3],
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
    'delta_max': [1.0, 1.5, 2.0],
    'batch_size': [128, 200, 256],
}

SIGNATURE_SEARCH_SPACE = {
    'sig_order': [2, 3, 4],
    'sig_window': [3, 5, 7, 10],
    'hidden_size': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'fc_layers': [1, 2, 3],
    'dropout': [0.0, 0.1, 0.2, 0.3],
    'learning_rate': [0.0001, 0.0005, 0.001],
    'delta_max': [1.0, 1.5, 2.0],
    'normalize_sig': [True, False],
}

TRANSFORMER_SEARCH_SPACE = {
    'd_model': [32, 64, 128],
    'n_heads': [2, 4, 8],
    'n_layers': [2, 4, 6],
    'dim_feedforward': [64, 128, 256],
    'dropout': [0.0, 0.1, 0.2],
    'learning_rate': [0.0001, 0.0005, 0.001],
    'delta_max': [1.0, 1.5, 2.0],
}

ATTENTION_LSTM_SEARCH_SPACE = {
    'hidden_size': [32, 50, 64, 128],
    'num_layers': [1, 2, 3],
    'attention_dim': [16, 32, 64],
    'memory_length': [5, 10, 15, 30],
    'combination': ['concat', 'sum', 'gate'],
    'dropout': [0.0, 0.1, 0.2],
    'learning_rate': [0.0001, 0.0005, 0.001],
    'delta_max': [1.0, 1.5, 2.0],
}

RL_SEARCH_SPACE = {
    'hidden_size': [64, 128, 256],
    'lr_policy': [1e-5, 3e-5, 1e-4, 3e-4],
    'lr_value': [1e-4, 3e-4, 1e-3],
    'gamma_discount': [0.95, 0.99, 0.995],
    'entropy_coef': [0.0, 0.01, 0.05],
    'action_penalty': [1e-4, 1e-3, 1e-2],
    'clip_epsilon': [0.1, 0.2, 0.3],
    'n_epochs': [3, 5, 10],
}

ENSEMBLE_SEARCH_SPACE = {
    'method': ['median', 'mean', 'weighted', 'learned'],
    'n_models': [3, 5, 7],
    'weight_decay': [0.0, 1e-4, 1e-3],
}
