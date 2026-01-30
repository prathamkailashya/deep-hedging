"""
Configuration and seed management for reproducibility.
"""

import os
import random
import yaml
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


@dataclass
class MarketConfig:
    """Market and model parameters."""
    # Time grid (Buehler et al. Section 5)
    n_steps: int = 30
    T: float = 30 / 365  # 30 days to expiry
    
    # Heston model parameters
    S0: float = 100.0
    v0: float = 0.04  # Initial variance
    r: float = 0.0    # Risk-free rate
    kappa: float = 1.0  # Mean reversion speed
    theta: float = 0.04  # Long-term variance
    sigma: float = 0.2   # Vol of vol
    rho: float = -0.7    # Correlation
    
    # Option parameters
    K: float = 100.0  # Strike price
    
    # Transaction costs
    cost_multiplier: float = 0.0  # Proportional cost κ


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Dataset sizes (Buehler et al.)
    n_train: int = 90000
    n_val: int = 10000
    n_test: int = 100000
    
    # Training parameters (Buehler et al.)
    batch_size: int = 256
    learning_rate: float = 0.005
    n_epochs: int = 100
    
    # Risk parameters
    lambda_risk: float = 1.0  # Entropic risk aversion
    cvar_alpha: float = 0.95  # CVaR confidence level
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-6
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Seed
    seed: int = 42


@dataclass
class KozyraConfig:
    """Kozyra-specific training parameters."""
    # RNN parameters
    hidden_size: int = 50
    num_layers: int = 2
    
    # Training (Kozyra)
    batch_size: int = 200
    learning_rate: float = 0.0005
    
    # Two-stage training parameters
    gamma: float = 1e-3  # Transaction cost penalty
    nu: float = 1e8      # No-transaction band penalty
    band_width: float = 0.15  # No-transaction band half-width
    
    # Extended network architecture
    layers: list = field(default_factory=lambda: [1, 50, 50, 30, 1, 50, 50, 30, 1])


@dataclass
class Config:
    """Master configuration."""
    market: MarketConfig = field(default_factory=MarketConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    kozyra: KozyraConfig = field(default_factory=KozyraConfig)
    
    # Experiment settings
    experiment_name: str = "deep_hedging"
    save_dir: str = "experiments"
    
    def __post_init__(self):
        set_seed(self.training.seed)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'market': self.market.__dict__,
            'training': self.training.__dict__,
            'kozyra': {k: v for k, v in self.kozyra.__dict__.items() if k != 'layers'} | {'layers': self.kozyra.layers},
            'experiment_name': self.experiment_name,
            'save_dir': self.save_dir
        }
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        if 'market' in data:
            config.market = MarketConfig(**data['market'])
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        if 'kozyra' in data:
            config.kozyra = KozyraConfig(**data['kozyra'])
        if 'experiment_name' in data:
            config.experiment_name = data['experiment_name']
        if 'save_dir' in data:
            config.save_dir = data['save_dir']
        
        return config
