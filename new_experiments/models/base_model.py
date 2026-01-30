"""
Base Model Class for Fair Comparison

All models inherit from this base and use:
- Bounded delta output
- Same initialization
- Same forward interface
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseHedgingModel(nn.Module, ABC):
    """
    Base class for all hedging models.
    
    Enforces:
    - Bounded deltas via tanh
    - Consistent interface
    - Proper initialization
    """
    
    def __init__(self, delta_max: float = 1.5):
        super().__init__()
        self.delta_max = delta_max
    
    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate hedging deltas.
        
        Args:
            features: (batch, n_steps, input_dim)
            
        Returns:
            deltas: (batch, n_steps) - bounded by delta_max
        """
        pass
    
    def bound_delta(self, raw_delta: torch.Tensor) -> torch.Tensor:
        """Apply tanh bounding to raw delta output."""
        return self.delta_max * torch.tanh(raw_delta)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for logging."""
        return {
            'class': self.__class__.__name__,
            'delta_max': self.delta_max,
            'n_parameters': self.count_parameters()
        }
    
    @staticmethod
    def init_weights(module: nn.Module):
        """Standard weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)


class LSTMHedger(BaseHedgingModel):
    """
    Paper-faithful LSTM hedger (Kozyra baseline).
    
    Architecture:
    - 2-layer LSTM with hidden_size=50
    - FC output to scalar delta
    - He initialization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.0,
        delta_max: float = 1.5
    ):
        super().__init__(delta_max)
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
        self.apply(self.init_weights)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate bounded deltas using LSTM."""
        batch, n_steps, _ = features.shape
        
        lstm_out, _ = self.lstm(features)
        raw_deltas = self.fc(lstm_out).squeeze(-1)
        
        return self.bound_delta(raw_deltas)
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        })
        return config


class DeepHedgingMLP(BaseHedgingModel):
    """
    Paper-faithful Deep Hedging MLP (per-step network).
    
    Architecture per paper:
    - 3 layers: N0=2d, N1=N2=d+15, N3=d
    - ReLU + BatchNorm
    """
    
    def __init__(
        self,
        input_dim: int,
        n_steps: int = 30,
        dropout: float = 0.0,
        delta_max: float = 1.5,
        shared_weights: bool = False
    ):
        super().__init__(delta_max)
        
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.shared_weights = shared_weights
        
        # Paper architecture: 3 layers
        d = input_dim
        layer_dims = [2 * d, d + 15, d + 15, d]
        
        if shared_weights:
            # Single network for all timesteps
            self.networks = nn.ModuleList([self._make_network(layer_dims, dropout)])
        else:
            # Separate network per timestep
            self.networks = nn.ModuleList([
                self._make_network(layer_dims, dropout) for _ in range(n_steps)
            ])
        
        self.apply(self.init_weights)
    
    def _make_network(self, layer_dims, dropout):
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(layer_dims[-1], 1))
        return nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate bounded deltas using per-step MLP."""
        batch, n_steps, _ = features.shape
        
        deltas = []
        for t in range(n_steps):
            net_idx = 0 if self.shared_weights else t
            raw_delta = self.networks[net_idx](features[:, t, :])
            deltas.append(raw_delta.squeeze(-1))
        
        raw_deltas = torch.stack(deltas, dim=1)
        return self.bound_delta(raw_deltas)


class IncrementalDeltaWrapper(BaseHedgingModel):
    """
    Wrapper that converts any model to use delta increments.
    
    δ_t = clip(δ_{t-1} + Δδ_t, [-δ_max, δ_max])
    
    This ensures smooth delta paths.
    """
    
    def __init__(
        self,
        base_model: BaseHedgingModel,
        max_increment: float = 0.3,
        delta_max: float = 1.5
    ):
        super().__init__(delta_max)
        self.base_model = base_model
        self.max_increment = max_increment
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate deltas via increments."""
        batch, n_steps, _ = features.shape
        device = features.device
        
        # Get raw outputs from base model (interpreted as increments)
        raw_outputs = self.base_model(features)
        
        # Convert to bounded increments
        increments = self.max_increment * torch.tanh(raw_outputs / self.delta_max)
        
        # Accumulate with clipping
        deltas = torch.zeros(batch, n_steps, device=device)
        prev_delta = torch.zeros(batch, device=device)
        
        for t in range(n_steps):
            new_delta = torch.clamp(prev_delta + increments[:, t], -self.delta_max, self.delta_max)
            deltas[:, t] = new_delta
            prev_delta = new_delta
        
        return deltas
