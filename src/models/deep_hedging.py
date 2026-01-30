"""
Deep Hedging Model Implementation (Buehler et al.)

Semi-recurrent policy:
    δ_k = F_{θ_k}(I_0, ..., I_k, δ_{k-1})

Per-time-step neural network architecture:
- Layers: 3
- Dimensions: N_0 = 2d, N_1 = N_2 = d + 15, N_3 = d
- Activation: ReLU
- Batch Normalization before activation

Reference: Buehler et al. (Deep Hedging), Section 5
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import numpy as np


class TimeStepNetwork(nn.Module):
    """
    Neural network for a single time step in the semi-recurrent model.
    
    Architecture (Buehler et al.):
    - Input: [market_features, previous_delta] -> dimension 2d
    - Layer 1: Linear(2d, d+15) -> BatchNorm -> ReLU
    - Layer 2: Linear(d+15, d+15) -> BatchNorm -> ReLU  
    - Layer 3: Linear(d+15, d) -> output delta
    
    Where d = input feature dimension
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        
        d = input_dim
        # Following Buehler et al. architecture exactly
        # N_0 = 2d (input includes previous delta)
        # N_1 = N_2 = d + 15
        # N_3 = d (output)
        
        hidden_dim = d + 15
        
        self.layers = nn.Sequential(
            # Layer 1
            nn.Linear(2 * d, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            # Layer 3 (output)
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, market_features: torch.Tensor, prev_delta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            market_features: Market features at time k, shape (batch, d)
            prev_delta: Previous delta position, shape (batch, d) or (batch, 1)
        
        Returns:
            delta: Hedging position, shape (batch, 1)
        """
        # Expand prev_delta if necessary
        if prev_delta.dim() == 1:
            prev_delta = prev_delta.unsqueeze(-1)
        
        # Repeat prev_delta to match market features dimension
        if prev_delta.size(-1) == 1 and market_features.size(-1) > 1:
            prev_delta = prev_delta.expand(-1, market_features.size(-1))
        
        # Concatenate inputs
        x = torch.cat([market_features, prev_delta], dim=-1)
        
        return self.layers(x)


class SemiRecurrentNetwork(nn.Module):
    """
    Semi-recurrent network for Deep Hedging.
    
    Uses separate neural networks for each time step, but shares the
    previous delta as a recurrent state.
    
    CRITICAL: Delta outputs are bounded using tanh * scale to prevent
    explosion (Kozyra thesis observation).
    """
    
    def __init__(self, input_dim: int, n_steps: int, share_weights: bool = False,
                 delta_scale: float = 1.5):
        """
        Args:
            input_dim: Dimension of market features at each time step
            n_steps: Number of hedging periods
            share_weights: If True, use same network for all time steps
            delta_scale: Maximum absolute delta value (tanh scaling)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.share_weights = share_weights
        self.delta_scale = delta_scale
        
        if share_weights:
            # Single network shared across all time steps
            self.network = TimeStepNetwork(input_dim, output_dim=1)
        else:
            # Separate network for each time step
            self.networks = nn.ModuleList([
                TimeStepNetwork(input_dim, output_dim=1)
                for _ in range(n_steps)
            ])
    
    def forward(self, features: torch.Tensor, 
                initial_delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through all time steps.
        
        Args:
            features: Market features, shape (batch, n_steps, input_dim)
            initial_delta: Initial position (default: 0), shape (batch, 1)
        
        Returns:
            deltas: Hedging positions, shape (batch, n_steps)
                    BOUNDED by tanh * delta_scale to prevent explosion
        """
        batch_size = features.size(0)
        device = features.device
        
        if initial_delta is None:
            initial_delta = torch.zeros(batch_size, 1, device=device)
        
        deltas = []
        prev_delta = initial_delta
        
        for k in range(self.n_steps):
            market_feat = features[:, k, :]  # (batch, input_dim)
            
            if self.share_weights:
                raw_delta = self.network(market_feat, prev_delta)
            else:
                raw_delta = self.networks[k](market_feat, prev_delta)
            
            # CRITICAL: Bound delta with tanh to prevent explosion
            delta = self.delta_scale * torch.tanh(raw_delta)
            
            deltas.append(delta)
            prev_delta = delta
        
        return torch.cat(deltas, dim=-1)  # (batch, n_steps)


class DeepHedgingModel(nn.Module):
    """
    Complete Deep Hedging model with P&L computation and loss functions.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_steps: int,
        cost_multiplier: float = 0.0,
        lambda_risk: float = 1.0,
        share_weights: bool = False,
        delta_scale: float = 1.5
    ):
        """
        Args:
            input_dim: Dimension of market features
            n_steps: Number of hedging periods
            cost_multiplier: Proportional transaction cost κ
            lambda_risk: Risk aversion parameter λ for entropic risk
            share_weights: Whether to share weights across time steps
            delta_scale: Maximum absolute delta (tanh bound), default 1.5
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.cost_multiplier = cost_multiplier
        self.lambda_risk = lambda_risk
        self.delta_scale = delta_scale
        
        self.network = SemiRecurrentNetwork(input_dim, n_steps, share_weights, delta_scale)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute hedging positions.
        
        Args:
            features: Market features, shape (batch, n_steps, input_dim)
        
        Returns:
            deltas: Hedging positions, shape (batch, n_steps)
        """
        return self.network(features)
    
    def compute_pnl(
        self,
        deltas: torch.Tensor,
        stock_paths: torch.Tensor,
        payoffs: torch.Tensor,
        include_costs: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute P&L for given hedging strategy.
        
        P&L = -Z + Σ δ_k(S_{k+1} - S_k) - Σ κ|δ_{k+1} - δ_k|
        
        Args:
            deltas: Hedging positions, shape (batch, n_steps)
            stock_paths: Stock prices, shape (batch, n_steps + 1)
            payoffs: Option payoffs, shape (batch,)
            include_costs: Whether to include transaction costs
        
        Returns:
            pnl: P&L for each path, shape (batch,)
            info: Dictionary with components
        """
        # Price changes: S_{k+1} - S_k
        price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]  # (batch, n_steps)
        
        # Hedging gains: Σ δ_k(S_{k+1} - S_k)
        hedging_gains = torch.sum(deltas * price_changes, dim=1)  # (batch,)
        
        # Transaction costs: Σ κ|δ_{k+1} - δ_k|
        # Include initial trade from 0 and final trade to 0
        deltas_extended = torch.cat([
            torch.zeros(deltas.size(0), 1, device=deltas.device),
            deltas,
            torch.zeros(deltas.size(0), 1, device=deltas.device)
        ], dim=1)
        delta_changes = torch.abs(deltas_extended[:, 1:] - deltas_extended[:, :-1])
        transaction_costs = self.cost_multiplier * torch.sum(delta_changes, dim=1)
        
        # P&L
        if include_costs:
            pnl = -payoffs + hedging_gains - transaction_costs
        else:
            pnl = -payoffs + hedging_gains
        
        info = {
            'hedging_gains': hedging_gains,
            'transaction_costs': transaction_costs,
            'trading_volume': torch.sum(delta_changes, dim=1)
        }
        
        return pnl, info
    
    def entropic_loss(self, pnl: torch.Tensor) -> torch.Tensor:
        """
        Compute entropic risk loss.
        
        J(θ) = E[exp(-λ * P&L)]
        
        We minimize this, which is equivalent to maximizing expected utility.
        """
        # Use log-sum-exp trick for numerical stability
        scaled_pnl = -self.lambda_risk * pnl
        max_val = torch.max(scaled_pnl)
        loss = max_val + torch.log(torch.mean(torch.exp(scaled_pnl - max_val)))
        return loss
    
    def cvar_loss(self, pnl: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
        """
        Compute CVaR loss (on negative P&L = losses).
        
        CVaR_α(L) = E[L | L >= VaR_α(L)]
        """
        losses = -pnl
        k = int((1 - alpha) * losses.size(0))
        if k == 0:
            k = 1
        
        # Get k largest losses
        top_losses, _ = torch.topk(losses, k)
        return torch.mean(top_losses)
    
    def combined_loss(
        self,
        pnl: torch.Tensor,
        alpha_cvar: float = 0.95,
        weight_entropic: float = 0.5,
        weight_cvar: float = 0.5
    ) -> torch.Tensor:
        """Combined entropic and CVaR loss."""
        entropic = self.entropic_loss(pnl)
        cvar = self.cvar_loss(pnl, alpha_cvar)
        return weight_entropic * entropic + weight_cvar * cvar
    
    def get_indifference_price(self, pnl: torch.Tensor) -> torch.Tensor:
        """
        Compute indifference price.
        
        π(-Z) = (1/λ) * log(E[exp(-λ * P&L)])
        """
        return self.entropic_loss(pnl) / self.lambda_risk


class DeepHedgingWithState(nn.Module):
    """
    Deep Hedging model with explicit state tracking for analysis.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_steps: int,
        hidden_dim: int = 32,
        cost_multiplier: float = 0.0,
        lambda_risk: float = 1.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
        self.cost_multiplier = cost_multiplier
        self.lambda_risk = lambda_risk
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Delta predictor
        self.delta_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with state tracking.
        
        Returns:
            deltas: Hedging positions
            states: List of hidden states at each time step
        """
        batch_size = features.size(0)
        device = features.device
        
        deltas = []
        states = []
        prev_delta = torch.zeros(batch_size, 1, device=device)
        
        for k in range(self.n_steps):
            market_feat = features[:, k, :]
            x = torch.cat([market_feat, prev_delta], dim=-1)
            
            state = self.encoder(x)
            delta = self.delta_head(state)
            
            deltas.append(delta)
            states.append(state)
            prev_delta = delta
        
        return torch.cat(deltas, dim=-1), states
