"""
Market Environment for Deep Hedging.

Implements the hedging problem with transaction costs as specified in 
Buehler et al. (Deep Hedging).
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .heston import HestonModel, HestonParams, BlackScholesModel


@dataclass
class HedgingState:
    """State representation for the hedging problem."""
    time_idx: int           # Current time step index
    stock_price: np.ndarray # Current stock price
    variance: np.ndarray    # Current variance (Heston)
    delta: np.ndarray       # Current position in underlying
    ttm: float              # Time to maturity
    
    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """Convert state to tensor for neural network input."""
        features = np.stack([
            self.stock_price,
            np.full_like(self.stock_price, self.ttm),
            self.delta
        ], axis=-1)
        return torch.tensor(features, dtype=torch.float32, device=device)


class MarketEnvironment:
    """
    Market environment for the hedging problem.
    
    Problem setup:
    - Short one European call option with payoff Z = max(S_T - K, 0)
    - Trading strategy δ_k in the underlying
    - Proportional transaction costs κ|δ_{k+1} - δ_k|
    
    Terminal P&L:
    P&L = -Z + Σ δ_k(S_{k+1} - S_k) - Σ κ|δ_{k+1} - δ_k|
    """
    
    def __init__(
        self,
        n_steps: int = 30,
        T: float = 30/365,
        S0: float = 100.0,
        K: float = 100.0,
        r: float = 0.0,
        cost_multiplier: float = 0.0,
        model_type: str = 'heston',
        heston_params: Optional[HestonParams] = None,
        bs_sigma: float = 0.2
    ):
        """
        Initialize market environment.
        
        Args:
            n_steps: Number of hedging periods (default: 30)
            T: Time to maturity in years (default: 30/365)
            S0: Initial stock price
            K: Option strike price
            r: Risk-free rate
            cost_multiplier: Proportional transaction cost κ
            model_type: 'heston' or 'bs' (Black-Scholes)
            heston_params: Heston model parameters
            bs_sigma: Black-Scholes volatility (if model_type='bs')
        """
        self.n_steps = n_steps
        self.T = T
        self.S0 = S0
        self.K = K
        self.r = r
        self.cost_multiplier = cost_multiplier
        self.model_type = model_type
        
        # Time grid
        self.dt = T / n_steps
        self.time_grid = np.linspace(0, T, n_steps + 1)
        
        # Initialize price model
        if model_type == 'heston':
            if heston_params is None:
                heston_params = HestonParams(S0=S0, r=r)
            self.price_model = HestonModel(heston_params)
        else:
            self.price_model = BlackScholesModel(S0=S0, r=r, sigma=bs_sigma)
        
        # For Black-Scholes delta computation
        self.bs_model = BlackScholesModel(S0=S0, r=r, sigma=bs_sigma)
        
        # State storage
        self.stock_paths = None
        self.variance_paths = None
        self.n_paths = 0
    
    def generate_paths(self, n_paths: int, seed: Optional[int] = None):
        """Generate price paths for simulation."""
        self.n_paths = n_paths
        
        if self.model_type == 'heston':
            self.stock_paths, self.variance_paths = self.price_model.simulate(
                T=self.T, n_steps=self.n_steps, n_paths=n_paths, seed=seed
            )
        else:
            self.stock_paths = self.price_model.simulate(
                T=self.T, n_steps=self.n_steps, n_paths=n_paths, seed=seed
            )
            # Constant variance for BS
            self.variance_paths = np.full_like(
                self.stock_paths, self.price_model.sigma**2
            )
    
    def get_payoff(self) -> np.ndarray:
        """
        Compute option payoff at maturity.
        Z = max(S_T - K, 0) for call option
        """
        if self.stock_paths is None:
            raise ValueError("Must generate paths first")
        
        S_T = self.stock_paths[:, -1]
        return np.maximum(S_T - self.K, 0)
    
    def compute_pnl(
        self,
        deltas: np.ndarray,
        include_costs: bool = True
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute P&L for given hedging strategy.
        
        Args:
            deltas: Hedging positions, shape (n_paths, n_steps)
            include_costs: Whether to include transaction costs
        
        Returns:
            pnl: Terminal P&L for each path
            info: Dictionary with additional information
        """
        if self.stock_paths is None:
            raise ValueError("Must generate paths first")
        
        n_paths, n_steps_plus_1 = self.stock_paths.shape
        n_steps = n_steps_plus_1 - 1
        
        if deltas.shape != (n_paths, n_steps):
            raise ValueError(f"Expected deltas shape {(n_paths, n_steps)}, got {deltas.shape}")
        
        # Option payoff (we are short the option)
        payoff = self.get_payoff()
        
        # Hedging gains: Σ δ_k(S_{k+1} - S_k)
        price_changes = np.diff(self.stock_paths, axis=1)  # (n_paths, n_steps)
        hedging_gains = np.sum(deltas * price_changes, axis=1)
        
        # Transaction costs: Σ κ|δ_{k+1} - δ_k|
        # Include initial trade from 0 and final trade to 0
        delta_extended = np.zeros((n_paths, n_steps + 2))
        delta_extended[:, 1:-1] = deltas
        delta_changes = np.abs(np.diff(delta_extended, axis=1))
        transaction_costs = self.cost_multiplier * np.sum(
            delta_changes * np.hstack([self.stock_paths[:, :1], self.stock_paths]),
            axis=1
        )
        
        # Simpler cost calculation (as in Buehler et al.)
        delta_changes_simple = np.abs(np.diff(
            np.hstack([np.zeros((n_paths, 1)), deltas, np.zeros((n_paths, 1))]),
            axis=1
        ))
        transaction_costs = self.cost_multiplier * np.sum(delta_changes_simple, axis=1)
        
        # P&L = -Z + hedging_gains - transaction_costs
        if include_costs:
            pnl = -payoff + hedging_gains - transaction_costs
        else:
            pnl = -payoff + hedging_gains
        
        info = {
            'payoff': payoff,
            'hedging_gains': hedging_gains,
            'transaction_costs': transaction_costs,
            'trading_volume': np.sum(delta_changes_simple, axis=1)
        }
        
        return pnl, info
    
    def get_features(self, normalize: bool = True) -> np.ndarray:
        """
        Get feature matrix for all time steps.
        
        Features at time k:
        - Normalized stock price: S_k / S_0
        - Time to maturity: (T - t_k) / T
        - Log-moneyness: log(S_k / K)
        - Variance (if Heston): v_k
        
        Returns:
            features: Shape (n_paths, n_steps, n_features)
        """
        if self.stock_paths is None:
            raise ValueError("Must generate paths first")
        
        n_paths = self.stock_paths.shape[0]
        
        # Compute features for each time step (excluding terminal)
        features_list = []
        
        for k in range(self.n_steps):
            t_k = self.time_grid[k]
            ttm = (self.T - t_k) / self.T  # Normalized TTM
            
            S_k = self.stock_paths[:, k]
            
            if normalize:
                norm_price = S_k / self.S0
            else:
                norm_price = S_k
            
            log_moneyness = np.log(S_k / self.K)
            
            if self.model_type == 'heston':
                v_k = self.variance_paths[:, k]
                step_features = np.stack([
                    norm_price,
                    np.full(n_paths, ttm),
                    log_moneyness,
                    v_k
                ], axis=-1)
            else:
                step_features = np.stack([
                    norm_price,
                    np.full(n_paths, ttm),
                    log_moneyness
                ], axis=-1)
            
            features_list.append(step_features)
        
        return np.stack(features_list, axis=1)  # (n_paths, n_steps, n_features)
    
    def get_bs_delta(self) -> np.ndarray:
        """
        Compute Black-Scholes delta for all paths and time steps.
        
        Returns:
            deltas: Shape (n_paths, n_steps)
        """
        if self.stock_paths is None:
            raise ValueError("Must generate paths first")
        
        deltas = np.zeros((self.n_paths, self.n_steps))
        
        for k in range(self.n_steps):
            t_k = self.time_grid[k]
            S_k = self.stock_paths[:, k]
            
            for i in range(self.n_paths):
                deltas[i, k] = self.bs_model.delta(S_k[i], self.K, self.T, t_k)
        
        return deltas
    
    def to_tensors(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Convert environment data to PyTorch tensors."""
        features = self.get_features()
        
        return {
            'features': torch.tensor(features, dtype=torch.float32, device=device),
            'stock_paths': torch.tensor(self.stock_paths, dtype=torch.float32, device=device),
            'variance_paths': torch.tensor(self.variance_paths, dtype=torch.float32, device=device),
            'payoff': torch.tensor(self.get_payoff(), dtype=torch.float32, device=device)
        }
