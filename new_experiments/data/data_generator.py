"""
Data Generation for Fair Comparison

Generates Heston paths with proper feature engineering.
All models use identical data splits.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class HestonParams:
    S0: float = 100.0
    v0: float = 0.04
    r: float = 0.0
    kappa: float = 1.0
    theta: float = 0.04
    sigma: float = 0.2
    rho: float = -0.7


class HestonSimulator:
    """Heston model simulation with full truncation scheme."""
    
    def __init__(self, params: HestonParams, n_steps: int, T: float):
        self.params = params
        self.n_steps = n_steps
        self.T = T
        self.dt = T / n_steps
    
    def simulate(self, n_paths: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston paths.
        
        Returns:
            stock_paths: (n_paths, n_steps + 1)
            variance_paths: (n_paths, n_steps + 1)
        """
        if seed is not None:
            np.random.seed(seed)
        
        p = self.params
        dt = self.dt
        
        # Initialize
        S = np.zeros((n_paths, self.n_steps + 1))
        v = np.zeros((n_paths, self.n_steps + 1))
        S[:, 0] = p.S0
        v[:, 0] = p.v0
        
        # Correlated Brownian motions
        for t in range(self.n_steps):
            Z1 = np.random.randn(n_paths)
            Z2 = np.random.randn(n_paths)
            W_S = Z1
            W_v = p.rho * Z1 + np.sqrt(1 - p.rho**2) * Z2
            
            # Full truncation scheme
            v_pos = np.maximum(v[:, t], 0)
            sqrt_v = np.sqrt(v_pos)
            
            # Variance process
            v[:, t + 1] = (v[:, t] + p.kappa * (p.theta - v_pos) * dt 
                          + p.sigma * sqrt_v * np.sqrt(dt) * W_v)
            v[:, t + 1] = np.maximum(v[:, t + 1], 0)
            
            # Stock process
            S[:, t + 1] = S[:, t] * np.exp(
                (p.r - 0.5 * v_pos) * dt + sqrt_v * np.sqrt(dt) * W_S
            )
        
        return S, v


class HedgingDataset(Dataset):
    """Dataset for deep hedging with full feature engineering."""
    
    def __init__(
        self,
        stock_paths: np.ndarray,
        variance_paths: np.ndarray,
        K: float,
        T: float,
        r: float = 0.0
    ):
        self.stock_paths = torch.FloatTensor(stock_paths)
        self.variance_paths = torch.FloatTensor(variance_paths)
        self.K = K
        self.T = T
        self.r = r
        self.n_paths, self.n_steps_plus1 = stock_paths.shape
        self.n_steps = self.n_steps_plus1 - 1
        
        # Compute features
        self.features = self._compute_features()
        self.payoffs = self._compute_payoffs()
        
        self.n_features = self.features.shape[-1]
    
    def _compute_features(self) -> torch.Tensor:
        """
        Compute time-augmented features for each timestep.
        
        Features: [S_t/S_0, log(S_t/K), sqrt(v_t), tau_t]
        """
        n_paths = self.n_paths
        n_steps = self.n_steps
        
        # Time to maturity
        tau = torch.linspace(self.T, 0, n_steps + 1)[:-1]  # Exclude final
        tau = tau.unsqueeze(0).expand(n_paths, -1)
        
        # Price features (normalized)
        S = self.stock_paths[:, :-1]  # Exclude final price
        S_normalized = S / self.stock_paths[:, 0:1]
        log_moneyness = torch.log(S / self.K)
        
        # Volatility feature
        vol = torch.sqrt(torch.clamp(self.variance_paths[:, :-1], min=1e-8))
        
        # Stack features: [S_norm, log_moneyness, vol, tau]
        features = torch.stack([S_normalized, log_moneyness, vol, tau], dim=-1)
        
        return features
    
    def _compute_payoffs(self) -> torch.Tensor:
        """Compute European call payoffs."""
        S_T = self.stock_paths[:, -1]
        payoff = torch.maximum(S_T - self.K, torch.zeros_like(S_T))
        return payoff
    
    def __len__(self) -> int:
        return self.n_paths
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'features': self.features[idx],
            'stock_paths': self.stock_paths[idx],
            'variance_paths': self.variance_paths[idx],
            'payoff': self.payoffs[idx]
        }


class DataGenerator:
    """Generate train/val/test splits with consistent seeds."""
    
    def __init__(
        self,
        heston_params: HestonParams,
        n_steps: int = 30,
        T: float = 30/365,
        K: float = 100.0,
        r: float = 0.0
    ):
        self.heston_params = heston_params
        self.n_steps = n_steps
        self.T = T
        self.K = K
        self.r = r
        self.simulator = HestonSimulator(heston_params, n_steps, T)
    
    def generate_dataset(
        self,
        n_paths: int,
        seed: int
    ) -> HedgingDataset:
        """Generate a single dataset."""
        S, v = self.simulator.simulate(n_paths, seed)
        return HedgingDataset(S, v, self.K, self.T, self.r)
    
    def generate_splits(
        self,
        n_train: int = 90000,
        n_val: int = 10000,
        n_test: int = 100000,
        base_seed: int = 42
    ) -> Tuple[HedgingDataset, HedgingDataset, HedgingDataset]:
        """Generate train/val/test with fixed seeds."""
        print(f"Generating data: train={n_train}, val={n_val}, test={n_test}")
        
        train_data = self.generate_dataset(n_train, base_seed)
        val_data = self.generate_dataset(n_val, base_seed + 1)
        test_data = self.generate_dataset(n_test, base_seed + 2)
        
        return train_data, val_data, test_data
    
    def get_dataloaders(
        self,
        train_data: HedgingDataset,
        val_data: HedgingDataset,
        test_data: HedgingDataset,
        batch_size: int = 256,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders."""
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        return train_loader, val_loader, test_loader


def compute_extended_features(
    stock_paths: torch.Tensor,
    variance_paths: torch.Tensor,
    K: float,
    T: float,
    window: int = 5
) -> torch.Tensor:
    """
    Compute extended features including realized volatility and returns.
    
    Features: [S_norm, log_moneyness, vol, tau, realized_vol, lagged_returns]
    """
    n_paths, n_steps_plus1 = stock_paths.shape
    n_steps = n_steps_plus1 - 1
    
    # Basic features
    tau = torch.linspace(T, 0, n_steps + 1)[:-1]
    tau = tau.unsqueeze(0).expand(n_paths, -1)
    
    S = stock_paths[:, :-1]
    S_normalized = S / stock_paths[:, 0:1]
    log_moneyness = torch.log(S / K)
    vol = torch.sqrt(torch.clamp(variance_paths[:, :-1], min=1e-8))
    
    # Compute log returns
    log_returns = torch.log(stock_paths[:, 1:] / stock_paths[:, :-1])
    
    # Realized volatility (rolling window)
    realized_vol = torch.zeros(n_paths, n_steps)
    for t in range(n_steps):
        start = max(0, t - window + 1)
        if t > 0:
            window_returns = log_returns[:, start:t+1]
            realized_vol[:, t] = window_returns.std(dim=1) * np.sqrt(252)
        else:
            realized_vol[:, t] = vol[:, t]
    
    # Lagged returns (pad with zeros)
    lagged_returns = torch.zeros(n_paths, n_steps)
    lagged_returns[:, 1:] = log_returns[:, :-1]
    
    # Stack all features
    features = torch.stack([
        S_normalized, log_moneyness, vol, tau, realized_vol, lagged_returns
    ], dim=-1)
    
    return features
