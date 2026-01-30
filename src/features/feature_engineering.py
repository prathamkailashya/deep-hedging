"""
Enhanced Feature Engineering for Deep Hedging.

Features include:
- Moneyness (S/K)
- Time-to-maturity (τ)
- Realized volatility (rolling windows)
- Lagged deltas and returns
- Implied volatility proxy
"""

import numpy as np
import torch
from typing import Optional, Tuple, List


class FeatureEngineer:
    """
    Generate enhanced features for deep hedging models.
    """
    
    def __init__(
        self,
        K: float = 100.0,
        T: float = 30/365,
        n_steps: int = 30,
        vol_windows: List[int] = [5, 10, 20],
        lag_steps: int = 3,
        include_implied_vol: bool = False
    ):
        self.K = K
        self.T = T
        self.n_steps = n_steps
        self.vol_windows = vol_windows
        self.lag_steps = lag_steps
        self.include_implied_vol = include_implied_vol
        self.dt = T / n_steps
        
    def compute_moneyness(self, stock_paths: np.ndarray) -> np.ndarray:
        """Compute moneyness S/K."""
        return stock_paths / self.K
    
    def compute_time_to_maturity(self, n_samples: int) -> np.ndarray:
        """Compute time-to-maturity at each step."""
        # Shape: (n_steps + 1,)
        tau = np.linspace(self.T, 0, self.n_steps + 1)
        # Broadcast to (n_samples, n_steps + 1)
        return np.broadcast_to(tau, (n_samples, self.n_steps + 1))
    
    def compute_log_returns(self, stock_paths: np.ndarray) -> np.ndarray:
        """Compute log returns."""
        # Shape: (n_samples, n_steps)
        returns = np.diff(np.log(stock_paths), axis=1)
        # Pad with zero at the start
        return np.concatenate([np.zeros((len(returns), 1)), returns], axis=1)
    
    def compute_realized_volatility(
        self, 
        stock_paths: np.ndarray,
        window: int
    ) -> np.ndarray:
        """
        Compute rolling realized volatility.
        
        RV_t = sqrt(sum(r_{t-w:t}^2) / w) * sqrt(252)
        """
        n_samples, n_steps_plus_1 = stock_paths.shape
        log_returns = np.diff(np.log(stock_paths), axis=1)
        
        rv = np.zeros((n_samples, n_steps_plus_1))
        
        for t in range(n_steps_plus_1):
            start = max(0, t - window)
            end = t
            if end > start and end <= log_returns.shape[1]:
                window_returns = log_returns[:, start:end]
                rv[:, t] = np.sqrt(np.mean(window_returns**2, axis=1)) * np.sqrt(252)
            else:
                rv[:, t] = 0.2  # Default to 20% vol
        
        return rv
    
    def compute_lagged_returns(
        self, 
        stock_paths: np.ndarray
    ) -> np.ndarray:
        """Compute lagged returns for each timestep."""
        log_returns = self.compute_log_returns(stock_paths)
        n_samples, n_steps_plus_1 = log_returns.shape
        
        # Shape: (n_samples, n_steps + 1, lag_steps)
        lagged = np.zeros((n_samples, n_steps_plus_1, self.lag_steps))
        
        for lag in range(self.lag_steps):
            shifted = np.roll(log_returns, lag + 1, axis=1)
            shifted[:, :lag + 1] = 0  # Zero out wrapped values
            lagged[:, :, lag] = shifted
        
        return lagged
    
    def compute_price_features(self, stock_paths: np.ndarray) -> np.ndarray:
        """Compute price-based features: normalized price, high/low ratios."""
        n_samples, n_steps_plus_1 = stock_paths.shape
        
        # Normalized price (relative to initial)
        norm_price = stock_paths / stock_paths[:, 0:1]
        
        # Rolling max/min ratios
        rolling_max = np.maximum.accumulate(stock_paths, axis=1)
        rolling_min = np.minimum.accumulate(stock_paths, axis=1)
        
        drawdown = (rolling_max - stock_paths) / rolling_max
        rally = (stock_paths - rolling_min) / (rolling_min + 1e-8)
        
        return np.stack([norm_price, drawdown, rally], axis=-1)
    
    def generate_features(
        self,
        stock_paths: np.ndarray,
        variance_paths: Optional[np.ndarray] = None,
        prev_deltas: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate all enhanced features.
        
        Returns:
            features: (n_samples, n_steps, n_features)
        """
        n_samples, n_steps_plus_1 = stock_paths.shape
        
        feature_list = []
        
        # 1. Moneyness (S/K)
        moneyness = self.compute_moneyness(stock_paths)[:, :-1]
        feature_list.append(moneyness[:, :, np.newaxis])
        
        # 2. Time-to-maturity
        tau = self.compute_time_to_maturity(n_samples)[:, :-1]
        feature_list.append(tau[:, :, np.newaxis])
        
        # 3. Log returns
        log_returns = self.compute_log_returns(stock_paths)[:, :-1]
        feature_list.append(log_returns[:, :, np.newaxis])
        
        # 4. Realized volatility (multiple windows)
        for window in self.vol_windows:
            rv = self.compute_realized_volatility(stock_paths, window)[:, :-1]
            feature_list.append(rv[:, :, np.newaxis])
        
        # 5. Lagged returns
        lagged_returns = self.compute_lagged_returns(stock_paths)[:, :-1, :]
        feature_list.append(lagged_returns)
        
        # 6. Price features (normalized, drawdown, rally)
        price_feats = self.compute_price_features(stock_paths)[:, :-1, :]
        feature_list.append(price_feats)
        
        # 7. Variance (if available from Heston)
        if variance_paths is not None:
            var_feats = variance_paths[:, :-1, np.newaxis]
            feature_list.append(var_feats)
            # Also add sqrt(variance) as volatility proxy
            vol_feats = np.sqrt(variance_paths[:, :-1, np.newaxis])
            feature_list.append(vol_feats)
        
        # 8. Previous deltas (if available)
        if prev_deltas is not None:
            # Shift deltas by 1 to get previous delta
            shifted_deltas = np.roll(prev_deltas, 1, axis=1)
            shifted_deltas[:, 0] = 0
            feature_list.append(shifted_deltas[:, :, np.newaxis])
        
        # Concatenate all features
        features = np.concatenate(feature_list, axis=-1)
        
        return features
    
    def get_feature_names(self, has_variance: bool = False, has_deltas: bool = False) -> List[str]:
        """Get names of all features."""
        names = ['moneyness', 'tau', 'log_return']
        names += [f'rv_{w}' for w in self.vol_windows]
        names += [f'lag_return_{i+1}' for i in range(self.lag_steps)]
        names += ['norm_price', 'drawdown', 'rally']
        if has_variance:
            names += ['variance', 'volatility']
        if has_deltas:
            names += ['prev_delta']
        return names


class EnhancedDataGenerator:
    """
    Data generator with enhanced features.
    """
    
    def __init__(
        self,
        base_generator,
        feature_engineer: FeatureEngineer
    ):
        self.base_generator = base_generator
        self.feature_engineer = feature_engineer
    
    def generate_enhanced_data(
        self,
        n_samples: int,
        seed: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate data with enhanced features.
        
        Returns:
            features: (n_samples, n_steps, n_features)
            stock_paths: (n_samples, n_steps + 1)
            payoffs: (n_samples,)
        """
        # Generate base data
        data = self.base_generator.generate(n_samples, seed)
        
        stock_paths = data.stock_paths.numpy()
        payoffs = data.payoffs.numpy()
        
        # Get variance paths if available
        variance_paths = None
        if hasattr(data, 'variance_paths') and data.variance_paths is not None:
            variance_paths = data.variance_paths.numpy()
        
        # Generate enhanced features
        features = self.feature_engineer.generate_features(
            stock_paths,
            variance_paths
        )
        
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(stock_paths, dtype=torch.float32),
            torch.tensor(payoffs, dtype=torch.float32)
        )


def compare_feature_sets(raw_features: np.ndarray, enhanced_features: np.ndarray) -> dict:
    """Compare raw vs enhanced feature statistics."""
    return {
        'raw': {
            'n_features': raw_features.shape[-1],
            'mean': np.mean(raw_features),
            'std': np.std(raw_features),
            'range': (np.min(raw_features), np.max(raw_features))
        },
        'enhanced': {
            'n_features': enhanced_features.shape[-1],
            'mean': np.mean(enhanced_features),
            'std': np.std(enhanced_features),
            'range': (np.min(enhanced_features), np.max(enhanced_features))
        }
    }
