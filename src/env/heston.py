"""
Heston Stochastic Volatility Model Implementation.

Model specification (risk-neutral):
    dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW_t^S
    dv_t = kappa * (theta - v_t) * dt + sigma * sqrt(v_t) * dW_t^v
    corr(W^S, W^v) = rho

Reference: Buehler et al. (Deep Hedging), Section 5
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class HestonParams:
    """Heston model parameters."""
    S0: float = 100.0      # Initial stock price
    v0: float = 0.04       # Initial variance (sigma^2 = 0.2^2)
    r: float = 0.0         # Risk-free rate
    kappa: float = 1.0     # Mean reversion speed
    theta: float = 0.04    # Long-term variance
    sigma: float = 0.2     # Volatility of volatility
    rho: float = -0.7      # Correlation between W^S and W^v


class HestonModel:
    """
    Heston stochastic volatility model for Monte Carlo simulation.
    
    Uses the Euler-Maruyama scheme with full truncation for variance process.
    """
    
    def __init__(self, params: Optional[HestonParams] = None):
        self.params = params or HestonParams()
        self._validate_params()
    
    def _validate_params(self):
        """Validate Feller condition and other constraints."""
        p = self.params
        
        # Feller condition: 2 * kappa * theta > sigma^2
        feller = 2 * p.kappa * p.theta
        if feller <= p.sigma ** 2:
            import warnings
            warnings.warn(
                f"Feller condition not satisfied: 2κθ={feller:.4f} <= σ²={p.sigma**2:.4f}. "
                "Variance process may hit zero."
            )
        
        if p.v0 <= 0:
            raise ValueError("Initial variance must be positive")
        if p.theta <= 0:
            raise ValueError("Long-term variance must be positive")
        if p.kappa <= 0:
            raise ValueError("Mean reversion speed must be positive")
        if not -1 <= p.rho <= 1:
            raise ValueError("Correlation must be in [-1, 1]")
    
    def simulate(self, T: float, n_steps: int, n_paths: int,
                seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston model paths.
        
        Args:
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of Monte Carlo paths
            seed: Random seed for reproducibility
        
        Returns:
            S: Stock prices, shape (n_paths, n_steps + 1)
            v: Variance process, shape (n_paths, n_steps + 1)
        """
        if seed is not None:
            np.random.seed(seed)
        
        p = self.params
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize arrays
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        
        S[:, 0] = p.S0
        v[:, 0] = p.v0
        
        # Generate correlated Brownian increments
        # Z1, Z2 are independent standard normals
        # W^S = Z1
        # W^v = rho * Z1 + sqrt(1 - rho^2) * Z2
        Z1 = np.random.randn(n_paths, n_steps)
        Z2 = np.random.randn(n_paths, n_steps)
        
        dW_S = Z1 * sqrt_dt
        dW_v = (p.rho * Z1 + np.sqrt(1 - p.rho**2) * Z2) * sqrt_dt
        
        # Euler-Maruyama simulation with full truncation
        for t in range(n_steps):
            v_t = np.maximum(v[:, t], 0)  # Full truncation
            sqrt_v_t = np.sqrt(v_t)
            
            # Stock price dynamics
            S[:, t + 1] = S[:, t] * np.exp(
                (p.r - 0.5 * v_t) * dt + sqrt_v_t * dW_S[:, t]
            )
            
            # Variance dynamics (with reflection at zero)
            v[:, t + 1] = (
                v_t + 
                p.kappa * (p.theta - v_t) * dt + 
                p.sigma * sqrt_v_t * dW_v[:, t]
            )
            v[:, t + 1] = np.maximum(v[:, t + 1], 0)  # Reflection
        
        return S, v
    
    def simulate_with_antithetic(self, T: float, n_steps: int, n_paths: int,
                                  seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate with antithetic variates for variance reduction.
        
        Returns 2 * n_paths paths (original + antithetic).
        """
        if seed is not None:
            np.random.seed(seed)
        
        p = self.params
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Generate random numbers for half the paths
        half_paths = n_paths // 2
        Z1 = np.random.randn(half_paths, n_steps)
        Z2 = np.random.randn(half_paths, n_steps)
        
        # Create antithetic pairs
        Z1_anti = np.vstack([Z1, -Z1])
        Z2_anti = np.vstack([Z2, -Z2])
        
        actual_paths = Z1_anti.shape[0]
        
        dW_S = Z1_anti * sqrt_dt
        dW_v = (p.rho * Z1_anti + np.sqrt(1 - p.rho**2) * Z2_anti) * sqrt_dt
        
        # Initialize arrays
        S = np.zeros((actual_paths, n_steps + 1))
        v = np.zeros((actual_paths, n_steps + 1))
        
        S[:, 0] = p.S0
        v[:, 0] = p.v0
        
        for t in range(n_steps):
            v_t = np.maximum(v[:, t], 0)
            sqrt_v_t = np.sqrt(v_t)
            
            S[:, t + 1] = S[:, t] * np.exp(
                (p.r - 0.5 * v_t) * dt + sqrt_v_t * dW_S[:, t]
            )
            
            v[:, t + 1] = (
                v_t + 
                p.kappa * (p.theta - v_t) * dt + 
                p.sigma * sqrt_v_t * dW_v[:, t]
            )
            v[:, t + 1] = np.maximum(v[:, t + 1], 0)
        
        return S, v
    
    def get_time_grid(self, T: float, n_steps: int) -> np.ndarray:
        """Return the time grid."""
        return np.linspace(0, T, n_steps + 1)


class BlackScholesModel:
    """
    Black-Scholes model (constant volatility special case).
    
    dS_t = r * S_t * dt + sigma * S_t * dW_t
    """
    
    def __init__(self, S0: float = 100.0, r: float = 0.0, sigma: float = 0.2):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
    
    def simulate(self, T: float, n_steps: int, n_paths: int,
                seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate Black-Scholes paths using exact simulation.
        
        Returns:
            S: Stock prices, shape (n_paths, n_steps + 1)
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        
        # Generate increments
        Z = np.random.randn(n_paths, n_steps)
        
        # Exact simulation using log-normal
        log_returns = (self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z
        
        # Cumulative sum for log prices
        log_S = np.zeros((n_paths, n_steps + 1))
        log_S[:, 0] = np.log(self.S0)
        log_S[:, 1:] = np.log(self.S0) + np.cumsum(log_returns, axis=1)
        
        return np.exp(log_S)
    
    def delta(self, S: float, K: float, T: float, t: float = 0) -> float:
        """
        Black-Scholes delta for European call option.
        
        Δ = N(d1)
        """
        from scipy.stats import norm
        
        tau = T - t
        if tau <= 0:
            return 1.0 if S > K else 0.0
        
        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
        return norm.cdf(d1)
    
    def gamma(self, S: float, K: float, T: float, t: float = 0) -> float:
        """Black-Scholes gamma."""
        from scipy.stats import norm
        
        tau = T - t
        if tau <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
        return norm.pdf(d1) / (S * self.sigma * np.sqrt(tau))
    
    def call_price(self, S: float, K: float, T: float, t: float = 0) -> float:
        """Black-Scholes call option price."""
        from scipy.stats import norm
        
        tau = T - t
        if tau <= 0:
            return max(S - K, 0)
        
        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
        d2 = d1 - self.sigma * np.sqrt(tau)
        
        return S * norm.cdf(d1) - K * np.exp(-self.r * tau) * norm.cdf(d2)
