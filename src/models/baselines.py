"""
Baseline Hedging Strategies.

Implements classical hedging strategies for comparison:
- Black-Scholes delta hedge
- Leland adjusted delta
- Whalley-Wilmott no-transaction band

References:
- Black, F. & Scholes, M. (1973)
- Leland, H. (1985) "Option Pricing and Replication with Transaction Costs"
- Whalley, A.E. & Wilmott, P. (1997) "An Asymptotic Analysis of an Optimal
  Hedging Model for Option Pricing with Transaction Costs"
"""

import numpy as np
import torch
from scipy.stats import norm
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class BaselineHedge(ABC):
    """Abstract base class for baseline hedging strategies."""
    
    @abstractmethod
    def compute_deltas(
        self,
        stock_paths: np.ndarray,
        time_grid: np.ndarray,
        K: float,
        T: float,
        **kwargs
    ) -> np.ndarray:
        """
        Compute hedging positions for all paths and time steps.
        
        Args:
            stock_paths: Stock prices, shape (n_paths, n_steps + 1)
            time_grid: Time points, shape (n_steps + 1,)
            K: Strike price
            T: Time to maturity
        
        Returns:
            deltas: Hedging positions, shape (n_paths, n_steps)
        """
        pass


class BlackScholesHedge(BaselineHedge):
    """
    Black-Scholes delta hedging strategy.
    
    Delta = N(d1) where
    d1 = [log(S/K) + (r + σ²/2)τ] / (σ√τ)
    """
    
    def __init__(self, sigma: float = 0.2, r: float = 0.0):
        """
        Args:
            sigma: Implied volatility
            r: Risk-free rate
        """
        self.sigma = sigma
        self.r = r
    
    def delta(self, S: float, K: float, tau: float) -> float:
        """
        Compute Black-Scholes delta for a single point.
        
        Args:
            S: Current stock price
            K: Strike price
            tau: Time to maturity
        
        Returns:
            Delta value
        """
        if tau <= 1e-10:
            return 1.0 if S > K else 0.0
        
        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
        return norm.cdf(d1)
    
    def gamma(self, S: float, K: float, tau: float) -> float:
        """Compute Black-Scholes gamma."""
        if tau <= 1e-10:
            return 0.0
        
        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
        return norm.pdf(d1) / (S * self.sigma * np.sqrt(tau))
    
    def compute_deltas(
        self,
        stock_paths: np.ndarray,
        time_grid: np.ndarray,
        K: float,
        T: float,
        **kwargs
    ) -> np.ndarray:
        """Compute BS deltas for all paths."""
        n_paths, n_steps_plus_1 = stock_paths.shape
        n_steps = n_steps_plus_1 - 1
        
        deltas = np.zeros((n_paths, n_steps))
        
        for k in range(n_steps):
            tau = T - time_grid[k]
            for i in range(n_paths):
                deltas[i, k] = self.delta(stock_paths[i, k], K, tau)
        
        return deltas
    
    def compute_deltas_vectorized(
        self,
        stock_paths: np.ndarray,
        time_grid: np.ndarray,
        K: float,
        T: float
    ) -> np.ndarray:
        """Vectorized computation of BS deltas."""
        n_paths, n_steps_plus_1 = stock_paths.shape
        n_steps = n_steps_plus_1 - 1
        
        deltas = np.zeros((n_paths, n_steps))
        
        for k in range(n_steps):
            tau = T - time_grid[k]
            if tau <= 1e-10:
                deltas[:, k] = np.where(stock_paths[:, k] > K, 1.0, 0.0)
            else:
                S = stock_paths[:, k]
                d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
                deltas[:, k] = norm.cdf(d1)
        
        return deltas


class LelandHedge(BaselineHedge):
    """
    Leland's modified delta for transaction costs.
    
    Uses modified volatility:
    σ_L = σ * √(1 + √(2/π) * κ/(σ√Δt))
    
    where κ is the proportional transaction cost.
    
    Reference: Leland (1985)
    """
    
    def __init__(self, sigma: float = 0.2, r: float = 0.0, cost: float = 0.0):
        """
        Args:
            sigma: Base volatility
            r: Risk-free rate
            cost: Proportional transaction cost κ
        """
        self.sigma = sigma
        self.r = r
        self.cost = cost
    
    def modified_volatility(self, dt: float) -> float:
        """
        Compute Leland's modified volatility.
        
        σ_L = σ * √(1 + √(2/π) * κ/(σ√Δt))
        """
        if self.cost <= 0 or dt <= 0:
            return self.sigma
        
        adjustment = np.sqrt(2 / np.pi) * self.cost / (self.sigma * np.sqrt(dt))
        sigma_L = self.sigma * np.sqrt(1 + adjustment)
        return sigma_L
    
    def delta(self, S: float, K: float, tau: float, dt: float) -> float:
        """Compute Leland-adjusted delta."""
        if tau <= 1e-10:
            return 1.0 if S > K else 0.0
        
        sigma_L = self.modified_volatility(dt)
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma_L**2) * tau) / (sigma_L * np.sqrt(tau))
        return norm.cdf(d1)
    
    def compute_deltas(
        self,
        stock_paths: np.ndarray,
        time_grid: np.ndarray,
        K: float,
        T: float,
        **kwargs
    ) -> np.ndarray:
        """Compute Leland deltas for all paths."""
        n_paths, n_steps_plus_1 = stock_paths.shape
        n_steps = n_steps_plus_1 - 1
        dt = T / n_steps
        
        deltas = np.zeros((n_paths, n_steps))
        
        for k in range(n_steps):
            tau = T - time_grid[k]
            for i in range(n_paths):
                deltas[i, k] = self.delta(stock_paths[i, k], K, tau, dt)
        
        return deltas


class WhalleyWilmottHedge(BaselineHedge):
    """
    Whalley-Wilmott no-transaction band strategy.
    
    The hedger only trades when the BS delta moves outside a band:
    [δ - H, δ + H]
    
    where H is the half-width of the no-transaction band:
    H = (3/2 * κ * exp(-rτ) * Γ * S² / λ)^(1/3)
    
    Reference: Whalley & Wilmott (1997)
    """
    
    def __init__(
        self,
        sigma: float = 0.2,
        r: float = 0.0,
        cost: float = 0.0,
        risk_aversion: float = 1.0
    ):
        """
        Args:
            sigma: Volatility
            r: Risk-free rate  
            cost: Proportional transaction cost κ
            risk_aversion: Risk aversion parameter λ
        """
        self.sigma = sigma
        self.r = r
        self.cost = cost
        self.risk_aversion = risk_aversion
        self.bs_hedge = BlackScholesHedge(sigma, r)
    
    def band_width(self, S: float, K: float, tau: float) -> float:
        """
        Compute the half-width of the no-transaction band.
        
        H = (3/2 * κ * exp(-rτ) * Γ * S² / λ)^(1/3)
        """
        if self.cost <= 0 or tau <= 1e-10:
            return 0.0
        
        gamma = self.bs_hedge.gamma(S, K, tau)
        if gamma <= 0:
            return 0.0
        
        H = np.power(
            1.5 * self.cost * np.exp(-self.r * tau) * gamma * S**2 / self.risk_aversion,
            1/3
        )
        return H
    
    def compute_deltas(
        self,
        stock_paths: np.ndarray,
        time_grid: np.ndarray,
        K: float,
        T: float,
        **kwargs
    ) -> np.ndarray:
        """
        Compute Whalley-Wilmott deltas with no-transaction bands.
        
        Trading rule:
        - If current position is within band, don't trade
        - If outside band, trade to the nearest band edge
        """
        n_paths, n_steps_plus_1 = stock_paths.shape
        n_steps = n_steps_plus_1 - 1
        
        deltas = np.zeros((n_paths, n_steps))
        
        for i in range(n_paths):
            current_delta = 0.0  # Start with no position
            
            for k in range(n_steps):
                S = stock_paths[i, k]
                tau = T - time_grid[k]
                
                # BS target delta
                bs_delta = self.bs_hedge.delta(S, K, tau)
                
                # Band width
                H = self.band_width(S, K, tau)
                
                # Check if current position is within band
                lower = bs_delta - H
                upper = bs_delta + H
                
                if current_delta < lower:
                    # Trade to lower band edge
                    current_delta = lower
                elif current_delta > upper:
                    # Trade to upper band edge
                    current_delta = upper
                # Otherwise, keep current position
                
                deltas[i, k] = current_delta
        
        return deltas


class NoHedge(BaselineHedge):
    """No hedging baseline (naked short position)."""
    
    def compute_deltas(
        self,
        stock_paths: np.ndarray,
        time_grid: np.ndarray,
        K: float,
        T: float,
        **kwargs
    ) -> np.ndarray:
        """Return zero deltas (no hedging)."""
        n_paths, n_steps_plus_1 = stock_paths.shape
        n_steps = n_steps_plus_1 - 1
        return np.zeros((n_paths, n_steps))


class PerfectHedge(BaselineHedge):
    """
    Perfect hedging (oracle that knows future prices).
    
    For testing purposes only - uses future information.
    """
    
    def compute_deltas(
        self,
        stock_paths: np.ndarray,
        time_grid: np.ndarray,
        K: float,
        T: float,
        **kwargs
    ) -> np.ndarray:
        """Compute perfect hedge using final payoff."""
        n_paths, n_steps_plus_1 = stock_paths.shape
        n_steps = n_steps_plus_1 - 1
        
        # Final payoff
        S_T = stock_paths[:, -1]
        in_the_money = (S_T > K).astype(float)
        
        # Hold 1 share if ITM at expiry, 0 otherwise
        deltas = np.tile(in_the_money[:, np.newaxis], (1, n_steps))
        
        return deltas


def evaluate_baseline(
    hedge: BaselineHedge,
    stock_paths: np.ndarray,
    time_grid: np.ndarray,
    K: float,
    T: float,
    cost_multiplier: float = 0.0
) -> Tuple[np.ndarray, dict]:
    """
    Evaluate a baseline hedging strategy.
    
    Returns:
        pnl: P&L for each path
        info: Dictionary with additional metrics
    """
    deltas = hedge.compute_deltas(stock_paths, time_grid, K, T)
    
    # Payoff (short call)
    S_T = stock_paths[:, -1]
    payoff = np.maximum(S_T - K, 0)
    
    # Hedging gains
    price_changes = np.diff(stock_paths, axis=1)
    hedging_gains = np.sum(deltas * price_changes, axis=1)
    
    # Transaction costs
    delta_ext = np.hstack([
        np.zeros((deltas.shape[0], 1)),
        deltas,
        np.zeros((deltas.shape[0], 1))
    ])
    delta_changes = np.abs(np.diff(delta_ext, axis=1))
    transaction_costs = cost_multiplier * np.sum(delta_changes, axis=1)
    
    # P&L
    pnl = -payoff + hedging_gains - transaction_costs
    
    info = {
        'payoff': payoff,
        'hedging_gains': hedging_gains,
        'transaction_costs': transaction_costs,
        'trading_volume': np.sum(delta_changes, axis=1),
        'mean_delta': np.mean(deltas, axis=1),
        'delta_std': np.std(deltas, axis=1)
    }
    
    return pnl, info
