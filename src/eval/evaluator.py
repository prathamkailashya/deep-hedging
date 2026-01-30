"""
Evaluation utilities for Deep Hedging strategies.

Provides comprehensive evaluation metrics and statistical comparisons.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

try:
    from ..utils.statistics import (
        compute_metrics, bootstrap_metrics, bootstrap_ci,
        paired_ttest, wilcoxon_test, compare_strategies as stat_compare
    )
except ImportError:
    from utils.statistics import (
        compute_metrics, bootstrap_metrics, bootstrap_ci,
        paired_ttest, wilcoxon_test, compare_strategies as stat_compare
    )


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    strategy_name: str
    pnl: np.ndarray
    deltas: np.ndarray
    metrics: Dict[str, float]
    bootstrap_ci: Optional[Dict[str, Dict[str, float]]] = None
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [f"Strategy: {self.strategy_name}"]
        lines.append("-" * 40)
        for key, value in self.metrics.items():
            lines.append(f"  {key}: {value:.4f}")
        return "\n".join(lines)


class HedgingEvaluator:
    """
    Comprehensive evaluator for hedging strategies.
    
    Computes all metrics specified in the requirements:
    - Mean P&L, Std deviation
    - VaR95, VaR99
    - CVaR (Expected Shortfall)
    - Entropic loss
    - Trading volume and total transaction cost
    - Indifference price
    """
    
    def __init__(
        self,
        cost_multiplier: float = 0.0,
        lambda_risk: float = 1.0,
        cvar_alpha: float = 0.95,
        n_bootstrap: int = 1000,
        device: str = 'cpu'
    ):
        self.cost_multiplier = cost_multiplier
        self.lambda_risk = lambda_risk
        self.cvar_alpha = cvar_alpha
        self.n_bootstrap = n_bootstrap
        self.device = device
    
    def compute_pnl(
        self,
        deltas: np.ndarray,
        stock_paths: np.ndarray,
        payoffs: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute P&L and components.
        
        Args:
            deltas: Hedging positions (n_paths, n_steps)
            stock_paths: Stock prices (n_paths, n_steps + 1)
            payoffs: Option payoffs (n_paths,)
        
        Returns:
            pnl: P&L array
            info: Dictionary with components
        """
        # Price changes
        price_changes = np.diff(stock_paths, axis=1)
        
        # Hedging gains
        hedging_gains = np.sum(deltas * price_changes, axis=1)
        
        # Transaction costs
        deltas_ext = np.hstack([
            np.zeros((deltas.shape[0], 1)),
            deltas,
            np.zeros((deltas.shape[0], 1))
        ])
        delta_changes = np.abs(np.diff(deltas_ext, axis=1))
        transaction_costs = self.cost_multiplier * np.sum(delta_changes, axis=1)
        
        # Trading volume
        trading_volume = np.sum(delta_changes, axis=1)
        
        # P&L
        pnl = -payoffs + hedging_gains - transaction_costs
        
        info = {
            'payoffs': payoffs,
            'hedging_gains': hedging_gains,
            'transaction_costs': transaction_costs,
            'trading_volume': trading_volume
        }
        
        return pnl, info
    
    def evaluate_strategy(
        self,
        deltas: np.ndarray,
        stock_paths: np.ndarray,
        payoffs: np.ndarray,
        strategy_name: str = "Strategy",
        compute_bootstrap: bool = True
    ) -> EvaluationResults:
        """
        Evaluate a hedging strategy.
        
        Args:
            deltas: Hedging positions
            stock_paths: Stock prices
            payoffs: Option payoffs
            strategy_name: Name of strategy
            compute_bootstrap: Whether to compute bootstrap CIs
        
        Returns:
            EvaluationResults object
        """
        pnl, info = self.compute_pnl(deltas, stock_paths, payoffs)
        
        # Base metrics
        metrics = compute_metrics(pnl, self.lambda_risk, self.cvar_alpha)
        
        # Additional metrics from info
        metrics['mean_payoff'] = np.mean(info['payoffs'])
        metrics['mean_hedging_gains'] = np.mean(info['hedging_gains'])
        metrics['mean_transaction_costs'] = np.mean(info['transaction_costs'])
        metrics['total_trading_volume'] = np.mean(info['trading_volume'])
        metrics['mean_delta'] = np.mean(deltas)
        metrics['delta_std'] = np.std(deltas)
        
        # Bootstrap confidence intervals
        bootstrap_results = None
        if compute_bootstrap:
            try:
                bootstrap_results = bootstrap_metrics(
                    pnl, self.n_bootstrap, confidence=0.95
                )
            except Exception as e:
                warnings.warn(f"Bootstrap failed: {e}")
        
        return EvaluationResults(
            strategy_name=strategy_name,
            pnl=pnl,
            deltas=deltas,
            metrics=metrics,
            bootstrap_ci=bootstrap_results
        )
    
    @torch.no_grad()
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        strategy_name: str = "Model",
        compute_bootstrap: bool = True
    ) -> EvaluationResults:
        """
        Evaluate a PyTorch model on test data.
        """
        model.eval()
        model.to(self.device)
        
        all_deltas = []
        all_stock_paths = []
        all_payoffs = []
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            stock_paths = batch['stock_paths']
            payoffs = batch['payoff']
            
            deltas = model(features).cpu().numpy()
            
            all_deltas.append(deltas)
            all_stock_paths.append(stock_paths.numpy())
            all_payoffs.append(payoffs.numpy())
        
        deltas = np.concatenate(all_deltas, axis=0)
        stock_paths = np.concatenate(all_stock_paths, axis=0)
        payoffs = np.concatenate(all_payoffs, axis=0)
        
        return self.evaluate_strategy(
            deltas, stock_paths, payoffs,
            strategy_name, compute_bootstrap
        )
    
    def evaluate_baseline(
        self,
        baseline_hedge,
        stock_paths: np.ndarray,
        time_grid: np.ndarray,
        K: float,
        T: float,
        payoffs: np.ndarray,
        strategy_name: str = "Baseline"
    ) -> EvaluationResults:
        """
        Evaluate a baseline hedging strategy.
        """
        deltas = baseline_hedge.compute_deltas(stock_paths, time_grid, K, T)
        
        return self.evaluate_strategy(
            deltas, stock_paths, payoffs,
            strategy_name, compute_bootstrap=True
        )


def compare_strategies(
    results_list: List[EvaluationResults],
    baseline_idx: int = 0
) -> Dict[str, Any]:
    """
    Compare multiple hedging strategies.
    
    Args:
        results_list: List of EvaluationResults
        baseline_idx: Index of baseline strategy for comparison
    
    Returns:
        Dictionary with comparison results
    """
    if len(results_list) < 2:
        raise ValueError("Need at least 2 strategies to compare")
    
    baseline = results_list[baseline_idx]
    comparison = {}
    
    for i, results in enumerate(results_list):
        if i == baseline_idx:
            continue
        
        name = results.strategy_name
        
        # Statistical tests
        t_stat, t_pval = paired_ttest(results.pnl, baseline.pnl)
        w_stat, w_pval = wilcoxon_test(results.pnl, baseline.pnl)
        
        # Mean difference with CI
        diff = results.pnl - baseline.pnl
        diff_mean, diff_lower, diff_upper = bootstrap_ci(diff, np.mean, 1000)
        
        comparison[name] = {
            'metrics': results.metrics,
            'vs_baseline': {
                'baseline': baseline.strategy_name,
                'mean_diff': diff_mean,
                'diff_ci_lower': diff_lower,
                'diff_ci_upper': diff_upper,
                't_statistic': t_stat,
                't_pvalue': t_pval,
                'wilcoxon_statistic': w_stat,
                'wilcoxon_pvalue': w_pval,
                'significant_5pct': t_pval < 0.05,
                'significant_1pct': t_pval < 0.01,
                'improvement': diff_mean > 0
            }
        }
    
    return comparison


def create_results_summary(
    results_list: List[EvaluationResults],
    metrics_to_show: Optional[List[str]] = None
) -> str:
    """
    Create a formatted summary table of results.
    """
    if metrics_to_show is None:
        metrics_to_show = [
            'mean_pnl', 'std_pnl', 'var_95', 'cvar_95',
            'entropic_risk', 'total_trading_volume'
        ]
    
    # Header
    col_width = 15
    header = "Strategy".ljust(20)
    for m in metrics_to_show:
        header += m.ljust(col_width)
    
    lines = [header, "=" * len(header)]
    
    # Results
    for results in results_list:
        line = results.strategy_name.ljust(20)
        for m in metrics_to_show:
            val = results.metrics.get(m, float('nan'))
            line += f"{val:.4f}".ljust(col_width)
        lines.append(line)
    
    return "\n".join(lines)


class StressTestEvaluator:
    """
    Evaluator for stress testing hedging strategies.
    
    Tests under:
    - Volatility regime shifts
    - 2008 financial crisis conditions
    - 2020 COVID market conditions
    """
    
    def __init__(self, base_evaluator: HedgingEvaluator):
        self.base_evaluator = base_evaluator
    
    def volatility_stress_test(
        self,
        model: torch.nn.Module,
        base_features: torch.Tensor,
        base_stock_paths: torch.Tensor,
        base_payoffs: torch.Tensor,
        vol_multipliers: List[float] = [0.5, 1.0, 1.5, 2.0, 3.0]
    ) -> Dict[float, EvaluationResults]:
        """
        Test strategy under different volatility regimes.
        """
        results = {}
        
        for mult in vol_multipliers:
            # Scale price changes by volatility multiplier
            price_changes = torch.diff(base_stock_paths, dim=1)
            scaled_changes = price_changes * mult
            
            # Reconstruct paths
            scaled_paths = torch.zeros_like(base_stock_paths)
            scaled_paths[:, 0] = base_stock_paths[:, 0]
            scaled_paths[:, 1:] = base_stock_paths[:, 0:1] + torch.cumsum(scaled_changes, dim=1)
            
            # Scale payoffs
            final_prices = scaled_paths[:, -1]
            K = 100.0  # Assuming ATM
            scaled_payoffs = torch.relu(final_prices - K)
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                deltas = model(base_features).numpy()
            
            result = self.base_evaluator.evaluate_strategy(
                deltas,
                scaled_paths.numpy(),
                scaled_payoffs.numpy(),
                f"Vol x{mult}",
                compute_bootstrap=False
            )
            results[mult] = result
        
        return results
    
    def crisis_regime_test(
        self,
        model: torch.nn.Module,
        features: torch.Tensor,
        stock_paths: torch.Tensor,
        regime: str = '2008'
    ) -> EvaluationResults:
        """
        Test under crisis conditions.
        
        Simulates:
        - 2008: High volatility, negative drift, correlation breakdown
        - 2020: Volatility spike, V-shaped recovery
        """
        if regime == '2008':
            # 2008 crisis: sustained high vol, downward drift
            vol_mult = 3.0
            drift_adj = -0.001  # Daily downward drift
        elif regime == '2020':
            # COVID: spike then recovery
            vol_mult = 2.5
            drift_adj = 0.0
        else:
            raise ValueError(f"Unknown regime: {regime}")
        
        # Apply regime adjustments
        n_steps = stock_paths.shape[1] - 1
        
        price_changes = torch.diff(stock_paths, dim=1)
        scaled_changes = price_changes * vol_mult
        
        # Add drift adjustment
        drift = torch.ones(stock_paths.shape[0], n_steps) * drift_adj * stock_paths[:, :-1]
        scaled_changes = scaled_changes + drift
        
        # Reconstruct
        crisis_paths = torch.zeros_like(stock_paths)
        crisis_paths[:, 0] = stock_paths[:, 0]
        crisis_paths[:, 1:] = stock_paths[:, 0:1] + torch.cumsum(scaled_changes, dim=1)
        crisis_paths = torch.relu(crisis_paths) + 1e-6  # Ensure positive
        
        # Payoffs
        K = 100.0
        crisis_payoffs = torch.relu(crisis_paths[:, -1] - K)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            deltas = model(features).numpy()
        
        return self.base_evaluator.evaluate_strategy(
            deltas,
            crisis_paths.numpy(),
            crisis_payoffs.numpy(),
            f"Crisis {regime}",
            compute_bootstrap=True
        )
