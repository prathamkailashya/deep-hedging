"""
Statistical utilities for evaluation and hypothesis testing.
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, List
import warnings


def compute_var(losses: np.ndarray, alpha: float = 0.95) -> float:
    """
    Compute Value-at-Risk at confidence level alpha.
    VaR_alpha = inf{x : P(X <= x) >= alpha}
    """
    return np.percentile(losses, alpha * 100)


def compute_cvar(losses: np.ndarray, alpha: float = 0.95) -> float:
    """
    Compute Conditional Value-at-Risk (Expected Shortfall).
    CVaR_alpha = E[X | X >= VaR_alpha]
    """
    var = compute_var(losses, alpha)
    return np.mean(losses[losses >= var])


def compute_entropic_risk(pnl: np.ndarray, lambda_risk: float = 1.0) -> float:
    """
    Compute entropic risk measure.
    ρ(X) = (1/λ) * log(E[exp(-λX)])
    """
    # Use log-sum-exp trick for numerical stability
    scaled_pnl = -lambda_risk * pnl
    max_val = np.max(scaled_pnl)
    entropic = max_val + np.log(np.mean(np.exp(scaled_pnl - max_val)))
    return entropic / lambda_risk


def compute_indifference_price(pnl: np.ndarray, lambda_risk: float = 1.0) -> float:
    """
    Compute indifference price.
    π(-Z) = (1/λ) * log(inf_θ J(θ))
    
    For the optimal hedging strategy:
    π = (1/λ) * log(E[exp(-λ * P&L)])
    """
    return compute_entropic_risk(pnl, lambda_risk)


def compute_metrics(pnl: np.ndarray, lambda_risk: float = 1.0, 
                   cvar_alpha: float = 0.95) -> Dict[str, float]:
    """
    Compute all evaluation metrics for P&L distribution.
    
    Args:
        pnl: Array of P&L values (positive = profit)
        lambda_risk: Risk aversion parameter for entropic risk
        cvar_alpha: Confidence level for VaR/CVaR
    
    Returns:
        Dictionary with all metrics
    """
    # Convert to losses for VaR/CVaR (negative P&L)
    losses = -pnl
    
    metrics = {
        'mean_pnl': np.mean(pnl),
        'std_pnl': np.std(pnl),
        'min_pnl': np.min(pnl),
        'max_pnl': np.max(pnl),
        'median_pnl': np.median(pnl),
        'skewness': stats.skew(pnl),
        'kurtosis': stats.kurtosis(pnl),
        'var_95': compute_var(losses, 0.95),
        'var_99': compute_var(losses, 0.99),
        'cvar_95': compute_cvar(losses, 0.95),
        'cvar_99': compute_cvar(losses, 0.99),
        'entropic_risk': compute_entropic_risk(pnl, lambda_risk),
        'indifference_price': compute_indifference_price(pnl, lambda_risk)
    }
    
    return metrics


def bootstrap_ci(data: np.ndarray, statistic_func, n_bootstrap: int = 1000,
                confidence: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Input data array
        statistic_func: Function to compute statistic
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level (default 0.95)
        seed: Random seed
    
    Returns:
        Tuple of (point_estimate, lower_ci, upper_ci)
    """
    np.random.seed(seed)
    n = len(data)
    
    # Point estimate
    point_estimate = statistic_func(data)
    
    # Bootstrap resamples
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample_idx = np.random.choice(n, size=n, replace=True)
        bootstrap_stats[i] = statistic_func(data[resample_idx])
    
    # Confidence interval (percentile method)
    alpha = 1 - confidence
    lower_ci = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper_ci = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return point_estimate, lower_ci, upper_ci


def bootstrap_metrics(pnl: np.ndarray, n_bootstrap: int = 1000,
                     confidence: float = 0.95, seed: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for all metrics.
    
    Returns:
        Dictionary with metric names as keys, each containing
        'estimate', 'ci_lower', 'ci_upper'
    """
    results = {}
    
    # Define metric functions
    metric_funcs = {
        'mean_pnl': np.mean,
        'std_pnl': np.std,
        'var_95': lambda x: compute_var(-x, 0.95),
        'var_99': lambda x: compute_var(-x, 0.99),
        'cvar_95': lambda x: compute_cvar(-x, 0.95),
        'cvar_99': lambda x: compute_cvar(-x, 0.99),
        'entropic_risk': lambda x: compute_entropic_risk(x, 1.0)
    }
    
    for name, func in metric_funcs.items():
        try:
            estimate, lower, upper = bootstrap_ci(
                pnl, func, n_bootstrap, confidence, seed
            )
            results[name] = {
                'estimate': estimate,
                'ci_lower': lower,
                'ci_upper': upper
            }
        except Exception as e:
            warnings.warn(f"Bootstrap failed for {name}: {e}")
            results[name] = {
                'estimate': func(pnl),
                'ci_lower': np.nan,
                'ci_upper': np.nan
            }
    
    return results


def paired_ttest(pnl1: np.ndarray, pnl2: np.ndarray, 
                alternative: str = 'two-sided') -> Tuple[float, float]:
    """
    Paired t-test for comparing two strategies.
    
    Args:
        pnl1: P&L from strategy 1
        pnl2: P&L from strategy 2
        alternative: 'two-sided', 'greater', or 'less'
    
    Returns:
        Tuple of (t_statistic, p_value)
    """
    if len(pnl1) != len(pnl2):
        raise ValueError("P&L arrays must have same length for paired test")
    
    t_stat, p_value = stats.ttest_rel(pnl1, pnl2, alternative=alternative)
    return t_stat, p_value


def wilcoxon_test(pnl1: np.ndarray, pnl2: np.ndarray,
                 alternative: str = 'two-sided') -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    """
    if len(pnl1) != len(pnl2):
        raise ValueError("P&L arrays must have same length")
    
    stat, p_value = stats.wilcoxon(pnl1, pnl2, alternative=alternative)
    return stat, p_value


def compare_strategies(pnl_dict: Dict[str, np.ndarray], 
                      baseline_name: str,
                      n_bootstrap: int = 1000) -> Dict[str, Dict]:
    """
    Compare multiple strategies against a baseline.
    
    Args:
        pnl_dict: Dictionary mapping strategy names to P&L arrays
        baseline_name: Name of baseline strategy
        n_bootstrap: Number of bootstrap resamples
    
    Returns:
        Dictionary with comparison results
    """
    if baseline_name not in pnl_dict:
        raise ValueError(f"Baseline {baseline_name} not in pnl_dict")
    
    baseline_pnl = pnl_dict[baseline_name]
    results = {}
    
    for name, pnl in pnl_dict.items():
        if name == baseline_name:
            continue
        
        # Compute metrics
        metrics = compute_metrics(pnl)
        bootstrap_results = bootstrap_metrics(pnl, n_bootstrap)
        
        # Statistical tests
        t_stat, t_pval = paired_ttest(pnl, baseline_pnl)
        w_stat, w_pval = wilcoxon_test(pnl, baseline_pnl)
        
        # Mean difference with CI
        diff = pnl - baseline_pnl
        diff_mean, diff_lower, diff_upper = bootstrap_ci(
            diff, np.mean, n_bootstrap
        )
        
        results[name] = {
            'metrics': metrics,
            'bootstrap_ci': bootstrap_results,
            'vs_baseline': {
                'mean_diff': diff_mean,
                'diff_ci_lower': diff_lower,
                'diff_ci_upper': diff_upper,
                't_statistic': t_stat,
                't_pvalue': t_pval,
                'wilcoxon_statistic': w_stat,
                'wilcoxon_pvalue': w_pval,
                'significant_5pct': t_pval < 0.05,
                'significant_1pct': t_pval < 0.01
            }
        }
    
    return results


def summary_table(comparison_results: Dict[str, Dict], 
                 metrics_to_show: Optional[List[str]] = None) -> str:
    """
    Generate a summary table of comparison results.
    """
    if metrics_to_show is None:
        metrics_to_show = ['mean_pnl', 'std_pnl', 'var_95', 'cvar_95', 'entropic_risk']
    
    lines = []
    header = "Strategy".ljust(20) + " | " + " | ".join([m.ljust(15) for m in metrics_to_show])
    lines.append(header)
    lines.append("-" * len(header))
    
    for strategy, data in comparison_results.items():
        values = []
        for metric in metrics_to_show:
            if metric in data['metrics']:
                val = data['metrics'][metric]
                values.append(f"{val:.4f}".ljust(15))
            else:
                values.append("N/A".ljust(15))
        
        line = strategy.ljust(20) + " | " + " | ".join(values)
        lines.append(line)
    
    return "\n".join(lines)
