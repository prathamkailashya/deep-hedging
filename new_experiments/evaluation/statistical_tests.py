"""
Statistical Testing for Fair Model Comparison

Implements:
- Bootstrap confidence intervals
- Paired statistical tests with multiplicity correction
- Effect size computation
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class StatisticalResult:
    """Result of statistical comparison."""
    metric: str
    baseline_value: float
    model_value: float
    difference: float
    ci_lower: float
    ci_upper: float
    p_value: float
    significant: bool
    effect_size: float


def bootstrap_ci(
    data: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Returns:
        point_estimate, ci_lower, ci_upper
    """
    np.random.seed(seed)
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        bootstrap_stats.append(metric_fn(data[idx]))
    
    bootstrap_stats = np.array(bootstrap_stats)
    point_estimate = metric_fn(data)
    
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def paired_bootstrap_test(
    baseline_data: np.ndarray,
    model_data: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float, float]:
    """
    Paired bootstrap test for difference in metrics.
    
    Returns:
        difference, ci_lower, ci_upper, p_value
    """
    np.random.seed(seed)
    n = len(baseline_data)
    
    baseline_stat = metric_fn(baseline_data)
    model_stat = metric_fn(model_data)
    observed_diff = model_stat - baseline_stat
    
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        b_stat = metric_fn(baseline_data[idx])
        m_stat = metric_fn(model_data[idx])
        bootstrap_diffs.append(m_stat - b_stat)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
    
    # Two-sided p-value
    centered_diffs = bootstrap_diffs - np.mean(bootstrap_diffs)
    p_value = np.mean(np.abs(centered_diffs) >= np.abs(observed_diff))
    
    return observed_diff, ci_lower, ci_upper, p_value


def cohens_d(baseline_data: np.ndarray, model_data: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(baseline_data), len(model_data)
    var1, var2 = np.var(baseline_data, ddof=1), np.var(model_data, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std < 1e-10:
        return 0.0
    
    return (np.mean(model_data) - np.mean(baseline_data)) / pooled_std


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Apply Bonferroni correction for multiple comparisons."""
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    return [p < adjusted_alpha for p in p_values]


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Apply Holm-Bonferroni step-down correction."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    significant = [False] * n
    
    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        adjusted_alpha = alpha / (n - i)
        if p < adjusted_alpha:
            significant[idx] = True
        else:
            break
    
    return significant


# Metric functions for bootstrap
def cvar_95(pnl: np.ndarray) -> float:
    """CVaR at 95% confidence."""
    losses = -pnl
    threshold = np.percentile(losses, 95)
    return np.mean(losses[losses >= threshold])


def cvar_99(pnl: np.ndarray) -> float:
    """CVaR at 99% confidence."""
    losses = -pnl
    threshold = np.percentile(losses, 99)
    return np.mean(losses[losses >= threshold])


def var_95(pnl: np.ndarray) -> float:
    """VaR at 95% confidence."""
    return np.percentile(-pnl, 95)


def entropic_risk(pnl: np.ndarray, lambda_risk: float = 1.0) -> float:
    """Entropic risk measure."""
    scaled = -lambda_risk * pnl
    max_val = np.max(scaled)
    return (max_val + np.log(np.mean(np.exp(scaled - max_val)))) / lambda_risk


def std_pnl(pnl: np.ndarray) -> float:
    """Standard deviation of P&L."""
    return np.std(pnl)


def trading_volume(deltas: np.ndarray) -> float:
    """Total trading volume."""
    changes = np.abs(np.diff(
        np.concatenate([np.zeros((len(deltas), 1)), deltas, np.zeros((len(deltas), 1))], axis=1),
        axis=1
    ))
    return np.mean(np.sum(changes, axis=1))


def compare_models(
    baseline_pnl: np.ndarray,
    baseline_deltas: np.ndarray,
    model_pnl: np.ndarray,
    model_deltas: np.ndarray,
    model_name: str,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Dict[str, StatisticalResult]:
    """
    Comprehensive statistical comparison of two models.
    
    Returns results for all key metrics with CIs and p-values.
    """
    results = {}
    
    # Define metrics to compare
    metrics = [
        ('cvar_95', cvar_95, baseline_pnl, model_pnl),
        ('cvar_99', cvar_99, baseline_pnl, model_pnl),
        ('var_95', var_95, baseline_pnl, model_pnl),
        ('std_pnl', std_pnl, baseline_pnl, model_pnl),
        ('entropic_risk', entropic_risk, baseline_pnl, model_pnl),
    ]
    
    p_values = []
    
    for metric_name, metric_fn, base_data, model_data in metrics:
        # Bootstrap comparison
        diff, ci_lower, ci_upper, p_value = paired_bootstrap_test(
            base_data, model_data, metric_fn,
            n_bootstrap=n_bootstrap, confidence=confidence
        )
        
        baseline_val = metric_fn(base_data)
        model_val = metric_fn(model_data)
        effect = cohens_d(base_data, model_data)
        
        p_values.append(p_value)
        
        results[metric_name] = StatisticalResult(
            metric=metric_name,
            baseline_value=baseline_val,
            model_value=model_val,
            difference=diff,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            significant=False,  # Will be updated after correction
            effect_size=effect
        )
    
    # Trading volume comparison (using deltas)
    base_vol = trading_volume(baseline_deltas)
    model_vol = trading_volume(model_deltas)
    
    results['trading_volume'] = StatisticalResult(
        metric='trading_volume',
        baseline_value=base_vol,
        model_value=model_vol,
        difference=model_vol - base_vol,
        ci_lower=model_vol - base_vol,
        ci_upper=model_vol - base_vol,
        p_value=1.0,  # Not tested
        significant=False,
        effect_size=0.0
    )
    
    # Apply Holm-Bonferroni correction
    significant = holm_bonferroni_correction(p_values)
    for i, metric_name in enumerate(['cvar_95', 'cvar_99', 'var_95', 'std_pnl', 'entropic_risk']):
        results[metric_name].significant = significant[i]
    
    return results


def generate_comparison_report(
    baseline_name: str,
    comparisons: Dict[str, Dict[str, StatisticalResult]]
) -> str:
    """Generate formatted comparison report."""
    lines = []
    lines.append("="*80)
    lines.append(f"STATISTICAL COMPARISON vs {baseline_name}")
    lines.append("="*80)
    
    for model_name, results in comparisons.items():
        lines.append(f"\n{model_name}:")
        lines.append("-"*60)
        lines.append(f"{'Metric':<15} {'Baseline':>10} {'Model':>10} {'Diff':>10} {'95% CI':>20} {'Sig':>5}")
        lines.append("-"*60)
        
        for metric_name, result in results.items():
            ci_str = f"[{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
            sig_str = "*" if result.significant else ""
            lines.append(
                f"{metric_name:<15} {result.baseline_value:>10.4f} {result.model_value:>10.4f} "
                f"{result.difference:>10.4f} {ci_str:>20} {sig_str:>5}"
            )
    
    lines.append("\n* = Significant after Holm-Bonferroni correction (α=0.05)")
    lines.append("="*80)
    
    return "\n".join(lines)


def is_improvement(
    results: Dict[str, StatisticalResult],
    primary_metric: str = 'cvar_95',
    volume_threshold: float = 1.1
) -> Tuple[bool, str]:
    """
    Check if model represents a genuine improvement.
    
    Criteria:
    - Primary metric significantly better
    - Trading volume not more than threshold × baseline
    """
    primary = results.get(primary_metric)
    volume = results.get('trading_volume')
    
    if primary is None:
        return False, "Primary metric not found"
    
    # For risk metrics, lower is better
    if primary.difference >= 0:
        return False, f"{primary_metric} not improved (diff={primary.difference:.4f})"
    
    if not primary.significant:
        return False, f"{primary_metric} improvement not statistically significant (p={primary.p_value:.4f})"
    
    if volume and volume.model_value > volume.baseline_value * volume_threshold:
        return False, f"Trading volume too high ({volume.model_value:.2f} > {volume.baseline_value * volume_threshold:.2f})"
    
    improvement_pct = -primary.difference / primary.baseline_value * 100
    return True, f"{primary_metric} improved by {improvement_pct:.1f}%"
