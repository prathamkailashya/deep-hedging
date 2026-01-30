"""
Comprehensive Statistical Analysis Module

Provides:
- Bootstrap confidence intervals
- Paired statistical tests
- Multiple comparison correction (Holm-Bonferroni)
- Effect size computation (Cohen's d)
- Seed robustness analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval result."""
    estimate: float
    ci_lower: float
    ci_upper: float
    se: float
    n_bootstrap: int


@dataclass
class PairedTestResult:
    """Result of paired statistical test."""
    model_a: str
    model_b: str
    metric: str
    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    effect_size: float  # Cohen's d
    significant: bool
    n_seeds: int


@dataclass
class ModelComparisonResult:
    """Complete comparison between two models."""
    model_a: str
    model_b: str
    metrics: Dict[str, PairedTestResult]
    overall_significant: bool
    n_significant_metrics: int


def bootstrap_ci(
    data: np.ndarray,
    statistic_func,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> BootstrapCI:
    """
    Compute bootstrap confidence interval for any statistic.
    
    Args:
        data: 1D array of observations
        statistic_func: Function to compute statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed
    
    Returns:
        BootstrapCI with estimate, CI, and SE
    """
    np.random.seed(seed)
    n = len(data)
    
    # Point estimate
    estimate = statistic_func(data)
    
    # Bootstrap
    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        boot_sample = data[np.random.randint(0, n, n)]
        boot_stats[i] = statistic_func(boot_sample)
    
    # Confidence interval (percentile method)
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    
    # Standard error
    se = np.std(boot_stats)
    
    return BootstrapCI(
        estimate=float(estimate),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        se=float(se),
        n_bootstrap=n_bootstrap
    )


def cvar_statistic(pnl: np.ndarray, alpha: float = 0.95) -> float:
    """Compute CVaR (Expected Shortfall)."""
    losses = -pnl
    var = np.percentile(losses, alpha * 100)
    return float(np.mean(losses[losses >= var]))


def paired_bootstrap_test(
    data_a: np.ndarray,
    data_b: np.ndarray,
    statistic_func,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float, float]:
    """
    Paired bootstrap test for difference in statistics.
    
    Returns:
        (mean_diff, ci_lower, ci_upper, p_value)
    """
    np.random.seed(seed)
    n = len(data_a)
    assert len(data_b) == n, "Arrays must have same length"
    
    # Point estimates
    stat_a = statistic_func(data_a)
    stat_b = statistic_func(data_b)
    observed_diff = stat_b - stat_a
    
    # Bootstrap differences
    boot_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        boot_a = statistic_func(data_a[idx])
        boot_b = statistic_func(data_b[idx])
        boot_diffs[i] = boot_b - boot_a
    
    # Confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    
    # Two-sided p-value (proportion of bootstrap diffs on opposite side of 0)
    if observed_diff >= 0:
        p_value = 2 * np.mean(boot_diffs <= 0)
    else:
        p_value = 2 * np.mean(boot_diffs >= 0)
    p_value = min(p_value, 1.0)
    
    return float(observed_diff), float(ci_lower), float(ci_upper), float(p_value)


def cohens_d(data_a: np.ndarray, data_b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for paired data.
    
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    diff = data_b - data_a
    return float(np.mean(diff) / (np.std(diff) + 1e-10))


def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.
    
    Returns list of booleans indicating significance after correction.
    """
    n = len(p_values)
    if n == 0:
        return []
    
    # Sort p-values with original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    
    # Apply correction
    significant = [False] * n
    for i, (orig_idx, p) in enumerate(indexed):
        corrected_alpha = alpha / (n - i)
        if p <= corrected_alpha:
            significant[orig_idx] = True
        else:
            # Once we fail to reject, all remaining are not significant
            break
    
    return significant


class SeedRobustnessAnalyzer:
    """
    Analyze robustness across multiple random seeds.
    
    Ensures results are not artifacts of specific initialization.
    """
    
    def __init__(self, n_seeds: int = 10, confidence: float = 0.95):
        self.n_seeds = n_seeds
        self.confidence = confidence
    
    def analyze_metric_across_seeds(
        self,
        seed_results: Dict[int, float]
    ) -> Dict[str, float]:
        """
        Analyze a single metric across seeds.
        
        Args:
            seed_results: {seed: metric_value}
        
        Returns:
            Statistics about the metric across seeds
        """
        values = np.array(list(seed_results.values()))
        n = len(values)
        
        # Compute statistics
        mean = np.mean(values)
        std = np.std(values, ddof=1) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 0 else 0.0
        
        # Confidence interval (t-distribution)
        if n > 1:
            t_crit = stats.t.ppf((1 + self.confidence) / 2, n - 1)
            ci_lower = mean - t_crit * se
            ci_upper = mean + t_crit * se
        else:
            ci_lower = ci_upper = mean
        
        return {
            'mean': float(mean),
            'std': float(std),
            'se': float(se),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'range': float(np.max(values) - np.min(values)),
            'cv': float(std / abs(mean)) if abs(mean) > 1e-10 else 0.0,  # Coefficient of variation
            'n_seeds': n
        }
    
    def paired_model_comparison(
        self,
        model_a_seeds: Dict[int, float],
        model_b_seeds: Dict[int, float],
        model_a_name: str,
        model_b_name: str,
        metric_name: str
    ) -> PairedTestResult:
        """
        Compare two models across seeds using paired t-test.
        
        Uses same seeds for both models for fair comparison.
        """
        # Get common seeds
        common_seeds = sorted(set(model_a_seeds.keys()) & set(model_b_seeds.keys()))
        n = len(common_seeds)
        
        if n < 2:
            warnings.warn(f"Only {n} common seeds - insufficient for paired test")
            return PairedTestResult(
                model_a=model_a_name,
                model_b=model_b_name,
                metric=metric_name,
                mean_diff=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                p_value=1.0,
                effect_size=0.0,
                significant=False,
                n_seeds=n
            )
        
        # Extract paired values
        vals_a = np.array([model_a_seeds[s] for s in common_seeds])
        vals_b = np.array([model_b_seeds[s] for s in common_seeds])
        
        # Paired t-test
        diff = vals_b - vals_a
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        se_diff = std_diff / np.sqrt(n)
        
        # t-statistic and p-value
        t_stat = mean_diff / (se_diff + 1e-10)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
        
        # Confidence interval
        t_crit = stats.t.ppf((1 + self.confidence) / 2, n - 1)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
        
        # Effect size
        effect = cohens_d(vals_a, vals_b)
        
        return PairedTestResult(
            model_a=model_a_name,
            model_b=model_b_name,
            metric=metric_name,
            mean_diff=float(mean_diff),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(p_value),
            effect_size=float(effect),
            significant=p_value < (1 - self.confidence),
            n_seeds=n
        )


class ComprehensiveComparison:
    """
    Complete statistical comparison framework.
    
    Compares all models against baseline with:
    - Multiple metrics
    - Seed robustness
    - Bootstrap CIs
    - Multiple comparison correction
    """
    
    def __init__(
        self,
        baseline_name: str = 'LSTM',
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        alpha: float = 0.05
    ):
        self.baseline_name = baseline_name
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.alpha = alpha
        self.seed_analyzer = SeedRobustnessAnalyzer(confidence=confidence)
    
    def compare_models(
        self,
        results: Dict[str, Dict[int, Dict[str, float]]],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare all models against baseline.
        
        Args:
            results: {model_name: {seed: {metric: value}}}
            metrics: List of metrics to compare
        
        Returns:
            Comprehensive comparison results
        """
        if metrics is None:
            metrics = ['cvar_95', 'cvar_99', 'std_pnl', 'entropic_risk', 'trading_volume']
        
        if self.baseline_name not in results:
            raise ValueError(f"Baseline {self.baseline_name} not found in results")
        
        baseline_results = results[self.baseline_name]
        comparison_results = {}
        all_p_values = []
        p_value_indices = []  # Track which comparison each p-value belongs to
        
        for model_name, model_results in results.items():
            if model_name == self.baseline_name:
                continue
            
            model_comparison = {}
            
            for metric in metrics:
                # Extract metric values per seed
                baseline_seeds = {s: r[metric] for s, r in baseline_results.items() if metric in r}
                model_seeds = {s: r[metric] for s, r in model_results.items() if metric in r}
                
                # Paired comparison
                test_result = self.seed_analyzer.paired_model_comparison(
                    baseline_seeds, model_seeds,
                    self.baseline_name, model_name, metric
                )
                
                model_comparison[metric] = test_result
                all_p_values.append(test_result.p_value)
                p_value_indices.append((model_name, metric))
            
            comparison_results[model_name] = model_comparison
        
        # Apply Holm-Bonferroni correction
        corrected_significant = holm_bonferroni_correction(all_p_values, self.alpha)
        
        # Update significance after correction
        for i, (model_name, metric) in enumerate(p_value_indices):
            comparison_results[model_name][metric].significant = corrected_significant[i]
        
        # Compute summary
        summary = self._compute_summary(comparison_results, metrics)
        
        return {
            'baseline': self.baseline_name,
            'comparisons': comparison_results,
            'summary': summary,
            'n_comparisons': len(all_p_values),
            'alpha': self.alpha,
            'confidence': self.confidence
        }
    
    def _compute_summary(
        self,
        comparisons: Dict[str, Dict[str, PairedTestResult]],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Compute summary statistics."""
        summary = {}
        
        for model_name, model_results in comparisons.items():
            n_sig = sum(1 for r in model_results.values() if r.significant)
            n_improved = sum(1 for m, r in model_results.items() 
                           if r.mean_diff < 0 and m in ['cvar_95', 'cvar_99', 'std_pnl', 'entropic_risk'])
            
            summary[model_name] = {
                'n_significant': n_sig,
                'n_improved': n_improved,
                'n_metrics': len(metrics),
                'pct_significant': n_sig / len(metrics) * 100 if metrics else 0,
                'overall_better': n_improved > len(metrics) / 2
            }
        
        return summary


def generate_comparison_table(
    comparison_results: Dict[str, Any],
    latex: bool = False
) -> str:
    """Generate formatted comparison table."""
    baseline = comparison_results['baseline']
    comparisons = comparison_results['comparisons']
    
    lines = []
    
    if latex:
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lrrrrr}")
        lines.append("\\toprule")
        lines.append(f"Model vs {baseline} & Mean Diff & 95\\% CI & p-value & Cohen's d & Sig \\\\")
        lines.append("\\midrule")
    else:
        lines.append(f"{'Model':<20} {'Metric':<15} {'Diff':>10} {'95% CI':>20} {'p':>8} {'d':>8} {'Sig':>5}")
        lines.append("-" * 90)
    
    for model_name, metrics in comparisons.items():
        for metric_name, result in metrics.items():
            sig_marker = "*" if result.significant else ""
            ci_str = f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
            
            if latex:
                lines.append(
                    f"{model_name} & {result.mean_diff:.4f} & {ci_str} & "
                    f"{result.p_value:.4f} & {result.effect_size:.3f} & {sig_marker} \\\\"
                )
            else:
                lines.append(
                    f"{model_name:<20} {metric_name:<15} {result.mean_diff:>10.4f} "
                    f"{ci_str:>20} {result.p_value:>8.4f} {result.effect_size:>8.3f} {sig_marker:>5}"
                )
    
    if latex:
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Model comparison with Holm-Bonferroni correction}")
        lines.append("\\end{table}")
    
    return "\n".join(lines)
