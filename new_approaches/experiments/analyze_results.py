#!/usr/bin/env python3
"""
Results Analysis Script for Novel Deep Hedging Experiments.

Generates:
- Publication-quality LaTeX tables
- Statistical comparison with Holm-Bonferroni correction
- Visualization plots
- Summary report

Usage:
    python analyze_results.py --results-file results_YYYYMMDD_HHMMSS.pkl
    python analyze_results.py --latest
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats

# Add paths
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


class ResultsAnalyzer:
    """Analyze experiment results and generate reports."""
    
    def __init__(self, results_path: Path):
        self.results_path = results_path
        self.results_dir = results_path.parent
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Load results
        with open(results_path, 'rb') as f:
            self.raw_results = pickle.load(f)
        
        # Statistical parameters (paper.tex aligned)
        self.n_bootstrap = 10000
        self.alpha = 0.05
        
    def get_model_metrics(self, model_name: str, regime: str = 'base') -> Dict[str, np.ndarray]:
        """Extract metrics for a model across all seeds."""
        regime_results = self.raw_results.get(regime, self.raw_results)
        model_results = regime_results.get(model_name, [])
        
        if not model_results:
            return {}
        
        metrics = {}
        metric_names = ['cvar_95', 'cvar_99', 'std_pnl', 'entropic_risk', 'trading_volume', 'train_time']
        
        for metric in metric_names:
            values = []
            for result in model_results:
                if isinstance(result, dict):
                    val = result.get(metric, result.get(metric.replace('_', ''), None))
                else:
                    val = getattr(result, metric, None)
                if val is not None:
                    values.append(val)
            if values:
                metrics[metric] = np.array(values)
        
        return metrics
    
    def bootstrap_ci(self, data: np.ndarray, stat_func=np.mean) -> Tuple[float, float, float]:
        """Compute bootstrap confidence interval."""
        n = len(data)
        if n == 0:
            return 0.0, 0.0, 0.0
            
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(stat_func(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        point = stat_func(data)
        ci_lower = np.percentile(bootstrap_stats, 100 * self.alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - self.alpha / 2))
        
        return point, ci_lower, ci_upper
    
    def paired_comparison(self, baseline: np.ndarray, model: np.ndarray) -> Dict[str, float]:
        """Perform paired statistical comparison."""
        if len(baseline) == 0 or len(model) == 0:
            return {'diff': 0, 'ci_lower': 0, 'ci_upper': 0, 'p_value': 1.0, 'cohens_d': 0}
        
        diff = model - baseline
        
        # Paired t-test
        if len(diff) > 1:
            _, p_value = stats.ttest_rel(model, baseline)
        else:
            p_value = 1.0
        
        # Cohen's d
        std_diff = diff.std() if diff.std() > 0 else 1e-8
        cohens_d = diff.mean() / std_diff
        
        # Bootstrap CI for difference
        point, ci_lower, ci_upper = self.bootstrap_ci(diff)
        
        return {
            'diff': float(diff.mean()),
            'diff_pct': float(diff.mean() / (baseline.mean() + 1e-8) * 100),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d)
        }
    
    def holm_bonferroni(self, p_values: List[float]) -> List[bool]:
        """Apply Holm-Bonferroni correction."""
        n = len(p_values)
        if n == 0:
            return []
        
        sorted_indices = np.argsort(p_values)
        sorted_pvalues = np.array(p_values)[sorted_indices]
        
        significant = [False] * n
        for i, (idx, pval) in enumerate(zip(sorted_indices, sorted_pvalues)):
            adjusted_alpha = self.alpha / (n - i)
            if pval <= adjusted_alpha:
                significant[idx] = True
            else:
                break
        
        return significant
    
    def generate_main_table(self, regime: str = 'base') -> str:
        """Generate main results LaTeX table."""
        regime_results = self.raw_results.get(regime, self.raw_results)
        models = list(regime_results.keys())
        
        latex = []
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(r"\caption{Performance Comparison of Deep Hedging Models (10 Seeds, 50K Training)}")
        latex.append(r"\label{tab:main_results}")
        latex.append(r"\small")
        latex.append(r"\begin{tabular}{lccccc}")
        latex.append(r"\toprule")
        latex.append(r"\textbf{Model} & \textbf{CVaR$_{95}$} & \textbf{CVaR$_{99}$} & \textbf{Std P\&L} & \textbf{Entropic} & \textbf{Time (s)} \\")
        latex.append(r"\midrule")
        
        # Baselines first
        baselines = ['LSTM', 'Transformer']
        novels = ['W-DRO-T', 'RVSN', 'SAC-CVaR', '3SCH', 'RSE']
        
        for model_list in [baselines, novels]:
            for model in model_list:
                if model not in models:
                    continue
                
                metrics = self.get_model_metrics(model, regime)
                if not metrics:
                    continue
                
                row = f"{model} & "
                
                for metric in ['cvar_95', 'cvar_99', 'std_pnl', 'entropic_risk', 'train_time']:
                    if metric in metrics and len(metrics[metric]) > 0:
                        mean = metrics[metric].mean()
                        std = metrics[metric].std()
                        if metric == 'train_time':
                            row += f"{mean:.1f} & "
                        else:
                            row += f"{mean:.3f} $\\pm$ {std:.3f} & "
                    else:
                        row += "-- & "
                
                row = row.rstrip(" & ") + r" \\"
                latex.append(row)
            
            if model_list == baselines and any(m in models for m in novels):
                latex.append(r"\midrule")
        
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        
        return '\n'.join(latex)
    
    def generate_comparison_table(self, baseline_name: str = 'LSTM', regime: str = 'base') -> str:
        """Generate statistical comparison table."""
        regime_results = self.raw_results.get(regime, self.raw_results)
        
        if baseline_name not in regime_results:
            return "% No baseline results available"
        
        baseline_metrics = self.get_model_metrics(baseline_name, regime)
        if 'cvar_95' not in baseline_metrics:
            return "% No baseline CVaR95 available"
        
        baseline_cvar = baseline_metrics['cvar_95']
        
        latex = []
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(r"\caption{Statistical Comparison vs " + baseline_name + r" Baseline (CVaR$_{95}$)}")
        latex.append(r"\label{tab:statistical}")
        latex.append(r"\small")
        latex.append(r"\begin{tabular}{lccccl}")
        latex.append(r"\toprule")
        latex.append(r"\textbf{Model} & \textbf{$\Delta$CVaR$_{95}$} & \textbf{95\% CI} & \textbf{$p$-value} & \textbf{Cohen's $d$} & \textbf{Sig.} \\")
        latex.append(r"\midrule")
        
        comparisons = []
        p_values = []
        
        for model in regime_results.keys():
            if model == baseline_name:
                continue
            
            metrics = self.get_model_metrics(model, regime)
            if 'cvar_95' not in metrics or len(metrics['cvar_95']) == 0:
                continue
            
            comparison = self.paired_comparison(baseline_cvar, metrics['cvar_95'])
            comparisons.append((model, comparison))
            p_values.append(comparison['p_value'])
        
        # Holm-Bonferroni correction
        significant = self.holm_bonferroni(p_values)
        
        for i, (model, comp) in enumerate(comparisons):
            sig_marker = r"$\ast$" if significant[i] else ""
            
            row = f"{model} & "
            row += f"{comp['diff']:+.4f} ({comp['diff_pct']:+.1f}\\%) & "
            row += f"[{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}] & "
            row += f"{comp['p_value']:.4f} & "
            row += f"{comp['cohens_d']:.2f} & "
            row += f"{sig_marker} \\\\"
            latex.append(row)
        
        latex.append(r"\bottomrule")
        latex.append(r"\multicolumn{6}{l}{\footnotesize $\ast$ Significant after Holm-Bonferroni correction ($\alpha=0.05$)} \\")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        
        return '\n'.join(latex)
    
    def generate_summary_report(self, regime: str = 'base') -> str:
        """Generate text summary report."""
        regime_results = self.raw_results.get(regime, self.raw_results)
        
        report = []
        report.append("=" * 70)
        report.append("NOVEL DEEP HEDGING EXPERIMENTS - RESULTS SUMMARY")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Results file: {self.results_path}")
        report.append("")
        
        # Model performance summary
        report.append("MODEL PERFORMANCE (CVaR95)")
        report.append("-" * 40)
        
        results_list = []
        for model in regime_results.keys():
            metrics = self.get_model_metrics(model, regime)
            if 'cvar_95' in metrics and len(metrics['cvar_95']) > 0:
                mean = metrics['cvar_95'].mean()
                std = metrics['cvar_95'].std()
                results_list.append((model, mean, std))
        
        # Sort by CVaR (lower is better)
        results_list.sort(key=lambda x: x[1])
        
        for rank, (model, mean, std) in enumerate(results_list, 1):
            report.append(f"  {rank}. {model}: {mean:.4f} ± {std:.4f}")
        
        report.append("")
        
        # Statistical significance
        if 'LSTM' in regime_results:
            baseline_metrics = self.get_model_metrics('LSTM', regime)
            if 'cvar_95' in baseline_metrics:
                baseline_cvar = baseline_metrics['cvar_95']
                
                report.append("STATISTICAL COMPARISON VS LSTM")
                report.append("-" * 40)
                
                for model in regime_results.keys():
                    if model == 'LSTM':
                        continue
                    
                    metrics = self.get_model_metrics(model, regime)
                    if 'cvar_95' not in metrics or len(metrics['cvar_95']) == 0:
                        continue
                    
                    comp = self.paired_comparison(baseline_cvar, metrics['cvar_95'])
                    sig = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp['p_value'] < 0.05 else ""
                    
                    report.append(f"  {model}:")
                    report.append(f"    Δ CVaR95: {comp['diff']:+.4f} ({comp['diff_pct']:+.1f}%)")
                    report.append(f"    95% CI: [{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}]")
                    report.append(f"    p-value: {comp['p_value']:.4f} {sig}")
                    report.append(f"    Cohen's d: {comp['cohens_d']:.3f}")
                    report.append("")
        
        # Key findings
        report.append("KEY FINDINGS")
        report.append("-" * 40)
        
        if results_list:
            best_model, best_cvar, _ = results_list[0]
            report.append(f"  • Best CVaR95: {best_model} ({best_cvar:.4f})")
            
            if 'LSTM' in [r[0] for r in results_list]:
                lstm_cvar = [r[1] for r in results_list if r[0] == 'LSTM'][0]
                improvements = [(m, (lstm_cvar - c) / lstm_cvar * 100) for m, c, _ in results_list if m != 'LSTM']
                improvements.sort(key=lambda x: -x[1])
                
                for model, imp in improvements[:3]:
                    if imp > 0:
                        report.append(f"  • {model}: {imp:+.2f}% improvement vs LSTM")
        
        report.append("")
        report.append("=" * 70)
        
        return '\n'.join(report)
    
    def save_all_outputs(self):
        """Save all analysis outputs."""
        output_dir = self.results_dir / 'analysis'
        output_dir.mkdir(exist_ok=True)
        
        # Main table
        main_table = self.generate_main_table()
        with open(output_dir / f'main_table_{self.timestamp}.tex', 'w') as f:
            f.write(main_table)
        
        # Comparison table
        comparison_table = self.generate_comparison_table()
        with open(output_dir / f'comparison_table_{self.timestamp}.tex', 'w') as f:
            f.write(comparison_table)
        
        # Summary report
        summary = self.generate_summary_report()
        with open(output_dir / f'summary_{self.timestamp}.txt', 'w') as f:
            f.write(summary)
        
        print(summary)
        print(f"\nOutputs saved to {output_dir}")
        
        return output_dir


def find_latest_results(results_dir: Path) -> Path:
    """Find the most recent results file."""
    pkl_files = list(results_dir.glob('results_*.pkl'))
    if not pkl_files:
        raise FileNotFoundError(f"No results files found in {results_dir}")
    
    # Sort by modification time
    pkl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return pkl_files[0]


def main():
    parser = argparse.ArgumentParser(description='Analyze Novel Deep Hedging Results')
    parser.add_argument('--results-file', type=str, help='Path to results pickle file')
    parser.add_argument('--latest', action='store_true', help='Use latest results file')
    args = parser.parse_args()
    
    results_dir = ROOT / 'new_approaches' / 'results'
    
    if args.results_file:
        results_path = Path(args.results_file)
    elif args.latest:
        results_path = find_latest_results(results_dir)
    else:
        results_path = find_latest_results(results_dir)
    
    print(f"Analyzing: {results_path}")
    
    analyzer = ResultsAnalyzer(results_path)
    analyzer.save_all_outputs()


if __name__ == '__main__':
    main()
