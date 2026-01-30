"""Evaluation utilities for deep hedging."""

from .statistical_tests import (
    bootstrap_ci, paired_bootstrap_test, cohens_d,
    bonferroni_correction, holm_bonferroni_correction,
    compare_models, generate_comparison_report, is_improvement,
    cvar_95, cvar_99, var_95, entropic_risk, std_pnl, trading_volume
)

__all__ = [
    'bootstrap_ci', 'paired_bootstrap_test', 'cohens_d',
    'bonferroni_correction', 'holm_bonferroni_correction',
    'compare_models', 'generate_comparison_report', 'is_improvement',
    'cvar_95', 'cvar_99', 'var_95', 'entropic_risk', 'std_pnl', 'trading_volume'
]
