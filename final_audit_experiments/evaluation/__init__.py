"""Evaluation module for statistical analysis."""
from .statistical_analysis import (
    BootstrapCI, PairedTestResult, ModelComparisonResult,
    bootstrap_ci, cvar_statistic, paired_bootstrap_test, cohens_d,
    holm_bonferroni_correction, SeedRobustnessAnalyzer, ComprehensiveComparison,
    generate_comparison_table
)
