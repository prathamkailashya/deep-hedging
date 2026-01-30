"""
PART 8: Financial, Accounting & Risk Analysis

Translates model outputs into:
- Capital requirement reduction
- Regulatory CVaR/VaR impact
- Earnings volatility smoothing
- Transaction cost budgets
- Hedging effectiveness ratios
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class CapitalRequirements:
    """Regulatory capital requirements."""
    var_99: float  # Market risk VaR
    cvar_99: float  # Expected shortfall
    stressed_var: float  # Stressed VaR
    incremental_risk: float
    total_capital: float
    capital_ratio: float


@dataclass
class HedgingEffectiveness:
    """Hedge effectiveness metrics per IAS 39 / IFRS 9."""
    dollar_offset_ratio: float  # Change in hedge / Change in hedged item
    variance_reduction: float  # Reduction in P&L variance
    regression_r_squared: float  # R² of hedge regression
    qualifies_for_hedge_accounting: bool


@dataclass
class RiskManagementReport:
    """Complete risk management report."""
    model_name: str
    
    # Capital metrics
    capital_requirements: CapitalRequirements
    capital_savings_vs_bs: float  # vs Black-Scholes delta
    
    # Hedge effectiveness
    effectiveness: HedgingEffectiveness
    
    # Trading metrics
    expected_transaction_costs: float
    turnover_ratio: float
    max_position: float
    
    # Risk metrics
    tail_risk_reduction: float
    earnings_volatility: float
    max_drawdown: float


class FinancialAnalyzer:
    """
    Comprehensive financial analysis of hedging strategies.
    
    Provides:
    - Regulatory capital calculations
    - Hedge accounting qualification
    - Risk management metrics
    - Economic value analysis
    """
    
    def __init__(
        self,
        confidence_level: float = 0.99,
        holding_period: int = 10,  # days
        stress_multiplier: float = 3.0,
        risk_free_rate: float = 0.05
    ):
        self.confidence = confidence_level
        self.holding_period = holding_period
        self.stress_multiplier = stress_multiplier
        self.rf = risk_free_rate
    
    def compute_capital_requirements(
        self,
        pnl: np.ndarray,
        notional: float = 1_000_000
    ) -> CapitalRequirements:
        """
        Compute regulatory capital requirements.
        
        Based on Basel III/IV framework:
        - VaR at 99% confidence
        - Expected Shortfall (CVaR)
        - Stressed VaR
        - Incremental risk charge
        """
        # Scale to notional
        scaled_pnl = pnl * notional / 100  # Assuming S0=100
        losses = -scaled_pnl
        
        # VaR (10-day, 99%)
        daily_var = np.percentile(losses, self.confidence * 100)
        var_99 = daily_var * np.sqrt(self.holding_period)
        
        # Expected Shortfall (CVaR)
        tail_losses = losses[losses >= daily_var]
        daily_cvar = np.mean(tail_losses) if len(tail_losses) > 0 else daily_var
        cvar_99 = daily_cvar * np.sqrt(self.holding_period)
        
        # Stressed VaR (multiplier for stress periods)
        stressed_var = var_99 * self.stress_multiplier
        
        # Incremental risk charge (simplified)
        incremental_risk = cvar_99 * 0.5
        
        # Total capital (simplified Basel formula)
        # Capital = max(VaR, 3 * avg_VaR) + SVaR + IRC
        total_capital = max(var_99, 3 * var_99) + stressed_var + incremental_risk
        
        # Capital ratio (as % of notional)
        capital_ratio = total_capital / notional * 100
        
        return CapitalRequirements(
            var_99=var_99,
            cvar_99=cvar_99,
            stressed_var=stressed_var,
            incremental_risk=incremental_risk,
            total_capital=total_capital,
            capital_ratio=capital_ratio
        )
    
    def compute_hedge_effectiveness(
        self,
        hedge_pnl: np.ndarray,
        hedged_item_pnl: np.ndarray
    ) -> HedgingEffectiveness:
        """
        Compute hedge effectiveness for accounting purposes.
        
        Per IAS 39 / IFRS 9:
        - Dollar offset method: ratio should be 80-125%
        - Variance reduction method
        - Regression method: R² should be high
        """
        # Dollar offset ratio
        hedge_change = np.sum(hedge_pnl)
        item_change = np.sum(hedged_item_pnl)
        
        if abs(item_change) > 1e-10:
            dollar_offset = -hedge_change / item_change
        else:
            dollar_offset = 1.0
        
        # Variance reduction
        unhedged_var = np.var(hedged_item_pnl)
        hedged_var = np.var(hedge_pnl + hedged_item_pnl)
        
        if unhedged_var > 1e-10:
            variance_reduction = 1 - hedged_var / unhedged_var
        else:
            variance_reduction = 0.0
        
        # Regression R²
        if len(hedge_pnl) > 2 and np.std(hedged_item_pnl) > 1e-10:
            correlation = np.corrcoef(hedge_pnl, -hedged_item_pnl)[0, 1]
            r_squared = correlation ** 2
        else:
            r_squared = 0.0
        
        # Qualification check (IAS 39: 80-125% effectiveness)
        qualifies = (0.80 <= dollar_offset <= 1.25) and (r_squared >= 0.80)
        
        return HedgingEffectiveness(
            dollar_offset_ratio=dollar_offset,
            variance_reduction=variance_reduction,
            regression_r_squared=r_squared,
            qualifies_for_hedge_accounting=qualifies
        )
    
    def compute_transaction_cost_budget(
        self,
        deltas: np.ndarray,
        stock_prices: np.ndarray,
        cost_bps: float = 3.0,
        notional: float = 1_000_000
    ) -> Dict[str, float]:
        """
        Compute expected transaction costs.
        
        Args:
            deltas: Delta positions over time
            stock_prices: Stock price path
            cost_bps: Transaction cost in basis points
            notional: Position notional
        """
        # Delta changes
        delta_changes = np.diff(deltas, axis=1, prepend=0)
        abs_changes = np.abs(delta_changes)
        
        # Cost per trade
        cost_rate = cost_bps / 10000
        
        # Total cost (scaled to notional)
        scale = notional / stock_prices[:, 0:1]
        costs = abs_changes * stock_prices[:, :-1] * cost_rate * scale
        
        total_cost = np.sum(costs, axis=1)
        
        return {
            'mean_cost': float(np.mean(total_cost)),
            'std_cost': float(np.std(total_cost)),
            'max_cost': float(np.max(total_cost)),
            'cost_as_pct_notional': float(np.mean(total_cost) / notional * 100),
            'n_trades_per_path': float(np.mean(np.sum(abs_changes > 0.01, axis=1)))
        }
    
    def generate_risk_report(
        self,
        model_name: str,
        pnl: np.ndarray,
        deltas: np.ndarray,
        stock_paths: np.ndarray,
        payoffs: np.ndarray,
        bs_pnl: Optional[np.ndarray] = None,
        notional: float = 1_000_000
    ) -> RiskManagementReport:
        """Generate comprehensive risk management report."""
        
        # Capital requirements
        capital = self.compute_capital_requirements(pnl, notional)
        
        # Capital savings vs BS
        if bs_pnl is not None:
            bs_capital = self.compute_capital_requirements(bs_pnl, notional)
            capital_savings = bs_capital.total_capital - capital.total_capital
        else:
            capital_savings = 0.0
        
        # Hedge effectiveness
        hedge_pnl = np.sum(deltas * np.diff(stock_paths, axis=1), axis=1)
        hedged_item_pnl = -payoffs
        effectiveness = self.compute_hedge_effectiveness(hedge_pnl, hedged_item_pnl)
        
        # Transaction costs
        tc_analysis = self.compute_transaction_cost_budget(
            deltas, stock_paths, cost_bps=3.0, notional=notional
        )
        
        # Trading metrics
        delta_changes = np.abs(np.diff(deltas, axis=1))
        turnover = np.mean(np.sum(delta_changes, axis=1))
        max_position = np.max(np.abs(deltas))
        
        # Risk metrics
        unhedged_std = np.std(-payoffs)
        hedged_std = np.std(pnl)
        tail_risk_reduction = 1 - capital.cvar_99 / (np.percentile(-payoffs * notional / 100, 99) * np.sqrt(10))
        
        # Max drawdown
        cumulative = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown)
        
        return RiskManagementReport(
            model_name=model_name,
            capital_requirements=capital,
            capital_savings_vs_bs=capital_savings,
            effectiveness=effectiveness,
            expected_transaction_costs=tc_analysis['mean_cost'],
            turnover_ratio=turnover,
            max_position=max_position,
            tail_risk_reduction=tail_risk_reduction,
            earnings_volatility=hedged_std,
            max_drawdown=max_drawdown
        )


def generate_managerial_summary(reports: Dict[str, RiskManagementReport]) -> str:
    """
    Generate executive summary for management.
    
    Focuses on:
    - Capital efficiency
    - Hedge accounting qualification
    - Cost-benefit analysis
    """
    lines = []
    lines.append("=" * 70)
    lines.append("EXECUTIVE SUMMARY: DEEP HEDGING RISK ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    
    # Capital comparison
    lines.append("1. CAPITAL REQUIREMENTS")
    lines.append("-" * 40)
    lines.append(f"{'Model':<20} {'Total Capital':>15} {'Capital Ratio':>15}")
    lines.append("-" * 50)
    
    for name, report in reports.items():
        cap = report.capital_requirements
        lines.append(f"{name:<20} ${cap.total_capital:>14,.0f} {cap.capital_ratio:>14.2f}%")
    
    # Best model
    best_capital = min(reports.items(), key=lambda x: x[1].capital_requirements.total_capital)
    lines.append(f"\nBest capital efficiency: {best_capital[0]}")
    
    # Hedge accounting
    lines.append("\n2. HEDGE ACCOUNTING QUALIFICATION (IAS 39 / IFRS 9)")
    lines.append("-" * 40)
    
    for name, report in reports.items():
        eff = report.effectiveness
        status = "QUALIFIES" if eff.qualifies_for_hedge_accounting else "DOES NOT QUALIFY"
        lines.append(f"{name}: {status}")
        lines.append(f"  - Dollar offset: {eff.dollar_offset_ratio:.2f} (require 0.80-1.25)")
        lines.append(f"  - R-squared: {eff.regression_r_squared:.3f} (require >= 0.80)")
    
    # Cost-benefit
    lines.append("\n3. COST-BENEFIT ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"{'Model':<20} {'Expected TC':>12} {'Turnover':>10} {'Tail Risk ↓':>12}")
    lines.append("-" * 55)
    
    for name, report in reports.items():
        lines.append(
            f"{name:<20} ${report.expected_transaction_costs:>11,.0f} "
            f"{report.turnover_ratio:>10.2f} {report.tail_risk_reduction:>11.1%}"
        )
    
    # Recommendation
    lines.append("\n4. RECOMMENDATION")
    lines.append("-" * 40)
    
    # Find model with best risk-adjusted performance
    scores = {}
    for name, report in reports.items():
        # Score = capital savings - transaction costs (simplified)
        score = report.capital_savings_vs_bs - report.expected_transaction_costs
        scores[name] = score
    
    best_model = max(scores.items(), key=lambda x: x[1])
    lines.append(f"Recommended model: {best_model[0]}")
    lines.append(f"Net benefit score: ${best_model[1]:,.0f}")
    
    return "\n".join(lines)


def run_financial_analysis(
    model_results: Dict[str, Dict[str, np.ndarray]],
    notional: float = 1_000_000
) -> Dict[str, Any]:
    """
    Run complete financial analysis.
    
    Args:
        model_results: {model_name: {'pnl': array, 'deltas': array, 'stock_paths': array, 'payoffs': array}}
    """
    
    print("=" * 70)
    print("FINANCIAL, ACCOUNTING & RISK ANALYSIS")
    print("=" * 70)
    
    analyzer = FinancialAnalyzer()
    reports = {}
    
    for model_name, data in model_results.items():
        print(f"\nAnalyzing {model_name}...")
        
        report = analyzer.generate_risk_report(
            model_name=model_name,
            pnl=data['pnl'],
            deltas=data['deltas'],
            stock_paths=data['stock_paths'],
            payoffs=data['payoffs'],
            bs_pnl=data.get('bs_pnl'),
            notional=notional
        )
        
        reports[model_name] = report
        
        # Print summary
        print(f"  Capital requirement: ${report.capital_requirements.total_capital:,.0f}")
        print(f"  Hedge accounting: {'QUALIFIES' if report.effectiveness.qualifies_for_hedge_accounting else 'NO'}")
        print(f"  Expected TC: ${report.expected_transaction_costs:,.0f}")
    
    # Generate summary
    summary = generate_managerial_summary(reports)
    print("\n" + summary)
    
    return {
        'reports': reports,
        'summary': summary
    }


if __name__ == '__main__':
    # Example usage with synthetic data
    np.random.seed(42)
    n = 10000
    
    model_results = {
        'LSTM': {
            'pnl': np.random.normal(0, 1, n),
            'deltas': np.random.uniform(-1, 1, (n, 30)),
            'stock_paths': 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, (n, 31)), axis=1)),
            'payoffs': np.maximum(100 * np.exp(np.random.normal(0, 0.1, n)) - 100, 0)
        },
        'AttentionLSTM': {
            'pnl': np.random.normal(0.02, 0.95, n),  # Slightly better
            'deltas': np.random.uniform(-1, 1, (n, 30)),
            'stock_paths': 100 * np.exp(np.cumsum(np.random.normal(0, 0.02, (n, 31)), axis=1)),
            'payoffs': np.maximum(100 * np.exp(np.random.normal(0, 0.1, n)) - 100, 0)
        }
    }
    
    results = run_financial_analysis(model_results)
