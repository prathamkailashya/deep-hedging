"""
PART 7: Novel Enhancements for Deep Hedging

Implements:
1. Ensemble Hedging: δ = median(δ_LSTM, δ_Attention, δ_Signature)
2. Distributionally Robust Hedging: optimize worst-case CVaR
3. Volatility-Regime Conditioning: condition on realized vol state
4. Market-Impact Aware Hedging: include temporary & permanent impact
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


class EnsembleHedger(nn.Module):
    """
    Ensemble of hedging models.
    
    Combines multiple models using:
    - Median (robust to outliers)
    - Mean (simple average)
    - Weighted (learned or fixed weights)
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        method: str = 'median',
        weights: Optional[List[float]] = None,
        delta_max: float = 1.5
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.method = method
        self.delta_max = delta_max
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)
        
        # Learnable weights for 'learned' method
        if method == 'learned':
            self.weight_net = nn.Sequential(
                nn.Linear(len(models), len(models)),
                nn.Softmax(dim=-1)
            )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Combine model outputs."""
        # Get all model predictions
        all_deltas = []
        for model in self.models:
            with torch.no_grad():
                deltas = model(features)
            all_deltas.append(deltas)
        
        # Stack: (n_models, batch, n_steps)
        stacked = torch.stack(all_deltas, dim=0)
        
        if self.method == 'median':
            combined = torch.median(stacked, dim=0)[0]
        elif self.method == 'mean':
            combined = torch.mean(stacked, dim=0)
        elif self.method == 'weighted':
            weights = self.weights.view(-1, 1, 1).to(stacked.device)
            combined = (stacked * weights).sum(dim=0)
        elif self.method == 'learned':
            # Use weight network
            model_outputs = stacked.permute(1, 2, 0)  # (batch, steps, n_models)
            weights = self.weight_net(model_outputs)  # (batch, steps, n_models)
            combined = (model_outputs * weights).sum(dim=-1)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return torch.clamp(combined, -self.delta_max, self.delta_max)
    
    def get_individual_predictions(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get predictions from each model for analysis."""
        predictions = {}
        for i, model in enumerate(self.models):
            with torch.no_grad():
                predictions[f'model_{i}'] = model(features)
        return predictions


class DistributionallyRobustHedger(nn.Module):
    """
    Distributionally Robust Optimization for hedging.
    
    Trains to minimize worst-case CVaR across multiple
    stochastic volatility models (Heston, SABR, etc.)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        n_scenarios: int = 5,
        ambiguity_radius: float = 0.1,
        delta_max: float = 1.5
    ):
        super().__init__()
        self.base_model = base_model
        self.n_scenarios = n_scenarios
        self.ambiguity_radius = ambiguity_radius
        self.delta_max = delta_max
        
        # Scenario weights (Wasserstein ball)
        self.scenario_weights = nn.Parameter(
            torch.ones(n_scenarios) / n_scenarios
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate robust hedge."""
        return self.base_model(features)
    
    def compute_robust_loss(
        self,
        features_list: List[torch.Tensor],
        stock_paths_list: List[torch.Tensor],
        payoffs_list: List[torch.Tensor],
        alpha: float = 0.95
    ) -> torch.Tensor:
        """
        Compute worst-case CVaR across scenarios.
        
        Args:
            features_list: Features from different stochastic models
            stock_paths_list: Stock paths from different models
            payoffs_list: Payoffs from different models
        """
        scenario_cvars = []
        
        for features, paths, payoffs in zip(features_list, stock_paths_list, payoffs_list):
            deltas = self.forward(features)
            
            # Compute P&L
            price_changes = paths[:, 1:] - paths[:, :-1]
            hedge_gains = torch.sum(deltas * price_changes, dim=1)
            pnl = -payoffs + hedge_gains
            
            # CVaR
            losses = -pnl
            var = torch.quantile(losses, alpha)
            cvar = torch.mean(losses[losses >= var])
            scenario_cvars.append(cvar)
        
        cvars = torch.stack(scenario_cvars)
        
        # Worst-case: maximize over probability weights
        weights = F.softmax(self.scenario_weights, dim=0)
        
        # Add entropy regularization for Wasserstein ball
        uniform = torch.ones_like(weights) / self.n_scenarios
        kl_div = torch.sum(weights * torch.log(weights / uniform + 1e-10))
        
        # Robust objective: weighted CVaR + penalty for deviation from uniform
        robust_loss = torch.sum(weights * cvars) + self.ambiguity_radius * kl_div
        
        return robust_loss


class RegimeConditionedHedger(nn.Module):
    """
    Volatility-regime conditioned hedging.
    
    Uses different sub-networks for different market regimes:
    - Low volatility regime
    - Normal volatility regime
    - High volatility regime
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        n_regimes: int = 3,
        vol_thresholds: Tuple[float, float] = (0.15, 0.30),
        delta_max: float = 1.5
    ):
        super().__init__()
        self.n_regimes = n_regimes
        self.vol_thresholds = vol_thresholds
        self.delta_max = delta_max
        
        # Regime-specific networks
        self.regime_nets = nn.ModuleList([
            nn.LSTM(input_dim, hidden_size, num_layers=2, batch_first=True)
            for _ in range(n_regimes)
        ])
        
        self.regime_fcs = nn.ModuleList([
            nn.Linear(hidden_size, 1)
            for _ in range(n_regimes)
        ])
        
        # Soft regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_regimes),
            nn.Softmax(dim=-1)
        )
    
    def _classify_regime(self, features: torch.Tensor) -> torch.Tensor:
        """Classify regime based on volatility features."""
        # Use soft classification
        return self.regime_classifier(features)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate regime-aware deltas."""
        batch, n_steps, _ = features.shape
        
        # Get regime weights for each timestep
        regime_weights = self._classify_regime(features)  # (batch, steps, n_regimes)
        
        # Get outputs from each regime network
        regime_outputs = []
        for i, (lstm, fc) in enumerate(zip(self.regime_nets, self.regime_fcs)):
            lstm_out, _ = lstm(features)
            raw_delta = fc(lstm_out).squeeze(-1)
            regime_outputs.append(raw_delta)
        
        # Stack: (batch, steps, n_regimes)
        stacked = torch.stack(regime_outputs, dim=-1)
        
        # Weighted combination
        combined = (stacked * regime_weights).sum(dim=-1)
        
        return self.delta_max * torch.tanh(combined)


class MarketImpactHedger(nn.Module):
    """
    Market-impact aware hedging.
    
    Accounts for:
    - Temporary impact: affects execution price
    - Permanent impact: affects future prices
    - Order book dynamics
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        temp_impact_coef: float = 0.001,
        perm_impact_coef: float = 0.0001,
        max_order_size: float = 0.5,
        delta_max: float = 1.5
    ):
        super().__init__()
        self.base_model = base_model
        self.temp_impact = temp_impact_coef
        self.perm_impact = perm_impact_coef
        self.max_order_size = max_order_size
        self.delta_max = delta_max
        
        # Impact-aware adjustment network
        self.impact_adjuster = nn.Sequential(
            nn.Linear(3, 16),  # [raw_delta, delta_change, vol]
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate impact-adjusted deltas."""
        # Get base delta
        raw_deltas = self.base_model(features)
        
        batch, n_steps = raw_deltas.shape
        adjusted_deltas = torch.zeros_like(raw_deltas)
        prev_delta = torch.zeros(batch, device=features.device)
        
        for t in range(n_steps):
            raw_delta = raw_deltas[:, t]
            delta_change = raw_delta - prev_delta
            vol = features[:, t, 2] if features.shape[-1] > 2 else torch.ones(batch)
            
            # Compute impact adjustment
            impact_input = torch.stack([raw_delta, delta_change, vol], dim=-1)
            adjustment = self.impact_adjuster(impact_input).squeeze(-1)
            
            # Limit order size to reduce impact
            clipped_change = torch.clamp(
                delta_change,
                -self.max_order_size,
                self.max_order_size
            )
            
            # Apply adjustment
            adjusted_delta = prev_delta + clipped_change * (1 - 0.1 * adjustment)
            adjusted_deltas[:, t] = torch.clamp(adjusted_delta, -self.delta_max, self.delta_max)
            prev_delta = adjusted_deltas[:, t]
        
        return adjusted_deltas
    
    def compute_execution_cost(
        self,
        deltas: torch.Tensor,
        prices: torch.Tensor,
        volumes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute total execution cost including market impact."""
        delta_changes = torch.diff(deltas, dim=1, prepend=torch.zeros_like(deltas[:, :1]))
        
        # Temporary impact
        temp_cost = self.temp_impact * torch.abs(delta_changes) * prices[:, :-1]
        
        # Permanent impact (accumulates)
        perm_impact = self.perm_impact * torch.cumsum(delta_changes, dim=1)
        perm_cost = perm_impact.abs() * prices[:, :-1]
        
        total_cost = temp_cost.sum(dim=1) + perm_cost.sum(dim=1)
        return total_cost


def run_enhancement_comparison(
    base_models: Dict[str, nn.Module],
    features: torch.Tensor,
    stock_paths: torch.Tensor,
    payoffs: torch.Tensor,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Compare novel enhancement methods.
    
    Args:
        base_models: {name: model} for ensemble
        features, stock_paths, payoffs: Test data
    """
    
    print("=" * 70)
    print("NOVEL ENHANCEMENT COMPARISON")
    print("=" * 70)
    
    results = {}
    
    # 1. Ensemble methods
    print("\n[1] Ensemble Methods")
    print("-" * 40)
    
    models_list = list(base_models.values())
    
    for method in ['median', 'mean', 'weighted']:
        ensemble = EnsembleHedger(models_list, method=method)
        ensemble.eval()
        
        with torch.no_grad():
            deltas = ensemble(features)
            price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]
            hedge_gains = torch.sum(deltas * price_changes, dim=1)
            pnl = -payoffs + hedge_gains
            
            losses = -pnl
            cvar = float(torch.mean(losses[losses >= torch.quantile(losses, 0.95)]))
        
        results[f'ensemble_{method}'] = {'cvar_95': cvar}
        print(f"  Ensemble ({method}): CVaR95 = {cvar:.4f}")
    
    # 2. Regime-conditioned (requires retraining, show structure only)
    print("\n[2] Regime-Conditioned Hedger")
    print("-" * 40)
    
    regime_model = RegimeConditionedHedger(input_dim=features.shape[-1])
    n_params = sum(p.numel() for p in regime_model.parameters())
    print(f"  Parameters: {n_params:,}")
    print("  (Requires separate training)")
    
    # 3. Market Impact Hedger
    print("\n[3] Market-Impact Aware Hedger")
    print("-" * 40)
    
    base_model = list(base_models.values())[0]
    impact_model = MarketImpactHedger(base_model)
    
    with torch.no_grad():
        deltas = impact_model(features)
        exec_cost = impact_model.compute_execution_cost(deltas, stock_paths)
        
        price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]
        hedge_gains = torch.sum(deltas * price_changes, dim=1)
        pnl = -payoffs + hedge_gains - exec_cost
        
        losses = -pnl
        cvar = float(torch.mean(losses[losses >= torch.quantile(losses, 0.95)]))
    
    results['market_impact'] = {
        'cvar_95': cvar,
        'mean_exec_cost': float(exec_cost.mean())
    }
    print(f"  CVaR95 (with impact): {cvar:.4f}")
    print(f"  Mean execution cost: {exec_cost.mean():.4f}")
    
    return results


if __name__ == '__main__':
    # Example usage
    from new_experiments.models.base_model import LSTMHedger
    from new_experiments.models.attention_lstm import AttentionLSTM
    from new_experiments.data.data_generator import DataGenerator, HestonParams
    
    # Generate test data
    data_gen = DataGenerator(HestonParams())
    _, _, test_data = data_gen.generate_splits(n_train=1000, n_val=1000, n_test=10000)
    
    # Create base models
    models = {
        'LSTM': LSTMHedger(input_dim=4, hidden_size=50),
        'AttentionLSTM': AttentionLSTM(input_dim=4, hidden_size=64)
    }
    
    # Run comparison
    results = run_enhancement_comparison(
        models,
        test_data.features,
        test_data.stock_paths,
        test_data.payoffs
    )
