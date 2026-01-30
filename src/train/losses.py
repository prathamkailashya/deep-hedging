"""
Loss Functions for Deep Hedging.

Implements risk-based loss functions:
- Entropic Risk (OCE objective)
- Conditional Value-at-Risk (CVaR)
- Combined losses with regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EntropicLoss(nn.Module):
    """
    Entropic risk loss function.
    
    J(θ) = E[exp(-λ * P&L)]
    
    This is the core objective from Buehler et al. (Deep Hedging).
    Minimizing this is equivalent to maximizing expected exponential utility.
    """
    
    def __init__(self, lambda_risk: float = 1.0):
        """
        Args:
            lambda_risk: Risk aversion parameter λ (default: 1.0)
        """
        super().__init__()
        self.lambda_risk = lambda_risk
    
    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """
        Compute entropic loss.
        
        Args:
            pnl: P&L values, shape (batch,)
        
        Returns:
            loss: Scalar loss value
        """
        # Use log-sum-exp trick for numerical stability
        scaled_pnl = -self.lambda_risk * pnl
        max_val = torch.max(scaled_pnl)
        
        # log(E[exp(-λ * P&L)]) = max + log(mean(exp(scaled - max)))
        loss = max_val + torch.log(torch.mean(torch.exp(scaled_pnl - max_val)))
        
        return loss
    
    def indifference_price(self, pnl: torch.Tensor) -> torch.Tensor:
        """
        Compute indifference price.
        
        π(-Z) = (1/λ) * log(E[exp(-λ * P&L)])
        """
        return self.forward(pnl) / self.lambda_risk


class CVaRLoss(nn.Module):
    """
    Conditional Value-at-Risk loss function.
    
    CVaR_α(L) = E[L | L >= VaR_α(L)]
    
    Where L = -P&L (losses).
    """
    
    def __init__(self, alpha: float = 0.95):
        """
        Args:
            alpha: Confidence level (default: 0.95)
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """
        Compute CVaR loss.
        
        Args:
            pnl: P&L values, shape (batch,)
        
        Returns:
            loss: CVaR of losses
        """
        losses = -pnl
        n = losses.size(0)
        k = max(1, int((1 - self.alpha) * n))
        
        # Get k largest losses (worst outcomes)
        top_losses, _ = torch.topk(losses, k)
        
        return torch.mean(top_losses)
    
    def var(self, pnl: torch.Tensor) -> torch.Tensor:
        """Compute VaR at confidence level alpha."""
        losses = -pnl
        n = losses.size(0)
        k = max(1, int((1 - self.alpha) * n))
        
        sorted_losses, _ = torch.sort(losses, descending=True)
        return sorted_losses[k - 1]


class SmoothCVaRLoss(nn.Module):
    """
    Smooth approximation of CVaR for better gradient flow.
    
    Uses the dual representation:
    CVaR_α(X) = inf_t {t + (1/(1-α)) * E[(X - t)_+]}
    """
    
    def __init__(self, alpha: float = 0.95, n_samples: int = 100):
        super().__init__()
        self.alpha = alpha
        self.n_samples = n_samples
    
    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        losses = -pnl
        
        # Search for optimal t
        t_min = losses.min()
        t_max = losses.max()
        t_values = torch.linspace(t_min, t_max, self.n_samples, device=losses.device)
        
        best_cvar = float('inf')
        
        for t in t_values:
            cvar = t + (1 / (1 - self.alpha)) * torch.mean(F.relu(losses - t))
            if cvar < best_cvar:
                best_cvar = cvar
        
        return best_cvar


class CombinedLoss(nn.Module):
    """
    Combined loss function with multiple objectives.
    
    L = w_entropic * Entropic + w_cvar * CVaR + regularization
    """
    
    def __init__(
        self,
        lambda_risk: float = 1.0,
        alpha_cvar: float = 0.95,
        weight_entropic: float = 0.5,
        weight_cvar: float = 0.5,
        weight_variance: float = 0.0
    ):
        super().__init__()
        
        self.entropic_loss = EntropicLoss(lambda_risk)
        self.cvar_loss = CVaRLoss(alpha_cvar)
        
        self.weight_entropic = weight_entropic
        self.weight_cvar = weight_cvar
        self.weight_variance = weight_variance
    
    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """Compute combined loss."""
        loss = 0.0
        
        if self.weight_entropic > 0:
            loss += self.weight_entropic * self.entropic_loss(pnl)
        
        if self.weight_cvar > 0:
            loss += self.weight_cvar * self.cvar_loss(pnl)
        
        if self.weight_variance > 0:
            loss += self.weight_variance * torch.var(pnl)
        
        return loss


class TransactionCostPenalty(nn.Module):
    """
    Transaction cost penalty for two-stage training.
    
    Penalizes large position changes.
    """
    
    def __init__(self, gamma: float = 1e-3):
        """
        Args:
            gamma: Penalty coefficient
        """
        super().__init__()
        self.gamma = gamma
    
    def forward(self, deltas: torch.Tensor) -> torch.Tensor:
        """
        Compute transaction cost penalty.
        
        Args:
            deltas: Hedging positions, shape (batch, n_steps)
        
        Returns:
            penalty: Average transaction cost
        """
        # Include initial and final trades
        deltas_ext = torch.cat([
            torch.zeros(deltas.size(0), 1, device=deltas.device),
            deltas,
            torch.zeros(deltas.size(0), 1, device=deltas.device)
        ], dim=1)
        
        delta_changes = torch.abs(deltas_ext[:, 1:] - deltas_ext[:, :-1])
        total_cost = torch.sum(delta_changes, dim=1)
        
        return self.gamma * torch.mean(total_cost)


class NoTransactionBandPenalty(nn.Module):
    """
    No-transaction band penalty (Kozyra two-stage training).
    
    Penalizes positions outside the band [δ* - h, δ* + h].
    """
    
    def __init__(self, nu: float = 1e8, band_width: float = 0.15):
        """
        Args:
            nu: Penalty coefficient
            band_width: Half-width of no-transaction band
        """
        super().__init__()
        self.nu = nu
        self.band_width = band_width
    
    def forward(self, deltas: torch.Tensor, reference_deltas: torch.Tensor) -> torch.Tensor:
        """
        Compute no-transaction band penalty.
        
        Args:
            deltas: Predicted hedging positions
            reference_deltas: Reference positions (e.g., from Stage 1)
        
        Returns:
            penalty: Band violation penalty
        """
        # Distance outside band: max(0, |δ - δ*| - h)
        violation = F.relu(torch.abs(deltas - reference_deltas) - self.band_width)
        
        return self.nu * torch.mean(violation)


class KozyraTwoStageLoss(nn.Module):
    """
    Complete two-stage loss for Kozyra model.
    
    Stage 1: CVaR_α(P&L)
    Stage 2: Entropic(P&L) + γ * Σ|δ_{k+1} - δ_k| + ν * d(δ, H_c)
    """
    
    def __init__(
        self,
        lambda_risk: float = 1.0,
        alpha_cvar: float = 0.95,
        gamma: float = 1e-3,
        nu: float = 1e8,
        band_width: float = 0.15
    ):
        super().__init__()
        
        self.entropic_loss = EntropicLoss(lambda_risk)
        self.cvar_loss = CVaRLoss(alpha_cvar)
        self.tc_penalty = TransactionCostPenalty(gamma)
        self.band_penalty = NoTransactionBandPenalty(nu, band_width)
        
        self.stage = 1
    
    def set_stage(self, stage: int):
        """Set training stage (1 or 2)."""
        if stage not in [1, 2]:
            raise ValueError("Stage must be 1 or 2")
        self.stage = stage
    
    def forward(
        self,
        pnl: torch.Tensor,
        deltas: torch.Tensor,
        reference_deltas: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss for current stage.
        
        Args:
            pnl: P&L values
            deltas: Current hedging positions
            reference_deltas: Reference positions (required for Stage 2)
        
        Returns:
            loss: Total loss for current stage
        """
        if self.stage == 1:
            return self.cvar_loss(pnl)
        
        else:  # Stage 2
            if reference_deltas is None:
                raise ValueError("reference_deltas required for Stage 2")
            
            entropic = self.entropic_loss(pnl)
            tc = self.tc_penalty(deltas)
            band = self.band_penalty(deltas, reference_deltas)
            
            return entropic + tc + band


class MeanVarianceLoss(nn.Module):
    """
    Mean-variance objective for comparison.
    
    L = -E[P&L] + λ * Var[P&L]
    """
    
    def __init__(self, risk_aversion: float = 1.0):
        super().__init__()
        self.risk_aversion = risk_aversion
    
    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        mean_pnl = torch.mean(pnl)
        var_pnl = torch.var(pnl)
        
        return -mean_pnl + self.risk_aversion * var_pnl


class HedgingLoss(nn.Module):
    """
    Unified hedging loss combining P&L computation and risk measure.
    """
    
    def __init__(
        self,
        cost_multiplier: float = 0.0,
        lambda_risk: float = 1.0,
        risk_measure: str = 'entropic'
    ):
        super().__init__()
        
        self.cost_multiplier = cost_multiplier
        
        if risk_measure == 'entropic':
            self.risk_loss = EntropicLoss(lambda_risk)
        elif risk_measure == 'cvar':
            self.risk_loss = CVaRLoss(0.95)
        elif risk_measure == 'mean_variance':
            self.risk_loss = MeanVarianceLoss(lambda_risk)
        else:
            raise ValueError(f"Unknown risk measure: {risk_measure}")
    
    def compute_pnl(
        self,
        deltas: torch.Tensor,
        stock_paths: torch.Tensor,
        payoffs: torch.Tensor
    ) -> torch.Tensor:
        """Compute P&L from hedging strategy."""
        # Price changes
        price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]
        
        # Hedging gains
        hedging_gains = torch.sum(deltas * price_changes, dim=1)
        
        # Transaction costs
        deltas_ext = torch.cat([
            torch.zeros(deltas.size(0), 1, device=deltas.device),
            deltas,
            torch.zeros(deltas.size(0), 1, device=deltas.device)
        ], dim=1)
        delta_changes = torch.abs(deltas_ext[:, 1:] - deltas_ext[:, :-1])
        transaction_costs = self.cost_multiplier * torch.sum(delta_changes, dim=1)
        
        # P&L
        pnl = -payoffs + hedging_gains - transaction_costs
        
        return pnl
    
    def forward(
        self,
        deltas: torch.Tensor,
        stock_paths: torch.Tensor,
        payoffs: torch.Tensor
    ) -> torch.Tensor:
        """Compute hedging loss."""
        pnl = self.compute_pnl(deltas, stock_paths, payoffs)
        return self.risk_loss(pnl)
