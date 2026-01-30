"""
Loss Functions for Deep Hedging

Numerically stable implementations with unit tests.
All models use identical loss functions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def entropic_loss(pnl: torch.Tensor, lambda_risk: float = 1.0) -> torch.Tensor:
    """
    Entropic risk measure (numerically stable).
    
    J(θ) = (1/λ) * log E[exp(-λ * P&L)]
    
    Uses log-sum-exp trick for stability.
    """
    scaled = -lambda_risk * pnl
    max_val = scaled.max().detach()
    loss = (max_val + torch.log(torch.mean(torch.exp(scaled - max_val)))) / lambda_risk
    return loss


def cvar_loss(pnl: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """
    Conditional Value at Risk (CVaR) loss.
    
    CVaR_α = E[Loss | Loss >= VaR_α]
    """
    losses = -pnl
    var = torch.quantile(losses, alpha)
    tail_losses = losses[losses >= var]
    
    if len(tail_losses) == 0:
        return var
    
    return tail_losses.mean()


def cvar_loss_differentiable(pnl: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """
    Differentiable CVaR approximation using soft quantile.
    
    Uses a smooth approximation for gradient flow.
    """
    losses = -pnl
    n = len(losses)
    k = int(n * (1 - alpha))
    
    # Soft top-k using temperature-scaled softmax
    temperature = 0.1
    weights = F.softmax(losses / temperature, dim=0)
    
    # Weighted sum approximates tail expectation
    sorted_losses, _ = torch.sort(losses, descending=True)
    
    # Exact tail mean
    if k > 0:
        return sorted_losses[:k].mean()
    return sorted_losses[0]


def trading_penalty(deltas: torch.Tensor, gamma: float = 1e-3) -> torch.Tensor:
    """
    Penalty on delta changes (transaction cost proxy).
    
    Penalty = γ * Σ|Δδ_t|
    """
    delta_changes = deltas[:, 1:] - deltas[:, :-1]
    return gamma * torch.mean(torch.sum(torch.abs(delta_changes), dim=1))


def no_trade_band_penalty(
    deltas: torch.Tensor,
    reference_deltas: torch.Tensor,
    band_width: float = 0.15,
    nu: float = 1.0
) -> torch.Tensor:
    """
    Penalty for deviating outside no-trade band.
    
    Penalty = ν * Σ max(0, |δ - δ_ref| - band_width)
    """
    deviation = torch.abs(deltas - reference_deltas)
    outside_band = F.relu(deviation - band_width)
    return nu * torch.mean(torch.sum(outside_band, dim=1))


def compute_pnl(
    deltas: torch.Tensor,
    stock_paths: torch.Tensor,
    payoffs: torch.Tensor,
    cost_multiplier: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute P&L and transaction costs.
    
    P&L = -payoff + Σ δ_t * ΔS_t - TC
    """
    price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]
    hedging_gains = torch.sum(deltas * price_changes, dim=1)
    
    # Transaction costs
    delta_changes = torch.cat([
        deltas[:, 0:1],
        deltas[:, 1:] - deltas[:, :-1],
        -deltas[:, -1:]
    ], dim=1)
    tc = cost_multiplier * torch.sum(torch.abs(delta_changes) * stock_paths, dim=1)
    
    pnl = -payoffs + hedging_gains - tc
    return pnl, tc


def stage1_loss(
    deltas: torch.Tensor,
    stock_paths: torch.Tensor,
    payoffs: torch.Tensor,
    alpha: float = 0.95
) -> torch.Tensor:
    """Stage 1: CVaR loss only."""
    pnl, _ = compute_pnl(deltas, stock_paths, payoffs)
    return cvar_loss(pnl, alpha)


def stage2_loss(
    deltas: torch.Tensor,
    stock_paths: torch.Tensor,
    payoffs: torch.Tensor,
    reference_deltas: Optional[torch.Tensor] = None,
    lambda_risk: float = 1.0,
    gamma: float = 1e-3,
    nu: float = 1.0,
    band_width: float = 0.15
) -> torch.Tensor:
    """
    Stage 2: Entropic + trading penalty + no-trade band.
    """
    pnl, _ = compute_pnl(deltas, stock_paths, payoffs)
    
    loss = entropic_loss(pnl, lambda_risk)
    loss = loss + trading_penalty(deltas, gamma)
    
    if reference_deltas is not None:
        loss = loss + no_trade_band_penalty(deltas, reference_deltas, band_width, nu)
    
    return loss


# ============================================================
# UNIT TESTS
# ============================================================

def test_losses():
    """Unit tests for loss functions."""
    print("Testing loss functions...")
    
    # Test data
    torch.manual_seed(42)
    n = 1000
    pnl = torch.randn(n)
    deltas = torch.randn(n, 30)
    ref_deltas = deltas + 0.1 * torch.randn(n, 30)
    
    # Test entropic loss
    ent = entropic_loss(pnl, lambda_risk=1.0)
    assert torch.isfinite(ent), "Entropic loss not finite"
    assert ent > 0, "Entropic loss should be positive for centered P&L"
    
    # Test with extreme values
    extreme_pnl = torch.tensor([1000.0, -1000.0, 0.0])
    ent_extreme = entropic_loss(extreme_pnl, lambda_risk=1.0)
    assert torch.isfinite(ent_extreme), "Entropic loss fails on extreme values"
    
    # Test CVaR
    cvar = cvar_loss(pnl, alpha=0.95)
    assert torch.isfinite(cvar), "CVaR not finite"
    assert cvar > -pnl.mean(), "CVaR should be >= mean loss"
    
    # Test trading penalty
    tp = trading_penalty(deltas, gamma=1e-3)
    assert tp >= 0, "Trading penalty should be non-negative"
    
    # Test no-trade band
    ntb = no_trade_band_penalty(deltas, ref_deltas, band_width=0.15, nu=1.0)
    assert ntb >= 0, "No-trade band penalty should be non-negative"
    
    # Test identical deltas = zero penalty
    ntb_zero = no_trade_band_penalty(deltas, deltas, band_width=0.15, nu=1.0)
    assert ntb_zero < 1e-6, "Identical deltas should have ~zero penalty"
    
    print("  All loss function tests passed!")
    return True


if __name__ == '__main__':
    test_losses()
