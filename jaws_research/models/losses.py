"""Risk measures (CVaR, entropic, mixed) for the hedger's loss = -PnL.

All functions accept a tensor ``pnl`` (shape (B,)) and return a scalar loss.
"""
from __future__ import annotations
import torch


def cvar_loss(pnl: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """CVaR of -PnL via Rockafellar--Uryasev superquantile."""
    losses = -pnl
    nu = torch.quantile(losses.detach(), alpha)
    excess = torch.clamp(losses - nu, min=0)
    return nu + excess.mean() / (1.0 - alpha)


def entropic_loss(pnl: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Entropic risk = (1/lam) * log E[exp(-lam * pnl)]."""
    return torch.logsumexp(-lam * pnl - torch.log(torch.tensor(float(pnl.numel()))), dim=0) / lam


def mixed_loss(pnl: torch.Tensor, alpha_cvar: float = 0.95,
               lam_ent: float = 1.0, weight_cvar: float = 0.5) -> torch.Tensor:
    return weight_cvar * cvar_loss(pnl, alpha_cvar) + (1 - weight_cvar) * entropic_loss(pnl, lam_ent)


def trading_penalty(deltas: torch.Tensor, prices: torch.Tensor, c_tc: float) -> torch.Tensor:
    """Mean over the batch of sum_k |Delta_k - Delta_{k-1}| * S_k * c_tc.

    deltas : (B, T, d)
    prices : (B, T+1, d)  (only the first T entries used)
    """
    diff = torch.cat([deltas[:, :1, :], deltas[:, 1:, :] - deltas[:, :-1, :]], dim=1)
    cost = (diff.abs() * prices[:, :-1, :] * c_tc).sum(dim=(1, 2))
    return cost.mean()
