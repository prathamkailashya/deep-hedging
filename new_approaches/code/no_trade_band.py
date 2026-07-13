"""
Volatility/gamma-scaled no-trade-band overlay (Whalley-Wilmott 1997;
Zakamouline 2006; Kallsen & Muhle-Karbe 2015).
=====================================================================

Under proportional transaction costs the *optimal* hedge is not to track
the target delta continuously but to hold it inside a no-trade band and
only rebalance when it exits. The asymptotic optimal half-width is

    h_t = K * ( c * S_t * Gamma_t / gamma )^{1/3}

where c is the proportional cost, Gamma_t the option gamma, gamma the
risk-aversion, and K an O(1) constant. Because Gamma_t = phi(d1)/(S sigma
sqrt(tau)) the spot S cancels:

    h_t = K' * ( c / (sigma_t sqrt(tau_t)) )^{1/3} * phi(d1_t)^{1/3},

so the band, in delta units, has microstructurally correct dependencies:
  * h ~ c^{1/3}                 -> WIDER band in high-friction markets
                                   (trade less where trading is expensive)
  * h ~ (phi(d1)/(sigma))^{1/3} -> the band tracks *local gamma*. At fixed
                                   (ATM) moneyness this gives h ~ sigma^{-1/3}
                                   (tighter, more responsive, when vol spikes);
                                   away from ATM the gamma term dominates, so
                                   the width follows where hedging error is
                                   actually risky.

This is the tension a *uniform* turnover penalty cannot resolve: the band
concentrates trading where it is worth the cost (high local gamma) and
suppresses it where it is not, with a cost-scaled threshold. Whether this
lowers CVaR net of cost is settled empirically, not assumed. The overlay is model-agnostic and applied
at inference, so it needs no retraining and preserves interpretability
and deployment simplicity.

Feature layout (walk-forward and synthetic pipelines share it):
    0: normalised price S/S0   1: log-moneyness   2: realised vol
    3: time-to-maturity tau    4: Black-Scholes delta N(d1)
The band needs only features (c, sigma, tau, and d1 via log-moneyness),
so BandedPolicy.forward(features) is a drop-in wrapper usable inside the
existing walk_forward evaluate()/PnL/turnover/bootstrap machinery.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def bs_gamma_delta_band(features: torch.Tensor, c: float, K: float,
                        gamma_ra: float = 1.0,
                        vol_idx: int = 2, tau_idx: int = 3,
                        logmny_idx: int = 1) -> torch.Tensor:
    """No-trade-band half-width h_t in delta units, shape [B, T]."""
    sigma = features[..., vol_idx].clamp(min=1e-3)
    tau = features[..., tau_idx].clamp(min=1e-4)
    logm = features[..., logmny_idx]
    sqrt_tau = torch.sqrt(tau)
    d1 = (logm + 0.5 * sigma ** 2 * tau) / (sigma * sqrt_tau)
    phi = torch.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)   # standard-normal pdf
    # h = K * ( c * phi/(sigma sqrt(tau)) / gamma )^{1/3}   (S cancels via Gamma)
    inner = (c * phi / (sigma * sqrt_tau)) / max(gamma_ra, 1e-8)
    h = K * torch.clamp(inner, min=0.0) ** (1.0 / 3.0)
    return h


def apply_no_trade_band(deltas: torch.Tensor, features: torch.Tensor,
                        c: float, K: float, gamma_ra: float = 1.0) -> torch.Tensor:
    """Rebalance-to-target no-trade band. Hold the position until the
    target delta leaves the band [held - h_t, held + h_t], then move to
    the target. K = 0 recovers the raw policy exactly."""
    if K <= 0:
        return deltas
    B, T = deltas.shape
    h = bs_gamma_delta_band(features, c, K, gamma_ra)          # [B, T]
    held = torch.zeros(B, device=deltas.device, dtype=deltas.dtype)
    out = []
    for t in range(T):
        target = deltas[:, t]
        trigger = (target - held).abs() > h[:, t]
        held = torch.where(trigger, target, held)
        out.append(held)
    return torch.stack(out, dim=1)


class BandedPolicy(nn.Module):
    """Wrap any hedger so forward(features) returns band-filtered deltas.
    Drop-in for walk_forward.evaluate (which calls model(features))."""

    def __init__(self, model: nn.Module, c: float, K: float, gamma_ra: float = 1.0):
        super().__init__()
        self.model = model
        self.c = c
        self.K = K
        self.gamma_ra = gamma_ra

    def parameters(self, *a, **k):
        return self.model.parameters(*a, **k)

    def eval(self):
        self.model.eval(); return self

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.model(features)
        return apply_no_trade_band(raw, features, self.c, self.K, self.gamma_ra)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 6, 30
    feats = torch.randn(B, T, 5)
    feats[..., 1] = feats[..., 1] * 0.1            # log-moneyness near ATM
    feats[..., 2] = feats[..., 2].abs() * 0.2 + 0.1  # vol > 0
    feats[..., 3] = torch.linspace(30/252, 0, T)[None].repeat(B, 1)  # tau
    raw = torch.rand(B, T) * 1.2                    # noisy target deltas in [0,1.2]

    # 1) band widens with cost c
    hi_c = bs_gamma_delta_band(feats, c=0.0018, K=0.5).mean().item()
    lo_c = bs_gamma_delta_band(feats, c=0.0003, K=0.5).mean().item()
    print(f"mean band  NIFTY c=18bps: {hi_c:.4f}   SPY c=3bps: {lo_c:.4f}   "
          f"ratio {hi_c/lo_c:.2f} (expect ~ (6)^(1/3)=1.82)")

    # 2) at fixed ATM moneyness the band tightens as vol rises (h ~ sigma^-1/3)
    atm = feats.clone(); atm[..., 1] = 0.0
    calm = atm.clone(); calm[..., 2] = 0.15
    crisis = atm.clone(); crisis[..., 2] = 0.60
    hb = bs_gamma_delta_band(calm, 0.0018, 0.5).mean().item()
    cb = bs_gamma_delta_band(crisis, 0.0018, 0.5).mean().item()
    print(f"ATM band   calm vol=15%: {hb:.4f}   crisis vol=60%: {cb:.4f}   "
          f"tighter-in-crisis (ATM): {cb < hb}")

    # 3) band reduces turnover; K=0 is identity
    for K in (0.0, 0.3, 0.8):
        b = apply_no_trade_band(raw, feats, c=0.0018, K=K)
        turn = (b[:, 1:] - b[:, :-1]).abs().sum(1).mean() + b[:, 0].abs().mean()
        print(f"K={K}: turnover={turn:.3f}  identity@K0={torch.equal(b, raw) if K==0 else '-'}")
    print("OK")
