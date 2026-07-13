"""
W-DRO-T with a volatility-conditioned (adaptive) Wasserstein radius.
====================================================================

Motivation (diagnosed empirically). The vanilla W-DRO-T penalty
    L_DRO = L_base + eps * E[ ||grad_I L_base||_2 ]
penalises the sensitivity of the loss to *every* input dimension
uniformly. In a crisis the hedger must be *more* responsive to the
state (larger, faster delta moves), yet a uniform gradient-norm
penalty suppresses exactly that responsiveness. The pass-5 walk-forward
pilot confirmed the failure mode: vanilla W-DRO-T produced the lowest
turnover and the *highest* CVaR of the four models on the SPY COVID
cell -- it under-hedged the crisis.

Fix (data-driven / state-dependent ambiguity radius). Wasserstein-DRO
theory does not require a constant radius: the ambiguity set should
reflect where distributional uncertainty actually is, and the radius is
naturally data-driven (Mohajerin Esfahani & Kuhn 2018; Gao & Kleywegt
2023; Blanchet & Murthy 2019). We make the *effective* radius shrink in
high-volatility states, so robustification concentrates in calm regimes
(where over-fitting to the nominal law is the real risk) and relaxes in
crises (where responsiveness matters):

    eps_eff(x) = eps * exp( -beta * relu( vol(x)/vol_ref - 1 ) )

so eps_eff = eps when realised vol <= vol_ref and decays smoothly above
it. vol(x) is the realised-volatility input feature (index 2 of the
five Heston-style features). This is a strict generalisation: beta = 0
recovers the vanilla constant-radius W-DRO-T exactly.

Everything else (the Blanchet-Murthy gradient-norm dual, the second-order
gradient, the 80-epoch two-stage schedule) is unchanged, so the
comparison isolates the effect of the adaptive radius.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Dict, Tuple

from w_dro_t import WassersteinDROLoss, WDROTransformerHedger
from src.train.losses import EntropicLoss


class AdaptiveWassersteinDROLoss(WassersteinDROLoss):
    """Wasserstein-DRO loss with a volatility-conditioned radius.

    Identical to :class:`WassersteinDROLoss` except that the per-sample
    gradient-norm penalty is weighted by
        w(x) = exp(-beta * relu(vol(x)/vol_ref - 1)),
    which equals 1 in calm states and decays in high-vol states.
    """

    def __init__(self, base_loss: nn.Module, epsilon: float = 0.1,
                 grad_penalty_weight: float = 1.0, vol_ref: float = 0.2,
                 beta: float = 3.0, vol_index: int = 2):
        super().__init__(base_loss, epsilon, grad_penalty_weight)
        self.vol_ref = vol_ref
        self.beta = beta
        self.vol_index = vol_index

    def forward(self, model, inputs, prices, option_payoff,
                transaction_cost: float = 0.001):
        inputs_grad = inputs.detach().clone().requires_grad_(True)
        sdp_ctx = torch.nn.attention.sdpa_kernel(
            [torch.nn.attention.SDPBackend.MATH]
        ) if hasattr(torch.nn.attention, 'sdpa_kernel') else nullcontext()
        with sdp_ctx:
            deltas = model(inputs_grad)
        pnl = self._compute_pnl(deltas, prices, option_payoff, transaction_cost)
        base_loss = self.base_loss(pnl)
        grad_inputs = torch.autograd.grad(
            base_loss, inputs_grad, create_graph=True, retain_graph=True)[0]

        per_elem = grad_inputs.norm(p=2, dim=-1)                 # [B, T]
        vol = inputs_grad[..., self.vol_index].detach()          # [B, T]
        w = torch.exp(-self.beta * F.relu(vol / self.vol_ref - 1.0))
        grad_norm = (w * per_elem).mean()
        dro_penalty = self.epsilon * self.grad_penalty_weight * grad_norm
        total_loss = base_loss + dro_penalty
        metrics = {
            'base_loss': base_loss.detach(),
            'dro_penalty': dro_penalty.detach(),
            'grad_norm': grad_norm.detach(),
            'mean_vol_weight': w.detach().mean(),
            'total_loss': total_loss.detach(),
        }
        return total_loss, metrics


class WDROTransformerHedgerAdaptive(WDROTransformerHedger):
    """W-DRO-T whose DRO loss uses a volatility-conditioned radius."""

    def __init__(self, input_dim: int = 5, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, d_ff: int = 256, dropout: float = 0.1,
                 delta_max: float = 1.5, epsilon: float = 0.1,
                 vol_ref: float = 0.2, beta: float = 3.0, vol_index: int = 2):
        super().__init__(input_dim=input_dim, d_model=d_model, n_heads=n_heads,
                         n_layers=n_layers, d_ff=d_ff, dropout=dropout,
                         delta_max=delta_max, epsilon=epsilon)
        # Replace the constant-radius DRO loss with the adaptive one.
        self.dro_loss = AdaptiveWassersteinDROLoss(
            base_loss=EntropicLoss(lambda_risk=1.0), epsilon=epsilon,
            vol_ref=vol_ref, beta=beta, vol_index=vol_index)


if __name__ == "__main__":
    # shape / equivalence smoke test (CPU, no training)
    torch.manual_seed(0)
    B, T, d = 16, 30, 5
    x = torch.randn(B, T, d); x[..., 2] = x[..., 2].abs() * 0.3  # vol feature >=0
    prices = 100 + torch.cumsum(torch.randn(B, T + 1) * 0.5, dim=1)
    payoff = F.relu(prices[:, -1] - 100)

    adaptive = WDROTransformerHedgerAdaptive(epsilon=0.1, beta=3.0)
    loss_a, m_a = adaptive.compute_dro_loss(x, prices, payoff)
    print("adaptive deltas ok; total=%.4f dro=%.5f vol_w=%.3f"
          % (m_a['total_loss'], m_a['dro_penalty'], m_a['mean_vol_weight']))

    # beta=0 must reproduce the vanilla constant-radius penalty exactly
    torch.manual_seed(1); van = WDROTransformerHedger(epsilon=0.1)
    torch.manual_seed(1); adp0 = WDROTransformerHedgerAdaptive(epsilon=0.1, beta=0.0)
    lv, mv = van.compute_dro_loss(x, prices, payoff)
    la, ma = adp0.compute_dro_loss(x, prices, payoff)
    print("beta=0 vs vanilla dro_penalty: %.6f vs %.6f  match=%s"
          % (ma['dro_penalty'], mv['dro_penalty'],
             torch.allclose(ma['dro_penalty'], mv['dro_penalty'], atol=1e-6)))
    print("OK")
