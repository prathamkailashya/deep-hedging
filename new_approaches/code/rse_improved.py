"""
Improved Regime-Switching Ensemble variants (pass-5 research investigation).
============================================================================

Two evidence-gated extensions of the baseline RSE, kept as subclasses so
the reported baseline (rse.py) is untouched and every comparison is a
clean ablation.

1. Friction-aware training  (RSETrainerFrictionAware)
   -----------------------------------------------------------------
   Diagnosis. The baseline RSETrainer computes its training P&L with a
   *fixed* transaction cost of 0.001, regardless of the deployment
   market. The pass-5 walk-forward showed RSE's only losses are the
   high-friction NIFTY cells (18 bps), and the mechanism is turnover:
   RSE trades ~1.5x the LSTM. A policy that never sees the true friction
   during training has no incentive to economise on turnover.

   Fix. (a) train the risk objective with the *deployment* friction
   (tc), and (b) optionally add an explicit turnover penalty
   lambda * E[ sum_t |delta_t - delta_{t-1}| ] to the gating loss. This
   is the transaction-cost-aware hedging objective of Buehler et al.
   (2019) and Kolm & Ritter (2019); the turnover penalty is the smooth
   analogue of the no-trade band of Zakamouline (2006). tc = 0.001 and
   lambda = 0 recover the baseline exactly.

2. Richer regime features   (RegimeFeatureExtractorRich / ...Rich)
   -----------------------------------------------------------------
   The baseline regime vector is 6-d (3 realised vols, trend, momentum,
   jump ratio). We add three econometrically-motivated, tail-sensitive
   descriptors (Barndorff-Nielsen & Shephard 2002; Ang et al. 2006 on
   downside risk): downside semi-deviation, realised skewness, and
   running drawdown -> a 9-d vector. Whether these help is an empirical
   question settled by ablation, not assumed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from rse import (RegimeFeatureExtractor, RegimeClassifier,
                 RegimeSwitchingEnsemble, RSETrainer)


# --------------------------------------------------------------------------
# 1. Friction-aware trainer
# --------------------------------------------------------------------------
class RSETrainerFrictionAware(RSETrainer):
    """RSE trainer that trains under the deployment friction and (optionally)
    an explicit turnover penalty. tc=0.001, turnover_lambda=0 == baseline."""

    def __init__(self, model, lr: float = 1e-3, device: str = 'cuda',
                 tc: float = 0.001, turnover_lambda: float = 0.0):
        super().__init__(model, lr=lr, device=device)
        self.tc = tc
        self.turnover_lambda = turnover_lambda

    def _compute_pnl(self, deltas, prices, payoff, tc=None):
        # ignore the caller's tc; always use the configured deployment friction
        return super()._compute_pnl(deltas, prices, payoff, tc=self.tc)

    @staticmethod
    def _turnover(deltas: torch.Tensor) -> torch.Tensor:
        # E[ |delta_0| + sum_{t>=1} |delta_t - delta_{t-1}| ]  (initial trade + rebalances)
        rebal = torch.abs(deltas[:, 1:] - deltas[:, :-1]).sum(dim=1)
        return (rebal + torch.abs(deltas[:, 0])).mean()

    def train_gating(self, train_loader, epochs: int = 50, loss_fn=None):
        """Same as the baseline gating loop, plus a turnover penalty."""
        loss_fn = loss_fn or self.entropic_loss_fn
        history = {'loss': []}
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                features = batch['features'].to(self.device)
                prices = batch.get('prices', batch.get('stock_paths')).to(self.device)
                payoff = batch['payoff'].to(self.device)

                self.optimizer.zero_grad()
                deltas = self.model(features, prices)
                pnl = self._compute_pnl(deltas, prices, payoff)
                loss = loss_fn(pnl)
                if self.turnover_lambda > 0:
                    loss = loss + self.turnover_lambda * self._turnover(deltas)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                epoch_loss += loss.item()
            history['loss'].append(epoch_loss / len(train_loader))
        return history


# --------------------------------------------------------------------------
# 2. Richer regime features
# --------------------------------------------------------------------------
class RegimeFeatureExtractorRich(RegimeFeatureExtractor):
    """Baseline 6 regime features + {downside semi-dev, realised skew,
    running drawdown} -> 9 features. All strictly causal (window ends at t)."""

    N_FEATURES = 9

    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        base = super().forward(prices)                      # [B, T, 6]
        B, T = prices.shape
        log_p = torch.log(prices + 1e-8)
        returns = log_p[:, 1:] - log_p[:, :-1]
        returns = torch.nn.functional.pad(returns, (1, 0), value=0)  # [B, T]

        dsemi = torch.zeros(B, T, device=prices.device)
        rskew = torch.zeros(B, T, device=prices.device)
        w = 20
        for t in range(1, T):
            seg = returns[:, max(0, t - w + 1):t + 1]
            neg = torch.clamp(seg, max=0.0)
            dsemi[:, t] = torch.sqrt((neg ** 2).mean(dim=1) + 1e-12) * np.sqrt(252)
            if seg.shape[1] >= 3:
                mu = seg.mean(dim=1, keepdim=True)
                sd = seg.std(dim=1) + 1e-8
                rskew[:, t] = (((seg - mu) ** 3).mean(dim=1)) / (sd ** 3)

        run_max, _ = torch.cummax(prices, dim=1)            # running peak, causal
        drawdown = (prices - run_max) / (run_max + 1e-8)    # <= 0

        extra = torch.stack([dsemi, rskew, drawdown], dim=-1)  # [B, T, 3]
        return torch.cat([base, extra], dim=-1)                 # [B, T, 9]


class RegimeSwitchingEnsembleRich(RegimeSwitchingEnsemble):
    """RSE with the 9-d rich regime representation (classifier resized)."""

    def __init__(self, input_dim: int = 5, n_regimes: int = 4,
                 delta_max: float = 1.5, pretrained_models=None):
        super().__init__(input_dim=input_dim, n_regimes=n_regimes,
                         delta_max=delta_max, pretrained_models=pretrained_models)
        self.feature_extractor = RegimeFeatureExtractorRich()
        self.regime_classifier = RegimeClassifier(
            input_dim=RegimeFeatureExtractorRich.N_FEATURES, n_regimes=n_regimes)


if __name__ == "__main__":
    import torch.nn.functional as F
    B, T = 8, 30
    feats = torch.randn(B, T, 5)
    prices = 100 + torch.cumsum(torch.randn(B, T + 1) * 0.5, dim=1)

    # rich model forward shape
    rich = RegimeSwitchingEnsembleRich(input_dim=5, n_regimes=4)
    d = rich(feats, prices)
    rf = rich.feature_extractor(prices[:, :-1])
    print("rich regime feats:", tuple(rf.shape), " deltas:", tuple(d.shape),
          " finite:", bool(torch.isfinite(d).all()))

    # friction-aware trainer: tc plumbs into pnl; baseline-equivalence at defaults
    base = RegimeSwitchingEnsemble(input_dim=5, n_regimes=4)
    tr = RSETrainerFrictionAware(base, device='cpu', tc=0.0018, turnover_lambda=1e-3)
    deltas = torch.randn(B, T)
    print("turnover stat:", round(float(tr._turnover(deltas)), 3))
    pnl = tr._compute_pnl(deltas, prices, F.relu(prices[:, -1] - 100))
    print("pnl uses tc=%.4f -> shape %s finite %s"
          % (tr.tc, tuple(pnl.shape), bool(torch.isfinite(pnl).all())))
    print("OK")
