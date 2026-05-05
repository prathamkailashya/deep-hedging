"""Hedger architectures: LSTM, Transformer, RSE, plus shared utilities.

All hedgers accept input ``X`` of shape (B, T, F) and produce per-step deltas of
shape (B, T, d) where ``d`` is the number of tradable assets.  Deltas are
bounded via tanh to ``+/- delta_max`` (default 1.5).

W-DRO-T = TransformerHedger trained with the DRO gradient penalty (handled in
trainer.py).  3SCH = LSTMHedger trained with the 3-stage curriculum (also in
trainer.py).  RSE wraps frozen base hedgers under a regime-dependent gating.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# LSTM hedger
# -----------------------------------------------------------------------------
class LSTMHedger(nn.Module):
    def __init__(self, in_dim: int, n_assets: int = 1,
                 hidden: int = 50, n_layers: int = 2,
                 delta_max: float = 1.5):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=n_layers, batch_first=True, dropout=0.0)
        self.head = nn.Linear(hidden, n_assets)
        self.delta_max = delta_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.delta_max * torch.tanh(self.head(h))


# -----------------------------------------------------------------------------
# Transformer hedger (causal)
# -----------------------------------------------------------------------------
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerHedger(nn.Module):
    def __init__(self, in_dim: int, n_assets: int = 1,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 128, dropout: float = 0.1,
                 delta_max: float = 1.5):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pe = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                           dim_feedforward=d_ff, dropout=dropout,
                                           activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, n_assets))
        self.delta_max = delta_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = self.pe(z)
        T = z.size(1)
        causal = torch.triu(torch.ones(T, T, device=z.device, dtype=torch.bool), diagonal=1)
        z = self.encoder(z, mask=causal)
        return self.delta_max * torch.tanh(self.head(z))


# -----------------------------------------------------------------------------
# Signature-feature hedger (for use as a base inside RSE).  We approximate the
# truncated path signature with running statistics: mean / std / max / min /
# realised quadratic variation of log-returns over a rolling window.
# -----------------------------------------------------------------------------
class SignatureHedger(nn.Module):
    def __init__(self, in_dim: int, n_assets: int = 1,
                 hidden: int = 64, delta_max: float = 1.5,
                 window: int = 5):
        super().__init__()
        # The "signature" extension produces 6 extra channels per asset
        self.window = window
        self.in_dim = in_dim
        sig_dim = 6 * n_assets
        self.head = nn.Sequential(
            nn.Linear(in_dim + sig_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_assets),
        )
        self.delta_max = delta_max
        self.n_assets = n_assets

    def _signature_stats(self, x: torch.Tensor) -> torch.Tensor:
        # Use the first column-block per asset (S/S_0) as the path proxy.
        B, T, F = x.shape
        d = self.n_assets
        feats_per_asset = F // d
        sig = []
        for j in range(d):
            px = x[:, :, j * feats_per_asset]   # (B, T)
            log_ret = px[:, 1:].log() - px[:, :-1].log()
            log_ret = torch.nn.functional.pad(log_ret, (1, 0), value=0.0)
            running = []
            for t in range(T):
                lo = max(0, t - self.window + 1)
                w = log_ret[:, lo:t + 1]
                w_safe = w if w.numel() > 0 else log_ret[:, :1]
                running.append(torch.stack([
                    w_safe.mean(dim=1),
                    w_safe.std(dim=1, unbiased=False),
                    w_safe.max(dim=1).values,
                    w_safe.min(dim=1).values,
                    (w_safe ** 2).sum(dim=1),
                    (w_safe.abs()).sum(dim=1),
                ], dim=-1))
            sig.append(torch.stack(running, dim=1))
        return torch.cat(sig, dim=-1)  # (B, T, 6*d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sig = self._signature_stats(x)
        z = torch.cat([x, sig], dim=-1)
        return self.delta_max * torch.tanh(self.head(z))


# -----------------------------------------------------------------------------
# Regime-Switching Ensemble
# -----------------------------------------------------------------------------
class RegimeClassifier(nn.Module):
    def __init__(self, regime_dim: int = 6, n_regimes: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(regime_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_regimes),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(r), dim=-1)


class RSEHedger(nn.Module):
    """Mixture-of-frozen-experts gated by regime probabilities."""
    def __init__(self, base_models: List[nn.Module],
                 regime_dim: int = 6, n_regimes: int = 4,
                 delta_max: float = 1.5, n_assets: int = 1):
        super().__init__()
        self.bases = nn.ModuleList(base_models)
        self.classifier = RegimeClassifier(regime_dim, n_regimes)
        # Affinity matrix [regimes x experts]
        self.A = nn.Parameter(torch.zeros(n_regimes, len(base_models)))
        self.n_assets = n_assets
        self.delta_max = delta_max
        # Freeze base models
        for m in self.bases:
            for p in m.parameters():
                p.requires_grad_(False)

    def expert_outputs(self, x: torch.Tensor) -> torch.Tensor:
        outs = [m(x) for m in self.bases]   # each (B, T, d)
        return torch.stack(outs, dim=-1)    # (B, T, d, M)

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # r : (B, T, regime_dim)
        regime_probs = self.classifier(r)   # (B, T, K)
        affinity = torch.softmax(self.A, dim=1)   # (K, M)
        weights = regime_probs @ affinity    # (B, T, M)
        experts = self.expert_outputs(x)     # (B, T, d, M)
        delta = (experts * weights.unsqueeze(2)).sum(dim=-1)  # (B, T, d)
        return self.delta_max * torch.tanh(delta / self.delta_max)
