"""Trainers for LSTM, Transformer, W-DRO-T, 3SCH, RSE.

All trainers share a common interface

    trainer.fit(features, prices, payoff)

and return a dict of training-time metrics. Hedging deltas at evaluation time
are computed by ``trainer.act(features)``.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from jaws_research.models.losses import (cvar_loss, entropic_loss, mixed_loss,
                                         trading_penalty)


# ---------------------------------------------------------------------------
def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_loader(features: np.ndarray, prices: np.ndarray, payoff: np.ndarray,
                 regime: Optional[np.ndarray] = None,
                 batch_size: int = 256, shuffle: bool = True) -> DataLoader:
    tensors = [torch.from_numpy(features).float(),
               torch.from_numpy(prices).float(),
               torch.from_numpy(payoff).float()]
    if regime is not None:
        tensors.append(torch.from_numpy(regime).float())
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def _pnl(deltas: torch.Tensor, prices: torch.Tensor, payoff: torch.Tensor,
         hedge_assets: List[int], c_tc: float) -> torch.Tensor:
    """deltas (B, T, d_h) on hedge assets only; prices (B, T+1, d) full asset stack."""
    p_hedge = prices[..., hedge_assets]
    dS = p_hedge[:, 1:, :] - p_hedge[:, :-1, :]
    hedge_gain = (deltas * dS).sum(dim=(1, 2))
    pad = torch.zeros_like(deltas[:, :1, :])
    diff = torch.cat([deltas[:, :1, :], deltas[:, 1:, :] - deltas[:, :-1, :]], dim=1)
    cost = (diff.abs() * p_hedge[:, :-1, :] * c_tc).sum(dim=(1, 2))
    return -payoff + hedge_gain - cost


# ---------------------------------------------------------------------------
class BaseTrainer:
    def __init__(self, model: nn.Module, hedge_assets: List[int], c_tc: float = 0.001,
                 device: Optional[torch.device] = None,
                 cvar_alpha: float = 0.95, lam_ent: float = 1.0):
        self.model = model.to(device or _device())
        self.device = device or _device()
        self.hedge_assets = hedge_assets
        self.c_tc = c_tc
        self.cvar_alpha = cvar_alpha
        self.lam_ent = lam_ent

    # ------------------------------------------------------------------
    def _step_pnl(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        x, S, Z = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        deltas = self._predict(x, batch)
        pnl = _pnl(deltas, S, Z, self.hedge_assets, self.c_tc)
        return deltas, pnl

    def _predict(self, x, batch):  # subclasses can override
        return self.model(x)

    # ------------------------------------------------------------------
    def fit(self, features: np.ndarray, prices: np.ndarray, payoff: np.ndarray,
            *, epochs_cvar: int = 25, epochs_ent: int = 15, batch_size: int = 256,
            lr_stage1: float = 1e-3, lr_stage2: float = 1e-4,
            grad_clip: float = 5.0,
            log_every: int = 5,
            patience: int = 8) -> Dict:
        loader = _make_loader(features, prices, payoff, batch_size=batch_size)
        history = {"stage1_loss": [], "stage2_loss": [], "epoch_time": []}

        opt1 = torch.optim.Adam(self.model.parameters(), lr=lr_stage1, weight_decay=1e-4)
        best, bad = math.inf, 0
        for ep in range(epochs_cvar):
            t0 = time.time()
            self.model.train()
            losses = []
            for batch in loader:
                opt1.zero_grad()
                _, pnl = self._step_pnl(batch)
                loss = cvar_loss(pnl, self.cvar_alpha)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                opt1.step()
                losses.append(loss.item())
            mean_loss = float(np.mean(losses))
            history["stage1_loss"].append(mean_loss)
            history["epoch_time"].append(time.time() - t0)
            if mean_loss + 1e-6 < best:
                best, bad = mean_loss, 0
            else:
                bad += 1
                if bad >= patience:
                    break

        opt2 = torch.optim.Adam(self.model.parameters(), lr=lr_stage2, weight_decay=1e-4)
        best, bad = math.inf, 0
        for ep in range(epochs_ent):
            t0 = time.time()
            self.model.train()
            losses = []
            for batch in loader:
                opt2.zero_grad()
                deltas, pnl = self._step_pnl(batch)
                loss = entropic_loss(pnl, self.lam_ent) + 1e-3 * trading_penalty(
                    deltas, batch[1].to(self.device)[..., self.hedge_assets], self.c_tc)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                opt2.step()
                losses.append(loss.item())
            mean_loss = float(np.mean(losses))
            history["stage2_loss"].append(mean_loss)
            history["epoch_time"].append(time.time() - t0)
            if mean_loss + 1e-6 < best:
                best, bad = mean_loss, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        return history

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.from_numpy(features).float().to(self.device)
        return self.model(x).cpu().numpy()


# ---------------------------------------------------------------------------
class WDROTrainer(BaseTrainer):
    """W-DRO-T: Stage-1 CVaR; Stage-2 entropic + Wasserstein-DRO gradient penalty."""
    def __init__(self, *args, eps_max: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps_max = eps_max

    def fit(self, features: np.ndarray, prices: np.ndarray, payoff: np.ndarray,
            *, epochs_cvar: int = 25, epochs_ent: int = 15, batch_size: int = 256,
            lr_stage1: float = 1e-3, lr_stage2: float = 1e-4,
            grad_clip: float = 5.0, patience: int = 8) -> Dict:
        loader = _make_loader(features, prices, payoff, batch_size=batch_size)
        history = {"stage1_loss": [], "stage2_loss": [], "grad_norm": [], "epoch_time": []}

        opt = torch.optim.Adam(self.model.parameters(), lr=lr_stage1, weight_decay=1e-4)
        best, bad = math.inf, 0
        for ep in range(epochs_cvar):
            t0 = time.time()
            self.model.train()
            ls = []
            for batch in loader:
                opt.zero_grad()
                _, pnl = self._step_pnl(batch)
                loss = cvar_loss(pnl, self.cvar_alpha)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                opt.step()
                ls.append(loss.item())
            history["stage1_loss"].append(float(np.mean(ls)))
            history["epoch_time"].append(time.time() - t0)
            if history["stage1_loss"][-1] + 1e-6 < best:
                best, bad = history["stage1_loss"][-1], 0
            else:
                bad += 1
                if bad >= patience:
                    break

        opt = torch.optim.Adam(self.model.parameters(), lr=lr_stage2, weight_decay=1e-4)
        best, bad = math.inf, 0
        # Use MATH backend for SDPA so that 2nd-order gradients work
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            def _sdp():
                return sdpa_kernel([SDPBackend.MATH])
        except Exception:  # pragma: no cover
            from contextlib import nullcontext
            def _sdp():
                return nullcontext()

        for ep in range(epochs_ent):
            t0 = time.time()
            self.model.train()
            eps = self.eps_max * (ep + 1) / max(1, epochs_ent)
            ls, gns = [], []
            for batch in loader:
                opt.zero_grad()
                x = batch[0].detach().clone().to(self.device).requires_grad_(True)
                S, Z = batch[1].to(self.device), batch[2].to(self.device)
                with _sdp():
                    deltas = self.model(x)
                pnl = _pnl(deltas, S, Z, self.hedge_assets, self.c_tc)
                base = entropic_loss(pnl, self.lam_ent)
                grad_x = torch.autograd.grad(base, x, create_graph=True, retain_graph=True)[0]
                grad_norm = grad_x.norm(p=2, dim=-1).mean()
                loss = base + eps * grad_norm
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                opt.step()
                ls.append(loss.item())
                gns.append(grad_norm.detach().item())
            history["stage2_loss"].append(float(np.mean(ls)))
            history["grad_norm"].append(float(np.mean(gns)))
            history["epoch_time"].append(time.time() - t0)
            if history["stage2_loss"][-1] + 1e-6 < best:
                best, bad = history["stage2_loss"][-1], 0
            else:
                bad += 1
                if bad >= patience:
                    break
        return history


# ---------------------------------------------------------------------------
class ThreeStageTrainer(BaseTrainer):
    """3SCH: CVaR -> mixed homotopy -> entropic + trading penalty."""
    def __init__(self, *args, alpha_start: float = 0.8, alpha_end: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end

    def fit(self, features: np.ndarray, prices: np.ndarray, payoff: np.ndarray,
            *, epochs_cvar: int = 20, epochs_mid: int = 10, epochs_ent: int = 10,
            batch_size: int = 256, lr_stage1: float = 1e-3,
            lr_stage2: float = 5e-4, lr_stage3: float = 1e-4,
            grad_clip: float = 5.0, patience: int = 8) -> Dict:
        loader = _make_loader(features, prices, payoff, batch_size=batch_size)
        history = {"stage1_loss": [], "stage2_loss": [], "stage3_loss": [],
                   "alpha_schedule": [], "epoch_time": []}

        # Stage 1
        opt = torch.optim.Adam(self.model.parameters(), lr=lr_stage1, weight_decay=1e-4)
        best, bad = math.inf, 0
        for ep in range(epochs_cvar):
            t0 = time.time(); self.model.train(); ls = []
            for batch in loader:
                opt.zero_grad()
                _, pnl = self._step_pnl(batch)
                loss = cvar_loss(pnl, self.cvar_alpha); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                opt.step(); ls.append(loss.item())
            history["stage1_loss"].append(float(np.mean(ls)))
            history["epoch_time"].append(time.time() - t0)
            if history["stage1_loss"][-1] + 1e-6 < best:
                best, bad = history["stage1_loss"][-1], 0
            else:
                bad += 1
                if bad >= patience: break

        # Stage 2 mixed-loss homotopy
        opt = torch.optim.Adam(self.model.parameters(), lr=lr_stage2, weight_decay=1e-4)
        for ep in range(epochs_mid):
            t0 = time.time(); self.model.train(); ls = []
            alpha = self.alpha_start - (self.alpha_start - self.alpha_end) * (ep / max(1, epochs_mid - 1))
            history["alpha_schedule"].append(alpha)
            for batch in loader:
                opt.zero_grad()
                _, pnl = self._step_pnl(batch)
                loss = mixed_loss(pnl, self.cvar_alpha, self.lam_ent, weight_cvar=alpha)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                opt.step(); ls.append(loss.item())
            history["stage2_loss"].append(float(np.mean(ls)))
            history["epoch_time"].append(time.time() - t0)

        # Stage 3 entropic + trading penalty
        opt = torch.optim.Adam(self.model.parameters(), lr=lr_stage3, weight_decay=1e-4)
        best, bad = math.inf, 0
        for ep in range(epochs_ent):
            t0 = time.time(); self.model.train(); ls = []
            for batch in loader:
                opt.zero_grad()
                deltas, pnl = self._step_pnl(batch)
                loss = entropic_loss(pnl, self.lam_ent) + 1e-3 * trading_penalty(
                    deltas, batch[1].to(self.device)[..., self.hedge_assets], self.c_tc)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                opt.step(); ls.append(loss.item())
            history["stage3_loss"].append(float(np.mean(ls)))
            history["epoch_time"].append(time.time() - t0)
            if history["stage3_loss"][-1] + 1e-6 < best:
                best, bad = history["stage3_loss"][-1], 0
            else:
                bad += 1
                if bad >= patience: break
        return history


# ---------------------------------------------------------------------------
class RSETrainer(BaseTrainer):
    """RSE: pretrains base experts via BaseTrainer, then trains gating only."""
    def __init__(self, ensemble, hedge_assets, c_tc=0.001, device=None,
                 cvar_alpha=0.95, lam_ent=1.0):
        self.ensemble = ensemble
        super().__init__(ensemble, hedge_assets, c_tc, device, cvar_alpha, lam_ent)

    def _predict(self, x, batch):
        # Last tensor in batch is regime features
        r = batch[-1].to(self.device)
        return self.ensemble(x, r)

    def fit(self, features, prices, payoff, *, regime,
            epochs_gate: int = 12, epochs_finetune: int = 8,
            batch_size: int = 256, lr_gate: float = 1e-3,
            lr_finetune: float = 1e-4, grad_clip: float = 5.0,
            patience: int = 6) -> Dict:
        loader = _make_loader(features, prices, payoff, regime=regime, batch_size=batch_size)
        history = {"gate_loss": [], "finetune_loss": [], "epoch_time": []}

        gate_params = list(self.ensemble.classifier.parameters()) + [self.ensemble.A]
        opt = torch.optim.Adam(gate_params, lr=lr_gate, weight_decay=1e-4)
        best, bad = math.inf, 0
        for ep in range(epochs_gate):
            t0 = time.time(); self.ensemble.train(); ls = []
            for batch in loader:
                opt.zero_grad()
                _, pnl = self._step_pnl(batch)
                loss = cvar_loss(pnl, self.cvar_alpha); loss.backward()
                nn.utils.clip_grad_norm_(gate_params, grad_clip)
                opt.step(); ls.append(loss.item())
            history["gate_loss"].append(float(np.mean(ls)))
            history["epoch_time"].append(time.time() - t0)
            if history["gate_loss"][-1] + 1e-6 < best:
                best, bad = history["gate_loss"][-1], 0
            else:
                bad += 1
                if bad >= patience: break

        opt = torch.optim.Adam(gate_params, lr=lr_finetune, weight_decay=1e-4)
        for ep in range(epochs_finetune):
            t0 = time.time(); self.ensemble.train(); ls = []
            for batch in loader:
                opt.zero_grad()
                deltas, pnl = self._step_pnl(batch)
                loss = entropic_loss(pnl, self.lam_ent) + 1e-3 * trading_penalty(
                    deltas, batch[1].to(self.device)[..., self.hedge_assets], self.c_tc)
                loss.backward()
                nn.utils.clip_grad_norm_(gate_params, grad_clip)
                opt.step(); ls.append(loss.item())
            history["finetune_loss"].append(float(np.mean(ls)))
            history["epoch_time"].append(time.time() - t0)
        return history

    @torch.no_grad()
    def act(self, features: np.ndarray, regime: np.ndarray) -> np.ndarray:
        self.ensemble.eval()
        x = torch.from_numpy(features).float().to(self.device)
        r = torch.from_numpy(regime).float().to(self.device)
        return self.ensemble(x, r).cpu().numpy()
