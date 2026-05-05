"""Hedging-task payoff functions: vanilla, barrier, digital, Asian, basket.

Each task accepts ``S`` of shape (n_paths, T+1, d) and returns a payoff vector
of shape (n_paths,).  All payoffs are in the *holder's* perspective; the hedger
shorts the option, so terminal P&L subtracts the payoff.

Tasks come with a default hedge-instrument set: for vanilla/barrier/digital/Asian
tasks the underlying asset is the only hedge instrument; for basket tasks all
``d`` underliers are tradable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

PayoffFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class HedgingTask:
    name: str
    payoff: PayoffFn
    hedge_assets: List[int]   # indices of tradable assets in S
    description: str
    moneyness: float = 1.0    # K = moneyness * S0


# ---------------------------------------------------------------------------
def european_call(K: float = 100.0, asset_idx: int = 0) -> HedgingTask:
    def fn(S: np.ndarray) -> np.ndarray:
        return np.maximum(S[:, -1, asset_idx] - K, 0.0)
    return HedgingTask("european_call", fn, [asset_idx],
                       f"vanilla European call, K={K}")


def european_put(K: float = 100.0, asset_idx: int = 0) -> HedgingTask:
    def fn(S: np.ndarray) -> np.ndarray:
        return np.maximum(K - S[:, -1, asset_idx], 0.0)
    return HedgingTask("european_put", fn, [asset_idx],
                       f"vanilla European put, K={K}")


# ---------------------------------------------------------------------------
def up_and_out_call(K: float = 100.0,
                    barrier: float = 130.0,
                    asset_idx: int = 0) -> HedgingTask:
    """Knock-out at barrier B; payoff = (S_T - K)+ if max_t S_t < B else 0."""
    def fn(S: np.ndarray) -> np.ndarray:
        path = S[..., asset_idx]
        max_path = np.max(path, axis=1)
        terminal = np.maximum(path[:, -1] - K, 0.0)
        return np.where(max_path < barrier, terminal, 0.0)
    return HedgingTask("up_and_out_call", fn, [asset_idx],
                       f"up-and-out call, K={K}, B={barrier}")


def down_and_in_put(K: float = 100.0,
                    barrier: float = 80.0,
                    asset_idx: int = 0) -> HedgingTask:
    def fn(S: np.ndarray) -> np.ndarray:
        path = S[..., asset_idx]
        min_path = np.min(path, axis=1)
        terminal = np.maximum(K - path[:, -1], 0.0)
        return np.where(min_path < barrier, terminal, 0.0)
    return HedgingTask("down_and_in_put", fn, [asset_idx],
                       f"down-and-in put, K={K}, B={barrier}")


# ---------------------------------------------------------------------------
def cash_or_nothing_call(K: float = 100.0,
                         payout: float = 10.0,
                         asset_idx: int = 0) -> HedgingTask:
    def fn(S: np.ndarray) -> np.ndarray:
        return payout * (S[:, -1, asset_idx] >= K).astype(np.float64)
    return HedgingTask("digital_call", fn, [asset_idx],
                       f"digital call, K={K}, payout={payout}")


# ---------------------------------------------------------------------------
def asian_call(K: float = 100.0, asset_idx: int = 0) -> HedgingTask:
    """Arithmetic-average Asian call: payoff = (mean(S) - K)+."""
    def fn(S: np.ndarray) -> np.ndarray:
        return np.maximum(S[..., asset_idx].mean(axis=1) - K, 0.0)
    return HedgingTask("asian_call", fn, [asset_idx],
                       f"Asian (arithmetic-avg) call, K={K}")


# ---------------------------------------------------------------------------
def basket_call(weights: Optional[np.ndarray] = None,
                K: float = 100.0) -> HedgingTask:
    """Equally-weighted basket European call: payoff = (<w, S_T> - K)+."""
    def fn(S: np.ndarray) -> np.ndarray:
        d = S.shape[-1]
        w = np.full(d, 1.0 / d) if weights is None else weights
        idx_basket = (S[:, -1, :] * w).sum(axis=1)
        return np.maximum(idx_basket - K, 0.0)
    d_hint = 0  # determined at runtime
    return HedgingTask("basket_call", fn, list(range(99)),
                       f"basket European call, K={K}")


# ---------------------------------------------------------------------------
TASK_REGISTRY = {
    "european_call":     european_call,
    "european_put":      european_put,
    "up_and_out_call":   up_and_out_call,
    "down_and_in_put":   down_and_in_put,
    "digital_call":      cash_or_nothing_call,
    "asian_call":        asian_call,
    "basket_call":       basket_call,
}


def list_tasks() -> List[str]:
    return list(TASK_REGISTRY.keys())
