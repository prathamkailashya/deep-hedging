"""Feature engineering for hedging episodes.

Inputs at each (path, time, asset):
    1. S_k / S_0
    2. log(S_k / K)
    3. sqrt(v_k)        (instantaneous vol proxy)
    4. tau_k            (normalised time-to-maturity)
    5. Delta_BS_k       (Black-Scholes delta at current vol/moneyness)

Optional regime features (for RSE) computed at the *path level* from spot only.
Multi-asset case stacks per-asset features along the last axis.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple
from scipy.stats import norm


def _bs_delta(S: np.ndarray, K: float, tau: np.ndarray, sigma: np.ndarray, r: float = 0.05) -> np.ndarray:
    """Vectorised Black-Scholes call delta.

    All inputs broadcastable to the same shape; tau, sigma must be > 0 element-wise.
    """
    sigma = np.maximum(sigma, 1e-3)
    tau = np.maximum(tau, 1e-6)
    d1 = (np.log(np.maximum(S / K, 1e-9)) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def build_features(S: np.ndarray,
                   v: np.ndarray,
                   T: float,
                   K: float = 100.0,
                   r: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-step feature tensor.

    Parameters
    ----------
    S : (n, T+1, d)
    v : (n, T+1, d) -- variance proxy (>= 0)
    Returns
    -------
    X : (n, T, F)         where F = 5 * d  (drops the terminal step)
    log_returns : (n, T, d)  for diagnostics / signature features
    """
    n, n_steps_p1, d = S.shape
    n_steps = n_steps_p1 - 1
    times = np.linspace(0.0, T, n_steps_p1)
    tau = (T - times) / T  # normalised

    feats = []
    log_ret = np.zeros((n, n_steps, d))
    for j in range(d):
        S_j = S[..., j]
        v_j = v[..., j]
        m = S_j / S_j[:, [0]]
        log_money = np.log(np.maximum(S_j / K, 1e-9))
        sqrt_v = np.sqrt(np.maximum(v_j, 1e-8))
        # Black-Scholes delta with running variance estimate
        delta_bs = _bs_delta(S_j, K, np.broadcast_to(tau, S_j.shape), sqrt_v, r=r)
        # Drop terminal step so each feature has length T (matches deltas)
        feats.extend([m[:, :-1], log_money[:, :-1], sqrt_v[:, :-1],
                      np.broadcast_to(tau[:-1], (n, n_steps)), delta_bs[:, :-1]])
        log_ret[:, :, j] = np.diff(np.log(np.maximum(S_j, 1e-9)), axis=1)
    X = np.stack(feats, axis=-1)  # (n, T, 5*d)
    return X.astype(np.float32), log_ret.astype(np.float32)


# -----------------------------------------------------------------------------
# Regime features (for RSE) -- single-asset only at this stage
# -----------------------------------------------------------------------------
def regime_features(S: np.ndarray, asset_idx: int = 0) -> np.ndarray:
    """Compute six regime features along time.

    Returns (n, T, 6).
    """
    px = S[..., asset_idx]
    log_ret = np.diff(np.log(np.maximum(px, 1e-9)), axis=1)  # (n, T)
    n, Tn = log_ret.shape

    rv5 = _rolling_std(log_ret, 5) * np.sqrt(252.0)
    rv10 = _rolling_std(log_ret, 10) * np.sqrt(252.0)
    rv20 = _rolling_std(log_ret, 20) * np.sqrt(252.0)
    sma5 = _rolling_mean(px, 5)[:, 1:]
    sma20 = _rolling_mean(px, 20)[:, 1:]
    trend = (sma5 - sma20) / np.maximum(sma20, 1e-9)
    mom = np.zeros_like(log_ret)
    mom[:, 5:] = (px[:, 5:-1] - px[:, :-6]) / np.maximum(px[:, :-6], 1e-9)
    bv = _bipower_var(log_ret, 20) * np.sqrt(252.0)
    jump = rv20 / np.maximum(bv, 1e-6)

    feats = np.stack([rv5, rv10, rv20, trend, mom, jump], axis=-1)
    return feats.astype(np.float32)


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    n, T = x.shape
    out = np.zeros((n, T))
    for t in range(T):
        lo = max(0, t - w + 1)
        out[:, t] = x[:, lo:t + 1].std(axis=1, ddof=0)
    return out


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    n, T = x.shape
    out = np.zeros((n, T))
    for t in range(T):
        lo = max(0, t - w + 1)
        out[:, t] = x[:, lo:t + 1].mean(axis=1)
    return out


def _bipower_var(log_ret: np.ndarray, w: int) -> np.ndarray:
    n, T = log_ret.shape
    abs_r = np.abs(log_ret)
    out = np.zeros((n, T))
    factor = np.pi / 2
    for t in range(T):
        lo = max(0, t - w + 1)
        if t > 0:
            r = abs_r[:, lo:t]
            r1 = abs_r[:, lo + 1:t + 1]
            length = min(r.shape[1], r1.shape[1])
            if length > 0:
                out[:, t] = factor * (r[:, :length] * r1[:, :length]).mean(axis=1)
        if out[:, t].max() == 0:
            out[:, t] = abs_r[:, max(0, t - 1):t + 1].mean(axis=1) ** 2
    return out
