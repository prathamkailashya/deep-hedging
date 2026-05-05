"""Evaluation metrics for hedging P&L distributions."""
from __future__ import annotations

import numpy as np
from typing import Dict


def cvar(pnl: np.ndarray, alpha: float = 0.95) -> float:
    losses = -np.asarray(pnl)
    nu = np.quantile(losses, alpha)
    return float(nu + np.mean(np.maximum(losses - nu, 0)) / (1 - alpha))


def var(pnl: np.ndarray, alpha: float = 0.95) -> float:
    return float(np.quantile(-np.asarray(pnl), alpha))


def entropic(pnl: np.ndarray, lam: float = 1.0) -> float:
    pnl = np.asarray(pnl)
    return float((1.0 / lam) * np.log(np.mean(np.exp(-lam * pnl)) + 1e-12))


def turnover(deltas: np.ndarray, prices: np.ndarray) -> float:
    """Average per-path turnover: sum_t |Delta_t - Delta_{t-1}| * S_t / S_0."""
    deltas = np.asarray(deltas)
    if deltas.ndim == 2:
        deltas = deltas[:, :, None]
    if prices.ndim == 2:
        prices = prices[:, :, None]
    pad = np.zeros_like(deltas[:, :1, :])
    diff = np.concatenate([deltas[:, :1, :], deltas[:, 1:, :] - deltas[:, :-1, :]], axis=1)
    s0 = prices[:, [0], :]
    norm_p = prices[:, :-1, :] / np.maximum(s0, 1e-9)
    return float((np.abs(diff) * norm_p).sum(axis=(1, 2)).mean())


def evaluate_pnl(pnl: np.ndarray, deltas: np.ndarray, prices: np.ndarray,
                 hedge_assets) -> Dict[str, float]:
    p_hedge = prices[..., hedge_assets]
    return {
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl":  float(np.std(pnl, ddof=1)),
        "cvar_95":  cvar(pnl, 0.95),
        "cvar_99":  cvar(pnl, 0.99),
        "var_95":   var(pnl, 0.95),
        "var_99":   var(pnl, 0.99),
        "entropic": entropic(pnl, 1.0),
        "turnover": turnover(deltas, p_hedge),
        "min_pnl":  float(np.min(pnl)),
        "max_pnl":  float(np.max(pnl)),
        "skew":     float(_skew(pnl)),
        "kurt":     float(_kurt(pnl)),
    }


def _skew(x: np.ndarray) -> float:
    x = np.asarray(x)
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurt(x: np.ndarray) -> float:
    x = np.asarray(x)
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def bootstrap_ci(values: np.ndarray, n_resample: int = 10000,
                 alpha: float = 0.05, seed: int = 0):
    """Bootstrap CI for a vector of seed-level metric values."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    boots = rng.choice(values, size=(n_resample, len(values)), replace=True)
    means = boots.mean(axis=1)
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(values.mean()), float(lo), float(hi)


def paired_test(values_a: np.ndarray, values_b: np.ndarray) -> Dict[str, float]:
    """Paired t-test + Cohen's d on per-seed metrics."""
    a, b = np.asarray(values_a, np.float64), np.asarray(values_b, np.float64)
    diff = a - b
    n = len(diff)
    sd = diff.std(ddof=1)
    mean = diff.mean()
    if sd == 0 or n < 2:
        return {"mean_diff": float(mean), "p_value": 1.0, "cohen_d": 0.0}
    t = mean / (sd / np.sqrt(n))
    # two-sided p-value via normal approx (fine for reporting given small n)
    from scipy.stats import t as tdist
    p = float(2 * (1 - tdist.cdf(abs(t), df=n - 1)))
    return {"mean_diff": float(mean), "p_value": p, "cohen_d": float(mean / sd)}


def holm_bonferroni(p_values: dict, alpha: float = 0.05) -> dict:
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    K = len(items)
    rejected = {}
    for j, (key, p) in enumerate(items):
        thresh = alpha / (K - j)
        rejected[key] = (p <= thresh)
    return rejected
