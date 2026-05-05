"""Synthetic market simulators: Heston (single-asset) and Bates (multi-asset jump-diffusion).

All simulators return:
    paths : ndarray (n_paths, n_steps + 1, n_assets)
    var   : ndarray (n_paths, n_steps + 1, n_assets)  -- instantaneous variance proxy

Discretisation: Euler--Maruyama with full-truncation reflection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np


# -----------------------------------------------------------------------------
# Heston
# -----------------------------------------------------------------------------
@dataclass
class HestonParams:
    S0: float = 100.0
    v0: float = 0.04
    r: float = 0.05
    kappa: float = 1.0
    theta: float = 0.04
    xi: float = 0.2
    rho: float = -0.7


def simulate_heston(p: HestonParams,
                    T: float,
                    n_steps: int,
                    n_paths: int,
                    seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Single-asset Heston paths."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sdt = np.sqrt(dt)

    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rng.standard_normal((n_paths, n_steps))
    dW_S = Z1 * sdt
    dW_v = (p.rho * Z1 + np.sqrt(max(1.0 - p.rho ** 2, 1e-12)) * Z2) * sdt

    S = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    v = np.empty_like(S)
    S[:, 0] = p.S0
    v[:, 0] = p.v0
    for t in range(n_steps):
        v_t = np.maximum(v[:, t], 0.0)
        sv = np.sqrt(v_t)
        S[:, t + 1] = S[:, t] * np.exp((p.r - 0.5 * v_t) * dt + sv * dW_S[:, t])
        v_next = v[:, t] + p.kappa * (p.theta - v_t) * dt + p.xi * sv * dW_v[:, t]
        v[:, t + 1] = np.maximum(v_next, 0.0)
    # add asset axis so downstream code is uniform
    return S[..., None], v[..., None]


# -----------------------------------------------------------------------------
# Bates (Heston + lognormal Poisson jumps), multi-asset
# -----------------------------------------------------------------------------
@dataclass
class BatesAssetParams:
    """Parameters for one asset under the Bates model."""
    S0: float = 100.0
    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    lam: float = 1.5      # jump intensity (per year)
    mu_J: float = -0.05   # mean log-jump size
    sigma_J: float = 0.10 # std of log-jump size


@dataclass
class BatesMultiAssetParams:
    """Multi-asset Bates: independent jump processes, correlated diffusions."""
    assets: List[BatesAssetParams] = field(default_factory=list)
    r: float = 0.05
    cross_corr: Optional[np.ndarray] = None  # (d, d) correlation matrix between W^{S,i}

    def n_assets(self) -> int:
        return len(self.assets)


def _ensure_corr_matrix(d: int, C: Optional[np.ndarray], default: float = 0.4) -> np.ndarray:
    if C is None:
        out = np.full((d, d), default)
        np.fill_diagonal(out, 1.0)
        return out
    out = np.array(C, dtype=np.float64)
    assert out.shape == (d, d), "cross_corr must be (d,d)"
    return out


def simulate_bates_multi(p: BatesMultiAssetParams,
                         T: float,
                         n_steps: int,
                         n_paths: int,
                         seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-asset Bates jump-diffusion.

    Returns
    -------
    S : (n_paths, n_steps+1, d)
    v : (n_paths, n_steps+1, d)
    jumps : (n_paths, n_steps, d)  binary jump-occurrence flag
    """
    rng = np.random.default_rng(seed)
    d = p.n_assets()
    if d == 0:
        raise ValueError("Bates parameters must contain at least one asset")
    dt = T / n_steps
    sdt = np.sqrt(dt)

    C = _ensure_corr_matrix(d, p.cross_corr)
    L = np.linalg.cholesky(C + 1e-10 * np.eye(d))

    # Pre-allocate
    S = np.empty((n_paths, n_steps + 1, d), dtype=np.float64)
    v = np.empty((n_paths, n_steps + 1, d), dtype=np.float64)
    jumps = np.zeros((n_paths, n_steps, d), dtype=np.float64)

    S[:, 0, :] = np.array([a.S0 for a in p.assets])
    v[:, 0, :] = np.array([a.v0 for a in p.assets])

    # Bates jump compensator m_J = E[e^J] - 1
    mJ = np.array([np.exp(a.mu_J + 0.5 * a.sigma_J ** 2) - 1.0 for a in p.assets])
    lam = np.array([a.lam for a in p.assets])

    for t in range(n_steps):
        # Asset-side Brownian increments (correlated across assets)
        Z = rng.standard_normal((n_paths, d))
        dW_S = (Z @ L.T) * sdt  # (n_paths, d)

        # Variance-side Brownians (correlated within-asset to dW_S via rho_i)
        Zv = rng.standard_normal((n_paths, d))
        rho = np.array([a.rho for a in p.assets])
        dW_v = (rho * Z + np.sqrt(np.clip(1.0 - rho ** 2, 0, 1)) * Zv) * sdt

        v_t = np.maximum(v[:, t, :], 0.0)
        sv = np.sqrt(v_t)

        # Diffusion increment in log price
        log_inc = (p.r - lam * mJ - 0.5 * v_t) * dt + sv * dW_S

        # Jump component: # of jumps ~ Poisson(lam dt) (we approximate to <=1 per step
        # at typical lam dt ~ 0.005 -- err on full Poisson for fidelity)
        N = rng.poisson(lam * dt, size=(n_paths, d)).astype(np.float64)
        # When N > 0 we sum N i.i.d. normals -> sum is N(N*mu, N*sigma^2)
        mu_J = np.array([a.mu_J for a in p.assets])
        sig_J = np.array([a.sigma_J for a in p.assets])
        jump_mean = N * mu_J
        jump_var = np.where(N > 0, N * sig_J ** 2, 0.0)
        jump_term = jump_mean + np.sqrt(jump_var) * rng.standard_normal((n_paths, d))
        jumps[:, t, :] = (N > 0).astype(np.float64)

        S[:, t + 1, :] = S[:, t, :] * np.exp(log_inc + jump_term)

        v_next = v_t + np.array([a.kappa for a in p.assets]) * (np.array([a.theta for a in p.assets]) - v_t) * dt \
                 + np.array([a.xi for a in p.assets]) * sv * dW_v
        v[:, t + 1, :] = np.maximum(v_next, 0.0)
    return S, v, jumps


# -----------------------------------------------------------------------------
# Crisis presets
# -----------------------------------------------------------------------------
def heston_calibration(scenario: str = "normal_us") -> HestonParams:
    """Returns Heston params calibrated to a regime.

    Scenarios derived from historical implied volatility levels.
    """
    s = scenario.lower()
    if s == "normal_us":
        return HestonParams(S0=100, v0=0.0225, r=0.04, kappa=1.0, theta=0.0225, xi=0.20, rho=-0.7)
    if s == "post_covid_us":
        return HestonParams(S0=100, v0=0.04, r=0.04, kappa=1.0, theta=0.04, xi=0.30, rho=-0.7)
    if s == "covid_us":
        return HestonParams(S0=100, v0=0.4225, r=0.01, kappa=1.5, theta=0.16, xi=0.55, rho=-0.85)
    if s == "gfc_2008":
        return HestonParams(S0=100, v0=0.64, r=0.02, kappa=1.5, theta=0.20, xi=0.70, rho=-0.9)
    if s == "normal_in":
        return HestonParams(S0=100, v0=0.0196, r=0.06, kappa=1.0, theta=0.0196, xi=0.20, rho=-0.6)
    if s == "covid_in":
        return HestonParams(S0=100, v0=0.3025, r=0.04, kappa=1.5, theta=0.16, xi=0.55, rho=-0.85)
    raise ValueError(f"Unknown scenario {scenario}")


def bates_calibration(scenario: str = "normal_us", n_assets: int = 1) -> BatesMultiAssetParams:
    """Returns Bates params for a regime, replicated across n_assets."""
    h = heston_calibration(scenario)
    s = scenario.lower()
    # Jump intensity grows in crisis regimes
    if "covid" in s:
        lam, muJ, sigJ = 6.0, -0.06, 0.18
    elif "gfc" in s:
        lam, muJ, sigJ = 8.0, -0.08, 0.22
    else:
        lam, muJ, sigJ = 1.5, -0.03, 0.10

    assets = [BatesAssetParams(
        S0=h.S0, v0=h.v0, kappa=h.kappa, theta=h.theta, xi=h.xi, rho=h.rho,
        lam=lam, mu_J=muJ, sigma_J=sigJ
    ) for _ in range(n_assets)]
    cross = None
    if n_assets > 1:
        # Plausible equity-equity correlation ~0.5 during normal, 0.75 in crisis
        rho = 0.75 if "covid" in s or "gfc" in s else 0.55
        cross = np.full((n_assets, n_assets), rho)
        np.fill_diagonal(cross, 1.0)
    return BatesMultiAssetParams(assets=assets, r=h.r, cross_corr=cross)
