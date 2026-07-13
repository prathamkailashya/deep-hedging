"""Regression test: walk-forward features must be causal.

At decision time t the policy may use closes[0..t] only. We perturb
every close strictly after time t and assert that window 0's features
at decision times <= t are unchanged. This test failed before the
2026-07-13 fix, when the realised-vol window included the return over
[t, t+1] (and, through it, the Black--Scholes delta feature).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from walk_forward_backtest import make_windows


def test_features_do_not_depend_on_future_closes():
    rng = np.random.default_rng(0)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=200)))
    base = make_windows(closes)
    for t in [0, 1, 5, 10, 20, 29]:
        pert = closes.copy()
        pert[t + 1:] *= 1.10
        shocked = make_windows(pert)
        f1 = base["features"][0, : t + 1].numpy()
        f2 = shocked["features"][0, : t + 1].numpy()
        assert np.array_equal(f1, f2), (
            f"feature leak at decision time {t}: "
            f"max diff {np.abs(f1 - f2).max():.3e}"
        )


def test_payoff_and_prices_shapes():
    rng = np.random.default_rng(1)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=80)))
    d = make_windows(closes)
    n_windows = len(closes) - 31
    assert d["features"].shape == (n_windows, 30, 5)
    assert d["prices"].shape == (n_windows, 31)
    assert d["payoff"].shape == (n_windows,)
