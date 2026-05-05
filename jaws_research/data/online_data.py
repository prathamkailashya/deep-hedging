"""Online market-data feeds with crisis-regime windows and walk-forward splits.

We use ``yfinance`` for daily OHLCV (free, no key). Data is cached as parquet/CSV
under ``jaws_research/outputs/cache`` to avoid repeated calls and to make the
pipeline reproducible offline.

Crisis windows (configurable):
    - GFC 2008    : 2007-07-01 -- 2009-06-30
    - COVID-19    : 2020-02-01 -- 2020-04-30 (acute) / 2020-02-01 -- 2020-12-31 (extended)
    - Vol-spike 2018-Q4
    - 2022 inflation regime : 2022-01-01 -- 2022-12-31
    - 2023 banking stress : 2023-03-01 -- 2023-05-31
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[1] / "outputs" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Crisis & regime windows
# -----------------------------------------------------------------------------
CRISIS_WINDOWS: Dict[str, Tuple[str, str]] = {
    "gfc_2008":      ("2007-07-01", "2009-06-30"),
    "covid_acute":   ("2020-02-01", "2020-04-30"),
    "covid_extended":("2020-02-01", "2020-12-31"),
    "vol_q4_2018":   ("2018-10-01", "2018-12-31"),
    "inflation_2022":("2022-01-01", "2022-12-31"),
    "banking_2023":  ("2023-03-01", "2023-05-31"),
    "calm_2017":     ("2017-01-01", "2017-12-31"),
    "calm_2019":     ("2019-04-01", "2019-12-31"),
    "calm_2024":     ("2024-01-01", "2024-12-31"),
}


# -----------------------------------------------------------------------------
# Tickers
# -----------------------------------------------------------------------------
DEFAULT_TICKERS_US = ["SPY", "QQQ", "IWM"]
DEFAULT_TICKERS_IN = ["^NSEI", "^NSEBANK", "RELIANCE.NS"]


# -----------------------------------------------------------------------------
# yfinance fetch with caching
# -----------------------------------------------------------------------------
def fetch_history(ticker: str,
                  start: str = "2005-01-01",
                  end: Optional[str] = None,
                  use_cache: bool = True,
                  force_refresh: bool = False) -> pd.DataFrame:
    """Download daily OHLCV via yfinance, cached to disk.

    Returns a DataFrame indexed by Date with columns Open/High/Low/Close/Volume.
    """
    import yfinance as yf

    end = end or datetime.now().strftime("%Y-%m-%d")
    safe = ticker.replace("^", "_").replace("/", "-")
    cache_path = CACHE_DIR / f"{safe}_{start}_{end}.parquet"

    if use_cache and not force_refresh and cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if not df.empty:
                return df
        except Exception:  # pragma: no cover
            pass

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"yfinance download failed for {ticker}: {exc}")

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker} between {start} and {end}")

    # yfinance newer versions return MultiIndex columns when one ticker is given
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    try:
        df.to_parquet(cache_path)
    except Exception:  # pragma: no cover
        df.to_csv(cache_path.with_suffix(".csv"))
    return df


# -----------------------------------------------------------------------------
# Synthetic fallback (offline-safe)
# -----------------------------------------------------------------------------
def synthetic_history(ticker: str,
                      start: str,
                      end: str,
                      seed: int = 0) -> pd.DataFrame:
    """Deterministic synthetic OHLCV used when network access is unavailable.

    Generates daily Heston-like returns with a vol-regime that loosely tracks the
    crisis windows above so downstream walk-forward tests are still meaningful.
    """
    rng = np.random.default_rng(abs(hash((ticker, seed))) % (2 ** 32 - 1))
    idx = pd.bdate_range(start, end)
    n = len(idx)
    if n == 0:
        raise ValueError("Empty date range")

    sigma = np.full(n, 0.012)  # ~ 19% annualised
    for name, (s, e) in CRISIS_WINDOWS.items():
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        mask = (idx >= s) & (idx <= e)
        if "covid" in name or "gfc" in name:
            sigma[mask] = 0.045
        elif "vol" in name or "banking" in name:
            sigma[mask] = 0.025
        elif "inflation" in name:
            sigma[mask] = 0.018

    rets = rng.normal(0.0002, sigma)
    # add Poisson jumps in crisis windows
    for name, (s, e) in CRISIS_WINDOWS.items():
        if "covid" not in name and "gfc" not in name:
            continue
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        mask = (idx >= s) & (idx <= e)
        n_jump = rng.poisson(0.05 * mask.sum())
        if n_jump > 0:
            jump_idx = rng.choice(np.where(mask)[0], size=n_jump, replace=False)
            rets[jump_idx] += rng.normal(-0.04, 0.02, size=n_jump)
    px = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({
        "Open":   px * (1 + rng.normal(0, 0.0008, n)),
        "High":   px * (1 + np.abs(rng.normal(0, 0.003, n))),
        "Low":    px * (1 - np.abs(rng.normal(0, 0.003, n))),
        "Close":  px,
        "Volume": rng.integers(1_000_000, 50_000_000, n),
    }, index=idx)
    df.index.name = "Date"
    return df


def fetch_with_fallback(ticker: str,
                        start: str = "2005-01-01",
                        end: Optional[str] = None,
                        seed: int = 0) -> Tuple[pd.DataFrame, str]:
    """Try online fetch; on failure, return synthetic data tagged with source.

    Returns (df, source) where source is "yfinance" or "synthetic".
    """
    end = end or datetime.now().strftime("%Y-%m-%d")
    try:
        df = fetch_history(ticker, start, end)
        return df, "yfinance"
    except Exception:
        return synthetic_history(ticker, start, end, seed=seed), "synthetic"


# -----------------------------------------------------------------------------
# Walk-forward / regime split
# -----------------------------------------------------------------------------
@dataclass
class WalkForwardSplit:
    train: Tuple[str, str]
    val:   Tuple[str, str]
    test:  Tuple[str, str]
    label: str


def walk_forward_splits(start: str = "2010-01-01",
                        end: str = "2025-01-01",
                        train_years: int = 4,
                        val_months: int = 6,
                        test_months: int = 6) -> List[WalkForwardSplit]:
    """Generate rolling walk-forward train/val/test splits."""
    splits: List[WalkForwardSplit] = []
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while True:
        tr_s = cursor
        tr_e = tr_s + pd.DateOffset(years=train_years)
        v_e = tr_e + pd.DateOffset(months=val_months)
        te_e = v_e + pd.DateOffset(months=test_months)
        if te_e > end_ts:
            break
        splits.append(WalkForwardSplit(
            train=(str(tr_s.date()), str(tr_e.date())),
            val=(str(tr_e.date()), str(v_e.date())),
            test=(str(v_e.date()), str(te_e.date())),
            label=f"WF_{tr_s.year}-{te_e.year}",
        ))
        cursor = cursor + pd.DateOffset(months=test_months)
    return splits


def regime_label(ts: pd.Timestamp) -> str:
    """Tag a date with the crisis-regime label it falls in (or 'calm')."""
    for name, (s, e) in CRISIS_WINDOWS.items():
        if pd.Timestamp(s) <= ts <= pd.Timestamp(e):
            return name
    return "other"


def crisis_blocks(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Slice a price DataFrame into the labelled crisis blocks defined above."""
    out = {}
    for name, (s, e) in CRISIS_WINDOWS.items():
        block = df.loc[(df.index >= s) & (df.index <= e)]
        if len(block) >= 30:
            out[name] = block
    return out


# -----------------------------------------------------------------------------
# Path windows (for hedging episodes)
# -----------------------------------------------------------------------------
def path_windows(close: np.ndarray,
                 window: int = 30,
                 step: int = 1) -> np.ndarray:
    """Cut a single price series into overlapping windows of fixed length.

    Returns array of shape (n_windows, window+1) -- each row is a price path
    starting at S_0.  We rebase prices so the first element equals 100 to keep
    feature scales consistent across regimes.
    """
    px = np.asarray(close, dtype=np.float64)
    if len(px) < window + 1:
        return np.empty((0, window + 1))
    starts = np.arange(0, len(px) - window, step)
    paths = np.stack([px[i:i + window + 1] for i in starts], axis=0)
    paths = 100.0 * paths / paths[:, [0]]
    return paths


def realised_var_proxy(paths: np.ndarray, window: int = 5) -> np.ndarray:
    """Estimate instantaneous variance via rolling realised variance of log-returns.

    paths : (n, T+1)  ->  var : (n, T+1) with the first ``window`` entries set
    to the global mean variance.
    """
    log_ret = np.diff(np.log(paths), axis=1)
    n, T = log_ret.shape
    rv = np.zeros((n, T + 1))
    annual_factor = 252.0
    for t in range(T):
        lo = max(0, t - window + 1)
        rv[:, t + 1] = np.var(log_ret[:, lo:t + 1], axis=1) * annual_factor
    rv[:, 0] = rv[:, 1]
    rv = np.maximum(rv, 1e-6)
    return rv
