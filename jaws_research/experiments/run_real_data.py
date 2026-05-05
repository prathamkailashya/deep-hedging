"""Real-market evaluation with crisis windows.

For each ticker, we:
  1. Pull daily OHLCV via yfinance (with synthetic fallback when offline).
  2. Cut the close-price series into rolling windows of length N.
  3. Compute realised-variance proxies for the path-level features.
  4. Train each model on a *calm* training window and evaluate on each
     crisis window (true OOS, not synthetic perturbation).

Outputs: jaws_research/outputs/runs/real_<label>.pkl  with the same schema as
``run_benchmark.py`` so it can be analysed with the same tools.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jaws_research.data.online_data import (CRISIS_WINDOWS, fetch_with_fallback,
                                            path_windows, realised_var_proxy)
from jaws_research.experiments.run_benchmark import (TASK_BUILDERS,
                                                     build_models,
                                                     build_rse_bases,
                                                     evaluate_one,
                                                     train_one_model)
from jaws_research.models.architectures import RSEHedger, SignatureHedger
from jaws_research.tasks.features import build_features, regime_features
from jaws_research.train.trainer import BaseTrainer, _device

OUT = ROOT / "jaws_research" / "outputs"
RUNS = OUT / "runs"
LOGS = OUT / "logs"
RUNS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def get_paths_for_window(close: np.ndarray, n_steps: int = 30) -> np.ndarray:
    p = path_windows(close, window=n_steps, step=1)
    return p[:, :, None]  # (n, T+1, 1)


def get_features(close_paths: np.ndarray, T: float) -> np.ndarray:
    rv = realised_var_proxy(close_paths[:, :, 0], window=10)[:, :, None]
    X, _ = build_features(close_paths, rv, T)
    return X, rv


# ---------------------------------------------------------------------------
def run_real_data(label: str = "real_v1",
                  tickers: List[str] = ("SPY", "QQQ", "^NSEI"),
                  train_window=("2017-01-01", "2019-12-31"),
                  test_windows=("calm_2019", "covid_acute", "vol_q4_2018",
                                "inflation_2022", "banking_2023"),
                  task_names=("european_call", "up_out_call",
                              "digital_call", "asian_call"),
                  seeds=(42, 142, 242),
                  n_steps=30, c_tc=0.001):
    log_path = LOGS / f"real_{label}.log"
    log = open(log_path, "w")

    def _log(msg):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    device = _device()
    _log(f"[INFO] real-data run -- device={device}")
    _log(f"[INFO] tickers={tickers}  test_windows={test_windows}")

    K = 100.0
    T = n_steps / 365.0

    all_results = {}
    started = time.time()

    for ticker in tickers:
        _log(f"\n[TICKER] {ticker}")
        df, source = fetch_with_fallback(ticker, start="2005-01-01")
        _log(f"   data source: {source}, range {df.index[0].date()} -> {df.index[-1].date()}, n={len(df)}")
        close = df["Close"].astype(np.float64).values

        # Training paths: from calm window
        idx_dates = df.index
        tr_mask = (idx_dates >= train_window[0]) & (idx_dates <= train_window[1])
        tr_close = close[tr_mask]
        if len(tr_close) < n_steps + 1:
            _log(f"   skipping {ticker}: insufficient training data")
            continue
        tr_paths = get_paths_for_window(tr_close, n_steps)
        if len(tr_paths) < 200:
            _log(f"   skipping {ticker}: only {len(tr_paths)} train windows")
            continue
        X_tr, _ = get_features(tr_paths, T)
        regime_tr = regime_features(tr_paths).astype(np.float32)

        for task_name in task_names:
            task = TASK_BUILDERS[task_name](1, K)
            Z_tr = task.payoff(tr_paths).astype(np.float32)

            for window_name in test_windows:
                # Reset per-window metric containers so each (ticker,window,task)
                # cell gets its own data.
                seed_metrics = {m: [] for m in ["LSTM", "Transformer", "WDRO_T", "3SCH", "RSE"]}
                seed_pnls = {m: [] for m in seed_metrics}
                seed_deltas = {m: [] for m in seed_metrics}
                if window_name not in CRISIS_WINDOWS:
                    continue
                w_s, w_e = CRISIS_WINDOWS[window_name]
                te_mask = (idx_dates >= w_s) & (idx_dates <= w_e)
                te_close = close[te_mask]
                if len(te_close) < n_steps + 1:
                    continue
                te_paths = get_paths_for_window(te_close, n_steps)
                X_te, _ = get_features(te_paths, T)
                Z_te = task.payoff(te_paths).astype(np.float32)
                regime_te = regime_features(te_paths).astype(np.float32)

                for seed in seeds:
                    t0 = time.time()
                    torch.manual_seed(seed); np.random.seed(seed)
                    models = build_models(X_tr.shape[-1], n_assets=1, device=device)
                    for name, model in models.items():
                        torch.manual_seed(seed); np.random.seed(seed)
                        tr, _ = train_one_model(name, model, X_tr,
                                                 tr_paths.astype(np.float32),
                                                 Z_tr, [0], c_tc,
                                                 epochs_cvar=10, epochs_ent=8,
                                                 batch_size=128)
                        pnl, deltas, metrics = evaluate_one(name, tr, X_te,
                                                            te_paths.astype(np.float32),
                                                            Z_te, [0], c_tc)
                        metrics["scenario"] = f"{ticker}:{window_name}"
                        seed_metrics[name].append(metrics)
                        seed_pnls[name].append(pnl)
                        seed_deltas[name].append(deltas)

                    # RSE -- reuse just-trained LSTM/Transformer + new Signature
                    sig_model = SignatureHedger(X_tr.shape[-1], n_assets=1, hidden=48)
                    sig_tr = BaseTrainer(sig_model, [0], c_tc=c_tc)
                    sig_tr.fit(X_tr, tr_paths.astype(np.float32), Z_tr,
                               epochs_cvar=6, epochs_ent=4, batch_size=128)
                    bases = [models["LSTM"], models["Transformer"], sig_model]
                    rse = RSEHedger(bases, regime_dim=6, n_regimes=4, n_assets=1).to(device)
                    rse_tr, _ = train_one_model("RSE", rse, X_tr,
                                                 tr_paths.astype(np.float32),
                                                 Z_tr, [0], c_tc,
                                                 epochs_cvar=10, epochs_ent=6,
                                                 batch_size=128, regime_tr=regime_tr)
                    pnl, deltas, metrics = evaluate_one("RSE", rse_tr, X_te,
                                                        te_paths.astype(np.float32),
                                                        Z_te, [0], c_tc,
                                                        regime_te=regime_te)
                    metrics["scenario"] = f"{ticker}:{window_name}"
                    seed_metrics["RSE"].append(metrics)
                    seed_pnls["RSE"].append(pnl)
                    seed_deltas["RSE"].append(deltas)
                    _log(f"   {ticker} {task_name} {window_name} seed={seed} done in {time.time()-t0:.1f}s")
                all_results[(f"{ticker}:{window_name}", task_name)] = {
                    "metrics": seed_metrics,
                    "pnls":   seed_pnls,
                    "deltas": seed_deltas,
                    "data_source": source,
                }
    elapsed = time.time() - started
    _log(f"\n[DONE] real-data total {elapsed/60:.1f}min")

    out_path = RUNS / f"real_{label}.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump({"results": all_results, "elapsed_sec": elapsed,
                      "tickers": list(tickers),
                      "train_window": train_window,
                      "test_windows": list(test_windows),
                      "tasks": list(task_names),
                      "seeds": list(seeds)}, fh)
    _log(f"[OUT] saved {out_path}")
    log.close()
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="real_v1")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "^NSEI"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 142])
    args = parser.parse_args()
    run_real_data(label=args.label, tickers=tuple(args.tickers), seeds=tuple(args.seeds))
