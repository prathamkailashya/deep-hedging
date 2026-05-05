"""End-to-end benchmark across the 5 models, 6 tasks, 6 regimes.

Run as a module so package imports resolve:
    cd <repo root>
    python -m jaws_research.experiments.run_benchmark --mode quick

Modes:
    quick : 2 seeds, 1500 paths/regime, 8 epochs/stage  (smoke test, ~5 min)
    full  : 5 seeds, 6000 paths/regime, 25 epochs/stage (~few hours on CPU/MPS)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# Make repo root importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jaws_research.data.simulators import (BatesAssetParams, HestonParams,
                                           bates_calibration, heston_calibration,
                                           simulate_bates_multi, simulate_heston)
from jaws_research.tasks.payoffs import (asian_call, basket_call,
                                         cash_or_nothing_call, down_and_in_put,
                                         european_call, european_put,
                                         up_and_out_call)
from jaws_research.tasks.features import build_features, regime_features
from jaws_research.models.architectures import (LSTMHedger, RSEHedger,
                                                SignatureHedger,
                                                TransformerHedger)
from jaws_research.train.trainer import (BaseTrainer, RSETrainer,
                                         ThreeStageTrainer, WDROTrainer,
                                         _device, _pnl)
from jaws_research.eval.metrics import (bootstrap_ci, evaluate_pnl,
                                        holm_bonferroni, paired_test)


OUT = ROOT / "jaws_research" / "outputs"
RUNS = OUT / "runs"
LOGS = OUT / "logs"
RUNS.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def _log(msg: str, log_file=None):
    print(msg, flush=True)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


# ---------------------------------------------------------------------------
def make_paths(scenario: str, n_paths: int, n_steps: int, T: float,
               n_assets: int, seed: int, dynamics: str = "heston"):
    """Generate paths.  dynamics in {'heston','bates'}."""
    if dynamics == "bates" or n_assets > 1:
        params = bates_calibration(scenario, n_assets=n_assets)
        S, v, jumps = simulate_bates_multi(params, T, n_steps, n_paths, seed=seed)
        return S, v, jumps, params.r
    params = heston_calibration(scenario)
    S, v = simulate_heston(params, T, n_steps, n_paths, seed=seed)
    return S, v, np.zeros((n_paths, n_steps, 1)), params.r


# ---------------------------------------------------------------------------
TASK_BUILDERS = {
    "european_call":   lambda d, K: european_call(K=K, asset_idx=0),
    "european_put":    lambda d, K: european_put(K=K, asset_idx=0),
    "up_out_call":     lambda d, K: up_and_out_call(K=K, barrier=K * 1.30, asset_idx=0),
    "down_in_put":     lambda d, K: down_and_in_put(K=K, barrier=K * 0.80, asset_idx=0),
    "digital_call":    lambda d, K: cash_or_nothing_call(K=K, payout=10.0, asset_idx=0),
    "asian_call":      lambda d, K: asian_call(K=K, asset_idx=0),
    "basket_call":     lambda d, K: basket_call(weights=None, K=K),
}


# ---------------------------------------------------------------------------
def build_models(in_dim: int, n_assets: int, device):
    lstm = LSTMHedger(in_dim, n_assets=n_assets, hidden=48, n_layers=2)
    transf = TransformerHedger(in_dim, n_assets=n_assets, d_model=48, n_heads=4,
                               n_layers=2, d_ff=96, dropout=0.1)
    return {"LSTM": lstm, "Transformer": transf,
            "WDRO_T": TransformerHedger(in_dim, n_assets=n_assets, d_model=48,
                                        n_heads=4, n_layers=2, d_ff=96),
            "3SCH": LSTMHedger(in_dim, n_assets=n_assets, hidden=48, n_layers=2)}


def build_rse_bases(in_dim: int, n_assets: int):
    return [LSTMHedger(in_dim, n_assets=n_assets, hidden=48, n_layers=2),
            TransformerHedger(in_dim, n_assets=n_assets, d_model=48, n_heads=4,
                              n_layers=2, d_ff=96, dropout=0.1),
            SignatureHedger(in_dim, n_assets=n_assets, hidden=48)]


# ---------------------------------------------------------------------------
def train_one_model(name: str, model, X_tr, S_tr, Z_tr, hedge_assets,
                    c_tc: float, epochs_cvar: int, epochs_ent: int,
                    batch_size: int, regime_tr=None, lam_ent=1.0):
    if name == "WDRO_T":
        tr = WDROTrainer(model, hedge_assets, c_tc=c_tc, lam_ent=lam_ent)
        history = tr.fit(X_tr, S_tr, Z_tr, epochs_cvar=epochs_cvar,
                         epochs_ent=epochs_ent, batch_size=batch_size)
        return tr, history
    if name == "3SCH":
        tr = ThreeStageTrainer(model, hedge_assets, c_tc=c_tc, lam_ent=lam_ent)
        history = tr.fit(X_tr, S_tr, Z_tr,
                         epochs_cvar=int(epochs_cvar * 0.7),
                         epochs_mid=max(3, int((epochs_cvar + epochs_ent) * 0.15)),
                         epochs_ent=int(epochs_ent), batch_size=batch_size)
        return tr, history
    if name == "RSE":
        tr = RSETrainer(model, hedge_assets, c_tc=c_tc, lam_ent=lam_ent)
        history = tr.fit(X_tr, S_tr, Z_tr, regime=regime_tr,
                         epochs_gate=epochs_cvar, epochs_finetune=epochs_ent,
                         batch_size=batch_size)
        return tr, history
    tr = BaseTrainer(model, hedge_assets, c_tc=c_tc, lam_ent=lam_ent)
    history = tr.fit(X_tr, S_tr, Z_tr, epochs_cvar=epochs_cvar,
                     epochs_ent=epochs_ent, batch_size=batch_size)
    return tr, history


def evaluate_one(name: str, trainer, X_te, S_te, payoff_te, hedge_assets,
                 c_tc, regime_te=None):
    if name == "RSE":
        deltas = trainer.act(X_te, regime_te)
    else:
        deltas = trainer.act(X_te)
    deltas_t = torch.from_numpy(deltas)
    S_t = torch.from_numpy(S_te)
    Z_t = torch.from_numpy(payoff_te)
    pnl = _pnl(deltas_t, S_t, Z_t, hedge_assets, c_tc).numpy()
    metrics = evaluate_pnl(pnl, deltas, S_te, hedge_assets)
    return pnl, deltas, metrics


# ---------------------------------------------------------------------------
def run_benchmark(mode: str = "quick", out_label: str = None):
    cfg = {
        "quick": dict(seeds=[42, 142], n_train=1500, n_test=2000,
                       epochs_cvar=8, epochs_ent=6, batch_size=128,
                       scenarios=["normal_us", "covid_us"],
                       multi_asset_scenarios=["normal_us"],
                       tasks_single=["european_call", "up_out_call", "asian_call"],
                       tasks_multi=["basket_call"]),
        "medium": dict(seeds=[42, 142, 242], n_train=3000, n_test=3000,
                       epochs_cvar=14, epochs_ent=10, batch_size=192,
                       scenarios=["normal_us", "post_covid_us", "covid_us", "gfc_2008"],
                       multi_asset_scenarios=["normal_us", "covid_us"],
                       tasks_single=["european_call", "european_put",
                                     "up_out_call", "down_in_put",
                                     "digital_call", "asian_call"],
                       tasks_multi=["basket_call"]),
        "full":  dict(seeds=[42, 142, 242, 342, 442], n_train=6000, n_test=4000,
                       epochs_cvar=20, epochs_ent=15, batch_size=256,
                       scenarios=["normal_us", "post_covid_us", "covid_us",
                                  "gfc_2008", "normal_in", "covid_in"],
                       multi_asset_scenarios=["normal_us", "covid_us"],
                       tasks_single=["european_call", "european_put",
                                     "up_out_call", "down_in_put",
                                     "digital_call", "asian_call"],
                       tasks_multi=["basket_call"]),
    }[mode]

    out_label = out_label or mode
    log_path = LOGS / f"benchmark_{out_label}.log"
    log_file = open(log_path, "w")

    device = _device()
    _log(f"[INFO] device={device}", log_file)
    _log(f"[INFO] mode={mode} cfg={cfg}", log_file)

    n_steps = 30
    T = 30 / 365.0
    K = 100.0
    c_tc = 0.001
    lam_ent = 1.0

    all_results = {}
    started = time.time()

    # ------------------------------------------------------------------
    # SINGLE-ASSET SUITE
    # ------------------------------------------------------------------
    for scenario in cfg["scenarios"]:
        c_tc_scenario = 0.0018 if scenario.endswith("_in") else c_tc
        for task_name in cfg["tasks_single"]:
            _log(f"\n=== Scenario={scenario}  Task={task_name} (single-asset) ===", log_file)
            task = TASK_BUILDERS[task_name](1, K)
            seed_metrics = {m: [] for m in ["LSTM", "Transformer", "WDRO_T", "3SCH", "RSE"]}
            seed_pnls = {m: [] for m in seed_metrics}
            seed_deltas = {m: [] for m in seed_metrics}
            for seed in cfg["seeds"]:
                t0 = time.time()
                # Generate train/test paths from the scenario calibration
                S_tr, v_tr, _, _ = make_paths(scenario, cfg["n_train"], n_steps, T,
                                              n_assets=1, seed=seed,
                                              dynamics="heston")
                S_te, v_te, _, _ = make_paths(scenario, cfg["n_test"], n_steps, T,
                                              n_assets=1, seed=seed + 9000,
                                              dynamics="heston")
                X_tr, _ = build_features(S_tr, v_tr, T, K)
                X_te, _ = build_features(S_te, v_te, T, K)
                Z_tr = task.payoff(S_tr).astype(np.float32)
                Z_te = task.payoff(S_te).astype(np.float32)

                models = build_models(X_tr.shape[-1], n_assets=1, device=device)

                for name, model in models.items():
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    tr, _ = train_one_model(name, model, X_tr, S_tr.astype(np.float32),
                                            Z_tr, [0], c_tc_scenario,
                                            cfg["epochs_cvar"], cfg["epochs_ent"],
                                            cfg["batch_size"], lam_ent=lam_ent)
                    pnl, deltas, metrics = evaluate_one(name, tr, X_te,
                                                       S_te.astype(np.float32),
                                                       Z_te, [0], c_tc_scenario)
                    seed_metrics[name].append(metrics)
                    seed_pnls[name].append(pnl)
                    seed_deltas[name].append(deltas)

                # RSE: build ensemble from already-trained LSTM/Transformer +
                # a freshly trained Signature, then train gating.
                torch.manual_seed(seed)
                np.random.seed(seed)
                sig_model = SignatureHedger(X_tr.shape[-1], n_assets=1, hidden=48)
                sig_tr = BaseTrainer(sig_model, [0], c_tc=c_tc_scenario, lam_ent=lam_ent)
                sig_tr.fit(X_tr, S_tr.astype(np.float32), Z_tr,
                           epochs_cvar=max(4, cfg["epochs_cvar"] // 2),
                           epochs_ent=max(3, cfg["epochs_ent"] // 2),
                           batch_size=cfg["batch_size"])
                # Detach base models (no gradient for them)
                bases = [models["LSTM"], models["Transformer"], sig_model]
                rse = RSEHedger(bases, regime_dim=6, n_regimes=4, n_assets=1).to(device)
                r_tr = regime_features(S_tr).astype(np.float32)
                r_te = regime_features(S_te).astype(np.float32)
                rse_tr, _ = train_one_model("RSE", rse, X_tr, S_tr.astype(np.float32),
                                            Z_tr, [0], c_tc_scenario,
                                            cfg["epochs_cvar"], cfg["epochs_ent"],
                                            cfg["batch_size"], regime_tr=r_tr,
                                            lam_ent=lam_ent)
                pnl, deltas, metrics = evaluate_one("RSE", rse_tr, X_te,
                                                   S_te.astype(np.float32),
                                                   Z_te, [0], c_tc_scenario,
                                                   regime_te=r_te)
                seed_metrics["RSE"].append(metrics)
                seed_pnls["RSE"].append(pnl)
                seed_deltas["RSE"].append(deltas)
                _log(f"  seed={seed}  done in {time.time()-t0:.1f}s", log_file)
            all_results[(scenario, task_name)] = {
                "metrics": seed_metrics,
                "pnls":   seed_pnls,
                "deltas": seed_deltas,
            }

    # ------------------------------------------------------------------
    # MULTI-ASSET SUITE (basket call on Bates)
    # ------------------------------------------------------------------
    for scenario in cfg["multi_asset_scenarios"]:
        c_tc_scenario = c_tc
        for task_name in cfg["tasks_multi"]:
            _log(f"\n=== Scenario={scenario}  Task={task_name} (multi-asset, d=3) ===", log_file)
            task = TASK_BUILDERS[task_name](3, K)
            seed_metrics = {m: [] for m in ["LSTM", "Transformer", "WDRO_T", "3SCH", "RSE"]}
            seed_pnls = {m: [] for m in seed_metrics}
            seed_deltas = {m: [] for m in seed_metrics}
            hedge_assets = [0, 1, 2]
            for seed in cfg["seeds"]:
                t0 = time.time()
                S_tr, v_tr, _, _ = make_paths(scenario, cfg["n_train"], n_steps, T,
                                              n_assets=3, seed=seed, dynamics="bates")
                S_te, v_te, _, _ = make_paths(scenario, cfg["n_test"], n_steps, T,
                                              n_assets=3, seed=seed + 9000, dynamics="bates")
                X_tr, _ = build_features(S_tr, v_tr, T, K)
                X_te, _ = build_features(S_te, v_te, T, K)
                Z_tr = task.payoff(S_tr).astype(np.float32)
                Z_te = task.payoff(S_te).astype(np.float32)

                models = build_models(X_tr.shape[-1], n_assets=3, device=device)
                for name, model in models.items():
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    tr, _ = train_one_model(name, model, X_tr, S_tr.astype(np.float32),
                                            Z_tr, hedge_assets, c_tc_scenario,
                                            cfg["epochs_cvar"], cfg["epochs_ent"],
                                            cfg["batch_size"], lam_ent=lam_ent)
                    pnl, deltas, metrics = evaluate_one(name, tr, X_te,
                                                       S_te.astype(np.float32),
                                                       Z_te, hedge_assets, c_tc_scenario)
                    seed_metrics[name].append(metrics)
                    seed_pnls[name].append(pnl)
                    seed_deltas[name].append(deltas)

                # RSE for multi-asset (regime features from asset 0 still informative)
                torch.manual_seed(seed)
                np.random.seed(seed)
                sig_model = SignatureHedger(X_tr.shape[-1], n_assets=3, hidden=48)
                sig_tr = BaseTrainer(sig_model, hedge_assets, c_tc=c_tc_scenario,
                                     lam_ent=lam_ent)
                sig_tr.fit(X_tr, S_tr.astype(np.float32), Z_tr,
                           epochs_cvar=max(4, cfg["epochs_cvar"] // 2),
                           epochs_ent=max(3, cfg["epochs_ent"] // 2),
                           batch_size=cfg["batch_size"])
                bases = [models["LSTM"], models["Transformer"], sig_model]
                rse = RSEHedger(bases, regime_dim=6, n_regimes=4, n_assets=3).to(device)
                r_tr = regime_features(S_tr).astype(np.float32)
                r_te = regime_features(S_te).astype(np.float32)
                rse_tr, _ = train_one_model("RSE", rse, X_tr, S_tr.astype(np.float32),
                                            Z_tr, hedge_assets, c_tc_scenario,
                                            cfg["epochs_cvar"], cfg["epochs_ent"],
                                            cfg["batch_size"], regime_tr=r_tr,
                                            lam_ent=lam_ent)
                pnl, deltas, metrics = evaluate_one("RSE", rse_tr, X_te,
                                                   S_te.astype(np.float32),
                                                   Z_te, hedge_assets, c_tc_scenario,
                                                   regime_te=r_te)
                seed_metrics["RSE"].append(metrics)
                seed_pnls["RSE"].append(pnl)
                seed_deltas["RSE"].append(deltas)
                _log(f"  seed={seed}  done in {time.time()-t0:.1f}s", log_file)
            all_results[(scenario, task_name)] = {
                "metrics": seed_metrics,
                "pnls":   seed_pnls,
                "deltas": seed_deltas,
            }

    # ------------------------------------------------------------------
    elapsed = time.time() - started
    _log(f"\n[DONE] Total time {elapsed/60:.1f}min", log_file)

    out_path = RUNS / f"benchmark_{out_label}.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump({
            "config": cfg,
            "results": all_results,
            "elapsed_sec": elapsed,
            "device": str(device),
        }, fh)
    _log(f"[OUT] saved {out_path}", log_file)
    log_file.close()
    return out_path


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "medium", "full"], default="quick")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    run_benchmark(args.mode, args.label)
