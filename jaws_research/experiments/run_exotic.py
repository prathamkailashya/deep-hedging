"""Exotic-only synthetic benchmark.

Runs the five hedgers on the four non-European payoffs
(up-and-out, down-and-in, digital, Asian) and the basket call,
under Heston (normal_us, covid_us) and Bates (normal_us, covid_us).

Output: outputs/runs/benchmark_exotic_v1.pkl with the same schema as
run_benchmark.py so the same analysis tools work.

Usage:
    python -m jaws_research.experiments.run_exotic --label exotic_v1
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jaws_research.experiments.run_benchmark import (TASK_BUILDERS,
                                                     build_models, evaluate_one,
                                                     make_paths,
                                                     train_one_model)
from jaws_research.models.architectures import RSEHedger, SignatureHedger
from jaws_research.tasks.features import build_features, regime_features
from jaws_research.train.trainer import BaseTrainer, _device

OUT = ROOT / "jaws_research" / "outputs"
RUNS = OUT / "runs"
LOGS = OUT / "logs"
RUNS.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)


def run(label: str = "exotic_v1",
        seeds=(42, 142, 242),
        n_train=3000, n_test=3000,
        epochs_cvar=14, epochs_ent=10, batch_size=192):
    log = open(LOGS / f"benchmark_{label}.log", "w")

    def _l(m):
        print(m, flush=True)
        log.write(m + "\n"); log.flush()

    device = _device()
    _l(f"[INFO] device={device} seeds={seeds}")

    n_steps = 30; T = 30 / 365.0; K = 100.0; c_tc = 0.001
    exotic_single = ["up_out_call", "down_in_put", "digital_call", "asian_call"]
    multi_asset = ["basket_call"]

    all_results = {}
    started = time.time()

    # --- Single-asset exotics under Heston and Bates ---
    for dyn in ["heston", "bates"]:
        for scenario in ["normal_us", "covid_us"]:
            for task_name in exotic_single:
                _l(f"\n=== dyn={dyn} sc={scenario} task={task_name} ===")
                seed_metrics = {m: [] for m in
                                ["LSTM", "Transformer", "WDRO_T", "3SCH", "RSE"]}
                seed_pnls = {m: [] for m in seed_metrics}
                seed_deltas = {m: [] for m in seed_metrics}
                for seed in seeds:
                    t0 = time.time()
                    S_tr, v_tr, _, _ = make_paths(scenario, n_train, n_steps, T,
                                                  n_assets=1, seed=seed,
                                                  dynamics=dyn)
                    S_te, v_te, _, _ = make_paths(scenario, n_test, n_steps, T,
                                                  n_assets=1, seed=seed + 9000,
                                                  dynamics=dyn)
                    X_tr, _ = build_features(S_tr, v_tr, T, K)
                    X_te, _ = build_features(S_te, v_te, T, K)
                    task = TASK_BUILDERS[task_name](1, K)
                    Z_tr = task.payoff(S_tr).astype(np.float32)
                    Z_te = task.payoff(S_te).astype(np.float32)

                    models = build_models(X_tr.shape[-1], n_assets=1, device=device)
                    for name, model in models.items():
                        torch.manual_seed(seed); np.random.seed(seed)
                        tr, _ = train_one_model(name, model, X_tr,
                                                 S_tr.astype(np.float32), Z_tr,
                                                 [0], c_tc, epochs_cvar,
                                                 epochs_ent, batch_size)
                        pnl, deltas, metrics = evaluate_one(name, tr, X_te,
                                                            S_te.astype(np.float32),
                                                            Z_te, [0], c_tc)
                        seed_metrics[name].append(metrics)
                        seed_pnls[name].append(pnl)
                        seed_deltas[name].append(deltas)

                    sig = SignatureHedger(X_tr.shape[-1], n_assets=1, hidden=48)
                    BaseTrainer(sig, [0], c_tc=c_tc).fit(X_tr,
                                                          S_tr.astype(np.float32),
                                                          Z_tr,
                                                          epochs_cvar=max(4, epochs_cvar // 2),
                                                          epochs_ent=max(3, epochs_ent // 2),
                                                          batch_size=batch_size)
                    bases = [models["LSTM"], models["Transformer"], sig]
                    rse = RSEHedger(bases, regime_dim=6, n_regimes=4,
                                     n_assets=1).to(device)
                    r_tr = regime_features(S_tr).astype(np.float32)
                    r_te = regime_features(S_te).astype(np.float32)
                    rse_tr, _ = train_one_model("RSE", rse, X_tr,
                                                 S_tr.astype(np.float32), Z_tr,
                                                 [0], c_tc, epochs_cvar,
                                                 epochs_ent, batch_size,
                                                 regime_tr=r_tr)
                    pnl, deltas, metrics = evaluate_one("RSE", rse_tr, X_te,
                                                        S_te.astype(np.float32),
                                                        Z_te, [0], c_tc,
                                                        regime_te=r_te)
                    seed_metrics["RSE"].append(metrics)
                    seed_pnls["RSE"].append(pnl)
                    seed_deltas["RSE"].append(deltas)
                    _l(f"   seed={seed} done in {time.time()-t0:.1f}s")
                all_results[(f"{dyn}:{scenario}", task_name)] = {
                    "metrics": seed_metrics,
                    "pnls": seed_pnls,
                    "deltas": seed_deltas,
                }

    # --- Multi-asset basket on Bates ---
    for scenario in ["normal_us", "covid_us"]:
        for task_name in multi_asset:
            _l(f"\n=== dyn=bates(d=3) sc={scenario} task={task_name} ===")
            seed_metrics = {m: [] for m in
                            ["LSTM", "Transformer", "WDRO_T", "3SCH", "RSE"]}
            seed_pnls = {m: [] for m in seed_metrics}
            seed_deltas = {m: [] for m in seed_metrics}
            hedge_assets = [0, 1, 2]
            for seed in seeds:
                t0 = time.time()
                S_tr, v_tr, _, _ = make_paths(scenario, n_train, n_steps, T,
                                              n_assets=3, seed=seed,
                                              dynamics="bates")
                S_te, v_te, _, _ = make_paths(scenario, n_test, n_steps, T,
                                              n_assets=3, seed=seed + 9000,
                                              dynamics="bates")
                X_tr, _ = build_features(S_tr, v_tr, T, K)
                X_te, _ = build_features(S_te, v_te, T, K)
                task = TASK_BUILDERS[task_name](3, K)
                Z_tr = task.payoff(S_tr).astype(np.float32)
                Z_te = task.payoff(S_te).astype(np.float32)
                models = build_models(X_tr.shape[-1], n_assets=3, device=device)
                for name, model in models.items():
                    torch.manual_seed(seed); np.random.seed(seed)
                    tr, _ = train_one_model(name, model, X_tr,
                                             S_tr.astype(np.float32), Z_tr,
                                             hedge_assets, c_tc, epochs_cvar,
                                             epochs_ent, batch_size)
                    pnl, deltas, metrics = evaluate_one(name, tr, X_te,
                                                        S_te.astype(np.float32),
                                                        Z_te, hedge_assets, c_tc)
                    seed_metrics[name].append(metrics)
                    seed_pnls[name].append(pnl)
                    seed_deltas[name].append(deltas)

                sig = SignatureHedger(X_tr.shape[-1], n_assets=3, hidden=48)
                BaseTrainer(sig, hedge_assets, c_tc=c_tc).fit(X_tr,
                                                                S_tr.astype(np.float32),
                                                                Z_tr,
                                                                epochs_cvar=max(4, epochs_cvar // 2),
                                                                epochs_ent=max(3, epochs_ent // 2),
                                                                batch_size=batch_size)
                bases = [models["LSTM"], models["Transformer"], sig]
                rse = RSEHedger(bases, regime_dim=6, n_regimes=4,
                                 n_assets=3).to(device)
                r_tr = regime_features(S_tr).astype(np.float32)
                r_te = regime_features(S_te).astype(np.float32)
                rse_tr, _ = train_one_model("RSE", rse, X_tr,
                                             S_tr.astype(np.float32), Z_tr,
                                             hedge_assets, c_tc, epochs_cvar,
                                             epochs_ent, batch_size,
                                             regime_tr=r_tr)
                pnl, deltas, metrics = evaluate_one("RSE", rse_tr, X_te,
                                                    S_te.astype(np.float32),
                                                    Z_te, hedge_assets, c_tc,
                                                    regime_te=r_te)
                seed_metrics["RSE"].append(metrics)
                seed_pnls["RSE"].append(pnl)
                seed_deltas["RSE"].append(deltas)
                _l(f"   seed={seed} done in {time.time()-t0:.1f}s")
            all_results[(f"bates3:{scenario}", task_name)] = {
                "metrics": seed_metrics,
                "pnls": seed_pnls,
                "deltas": seed_deltas,
            }

    elapsed = time.time() - started
    _l(f"\n[DONE] total {elapsed/60:.1f}min")
    out_path = RUNS / f"benchmark_{label}.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump({"results": all_results, "elapsed_sec": elapsed,
                      "device": str(device),
                      "config": {"seeds": list(seeds), "n_train": n_train,
                                  "n_test": n_test,
                                  "epochs_cvar": epochs_cvar,
                                  "epochs_ent": epochs_ent}}, fh)
    _l(f"[OUT] saved {out_path}")
    log.close()
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="exotic_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 142, 242])
    args = parser.parse_args()
    run(label=args.label, seeds=tuple(args.seeds))
