"""Ablation studies referenced in thesis.tex Section 9.

Runs:
  (a) DRO radius sweep for W-DRO-T  (eps in {0, 0.05, 0.1, 0.15, 0.2})
  (b) RSE base-expert ablation       (drop one of LSTM/Transformer/Signature)
  (c) Bates-jumps in training        (Heston-trained eval'd on Bates)

Saves:
  outputs/runs/ablation_<name>.pkl
  outputs/tables/ablation_<name>.csv
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jaws_research.data.simulators import (heston_calibration, simulate_heston,
                                           bates_calibration, simulate_bates_multi)
from jaws_research.tasks.payoffs import european_call
from jaws_research.tasks.features import build_features, regime_features
from jaws_research.models.architectures import (LSTMHedger, TransformerHedger,
                                                SignatureHedger, RSEHedger)
from jaws_research.train.trainer import (BaseTrainer, WDROTrainer, RSETrainer,
                                         _device, _pnl)
from jaws_research.eval.metrics import evaluate_pnl

OUT = ROOT / "jaws_research" / "outputs"
RUNS = OUT / "runs"
TABLES = OUT / "tables"


def gen_data(scenario: str, n_paths: int, n_steps: int, T: float, seed: int,
             dynamics: str = "heston", n_assets: int = 1):
    if dynamics == "bates" or n_assets > 1:
        params = bates_calibration(scenario, n_assets=n_assets)
        S, v, _ = simulate_bates_multi(params, T, n_steps, n_paths, seed=seed)
        return S.astype(np.float32), v.astype(np.float32)
    p = heston_calibration(scenario)
    S, v = simulate_heston(p, T, n_steps, n_paths, seed=seed)
    return S.astype(np.float32), v.astype(np.float32)


# ---------------------------------------------------------------------------
def ablation_dro_radius(eps_list=(0.0, 0.05, 0.1, 0.15, 0.2),
                        seeds=(42, 142, 242), label: str = "dro_eps"):
    n_steps = 30; T = n_steps / 365.0; K = 100.0; c_tc = 0.001
    rows = []
    for eps in eps_list:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            S_tr, v_tr = gen_data("normal_us", 2000, n_steps, T, seed)
            S_te, v_te = gen_data("covid_us", 2500, n_steps, T, seed + 9000)
            X_tr, _ = build_features(S_tr, v_tr, T, K)
            X_te, _ = build_features(S_te, v_te, T, K)
            Z_tr = european_call(K=K).payoff(S_tr).astype(np.float32)
            Z_te = european_call(K=K).payoff(S_te).astype(np.float32)
            model = TransformerHedger(X_tr.shape[-1], n_assets=1, d_model=48,
                                       n_heads=4, n_layers=2, d_ff=96, dropout=0.1)
            tr = WDROTrainer(model, [0], c_tc=c_tc, eps_max=eps)
            tr.fit(X_tr, S_tr, Z_tr, epochs_cvar=10, epochs_ent=8, batch_size=128)
            deltas = tr.act(X_te)
            pnl = _pnl(torch.from_numpy(deltas), torch.from_numpy(S_te),
                        torch.from_numpy(Z_te), [0], c_tc).numpy()
            metrics = evaluate_pnl(pnl, deltas, S_te, [0])
            rows.append({"eps": eps, "seed": seed, **metrics})
            print(f"   eps={eps:.2f} seed={seed} cvar95={metrics['cvar_95']:.3f}")
    import pandas as pd
    pd.DataFrame(rows).to_csv(TABLES / f"ablation_{label}.csv", index=False)
    with open(RUNS / f"ablation_{label}.pkl", "wb") as fh:
        pickle.dump({"rows": rows}, fh)
    print(f"[OK] saved ablation_{label}")


# ---------------------------------------------------------------------------
def ablation_rse_experts(seeds=(42, 142, 242), label: str = "rse_experts"):
    n_steps = 30; T = n_steps / 365.0; K = 100.0; c_tc = 0.001
    expert_pool = ["LSTM", "Transformer", "Signature"]
    drop_combos = [None] + [(e,) for e in expert_pool]  # None = full ensemble
    rows = []
    for drop in drop_combos:
        kept = [e for e in expert_pool if drop is None or e not in drop]
        if len(kept) < 2:
            continue
        tag = "full" if drop is None else f"drop_{drop[0]}"
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            S_tr, v_tr = gen_data("normal_us", 2000, n_steps, T, seed)
            S_te, v_te = gen_data("normal_us", 2500, n_steps, T, seed + 9000)
            X_tr, _ = build_features(S_tr, v_tr, T, K)
            X_te, _ = build_features(S_te, v_te, T, K)
            Z_tr = european_call(K=K).payoff(S_tr).astype(np.float32)
            Z_te = european_call(K=K).payoff(S_te).astype(np.float32)
            r_tr = regime_features(S_tr).astype(np.float32)
            r_te = regime_features(S_te).astype(np.float32)
            bases = []
            for e in kept:
                if e == "LSTM":
                    m = LSTMHedger(X_tr.shape[-1], n_assets=1, hidden=48)
                elif e == "Transformer":
                    m = TransformerHedger(X_tr.shape[-1], n_assets=1, d_model=48,
                                           n_heads=4, n_layers=2, d_ff=96, dropout=0.1)
                else:
                    m = SignatureHedger(X_tr.shape[-1], n_assets=1, hidden=48)
                BaseTrainer(m, [0], c_tc=c_tc).fit(X_tr, S_tr, Z_tr,
                                                    epochs_cvar=8, epochs_ent=6,
                                                    batch_size=128)
                bases.append(m)
            rse = RSEHedger(bases, regime_dim=6, n_regimes=4, n_assets=1)
            rt = RSETrainer(rse, [0], c_tc=c_tc)
            rt.fit(X_tr, S_tr, Z_tr, regime=r_tr, epochs_gate=8, epochs_finetune=4,
                   batch_size=128)
            deltas = rt.act(X_te, r_te)
            pnl = _pnl(torch.from_numpy(deltas), torch.from_numpy(S_te),
                        torch.from_numpy(Z_te), [0], c_tc).numpy()
            metrics = evaluate_pnl(pnl, deltas, S_te, [0])
            rows.append({"experts": "+".join(kept), "tag": tag, "seed": seed,
                          **metrics})
            print(f"   {tag} seed={seed} cvar95={metrics['cvar_95']:.3f}")
    import pandas as pd
    pd.DataFrame(rows).to_csv(TABLES / f"ablation_{label}.csv", index=False)
    with open(RUNS / f"ablation_{label}.pkl", "wb") as fh:
        pickle.dump({"rows": rows}, fh)
    print(f"[OK] saved ablation_{label}")


# ---------------------------------------------------------------------------
def ablation_bates_train_test(seeds=(42, 142, 242), label: str = "bates_xfer"):
    n_steps = 30; T = n_steps / 365.0; K = 100.0; c_tc = 0.001
    rows = []
    for train_dyn in ["heston", "bates"]:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            S_tr, v_tr = gen_data("normal_us", 2000, n_steps, T, seed,
                                   dynamics=train_dyn)
            S_te, v_te = gen_data("covid_us", 2500, n_steps, T, seed + 9000,
                                   dynamics="bates")
            X_tr, _ = build_features(S_tr, v_tr, T, K)
            X_te, _ = build_features(S_te, v_te, T, K)
            Z_tr = european_call(K=K).payoff(S_tr).astype(np.float32)
            Z_te = european_call(K=K).payoff(S_te).astype(np.float32)
            model = TransformerHedger(X_tr.shape[-1], n_assets=1, d_model=48,
                                       n_heads=4, n_layers=2, d_ff=96, dropout=0.1)
            tr = WDROTrainer(model, [0], c_tc=c_tc, eps_max=0.1)
            tr.fit(X_tr, S_tr, Z_tr, epochs_cvar=10, epochs_ent=8, batch_size=128)
            deltas = tr.act(X_te)
            pnl = _pnl(torch.from_numpy(deltas), torch.from_numpy(S_te),
                        torch.from_numpy(Z_te), [0], c_tc).numpy()
            metrics = evaluate_pnl(pnl, deltas, S_te, [0])
            rows.append({"train_dynamics": train_dyn, "seed": seed, **metrics})
            print(f"   train={train_dyn} seed={seed} cvar95={metrics['cvar_95']:.3f}")
    import pandas as pd
    pd.DataFrame(rows).to_csv(TABLES / f"ablation_{label}.csv", index=False)
    with open(RUNS / f"ablation_{label}.pkl", "wb") as fh:
        pickle.dump({"rows": rows}, fh)
    print(f"[OK] saved ablation_{label}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["dro", "rse", "bates", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 142, 242])
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if args.which in ("dro", "all"):
        ablation_dro_radius(seeds=seeds)
    if args.which in ("rse", "all"):
        ablation_rse_experts(seeds=seeds)
    if args.which in ("bates", "all"):
        ablation_bates_train_test(seeds=seeds)
