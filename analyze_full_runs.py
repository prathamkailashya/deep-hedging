#!/usr/bin/env python3
"""
Re-compute summary statistics, bootstrap CIs and paired tests on
the TC-inclusive 10-seed rerun (``full_tc_validation_results.json``)
and on the SPY/NIFTY full-model rerun
(``spy_nifty_full_validation_results.json``).

Output: ``full_runs_analysis.json`` and a markdown summary printed
to stdout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
TC_FILE = ROOT / "full_tc_validation_results.json"
SPY_FILE = ROOT / "spy_nifty_full_validation_results.json"
OUT = ROOT / "full_runs_analysis.json"


def bootstrap_ci(values: List[float], n: int = 10000,
                  alpha: float = 0.05) -> Dict[str, float]:
    rng = np.random.default_rng(0)
    boots = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "std":  float(np.std(values, ddof=0)),
        "lo":   float(np.quantile(boots, alpha / 2)),
        "hi":   float(np.quantile(boots, 1 - alpha / 2)),
        "n":    len(values),
    }


def paired(a: List[float], b: List[float]) -> Dict[str, float]:
    a = np.asarray(a); b = np.asarray(b)
    diff = a - b
    if len(diff) >= 3 and stats.shapiro(diff).pvalue >= 0.05:
        t, p = stats.ttest_rel(a, b)
        test = "ttest_rel"
    else:
        if len(diff) < 6:
            t, p = stats.ttest_rel(a, b)
            test = "ttest_rel (Wilcoxon under-powered)"
        else:
            t, p = stats.wilcoxon(a, b)
            test = "wilcoxon"
    pooled = float(np.std(diff, ddof=1)) or 1e-12
    d = float(diff.mean() / pooled)
    return {"test": test, "stat": float(t), "p": float(p),
            "cohens_d": d, "mean_diff": float(diff.mean()),
            "pct_diff": float(100 * diff.mean() / b.mean())}


def analyse_tc() -> Dict:
    if not TC_FILE.exists():
        return {"status": "missing"}
    raw = json.loads(TC_FILE.read_text())
    seeds = sorted(raw.keys(), key=int)
    by_model = {}
    for s in seeds:
        for m, met in raw[s].items():
            by_model.setdefault(m, {}).setdefault("cvar_95_tc", []).append(met["cvar_95_tc"])
            by_model[m].setdefault("cvar_95_notc", []).append(met["cvar_95_notc"])
            by_model[m].setdefault("turnover", []).append(met["turnover"])
            by_model[m].setdefault("std_pnl_tc", []).append(met["std_pnl_tc"])
    out = {"n_seeds": len(seeds), "models": {}}
    for m, mets in by_model.items():
        out["models"][m] = {
            "cvar_95_tc":   bootstrap_ci(mets["cvar_95_tc"]),
            "cvar_95_notc": bootstrap_ci(mets["cvar_95_notc"]),
            "turnover":     bootstrap_ci(mets["turnover"]),
            "std_pnl_tc":   bootstrap_ci(mets["std_pnl_tc"]),
            "cv_pct":       100 * np.std(mets["cvar_95_tc"], ddof=0)
                                 / np.mean(mets["cvar_95_tc"]),
        }
    if "RSE" in by_model and "LSTM" in by_model:
        if len(by_model["RSE"]["cvar_95_tc"]) >= 3:
            out["RSE_vs_LSTM_tc"] = paired(
                by_model["RSE"]["cvar_95_tc"],
                by_model["LSTM"]["cvar_95_tc"],
            )
            out["RSE_vs_LSTM_notc"] = paired(
                by_model["RSE"]["cvar_95_notc"],
                by_model["LSTM"]["cvar_95_notc"],
            )
    return out


def analyse_spy() -> Dict:
    if not SPY_FILE.exists():
        return {"status": "missing"}
    raw = json.loads(SPY_FILE.read_text())
    return raw


def main() -> None:
    res = {
        "tc_validation": analyse_tc(),
        "spy_nifty":     analyse_spy(),
    }
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"Wrote {OUT}")
    if res["tc_validation"].get("status") != "missing":
        n = res["tc_validation"]["n_seeds"]
        print(f"\nTC validation: {n} seeds")
        for m, met in res["tc_validation"]["models"].items():
            tc = met["cvar_95_tc"]
            print(f"  {m:8s}  CVaR95_tc = {tc['mean']:.4f} ± {tc['std']:.4f}  "
                  f"[{tc['lo']:.4f}, {tc['hi']:.4f}]  "
                  f"CV={met['cv_pct']:.2f}%")
        if "RSE_vs_LSTM_tc" in res["tc_validation"]:
            p = res["tc_validation"]["RSE_vs_LSTM_tc"]
            print(f"\n  RSE vs LSTM (TC-inclusive): "
                  f"{p['test']}  t={p['stat']:.2f}  p={p['p']:.2e}  "
                  f"d={p['cohens_d']:.2f}  pct={p['pct_diff']:.2f}%")
    if res["spy_nifty"].get("status") != "missing":
        print("\nSPY/NIFTY full-model results:")
        for market, scens in res["spy_nifty"].items():
            for sname, mods in scens.items():
                line = f"  {market} {sname}: "
                line += " | ".join(f"{m}={v['cvar_95']:.2f}" for m, v in mods.items())
                print(line)


if __name__ == "__main__":
    main()
