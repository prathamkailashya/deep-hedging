"""Aggregate benchmark pickles into summary tables, statistical tests, and figures.

Usage:
    python -m jaws_research.experiments.analyze_results --label quick_v1
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jaws_research.eval.metrics import bootstrap_ci, holm_bonferroni, paired_test

OUT = ROOT / "jaws_research" / "outputs"
RUNS = OUT / "runs"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


METRICS_OF_INTEREST = ["cvar_95", "cvar_99", "std_pnl", "mean_pnl", "entropic", "turnover"]


def load_results(label: str) -> dict:
    path = RUNS / f"benchmark_{label}.pkl"
    with open(path, "rb") as fh:
        return pickle.load(fh)


def metrics_to_dataframe(blob: dict) -> pd.DataFrame:
    rows = []
    for (scenario, task), entry in blob["results"].items():
        for model, seed_metrics in entry["metrics"].items():
            for s_idx, m in enumerate(seed_metrics):
                row = {"scenario": scenario, "task": task, "model": model,
                       "seed_idx": s_idx}
                row.update(m)
                rows.append(row)
    return pd.DataFrame(rows)


def aggregate_table(df: pd.DataFrame, metric: str = "cvar_95") -> pd.DataFrame:
    """Compute mean/std/95% bootstrap CI of `metric` across seeds for each (scenario, task, model)."""
    rows = []
    for (scenario, task, model), grp in df.groupby(["scenario", "task", "model"]):
        vals = grp[metric].values
        mean, lo, hi = bootstrap_ci(vals, n_resample=2000)
        rows.append({"scenario": scenario, "task": task, "model": model,
                      "n_seeds": len(vals),
                      f"{metric}_mean": mean,
                      f"{metric}_std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
                      f"{metric}_lo":  lo,
                      f"{metric}_hi":  hi})
    return pd.DataFrame(rows)


def paired_vs_baseline(df: pd.DataFrame, baseline: str = "LSTM",
                       metric: str = "cvar_95") -> pd.DataFrame:
    """For each (scenario, task), paired test of each model vs baseline."""
    rows = []
    for (scenario, task), grp in df.groupby(["scenario", "task"]):
        bvals = grp[grp["model"] == baseline].sort_values("seed_idx")[metric].values
        for model, mgrp in grp.groupby("model"):
            if model == baseline:
                continue
            mvals = mgrp.sort_values("seed_idx")[metric].values
            if len(mvals) != len(bvals) or len(mvals) < 2:
                continue
            test = paired_test(mvals, bvals)
            rows.append({"scenario": scenario, "task": task, "model": model,
                          "baseline": baseline, "metric": metric,
                          "delta_pct": 100.0 * (mvals.mean() - bvals.mean()) / abs(bvals.mean() + 1e-9),
                          **test})
    out = pd.DataFrame(rows)
    if not out.empty:
        # Apply Holm-Bonferroni per scenario across tested models for clarity
        out["adjusted_significant"] = False
        for (scenario, task), g in out.groupby(["scenario", "task"]):
            pvs = dict(zip(g.index, g["p_value"].values))
            rej = holm_bonferroni(pvs, alpha=0.05)
            for idx, ok in rej.items():
                out.loc[idx, "adjusted_significant"] = ok
    return out


def winner_summary(df: pd.DataFrame, metric: str = "cvar_95") -> pd.DataFrame:
    """For each (scenario, task), report which model has the lowest mean metric."""
    out = []
    for (scenario, task), grp in df.groupby(["scenario", "task"]):
        means = grp.groupby("model")[metric].mean()
        winner = means.idxmin()
        out.append({"scenario": scenario, "task": task,
                     "winner": winner, f"{winner}_{metric}": means[winner]})
    return pd.DataFrame(out)


def export_latex_table(df: pd.DataFrame, out_path: Path, caption: str,
                       label: str, metric: str = "cvar_95"):
    """Write a LaTeX table with mean +/- std per (scenario, model) for one task."""
    pivot = df.pivot_table(index="scenario", columns="model",
                            values=f"{metric}_mean", aggfunc="first")
    err = df.pivot_table(index="scenario", columns="model",
                          values=f"{metric}_std", aggfunc="first")
    cols = list(pivot.columns)
    cols_tex = [c.replace("_", "-") for c in cols]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{l" + "c" * len(cols) + "}")
    lines.append(r"\toprule")
    lines.append("Scenario & " + " & ".join(cols_tex) + r" \\")
    lines.append(r"\midrule")
    for sc in pivot.index:
        cells = []
        for c in cols:
            val = pivot.at[sc, c]
            std = err.at[sc, c]
            cells.append(f"{val:.3f}\\,$\\pm$\\,{std:.3f}")
        lines.append(f"{sc.replace('_', ' ')} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines))


def main(label: str):
    blob = load_results(label)
    df = metrics_to_dataframe(blob)
    df.to_csv(TABLES / f"raw_metrics_{label}.csv", index=False)

    summaries = {}
    for metric in METRICS_OF_INTEREST:
        agg = aggregate_table(df, metric)
        agg.to_csv(TABLES / f"agg_{metric}_{label}.csv", index=False)
        summaries[metric] = agg

    paired = paired_vs_baseline(df, "LSTM", "cvar_95")
    paired.to_csv(TABLES / f"paired_LSTM_cvar95_{label}.csv", index=False)

    winners = winner_summary(df, "cvar_95")
    winners.to_csv(TABLES / f"winners_cvar95_{label}.csv", index=False)

    # Export European call LaTeX table
    eu = summaries["cvar_95"]
    eu_call = eu[eu["task"] == "european_call"].copy()
    if not eu_call.empty:
        export_latex_table(eu_call, TABLES / f"latex_eu_call_{label}.tex",
                            caption=f"CVaR$_{{95}}$ on European call across regimes ({label}).",
                            label=f"tab:eu_call_{label}",
                            metric="cvar_95")

    print(f"\n=== Winner summary ({metric}=cvar_95) ===")
    print(winners.to_string(index=False))
    print("\n=== Paired vs LSTM (cvar_95) ===")
    if not paired.empty:
        print(paired[["scenario", "task", "model", "delta_pct", "p_value",
                       "cohen_d", "adjusted_significant"]].round(4).to_string(index=False))
    print(f"\nArtifacts in {TABLES}")
    return df, summaries, paired, winners


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    main(args.label)
