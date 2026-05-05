"""Generate publication-quality figures from the saved benchmark pickle."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jaws_research.experiments.analyze_results import load_results, metrics_to_dataframe

OUT = ROOT / "jaws_research" / "outputs"
RUNS = OUT / "runs"
FIGS = OUT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


COLORS = {"LSTM": "#1f77b4", "Transformer": "#ff7f0e",
          "WDRO_T": "#d62728", "3SCH": "#2ca02c", "RSE": "#9467bd"}


def fig_cvar_by_scenario(df: pd.DataFrame, task: str, label: str):
    sub = df[df["task"] == task]
    if sub.empty:
        return None
    pivot = sub.groupby(["scenario", "model"])["cvar_95"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    scenarios = pivot["scenario"].unique().tolist()
    models = pivot["model"].unique().tolist()
    width = 0.8 / len(models)
    for j, m in enumerate(models):
        sub_m = pivot[pivot["model"] == m].set_index("scenario").reindex(scenarios)
        x = np.arange(len(scenarios)) + j * width - 0.4 + width / 2
        ax.bar(x, sub_m["mean"].values, width=width, yerr=sub_m["std"].values,
               capsize=2, label=m, color=COLORS.get(m, None))
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"CVaR$_{95}$  (lower is better)")
    ax.set_title(f"{task}  -  CVaR$_{{95}}$ across regimes")
    ax.legend(ncol=len(models), fontsize=8, loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fp = FIGS / f"cvar_by_scenario_{task}_{label}.pdf"
    fig.savefig(fp)
    fig.savefig(fp.with_suffix(".png"), dpi=160)
    plt.close(fig)
    return fp


def fig_pnl_distributions(blob: dict, scenario: str, task: str, label: str):
    if (scenario, task) not in blob["results"]:
        return None
    entry = blob["results"][(scenario, task)]
    pnl_dict = entry["pnls"]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    for name in ["LSTM", "Transformer", "WDRO_T", "3SCH", "RSE"]:
        if name not in pnl_dict or len(pnl_dict[name]) == 0:
            continue
        all_pnl = np.concatenate(pnl_dict[name])
        ax.hist(all_pnl, bins=80, alpha=0.45, label=name,
                color=COLORS.get(name, None), histtype="stepfilled")
    ax.set_xlabel("P&L (per path)")
    ax.set_ylabel("frequency")
    ax.set_title(f"P&L distribution -- {scenario} / {task}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = FIGS / f"pnl_dist_{scenario}_{task}_{label}.pdf"
    fig.savefig(fp); fig.savefig(fp.with_suffix(".png"), dpi=160)
    plt.close(fig)
    return fp


def fig_turnover_vs_cvar(df: pd.DataFrame, label: str):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    for m in df["model"].unique():
        sub = df[df["model"] == m]
        ax.scatter(sub["turnover"], sub["cvar_95"], label=m,
                   color=COLORS.get(m, None), s=18, alpha=0.6)
    ax.set_xlabel("turnover  (per-path Sum |Delta_t - Delta_{t-1}| S_t / S_0)")
    ax.set_ylabel("CVaR$_{95}$")
    ax.set_title("Risk-cost frontier")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = FIGS / f"turnover_vs_cvar_{label}.pdf"
    fig.savefig(fp); fig.savefig(fp.with_suffix(".png"), dpi=160)
    plt.close(fig)
    return fp


def fig_crisis_robustness(df: pd.DataFrame, label: str):
    """Crisis-to-normal CVaR ratio per model (single asset, european_call)."""
    sub = df[df["task"] == "european_call"]
    rows = []
    for model, grp in sub.groupby("model"):
        normals = grp[grp["scenario"] == "normal_us"]["cvar_95"].mean()
        for sc in ["covid_us", "gfc_2008", "covid_in"]:
            if sc not in grp["scenario"].unique():
                continue
            crisis = grp[grp["scenario"] == sc]["cvar_95"].mean()
            if normals and not np.isnan(normals) and normals > 0:
                rows.append({"model": model, "scenario": sc,
                              "ratio": crisis / normals})
    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return None
    pivot = rdf.pivot(index="model", columns="scenario", values="ratio")
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    pivot.plot(kind="bar", ax=ax, color=["#d62728", "#9467bd", "#ff7f0e"][:pivot.shape[1]])
    ax.set_ylabel(r"CVaR$_{95}^{crisis}$ / CVaR$_{95}^{normal}$")
    ax.set_title("Crisis-to-normal CVaR ratio")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fp = FIGS / f"crisis_ratio_{label}.pdf"
    fig.savefig(fp); fig.savefig(fp.with_suffix(".png"), dpi=160)
    plt.close(fig)
    return fp


def main(label: str):
    blob = load_results(label)
    df = metrics_to_dataframe(blob)

    artifacts = []
    for task in df["task"].unique():
        artifacts.append(fig_cvar_by_scenario(df, task, label))
    for sc, t in [("normal_us", "european_call"), ("covid_us", "european_call"),
                   ("gfc_2008", "european_call"), ("normal_in", "european_call"),
                   ("normal_us", "basket_call"), ("normal_us", "up_out_call")]:
        artifacts.append(fig_pnl_distributions(blob, sc, t, label))
    artifacts.append(fig_turnover_vs_cvar(df, label))
    artifacts.append(fig_crisis_robustness(df, label))
    print("[OK] figures written:")
    for a in artifacts:
        if a is not None:
            print("  ", a)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    main(args.label)
