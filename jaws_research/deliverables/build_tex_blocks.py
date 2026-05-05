"""Refresh auto-generated LaTeX result blocks from saved pickles.

Reads:
    outputs/runs/benchmark_medium_v1.pkl  (or any --label)
    outputs/runs/real_real_v1.pkl

Writes:
    deliverables/auto_results_block.tex
    deliverables/auto_real_block.tex

Run as a module:
    python -m jaws_research.deliverables.build_tex_blocks
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jaws_research.experiments.analyze_results import metrics_to_dataframe  # noqa: E402

DEL = ROOT / "jaws_research" / "deliverables"
RUNS = ROOT / "jaws_research" / "outputs" / "runs"


def _fmt(value, std=None):
    if std is None:
        return f"{value:.3f}"
    return f"{value:.3f}\\,$\\pm$\\,{std:.3f}"


def build_results_block(label: str = "medium_v1") -> str:
    pkl = RUNS / f"benchmark_{label}.pkl"
    if not pkl.exists():
        return _placeholder_results_block(label)
    with open(pkl, "rb") as fh:
        blob = pickle.load(fh)
    df = metrics_to_dataframe(blob)
    pivot = df.groupby(["scenario", "task", "model"])["cvar_95"].agg(
        ["mean", "std", "count"]).reset_index()
    pivot["count"] = pivot["count"].astype(int)
    n_seeds = pivot["count"].max()

    scenarios = list(dict.fromkeys(df["scenario"].tolist()))
    tasks = list(dict.fromkeys(df["task"].tolist()))
    models = ["LSTM", "Transformer", "WDRO_T", "3SCH", "RSE"]

    lines = []
    lines.append(r"\begin{table}[h]\centering\scriptsize")
    lines.append(rf"\caption{{Medium benchmark CVaR$_{{95}}$ "
                  rf"(mean $\pm$ std across $n={n_seeds}$ seeds, lower is better).  "
                  rf"\textbf{{Bold}} = best in row.}}")
    lines.append(r"\label{tab:medium_results}")
    lines.append(r"\begin{tabular}{ll" + "c" * len(models) + "}")
    lines.append(r"\toprule")
    models_tex = [m.replace("_", "-") for m in models]
    lines.append("Regime & Task & " + " & ".join(models_tex) + r" \\")
    lines.append(r"\midrule")
    for sc in scenarios:
        for tk in tasks:
            row_means = {}
            for m in models:
                cell = pivot[(pivot["scenario"] == sc) &
                             (pivot["task"] == tk) &
                             (pivot["model"] == m)]
                if cell.empty:
                    row_means[m] = (None, None)
                else:
                    row_means[m] = (float(cell["mean"].iloc[0]),
                                     float(cell["std"].iloc[0] or 0.0))
            valid_means = {m: v[0] for m, v in row_means.items() if v[0] is not None}
            if not valid_means:
                continue
            best_model = min(valid_means, key=valid_means.get)
            cells = []
            for m in models:
                v, s = row_means[m]
                if v is None:
                    cells.append("--")
                elif m == best_model:
                    cells.append(rf"\textbf{{{_fmt(v, s)}}}")
                else:
                    cells.append(_fmt(v, s))
            lines.append(f"{sc.replace('_', ' ')} & {tk.replace('_', ' ')} & "
                          + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    # Add a paired-test table
    paired_rows = []
    for (sc, tk), grp in df.groupby(["scenario", "task"]):
        bsub = grp[grp["model"] == "LSTM"].sort_values("seed_idx")["cvar_95"].values
        for m in models:
            if m == "LSTM":
                continue
            msub = grp[grp["model"] == m].sort_values("seed_idx")["cvar_95"].values
            if len(bsub) < 2 or len(bsub) != len(msub):
                continue
            d = msub - bsub
            mean = d.mean(); sd = d.std(ddof=1) if len(d) > 1 else 0.0
            from scipy.stats import t as tdist
            if sd == 0:
                p = 1.0; cd = 0.0
            else:
                tstat = mean / (sd / np.sqrt(len(d)))
                p = float(2 * (1 - tdist.cdf(abs(tstat), df=len(d) - 1)))
                cd = float(mean / sd)
            paired_rows.append({"scenario": sc, "task": tk, "model": m,
                                 "delta_pct": 100 * mean / max(abs(bsub.mean()), 1e-9),
                                 "p_value": p, "cohen_d": cd})
    if paired_rows:
        pdf = pd.DataFrame(paired_rows)
        lines.append("")
        lines.append(r"\begin{table}[h]\centering\scriptsize")
        lines.append(r"\caption{Paired comparison vs.\ LSTM (Holm--Bonferroni-style raw $p$-values; full corrected table in the appendix).}")
        lines.append(r"\label{tab:paired_medium}")
        lines.append(r"\begin{tabular}{lllrrr}")
        lines.append(r"\toprule")
        lines.append(r"Regime & Task & Model & $\Delta\%$ & $p$ & Cohen's $d$ \\\midrule")
        for _, r in pdf.iterrows():
            mname = r['model'].replace('_','-')
            lines.append(f"{r['scenario'].replace('_',' ')} & {r['task'].replace('_',' ')} "
                          f"& {mname} & {r['delta_pct']:+.1f} & "
                          f"{r['p_value']:.3f} & {r['cohen_d']:+.2f} \\\\")
        lines.append(r"\bottomrule\end{tabular}\end{table}")

    return "\n".join(lines) + "\n"


def _placeholder_results_block(label: str) -> str:
    return (rf"% Placeholder: medium pickle not found at outputs/runs/"
            rf"benchmark_{label}.pkl"
            "\n\\textit{The medium-scale ($5$-seed) benchmark is in progress.  "
            "When complete, run \\texttt{python -m jaws\\_research.deliverables."
            "build\\_tex\\_blocks} to refresh this section with the live numbers.}\n")


def build_real_block(label: str = "real_v1") -> str:
    pkl = RUNS / f"real_{label}.pkl"
    if not pkl.exists():
        return _placeholder_real_block(label)
    with open(pkl, "rb") as fh:
        blob = pickle.load(fh)

    rows = []
    for (key, task), entry in blob["results"].items():
        ticker, window = key.split(":")
        for model, m_list in entry["metrics"].items():
            cvar = float(np.mean([m["cvar_95"] for m in m_list]))
            rows.append({"ticker": ticker, "window": window, "task": task,
                         "model": model, "cvar_95": cvar})
    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return _placeholder_real_block(label)

    pivot = rdf.pivot_table(index=["ticker", "window", "task"],
                             columns="model", values="cvar_95")
    models = ["LSTM", "Transformer", "WDRO_T", "3SCH", "RSE"]
    lines = []
    lines.append(r"\begin{table}[h]\centering\scriptsize")
    lines.append(r"\caption{Real-market crisis window CVaR$_{95}$ (lower better).  Source: \texttt{yfinance}.}")
    lines.append(r"\label{tab:real_results}")
    lines.append(r"\begin{tabular}{lllccccc}\toprule")
    lines.append(r"Ticker & Window & Task & " + " & ".join([m.replace("_","-") for m in models]) + r" \\\midrule")
    for (tk, w, t), row in pivot.iterrows():
        cells = [f"{row.get(m, np.nan):.2f}" if not pd.isna(row.get(m, np.nan)) else "--" for m in models]
        lines.append(f"{tk} & {w.replace('_',' ')} & {t.replace('_',' ')} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    return "\n".join(lines) + "\n"


def _placeholder_real_block(label: str) -> str:
    return (rf"% Placeholder: real-data pickle not found at outputs/runs/real_{label}.pkl"
            "\n\\textit{The real-market crisis-window benchmark is in progress.  "
            "When complete, run \\texttt{python -m jaws\\_research.deliverables."
            "build\\_tex\\_blocks} to refresh this section.}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--medium", default="medium_v1")
    parser.add_argument("--real", default="real_v1")
    args = parser.parse_args()

    (DEL / "auto_results_block.tex").write_text(build_results_block(args.medium))
    (DEL / "auto_real_block.tex").write_text(build_real_block(args.real))
    print(f"Updated auto blocks in {DEL}")


if __name__ == "__main__":
    main()
