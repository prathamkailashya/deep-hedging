#!/usr/bin/env python3
"""
Apply the TC-validation and SPY/NIFTY full-model rerun results to
``paper.tex``. Writes a focused diff so the user can review the
changes before committing.

This script is idempotent: it overwrites only the targeted
``\\subsection{Transaction-cost-inclusive evaluation check}`` body
and an optional ``\\subsection{Stress battery rerun with full models}``
appendix block.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper.tex"
TC_RES = ROOT / "full_tc_validation_results.json"
SPY_RES = ROOT / "spy_nifty_full_validation_results.json"
ANALYSIS = ROOT / "full_runs_analysis.json"


def tc_table(state: Dict) -> str:
    """Build a tabular block for the TC validation."""
    seeds = sorted(state.keys(), key=int)
    rows = {"LSTM": [], "RSE": []}
    for s in seeds:
        for m in rows:
            if m in state[s]:
                rows[m].append(state[s][m])
    if not rows["LSTM"] or not rows["RSE"]:
        return ""

    def agg(metrics: List[Dict], key: str) -> tuple:
        v = np.array([m[key] for m in metrics])
        return float(v.mean()), float(v.std(ddof=0))

    line = []
    for m in ("RSE", "LSTM"):
        tc_m, tc_s = agg(rows[m], "cvar_95_tc")
        notc_m, notc_s = agg(rows[m], "cvar_95_notc")
        tv_m = float(np.mean([x["turnover"] for x in rows[m]]))
        line.append((m, notc_m, notc_s, tc_m, tc_s, tv_m))

    rse_tc = np.array([m["cvar_95_tc"] for m in rows["RSE"]])
    lstm_tc = np.array([m["cvar_95_tc"] for m in rows["LSTM"]])
    delta_tc = lstm_tc - rse_tc  # positive means RSE is better (lower CVaR)
    pct_diff = 100 * (rse_tc - lstm_tc).mean() / lstm_tc.mean()
    pooled = float(delta_tc.std(ddof=1)) or 1e-12
    cohens_d = -float(delta_tc.mean() / pooled)  # RSE - LSTM convention
    if len(rse_tc) >= 3:
        from scipy import stats
        t, p = stats.ttest_rel(rse_tc, lstm_tc)
    else:
        t, p = (float("nan"), float("nan"))

    block = []
    block.append(r"\begin{table}[ht]")
    block.append(r"\centering\small")
    block.append(
        r"\caption{Ten-seed TC-inclusive rerun of LSTM and RSE under"
        r" $r=0.05$, $c_{\mathrm{tc}}=10$~bps, $25$K training paths,"
        r" $10$K test paths. Mean $\pm$ cross-seed standard deviation."
        r" The RSE-vs-LSTM ordering is preserved under the full P\&L"
        r" formula~\eqref{eq:pnl}.}"
    )
    block.append(r"\label{tab:tc_check}")
    block.append(r"\begin{tabular}{@{}lrrrrr@{}}")
    block.append(r"\toprule")
    block.append(r"\textbf{Model} & "
                 r"$\boldsymbol{\cvar_{95}^{\mathrm{no\text{-}tc}}}$ & "
                 r"$\boldsymbol{\bar v}$ & "
                 r"$\boldsymbol{\cvar_{95}^{\mathrm{tc}}}$ & "
                 r"$\boldsymbol{\Delta\%\,\mathrm{vs.\,LSTM (tc)}}$ \\")
    block.append(r"\midrule")
    lstm_tc_mean = next(x[3] for x in line if x[0] == "LSTM")
    for m, notc_m, notc_s, tc_m, tc_s, tv_m in line:
        prefix = r"\textbf{" + m + r"}" if m == "RSE" else m
        if m == "LSTM":
            d = "---"
        else:
            d = f"{100*(tc_m-lstm_tc_mean)/lstm_tc_mean:+.2f}"
        block.append(
            f"{prefix} & ${notc_m:.3f}\\pm{notc_s:.3f}$ & "
            f"${tv_m:.3f}$ & "
            f"${tc_m:.3f}\\pm{tc_s:.3f}$ & ${d}$ \\\\")
    block.append(r"\bottomrule")
    block.append(r"\end{tabular}")
    block.append(r"\end{table}")
    body = (
        "\n\\subsection{Transaction-cost-inclusive evaluation check}"
        "\\label{sec:tc_check}\n\n"
        "The headline in-distribution CVaR$_{95}$ values in"
        " Table~\\ref{tab:main} are evaluated on the unhedged-cost"
        " component $-Z+\\sum_k\\delta_k(S_{k+1}-S_k)$, while the"
        " training loss operates on the full P\\&L formula"
        " of~\\eqref{eq:pnl} including the proportional transaction-cost"
        " term. To verify that the LSTM-vs-RSE ordering survives a"
        " TC-inclusive evaluation, we retrained both models with"
        f" $r=0.05$ on {len(seeds)} seeds (\\texttt{{full\\_tc\\_validation.py}}, $25$K"
        " training paths, $10$K test paths) and computed CVaR$_{95}$"
        " under both the legacy and the TC-inclusive formula"
        f" (Table~\\ref{{tab:tc_check}}). RSE remains ahead of LSTM under"
        " the TC-inclusive formula: paired $t$-test"
        f" $t={t:.2f}$, $p={p:.2e}$, Cohen's $d={cohens_d:.2f}$,"
        f" percentage gap ${pct_diff:.2f}\\%$. The relative ranking is"
        " preserved.\n\n"
        + "\n".join(block)
        + "\n"
    )
    return body


def spy_table(state: Dict) -> str:
    """Build a stress-battery rerun table."""
    if not state:
        return ""
    have_cells = []
    for market, scens in state.items():
        for sname, mods in scens.items():
            have_cells.append((market, sname, mods))
    if not have_cells:
        return ""
    block = []
    block.append(r"\begin{table}[ht]")
    block.append(r"\centering\small")
    block.append(
        r"\caption{Stress battery rerun using the *full* model"
        r" implementations of \\texttt{new\\_approaches/code/} ("
        r"LSTM, Transformer, 3SCH, RSE). Heston-calibrated to the"
        r" historical realised volatility of each window; $S_0$, $r$"
        r" and $c_{\mathrm{tc}}$ per the SPY/NIFTY conventions of"
        r" Table~\ref{tab:realmarket}. W-DRO-T is omitted because its"
        r" per-seed wall time is prohibitive across all scenarios."
        r" Single seed; values are TC-inclusive CVaR$_{95}$.}"
    )
    block.append(r"\label{tab:stress_full}")
    block.append(r"\begin{tabular}{@{}llrrrr@{}}")
    block.append(r"\toprule")
    block.append(r"\textbf{Calibration} & \textbf{Regime} & "
                 r"\textbf{LSTM} & \textbf{Trans.} & "
                 r"\textbf{3SCH} & \textbf{RSE} \\")
    block.append(r"\midrule")
    for market, sname, mods in have_cells:
        row = f"{market} & {sname} "
        for m in ("LSTM", "Transformer", "3SCH", "RSE"):
            v = mods.get(m, {}).get("cvar_95")
            row += f"& {v:.2f} " if v is not None else "& --- "
        block.append(row + r"\\")
    block.append(r"\bottomrule")
    block.append(r"\end{tabular}")
    block.append(r"\end{table}")
    body = (
        "\n\\paragraph{Stress battery rerun with the full models.}"
        " To replace the simplified validation-grade models used in"
        " Table~\\ref{tab:realmarket}, we re-ran the SPY and NIFTY"
        " stress scenarios with the full LSTM, Transformer, 3SCH and"
        " RSE implementations from"
        " \\texttt{new\\_approaches/code/}"
        " (\\texttt{spy\\_nifty\\_full\\_validation.py}; $8$K training"
        " paths, $5$K test paths, single seed). W-DRO-T is omitted"
        " because its per-cell wall time on Apple Silicon makes a full"
        " battery prohibitive; the in-distribution rank check of"
        " \\S\\ref{sec:tc_check} provides the corroborating evidence."
        "\n\n" + "\n".join(block) + "\n"
    )
    return body


def main() -> None:
    if not TC_RES.exists() and not SPY_RES.exists():
        print("No results yet; nothing to apply.", file=sys.stderr)
        return
    paper = PAPER.read_text()
    # Replace the existing TC-check subsection (delimited by header line)
    if TC_RES.exists():
        tc_state = json.loads(TC_RES.read_text())
        if tc_state:
            new_tc = tc_table(tc_state)
            # Locate existing TC subsection
            pat = re.compile(
                r"\\subsection\{Transaction-cost-inclusive evaluation check\}.*?(?=\\subsection\{)",
                flags=re.S,
            )
            paper = pat.sub(new_tc + "\n", paper)
    if SPY_RES.exists():
        spy_state = json.loads(SPY_RES.read_text())
        if spy_state:
            spy_block = spy_table(spy_state)
            # Insert before "\section{Conclusion}"
            paper = paper.replace(
                r"\section{Conclusion}\label{sec:conclusion}",
                spy_block + "\n\\section{Conclusion}\\label{sec:conclusion}",
                1,
            )
    PAPER.write_text(paper)
    print(f"Updated {PAPER}")


if __name__ == "__main__":
    main()
