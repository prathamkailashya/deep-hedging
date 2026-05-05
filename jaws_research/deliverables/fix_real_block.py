"""Re-slice the real-data pickle into proper per-window cells.

The earlier run_real_data.py accumulated seed_metrics across all crisis
windows for the same (ticker, task), so every (ticker, window, task)
key currently holds 10 entries = 5 windows x 2 seeds in window-major,
seed-minor order.  We split them back here so each window cell shows its
own metrics.

Writes:
    jaws_research/deliverables/auto_real_block.tex
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEL = ROOT / "jaws_research" / "deliverables"
RUNS = ROOT / "jaws_research" / "outputs" / "runs"


WINDOW_ORDER = ["calm_2019", "covid_acute", "vol_q4_2018",
                "inflation_2022", "banking_2023"]
N_SEEDS = 2


def main(label: str = "real_v1"):
    pkl = RUNS / f"real_{label}.pkl"
    with open(pkl, "rb") as fh:
        blob = pickle.load(fh)

    # Group entries by (ticker, task)
    grouped = {}
    for (key, task), entry in blob["results"].items():
        if ":" not in key:
            continue
        ticker, _ = key.split(":", 1)
        grouped.setdefault((ticker, task), entry)

    rows = []
    for (ticker, task), entry in grouped.items():
        for model, m_list in entry["metrics"].items():
            # Each m_list contains 5 windows x 2 seeds = 10 entries (window-major)
            n_per = N_SEEDS
            for wi, window in enumerate(WINDOW_ORDER):
                lo, hi = wi * n_per, (wi + 1) * n_per
                slice_ = m_list[lo:hi]
                if not slice_:
                    continue
                rows.append({
                    "ticker": ticker,
                    "window": window,
                    "task": task,
                    "model": model,
                    "cvar_95": float(np.mean([m["cvar_95"] for m in slice_])),
                    "cvar_95_std": float(np.std([m["cvar_95"] for m in slice_], ddof=1)) if len(slice_) > 1 else 0.0,
                })

    import pandas as pd
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["ticker", "window", "task"],
                            columns="model", values="cvar_95")
    models = ["LSTM", "Transformer", "WDRO_T", "3SCH", "RSE"]
    models_tex = [m.replace("_", "-") for m in models]

    lines = []
    lines.append(r"\begin{table}[H]\centering\scriptsize")
    lines.append(r"\caption{Walk-forward real-market $\cvar_{95}$ on \texttt{yfinance} OHLCV across crisis windows. Models trained on the calm 2017-2019 calm window and evaluated on each crisis window of Table~\ref{tab:crisis_windows}.  Lower is better.}")
    lines.append(r"\label{tab:real_results}")
    lines.append(r"\begin{tabular}{lll" + "c" * len(models) + "}")
    lines.append(r"\toprule")
    lines.append(r"Ticker & Window & Task & " + " & ".join(models_tex) + r" \\")
    lines.append(r"\midrule")

    last_ticker = None
    last_window = None
    for (tk, w, t), row in pivot.iterrows():
        cells = []
        for m in models:
            v = row.get(m, np.nan)
            cells.append(f"{v:.2f}" if not pd.isna(v) else "--")
        # Mark the best (lowest)
        try:
            valid = {m: row[m] for m in models if not pd.isna(row.get(m, np.nan))}
            best = min(valid, key=valid.get)
            best_idx = models.index(best)
            cells[best_idx] = rf"\textbf{{{cells[best_idx]}}}"
        except Exception:
            pass
        # Group ticker rows
        ticker_str = tk if tk != last_ticker else ""
        window_str = w.replace("_", " ") if (tk, w) != (last_ticker, last_window) else ""
        if ticker_str and last_ticker is not None:
            lines.append(r"\midrule")
        lines.append(f"{ticker_str} & {window_str} & {t.replace('_', ' ')} & "
                      + " & ".join(cells) + r" \\")
        last_ticker = tk; last_window = w
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = "\n".join(lines) + "\n"
    (DEL / "auto_real_block.tex").write_text(out)
    print(f"Wrote {DEL / 'auto_real_block.tex'} with {len(pivot)} rows")


if __name__ == "__main__":
    main()
