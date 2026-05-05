"""Forest plot of paired Delta% vs LSTM across (regime, task, architecture).

Loads benchmark_medium_v1.pkl, computes per-cell Delta% on CVaR_95 for each
non-LSTM architecture vs LSTM, runs a paired t-test across the three seeds,
applies Holm-Bonferroni correction within each (regime, task) family of
size 4, and plots a forest of the resulting Delta% with significance stars.
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PKL  = os.path.join(ROOT, "jaws_research/outputs/runs/benchmark_medium_v1.pkl")
OUT  = os.path.join(os.path.dirname(__file__), "paired_test_forest.pdf")

with open(PKL, "rb") as f:
    pkl = pickle.load(f)
res = pkl["results"]

REGIMES = [("normal_us","Normal US"), ("post_covid_us","Post-COVID US"),
           ("covid_us","COVID US"), ("gfc_2008","GFC 2008")]
TASKS = [("european_call","Euro call"), ("european_put","Euro put"),
         ("up_out_call","Up-out call"), ("down_in_put","Down-in put"),
         ("digital_call","Digital call"), ("asian_call","Asian call"),
         ("basket_call","Basket call")]
MODELS = [("Transformer","Transformer","#1f77b4"),
          ("WDRO_T","W-DRO-T","#d62728"),
          ("3SCH","3SCH","#2ca02c"),
          ("RSE","RSE","#9467bd")]

def holm(pvals):
    """Holm-Bonferroni at alpha=0.05 within a family."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    rejected = np.zeros(n, bool)
    for k, i in enumerate(order):
        if p[i] <= 0.05 / (n - k):
            rejected[i] = True
        else:
            break
    return rejected

# Build records per (regime, task)
families = []
for rk, rn in REGIMES:
    for tk, tn in TASKS:
        cell = res.get((rk, tk))
        if cell is None: continue
        m = cell["metrics"]
        if "LSTM" not in m: continue
        base = np.array([s["cvar_95"] for s in m["LSTM"]], float)
        rec = {"label": f"{rn} - {tn}", "entries": []}
        pvals = []
        for mk, mn, _ in MODELS:
            if mk not in m or len(m[mk]) != len(base):
                rec["entries"].append(None)
                pvals.append(1.0)
                continue
            arr = np.array([s["cvar_95"] for s in m[mk]], float)
            delta_pct = 100.0 * (arr - base).mean() / base.mean()
            t, p = stats.ttest_rel(arr, base)
            rec["entries"].append({"delta_pct": delta_pct, "p": float(p)})
            pvals.append(float(p))
        rec["sig"] = holm(pvals)
        families.append(rec)

n = len(families)
labels = [f["label"] for f in families]

plt.rcParams.update({"font.family":"serif","font.size":9,"savefig.bbox":"tight"})
fig, ax = plt.subplots(figsize=(7.0, 0.32 * n + 1.5))

ys = np.arange(n)
offsets = np.linspace(-0.28, 0.28, len(MODELS))
for j, (mk, mn, color) in enumerate(MODELS):
    xs, sigs = [], []
    for i, fam in enumerate(families):
        e = fam["entries"][j]
        if e is None:
            xs.append(np.nan); sigs.append(False); continue
        xs.append(e["delta_pct"])
        sigs.append(bool(fam["sig"][j]))
    xs = np.array(xs)
    yy = ys + offsets[j]
    ax.scatter(xs, yy, s=22, color=color, label=mn, zorder=3)
    # significance stars
    for x, y, s in zip(xs, yy, sigs):
        if not np.isnan(x) and s:
            ax.text(x, y, "  *", va="center", fontsize=10, color=color)

ax.axvline(0, color="black", linewidth=0.6)
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel(r"$\Delta\%$ in $\mathrm{CVaR}_{95}$ vs. LSTM (negative = better)")
ax.set_xlim(-25, 10)
ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=4)
ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.set_title(r"Paired comparison vs. LSTM ($*$ = Holm--Bonferroni significant at $\alpha=0.05$)")
plt.tight_layout()
fig.savefig(OUT)
print("Wrote", OUT)
