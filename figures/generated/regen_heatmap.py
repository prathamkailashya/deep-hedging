"""Regenerate effect_size_heatmap.pdf from benchmark_medium_v1.pkl.

Cohen's d is computed per (regime, task, architecture) on CVaR_95 across
the three seeds, against the LSTM baseline.  Negative d means the model
beats LSTM (lower CVaR is better); the colormap is inverted accordingly so
that blue = better, red = worse.
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PKL  = os.path.join(ROOT, "jaws_research/outputs/runs/benchmark_medium_v1.pkl")
OUT  = os.path.join(os.path.dirname(__file__), "effect_size_heatmap.pdf")

with open(PKL, "rb") as f:
    pkl = pickle.load(f)
res = pkl["results"]

REGIMES = [("normal_us","Normal US"), ("post_covid_us","Post-COVID US"),
           ("covid_us","COVID US"), ("gfc_2008","GFC 2008")]
TASKS = [("european_call","European call"), ("european_put","European put"),
         ("up_out_call","Up-out call"), ("down_in_put","Down-in put"),
         ("digital_call","Digital call"), ("asian_call","Asian call"),
         ("basket_call","Basket call")]
MODELS = [("Transformer","Transformer"), ("WDRO_T","W-DRO-T"),
          ("3SCH","3SCH"), ("RSE","RSE")]

def cohend(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    return diff.mean() / sd if sd else np.nan

rows = [(rk, rn, tk, tn) for rk, rn in REGIMES for tk, tn in TASKS]
mat = np.full((len(rows), len(MODELS)), np.nan)
for i, (rk, rn, tk, tn) in enumerate(rows):
    cell = res.get((rk, tk))
    if cell is None: continue
    m = cell["metrics"]
    if "LSTM" not in m: continue
    base = [s["cvar_95"] for s in m["LSTM"]]
    for j, (mk, _) in enumerate(MODELS):
        if mk in m and len(m[mk]) == len(base) and len(base) > 1:
            mat[i, j] = cohend([s["cvar_95"] for s in m[mk]], base)

plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.bbox":"tight"})
vmax = 8.0
clipped = np.clip(mat, -vmax, vmax)

fig, ax = plt.subplots(figsize=(6.4, 8.6))
im = ax.imshow(clipped, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([m[1] for m in MODELS])
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{r[1]} -- {r[3]}" for r in rows], fontsize=7.5)
ax.set_title(r"Cohen's $d$ vs. LSTM on $\mathrm{CVaR}_{95}$ (clipped to $\pm 8$;"
             "\n" r"blue $=$ better, red $=$ worse)")

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        if np.isnan(v):
            ax.text(j, i, "--", ha="center", va="center", fontsize=7, color="grey")
        else:
            colr = "white" if abs(clipped[i, j]) > 4 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color=colr)

for k in range(1, 4):
    ax.axhline(7 * k - 0.5, color="black", linewidth=0.7)
ax.set_xlabel("Architecture")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Cohen's $d$ (clipped)")
plt.tight_layout()
fig.savefig(OUT)
print("Wrote", OUT)
