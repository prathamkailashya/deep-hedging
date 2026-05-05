"""
Generate the two figures missing from combined_report.tex:
  1. capital_requirement_comparison.pdf  - Basel-IV capital per $100M (Table 9.1)
  2. effect_size_heatmap.pdf              - Cohen's d vs LSTM across (regime, task, model)
                                            from Appendix Table tab:paired_medium_full

Numbers are taken verbatim from the thesis tables so the plots and the prose stay
consistent.  No external data is used.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# 1. Capital requirement comparison (Table 9.1, SPY normal scenario)
# ---------------------------------------------------------------------------
models  = ["W-DRO-T", "Transformer", "3SCH", "RSE", "LSTM"]
capital = [4.79, 5.79, 6.32, 6.76, 8.11]      # $ millions per $100M notional
savings = [(c - 8.11) / 8.11 * 100 for c in capital]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))

colors = ["#2b6cb0", "#4a90c2", "#7aa6c9", "#a9bcd0", "#cfd6dd"]
bars = ax1.bar(models, capital, color=colors, edgecolor="black", linewidth=0.6)
ax1.set_ylabel(r"Capital per \$100\,M notional (\$M)")
ax1.set_title("Basel-IV capital, SPY normal scenario")
for b, c in zip(bars, capital):
    ax1.text(b.get_x() + b.get_width() / 2, c + 0.12, f"\\${c:.2f}M",
             ha="center", va="bottom", fontsize=9)
ax1.set_ylim(0, max(capital) * 1.18)

bars2 = ax2.bar(models[:-1], savings[:-1],
                color=["#2b6cb0", "#4a90c2", "#7aa6c9", "#a9bcd0"],
                edgecolor="black", linewidth=0.6)
ax2.axhline(0, color="black", linewidth=0.6)
ax2.set_ylabel("Capital saving vs.\\ LSTM (\\%)")
ax2.set_title("Relative capital saving")
for b, s in zip(bars2, savings[:-1]):
    ax2.text(b.get_x() + b.get_width() / 2, s - 1.2, f"{s:+.1f}\\%",
             ha="center", va="top", fontsize=9)
ax2.set_ylim(min(savings) * 1.15, 5)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "capital_requirement_comparison.pdf"))
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Effect-size heatmap (Cohen's d vs LSTM, COVID & GFC stress regimes)
#    Numbers taken from Appendix Table tab:paired_medium_full
# ---------------------------------------------------------------------------
tasks = ["European call", "European put", "Up-out call",
         "Down-in put", "Digital call", "Asian call", "Basket call"]
models_h = ["Transformer", "WDRO-T", "3SCH", "RSE"]

# rows = regime x task (4*7 = 28 rows), cols = model.
# A NaN entry means the cell is not reported in the truncated appendix table
# (basket-call rows are absent for some regimes).
nan = np.nan
covid_us = [
    [-4.18, -4.40, +1.79, +1.59],   # european call
    [-3.58, -4.23, +0.61, -0.54],   # european put
    [-5.89, -5.44, +1.48, -2.87],   # up out call
    [-10.73, -14.70, +1.82, -1.19], # down in put
    [-13.91, -14.60, -0.20, -12.69],# digital call
    [-6.02, -19.11, +2.23, -5.39],  # asian call
    [-3.68,  -9.45, +1.07, -4.44],  # basket call
]
gfc_2008 = [
    [-5.98, -9.37, +7.76, +4.73],
    [-5.85, -9.19, +2.30, -0.06],
    [-4.91, -5.98, +1.13, -3.79],
    [-7.58, -6.53, +1.72, -0.40],
    [-10.08, -12.69, -0.35, -8.71],
    [-5.15, -34.16, +1.75, -17.47],
    [   nan,   nan,   nan,   nan],
]
normal_us = [
    [-4.90, -3.27, +2.51, -5.13],
    [-2.57, -2.54, +0.89, +0.20],
    [-3.00, -2.38, +4.79, -1.92],
    [+0.45, +0.74, +0.19, +0.37],
    [-12.68, -8.96, +0.50, -11.01],
    [-6.89, -8.64, +0.59, -16.50],
    [+0.92, +1.53, +4.05, -3.47],
]
post_covid_us = [
    [-3.41, -2.63, +3.11, -2.25],
    [-3.08, -2.54, +1.50, -0.00],
    [-3.41, -2.63, +3.11, -2.25],
    [+0.55, +0.54, -0.92, +1.09],
    [-112.22, -23.61, +0.02, -9.71],
    [-4.58, -8.28, +0.85, -6.07],
    [   nan,   nan,   nan,   nan],
]

regimes = [("Normal US", normal_us),
           ("Post-COVID US", post_covid_us),
           ("COVID US", covid_us),
           ("GFC 2008", gfc_2008)]

# stack into a single matrix (28 x 4)
matrix = np.vstack([np.asarray(d, dtype=float) for _, d in regimes])
row_labels = []
for name, _ in regimes:
    for t in tasks:
        row_labels.append(f"{name} - {t}")

# clip the heatmap so a single huge negative (-112) does not flatten the rest
vmax = 8.0
clipped = np.clip(matrix, -vmax, vmax)

fig, ax = plt.subplots(figsize=(6.6, 9.2))
im = ax.imshow(clipped, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(models_h)))
ax.set_xticklabels(models_h)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=8)
ax.set_title("Cohen's $d$ vs.\\ LSTM (clipped to $\\pm 8$; blue = better, red = worse)")

# annotate every cell with the raw (unclipped) value
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        v = matrix[i, j]
        if np.isnan(v):
            ax.text(j, i, "-", ha="center", va="center", fontsize=7, color="grey")
        else:
            txt = f"{v:.1f}"
            colr = "white" if abs(clipped[i, j]) > 4 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=colr)

# horizontal separators between regime blocks
for k in range(1, 4):
    ax.axhline(7 * k - 0.5, color="black", linewidth=0.7)

cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("Cohen's $d$ (clipped)")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "effect_size_heatmap.pdf"))
plt.close(fig)

print("Wrote:")
for f in ("capital_requirement_comparison.pdf", "effect_size_heatmap.pdf"):
    print(" ", os.path.join(OUT, f))
