#!/usr/bin/env python3
"""Part 6: Economic/regulatory analysis figures."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, warnings
warnings.filterwarnings('ignore')

FIGDIR = os.path.dirname(os.path.abspath(__file__))
C = {'purple':'#7B1FA2','purple_bg':'#F3E5F5','blue':'#1565C0','blue_bg':'#E3F2FD',
     'orange':'#E65100','orange_bg':'#FFF3E0','red':'#C62828','red_bg':'#FFEBEE',
     'green':'#2E7D32','green_bg':'#E8F5E9','teal':'#00695C','gray':'#616161',
     'gray_l':'#BDBDBD','dark':'#212121','bg':'#FAFAFA'}

plt.rcParams.update({'font.size':12,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
    'figure.facecolor':'white','axes.facecolor':'white'})

def save(fig,name):
    fig.savefig(os.path.join(FIGDIR,name),bbox_inches='tight',pad_inches=0.15); plt.close(fig); print(f"  ✓ {name}")

def box(ax,x,y,w,h,txt,col,tc='white',fs=10,a=0.95):
    b=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.12",facecolor=col,edgecolor='#333',lw=1.5,alpha=a)
    ax.add_patch(b)
    for i,line in enumerate(txt.split('\n')):
        off=(len(txt.split('\n'))-1)*0.5*(fs/72)*2.2
        ax.text(x+w/2,y+h/2+off-i*(fs/72)*2.2,line,ha='center',va='center',fontsize=fs,fontweight='bold',color=tc)

def arr(ax,x1,y1,x2,y2,col='#333',lw=2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color=col,lw=lw))

# ═════════════════════════════════════════
# 29: Capital Requirement Analysis
# ═════════════════════════════════════════
print("\n[29] Capital requirement analysis...")
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# FRTB IMA capital charges
models = ['LSTM', 'Transformer', 'RSE', 'W-DRO-T', '3SCH']
colors = ['#9E9E9E', '#78909C', '#1565C0', '#7B1FA2', '#E65100']
# ES at 97.5% ≈ CVaR × scaling factor, supervisory multiplier = 1.5
cvar_vals = [3.215, 3.234, 3.109, 3.227, 3.219]
es975 = [cv * 1.12 for cv in cvar_vals]  # approximate ES 97.5 from CVaR 95
capital = [es * 1.5 for es in es975]  # supervisory multiplier

bars = axes[0].bar(models, capital, color=colors, edgecolor='white', lw=2, width=0.6, alpha=0.9)
for b, c in zip(bars, capital):
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                 f'{c:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Capital Charge (ES × multiplier)', fontweight='bold')
axes[0].set_title('FRTB IMA Capital Requirements', fontsize=13, fontweight='bold')
axes[0].set_ylim(top=max(capital)*1.08)
axes[0].axhline(min(capital), color=C['green'], ls='--', lw=1.5, alpha=0.5, label=f'Best: RSE ({min(capital):.2f})')
axes[0].legend(fontsize=10)

# Capital savings vs baseline
baseline = capital[0]  # LSTM as baseline
savings = [(baseline - c) / baseline * 100 for c in capital]
bars2 = axes[1].bar(models, savings, color=colors, edgecolor='white', lw=2, width=0.6, alpha=0.9)
for b, s in zip(bars2, savings):
    y_pos = max(b.get_height(), 0) + 0.3
    axes[1].text(b.get_x()+b.get_width()/2, y_pos,
                 f'{s:+.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Capital Saving vs LSTM (%)', fontweight='bold')
axes[1].set_title('Capital Savings Relative to LSTM Baseline', fontsize=13, fontweight='bold')
axes[1].axhline(0, color='black', lw=1, alpha=0.3)
axes[1].set_ylim(top=max(savings)*1.25)
fig.suptitle('Basel III/IV FRTB Capital Analysis', fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94]); save(fig, '29_capital_requirement_analysis.pdf')

# ═════════════════════════════════════════
# 30: Hedge Accounting Workflow
# ═════════════════════════════════════════
print("[30] Hedge accounting workflow...")
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')
ax.set_title('Hedge Accounting Workflow (IFRS 9 / Ind-AS 109)', fontsize=16, fontweight='bold', pad=15)

# Top row: Designation
box(ax, 0.3, 7, 3, 1.5, 'Hedge Designation\nDocument relationship\nRisk management objective', C['blue'], fs=9)
arr(ax, 3.3, 7.75, 4.5, 7.75)
box(ax, 4.5, 7, 3, 1.5, 'Effectiveness Testing\nProspective: regression\nRetrospective: dollar offset', C['purple'], fs=9)
arr(ax, 7.5, 7.75, 8.7, 7.75)
box(ax, 8.7, 7, 3, 1.5, 'Ongoing Assessment\n80-125% effectiveness\nQuarterly re-validation', C['orange'], fs=9)
arr(ax, 11.7, 7.75, 12.8, 7.75)
box(ax, 12.8, 7, 2.8, 1.5, 'Accounting Treatment\nFair value / Cash flow\nOCI / P&L recognition', C['green'], fs=9)

# Model-specific row
box(ax, 0.3, 4.5, 3, 2, 'LSTM / 3SCH\n✓ High interpretability\n✓ Simple documentation\n✓ Easy effectiveness test\nRecommended: Tier 1', C['green_bg'], tc=C['dark'], fs=9, a=0.85)
box(ax, 4, 4.5, 3, 2, 'RSE (Ensemble)\n△ Moderate complexity\n△ Regime justification needed\n△ Multiple sub-models\nRecommended: Tier 2', C['orange_bg'], tc=C['dark'], fs=9, a=0.85)
box(ax, 7.7, 4.5, 3.5, 2, 'W-DRO-T\n✗ High complexity\n✗ DRO regularization opaque\n✗ Harder to document\nRecommended: Tier 3', C['red_bg'], tc=C['dark'], fs=9, a=0.85)
box(ax, 11.8, 4.5, 3.8, 2, 'De-designation Trigger\n• Effectiveness < 80%\n• Model retrained\n• Regime shift detected\n• Material model change', C['purple_bg'], tc=C['dark'], fs=9, a=0.85)

# Bottom: Regulatory framework
box(ax, 0.3, 1.5, 15.3, 2.5, '', C['bg'], a=0.5)
ax.text(8, 3.5, 'Regulatory Framework Mapping', fontsize=13, fontweight='bold', ha='center', color=C['dark'])
regs = [('IFRS 9 / Ind-AS 109', 'Hedge accounting standard'), ('FRTB IMA', 'ES 97.5% capital charge'),
        ('SR 11-7 / SS1/23', 'Model risk governance'), ('IFRS 7', 'Risk disclosure')]
for i, (reg, desc) in enumerate(regs):
    x_pos = 1 + i * 3.8
    ax.text(x_pos, 2.5, reg, fontsize=11, fontweight='bold', color=C['purple'])
    ax.text(x_pos, 2.0, desc, fontsize=9, color=C['gray'])
save(fig, '30_hedge_accounting_workflow.pdf')

# ═════════════════════════════════════════
# 31: Risk Governance Framework
# ═════════════════════════════════════════
print("[31] Risk governance framework...")
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title('Model Risk Governance Framework (SR 11-7 / SS1/23)', fontsize=16, fontweight='bold', pad=15)

# Four pillars
pillars = [
    ('Pillar 1:\nModel Development', C['blue'],
     'Architecture selection\nLoss function design\nData quality checks\nFeature engineering\nHyperparameter tuning'),
    ('Pillar 2:\nModel Validation', C['purple'],
     'Independent testing\nOut-of-sample backtest\nStress testing (GFC, COVID)\nBenchmark comparison\n10-seed stability'),
    ('Pillar 3:\nModel Monitoring', C['orange'],
     'Real-time P&L tracking\nRegime shift detection\nDrift monitoring\nEffectiveness testing\nPerformance alerts'),
    ('Pillar 4:\nModel Governance', C['red'],
     'Documentation standards\nApproval workflows\nRetraining schedule\nRollback procedures\nAudit trail'),
]
for i, (title, color, details) in enumerate(pillars):
    x = 0.3 + i * 3.9
    box(ax, x, 7, 3.4, 2, title, color, fs=11)
    box(ax, x, 3.5, 3.4, 3.2, details, color + '22', tc=C['dark'], fs=9, a=0.85)

# Arrows from pillars down
for i in range(4):
    x = 0.3 + i * 3.9 + 1.7
    arr(ax, x, 7, x, 6.7)

# Bottom: Deployment tiers
box(ax, 0.3, 0.5, 15.3, 2.5, '', C['bg'], a=0.5)
ax.text(8, 2.5, 'Tiered Deployment Strategy', fontsize=13, fontweight='bold', ha='center', color=C['dark'])
tiers = [
    ('Tier 1: Shadow', 'LSTM/3SCH alongside BS delta\nMonitor for 6 months', C['green']),
    ('Tier 2: Limited', 'RSE on subset of book\nDaily reconciliation', C['orange']),
    ('Tier 3: Full', 'W-DRO-T after Tier 2 pass\nContinuous monitoring', C['red']),
]
for i, (tier, desc, color) in enumerate(tiers):
    x = 1 + i * 5
    ax.text(x, 1.9, tier, fontsize=11, fontweight='bold', color=color)
    ax.text(x, 1.1, desc, fontsize=9, color=C['gray'])
save(fig, '31_risk_governance_framework.pdf')

# ═════════════════════════════════════════
# 32: Pareto Frontier: Risk vs Cost
# ═════════════════════════════════════════
print("[32] Pareto frontier...")
fig, ax = plt.subplots(figsize=(10, 7))
models_all = ['LSTM', 'Transformer', 'RSE', 'W-DRO-T', '3SCH']
cvar = [3.215, 3.234, 3.109, 3.227, 3.219]
train_time = [273/60, 7534/60, 924/60, 7119/60, 295/60]  # minutes
colors_all = ['#9E9E9E', '#78909C', '#1565C0', '#7B1FA2', '#E65100']

for m, cv, tt, clr in zip(models_all, cvar, train_time, colors_all):
    ax.scatter(tt, cv, s=200, color=clr, edgecolor='white', linewidth=2, zorder=5)
    ax.annotate(m, (tt, cv), textcoords="offset points", xytext=(10, 8),
                fontsize=12, fontweight='bold', color=clr)

# Pareto frontier line (connect non-dominated points)
pareto_idx = [0, 4, 2]  # LSTM, 3SCH, RSE (sorted by time)
px = [train_time[i] for i in pareto_idx]
py = [cvar[i] for i in pareto_idx]
ax.plot(px, py, '--', lw=2, color=C['green'], alpha=0.5, label='Pareto frontier')

ax.set_xlabel('Training Time (minutes)', fontsize=13, fontweight='bold')
ax.set_ylabel('CVaR$_{0.95}$ (lower is better)', fontsize=13, fontweight='bold')
ax.set_title('Pareto Frontier: Risk vs Computational Cost', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.invert_yaxis()
save(fig, '32_pareto_frontier_risk_vs_cost.pdf')

# ═════════════════════════════════════════
# 33: Model Interpretability Trade-off
# ═════════════════════════════════════════
print("[33] Model interpretability trade-off...")
fig, ax = plt.subplots(figsize=(10, 7))
models_all = ['LSTM', 'Transformer', 'RSE', 'W-DRO-T', '3SCH']
interpretability = [0.85, 0.5, 0.65, 0.35, 0.80]  # subjective scores
performance = [3.215, 3.234, 3.109, 3.227, 3.219]  # CVaR (lower=better)
perf_norm = [1 - (p - min(performance)) / (max(performance) - min(performance)) for p in performance]  # normalized 0-1

for m, interp, perf, clr in zip(models_all, interpretability, perf_norm, colors_all):
    ax.scatter(interp, perf, s=250, color=clr, edgecolor='white', linewidth=2, zorder=5)
    ax.annotate(m, (interp, perf), textcoords="offset points", xytext=(10, 8),
                fontsize=12, fontweight='bold', color=clr)

ax.set_xlabel('Interpretability Score', fontsize=13, fontweight='bold')
ax.set_ylabel('Normalized Performance (higher=better)', fontsize=13, fontweight='bold')
ax.set_title('Interpretability vs Performance Trade-off', fontsize=14, fontweight='bold')
ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.1, 1.1)
# Ideal quadrant
ax.axhspan(0.5, 1.1, xmin=0.5, xmax=1.0, alpha=0.08, color=C['green'])
ax.text(0.85, 0.95, 'Ideal Zone', fontsize=10, color=C['green'], fontweight='bold', ha='center')
save(fig, '33_interpretability_vs_performance.pdf')

print("\n✅ Part 6 complete: 5 figures generated.")
