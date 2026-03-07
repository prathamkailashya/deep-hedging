#!/usr/bin/env python3
"""Part 4: Risk metric visualizations + statistical analysis."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, warnings
warnings.filterwarnings('ignore')

FIGDIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(FIGDIR), 'new_approaches', 'results')
C = {'purple':'#7B1FA2','blue':'#1565C0','orange':'#E65100','red':'#C62828',
     'green':'#2E7D32','teal':'#00695C','gray':'#616161','dark':'#212121'}
MC = {'LSTM':'#9E9E9E','Transformer':'#78909C','RSE':'#1565C0','W-DRO-T':'#7B1FA2','3SCH':'#E65100'}

plt.rcParams.update({'font.size':12,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--'})

def save(fig,name):
    fig.savefig(os.path.join(FIGDIR,name),bbox_inches='tight',pad_inches=0.15); plt.close(fig); print(f"  ✓ {name}")

with open(os.path.join(BASE,'audit_summary.json')) as f: audit=json.load(f)
with open(os.path.join(BASE,'statistical_analysis.json')) as f: sta=json.load(f)

models=['LSTM','Transformer','RSE','3SCH','W-DRO-T']
colors=[MC[m] for m in models]

# ═════════════════════════════════════════
# 18: CVaR comparison with bootstrap CIs
# ═════════════════════════════════════════
print("\n[18] CVaR comparison with CIs...")
fig,ax=plt.subplots(figsize=(10,7))
means=[audit[m]['cvar_95_mean'] for m in models]
ci=sta['bootstrap_ci']
lo=[ci[m]['lower'] for m in models]; hi=[ci[m]['upper'] for m in models]
errs=[[m-l for m,l in zip(means,lo)],[u-m for m,u in zip(means,hi)]]
bars=ax.bar(models,means,color=colors,edgecolor='white',lw=2,width=0.6,alpha=0.9)
ax.errorbar(models,means,yerr=errs,fmt='none',capsize=6,capthick=2,color='black',lw=2)
for b,m in zip(bars,means):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.012,f'{m:.3f}',ha='center',va='bottom',fontsize=11,fontweight='bold')
ax.set_ylabel('CVaR$_{0.95}$',fontsize=13,fontweight='bold')
ax.set_title('CVaR$_{0.95}$ Comparison (10 Seeds, 95% Bootstrap CI)',fontsize=14,fontweight='bold',pad=15)
ax.set_ylim(3.05,3.32)
fig.tight_layout()
save(fig,'18_cvar95_comparison_with_ci.pdf')

# ═════════════════════════════════════════
# 19: Multi-metric comparison
# ═════════════════════════════════════════
print("[19] Multi-metric comparison...")
fig,axes=plt.subplots(1,3,figsize=(18,6))
metrics=[('cvar_95_mean','CVaR$_{0.95}$','lower is better'),
         ('std_pnl_mean','Std P&L','lower is better'),
         ('entropic_mean','Entropic Risk','lower is better')]
for ax,(key,label,note) in zip(axes,metrics):
    vals=[audit[m][key] for m in models]
    bars=ax.barh(models,vals,color=colors,edgecolor='white',lw=1.5,height=0.6)
    max_val=max(vals)
    for b,v in zip(bars,vals):
        ax.text(b.get_width()+max_val*0.01,b.get_y()+b.get_height()/2,f'{v:.4f}',va='center',fontsize=10,fontweight='bold')
    ax.set_xlabel(label,fontsize=11,fontweight='bold')
    ax.set_title(f'{label}\n({note})',fontsize=12,fontweight='bold')
    ax.set_xlim(right=max_val*1.12)
fig.suptitle('Risk Metrics Comparison — All Models',fontsize=15,fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94]); save(fig,'19_risk_metrics_multi_comparison.pdf')

# ═════════════════════════════════════════
# 20: Seed-by-seed CVaR consistency
# ═════════════════════════════════════════
print("[20] Seed consistency...")
fig,ax=plt.subplots(figsize=(12,6))
seeds=list(range(10))
for m in models:
    ax.plot(seeds,audit[m]['cvar_95_values'],'o-',lw=2,ms=8,label=m,color=MC[m])
ax.set_xlabel('Seed Index',fontweight='bold'); ax.set_ylabel('CVaR$_{0.95}$',fontweight='bold')
ax.set_title('CVaR$_{0.95}$ Across Seeds — Consistency',fontsize=14,fontweight='bold')
ax.legend(fontsize=10); ax.set_xticks(seeds)
save(fig,'20_cvar_seed_consistency.pdf')

# ═════════════════════════════════════════
# 21: Statistical significance heatmap
# ═════════════════════════════════════════
print("[21] Statistical significance heatmap...")
fig,ax=plt.subplots(figsize=(8,6))
comp=sta['comparisons']
cm=['RSE','3SCH','W-DRO-T']; bl=['vs_lstm','vs_transformer']
pm=np.zeros((3,2)); dm=np.zeros((3,2))
for i,m in enumerate(cm):
    for j,b in enumerate(bl):
        pm[i,j]=comp[m][b]['p_value']; dm[i,j]=comp[m][b]['cohens_d']
im=ax.imshow(dm,cmap='RdBu_r',vmin=-8,vmax=2,aspect='auto')
ax.set_xticks(range(2)); ax.set_xticklabels(['vs LSTM','vs Transformer'],fontsize=11)
ax.set_yticks(range(3)); ax.set_yticklabels(cm,fontsize=11)
for i in range(3):
    for j in range(2):
        sig='***' if pm[i,j]<0.001 else '**' if pm[i,j]<0.01 else '*' if pm[i,j]<0.05 else 'ns'
        ax.text(j,i,f'd={dm[i,j]:.2f}\np={pm[i,j]:.2e}\n{sig}',ha='center',va='center',fontsize=9,fontweight='bold',
            color='white' if abs(dm[i,j])>3 else 'black')
plt.colorbar(im,ax=ax,label="Cohen's d")
ax.set_title("Statistical Significance — Cohen's d",fontsize=14,fontweight='bold')
save(fig,'21_statistical_significance_heatmap.pdf')

# ═════════════════════════════════════════
# 22: CV% stability analysis
# ═════════════════════════════════════════
print("[22] CV% stability...")
fig,ax=plt.subplots(figsize=(10,6))
cvs=[audit[m]['cvar_95_std']/audit[m]['cvar_95_mean']*100 for m in models]
bars=ax.bar(models,cvs,color=colors,edgecolor='white',lw=2,width=0.6,alpha=0.9)
for b,cv in zip(bars,cvs):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f'{cv:.2f}%',ha='center',va='bottom',fontsize=11,fontweight='bold')
ax.set_ylabel('Coefficient of Variation (%)',fontweight='bold')
ax.set_title('CVaR$_{0.95}$ Coefficient of Variation — Stability',fontsize=14,fontweight='bold')
save(fig,'22_cvar_cv_percent.pdf')

# ═════════════════════════════════════════
# EXTRA: Training time comparison
# ═════════════════════════════════════════
print("[22b] Training time comparison...")
fig,ax=plt.subplots(figsize=(10,6))
times=[audit[m]['train_time_mean']/60 for m in models]  # minutes
bars=ax.bar(models,times,color=colors,edgecolor='white',lw=2,width=0.6,alpha=0.9)
for b,t in zip(bars,times):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.5,f'{t:.1f}m',ha='center',va='bottom',fontsize=11,fontweight='bold')
ax.set_ylabel('Training Time (minutes)',fontweight='bold')
ax.set_title('Mean Training Time Per Seed',fontsize=14,fontweight='bold')
save(fig,'22b_training_time_comparison.pdf')

print("\n✅ Part 4 complete: 6 figures generated.")
