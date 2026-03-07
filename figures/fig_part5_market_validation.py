#!/usr/bin/env python3
"""Part 5: Market validation (SPY, NIFTY, yfinance) + crisis testing."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import json, os, warnings
warnings.filterwarnings('ignore')

FIGDIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(FIGDIR), 'new_approaches', 'results')
C = {'purple':'#7B1FA2','blue':'#1565C0','orange':'#E65100','red':'#C62828',
     'green':'#2E7D32','teal':'#00695C','gray':'#616161'}
MC = {'LSTM':'#9E9E9E','Transformer':'#78909C','RSE':'#1565C0','W-DRO-T':'#7B1FA2','3SCH':'#E65100'}

plt.rcParams.update({'font.size':12,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--'})

def save(fig,name):
    fig.savefig(os.path.join(FIGDIR,name),bbox_inches='tight',pad_inches=0.15); plt.close(fig); print(f"  ✓ {name}")

def clean(d):
    if isinstance(d,dict): return {k:clean(v) for k,v in d.items()}
    if isinstance(d,float) and (d==float('inf') or d>1e10): return np.nan
    return d

with open(os.path.join(BASE,'extended_real_market_validation.json')) as f:
    mkt=clean(json.load(f))

models=['LSTM','Transformer','RSE','W-DRO-T','3SCH']

# ═════════════════════════════════════════
# 23: SPY crisis stress testing
# ═════════════════════════════════════════
print("\n[23] SPY crisis stress testing...")
fig,axes=plt.subplots(2,2,figsize=(14,10))
spy_sc=[('normal_2019','Normal 2019'),('covid_2020','COVID-19 2020'),
        ('post_covid_2021','Post-COVID 2021'),('crisis_2008','GFC 2008')]
for ax,(sc,title) in zip(axes.flat,spy_sc):
    vals=[mkt['SPY'][sc][m]['cvar95'] for m in models]
    bars=ax.barh(models,vals,color=[MC[m] for m in models],edgecolor='white',lw=1.5,height=0.6)
    for b,v in zip(bars,vals):
        ax.text(b.get_width()+0.5,b.get_y()+b.get_height()/2,f'{v:.1f}',va='center',fontsize=10,fontweight='bold')
    ax.set_title(f'SPY — {title}',fontsize=13,fontweight='bold')
    ax.set_xlabel('CVaR$_{0.95}$',fontsize=11)
fig.suptitle('SPY Crisis Stress Testing — CVaR$_{0.95}$',fontsize=16,fontweight='bold',y=1.01)
fig.tight_layout(); save(fig,'23_spy_crisis_stress_testing.pdf')

# ═════════════════════════════════════════
# 24: NIFTY stress testing
# ═════════════════════════════════════════
print("[24] NIFTY stress testing...")
fig,axes=plt.subplots(1,3,figsize=(16,5))
nifty_sc=[('normal_2019','Normal 2019'),('covid_2020','COVID-19 2020'),('crisis_2008','GFC 2008')]
for ax,(sc,title) in zip(axes,nifty_sc):
    vals=[mkt['NIFTY'][sc][m]['cvar95'] for m in models]
    bars=ax.barh(models,vals,color=[MC[m] for m in models],edgecolor='white',lw=1.5,height=0.6)
    for b,v in zip(bars,vals):
        ax.text(b.get_width()+5,b.get_y()+b.get_height()/2,f'{v:.0f}',va='center',fontsize=10,fontweight='bold')
    ax.set_title(f'NIFTY — {title}',fontsize=13,fontweight='bold')
    ax.set_xlabel('CVaR$_{0.95}$',fontsize=11)
fig.suptitle('NIFTY 50 Crisis Stress Testing — CVaR$_{0.95}$',fontsize=16,fontweight='bold',y=1.02)
fig.tight_layout(); save(fig,'24_nifty_crisis_stress_testing.pdf')

# ═════════════════════════════════════════
# 25: Crisis-to-normal degradation ratio
# ═════════════════════════════════════════
print("[25] Crisis degradation ratio...")
fig,ax=plt.subplots(figsize=(10,6))
sn={m:mkt['SPY']['normal_2019'][m]['cvar95'] for m in models}
sc={m:mkt['SPY']['covid_2020'][m]['cvar95'] for m in models}
sg={m:mkt['SPY']['crisis_2008'][m]['cvar95'] for m in models}
x=np.arange(len(models)); w=0.35
cr=[sc[m]/sn[m] for m in models]; gr=[sg[m]/sn[m] for m in models]
ax.bar(x-w/2,cr,w,color=C['orange'],label='COVID/Normal',edgecolor='white')
ax.bar(x+w/2,gr,w,color=C['red'],label='GFC/Normal',edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel('Crisis-to-Normal Ratio',fontweight='bold')
ax.set_title('SPY — Crisis Degradation Ratio',fontsize=14,fontweight='bold')
ax.legend(fontsize=11)
for i,(c,g) in enumerate(zip(cr,gr)):
    ax.text(i-w/2,c+0.1,f'{c:.1f}×',ha='center',fontsize=9,fontweight='bold')
    ax.text(i+w/2,g+0.1,f'{g:.1f}×',ha='center',fontsize=9,fontweight='bold')
save(fig,'25_spy_crisis_degradation_ratio.pdf')

# ═════════════════════════════════════════
# 26: Std P&L across regimes
# ═════════════════════════════════════════
print("[26] P&L volatility across regimes...")
fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,(market,title) in zip(axes,[('SPY','SPY'),('NIFTY','NIFTY')]):
    scenarios=list(mkt[market].keys())
    xp=np.arange(len(scenarios)); w=0.15
    for i,m in enumerate(models):
        vals=[mkt[market][s][m]['std_pnl'] for s in scenarios]
        ax.bar(xp+i*w-2*w,vals,w,color=MC[m],label=m if ax==axes[0] else None,edgecolor='white')
    ax.set_xticks(xp); ax.set_xticklabels([s.replace('_','\n') for s in scenarios],fontsize=9)
    ax.set_ylabel('Std P&L',fontweight='bold')
    ax.set_title(f'{title} — P&L Volatility',fontsize=13,fontweight='bold')
axes[0].legend(fontsize=9,loc='upper left')
fig.suptitle('P&L Volatility Across Market Regimes',fontsize=15,fontweight='bold',y=1.02)
fig.tight_layout(); save(fig,'26_pnl_volatility_across_regimes.pdf')

# ═════════════════════════════════════════
# 27: YFinance real market data
# ═════════════════════════════════════════
print("[27] Downloading SPY & NIFTY from yfinance...")
try:
    import yfinance as yf

    spy = yf.download('SPY', start='2007-01-01', end='2025-12-31', progress=False)
    nifty = yf.download('^NSEI', start='2007-01-01', end='2025-12-31', progress=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # SPY price
    spy_close = spy['Close'].values.flatten() if 'Close' in spy.columns else spy.iloc[:, 3].values.flatten()
    axes[0, 0].plot(spy.index, spy_close, lw=1.5, color=C['blue'])
    axes[0, 0].axvspan('2008-09-01', '2009-03-31', alpha=0.2, color=C['red'], label='GFC')
    axes[0, 0].axvspan('2020-02-01', '2020-06-30', alpha=0.2, color=C['orange'], label='COVID')
    axes[0, 0].set_title('SPY — Historical Price', fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel('Price ($)'); axes[0, 0].legend(fontsize=9)

    # SPY returns
    spy_ret = np.diff(np.log(spy_close))
    spy_ret = spy_ret[~np.isnan(spy_ret)]
    axes[0, 1].hist(spy_ret, bins=100, density=True, alpha=0.7, color=C['blue'], edgecolor='white', lw=0.3)
    xn = np.linspace(spy_ret.min(), spy_ret.max(), 200)
    axes[0, 1].plot(xn, stats.norm.pdf(xn, spy_ret.mean(), spy_ret.std()), lw=2, color=C['red'], label='Normal fit')
    axes[0, 1].set_title('SPY Log Return Distribution', fontsize=13, fontweight='bold'); axes[0, 1].legend()

    # NIFTY
    if len(nifty) > 0:
        n_close = nifty['Close'].values.flatten() if 'Close' in nifty.columns else nifty.iloc[:, 3].values.flatten()
        axes[1, 0].plot(nifty.index, n_close, lw=1.5, color=C['purple'])
        axes[1, 0].axvspan('2008-09-01', '2009-03-31', alpha=0.2, color=C['red'], label='GFC')
        axes[1, 0].axvspan('2020-02-01', '2020-06-30', alpha=0.2, color=C['orange'], label='COVID')
        axes[1, 0].set_title('NIFTY 50 — Historical Price', fontsize=13, fontweight='bold')
        axes[1, 0].set_ylabel('Price (₹)'); axes[1, 0].legend(fontsize=9)

        n_ret = np.diff(np.log(n_close)); n_ret = n_ret[~np.isnan(n_ret)]
        axes[1, 1].hist(n_ret, bins=100, density=True, alpha=0.7, color=C['purple'], edgecolor='white', lw=0.3)
        xn2 = np.linspace(n_ret.min(), n_ret.max(), 200)
        axes[1, 1].plot(xn2, stats.norm.pdf(xn2, n_ret.mean(), n_ret.std()), lw=2, color=C['red'], label='Normal fit')
        axes[1, 1].set_title('NIFTY Log Return Distribution', fontsize=13, fontweight='bold'); axes[1, 1].legend()

    fig.suptitle('Real Market Data — SPY & NIFTY 50', fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout(); save(fig, '27_real_market_spy_nifty_yfinance.pdf')

    # 28: Rolling volatility
    print("[28] Rolling volatility comparison...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    rv_spy = np.array([np.nanstd(spy_ret[max(0,i-20):i])*np.sqrt(252) for i in range(1, len(spy_ret))])
    axes[0].plot(spy.index[2:len(rv_spy)+2], rv_spy, lw=0.8, color=C['blue'])
    axes[0].axhline(0.2, color=C['red'], ls='--', lw=2, label='Heston $\\sqrt{\\theta}=0.2$')
    axes[0].set_title('SPY — 20-Day Rolling Annualized Vol', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Volatility'); axes[0].legend()
    axes[0].axvspan('2008-09-01', '2009-03-31', alpha=0.15, color=C['red'])
    axes[0].axvspan('2020-02-01', '2020-06-30', alpha=0.15, color=C['orange'])

    if len(nifty) > 0:
        n_ret2 = np.diff(np.log(n_close)); n_ret2 = n_ret2[~np.isnan(n_ret2)]
        rv_n = np.array([np.nanstd(n_ret2[max(0,i-20):i])*np.sqrt(252) for i in range(1, len(n_ret2))])
        axes[1].plot(nifty.index[2:len(rv_n)+2], rv_n, lw=0.8, color=C['purple'])
        axes[1].axhline(0.2, color=C['red'], ls='--', lw=2, label='Heston $\\sqrt{\\theta}=0.2$')
        axes[1].set_title('NIFTY — 20-Day Rolling Annualized Vol', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('Volatility'); axes[1].legend()
        axes[1].axvspan('2008-09-01', '2009-03-31', alpha=0.15, color=C['red'])
        axes[1].axvspan('2020-02-01', '2020-06-30', alpha=0.15, color=C['orange'])

    fig.suptitle('Rolling Volatility vs Heston Calibration', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout(); save(fig, '28_rolling_volatility_comparison.pdf')

except Exception as e:
    print(f"  ⚠ yfinance download failed: {e}")
    print("  Skipping figures 27-28. Market validation bar charts (23-26) already generated.")

print("\n✅ Part 5 complete.")
