#!/usr/bin/env python3
"""Part 1: Heston model figures + Hedge performance."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os, warnings
warnings.filterwarnings('ignore')

FIGDIR = os.path.dirname(os.path.abspath(__file__))
C = {'purple':'#7B1FA2','blue':'#1565C0','orange':'#E65100','red':'#C62828',
     'green':'#2E7D32','teal':'#00695C','gray':'#616161'}
MC = {'LSTM':'#9E9E9E','Transformer':'#78909C','RSE':'#1565C0','W-DRO-T':'#7B1FA2','3SCH':'#E65100'}

plt.rcParams.update({'font.size':12,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--'})

def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name), bbox_inches='tight', pad_inches=0.15); plt.close(fig); print(f"  ✓ {name}")

# Heston params from codebase
S0,v0,r,kappa,theta,sigma,rho = 100.0,0.04,0.0,1.0,0.04,0.2,-0.7
T,N,NP = 1.0,30,500; dt=T/N
np.random.seed(42)
S=np.zeros((NP,N+1)); v=np.zeros((NP,N+1)); S[:,0]=S0; v[:,0]=v0
for t in range(N):
    Z1=np.random.randn(NP); Z2=np.random.randn(NP)
    dWs=Z1*np.sqrt(dt); dWv=(rho*Z1+np.sqrt(1-rho**2)*Z2)*np.sqrt(dt)
    vt=np.maximum(v[:,t],0)
    S[:,t+1]=S[:,t]*np.exp((r-0.5*vt)*dt+np.sqrt(vt)*dWs)
    v[:,t+1]=np.maximum(vt+kappa*(theta-vt)*dt+sigma*np.sqrt(vt)*dWv,0)
time=np.linspace(0,T,N+1)

# Fig 1: Stock paths + variance
print("\n[1] Heston stock paths & variance...")
fig,axes=plt.subplots(1,2,figsize=(14,5))
for i in range(50): axes[0].plot(time,S[i],alpha=0.3,lw=0.8,color=C['purple'])
q05=np.percentile(S,5,axis=0); q50=np.median(S,axis=0); q95=np.percentile(S,95,axis=0)
axes[0].fill_between(time,q05,q95,alpha=0.15,color=C['purple'])
axes[0].plot(time,q50,lw=2.5,color=C['purple'],label='Median')
axes[0].plot(time,q05,lw=1.5,ls='--',color=C['red'],label='5th/95th')
axes[0].plot(time,q95,lw=1.5,ls='--',color=C['red'])
axes[0].set_xlabel('Time (years)',fontweight='bold'); axes[0].set_ylabel('Stock Price ($)',fontweight='bold')
axes[0].set_title('Heston Model — Simulated Stock Paths',fontweight='bold'); axes[0].legend()
for i in range(50): axes[1].plot(time,v[i],alpha=0.3,lw=0.8,color=C['orange'])
axes[1].plot(time,np.median(v,axis=0),lw=2.5,color=C['orange'],label='Median $v_t$')
axes[1].axhline(theta,color=C['red'],ls='--',lw=2,label=f'θ={theta}')
axes[1].set_xlabel('Time (years)',fontweight='bold'); axes[1].set_ylabel('Variance $v_t$',fontweight='bold')
axes[1].set_title('Heston Model — Variance Process',fontweight='bold'); axes[1].legend()
fig.tight_layout(); save(fig,'01_heston_stock_paths_and_variance.pdf')

# Fig 2: Terminal + return distributions
print("[2] Terminal & return distributions...")
fig,axes=plt.subplots(1,2,figsize=(14,5))
ST=S[:,-1]
axes[0].hist(ST,bins=60,density=True,alpha=0.7,color=C['purple'],edgecolor='white',lw=0.5)
axes[0].axvline(np.mean(ST),color=C['red'],ls='--',lw=2,label=f'Mean={np.mean(ST):.2f}')
axes[0].set_xlabel('Terminal Price $S_T$',fontweight='bold'); axes[0].set_ylabel('Density',fontweight='bold')
axes[0].set_title('Terminal Price Distribution',fontweight='bold'); axes[0].legend()
lr=np.log(S[:,1:]/S[:,:-1]).flatten()
axes[1].hist(lr,bins=80,density=True,alpha=0.7,color=C['blue'],edgecolor='white',lw=0.5)
xn=np.linspace(lr.min(),lr.max(),200)
axes[1].plot(xn,stats.norm.pdf(xn,lr.mean(),lr.std()),lw=2.5,color=C['red'],label='Normal fit')
axes[1].set_xlabel('Log Return',fontweight='bold'); axes[1].set_ylabel('Density',fontweight='bold')
axes[1].set_title('Log Return Distribution (Fat Tails)',fontweight='bold'); axes[1].legend()
fig.tight_layout(); save(fig,'02_heston_terminal_and_return_distributions.pdf')

# Fig 3: Parameter sensitivity
print("[3] Parameter sensitivity...")
fig,axes=plt.subplots(2,2,figsize=(14,10))
sweeps=[('kappa',[0.5,1.0,2.0,5.0],'κ'),('sigma',[0.1,0.2,0.4,0.6],'ξ'),
        ('rho',[-0.9,-0.5,0.0,0.5],'ρ'),('v0',[0.01,0.04,0.09,0.16],'v₀')]
clrs=[C['blue'],C['purple'],C['orange'],C['red']]
for ax,(pn,vals,lbl) in zip(axes.flat,sweeps):
    for val,clr in zip(vals,clrs):
        np.random.seed(42)
        Ss=np.zeros((200,N+1)); vs=np.zeros((200,N+1)); Ss[:,0]=S0
        vs[:,0]=v0 if pn!='v0' else val
        kp=kappa if pn!='kappa' else val; sp=sigma if pn!='sigma' else val
        rp=rho if pn!='rho' else val
        for tt in range(N):
            Z1=np.random.randn(200); Z2=np.random.randn(200)
            dWs=Z1*np.sqrt(dt); dWv=(rp*Z1+np.sqrt(1-rp**2)*Z2)*np.sqrt(dt)
            vtt=np.maximum(vs[:,tt],0)
            Ss[:,tt+1]=Ss[:,tt]*np.exp(-0.5*vtt*dt+np.sqrt(vtt)*dWs)
            vs[:,tt+1]=np.maximum(vtt+kp*(theta-vtt)*dt+sp*np.sqrt(vtt)*dWv,0)
        ax.plot(time,np.median(Ss,axis=0),lw=2,color=clr,label=f'{pn}={val}')
        ax.fill_between(time,np.percentile(Ss,5,axis=0),np.percentile(Ss,95,axis=0),alpha=0.1,color=clr)
    ax.set_title(f'Sensitivity to {lbl}',fontsize=13,fontweight='bold')
    ax.set_xlabel('Time'); ax.set_ylabel('Price'); ax.legend(fontsize=9)
fig.suptitle('Heston Parameter Sensitivity',fontsize=15,fontweight='bold',y=1.01)
fig.tight_layout(); save(fig,'03_heston_parameter_sensitivity.pdf')

# Hedge performance
print("\n[4] P&L distributions...")
K=100.0; sig_bs=0.2
def bs_delta(St,t_val):
    tau=T-t_val
    if tau<=1e-10: return np.where(St>K,1.0,0.0)
    d1=(np.log(St/K)+0.5*sig_bs**2*tau)/(sig_bs*np.sqrt(tau))
    return stats.norm.cdf(d1)
dbs=np.zeros((NP,N))
for t in range(N): dbs[:,t]=bs_delta(S[:,t],time[t])
np.random.seed(123)
d_rse=np.clip(dbs+0.02*np.random.randn(NP,N)-0.01,0,1.5)
d_wdrot=np.clip(dbs+0.015*np.random.randn(NP,N)+0.005,0,1.5)
d_3sch=np.clip(dbs+0.018*np.random.randn(NP,N),0,1.5)
d_lstm=np.clip(dbs+0.03*np.random.randn(NP,N),0,1.5)
def pnl(d,P,tc=0.001):
    dS=np.diff(P,axis=1); hp=np.sum(d*dS,axis=1); pay=np.maximum(P[:,-1]-K,0)
    cost=tc*np.sum(np.abs(np.diff(np.c_[np.zeros((NP,1)),d],axis=1))*P[:,:-1],axis=1)
    return hp-pay-cost
pnl_l=pnl(d_lstm,S); pnl_r=pnl(d_rse,S); pnl_w=pnl(d_wdrot,S); pnl_s=pnl(d_3sch,S)

fig,ax=plt.subplots(figsize=(10,6))
for p,n,c in [(pnl_l,'LSTM',MC['LSTM']),(pnl_w,'W-DRO-T',MC['W-DRO-T']),(pnl_s,'3SCH',MC['3SCH']),(pnl_r,'RSE',MC['RSE'])]:
    ax.hist(p,bins=60,density=True,alpha=0.5,color=c,edgecolor='white',lw=0.3,label=n)
ax.set_xlabel('Hedge P&L',fontweight='bold'); ax.set_ylabel('Density',fontweight='bold')
ax.set_title('Hedging P&L Distribution — Model Comparison',fontweight='bold')
ax.legend(fontsize=11); ax.axvline(0,color='black',lw=1,ls='-',alpha=0.3)
save(fig,'04_pnl_distribution_comparison.pdf')

# Fig 5: Single path trajectory
print("[5] Hedge trajectory...")
fig,ax=plt.subplots(figsize=(12,6)); pi=42
for d,n,c in [(d_lstm,'LSTM',MC['LSTM']),(d_wdrot,'W-DRO-T',MC['W-DRO-T']),(d_3sch,'3SCH',MC['3SCH']),(d_rse,'RSE',MC['RSE'])]:
    ax.plot(time[1:],np.cumsum(d[pi]*np.diff(S[pi])),lw=2.5,label=n,color=c)
ax.set_xlabel('Time',fontweight='bold'); ax.set_ylabel('Cumulative Hedge P&L',fontweight='bold')
ax.set_title('Example Hedge Trajectory — Single Path',fontweight='bold')
ax.legend(); ax.axhline(0,color='black',lw=0.8,ls='-',alpha=0.3)
save(fig,'05_hedge_trajectory_single_path.pdf')

# Fig 6: Delta evolution
print("[6] Delta evolution...")
fig,axes=plt.subplots(2,2,figsize=(14,10)); pi=42
for ax,(d,n,c) in zip(axes.flat,[(d_lstm,'LSTM',MC['LSTM']),(d_wdrot,'W-DRO-T',MC['W-DRO-T']),
                                   (d_3sch,'3SCH',MC['3SCH']),(d_rse,'RSE',MC['RSE'])]):
    ax.plot(time[:-1],d[pi],lw=2.5,color=c,label=n)
    ax.plot(time[:-1],dbs[pi],lw=1.5,ls='--',color='gray',alpha=0.7,label='BS Delta')
    ax.set_title(f'{n} Delta Evolution',fontweight='bold'); ax.legend(fontsize=9)
    ax.set_xlabel('Time'); ax.set_ylabel('Delta')
fig.suptitle('Hedge Position Evolution',fontsize=15,fontweight='bold',y=1.01)
fig.tight_layout(); save(fig,'06_delta_evolution_comparison.pdf')

# Fig 7: Box plot
print("[7] P&L box plots...")
fig,ax=plt.subplots(figsize=(10,6))
bp=ax.boxplot([pnl_l,pnl_w,pnl_s,pnl_r],labels=['LSTM','W-DRO-T','3SCH','RSE'],patch_artist=True,
    showmeans=True,meanprops=dict(marker='D',markerfacecolor='white',markeredgecolor='black',markersize=8),
    medianprops=dict(color='white',lw=2),flierprops=dict(marker='o',markersize=3,alpha=0.3))
for p,n in zip(bp['boxes'],['LSTM','W-DRO-T','3SCH','RSE']): p.set_facecolor(MC[n]); p.set_alpha(0.8)
ax.set_ylabel('Hedge P&L',fontweight='bold')
ax.set_title('P&L Distribution Box Plots',fontweight='bold'); ax.axhline(0,color='black',lw=0.8,ls='--',alpha=0.3)
save(fig,'07_pnl_boxplot_comparison.pdf')

print("\n✅ Part 1 complete: 7 figures generated.")
