#!/usr/bin/env python3
"""Part 3: RSE regime features, signature diagram, training flowcharts."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, warnings
warnings.filterwarnings('ignore')

FIGDIR = os.path.dirname(os.path.abspath(__file__))
C = {'purple':'#7B1FA2','purple_bg':'#F3E5F5','blue':'#1565C0','blue_bg':'#E3F2FD',
     'orange':'#E65100','orange_bg':'#FFF3E0','red':'#C62828','red_bg':'#FFEBEE',
     'green':'#2E7D32','green_bg':'#E8F5E9','teal':'#00695C','gray':'#616161','dark':'#212121','bg':'#FAFAFA'}

plt.rcParams.update({'font.size':12,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--'})

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
# Simulate Heston for regime features
# ═════════════════════════════════════════
S0,v0,kappa,theta,sigma,rho = 100.0,0.04,1.0,0.04,0.2,-0.7
T,N,NP = 1.0,30,500; dt=T/N
np.random.seed(42)
S=np.zeros((NP,N+1)); v=np.zeros((NP,N+1)); S[:,0]=S0; v[:,0]=v0
for t in range(N):
    Z1=np.random.randn(NP); Z2=np.random.randn(NP)
    dWs=Z1*np.sqrt(dt); dWv=(rho*Z1+np.sqrt(1-rho**2)*Z2)*np.sqrt(dt)
    vt=np.maximum(v[:,t],0)
    S[:,t+1]=S[:,t]*np.exp(-0.5*vt*dt+np.sqrt(vt)*dWs)
    v[:,t+1]=np.maximum(vt+kappa*(theta-vt)*dt+sigma*np.sqrt(vt)*dWv,0)

lr = np.diff(np.log(S),axis=1)

def roll_std(x,w):
    r=np.zeros_like(x)
    for t in range(x.shape[1]):
        s=max(0,t-w+1)
        if t-s>=1: r[:,t]=np.std(x[:,s:t+1],axis=1)
    return r

def roll_mean(x,w):
    r=np.zeros_like(x)
    for t in range(x.shape[1]):
        s=max(0,t-w+1)
        r[:,t]=np.mean(x[:,s:t+1],axis=1)
    return r

rv5=roll_std(lr,5); rv10=roll_std(lr,10); rv20=roll_std(lr,20)
sma5=roll_mean(S[:,1:],5); sma20=roll_mean(S[:,1:],20)
trend=(sma5-sma20)/(sma20+1e-8)
bv=np.zeros_like(lr)
for t in range(20,lr.shape[1]):
    r=lr[:,t-20:t]; bv[:,t]=np.sum(np.abs(r[:,:-1])*np.abs(r[:,1:]),axis=1)*np.pi/2

# ═════════════════════════════════════════
# 13: RSE Regime Features
# ═════════════════════════════════════════
print("\n[13] RSE Regime Features...")
fig,axes=plt.subplots(3,2,figsize=(14,12))
ts=np.arange(lr.shape[1])/lr.shape[1]; pi=42

axes[0,0].plot(ts,rv5[pi],lw=2,color=C['blue'],label='RV(5)')
axes[0,0].plot(ts,rv10[pi],lw=2,color=C['purple'],label='RV(10)')
axes[0,0].plot(ts,rv20[pi],lw=2,color=C['orange'],label='RV(20)')
axes[0,0].set_title('Realized Volatility (Multi-Window)',fontsize=13,fontweight='bold')
axes[0,0].legend(); axes[0,0].set_ylabel('Volatility')

axes[0,1].plot(ts,trend[pi],lw=2,color=C['teal'])
axes[0,1].axhline(0,color='gray',ls='--',alpha=0.5)
axes[0,1].set_title('Trend Indicator (SMA Crossover)',fontsize=13,fontweight='bold')
axes[0,1].set_ylabel('Trend Strength')

axes[1,0].plot(ts,bv[pi],lw=2,color=C['red'])
axes[1,0].set_title('Bipower Variation (Jump Detection)',fontsize=13,fontweight='bold')
axes[1,0].set_ylabel('BiPV')

rv_sq=rv20[pi]**2; jr=rv_sq/(bv[pi]+1e-8)
axes[1,1].plot(ts,jr,lw=2,color=C['orange'])
axes[1,1].axhline(1.0,color=C['red'],ls='--',lw=1.5,label='Jump threshold')
axes[1,1].set_title('Jump Ratio (RV²/BiPV)',fontsize=13,fontweight='bold'); axes[1,1].legend()

im=axes[2,0].imshow(rv20[:50],aspect='auto',cmap='YlOrRd',interpolation='nearest')
axes[2,0].set_title('RV(20) Heatmap (50 Paths)',fontsize=13,fontweight='bold')
axes[2,0].set_xlabel('Time Step'); axes[2,0].set_ylabel('Path')
plt.colorbar(im,ax=axes[2,0],label='RV(20)')

# Regime classification
regimes=np.zeros(NP)
frv=rv20[:,-1]; ft=trend[:,-1]
p25,p75=np.percentile(frv,25),np.percentile(frv,75)
for i in range(NP):
    if frv[i]<p25: regimes[i]=0
    elif frv[i]>p75: regimes[i]=1
    elif ft[i]>0: regimes[i]=2
    else: regimes[i]=3
rn=['Low Vol','High Vol','Trending','Mean-Rev']
rc=[C['green'],C['red'],C['blue'],C['orange']]
axes[2,1].bar(rn,[np.sum(regimes==i) for i in range(4)],color=rc,edgecolor='white',lw=1.5)
axes[2,1].set_title('Regime Classification',fontsize=13,fontweight='bold'); axes[2,1].set_ylabel('Count')

fig.suptitle('RSE — Econometric Regime Features',fontsize=16,fontweight='bold',y=1.01)
fig.tight_layout(); save(fig,'13_rse_regime_features.pdf')

# ═════════════════════════════════════════
# 14: Signature Transform Diagram
# ═════════════════════════════════════════
print("[14] Signature transform diagram...")
fig,ax=plt.subplots(figsize=(15,6)); ax.set_xlim(0,15); ax.set_ylim(0,6); ax.axis('off')
ax.set_title('Signature Transform Component in RSE',fontsize=16,fontweight='bold',pad=15)
box(ax,0.2,2,2.5,2,'Price Path\n$X:[0,T]\\to\\mathbb{R}^d$\n(S, v, features)',C['blue'],fs=9)
arr(ax,2.7,3,3.8,3)
box(ax,3.8,1.3,3.5,3.4,'Signature Transform\n$\\mathrm{Sig}(X)_{[s,t]}$\n$=(1,\\int dX,$\n$\\int\\!\\int dXdX,\\ldots)$\nTruncated at depth $m$',C['purple'],fs=9)
arr(ax,7.3,3,8.3,3)
box(ax,8.3,1.5,3,3,'Running Statistics\nMean $\\mu_t$\nStd $\\sigma_t$\nMax/Min\nPath signature features',C['orange'],fs=9)
arr(ax,11.3,3,12.3,3)
box(ax,12.3,2,2.2,2,'MLP\n$\\to \\delta_k$\nPath-dependent\nhedge ratio',C['green'],fs=9)
ax.text(7.5,0.3,'Captures: order of events • oscillation signature • roughness • non-Markovian path effects',
    fontsize=11,fontweight='bold',color=C['gray'],ha='center',
    bbox=dict(boxstyle='round,pad=0.3',facecolor='#F5F5F5',edgecolor=C['gray'],lw=1))
save(fig,'14_rse_signature_transform.pdf')

# ═════════════════════════════════════════
# 15: RSE Training Pipeline
# ═════════════════════════════════════════
print("[15] RSE training pipeline...")
fig,ax=plt.subplots(figsize=(16,8)); ax.set_xlim(0,16); ax.set_ylim(0,8); ax.axis('off')
ax.set_title('RSE Training Pipeline',fontsize=16,fontweight='bold',pad=15)
# Phases
box(ax,0.2,6,3.2,1.5,'Phase 1: Pre-train\nBase Models\n(LSTM, Transformer, Sig)',C['blue'],fs=9)
arr(ax,3.4,6.75,4.3,6.75)
box(ax,4.3,6,3.2,1.5,'Phase 2: Freeze\nExtract Regime\nFeatures (RV, SMA, BiPV)',C['purple'],fs=9)
arr(ax,7.5,6.75,8.3,6.75)
box(ax,8.3,6,3.2,1.5,'Phase 3: Train\nGating Network\n+ Regime Classifier',C['orange'],fs=9)
arr(ax,11.5,6.75,12.3,6.75)
box(ax,12.3,6,3.2,1.5,'Phase 4: Optional\nEnd-to-end\nFine-tuning',C['green'],fs=9)
# Detail boxes
box(ax,0.2,3.5,3.2,2,'LSTM: 50+30 ep\nTransformer: 50+30\nSignature: 50+30\nCVaR→Entropic',C['blue_bg'],tc=C['dark'],fs=8,a=0.8)
box(ax,4.3,3.5,3.2,2,'RV(5,10,20)\nSMA crossover (5/20)\nBiPV jump indicator\nMomentum (5-step)',C['purple_bg'],tc=C['dark'],fs=8,a=0.8)
box(ax,8.3,3.5,3.2,2,'Affinity A ∈ R⁴ˣ³\nSoftmax τ temperature\nRegime probs p_j\nCVaR₉₅ ensemble loss',C['orange_bg'],tc=C['dark'],fs=8,a=0.8)
box(ax,12.3,3.5,3.2,2,'Unfreeze base models\nLower LR: 10⁻⁴→10⁻⁵\n10 epochs\nEnsemble CVaR loss',C['green_bg'],tc=C['dark'],fs=8,a=0.8)
# Config bar
box(ax,0.2,0.5,15.3,2.5,'',C['bg'],a=0.5)
ax.text(8,2.5,'Training Configuration',fontsize=12,fontweight='bold',ha='center',color=C['dark'])
ax.text(8,1.3,'80K Heston paths (50K train / 10K val / 20K test)  •  N=30 steps  •  10 seeds\n'
    'LR: 10⁻³→10⁻⁴  •  Adam  •  Batch: 256  •  Bootstrap: 10K  •  Holm–Bonferroni α=0.05',
    fontsize=10,ha='center',color=C['gray'])
save(fig,'15_training_pipeline_rse.pdf')

# ═════════════════════════════════════════
# 16: W-DRO-T Training Procedure
# ═════════════════════════════════════════
print("[16] W-DRO-T training procedure...")
fig,ax=plt.subplots(figsize=(15,7)); ax.set_xlim(0,15); ax.set_ylim(0,7); ax.axis('off')
ax.set_title('W-DRO-T Training Procedure',fontsize=16,fontweight='bold',pad=15)
box(ax,0.3,4,3.2,2,'Stage 1: CVaR Pre-train\n50 epochs\n$\\mathcal{L}=CVaR_{0.95}$\nLR=$10^{-3}$',C['red'])
arr(ax,3.5,5,4.5,5)
box(ax,4.5,4,3.5,2,'Stage 2: Entropic + DRO\n30 epochs\n$\\mathcal{L}=\\rho_\\lambda+\\epsilon\\|\\nabla\\|_2$\n$\\epsilon:0\\to0.1$',C['purple'])
arr(ax,8,5,9,5)
box(ax,9,4,3.5,2,'ε-Annealing Schedule\nLinear warmup 15ep\nSDPA Math backend\ncreate_graph=True',C['orange'])
# Notes
box(ax,0.5,1,4.5,2.2,'Key Implementation Detail:\nDRO gradient penalty needs\ncreate_graph=True for ∇²\n→ forces SDPA Math backend\n(no Flash/Efficient Attention)',C['red_bg'],tc=C['dark'],fs=9,a=0.85)
box(ax,6,1,5.5,2.2,'Robustness Guarantee (Blanchet & Murthy 2019):\n$\\sup_{Q:W(P,Q)\\leq\\epsilon} E_Q[\\ell]$\n$\\approx E_P[\\ell]+\\epsilon\\cdot E_P[\\|\\nabla\\ell\\|_2]$\nFirst-order approximation of worst-case risk',C['purple_bg'],tc=C['dark'],fs=9,a=0.85)
save(fig,'16_training_procedure_wdrot.pdf')

# ═════════════════════════════════════════
# 17: 3SCH Training Schedule
# ═════════════════════════════════════════
print("[17] 3SCH training schedule...")
fig,axes=plt.subplots(1,2,figsize=(14,6))

epochs=np.arange(100)
alpha=np.ones(100)
alpha[:50]=1.0
for i in range(50,70): alpha[i]=0.8-(0.8-0.2)*(i-50)/20
alpha[70:]=0.0

axes[0].fill_between(range(50),0,1,alpha=0.2,color=C['red'],label='Stage 1: CVaR₉₅')
axes[0].fill_between(range(50,70),0,1,alpha=0.2,color=C['orange'],label='Stage 2: Mixed')
axes[0].fill_between(range(70,100),0,1,alpha=0.2,color=C['green'],label='Stage 3: Entropic')
axes[0].plot(epochs,alpha,lw=3,color=C['purple'],label='α (CVaR weight)')
axes[0].set_xlabel('Epoch',fontweight='bold'); axes[0].set_ylabel('CVaR Weight α',fontweight='bold')
axes[0].set_title('3SCH Curriculum Schedule',fontsize=14,fontweight='bold')
axes[0].legend(fontsize=9,loc='center right'); axes[0].set_ylim(-0.05,1.1)

x=np.linspace(-3,3,200)
cvar_l=0.5*x**2+0.3*np.abs(x)
ent_l=np.log(1+np.exp(x))+0.1*x**2
mixed=0.5*cvar_l+0.5*ent_l
axes[1].plot(x,cvar_l,lw=2.5,color=C['red'],label='CVaR Loss')
axes[1].plot(x,ent_l,lw=2.5,color=C['green'],label='Entropic Loss')
axes[1].plot(x,mixed,lw=2.5,color=C['orange'],ls='--',label='Mixed Loss')
axes[1].set_xlabel('P&L',fontweight='bold'); axes[1].set_ylabel('Loss',fontweight='bold')
axes[1].set_title('Loss Function Landscape',fontsize=14,fontweight='bold'); axes[1].legend()
fig.tight_layout(); save(fig,'17_training_schedule_3sch.pdf')

print("\n✅ Part 3 complete: 5 figures generated.")
