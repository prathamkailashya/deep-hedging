#!/usr/bin/env python3
"""Part 2: Architecture diagrams for all 5 models."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os, warnings
warnings.filterwarnings('ignore')

FIGDIR = os.path.dirname(os.path.abspath(__file__))
C = {'purple':'#7B1FA2','purple_l':'#CE93D8','purple_bg':'#F3E5F5',
     'blue':'#1565C0','blue_l':'#64B5F6','blue_bg':'#E3F2FD',
     'orange':'#E65100','orange_l':'#FFB74D','orange_bg':'#FFF3E0',
     'red':'#C62828','red_l':'#EF9A9A','red_bg':'#FFEBEE',
     'green':'#2E7D32','green_l':'#81C784','green_bg':'#E8F5E9',
     'teal':'#00695C','gray':'#616161','gray_l':'#BDBDBD','dark':'#212121','bg':'#FAFAFA'}

plt.rcParams.update({'font.size':11,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
    'figure.facecolor':'white','axes.facecolor':'white'})

def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name), bbox_inches='tight', pad_inches=0.15); plt.close(fig); print(f"  ✓ {name}")

def box(ax, x, y, w, h, txt, col, tc='white', fs=10, a=0.95):
    b = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.12",facecolor=col,edgecolor='#333',lw=1.5,alpha=a)
    ax.add_patch(b)
    for i,line in enumerate(txt.split('\n')):
        off = (len(txt.split('\n'))-1)*0.5*(fs/72)*2.2
        ax.text(x+w/2, y+h/2+off-i*(fs/72)*2.2, line, ha='center',va='center',fontsize=fs,fontweight='bold',color=tc)

def arr(ax, x1, y1, x2, y2, col='#333', lw=2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color=col,lw=lw))

# ═══════════════════════════════════════════════════
# 08: LSTM Hedger Architecture
# ═══════════════════════════════════════════════════
print("\n[8] LSTM Architecture...")
fig,ax=plt.subplots(figsize=(14,5.5)); ax.set_xlim(0,14); ax.set_ylim(0,5.5); ax.axis('off')
ax.set_title('LSTM Deep Hedger Architecture',fontsize=16,fontweight='bold',pad=15)
box(ax,0.4,1.8,2,2,'Input Features\n$I_k \\in \\mathbb{R}^5$\n(S/K, τ, σ, δ_{k-1})',C['blue'],fs=9)
arr(ax,2.4,2.8,3.3,2.8)
box(ax,3.3,1.3,2.8,3,'LSTM Cell\n2 layers, h=50\nForget/Input/Output\nGates',C['purple'],fs=9)
arr(ax,6.1,2.8,7,2.8)
box(ax,7,1.8,2,2,'Linear Layer\n$\\mathbb{R}^{50} \\to \\mathbb{R}^1$',C['orange'],fs=9)
arr(ax,9,2.8,10,2.8)
box(ax,10,1.8,2.5,2,'Activation\n$\\delta_k = 1.5 \\cdot \\tanh(\\cdot)$\nClipped output',C['green'],fs=9)
arr(ax,12.5,2.8,13.3,2.8,col=C['red'])
ax.text(13.5,2.8,'$\\delta_k$',fontsize=18,fontweight='bold',color=C['red'],va='center')
# Recurrence
ax.annotate('$\\delta_{k-1}$ feedback',xy=(3.3,1.3),xytext=(5,0.4),fontsize=10,color=C['gray'],
    arrowprops=dict(arrowstyle='->',color=C['gray'],lw=1.5,ls='--'))
# Hidden state
ax.annotate('$h_{k-1}, c_{k-1}$ recurrence',xy=(3.3,4.3),xytext=(1,5),fontsize=10,color=C['gray'],
    arrowprops=dict(arrowstyle='->',color=C['gray'],lw=1.5,ls='--'))
save(fig,'08_architecture_lstm_hedger.pdf')

# ═══════════════════════════════════════════════════
# 09: Transformer Hedger Architecture
# ═══════════════════════════════════════════════════
print("[9] Transformer Architecture...")
fig,ax=plt.subplots(figsize=(15,7)); ax.set_xlim(0,15); ax.set_ylim(0,7); ax.axis('off')
ax.set_title('Transformer Deep Hedger Architecture',fontsize=16,fontweight='bold',pad=15)
box(ax,0.2,2.5,2,2,'Input\n$I_k \\in \\mathbb{R}^5$',C['blue'])
arr(ax,2.2,3.5,3,3.5)
box(ax,3,2.5,2,2,'Linear\nProjection\n$d_{model}=64$',C['teal'])
arr(ax,5,3.5,5.8,3.5)
# Transformer block
b=FancyBboxPatch((5.8,0.8),3.5,5.5,boxstyle="round,pad=0.15",facecolor=C['purple_bg'],edgecolor=C['purple'],lw=2)
ax.add_patch(b)
ax.text(7.55,5.8,'Transformer Encoder × 3',fontsize=11,fontweight='bold',color=C['purple'],ha='center')
box(ax,6.1,4.2,2.9,1.2,'Multi-Head Self-Attention\n4 heads, causal mask',C['purple'],fs=9)
box(ax,6.1,2.5,2.9,1.2,'Feed-Forward\n$d_{ff} = 256$, ReLU',C['orange'],fs=9)
box(ax,6.1,1.1,2.9,1,'LayerNorm + Dropout(0.1)',C['gray'],tc='white',fs=9)
arr(ax,7.55,4.2,7.55,3.7); arr(ax,7.55,2.5,7.55,2.1)
arr(ax,9.3,3.5,10.2,3.5)
box(ax,10.2,2.5,2.3,2,'Linear\n$d \\to 1$\n$+ \\tanh$',C['orange'])
arr(ax,12.5,3.5,13.3,3.5,col=C['red'])
ax.text(13.6,3.5,'$\\delta_k$',fontsize=18,fontweight='bold',color=C['red'],va='center')
# Positional encoding
box(ax,3,5.5,2,1,'Positional\nEncoding (sin/cos)',C['blue_l'],tc=C['dark'],fs=9)
arr(ax,4,5.5,4,4.5,col=C['blue'])
save(fig,'09_architecture_transformer_hedger.pdf')

# ═══════════════════════════════════════════════════
# 10: RSE Architecture
# ═══════════════════════════════════════════════════
print("[10] RSE Architecture...")
fig,ax=plt.subplots(figsize=(16,9)); ax.set_xlim(0,16); ax.set_ylim(-0.5,9); ax.axis('off')
ax.set_title('Regime-Switching Ensemble (RSE) Architecture',fontsize=16,fontweight='bold',pad=15)
# Input
box(ax,0.2,3.5,2.2,2,'Market Data\n$S_k, v_k, \\tau$',C['blue'])
# Branches
arr(ax,2.4,5,3.5,7); arr(ax,2.4,4.5,3.5,4.5); arr(ax,2.4,4,3.5,2.5); arr(ax,2.4,3.5,3.5,0.5)
# Base models
box(ax,3.5,6.3,2.5,1.5,'Regime Feature\nExtractor\nRV, SMA, BiPV',C['teal'],fs=9)
box(ax,3.5,3.8,2.5,1.5,'LSTM Hedger\n(frozen)',C['gray'],fs=9)
box(ax,3.5,1.8,2.5,1.5,'Transformer\nHedger (frozen)','#78909C',fs=9)
box(ax,3.5,-0.2,2.5,1.5,'Signature\nHedger (frozen)',C['green'],fs=9)
# Regime classifier
arr(ax,6,7,7.5,7)
box(ax,7.5,6,2.5,2,'Regime\nClassifier\nFC(64)+BN\n$K=4$ regimes',C['red'],fs=9)
ax.text(8.75,5.6,'Low vol | High vol\nTrending | Mean-rev',fontsize=8,ha='center',color=C['red_l'])
# Gating
arr(ax,10,7,11,5.5)
arr(ax,6,4.55,11,5); arr(ax,6,2.55,11,4.5); arr(ax,6,0.55,11,4)
box(ax,11,3,3.2,4,'Gating Network\n\n$A \\in \\mathbb{R}^{4 \\times 3}$\n$w_m = \\sum_j p_j \\cdot$\n$[\\mathrm{softmax}(A/\\tau)]_{jm}$\n\n$\\delta^{RSE} = \\sum_m w_m \\delta_m$',C['purple'],fs=9)
# Output
arr(ax,14.2,5,15,5,col=C['red'])
ax.text(15.3,5,'$\\delta_k^{RSE}$',fontsize=18,fontweight='bold',color=C['red'],va='center')
save(fig,'10_architecture_rse.pdf')

# ═══════════════════════════════════════════════════
# 11: W-DRO-T Architecture
# ═══════════════════════════════════════════════════
print("[11] W-DRO-T Architecture...")
fig,ax=plt.subplots(figsize=(15,7)); ax.set_xlim(0,15); ax.set_ylim(0,7); ax.axis('off')
ax.set_title('Wasserstein DRO Transformer (W-DRO-T) Architecture',fontsize=16,fontweight='bold',pad=15)
box(ax,0.2,2.5,2.2,2,'Input $I_k$\n(grad enabled\nfor DRO)',C['blue'],fs=9)
arr(ax,2.4,3.5,3.5,3.5)
box(ax,3.5,2,3,3,'Transformer\nEncoder\n(SDPA Math\nBackend)',C['purple'])
arr(ax,6.5,3.5,7.5,3.5)
box(ax,7.5,2.5,2.3,2,'P&L\nComputation\n$\\sum \\delta_k \\Delta S_k$',C['teal'],fs=9)
# Split into two losses
arr(ax,9.8,4,10.8,5); arr(ax,9.8,3,10.8,1.8)
box(ax,10.8,4.5,3,1.5,'Base Loss\n$\\mathcal{L}_{ent} = \\frac{1}{\\lambda}\\log E[e^{-\\lambda \\cdot PnL}]$',C['orange'],fs=9)
box(ax,10.8,1,3,1.5,'DRO Gradient Penalty\n$\\epsilon \\cdot E[\\|\\nabla_I \\mathcal{L}\\|_2]$',C['red'],fs=9)
# Sum
arr(ax,13.8,5.25,14.3,4); arr(ax,13.8,1.75,14.3,3.5)
box(ax,14.2,3.2,0.7,1.2,'+',C['dark'],fs=14)
# Annealing note
ax.text(12.3,0.3,'$\\epsilon$-annealing: $0 \\to \\epsilon_{max}=0.1$\nover first 15 epochs (linear warmup)',
    fontsize=10,ha='center',color=C['red'],fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3',facecolor=C['red_bg'],edgecolor=C['red'],lw=1))
save(fig,'11_architecture_wdrot.pdf')

# ═══════════════════════════════════════════════════
# 12: 3SCH Architecture
# ═══════════════════════════════════════════════════
print("[12] 3SCH Architecture...")
fig,ax=plt.subplots(figsize=(15,8)); ax.set_xlim(0,15); ax.set_ylim(0,8); ax.axis('off')
ax.set_title('Three-Stage Curriculum Hedger (3SCH) Architecture',fontsize=16,fontweight='bold',pad=15)
# Base model
box(ax,0.3,3,2.5,2,'LSTM Hedger\nBase Model\n$h=50, L=2$',C['purple'])
arr(ax,2.8,4,3.8,4)
# Three stages
box(ax,3.8,5.8,2.8,1.5,'Stage 1: 50 epochs\n$\\mathcal{L} = CVaR_{0.95}(-PnL)$',C['red'],fs=9)
box(ax,3.8,3.5,2.8,1.5,'Stage 2: 20 epochs\n$\\mathcal{L} = \\alpha \\cdot CVaR + (1-\\alpha)\\rho_\\lambda$',C['orange'],fs=9)
box(ax,3.8,1.2,2.8,1.5,'Stage 3: 30 epochs\n$\\mathcal{L} = \\rho_\\lambda + \\gamma|\\Delta\\delta|$',C['green'],fs=9)
# Arrows + detail boxes
arr(ax,6.6,6.55,7.8,6.55); arr(ax,6.6,4.25,7.8,4.25); arr(ax,6.6,1.95,7.8,1.95)
box(ax,7.8,5.8,3.5,1.5,'Tail Risk Awareness\n$\\alpha=1.0$ (pure CVaR)\nLR = $10^{-3}$',C['red_bg'],tc=C['dark'],fs=9,a=0.8)
box(ax,7.8,3.5,3.5,1.5,'Smooth Transition\n$\\alpha: 0.8 \\to 0.2$ (annealing)\nMixed objective',C['orange_bg'],tc=C['dark'],fs=9,a=0.8)
box(ax,7.8,1.2,3.5,1.5,'Fine-Tuning\nEntropic + trading penalty\n$\\gamma = 10^{-3}$, LR = $10^{-4}$',C['green_bg'],tc=C['dark'],fs=9,a=0.8)
# Final output
arr(ax,11.3,6.55,12.3,4.5); arr(ax,11.3,4.25,12.3,4.5); arr(ax,11.3,1.95,12.3,4.5)
box(ax,12.3,3.5,2.2,2,'Final Hedger\n$\\delta_k^{3SCH}$\nRefined via\ncurriculum',C['purple'])
save(fig,'12_architecture_3sch.pdf')

print("\n✅ Part 2 complete: 5 architecture diagrams generated.")
