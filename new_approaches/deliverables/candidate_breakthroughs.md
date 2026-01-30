# Candidate Breakthrough Algorithms for Deep Hedging

**Generated:** 2026-01-28  
**Based on:** 
- OCR extraction from 7 reference papers (180 pages)
- Comprehensive analysis of existing pipeline (paper.tex, AUDIT_SUMMARY.md, FINAL_REPORT.md)
- Publication-quality baseline results (10 seeds, 50K samples)

**Status:** REFINED DESIGN PHASE

---

## Executive Summary

### Existing Pipeline Key Results (Baseline to Beat)

From `AUDIT_SUMMARY.md` (publication-quality, 10 seeds):

| Model | CVaR95 | vs LSTM | p-value | Cohen's d |
|-------|--------|---------|---------|-----------|
| **Transformer** | 4.41 ± 0.03 | **-3.1%** | 0.029 | -0.87 |
| LSTM (baseline) | 4.43 ± 0.02 | — | — | — |
| AttentionLSTM | 4.44 ± 0.03 | +0.2% | 0.322 | 0.35 |
| SignatureLSTM | 4.44 ± 0.02 | +0.2% | 0.275 | 0.39 |

**Critical Insight:** Transformer already achieves best CVaR95 (-3.1%). Novel approaches must demonstrate >5% improvement to be meaningfully better.

### Grounded Novel Approaches

Based on comprehensive literature review and gap analysis, we propose **5 novel algorithmic approaches** that address specific limitations identified in the existing pipeline:

| Rank | Algorithm | Grounding | Target Gap | Expected Additional Improvement |
|------|-----------|-----------|------------|--------------------------------|
| 1 | Wasserstein DRO Transformer (W-DRO-T) | Lütkebohmert 2021 | Parameter uncertainty | 3-8% additional CVaR reduction |
| 2 | Rough Volatility Signature Network (RVSN) | Abi Jaber 2025, SigFormer 2023 | Non-Markovian dynamics | 5-10% on rBergomi paths |
| 3 | SAC with CVaR Constraints (SAC-CVaR) | Huang 2025, Neagu 2025 | Explicit tail constraints | 5-12% with market impact |
| 4 | Three-Stage Curriculum Hedger (3SCH) | Kozyra 2018 extended | Training stability | 2-5% via better curriculum |
| 5 | Regime-Switching Ensemble (RSE) | Pipeline gap | Distribution shift | 3-7% in stress scenarios |

---

## Why These Specific Approaches?

### Gap Analysis from Existing Pipeline

| Gap | Evidence | Reference |
|-----|----------|-----------|
| **No robustness to parameter uncertainty** | Heston params fixed during training | Lütkebohmert 2021: "robust deep hedging outperforms in volatile periods" |
| **No rough volatility** | Only Heston (Markovian) tested | Abi Jaber 2025: "signatures outperform LSTMs on non-Markovian paths" |
| **No explicit CVaR constraints in RL** | MCPG/PPO in codebase use expected reward | Huang 2025: "SAC achieves 50% CVaR reduction" |
| **Two-stage may be suboptimal** | Current: CVaR→Entropic | Could add intermediate risk-aware stage |
| **No regime-aware adaptation** | Single model for all market conditions | Stress periods need different behavior |

### Key Insight from FINAL_REPORT.md

> "Why Complex Models Failed:
> 1. Signature features add noise
> 2. Attention over-parameterizes
> 3. LSTM is near-optimal for this task"

**Implication:** Novel approaches must be *principled* improvements, not just architectural complexity. Each proposal below addresses a specific, identified gap with theoretical grounding.

---

## Candidate 1: Wasserstein DRO Transformer (W-DRO-T)

### 1.1 Motivation

**From Lütkebohmert et al. 2021 (pages 154-180):** "The robust deep hedging approach leads to a remarkably smaller hedging error in comparison to classical hedging strategies" during the COVID-19 period.

**Gap in Current Pipeline:** The existing Transformer is trained on fixed Heston parameters. When deployed in real markets with parameter uncertainty, performance degrades. We propose incorporating **Wasserstein Distributionally Robust Optimization (DRO)** into training.

### 1.2 Mathematical Formulation

**Standard Deep Hedging Objective:**
$$\min_\theta \mathbb{E}_{P}[\rho(-\text{P\&L}_\theta)]$$

**Wasserstein DRO Objective:**
$$\min_\theta \sup_{Q \in \mathcal{B}_\epsilon(P)} \mathbb{E}_{Q}[\rho(-\text{P\&L}_\theta)]$$

where $\mathcal{B}_\epsilon(P) = \{Q : W_2(P, Q) \leq \epsilon\}$ is a Wasserstein-2 ball around training distribution.

**Tractable Dual Form (Blanchet & Murthy 2019):**
$$\sup_{Q \in \mathcal{B}_\epsilon(P)} \mathbb{E}_Q[f(X)] \approx \mathbb{E}_P[f(X)] + \epsilon \cdot \|\nabla_X f(X)\|_2$$

**W-DRO-T Training Loss:**
$$\mathcal{L}_{DRO}(\theta) = \mathcal{L}_{entropic}(\theta) + \epsilon \cdot \mathbb{E}\left[\|\nabla_{I} \mathcal{L}_{entropic}(\theta; I)\|_2\right]$$

where $I = (S, v, \tau, \delta_{prev})$ are input features and $\epsilon$ is the robustness radius.

### 1.3 Uncertainty Set for Heston Parameters

Following Lütkebohmert et al., we define parameter uncertainty:
$$\Theta = \{(\kappa, \theta, \xi, \rho) : \kappa \in [0.5, 4.0], \theta \in [0.02, 0.08], \xi \in [0.1, 0.4], \rho \in [-0.9, -0.5]\}$$

During training, we sample parameters uniformly from $\Theta$ to create distributional shift.

### 1.4 Architecture

Base architecture: **Existing Transformer** from `src/models/transformer.py` with DRO wrapper.

```
W-DRO-T Architecture:

1. Base Model: TransformerHedge (d_model=64, n_heads=4, n_layers=3)
2. DRO Wrapper:
   - Forward pass: δ = Transformer(X)
   - Compute loss: L = entropic_loss(δ, X)
   - Compute input gradients: g = ∇_X L
   - DRO regularizer: R = ε * ||g||_2
   - Total loss: L_DRO = L + R
```

### 1.5 Training Protocol

**Three-Phase Training (extends existing two-stage):**
1. **Phase 1 (CVaR Pretraining):** 50 epochs, standard CVaR loss (as in existing pipeline)
2. **Phase 2 (Entropic + DRO):** 30 epochs, $\mathcal{L}_{DRO}$ with $\epsilon$ annealed from 0 to 0.1
3. **Phase 3 (Stress Testing):** 10 epochs, $\epsilon = 0.2$ on adversarial Heston parameters

### 1.6 Expected Improvement

- **Target:** 3-8% additional CVaR95 reduction over vanilla Transformer (4.41 → ~4.10)
- **Mechanism:** Robustness to parameter uncertainty reduces tail losses during regime shifts
- **Computational overhead:** ~20% additional training time (gradient computation)

### 1.7 Pseudocode

```python
class WassersteinDROLoss(nn.Module):
    def __init__(self, base_loss, epsilon=0.1):
        self.base_loss = base_loss  # EntropicLoss from src/train/losses.py
        self.epsilon = epsilon
    
    def forward(self, model, inputs, targets):
        inputs.requires_grad_(True)
        
        # Forward pass
        outputs = model(inputs)
        loss = self.base_loss(outputs, targets)
        
        # Compute input gradients for DRO regularization
        grad_inputs = torch.autograd.grad(loss, inputs, create_graph=True)[0]
        dro_penalty = self.epsilon * grad_inputs.norm(p=2, dim=-1).mean()
        
        return loss + dro_penalty
```

---

## Candidate 2: Rough Volatility Signature Network (RVSN)

### 2.1 Motivation

**From Abi Jaber & Gérard 2025 (pages 33-64):** "Signatures outperform LSTMs with orders of magnitude less training compute" on non-Markovian stochastic volatility models like rough Bergomi.

**Gap in Current Pipeline:** 
- Existing SignatureLSTM/SignatureMLP showed **no improvement** over LSTM (CVaR95 +0.2-0.7%)
- **Reason:** Tested only on Heston (Markovian) - signatures excel on non-Markovian paths
- Current implementation uses fixed depth-3 signatures

**Our Proposal:** Test signatures on rough volatility models where they theoretically excel, with adaptive truncation.

### 2.2 Mathematical Formulation

**Rough Bergomi Model (from Abi Jaber 2025, SigFormer 2023):**
$$dS_t = S_t\sqrt{V_t}dW_t$$
$$V_t = \xi \exp\left(\eta W_t^H - \frac{1}{2}\eta^2 t^{2H}\right)$$

where $W^H$ is fractional Brownian motion with Hurst parameter $H \in (0, 0.5)$ (rough regime).

**Path Signature (truncated to order M):**
$$S^{(M)}(X)_{[0,t]} = \left(1, \int_0^t dX, \int_0^t \int_0^{s_1} dX \otimes dX, \ldots\right)$$

**Key Insight from Abi Jaber:** Signatures capture path-dependent information without recurrence, making them ideal for non-Markovian dynamics.

### 2.3 Adaptive Signature Truncation

**Problem:** Fixed depth-3 may be suboptimal. Higher orders needed for rough paths.

**Solution:** Learn optimal truncation per market regime:
$$\hat{S}(X)_t = \sum_{m=1}^{M_{max}} w_m(\text{regime}_t) \cdot S^{(m)}(X)_{[0,t]}$$

where $w_m$ are gating weights based on local Hurst estimate:
$$\hat{H}_t = \frac{\log(\text{RV}_t^{(2)}/\text{RV}_t^{(1)})}{\log(2)}$$

### 2.4 Architecture

```
RVSN Architecture:

1. Rough Vol Simulator: Generate rBergomi paths (H=0.1, 0.2, 0.3)
2. Signature Computation: signatory.signature(path, depth=1..5)
3. Hurst Estimator: Estimate local H from realized variation ratio
4. Adaptive Gating: w = softmax(MLP([H_local, vol_local]))
5. Weighted Signature: S_adaptive = Σ w_m * Sig^m
6. Hedging MLP: δ = δ_max * tanh(MLP(S_adaptive))
```

### 2.5 Training Protocol

**Data Generation:**
- Train on mixture: 50% Heston, 25% rBergomi (H=0.1), 25% rBergomi (H=0.3)
- This forces model to learn regime-adaptive behavior

**Loss:** Standard two-stage (CVaR → Entropic) from existing pipeline

### 2.6 Expected Improvement

- **Target:** 5-10% CVaR improvement **on rBergomi paths** (where signatures excel)
- **On Heston:** Expect similar to baseline (signatures don't help for Markovian)
- **Key test:** Real market data often exhibits rough volatility (H ≈ 0.1)

### 2.7 Pseudocode

```python
import signatory

class AdaptiveSignatureHedger(nn.Module):
    def __init__(self, max_depth=5, hidden_dim=64):
        self.max_depth = max_depth
        self.hurst_estimator = nn.Sequential(
            nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.gating = nn.Linear(2, max_depth)  # [H, vol] -> weights
        self.hedger = nn.Sequential(
            nn.Linear(self._sig_dim(), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        self.delta_max = 1.5
    
    def forward(self, path, returns):
        # Estimate local Hurst
        H_local = self.hurst_estimator(returns[:, -10:])
        vol_local = returns[:, -20:].std(dim=-1, keepdim=True)
        
        # Compute signatures at multiple depths
        sigs = [signatory.signature(path, depth=d) for d in range(1, self.max_depth+1)]
        
        # Adaptive weighting
        weights = F.softmax(self.gating(torch.cat([H_local, vol_local], dim=-1)), dim=-1)
        sig_adaptive = sum(w * s for w, s in zip(weights.T, sigs))
        
        # Hedging decision
        return self.delta_max * self.hedger(sig_adaptive)
```

### 2.8 Why This Addresses Pipeline Gap

From FINAL_REPORT.md: "Signature features add noise" - **True for Heston, but signatures excel on rough volatility.** This candidate tests the hypothesis that signatures are being evaluated on the wrong market model.

---

## Candidate 3: CVaR-Constrained Soft Actor-Critic (SAC-CVaR)

### 3.1 Motivation

**From Huang & Lawryshyn 2025 (pages 129-153):** "SAC achieves 50% reduction in CVaR" compared to traditional delta hedging when market impact is modeled.

**From Neagu et al. 2025 (pages 120-128):** "MCPG obtains best performance overall... only algorithm to outperform BS delta baseline."

**Gap in Current Pipeline:**
- Existing RL agents (MCPG, PPO, DDPG/TD3 in `src/models/rl_agents.py`) optimize expected reward
- No explicit CVaR constraint in policy optimization
- From FINAL_REPORT.md: "MCPG needs tuning" - RL not properly configured

**Our Proposal:** Implement SAC with explicit CVaR constraints via Lagrangian relaxation.

### 3.2 Mathematical Formulation

**Standard SAC Objective:**
$$\max_\pi \mathbb{E}_\pi\left[\sum_t \gamma^t (r_t + \alpha H(\pi(\cdot|s_t)))\right]$$

where $H(\pi)$ is entropy bonus and $\alpha$ is temperature.

**CVaR-Constrained SAC:**
$$\max_\pi \mathbb{E}_\pi\left[\sum_t \gamma^t r_t\right] \quad \text{s.t.} \quad \text{CVaR}_{0.95}(-R_{episode}) \leq c$$

**Lagrangian Relaxation:**
$$\mathcal{L}(\pi, \lambda) = \mathbb{E}_\pi[R] + \alpha H(\pi) - \lambda \cdot (\text{CVaR}_{0.95}(-R) - c)$$

where $\lambda \geq 0$ is learned via dual gradient ascent.

### 3.3 CVaR Estimation via Distributional RL

Use quantile regression to estimate return distribution:
$$Z_\theta(s, a) = \{z_1, z_2, \ldots, z_N\}$$ (N quantile atoms)

CVaR computed from quantile estimates:
$$\text{CVaR}_{0.95} \approx \frac{1}{0.05 \cdot N} \sum_{i: \tau_i \geq 0.95} z_i$$

### 3.4 Architecture

Build on existing `src/models/rl_agents.py` with distributional critic:

```
SAC-CVaR Architecture:

1. Policy Network: π_φ(a|s) - Gaussian policy (existing)
2. Twin Q-Networks: Q_θ1, Q_θ2 (existing SAC structure)
3. Quantile Critic: Z_ψ(s,a) = {z_1, ..., z_51} (NEW - 51 quantile atoms)
4. Lagrange Multiplier: λ (learnable, initialized to 0.1)
5. CVaR Threshold: c = baseline_CVaR * 0.9 (target 10% improvement)
```

### 3.5 Training Protocol

**Reward Shaping for Hedging:**
$$r_t = -|\delta_t(S_{t+1} - S_t) - \Delta\text{Option}_t| - \kappa|\delta_t - \delta_{t-1}|S_t$$

**Episode:** One hedging period (30 steps). Terminal reward includes option payoff.

**Hyperparameters (from Neagu 2025, Huang 2025):**
- Learning rate: 0.0001 (slower than supervised)
- Batch size: 256
- Replay buffer: 100,000
- Target update: τ = 0.005
- Entropy: α auto-tuned

### 3.6 Expected Improvement

- **Target:** 5-12% CVaR reduction over Transformer baseline
- **Mechanism:** Explicit CVaR constraint forces policy to avoid tail losses
- **Key advantage:** Handles market impact (Huang 2025 showed 50% reduction)

### 3.7 Pseudocode

```python
class CVaRConstrainedSAC:
    def __init__(self, state_dim, action_dim, cvar_threshold):
        self.actor = GaussianPolicy(state_dim, action_dim)
        self.critic1 = QNetwork(state_dim, action_dim)
        self.critic2 = QNetwork(state_dim, action_dim)
        self.quantile_critic = QuantileNetwork(state_dim, action_dim, n_quantiles=51)
        self.log_lambda = nn.Parameter(torch.tensor(-2.0))  # λ = exp(log_λ)
        self.cvar_threshold = cvar_threshold
    
    def compute_cvar(self, states, actions):
        quantiles = self.quantile_critic(states, actions)  # [B, 51]
        # CVaR95 = mean of top 5% quantiles (indices 48-50)
        return quantiles[:, 48:].mean(dim=-1)
    
    def update(self, batch):
        # Standard SAC critic and actor updates...
        
        # Quantile critic update (IQN-style)
        quantile_loss = self._quantile_huber_loss(batch)
        
        # Lagrange multiplier update
        cvar = self.compute_cvar(batch.states, batch.actions)
        constraint_violation = cvar.mean() - self.cvar_threshold
        lambda_loss = -self.log_lambda * constraint_violation.detach()
        
        # Policy includes CVaR penalty
        lmbda = self.log_lambda.exp()
        policy_loss = -Q_value + self.alpha * log_prob + lmbda * cvar
```

---

## Candidate 4: Three-Stage Curriculum Hedger (3SCH)

### 4.1 Motivation

**From Existing Pipeline (paper.tex, lines 242-265):** Two-stage training (CVaR → Entropic) is critical for performance.

**Gap:** Current protocol jumps from CVaR to entropic risk. An intermediate stage could smooth the transition and improve final performance.

**Our Proposal:** Add a third stage with mixed CVaR+Entropic objective before pure entropic fine-tuning.

### 4.2 Three-Stage Protocol

| Stage | Epochs | Loss | Description |
|-------|--------|------|-------------|
| 1 | 50 | CVaR95 | Tail risk awareness (existing) |
| 2 | 20 | 0.5*CVaR + 0.5*Entropic | Smooth transition (NEW) |
| 3 | 30 | Entropic + trading penalty | Fine-tuning (existing) |

### 4.3 Mathematical Formulation

**Stage 2 Mixed Loss:**
$$\mathcal{L}_2 = \alpha \cdot \text{CVaR}_{0.95}(-\text{P\&L}) + (1-\alpha) \cdot \rho_\lambda(-\text{P\&L})$$

with $\alpha$ linearly annealed from 0.8 to 0.2 over the stage.

### 4.4 Expected Improvement

- **Target:** 2-5% additional CVaR reduction
- **Mechanism:** Smoother loss landscape transition reduces local minima trapping
- **Computational cost:** ~25% more training time (extra 20 epochs)

### 4.5 Implementation

Minimal change to existing `src/train/trainer.py`:

```python
class ThreeStageTrainer(DeepHedgingTrainer):
    def train(self, model, train_loader, val_loader):
        # Stage 1: CVaR pretraining (existing)
        self.loss_fn = CVaRLoss(alpha=0.95)
        self._train_stage(model, train_loader, val_loader, epochs=50)
        
        # Stage 2: Mixed (NEW)
        for epoch in range(20):
            alpha = 0.8 - 0.6 * (epoch / 20)  # 0.8 → 0.2
            self.loss_fn = MixedLoss(cvar_weight=alpha)
            self._train_epoch(model, train_loader)
        
        # Stage 3: Entropic fine-tuning (existing)
        self.loss_fn = EntropicLoss(lambda_=1.0)
        self._train_stage(model, train_loader, val_loader, epochs=30)
```

---

## Candidate 5: Regime-Switching Ensemble (RSE)

### 5.1 Motivation

Different models excel in different market regimes (from AUDIT_SUMMARY.md):
- **LSTM:** Best in trending markets
- **Transformer:** Best for tail risk in volatile markets
- **Signature models:** Best for path-dependent exotic payoffs

We propose a **regime-detection ensemble** that dynamically weights model contributions.

### 5.2 Mathematical Formulation

**Ensemble Delta:**
$$\delta_t^{ensemble} = \sum_{m=1}^{M} w_m(s_t) \cdot \delta_t^{(m)}$$

where $w_m(s_t)$ are regime-dependent weights and $s_t$ is market state.

**Regime Classifier:**
$$p(regime | s_t) = \text{softmax}(\text{MLP}_{regime}(s_t))$$

**Regime-to-Model Mapping (learned):**
$$w_m(s_t) = \sum_{r=1}^{R} p(regime=r | s_t) \cdot W_{r,m}$$

where $W \in \mathbb{R}^{R \times M}$ is a learnable regime-model affinity matrix.

### 5.3 Regime Features

```python
def compute_regime_features(prices, window=20):
    returns = np.diff(np.log(prices), axis=1)
    
    # Volatility regime
    realized_vol = returns[:, -window:].std(axis=1)
    vol_regime = (realized_vol > realized_vol.mean()).astype(float)
    
    # Trend regime
    sma_short = prices[:, -5:].mean(axis=1)
    sma_long = prices[:, -20:].mean(axis=1)
    trend_regime = (sma_short > sma_long).astype(float)
    
    # Jump regime (using Barndorff-Nielsen statistic)
    bv = np.abs(returns[:, :-1] * returns[:, 1:]).sum(axis=1) * np.pi / 2
    rv = (returns ** 2).sum(axis=1)
    jump_regime = (rv > 2 * bv).astype(float)
    
    return np.stack([vol_regime, trend_regime, jump_regime], axis=1)
```

### 5.4 Training Protocol

1. **Pre-train individual models:** Train LSTM, Transformer, SignatureMLP independently
2. **Freeze base models:** Lock weights of pre-trained hedgers
3. **Train gating network:** Learn regime classifier and affinity matrix end-to-end

### 5.5 Expected Advantages

- **Best of all worlds:** Leverage strengths of different architectures
- **Interpretability:** Regime detection provides market insights
- **Robustness:** Ensemble reduces variance of hedging errors

---

## Implementation Priority

Based on novelty, feasibility, and expected impact, we recommend implementing in this order:

| Priority | Algorithm | Rationale | Risk Level |
|----------|-----------|-----------|------------|
| 1 | **W-DRO-T** | Minimal code change, wraps existing Transformer | Low |
| 2 | **3SCH** | Simple training protocol change, no new architecture | Low |
| 3 | **SAC-CVaR** | Extends existing RL agents, addresses key gap | Medium |
| 4 | **RVSN** | Requires rough volatility simulator, higher risk | Medium |
| 5 | **RSE** | Requires pre-trained models, orchestration complexity | Medium |

### Required Dependencies

```python
# Core (already in requirements.txt)
torch>=2.0.0
signatory>=1.2.0  # For signature computation

# RL (already in requirements.txt)
stable-baselines3>=2.0.0

# NEW dependencies
fbm>=0.3.0  # For fractional Brownian motion (rough vol)
```

### Estimated Implementation Time

| Algorithm | Implementation | Testing | Total | Files Modified |
|-----------|----------------|---------|-------|----------------|
| W-DRO-T | 2 hours | 2 hours | 4 hours | `src/train/losses.py` |
| 3SCH | 1 hour | 1 hour | 2 hours | `src/train/trainer.py` |
| SAC-CVaR | 4 hours | 3 hours | 7 hours | `src/models/rl_agents.py` |
| RVSN | 3 hours | 2 hours | 5 hours | `src/env/`, `src/models/` |
| RSE | 3 hours | 2 hours | 5 hours | New file |

---

## Summary: Grounded in Pipeline Understanding

### What We Learned from Existing Pipeline

| Finding | Source | Implication |
|---------|--------|-------------|
| Transformer best CVaR95 (4.41) | AUDIT_SUMMARY.md | Beat this baseline |
| Signatures failed on Heston | FINAL_REPORT.md | Test on rough vol instead |
| Two-stage training critical | paper.tex | Extend, don't replace |
| LSTM near-optimal for Markovian | FINAL_REPORT.md | Focus on non-Markovian |
| All models reduce trading vol | AUDIT_SUMMARY.md | Keep this property |

### How Each Candidate Addresses Gaps

| Candidate | Gap Addressed | Reference Paper | Expected Improvement |
|-----------|---------------|-----------------|---------------------|
| W-DRO-T | Parameter uncertainty | Lütkebohmert 2021 | 3-8% CVaR reduction |
| RVSN | Non-Markovian dynamics | Abi Jaber 2025 | 5-10% on rough vol |
| SAC-CVaR | Explicit tail constraints | Huang 2025, Neagu 2025 | 5-12% with impact |
| 3SCH | Training stability | Kozyra 2018 | 2-5% CVaR reduction |
| RSE | Regime adaptation | Pipeline gap | 3-7% in stress |

### Theoretical Guarantees

**W-DRO-T Generalization Bound (Blanchet & Murthy 2019):**
$$\sup_{Q: W_2(P_{train}, Q) \leq \epsilon} \mathbb{E}_Q[\mathcal{L}(\theta^*)] \leq \mathbb{E}_{P_{train}}[\mathcal{L}(\theta^*)] + \epsilon \cdot L_\theta$$

where $L_\theta$ is the Lipschitz constant of the loss.

**RVSN Universal Approximation (from rough path theory):**
Path signatures provide a universal basis for continuous path functionals, making them ideal for non-Markovian hedging problems.

---

**Document Status:** ✅ REFINED & COMPLETE

**Grounding:**
- ✅ Reference papers (7 papers, 180 pages OCR'd)
- ✅ Existing pipeline (paper.tex, AUDIT_SUMMARY.md, FINAL_REPORT.md)
- ✅ Codebase understanding (src/models/, src/train/, experiments/)

*Ready for implementation. Top 3 candidates: W-DRO-T, 3SCH, SAC-CVaR.*
