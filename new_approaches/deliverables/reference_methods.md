# Consolidated Methods from Reference Literature

**Generated:** 2026-01-28  
**Source:** OCR extraction from `reference/` folder (180 pages)

---

## 1. Neural Network Architectures for Hedging

### 1.1 Semi-Recurrent Network (Buehler et al., 2019)
**Source:** Deep Hedging, arXiv:1802.03042, Page 20

**Architecture:**
- Per-timestep MLP: $\delta_k = F_{\theta_k}(I_0, \ldots, I_k, \delta_{k-1})$
- Layers: L = 3
- Dimensions: $N_0 = 2d$, $N_1 = N_2 = d + 15$, $N_3 = d$
- Activation: ReLU with Batch Normalization

**Key Insight:** Weight sharing across time optional but not required.

### 1.2 LSTM Hedger (Kozyra, 2018)
**Source:** Oxford MSc Thesis, Pages 75-80

**Architecture:**
```
LSTM(input_size=state_dim, hidden_size=50, num_layers=2)
→ Linear(hidden_size, 1)
→ δ_max · tanh(·)  # CRITICAL: Bounding
```

**Critical Observation:** Without delta bounding, training explodes due to unbounded positions creating extreme P&L.

### 1.3 Extended Network (Kozyra, 2018)
**Source:** Oxford MSc Thesis, Page 80

**Architecture:** Deep feedforward with bottlenecks
```
Layers: [1, 50, 50, 30, 1, 50, 50, 30, 1]
```

---

## 2. Risk Measures

### 2.1 Entropic Risk Measure
**Source:** Buehler et al., Page 10

$$\rho_\lambda(X) = \frac{1}{\lambda} \log \mathbb{E}[\exp(-\lambda X)]$$

**Properties:**
- Convex ✓
- Cash-invariant: $\rho(X + c) = \rho(X) - c$ ✓
- Monotone: $X \leq Y \Rightarrow \rho(X) \geq \rho(Y)$ ✓
- Differentiable everywhere (good for SGD)

**Gradient Estimator:**
$$\nabla_\theta \rho_\lambda = \mathbb{E}\left[\frac{\exp(-\lambda X)}{\mathbb{E}[\exp(-\lambda X)]} \nabla_\theta(-X)\right]$$

### 2.2 Conditional Value-at-Risk (CVaR / Expected Shortfall)
**Source:** Buehler et al., Page 20

$$\text{CVaR}_\alpha(L) = \frac{1}{1-\alpha} \int_0^{1-\alpha} \text{VaR}_\gamma(L) \, d\gamma = \mathbb{E}[L | L \geq \text{VaR}_\alpha(L)]$$

**Dual Representation (for optimization):**
$$\text{CVaR}_\alpha(X) = \inf_{\eta \in \mathbb{R}} \left\{ \eta + \frac{1}{1-\alpha} \mathbb{E}[(X - \eta)_+] \right\}$$

**Properties:**
- Coherent risk measure ✓
- Sub-additive ✓
- Focuses on tail (worst 1-α fraction)

### 2.3 Optimized Certainty Equivalent (OCE)
**Source:** Buehler et al., Page 10

$$\rho(X) = \inf_{w \in \mathbb{R}} \{w + \mathbb{E}[\ell(-X - w)]\}$$

where $\ell: \mathbb{R} \to \mathbb{R}$ is a loss function (convex, non-decreasing).

**Special case:** $\ell(x) = \exp(\lambda x) - \frac{1+\log(\lambda)}{\lambda}$ gives entropic risk.

---

## 3. Training Protocols

### 3.1 Single-Stage Entropic Training
**Source:** Buehler et al., Page 20

**Objective:**
$$\min_\theta J(\theta) = \mathbb{E}[\exp(-\lambda \cdot \text{P\&L}_\theta)]$$

**Hyperparameters:**
- Optimizer: Adam
- Learning rate: 0.005
- Batch size: 256
- Epochs: Until convergence

### 3.2 Two-Stage Training (Kozyra)
**Source:** Oxford MSc Thesis, Page 85

**Stage 1 (CVaR Pretraining):**
$$\min_\theta \text{CVaR}_{0.95}(-\text{P\&L}_\theta)$$

- Train for 50 epochs
- No transaction cost penalty
- Purpose: Initialize for tail risk awareness

**Stage 2 (Entropic Fine-tuning):**
$$\min_\theta \rho_\lambda(-\text{P\&L}_\theta) + \gamma \sum_k |\delta_{k+1} - \delta_k| + \nu \cdot d(\delta, \mathcal{H}_c)$$

where:
- $\gamma = 10^{-3}$ (trading cost penalty)
- $\nu = 10^8$ (no-trade band penalty)
- $\mathcal{H}_c = [\delta^* - 0.15, \delta^* + 0.15]$ (band around Stage 1 solution)
- $d(\delta, \mathcal{H}_c) = \max(0, |\delta - \delta^*| - 0.15)$

---

## 4. Market Models

### 4.1 Heston Stochastic Volatility
**Source:** Buehler et al., Page 20

$$dS_t = r S_t \, dt + \sqrt{v_t} S_t \, dW_t^S$$
$$dv_t = \kappa(\theta - v_t) \, dt + \sigma \sqrt{v_t} \, dW_t^v$$
$$\text{corr}(W^S, W^v) = \rho$$

**Standard Parameters:**
| Parameter | Value | Meaning |
|-----------|-------|---------|
| $S_0$ | 100 | Initial stock price |
| $v_0$ | 0.04 | Initial variance (σ = 20%) |
| $\kappa$ | 1.0 | Mean reversion speed |
| $\theta$ | 0.04 | Long-term variance |
| $\sigma$ | 0.2 | Vol of vol |
| $\rho$ | -0.7 | Price-vol correlation |
| $r$ | 0.0 | Risk-free rate |

**Feller Condition:** $2\kappa\theta > \sigma^2$ ensures v_t > 0.

### 4.2 Black-Scholes (Baseline)
**Source:** Buehler et al., Page 3

$$dS_t = r S_t \, dt + \sigma S_t \, dW_t$$

**Analytical Delta:** $\Delta^{BS} = N(d_1)$ where
$$d_1 = \frac{\log(S/K) + (r + \frac{\sigma^2}{2})\tau}{\sigma\sqrt{\tau}}$$

---

## 5. Transaction Cost Models

### 5.1 Proportional Costs
**Source:** Buehler et al., Page 5

$$C_T(\delta) = \sum_{k=0}^{N} \kappa |\delta_k - \delta_{k-1}| S_{t_k}$$

where $\kappa$ is the proportional cost (typically 0.001 = 10 bps).

### 5.2 Leland Adjusted Delta
**Source:** Kozyra thesis, Page 70

Modify BS volatility to account for rebalancing costs:
$$\tilde{\sigma} = \sigma \sqrt{1 + \sqrt{\frac{2}{\pi}} \frac{\kappa}{\sigma\sqrt{\Delta t}}}$$

Then use $\Delta^{Leland} = N(\tilde{d}_1)$.

### 5.3 Whalley-Wilmott No-Trade Band
**Source:** Kozyra thesis, Page 70

Don't trade if delta is within band:
$$|\Delta^{current} - \Delta^{BS}| < H$$

where $H \propto (\kappa / \gamma)^{1/3}$ (transaction cost / gamma ratio).

---

## 6. Distributionally Robust Methods

### 6.1 Robust Deep Hedging (Lütkebohmert et al., 2021)
**Source:** arXiv:2106.10024, Page 154

**Key Idea:** Handle parameter uncertainty via worst-case optimization.

**Uncertainty Set:** Affine processes with parameter ranges
$$\mathcal{P} = \{P_\theta : \theta \in \Theta\}$$

**Robust Objective:**
$$\min_\delta \sup_{P \in \mathcal{P}} \rho_P(-\text{P\&L}(\delta))$$

**Connection to DRO:** Uses nonlinear expectation framework linked to Kolmogorov equation.

---

## 7. Signature Methods (from Rough Path Theory)

### 7.1 Path Signature
**Source:** Referenced in multiple papers

For path $X: [0,T] \to \mathbb{R}^d$, the signature of depth $m$:
$$S^{(m)}(X)_{[s,t]} = \left(1, \int_s^t dX^{i_1}, \int_s^t \int_s^{u_2} dX^{i_1} dX^{i_2}, \ldots\right)$$

**Properties:**
- Universal nonlinearity (characterizes paths up to tree-like equivalence)
- Signature kernel: $k(X, Y) = \langle S(X), S(Y) \rangle$

### 7.2 Log-Signature
More compact than full signature (dimension $\approx d^m / m$ vs $d^m$).

---

## 8. Reinforcement Learning Approaches

### 8.1 Market Impact RL (JRFM 2025)
**Source:** J. Risk Financial Manag., Page 130

**Key Innovation:** Model hedging feedback loop explicitly.

**Problem:** Buying shares to hedge → increases underlying price → increases option value → need more hedging

**Approach:** RL agent learns to minimize combined:
- Hedging error
- Market impact cost
- Transaction costs

---

## 9. Evaluation Metrics (Standard)

| Metric | Formula | Purpose |
|--------|---------|---------|
| Mean P&L | $\mathbb{E}[\text{P\&L}]$ | Average performance |
| Std P&L | $\sqrt{\text{Var}[\text{P\&L}]}$ | Risk |
| VaR_α | $\inf\{x: P(\text{P\&L} \leq x) \geq \alpha\}$ | Tail threshold |
| CVaR_α | $\mathbb{E}[\text{P\&L} | \text{P\&L} \leq \text{VaR}_\alpha]$ | Expected tail loss |
| Entropic Risk | $(1/\lambda)\log\mathbb{E}[\exp(-\lambda \text{P\&L})]$ | Utility-based |
| Trading Volume | $\sum_k |\delta_{k+1} - \delta_k|$ | Turnover |
| Max Drawdown | $\max_t (\text{peak}_t - \text{value}_t)$ | Worst peak-to-trough |

---

## 10. Gap Analysis: Methods NOT in Current Codebase

| Method | Source | Priority | Implementation Effort |
|--------|--------|----------|----------------------|
| DRO with adversarial training | Lütkebohmert 2021 | HIGH | Medium |
| Rough volatility (rBergomi) | Gatheral 2018 | MEDIUM | Medium |
| Merton jump-diffusion | Standard | HIGH | Low |
| Meta-learning (MAML) | Finn 2017 | MEDIUM | High |
| Risk-aware attention | Novel | HIGH | Medium |
| Signature kernels | Kidger 2020 | LOW | High |
| Market impact modeling | JRFM 2025 | MEDIUM | Medium |

---

*This document consolidates all methods identified via OCR from the reference papers. Use as authoritative source for method citations.*
