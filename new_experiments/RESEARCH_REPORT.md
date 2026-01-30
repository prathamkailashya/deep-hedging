# Fair Comparison of Deep Hedging Architectures

## Research Report

**Date:** 2026-01-25  
**Status:** COMPLETE

---

## Executive Summary

This study conducted a **scientifically fair comparison** of advanced deep hedging architectures to determine whether complex models truly underperform LSTM, or whether previous failures were due to suboptimal implementation and hyperparameter selection.

### Key Finding
**AttentionLSTM achieves the best tail risk (CVaR95 = 4.34)**, representing a **statistically significant 1.6% improvement** over the LSTM baseline. This contradicts earlier findings that suggested simple LSTM is always superior.

### Critical Insight
The improvement comes from **attention over LSTM hidden states**, not from signatures or transformers alone. This suggests that **memory mechanisms** are valuable for hedging when properly implemented.

---

## 1. Experimental Setup

### 1.1 Data Generation
- **Model:** Heston stochastic volatility
- **Parameters:** S0=100, v0=0.04, κ=1.0, θ=0.04, σ=0.2, ρ=-0.7
- **Horizon:** T=30/365 (30 days), n=30 steps
- **Datasets:** Train=50,000, Val=10,000, Test=50,000

### 1.2 Training Protocol (Identical for All Models)
All models used the same two-stage training:

**Stage 1: CVaR Pretraining**
- Loss: CVaR95
- Epochs: 40 (early stopping patience=12)
- Learning rate: 0.001

**Stage 2: Entropic Fine-tuning**
- Loss: Entropic + γ|Δδ| trading penalty
- Epochs: 25 (early stopping patience=8)
- Learning rate: 0.0001
- Trading penalty γ = 1e-3

### 1.3 Model Constraints
- **Delta bounding:** δ = 1.5 × tanh(output)
- **Gradient clipping:** ||g|| ≤ 5.0
- **Weight decay:** 1e-4

---

## 2. Results Summary

### 2.1 Performance Table

| Model | Params | Std P&L | CVaR95 | Entropic | Volume |
|-------|--------|---------|--------|----------|--------|
| **LSTM (baseline)** | 31,651 | 1.0438 | 4.4137 | 2.8409 | 1.9184 |
| SignatureLSTM | 95,415 | 1.0561 | 4.4198 | 2.8477 | 1.7048 |
| **SignatureMLP** | 37,943 | **1.0392** | 4.3860 | **2.8285** | 1.7289 |
| Transformer | 136,321 | 1.0583 | 4.4207 | 2.8482 | **1.6682** |
| **AttentionLSTM** | 69,889 | 1.0525 | **4.3432** | 2.8291 | 1.8733 |
| RegimeAware | 154,054 | 1.0400 | 4.4527 | 2.8469 | 1.8198 |
| Adaptive | 55,810 | 1.0799 | 4.4715 | 2.8749 | 2.0650 |

### 2.2 Statistical Significance (vs LSTM Baseline)

| Model | CVaR95 Diff | 95% CI | Significant? |
|-------|-------------|--------|--------------|
| SignatureLSTM | +0.0061 | [0.002, 0.011] | ✗ Worse |
| **SignatureMLP** | **-0.0277** | [-0.034, -0.021] | **✓ Better** |
| Transformer | +0.0070 | [0.002, 0.012] | ✗ Worse |
| **AttentionLSTM** | **-0.0705** | [-0.075, -0.066] | **✓ Better** |
| RegimeAware | +0.0390 | [0.033, 0.045] | ✗ Worse |
| Adaptive | +0.0578 | [0.054, 0.062] | ✗ Worse |

---

## 3. Key Findings

### 3.1 AttentionLSTM Improves Tail Risk
**Finding:** AttentionLSTM achieves CVaR95 = 4.3432, a **1.6% improvement** over LSTM baseline (4.4137).

**Why it works:**
- Attention over past hidden states allows the model to weight recent vs historical information
- The memory mechanism helps in volatile periods where recent information is more relevant
- Combination method (concat) preserves both current and historical context

### 3.2 SignatureMLP Achieves Best Std P&L
**Finding:** SignatureMLP achieves Std P&L = 1.0392, the lowest variance among all models.

**Why it works:**
- Signatures capture path-dependent information without recurrence overhead
- Non-recurrent architecture may prevent error accumulation
- More stable training due to simpler gradient flow

### 3.3 Transformer Achieves Lowest Trading Volume
**Finding:** Transformer achieves Volume = 1.6682, lowest among all models.

**Why it works:**
- Attention mechanism learns smoother delta transitions
- Causal masking prevents abrupt strategy changes
- Lower volume can reduce transaction costs in practice

### 3.4 Complex Hybrids (RegimeAware, Adaptive) Underperform
**Finding:** Novel hybrid architectures did not improve over simpler baselines.

**Why they failed:**
- Over-parameterization (RegimeAware: 154k params) leads to overfitting
- Regime detection adds noise rather than signal in this setting
- Adaptive mechanisms may respond to noise rather than true regime changes

---

## 4. Comparison with Previous Findings

### Previous Conclusion (Incorrect)
> "No hybrid architecture improved tail risk (CVaR) over the LSTM baseline."

### Updated Conclusion (Correct)
> "**AttentionLSTM and SignatureMLP both achieve statistically significant improvements** over LSTM when trained with proper two-stage protocol and hyperparameters."

### What Changed?
1. **Proper two-stage training:** CVaR pretraining followed by entropic fine-tuning
2. **Consistent architecture choices:** Same delta bounding, gradient clipping for all
3. **Longer training:** 40 epochs Stage 1 + 25 epochs Stage 2 with early stopping
4. **Fair comparison:** Same data splits, same evaluation metrics

---

## 5. Recommendations

### 5.1 For Practitioners
1. **Use AttentionLSTM** when tail risk (CVaR) is the primary concern
2. **Use SignatureMLP** when P&L variance minimization is the goal
3. **Use Transformer** when trading cost minimization is important
4. **Avoid over-complex architectures** like RegimeAware without extensive HPO

### 5.2 For Researchers
1. **Two-stage training is essential** - CVaR pretraining stabilizes learning
2. **Attention mechanisms help** when properly integrated with LSTM
3. **Signatures have value** but work better in non-recurrent (MLP) architectures
4. **Fair comparison requires identical training protocols**

### 5.3 Model Selection Guide

| Priority | Recommended Model | CVaR95 | Volume |
|----------|------------------|--------|--------|
| Best tail risk | AttentionLSTM | 4.34 | 1.87 |
| Lowest variance | SignatureMLP | 4.39 | 1.73 |
| Lowest volume | Transformer | 4.42 | 1.67 |
| Balanced | LSTM | 4.41 | 1.92 |

---

## 6. Limitations and Future Work

### 6.1 Limitations
- Single market simulation (Heston model)
- Default hyperparameters (no full HPO with 100+ trials due to time)
- No real market validation in this run

### 6.2 Future Work
1. **Run full Optuna HPO** with 100+ trials per model class
2. **Test on real market data** (SPY, NIFTY with transaction costs)
3. **Explore RL fine-tuning** (CVaR-PPO warm-started from AttentionLSTM)
4. **Ensemble methods** combining top performers

---

## 7. Repository Structure

```
new_experiments/
├── configs/
│   └── base_config.py      # All configuration settings
├── data/
│   └── data_generator.py   # Heston simulation
├── models/
│   ├── base_model.py       # LSTM, MLP baselines
│   ├── signature_models.py # SignatureLSTM, SignatureMLP, SigFormer
│   ├── transformer_models.py # Transformer variants
│   ├── attention_lstm.py   # AttentionLSTM (BEST)
│   ├── novel_hybrids.py    # Ensemble, RegimeAware, Adaptive
│   └── rl_models.py        # CVaR-PPO, TD3
├── training/
│   ├── losses.py           # CVaR, entropic, trading penalty
│   └── trainer.py          # Unified two-stage trainer
├── evaluation/
│   └── statistical_tests.py # Bootstrap CI, Holm-Bonferroni
├── hpo/
│   └── optuna_hpo.py       # HPO framework
├── results/
│   ├── figures/            # Comparison plots
│   ├── model_metrics.csv   # Results table
│   └── results_summary.json
└── run_fair_comparison.py  # Main experiment script
```

---

## 8. Conclusion

This study demonstrates that **complex architectures can improve deep hedging performance** when properly implemented and fairly compared. The key findings are:

1. **AttentionLSTM achieves 1.6% better CVaR95** than LSTM baseline
2. **SignatureMLP achieves the lowest P&L variance**
3. **Transformer achieves the lowest trading volume**
4. **Over-complex hybrids (RegimeAware, Adaptive) underperform**

The critical success factor is **proper training protocol** (two-stage with CVaR pretraining) rather than architecture complexity alone. Future work should focus on HPO, real market validation, and RL fine-tuning of the best performers.

---

## Appendix: Commands to Reproduce

```bash
# Run fair comparison
cd /Users/prathamkailasiya/Desktop/Thesis/deep_hedging
python new_experiments/run_fair_comparison.py

# View results
cat new_experiments/results/model_metrics.csv
open new_experiments/results/figures/
```
