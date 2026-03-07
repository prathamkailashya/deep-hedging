# Reproducibility Guide

**Author:** Pratham Kailasiya, IIT Roorkee  
**Repository:** `deep_hedging/`

---

## 1. Environment Setup

### Requirements
- **Python:** 3.10+
- **PyTorch:** 2.0+ (with CUDA 11.8+ for GPU)
- **OS:** Linux/macOS recommended; Windows supported
- **GPU:** NVIDIA GPU with ≥4GB VRAM recommended (CPU training possible but slow)

### Installation

```bash
cd deep_hedging
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### Key Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.0.0 | Deep learning framework |
| numpy | ≥1.24.0 | Numerical computing |
| scipy | ≥1.10.0 | Statistical tests |
| pandas | ≥2.0.0 | Data manipulation |
| yfinance | ≥0.2.28 | Market data (SPY) |
| optuna | ≥3.2.0 | Hyperparameter tuning |
| statsmodels | ≥0.14.0 | Statistical analysis |
| matplotlib | ≥3.7.0 | Plotting |

---

## 2. Data Generation

### Heston Model Simulation

Data is generated automatically by the experiment scripts. Parameters:

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Initial price | $S_0$ | 100.0 |
| Strike | $K$ | 100.0 (ATM) |
| Initial variance | $v_0$ | 0.04 (σ₀ = 20%) |
| Mean reversion | κ | 1.0 |
| Long-run variance | θ | 0.04 |
| Vol-of-vol | ξ | 0.2 |
| Correlation | ρ | -0.7 |
| Risk-free rate | r | 0.05 |
| Time steps | N | 30 |
| Horizon | T | 30/365 |
| Transaction cost | c_tc | 0.001 |

### Dataset Sizes
- **Training:** 50,000 paths
- **Validation:** 10,000 paths
- **Test:** 20,000 paths
- **Total:** 80,000 paths

---

## 3. Running Experiments

### Full Experiment Suite (All Models, 10 Seeds)

```bash
cd new_approaches
python experiments/run_full_experiments.py \
    --seeds 42 142 242 342 442 542 642 742 842 942 \
    --n_train 50000 \
    --n_val 10000 \
    --n_test 20000 \
    --epochs 80 \
    --batch_size 256
```

### Individual Models

```bash
# LSTM baseline
python experiments/run_full_experiments.py --models LSTM --seeds 42

# Transformer
python experiments/run_full_experiments.py --models Transformer --seeds 42

# W-DRO-T (requires MATH SDPA backend)
python experiments/run_full_experiments.py --models W-DRO-T --seeds 42

# 3SCH (Three-Stage Curriculum)
python experiments/run_full_experiments.py --models 3SCH --seeds 42

# RSE (Regime-Switching Ensemble)
python experiments/run_full_experiments.py --models RSE --seeds 42

# RVSN (Rough Volatility Signature Network)
python experiments/run_full_experiments.py --models RVSN --seeds 42

# SAC-CVaR (RL-based)
python experiments/run_full_experiments.py --models SAC-CVaR --seeds 42
```

### Real Market Validation

```bash
python experiments/extended_real_market_validation.py
```

This runs all models on SPY and NIFTY across four scenarios:
- Normal 2019 (σ ≈ 15%)
- Post-COVID 2021 (σ ≈ 20%)
- COVID Crisis 2020 (σ ≈ 65%)
- 2008 Financial Crisis (σ ≈ 80%)

### Statistical Analysis

```bash
python experiments/analyze_results.py
```

Generates:
- Bootstrap confidence intervals (10,000 resamples)
- Paired t-tests with Holm-Bonferroni correction
- Cohen's d effect sizes

---

## 4. Expected Runtimes

| Model | Time/Seed (GPU) | Time/Seed (CPU) | Total (10 seeds, GPU) |
|-------|-----------------|-----------------|----------------------|
| LSTM | ~273s (4.5 min) | ~20 min | ~46 min |
| 3SCH | ~295s (5 min) | ~22 min | ~49 min |
| RSE | ~924s (15 min) | ~60 min | ~154 min |
| W-DRO-T | ~7,119s (2h) | ~12h | ~20h |
| Transformer | ~7,534s (2.1h) | ~13h | ~21h |
| RVSN | ~1,791s (30 min) | ~3h | ~5h |
| SAC-CVaR | ~486s (8 min) | ~45 min | ~81 min |

**Total estimated time:** ~24 hours (GPU) / ~72 hours (CPU)

---

## 5. Expected Disk Usage

| Item | Size |
|------|------|
| Per checkpoint (.pkl) | ~500KB - 15MB |
| All checkpoints (70 runs) | ~500MB |
| Results JSON files | ~50MB |
| Generated figures | ~100MB |
| Total | ~650MB |

---

## 6. GPU/CPU Requirements

### Minimum (CPU)
- 8GB RAM
- 4-core CPU
- ~72 hours runtime

### Recommended (GPU)
- NVIDIA GPU with ≥4GB VRAM
- 16GB system RAM
- CUDA 11.8+
- ~24 hours runtime

### W-DRO-T Specific
- Requires PyTorch MATH SDPA backend (not flash/efficient attention)
- Second-order gradients increase memory by ~2×
- Recommended: ≥6GB VRAM

---

## 7. Random Seeds

All experiments use the following 10 seeds for reproducibility:

```
42, 142, 242, 342, 442, 542, 642, 742, 842, 942
```

Each seed controls:
- Data generation (path simulation)
- Weight initialization
- Batch ordering
- Dropout (where applicable)

---

## 8. Training Protocol (All Models)

### Two-Stage Curriculum (Baseline)
| Stage | Epochs | Loss | LR | Patience | Grad Clip |
|-------|--------|------|-------|----------|-----------|
| 1 (CVaR) | 50 | CVaR₉₅ | 1e-3 | 15 | 5.0 |
| 2 (Entropic) | 30 | ρ_λ + penalties | 1e-4 | 10 | 5.0 |

### Model-Specific Additions
- **W-DRO-T:** DRO penalty in Stage 2 (ε: 0→0.1)
- **3SCH:** Intermediate mixed stage (40+15+25 epochs)
- **RSE:** Pre-train base models (30 ep each), then train gating

---

## 9. Output Files

### Checkpoints
```
new_approaches/results/checkpoint_full.pkl
```

### Results
```
new_approaches/results/audit_summary.json          # 10-seed results
new_approaches/results/statistical_analysis.json    # Statistical tests
new_approaches/results/extended_real_market_validation.json  # Crisis testing
new_approaches/results/real_market_validation.json  # Market validation
```

### Analysis
```
new_approaches/results/analysis/                    # Detailed breakdowns
```

---

## 10. Verification

After running experiments, verify results match:

| Model | CVaR₉₅ (expected) | Tolerance |
|-------|-------------------|-----------|
| RSE | 3.109 ± 0.010 | ±0.05 |
| LSTM | 3.215 ± 0.018 | ±0.05 |
| 3SCH | 3.219 ± 0.022 | ±0.05 |
| W-DRO-T | 3.227 ± 0.024 | ±0.05 |
| Transformer | 3.234 ± 0.030 | ±0.05 |

Small deviations are expected due to hardware-specific floating point differences.

---

## 11. Compiling LaTeX Deliverables

```bash
cd deliverable
make all
# Or individually:
pdflatex W-DRO-T_paper.tex && bibtex W-DRO-T_paper && pdflatex W-DRO-T_paper.tex && pdflatex W-DRO-T_paper.tex
```

---

## 12. Troubleshooting

### W-DRO-T SDPA Error
```
RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented
```
**Fix:** The code automatically uses MATH backend. Ensure PyTorch ≥ 2.0.

### RVSN Anomalous Results
CVaR₉₅ ≈ 0.44 with high variance is a known issue. The model may produce biased hedges. Further hyperparameter tuning is needed.

### SAC-CVaR Poor Performance
CVaR₉₅ ≈ 14.9 indicates the RL agent did not learn an effective policy within the compute budget. Increasing training episodes or adjusting reward shaping may help.

### Memory Issues
Reduce batch size: `--batch_size 128` or `--batch_size 64`.
