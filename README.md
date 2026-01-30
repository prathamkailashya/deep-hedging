# Deep Hedging Research Pipeline

A fully reproducible research pipeline for replicating and extending deep hedging methodologies.

## Overview

This repository implements:

1. **Buehler et al. (Deep Hedging)** - Section 5: Numerical Experiments
2. **Kozyra (Oxford MSc thesis)** - RNN/LSTM and two-stage training
3. **Enhanced models** - Transformer, Signature features, RL agents
4. **Real data validation** - US and Indian markets

## Mathematical Setup

### Time Grid
$$0 = t_0 < t_1 < \dots < t_n = T, \quad n = 30, \quad T = \frac{30}{365}$$

### Heston Model (Risk-Neutral)
$$dS_t = r S_t \, dt + \sqrt{v_t} S_t \, dW_t^S$$
$$dv_t = \kappa(\theta - v_t)\,dt + \sigma \sqrt{v_t}\, dW_t^v$$
$$\text{corr}(W^S, W^v) = \rho$$

### Terminal P&L
$$\text{P\&L} = -Z + \sum_{k=0}^{n-1} \delta_k (S_{k+1}-S_k) - \sum_{k=0}^{n-1} \kappa |\delta_{k+1}-\delta_k|$$

### Risk Objectives

**Entropic Risk (OCE):**
$$J(\theta) = \mathbb{E}\left[\exp\left(-\lambda \cdot \text{P\&L}_\theta\right)\right]$$

**Conditional Value-at-Risk:**
$$\text{CVaR}_\alpha(X) = \frac{1}{1-\alpha}\int_0^{1-\alpha} \text{VaR}_u(X)\, du$$

## Installation

```bash
# Clone repository
cd deep_hedging

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Repository Structure

```
deep_hedging/
├── data/
│   ├── raw/              # Raw market data
│   └── processed/        # Processed datasets
├── src/
│   ├── env/              # Market simulation
│   │   ├── heston.py     # Heston model
│   │   ├── market_env.py # Trading environment
│   │   └── data_generator.py
│   ├── models/           # Neural network models
│   │   ├── deep_hedging.py    # Buehler et al.
│   │   ├── kozyra_models.py   # RNN/LSTM
│   │   ├── baselines.py       # BS, Leland, WW
│   │   ├── transformer.py     # Attention models
│   │   ├── signature_models.py # Path signatures
│   │   └── rl_agents.py       # PPO, DDPG, etc.
│   ├── train/            # Training utilities
│   │   ├── trainer.py
│   │   ├── kozyra_trainer.py
│   │   ├── losses.py
│   │   └── optuna_tuning.py
│   ├── eval/             # Evaluation
│   │   ├── evaluator.py
│   │   └── plotting.py
│   └── utils/            # Utilities
│       ├── config.py
│       ├── logging_utils.py
│       ├── statistics.py
│       └── real_data.py
├── notebooks/            # Jupyter notebooks
├── experiments/          # Experiment scripts
├── figures/              # Output figures
├── reports/              # Technical report
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Run All Experiments

```bash
cd experiments
python run_experiments.py --experiment all --seed 42
```

### 2. Run Specific Experiments

```bash
# Deep Hedging replication
python run_experiments.py --experiment deep_hedging

# Kozyra RNN/LSTM
python run_experiments.py --experiment kozyra

# With transaction costs
python run_experiments.py --experiment all --cost 0.001
```

### 3. Use in Python

```python
from src.utils.config import Config, set_seed
from src.env.data_generator import DataGenerator
from src.models.deep_hedging import DeepHedgingModel
from src.train.trainer import DeepHedgingTrainer

# Setup
set_seed(42)
config = Config()

# Generate data
generator = DataGenerator(n_steps=30, T=30/365)
train_data, val_data, test_data = generator.generate_train_val_test()
train_loader, val_loader, test_loader = generator.get_dataloaders(
    train_data, val_data, test_data
)

# Create and train model
model = DeepHedgingModel(
    input_dim=train_data.n_features,
    n_steps=30,
    lambda_risk=1.0
)

trainer = DeepHedgingTrainer(model, learning_rate=0.005)
history = trainer.fit(train_loader, val_loader, n_epochs=100)

# Evaluate
metrics, pnl, deltas = trainer.evaluate(test_loader)
print(f"Mean P&L: {metrics['mean_pnl']:.4f}")
print(f"CVaR 95%: {metrics['cvar_95']:.4f}")
```

## Model Architectures

### Buehler et al. Semi-Recurrent Network
- Per-time-step neural network
- 3 layers: N₀=2d, N₁=N₂=d+15, N₃=d
- ReLU activation with Batch Normalization
- Adam optimizer, lr=0.005, batch_size=256

### Kozyra RNN/LSTM
```python
from src.models.kozyra_models import HedgingRNN

model = HedgingRNN(
    state_dim=4,
    hidden_size=50,
    num_layers=2
)
```

### Two-Stage Training
```python
from src.train.kozyra_trainer import KozyraTwoStageTrainer

trainer = KozyraTwoStageTrainer(
    model,
    gamma=1e-3,      # Transaction cost penalty
    nu=1e8,          # Band penalty
    band_width=0.15  # No-transaction band
)

# Stage 1: CVaR pretraining
trainer.train_stage1(train_loader, val_loader)

# Stage 2: Fine-tuning with constraints
trainer.train_stage2(train_loader, val_loader)
```

## Baselines

- **Black-Scholes Delta**: Δ = N(d₁)
- **Leland Adjusted Delta**: Modified volatility for transaction costs
- **Whalley-Wilmott**: No-transaction band strategy

## Evaluation Metrics

- Mean P&L, Standard Deviation
- VaR (95%, 99%)
- CVaR / Expected Shortfall
- Entropic Risk
- Indifference Price
- Trading Volume, Transaction Costs
- Bootstrap confidence intervals (1000 resamples)

## Real Data

```python
from src.utils.real_data import get_hedging_data_for_ticker

# US market
data = get_hedging_data_for_ticker('SPY', lookback_days=365)

# Indian market
data = get_hedging_data_for_ticker('RELIANCE.NS', is_indian=True)
```

## Citations

```bibtex
@article{buehler2019deep,
  title={Deep hedging},
  author={Buehler, Hans and Gonon, Lukas and Teichmann, Josef and Wood, Ben},
  journal={Quantitative Finance},
  year={2019}
}

@mastersthesis{kozyra2021deep,
  title={Deep Hedging with RNNs},
  author={Kozyra, Michal},
  school={University of Oxford},
  year={2021}
}
```

## License

MIT License

## Authors

Research Pipeline for Thesis Work
