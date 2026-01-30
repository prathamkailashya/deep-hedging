# Novel Deep Hedging Algorithms

This directory contains implementations of 5 novel algorithms for deep hedging, grounded in reference literature and addressing gaps identified in the existing pipeline.

## Candidates Overview

| # | Algorithm | File | Reference | Target Gap |
|---|-----------|------|-----------|------------|
| 1 | **W-DRO-T** | `w_dro_t.py` | Lütkebohmert 2021 | Parameter uncertainty |
| 2 | **RVSN** | `rvsn.py` | Abi Jaber 2025 | Non-Markovian dynamics |
| 3 | **SAC-CVaR** | `sac_cvar.py` | Huang 2025, Neagu 2025 | Explicit tail constraints |
| 4 | **3SCH** | `three_stage.py` | Kozyra 2018 extended | Training stability |
| 5 | **RSE** | `rse.py` | Pipeline gap | Regime adaptation |

## Installation

All dependencies are in the main `requirements.txt`. Additional requirement:
```bash
pip install fbm  # For fractional Brownian motion (RVSN)
```

## Usage

### 1. W-DRO-T (Wasserstein DRO Transformer)

```python
from new_approaches.code import WDROTransformerHedger, WDROTrainer

model = WDROTransformerHedger(
    input_dim=5,
    d_model=64,
    n_heads=4,
    n_layers=3,
    epsilon=0.1  # DRO robustness radius
)

trainer = WDROTrainer(model, lr=1e-3, device='cuda')
results = trainer.train_full(train_loader)
```

### 2. RVSN (Rough Volatility Signature Network)

```python
from new_approaches.code import RoughVolatilitySimulator, AdaptiveSignatureHedger, RoughBergomiParams

# Generate rough volatility data
params = RoughBergomiParams(H=0.1)  # Hurst parameter for rough regime
simulator = RoughVolatilitySimulator(params)
data = simulator.generate_hedging_data(n_paths=50000)

# Train adaptive signature hedger
model = AdaptiveSignatureHedger(
    input_dim=5,
    max_depth=4,
    hidden_dim=64
)
```

### 3. SAC-CVaR (CVaR-Constrained Soft Actor-Critic)

```python
from new_approaches.code import CVaRConstrainedSAC, HedgingEnvironment

env = HedgingEnvironment(prices, payoffs, delta_max=1.5)

agent = CVaRConstrainedSAC(
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    cvar_threshold=5.0,  # Target CVaR constraint
    device='cuda'
)

# Training loop
for episode in range(n_episodes):
    state = env.reset()
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.buffer.push(state, action, reward, next_state, done)
        metrics = agent.update(batch_size=256)
        state = next_state
```

### 4. 3SCH (Three-Stage Curriculum Hedger)

```python
from new_approaches.code import ThreeStageTrainer
from src.models.kozyra_models import HedgingLSTM

model = HedgingLSTM(input_dim=5, hidden_dim=50)
trainer = ThreeStageTrainer(model, device='cuda')

# Full three-stage training
results = trainer.train_full(train_loader, val_loader)
# Stage 1: CVaR (50 epochs)
# Stage 2: Mixed (20 epochs) - NEW intermediate stage
# Stage 3: Entropic (30 epochs)
```

### 5. RSE (Regime-Switching Ensemble)

```python
from new_approaches.code import RegimeSwitchingEnsemble, RSETrainer

model = RegimeSwitchingEnsemble(
    input_dim=5,
    n_regimes=4,
    delta_max=1.5
)

trainer = RSETrainer(model, device='cuda')

# Pre-train base models, then train gating
trainer.train_full(train_loader)

# Get regime analysis for interpretability
analysis = model.get_regime_analysis(features, prices)
print(f"Regime probabilities: {analysis['regime_probs']}")
print(f"Model weights: {analysis['model_weights']}")
```

## Expected Improvements

Based on literature and gap analysis:

| Algorithm | Expected CVaR Improvement | Best Use Case |
|-----------|--------------------------|---------------|
| W-DRO-T | 3-8% | Uncertain market parameters |
| RVSN | 5-10% (on rough vol) | Non-Markovian volatility |
| SAC-CVaR | 5-12% | Market impact, explicit tail control |
| 3SCH | 2-5% | General training stability |
| RSE | 3-7% | Regime-dependent performance |

## Testing

Run tests for each implementation:

```bash
python new_approaches/code/w_dro_t.py
python new_approaches/code/rvsn.py
python new_approaches/code/sac_cvar.py
python new_approaches/code/three_stage.py
python new_approaches/code/rse.py
```

## Integration with Existing Pipeline

These implementations are designed to integrate with the existing `src/` codebase:

- **Losses**: Uses `src/train/losses.py` (EntropicLoss, CVaRLoss)
- **Models**: Extends `src/models/` architectures
- **Training**: Compatible with existing data loaders and evaluation

## References

1. **Lütkebohmert et al. 2021**: "Robust deep hedging" - DRO for parameter uncertainty
2. **Abi Jaber & Gérard 2025**: "Hedging with memory" - Signatures for rough volatility
3. **Huang & Lawryshyn 2025**: "Deep Hedging Under Market Frictions" - SAC with impact
4. **Neagu et al. 2025**: "Deep RL Algorithms for Option Hedging" - MCPG/PPO comparison
5. **Kozyra 2018**: Oxford MSc Thesis - Two-stage training protocol

## Files

```
new_approaches/code/
├── __init__.py           # Package exports
├── README.md             # This file
├── w_dro_t.py           # Candidate 1: Wasserstein DRO Transformer
├── rvsn.py              # Candidate 2: Rough Volatility Signature Network
├── sac_cvar.py          # Candidate 3: CVaR-Constrained SAC
├── three_stage.py       # Candidate 4: Three-Stage Curriculum Hedger
└── rse.py               # Candidate 5: Regime-Switching Ensemble
```
