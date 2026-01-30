"""
Novel Deep Hedging Algorithms
=============================

This module contains implementations of 5 novel algorithms for deep hedging,
grounded in reference literature and addressing gaps in the existing pipeline.

Candidates:
1. W-DRO-T: Wasserstein DRO Transformer (Lütkebohmert 2021)
2. RVSN: Rough Volatility Signature Network (Abi Jaber 2025)
3. SAC-CVaR: CVaR-Constrained Soft Actor-Critic (Huang 2025, Neagu 2025)
4. 3SCH: Three-Stage Curriculum Hedger (Kozyra 2018 extended)
5. RSE: Regime-Switching Ensemble (Pipeline gap)
"""

from .w_dro_t import WassersteinDROLoss, WDROTransformerHedger
from .rvsn import RoughVolatilitySimulator, AdaptiveSignatureHedger
from .sac_cvar import CVaRConstrainedSAC, QuantileNetwork
from .three_stage import ThreeStageTrainer, MixedLoss
from .rse import RegimeSwitchingEnsemble, RegimeClassifier

__all__ = [
    'WassersteinDROLoss',
    'WDROTransformerHedger',
    'RoughVolatilitySimulator', 
    'AdaptiveSignatureHedger',
    'CVaRConstrainedSAC',
    'QuantileNetwork',
    'ThreeStageTrainer',
    'MixedLoss',
    'RegimeSwitchingEnsemble',
    'RegimeClassifier',
]
