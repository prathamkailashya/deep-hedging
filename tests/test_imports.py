#!/usr/bin/env python3
"""
Test all imports to validate the package structure.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_utils():
    """Test utils module imports."""
    from utils.config import Config, MarketConfig, TrainingConfig, set_seed
    from utils.logging_utils import ExperimentLogger, setup_logger
    from utils.statistics import compute_metrics, bootstrap_ci, paired_ttest
    print("✓ utils imports OK")


def test_env():
    """Test environment module imports."""
    from env.heston import HestonModel, HestonParams, BlackScholesModel
    from env.market_env import MarketEnvironment
    from env.data_generator import DataGenerator, HedgingDataset
    print("✓ env imports OK")


def test_models():
    """Test models module imports."""
    from models.deep_hedging import DeepHedgingModel, SemiRecurrentNetwork
    from models.kozyra_models import HedgingRNN, HedgingLSTM, KozyraTwoStageModel
    from models.baselines import BlackScholesHedge, LelandHedge, WhalleyWilmottHedge
    from models.transformer import TransformerHedge, SigFormer
    from models.signature_models import SignatureHedge, LogSignatureHedge
    from models.rl_agents import PPOHedge, DDPGHedge, MCPGHedge
    print("✓ models imports OK")


def test_train():
    """Test training module imports."""
    from train.losses import EntropicLoss, CVaRLoss, HedgingLoss
    from train.trainer import DeepHedgingTrainer, EarlyStopping
    from train.kozyra_trainer import KozyraTwoStageTrainer
    from train.optuna_tuning import OptunaHyperparameterTuner
    print("✓ train imports OK")


def test_eval():
    """Test evaluation module imports."""
    from eval.evaluator import HedgingEvaluator, compare_strategies
    from eval.plotting import plot_pnl_histogram, plot_pnl_boxplot
    print("✓ eval imports OK")


def test_basic_functionality():
    """Test basic functionality."""
    import torch
    import numpy as np
    
    from utils.config import Config, set_seed
    from env.heston import HestonModel, HestonParams
    from models.deep_hedging import DeepHedgingModel
    
    set_seed(42)
    
    # Test Heston simulation
    params = HestonParams(S0=100, v0=0.04, r=0.0, kappa=1.0, theta=0.04, sigma=0.2, rho=-0.7)
    heston = HestonModel(params)
    paths, vol = heston.simulate(n_paths=100, n_steps=30, T=30/365)
    assert paths.shape == (100, 31), f"Expected (100, 31), got {paths.shape}"
    print("✓ Heston simulation OK")
    
    # Test Deep Hedging model
    model = DeepHedgingModel(input_dim=4, n_steps=30)
    features = torch.randn(10, 30, 4)
    deltas = model(features)
    assert deltas.shape == (10, 30), f"Expected (10, 30), got {deltas.shape}"
    print("✓ Deep Hedging model OK")
    
    print("\n✓ All basic functionality tests passed!")


def main():
    print("=" * 50)
    print("Deep Hedging Package Validation")
    print("=" * 50)
    
    try:
        test_utils()
        test_env()
        test_models()
        test_train()
        test_eval()
        print("\n" + "=" * 50)
        test_basic_functionality()
        print("=" * 50)
        print("\n✓ ALL TESTS PASSED")
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
