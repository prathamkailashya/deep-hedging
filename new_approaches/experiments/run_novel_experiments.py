#!/usr/bin/env python3
"""
Comprehensive Experiment Runner for Novel Deep Hedging Algorithms.

Runs full experiments for all 5 novel candidates:
1. W-DRO-T: Wasserstein DRO Transformer
2. RVSN: Rough Volatility Signature Network
3. SAC-CVaR: CVaR-Constrained Soft Actor-Critic
4. 3SCH: Three-Stage Curriculum Hedger
5. RSE: Regime-Switching Ensemble

Academic Standards:
- 10 random seeds for reproducibility
- Bootstrap confidence intervals (10,000 resamples)
- Paired statistical tests with Holm-Bonferroni correction
- Multiple data regimes (low/medium/high volatility)
- Fair comparison against LSTM and Transformer baselines

Usage:
    python run_novel_experiments.py --all
    python run_novel_experiments.py --model w_dro_t
    python run_novel_experiments.py --baseline-only
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add paths
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'new_approaches' / 'code'))
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from scipy import stats
from tqdm import tqdm

# Existing pipeline imports
from utils.config import set_seed
from env.heston import HestonParams, HestonModel
from env.data_generator import DataGenerator
from train.losses import EntropicLoss, CVaRLoss
from models.kozyra_models import HedgingLSTM
from models.transformer import TransformerHedge

# Novel algorithm imports
from w_dro_t import WDROTransformerHedger, WDROTrainer
from rvsn import RoughVolatilitySimulator, AdaptiveSignatureHedger, RoughBergomiParams, RSVNTrainer
from sac_cvar import CVaRConstrainedSAC, SACHyperparams, HedgingEnvironment
from three_stage import ThreeStageTrainer
from rse import RegimeSwitchingEnsemble, RSETrainer


@dataclass
class ExperimentConfig:
    """
    Configuration for experiments - aligned with paper.tex Section 5.1.
    
    Features (5D): [S/S0, log(S/K), sqrt(v), tau, BS_delta]
    """
    # Seeds for reproducibility (paper.tex: 10 seeds)
    seeds: List[int] = None
    
    # Data parameters (paper.tex: 80,000 total)
    n_train: int = 50000
    n_val: int = 10000
    n_test: int = 20000
    
    # Market parameters (Heston)
    S0: float = 100.0
    K: float = 100.0
    r: float = 0.0
    T: float = 30/365  # 30 days
    n_steps: int = 30
    
    # Heston parameters
    v0: float = 0.04
    kappa: float = 1.0
    theta: float = 0.04
    xi: float = 0.2
    rho: float = -0.7
    
    # Transaction costs
    cost_multiplier: float = 0.0
    
    # Training (paper.tex Section 5.1)
    batch_size: int = 256
    n_epochs_stage1: int = 50   # Stage 1: CVaR pretraining
    n_epochs_stage2: int = 30   # Stage 2: Entropic fine-tuning
    lr_stage1: float = 1e-3     # paper.tex: 10^-3
    lr_stage2: float = 1e-4     # paper.tex: 10^-4
    weight_decay: float = 1e-4  # paper.tex: Adam with weight decay 10^-4
    patience_stage1: int = 15   # paper.tex: early stopping patience 15
    patience_stage2: int = 10   # paper.tex: early stopping patience 10
    grad_clip: float = 5.0      # paper.tex: gradient clipping ||∇|| ≤ 5.0
    
    # Risk parameters
    cvar_alpha: float = 0.95
    lambda_risk: float = 1.0
    
    # Bootstrap (paper.tex: 10,000 resamples)
    n_bootstrap: int = 10000
    
    # Device
    device: str = 'cpu'
    
    def __post_init__(self):
        if self.seeds is None:
            # paper.tex: 10 independent random seeds (42, 142, 242, ..., 942)
            self.seeds = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]


@dataclass
class ExperimentResults:
    """Results from a single experiment run."""
    model_name: str
    seed: int
    
    # Core metrics
    mean_pnl: float
    std_pnl: float
    var_95: float
    cvar_95: float
    cvar_99: float
    entropic_risk: float
    
    # Trading metrics
    trading_volume: float
    avg_delta: float
    max_delta: float
    
    # Training info
    train_time: float
    final_loss: float
    
    # Raw data for bootstrap
    pnl_distribution: np.ndarray = None


class StatisticalAnalyzer:
    """Statistical analysis with bootstrap CI and multiple comparison correction."""
    
    def __init__(self, n_bootstrap: int = 10000, alpha: float = 0.05):
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
    
    def bootstrap_ci(self, data: np.ndarray, stat_func=np.mean) -> Tuple[float, float, float]:
        """Compute bootstrap confidence interval."""
        n = len(data)
        bootstrap_stats = []
        
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(stat_func(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        point_estimate = stat_func(data)
        ci_lower = np.percentile(bootstrap_stats, 100 * self.alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - self.alpha / 2))
        
        return point_estimate, ci_lower, ci_upper
    
    def compute_cvar(self, pnl: np.ndarray, alpha: float = 0.95) -> float:
        """Compute CVaR (on losses = -P&L)."""
        losses = -pnl
        var = np.percentile(losses, alpha * 100)
        cvar = losses[losses >= var].mean()
        return cvar
    
    def paired_ttest(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Paired t-test."""
        stat, pval = stats.ttest_rel(x, y)
        return stat, pval
    
    def wilcoxon_test(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Wilcoxon signed-rank test."""
        try:
            stat, pval = stats.wilcoxon(x - y)
        except ValueError:
            stat, pval = 0.0, 1.0
        return stat, pval
    
    def cohens_d(self, x: np.ndarray, y: np.ndarray) -> float:
        """Cohen's d effect size."""
        diff = x - y
        return diff.mean() / (diff.std() + 1e-8)
    
    def holm_bonferroni(self, pvalues: List[float]) -> List[bool]:
        """Holm-Bonferroni correction for multiple comparisons."""
        n = len(pvalues)
        sorted_indices = np.argsort(pvalues)
        sorted_pvalues = np.array(pvalues)[sorted_indices]
        
        significant = [False] * n
        for i, (idx, pval) in enumerate(zip(sorted_indices, sorted_pvalues)):
            adjusted_alpha = self.alpha / (n - i)
            if pval <= adjusted_alpha:
                significant[idx] = True
            else:
                break
        
        return significant


class DataManager:
    """Manages data generation for different regimes."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def get_heston_params(self, regime: str = 'base') -> HestonParams:
        """Get Heston parameters for different market regimes."""
        if regime == 'low_vol':
            return HestonParams(
                S0=self.config.S0, v0=0.01, r=self.config.r,
                kappa=2.0, theta=0.01, sigma=0.1, rho=-0.5
            )
        elif regime == 'high_vol':
            return HestonParams(
                S0=self.config.S0, v0=0.09, r=self.config.r,
                kappa=0.5, theta=0.09, sigma=0.4, rho=-0.8
            )
        elif regime == 'crisis':
            return HestonParams(
                S0=self.config.S0, v0=0.16, r=self.config.r,
                kappa=0.3, theta=0.16, sigma=0.6, rho=-0.9
            )
        else:  # base
            return HestonParams(
                S0=self.config.S0, v0=self.config.v0, r=self.config.r,
                kappa=self.config.kappa, theta=self.config.theta,
                sigma=self.config.xi, rho=self.config.rho
            )
    
    def generate_data(self, seed: int, regime: str = 'base'):
        """
        Generate train/val/test data with 5 features as per paper.tex.
        
        Features: [S/S0, log(S/K), sqrt(v), tau, BS_delta]
        """
        set_seed(seed)
        
        heston_params = self.get_heston_params(regime)
        
        generator = DataGenerator(
            n_steps=self.config.n_steps,
            T=self.config.T,
            S0=self.config.S0,
            K=self.config.K,
            r=self.config.r,
            cost_multiplier=self.config.cost_multiplier,
            model_type='heston',
            heston_params=heston_params
        )
        
        train_data, val_data, test_data = generator.generate_train_val_test(
            n_train=self.config.n_train,
            n_val=self.config.n_val,
            n_test=self.config.n_test,
            base_seed=seed,
            compute_bs_delta=True  # Need BS delta for 5-feature input
        )
        
        # Augment features with BS delta to get 5D input as per paper.tex
        train_data = self._augment_features_with_bs_delta(train_data)
        val_data = self._augment_features_with_bs_delta(val_data)
        test_data = self._augment_features_with_bs_delta(test_data)
        
        train_loader, val_loader, test_loader = generator.get_dataloaders(
            train_data, val_data, test_data,
            batch_size=self.config.batch_size
        )
        
        return train_loader, val_loader, test_loader, test_data
    
    def _augment_features_with_bs_delta(self, dataset):
        """
        Augment features with BS delta to create 5D input.
        
        Original features (4D): [S/S0, tau, log(S/K), v]
        Augmented features (5D): [S/S0, log(S/K), sqrt(v), tau, BS_delta]
        """
        if dataset.bs_deltas is not None:
            # Reshape BS delta to (n_samples, n_steps, 1)
            bs_delta = dataset.bs_deltas.unsqueeze(-1)
            # Concatenate with existing features
            dataset.features = torch.cat([dataset.features, bs_delta], dim=-1)
            dataset.n_features = dataset.features.shape[2]
        return dataset


class ModelTrainer:
    """Trains and evaluates models."""
    
    def __init__(self, config: ExperimentConfig, analyzer: StatisticalAnalyzer):
        self.config = config
        self.analyzer = analyzer
        self.device = torch.device(config.device)
    
    def compute_pnl(self, model, test_loader, test_data):
        """
        Compute P&L and deltas for a model on test data.
        
        Returns:
            pnl: numpy array of P&L per path
            deltas: numpy array of hedge positions [N_paths, T]
        """
        model.eval()
        all_pnl = []
        all_deltas = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                # Handle both 'prices' and 'stock_paths' keys
                prices = batch.get('prices', batch.get('stock_paths')).to(self.device)
                payoffs = batch.get('payoffs', batch.get('payoff')).to(self.device)
                
                # Get deltas
                deltas = model(features)
                if deltas.dim() == 3:
                    deltas = deltas.squeeze(-1)
                
                # Compute hedge gains
                price_changes = prices[:, 1:] - prices[:, :-1]
                hedge_gains = (deltas * price_changes).sum(dim=1)
                
                # P&L = -payoff + hedge_gains
                pnl = -payoffs + hedge_gains
                all_pnl.append(pnl.cpu().numpy())
                all_deltas.append(deltas.cpu().numpy())
        
        return np.concatenate(all_pnl), np.concatenate(all_deltas)
    
    def compute_metrics(self, pnl: np.ndarray, deltas: np.ndarray = None) -> Dict[str, float]:
        """Compute all evaluation metrics."""
        losses = -pnl
        
        metrics = {
            'mean_pnl': float(pnl.mean()),
            'std_pnl': float(pnl.std()),
            'var_95': float(np.percentile(losses, 95)),
            'cvar_95': float(self.analyzer.compute_cvar(pnl, 0.95)),
            'cvar_99': float(self.analyzer.compute_cvar(pnl, 0.99)),
            'entropic_risk': float(np.log(np.mean(np.exp(-pnl)))),
        }
        
        if deltas is not None:
            delta_changes = np.abs(np.diff(deltas, axis=1))
            metrics['trading_volume'] = float(delta_changes.sum(axis=1).mean())
            metrics['avg_delta'] = float(np.abs(deltas).mean())
            metrics['max_delta'] = float(np.abs(deltas).max())
        else:
            metrics['trading_volume'] = 0.0
            metrics['avg_delta'] = 0.0
            metrics['max_delta'] = 0.0
        
        return metrics
    
    def train_lstm_baseline(self, train_loader, val_loader, test_loader, test_data, seed: int) -> ExperimentResults:
        """Train LSTM baseline."""
        set_seed(seed)
        import time
        start_time = time.time()
        
        state_dim = 5  # [S/S0, log(S/K), sqrt(v), tau, BS_delta] per paper.tex
        model = HedgingLSTM(
            state_dim=state_dim,
            hidden_size=50,
            num_layers=2,
            delta_scale=1.5
        ).to(self.device)
        
        # Three-stage training per paper.tex protocol
        trainer = ThreeStageTrainer(
            model,
            lr_stage1=self.config.lr_stage1,
            lr_stage3=self.config.lr_stage2,  # Stage 3 uses stage2 lr from paper.tex
            weight_decay=self.config.weight_decay,
            epochs_stage1=self.config.n_epochs_stage1,
            epochs_stage3=self.config.n_epochs_stage2,
            patience_stage1=self.config.patience_stage1,
            patience_stage3=self.config.patience_stage2,
            grad_clip=self.config.grad_clip,
            device=self.config.device
        )
        trainer.train_full(train_loader, val_loader)
        
        train_time = time.time() - start_time
        
        # Evaluate
        pnl, deltas = self.compute_pnl(model, test_loader, test_data)
        metrics = self.compute_metrics(pnl, deltas)
        
        return ExperimentResults(
            model_name='LSTM',
            seed=seed,
            mean_pnl=metrics['mean_pnl'],
            std_pnl=metrics['std_pnl'],
            var_95=metrics['var_95'],
            cvar_95=metrics['cvar_95'],
            cvar_99=metrics['cvar_99'],
            entropic_risk=metrics['entropic_risk'],
            trading_volume=metrics['trading_volume'],
            avg_delta=metrics['avg_delta'],
            max_delta=metrics['max_delta'],
            train_time=train_time,
            final_loss=metrics['entropic_risk'],
            pnl_distribution=pnl
        )
    
    def train_transformer_baseline(self, train_loader, val_loader, test_loader, test_data, seed: int) -> ExperimentResults:
        """Train Transformer baseline."""
        set_seed(seed)
        import time
        start_time = time.time()
        
        model = TransformerHedge(
            input_dim=5,  # [S/S0, log(S/K), sqrt(v), tau, BS_delta] per paper.tex
            d_model=64,
            n_heads=4,
            n_layers=3,
            dropout=0.1
        ).to(self.device)
        
        # Three-stage training per paper.tex protocol
        trainer = ThreeStageTrainer(
            model,
            lr_stage1=self.config.lr_stage1,
            lr_stage3=self.config.lr_stage2,
            weight_decay=self.config.weight_decay,
            epochs_stage1=self.config.n_epochs_stage1,
            epochs_stage3=self.config.n_epochs_stage2,
            patience_stage1=self.config.patience_stage1,
            patience_stage3=self.config.patience_stage2,
            grad_clip=self.config.grad_clip,
            device=self.config.device
        )
        trainer.train_full(train_loader, val_loader)
        
        train_time = time.time() - start_time
        
        pnl, deltas = self.compute_pnl(model, test_loader, test_data)
        metrics = self.compute_metrics(pnl, deltas)
        
        return ExperimentResults(
            model_name='Transformer',
            seed=seed,
            mean_pnl=metrics['mean_pnl'],
            std_pnl=metrics['std_pnl'],
            var_95=metrics['var_95'],
            cvar_95=metrics['cvar_95'],
            cvar_99=metrics['cvar_99'],
            entropic_risk=metrics['entropic_risk'],
            trading_volume=metrics['trading_volume'],
            avg_delta=metrics['avg_delta'],
            max_delta=metrics['max_delta'],
            train_time=train_time,
            final_loss=metrics['entropic_risk'],
            pnl_distribution=pnl
        )
    
    def train_w_dro_t(self, train_loader, val_loader, test_loader, test_data, seed: int) -> ExperimentResults:
        """Train W-DRO-T model."""
        set_seed(seed)
        import time
        start_time = time.time()
        
        model = WDROTransformerHedger(
            input_dim=5,  # 5 features per paper.tex
            d_model=64,
            n_heads=4,
            n_layers=3,
            epsilon=0.1,
            delta_max=1.5
        ).to(self.device)
        
        trainer = WDROTrainer(model, lr=self.config.learning_rate, device=self.config.device)
        trainer.train_full(train_loader)
        
        train_time = time.time() - start_time
        
        pnl, deltas = self.compute_pnl(model, test_loader, test_data)
        metrics = self.compute_metrics(pnl, deltas)
        
        return ExperimentResults(
            model_name='W-DRO-T',
            seed=seed,
            mean_pnl=metrics['mean_pnl'],
            std_pnl=metrics['std_pnl'],
            var_95=metrics['var_95'],
            cvar_95=metrics['cvar_95'],
            cvar_99=metrics['cvar_99'],
            entropic_risk=metrics['entropic_risk'],
            trading_volume=metrics['trading_volume'],
            avg_delta=metrics['avg_delta'],
            max_delta=metrics['max_delta'],
            train_time=train_time,
            final_loss=metrics['entropic_risk'],
            pnl_distribution=pnl
        )
    
    def train_rvsn(self, train_loader, val_loader, test_loader, test_data, seed: int) -> ExperimentResults:
        """Train RVSN model."""
        set_seed(seed)
        import time
        start_time = time.time()
        
        model = AdaptiveSignatureHedger(
            input_dim=5,  # 5 features per paper.tex
            max_depth=4,
            hidden_dim=64,
            delta_max=1.5
        ).to(self.device)
        
        trainer = RSVNTrainer(model, lr=self.config.learning_rate, device=self.config.device)
        trainer.train(train_loader, n_epochs=self.config.n_epochs_stage1 + self.config.n_epochs_stage2)
        
        train_time = time.time() - start_time
        
        pnl, deltas = self.compute_pnl(model, test_loader, test_data)
        metrics = self.compute_metrics(pnl, deltas)
        
        return ExperimentResults(
            model_name='RVSN',
            seed=seed,
            mean_pnl=metrics['mean_pnl'],
            std_pnl=metrics['std_pnl'],
            var_95=metrics['var_95'],
            cvar_95=metrics['cvar_95'],
            cvar_99=metrics['cvar_99'],
            entropic_risk=metrics['entropic_risk'],
            trading_volume=metrics['trading_volume'],
            avg_delta=metrics['avg_delta'],
            max_delta=metrics['max_delta'],
            train_time=train_time,
            final_loss=metrics['entropic_risk'],
            pnl_distribution=pnl
        )
    
    def train_sac_cvar(self, train_loader, val_loader, test_loader, test_data, seed: int) -> ExperimentResults:
        """Train SAC-CVaR model."""
        set_seed(seed)
        import time
        start_time = time.time()
        
        # Get sample data for environment
        sample_batch = next(iter(train_loader))
        prices = sample_batch.get('prices', sample_batch.get('stock_paths')).numpy()
        payoffs = sample_batch.get('payoffs', sample_batch.get('payoff')).numpy()
        
        # Create environment
        env = HedgingEnvironment(
            prices=prices[:1000],
            payoffs=payoffs[:1000],
            transaction_cost=self.config.cost_multiplier,
            delta_max=1.5
        )
        
        # Create agent
        agent = CVaRConstrainedSAC(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            cvar_threshold=5.0,
            device=self.config.device
        )
        
        # Train with episodes
        n_episodes = min(500, self.config.n_epochs_stage1 * 10)
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.buffer.push(state, action, reward, next_state, done)
                if len(agent.buffer) >= 256:
                    agent.update(batch_size=256)
                state = next_state
        
        train_time = time.time() - start_time
        
        # Evaluate: run agent on test data, collecting both P&L and deltas
        test_pnl = []
        all_deltas = []
        for batch in test_loader:
            prices_batch = batch.get('prices', batch.get('stock_paths')).numpy()
            payoffs_batch = batch.get('payoffs', batch.get('payoff')).numpy()
            
            for i in range(len(prices_batch)):
                test_env = HedgingEnvironment(
                    prices=prices_batch[i:i+1],
                    payoffs=payoffs_batch[i:i+1],
                    transaction_cost=self.config.cost_multiplier,
                    delta_max=1.5
                )
                state = test_env.reset()
                done = False
                path_deltas = []
                while not done:
                    action = agent.select_action(state, evaluate=True)
                    path_deltas.append(action[0] if hasattr(action, '__len__') else action)
                    state, reward, done, info = test_env.step(action)
                test_pnl.append(info.get('pnl', reward))
                all_deltas.append(path_deltas)
        
        pnl = np.array(test_pnl)
        # Pad deltas to same length for proper array
        max_len = max(len(d) for d in all_deltas) if all_deltas else 1
        deltas = np.array([d + [d[-1]] * (max_len - len(d)) if d else [0] * max_len for d in all_deltas])
        metrics = self.compute_metrics(pnl, deltas)
        
        return ExperimentResults(
            model_name='SAC-CVaR',
            seed=seed,
            mean_pnl=metrics['mean_pnl'],
            std_pnl=metrics['std_pnl'],
            var_95=metrics['var_95'],
            cvar_95=metrics['cvar_95'],
            cvar_99=metrics['cvar_99'],
            entropic_risk=metrics['entropic_risk'],
            trading_volume=metrics['trading_volume'],
            avg_delta=metrics['avg_delta'],
            max_delta=metrics['max_delta'],
            train_time=train_time,
            final_loss=metrics['entropic_risk'],
            pnl_distribution=pnl
        )
    
    def train_three_stage(self, train_loader, val_loader, test_loader, test_data, seed: int) -> ExperimentResults:
        """Train Three-Stage Curriculum Hedger."""
        set_seed(seed)
        import time
        start_time = time.time()
        
        model = HedgingLSTM(
            state_dim=5,  # 5 features per paper.tex
            hidden_size=50,
            num_layers=2,
            delta_scale=1.5
        ).to(self.device)
        
        trainer = ThreeStageTrainer(
            model,
            device=self.config.device,
            epochs_stage1=self.config.n_epochs_stage1,
            epochs_stage2=20,  # Mixed stage
            epochs_stage3=self.config.n_epochs_stage2
        )
        trainer.train_full(train_loader, val_loader)
        
        train_time = time.time() - start_time
        
        pnl, deltas = self.compute_pnl(model, test_loader, test_data)
        metrics = self.compute_metrics(pnl, deltas)
        
        return ExperimentResults(
            model_name='3SCH',
            seed=seed,
            mean_pnl=metrics['mean_pnl'],
            std_pnl=metrics['std_pnl'],
            var_95=metrics['var_95'],
            cvar_95=metrics['cvar_95'],
            cvar_99=metrics['cvar_99'],
            entropic_risk=metrics['entropic_risk'],
            trading_volume=metrics['trading_volume'],
            avg_delta=metrics['avg_delta'],
            max_delta=metrics['max_delta'],
            train_time=train_time,
            final_loss=metrics['entropic_risk'],
            pnl_distribution=pnl
        )
    
    def train_rse(self, train_loader, val_loader, test_loader, test_data, seed: int) -> ExperimentResults:
        """Train Regime-Switching Ensemble."""
        set_seed(seed)
        import time
        start_time = time.time()
        
        model = RegimeSwitchingEnsemble(
            input_dim=5,  # 5 features per paper.tex
            n_regimes=4,
            delta_max=1.5
        ).to(self.device)
        
        trainer = RSETrainer(model, device=self.config.device)
        trainer.train_full(train_loader)
        
        train_time = time.time() - start_time
        
        pnl, deltas = self.compute_pnl(model, test_loader, test_data)
        metrics = self.compute_metrics(pnl, deltas)
        
        return ExperimentResults(
            model_name='RSE',
            seed=seed,
            mean_pnl=metrics['mean_pnl'],
            std_pnl=metrics['std_pnl'],
            var_95=metrics['var_95'],
            cvar_95=metrics['cvar_95'],
            cvar_99=metrics['cvar_99'],
            entropic_risk=metrics['entropic_risk'],
            trading_volume=metrics['trading_volume'],
            avg_delta=metrics['avg_delta'],
            max_delta=metrics['max_delta'],
            train_time=train_time,
            final_loss=metrics['entropic_risk'],
            pnl_distribution=pnl
        )


class ExperimentRunner:
    """Main experiment runner."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.analyzer = StatisticalAnalyzer(n_bootstrap=config.n_bootstrap)
        self.data_manager = DataManager(config)
        self.trainer = ModelTrainer(config, self.analyzer)
        self.results_dir = ROOT / 'new_approaches' / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_single_model(self, model_name: str, seed: int, regime: str = 'base') -> ExperimentResults:
        """Run experiment for a single model and seed."""
        print(f"  Running {model_name} with seed {seed} ({regime} regime)...")
        
        # Generate data
        train_loader, val_loader, test_loader, test_data = self.data_manager.generate_data(seed, regime)
        
        # Train based on model name
        if model_name == 'LSTM':
            return self.trainer.train_lstm_baseline(train_loader, val_loader, test_loader, test_data, seed)
        elif model_name == 'Transformer':
            return self.trainer.train_transformer_baseline(train_loader, val_loader, test_loader, test_data, seed)
        elif model_name == 'W-DRO-T':
            return self.trainer.train_w_dro_t(train_loader, val_loader, test_loader, test_data, seed)
        elif model_name == 'RVSN':
            return self.trainer.train_rvsn(train_loader, val_loader, test_loader, test_data, seed)
        elif model_name == 'SAC-CVaR':
            return self.trainer.train_sac_cvar(train_loader, val_loader, test_loader, test_data, seed)
        elif model_name == '3SCH':
            return self.trainer.train_three_stage(train_loader, val_loader, test_loader, test_data, seed)
        elif model_name == 'RSE':
            return self.trainer.train_rse(train_loader, val_loader, test_loader, test_data, seed)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def run_all_seeds(self, model_name: str, regime: str = 'base') -> List[ExperimentResults]:
        """Run experiment across all seeds."""
        results = []
        for seed in tqdm(self.config.seeds, desc=f"{model_name}"):
            result = self.run_single_model(model_name, seed, regime)
            results.append(result)
        return results
    
    def run_full_comparison(self, models: List[str] = None, regimes: List[str] = None):
        """Run full comparison across all models and regimes."""
        if models is None:
            models = ['LSTM', 'Transformer', 'W-DRO-T', 'RVSN', 'SAC-CVaR', '3SCH', 'RSE']
        if regimes is None:
            regimes = ['base']
        
        all_results = {}
        
        for regime in regimes:
            print(f"\n{'='*60}")
            print(f"REGIME: {regime.upper()}")
            print(f"{'='*60}")
            
            regime_results = {}
            for model_name in models:
                print(f"\nTraining {model_name}...")
                try:
                    results = self.run_all_seeds(model_name, regime)
                    regime_results[model_name] = results
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue
            
            all_results[regime] = regime_results
        
        return all_results
    
    def compute_summary_stats(self, results: List[ExperimentResults]) -> Dict[str, Any]:
        """Compute summary statistics across seeds."""
        metrics = ['cvar_95', 'cvar_99', 'std_pnl', 'entropic_risk', 'trading_volume']
        summary = {}
        
        for metric in metrics:
            values = [getattr(r, metric) for r in results]
            point, ci_lower, ci_upper = self.analyzer.bootstrap_ci(np.array(values))
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        
        summary['train_time_mean'] = np.mean([r.train_time for r in results])
        
        return summary
    
    def compare_to_baseline(self, baseline_results: List[ExperimentResults], 
                           model_results: List[ExperimentResults]) -> Dict[str, Any]:
        """Statistical comparison to baseline."""
        comparison = {}
        metrics = ['cvar_95', 'cvar_99', 'std_pnl', 'entropic_risk', 'trading_volume']
        
        for metric in metrics:
            baseline_vals = np.array([getattr(r, metric) for r in baseline_results])
            model_vals = np.array([getattr(r, metric) for r in model_results])
            
            diff = model_vals - baseline_vals
            _, pval = self.analyzer.paired_ttest(model_vals, baseline_vals)
            cohens_d = self.analyzer.cohens_d(model_vals, baseline_vals)
            
            point, ci_lower, ci_upper = self.analyzer.bootstrap_ci(diff)
            
            comparison[metric] = {
                'diff_mean': float(diff.mean()),
                'diff_pct': float(diff.mean() / (baseline_vals.mean() + 1e-8) * 100),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'p_value': float(pval),
                'cohens_d': float(cohens_d)
            }
        
        return comparison
    
    def generate_latex_table(self, all_results: Dict, baseline_name: str = 'LSTM') -> str:
        """Generate LaTeX table of results."""
        latex = []
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(r"\caption{Performance Comparison of Deep Hedging Models (10 Seeds)}")
        latex.append(r"\label{tab:main_results}")
        latex.append(r"\begin{tabular}{lccccc}")
        latex.append(r"\toprule")
        latex.append(r"Model & CVaR$_{95}$ & CVaR$_{99}$ & Std P\&L & Entropic & Trading Vol \\")
        latex.append(r"\midrule")
        
        regime_results = all_results.get('base', all_results)
        
        for model_name, results in regime_results.items():
            summary = self.compute_summary_stats(results)
            
            row = f"{model_name} & "
            row += f"{summary['cvar_95']['mean']:.3f} $\\pm$ {summary['cvar_95']['std']:.3f} & "
            row += f"{summary['cvar_99']['mean']:.3f} $\\pm$ {summary['cvar_99']['std']:.3f} & "
            row += f"{summary['std_pnl']['mean']:.3f} $\\pm$ {summary['std_pnl']['std']:.3f} & "
            row += f"{summary['entropic_risk']['mean']:.3f} $\\pm$ {summary['entropic_risk']['std']:.3f} & "
            row += f"{summary['trading_volume']['mean']:.3f} $\\pm$ {summary['trading_volume']['std']:.3f} \\\\"
            latex.append(row)
        
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        
        return '\n'.join(latex)
    
    def save_results(self, all_results: Dict, filename: str = 'experiment_results.pkl'):
        """Save results to disk."""
        filepath = self.results_dir / filename
        
        # Convert to serializable format
        serializable = {}
        for regime, regime_results in all_results.items():
            serializable[regime] = {}
            for model_name, results in regime_results.items():
                serializable[regime][model_name] = [
                    {k: v.tolist() if isinstance(v, np.ndarray) else v 
                     for k, v in asdict(r).items()}
                    for r in results
                ]
        
        with open(filepath, 'wb') as f:
            pickle.dump(serializable, f)
        
        # Also save as JSON (without pnl_distribution)
        json_results = {}
        for regime, regime_results in all_results.items():
            json_results[regime] = {}
            for model_name, results in regime_results.items():
                json_results[regime][model_name] = self.compute_summary_stats(results)
        
        json_path = self.results_dir / filename.replace('.pkl', '.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {filepath}")
        return filepath


def main():
    parser = argparse.ArgumentParser(description='Run Novel Deep Hedging Experiments')
    parser.add_argument('--all', action='store_true', help='Run all models')
    parser.add_argument('--model', type=str, choices=['LSTM', 'Transformer', 'W-DRO-T', 'RVSN', 'SAC-CVaR', '3SCH', 'RSE'],
                       help='Run specific model')
    parser.add_argument('--baseline-only', action='store_true', help='Run baselines only')
    parser.add_argument('--novel-only', action='store_true', help='Run novel models only')
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds')
    parser.add_argument('--n-train', type=int, default=50000, help='Training samples')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--quick', action='store_true', help='Quick test with 3 seeds')
    args = parser.parse_args()
    
    # Configuration
    config = ExperimentConfig(
        n_train=args.n_train,
        device=args.device
    )
    
    if args.quick:
        config.seeds = [42, 142, 242]
        config.n_train = 10000
        config.n_epochs_stage1 = 10
        config.n_epochs_stage2 = 5
        config.patience_stage1 = 5
        config.patience_stage2 = 3
    elif args.seeds != 10:
        config.seeds = list(range(42, 42 + args.seeds * 100, 100))
    
    # Run experiments
    runner = ExperimentRunner(config)
    
    if args.model:
        models = [args.model]
    elif args.baseline_only:
        models = ['LSTM', 'Transformer']
    elif args.novel_only:
        models = ['W-DRO-T', 'RVSN', 'SAC-CVaR', '3SCH', 'RSE']
    else:
        models = ['LSTM', 'Transformer', 'W-DRO-T', 'RVSN', 'SAC-CVaR', '3SCH', 'RSE']
    
    print(f"\n{'='*60}")
    print("NOVEL DEEP HEDGING EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Models: {models}")
    print(f"Seeds: {config.seeds}")
    print(f"Training samples: {config.n_train}")
    print(f"Device: {config.device}")
    print(f"{'='*60}\n")
    
    # Run experiments
    all_results = runner.run_full_comparison(models=models)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    runner.save_results(all_results, f'results_{timestamp}.pkl')
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    regime_results = all_results.get('base', all_results)
    
    for model_name, results in regime_results.items():
        summary = runner.compute_summary_stats(results)
        print(f"\n{model_name}:")
        print(f"  CVaR95: {summary['cvar_95']['mean']:.4f} ± {summary['cvar_95']['std']:.4f}")
        print(f"  CVaR99: {summary['cvar_99']['mean']:.4f} ± {summary['cvar_99']['std']:.4f}")
        print(f"  Std P&L: {summary['std_pnl']['mean']:.4f} ± {summary['std_pnl']['std']:.4f}")
    
    # Statistical comparison
    if 'LSTM' in regime_results and len(regime_results) > 1:
        print(f"\n{'='*60}")
        print("STATISTICAL COMPARISON VS LSTM BASELINE")
        print(f"{'='*60}")
        
        baseline_results = regime_results['LSTM']
        pvalues = []
        comparisons = []
        
        for model_name, results in regime_results.items():
            if model_name == 'LSTM':
                continue
            
            comparison = runner.compare_to_baseline(baseline_results, results)
            comparisons.append((model_name, comparison))
            pvalues.append(comparison['cvar_95']['p_value'])
        
        # Holm-Bonferroni correction
        significant = runner.analyzer.holm_bonferroni(pvalues)
        
        for i, (model_name, comparison) in enumerate(comparisons):
            cvar = comparison['cvar_95']
            sig = '*' if significant[i] else ''
            print(f"\n{model_name} vs LSTM:")
            print(f"  CVaR95 diff: {cvar['diff_mean']:+.4f} ({cvar['diff_pct']:+.1f}%)")
            print(f"  95% CI: [{cvar['ci_lower']:.4f}, {cvar['ci_upper']:.4f}]")
            print(f"  p-value: {cvar['p_value']:.4f} {sig}")
            print(f"  Cohen's d: {cvar['cohens_d']:.3f}")
    
    # Generate LaTeX table
    latex_table = runner.generate_latex_table(all_results)
    latex_path = runner.results_dir / f'results_table_{timestamp}.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {latex_path}")
    
    print(f"\n{'='*60}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
