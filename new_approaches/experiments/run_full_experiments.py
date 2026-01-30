#!/usr/bin/env python3
"""
Optimized Full Experiment Runner with Checkpointing.

Runs all models with 10 seeds per paper.tex protocol with:
- Incremental saving after each seed
- Resume capability from checkpoints
- Progress tracking

Usage:
    python run_full_experiments.py --all
    python run_full_experiments.py --model LSTM --resume
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Auto-detect best device (MPS for M1 Mac, CUDA for NVIDIA, else CPU)
def get_best_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

DEFAULT_DEVICE = get_best_device()

# Add paths
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'new_approaches' / 'code'))

try:
    from src.utils.seed import set_seed
except ImportError:
    def set_seed(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

from src.env.heston import HestonParams
from src.env.data_generator import DataGenerator
from src.train.losses import EntropicLoss, CVaRLoss
from src.models.kozyra_models import HedgingLSTM
from src.models.transformer import TransformerHedge

# Novel algorithms
from three_stage import ThreeStageTrainer
from w_dro_t import WDROTransformerHedger, WDROTrainer
from rvsn import AdaptiveSignatureHedger, RSVNTrainer
from rse import RegimeSwitchingEnsemble, RSETrainer
from sac_cvar import CVaRConstrainedSAC, HedgingEnvironment


# Paper.tex protocol
SEEDS = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]
N_TRAIN = 50000
N_VAL = 10000
N_TEST = 20000
N_STEPS = 30
BATCH_SIZE = 256

# Training protocol (paper.tex Section 5.1)
EPOCHS_STAGE1 = 50
EPOCHS_STAGE2 = 30
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-4
PATIENCE_STAGE1 = 15
PATIENCE_STAGE2 = 10
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0


def get_heston_params():
    return HestonParams(
        S0=100.0, v0=0.04, r=0.0,
        kappa=1.0, theta=0.04, sigma=0.2, rho=-0.7
    )


def generate_data(seed: int, n_train: int = N_TRAIN):
    """Generate data with 5 features per paper.tex."""
    set_seed(seed)
    
    generator = DataGenerator(
        n_steps=N_STEPS,
        T=30/365,
        S0=100.0,
        K=100.0,
        r=0.0,
        model_type='heston',
        heston_params=get_heston_params()
    )
    
    train_data, val_data, test_data = generator.generate_train_val_test(
        n_train=n_train,
        n_val=N_VAL,
        n_test=N_TEST,
        base_seed=seed,
        compute_bs_delta=True
    )
    
    # Augment with BS delta for 5D features
    for dataset in [train_data, val_data, test_data]:
        if dataset.bs_deltas is not None:
            bs_delta = dataset.bs_deltas.unsqueeze(-1)
            dataset.features = torch.cat([dataset.features, bs_delta], dim=-1)
            dataset.n_features = dataset.features.shape[2]
    
    train_loader, val_loader, test_loader = generator.get_dataloaders(
        train_data, val_data, test_data, batch_size=BATCH_SIZE
    )
    
    return train_loader, val_loader, test_loader, test_data


def compute_pnl(model, test_loader, test_data, device='cpu'):
    """
    Compute P&L and deltas for evaluation.
    
    Returns:
        pnl: numpy array of P&L per path
        deltas: numpy array of hedge positions [N_paths, T]
    """
    model.eval()
    all_deltas = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            deltas = model(features)
            all_deltas.append(deltas.cpu())
    
    deltas = torch.cat(all_deltas, dim=0)
    prices = test_data.stock_paths
    payoffs = test_data.payoffs
    
    # P&L calculation
    price_changes = prices[:, 1:] - prices[:, :-1]
    n_steps = min(deltas.shape[1], price_changes.shape[1])
    hedge_gains = (deltas[:, :n_steps] * price_changes[:, :n_steps]).sum(dim=1)
    pnl = -payoffs + hedge_gains
    
    return pnl.numpy(), deltas.numpy()


def compute_metrics(pnl: np.ndarray, deltas: np.ndarray = None) -> Dict[str, float]:
    """
    Compute evaluation metrics including trading volume.
    
    Args:
        pnl: P&L array [N_paths]
        deltas: Hedge positions [N_paths, T] - required for trading volume
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mean_pnl': float(np.mean(pnl)),
        'std_pnl': float(np.std(pnl)),
        'var_95': float(np.percentile(-pnl, 95)),
        'cvar_95': float(-pnl[pnl <= np.percentile(pnl, 5)].mean()),
        'cvar_99': float(-pnl[pnl <= np.percentile(pnl, 1)].mean()),
        'entropic_risk': float(np.log(np.mean(np.exp(-pnl)))),
    }
    
    # Compute trading volume from deltas
    if deltas is not None:
        # Trading volume: sum of absolute delta changes per path
        # TV_i = Σ_t |Δ_{i,t+1} − Δ_{i,t}|
        delta_changes = np.abs(np.diff(deltas, axis=1))
        trading_volume_per_path = np.sum(delta_changes, axis=1)
        
        metrics['trading_volume'] = float(np.mean(trading_volume_per_path))
        metrics['trading_volume_std'] = float(np.std(trading_volume_per_path))
        metrics['avg_delta'] = float(np.mean(np.abs(deltas)))
        metrics['max_delta'] = float(np.max(np.abs(deltas)))
        metrics['delta_smoothness'] = float(np.mean(delta_changes))
        
        # Sanity checks
        if metrics['trading_volume'] < 1e-6:
            warnings.warn(f"Trading volume suspiciously low: {metrics['trading_volume']:.6f}")
        if np.any(np.isnan(deltas)) or np.any(np.isinf(deltas)):
            raise ValueError("Deltas contain NaN or Inf values")
        if np.all(deltas == 0):
            raise ValueError("All deltas are zero - model not learning")
    else:
        # Fallback if deltas not provided (should not happen in proper usage)
        warnings.warn("Deltas not provided - trading volume metrics unavailable")
        metrics['trading_volume'] = 0.0
        metrics['trading_volume_std'] = 0.0
        metrics['avg_delta'] = 0.0
        metrics['max_delta'] = 0.0
        metrics['delta_smoothness'] = 0.0
    
    return metrics


def train_lstm(train_loader, val_loader, test_loader, test_data, seed, device='cpu'):
    """Train LSTM baseline."""
    set_seed(seed)
    import time
    start = time.time()
    
    model = HedgingLSTM(
        state_dim=5,
        hidden_size=50,
        num_layers=2,
        delta_scale=1.5
    ).to(device)
    
    trainer = ThreeStageTrainer(
        model,
        lr_stage1=LR_STAGE1,
        lr_stage3=LR_STAGE2,
        weight_decay=WEIGHT_DECAY,
        epochs_stage1=EPOCHS_STAGE1,
        epochs_stage3=EPOCHS_STAGE2,
        patience_stage1=PATIENCE_STAGE1,
        patience_stage3=PATIENCE_STAGE2,
        grad_clip=GRAD_CLIP,
        device=device
    )
    trainer.train_full(train_loader, val_loader)
    
    train_time = time.time() - start
    pnl, deltas = compute_pnl(model, test_loader, test_data, device)
    metrics = compute_metrics(pnl, deltas)
    metrics['train_time'] = train_time
    metrics['seed'] = seed
    
    return metrics


def train_transformer(train_loader, val_loader, test_loader, test_data, seed, device='cpu'):
    """Train Transformer baseline."""
    set_seed(seed)
    import time
    start = time.time()
    
    model = TransformerHedge(
        input_dim=5,
        d_model=64,
        n_heads=4,
        n_layers=3,
        dropout=0.1
    ).to(device)
    
    trainer = ThreeStageTrainer(
        model,
        lr_stage1=LR_STAGE1,
        lr_stage3=LR_STAGE2,
        weight_decay=WEIGHT_DECAY,
        epochs_stage1=EPOCHS_STAGE1,
        epochs_stage3=EPOCHS_STAGE2,
        patience_stage1=PATIENCE_STAGE1,
        patience_stage3=PATIENCE_STAGE2,
        grad_clip=GRAD_CLIP,
        device=device
    )
    trainer.train_full(train_loader, val_loader)
    
    train_time = time.time() - start
    pnl, deltas = compute_pnl(model, test_loader, test_data, device)
    metrics = compute_metrics(pnl, deltas)
    metrics['train_time'] = train_time
    metrics['seed'] = seed
    
    return metrics


def train_wdrot(train_loader, val_loader, test_loader, test_data, seed, device='cpu'):
    """Train W-DRO-T model."""
    set_seed(seed)
    import time
    start = time.time()
    
    model = WDROTransformerHedger(
        input_dim=5,
        d_model=64,
        n_heads=4,
        n_layers=3,
        epsilon=0.1,
        delta_max=1.5
    ).to(device)
    
    trainer = WDROTrainer(model, lr=LR_STAGE1, device=device)
    trainer.train(train_loader, val_loader, epochs=EPOCHS_STAGE1 + EPOCHS_STAGE2)
    
    train_time = time.time() - start
    pnl, deltas = compute_pnl(model, test_loader, test_data, device)
    metrics = compute_metrics(pnl, deltas)
    metrics['train_time'] = train_time
    metrics['seed'] = seed
    
    return metrics


def train_3sch(train_loader, val_loader, test_loader, test_data, seed, device='cpu'):
    """Train 3SCH (Three-Stage Curriculum Hedger)."""
    set_seed(seed)
    import time
    start = time.time()
    
    model = HedgingLSTM(
        state_dim=5,
        hidden_size=50,
        num_layers=2,
        delta_scale=1.5
    ).to(device)
    
    # Full three-stage training
    trainer = ThreeStageTrainer(
        model,
        lr_stage1=LR_STAGE1,
        lr_stage2=5e-4,  # Intermediate rate for stage 2
        lr_stage3=LR_STAGE2,
        weight_decay=WEIGHT_DECAY,
        epochs_stage1=EPOCHS_STAGE1,
        epochs_stage2=20,  # Explicit stage 2
        epochs_stage3=EPOCHS_STAGE2,
        patience_stage1=PATIENCE_STAGE1,
        patience_stage3=PATIENCE_STAGE2,
        grad_clip=GRAD_CLIP,
        device=device
    )
    trainer.train_full(train_loader, val_loader)
    
    train_time = time.time() - start
    pnl, deltas = compute_pnl(model, test_loader, test_data, device)
    metrics = compute_metrics(pnl, deltas)
    metrics['train_time'] = train_time
    metrics['seed'] = seed
    
    return metrics


def train_rse(train_loader, val_loader, test_loader, test_data, seed, device='cpu'):
    """Train RSE (Regime-Switching Ensemble)."""
    set_seed(seed)
    import time
    start = time.time()
    
    model = RegimeSwitchingEnsemble(
        input_dim=5,
        n_regimes=4,
        delta_max=1.5
    ).to(device)
    
    trainer = RSETrainer(model, device=device)
    trainer.train(train_loader, val_loader, epochs=EPOCHS_STAGE1)
    
    train_time = time.time() - start
    pnl, deltas = compute_pnl(model, test_loader, test_data, device)
    metrics = compute_metrics(pnl, deltas)
    metrics['train_time'] = train_time
    metrics['seed'] = seed
    
    return metrics


def train_rvsn(train_loader, val_loader, test_loader, test_data, seed, device='cpu'):
    """Train RVSN (Rough Volatility Signature Network)."""
    set_seed(seed)
    import time
    start = time.time()
    
    model = AdaptiveSignatureHedger(
        input_dim=5,
        max_depth=4,
        hidden_dim=64,
        delta_max=1.5
    ).to(device)
    
    trainer = RSVNTrainer(model, device=device)
    trainer.train(train_loader, val_loader, epochs=EPOCHS_STAGE1)
    
    train_time = time.time() - start
    pnl, deltas = compute_pnl(model, test_loader, test_data, device)
    metrics = compute_metrics(pnl, deltas)
    metrics['train_time'] = train_time
    metrics['seed'] = seed
    
    return metrics


def train_sac_cvar(train_loader, val_loader, test_loader, test_data, seed, device='cpu'):
    """Train SAC-CVaR (Soft Actor-Critic with CVaR constraints)."""
    set_seed(seed)
    import time
    start = time.time()
    
    # Extract data for RL environment
    prices_list = []
    payoffs_list = []
    for batch in test_loader:
        prices_list.append(batch.get('prices', batch.get('stock_paths')))
        payoffs_list.append(batch['payoff'])
    
    prices = torch.cat(prices_list, dim=0)
    payoffs = torch.cat(payoffs_list, dim=0)
    
    # Create environment and agent
    env = HedgingEnvironment(prices, payoffs, delta_max=1.5)
    
    agent = CVaRConstrainedSAC(
        state_dim=5,
        action_dim=1,
        cvar_threshold=4.0,  # Target CVaR
        device=device
    )
    
    # RL training loop
    n_episodes = 500
    for ep in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            # Update after enough samples
            if len(agent.buffer) >= 256:
                agent.update(batch_size=256)
    
    train_time = time.time() - start
    
    # Evaluate: Run through test data collecting P&L and deltas
    pnl_list = []
    all_deltas = []
    
    for batch in test_loader:
        batch_prices = batch.get('prices', batch.get('stock_paths'))
        batch_payoffs = batch['payoff']
        
        for i in range(len(batch_prices)):
            env_eval = HedgingEnvironment(
                batch_prices[i:i+1], 
                batch_payoffs[i:i+1],
                delta_max=1.5
            )
            state = env_eval.reset()
            total_pnl = 0
            done = False
            path_deltas = []
            
            while not done:
                action = agent.select_action(state, evaluate=True)
                path_deltas.append(action[0])  # Store delta
                state, reward, done, _ = env_eval.step(action)
                total_pnl += reward
            
            pnl_list.append(total_pnl)
            all_deltas.append(path_deltas)
    
    pnl = np.array(pnl_list)
    # Pad deltas to same length
    max_len = max(len(d) for d in all_deltas)
    deltas = np.array([d + [d[-1]] * (max_len - len(d)) for d in all_deltas])
    
    metrics = compute_metrics(pnl, deltas)
    metrics['train_time'] = train_time
    metrics['seed'] = seed
    
    return metrics


TRAIN_FUNCS = {
    'LSTM': train_lstm,
    'Transformer': train_transformer,
    'W-DRO-T': train_wdrot,
    'RVSN': train_rvsn,
    'SAC-CVaR': train_sac_cvar,
    '3SCH': train_3sch,
    'RSE': train_rse,
}


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """Load existing checkpoint."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return {'results': {}, 'completed': set()}


def save_checkpoint(checkpoint_path: Path, data: Dict):
    """Save checkpoint."""
    # Convert set to list for pickling
    save_data = {
        'results': data['results'],
        'completed': list(data['completed'])
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(save_data, f)


def run_experiments(
    models: List[str],
    seeds: List[int] = SEEDS,
    n_train: int = N_TRAIN,
    device: str = 'cpu',
    resume: bool = True
):
    """Run full experiments with checkpointing."""
    results_dir = ROOT / 'new_approaches' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = results_dir / 'checkpoint_full.pkl'
    
    # Load checkpoint
    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        checkpoint['completed'] = set(checkpoint.get('completed', []))
    else:
        checkpoint = {'results': {}, 'completed': set()}
    
    print("=" * 60)
    print("FULL DEEP HEDGING EXPERIMENTS (10 seeds per paper.tex)")
    print("=" * 60)
    print(f"Models: {models}")
    print(f"Seeds: {seeds}")
    print(f"Training samples: {n_train}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)
    
    total_runs = len(models) * len(seeds)
    completed_runs = len(checkpoint['completed'])
    print(f"Progress: {completed_runs}/{total_runs} runs completed")
    
    for model_name in models:
        if model_name not in TRAIN_FUNCS:
            print(f"Unknown model: {model_name}, skipping")
            continue
        
        if model_name not in checkpoint['results']:
            checkpoint['results'][model_name] = []
        
        train_func = TRAIN_FUNCS[model_name]
        
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        for seed in tqdm(seeds, desc=model_name):
            run_key = f"{model_name}_{seed}"
            
            if run_key in checkpoint['completed']:
                print(f"  Seed {seed}: Already completed, skipping")
                continue
            
            print(f"\n  Running {model_name} with seed {seed}...")
            
            # Generate data
            print(f"  Generating data...")
            train_loader, val_loader, test_loader, test_data = generate_data(seed, n_train)
            
            # Train
            try:
                metrics = train_func(
                    train_loader, val_loader, test_loader, test_data,
                    seed=seed, device=device
                )
                
                checkpoint['results'][model_name].append(metrics)
                checkpoint['completed'].add(run_key)
                
                # Save checkpoint after each run
                save_checkpoint(checkpoint_path, checkpoint)
                
                print(f"  Seed {seed} completed: CVaR95={metrics['cvar_95']:.4f}, Time={metrics['train_time']:.1f}s")
                
            except Exception as e:
                print(f"  ERROR on seed {seed}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Final summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    summary = {}
    for model_name, results in checkpoint['results'].items():
        if not results:
            continue
        
        cvar95_values = [r['cvar_95'] for r in results]
        summary[model_name] = {
            'cvar_95': {
                'mean': np.mean(cvar95_values),
                'std': np.std(cvar95_values),
                'n_seeds': len(cvar95_values)
            },
            'std_pnl': {
                'mean': np.mean([r['std_pnl'] for r in results]),
                'std': np.std([r['std_pnl'] for r in results])
            },
            'entropic_risk': {
                'mean': np.mean([r['entropic_risk'] for r in results]),
                'std': np.std([r['entropic_risk'] for r in results])
            },
            'train_time_mean': np.mean([r['train_time'] for r in results])
        }
        
        print(f"\n{model_name} ({len(results)} seeds):")
        print(f"  CVaR95: {summary[model_name]['cvar_95']['mean']:.4f} ± {summary[model_name]['cvar_95']['std']:.4f}")
        print(f"  Std P&L: {summary[model_name]['std_pnl']['mean']:.4f} ± {summary[model_name]['std_pnl']['std']:.4f}")
        print(f"  Entropic: {summary[model_name]['entropic_risk']['mean']:.4f}")
        print(f"  Avg Time: {summary[model_name]['train_time_mean']:.1f}s")
    
    # Save final results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_path = results_dir / f'results_full_{timestamp}.pkl'
    with open(final_path, 'wb') as f:
        pickle.dump({'raw': checkpoint['results'], 'summary': summary}, f)
    
    # Save JSON summary
    with open(results_dir / f'results_full_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {final_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Full Deep Hedging Experiments')
    parser.add_argument('--model', type=str, help='Single model to run')
    parser.add_argument('--all', action='store_true', help='Run all models')
    parser.add_argument('--baselines', action='store_true', help='Run baselines only')
    parser.add_argument('--novel', action='store_true', help='Run novel models only')
    parser.add_argument('--n-train', type=int, default=N_TRAIN, help='Training samples')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, help='Device (auto-detects MPS/CUDA)')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from checkpoint')
    parser.add_argument('--fresh', action='store_true', help='Start fresh (ignore checkpoint)')
    args = parser.parse_args()
    
    if args.all:
        models = ['LSTM', 'Transformer', 'W-DRO-T', 'RVSN', 'SAC-CVaR', '3SCH', 'RSE']
    elif args.baselines:
        models = ['LSTM', 'Transformer']
    elif args.novel:
        models = ['W-DRO-T', 'RVSN', 'SAC-CVaR', '3SCH', 'RSE']
    elif args.model:
        models = [args.model]
    else:
        models = ['LSTM', 'Transformer']
    
    resume = not args.fresh
    
    run_experiments(
        models=models,
        n_train=args.n_train,
        device=args.device,
        resume=resume
    )


if __name__ == '__main__':
    main()
