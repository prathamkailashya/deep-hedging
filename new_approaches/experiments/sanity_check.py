#!/usr/bin/env python
"""
PRIORITY 1b: 1-seed sanity checks for all novel models.

Validates that all models produce numerically reasonable results
relative to Transformer baseline (CVaR95 ≈ 3.23, Std ≈ 0.45, Entropic ≈ 2.38).

Models tested: RVSN, W-DRO-T, SAC-CVaR, 3SCH, RSE
"""

import sys
import os

# Get absolute paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..', 'code'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src', 'models'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src', 'utils'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src', 'training'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src', 'env'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))

import torch
import numpy as np
import time
from datetime import datetime

# Baseline reference values (Transformer 10-seed average)
BASELINE = {
    'CVaR95': 3.234,
    'Std': 0.450,
    'Entropic': 2.380,
}

# Acceptable deviation from baseline (20% tolerance for sanity check)
TOLERANCE = 0.20

def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_data(seed, n_train=20000, n_val=5000, n_test=10000):
    """Generate training data using Heston model."""
    from data_generator import DataGenerator, get_heston_params
    
    set_seed(seed)
    generator = DataGenerator(
        n_steps=30,
        T=30/365,
        S0=100.0,
        K=100.0,
        r=0.0,
        model_type='heston',
        heston_params=get_heston_params()
    )
    
    train_data, val_data, test_data = generator.generate_train_val_test(
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
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
        train_data, val_data, test_data, batch_size=256
    )
    
    return train_loader, val_loader, test_loader, test_data

def compute_pnl(model, test_loader, test_data, device):
    """Compute P&L for evaluation."""
    model.eval()
    all_deltas = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            deltas = model(features)
            all_deltas.append(deltas.cpu())
    
    deltas = torch.cat(all_deltas, dim=0).numpy()
    prices = test_data.stock_paths.numpy()
    payoff = test_data.payoff.numpy()
    
    # P&L calculation
    price_changes = prices[:, 1:] - prices[:, :-1]
    hedge_gains = (deltas * price_changes).sum(axis=1)
    
    delta_changes = np.zeros_like(deltas)
    delta_changes[:, 0] = deltas[:, 0]
    delta_changes[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
    transaction_costs = (np.abs(delta_changes) * prices[:, :-1] * 0.001).sum(axis=1)
    
    pnl = hedge_gains - payoff - transaction_costs
    return pnl, deltas

def compute_metrics(pnl, deltas):
    """Compute evaluation metrics."""
    metrics = {
        'mean_pnl': float(np.mean(pnl)),
        'std_pnl': float(np.std(pnl)),
        'var_95': float(np.percentile(-pnl, 95)),
        'cvar_95': float(-np.mean(pnl[pnl <= np.percentile(pnl, 5)])),
        'cvar_99': float(-np.mean(pnl[pnl <= np.percentile(pnl, 1)])),
        'entropic_risk': float(np.log(np.mean(np.exp(-pnl)))),
        'trading_volume': float(np.mean(np.abs(np.diff(deltas, axis=1)).sum(axis=1))),
    }
    return metrics

def check_sanity(metrics, model_name):
    """Check if metrics are within tolerance of baseline."""
    issues = []
    
    cvar_ratio = metrics['cvar_95'] / BASELINE['CVaR95']
    if cvar_ratio > (1 + TOLERANCE) or cvar_ratio < (1 - TOLERANCE):
        issues.append(f"CVaR95 {metrics['cvar_95']:.4f} deviates {(cvar_ratio-1)*100:.1f}% from baseline")
    
    std_ratio = metrics['std_pnl'] / BASELINE['Std']
    if std_ratio > (1 + TOLERANCE) or std_ratio < (1 - TOLERANCE):
        issues.append(f"Std {metrics['std_pnl']:.4f} deviates {(std_ratio-1)*100:.1f}% from baseline")
    
    entropic_ratio = metrics['entropic_risk'] / BASELINE['Entropic']
    if entropic_ratio > (1 + TOLERANCE) or entropic_ratio < (1 - TOLERANCE):
        issues.append(f"Entropic {metrics['entropic_risk']:.4f} deviates {(entropic_ratio-1)*100:.1f}% from baseline")
    
    return issues

def run_sanity_check_3sch(train_loader, val_loader, test_loader, test_data, device):
    """Sanity check for 3SCH."""
    print("\n" + "="*60)
    print("3SCH (Three-Stage Curriculum Hedger)")
    print("="*60)
    
    from hedging_models import HedgingLSTM
    from three_stage_trainer import ThreeStageTrainer
    
    set_seed(42)
    start = time.time()
    
    model = HedgingLSTM(
        state_dim=5,
        hidden_size=50,
        num_layers=2,
        delta_scale=1.5
    ).to(device)
    
    # 3-stage curriculum (40+15+25=80 epochs, scaled for sanity check)
    trainer = ThreeStageTrainer(
        model,
        lr_stage1=1e-3,
        lr_stage2=5e-4,
        lr_stage3=1e-4,
        weight_decay=1e-5,
        epochs_stage1=10,  # Scaled down for sanity check
        epochs_stage2=4,
        epochs_stage3=6,
        patience_stage1=5,
        patience_stage3=5,
        grad_clip=1.0,
        device=device
    )
    trainer.train_full(train_loader, val_loader)
    
    train_time = time.time() - start
    pnl, deltas = compute_pnl(model, test_loader, test_data, device)
    metrics = compute_metrics(pnl, deltas)
    metrics['train_time'] = train_time
    
    print(f"  CVaR95: {metrics['cvar_95']:.4f} (baseline: {BASELINE['CVaR95']:.4f})")
    print(f"  Std P&L: {metrics['std_pnl']:.4f} (baseline: {BASELINE['Std']:.4f})")
    print(f"  Entropic: {metrics['entropic_risk']:.4f} (baseline: {BASELINE['Entropic']:.4f})")
    print(f"  Trading Vol: {metrics['trading_volume']:.4f}")
    print(f"  Runtime: {train_time:.1f}s")
    
    issues = check_sanity(metrics, '3SCH')
    return metrics, issues

def run_sanity_check_wdrot(train_loader, val_loader, test_loader, test_data, device):
    """Sanity check for W-DRO-T."""
    print("\n" + "="*60)
    print("W-DRO-T (Wasserstein DRO Transformer)")
    print("="*60)
    
    from w_dro_t import WDROTransformerHedger, WDROTrainer
    
    set_seed(42)
    start = time.time()
    
    model = WDROTransformerHedger(
        input_dim=5,
        d_model=64,
        n_heads=4,
        n_layers=3,
        epsilon=0.1,
        delta_max=1.5
    ).to(device)
    
    trainer = WDROTrainer(model, lr=1e-3, device=device)
    trainer.train(train_loader, val_loader, epochs=20)  # Scaled down
    
    train_time = time.time() - start
    pnl, deltas = compute_pnl(model, test_loader, test_data, device)
    metrics = compute_metrics(pnl, deltas)
    metrics['train_time'] = train_time
    
    print(f"  CVaR95: {metrics['cvar_95']:.4f} (baseline: {BASELINE['CVaR95']:.4f})")
    print(f"  Std P&L: {metrics['std_pnl']:.4f} (baseline: {BASELINE['Std']:.4f})")
    print(f"  Entropic: {metrics['entropic_risk']:.4f} (baseline: {BASELINE['Entropic']:.4f})")
    print(f"  Trading Vol: {metrics['trading_volume']:.4f}")
    print(f"  Runtime: {train_time:.1f}s")
    
    issues = check_sanity(metrics, 'W-DRO-T')
    return metrics, issues

def run_sanity_check_rvsn(train_loader, val_loader, test_loader, test_data, device):
    """Sanity check for RVSN."""
    print("\n" + "="*60)
    print("RVSN (Rough Volatility Signature Network)")
    print("="*60)
    
    from rvsn import AdaptiveSignatureHedger, RSVNTrainer
    
    set_seed(42)
    start = time.time()
    
    model = AdaptiveSignatureHedger(
        input_dim=5,
        max_depth=4,
        hidden_dim=64,
        delta_max=1.5
    ).to(device)
    
    trainer = RSVNTrainer(model, lr=1e-3, device=device)
    trainer.train(train_loader, val_loader, epochs=20)  # Scaled down
    
    train_time = time.time() - start
    pnl, deltas = compute_pnl(model, test_loader, test_data, device)
    metrics = compute_metrics(pnl, deltas)
    metrics['train_time'] = train_time
    
    print(f"  CVaR95: {metrics['cvar_95']:.4f} (baseline: {BASELINE['CVaR95']:.4f})")
    print(f"  Std P&L: {metrics['std_pnl']:.4f} (baseline: {BASELINE['Std']:.4f})")
    print(f"  Entropic: {metrics['entropic_risk']:.4f} (baseline: {BASELINE['Entropic']:.4f})")
    print(f"  Trading Vol: {metrics['trading_volume']:.4f}")
    print(f"  Runtime: {train_time:.1f}s")
    
    issues = check_sanity(metrics, 'RVSN')
    return metrics, issues

def run_sanity_check_rse(train_loader, val_loader, test_loader, test_data, device):
    """Sanity check for RSE."""
    print("\n" + "="*60)
    print("RSE (Regime-Switching Ensemble)")
    print("="*60)
    
    from rse import RegimeSwitchingEnsemble, RSETrainer
    
    set_seed(42)
    start = time.time()
    
    model = RegimeSwitchingEnsemble(
        input_dim=5,
        n_regimes=4,
        delta_max=1.5
    ).to(device)
    
    trainer = RSETrainer(model, lr=1e-3, device=device)
    trainer.train(train_loader, val_loader, epochs=20)  # Scaled down
    
    train_time = time.time() - start
    pnl, deltas = compute_pnl(model, test_loader, test_data, device)
    metrics = compute_metrics(pnl, deltas)
    metrics['train_time'] = train_time
    
    print(f"  CVaR95: {metrics['cvar_95']:.4f} (baseline: {BASELINE['CVaR95']:.4f})")
    print(f"  Std P&L: {metrics['std_pnl']:.4f} (baseline: {BASELINE['Std']:.4f})")
    print(f"  Entropic: {metrics['entropic_risk']:.4f} (baseline: {BASELINE['Entropic']:.4f})")
    print(f"  Trading Vol: {metrics['trading_volume']:.4f}")
    print(f"  Runtime: {train_time:.1f}s")
    
    issues = check_sanity(metrics, 'RSE')
    return metrics, issues

def run_sanity_check_sac_cvar(train_loader, val_loader, test_loader, test_data, device):
    """Sanity check for SAC-CVaR (episodic RL)."""
    print("\n" + "="*60)
    print("SAC-CVaR (Soft Actor-Critic with CVaR Constraints)")
    print("="*60)
    
    from sac_cvar import CVaRConstrainedSAC, HedgingEnvironment
    
    set_seed(42)
    start = time.time()
    
    # Create environment
    env = HedgingEnvironment(
        train_loader=train_loader,
        device=device,
        transaction_cost=0.001
    )
    
    # Create SAC agent
    agent = CVaRConstrainedSAC(
        state_dim=5,
        action_dim=1,
        hidden_dim=64,
        cvar_alpha=0.95,
        cvar_limit=5.0,
        device=device
    )
    
    # Train for limited episodes (sanity check)
    n_episodes = 50
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > 256:
                agent.update()
            
            state = next_state
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}")
    
    train_time = time.time() - start
    
    # Evaluate
    all_pnls = []
    all_deltas = []
    
    for batch in test_loader:
        features = batch['features'].to(device)
        prices = batch.get('prices', batch.get('stock_paths')).to(device)
        payoff = batch['payoff'].to(device)
        
        with torch.no_grad():
            batch_deltas = []
            for t in range(features.shape[1]):
                state = features[:, t, :]
                action = agent.select_action(state.cpu().numpy(), evaluate=True)
                action = np.clip(action, -1.5, 1.5)
                batch_deltas.append(torch.tensor(action))
            
            deltas = torch.stack(batch_deltas, dim=1).squeeze(-1)
            all_deltas.append(deltas)
    
    deltas = torch.cat(all_deltas, dim=0).numpy()
    prices_np = test_data.stock_paths.numpy()
    payoff_np = test_data.payoff.numpy()
    
    price_changes = prices_np[:, 1:] - prices_np[:, :-1]
    hedge_gains = (deltas * price_changes).sum(axis=1)
    delta_changes = np.zeros_like(deltas)
    delta_changes[:, 0] = deltas[:, 0]
    delta_changes[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
    transaction_costs = (np.abs(delta_changes) * prices_np[:, :-1] * 0.001).sum(axis=1)
    
    pnl = hedge_gains - payoff_np - transaction_costs
    metrics = compute_metrics(pnl, deltas)
    metrics['train_time'] = train_time
    
    print(f"  CVaR95: {metrics['cvar_95']:.4f} (baseline: {BASELINE['CVaR95']:.4f})")
    print(f"  Std P&L: {metrics['std_pnl']:.4f} (baseline: {BASELINE['Std']:.4f})")
    print(f"  Entropic: {metrics['entropic_risk']:.4f} (baseline: {BASELINE['Entropic']:.4f})")
    print(f"  Trading Vol: {metrics['trading_volume']:.4f}")
    print(f"  Runtime: {train_time:.1f}s")
    
    # RL models have higher tolerance due to exploration
    issues = []
    if metrics['cvar_95'] > BASELINE['CVaR95'] * 2:
        issues.append(f"CVaR95 {metrics['cvar_95']:.4f} too high (>2x baseline)")
    
    return metrics, issues

def main():
    print("="*60)
    print("SANITY CHECK: 1-Seed Tests for Novel Models")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"\nBaseline (Transformer 10-seed avg):")
    print(f"  CVaR95: {BASELINE['CVaR95']:.4f}")
    print(f"  Std P&L: {BASELINE['Std']:.4f}")
    print(f"  Entropic: {BASELINE['Entropic']:.4f}")
    print(f"\nTolerance: ±{TOLERANCE*100:.0f}%")
    
    device = get_device()
    print(f"\nDevice: {device}")
    
    # Generate data once
    print("\nGenerating data...")
    train_loader, val_loader, test_loader, test_data = generate_data(
        seed=42, n_train=10000, n_val=2000, n_test=5000  # Smaller for sanity check
    )
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    results = {}
    all_issues = {}
    
    # Run sanity checks
    try:
        results['3SCH'], all_issues['3SCH'] = run_sanity_check_3sch(
            train_loader, val_loader, test_loader, test_data, device
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        all_issues['3SCH'] = [str(e)]
    
    try:
        results['W-DRO-T'], all_issues['W-DRO-T'] = run_sanity_check_wdrot(
            train_loader, val_loader, test_loader, test_data, device
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        all_issues['W-DRO-T'] = [str(e)]
    
    try:
        results['RVSN'], all_issues['RVSN'] = run_sanity_check_rvsn(
            train_loader, val_loader, test_loader, test_data, device
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        all_issues['RVSN'] = [str(e)]
    
    try:
        results['RSE'], all_issues['RSE'] = run_sanity_check_rse(
            train_loader, val_loader, test_loader, test_data, device
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        all_issues['RSE'] = [str(e)]
    
    try:
        results['SAC-CVaR'], all_issues['SAC-CVaR'] = run_sanity_check_sac_cvar(
            train_loader, val_loader, test_loader, test_data, device
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        all_issues['SAC-CVaR'] = [str(e)]
    
    # Summary
    print("\n" + "="*60)
    print("SANITY CHECK SUMMARY")
    print("="*60)
    
    all_passed = True
    for model, issues in all_issues.items():
        if issues:
            print(f"\n{model}: ISSUES FOUND")
            for issue in issues:
                print(f"  - {issue}")
            all_passed = False
        else:
            print(f"\n{model}: ✓ PASSED")
    
    if all_passed:
        print("\n" + "="*60)
        print("ALL SANITY CHECKS PASSED - Ready for full experiments")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("SOME CHECKS FAILED - Review issues before proceeding")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
