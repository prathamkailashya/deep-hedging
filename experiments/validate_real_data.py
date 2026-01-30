#!/usr/bin/env python3
"""
Real Data Validation - US and Indian Markets.

Tests hedging models on real market data:
- US: SPY, QQQ via yfinance
- India: NIFTY50, BANKNIFTY via yfinance

Includes market-specific transaction costs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from tqdm import tqdm

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Warning: yfinance not installed. Run: pip install yfinance")

from utils.config import set_seed
from models.kozyra_models import HedgingLSTM
from models.baselines import BlackScholesHedge
from train.trainer import DeepHedgingTrainer

# Output
RESULTS_DIR = Path(__file__).parent / "real_data_results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Settings
TRAIN_YEARS = 3
TEST_MONTHS = 6
HEDGE_HORIZON = 30  # days
DELTA_SCALE = 1.5
BATCH_SIZE = 64

set_seed(42)
device = 'cpu'

# Transaction costs
US_COSTS = {'brokerage': 0.0001, 'slippage': 0.0002}  # ~3 bps total
INDIA_COSTS = {'brokerage': 0.0003, 'stt': 0.001, 'slippage': 0.0005}  # ~18 bps total


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical data from yfinance."""
    if not HAS_YFINANCE:
        raise ImportError("yfinance not installed")
    
    print(f"  Downloading {ticker} from {start} to {end}...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    
    if data.empty:
        raise ValueError(f"No data for {ticker}")
    
    return data


def compute_realized_vol(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute rolling realized volatility."""
    log_returns = np.diff(np.log(prices))
    vol = np.zeros(len(prices))
    
    for i in range(window, len(prices)):
        vol[i] = np.std(log_returns[i-window:i]) * np.sqrt(252)
    
    vol[:window] = vol[window]  # Fill initial values
    return vol


def create_hedging_windows(prices: np.ndarray, vol: np.ndarray, 
                           n_steps: int = 30, stride: int = 5) -> tuple:
    """Create overlapping hedging windows from price data."""
    n = len(prices)
    windows = []
    
    for start in range(0, n - n_steps, stride):
        end = start + n_steps + 1
        window_prices = prices[start:end]
        window_vol = vol[start:end]
        
        # Normalize prices to start at 100
        S0 = window_prices[0]
        norm_prices = window_prices / S0 * 100
        
        # Features: moneyness, tau, vol
        moneyness = norm_prices[:-1] / 100  # S/K where K=100
        tau = np.linspace(30/365, 0, n_steps)
        
        features = np.stack([moneyness, tau, window_vol[:-1]], axis=1)
        
        # ATM call payoff
        payoff = max(0, norm_prices[-1] - 100)
        
        windows.append({
            'features': features,
            'prices': norm_prices,
            'payoff': payoff,
            'S0': S0
        })
    
    return windows


def prepare_dataloaders(windows: list, train_ratio: float = 0.8):
    """Prepare train/test dataloaders from windows."""
    n = len(windows)
    n_train = int(n * train_ratio)
    
    train_windows = windows[:n_train]
    test_windows = windows[n_train:]
    
    def collate(windows):
        features = torch.tensor(np.array([w['features'] for w in windows]), dtype=torch.float32)
        prices = torch.tensor(np.array([w['prices'] for w in windows]), dtype=torch.float32)
        payoffs = torch.tensor(np.array([w['payoff'] for w in windows]), dtype=torch.float32)
        
        return {
            'features': features,
            'stock_paths': prices,
            'payoff': payoffs
        }
    
    train_loader = torch.utils.data.DataLoader(
        train_windows, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_windows, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )
    
    return train_loader, test_loader, len(train_windows), len(test_windows)


def compute_metrics(pnl: np.ndarray, deltas: np.ndarray) -> dict:
    """Compute hedging metrics."""
    losses = -pnl
    delta_changes = np.abs(np.diff(
        np.concatenate([np.zeros((len(deltas), 1)), deltas, np.zeros((len(deltas), 1))], axis=1), axis=1
    ))
    volume = np.mean(np.sum(delta_changes, axis=1))
    
    return {
        'mean_pnl': np.mean(pnl),
        'std_pnl': np.std(pnl),
        'var_95': np.percentile(losses, 95),
        'cvar_95': np.mean(losses[losses >= np.percentile(losses, 95)]) if len(losses[losses >= np.percentile(losses, 95)]) > 0 else 0,
        'trading_volume': volume,
        'max_delta': np.max(np.abs(deltas)),
        'n_samples': len(pnl)
    }


def train_and_evaluate(train_loader, test_loader, input_dim: int, cost_multiplier: float = 0.0):
    """Train LSTM and evaluate."""
    model = HedgingLSTM(
        state_dim=input_dim,
        hidden_size=50,
        num_layers=2,
        delta_scale=DELTA_SCALE
    ).to(device)
    
    trainer = DeepHedgingTrainer(
        model=model,
        lambda_risk=1.0,
        cost_multiplier=cost_multiplier,
        learning_rate=0.0005,
        device=device
    )
    
    trainer.fit(train_loader, train_loader, n_epochs=30, patience=10, verbose=True)
    metrics, pnl, deltas = trainer.evaluate(test_loader)
    
    return compute_metrics(pnl, deltas), pnl, deltas


def evaluate_bs_hedge(test_loader, sigma: float = 0.2):
    """Evaluate Black-Scholes delta hedge."""
    bs = BlackScholesHedge(sigma=sigma, r=0.0)
    
    all_pnl, all_deltas = [], []
    
    for batch in test_loader:
        prices = batch['stock_paths'].numpy()
        payoffs = batch['payoff'].numpy()
        
        time_grid = np.linspace(0, 30/365, 31)
        deltas = bs.compute_deltas_vectorized(prices, time_grid, K=100.0, T=30/365)
        
        price_changes = np.diff(prices, axis=1)
        hedging_gains = np.sum(deltas * price_changes, axis=1)
        pnl = -payoffs + hedging_gains
        
        all_pnl.append(pnl)
        all_deltas.append(deltas)
    
    pnl = np.concatenate(all_pnl)
    deltas = np.concatenate(all_deltas)
    
    return compute_metrics(pnl, deltas), pnl, deltas


def run_market_validation(ticker: str, market: str, costs: dict):
    """Run validation for a specific market/ticker."""
    print(f"\n{'='*60}")
    print(f"{market}: {ticker}")
    print(f"{'='*60}")
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=TRAIN_YEARS * 365 + TEST_MONTHS * 30)
    
    try:
        data = download_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    except Exception as e:
        print(f"  Failed to download data: {e}")
        return None
    
    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close'][ticker].values if ticker in data['Close'].columns else data['Close'].iloc[:, 0].values
    else:
        prices = data['Close'].values
    
    print(f"  Data points: {len(prices)}")
    print(f"  Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    # Compute realized volatility
    vol = compute_realized_vol(prices)
    avg_vol = np.mean(vol[vol > 0])
    print(f"  Average volatility: {avg_vol:.2%}")
    
    # Create hedging windows
    windows = create_hedging_windows(prices, vol, n_steps=HEDGE_HORIZON)
    print(f"  Hedging windows: {len(windows)}")
    
    if len(windows) < 100:
        print(f"  Insufficient data for training")
        return None
    
    # Prepare data
    train_loader, test_loader, n_train, n_test = prepare_dataloaders(windows)
    print(f"  Train: {n_train}, Test: {n_test}")
    
    input_dim = 3  # moneyness, tau, vol
    cost_mult = sum(costs.values())
    
    results = {}
    
    # Train LSTM
    print(f"\n  Training LSTM (cost={cost_mult:.4f})...")
    lstm_metrics, lstm_pnl, lstm_deltas = train_and_evaluate(
        train_loader, test_loader, input_dim, cost_mult
    )
    results['LSTM'] = {'metrics': lstm_metrics, 'pnl': lstm_pnl, 'deltas': lstm_deltas}
    print(f"    Std P&L: {lstm_metrics['std_pnl']:.4f}, CVaR95: {lstm_metrics['cvar_95']:.4f}")
    
    # BS Hedge baseline
    print(f"\n  Evaluating BS Delta (sigma={avg_vol:.2%})...")
    bs_metrics, bs_pnl, bs_deltas = evaluate_bs_hedge(test_loader, sigma=avg_vol)
    results['BS Delta'] = {'metrics': bs_metrics, 'pnl': bs_pnl, 'deltas': bs_deltas}
    print(f"    Std P&L: {bs_metrics['std_pnl']:.4f}, CVaR95: {bs_metrics['cvar_95']:.4f}")
    
    return {
        'ticker': ticker,
        'market': market,
        'n_samples': len(prices),
        'avg_vol': avg_vol,
        'results': results
    }


def generate_report(all_results: dict):
    """Generate figures and save results."""
    print(f"\n{'='*60}")
    print("GENERATING REPORT")
    print(f"{'='*60}")
    
    # Summary table
    rows = []
    for market, data in all_results.items():
        if data is None:
            continue
        for model, res in data['results'].items():
            rows.append({
                'Market': market,
                'Ticker': data['ticker'],
                'Model': model,
                'Avg Vol': data['avg_vol'],
                **res['metrics']
            })
    
    if not rows:
        print("No results to report")
        return
    
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "real_data_results.csv", index=False, float_format='%.4f')
    
    # Print summary
    print("\n" + "="*100)
    print("REAL DATA VALIDATION SUMMARY")
    print("="*100)
    print(f"{'Market':<12} {'Ticker':<10} {'Model':<12} {'Vol':>8} {'Std P&L':>10} {'CVaR95':>10} {'Volume':>10}")
    print("-"*100)
    for _, row in df.iterrows():
        print(f"{row['Market']:<12} {row['Ticker']:<10} {row['Model']:<12} "
              f"{row['Avg Vol']:>8.2%} {row['std_pnl']:>10.4f} {row['cvar_95']:>10.4f} {row['trading_volume']:>10.4f}")
    print("="*100)
    
    # Generate figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    markets = df['Market'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    
    for i, (_, row) in enumerate(df.iterrows()):
        label = f"{row['Market']}-{row['Model']}"
        axes[0].bar(i, row['std_pnl'], color=colors[i], label=label)
        axes[1].bar(i, row['cvar_95'], color=colors[i])
    
    axes[0].set_ylabel('Std P&L')
    axes[0].set_title('P&L Volatility by Market/Model')
    axes[0].set_xticks(range(len(df)))
    axes[0].set_xticklabels([f"{r['Market'][:3]}-{r['Model'][:4]}" for _, r in df.iterrows()], rotation=45, ha='right')
    
    axes[1].set_ylabel('CVaR 95%')
    axes[1].set_title('Tail Risk by Market/Model')
    axes[1].set_xticks(range(len(df)))
    axes[1].set_xticklabels([f"{r['Market'][:3]}-{r['Model'][:4]}" for _, r in df.iterrows()], rotation=45, ha='right')
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "real_data_comparison.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "real_data_comparison.png", dpi=150)
    plt.close()
    
    print(f"\nResults saved to: {RESULTS_DIR}")


def main():
    print("="*60)
    print("REAL DATA VALIDATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    if not HAS_YFINANCE:
        print("\nERROR: yfinance not installed. Install with: pip install yfinance")
        return
    
    all_results = {}
    
    # US Markets
    print("\n" + "="*60)
    print("US MARKETS")
    print("="*60)
    
    all_results['US-SPY'] = run_market_validation('SPY', 'US', US_COSTS)
    
    # Indian Markets
    print("\n" + "="*60)
    print("INDIAN MARKETS")
    print("="*60)
    
    # Try NIFTY50 via yfinance
    all_results['India-NIFTY'] = run_market_validation('^NSEI', 'India', INDIA_COSTS)
    
    # Generate report
    generate_report(all_results)
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✓ REAL DATA VALIDATION COMPLETE")


if __name__ == '__main__':
    main()
