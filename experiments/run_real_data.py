#!/usr/bin/env python3
"""
Real Data Experiments for Deep Hedging.

Validates models on real market data:
- US markets (SPY, QQQ) via yfinance
- Indian markets (NIFTY, Bank NIFTY) via yfinance/nsepy
- Crisis periods (2008, 2020)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import numpy as np
import torch
from datetime import datetime, timedelta

from utils.config import Config, set_seed
from utils.logging_utils import ExperimentLogger
from utils.real_data import (
    MarketDataFetcher, RealDataProcessor, IndianMarketData,
    prepare_crisis_data, get_hedging_data_for_ticker
)
from models.deep_hedging import DeepHedgingModel
from models.kozyra_models import HedgingRNN
from models.baselines import BlackScholesHedge, evaluate_baseline
from train.trainer import DeepHedgingTrainer
from eval.evaluator import HedgingEvaluator
from eval.plotting import plot_pnl_histogram, plot_pnl_boxplot


def run_us_market_experiment(logger: ExperimentLogger, device: str = 'cpu'):
    """Run experiments on US market data."""
    logger.info("=" * 60)
    logger.info("US MARKET EXPERIMENT")
    logger.info("=" * 60)
    
    results = {}
    
    for ticker in ['SPY', 'QQQ', 'AAPL']:
        logger.info(f"\n--- Processing {ticker} ---")
        
        try:
            # Fetch and prepare data
            data = get_hedging_data_for_ticker(
                ticker=ticker,
                lookback_days=500,
                K='ATM',
                is_indian=False
            )
            
            logger.info(f"  Paths: {data['n_paths']}")
            logger.info(f"  Avg volatility: {data['avg_volatility']:.4f}")
            
            # Convert to tensors
            features = torch.tensor(data['features'], dtype=torch.float32)
            stock_paths = torch.tensor(data['stock_paths'], dtype=torch.float32)
            payoffs = torch.tensor(data['payoffs'], dtype=torch.float32)
            
            # Train a simple model
            input_dim = features.shape[-1]
            n_steps = features.shape[1]
            
            model = DeepHedgingModel(
                input_dim=input_dim,
                n_steps=n_steps,
                lambda_risk=1.0,
                share_weights=True
            ).to(device)
            
            # Quick training loop
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            for epoch in range(20):
                model.train()
                optimizer.zero_grad()
                
                deltas = model(features.to(device))
                
                # Compute P&L
                price_changes = stock_paths[:, 1:] - stock_paths[:, :-1]
                hedging_gains = torch.sum(deltas.cpu() * price_changes, dim=1)
                pnl = -payoffs + hedging_gains
                
                # Entropic loss
                scaled = -1.0 * pnl
                loss = torch.logsumexp(scaled, dim=0) - np.log(len(pnl))
                
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                deltas = model(features.to(device)).cpu().numpy()
            
            pnl_np = -payoffs.numpy() + np.sum(deltas * (stock_paths[:, 1:] - stock_paths[:, :-1]).numpy(), axis=1)
            
            results[ticker] = {
                'mean_pnl': np.mean(pnl_np),
                'std_pnl': np.std(pnl_np),
                'var_95': np.percentile(-pnl_np, 95),
                'n_paths': data['n_paths']
            }
            
            logger.info(f"  Mean P&L: {results[ticker]['mean_pnl']:.4f}")
            logger.info(f"  Std P&L: {results[ticker]['std_pnl']:.4f}")
            
        except Exception as e:
            logger.warning(f"  Failed for {ticker}: {e}")
            continue
    
    return results


def run_indian_market_experiment(logger: ExperimentLogger, device: str = 'cpu'):
    """Run experiments on Indian market data."""
    logger.info("=" * 60)
    logger.info("INDIAN MARKET EXPERIMENT")
    logger.info("=" * 60)
    
    results = {}
    indian_data = IndianMarketData()
    
    # Get Indian transaction costs
    tc = indian_data.get_total_cost()
    logger.info(f"Indian market transaction cost: {tc*100:.3f}%")
    
    # Test tickers
    tickers = ['^NSEI', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
    
    for ticker in tickers:
        logger.info(f"\n--- Processing {ticker} ---")
        
        try:
            data = get_hedging_data_for_ticker(
                ticker=ticker,
                lookback_days=365,
                K='ATM',
                is_indian=True
            )
            
            logger.info(f"  Paths: {data['n_paths']}")
            
            # Simple evaluation with BS delta
            features = data['features']
            stock_paths = data['stock_paths']
            payoffs = data['payoffs']
            
            # BS baseline
            bs_hedge = BlackScholesHedge(sigma=data['avg_volatility'], r=0.0)
            bs_deltas = bs_hedge.compute_deltas_vectorized(
                stock_paths, data['time_grid'], data['K'], data['T']
            )
            
            # P&L
            price_changes = np.diff(stock_paths, axis=1)
            hedging_gains = np.sum(bs_deltas * price_changes, axis=1)
            pnl = -payoffs + hedging_gains
            
            results[ticker] = {
                'mean_pnl': np.mean(pnl),
                'std_pnl': np.std(pnl),
                'var_95': np.percentile(-pnl, 95),
                'n_paths': data['n_paths']
            }
            
            logger.info(f"  BS Delta Mean P&L: {results[ticker]['mean_pnl']:.4f}")
            
        except Exception as e:
            logger.warning(f"  Failed for {ticker}: {e}")
            continue
    
    return results


def run_crisis_experiment(logger: ExperimentLogger, device: str = 'cpu'):
    """Run experiments on crisis period data."""
    logger.info("=" * 60)
    logger.info("CRISIS PERIOD EXPERIMENT")
    logger.info("=" * 60)
    
    results = {}
    
    for crisis in ['2008', '2020']:
        logger.info(f"\n--- {crisis} Crisis ---")
        
        try:
            data = prepare_crisis_data(crisis=crisis, ticker='SPY')
            
            logger.info(f"  Paths: {data['n_paths']}")
            logger.info(f"  Avg volatility: {data['avg_volatility']:.4f}")
            
            # Evaluate BS delta during crisis
            bs_hedge = BlackScholesHedge(sigma=data['avg_volatility'], r=0.0)
            bs_deltas = bs_hedge.compute_deltas_vectorized(
                data['stock_paths'], data['time_grid'], data['K'], data['T']
            )
            
            price_changes = np.diff(data['stock_paths'], axis=1)
            hedging_gains = np.sum(bs_deltas * price_changes, axis=1)
            pnl = -data['payoffs'] + hedging_gains
            
            results[crisis] = {
                'mean_pnl': np.mean(pnl),
                'std_pnl': np.std(pnl),
                'var_95': np.percentile(-pnl, 95),
                'volatility': data['avg_volatility']
            }
            
            logger.info(f"  Mean P&L: {results[crisis]['mean_pnl']:.4f}")
            logger.info(f"  Std P&L: {results[crisis]['std_pnl']:.4f}")
            
        except Exception as e:
            logger.warning(f"  Failed for {crisis}: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Real Data Experiments")
    parser.add_argument('--market', type=str, default='all',
                       choices=['all', 'us', 'india', 'crisis'],
                       help='Market to test')
    parser.add_argument('--save_dir', type=str, default='../experiments',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger = ExperimentLogger("real_data", base_dir=args.save_dir)
    
    all_results = {}
    
    if args.market in ['all', 'us']:
        all_results['us'] = run_us_market_experiment(logger, device)
    
    if args.market in ['all', 'india']:
        all_results['india'] = run_indian_market_experiment(logger, device)
    
    if args.market in ['all', 'crisis']:
        all_results['crisis'] = run_crisis_experiment(logger, device)
    
    # Save results
    logger.save_results(all_results)
    
    logger.info("\n" + "=" * 60)
    logger.info("REAL DATA EXPERIMENTS COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
