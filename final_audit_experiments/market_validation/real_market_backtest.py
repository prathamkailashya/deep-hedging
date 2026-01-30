"""
PART 4: Real Market Validation

Backtest deep hedging strategies on:
- SPY options (US market)
- NIFTY options (Indian market)

Includes:
- Discrete strikes
- Transaction costs
- Slippage
- Rebalancing constraints
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    warnings.warn("yfinance not installed. Real market data will use synthetic fallback.")


@dataclass
class MarketConfig:
    """Market-specific configuration."""
    name: str
    ticker: str
    transaction_cost_bps: float  # Basis points
    slippage_bps: float
    min_trade_size: float
    rebalance_frequency: str  # 'daily', 'hourly'
    trading_hours: Tuple[int, int]  # (start_hour, end_hour)


# Market configurations
US_MARKET = MarketConfig(
    name='US_SPY',
    ticker='SPY',
    transaction_cost_bps=3.0,  # ~3 bps for SPY options
    slippage_bps=2.0,
    min_trade_size=0.01,
    rebalance_frequency='daily',
    trading_hours=(9, 16)
)

INDIA_MARKET = MarketConfig(
    name='INDIA_NIFTY',
    ticker='^NSEI',  # NIFTY 50 index
    transaction_cost_bps=18.0,  # Higher costs in India
    slippage_bps=5.0,
    min_trade_size=0.01,
    rebalance_frequency='daily',
    trading_hours=(9, 15)
)


@dataclass
class BacktestResult:
    """Complete backtest results."""
    market: str
    model_name: str
    
    # P&L metrics
    total_pnl: float
    mean_daily_pnl: float
    std_daily_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Risk metrics
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    
    # Trading metrics
    total_trades: int
    avg_trade_size: float
    total_transaction_costs: float
    total_slippage: float
    turnover: float
    
    # Time series
    pnl_series: np.ndarray
    delta_series: np.ndarray
    
    # Comparison to benchmark
    vs_bs_delta: float  # Improvement over BS delta


class BlackScholesDelta:
    """Black-Scholes delta calculator for benchmark."""
    
    @staticmethod
    def call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Compute BS call delta."""
        from scipy.stats import norm
        
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return float(norm.cdf(d1))
    
    @staticmethod
    def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Compute BS put delta."""
        return BlackScholesDelta.call_delta(S, K, T, r, sigma) - 1.0


class LelandDelta:
    """Leland's modified delta with transaction cost adjustment."""
    
    @staticmethod
    def call_delta(
        S: float, K: float, T: float, r: float, sigma: float,
        kappa: float, dt: float
    ) -> float:
        """
        Leland's delta adjustment.
        
        kappa = transaction cost (proportional)
        """
        from scipy.stats import norm
        
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        # Adjusted volatility
        sigma_adj = sigma * np.sqrt(1 + np.sqrt(2/np.pi) * kappa / (sigma * np.sqrt(dt)))
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma_adj**2) * T) / (sigma_adj * np.sqrt(T))
        return float(norm.cdf(d1))


class MarketDataLoader:
    """Load real market data."""
    
    def __init__(self, config: MarketConfig):
        self.config = config
    
    def load_data(
        self,
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """Load market data from yfinance."""
        
        if not HAS_YFINANCE:
            return self._generate_synthetic_data(start_date, end_date)
        
        try:
            ticker = yf.Ticker(self.config.ticker)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                warnings.warn(f"No data for {self.config.ticker}, using synthetic")
                return self._generate_synthetic_data(start_date, end_date)
            
            # Add implied volatility estimate (using realized vol as proxy)
            df['returns'] = df['Close'].pct_change()
            df['realized_vol'] = df['returns'].rolling(20).std() * np.sqrt(252)
            df['implied_vol'] = df['realized_vol'].fillna(0.2)  # Default 20%
            
            return df
            
        except Exception as e:
            warnings.warn(f"Failed to load {self.config.ticker}: {e}")
            return self._generate_synthetic_data(start_date, end_date)
    
    def _generate_synthetic_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Generate synthetic data matching market characteristics."""
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='B')  # Business days
        
        n = len(dates)
        np.random.seed(42)
        
        # GBM with realistic parameters
        S0 = 400 if 'SPY' in self.config.ticker else 20000
        mu = 0.08  # 8% annual drift
        sigma = 0.18  # 18% annual vol
        
        dt = 1/252
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n)
        prices = S0 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            'High': prices * (1 + np.random.uniform(0, 0.02, n)),
            'Low': prices * (1 - np.random.uniform(0, 0.02, n)),
            'Close': prices,
            'Volume': np.random.uniform(1e6, 5e6, n),
            'returns': returns,
            'realized_vol': pd.Series(returns).rolling(20).std() * np.sqrt(252),
            'implied_vol': sigma + np.random.uniform(-0.02, 0.02, n)
        }, index=dates)
        
        df['realized_vol'] = df['realized_vol'].fillna(sigma)
        
        return df


class RealMarketBacktester:
    """
    Backtest hedging strategies on real market data.
    
    Implements realistic constraints:
    - Transaction costs
    - Slippage
    - Discrete rebalancing
    - Position limits
    """
    
    def __init__(
        self,
        market_config: MarketConfig,
        model: nn.Module,
        model_name: str,
        device: str = 'cpu'
    ):
        self.config = market_config
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.model.eval()
    
    def run_backtest(
        self,
        market_data: pd.DataFrame,
        option_type: str = 'call',
        moneyness: float = 1.0,  # ATM
        days_to_expiry: int = 30,
        r: float = 0.05
    ) -> BacktestResult:
        """
        Run complete backtest.
        
        Simulates hedging a portfolio of options over the data period.
        """
        
        prices = market_data['Close'].values
        vols = market_data['implied_vol'].values
        n_days = len(prices)
        
        # Option parameters
        K = prices[0] * moneyness
        T_initial = days_to_expiry / 252
        
        # Initialize tracking arrays
        pnl_series = np.zeros(n_days)
        delta_series = np.zeros(n_days)
        tc_series = np.zeros(n_days)
        slippage_series = np.zeros(n_days)
        
        # Initial position
        prev_delta = 0.0
        cumulative_pnl = 0.0
        
        for t in range(n_days):
            S = prices[t]
            sigma = vols[t]
            T = max(T_initial - t / 252, 1e-6)
            
            # Get model delta
            features = self._prepare_features(prices[:t+1], vols[:t+1], K, T)
            with torch.no_grad():
                model_delta = self._get_model_delta(features)
            
            # Apply position limits
            model_delta = np.clip(model_delta, -1.5, 1.5)
            
            # Rebalancing decision (with min trade size)
            delta_change = model_delta - prev_delta
            if abs(delta_change) < self.config.min_trade_size:
                delta_change = 0
                model_delta = prev_delta
            
            # Transaction costs
            tc = abs(delta_change) * S * self.config.transaction_cost_bps / 10000
            
            # Slippage
            slippage = abs(delta_change) * S * self.config.slippage_bps / 10000
            
            # P&L from delta hedging
            if t > 0:
                price_change = prices[t] - prices[t-1]
                hedge_pnl = prev_delta * price_change
                
                # Option value change (approximation)
                # For simplicity, use delta as proxy
                option_pnl = -BlackScholesDelta.call_delta(S, K, T, r, sigma) * price_change
                
                daily_pnl = hedge_pnl + option_pnl - tc - slippage
                cumulative_pnl += daily_pnl
            else:
                daily_pnl = 0
            
            # Store
            pnl_series[t] = cumulative_pnl
            delta_series[t] = model_delta
            tc_series[t] = tc
            slippage_series[t] = slippage
            
            prev_delta = model_delta
        
        # Compute metrics
        daily_pnl = np.diff(pnl_series, prepend=0)
        
        # Risk metrics
        var_95 = np.percentile(-daily_pnl, 95)
        var_99 = np.percentile(-daily_pnl, 99)
        cvar_95 = np.mean(-daily_pnl[-daily_pnl >= -var_95]) if var_95 > 0 else var_95
        cvar_99 = np.mean(-daily_pnl[-daily_pnl >= -var_99]) if var_99 > 0 else var_99
        
        # Sharpe ratio (annualized)
        sharpe = np.mean(daily_pnl) / (np.std(daily_pnl) + 1e-10) * np.sqrt(252)
        
        # Max drawdown
        cummax = np.maximum.accumulate(pnl_series)
        drawdown = cummax - pnl_series
        max_drawdown = np.max(drawdown)
        
        # Trading metrics
        delta_changes = np.abs(np.diff(delta_series, prepend=0))
        n_trades = np.sum(delta_changes > self.config.min_trade_size)
        turnover = np.sum(delta_changes)
        
        # Compare to BS delta benchmark
        bs_pnl = self._run_bs_benchmark(prices, vols, K, T_initial, r)
        vs_bs = cumulative_pnl - bs_pnl
        
        return BacktestResult(
            market=self.config.name,
            model_name=self.model_name,
            total_pnl=float(cumulative_pnl),
            mean_daily_pnl=float(np.mean(daily_pnl)),
            std_daily_pnl=float(np.std(daily_pnl)),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_drawdown),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            var_99=float(var_99),
            cvar_99=float(cvar_99),
            total_trades=int(n_trades),
            avg_trade_size=float(np.mean(delta_changes[delta_changes > 0])) if n_trades > 0 else 0,
            total_transaction_costs=float(np.sum(tc_series)),
            total_slippage=float(np.sum(slippage_series)),
            turnover=float(turnover),
            pnl_series=pnl_series,
            delta_series=delta_series,
            vs_bs_delta=float(vs_bs)
        )
    
    def _prepare_features(
        self,
        prices: np.ndarray,
        vols: np.ndarray,
        K: float,
        T: float
    ) -> torch.Tensor:
        """Prepare features for model input."""
        
        n = len(prices)
        target_len = 30  # Model expects 30 steps
        
        # Pad or truncate
        if n < target_len:
            prices = np.pad(prices, (target_len - n, 0), mode='edge')
            vols = np.pad(vols, (target_len - n, 0), mode='edge')
        else:
            prices = prices[-target_len:]
            vols = vols[-target_len:]
        
        # Compute features: [S_norm, log_moneyness, vol, tau]
        S_norm = prices / prices[0]
        log_money = np.log(prices / K)
        tau = np.linspace(T, 0, target_len)
        
        features = np.stack([S_norm, log_money, vols, tau], axis=-1)
        
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)
    
    def _get_model_delta(self, features: torch.Tensor) -> float:
        """Get delta from model."""
        with torch.no_grad():
            deltas = self.model(features)
            return float(deltas[0, -1].cpu().numpy())
    
    def _run_bs_benchmark(
        self,
        prices: np.ndarray,
        vols: np.ndarray,
        K: float,
        T_initial: float,
        r: float
    ) -> float:
        """Run Black-Scholes delta hedge benchmark."""
        
        n_days = len(prices)
        cumulative_pnl = 0.0
        prev_delta = 0.0
        
        for t in range(n_days):
            S = prices[t]
            sigma = vols[t]
            T = max(T_initial - t / 252, 1e-6)
            
            bs_delta = BlackScholesDelta.call_delta(S, K, T, r, sigma)
            
            if t > 0:
                price_change = prices[t] - prices[t-1]
                hedge_pnl = prev_delta * price_change
                option_pnl = -bs_delta * price_change
                
                tc = abs(bs_delta - prev_delta) * S * self.config.transaction_cost_bps / 10000
                cumulative_pnl += hedge_pnl + option_pnl - tc
            
            prev_delta = bs_delta
        
        return cumulative_pnl


def run_market_validation(
    models: Dict[str, nn.Module],
    markets: List[MarketConfig] = None,
    start_date: str = '2023-01-01',
    end_date: str = '2024-01-01',
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Run complete market validation for all models.
    
    Args:
        models: {model_name: model}
        markets: List of market configs
        start_date, end_date: Backtest period
    
    Returns:
        Comprehensive backtest results
    """
    
    if markets is None:
        markets = [US_MARKET, INDIA_MARKET]
    
    print("=" * 70)
    print("REAL MARKET VALIDATION")
    print("=" * 70)
    
    all_results = {}
    
    for market in markets:
        print(f"\n{'='*50}")
        print(f"Market: {market.name}")
        print(f"{'='*50}")
        
        # Load data
        loader = MarketDataLoader(market)
        data = loader.load_data(start_date, end_date)
        print(f"  Data points: {len(data)}")
        
        market_results = {}
        
        for model_name, model in models.items():
            print(f"\n  Testing {model_name}...")
            
            backtester = RealMarketBacktester(
                market_config=market,
                model=model,
                model_name=model_name,
                device=device
            )
            
            result = backtester.run_backtest(data)
            market_results[model_name] = result
            
            print(f"    P&L: {result.total_pnl:.2f}")
            print(f"    Sharpe: {result.sharpe_ratio:.3f}")
            print(f"    CVaR95: {result.cvar_95:.4f}")
            print(f"    vs BS: {result.vs_bs_delta:+.2f}")
        
        all_results[market.name] = market_results
    
    return all_results


if __name__ == '__main__':
    # Example usage
    from new_experiments.models.base_model import LSTMHedger
    from new_experiments.models.attention_lstm import AttentionLSTM
    
    # Create models
    models = {
        'LSTM': LSTMHedger(input_dim=4, hidden_size=50),
        'AttentionLSTM': AttentionLSTM(input_dim=4, hidden_size=64)
    }
    
    # Run validation
    results = run_market_validation(models)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for market, market_results in results.items():
        print(f"\n{market}:")
        for model_name, result in market_results.items():
            print(f"  {model_name}: Sharpe={result.sharpe_ratio:.3f}, CVaR95={result.cvar_95:.4f}")
