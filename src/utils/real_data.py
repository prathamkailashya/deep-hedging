"""
Real Market Data Integration.

Fetches and processes real market data from:
- yfinance for US markets
- nsepy/nsetools for Indian markets (NSE)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from datetime import datetime, timedelta
import warnings

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    warnings.warn("yfinance not installed. US market data unavailable.")

try:
    from nsepy import get_history
    HAS_NSEPY = True
except ImportError:
    HAS_NSEPY = False

try:
    from nsetools import Nse
    HAS_NSETOOLS = True
except ImportError:
    HAS_NSETOOLS = False


class MarketDataFetcher:
    """
    Fetch real market data from various sources.
    """
    
    def __init__(self):
        self.cache = {}
    
    def fetch_us_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch US market data using yfinance.
        
        Args:
            ticker: Stock ticker (e.g., 'SPY', 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', '5m', etc.)
        
        Returns:
            DataFrame with OHLCV data
        """
        if not HAS_YFINANCE:
            raise ImportError("yfinance is required for US data")
        
        cache_key = f"{ticker}_{start_date}_{end_date}_{interval}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        
        self.cache[cache_key] = df
        return df
    
    def fetch_indian_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        index: bool = False
    ) -> pd.DataFrame:
        """
        Fetch Indian market data from NSE.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS') or index ('^NSEI')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            index: Whether the symbol is an index
        
        Returns:
            DataFrame with OHLCV data
        """
        # Try yfinance first (works for indices like ^NSEI)
        if HAS_YFINANCE and symbol.startswith('^'):
            try:
                df = self.fetch_us_data(symbol, start_date, end_date)
                return df
            except:
                pass
        
        # Try nsepy
        if HAS_NSEPY:
            try:
                start = datetime.strptime(start_date, '%Y-%m-%d')
                end = datetime.strptime(end_date, '%Y-%m-%d')
                
                df = get_history(
                    symbol=symbol,
                    start=start,
                    end=end,
                    index=index
                )
                
                if not df.empty:
                    return df
            except Exception as e:
                warnings.warn(f"nsepy failed: {e}")
        
        # Try yfinance with .NS suffix
        if HAS_YFINANCE:
            try:
                ticker = f"{symbol}.NS"
                df = self.fetch_us_data(ticker, start_date, end_date)
                return df
            except:
                pass
        
        raise ValueError(f"Could not fetch data for {symbol}")
    
    def fetch_options_data(
        self,
        ticker: str,
        expiry_date: Optional[str] = None
    ) -> Dict:
        """
        Fetch options chain data.
        
        Args:
            ticker: Underlying ticker
            expiry_date: Option expiry date
        
        Returns:
            Dictionary with calls and puts DataFrames
        """
        if not HAS_YFINANCE:
            raise ImportError("yfinance required for options data")
        
        stock = yf.Ticker(ticker)
        
        # Get available expiries
        expiries = stock.options
        
        if not expiries:
            raise ValueError(f"No options data for {ticker}")
        
        if expiry_date is None:
            expiry_date = expiries[0]
        elif expiry_date not in expiries:
            # Find closest expiry
            expiry_date = min(expiries, key=lambda x: abs(
                datetime.strptime(x, '%Y-%m-%d') - 
                datetime.strptime(expiry_date, '%Y-%m-%d')
            ))
        
        opt = stock.option_chain(expiry_date)
        
        return {
            'calls': opt.calls,
            'puts': opt.puts,
            'expiry': expiry_date
        }


class RealDataProcessor:
    """
    Process real market data for hedging experiments.
    """
    
    def __init__(self, fetcher: Optional[MarketDataFetcher] = None):
        self.fetcher = fetcher or MarketDataFetcher()
    
    def compute_realized_volatility(
        self,
        prices: pd.Series,
        window: int = 30,
        annualize: bool = True
    ) -> pd.Series:
        """
        Compute realized volatility.
        
        Args:
            prices: Price series
            window: Rolling window size
            annualize: Whether to annualize volatility
        
        Returns:
            Realized volatility series
        """
        log_returns = np.log(prices / prices.shift(1))
        vol = log_returns.rolling(window=window).std()
        
        if annualize:
            vol = vol * np.sqrt(252)  # Assuming daily data
        
        return vol
    
    def compute_implied_volatility_proxy(
        self,
        prices: pd.Series,
        K: float,
        T: float,
        r: float = 0.0
    ) -> float:
        """
        Estimate implied volatility from historical data.
        
        Uses the simple approach of recent realized volatility
        as a proxy for implied volatility.
        """
        recent_vol = self.compute_realized_volatility(prices[-60:], window=30)
        return recent_vol.iloc[-1] if not recent_vol.empty else 0.2
    
    def prepare_hedging_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        K: float,
        T_days: int = 30,
        n_steps: int = 30,
        cost_multiplier: float = 0.001,
        is_indian: bool = False
    ) -> Dict:
        """
        Prepare real market data for hedging experiments.
        
        Constructs synthetic option payoffs using realized volatility.
        
        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            K: Strike price (can be 'ATM' for at-the-money)
            T_days: Option maturity in days
            n_steps: Number of hedging steps
            cost_multiplier: Transaction cost
            is_indian: Whether this is Indian market data
        
        Returns:
            Dictionary with processed data
        """
        # Fetch data
        if is_indian:
            df = self.fetcher.fetch_indian_data(ticker, start_date, end_date)
        else:
            df = self.fetcher.fetch_us_data(ticker, start_date, end_date)
        
        prices = df['Close'].values
        
        # Compute volatility
        vol = self.compute_realized_volatility(df['Close'])
        
        # Create rolling windows as "paths"
        window_size = n_steps + 1
        n_paths = len(prices) - window_size + 1
        
        if n_paths < 100:
            warnings.warn(f"Only {n_paths} paths available. Consider longer date range.")
        
        stock_paths = np.zeros((n_paths, window_size))
        for i in range(n_paths):
            stock_paths[i] = prices[i:i + window_size]
        
        # Normalize paths to start at 100
        S0 = stock_paths[:, 0:1]
        normalized_paths = stock_paths / S0 * 100
        
        # Set strike
        if K == 'ATM':
            K = 100.0  # ATM for normalized paths
        
        # Compute payoffs
        S_T = normalized_paths[:, -1]
        payoffs = np.maximum(S_T - K, 0)
        
        # Create features
        T = T_days / 365
        dt = T / n_steps
        time_grid = np.linspace(0, T, n_steps + 1)
        
        features = []
        for k in range(n_steps):
            ttm = (T - time_grid[k]) / T
            S_k = normalized_paths[:, k]
            
            norm_price = S_k / 100
            log_moneyness = np.log(S_k / K)
            
            step_features = np.stack([
                norm_price,
                np.full(n_paths, ttm),
                log_moneyness
            ], axis=-1)
            
            features.append(step_features)
        
        features = np.stack(features, axis=1)  # (n_paths, n_steps, n_features)
        
        # Estimate volatility for BS delta
        avg_vol = vol.mean()
        if np.isnan(avg_vol):
            avg_vol = 0.2
        
        return {
            'features': features,
            'stock_paths': normalized_paths,
            'payoffs': payoffs,
            'raw_prices': prices,
            'volatility': vol.values,
            'avg_volatility': avg_vol,
            'time_grid': time_grid,
            'K': K,
            'T': T,
            'n_paths': n_paths,
            'n_steps': n_steps,
            'cost_multiplier': cost_multiplier,
            'ticker': ticker
        }


class IndianMarketData:
    """
    Specialized handler for Indian market data.
    
    Handles:
    - NIFTY 50 (^NSEI)
    - Bank NIFTY (BANKNIFTY / ^NSEBANK)
    - Individual stocks with .NS suffix
    """
    
    # Indian market transaction cost assumptions
    TRANSACTION_COSTS = {
        'brokerage': 0.0003,  # 0.03% typical discount broker
        'stt': 0.001,         # Securities Transaction Tax (sell side)
        'exchange': 0.0000325, # Exchange charges
        'gst': 0.18,          # GST on brokerage
        'stamp_duty': 0.00015, # Stamp duty
        'sebi': 0.000001      # SEBI charges
    }
    
    @classmethod
    def get_total_cost(cls) -> float:
        """Get total transaction cost as fraction of trade value."""
        costs = cls.TRANSACTION_COSTS
        brokerage_with_gst = costs['brokerage'] * (1 + costs['gst'])
        total = brokerage_with_gst + costs['stt'] + costs['exchange'] + \
                costs['stamp_duty'] + costs['sebi']
        return total  # Approximately 0.15-0.2%
    
    @staticmethod
    def get_nifty_tickers() -> List[str]:
        """Get list of NIFTY 50 component tickers."""
        return [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS',
            'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
            'SUNPHARMA.NS', 'BAJFINANCE.NS', 'WIPRO.NS', 'ULTRACEMCO.NS', 'HCLTECH.NS'
        ]
    
    def fetch_nifty_index(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch NIFTY 50 index data."""
        fetcher = MarketDataFetcher()
        
        # Try multiple symbols
        for symbol in ['^NSEI', 'NIFTY_50.NS', '^NSEI']:
            try:
                return fetcher.fetch_us_data(symbol, start_date, end_date)
            except:
                continue
        
        raise ValueError("Could not fetch NIFTY data")
    
    def fetch_banknifty(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch Bank NIFTY index data."""
        fetcher = MarketDataFetcher()
        
        for symbol in ['^NSEBANK', 'BANKNIFTY.NS']:
            try:
                return fetcher.fetch_us_data(symbol, start_date, end_date)
            except:
                continue
        
        raise ValueError("Could not fetch Bank NIFTY data")


def prepare_crisis_data(
    crisis: str = '2008',
    ticker: str = 'SPY'
) -> Dict:
    """
    Prepare data for crisis period testing.
    
    Args:
        crisis: '2008' for financial crisis, '2020' for COVID
        ticker: Stock ticker
    
    Returns:
        Processed data dictionary
    """
    if crisis == '2008':
        start_date = '2007-01-01'
        end_date = '2009-12-31'
    elif crisis == '2020':
        start_date = '2019-01-01'
        end_date = '2022-12-31'
    else:
        raise ValueError(f"Unknown crisis: {crisis}")
    
    processor = RealDataProcessor()
    return processor.prepare_hedging_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        K='ATM',
        T_days=30,
        n_steps=30
    )


def get_hedging_data_for_ticker(
    ticker: str,
    lookback_days: int = 365,
    K: float = 'ATM',
    is_indian: bool = False
) -> Dict:
    """
    Convenience function to get hedging data for any ticker.
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    processor = RealDataProcessor()
    return processor.prepare_hedging_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        K=K,
        is_indian=is_indian
    )
