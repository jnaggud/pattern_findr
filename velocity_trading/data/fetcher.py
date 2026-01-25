"""
Price data fetching for velocity trading.

Provides unified interface for fetching price data from multiple sources:
- yfinance (stocks, ETFs, crypto)
- Polygon (premium crypto data)
- Databento (futures)

This module re-uses existing infrastructure from the Pattern_FindR codebase
where possible, providing a clean interface for the new trading system.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pandas as pd

# Add parent directory to path to import existing modules
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import from existing infrastructure
try:
    from data_cache import fetch_and_cache
    HAS_DATA_CACHE = True
except ImportError:
    HAS_DATA_CACHE = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import databento as db
    HAS_DATABENTO = True
except ImportError:
    HAS_DATABENTO = False


# Futures ticker mapping
FUTURES_TICKERS = {
    'ES=F': 'ES.FUT',    # S&P 500 E-mini
    'NQ=F': 'NQ.FUT',    # Nasdaq 100 E-mini
    'YM=F': 'YM.FUT',    # Dow E-mini
    'GC=F': 'GC.FUT',    # Gold
    'CL=F': 'CL.FUT',    # Crude Oil
    'SI=F': 'SI.FUT',    # Silver
    'RTY=F': 'RTY.FUT',  # Russell 2000 E-mini
}

# Crypto tickers
CRYPTO_TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'XRP-USD']


def is_futures_ticker(ticker: str) -> bool:
    """Check if ticker is a futures contract."""
    return ticker in FUTURES_TICKERS or ticker.endswith('=F')


def is_crypto_ticker(ticker: str) -> bool:
    """Check if ticker is a cryptocurrency."""
    return ticker in CRYPTO_TICKERS or ticker.endswith('-USD')


def fetch_price_data(
    ticker: str,
    days: int = 200,
    interval: str = '1d',
    use_cache: bool = True
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV price data for a ticker.

    Args:
        ticker: Trading symbol (e.g., 'SPY', 'ES=F', 'BTC-USD')
        days: Number of days of history to fetch
        interval: Bar interval ('1d', '15m', '1h', etc.)
        use_cache: Whether to use the data_cache module

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index is datetime
        Returns None if fetch fails
    """
    # Try cached fetch first (if available)
    if use_cache and HAS_DATA_CACHE:
        try:
            df = fetch_and_cache(ticker, days=days, interval=interval)
            if df is not None and not df.empty:
                return _normalize_dataframe(df)
        except Exception as e:
            print(f"   Cache fetch failed for {ticker}: {e}")

    # Determine fetch method based on ticker type
    if is_futures_ticker(ticker):
        df = _fetch_futures(ticker, days, interval)
    elif is_crypto_ticker(ticker):
        df = _fetch_crypto(ticker, days, interval)
    else:
        df = _fetch_stock(ticker, days, interval)

    if df is not None and not df.empty:
        return _normalize_dataframe(df)

    return None


def _fetch_stock(ticker: str, days: int, interval: str) -> Optional[pd.DataFrame]:
    """Fetch stock/ETF data via yfinance."""
    if not HAS_YFINANCE:
        print("   yfinance not available")
        return None

    try:
        yf_ticker = yf.Ticker(ticker)

        # For intraday intervals, use period (yfinance has strict 60-day limit)
        intraday_intervals = {'1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h'}

        if interval in intraday_intervals:
            # Use period for intraday - max 60 days
            period = '60d' if days >= 60 else f'{days}d'
            df = yf_ticker.history(period=period, interval=interval)
        else:
            # Use date range for daily+
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            df = yf_ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval
            )

        return df if not df.empty else None

    except Exception as e:
        print(f"   yfinance fetch failed for {ticker}: {e}")
        return None


def _fetch_crypto(ticker: str, days: int, interval: str) -> Optional[pd.DataFrame]:
    """Fetch cryptocurrency data via yfinance (or Polygon if available)."""
    # First try yfinance (free, reliable for crypto)
    df = _fetch_stock(ticker, days, interval)
    if df is not None:
        return df

    # Could add Polygon fallback here
    return None


def _fetch_futures(ticker: str, days: int, interval: str) -> Optional[pd.DataFrame]:
    """Fetch futures data via Databento (preferred) or yfinance (fallback).

    Databento Historical API has ~1 hour delay, so for intraday data we check
    freshness and fall back to yfinance if data is too stale.
    """
    # Minimum bars needed for oscillator calculation (need at least 50 for proper signals)
    MIN_BARS_REQUIRED = 50
    # Maximum data age before falling back to yfinance (90 minutes)
    MAX_DATA_AGE_MINUTES = 90

    # Try Databento first for better futures data
    # Skip on weekends when CME is closed (avoids async errors)
    from datetime import datetime as dt_check
    now_check = dt_check.now()
    weekday = now_check.weekday()  # 0=Monday, 5=Saturday, 6=Sunday
    skip_databento = (weekday == 5) or (weekday == 6 and now_check.hour < 17)

    databento_df = None
    if HAS_DATABENTO and not skip_databento:
        try:
            databento_df = _fetch_futures_databento(ticker, days, interval)
            if databento_df is not None and len(databento_df) >= MIN_BARS_REQUIRED:
                # Check data freshness for intraday intervals
                if interval not in ['1d', '1wk', '1mo']:
                    data_age_minutes = (datetime.now(databento_df.index[-1].tzinfo or None) -
                                       databento_df.index[-1]).total_seconds() / 60
                    if databento_df.index[-1].tzinfo is None:
                        # Naive timestamp - assume UTC
                        import pytz
                        data_age_minutes = (datetime.now(pytz.UTC).replace(tzinfo=None) -
                                           databento_df.index[-1]).total_seconds() / 60

                    if data_age_minutes > MAX_DATA_AGE_MINUTES:
                        print(f"   Databento data is {data_age_minutes:.0f} min old (>{MAX_DATA_AGE_MINUTES}), checking yfinance for fresher data")
                    else:
                        return databento_df
                else:
                    # Daily data - Databento is fine
                    return databento_df
            elif databento_df is not None:
                print(f"   Databento returned only {len(databento_df)} bars, need {MIN_BARS_REQUIRED}+, falling back to yfinance")
        except Exception as e:
            print(f"   Databento fetch failed for {ticker}: {e}")

    # Fallback/supplement with yfinance (has fresher intraday data)
    yf_df = _fetch_stock(ticker, days, interval)

    # If we have both, use whichever is fresher for recent data
    if databento_df is not None and yf_df is not None and len(yf_df) > 0:
        yf_latest = yf_df.index[-1]
        db_latest = databento_df.index[-1]
        # Normalize to compare
        if yf_latest.tzinfo is not None and db_latest.tzinfo is not None:
            if yf_latest > db_latest:
                print(f"   Using yfinance (fresher: {yf_latest} vs {db_latest})")
                return yf_df
            else:
                return databento_df

    return yf_df


def _fetch_futures_databento(ticker: str, days: int, interval: str) -> Optional[pd.DataFrame]:
    """Fetch futures data from Databento."""
    if not HAS_DATABENTO:
        return None

    api_key = os.environ.get('DATABENTO_API_KEY', '')
    if not api_key:
        return None

    try:
        client = db.Historical(api_key)

        # Map yfinance ticker to Databento continuous contract symbol
        # Databento continuous contract formats:
        #   .c.0 = calendar roll (front month) - works for ES, NQ, etc.
        #   .n.0 = open interest roll - works better for COMEX metals (GC, SI)
        # COMEX metals have issues with .c.0 returning very few bars
        base_symbol = ticker.replace('=F', '')  # ES=F -> ES

        # Use .n.0 (open interest roll) for COMEX metals, .c.0 for others
        COMEX_METALS = ['GC', 'SI', 'HG', 'PL', 'PA']  # Gold, Silver, Copper, Platinum, Palladium
        if base_symbol in COMEX_METALS:
            symbol = f"{base_symbol}.n.0"  # GC -> GC.n.0 (open interest roll)
        else:
            symbol = f"{base_symbol}.c.0"  # ES -> ES.c.0 (calendar roll)

        # Determine schema based on interval
        if interval == '1d':
            schema = 'ohlcv-1d'
        elif interval == '1h':
            schema = 'ohlcv-1h'
        elif interval in ['15m', '15min']:
            schema = 'ohlcv-1m'  # Will resample
        else:
            schema = 'ohlcv-1d'

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        data = client.timeseries.get_range(
            dataset='GLBX.MDP3',
            symbols=[symbol],
            stype_in='continuous',  # Required for continuous contracts
            schema=schema,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )

        df = data.to_df()
        if df.empty:
            return None

        # Rename columns to match expected format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        # Resample if needed
        if interval in ['15m', '15min'] and schema == 'ohlcv-1m':
            df = df.resample('15T').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

        return df

    except Exception as e:
        print(f"   Databento error: {e}")
        return None


def fetch_realtime_price(ticker: str) -> Optional[float]:
    """
    Fetch current real-time price for a ticker.

    Args:
        ticker: Trading symbol

    Returns:
        Current price as float, or None if unavailable
    """
    import io
    import sys

    # Check if market is likely closed (weekends for futures)
    from datetime import datetime as dt_check
    now_check = dt_check.now()
    weekday = now_check.weekday()  # 0=Monday, 5=Saturday, 6=Sunday
    is_weekend = (weekday == 5) or (weekday == 6 and now_check.hour < 17)

    # For futures on weekend, skip realtime fetch (market closed)
    if is_futures_ticker(ticker) and is_weekend:
        return None

    # For futures, try Databento first
    if is_futures_ticker(ticker) and HAS_DATABENTO:
        try:
            price = _fetch_realtime_databento(ticker)
            if price is not None:
                return price
        except Exception:
            pass

    # Fallback to yfinance (suppress "possibly delisted" warnings)
    if HAS_YFINANCE:
        try:
            # Temporarily suppress yfinance warnings to stderr
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                yf_ticker = yf.Ticker(ticker)
                data = yf_ticker.history(period='1d', interval='1m')
            finally:
                sys.stderr = old_stderr

            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception:
            pass

    return None


def _fetch_realtime_databento(ticker: str) -> Optional[float]:
    """Fetch real-time price from Databento."""
    if not HAS_DATABENTO:
        return None

    api_key = os.environ.get('DATABENTO_API_KEY', '')
    if not api_key:
        return None

    try:
        client = db.Historical(api_key)

        # Get last 5 minutes of data
        base_symbol = ticker.replace('=F', '')

        # Use .n.0 (open interest roll) for COMEX metals, .c.0 for others
        COMEX_METALS = ['GC', 'SI', 'HG', 'PL', 'PA']
        if base_symbol in COMEX_METALS:
            symbol = f"{base_symbol}.n.0"
        else:
            symbol = f"{base_symbol}.c.0"

        end = datetime.now()
        start = end - timedelta(minutes=5)

        data = client.timeseries.get_range(
            dataset='GLBX.MDP3',
            symbols=[symbol],
            stype_in='continuous',  # Required for continuous contracts
            schema='ohlcv-1m',
            start=start.strftime('%Y-%m-%dT%H:%M:%S'),
            end=end.strftime('%Y-%m-%dT%H:%M:%S')
        )

        df = data.to_df()
        if not df.empty:
            return float(df['close'].iloc[-1])

    except Exception:
        pass

    return None


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame to standard format.

    Ensures:
    - Column names are capitalized (Open, High, Low, Close, Volume)
    - Index is datetime
    - No duplicate rows
    - Sorted by date ascending
    """
    # Standardize column names
    col_map = {
        'open': 'Open', 'Open': 'Open',
        'high': 'High', 'High': 'High',
        'low': 'Low', 'Low': 'Low',
        'close': 'Close', 'Close': 'Close',
        'volume': 'Volume', 'Volume': 'Volume',
        'adj close': 'Adj Close', 'Adj Close': 'Adj Close'
    }

    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure required columns exist
    required = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required):
        return df

    # Handle index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        elif 'ts_event' in df.columns:
            df['ts_event'] = pd.to_datetime(df['ts_event'])
            df = df.set_index('ts_event')

    # Remove duplicates and sort
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    return df


def get_latest_bar(df: pd.DataFrame) -> Tuple[pd.Series, datetime]:
    """
    Get the most recent bar from a DataFrame.

    Returns:
        Tuple of (bar_data: Series, bar_time: datetime)
    """
    if df is None or df.empty:
        return None, None

    return df.iloc[-1], df.index[-1]


def is_data_fresh(df: pd.DataFrame, max_age_minutes: int = 30) -> bool:
    """
    Check if the latest data is recent enough.

    Args:
        df: Price DataFrame
        max_age_minutes: Maximum age in minutes

    Returns:
        True if latest bar is within max_age_minutes
    """
    if df is None or df.empty:
        return False

    latest_time = df.index[-1]
    if not isinstance(latest_time, pd.Timestamp):
        latest_time = pd.to_datetime(latest_time)

    # Make timezone-naive for comparison
    if latest_time.tzinfo is not None:
        latest_time = latest_time.tz_localize(None)

    age = datetime.now() - latest_time
    return age.total_seconds() < (max_age_minutes * 60)
