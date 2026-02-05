"""
Incremental Data Pipeline for Velocity Trading.

Architecture:
1. Historical API (db.Historical) → Initial backfill to SQLite
2. Live API (db.Live) → Stream new bars → Append to SQLite
3. Trader reads from SQLite (fast, complete data)

This ensures:
- Historical depth for backtesting/oscillator calculation
- Real-time freshness for live trading signals
- No duplicate fetches - data is persisted incrementally
"""

import os
import sys
import sqlite3
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
import pandas as pd
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Add parent directory to path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    import databento as db
    HAS_DATABENTO = True
except ImportError:
    HAS_DATABENTO = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# Databento API key
DATABENTO_API_KEY = os.environ.get('DATABENTO_API_KEY', '')

# Database directory
DB_DIR = os.path.join(os.path.dirname(__file__), '..', 'db')
os.makedirs(DB_DIR, exist_ok=True)

# Futures ticker mapping (yfinance -> Databento)
# Note: Use .c.1 for COMEX metals (GC, SI) to match yfinance front month pricing
# The .n.0 (open interest roll) tracks a different contract that doesn't match yfinance
FUTURES_TICKER_MAP = {
    'ES=F': 'ES.n.0',   # S&P 500 E-mini
    'NQ=F': 'NQ.n.0',   # Nasdaq 100 E-mini
    'YM=F': 'YM.n.0',   # Dow E-mini
    'RTY=F': 'RTY.n.0', # Russell 2000 E-mini
    'GC=F': 'GC.c.1',   # Gold (COMEX) - use .c.1 to match yfinance pricing
    'SI=F': 'SI.c.1',   # Silver (COMEX) - use .c.1 to match yfinance pricing
    'CL=F': 'CL.n.0',   # Crude Oil
    'NG=F': 'NG.n.0',   # Natural Gas
    'ZB=F': 'ZB.n.0',   # 30-Year Treasury
    'ZN=F': 'ZN.n.0',   # 10-Year Treasury
    '6E=F': '6E.n.0',   # Euro FX
}


def _retry_with_backoff(func, max_retries=3, base_delay=1.0, max_delay=30.0, description="API call"):
    """
    Retry a function with exponential backoff.

    Args:
        func: Callable to retry (should return result or raise)
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        description: Description for logging

    Returns:
        Result of func(), or None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(kw in error_str for kw in ['rate limit', 'too many', '429', 'throttl'])
            is_connection = any(kw in error_str for kw in ['connection', 'timeout', 'refused', 'reset'])

            if attempt == max_retries - 1:
                logger.warning(f"{description} failed after {max_retries} attempts: {e}")
                return None

            delay = min(base_delay * (2 ** attempt), max_delay)
            if is_rate_limit:
                delay = min(delay * 2, max_delay)  # Extra delay for rate limits
                logger.info(f"{description} rate limited, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            elif is_connection:
                logger.info(f"{description} connection error, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            else:
                logger.info(f"{description} failed: {e}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")

            time.sleep(delay)

    return None


class DataPipeline:
    """
    Manages incremental data fetching and storage.

    Usage:
        pipeline = DataPipeline('GC=F', '15m')

        # Initial backfill (run once)
        pipeline.backfill(days=60)

        # Incremental update (run periodically or on-demand)
        pipeline.update()

        # Get data for trading
        df = pipeline.get_data(lookback_bars=500)
    """

    def __init__(self, ticker: str, interval: str):
        """
        Initialize data pipeline for a ticker/interval combination.

        Args:
            ticker: Trading symbol (e.g., 'GC=F', 'SPY')
            interval: Bar interval ('1m', '5m', '15m', '1h', '1d')
        """
        self.ticker = ticker
        self.interval = interval
        self.db_symbol = FUTURES_TICKER_MAP.get(ticker, ticker)
        self.is_futures = ticker.endswith('=F') or ticker in FUTURES_TICKER_MAP

        # Database path
        safe_ticker = ticker.replace('=', '_').replace('-', '_')
        self.db_path = os.path.join(DB_DIR, f'ohlcv_{safe_ticker}_{interval}.db')

        # Initialize database
        self._init_db()

        # Live streaming state
        self._live_client = None
        self._live_thread = None
        self._live_running = False

        # API call caching to prevent excessive requests
        self._last_api_fetch_time = None
        self._last_api_fetch_result = None
        self._min_fetch_interval_seconds = self._get_min_fetch_interval()

    def _get_min_fetch_interval(self) -> int:
        """Get minimum seconds between API calls based on interval."""
        # For longer intervals, we can wait longer between API calls
        interval_seconds = {
            '1m': 15,    # Check every 15s for 1m bars
            '5m': 30,    # Check every 30s for 5m bars
            '15m': 30,   # Check every 30s for 15m bars
            '30m': 60,   # Check every 60s for 30m bars
            '1h': 120,   # Check every 2min for hourly bars
            '1d': 300,   # Check every 5min for daily bars
        }
        return interval_seconds.get(self.interval, 30)

    def _init_db(self):
        """Initialize SQLite database with OHLCV table."""
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv (
                    timestamp TEXT PRIMARY KEY,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'historical',
                    created_at TEXT DEFAULT (datetime('now'))
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv(timestamp)')
            conn.commit()

    @contextmanager
    def _get_conn(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            yield conn
        finally:
            conn.close()

    def backfill(self, days: int = 60, force: bool = False) -> Tuple[bool, Dict]:
        """
        Backfill historical data from Databento Historical API.

        Args:
            days: Number of days to backfill
            force: If True, clear existing data and refetch

        Returns:
            Tuple of (success, result_dict)
        """
        if force:
            with self._get_conn() as conn:
                conn.execute('DELETE FROM ohlcv')
                conn.commit()
                print(f"   Cleared existing data for {self.ticker}")

        # Check existing data
        existing_range = self._get_data_range()
        if existing_range[0] and not force:
            # Already have data - just update
            print(f"   Data exists from {existing_range[0]} to {existing_range[1]}, running incremental update")
            return self.update()

        print(f"📊 Backfilling {self.ticker} ({days} days, {self.interval})...")

        # Fetch from Databento Historical
        if self.is_futures and HAS_DATABENTO and DATABENTO_API_KEY:
            df = self._fetch_historical_databento(days)
        elif HAS_YFINANCE:
            df = self._fetch_historical_yfinance(days)
        else:
            return False, {'error': 'No data source available'}

        if df is None or df.empty:
            return False, {'error': 'No data returned'}

        # Store in database
        bars_stored = self._store_bars(df, source='historical')

        return True, {
            'bars_stored': bars_stored,
            'start': str(df.index[0]),
            'end': str(df.index[-1])
        }

    def update(self) -> Tuple[bool, Dict]:
        """
        Incrementally update with latest data from Live API.

        Uses yfinance as primary source for equity index futures (ES, NQ, YM)
        since it matches TradingView pricing. Uses Databento for metals (GC, SI)
        where it provides better contract matching.

        Returns:
            Tuple of (success, result_dict)
        """
        # Get latest timestamp in database
        existing_range = self._get_data_range()
        if not existing_range[1]:
            # No existing data - need backfill first
            print(f"   No existing data, running backfill first")
            return self.backfill()

        last_timestamp = pd.to_datetime(existing_range[1])
        now = datetime.now(timezone.utc)

        # Calculate how many minutes of data we need
        if last_timestamp.tzinfo is None:
            last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)

        gap_minutes = (now - last_timestamp).total_seconds() / 60

        if gap_minutes < 1:
            return True, {'message': 'Data is current', 'new_bars': 0}

        # Only print fetch message if we're actually going to hit the API
        cache_age = time.time() - (self._last_api_fetch_time or 0)
        will_use_cache = (self._last_api_fetch_time is not None and
                          self._last_api_fetch_result is not None and
                          cache_age < self._min_fetch_interval_seconds)

        if not will_use_cache:
            print(f"   Fetching {gap_minutes:.0f} minutes of new data...")

        # Get last known price for validation
        last_price = self._get_last_price()

        df = None
        source_used = None

        # Data source priority by ticker:
        # - GC=F: Use Polygon/Databento as PRIMARY (more complete overnight coverage)
        # - ES=F, NQ=F, etc: Use yfinance as PRIMARY (matches TradingView pricing)
        #
        # GC=F specifically has 204 extra overnight bars in Polygon that yfinance misses,
        # which causes oscillator calculations to diverge significantly.
        POLYGON_PRIMARY_FUTURES = ['GC=F']  # Use Polygon for complete data coverage
        YFINANCE_PRIMARY_FUTURES = ['ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'SI=F', 'CL=F']

        prefer_polygon = self.ticker in POLYGON_PRIMARY_FUTURES
        prefer_yfinance = self.ticker in YFINANCE_PRIMARY_FUTURES and not prefer_polygon

        # CME futures: closed Saturday, Sunday until 5 PM CT
        now_local = datetime.now()
        weekday = now_local.weekday()  # 0=Monday, 5=Saturday, 6=Sunday
        hour = now_local.hour
        skip_databento = (weekday == 5) or (weekday == 6 and hour < 17)

        # Helper function to filter and validate data
        def filter_and_validate(df, source_name):
            if df is None or df.empty:
                return None, None

            # CRITICAL FIX: Filter to new bars FIRST, then validate
            # Previously validation happened before filtering, causing the first bar
            # (from days ago) to be compared against current price - always failing!
            compare_ts = last_timestamp
            if df.index.tz is None and compare_ts.tzinfo is not None:
                compare_ts = compare_ts.replace(tzinfo=None)
            elif df.index.tz is not None and compare_ts.tzinfo is None:
                compare_ts = compare_ts.tz_localize(df.index.tz)

            df_filtered = df[df.index > compare_ts]

            if df_filtered.empty:
                return df, source_name  # No new bars but data is valid

            # Validate only the NEW bars (not the full historical fetch)
            if self._is_data_valid(df_filtered, last_price):
                return df_filtered, source_name
            else:
                print(f"   ⚠️ {source_name} data looks suspicious after filtering")
                return None, None

        df = None
        source_used = None

        # For Polygon-primary futures (GC=F), try Databento first for complete overnight coverage
        if prefer_polygon and self.is_futures and HAS_DATABENTO and DATABENTO_API_KEY and not skip_databento:
            lookback = max(120, int(gap_minutes) + 30)
            raw_df = _retry_with_backoff(
                lambda: self._fetch_live_databento(lookback_minutes=lookback),
                max_retries=2, description=f"Databento Live ({self.ticker})"
            )
            df, source_used = filter_and_validate(raw_df, 'polygon/databento')
            if df is not None:
                print(f"   📊 Using Polygon/Databento for {self.ticker} (complete overnight coverage)")

        # For equity index futures, try yfinance first
        if df is None and prefer_yfinance and HAS_YFINANCE:
            raw_df = _retry_with_backoff(
                self._fetch_recent_yfinance,
                max_retries=2, description=f"yfinance ({self.ticker})"
            )
            df, source_used = filter_and_validate(raw_df, 'yfinance')

        # Try Databento Live for other futures or as fallback
        if df is None and self.is_futures and HAS_DATABENTO and DATABENTO_API_KEY and not skip_databento:
            lookback = max(120, int(gap_minutes) + 30)
            raw_df = _retry_with_backoff(
                lambda: self._fetch_live_databento(lookback_minutes=lookback),
                max_retries=2, description=f"Databento Live fallback ({self.ticker})"
            )
            df, source_used = filter_and_validate(raw_df, 'databento')

        # If gap is large (e.g., over a weekend) and Live API didn't work,
        # try Databento Historical API for backfill
        if df is None and self.is_futures and HAS_DATABENTO and DATABENTO_API_KEY and gap_minutes > 720:
            # Gap > 12 hours - use Historical API
            gap_days = max(3, int(gap_minutes / 1440) + 1)  # At least 3 days
            print(f"   📊 Backfilling {gap_days} days via Databento Historical...")
            raw_df = _retry_with_backoff(
                lambda: self._fetch_historical_databento(days=gap_days),
                max_retries=3, description=f"Databento Historical ({self.ticker})"
            )
            if raw_df is not None and not raw_df.empty:
                # Filter to new bars only
                compare_ts = last_timestamp
                if raw_df.index.tz is None and compare_ts.tzinfo is not None:
                    compare_ts = compare_ts.replace(tzinfo=None)
                elif raw_df.index.tz is not None and compare_ts.tzinfo is None:
                    compare_ts = compare_ts.tz_localize(raw_df.index.tz)
                df = raw_df[raw_df.index > compare_ts]
                if not df.empty:
                    source_used = 'databento-historical'
                    print(f"   ✓ Backfilled {len(df)} bars via Databento Historical")
                else:
                    df = None

        # Final fallback to yfinance if nothing worked yet
        if df is None and HAS_YFINANCE:
            raw_df = _retry_with_backoff(
                self._fetch_recent_yfinance,
                max_retries=2, description=f"yfinance fallback ({self.ticker})"
            )
            if raw_df is not None and not raw_df.empty:
                # Filter to new bars
                compare_ts = last_timestamp
                if raw_df.index.tz is None and compare_ts.tzinfo is not None:
                    compare_ts = compare_ts.replace(tzinfo=None)
                elif raw_df.index.tz is not None and compare_ts.tzinfo is None:
                    compare_ts = compare_ts.tz_localize(raw_df.index.tz)
                df = raw_df[raw_df.index > compare_ts]
                if not df.empty:
                    source_used = 'yfinance (unvalidated)'
                else:
                    df = None

        if df is None or df.empty:
            return True, {'message': 'No new bars available', 'new_bars': 0}

        if df.empty:
            return True, {'message': 'No new bars', 'new_bars': 0}

        # CRITICAL: Exclude today's incomplete bar for daily intervals
        # Today's bar is still forming and may have wrong OHLC values
        # that will corrupt the database when stored with INSERT OR REPLACE
        if self.interval in ['1d', '1wk', '1mo']:
            today = pd.Timestamp.now(tz='UTC').normalize()
            if df.index.tz is None:
                today = today.tz_localize(None)
            elif df.index.tz != today.tz:
                today = today.tz_convert(df.index.tz)

            # Filter out today's bar
            df_filtered = df[df.index < today]
            if len(df_filtered) < len(df):
                excluded = len(df) - len(df_filtered)
                print(f"   ⚠️ Excluded {excluded} incomplete daily bar(s) (today's data)")
            df = df_filtered

            if df.empty:
                return True, {'message': 'No completed bars to store', 'new_bars': 0}

        # Store new bars
        bars_stored = self._store_bars(df, source='live')

        return True, {
            'new_bars': bars_stored,
            'latest': str(df.index[-1]),
            'source': source_used
        }

    def _get_last_price(self) -> Optional[float]:
        """Get the last known close price from database."""
        with self._get_conn() as conn:
            cursor = conn.execute('SELECT close FROM ohlcv ORDER BY timestamp DESC LIMIT 1')
            row = cursor.fetchone()
        return row[0] if row else None

    def _is_data_valid(self, df: pd.DataFrame, last_known_price: float, max_gap_pct: float = 2.0) -> bool:
        """
        Check if new data is valid.

        For commodity futures (GC, SI, CL, etc.) and crypto, only reject true placeholder data.
        These markets are volatile and yfinance quotes are often stale/broken.

        For other tickers, apply stricter validation.

        Returns:
            True if data looks valid, False if suspicious
        """
        if df is None or df.empty:
            return False

        # Normalize column names
        close_col = 'close' if 'close' in df.columns else 'Close'
        open_col = 'open' if 'open' in df.columns else 'Open'
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col = 'low' if 'low' in df.columns else 'Low'

        if close_col not in df.columns:
            return True  # Can't validate

        # Tickers that skip strict validation (volatile markets, unreliable yfinance quotes)
        SKIP_STRICT_VALIDATION = [
            # CME Futures (all use Databento, yfinance quotes unreliable)
            'ES=F', 'NQ=F', 'YM=F', 'RTY=F',  # Index futures
            'GC=F', 'SI=F', 'HG=F',            # Metals
            'CL=F', 'NG=F',                    # Energy
            'ZB=F', 'ZN=F',                    # Treasuries
            # Crypto (24/7, volatile)
            'BTC-USD', 'ETH-USD', 'SOL-USD',
        ]

        # For volatile tickers: only check for TRUE placeholder data
        if self.ticker in SKIP_STRICT_VALIDATION:
            # Check: All bars have identical OHLC (placeholder data)
            if all(col in df.columns for col in [open_col, high_col, low_col, close_col]):
                all_same = (
                    (df[open_col] == df[high_col]) &
                    (df[high_col] == df[low_col]) &
                    (df[low_col] == df[close_col])
                )
                if all_same.all() and len(df) > 1:
                    print(f"   All bars have identical OHLC (placeholder data)")
                    return False
            return True  # Accept the data

        # For other tickers: apply stricter validation
        # Check 1: Compare to yfinance quote
        realtime_price = self._get_realtime_quote()
        if realtime_price is not None:
            latest_close = df[close_col].iloc[-1]
            quote_gap_pct = abs(latest_close - realtime_price) / realtime_price * 100
            if quote_gap_pct > 1.0:
                print(f"   ⚠️ Data doesn't match yfinance quote: ${latest_close:.2f} vs ${realtime_price:.2f} ({quote_gap_pct:.2f}% off)")
                return False

        # Check 2: Price gap from last known
        if last_known_price is not None:
            first_close = df[close_col].iloc[0]
            gap_pct = abs(first_close - last_known_price) / last_known_price * 100
            if gap_pct > max_gap_pct:
                print(f"   Price gap too large: {last_known_price:.2f} -> {first_close:.2f} ({gap_pct:.1f}%)")
                return False

        # Check 3: All bars identical close (placeholder)
        if df[close_col].nunique() == 1 and len(df) > 3:
            print(f"   All bars have identical close price (placeholder data)")
            return False

        return True

    def _get_realtime_quote(self) -> Optional[float]:
        """Get real-time quote price for validation (not OHLCV, just current price)."""
        if not HAS_YFINANCE:
            return None

        try:
            import yfinance as yf
            ticker = yf.Ticker(self.ticker)
            info = ticker.fast_info
            if hasattr(info, 'last_price') and info.last_price:
                return float(info.last_price)
        except Exception:
            pass

        return None

    def get_data(self, lookback_bars: int = 500, add_synthetic_current: bool = True) -> pd.DataFrame:
        """
        Get OHLCV data from database.

        Args:
            lookback_bars: Number of most recent bars to return
            add_synthetic_current: If True, add synthetic bar with real-time quote
                                   when OHLCV data is stale (> 30 min old)

        Returns:
            DataFrame with OHLCV data, index is datetime
        """
        # First, detect and fill any internal gaps (e.g., missing weekend bars)
        if self.is_futures and HAS_DATABENTO and DATABENTO_API_KEY:
            self._detect_and_fill_internal_gaps()

        with self._get_conn() as conn:
            query = f'''
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                ORDER BY timestamp DESC
                LIMIT {lookback_bars}
            '''
            df = pd.read_sql_query(query, conn)

        if df.empty:
            return df

        # Convert timestamp and set as index
        # Use format='mixed' with utc=True to handle both timezone-aware and naive timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        # Convert to naive UTC for consistent comparison
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        df = df.set_index('timestamp').sort_index()

        # Remove duplicate timestamps (keep last = most recent data source)
        df = df[~df.index.duplicated(keep='last')]

        # Rename columns to match expected format
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Add synthetic current bar if data is stale
        if add_synthetic_current and len(df) > 0:
            df = self._add_synthetic_bar_if_needed(df)

        return df

    def _detect_and_fill_internal_gaps(self):
        """
        Detect and fill internal gaps in OHLCV data.

        This handles cases where:
        - Historical data ends Friday
        - Live API fetches Monday's data
        - But Sunday evening bars (when CME reopens) are missing

        For CME futures, markets reopen Sunday 5 PM CT (23:00 UTC).
        """
        # Get recent bars to check for gaps
        with self._get_conn() as conn:
            # Get last 500 bars to check for gaps
            df = pd.read_sql_query(
                '''SELECT timestamp FROM ohlcv ORDER BY timestamp DESC LIMIT 500''',
                conn
            )

        if df.empty or len(df) < 2:
            return

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        # Convert to naive UTC for consistent comparison
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        df = df.sort_values('timestamp')
        timestamps = df['timestamp'].tolist()

        # Calculate expected gap between bars
        interval_minutes = self._get_interval_minutes()
        # Allow up to 3x interval for minor gaps (e.g., daily maintenance window)
        # For CME futures, the daily break is 4-5 PM CT = 1 hour
        # Weekend gap is ~60 hours (Fri 5PM to Sun 5PM CT)
        max_normal_gap_minutes = interval_minutes * 4  # 4x interval = about 1 hour for 15m

        # Check for large gaps
        gaps_found = []
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).total_seconds() / 60
            if gap > max_normal_gap_minutes:
                # Check if this is a weekend gap that should have Sunday evening bars
                prev_ts = timestamps[i-1]
                next_ts = timestamps[i]

                # Weekend check: Friday to Monday gap
                prev_weekday = prev_ts.weekday()  # 0=Mon, 4=Fri
                next_weekday = next_ts.weekday()

                # If gap spans Fri->Mon, CME reopens Sunday 5 PM CT
                # That's ~2040 minutes (34 hours) from Friday close to Sunday reopen
                # Plus Sunday evening to Monday = ~2880+ total minutes
                if prev_weekday == 4 and next_weekday == 0:  # Fri -> Mon
                    # This is a weekend gap - should have Sunday evening bars
                    gaps_found.append({
                        'start': prev_ts,
                        'end': next_ts,
                        'gap_minutes': gap,
                        'is_weekend': True
                    })
                elif gap > 720:  # 12+ hour gap that's not explained
                    gaps_found.append({
                        'start': prev_ts,
                        'end': next_ts,
                        'gap_minutes': gap,
                        'is_weekend': False
                    })

        if not gaps_found:
            return

        # Fill the gaps using Databento Historical
        for gap in gaps_found:
            print(f"   📊 Detected {gap['gap_minutes']:.0f} min gap: {gap['start']} -> {gap['end']}")

            # Calculate how many days to fetch
            gap_days = max(3, int(gap['gap_minutes'] / 1440) + 1)

            try:
                raw_df = self._fetch_historical_databento(days=gap_days)
                if raw_df is not None and not raw_df.empty:
                    # Filter to bars within the gap
                    start_ts = gap['start']
                    end_ts = gap['end']

                    # Handle timezone
                    if raw_df.index.tz is None:
                        if start_ts.tzinfo is not None:
                            start_ts = start_ts.replace(tzinfo=None)
                        if end_ts.tzinfo is not None:
                            end_ts = end_ts.replace(tzinfo=None)
                    else:
                        if start_ts.tzinfo is None:
                            start_ts = start_ts.tz_localize(raw_df.index.tz)
                        if end_ts.tzinfo is None:
                            end_ts = end_ts.tz_localize(raw_df.index.tz)

                    gap_bars = raw_df[(raw_df.index > start_ts) & (raw_df.index < end_ts)]

                    if not gap_bars.empty:
                        bars_stored = self._store_bars(gap_bars, source='gap-fill')
                        print(f"   ✓ Filled gap with {bars_stored} bars")
                    else:
                        print(f"   ⚠️ No bars available for gap period")

            except Exception as e:
                print(f"   ⚠️ Failed to fill gap: {e}")

    def _add_synthetic_bar_if_needed(self, df: pd.DataFrame, max_age_minutes: int = 30) -> pd.DataFrame:
        """
        Add a synthetic bar with real-time quote if OHLCV data is stale.

        This handles cases where:
        - Databento returns bad data (rejected by validation)
        - yfinance doesn't have current intraday data yet

        The synthetic bar has O=H=L=C=current_quote, volume=0.
        It's marked with a special timestamp rounded to current bar.

        Args:
            df: Existing OHLCV dataframe
            max_age_minutes: Data older than this gets synthetic bar added

        Returns:
            DataFrame with synthetic bar appended if needed
        """
        if df.empty:
            return df

        # Check data freshness
        latest_time = df.index[-1]
        if latest_time.tzinfo is not None:
            latest_time = latest_time.tz_convert('UTC').tz_localize(None)

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        age_minutes = (now - latest_time).total_seconds() / 60

        if age_minutes <= max_age_minutes:
            return df  # Data is fresh enough

        # Get real-time quote
        realtime_price = self._get_realtime_quote()
        if realtime_price is None:
            return df  # Can't add synthetic without quote

        # Check if real-time price is reasonable vs last known
        last_close = df['Close'].iloc[-1]
        gap_pct = abs(realtime_price - last_close) / last_close * 100
        if gap_pct > 5:  # More than 5% gap - something's wrong
            print(f"   ⚠️ Real-time quote ({realtime_price:.2f}) too far from last close ({last_close:.2f})")
            return df

        # Create synthetic bar timestamp (rounded to interval)
        # Match timezone of existing data
        interval_minutes = self._get_interval_minutes()
        synthetic_time = now.replace(second=0, microsecond=0)
        # Round down to interval
        minutes = synthetic_time.minute
        rounded_minutes = (minutes // interval_minutes) * interval_minutes
        synthetic_time = synthetic_time.replace(minute=rounded_minutes)

        # Match timezone format of existing data
        if df.index.tz is not None:
            # Existing data has timezone - make synthetic match
            synthetic_time = pd.Timestamp(synthetic_time).tz_localize('UTC')
        else:
            # Existing data is naive - keep synthetic naive
            synthetic_time = pd.Timestamp(synthetic_time)

        # Create synthetic bar (doji at current price)
        synthetic_bar = pd.DataFrame({
            'Open': [realtime_price],
            'High': [realtime_price],
            'Low': [realtime_price],
            'Close': [realtime_price],
            'Volume': [0]  # Zero volume indicates synthetic
        }, index=[synthetic_time])

        # Append synthetic bar
        df = pd.concat([df, synthetic_bar])

        print(f"   📍 Added synthetic bar at {synthetic_time.strftime('%H:%M')} with price {realtime_price:.2f}")

        return df

    def _get_interval_minutes(self) -> int:
        """Get interval in minutes."""
        interval_map = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '1d': 1440
        }
        return interval_map.get(self.interval, 15)

    def get_latest_bar(self) -> Optional[Dict]:
        """Get the most recent bar from database."""
        with self._get_conn() as conn:
            cursor = conn.execute('''
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                ORDER BY timestamp DESC
                LIMIT 1
            ''')
            row = cursor.fetchone()

        if not row:
            return None

        return {
            'timestamp': row[0],
            'open': row[1],
            'high': row[2],
            'low': row[3],
            'close': row[4],
            'volume': row[5]
        }

    def _get_data_range(self) -> Tuple[Optional[str], Optional[str]]:
        """Get the date range of existing data."""
        with self._get_conn() as conn:
            cursor = conn.execute('SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv')
            row = cursor.fetchone()
        return row if row else (None, None)

    def _validate_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean OHLCV data to remove bad ticks.

        Data sources (yfinance, Databento Live) occasionally return erroneous data:
        - Bad ticks with impossible OHLC relationships
        - Aggregation errors where Open=High with large downward range
        - Placeholder bars with volume=1 and no price movement
        - Outlier moves that don't match real market behavior

        Filters out:
        1. Candles with zero/negative OHLC values
        2. Candles where low > high (impossible)
        3. Candles where open/close outside high/low range

        Only structural validation is performed. We trust data from yfinance/Polygon
        and don't filter based on market behavior assumptions (% moves, volume, etc.)
        """
        if df.empty:
            return df

        df = df.copy()

        # Normalize column names for checking
        col_map = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if col_lower in ['open', 'high', 'low', 'close', 'volume']:
                col_map[col_lower] = col

        if not all(k in col_map for k in ['open', 'high', 'low', 'close']):
            return df  # Can't validate without OHLC

        open_col = col_map['open']
        high_col = col_map['high']
        low_col = col_map['low']
        close_col = col_map['close']

        initial_len = len(df)

        # Filter 1: Remove rows with zero or negative OHLC
        mask_valid = (
            (df[open_col] > 0) &
            (df[high_col] > 0) &
            (df[low_col] > 0) &
            (df[close_col] > 0)
        )
        df = df[mask_valid]

        # Filter 2: Remove rows where low > high (impossible)
        mask_valid = df[low_col] <= df[high_col]
        df = df[mask_valid]

        # Filter 3: Remove rows where open/close outside high/low range
        mask_valid = (
            (df[open_col] >= df[low_col]) &
            (df[open_col] <= df[high_col]) &
            (df[close_col] >= df[low_col]) &
            (df[close_col] <= df[high_col])
        )
        df = df[mask_valid]

        # NOTE: Removed filters 4-6 (Open=High check, low volume check, % move check)
        # These were opinionated filters that assumed "normal" market behavior and
        # were incorrectly filtering out valid data during volatile conditions.
        # If OHLC structure is valid, trust the data from yfinance/Polygon.

        removed = initial_len - len(df)
        if removed > 0:
            print(f"   ⚠️ Filtered {removed} bad bars (outliers/invalid data)")

        return df

    def _store_bars(self, df: pd.DataFrame, source: str = 'historical') -> int:
        """Store bars in database, handling duplicates."""
        if df.empty:
            return 0

        # Skip validation entirely for GC=F when using Polygon data
        # Polygon data is trusted and validation was causing bar count mismatch
        # with backtest bundle (318 bars filtered that were in original backtest)
        skip_validation = self.ticker == 'GC=F' and source in ['historical', 'live', 'polygon/databento']

        if not skip_validation:
            # Validate data before storing (removes bad ticks)
            df = self._validate_bars(df)
            if df.empty:
                return 0

        # Prepare data for insertion
        records = []
        for idx, row in df.iterrows():
            # CRITICAL: Normalize timestamp to naive UTC for consistent storage
            # This ensures all sources (Databento, yfinance, etc.) use the same format
            if hasattr(idx, 'tzinfo') and idx.tzinfo is not None:
                # Convert to UTC then strip timezone
                idx = idx.tz_convert('UTC').tz_localize(None)
            # Use consistent format: YYYY-MM-DDTHH:MM:SS (no timezone suffix)
            timestamp = idx.strftime('%Y-%m-%dT%H:%M:%S') if hasattr(idx, 'strftime') else str(idx)[:19]
            records.append((
                timestamp,
                float(row.get('open', row.get('Open', 0))),
                float(row.get('high', row.get('High', 0))),
                float(row.get('low', row.get('Low', 0))),
                float(row.get('close', row.get('Close', 0))),
                int(row.get('volume', row.get('Volume', 0))),
                source
            ))

        # Insert with conflict resolution (update if newer)
        with self._get_conn() as conn:
            conn.executemany('''
                INSERT OR REPLACE INTO ohlcv (timestamp, open, high, low, close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', records)
            conn.commit()

        return len(records)

    def _fetch_historical_databento(self, days: int) -> Optional[pd.DataFrame]:
        """Fetch historical data from Databento Historical API."""
        if not HAS_DATABENTO or not DATABENTO_API_KEY:
            return None

        # Map interval to schema
        schema_map = {
            '1m': 'ohlcv-1m',
            '5m': 'ohlcv-1m',
            '15m': 'ohlcv-1m',
            '30m': 'ohlcv-1m',
            '1h': 'ohlcv-1h',
            '1d': 'ohlcv-1d',
        }
        schema = schema_map.get(self.interval, 'ohlcv-1m')

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        try:
            client = db.Historical(DATABENTO_API_KEY)

            try:
                data = client.timeseries.get_range(
                    dataset='GLBX.MDP3',
                    symbols=[self.db_symbol],
                    stype_in='continuous',
                    schema=schema,
                    start=start_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    end=end_date.strftime('%Y-%m-%dT%H:%M:%S'),
                )
            except Exception as e:
                # Handle "data_end_after_available_end" error
                error_str = str(e)
                if 'available up to' in error_str:
                    import re
                    match = re.search(r"available up to '(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})", error_str)
                    if match:
                        available_end = f"{match.group(1)}T{match.group(2)}"
                        print(f"   Databento data available up to {available_end}")
                        data = client.timeseries.get_range(
                            dataset='GLBX.MDP3',
                            symbols=[self.db_symbol],
                            stype_in='continuous',
                            schema=schema,
                            start=start_date.strftime('%Y-%m-%dT%H:%M:%S'),
                            end=available_end,
                        )
                    else:
                        raise
                else:
                    raise

            df = data.to_df()
            if df.empty:
                return None

            # Extract OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()

            # Resample if needed
            if self.interval in ['5m', '15m', '30m']:
                resample_rule = self.interval.replace('m', 'min')
                df = df.resample(resample_rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

            print(f"   ✓ Historical: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
            return df

        except Exception as e:
            print(f"   ⚠️ Databento Historical error: {e}")
            return None

    def _fetch_live_databento(self, lookback_minutes: int = 120) -> Optional[pd.DataFrame]:
        """Fetch recent data from Databento Live API."""
        if not HAS_DATABENTO or not DATABENTO_API_KEY:
            return None

        # Databento Live API only allows start times from today UTC 00:00:00
        # Check if our requested start time is valid
        now_utc = datetime.now(timezone.utc)
        start_time = now_utc - timedelta(minutes=lookback_minutes)
        today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

        if start_time < today_start:
            # Clamp to today's start to avoid "Invalid start time" error
            start_time = today_start
            # If we're early in the day and need more lookback, skip Databento
            if lookback_minutes > (now_utc - today_start).total_seconds() / 60:
                return None  # Not enough data available today, fall back to yfinance

        try:
            client = db.Live(key=DATABENTO_API_KEY)

            client.subscribe(
                dataset='GLBX.MDP3',
                schema='ohlcv-1m',
                stype_in='continuous',
                symbols=[self.db_symbol],
                start=start_time.strftime('%Y-%m-%dT%H:%M:%S'),
            )

            # Collect records
            records = []
            error_occurred = [False]  # Use list to allow mutation in callback

            def callback(record):
                if hasattr(record, 'open') and hasattr(record, 'close'):
                    records.append({
                        'ts': record.ts_event,
                        'open': record.open / 1e9,
                        'high': record.high / 1e9,
                        'low': record.low / 1e9,
                        'close': record.close / 1e9,
                        'volume': record.volume
                    })

            def error_callback(error):
                error_occurred[0] = True

            client.add_callback(callback)
            # Suppress the async exception warning by handling errors
            try:
                client.add_stream_error_callback(error_callback)
            except AttributeError:
                pass  # Older databento versions may not have this

            client.start()

            # Wait for data
            max_wait = 15
            waited = 0
            while waited < max_wait and not error_occurred[0]:
                time.sleep(0.5)
                waited += 0.5
                if len(records) >= lookback_minutes * 0.5:
                    break

            try:
                client.stop()
            except Exception:
                pass  # Ignore errors on stop

            if not records or error_occurred[0]:
                return None

            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['ts'], unit='ns', utc=True)
            df = df.set_index('timestamp').sort_index()
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df[~df.index.duplicated(keep='last')]

            # Resample if needed
            if self.interval in ['5m', '15m', '30m']:
                resample_rule = self.interval.replace('m', 'min')
                df = df.resample(resample_rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

            print(f"   ⚡ Live: {len(df)} bars up to {df.index[-1]}")
            return df

        except Exception as e:
            print(f"   ⚠️ Databento Live error: {e}")
            return None

    def _fetch_historical_yfinance(self, days: int) -> Optional[pd.DataFrame]:
        """Fetch historical data from yfinance."""
        if not HAS_YFINANCE:
            return None

        try:
            # For intraday intervals, use period instead of date range
            # yfinance is strict about 60-day limit for intraday
            intraday_intervals = {'1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h'}

            if self.interval in intraday_intervals:
                # Use period for intraday (more reliable)
                period = '60d' if days >= 60 else f'{days}d'
                df = yf.download(
                    self.ticker,
                    period=period,
                    interval=self.interval,
                    progress=False
                )
            else:
                # Use date range for daily+
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                df = yf.download(
                    self.ticker,
                    start=start_date,
                    end=end_date,
                    interval=self.interval,
                    progress=False
                )

            if df.empty:
                return None

            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Normalize column names
            df.columns = df.columns.str.lower()

            print(f"   ✓ yfinance: {len(df)} bars")
            return df

        except Exception as e:
            print(f"   ⚠️ yfinance error: {e}")
            return None

    def _fetch_recent_yfinance(self) -> Optional[pd.DataFrame]:
        """
        Fetch recent data from yfinance for incremental updates.

        Period selection by interval:
        - Daily (1d): Use '7d' to ensure we capture gaps over weekends/holidays
        - Intraday: Use '2d' to capture bars across midnight boundary

        Includes caching to prevent excessive API calls (max once per 30s).
        """
        if not HAS_YFINANCE:
            return None

        # Check cache - don't call API more than once per min_fetch_interval
        now = time.time()
        min_interval = self._min_fetch_interval_seconds
        if (self._last_api_fetch_time is not None and
            self._last_api_fetch_result is not None and
            (now - self._last_api_fetch_time) < min_interval):
            # Return cached result (silent - don't spam console)
            return self._last_api_fetch_result.copy() if self._last_api_fetch_result is not None else None

        try:
            # Period selection:
            # - Daily bars: Use '7d' to ensure we capture gaps over weekends/holidays
            #   (2d was causing missed bars when updates were skipped)
            # - Intraday: Use '2d' to capture bars across midnight boundary
            if self.interval in ['1d', '1wk', '1mo']:
                fetch_period = '7d'
            else:
                fetch_period = '2d'

            df = yf.download(
                self.ticker,
                period=fetch_period,
                interval=self.interval,
                progress=False
            )

            if df.empty:
                return None

            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Normalize column names
            df.columns = df.columns.str.lower()

            print(f"   ✓ yfinance: {len(df)} bars ({self.interval})")

            # Cache the result
            self._last_api_fetch_time = now
            self._last_api_fetch_result = df.copy()

            return df

        except Exception as e:
            print(f"   ⚠️ yfinance error: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get statistics about stored data."""
        with self._get_conn() as conn:
            cursor = conn.execute('''
                SELECT
                    COUNT(*) as total_bars,
                    MIN(timestamp) as first_bar,
                    MAX(timestamp) as last_bar,
                    SUM(CASE WHEN source = 'historical' THEN 1 ELSE 0 END) as historical_bars,
                    SUM(CASE WHEN source = 'live' THEN 1 ELSE 0 END) as live_bars
                FROM ohlcv
            ''')
            row = cursor.fetchone()

        return {
            'total_bars': row[0],
            'first_bar': row[1],
            'last_bar': row[2],
            'historical_bars': row[3],
            'live_bars': row[4]
        }

    def clean_outliers(self, max_move_pct: float = 3.0) -> int:
        """
        Remove outlier bars from existing database.

        Identifies and removes bars with abnormal price moves that indicate bad data.

        Args:
            max_move_pct: Maximum allowed percentage move from previous close

        Returns:
            Number of bars removed
        """
        with self._get_conn() as conn:
            # Get all bars ordered by time
            df = pd.read_sql_query(
                'SELECT timestamp, open, high, low, close FROM ohlcv ORDER BY timestamp',
                conn
            )

            if df.empty:
                return 0

            # Find outliers
            prev_close = df['close'].shift(1)
            pct_change_high = abs(df['high'] - prev_close) / prev_close * 100
            pct_change_low = abs(df['low'] - prev_close) / prev_close * 100

            # Identify bad bars (excluding first bar)
            bad_mask = (
                (pct_change_high > max_move_pct) |
                (pct_change_low > max_move_pct)
            ) & ~prev_close.isna()

            bad_timestamps = df[bad_mask]['timestamp'].tolist()

            if bad_timestamps:
                # Delete bad bars
                placeholders = ','.join('?' * len(bad_timestamps))
                conn.execute(
                    f'DELETE FROM ohlcv WHERE timestamp IN ({placeholders})',
                    bad_timestamps
                )
                conn.commit()

                print(f"   🧹 Removed {len(bad_timestamps)} outlier bars:")
                for ts in bad_timestamps[:5]:  # Show first 5
                    print(f"      - {ts}")
                if len(bad_timestamps) > 5:
                    print(f"      ... and {len(bad_timestamps) - 5} more")

                return len(bad_timestamps)

            return 0


def get_pipeline(ticker: str, interval: str) -> DataPipeline:
    """Factory function to get or create a data pipeline."""
    return DataPipeline(ticker, interval)


# Convenience function for quick data fetch
def fetch_data(ticker: str, interval: str = '15m', lookback_bars: int = 500,
               auto_update: bool = True) -> pd.DataFrame:
    """
    Fetch data using the pipeline, with automatic backfill/update.

    Args:
        ticker: Trading symbol
        interval: Bar interval
        lookback_bars: Number of bars to return
        auto_update: Whether to automatically update to latest

    Returns:
        DataFrame with OHLCV data
    """
    pipeline = DataPipeline(ticker, interval)

    # Check if we have enough data
    stats = pipeline.get_stats()
    if stats['total_bars'] < lookback_bars:
        # Need backfill
        days = max(60, lookback_bars // (24 * 4))  # Estimate days needed for 15m bars
        pipeline.backfill(days=days)
    elif auto_update:
        # Just update
        pipeline.update()

    return pipeline.get_data(lookback_bars)


if __name__ == '__main__':
    # Test the pipeline
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else 'GC=F'
    interval = sys.argv[2] if len(sys.argv) > 2 else '15m'

    print(f"\n{'='*60}")
    print(f"  Data Pipeline Test: {ticker} ({interval})")
    print(f"{'='*60}\n")

    pipeline = DataPipeline(ticker, interval)

    # Check existing data
    stats = pipeline.get_stats()
    print(f"Existing data: {stats['total_bars']} bars")

    if stats['total_bars'] == 0:
        print("\nRunning backfill...")
        success, result = pipeline.backfill(days=30)
        print(f"Backfill: {result}")
    else:
        print("\nRunning update...")
        success, result = pipeline.update()
        print(f"Update: {result}")

    # Get data
    df = pipeline.get_data(lookback_bars=100)
    print(f"\nData retrieved: {len(df)} bars")
    print(f"Range: {df.index[0]} to {df.index[-1]}")
    print(f"\nLast 5 bars:")
    print(df.tail())
