"""
Market Data Database Manager

SQLite-based caching layer for market data used in price prediction.
- Historical data: Read from database (fast, no rate limits)
- Today's data: Always fetch fresh from yfinance (live trading)
- Auto-sync: Updates database with missing historical data

Tables:
- vix: VIX volatility index
- vvix: VVIX (volatility of VIX)
- sectors: Sector ETF prices (XLK, XLF, XLE, etc.)
- treasury: 10Y Treasury yield
- credit: HYG and LQD for credit spreads
- dollar: Dollar index (DXY)
- qqq: QQQ for SPY correlation
- spy: SPY price data (main ticker)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("Warning: yfinance not available")


# Database configuration
DB_PATH = "market_data.db"

# Symbols we track
SYMBOLS = {
    'vix': '^VIX',
    'vvix': '^VVIX',
    'qqq': 'QQQ',
    'spy': 'SPY',
    'treasury_10y': '^TNX',
    'hyg': 'HYG',
    'lqd': 'LQD',
    'dollar': 'DX-Y.NYB',
}

# Sector ETFs
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services',
}


class MarketDataDB:
    """
    SQLite database manager for market data caching.

    Usage:
        db = MarketDataDB()
        vix_data = db.get_vix(start_date='2024-01-01', end_date='2025-01-10')
        sectors = db.get_sectors(start_date='2024-01-01', end_date='2025-01-10')
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection (creates if needed)."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self.conn

    def _init_database(self):
        """Initialize database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Main price data table (for VIX, VVIX, QQQ, SPY, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Sector ETF data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sector_data (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL,
                sma_50 REAL,
                above_sma BOOLEAN,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Metadata table to track last sync
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_metadata (
                symbol TEXT PRIMARY KEY,
                last_sync_date TEXT,
                last_sync_time TEXT
            )
        """)

        # IV (Implied Volatility) data table - for historical IV caching
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS iv_data (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                iv_weighted REAL,
                iv_call REAL,
                iv_put REAL,
                iv_skew REAL,
                atm_iv REAL,
                iv_rank REAL,
                iv_percentile REAL,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Options snapshot data (daily zones from SPY options chain)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS options_daily_zones (
                date TEXT PRIMARY KEY,
                spy_price REAL,
                max_pain REAL,
                max_pain_zone REAL,
                gamma_zone REAL,
                wall_zone REAL,
                combined_zone REAL,
                call_wall_strike REAL,
                call_wall_oi INTEGER,
                put_wall_strike REAL,
                put_wall_oi INTEGER,
                net_gamma REAL,
                pcr_volume REAL,
                pcr_oi REAL,
                atm_iv REAL,
                iv_skew REAL,
                raw_json TEXT,
                created_at TEXT
            )
        """)

        # Per-strike OI snapshots (for detailed zone reconstruction)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS options_strike_oi (
                date TEXT,
                strike REAL,
                call_oi INTEGER DEFAULT 0,
                put_oi INTEGER DEFAULT 0,
                call_gamma REAL DEFAULT 0,
                put_gamma REAL DEFAULT 0,
                PRIMARY KEY (date, strike)
            )
        """)

        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_date ON price_data(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_symbol ON price_data(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sector_date ON sector_data(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_options_zones_date ON options_daily_zones(date)")

        conn.commit()
        # Only print init message in main process (not worker processes)
        if not hasattr(self, '_suppress_init_message') or not self._suppress_init_message:
            print(f"[MarketDataDB] Database initialized: {self.db_path}")

    def _is_today(self, date_str: str) -> bool:
        """Check if a date string is today."""
        today = datetime.now().strftime('%Y-%m-%d')
        return date_str == today

    def _is_market_open(self) -> bool:
        """Check if US market is currently open (rough estimate)."""
        now = datetime.now()
        # Market hours: 9:30 AM - 4:00 PM ET (approximate)
        # This is a rough check - for production, use a proper market calendar
        hour = now.hour
        weekday = now.weekday()

        # Weekend
        if weekday >= 5:
            return False

        # Before 9:30 AM or after 4:00 PM (rough ET estimate)
        if hour < 9 or hour >= 16:
            return False

        return True

    def _is_trading_day(self, date_str: str = None) -> bool:
        """Check if a date is a trading day (not weekend)."""
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')

        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            # Weekend check
            if dt.weekday() >= 5:
                return False
            return True
        except:
            return False

    def _get_last_trading_day(self) -> str:
        """Get the most recent trading day (for fetching fresh data)."""
        today = datetime.now()
        # If today is weekend, go back to Friday
        if today.weekday() == 5:  # Saturday
            today = today - timedelta(days=1)
        elif today.weekday() == 6:  # Sunday
            today = today - timedelta(days=2)
        return today.strftime('%Y-%m-%d')

    def _fetch_from_yfinance(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance."""
        if not YF_AVAILABLE:
            print(f"[MarketDataDB] yfinance not available, cannot fetch {symbol}")
            return None

        try:
            # Add buffer to end date to ensure we get today if needed
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
            end_date_fetch = end_dt.strftime('%Y-%m-%d')

            df = yf.download(
                symbol,
                start=start_date,
                end=end_date_fetch,
                progress=False,
                auto_adjust=True
            )

            if df.empty:
                return None

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            # Ensure date is string format
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            return df

        except Exception as e:
            print(f"[MarketDataDB] Error fetching {symbol}: {e}")
            return None

    def _get_cached_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get data from cache for a symbol."""
        conn = self._get_connection()

        query = """
            SELECT date, open, high, low, close, volume
            FROM price_data
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date
        """

        df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
        return df

    def _save_to_cache(self, symbol: str, df: pd.DataFrame):
        """Save data to cache."""
        if df is None or df.empty:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        for _, row in df.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO price_data (symbol, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                row.get('date', ''),
                row.get('open', None),
                row.get('high', None),
                row.get('low', None),
                row.get('close', None),
                row.get('volume', None)
            ))

        # Update sync metadata
        cursor.execute("""
            INSERT OR REPLACE INTO sync_metadata (symbol, last_sync_date, last_sync_time)
            VALUES (?, ?, ?)
        """, (symbol, datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M:%S')))

        conn.commit()

    def _get_missing_dates(self, symbol: str, start_date: str, end_date: str) -> List[str]:
        """Find dates missing from cache."""
        cached = self._get_cached_data(symbol, start_date, end_date)
        cached_dates = set(cached['date'].tolist()) if not cached.empty else set()

        # Generate all business days in range
        all_dates = pd.bdate_range(start=start_date, end=end_date)
        all_dates_str = set(d.strftime('%Y-%m-%d') for d in all_dates)

        missing = all_dates_str - cached_dates
        return sorted(list(missing))

    def get_data(self, symbol: str, start_date: str, end_date: str,
                 force_refresh_today: bool = True, verbose: bool = False) -> pd.DataFrame:
        """
        Get data for a symbol, using cache for historical and fresh fetch for today.

        Args:
            symbol: The ticker symbol (e.g., '^VIX', 'SPY')
            start_date: Start date string 'YYYY-MM-DD'
            end_date: End date string 'YYYY-MM-DD'
            force_refresh_today: If True, always fetch fresh data for today (if trading day)
            verbose: If True, print fetch messages (default False to reduce noise in parallel)

        Returns:
            DataFrame with date, open, high, low, close, volume columns
        """
        today = datetime.now().strftime('%Y-%m-%d')
        last_trading_day = self._get_last_trading_day()
        is_trading_day_today = self._is_trading_day(today)

        # On weekends, don't try to fetch "today" - use last trading day from cache
        if not is_trading_day_today:
            # Weekend: just get from cache up to the last trading day
            cache_end = end_date if end_date <= last_trading_day else last_trading_day
            need_fresh_today = False
        elif end_date >= today and force_refresh_today:
            # Weekday: need fresh data for today
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            cache_end = yesterday
            need_fresh_today = True
        else:
            cache_end = end_date
            need_fresh_today = False

        # Get historical data from cache
        cached_df = self._get_cached_data(symbol, start_date, cache_end)

        # Check for missing historical dates (only if cache_end is valid)
        if start_date <= cache_end:
            missing_dates = self._get_missing_dates(symbol, start_date, cache_end)

            # Filter out any weekend dates from missing (they don't exist in market data)
            missing_dates = [d for d in missing_dates if self._is_trading_day(d)]

            if missing_dates:
                # Fetch missing historical data
                fetch_start = min(missing_dates)
                fetch_end = max(missing_dates)

                if verbose:
                    print(f"[MarketDataDB] Fetching missing {symbol} data: {fetch_start} to {fetch_end}")

                fresh_df = self._fetch_from_yfinance(symbol, fetch_start, fetch_end)
                if fresh_df is not None and not fresh_df.empty:
                    self._save_to_cache(symbol, fresh_df)
                    # Re-query cache
                    cached_df = self._get_cached_data(symbol, start_date, cache_end)

        # Fetch today's data fresh (only on trading days for live trading)
        if need_fresh_today and is_trading_day_today:
            if verbose:
                print(f"[MarketDataDB] Fetching live {symbol} data for {today}")
            today_df = self._fetch_from_yfinance(symbol, today, today)

            if today_df is not None and not today_df.empty:
                # Save today's data to cache (will be overwritten on next fetch)
                self._save_to_cache(symbol, today_df)

                # Combine with historical
                if not cached_df.empty:
                    result = pd.concat([cached_df, today_df], ignore_index=True)
                    result = result.drop_duplicates(subset=['date'], keep='last')
                    result = result.sort_values('date').reset_index(drop=True)
                else:
                    result = today_df
            else:
                result = cached_df
        else:
            result = cached_df

        return result

    # ========================================================================
    # Convenience methods for specific data types
    # ========================================================================

    def get_vix(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get VIX data."""
        return self.get_data(SYMBOLS['vix'], start_date, end_date)

    def get_vvix(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get VVIX data."""
        return self.get_data(SYMBOLS['vvix'], start_date, end_date)

    def get_qqq(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get QQQ data."""
        return self.get_data(SYMBOLS['qqq'], start_date, end_date)

    def get_spy(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get SPY data."""
        return self.get_data(SYMBOLS['spy'], start_date, end_date)

    def get_treasury_10y(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get 10Y Treasury yield data."""
        return self.get_data(SYMBOLS['treasury_10y'], start_date, end_date)

    def get_credit_spreads(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Get HYG and LQD data for credit spread calculation."""
        return {
            'hyg': self.get_data(SYMBOLS['hyg'], start_date, end_date),
            'lqd': self.get_data(SYMBOLS['lqd'], start_date, end_date)
        }

    def get_dollar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get Dollar index data."""
        return self.get_data(SYMBOLS['dollar'], start_date, end_date)

    def get_sectors(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Get all sector ETF data."""
        sectors = {}
        for symbol in SECTOR_ETFS.keys():
            sectors[symbol] = self.get_data(symbol, start_date, end_date)
        return sectors

    # ========================================================================
    # Bulk operations
    # ========================================================================

    def sync_all(self, start_date: str = None, end_date: str = None, verbose: bool = True):
        """
        Sync all tracked symbols to the database.

        Args:
            start_date: Start date (default: 3 years ago)
            end_date: End date (default: today)
            verbose: Print progress
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if verbose:
            print(f"\n{'='*60}")
            print(f"SYNCING MARKET DATA: {start_date} to {end_date}")
            print(f"{'='*60}")

        # Sync main symbols
        all_symbols = list(SYMBOLS.values()) + list(SECTOR_ETFS.keys())
        total = len(all_symbols)

        for i, symbol in enumerate(all_symbols, 1):
            if verbose:
                print(f"[{i}/{total}] Syncing {symbol}...")

            try:
                self.get_data(symbol, start_date, end_date, force_refresh_today=True, verbose=True)
            except Exception as e:
                print(f"  Error syncing {symbol}: {e}")

        if verbose:
            print(f"\n{'='*60}")
            print(f"SYNC COMPLETE")
            print(f"{'='*60}\n")

    def get_sync_status(self) -> pd.DataFrame:
        """Get sync status for all symbols."""
        conn = self._get_connection()

        query = """
            SELECT
                symbol,
                last_sync_date,
                last_sync_time,
                (SELECT COUNT(*) FROM price_data WHERE price_data.symbol = sync_metadata.symbol) as row_count,
                (SELECT MIN(date) FROM price_data WHERE price_data.symbol = sync_metadata.symbol) as earliest_date,
                (SELECT MAX(date) FROM price_data WHERE price_data.symbol = sync_metadata.symbol) as latest_date
            FROM sync_metadata
            ORDER BY symbol
        """

        return pd.read_sql_query(query, conn)

    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Count rows in each table
        cursor.execute("SELECT COUNT(*) FROM price_data")
        price_rows = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM sector_data")
        sector_rows = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM price_data")
        unique_symbols = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(date), MAX(date) FROM price_data")
        date_range = cursor.fetchone()

        # Get file size
        file_size_mb = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0

        return {
            'price_data_rows': price_rows,
            'sector_data_rows': sector_rows,
            'unique_symbols': unique_symbols,
            'earliest_date': date_range[0],
            'latest_date': date_range[1],
            'file_size_mb': round(file_size_mb, 2)
        }

    def clear_cache(self, symbol: str = None):
        """
        Clear cached data.

        Args:
            symbol: Specific symbol to clear, or None for all
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if symbol:
            cursor.execute("DELETE FROM price_data WHERE symbol = ?", (symbol,))
            cursor.execute("DELETE FROM sync_metadata WHERE symbol = ?", (symbol,))
            print(f"[MarketDataDB] Cleared cache for {symbol}")
        else:
            cursor.execute("DELETE FROM price_data")
            cursor.execute("DELETE FROM sector_data")
            cursor.execute("DELETE FROM sync_metadata")
            print("[MarketDataDB] Cleared all cached data")

        conn.commit()

    # ========================================================================
    # IV (Implied Volatility) Data Methods
    # ========================================================================

    def save_iv_data(self, symbol: str, iv_data: Dict, date: str = None):
        """
        Save IV data for a symbol to the database.

        Args:
            symbol: Ticker symbol (e.g., 'SPY')
            iv_data: Dict with iv_weighted, iv_call, iv_put, iv_skew, atm_iv, etc.
            date: Date string (default: today)
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO iv_data
            (symbol, date, iv_weighted, iv_call, iv_put, iv_skew, atm_iv, iv_rank, iv_percentile)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            date,
            iv_data.get('iv_weighted', 0),
            iv_data.get('iv_call', 0),
            iv_data.get('iv_put', 0),
            iv_data.get('iv_skew', 0),
            iv_data.get('atm_iv', 0),
            iv_data.get('iv_rank', 0),
            iv_data.get('iv_percentile', 0)
        ))

        conn.commit()
        print(f"[MarketDataDB] Saved IV data for {symbol} on {date}: IV={iv_data.get('iv_weighted', 0):.2f}%")

    def get_iv_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get historical IV data for a symbol.

        Args:
            symbol: Ticker symbol
            start_date: Start date (default: 1 year ago)
            end_date: End date (default: today)

        Returns:
            DataFrame with IV data indexed by date
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        conn = self._get_connection()

        query = """
            SELECT date, iv_weighted, iv_call, iv_put, iv_skew, atm_iv, iv_rank, iv_percentile
            FROM iv_data
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date
        """

        df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        return df

    def get_iv_stats(self, symbol: str) -> Dict:
        """Get IV data statistics for a symbol."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as row_count,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                AVG(iv_weighted) as avg_iv,
                MIN(iv_weighted) as min_iv,
                MAX(iv_weighted) as max_iv
            FROM iv_data
            WHERE symbol = ?
        """, (symbol,))

        row = cursor.fetchone()
        return {
            'row_count': row[0],
            'earliest_date': row[1],
            'latest_date': row[2],
            'avg_iv': row[3],
            'min_iv': row[4],
            'max_iv': row[5]
        }

    # ========================================================================
    # Options Snapshot Methods (daily zones from SPY options chain)
    # ========================================================================

    def save_options_snapshot(self, date: str, snapshot: dict):
        """
        Save a daily options snapshot with computed influence zones.

        Args:
            date: Date string (YYYY-MM-DD)
            snapshot: Dict from polygon_manager.get_full_options_analysis() + zones
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        zones = snapshot.get('zones', {})
        raw = snapshot.get('raw', {})
        high_oi = raw.get('high_oi', {})

        top_calls = high_oi.get('high_oi_calls', [])
        top_puts = high_oi.get('high_oi_puts', [])

        cursor.execute("""
            INSERT OR REPLACE INTO options_daily_zones
            (date, spy_price, max_pain, max_pain_zone, gamma_zone, wall_zone, combined_zone,
             call_wall_strike, call_wall_oi, put_wall_strike, put_wall_oi, net_gamma,
             pcr_volume, pcr_oi, atm_iv, iv_skew, raw_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date,
            snapshot.get('current_price'),
            snapshot.get('max_pain'),
            zones.get('max_pain_zone', 0.0),
            zones.get('gamma_zone', 0.0),
            zones.get('wall_zone', 0.0),
            zones.get('combined_zone', 0.0),
            top_calls[0]['strike'] if top_calls else None,
            top_calls[0]['oi'] if top_calls else 0,
            top_puts[0]['strike'] if top_puts else None,
            top_puts[0]['oi'] if top_puts else 0,
            snapshot.get('net_gamma', 0),
            snapshot.get('pcr_volume', 0),
            snapshot.get('pcr_oi', 0),
            snapshot.get('atm_iv', 0),
            snapshot.get('iv_skew', 0),
            None,  # Skip raw_json to save space (can enable later)
            datetime.now().isoformat(),
        ))

        # Save per-strike OI for the top strikes
        for strike_list in [top_calls, top_puts]:
            for item in strike_list:
                strike = item.get('strike')
                oi = item.get('oi', 0)
                if strike:
                    # Determine if call or put based on which list
                    is_call = strike_list is top_calls
                    if is_call:
                        cursor.execute("""
                            INSERT OR REPLACE INTO options_strike_oi
                            (date, strike, call_oi, put_oi)
                            VALUES (?, ?, ?,
                                    COALESCE((SELECT put_oi FROM options_strike_oi
                                              WHERE date=? AND strike=?), 0))
                        """, (date, strike, oi, date, strike))
                    else:
                        cursor.execute("""
                            INSERT OR REPLACE INTO options_strike_oi
                            (date, strike, call_oi, put_oi)
                            VALUES (?, ?,
                                    COALESCE((SELECT call_oi FROM options_strike_oi
                                              WHERE date=? AND strike=?), 0), ?)
                        """, (date, strike, date, strike, oi))

        conn.commit()
        print(f"[MarketDataDB] Saved options snapshot for {date}: "
              f"max_pain={snapshot.get('max_pain')}, "
              f"gamma_zone={zones.get('gamma_zone', 0):.3f}")

    def get_options_snapshot(self, date: str) -> Optional[Dict]:
        """Get options snapshot for a specific date."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM options_daily_zones WHERE date = ?', (date,))
        row = cursor.fetchone()
        if not row:
            return None
        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))

    def get_latest_options_snapshot(self) -> Optional[Dict]:
        """Get most recent options snapshot."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM options_daily_zones ORDER BY date DESC LIMIT 1')
        row = cursor.fetchone()
        if not row:
            return None
        cols = [desc[0] for desc in cursor.description]
        return dict(zip(cols, row))

    def get_options_snapshots_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get options snapshots for a date range (for backtesting)."""
        conn = self._get_connection()
        df = pd.read_sql_query(
            'SELECT * FROM options_daily_zones WHERE date >= ? AND date <= ? ORDER BY date',
            conn,
            params=(start_date, end_date),
        )
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        return df

    def options_snapshot_days(self) -> int:
        """How many days of options snapshots we have."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM options_daily_zones')
        return cursor.fetchone()[0]

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


# ============================================================================
# Process-safe instance for global access
# ============================================================================

import os as _os
import threading as _threading

# Store instance per process ID to handle multiprocessing
_db_instances = {}
_db_lock = _threading.Lock()

def get_market_db(suppress_init_message: bool = False) -> MarketDataDB:
    """
    Get a MarketDataDB instance for the current process.

    Each process gets its own database connection to avoid SQLite
    threading issues and to support multiprocessing with joblib/loky.

    Args:
        suppress_init_message: If True, suppress the "Database initialized" message
    """
    global _db_instances

    # Use process ID as key to ensure each worker process gets its own connection
    pid = _os.getpid()

    with _db_lock:
        if pid not in _db_instances:
            # Create new instance for this process
            db = MarketDataDB.__new__(MarketDataDB)
            db._suppress_init_message = suppress_init_message
            db.db_path = DB_PATH
            db.conn = None
            db._init_database()
            _db_instances[pid] = db

        return _db_instances[pid]


def reset_db_instance():
    """Reset the database instance for the current process. Useful for testing."""
    global _db_instances
    pid = _os.getpid()
    with _db_lock:
        if pid in _db_instances:
            _db_instances[pid].close()
            del _db_instances[pid]


# ============================================================================
# CLI for manual operations
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Market Data Database Manager")
    parser.add_argument('--sync', action='store_true', help='Sync all data')
    parser.add_argument('--status', action='store_true', help='Show sync status')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--clear', type=str, help='Clear cache (symbol or "all")')
    parser.add_argument('--start', type=str, help='Start date for sync (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date for sync (YYYY-MM-DD)')

    args = parser.parse_args()

    db = MarketDataDB()

    if args.sync:
        db.sync_all(start_date=args.start, end_date=args.end)

    if args.status:
        print("\nSync Status:")
        print(db.get_sync_status().to_string())

    if args.stats:
        print("\nDatabase Statistics:")
        stats = db.get_database_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")

    if args.clear:
        if args.clear.lower() == 'all':
            db.clear_cache()
        else:
            db.clear_cache(args.clear)

    if not any([args.sync, args.status, args.stats, args.clear]):
        # Default: show stats
        print("\nDatabase Statistics:")
        stats = db.get_database_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print("\nUse --help for available commands")

    db.close()
