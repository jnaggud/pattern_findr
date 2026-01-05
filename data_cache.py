"""
Data Cache - SQLite database for storing historical price data.

This module provides caching for yfinance data to:
1. Reduce API calls and avoid rate limiting
2. Speed up startup time
3. Provide reliable data access even when API is unavailable
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf

# Default database path
DB_PATH = Path(__file__).parent / "price_data.db"


def get_connection(db_path: str = None) -> sqlite3.Connection:
    """Get a connection to the SQLite database."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    return conn


def init_database(db_path: str = None):
    """Initialize the database with required tables."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Create price data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            interval TEXT DEFAULT '1d',
            updated_at TEXT,
            PRIMARY KEY (ticker, date, interval)
        )
    """)

    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker_date
        ON price_data (ticker, date)
    """)

    # Create metadata table to track last update times
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            ticker TEXT NOT NULL,
            interval TEXT DEFAULT '1d',
            last_update TEXT,
            first_date TEXT,
            last_date TEXT,
            row_count INTEGER,
            PRIMARY KEY (ticker, interval)
        )
    """)

    conn.commit()
    conn.close()
    print(f"📦 Database initialized: {db_path or DB_PATH}")


def save_price_data(df: pd.DataFrame, ticker: str, interval: str = "1d", db_path: str = None):
    """Save price data to the database."""
    if df.empty:
        return

    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Prepare data for insertion
    now = datetime.now().isoformat()

    for idx, row in df.iterrows():
        date_str = idx.strftime('%Y-%m-%d %H:%M:%S') if hasattr(idx, 'strftime') else str(idx)

        cursor.execute("""
            INSERT OR REPLACE INTO price_data
            (ticker, date, open, high, low, close, volume, interval, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker,
            date_str,
            float(row.get('open', row.get('Open', 0))),
            float(row.get('high', row.get('High', 0))),
            float(row.get('low', row.get('Low', 0))),
            float(row.get('close', row.get('Close', 0))),
            float(row.get('volume', row.get('Volume', 0))),
            interval,
            now
        ))

    # Update metadata
    cursor.execute("""
        INSERT OR REPLACE INTO metadata
        (ticker, interval, last_update, first_date, last_date, row_count)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        ticker,
        interval,
        now,
        df.index[0].strftime('%Y-%m-%d') if hasattr(df.index[0], 'strftime') else str(df.index[0])[:10],
        df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])[:10],
        len(df)
    ))

    conn.commit()
    conn.close()


def load_price_data(ticker: str, interval: str = "1d", start_date: str = None,
                    end_date: str = None, db_path: str = None) -> pd.DataFrame:
    """Load price data from the database."""
    conn = get_connection(db_path)

    query = """
        SELECT date, open, high, low, close, volume
        FROM price_data
        WHERE ticker = ? AND interval = ?
    """
    params = [ticker, interval]

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " ORDER BY date"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return pd.DataFrame()

    # Convert to proper format
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.columns = ['open', 'high', 'low', 'close', 'volume']

    return df


def get_cached_date_range(ticker: str, interval: str = "1d", db_path: str = None) -> tuple:
    """Get the date range of cached data for a ticker."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT first_date, last_date, row_count, last_update
        FROM metadata
        WHERE ticker = ? AND interval = ?
    """, (ticker, interval))

    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            'first_date': row[0],
            'last_date': row[1],
            'row_count': row[2],
            'last_update': row[3]
        }
    return None


def fetch_and_cache(ticker: str, days: int = 200, interval: str = "1d",
                    force_refresh: bool = False, db_path: str = None) -> pd.DataFrame:
    """
    Fetch price data, using cache when possible.

    Strategy:
    1. Check if we have cached data for this ticker
    2. If cached and recent enough, use cached data
    3. If cached but stale, fetch only new data and append
    4. If not cached, fetch all and cache

    Args:
        ticker: The ticker symbol (e.g., 'BTC-USD', 'SPY')
        days: Number of days of data needed
        interval: Data interval ('1d', '1h', etc.)
        force_refresh: If True, ignore cache and fetch fresh
        db_path: Optional custom database path

    Returns:
        DataFrame with OHLCV data
    """
    init_database(db_path)  # Ensure DB exists

    today = datetime.now().date()
    start_date = today - timedelta(days=days)

    cached_info = get_cached_date_range(ticker, interval, db_path)

    # Check if we have recent cached data
    if cached_info and not force_refresh:
        cached_last = datetime.strptime(cached_info['last_date'], '%Y-%m-%d').date()
        cache_age_days = (today - cached_last).days

        # If cache is fresh (within 1 day for daily data), use it
        if cache_age_days <= 1 and interval == '1d':
            print(f"   📦 Using cached data for {ticker} (last update: {cached_info['last_date']})")
            df = load_price_data(ticker, interval, start_date.strftime('%Y-%m-%d'), db_path=db_path)
            if len(df) >= days * 0.9:  # Have at least 90% of requested data
                return df

        # If cache is slightly stale, fetch only recent data
        if cache_age_days <= 7:
            print(f"   📦 Updating cache for {ticker} (fetching last {cache_age_days + 5} days)")
            try:
                # Fetch recent data
                new_df = _fetch_from_yfinance(ticker, days=cache_age_days + 5, interval=interval)
                if not new_df.empty:
                    save_price_data(new_df, ticker, interval, db_path)
                    # Return combined cached + new data
                    df = load_price_data(ticker, interval, start_date.strftime('%Y-%m-%d'), db_path=db_path)
                    return df
            except Exception as e:
                print(f"   ⚠️ Failed to update cache: {e}, using existing cache")
                return load_price_data(ticker, interval, start_date.strftime('%Y-%m-%d'), db_path=db_path)

    # No usable cache, fetch full data
    print(f"   🌐 Fetching {days} days of {ticker} data from yfinance...")
    try:
        df = _fetch_from_yfinance(ticker, days=days, interval=interval)
        if not df.empty:
            save_price_data(df, ticker, interval, db_path)
            print(f"   💾 Cached {len(df)} bars for {ticker}")
        return df
    except Exception as e:
        print(f"   ⚠️ Failed to fetch from yfinance: {e}")
        # Try to return whatever cache we have
        if cached_info:
            print(f"   📦 Falling back to cached data")
            return load_price_data(ticker, interval, db_path=db_path)
        return pd.DataFrame()


def _fetch_from_yfinance(ticker: str, days: int = 200, interval: str = "1d") -> pd.DataFrame:
    """Internal function to fetch data from yfinance."""
    import time
    time.sleep(0.5)  # Rate limiting

    print(f"   📊 Fetching {ticker} via yfinance ({days} days, {interval})")

    # Calculate date range
    end_date = datetime.now()

    if interval == "1d":
        start_date = end_date - timedelta(days=days + 10)  # Extra buffer
    elif interval == "1h":
        start_date = end_date - timedelta(days=min(days, 729))  # yfinance limit
    else:
        start_date = end_date - timedelta(days=days)

    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start_date, end=end_date, interval=interval)

        if df.empty:
            print(f"   ⚠️ No data returned for {ticker}")
            return pd.DataFrame()

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]

        # Keep only OHLCV
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols_to_keep if c in df.columns]]

        print(f"   ✓ Loaded {len(df)} bars: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

        return df

    except Exception as e:
        print(f"   ⚠️ yfinance error: {e}")
        return pd.DataFrame()


def list_cached_tickers(db_path: str = None) -> list:
    """List all tickers in the cache with their info."""
    try:
        conn = get_connection(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ticker, interval, first_date, last_date, row_count, last_update
            FROM metadata
            ORDER BY ticker, interval
        """)

        rows = cursor.fetchall()
        conn.close()

        result = []
        for row in rows:
            result.append({
                'ticker': row[0],
                'interval': row[1],
                'first_date': row[2],
                'last_date': row[3],
                'row_count': row[4],
                'last_update': row[5]
            })
        return result
    except:
        return []


def clear_cache(ticker: str = None, db_path: str = None):
    """Clear cache for a specific ticker or all tickers."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    if ticker:
        cursor.execute("DELETE FROM price_data WHERE ticker = ?", (ticker,))
        cursor.execute("DELETE FROM metadata WHERE ticker = ?", (ticker,))
        print(f"🗑️ Cleared cache for {ticker}")
    else:
        cursor.execute("DELETE FROM price_data")
        cursor.execute("DELETE FROM metadata")
        print("🗑️ Cleared all cached data")

    conn.commit()
    conn.close()


# CLI interface for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "list":
            init_database()
            tickers = list_cached_tickers()
            if tickers:
                print("\n📦 Cached Data:")
                for t in tickers:
                    print(f"   {t['ticker']} ({t['interval']}): {t['first_date']} to {t['last_date']} ({t['row_count']} bars)")
            else:
                print("No cached data found")

        elif cmd == "fetch" and len(sys.argv) > 2:
            ticker = sys.argv[2]
            days = int(sys.argv[3]) if len(sys.argv) > 3 else 200
            df = fetch_and_cache(ticker, days=days)
            print(f"\nFetched {len(df)} bars for {ticker}")

        elif cmd == "clear":
            ticker = sys.argv[2] if len(sys.argv) > 2 else None
            clear_cache(ticker)

        else:
            print("Usage:")
            print("  python data_cache.py list              - List cached tickers")
            print("  python data_cache.py fetch TICKER [days] - Fetch and cache data")
            print("  python data_cache.py clear [TICKER]    - Clear cache")
    else:
        # Demo
        print("Data Cache Demo")
        print("="*50)
        init_database()

        # Test with BTC
        df = fetch_and_cache("BTC-USD", days=30)
        print(f"\nBTC-USD: {len(df)} bars")
        if not df.empty:
            print(df.tail())
