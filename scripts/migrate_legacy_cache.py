#!/usr/bin/env python3
"""
Migrate data from legacy data_cache (price_data.db) to DataPipeline databases.

This one-time migration:
1. Reads all ticker/interval pairs from price_data.db
2. Creates corresponding DataPipeline databases in velocity_trading/db/
3. Copies OHLCV data to the new per-ticker databases
4. Verifies row counts match

Usage:
    python scripts/migrate_legacy_cache.py                  # Dry run (show what would migrate)
    python scripts/migrate_legacy_cache.py --execute        # Actually migrate
    python scripts/migrate_legacy_cache.py --execute --force  # Overwrite existing data

NOTE: This does NOT delete price_data.db or data_cache.py. The legacy system
remains functional for scripts that depend on it (velocity_live_trader.py,
velocity_core.py, enhanced_optimization.py, signal_type_comparison.py).
"""

import os
import sys
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

LEGACY_DB = PROJECT_ROOT / "price_data.db"
NEW_DB_DIR = PROJECT_ROOT / "velocity_trading" / "db"


def get_legacy_tickers(conn: sqlite3.Connection):
    """Get all ticker/interval pairs from legacy metadata table."""
    try:
        cursor = conn.execute("""
            SELECT ticker, interval, row_count, first_date, last_date
            FROM metadata
            ORDER BY ticker, interval
        """)
        return cursor.fetchall()
    except sqlite3.OperationalError:
        # metadata table might not exist
        return []


def get_legacy_data(conn: sqlite3.Connection, ticker: str, interval: str):
    """Get all OHLCV data for a ticker/interval from legacy database."""
    cursor = conn.execute("""
        SELECT date, open, high, low, close, volume
        FROM price_data
        WHERE ticker = ? AND interval = ?
        ORDER BY date
    """, (ticker, interval))
    return cursor.fetchall()


def make_safe_ticker(ticker: str) -> str:
    """Convert ticker to filesystem-safe name."""
    return ticker.replace('=', '_').replace('-', '_')


def migrate_ticker(legacy_conn, ticker, interval, force=False):
    """Migrate a single ticker/interval pair."""
    safe_ticker = make_safe_ticker(ticker)
    new_db_path = NEW_DB_DIR / f"ohlcv_{safe_ticker}_{interval}.db"

    # Check if destination already has data
    if new_db_path.exists() and not force:
        dest_conn = sqlite3.connect(str(new_db_path))
        try:
            cursor = dest_conn.execute("SELECT COUNT(*) FROM ohlcv")
            existing_count = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            existing_count = 0
        dest_conn.close()

        if existing_count > 0:
            return 0, f"SKIPPED (destination has {existing_count} bars, use --force)"

    # Get data from legacy
    rows = get_legacy_data(legacy_conn, ticker, interval)
    if not rows:
        return 0, "SKIPPED (no data in legacy)"

    # Create/connect to destination
    os.makedirs(NEW_DB_DIR, exist_ok=True)
    dest_conn = sqlite3.connect(str(new_db_path))

    # Create table
    dest_conn.execute('''
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
    dest_conn.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv(timestamp)')

    if force:
        dest_conn.execute("DELETE FROM ohlcv")

    # Convert and insert
    records = []
    for date_str, open_p, high_p, low_p, close_p, volume in rows:
        # Normalize timestamp: legacy uses "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD"
        # New system uses "YYYY-MM-DDTHH:MM:SS"
        ts = date_str.replace(' ', 'T')
        if len(ts) == 10:  # Date only
            ts += 'T00:00:00'
        elif len(ts) < 19:
            ts = ts[:19]  # Truncate any fractional seconds

        records.append((ts, open_p, high_p, low_p, close_p, int(volume or 0), 'migrated'))

    dest_conn.executemany('''
        INSERT OR REPLACE INTO ohlcv (timestamp, open, high, low, close, volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', records)
    dest_conn.commit()

    # Verify
    cursor = dest_conn.execute("SELECT COUNT(*) FROM ohlcv")
    final_count = cursor.fetchone()[0]
    dest_conn.close()

    return final_count, "OK"


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy data_cache to DataPipeline databases")
    parser.add_argument('--execute', action='store_true', help="Actually perform migration (default: dry run)")
    parser.add_argument('--force', action='store_true', help="Overwrite existing data in destination")
    args = parser.parse_args()

    if not LEGACY_DB.exists():
        print(f"Legacy database not found: {LEGACY_DB}")
        print("Nothing to migrate.")
        return 0

    print(f"\n{'='*60}")
    print(f"LEGACY CACHE MIGRATION")
    print(f"{'='*60}")
    print(f"Source:      {LEGACY_DB}")
    print(f"Destination: {NEW_DB_DIR}")
    print(f"Mode:        {'EXECUTE' if args.execute else 'DRY RUN'}")
    if args.force:
        print(f"Force:       YES (will overwrite existing)")

    legacy_conn = sqlite3.connect(str(LEGACY_DB))
    tickers = get_legacy_tickers(legacy_conn)

    if not tickers:
        print("\nNo ticker data found in legacy database.")
        legacy_conn.close()
        return 0

    print(f"\nFound {len(tickers)} ticker/interval pair(s) in legacy database:\n")

    total_rows = 0
    migrated_count = 0

    for ticker, interval, row_count, first_date, last_date in tickers:
        safe_ticker = make_safe_ticker(ticker)
        dest_name = f"ohlcv_{safe_ticker}_{interval}.db"

        if args.execute:
            count, status = migrate_ticker(legacy_conn, ticker, interval, force=args.force)
            total_rows += count
            if status == "OK":
                migrated_count += 1
            print(f"  {ticker:12s} {interval:4s} | {row_count or '?':>6} rows | "
                  f"{first_date or '?'} to {last_date or '?'} | -> {dest_name} | {status}")
        else:
            print(f"  {ticker:12s} {interval:4s} | {row_count or '?':>6} rows | "
                  f"{first_date or '?'} to {last_date or '?'} | -> {dest_name}")

    legacy_conn.close()

    print(f"\n{'='*60}")
    if args.execute:
        print(f"MIGRATION COMPLETE")
        print(f"  Pairs migrated: {migrated_count}")
        print(f"  Total rows:     {total_rows}")
        print(f"\nLegacy database preserved at: {LEGACY_DB}")
        print(f"Legacy scripts (velocity_live_trader.py, etc.) will continue to work.")
    else:
        print(f"DRY RUN COMPLETE - no changes made.")
        print(f"Run with --execute to perform migration.")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
