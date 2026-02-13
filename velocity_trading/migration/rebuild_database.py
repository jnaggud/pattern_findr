"""
Rebuild a strategy database from fresh data.

This script:
1. Fetches fresh market data
2. Runs backtest with the strategy's signal logic
3. Creates a clean SQLite database with proper schema
4. Calculates and caches stats

Usage:
    python -m velocity_trading.migration.rebuild_database --strategy velocity_ES=F_any_reversal_sl.72
"""

import os
import sys
import sqlite3
import pandas as pd
import yfinance as yf
import pytz
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent to path
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


# Strategy configurations (same as intraday_trader.py)
STRATEGY_CONFIGS = {
    'velocity_GC=F_any_reversal_JD1': {
        'ticker': 'GC=F',
        'interval': '15m',
        'signal_type': 'any_reversal',
        'stop_loss_pct': 1.5,
        'take_profit_pct': 3.0,
        'oversold_threshold': -0.3,
        'overbought_threshold': 0.3,
    },
    'velocity_ES=F_any_reversal_sl.72': {
        'ticker': 'ES=F',
        'interval': '15m',
        'signal_type': 'any_reversal',
        'stop_loss_pct': 0.72,
        'take_profit_pct': 1.5,
        'oversold_threshold': -0.3,
        'overbought_threshold': 0.3,
    },
    'velocity_BTC-USD_15m': {
        'ticker': 'BTC-USD',
        'interval': '15m',
        'signal_type': 'any_reversal',
        'stop_loss_pct': 1.0,
        'take_profit_pct': 2.0,
        'oversold_threshold': -0.3,
        'overbought_threshold': 0.3,
    },
}


def get_market_timezone(ticker: str) -> pytz.timezone:
    """Get the appropriate timezone for a ticker."""
    if ticker in ['ES=F', 'GC=F', 'NQ=F', 'CL=F']:
        return pytz.timezone('America/Chicago')  # CME futures
    elif ticker.endswith('-USD') or ticker in ['BTC', 'ETH']:
        return pytz.UTC  # Crypto
    else:
        return pytz.timezone('America/New_York')  # Stocks


def create_database_schema(db_path: str):
    """Create all tables with proper schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Trades table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            entry_price REAL NOT NULL,
            entry_signal_bar TEXT,
            position_type TEXT DEFAULT 'long',
            exit_date TEXT,
            exit_price REAL,
            exit_signal_bar TEXT,
            exit_reason TEXT,
            pnl_pct REAL,
            pnl_dollars REAL,
            is_missed BOOLEAN DEFAULT FALSE,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # Positions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT UNIQUE NOT NULL,
            ticker TEXT NOT NULL,
            position_type TEXT NOT NULL,
            entry_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            entry_signal_bar TEXT,
            last_signal_time TEXT,
            trade_id INTEGER REFERENCES trades(id),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # Daily alerts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            alert_key TEXT NOT NULL,
            sent_at TEXT DEFAULT (datetime('now')),
            UNIQUE (strategy_name, alert_key)
        )
    """)

    # Strategy stats table (with ALL columns)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strategy_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT UNIQUE NOT NULL,
            num_trades INTEGER DEFAULT 0,
            num_wins INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0.0,
            total_return REAL DEFAULT 0.0,
            profit_factor REAL DEFAULT 0.0,
            avg_win REAL DEFAULT 0.0,
            avg_loss REAL DEFAULT 0.0,
            num_missed INTEGER DEFAULT 0,
            missed_return REAL DEFAULT 0.0,
            last_updated TEXT DEFAULT (datetime('now'))
        )
    """)

    # Schema version
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY
        )
    """)
    cursor.execute("INSERT OR IGNORE INTO schema_version VALUES (1)")

    conn.commit()
    conn.close()


def calculate_signals(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Calculate trading signals using any_reversal logic."""
    from oscillator_predictor_page import create_composite_oscillator

    df = create_composite_oscillator(df)

    osc_col = 'composite_smooth' if 'composite_smooth' in df.columns else 'osc_smooth'
    df['velocity'] = df[osc_col].diff().fillna(0)
    df['acceleration'] = df['velocity'].diff().fillna(0)

    # Velocity crossings
    df['vel_cross_up'] = (df['velocity'] > 0) & (df['velocity'].shift(1) <= 0)
    df['vel_cross_down'] = (df['velocity'] < 0) & (df['velocity'].shift(1) >= 0)

    # Thresholds
    oversold = config.get('oversold_threshold', -0.3)
    overbought = config.get('overbought_threshold', 0.3)
    extreme_mult = config.get('extreme_zone_mult', 1.5)

    # Zone conditions
    in_oversold = df[osc_col] < oversold
    in_overbought = df[osc_col] > overbought
    extreme_oversold = df[osc_col] < (oversold * extreme_mult)
    extreme_overbought = df[osc_col] > (overbought * extreme_mult)

    # Strong momentum
    vel_std = df['velocity'].rolling(10, min_periods=1).std().fillna(df['velocity'].std())
    strong_momentum_up = df['velocity'] > vel_std * 1.5
    strong_momentum_down = df['velocity'] < -vel_std * 1.5

    # Signal type
    signal_type = config.get('signal_type', 'any_reversal')

    if signal_type == 'any_reversal':
        df['buy_signal'] = df['vel_cross_up'] | extreme_oversold | (strong_momentum_up & in_oversold)
        df['sell_signal'] = df['vel_cross_down'] | extreme_overbought | (strong_momentum_down & in_overbought)
    elif signal_type == 'velocity_crossover_or_zone':
        df['buy_signal'] = df['vel_cross_up'] | extreme_oversold
        df['sell_signal'] = df['vel_cross_down'] | extreme_overbought
    else:
        df['buy_signal'] = df['vel_cross_up'] & in_oversold
        df['sell_signal'] = df['vel_cross_down'] & in_overbought

    return df


def run_backtest(df: pd.DataFrame, config: Dict, ticker: str) -> List[Dict]:
    """Run backtest and return list of trades."""
    tz = get_market_timezone(ticker)
    trades = []
    in_position = False

    for idx, row in df.iterrows():
        # Convert to market timezone
        if idx.tzinfo is not None:
            idx_local = idx.astimezone(tz)
        else:
            idx_local = idx.tz_localize('UTC').astimezone(tz)

        date_str = idx_local.strftime('%Y-%m-%d %H:%M:%S%z')
        # Format as -0600 not -06:00
        if len(date_str) > 5 and date_str[-3] == ':':
            date_str = date_str[:-3] + date_str[-2:]

        if not in_position and row.get('buy_signal', False):
            trades.append({
                'entry_date': date_str,
                'entry_price': row['close'],
                'exit_date': None,
                'exit_price': None,
                'pnl_pct': None
            })
            in_position = True
        elif in_position and row.get('sell_signal', False):
            entry_price = trades[-1]['entry_price']
            exit_price = row['close']
            pnl = (exit_price - entry_price) / entry_price * 100
            trades[-1]['exit_date'] = date_str
            trades[-1]['exit_price'] = exit_price
            trades[-1]['pnl_pct'] = pnl
            in_position = False

    return trades


def rebuild_database(strategy_name: str, days: int = 60) -> Dict:
    """Rebuild a strategy database from scratch."""
    if strategy_name not in STRATEGY_CONFIGS:
        return {'success': False, 'error': f'Unknown strategy: {strategy_name}'}

    config = STRATEGY_CONFIGS[strategy_name]
    ticker = config['ticker']
    interval = config['interval']

    print(f"Rebuilding {strategy_name}...")
    print(f"  Ticker: {ticker}, Interval: {interval}")

    # Fetch data
    print(f"  Fetching {days} days of data...")
    df = yf.download(ticker, period=f'{days}d', interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.columns = df.columns.str.lower()
    print(f"  Fetched {len(df)} bars")

    # Calculate signals
    print("  Calculating signals...")
    df = calculate_signals(df, config)

    # Run backtest
    print("  Running backtest...")
    trades = run_backtest(df, config, ticker)
    print(f"  Generated {len(trades)} trades")

    # Create database
    db_dir = os.path.join(PARENT_DIR, 'velocity_trading', 'db')
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, f'{strategy_name}.db')

    # Remove old database
    if os.path.exists(db_path):
        os.remove(db_path)
        print("  Removed old database")

    # Create schema
    create_database_schema(db_path)

    # Insert trades
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for trade in trades:
        cursor.execute("""
            INSERT INTO trades (strategy_name, ticker, entry_date, entry_price,
                               entry_signal_bar, position_type, exit_date, exit_price,
                               exit_reason, pnl_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy_name, ticker, trade['entry_date'], trade['entry_price'],
            trade['entry_date'], 'long', trade['exit_date'], trade['exit_price'],
            'Signal' if trade['exit_date'] else None, trade['pnl_pct']
        ))

    # Handle open position
    if trades and trades[-1]['exit_date'] is None:
        trade_id = cursor.lastrowid
        last_trade = trades[-1]
        cursor.execute("""
            INSERT INTO positions (strategy_name, ticker, position_type, entry_price,
                                  entry_date, entry_signal_bar, trade_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy_name, ticker, 'long', last_trade['entry_price'],
            last_trade['entry_date'], last_trade['entry_date'], trade_id
        ))

    conn.commit()
    conn.close()

    # Recalculate stats using PositionManager
    from velocity_trading.core.position_manager import PositionManager
    pm = PositionManager(strategy_name)
    stats = pm.recalculate_stats()

    # Verify
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trades")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_date IS NULL")
    open_count = cursor.fetchone()[0]
    conn.close()

    print(f"\n  Results:")
    print(f"    Total trades: {total}")
    print(f"    Open positions: {open_count}")
    print(f"    Win rate: {stats['win_rate']:.1f}%")
    print(f"    Total return: {stats['total_return']:.2f}%")
    print(f"    Profit factor: {stats['profit_factor']:.2f}")

    return {
        'success': True,
        'trades': total,
        'open_positions': open_count,
        'stats': stats
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Rebuild strategy database')
    parser.add_argument('--strategy', '-s', required=True, help='Strategy name')
    parser.add_argument('--days', '-d', type=int, default=60, help='Days of data')
    parser.add_argument('--list', action='store_true', help='List strategies')

    args = parser.parse_args()

    if args.list:
        print("Available strategies:")
        for name in STRATEGY_CONFIGS:
            print(f"  {name}")
        sys.exit(0)

    result = rebuild_database(args.strategy, args.days)
    if result['success']:
        print("\n✅ Database rebuilt successfully")
    else:
        print(f"\n❌ Failed: {result['error']}")
        sys.exit(1)
