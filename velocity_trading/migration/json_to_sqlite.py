"""
Migration utilities for converting JSON files to SQLite database.

Migrates data from the old velocity_live_trader.py JSON files:
- velocity_trade_state_*.json -> positions table
- velocity_locked_backtest_*.json -> trades table
- velocity_trade_history_*.json -> trades table
- daily_alerts_*.json -> daily_alerts table

IMPORTANT: This is NON-DESTRUCTIVE migration.
- Original JSON files are moved to an archive folder (not deleted)
- SQLite database is created fresh
- Migration can be re-run if needed
"""

import os
import sys
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add parent to path for imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from ..core.database import TradingDatabase
from ..core.position_manager import PositionManager


def find_json_files(base_dir: str) -> Dict[str, List[str]]:
    """
    Find all velocity JSON files in the given directory.

    Returns dict with keys: 'trade_state', 'locked_backtest', 'trade_history', 'daily_alerts'
    """
    files = {
        'trade_state': [],
        'locked_backtest': [],
        'trade_history': [],
        'daily_alerts': []
    }

    base_path = Path(base_dir)

    for f in base_path.glob('velocity_trade_state_*.json'):
        files['trade_state'].append(str(f))

    for f in base_path.glob('velocity_locked_backtest_*.json'):
        files['locked_backtest'].append(str(f))

    for f in base_path.glob('velocity_trade_history_*.json'):
        files['trade_history'].append(str(f))

    for f in base_path.glob('daily_alerts_*.json'):
        files['daily_alerts'].append(str(f))

    return files


def extract_strategy_name(filename: str, prefix: str) -> str:
    """
    Extract strategy name from filename.

    Example: velocity_trade_state_velocity_SPY_5y.json -> velocity_SPY_5y
    """
    basename = os.path.basename(filename)
    # Remove prefix and .json extension
    name = basename.replace(prefix, '').replace('.json', '')
    return name


def load_json_file(filepath: str) -> Optional[Dict]:
    """Load and parse a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"   Error loading {filepath}: {e}")
        return None


def migrate_locked_backtest(
    filepath: str,
    strategy_name: str,
    ticker: str,
    db: TradingDatabase
) -> Tuple[int, int]:
    """
    Migrate a locked_backtest JSON file to SQLite.

    The locked_backtest format contains matched entries and exits.
    Each exit has corresponding entry_price and entry_date.

    Returns (trades_migrated, errors)
    """
    data = load_json_file(filepath)
    if data is None:
        return 0, 1

    trades_migrated = 0
    errors = 0

    # Get exits (completed trades)
    exits = data.get('exits', [])

    for exit_trade in exits:
        try:
            entry_date = exit_trade.get('entry_date')
            entry_price = exit_trade.get('entry_price')
            exit_date = exit_trade.get('date')
            exit_price = exit_trade.get('price')
            pnl_pct = exit_trade.get('pnl', 0)
            exit_reason = exit_trade.get('reason', 'Unknown')

            if not all([entry_date, entry_price, exit_date, exit_price]):
                continue

            # Calculate pnl_dollars (assuming $10k position)
            pnl_dollars = pnl_pct * 100  # $10k * pnl_pct / 100

            # Insert into trades table
            db.insert('trades', {
                'strategy_name': strategy_name,
                'ticker': ticker,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'entry_signal_bar': entry_date,
                'position_type': 'long',
                'exit_date': exit_date,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl_pct': pnl_pct,
                'pnl_dollars': pnl_dollars
            })

            trades_migrated += 1

        except Exception as e:
            print(f"   Error migrating trade: {e}")
            errors += 1

    # Handle current open position if any
    current_position = data.get('current_position')
    entries = data.get('entries', [])

    # If there are more entries than exits, the last entry is open
    if len(entries) > len(exits):
        last_entry = entries[-1]
        try:
            entry_date = last_entry.get('date')
            entry_price = last_entry.get('price')
            position_type = last_entry.get('position', 'long')

            # Insert open trade (no exit info)
            trade_id = db.insert('trades', {
                'strategy_name': strategy_name,
                'ticker': ticker,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'entry_signal_bar': entry_date,
                'position_type': position_type
            })

            # Check if position already exists before inserting
            existing_pos = db.execute_one(
                "SELECT id FROM positions WHERE strategy_name = ?",
                (strategy_name,)
            )
            if not existing_pos:
                db.insert('positions', {
                    'strategy_name': strategy_name,
                    'ticker': ticker,
                    'position_type': position_type,
                    'entry_price': entry_price,
                    'entry_date': entry_date,
                    'entry_signal_bar': entry_date,
                    'trade_id': trade_id
                })

            trades_migrated += 1

        except Exception as e:
            print(f"   Error migrating open position: {e}")
            errors += 1

    # Migrate stats using db.update_stats or insert
    try:
        db.update_stats({
            'num_trades': data.get('num_trades', 0),
            'win_rate': data.get('win_rate', 0),
            'total_return': data.get('total_return', 0),
            'profit_factor': data.get('profit_factor', 0)
        })
    except Exception as e:
        print(f"   Error migrating stats: {e}")

    return trades_migrated, errors


def migrate_trade_history(
    filepath: str,
    db: TradingDatabase
) -> Tuple[int, int]:
    """
    Migrate a trade_history JSON file to SQLite.

    Trade history files contain detailed trade records.
    We check for duplicates to avoid double-inserting.

    Returns (trades_migrated, errors)
    """
    data = load_json_file(filepath)
    if data is None or not isinstance(data, list):
        return 0, 1

    trades_migrated = 0
    errors = 0

    for trade in data:
        try:
            strategy_name = trade.get('strategy_name', '')
            ticker = trade.get('ticker', '')
            entry_date = trade.get('entry_time')
            entry_price = trade.get('entry_price')
            exit_date = trade.get('exit_time')
            exit_price = trade.get('exit_price')
            exit_reason = trade.get('exit_reason', 'Unknown')
            pnl_pct = trade.get('pnl_pct', 0)
            pnl_dollars = trade.get('pnl_dollars', 0)
            entry_signal_bar = trade.get('entry_signal_bar', entry_date)
            exit_signal_bar = trade.get('exit_signal_bar', exit_date)
            position_type = trade.get('type', 'LONG').lower()

            if not all([strategy_name, entry_date, entry_price]):
                continue

            # Check for duplicate (same strategy, entry_date, entry_price)
            existing = db.execute_one("""
                SELECT id FROM trades
                WHERE strategy_name = ? AND entry_date = ? AND entry_price = ?
            """, (strategy_name, entry_date, entry_price))

            if existing:
                continue  # Skip duplicate

            # Insert trade
            db.insert('trades', {
                'strategy_name': strategy_name,
                'ticker': ticker,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'entry_signal_bar': entry_signal_bar,
                'position_type': position_type,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'exit_signal_bar': exit_signal_bar,
                'exit_reason': exit_reason,
                'pnl_pct': pnl_pct,
                'pnl_dollars': pnl_dollars
            })

            trades_migrated += 1

        except Exception as e:
            print(f"   Error migrating trade from history: {e}")
            errors += 1

    return trades_migrated, errors


def migrate_trade_state(
    filepath: str,
    strategy_name: str,
    ticker: str,
    db: TradingDatabase
) -> Tuple[bool, str]:
    """
    Migrate a trade_state JSON file to SQLite positions table.

    Only migrates if there's an open position.

    Returns (success, message)
    """
    data = load_json_file(filepath)
    if data is None:
        return False, "Failed to load file"

    position = data.get('position')
    entry_price = data.get('entry_price')
    entry_time = data.get('entry_time')
    last_signal_time = data.get('last_signal_time')

    if not position or not entry_price:
        return True, "No open position to migrate"

    try:
        # Check if position already exists
        existing_pos = db.execute_one("""
            SELECT id FROM positions WHERE strategy_name = ?
        """, (strategy_name,))

        if existing_pos:
            return True, "Position already exists in database"

        # Check if an open trade already exists with same entry (from locked_backtest migration)
        existing_trade = db.execute_one("""
            SELECT id FROM trades
            WHERE strategy_name = ? AND exit_date IS NULL
            ORDER BY id DESC LIMIT 1
        """, (strategy_name,))

        if existing_trade:
            # Use existing open trade instead of creating duplicate
            trade_id = existing_trade['id']
        else:
            # Insert into trades table (open trade)
            trade_id = db.insert('trades', {
                'strategy_name': strategy_name,
                'ticker': ticker,
                'entry_date': entry_time,
                'entry_price': entry_price,
                'entry_signal_bar': entry_time,
                'position_type': position
            })

        # Insert into positions table
        db.insert('positions', {
            'strategy_name': strategy_name,
            'ticker': ticker,
            'position_type': position,
            'entry_price': entry_price,
            'entry_date': entry_time,
            'entry_signal_bar': entry_time,
            'last_signal_time': last_signal_time,
            'trade_id': trade_id
        })

        return True, f"Migrated open {position} position"

    except Exception as e:
        return False, f"Error: {e}"


def migrate_daily_alerts(
    filepath: str,
    strategy_name: str,
    db: TradingDatabase
) -> Tuple[int, int]:
    """
    Migrate daily_alerts JSON file to SQLite.

    Returns (alerts_migrated, errors)
    """
    data = load_json_file(filepath)
    if data is None or not isinstance(data, dict):
        return 0, 1

    alerts_migrated = 0
    errors = 0

    for alert_key, alert_data in data.items():
        try:
            sent_at = alert_data.get('sent_at', datetime.now().isoformat())

            db.execute("""
                INSERT OR IGNORE INTO daily_alerts (strategy_name, alert_key, sent_at)
                VALUES (?, ?, ?)
            """, (strategy_name, alert_key, sent_at))

            alerts_migrated += 1

        except Exception as e:
            print(f"   Error migrating alert {alert_key}: {e}")
            errors += 1

    return alerts_migrated, errors


def archive_json_files(files: List[str], archive_dir: str) -> int:
    """
    Move JSON files to archive directory.

    Returns number of files archived.
    """
    os.makedirs(archive_dir, exist_ok=True)
    archived = 0

    for filepath in files:
        try:
            filename = os.path.basename(filepath)
            dest = os.path.join(archive_dir, filename)

            # Don't overwrite existing archives
            if os.path.exists(dest):
                # Add timestamp to filename
                base, ext = os.path.splitext(filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                dest = os.path.join(archive_dir, f"{base}_{timestamp}{ext}")

            shutil.copy2(filepath, dest)
            archived += 1
            print(f"   Archived: {filename}")

        except Exception as e:
            print(f"   Error archiving {filepath}: {e}")

    return archived


def infer_ticker_from_strategy(strategy_name: str) -> str:
    """
    Infer ticker symbol from strategy name.

    Examples:
        velocity_SPY_5y -> SPY
        velocity_ES=F_any_reversal_sl.72 -> ES=F
        velocity_BTC-USD_velocity_crossover -> BTC-USD
    """
    # Common tickers to check
    tickers = ['SPY', 'ES=F', 'GC=F', 'NQ=F', 'BTC-USD', 'BTC', 'ETH-USD', 'QQQ', 'IWM']

    for ticker in tickers:
        if ticker in strategy_name or ticker.replace('-', '') in strategy_name:
            return ticker

    # Default fallback
    parts = strategy_name.split('_')
    if len(parts) >= 2:
        return parts[1]

    return 'UNKNOWN'


def migrate_strategy(
    strategy_name: str,
    base_dir: str,
    db_dir: str = None,
    archive: bool = True
) -> Dict:
    """
    Migrate all JSON files for a single strategy to SQLite.

    Args:
        strategy_name: Strategy identifier (e.g., velocity_SPY_5y)
        base_dir: Directory containing JSON files
        db_dir: Directory for SQLite databases (default: base_dir/velocity_trading/db)
        archive: Whether to archive JSON files after migration

    Returns:
        Migration results dict
    """
    if db_dir is None:
        db_dir = os.path.join(base_dir, 'velocity_trading', 'db')

    os.makedirs(db_dir, exist_ok=True)

    results = {
        'strategy_name': strategy_name,
        'trades_migrated': 0,
        'positions_migrated': 0,
        'alerts_migrated': 0,
        'files_archived': 0,
        'errors': 0
    }

    # Initialize database
    db_path = os.path.join(db_dir, f"{strategy_name}.db")
    db = TradingDatabase(strategy_name, db_path=db_path)

    ticker = infer_ticker_from_strategy(strategy_name)
    print(f"\nMigrating {strategy_name} (ticker: {ticker})")

    files_to_archive = []

    # Migrate locked_backtest (primary source of trade history)
    locked_file = os.path.join(base_dir, f"velocity_locked_backtest_{strategy_name}.json")
    if os.path.exists(locked_file):
        print(f"   Migrating locked backtest...")
        with db.transaction():
            trades, errors = migrate_locked_backtest(locked_file, strategy_name, ticker, db)
        results['trades_migrated'] += trades
        results['errors'] += errors
        files_to_archive.append(locked_file)

    # Migrate trade_history (may have additional records)
    history_file = os.path.join(base_dir, f"velocity_trade_history_{strategy_name}.json")
    if os.path.exists(history_file):
        print(f"   Migrating trade history...")
        with db.transaction():
            trades, errors = migrate_trade_history(history_file, db)
        results['trades_migrated'] += trades
        results['errors'] += errors
        files_to_archive.append(history_file)

    # Migrate trade_state (current position)
    state_file = os.path.join(base_dir, f"velocity_trade_state_{strategy_name}.json")
    if os.path.exists(state_file):
        print(f"   Migrating trade state...")
        with db.transaction():
            success, msg = migrate_trade_state(state_file, strategy_name, ticker, db)
        if success and 'position' in msg.lower():
            results['positions_migrated'] += 1
        elif not success:
            results['errors'] += 1
        print(f"      {msg}")
        files_to_archive.append(state_file)

    # Migrate daily_alerts
    alerts_file = os.path.join(base_dir, f"daily_alerts_{strategy_name}.json")
    if os.path.exists(alerts_file):
        print(f"   Migrating daily alerts...")
        with db.transaction():
            alerts, errors = migrate_daily_alerts(alerts_file, strategy_name, db)
        results['alerts_migrated'] += alerts
        results['errors'] += errors
        files_to_archive.append(alerts_file)

    # Archive files
    if archive and files_to_archive:
        archive_dir = os.path.join(base_dir, 'velocity_trading', 'json_archive')
        results['files_archived'] = archive_json_files(files_to_archive, archive_dir)

    print(f"   Results: {results['trades_migrated']} trades, "
          f"{results['positions_migrated']} positions, "
          f"{results['alerts_migrated']} alerts, "
          f"{results['errors']} errors")

    return results


def migrate_all(
    base_dir: str = None,
    db_dir: str = None,
    archive: bool = True,
    dry_run: bool = False
) -> Dict:
    """
    Migrate all velocity JSON files to SQLite.

    Args:
        base_dir: Directory containing JSON files (default: parent of velocity_trading)
        db_dir: Directory for SQLite databases
        archive: Whether to archive JSON files after migration
        dry_run: If True, only report what would be migrated

    Returns:
        Migration summary dict
    """
    if base_dir is None:
        base_dir = PARENT_DIR

    print(f"\n{'='*60}")
    print("  VELOCITY TRADING - JSON TO SQLITE MIGRATION")
    print(f"{'='*60}")
    print(f"Source: {base_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE MIGRATION'}")
    print(f"Archive: {'Yes' if archive else 'No'}")

    # Find all JSON files
    files = find_json_files(base_dir)

    print(f"\nFound files:")
    print(f"   Trade state: {len(files['trade_state'])}")
    print(f"   Locked backtest: {len(files['locked_backtest'])}")
    print(f"   Trade history: {len(files['trade_history'])}")
    print(f"   Daily alerts: {len(files['daily_alerts'])}")

    if dry_run:
        print("\nDry run - no changes made")
        return {'dry_run': True, 'files': files}

    # Extract unique strategy names from locked_backtest files (primary source)
    strategies = set()
    for f in files['locked_backtest']:
        name = extract_strategy_name(f, 'velocity_locked_backtest_')
        strategies.add(name)

    # Also check trade_state files for strategies without locked_backtest
    for f in files['trade_state']:
        name = extract_strategy_name(f, 'velocity_trade_state_')
        strategies.add(name)

    print(f"\nFound {len(strategies)} strategies to migrate:")
    for s in sorted(strategies):
        print(f"   - {s}")

    # Migrate each strategy
    total_results = {
        'strategies': len(strategies),
        'trades_migrated': 0,
        'positions_migrated': 0,
        'alerts_migrated': 0,
        'files_archived': 0,
        'errors': 0
    }

    for strategy_name in sorted(strategies):
        results = migrate_strategy(strategy_name, base_dir, db_dir, archive)
        total_results['trades_migrated'] += results['trades_migrated']
        total_results['positions_migrated'] += results['positions_migrated']
        total_results['alerts_migrated'] += results['alerts_migrated']
        total_results['files_archived'] += results['files_archived']
        total_results['errors'] += results['errors']

    print(f"\n{'='*60}")
    print("  MIGRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Strategies migrated: {total_results['strategies']}")
    print(f"Total trades: {total_results['trades_migrated']}")
    print(f"Open positions: {total_results['positions_migrated']}")
    print(f"Alerts: {total_results['alerts_migrated']}")
    print(f"Files archived: {total_results['files_archived']}")
    print(f"Errors: {total_results['errors']}")

    return total_results


def verify_migration(strategy_name: str, base_dir: str = None) -> Dict:
    """
    Verify that migration was successful by comparing JSON and SQLite data.

    Returns verification results.
    """
    if base_dir is None:
        base_dir = PARENT_DIR

    db_dir = os.path.join(base_dir, 'velocity_trading', 'db')
    db_path = os.path.join(db_dir, f"{strategy_name}.db")

    if not os.path.exists(db_path):
        return {'error': 'Database not found', 'verified': False}

    db = TradingDatabase(strategy_name, db_path=db_path)

    # Get SQLite counts
    trades_result = db.execute_one("SELECT COUNT(*) as cnt FROM trades WHERE strategy_name = ?",
                                   (strategy_name,))
    trades_count = trades_result['cnt'] if trades_result else 0

    positions_result = db.execute_one("SELECT COUNT(*) as cnt FROM positions WHERE strategy_name = ?",
                                      (strategy_name,))
    positions_count = positions_result['cnt'] if positions_result else 0

    # Get JSON counts
    locked_file = os.path.join(base_dir, f"velocity_locked_backtest_{strategy_name}.json")
    json_exits = 0
    json_open = False

    if os.path.exists(locked_file):
        data = load_json_file(locked_file)
        if data:
            json_exits = len(data.get('exits', []))
            entries = len(data.get('entries', []))
            json_open = entries > json_exits

    return {
        'strategy_name': strategy_name,
        'sqlite_trades': trades_count,
        'sqlite_positions': positions_count,
        'json_exits': json_exits,
        'json_open_position': json_open,
        'verified': trades_count >= json_exits
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Migrate velocity JSON files to SQLite')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be migrated')
    parser.add_argument('--no-archive', action='store_true', help='Do not archive JSON files')
    parser.add_argument('--strategy', '-s', help='Migrate single strategy')
    parser.add_argument('--verify', '-v', help='Verify migration for strategy')
    parser.add_argument('--base-dir', help='Base directory containing JSON files')

    args = parser.parse_args()

    if args.verify:
        result = verify_migration(args.verify, args.base_dir)
        print(f"\nVerification for {args.verify}:")
        for k, v in result.items():
            print(f"   {k}: {v}")

    elif args.strategy:
        migrate_strategy(
            args.strategy,
            args.base_dir or PARENT_DIR,
            archive=not args.no_archive
        )

    else:
        migrate_all(
            base_dir=args.base_dir,
            archive=not args.no_archive,
            dry_run=args.dry_run
        )
