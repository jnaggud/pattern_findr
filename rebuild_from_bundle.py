#!/usr/bin/env python3
"""
Rebuild database from strategy bundle's locked_backtest.json (Ground Truth).

This is the SAFEST way to rebuild a corrupted database - it uses the exact trades
from the Streamlit UI backtest rather than re-running simulation (which could
produce different results due to parameter mismatches).

Usage:
    python rebuild_from_bundle.py --strategy velocity_GC=F_any_reversal_JD1
    python rebuild_from_bundle.py --list  # List available bundles
"""

import os
import sys
import json
import glob
import argparse

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from velocity_trading.core.position_manager import PositionManager
from velocity_trading.core.database import get_db_path

# Directory where Streamlit UI saves optimized strategies
VELOCITY_STRATEGIES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "velocity_strategies")


def find_newest_bundle(strategy_name: str) -> dict:
    """
    Find the newest bundle directory for a strategy.

    Bundles are named like: velocity_GC=F_any_reversal_JD1_20260117_095753/
    This finds the most recent one by the timestamp suffix.

    For bundles without backtest results (daily strategies), falls back to
    root-level velocity_locked_backtest_{strategy}.json file.
    """
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(VELOCITY_STRATEGIES_DIR, f"{strategy_name}_*")
    matches = glob.glob(pattern)

    if not matches:
        return None

    # Sort by modification time (newest first)
    matches.sort(key=os.path.getmtime, reverse=True)
    bundle_dir = matches[0]

    # Load config
    config_path = os.path.join(bundle_dir, "velocity_config.json")
    if not os.path.exists(config_path):
        return None

    with open(config_path) as f:
        config = json.load(f)

    # Try to find backtest results - multiple locations
    backtest_path = None
    backtest_source = None

    # 1. Try bundle's locked_backtest.json
    candidate = os.path.join(bundle_dir, "locked_backtest.json")
    if os.path.exists(candidate):
        backtest_path = candidate
        backtest_source = "bundle/locked_backtest.json"

    # 2. Try bundle's backtest_results.json
    if not backtest_path:
        candidate = os.path.join(bundle_dir, "backtest_results.json")
        if os.path.exists(candidate):
            backtest_path = candidate
            backtest_source = "bundle/backtest_results.json"

    # 3. Fall back to root-level velocity_locked_backtest_{strategy}.json
    if not backtest_path:
        candidate = os.path.join(ROOT_DIR, f"velocity_locked_backtest_{strategy_name}.json")
        if os.path.exists(candidate):
            backtest_path = candidate
            backtest_source = f"root/velocity_locked_backtest_{strategy_name}.json"

    if not backtest_path:
        print(f"  WARNING: No backtest results found for {strategy_name}")
        print(f"  Checked: bundle/locked_backtest.json, bundle/backtest_results.json,")
        print(f"           velocity_locked_backtest_{strategy_name}.json")
        return None

    with open(backtest_path) as f:
        locked_backtest = json.load(f)

    return {
        'bundle_dir': bundle_dir,
        'config': config,
        'locked_backtest': locked_backtest,
        'backtest_source': backtest_source
    }


def list_available_bundles():
    """List all available strategy bundles."""
    import re

    if not os.path.exists(VELOCITY_STRATEGIES_DIR):
        print("No velocity_strategies directory found")
        return

    # Find all velocity_config.json files
    pattern = os.path.join(VELOCITY_STRATEGIES_DIR, "*", "velocity_config.json")
    config_files = glob.glob(pattern)

    # Group by strategy name
    bundles = {}
    for config_path in config_files:
        try:
            with open(config_path) as f:
                config = json.load(f)

            dir_name = os.path.basename(os.path.dirname(config_path))
            # Remove timestamp suffix
            strategy_name = re.sub(r'_\d{8}_\d{6}$', '', dir_name)

            ticker = config.get('ticker', '?')
            interval = config.get('interval', '?')

            if strategy_name not in bundles:
                bundles[strategy_name] = {
                    'ticker': ticker,
                    'interval': interval,
                    'count': 1,
                    'newest': dir_name
                }
            else:
                bundles[strategy_name]['count'] += 1
                # Keep newest by checking timestamp
                if dir_name > bundles[strategy_name]['newest']:
                    bundles[strategy_name]['newest'] = dir_name

        except Exception as e:
            continue

    if not bundles:
        print("No strategy bundles found")
        return

    print("\nAvailable strategy bundles:")
    print("-" * 70)
    for name in sorted(bundles.keys()):
        b = bundles[name]
        print(f"  {name}")
        print(f"      Ticker: {b['ticker']} | Interval: {b['interval']} | Versions: {b['count']}")
        print()


def rebuild_database(strategy_name: str, backup: bool = True):
    """
    Rebuild database from bundle's locked_backtest.json.

    Args:
        strategy_name: Strategy name (e.g., velocity_GC=F_any_reversal_JD1)
        backup: If True, backup existing database before rebuild
    """
    print("=" * 70)
    print(f"  REBUILD DATABASE FROM BUNDLE")
    print(f"  Strategy: {strategy_name}")
    print("=" * 70)

    # Find the bundle
    bundle = find_newest_bundle(strategy_name)
    if bundle is None:
        print(f"\nERROR: No bundle found for strategy '{strategy_name}'")
        print("Run 'python rebuild_from_bundle.py --list' to see available strategies")
        return False

    print(f"\nBundle: {bundle['bundle_dir']}")
    print(f"Backtest source: {bundle.get('backtest_source', 'unknown')}")

    config = bundle['config']
    locked_backtest = bundle['locked_backtest']

    print(f"Config:")
    print(f"  ticker: {config.get('ticker')}")
    print(f"  interval: {config.get('interval')}")
    print(f"  signal_type: {config.get('signal_type')}")
    print(f"  stop_loss_pct: {config.get('stop_loss_pct')}")
    print(f"  take_profit_pct: {config.get('take_profit_pct')}")
    print(f"  oversold_threshold: {config.get('oversold_threshold')}")
    print(f"  overbought_threshold: {config.get('overbought_threshold')}")

    print(f"\nLocked backtest (Ground Truth):")
    print(f"  Entries: {len(locked_backtest.get('entries', []))}")
    print(f"  Exits: {len(locked_backtest.get('exits', []))}")

    # Try to get last bar date from data.parquet (for setting last_processed_bar)
    last_bar_date = None
    data_parquet_path = os.path.join(bundle['bundle_dir'], 'data.parquet')
    if os.path.exists(data_parquet_path):
        try:
            import pandas as pd
            df = pd.read_parquet(data_parquet_path)
            if not df.empty:
                last_bar_date = str(df.index[-1])
                print(f"  Data range: {str(df.index[0])[:10]} to {last_bar_date[:10]}")
        except Exception as e:
            print(f"  Warning: Could not read data.parquet: {e}")

    # Backup existing database
    db_path = get_db_path(strategy_name)
    if backup and os.path.exists(db_path):
        backup_path = db_path + '.backup_before_rebuild'
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"\nBackup created: {backup_path}")

    # Create PositionManager and import
    print(f"\nImporting trades to database...")
    pm = PositionManager(strategy_name, db_path)

    success, result = pm.import_from_locked_backtest(
        locked_backtest=locked_backtest,
        ticker=config.get('ticker'),
        clear_existing=True,
        last_bar_date=last_bar_date
    )

    if success:
        print(f"\n✅ REBUILD COMPLETE")
        print(f"   Imported: {result['imported_trades']} trades")
        print(f"   Win rate: {result['stats']['win_rate']:.1f}%")
        print(f"   Total return: {result['stats']['total_return']:.1f}%")
        if result.get('open_position'):
            pos = result['open_position']
            print(f"   Open position: LONG @ ${pos['entry_price']:,.2f}")
    else:
        print(f"\n❌ REBUILD FAILED")
        print(f"   Error: {result.get('error')}")
        return False

    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild database from strategy bundle (Ground Truth)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python rebuild_from_bundle.py --list
    python rebuild_from_bundle.py --strategy velocity_GC=F_any_reversal_JD1
    python rebuild_from_bundle.py --strategy velocity_ES=F_any_reversal_sl.72 --no-backup
        """
    )

    parser.add_argument('--strategy', '-s', help='Strategy name to rebuild')
    parser.add_argument('--list', '-l', action='store_true', help='List available bundles')
    parser.add_argument('--no-backup', action='store_true', help='Skip database backup')

    args = parser.parse_args()

    if args.list:
        list_available_bundles()
        return

    if not args.strategy:
        print("ERROR: --strategy is required (or use --list to see options)")
        parser.print_help()
        sys.exit(1)

    success = rebuild_database(args.strategy, backup=not args.no_backup)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
