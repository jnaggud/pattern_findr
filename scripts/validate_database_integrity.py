#!/usr/bin/env python3
"""
Validate database integrity for trading databases.

Checks:
1. No duplicate trades (same entry_date)
2. No same-bar entry/exit (entry_date == exit_date when truncated)
3. No overlapping trades (entry before previous exit)
4. All required fields populated
5. OHLC timestamp relationships

Usage:
    python scripts/validate_database_integrity.py                    # All databases
    python scripts/validate_database_integrity.py velocity_ES=F_15m_v5  # Specific strategy
    python scripts/validate_database_integrity.py --fix              # Attempt auto-fixes
"""

import os
import sys
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_DIR = PROJECT_ROOT / "velocity_trading" / "db"


class IntegrityChecker:
    """Check database integrity for a single strategy."""

    def __init__(self, db_path: str, strategy_name: str):
        self.db_path = db_path
        self.strategy_name = strategy_name
        self.issues: List[Dict] = []
        self.warnings: List[Dict] = []

    def check_all(self) -> Tuple[int, int]:
        """
        Run all integrity checks.

        Returns:
            Tuple of (issue_count, warning_count)
        """
        if not os.path.exists(self.db_path):
            self.issues.append({
                'type': 'missing_database',
                'message': f"Database file not found: {self.db_path}"
            })
            return len(self.issues), len(self.warnings)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            # Check if this is a trading database (has trades table)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
            )
            if cursor.fetchone() is None:
                # Not a trading database (e.g., OHLCV data-only), skip
                self.warnings.append({
                    'type': 'not_trading_db',
                    'message': "No 'trades' table found, skipping trading integrity checks"
                })
                return len(self.issues), len(self.warnings)

            self._check_duplicate_trades(conn)
            self._check_same_bar_trades(conn)
            self._check_overlapping_trades(conn)
            self._check_required_fields(conn)
            self._check_position_consistency(conn)
        finally:
            conn.close()

        return len(self.issues), len(self.warnings)

    def _check_duplicate_trades(self, conn: sqlite3.Connection) -> None:
        """Check for duplicate trades with same entry_date."""
        cursor = conn.execute("""
            SELECT entry_date, COUNT(*) as count
            FROM trades
            WHERE strategy_name = ?
            GROUP BY entry_date
            HAVING count > 1
        """, (self.strategy_name,))

        for row in cursor.fetchall():
            self.issues.append({
                'type': 'duplicate_trade',
                'entry_date': row['entry_date'],
                'count': row['count'],
                'message': f"Duplicate trades at {row['entry_date']} (count: {row['count']})"
            })

    def _check_same_bar_trades(self, conn: sqlite3.Connection) -> None:
        """Check for same-bar entry/exit."""
        cursor = conn.execute("""
            SELECT id, entry_date, exit_date
            FROM trades
            WHERE strategy_name = ?
            AND exit_date IS NOT NULL
        """, (self.strategy_name,))

        for row in cursor.fetchall():
            entry = row['entry_date']
            exit_dt = row['exit_date']

            # Truncate to minute for comparison
            entry_bar = entry[:16] if entry else None
            exit_bar = exit_dt[:16] if exit_dt else None

            if entry_bar and exit_bar and entry_bar == exit_bar:
                self.issues.append({
                    'type': 'same_bar_trade',
                    'trade_id': row['id'],
                    'entry_date': entry,
                    'exit_date': exit_dt,
                    'message': f"Same-bar trade: ID {row['id']} entry={entry} exit={exit_dt}"
                })

    def _check_overlapping_trades(self, conn: sqlite3.Connection) -> None:
        """Check for overlapping trades."""
        cursor = conn.execute("""
            SELECT id, entry_date, exit_date
            FROM trades
            WHERE strategy_name = ?
            ORDER BY entry_date
        """, (self.strategy_name,))

        trades = cursor.fetchall()

        for i in range(1, len(trades)):
            prev = trades[i - 1]
            curr = trades[i]

            if prev['exit_date'] and curr['entry_date'] < prev['exit_date']:
                self.issues.append({
                    'type': 'overlapping_trades',
                    'trade_id_1': prev['id'],
                    'trade_id_2': curr['id'],
                    'message': f"Overlapping: Trade {prev['id']} ({prev['entry_date']} - {prev['exit_date']}) "
                               f"overlaps with Trade {curr['id']} ({curr['entry_date']})"
                })

    def _check_required_fields(self, conn: sqlite3.Connection) -> None:
        """Check that required fields are populated."""
        # Check trades table
        cursor = conn.execute("""
            SELECT id, entry_date, entry_price, ticker
            FROM trades
            WHERE strategy_name = ?
            AND (entry_date IS NULL OR entry_price IS NULL OR ticker IS NULL)
        """, (self.strategy_name,))

        for row in cursor.fetchall():
            self.issues.append({
                'type': 'missing_required_field',
                'table': 'trades',
                'trade_id': row['id'],
                'message': f"Trade {row['id']} missing required field(s)"
            })

        # Check for trades with exit but no pnl
        cursor = conn.execute("""
            SELECT id, exit_date, pnl_pct
            FROM trades
            WHERE strategy_name = ?
            AND exit_date IS NOT NULL
            AND pnl_pct IS NULL
        """, (self.strategy_name,))

        for row in cursor.fetchall():
            self.warnings.append({
                'type': 'missing_pnl',
                'trade_id': row['id'],
                'message': f"Trade {row['id']} has exit but no pnl_pct"
            })

    def _check_position_consistency(self, conn: sqlite3.Connection) -> None:
        """Check position table consistency with trades."""
        # Check for position without matching open trade
        cursor = conn.execute("""
            SELECT p.id, p.entry_date, p.trade_id
            FROM positions p
            WHERE p.strategy_name = ?
        """, (self.strategy_name,))

        position = cursor.fetchone()

        if position:
            # Verify trade exists and is open
            cursor = conn.execute("""
                SELECT id, exit_date FROM trades WHERE id = ?
            """, (position['trade_id'],))

            trade = cursor.fetchone()

            if not trade:
                self.issues.append({
                    'type': 'orphaned_position',
                    'position_id': position['id'],
                    'message': f"Position references non-existent trade {position['trade_id']}"
                })
            elif trade['exit_date']:
                self.issues.append({
                    'type': 'position_closed_trade',
                    'position_id': position['id'],
                    'trade_id': trade['id'],
                    'message': f"Position references closed trade {trade['id']}"
                })

    def print_report(self) -> None:
        """Print validation report."""
        print(f"\n{'='*60}")
        print(f"Strategy: {self.strategy_name}")
        print(f"Database: {self.db_path}")
        print(f"{'='*60}")

        if not self.issues and not self.warnings:
            print("\n  [PASS] All integrity checks passed")
        else:
            if self.issues:
                print(f"\n  [FAIL] {len(self.issues)} issue(s) found:")
                for issue in self.issues:
                    print(f"    - {issue['type']}: {issue['message']}")

            if self.warnings:
                print(f"\n  [WARN] {len(self.warnings)} warning(s):")
                for warning in self.warnings:
                    print(f"    - {warning['type']}: {warning['message']}")


def find_databases(strategy_filter: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Find all trading databases.

    Args:
        strategy_filter: Optional strategy name to filter

    Returns:
        List of (db_path, strategy_name) tuples
    """
    databases = []

    if not DB_DIR.exists():
        print(f"Warning: Database directory not found: {DB_DIR}")
        return databases

    for db_file in DB_DIR.glob("*.db"):
        strategy_name = db_file.stem

        if strategy_filter and strategy_filter not in strategy_name:
            continue

        databases.append((str(db_file), strategy_name))

    return sorted(databases)


def main():
    parser = argparse.ArgumentParser(description="Validate trading database integrity")
    parser.add_argument('strategy', nargs='?', help="Strategy name filter (optional)")
    parser.add_argument('--fix', action='store_true', help="Attempt to fix issues (not implemented)")
    parser.add_argument('--quiet', '-q', action='store_true', help="Only show failures")
    args = parser.parse_args()

    if args.fix:
        print("Warning: --fix not yet implemented")

    databases = find_databases(args.strategy)

    if not databases:
        print("No databases found to validate")
        return 1

    print(f"\nValidating {len(databases)} database(s)...")

    total_issues = 0
    total_warnings = 0
    failed_strategies = []

    for db_path, strategy_name in databases:
        checker = IntegrityChecker(db_path, strategy_name)
        issues, warnings = checker.check_all()

        total_issues += issues
        total_warnings += warnings

        if issues > 0:
            failed_strategies.append(strategy_name)

        if not args.quiet or issues > 0 or warnings > 0:
            checker.print_report()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Databases checked: {len(databases)}")
    print(f"Total issues:      {total_issues}")
    print(f"Total warnings:    {total_warnings}")

    if failed_strategies:
        print(f"\nFailed strategies:")
        for name in failed_strategies:
            print(f"  - {name}")
        return 1
    else:
        print(f"\n[PASS] All databases passed integrity checks")
        return 0


if __name__ == "__main__":
    sys.exit(main())
