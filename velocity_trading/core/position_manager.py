"""
Position Manager - Atomic state transitions for velocity trading.

This is the CRITICAL component that prevents the desync bugs that plagued
the old system. All position changes go through this class, which ensures:

INVARIANTS:
1. positions table and trades table are ALWAYS in sync
2. trades table is APPEND-ONLY (historical trades never modified)
3. Only ONE position can be open per strategy at any time
4. Entry and exit are atomic (single SQLite transaction)

USAGE PATTERN:
    pm = PositionManager(strategy_name)

    # Entry signal detected
    success, result = pm.enter_position(...)
    if success:
        send_discord_entry_alert(result)  # Only send if DB updated
    else:
        print(f"Entry rejected: {result['error']}")  # No Discord, no desync

    # Exit signal detected
    success, result = pm.exit_position(...)
    if success:
        send_discord_exit_alert(result)  # Only send if DB updated
    else:
        print(f"Exit rejected: {result['error']}")  # No Discord, no desync
"""

import sqlite3
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, asdict

from .database import TradingDatabase, get_db_path, init_database
import pandas as pd

# Base directory for JSON files (parent of velocity_trading/)


# =============================================================================
# Timestamp Utilities - CRITICAL for consistent timestamp handling
# =============================================================================

def normalize_timestamp(ts: str) -> str:
    """
    Convert any timestamp format to standard ISO: YYYY-MM-DDTHH:MM:SS

    Handles:
    - "2026-01-23 11:15:00" (space separator)
    - "2026-01-23T03:45:05" (ISO with T)
    - "2026-01-23 11:15:00+00:00" (with TZ)
    - "2026-01-23T11:15:00-06:00" (CST offset)

    All outputs are naive UTC for consistent database storage.
    """
    dt = pd.to_datetime(ts)
    if dt.tzinfo is not None:
        dt = dt.tz_convert('UTC').tz_localize(None)
    return dt.strftime('%Y-%m-%dT%H:%M:%S')


def compare_timestamps(ts1: str, ts2: str) -> int:
    """
    Compare two timestamps reliably using datetime parsing.

    Returns:
        -1 if ts1 < ts2
         0 if ts1 == ts2
         1 if ts1 > ts2

    NEVER use string comparison for timestamps - ASCII ordering fails:
    "2026-01-23 11:15:00" (space=32) < "2026-01-23T03:45:05" (T=84)
    even though 11:15 > 03:45
    """
    dt1 = pd.to_datetime(ts1)
    dt2 = pd.to_datetime(ts2)

    # Normalize to naive UTC for comparison
    if dt1.tzinfo is not None:
        dt1 = dt1.tz_convert('UTC').tz_localize(None)
    if dt2.tzinfo is not None:
        dt2 = dt2.tz_convert('UTC').tz_localize(None)

    if dt1 < dt2:
        return -1
    elif dt1 > dt2:
        return 1
    return 0


def get_bar_key(ts: str, precision: str = 'minute') -> str:
    """
    Get a bar key for same-bar detection.

    Args:
        ts: Timestamp string
        precision: 'minute' (default), 'hour', or 'day'

    Returns:
        Truncated timestamp string for bar comparison
    """
    dt = pd.to_datetime(ts)
    if dt.tzinfo is not None:
        dt = dt.tz_convert('UTC').tz_localize(None)

    if precision == 'day':
        return dt.strftime('%Y-%m-%d')
    elif precision == 'hour':
        return dt.strftime('%Y-%m-%dT%H')
    else:  # minute
        return dt.strftime('%Y-%m-%dT%H:%M')
BASE_DIR = Path(__file__).parent.parent.parent


@dataclass
class Position:
    """Current open position."""
    strategy_name: str
    ticker: str
    position_type: str  # 'long' or 'short'
    entry_price: float
    entry_date: str
    entry_signal_bar: Optional[str] = None
    last_signal_time: Optional[str] = None
    trade_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TradeResult:
    """Result of a completed trade."""
    trade_id: int
    ticker: str
    position_type: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    pnl_pct: float
    pnl_dollars: float
    exit_reason: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class PositionManagerError(Exception):
    """Base exception for position management errors."""
    pass


class DuplicateEntryError(PositionManagerError):
    """Raised when trying to enter position while already in one."""
    pass


class NoPositionError(PositionManagerError):
    """Raised when trying to exit without an open position."""
    pass


class EntryNotFoundError(PositionManagerError):
    """Raised when exit references non-existent entry."""
    pass


class PositionManager:
    """
    Manages position state with atomic SQLite transactions.

    This class is the single source of truth for position state.
    All public methods return (success: bool, result: dict) tuples.
    Callers MUST check success BEFORE sending Discord notifications
    to prevent state desync.
    """

    def __init__(self, strategy_name: str, db_path: str = None, auto_rebuild: bool = False):
        """
        Initialize position manager for a strategy.

        Args:
            strategy_name: Name of the trading strategy
            db_path: Optional explicit database path
            auto_rebuild: If True, rebuild trades from backtest if database is empty
        """
        self.strategy_name = strategy_name
        self.db_path = db_path or get_db_path(strategy_name)

        # Ensure database is initialized
        init_database(self.db_path)

        self._db = TradingDatabase(strategy_name, self.db_path)

        # Auto-rebuild from backtest if database is empty
        if auto_rebuild and self.get_trade_count() == 0:
            print(f"   ℹ️  Database empty for {strategy_name}, will rebuild from backtest")

        # Always sync positions from trades at startup (fixes desync issues)
        self._sync_positions_from_trades()

    def _sync_positions_from_trades(self):
        """
        Sync positions table from trades table.

        Handles three cases:
        1. Open trade exists but position doesn't → create position
        2. Position exists but no open trade → delete orphan position
        3. Position exists but references wrong/closed trade → update to correct trade

        This fixes desync issues from migration, crashes, or bugs.
        """
        # Check for open trades without position records
        open_trade = self._db.execute_one(
            """SELECT id, entry_date, entry_price, position_type, ticker
               FROM trades
               WHERE strategy_name = ? AND exit_date IS NULL
               ORDER BY id DESC LIMIT 1""",
            (self.strategy_name,)
        )

        current_position = self.get_current_position()

        if open_trade and not current_position:
            # Trade exists but position doesn't - fix the desync
            print(f"   ⚠️  Syncing position from trade #{open_trade['id']}")
            self._db.execute(
                """INSERT OR REPLACE INTO positions
                   (strategy_name, ticker, position_type, entry_price, entry_date, trade_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (self.strategy_name, open_trade['ticker'], open_trade['position_type'],
                 open_trade['entry_price'], open_trade['entry_date'], open_trade['id'])
            )
        elif current_position and not open_trade:
            # Position exists but no open trade - clear the orphan position
            print(f"   ⚠️  Clearing orphan position (no matching open trade)")
            self._db.execute(
                "DELETE FROM positions WHERE strategy_name = ?",
                (self.strategy_name,)
            )
        elif open_trade and current_position:
            # Both exist - check if they match
            if current_position.trade_id != open_trade['id']:
                # Position references wrong trade - update to correct one
                print(f"   ⚠️  Position mismatch! Position has trade #{current_position.trade_id}, "
                      f"but actual open trade is #{open_trade['id']}")
                print(f"   ⚠️  Updating position: ${current_position.entry_price:,.2f} → ${open_trade['entry_price']:,.2f}")
                self._db.execute(
                    """UPDATE positions SET
                           ticker = ?, position_type = ?, entry_price = ?,
                           entry_date = ?, trade_id = ?, updated_at = datetime('now')
                       WHERE strategy_name = ?""",
                    (open_trade['ticker'], open_trade['position_type'],
                     open_trade['entry_price'], open_trade['entry_date'],
                     open_trade['id'], self.strategy_name)
                )

    # =========================================================================
    # Core Position Operations
    # =========================================================================

    def get_current_position(self) -> Optional[Position]:
        """
        Get current open position, or None if flat.

        Thread-safe read operation.
        """
        row = self._db.execute_one(
            "SELECT * FROM positions WHERE strategy_name = ?",
            (self.strategy_name,)
        )

        if row:
            return Position(
                strategy_name=row['strategy_name'],
                ticker=row['ticker'],
                position_type=row['position_type'],
                entry_price=row['entry_price'],
                entry_date=row['entry_date'],
                entry_signal_bar=row.get('entry_signal_bar'),
                last_signal_time=row.get('last_signal_time'),
                trade_id=row.get('trade_id')
            )
        return None

    def enter_position(
        self,
        ticker: str,
        position_type: str,
        entry_price: float,
        entry_date: str,
        entry_signal_bar: str = None,
        is_missed: bool = False
    ) -> Tuple[bool, Dict]:
        """
        Atomically enter a new position.

        This method:
        1. Checks for existing position (rejects if one exists)
        2. Inserts trade record into trades table
        3. Inserts position record into positions table
        4. All in a single atomic transaction

        Args:
            ticker: Trading instrument symbol
            position_type: 'long' or 'short'
            entry_price: Entry price
            entry_date: Entry timestamp (ISO format)
            entry_signal_bar: Bar that generated the signal
            is_missed: True if this is a missed signal being recovered

        Returns:
            (success: bool, result: dict)
            If success=False, result['error'] contains reason code
            If success=True, result contains trade_id, entry details

        CRITICAL: Caller should only send Discord if success=True
        """
        try:
            with self._db.transaction() as conn:
                # Check for existing position
                existing = conn.execute(
                    "SELECT * FROM positions WHERE strategy_name = ?",
                    (self.strategy_name,)
                ).fetchone()

                if existing:
                    # Verify the position is actually valid (trade exists and is open)
                    trade_row = conn.execute(
                        "SELECT id, exit_date FROM trades WHERE id = ?",
                        (existing['trade_id'],)
                    ).fetchone()

                    if trade_row and trade_row['exit_date'] is None:
                        # Valid open position - reject entry
                        return False, {
                            'error': 'DuplicateEntry',
                            'message': f"Already in {existing['position_type']} position from {existing['entry_date']}",
                            'existing_position': dict(existing)
                        }
                    else:
                        # Orphaned position - clean up and continue with entry
                        print(f"   Cleaning up orphaned position record (trade {existing['trade_id']} closed or missing)")
                        conn.execute(
                            "DELETE FROM positions WHERE strategy_name = ?",
                            (self.strategy_name,)
                        )

                # CRITICAL: Double-check for ANY open trades (belt and suspenders)
                # This catches cases where positions table is out of sync with trades
                open_trade = conn.execute(
                    """SELECT id, entry_date, entry_price FROM trades
                       WHERE strategy_name = ? AND exit_date IS NULL
                       ORDER BY id DESC LIMIT 1""",
                    (self.strategy_name,)
                ).fetchone()

                if open_trade:
                    # There's an open trade but no position record - fix the desync and reject
                    print(f"   ⚠️ Found open trade #{open_trade['id']} without position record, rejecting new entry")
                    return False, {
                        'error': 'OpenTradeExists',
                        'message': f"Open trade exists from {open_trade['entry_date']} @ ${open_trade['entry_price']:.2f}",
                        'open_trade_id': open_trade['id']
                    }

                # Check for duplicate entry by date
                duplicate = conn.execute(
                    """SELECT id FROM trades
                       WHERE strategy_name = ? AND entry_date = ?""",
                    (self.strategy_name, entry_date[:19])
                ).fetchone()

                if duplicate:
                    return False, {
                        'error': 'DuplicateEntryDate',
                        'message': f"Entry already exists for {entry_date}"
                    }

                # CRITICAL: Check for overlapping trades (new entry falls within existing trade's duration)
                # This prevents creating duplicate entries when process_new_bars runs on restart
                # and processes bars that already have closed trades covering them
                overlapping = conn.execute(
                    """SELECT id, entry_date, exit_date FROM trades
                       WHERE strategy_name = ?
                         AND entry_date < ?
                         AND (exit_date IS NULL OR exit_date > ?)
                       ORDER BY entry_date DESC LIMIT 1""",
                    (self.strategy_name, entry_date, entry_date)
                ).fetchone()

                if overlapping:
                    return False, {
                        'error': 'OverlappingTrade',
                        'message': f"Entry at {entry_date} overlaps with trade from {overlapping['entry_date']} to {overlapping['exit_date'] or 'open'}"
                    }

                # Insert into trades table
                cursor = conn.execute(
                    """INSERT INTO trades
                       (strategy_name, ticker, entry_date, entry_price,
                        entry_signal_bar, position_type, is_missed)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (self.strategy_name, ticker, entry_date, entry_price,
                     entry_signal_bar, position_type, is_missed)
                )
                trade_id = cursor.lastrowid

                # Insert into positions table
                conn.execute(
                    """INSERT INTO positions
                       (strategy_name, ticker, position_type, entry_price,
                        entry_date, entry_signal_bar, last_signal_time, trade_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (self.strategy_name, ticker, position_type, entry_price,
                     entry_date, entry_signal_bar, entry_date, trade_id)
                )

                return True, {
                    'trade_id': trade_id,
                    'ticker': ticker,
                    'position_type': position_type,
                    'entry_price': entry_price,
                    'entry_date': entry_date,
                    'entry_signal_bar': entry_signal_bar,
                    'is_missed': is_missed
                }

        except sqlite3.IntegrityError as e:
            return False, {
                'error': 'IntegrityError',
                'message': str(e)
            }
        except Exception as e:
            return False, {
                'error': 'Exception',
                'message': str(e)
            }

    def exit_position(
        self,
        exit_price: float,
        exit_date: str,
        exit_reason: str,
        exit_signal_bar: str = None,
        is_missed: bool = False
    ) -> Tuple[bool, Dict]:
        """
        Atomically exit current position.

        This method:
        1. Gets current position (fails if none exists)
        2. Calculates P&L
        3. Updates trade record with exit data
        4. Deletes position record
        5. Recalculates strategy stats
        6. All in a single atomic transaction

        Args:
            exit_price: Exit price
            exit_date: Exit timestamp (ISO format)
            exit_reason: Reason for exit (e.g., "Opposite Signal", "Stop Loss")
            exit_signal_bar: Bar that generated the exit signal
            is_missed: True if this is a missed exit being recovered

        Returns:
            (success: bool, result: dict)
            If success=False, result['error'] contains reason code
            If success=True, result contains full trade details with P&L

        CRITICAL: Caller should only send Discord if success=True
        """
        try:
            with self._db.transaction() as conn:
                # Get current position
                pos_row = conn.execute(
                    "SELECT * FROM positions WHERE strategy_name = ?",
                    (self.strategy_name,)
                ).fetchone()

                if not pos_row:
                    return False, {
                        'error': 'NoPosition',
                        'message': 'No open position to exit'
                    }

                trade_id = pos_row['trade_id']
                entry_price = pos_row['entry_price']
                entry_date = pos_row['entry_date']
                position_type = pos_row['position_type']
                ticker = pos_row['ticker']

                # Validate exit date is after entry date using proper datetime comparison
                cmp_result = compare_timestamps(exit_date, entry_date)
                if cmp_result <= 0:
                    return False, {
                        'error': 'InvalidExitDate',
                        'message': f"Exit date ({exit_date}) must be after entry date ({entry_date})"
                    }

                # Same-bar protection: prevent exit on same bar as entry
                exit_bar_key = get_bar_key(exit_date, 'minute')
                entry_bar_key = get_bar_key(entry_date, 'minute')
                if exit_bar_key == entry_bar_key:
                    return False, {
                        'error': 'SameBarExit',
                        'message': f"Cannot exit on same bar as entry ({exit_bar_key})"
                    }

                # Calculate P&L
                if position_type == 'long':
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                else:  # short
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100

                # Assume $10,000 position for dollar P&L (can be made configurable)
                position_size = 10000
                pnl_dollars = position_size * (pnl_pct / 100)

                # Verify trade exists and has no exit yet
                trade_row = conn.execute(
                    "SELECT * FROM trades WHERE id = ?",
                    (trade_id,)
                ).fetchone()

                if not trade_row:
                    return False, {
                        'error': 'TradeNotFound',
                        'message': f'Trade ID {trade_id} not found in trades table'
                    }

                if trade_row['exit_date'] is not None:
                    return False, {
                        'error': 'AlreadyExited',
                        'message': f'Trade {trade_id} already has exit at {trade_row["exit_date"]}'
                    }

                # Update trades table with exit data
                conn.execute(
                    """UPDATE trades SET
                           exit_date = ?,
                           exit_price = ?,
                           exit_signal_bar = ?,
                           exit_reason = ?,
                           pnl_pct = ?,
                           pnl_dollars = ?
                       WHERE id = ?""",
                    (exit_date, exit_price, exit_signal_bar, exit_reason,
                     pnl_pct, pnl_dollars, trade_id)
                )

                # Remove from positions table
                conn.execute(
                    "DELETE FROM positions WHERE strategy_name = ?",
                    (self.strategy_name,)
                )

                # Recalculate and cache stats
                self._update_stats_internal(conn)

                return True, {
                    'trade_id': trade_id,
                    'ticker': ticker,
                    'position_type': position_type,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': exit_date,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_pct,
                    'pnl_dollars': pnl_dollars,
                    'is_missed': is_missed
                }

        except Exception as e:
            return False, {
                'error': 'Exception',
                'message': str(e)
            }

    def update_last_signal_time(self, signal_time: str) -> bool:
        """
        Update last_signal_time for deduplication.

        Used to prevent re-processing the same signal.
        """
        try:
            with self._db.transaction() as conn:
                cursor = conn.execute(
                    """UPDATE positions
                       SET last_signal_time = ?, updated_at = ?
                       WHERE strategy_name = ?""",
                    (signal_time, datetime.now().isoformat(), self.strategy_name)
                )
                return cursor.rowcount > 0
        except Exception:
            return False

    # =========================================================================
    # Trade History
    # =========================================================================

    def get_trades(
        self,
        limit: int = None,
        include_missed: bool = False,
        include_open: bool = False
    ) -> List[dict]:
        """
        Get historical trades for this strategy.

        Args:
            limit: Max number of trades to return (most recent first)
            include_missed: Include trades marked as missed
            include_open: Include open trade (no exit yet)

        Returns:
            List of trade dictionaries
        """
        conditions = ["strategy_name = ?"]
        params = [self.strategy_name]

        if not include_open:
            conditions.append("exit_date IS NOT NULL")

        if not include_missed:
            conditions.append("(is_missed = FALSE OR is_missed IS NULL)")

        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT * FROM trades
            WHERE {where_clause}
            ORDER BY COALESCE(exit_date, entry_date) DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        return self._db.execute(query, tuple(params))

    def get_entries_and_exits(self) -> Dict:
        """
        Get entries and exits in locked_backtest format for chart generation.

        Returns a dict compatible with the old locked_backtest JSON structure:
        {
            'entries': [...],
            'exits': [...],
            'current_position': {...} or None,
            'num_trades': int,
            'win_rate': float,
            'total_return': float,
            'profit_factor': float
        }
        """
        # All entries (including open position)
        entries = self._db.execute(
            """SELECT entry_date as date, entry_price as price,
                      position_type as position, is_missed as missed
               FROM trades
               WHERE strategy_name = ?
               ORDER BY entry_date""",
            (self.strategy_name,)
        )

        # Format entries
        formatted_entries = []
        for e in entries:
            entry = {
                'date': e['date'],
                'price': e['price'],
                'position': e['position']
            }
            if e.get('missed'):
                entry['missed'] = True
            formatted_entries.append(entry)

        # Completed exits only
        exits = self._db.execute(
            """SELECT exit_date as date, exit_price as price, pnl_pct as pnl,
                      exit_reason as reason, entry_price, entry_date,
                      is_missed as missed
               FROM trades
               WHERE strategy_name = ? AND exit_date IS NOT NULL
               ORDER BY exit_date""",
            (self.strategy_name,)
        )

        # Format exits
        formatted_exits = []
        for e in exits:
            exit_rec = {
                'date': e['date'],
                'price': e['price'],
                'pnl': e['pnl'],
                'reason': e['reason'],
                'entry_price': e['entry_price'],
                'entry_date': e['entry_date']
            }
            if e.get('missed'):
                exit_rec['missed'] = True
            formatted_exits.append(exit_rec)

        # Current position
        pos = self.get_current_position()
        current_position = None
        if pos:
            current_position = {
                'position': pos.position_type,
                'entry_price': pos.entry_price,
                'entry_date': pos.entry_date
            }

        # Get cached stats
        stats = self._db.get_stats() or {}

        return {
            'entries': formatted_entries,
            'exits': formatted_exits,
            'current_position': current_position,
            'num_trades': stats.get('num_trades', 0),
            'win_rate': stats.get('win_rate', 0.0),
            'total_return': stats.get('total_return', 0.0),
            'profit_factor': stats.get('profit_factor', 0.0)
        }

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get cached strategy statistics."""
        return self._db.get_stats() or {
            'num_trades': 0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'profit_factor': 0.0
        }

    def recalculate_stats(self) -> Dict:
        """Force recalculation of strategy statistics."""
        with self._db.transaction() as conn:
            return self._update_stats_internal(conn)

    def _update_stats_internal(self, conn: sqlite3.Connection) -> Dict:
        """
        Recalculate and cache strategy statistics.

        Called internally within a transaction.
        """
        # Get all completed trades (excluding missed)
        trades = conn.execute(
            """SELECT * FROM trades
               WHERE strategy_name = ? AND exit_date IS NOT NULL
                     AND (is_missed = FALSE OR is_missed IS NULL)
               ORDER BY exit_date""",
            (self.strategy_name,)
        ).fetchall()

        if not trades:
            stats = {
                'num_trades': 0,
                'num_wins': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
        else:
            wins = [t for t in trades if t['pnl_pct'] > 0]
            losses = [t for t in trades if t['pnl_pct'] <= 0]

            num_trades = len(trades)
            num_wins = len(wins)
            win_rate = (num_wins / num_trades) * 100 if num_trades > 0 else 0

            # Compounded return
            equity = 1.0
            for t in trades:
                equity *= (1 + t['pnl_pct'] / 100)
            total_return = (equity - 1) * 100

            # Profit factor
            gross_profit = sum(t['pnl_pct'] for t in wins) if wins else 0
            gross_loss = abs(sum(t['pnl_pct'] for t in losses)) if losses else 0.001
            profit_factor = min(gross_profit / gross_loss, 999.99) if gross_loss > 0 else 999.99

            # Averages
            avg_win = sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t['pnl_pct'] for t in losses) / len(losses) if losses else 0

            stats = {
                'num_trades': num_trades,
                'num_wins': num_wins,
                'win_rate': win_rate,
                'total_return': total_return,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            }

        # Get missed trade stats
        missed_trades = conn.execute(
            """SELECT * FROM trades
               WHERE strategy_name = ? AND exit_date IS NOT NULL
                     AND is_missed = TRUE
               ORDER BY exit_date""",
            (self.strategy_name,)
        ).fetchall()

        if missed_trades:
            stats['num_missed'] = len(missed_trades)
            stats['missed_return'] = sum(t['pnl_pct'] for t in missed_trades)
        else:
            stats['num_missed'] = 0
            stats['missed_return'] = 0.0

        # Update cache
        stats['strategy_name'] = self.strategy_name
        stats['last_updated'] = datetime.now().isoformat()

        conn.execute(
            """INSERT OR REPLACE INTO strategy_stats
               (strategy_name, num_trades, num_wins, win_rate, total_return,
                profit_factor, avg_win, avg_loss, num_missed, missed_return, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.strategy_name,
                stats['num_trades'],
                stats['num_wins'],
                stats['win_rate'],
                stats['total_return'],
                stats['profit_factor'],
                stats['avg_win'],
                stats['avg_loss'],
                stats['num_missed'],
                stats['missed_return'],
                stats['last_updated']
            )
        )

        return stats

    # =========================================================================
    # Daily Alerts
    # =========================================================================

    def has_alert(self, alert_key: str) -> bool:
        """Check if an alert has already been sent today."""
        return self._db.has_alert(alert_key)

    def add_alert(self, alert_key: str) -> bool:
        """Add an alert key to prevent duplicate sends."""
        return self._db.add_alert(alert_key)

    def clear_old_alerts(self, days_to_keep: int = 7) -> int:
        """Clear alerts older than specified days."""
        return self._db.clear_old_alerts(days_to_keep)

    # =========================================================================
    # Database Rebuild from Backtest
    # =========================================================================

    def get_trade_count(self) -> int:
        """Get total number of trades in database."""
        row = self._db.execute_one(
            "SELECT COUNT(*) as cnt FROM trades WHERE strategy_name = ?",
            (self.strategy_name,)
        )
        return row['cnt'] if row else 0

    def clear_all_trades(self) -> int:
        """
        Clear all trades and positions for this strategy.
        Returns number of trades deleted.

        WARNING: This is destructive - use with caution.
        """
        with self._db.transaction() as conn:
            # Get count before delete
            count = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE strategy_name = ?",
                (self.strategy_name,)
            ).fetchone()[0]

            # Delete position
            conn.execute(
                "DELETE FROM positions WHERE strategy_name = ?",
                (self.strategy_name,)
            )

            # Delete trades
            conn.execute(
                "DELETE FROM trades WHERE strategy_name = ?",
                (self.strategy_name,)
            )

            # Clear stats
            conn.execute(
                "DELETE FROM strategy_stats WHERE strategy_name = ?",
                (self.strategy_name,)
            )

            return count

    def rebuild_from_backtest(
        self,
        ticker: str,
        config: dict,
        days: int = 365,
        clear_existing: bool = True,
        interval: str = None
    ) -> Tuple[bool, Dict]:
        """
        Rebuild database from fresh backtest using data providers.

        This fetches price data, runs backtest, and populates SQLite directly.
        No JSON files involved - completely independent of old system.

        Args:
            ticker: Symbol to fetch data for
            config: Strategy config with signal parameters
            days: Number of days of historical data
            clear_existing: If True, clear existing trades before rebuild
            interval: Bar interval ('15m', '1d', etc.) - extracted from config if not provided

        Returns:
            (success, result) tuple with trade counts and stats
        """
        try:
            # Import data and indicator modules
            from ..data.fetcher import fetch_price_data
            from ..indicators.oscillators import calculate_composite_oscillator, calculate_composite_oscillator_legacy
            from ..indicators.velocity import calculate_velocity_signals, calculate_velocity_signals_legacy

            # CRITICAL FIX: Get interval from config if not provided
            # This ensures intraday strategies use 15m bars, not daily bars
            if interval is None:
                interval = config.get('interval', '1d')

            # Check if legacy mode requested
            use_legacy = config.get('use_legacy', False)

            print(f"🔄 Rebuilding {self.strategy_name} from backtest...")
            print(f"   Fetching {days} days of {ticker} data ({interval} bars)...")
            if use_legacy:
                print(f"   📌 Using LEGACY calculation (matches old system)")

            # Fetch price data from providers - CRITICAL: include interval!
            df = fetch_price_data(ticker, days=days, interval=interval)
            if df is None or df.empty:
                return False, {'error': f'Failed to fetch data for {ticker}'}

            print(f"   ✓ Loaded {len(df)} bars: {df.index[0].date()} to {df.index[-1].date()}")

            # Validate required config parameters - NO DEFAULTS
            required_params = ['signal_type', 'oversold_threshold', 'overbought_threshold',
                               'stop_loss_pct', 'take_profit_pct']
            missing = [p for p in required_params if p not in config]
            if missing:
                return False, {
                    'error': 'MissingConfig',
                    'message': f"Missing required config parameters: {missing}. "
                               "Config must come from strategy bundle, not hardcoded defaults."
                }

            # Calculate indicators
            if use_legacy:
                # Use old system's exact calculations
                df = calculate_composite_oscillator_legacy(df)
                df = calculate_velocity_signals_legacy(df, config)
            else:
                # Use new calculations - CRITICAL: pass config for oscillator_type (arwo, etc.)
                oscillator_type = config.get('oscillator_type', 'composite')
                df = calculate_composite_oscillator(df, oscillator_type=oscillator_type, config=config)
                df = calculate_velocity_signals(
                    df,
                    signal_type=config['signal_type'],
                    oversold_threshold=config['oversold_threshold'],
                    overbought_threshold=config['overbought_threshold']
                )

            # Extract trades from backtest
            entries = []
            exits = []
            in_position = False
            entry_price = None
            entry_date = None

            # Use config values directly - no defaults!
            stop_loss_pct = config['stop_loss_pct']
            take_profit_pct = config['take_profit_pct']

            for i, row in df.iterrows():
                if not in_position and row.get('buy_signal', False):
                    # Enter position
                    in_position = True
                    entry_price = row['Close']
                    entry_date = str(i)
                    entries.append({
                        'date': entry_date,
                        'price': entry_price,
                        'position': 'long'
                    })

                elif in_position:
                    current_price = row['Close']
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100

                    exit_reason = None

                    # Check exit conditions
                    if pnl_pct <= -stop_loss_pct:
                        exit_reason = f'Stop Loss ({pnl_pct:.2f}%)'
                    elif pnl_pct >= take_profit_pct:
                        exit_reason = f'Take Profit ({pnl_pct:.2f}%)'
                    elif row.get('sell_signal', False):
                        exit_reason = f'Opposite Signal ({pnl_pct:.2f}%)'

                    if exit_reason:
                        exits.append({
                            'date': str(i),
                            'price': current_price,
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'pnl': pnl_pct,
                            'reason': exit_reason
                        })
                        in_position = False
                        entry_price = None
                        entry_date = None

            print(f"   ✓ Backtest complete: {len(entries)} entries, {len(exits)} exits")

            # Clear existing trades if requested
            if clear_existing:
                deleted = self.clear_all_trades()
                if deleted > 0:
                    print(f"   ✓ Cleared {deleted} existing trades")

            # Insert trades into database
            with self._db.transaction() as conn:
                for i, (entry, exit) in enumerate(zip(entries, exits)):
                    conn.execute(
                        """INSERT INTO trades
                           (strategy_name, ticker, entry_date, entry_price, position_type,
                            exit_date, exit_price, exit_reason, pnl_pct, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            self.strategy_name,
                            ticker,
                            entry['date'],
                            entry['price'],
                            'long',
                            exit['date'],
                            exit['price'],
                            exit['reason'],
                            exit['pnl'],
                            datetime.now().isoformat()
                        )
                    )

                # If still in position, add open trade
                if in_position and len(entries) > len(exits):
                    last_entry = entries[-1]
                    trade_id = conn.execute(
                        """INSERT INTO trades
                           (strategy_name, ticker, entry_date, entry_price, position_type, created_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            self.strategy_name,
                            ticker,
                            last_entry['date'],
                            last_entry['price'],
                            'long',
                            datetime.now().isoformat()
                        )
                    ).lastrowid

                    # Add to positions table
                    conn.execute(
                        """INSERT OR REPLACE INTO positions
                           (strategy_name, ticker, position_type, entry_price, entry_date, trade_id, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            self.strategy_name,
                            ticker,
                            'long',
                            last_entry['price'],
                            last_entry['date'],
                            trade_id,
                            datetime.now().isoformat()
                        )
                    )
                    print(f"   ✓ Open position: LONG @ ${last_entry['price']:,.2f}")

            # Recalculate stats
            stats = self.recalculate_stats()

            print(f"   ✓ Database rebuilt: {len(exits)} completed trades, {stats['win_rate']:.1f}% win rate")

            return True, {
                'entries': len(entries),
                'exits': len(exits),
                'open_position': in_position,
                'stats': stats
            }

        except Exception as e:
            import traceback
            return False, {'error': str(e), 'traceback': traceback.format_exc()}

    def get_last_trade_date(self) -> Optional[str]:
        """Get the most recent trade date (entry or exit) in the database."""
        row = self._db.execute_one(
            """SELECT MAX(date) as last_date FROM (
                SELECT entry_date as date FROM trades WHERE strategy_name = ?
                UNION ALL
                SELECT exit_date as date FROM trades WHERE strategy_name = ? AND exit_date IS NOT NULL
            )""",
            (self.strategy_name, self.strategy_name)
        )
        return row['last_date'] if row and row['last_date'] else None

    def process_new_bars(
        self,
        ticker: str,
        config: dict,
        interval: str = '1d',
        lookback_bars: int = 50
    ) -> Tuple[bool, Dict]:
        """
        Process any unprocessed COMPLETED bars since last check.

        FORWARD-ONLY TRADING: Each bar is evaluated exactly ONCE, ever.
        This prevents repainting - historical trades never change.

        Key behaviors:
        1. Uses last_processed_bar (not last trade date) to track progress
        2. Only processes COMPLETED bars (excludes today's partial bar)
        3. Updates last_processed_bar after EACH bar (even if no signal)
        4. Catches up on missed bars if trader was offline

        Args:
            ticker: Symbol to fetch data for
            config: Strategy config with signal parameters
            interval: Bar interval ('1d', '15m', etc.)
            lookback_bars: Bars for indicator warmup

        Returns:
            (success, result) tuple with new trade counts
        """
        try:
            from ..data.fetcher import fetch_price_data
            from ..indicators.oscillators import calculate_composite_oscillator
            from ..indicators.velocity import calculate_velocity_signals
            import pandas as pd

            # Get last processed bar from strategy_state table
            last_processed = self._db.get_last_processed_bar()

            # Fallback: if no last_processed, use last trade date from database
            if last_processed is None:
                last_processed = self.get_last_trade_date()

            if last_processed is None:
                # Fresh database - nothing processed yet
                print(f"   ℹ️  No processed bars yet, will process available completed bars")
                # Set a far-back date to process all available data
                last_processed = '2020-01-01T00:00:00'

            print(f"   🔄 Checking for new bars since {last_processed[:10]}...")

            # Parse last_processed to datetime
            last_dt = pd.to_datetime(last_processed)
            if last_dt.tzinfo is None:
                last_dt = last_dt.tz_localize('UTC')
            else:
                last_dt = last_dt.tz_convert('UTC')

            # Fetch enough data for indicators + new bars
            days_since = (pd.Timestamp.now(tz='UTC') - last_dt).days + lookback_bars
            df = fetch_price_data(ticker, days=max(days_since, 60), interval=interval)
            if df is None or df.empty:
                return False, {'error': f'Failed to fetch data for {ticker}'}

            # Validate required config parameters - NO DEFAULTS
            required_params = ['signal_type', 'oversold_threshold', 'overbought_threshold',
                               'stop_loss_pct', 'take_profit_pct']
            missing = [p for p in required_params if p not in config]
            if missing:
                return False, {
                    'error': 'MissingConfig',
                    'message': f"Missing required config parameters: {missing}. "
                               "Config must come from strategy bundle, not hardcoded defaults."
                }

            # Calculate indicators on full fetched data
            # Support novel oscillators (ARWO, PRF, etc.) and V2 filters
            oscillator_type = config.get('oscillator_type', 'composite')
            df = calculate_composite_oscillator(df, oscillator_type=oscillator_type, config=config)
            df = calculate_velocity_signals(
                df,
                signal_type=config['signal_type'],
                oversold_threshold=config['oversold_threshold'],
                overbought_threshold=config['overbought_threshold'],
                vel_smoothing=config.get('vel_smoothing', 1),
                extreme_zone_mult=config.get('extreme_zone_mult', 1.5),
                require_accel=config.get('require_accel', False),
                use_regime_filter=config.get('use_regime_filter', False),
                regime_threshold=config.get('regime_threshold', -0.15),
                use_fragility_filter=config.get('use_fragility_filter', False),
                fragility_threshold=config.get('fragility_threshold', 0.5),
                use_entropy_filter=config.get('use_entropy_filter', False),
                entropy_threshold=config.get('entropy_threshold', 0.7)
            )

            # Filter to bars AFTER last_processed
            compare_dt = last_dt
            if df.index.tz is None and compare_dt.tzinfo is not None:
                compare_dt = compare_dt.replace(tzinfo=None)
            elif df.index.tz is not None and compare_dt.tzinfo is None:
                compare_dt = compare_dt.tz_localize(df.index.tz)

            df_new = df[df.index > compare_dt]

            # CRITICAL: Exclude today's incomplete bar for daily intervals
            if interval == '1d' and not df_new.empty:
                today = pd.Timestamp.now(tz='UTC').normalize()
                if df_new.index.tz is None:
                    today = today.tz_localize(None)
                elif df_new.index.tz != today.tz:
                    today = today.tz_convert(df_new.index.tz)
                df_new = df_new[df_new.index < today]

            # For intraday: exclude current incomplete bar
            elif interval != '1d' and not df_new.empty:
                # Get interval in minutes
                interval_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240}
                interval_mins = interval_map.get(interval, 15)

                now = pd.Timestamp.now(tz='UTC')
                if df_new.index.tz is None:
                    now = now.tz_localize(None)
                elif df_new.index.tz != now.tz:
                    now = now.tz_convert(df_new.index.tz)

                # A bar is complete if its end time (start + interval) is in the past
                # The bar timestamp is the START of the bar
                bar_end_times = df_new.index + pd.Timedelta(minutes=interval_mins)
                df_new = df_new[bar_end_times <= now]

            if df_new.empty:
                print(f"   ✓ No new completed bars to process")
                return True, {'new_entries': 0, 'new_exits': 0, 'message': 'No new bars'}

            print(f"   ✓ Processing {len(df_new)} new completed bars")

            # Get current position state
            current_pos = self.get_current_position()
            in_position = current_pos is not None
            entry_price = current_pos.entry_price if current_pos else None

            # Config values
            stop_loss_pct = config['stop_loss_pct']
            take_profit_pct = config['take_profit_pct']

            new_entries = 0
            new_exits = 0

            # Process each bar ONCE, in chronological order
            for bar_time, row in df_new.iterrows():
                # Use bar CLOSE time for consistency with live trading
                # For intraday: bar_time + interval = close time
                # For daily: keep bar_time as-is (date represents trading day)
                if interval in ['1d', '1wk', '1mo']:
                    bar_close_time = bar_time
                else:
                    bar_close_time = bar_time + pd.Timedelta(minutes=interval_mins)
                bar_timestamp = normalize_timestamp(bar_close_time)

                # Evaluate this bar for signals
                if not in_position and row.get('buy_signal', False):
                    success, result = self.enter_position(
                        ticker=ticker,
                        position_type='long',
                        entry_price=row['Close'],
                        entry_date=bar_timestamp
                    )
                    if success:
                        in_position = True
                        entry_price = row['Close']
                        new_entries += 1
                        print(f"      📈 Entry @ ${row['Close']:,.2f} on {bar_timestamp[:10]}")

                elif in_position:
                    current_price = row['Close']
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100

                    exit_reason = None
                    if pnl_pct <= -stop_loss_pct:
                        exit_reason = f'Stop Loss ({pnl_pct:.2f}%)'
                    elif pnl_pct >= take_profit_pct:
                        exit_reason = f'Take Profit ({pnl_pct:.2f}%)'
                    elif row.get('sell_signal', False):
                        exit_reason = f'Opposite Signal ({pnl_pct:.2f}%)'

                    if exit_reason:
                        success, result = self.exit_position(
                            exit_price=current_price,
                            exit_date=bar_timestamp,
                            exit_reason=exit_reason
                        )
                        if success:
                            in_position = False
                            entry_price = None
                            new_exits += 1
                            print(f"      📉 Exit @ ${current_price:,.2f} on {bar_timestamp[:10]} ({exit_reason})")

                # CRITICAL: Mark this bar as processed (even if no signal)
                # This ensures we never re-evaluate it
                self._db.set_last_processed_bar(bar_timestamp)

            # Recalculate stats
            if new_entries > 0 or new_exits > 0:
                stats = self.recalculate_stats()
                print(f"   ✓ Added {new_entries} entries, {new_exits} exits")
            else:
                stats = self.get_stats()
                print(f"   ✓ No new signals (position: {'LONG' if in_position else 'FLAT'})")

            return True, {
                'new_entries': new_entries,
                'new_exits': new_exits,
                'total_trades': stats.get('num_trades', 0),
                'open_position': in_position,
                'stats': stats
            }

        except Exception as e:
            import traceback
            return False, {'error': str(e), 'traceback': traceback.format_exc()}

    # Keep old name as alias for backwards compatibility
    def update_from_latest(self, ticker: str, config: dict, interval: str = '1d',
                           lookback_bars: int = 50) -> Tuple[bool, Dict]:
        """DEPRECATED: Use process_new_bars() instead."""
        return self.process_new_bars(ticker, config, interval, lookback_bars)

    def import_from_locked_backtest(
        self,
        locked_backtest: dict,
        ticker: str,
        clear_existing: bool = True,
        last_bar_date: str = None
    ) -> Tuple[bool, Dict]:
        """
        Import trades from a Streamlit UI bundle's locked_backtest.json.

        This is the SAFEST way to rebuild a database - it uses the exact trades
        from the ground truth backtest, avoiding any recalculation that could
        produce different results.

        Args:
            locked_backtest: Dict loaded from locked_backtest.json containing:
                - entries: List of {date, price, position}
                - exits: List of {date, price, pnl, reason, entry_date, entry_price}
            ticker: Trading symbol
            clear_existing: If True, clear existing trades before import
            last_bar_date: Last bar date in the backtest data - sets last_processed_bar
                           to prevent re-processing historical bars

        Returns:
            (success, result) tuple with import counts
        """
        try:
            entries = locked_backtest.get('entries', [])
            exits = locked_backtest.get('exits', [])

            print(f"🔄 Importing from locked_backtest: {len(entries)} entries, {len(exits)} exits")

            if clear_existing:
                deleted = self.clear_all_trades()
                if deleted > 0:
                    print(f"   ✓ Cleared {deleted} existing trades")

            # Build a map of entry_date -> entry for matching
            entry_map = {}
            for entry in entries:
                entry_date = normalize_timestamp(entry['date'])
                entry_map[entry_date] = entry

            # Insert trades
            imported_trades = 0
            open_position = None

            with self._db.transaction() as conn:
                # Match entries with exits and insert as complete trades
                for exit_rec in exits:
                    exit_date = normalize_timestamp(exit_rec['date'])
                    entry_date = normalize_timestamp(exit_rec.get('entry_date', ''))
                    entry_price = exit_rec.get('entry_price')
                    exit_price = exit_rec.get('price')
                    pnl = exit_rec.get('pnl', 0)
                    reason = exit_rec.get('reason', 'Signal')

                    conn.execute(
                        """INSERT INTO trades
                           (strategy_name, ticker, entry_date, entry_price, position_type,
                            exit_date, exit_price, exit_reason, pnl_pct, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            self.strategy_name,
                            ticker,
                            entry_date,
                            entry_price,
                            'long',
                            exit_date,
                            exit_price,
                            reason,
                            pnl,
                            datetime.now().isoformat()
                        )
                    )
                    imported_trades += 1

                    # Remove from entry_map (to track which entries don't have exits)
                    if entry_date in entry_map:
                        del entry_map[entry_date]

                # Check for open position (entries without matching exits)
                if entry_map:
                    # Get the most recent unmatched entry (should be the open position)
                    remaining_entries = sorted(entry_map.values(),
                                               key=lambda e: e['date'], reverse=True)
                    last_entry = remaining_entries[0]
                    entry_date = normalize_timestamp(last_entry['date'])
                    entry_price = last_entry['price']

                    # Insert as open trade
                    trade_id = conn.execute(
                        """INSERT INTO trades
                           (strategy_name, ticker, entry_date, entry_price, position_type, created_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            self.strategy_name,
                            ticker,
                            entry_date,
                            entry_price,
                            'long',
                            datetime.now().isoformat()
                        )
                    ).lastrowid

                    # Insert position record
                    conn.execute(
                        """INSERT OR REPLACE INTO positions
                           (strategy_name, ticker, position_type, entry_price, entry_date, trade_id, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            self.strategy_name,
                            ticker,
                            'long',
                            entry_price,
                            entry_date,
                            trade_id,
                            datetime.now().isoformat()
                        )
                    )

                    open_position = {
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'trade_id': trade_id
                    }
                    print(f"   ✓ Open position: LONG @ ${entry_price:,.2f}")

            # Recalculate stats
            stats = self.recalculate_stats()

            print(f"   ✓ Imported {imported_trades} trades, {stats['win_rate']:.1f}% win rate")

            # Set last_processed_bar to prevent re-processing historical bars
            # This is CRITICAL for forward-only trading
            if last_bar_date:
                last_bar_normalized = normalize_timestamp(last_bar_date)
                self._db.set_last_processed_bar(last_bar_normalized)
                print(f"   ✓ Set last_processed_bar to {last_bar_normalized[:10]}")
            else:
                # Fallback: use the most recent trade date as last_processed
                all_dates = [e['date'] for e in entries] + [x['date'] for x in exits]
                if all_dates:
                    latest = max(all_dates)
                    latest_normalized = normalize_timestamp(latest)
                    self._db.set_last_processed_bar(latest_normalized)
                    print(f"   ✓ Set last_processed_bar to {latest_normalized[:10]} (from trades)")

            return True, {
                'imported_trades': imported_trades,
                'total_entries': len(entries),
                'total_exits': len(exits),
                'open_position': open_position,
                'stats': stats
            }

        except Exception as e:
            import traceback
            return False, {'error': str(e), 'traceback': traceback.format_exc()}

    def get_recent_stats(self, days: int = 2) -> Dict:
        """
        Get statistics for trades in the recent N trading days.

        Args:
            days: Number of TRADING days to look back (default 2)
                  Trading days = weekdays (Mon-Fri), excludes weekends

        Returns:
            Dict with num_trades, win_rate, total_return, profit_factor
        """
        from datetime import datetime, timedelta
        import pandas as pd

        # Calculate cutoff based on TRADING days (weekdays), not calendar days
        def get_trading_day_cutoff(trading_days: int) -> datetime:
            """Go back N trading days (weekdays only)."""
            now = datetime.utcnow()
            cutoff = now
            days_counted = 0

            while days_counted < trading_days:
                cutoff -= timedelta(days=1)
                # Count only weekdays (Monday=0 through Friday=4)
                if cutoff.weekday() < 5:
                    days_counted += 1

            # Set to start of that trading day
            return cutoff.replace(hour=0, minute=0, second=0, microsecond=0)

        cutoff_dt = get_trading_day_cutoff(days)

        with self._db.connection() as conn:
            # Get all recent completed trades - filter in Python to handle mixed timestamp formats
            rows = conn.execute(
                """SELECT exit_date, pnl_pct FROM trades
                   WHERE strategy_name = ? AND exit_date IS NOT NULL
                   ORDER BY exit_date DESC""",
                (self.strategy_name,)
            ).fetchall()

        # Filter by date using proper datetime comparison (handles mixed formats)
        filtered_rows = []
        for row in rows:
            try:
                exit_dt = pd.to_datetime(row['exit_date'])
                if exit_dt.tzinfo is not None:
                    exit_dt = exit_dt.tz_convert('UTC').tz_localize(None)
                if exit_dt >= cutoff_dt:
                    filtered_rows.append(row)
            except Exception:
                continue
        rows = filtered_rows

        if not rows:
            return {
                'num_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'profit_factor': 0.0
            }

        pnls = [r['pnl_pct'] or 0 for r in rows]
        num_trades = len(pnls)
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        win_rate = (len(winners) / num_trades * 100) if num_trades > 0 else 0
        total_return = sum(pnls)

        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        return {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'profit_factor': profit_factor if profit_factor != float('inf') else 999.9
        }

    def get_trade_log_csv(self, limit: int = 10, local_tz: str = 'America/Chicago') -> str:
        """
        Generate a CSV trade log with the last N trades.

        Format:
        #,Entry Time,Entry $,Exit Time,Exit $,P&L,Result

        Times are shown in both UTC and local timezone.

        Args:
            limit: Number of trades to include (default 10)
            local_tz: Local timezone for display (default CT for futures)

        Returns:
            CSV string
        """
        import pytz

        with self._db.connection() as conn:
            rows = conn.execute(
                """SELECT entry_date, entry_price, exit_date, exit_price, pnl_pct
                   FROM trades
                   WHERE strategy_name = ? AND exit_date IS NOT NULL
                   ORDER BY exit_date DESC
                   LIMIT ?""",
                (self.strategy_name, limit)
            ).fetchall()

        if not rows:
            return "#,Entry Time,Entry $,Exit Time,Exit $,P&L,Result\n"

        # Reverse to show oldest first (1 = oldest of the 10)
        rows = list(reversed(rows))

        # Get timezone object
        try:
            local_tz_obj = pytz.timezone(local_tz)
            tz_abbrev = 'CT' if 'Chicago' in local_tz else 'ET' if 'New_York' in local_tz else 'LT'
        except:
            local_tz_obj = pytz.UTC
            tz_abbrev = 'UTC'

        lines = ["#,Entry Time,Entry $,Exit Time,Exit $,P&L,Result"]

        for i, row in enumerate(rows, 1):
            entry_date = row['entry_date']
            exit_date = row['exit_date']
            entry_price = row['entry_price']
            exit_price = row['exit_price']
            pnl = row['pnl_pct'] or 0

            # Format times with both UTC and local
            def format_time(ts_str):
                try:
                    dt = pd.to_datetime(ts_str)
                    if dt.tzinfo is None:
                        dt = dt.tz_localize('UTC')
                    else:
                        dt = dt.tz_convert('UTC')
                    utc_str = dt.strftime('%Y-%m-%d %H:%M UTC')

                    local_dt = dt.tz_convert(local_tz_obj)
                    local_str = local_dt.strftime(f'%I:%M %p {tz_abbrev}')

                    return f"{utc_str} ({local_str})"
                except:
                    return ts_str[:19] if ts_str else 'N/A'

            entry_time_str = format_time(entry_date)
            exit_time_str = format_time(exit_date)

            result = "WIN" if pnl > 0 else "LOSS"
            pnl_str = f"+{pnl:.2f}%" if pnl > 0 else f"{pnl:.2f}%"

            # Format prices
            entry_price_str = f"${entry_price:,.2f}"
            exit_price_str = f"${exit_price:,.2f}"

            lines.append(f"{i},{entry_time_str},{entry_price_str},{exit_time_str},{exit_price_str},{pnl_str},{result}")

        return "\n".join(lines)

    def get_max_drawdown(self, days: int = None) -> float:
        """
        Calculate maximum drawdown percentage from trade history.

        Args:
            days: If provided, only consider trades from the last N days

        Returns:
            Max drawdown as a positive percentage (e.g., 5.2 for 5.2% drawdown)
        """
        with self._db.connection() as conn:
            if days:
                cutoff = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
                rows = conn.execute(
                    """SELECT pnl_pct FROM trades
                       WHERE strategy_name = ? AND exit_date IS NOT NULL
                       AND exit_date >= ?
                       ORDER BY exit_date ASC""",
                    (self.strategy_name, cutoff)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT pnl_pct FROM trades
                       WHERE strategy_name = ? AND exit_date IS NOT NULL
                       ORDER BY exit_date ASC""",
                    (self.strategy_name,)
                ).fetchall()

        if not rows:
            return 0.0

        # Build equity curve
        equity = [100000.0]  # Starting capital
        for row in rows:
            pnl = row['pnl_pct'] or 0
            equity.append(equity[-1] * (1 + pnl / 100))

        # Calculate max drawdown
        peak = equity[0]
        max_dd = 0.0

        for value in equity:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100 if peak > 0 else 0
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    def get_enhanced_stats(self, ticker: str, config: Dict, current_price: float = None) -> Dict:
        """
        Get enhanced statistics for Discord notifications.

        Returns a dict with all info needed for the enhanced format:
        - strategy_name, ticker, signal_type
        - current_price, sl_price, tp_price
        - all_time stats (including max_drawdown)
        - recent_2day stats (including max_drawdown)
        - position info
        - trade_log_csv

        Args:
            ticker: Trading symbol
            config: Strategy config with signal_type, stop_loss_pct, take_profit_pct
            current_price: Current market price (optional)

        Returns:
            Dict with all enhanced stats
        """
        # Get all-time stats
        all_stats = self.get_stats()
        if not all_stats:
            all_stats = {'num_trades': 0, 'win_rate': 0, 'total_return': 0, 'profit_factor': 0}

        # Get recent 2-day stats
        recent_stats = self.get_recent_stats(days=2)

        # Get max drawdown (all-time and recent)
        all_time_dd = self.get_max_drawdown()
        recent_dd = self.get_max_drawdown(days=2)

        # Get current position
        position = self.get_current_position()

        # Calculate SL/TP prices
        sl_pct = config.get('stop_loss_pct', 2.0)
        tp_pct = config.get('take_profit_pct', 5.0)

        if current_price:
            sl_price = current_price * (1 - sl_pct / 100)
            tp_price = current_price * (1 + tp_pct / 100)
        else:
            sl_price = None
            tp_price = None

        # Determine local timezone based on ticker
        if ticker in ['ES=F', 'GC=F', 'NQ=F', 'CL=F']:
            local_tz = 'America/Chicago'
        elif ticker in ['BTC-USD', 'ETH-USD']:
            local_tz = 'UTC'
        else:
            local_tz = 'America/New_York'

        # Get trade log CSV
        trade_log = self.get_trade_log_csv(limit=10, local_tz=local_tz)

        return {
            'strategy_name': self.strategy_name,
            'ticker': ticker,
            'signal_type': config.get('signal_type', 'unknown'),
            'current_price': current_price,
            'sl_pct': sl_pct,
            'tp_pct': tp_pct,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'all_time': {
                'num_trades': all_stats.get('num_trades', 0),
                'win_rate': all_stats.get('win_rate', 0),
                'total_return': all_stats.get('total_return', 0),
                'profit_factor': all_stats.get('profit_factor', 0),
                'max_drawdown': all_time_dd
            },
            'recent_2day': {
                'num_trades': recent_stats.get('num_trades', 0),
                'win_rate': recent_stats.get('win_rate', 0),
                'total_return': recent_stats.get('total_return', 0),
                'profit_factor': recent_stats.get('profit_factor', 0),
                'max_drawdown': recent_dd
            },
            'position': {
                'is_long': position is not None,
                'entry_price': position.entry_price if position else None,
                'entry_date': position.entry_date if position else None
            } if position else None,
            'trade_log_csv': trade_log
        }
