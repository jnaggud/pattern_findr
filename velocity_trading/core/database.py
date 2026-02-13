"""
SQLite database management for velocity trading.

Provides schema creation, connection management, and utilities
for the trading database.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

# Database directory (relative to package)
DB_DIR = Path(__file__).parent.parent / "db"


def get_db_path(strategy_name: str) -> str:
    """Get the database path for a strategy."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    # Sanitize strategy name for filesystem
    safe_name = strategy_name.replace("/", "_").replace("\\", "_")
    return str(DB_DIR / f"{safe_name}.db")


def init_database(db_path: str) -> None:
    """
    Initialize database with schema.

    Creates all tables if they don't exist. Safe to call multiple times.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # trades: Complete trade lifecycle (append-only for history)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            ticker TEXT NOT NULL,

            -- Entry details
            entry_date TEXT NOT NULL,
            entry_price REAL NOT NULL,
            entry_signal_bar TEXT,
            position_type TEXT DEFAULT 'long',

            -- Exit details (NULL if position still open)
            exit_date TEXT,
            exit_price REAL,
            exit_signal_bar TEXT,
            exit_reason TEXT,
            pnl_pct REAL,
            pnl_dollars REAL,

            -- Regime (v9)
            entry_regime TEXT,

            -- Metadata
            is_missed BOOLEAN DEFAULT FALSE,
            created_at TEXT DEFAULT (datetime('now')),

            CHECK (position_type IN ('long', 'short'))
        )
    """)

    # Indexes for performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_strategy
        ON trades(strategy_name)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_entry_date
        ON trades(entry_date)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_trades_open
        ON trades(strategy_name, exit_date)
        WHERE exit_date IS NULL
    """)

    # positions: Current open position (0 or 1 row per strategy)
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
            entry_regime TEXT,
            trade_id INTEGER REFERENCES trades(id),
            updated_at TEXT DEFAULT (datetime('now')),

            CHECK (position_type IN ('long', 'short'))
        )
    """)

    # daily_alerts: Deduplication for scheduled notifications
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            alert_key TEXT NOT NULL,
            sent_at TEXT DEFAULT (datetime('now')),

            UNIQUE (strategy_name, alert_key)
        )
    """)

    # strategy_stats: Cached statistics
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
            max_drawdown REAL DEFAULT 0.0,
            num_missed INTEGER DEFAULT 0,
            missed_return REAL DEFAULT 0.0,
            last_updated TEXT DEFAULT (datetime('now'))
        )
    """)

    # strategy_state: Track last processed bar to prevent repainting
    # This ensures each bar is evaluated exactly once, ever
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strategy_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT UNIQUE NOT NULL,
            last_processed_bar TEXT,
            last_check_time TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # schema_version: For future migrations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # Insert initial schema version if not exists
    cursor.execute("""
        INSERT OR IGNORE INTO schema_version (version) VALUES (1)
    """)

    # Migration: Add entry_regime column (v9 regime-aware trading)
    for table in ('trades', 'positions'):
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN entry_regime TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

    conn.commit()
    conn.close()


class TradingDatabase:
    """
    Database connection manager with context management.

    Usage:
        db = TradingDatabase(strategy_name)
        with db.connection() as conn:
            cursor = conn.execute("SELECT * FROM trades")
            rows = cursor.fetchall()
    """

    def __init__(self, strategy_name: str, db_path: str = None):
        """
        Initialize database for a strategy.

        Args:
            strategy_name: Name of the trading strategy
            db_path: Optional explicit path (default: auto-generated)
        """
        self.strategy_name = strategy_name
        self.db_path = db_path or get_db_path(strategy_name)

        # Ensure database is initialized
        init_database(self.db_path)

    @contextmanager
    def connection(self):
        """
        Context manager for database connections with automatic commit/rollback.

        Usage:
            with db.connection() as conn:
                conn.execute("INSERT INTO ...")
                # Commits automatically on success
                # Rolls back on exception
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def transaction(self):
        """
        Explicit transaction context for multi-statement operations.

        Same as connection() but makes intent clearer.
        """
        with self.connection() as conn:
            yield conn

    def execute(self, sql: str, params: tuple = ()) -> list:
        """
        Execute a query and return all results.

        For simple queries that don't need transaction control.
        """
        with self.connection() as conn:
            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def execute_one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        """
        Execute a query and return first result, or None.
        """
        results = self.execute(sql, params)
        return results[0] if results else None

    def insert(self, table: str, data: dict) -> int:
        """
        Insert a row and return the new row id.
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        with self.connection() as conn:
            cursor = conn.execute(sql, tuple(data.values()))
            return cursor.lastrowid

    def update(self, table: str, data: dict, where: str, where_params: tuple) -> int:
        """
        Update rows and return number affected.
        """
        set_clause = ", ".join(f"{k} = ?" for k in data.keys())
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        params = tuple(data.values()) + where_params

        with self.connection() as conn:
            cursor = conn.execute(sql, params)
            return cursor.rowcount

    # =========================================================================
    # Daily Alerts Management
    # =========================================================================

    def has_alert(self, alert_key: str) -> bool:
        """Check if an alert has already been sent."""
        result = self.execute_one(
            "SELECT id FROM daily_alerts WHERE strategy_name = ? AND alert_key = ?",
            (self.strategy_name, alert_key)
        )
        return result is not None

    def add_alert(self, alert_key: str) -> bool:
        """
        Add an alert key to prevent duplicate sends.

        Returns True if added, False if already exists.
        """
        try:
            self.insert("daily_alerts", {
                "strategy_name": self.strategy_name,
                "alert_key": alert_key
            })
            return True
        except sqlite3.IntegrityError:
            return False  # Already exists

    def clear_old_alerts(self, days_to_keep: int = 7) -> int:
        """
        Clear alerts older than specified days.

        Returns number of alerts deleted.
        """
        cutoff = datetime.now().strftime("%Y-%m-%d")
        with self.connection() as conn:
            cursor = conn.execute(
                """DELETE FROM daily_alerts
                   WHERE strategy_name = ?
                   AND date(sent_at) < date(?, '-' || ? || ' days')""",
                (self.strategy_name, cutoff, days_to_keep)
            )
            return cursor.rowcount

    # =========================================================================
    # Strategy Stats
    # =========================================================================

    def get_stats(self) -> Optional[dict]:
        """Get cached strategy statistics."""
        return self.execute_one(
            "SELECT * FROM strategy_stats WHERE strategy_name = ?",
            (self.strategy_name,)
        )

    def update_stats(self, stats: dict) -> None:
        """Update cached strategy statistics."""
        stats["strategy_name"] = self.strategy_name
        stats["last_updated"] = datetime.now().isoformat()

        with self.connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO strategy_stats
                   (strategy_name, num_trades, num_wins, win_rate, total_return,
                    profit_factor, avg_win, avg_loss, max_drawdown,
                    num_missed, missed_return, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    stats.get("strategy_name"),
                    stats.get("num_trades", 0),
                    stats.get("num_wins", 0),
                    stats.get("win_rate", 0.0),
                    stats.get("total_return", 0.0),
                    stats.get("profit_factor", 0.0),
                    stats.get("avg_win", 0.0),
                    stats.get("avg_loss", 0.0),
                    stats.get("max_drawdown", 0.0),
                    stats.get("num_missed", 0),
                    stats.get("missed_return", 0.0),
                    stats.get("last_updated")
                )
            )

    # =========================================================================
    # Strategy State (Last Processed Bar Tracking)
    # =========================================================================

    def get_last_processed_bar(self) -> Optional[str]:
        """
        Get the timestamp of the last bar that was evaluated for signals.

        This is CRITICAL for forward-only trading - ensures each bar
        is evaluated exactly once, preventing repainting.

        Returns:
            ISO timestamp string of last processed bar, or None if never processed
        """
        result = self.execute_one(
            "SELECT last_processed_bar FROM strategy_state WHERE strategy_name = ?",
            (self.strategy_name,)
        )
        return result["last_processed_bar"] if result else None

    def set_last_processed_bar(self, bar_timestamp: str) -> None:
        """
        Record that a bar has been evaluated for signals.

        Once set, this bar will NEVER be re-evaluated, ensuring
        no repainting of historical signals.

        Args:
            bar_timestamp: ISO timestamp of the bar that was just processed
        """
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO strategy_state (strategy_name, last_processed_bar, updated_at)
                   VALUES (?, ?, datetime('now'))
                   ON CONFLICT(strategy_name) DO UPDATE SET
                   last_processed_bar = excluded.last_processed_bar,
                   updated_at = datetime('now')""",
                (self.strategy_name, bar_timestamp)
            )

    def get_strategy_state(self) -> Optional[dict]:
        """Get full strategy state including last processed bar."""
        return self.execute_one(
            "SELECT * FROM strategy_state WHERE strategy_name = ?",
            (self.strategy_name,)
        )
