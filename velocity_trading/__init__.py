"""
Velocity Trading System - Modular trading package.

This package provides a modular, SQLite-backed trading system that runs
in parallel with the legacy velocity_live_trader.py.

Key components:
- PositionManager: Atomic state transitions with SQLite
- DailyTrader: Trading on daily bars (SPY, BTC daily)
- IntradayTrader: Trading on intraday bars (ES=F 15m, etc.)

Usage:
    from velocity_trading import PositionManager, DailyTrader, IntradayTrader

    # Or import specific modules
    from velocity_trading.core import TradingDatabase
    from velocity_trading.data import fetch_price_data
"""

__version__ = "1.0.0"

# Core components
from .core import TradingDatabase, PositionManager, Position

# Traders
from .traders import DailyTrader, IntradayTrader, run_daily_trader, run_intraday_trader

# Migration
from .migration import migrate_all, migrate_strategy

__all__ = [
    # Core
    'TradingDatabase',
    'PositionManager',
    'Position',
    # Traders
    'DailyTrader',
    'IntradayTrader',
    'run_daily_trader',
    'run_intraday_trader',
    # Migration
    'migrate_all',
    'migrate_strategy',
]
