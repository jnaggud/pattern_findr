"""
Shared pytest fixtures for Pattern_FindR trading system tests.

These fixtures provide:
- Temporary databases for isolated testing
- Sample OHLC data generation
- Mock position managers
- Test utilities
"""

import os
import sys
import pytest
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Generator, Dict, Any

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from velocity_trading.core.database import TradingDatabase, init_database
from velocity_trading.core.position_manager import PositionManager


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """
    Create a temporary database file path.

    Yields:
        Path to temporary .db file (cleaned up after test)
    """
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def temp_database(temp_db_path: str) -> Generator[TradingDatabase, None, None]:
    """
    Create a TradingDatabase with temporary storage.

    Yields:
        Initialized TradingDatabase instance
    """
    db = TradingDatabase("test_strategy", db_path=temp_db_path)
    yield db


@pytest.fixture
def mock_position_manager(temp_db_path: str) -> Generator[PositionManager, None, None]:
    """
    Create a PositionManager with temporary database.

    Yields:
        PositionManager instance for testing
    """
    pm = PositionManager("test_strategy", db_path=temp_db_path)
    yield pm


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlc_data() -> pd.DataFrame:
    """
    Generate sample OHLC data for testing.

    Returns:
        DataFrame with 100 15-minute bars of synthetic price data
    """
    np.random.seed(42)  # Reproducible randomness

    n_bars = 100
    dates = pd.date_range('2024-01-02 09:30', periods=n_bars, freq='15min')

    # Generate realistic price movement
    base_price = 500.0
    returns = np.random.normal(0, 0.002, n_bars)  # 0.2% std per bar
    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close prices
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Intrabar volatility
        range_pct = abs(np.random.normal(0, 0.003))
        high = close * (1 + range_pct / 2)
        low = close * (1 - range_pct / 2)

        # Open is previous close (or random for first bar)
        if i == 0:
            open_price = close * (1 + np.random.normal(0, 0.001))
        else:
            open_price = prices[i - 1]

        # Ensure OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        data.append({
            'timestamp': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': int(np.random.uniform(10000, 100000))
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


@pytest.fixture
def sample_ohlc_data_daily() -> pd.DataFrame:
    """
    Generate sample daily OHLC data for testing.

    Returns:
        DataFrame with 252 daily bars (1 trading year)
    """
    np.random.seed(42)

    n_bars = 252
    dates = pd.bdate_range('2024-01-02', periods=n_bars)

    base_price = 500.0
    returns = np.random.normal(0.0003, 0.01, n_bars)  # ~7.5% annual return
    prices = base_price * np.cumprod(1 + returns)

    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        range_pct = abs(np.random.normal(0.01, 0.005))
        high = close * (1 + range_pct / 2)
        low = close * (1 - range_pct / 2)
        open_price = prices[i - 1] if i > 0 else close * 0.999

        high = max(high, open_price, close)
        low = min(low, open_price, close)

        data.append({
            'timestamp': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': int(np.random.uniform(1e6, 1e7))
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


# =============================================================================
# Trade Fixtures
# =============================================================================

@pytest.fixture
def sample_trade() -> Dict[str, Any]:
    """
    Generate a sample completed trade record.

    Returns:
        Dict with trade data matching database schema
    """
    return {
        'strategy_name': 'test_strategy',
        'ticker': 'ES=F',
        'entry_date': '2024-01-02T10:00:00',
        'entry_price': 500.00,
        'entry_signal_bar': '2024-01-02T09:45:00',
        'position_type': 'long',
        'exit_date': '2024-01-02T14:30:00',
        'exit_price': 505.00,
        'exit_signal_bar': '2024-01-02T14:15:00',
        'exit_reason': 'signal',
        'pnl_pct': 1.0,
        'pnl_dollars': 250.0,
        'is_missed': False
    }


@pytest.fixture
def sample_open_position() -> Dict[str, Any]:
    """
    Generate a sample open position record.

    Returns:
        Dict with position data matching database schema
    """
    return {
        'strategy_name': 'test_strategy',
        'ticker': 'ES=F',
        'position_type': 'long',
        'entry_price': 500.00,
        'entry_date': '2024-01-02T10:00:00',
        'entry_signal_bar': '2024-01-02T09:45:00',
        'last_signal_time': '2024-01-02T10:00:00'
    }


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def sample_strategy_config() -> Dict[str, Any]:
    """
    Generate a sample strategy configuration.

    Returns:
        Dict matching velocity_config.json structure
    """
    return {
        'ticker': 'ES=F',
        'interval': '15m',
        'strategy_type': 'velocity_crossover_or_zone',
        'stop_loss_pct': 0.72,
        'take_profit_pct': None,
        'lookback_bars': 200,
        'bar_completion_buffer': 3,
        'position_check_interval': 3,
        'entry_rules': {
            'min_velocity': 0.5,
            'confirm_bars': 1
        },
        'exit_rules': {
            'use_trailing_stop': False,
            'exit_on_reverse_signal': True
        }
    }


@pytest.fixture
def sample_daily_config() -> Dict[str, Any]:
    """
    Generate a sample daily strategy configuration.

    Returns:
        Dict matching velocity_config.json structure for daily strategies
    """
    return {
        'ticker': 'SPY',
        'interval': '1d',
        'strategy_type': 'velocity_daily',
        'stop_loss_pct': 2.0,
        'take_profit_pct': 5.0,
        'lookback_bars': 100,
        'entry_rules': {
            'min_velocity': 0.3
        },
        'exit_rules': {
            'exit_on_reverse_signal': True
        }
    }


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def now() -> datetime:
    """Current datetime for tests."""
    return datetime.now()


@pytest.fixture
def market_open_time() -> datetime:
    """A datetime during regular market hours (10:00 AM ET)."""
    return datetime(2024, 1, 2, 10, 0, 0)


@pytest.fixture
def market_close_time() -> datetime:
    """A datetime just after market close (4:00 PM ET)."""
    return datetime(2024, 1, 2, 16, 0, 0)


@pytest.fixture
def weekend_time() -> datetime:
    """A datetime on a weekend (Saturday)."""
    return datetime(2024, 1, 6, 12, 0, 0)


# =============================================================================
# Assertion Helpers
# =============================================================================

def assert_valid_ohlc(df: pd.DataFrame) -> None:
    """
    Assert that OHLC data is valid.

    Checks:
    - Low <= Open <= High
    - Low <= Close <= High
    - Low <= High
    - No NaN values
    """
    assert not df.isnull().any().any(), "OHLC data contains NaN values"
    assert (df['Low'] <= df['Open']).all(), "Low > Open in some bars"
    assert (df['Open'] <= df['High']).all(), "Open > High in some bars"
    assert (df['Low'] <= df['Close']).all(), "Low > Close in some bars"
    assert (df['Close'] <= df['High']).all(), "Close > High in some bars"
    assert (df['Low'] <= df['High']).all(), "Low > High in some bars"


def assert_no_same_bar_trades(trades: list) -> None:
    """
    Assert that no trades have same-bar entry and exit.

    Args:
        trades: List of trade dicts with entry_date and exit_date
    """
    for trade in trades:
        if trade.get('exit_date'):
            entry_bar = trade['entry_date'][:16]  # Truncate to minute
            exit_bar = trade['exit_date'][:16]
            assert entry_bar != exit_bar, \
                f"Same-bar trade found: entry={entry_bar}, exit={exit_bar}"


def assert_no_overlapping_trades(trades: list) -> None:
    """
    Assert that no trades overlap (entry before previous exit).

    Args:
        trades: List of trade dicts sorted by entry_date
    """
    sorted_trades = sorted(trades, key=lambda t: t['entry_date'])

    for i in range(1, len(sorted_trades)):
        prev = sorted_trades[i - 1]
        curr = sorted_trades[i]

        if prev.get('exit_date'):
            assert curr['entry_date'] >= prev['exit_date'], \
                f"Overlapping trades: {prev['entry_date']} - {prev['exit_date']} " \
                f"overlaps with {curr['entry_date']}"


# Export helpers for use in tests
__all__ = [
    'assert_valid_ohlc',
    'assert_no_same_bar_trades',
    'assert_no_overlapping_trades'
]
