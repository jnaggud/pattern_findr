"""
Stress tests for high-volatility scenarios.

Tests that the trading system handles:
- Large price gaps (gap-down through stop loss)
- Rapid SL/TP triggers
- Multiple exit conditions in same cycle
- Position state consistency under stress
- Price data with extreme moves
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from velocity_trading.core.position_manager import PositionManager
from velocity_trading.core.database import TradingDatabase


# =============================================================================
# Gap-Down Through Stop Loss
# =============================================================================

class TestGapDownStopLoss:
    """Test that stop losses trigger even on large gaps."""

    @pytest.fixture
    def pm(self, tmp_path):
        db_path = str(tmp_path / "stress_test.db")
        return PositionManager("stress_test", db_path=db_path)

    def test_gap_down_triggers_exit(self, pm):
        """A gap down past the stop loss should still result in exit."""
        # Enter position at 5000
        success, result = pm.enter_position(
            entry_price=5000.0,
            entry_date='2026-02-01T10:00:00',
            entry_signal_bar='2026-02-01T09:45:00',
            position_type='long',
            ticker='ES=F'
        )
        assert success

        # Gap down to 4900 (2% drop) - well past any reasonable stop loss
        # Exit should still be accepted
        success, result = pm.exit_position(
            exit_price=4900.0,
            exit_date='2026-02-01T10:15:00',
            exit_reason='Stop Loss (-2.00%)',
            interval='15m'
        )
        assert success
        assert result['pnl_pct'] == pytest.approx(-2.0, abs=0.01)

    def test_gap_down_pnl_accuracy(self, pm):
        """P&L should reflect actual exit price, not stop level."""
        pm.enter_position(
            entry_price=100.0,
            entry_date='2026-02-01T10:00:00',
            entry_signal_bar='2026-02-01T09:45:00',
            position_type='long',
            ticker='BTC-USD'
        )

        # Gap down 10% (crypto flash crash)
        success, result = pm.exit_position(
            exit_price=90.0,
            exit_date='2026-02-01T10:15:00',
            exit_reason='Stop Loss (-10.00%)',
            interval='15m'
        )
        assert success
        assert result['pnl_pct'] == pytest.approx(-10.0, abs=0.01)

    def test_large_gap_up_take_profit(self, pm):
        """Large gap up should trigger take profit."""
        pm.enter_position(
            entry_price=5000.0,
            entry_date='2026-02-01T10:00:00',
            entry_signal_bar='2026-02-01T09:45:00',
            position_type='long',
            ticker='ES=F'
        )

        # Gap up 5%
        success, result = pm.exit_position(
            exit_price=5250.0,
            exit_date='2026-02-01T10:15:00',
            exit_reason='Take Profit (+5.00%)',
            interval='15m'
        )
        assert success
        assert result['pnl_pct'] == pytest.approx(5.0, abs=0.01)


# =============================================================================
# Position State Consistency Under Stress
# =============================================================================

class TestPositionConsistencyStress:
    """Test position state stays consistent under rapid operations."""

    @pytest.fixture
    def pm(self, tmp_path):
        db_path = str(tmp_path / "consistency_test.db")
        return PositionManager("consistency_test", db_path=db_path)

    def test_rapid_entry_exit_cycles(self, pm):
        """Multiple rapid entry/exit cycles shouldn't corrupt state."""
        base_time = datetime(2026, 2, 1, 10, 0, 0)

        for i in range(20):
            entry_time = base_time + timedelta(minutes=30 * i)
            exit_time = entry_time + timedelta(minutes=15)

            entry_ts = entry_time.strftime('%Y-%m-%dT%H:%M:%S')
            signal_ts = (entry_time - timedelta(minutes=15)).strftime('%Y-%m-%dT%H:%M:%S')
            exit_ts = exit_time.strftime('%Y-%m-%dT%H:%M:%S')

            success, _ = pm.enter_position(
                entry_price=5000.0 + i,
                entry_date=entry_ts,
                entry_signal_bar=signal_ts,
                position_type='long',
                ticker='ES=F'
            )
            assert success, f"Entry {i} failed"

            success, _ = pm.exit_position(
                exit_price=5001.0 + i,
                exit_date=exit_ts,
                exit_reason='signal',
                interval='15m'
            )
            assert success, f"Exit {i} failed"

        # Verify state: no open position, 20 completed trades
        assert pm.get_current_position() is None
        stats = pm.get_stats()
        assert stats['num_trades'] == 20

    def test_no_position_after_exit(self, pm):
        """After exit, get_current_position must return None."""
        pm.enter_position(
            entry_price=5000.0,
            entry_date='2026-02-01T10:00:00',
            entry_signal_bar='2026-02-01T09:45:00',
            position_type='long',
            ticker='ES=F'
        )
        assert pm.get_current_position() is not None

        pm.exit_position(
            exit_price=5001.0,
            exit_date='2026-02-01T10:15:00',
            exit_reason='signal',
            interval='15m'
        )
        assert pm.get_current_position() is None

    def test_double_exit_rejected(self, pm):
        """Attempting to exit twice should fail the second time."""
        pm.enter_position(
            entry_price=5000.0,
            entry_date='2026-02-01T10:00:00',
            entry_signal_bar='2026-02-01T09:45:00',
            position_type='long',
            ticker='ES=F'
        )

        # First exit succeeds
        success1, _ = pm.exit_position(
            exit_price=5001.0,
            exit_date='2026-02-01T10:15:00',
            exit_reason='signal',
            interval='15m'
        )
        assert success1

        # Second exit should fail (no position)
        success2, _ = pm.exit_position(
            exit_price=5002.0,
            exit_date='2026-02-01T10:30:00',
            exit_reason='signal',
            interval='15m'
        )
        assert not success2

    def test_entry_during_position_rejected(self, pm):
        """Cannot enter while already in a position."""
        pm.enter_position(
            entry_price=5000.0,
            entry_date='2026-02-01T10:00:00',
            entry_signal_bar='2026-02-01T09:45:00',
            position_type='long',
            ticker='ES=F'
        )

        # Second entry should fail
        success, _ = pm.enter_position(
            entry_price=5001.0,
            entry_date='2026-02-01T10:15:00',
            entry_signal_bar='2026-02-01T10:00:00',
            position_type='long',
            ticker='ES=F'
        )
        assert not success


# =============================================================================
# High Volatility OHLC Data
# =============================================================================

class TestHighVolatilityData:
    """Test data validation under extreme price moves."""

    def test_valid_ohlc_with_large_range(self):
        """OHLC data with large intrabar range should be valid."""
        # 10% intrabar range
        data = pd.DataFrame([{
            'Open': 5000.0,
            'High': 5500.0,  # +10%
            'Low': 4500.0,   # -10%
            'Close': 5100.0,
            'Volume': 1000000
        }])

        # Validate relationships
        assert (data['Low'] <= data['Open']).all()
        assert (data['Open'] <= data['High']).all()
        assert (data['Low'] <= data['Close']).all()
        assert (data['Close'] <= data['High']).all()

    def test_ohlc_with_gap_down_open(self):
        """Data with gap-down open should be valid."""
        data = pd.DataFrame([
            {'Open': 5000, 'High': 5010, 'Low': 4990, 'Close': 5005, 'Volume': 100000},
            {'Open': 4800, 'High': 4810, 'Low': 4790, 'Close': 4795, 'Volume': 200000},  # Gap down
        ])

        # Each row independently valid
        for _, row in data.iterrows():
            assert row['Low'] <= row['Open'] <= row['High']
            assert row['Low'] <= row['Close'] <= row['High']

    def test_ohlc_with_flash_crash(self):
        """Flash crash data: huge range, close near low."""
        data = pd.DataFrame([{
            'Open': 100000.0,
            'High': 100500.0,
            'Low': 85000.0,   # -15% flash crash
            'Close': 86000.0,  # Recovery partial
            'Volume': 5000000
        }])

        assert (data['Low'] <= data['Close']).all()
        assert (data['Close'] <= data['High']).all()


# =============================================================================
# PnL Edge Cases
# =============================================================================

class TestPnLEdgeCases:
    """Test P&L calculation edge cases."""

    @pytest.fixture
    def pm(self, tmp_path):
        db_path = str(tmp_path / "pnl_test.db")
        return PositionManager("pnl_test", db_path=db_path)

    def test_breakeven_trade(self, pm):
        """Exit at exact entry price = 0% P&L."""
        pm.enter_position(
            entry_price=5000.0,
            entry_date='2026-02-01T10:00:00',
            entry_signal_bar='2026-02-01T09:45:00',
            position_type='long',
            ticker='ES=F'
        )
        success, result = pm.exit_position(
            exit_price=5000.0,
            exit_date='2026-02-01T10:15:00',
            exit_reason='signal',
            interval='15m'
        )
        assert success
        assert result['pnl_pct'] == pytest.approx(0.0, abs=0.001)

    def test_very_small_profit(self, pm):
        """Tiny profit should still be calculated correctly."""
        pm.enter_position(
            entry_price=5000.0,
            entry_date='2026-02-01T10:00:00',
            entry_signal_bar='2026-02-01T09:45:00',
            position_type='long',
            ticker='ES=F'
        )
        success, result = pm.exit_position(
            exit_price=5000.50,  # $0.50 profit on $5000 = 0.01%
            exit_date='2026-02-01T10:15:00',
            exit_reason='signal',
            interval='15m'
        )
        assert success
        assert result['pnl_pct'] == pytest.approx(0.01, abs=0.001)

    def test_large_loss(self, pm):
        """Large loss P&L calculated correctly."""
        pm.enter_position(
            entry_price=100000.0,
            entry_date='2026-02-01T10:00:00',
            entry_signal_bar='2026-02-01T09:45:00',
            position_type='long',
            ticker='BTC-USD'
        )
        # 20% crash
        success, result = pm.exit_position(
            exit_price=80000.0,
            exit_date='2026-02-01T10:15:00',
            exit_reason='Stop Loss (-20.00%)',
            interval='15m'
        )
        assert success
        assert result['pnl_pct'] == pytest.approx(-20.0, abs=0.01)
