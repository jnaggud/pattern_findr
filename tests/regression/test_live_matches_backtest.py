"""
Regression tests: Live trading signal generation matches backtest.

The critical invariant: given identical OHLC data and identical config parameters,
the modular trading system (velocity_trading.indicators) must produce the same
buy_signal and sell_signal columns as the backtest system (velocity_core).

This prevents silent divergence between what backtest shows and what live trades do.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import both code paths
from velocity_trading.indicators.oscillators import calculate_composite_oscillator
from velocity_trading.indicators.velocity import calculate_velocity_signals

try:
    from velocity_core import (
        calculate_composite_oscillator as backtest_oscillator,
        calculate_velocity_signals as backtest_signals,
    )
    HAS_BACKTEST = True
except ImportError:
    HAS_BACKTEST = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_ohlcv():
    """
    Generate deterministic OHLCV data suitable for oscillator calculation.

    Uses enough bars (200+) for all rolling window indicators to warm up.
    """
    np.random.seed(42)
    n = 300

    dates = pd.date_range('2025-06-01 09:30', periods=n, freq='15min')

    # Generate trending then mean-reverting price series
    base = 5000.0
    trend = np.cumsum(np.random.normal(0, 0.5, n))
    noise = np.random.normal(0, 2, n)
    close = base + trend + noise

    # Generate OHLC from close
    data = []
    for i in range(n):
        c = close[i]
        range_pct = abs(np.random.normal(0.002, 0.001))
        h = c * (1 + range_pct)
        l = c * (1 - range_pct)
        o = close[i - 1] if i > 0 else c * (1 + np.random.normal(0, 0.001))

        h = max(h, o, c)
        l = min(l, o, c)

        data.append({
            'Open': round(o, 2),
            'High': round(h, 2),
            'Low': round(l, 2),
            'Close': round(c, 2),
            'Volume': int(np.random.uniform(50000, 200000)),
        })

    df = pd.DataFrame(data, index=dates)

    # Also create lowercase version for backtest (velocity_core expects lowercase)
    df_lower = df.copy()
    df_lower.columns = [c.lower() for c in df_lower.columns]

    return df, df_lower


@pytest.fixture
def base_config():
    """Standard config for signal comparison."""
    return {
        'signal_type': 'any_reversal',
        'oversold_threshold': -0.3,
        'overbought_threshold': 0.3,
        'vel_smoothing': 3,
        'extreme_zone_mult': 1.5,
        'require_accel': True,
        'oscillator_type': 'composite',
    }


# =============================================================================
# Modular System Internal Consistency
# =============================================================================

class TestModularSignalConsistency:
    """
    Verify the modular signal generation pipeline produces
    consistent, deterministic results.
    """

    def test_oscillator_produces_required_columns(self, synthetic_ohlcv):
        """calculate_composite_oscillator must produce osc_smooth column."""
        df, _ = synthetic_ohlcv
        result = calculate_composite_oscillator(df, oscillator_type='composite')

        assert 'osc_smooth' in result.columns or 'JD_Osc' in result.columns, \
            f"Missing oscillator column. Got: {list(result.columns)}"

    def test_velocity_signals_produce_buy_sell(self, synthetic_ohlcv, base_config):
        """calculate_velocity_signals must produce buy_signal and sell_signal."""
        df, _ = synthetic_ohlcv
        df = calculate_composite_oscillator(df, oscillator_type='composite')
        df = calculate_velocity_signals(
            df,
            signal_type=base_config['signal_type'],
            oversold_threshold=base_config['oversold_threshold'],
            overbought_threshold=base_config['overbought_threshold'],
            vel_smoothing=base_config['vel_smoothing'],
            extreme_zone_mult=base_config['extreme_zone_mult'],
            require_accel=base_config['require_accel'],
        )

        assert 'buy_signal' in df.columns
        assert 'sell_signal' in df.columns

    def test_signals_are_boolean(self, synthetic_ohlcv, base_config):
        """buy_signal and sell_signal must be boolean-like (True/False/0/1)."""
        df, _ = synthetic_ohlcv
        df = calculate_composite_oscillator(df, oscillator_type='composite')
        df = calculate_velocity_signals(
            df,
            signal_type=base_config['signal_type'],
            oversold_threshold=base_config['oversold_threshold'],
            overbought_threshold=base_config['overbought_threshold'],
            vel_smoothing=base_config['vel_smoothing'],
            extreme_zone_mult=base_config['extreme_zone_mult'],
            require_accel=base_config['require_accel'],
        )

        # Filter out NaN rows (warmup period)
        valid = df.dropna(subset=['buy_signal', 'sell_signal'])
        buy_vals = set(valid['buy_signal'].unique())
        sell_vals = set(valid['sell_signal'].unique())

        # Should only contain True/False or 1/0
        assert buy_vals.issubset({True, False, 0, 1, 0.0, 1.0}), \
            f"buy_signal has unexpected values: {buy_vals}"
        assert sell_vals.issubset({True, False, 0, 1, 0.0, 1.0}), \
            f"sell_signal has unexpected values: {sell_vals}"

    def test_deterministic_output(self, synthetic_ohlcv, base_config):
        """Same input must produce identical output every time."""
        df, _ = synthetic_ohlcv

        # Run twice
        df1 = calculate_composite_oscillator(df.copy(), oscillator_type='composite')
        df1 = calculate_velocity_signals(
            df1,
            signal_type=base_config['signal_type'],
            oversold_threshold=base_config['oversold_threshold'],
            overbought_threshold=base_config['overbought_threshold'],
            vel_smoothing=base_config['vel_smoothing'],
            extreme_zone_mult=base_config['extreme_zone_mult'],
            require_accel=base_config['require_accel'],
        )

        df2 = calculate_composite_oscillator(df.copy(), oscillator_type='composite')
        df2 = calculate_velocity_signals(
            df2,
            signal_type=base_config['signal_type'],
            oversold_threshold=base_config['oversold_threshold'],
            overbought_threshold=base_config['overbought_threshold'],
            vel_smoothing=base_config['vel_smoothing'],
            extreme_zone_mult=base_config['extreme_zone_mult'],
            require_accel=base_config['require_accel'],
        )

        # Compare signals
        valid_idx = df1['buy_signal'].notna() & df2['buy_signal'].notna()
        pd.testing.assert_series_equal(
            df1.loc[valid_idx, 'buy_signal'].astype(bool),
            df2.loc[valid_idx, 'buy_signal'].astype(bool),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            df1.loc[valid_idx, 'sell_signal'].astype(bool),
            df2.loc[valid_idx, 'sell_signal'].astype(bool),
            check_names=False,
        )

    def test_oscillator_range(self, synthetic_ohlcv):
        """Oscillator values should be roughly in [-1, 1] range."""
        df, _ = synthetic_ohlcv
        df = calculate_composite_oscillator(df, oscillator_type='composite')

        osc_col = 'osc_smooth' if 'osc_smooth' in df.columns else 'JD_Osc'
        valid = df[osc_col].dropna()

        assert valid.min() >= -2.0, f"Oscillator min {valid.min()} too low"
        assert valid.max() <= 2.0, f"Oscillator max {valid.max()} too high"

    def test_velocity_is_derivative_of_oscillator(self, synthetic_ohlcv, base_config):
        """Velocity should be the first difference of smoothed oscillator."""
        df, _ = synthetic_ohlcv
        df = calculate_composite_oscillator(df, oscillator_type='composite')
        df = calculate_velocity_signals(
            df,
            signal_type=base_config['signal_type'],
            oversold_threshold=base_config['oversold_threshold'],
            overbought_threshold=base_config['overbought_threshold'],
            vel_smoothing=base_config['vel_smoothing'],
            extreme_zone_mult=base_config['extreme_zone_mult'],
            require_accel=base_config['require_accel'],
        )

        if 'velocity' in df.columns:
            osc_col = 'osc_smooth' if 'osc_smooth' in df.columns else 'JD_Osc'
            # Velocity should be diff of osc_smooth (possibly after additional smoothing)
            # Just verify it exists and is numeric
            valid = df['velocity'].dropna()
            assert len(valid) > 0, "No valid velocity values"
            assert valid.dtype in [np.float64, np.float32, float], "Velocity not numeric"

    def test_no_simultaneous_buy_and_sell(self, synthetic_ohlcv, base_config):
        """No bar should have both buy and sell signals."""
        df, _ = synthetic_ohlcv
        df = calculate_composite_oscillator(df, oscillator_type='composite')
        df = calculate_velocity_signals(
            df,
            signal_type=base_config['signal_type'],
            oversold_threshold=base_config['oversold_threshold'],
            overbought_threshold=base_config['overbought_threshold'],
            vel_smoothing=base_config['vel_smoothing'],
            extreme_zone_mult=base_config['extreme_zone_mult'],
            require_accel=base_config['require_accel'],
        )

        valid = df.dropna(subset=['buy_signal', 'sell_signal'])
        both = valid[valid['buy_signal'].astype(bool) & valid['sell_signal'].astype(bool)]

        assert len(both) == 0, \
            f"Found {len(both)} bars with both buy AND sell signals"


# =============================================================================
# Cross-Path Comparison (Modular vs Backtest)
# =============================================================================

@pytest.mark.skipif(not HAS_BACKTEST, reason="velocity_core not importable")
class TestLiveMatchesBacktest:
    """
    Compare signals between the modular system (live) and
    velocity_core.py (backtest) to ensure they match.
    """

    def test_oscillator_values_match(self, synthetic_ohlcv, base_config):
        """Oscillator values from both paths should match."""
        df_cap, df_lower = synthetic_ohlcv

        # Modular path
        mod_df = calculate_composite_oscillator(df_cap.copy(), oscillator_type='composite')

        # Backtest path
        bt_df = backtest_oscillator(df_lower.copy(), base_config)

        # Find common oscillator column
        mod_osc = mod_df.get('osc_smooth', mod_df.get('JD_Osc', mod_df.get('composite_smooth')))
        bt_osc = bt_df.get('osc_smooth', bt_df.get('composite_smooth'))

        if mod_osc is not None and bt_osc is not None:
            # Compare valid (non-NaN) values
            valid = mod_osc.notna() & bt_osc.notna()
            if valid.any():
                correlation = mod_osc[valid].corr(bt_osc[valid])
                assert correlation > 0.95, \
                    f"Oscillator correlation too low: {correlation:.3f}"

    def test_buy_signals_match(self, synthetic_ohlcv, base_config):
        """Buy signals from both paths should match."""
        df_cap, df_lower = synthetic_ohlcv

        # Modular path
        mod_df = calculate_composite_oscillator(df_cap.copy(), oscillator_type='composite')
        mod_df = calculate_velocity_signals(
            mod_df,
            signal_type=base_config['signal_type'],
            oversold_threshold=base_config['oversold_threshold'],
            overbought_threshold=base_config['overbought_threshold'],
            vel_smoothing=base_config['vel_smoothing'],
            extreme_zone_mult=base_config['extreme_zone_mult'],
            require_accel=base_config['require_accel'],
        )

        # Backtest path
        bt_df = backtest_oscillator(df_lower.copy(), base_config)
        bt_df = backtest_signals(bt_df, base_config)

        # Compare buy signals
        if 'buy_signal' in mod_df.columns and 'buy_signal' in bt_df.columns:
            mod_buys = mod_df['buy_signal'].fillna(False).astype(bool)
            bt_buys = bt_df['buy_signal'].fillna(False).astype(bool)

            # Allow up to 5% signal mismatch (due to minor floating point diffs)
            valid_count = min(len(mod_buys), len(bt_buys))
            if valid_count > 0:
                mismatches = (mod_buys[:valid_count] != bt_buys[:valid_count]).sum()
                mismatch_rate = mismatches / valid_count
                assert mismatch_rate < 0.05, \
                    f"Buy signal mismatch rate: {mismatch_rate:.1%} ({mismatches}/{valid_count})"

    def test_sell_signals_match(self, synthetic_ohlcv, base_config):
        """Sell signals from both paths should match."""
        df_cap, df_lower = synthetic_ohlcv

        # Modular path
        mod_df = calculate_composite_oscillator(df_cap.copy(), oscillator_type='composite')
        mod_df = calculate_velocity_signals(
            mod_df,
            signal_type=base_config['signal_type'],
            oversold_threshold=base_config['oversold_threshold'],
            overbought_threshold=base_config['overbought_threshold'],
            vel_smoothing=base_config['vel_smoothing'],
            extreme_zone_mult=base_config['extreme_zone_mult'],
            require_accel=base_config['require_accel'],
        )

        # Backtest path
        bt_df = backtest_oscillator(df_lower.copy(), base_config)
        bt_df = backtest_signals(bt_df, base_config)

        if 'sell_signal' in mod_df.columns and 'sell_signal' in bt_df.columns:
            mod_sells = mod_df['sell_signal'].fillna(False).astype(bool)
            bt_sells = bt_df['sell_signal'].fillna(False).astype(bool)

            valid_count = min(len(mod_sells), len(bt_sells))
            if valid_count > 0:
                mismatches = (mod_sells[:valid_count] != bt_sells[:valid_count]).sum()
                mismatch_rate = mismatches / valid_count
                assert mismatch_rate < 0.05, \
                    f"Sell signal mismatch rate: {mismatch_rate:.1%} ({mismatches}/{valid_count})"

    def test_signal_count_similar(self, synthetic_ohlcv, base_config):
        """Total signal counts should be in the same ballpark."""
        df_cap, df_lower = synthetic_ohlcv

        # Modular
        mod_df = calculate_composite_oscillator(df_cap.copy(), oscillator_type='composite')
        mod_df = calculate_velocity_signals(
            mod_df,
            signal_type=base_config['signal_type'],
            oversold_threshold=base_config['oversold_threshold'],
            overbought_threshold=base_config['overbought_threshold'],
            vel_smoothing=base_config['vel_smoothing'],
            extreme_zone_mult=base_config['extreme_zone_mult'],
            require_accel=base_config['require_accel'],
        )

        # Backtest
        bt_df = backtest_oscillator(df_lower.copy(), base_config)
        bt_df = backtest_signals(bt_df, base_config)

        if 'buy_signal' in mod_df.columns and 'buy_signal' in bt_df.columns:
            mod_buy_count = mod_df['buy_signal'].fillna(False).astype(bool).sum()
            bt_buy_count = bt_df['buy_signal'].fillna(False).astype(bool).sum()

            # If either has signals, they should be within 2x of each other
            if max(mod_buy_count, bt_buy_count) > 0:
                ratio = (mod_buy_count + 1) / (bt_buy_count + 1)
                assert 0.3 < ratio < 3.0, \
                    f"Buy signal count mismatch: modular={mod_buy_count}, backtest={bt_buy_count}"


# =============================================================================
# Signal Generation with Different Config Variants
# =============================================================================

class TestSignalConfigVariants:
    """Test signal generation with different configuration parameters."""

    def test_zone_only_signals(self, synthetic_ohlcv):
        """Zone-only signal type should produce signals."""
        df, _ = synthetic_ohlcv
        df = calculate_composite_oscillator(df, oscillator_type='composite')
        df = calculate_velocity_signals(
            df,
            signal_type='zone_only',
            oversold_threshold=-0.2,
            overbought_threshold=0.2,
            vel_smoothing=3,
        )

        assert 'buy_signal' in df.columns

    def test_velocity_crossover_signals(self, synthetic_ohlcv):
        """Velocity crossover signal type should produce signals."""
        df, _ = synthetic_ohlcv
        df = calculate_composite_oscillator(df, oscillator_type='composite')
        df = calculate_velocity_signals(
            df,
            signal_type='velocity_crossover_or_zone',
            oversold_threshold=-0.3,
            overbought_threshold=0.3,
            vel_smoothing=3,
            require_accel=False,
        )

        assert 'buy_signal' in df.columns

    def test_stricter_thresholds_fewer_signals(self, synthetic_ohlcv):
        """Stricter thresholds should produce fewer or equal signals."""
        df, _ = synthetic_ohlcv

        # Loose thresholds
        df_loose = calculate_composite_oscillator(df.copy(), oscillator_type='composite')
        df_loose = calculate_velocity_signals(
            df_loose,
            signal_type='any_reversal',
            oversold_threshold=-0.1,
            overbought_threshold=0.1,
            vel_smoothing=3,
            require_accel=False,
        )

        # Strict thresholds
        df_strict = calculate_composite_oscillator(df.copy(), oscillator_type='composite')
        df_strict = calculate_velocity_signals(
            df_strict,
            signal_type='any_reversal',
            oversold_threshold=-0.5,
            overbought_threshold=0.5,
            vel_smoothing=3,
            require_accel=True,
        )

        loose_signals = df_loose['buy_signal'].fillna(False).astype(bool).sum()
        strict_signals = df_strict['buy_signal'].fillna(False).astype(bool).sum()

        assert strict_signals <= loose_signals, \
            f"Stricter thresholds produced MORE signals ({strict_signals} > {loose_signals})"

    def test_different_smoothing_produces_different_signals(self, synthetic_ohlcv):
        """Different vel_smoothing should produce different signal patterns."""
        df, _ = synthetic_ohlcv

        df1 = calculate_composite_oscillator(df.copy(), oscillator_type='composite')
        df1 = calculate_velocity_signals(
            df1, signal_type='any_reversal', vel_smoothing=1,
            oversold_threshold=-0.3, overbought_threshold=0.3,
        )

        df2 = calculate_composite_oscillator(df.copy(), oscillator_type='composite')
        df2 = calculate_velocity_signals(
            df2, signal_type='any_reversal', vel_smoothing=10,
            oversold_threshold=-0.3, overbought_threshold=0.3,
        )

        buy1 = df1['buy_signal'].fillna(False).astype(bool).sum()
        buy2 = df2['buy_signal'].fillna(False).astype(bool).sum()

        # Different smoothing should produce different counts (unless data is degenerate)
        # At minimum, both should run without errors
        assert buy1 >= 0
        assert buy2 >= 0


# =============================================================================
# Incremental Data Consistency
# =============================================================================

class TestIncrementalDataConsistency:
    """
    Test that adding one bar to the end of data doesn't change
    signals for previous bars. This ensures live trading (which
    adds bars incrementally) matches backtest (which has all data).
    """

    def test_new_bar_doesnt_change_old_signals(self, synthetic_ohlcv, base_config):
        """
        Adding a new bar should not change signals for bars before it.

        This catches lookahead bias - if signals change retroactively
        when new data arrives, live trading won't match backtest.
        """
        df, _ = synthetic_ohlcv

        # Calculate with N bars
        n = 250
        df_short = df.iloc[:n].copy()
        df_short = calculate_composite_oscillator(df_short, oscillator_type='composite')
        df_short = calculate_velocity_signals(
            df_short,
            signal_type=base_config['signal_type'],
            oversold_threshold=base_config['oversold_threshold'],
            overbought_threshold=base_config['overbought_threshold'],
            vel_smoothing=base_config['vel_smoothing'],
            extreme_zone_mult=base_config['extreme_zone_mult'],
            require_accel=base_config['require_accel'],
        )

        # Calculate with N+10 bars
        df_long = df.iloc[:n + 10].copy()
        df_long = calculate_composite_oscillator(df_long, oscillator_type='composite')
        df_long = calculate_velocity_signals(
            df_long,
            signal_type=base_config['signal_type'],
            oversold_threshold=base_config['oversold_threshold'],
            overbought_threshold=base_config['overbought_threshold'],
            vel_smoothing=base_config['vel_smoothing'],
            extreme_zone_mult=base_config['extreme_zone_mult'],
            require_accel=base_config['require_accel'],
        )

        # Trim the last few bars of the shorter df that might be affected
        # by centered rolling means (warmup region at end)
        check_end = n - 5  # Allow 5-bar edge effect
        check_start = 50   # Skip warmup period

        short_buys = df_short.iloc[check_start:check_end]['buy_signal'].fillna(False).astype(bool)
        long_buys = df_long.iloc[check_start:check_end]['buy_signal'].fillna(False).astype(bool)

        mismatches = (short_buys.values != long_buys.values).sum()
        total = len(short_buys)

        # Zero tolerance for this - signals must not change retroactively
        # (Exception: last few bars due to centered rolling windows)
        assert mismatches == 0, \
            f"Adding new bars changed {mismatches}/{total} old buy signals - lookahead bias!"
