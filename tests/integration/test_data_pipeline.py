"""
Integration tests for the data pipeline.

These tests verify:
1. Database schema and initialization
2. Bar storage and retrieval (data continuity)
3. OHLC validation (_validate_bars)
4. Internal gap detection logic
5. Rate limiting (API call caching)
6. Timestamp normalization in storage
7. Data range tracking
8. GC=F validation skip behavior

Phase D: These tests are written BEFORE implementation changes.
Tests use temporary databases and mock external API calls.
"""

import os
import sys
import pytest
import sqlite3
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from velocity_trading.data.data_pipeline import DataPipeline


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def pipeline(tmp_path):
    """Create a DataPipeline with a temporary database directory."""
    with patch.object(DataPipeline, '__init__', lambda self, *a, **k: None):
        p = DataPipeline.__new__(DataPipeline)

    p.ticker = 'ES=F'
    p.interval = '15m'
    p.db_symbol = 'ES.n.0'
    p.is_futures = True
    p.db_path = str(tmp_path / 'ohlcv_ES_F_15m.db')
    p._live_client = None
    p._live_thread = None
    p._live_running = False
    p._last_api_fetch_time = None
    p._last_api_fetch_result = None
    p._min_fetch_interval_seconds = 30
    p._init_db()
    return p


@pytest.fixture
def pipeline_gcf(tmp_path):
    """Create a DataPipeline for GC=F (to test validation skip)."""
    with patch.object(DataPipeline, '__init__', lambda self, *a, **k: None):
        p = DataPipeline.__new__(DataPipeline)

    p.ticker = 'GC=F'
    p.interval = '15m'
    p.db_symbol = 'GC.c.1'
    p.is_futures = True
    p.db_path = str(tmp_path / 'ohlcv_GC_F_15m.db')
    p._live_client = None
    p._live_thread = None
    p._live_running = False
    p._last_api_fetch_time = None
    p._last_api_fetch_result = None
    p._min_fetch_interval_seconds = 30
    p._init_db()
    return p


@pytest.fixture
def sample_bars():
    """Generate sample 15m OHLCV bars for testing."""
    np.random.seed(42)
    n_bars = 50
    # Use Tuesday to avoid weekend complications
    dates = pd.date_range('2024-01-02 09:30', periods=n_bars, freq='15min')

    base_price = 5000.0
    returns = np.random.normal(0, 0.001, n_bars)
    prices = base_price * np.cumprod(1 + returns)

    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        range_pct = abs(np.random.normal(0, 0.002))
        high = close * (1 + range_pct / 2)
        low = close * (1 - range_pct / 2)
        open_price = prices[i - 1] if i > 0 else close * 0.999
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': int(np.random.uniform(10000, 100000))
        })

    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def bars_with_gap():
    """
    Generate bars with a gap (simulating Friday close to Monday open).

    Creates bars on Friday afternoon, then Monday morning, with nothing
    in between (missing Sunday evening CME reopen bars).
    """
    # Friday bars: Jan 5, 2024 is a Friday
    friday_dates = pd.date_range('2024-01-05 14:00', periods=8, freq='15min')
    # Monday bars: Jan 8, 2024 is a Monday
    monday_dates = pd.date_range('2024-01-08 09:30', periods=8, freq='15min')

    all_dates = friday_dates.append(monday_dates)

    base_price = 5000.0
    data = []
    for i, date in enumerate(all_dates):
        close = base_price + i * 0.5
        data.append({
            'open': close - 0.1,
            'high': close + 1.0,
            'low': close - 1.0,
            'close': close,
            'volume': 50000
        })

    return pd.DataFrame(data, index=all_dates)


# =============================================================================
# Test: Database Schema and Initialization
# =============================================================================

class TestDatabaseInit:
    """Verify database is initialized correctly."""

    def test_db_file_created(self, pipeline):
        """Database file should be created on init."""
        assert os.path.exists(pipeline.db_path)

    def test_ohlcv_table_exists(self, pipeline):
        """OHLCV table should exist after init."""
        conn = sqlite3.connect(pipeline.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_ohlcv_table_columns(self, pipeline):
        """OHLCV table should have correct columns."""
        conn = sqlite3.connect(pipeline.db_path)
        cursor = conn.execute("PRAGMA table_info(ohlcv)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {'timestamp', 'open', 'high', 'low', 'close', 'volume', 'source', 'created_at'}
        assert expected == columns
        conn.close()

    def test_timestamp_index_exists(self, pipeline):
        """Timestamp index should exist for fast queries."""
        conn = sqlite3.connect(pipeline.db_path)
        cursor = conn.execute("PRAGMA index_list(ohlcv)")
        indexes = [row[1] for row in cursor.fetchall()]
        assert 'idx_ohlcv_timestamp' in indexes
        conn.close()

    def test_empty_db_range(self, pipeline):
        """Empty database should return None for data range."""
        min_ts, max_ts = pipeline._get_data_range()
        assert min_ts is None
        assert max_ts is None


# =============================================================================
# Test: Bar Storage and Retrieval (Data Continuity)
# =============================================================================

class TestBarStorageAndRetrieval:
    """Verify bars are stored and retrieved correctly."""

    def test_store_bars(self, pipeline, sample_bars):
        """Bars should be stored in database."""
        bars_stored = pipeline._store_bars(sample_bars, source='historical')
        assert bars_stored == len(sample_bars)

    def test_stored_bars_retrievable(self, pipeline, sample_bars):
        """Stored bars should be retrievable via get_data."""
        pipeline._store_bars(sample_bars, source='historical')

        # Disable gap detection and synthetic bar for isolated test
        with patch.object(pipeline, '_detect_and_fill_internal_gaps'):
            with patch.object(pipeline, '_add_synthetic_bar_if_needed', side_effect=lambda df, **kw: df):
                df = pipeline.get_data(lookback_bars=100, add_synthetic_current=False)

        assert len(df) == len(sample_bars)

    def test_data_range_after_store(self, pipeline, sample_bars):
        """Data range should reflect stored bars."""
        pipeline._store_bars(sample_bars, source='historical')
        min_ts, max_ts = pipeline._get_data_range()
        assert min_ts is not None
        assert max_ts is not None
        assert min_ts < max_ts

    def test_latest_bar_matches(self, pipeline, sample_bars):
        """get_latest_bar should return the most recent bar."""
        pipeline._store_bars(sample_bars, source='historical')
        latest = pipeline.get_latest_bar()
        assert latest is not None
        assert latest['close'] == pytest.approx(sample_bars.iloc[-1]['close'], abs=0.01)

    def test_no_duplicate_timestamps(self, pipeline, sample_bars):
        """Storing same bars twice should not create duplicates."""
        pipeline._store_bars(sample_bars, source='historical')
        pipeline._store_bars(sample_bars, source='historical')

        conn = sqlite3.connect(pipeline.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM ohlcv")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == len(sample_bars)

    def test_incremental_append(self, pipeline, sample_bars):
        """New bars should append without overwriting existing."""
        # Store first half
        first_half = sample_bars.iloc[:25]
        pipeline._store_bars(first_half, source='historical')

        # Store second half
        second_half = sample_bars.iloc[25:]
        pipeline._store_bars(second_half, source='live')

        conn = sqlite3.connect(pipeline.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM ohlcv")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == len(sample_bars)

    def test_data_survives_new_pipeline_instance(self, pipeline, sample_bars, tmp_path):
        """Data should persist across DataPipeline instances."""
        pipeline._store_bars(sample_bars, source='historical')

        # Create new pipeline pointing to same db
        with patch.object(DataPipeline, '__init__', lambda self, *a, **k: None):
            p2 = DataPipeline.__new__(DataPipeline)
        p2.ticker = 'ES=F'
        p2.interval = '15m'
        p2.db_symbol = 'ES.n.0'
        p2.is_futures = True
        p2.db_path = pipeline.db_path
        p2._live_client = None
        p2._live_thread = None
        p2._live_running = False
        p2._last_api_fetch_time = None
        p2._last_api_fetch_result = None
        p2._min_fetch_interval_seconds = 30
        p2._init_db()

        min_ts, max_ts = p2._get_data_range()
        assert min_ts is not None
        assert max_ts is not None

    def test_lookback_bars_limits_result(self, pipeline, sample_bars):
        """get_data should respect lookback_bars parameter."""
        pipeline._store_bars(sample_bars, source='historical')

        with patch.object(pipeline, '_detect_and_fill_internal_gaps'):
            with patch.object(pipeline, '_add_synthetic_bar_if_needed', side_effect=lambda df, **kw: df):
                df = pipeline.get_data(lookback_bars=10, add_synthetic_current=False)

        assert len(df) == 10

    def test_data_sorted_by_timestamp(self, pipeline, sample_bars):
        """Retrieved data should be sorted by timestamp ascending."""
        pipeline._store_bars(sample_bars, source='historical')

        with patch.object(pipeline, '_detect_and_fill_internal_gaps'):
            with patch.object(pipeline, '_add_synthetic_bar_if_needed', side_effect=lambda df, **kw: df):
                df = pipeline.get_data(lookback_bars=100, add_synthetic_current=False)

        assert df.index.is_monotonic_increasing


# =============================================================================
# Test: OHLC Validation
# =============================================================================

class TestOHLCValidation:
    """Verify OHLC structural validation catches bad data."""

    def test_valid_bars_pass(self, pipeline, sample_bars):
        """Valid OHLC data should pass validation unchanged."""
        result = pipeline._validate_bars(sample_bars)
        assert len(result) == len(sample_bars)

    def test_negative_prices_removed(self, pipeline):
        """Bars with negative prices should be removed."""
        dates = pd.date_range('2024-01-02 10:00', periods=3, freq='15min')
        df = pd.DataFrame({
            'open': [100.0, -1.0, 100.0],
            'high': [101.0, -0.5, 101.0],
            'low': [99.0, -1.5, 99.0],
            'close': [100.5, -0.8, 100.5],
            'volume': [1000, 1000, 1000]
        }, index=dates)

        result = pipeline._validate_bars(df)
        assert len(result) == 2

    def test_zero_prices_removed(self, pipeline):
        """Bars with zero prices should be removed."""
        dates = pd.date_range('2024-01-02 10:00', periods=3, freq='15min')
        df = pd.DataFrame({
            'open': [100.0, 0.0, 100.0],
            'high': [101.0, 0.0, 101.0],
            'low': [99.0, 0.0, 99.0],
            'close': [100.5, 0.0, 100.5],
            'volume': [1000, 0, 1000]
        }, index=dates)

        result = pipeline._validate_bars(df)
        assert len(result) == 2

    def test_low_greater_than_high_removed(self, pipeline):
        """Bars where low > high (impossible) should be removed."""
        dates = pd.date_range('2024-01-02 10:00', periods=3, freq='15min')
        df = pd.DataFrame({
            'open': [100.0, 100.0, 100.0],
            'high': [101.0, 99.0, 101.0],   # Second bar: high < low
            'low': [99.0, 101.0, 99.0],
            'close': [100.5, 100.0, 100.5],
            'volume': [1000, 1000, 1000]
        }, index=dates)

        result = pipeline._validate_bars(df)
        assert len(result) == 2

    def test_open_outside_range_removed(self, pipeline):
        """Bars where open is outside high/low range should be removed."""
        dates = pd.date_range('2024-01-02 10:00', periods=3, freq='15min')
        df = pd.DataFrame({
            'open': [100.0, 110.0, 100.0],   # Second bar: open > high
            'high': [101.0, 101.0, 101.0],
            'low': [99.0, 99.0, 99.0],
            'close': [100.5, 100.0, 100.5],
            'volume': [1000, 1000, 1000]
        }, index=dates)

        result = pipeline._validate_bars(df)
        assert len(result) == 2

    def test_close_outside_range_removed(self, pipeline):
        """Bars where close is outside high/low range should be removed."""
        dates = pd.date_range('2024-01-02 10:00', periods=3, freq='15min')
        df = pd.DataFrame({
            'open': [100.0, 100.0, 100.0],
            'high': [101.0, 101.0, 101.0],
            'low': [99.0, 99.0, 99.0],
            'close': [100.5, 95.0, 100.5],   # Second bar: close < low
            'volume': [1000, 1000, 1000]
        }, index=dates)

        result = pipeline._validate_bars(df)
        assert len(result) == 2

    def test_empty_dataframe_returns_empty(self, pipeline):
        """Empty dataframe should return empty."""
        df = pd.DataFrame()
        result = pipeline._validate_bars(df)
        assert result.empty

    def test_large_moves_accepted(self, pipeline):
        """Large percentage moves should be accepted (not filtered)."""
        dates = pd.date_range('2024-01-02 10:00', periods=3, freq='15min')
        df = pd.DataFrame({
            'open': [100.0, 90.0, 80.0],     # 10% gaps between bars
            'high': [101.0, 91.0, 81.0],
            'low': [99.0, 89.0, 79.0],
            'close': [100.5, 90.5, 80.5],
            'volume': [1000, 1000, 1000]
        }, index=dates)

        result = pipeline._validate_bars(df)
        assert len(result) == 3, "Large but structurally valid moves should not be filtered"


# =============================================================================
# Test: GC=F Validation Skip
# =============================================================================

class TestGCFValidationSkip:
    """Verify GC=F skips validation when using Polygon data."""

    def test_gcf_skips_validation_for_historical(self, pipeline_gcf):
        """GC=F should skip validation for 'historical' source."""
        dates = pd.date_range('2024-01-02 10:00', periods=3, freq='15min')
        # Create data with low > high (would normally be filtered)
        df = pd.DataFrame({
            'open': [2050.0, 2050.0, 2050.0],
            'high': [2049.0, 2055.0, 2055.0],  # First bar: high < low (bad)
            'low': [2051.0, 2045.0, 2045.0],
            'close': [2050.0, 2052.0, 2052.0],
            'volume': [1000, 1000, 1000]
        }, index=dates)

        bars_stored = pipeline_gcf._store_bars(df, source='historical')
        assert bars_stored == 3, "GC=F should store all bars without validation for 'historical' source"

    def test_gcf_skips_validation_for_polygon(self, pipeline_gcf):
        """GC=F should skip validation for 'polygon/databento' source."""
        dates = pd.date_range('2024-01-02 10:00', periods=3, freq='15min')
        df = pd.DataFrame({
            'open': [2050.0, 2050.0, 2050.0],
            'high': [2049.0, 2055.0, 2055.0],
            'low': [2051.0, 2045.0, 2045.0],
            'close': [2050.0, 2052.0, 2052.0],
            'volume': [1000, 1000, 1000]
        }, index=dates)

        bars_stored = pipeline_gcf._store_bars(df, source='polygon/databento')
        assert bars_stored == 3

    def test_non_gcf_validates(self, pipeline):
        """ES=F should validate and filter bad bars."""
        dates = pd.date_range('2024-01-02 10:00', periods=3, freq='15min')
        df = pd.DataFrame({
            'open': [5000.0, 5000.0, 5000.0],
            'high': [4999.0, 5005.0, 5005.0],  # First bar: high < low
            'low': [5001.0, 4995.0, 4995.0],
            'close': [5000.0, 5002.0, 5002.0],
            'volume': [1000, 1000, 1000]
        }, index=dates)

        bars_stored = pipeline._store_bars(df, source='historical')
        assert bars_stored == 2, "ES=F should validate and filter the bad bar"


# =============================================================================
# Test: Timestamp Normalization in Storage
# =============================================================================

class TestTimestampStorage:
    """Verify timestamps are normalized to naive UTC ISO format."""

    def test_timezone_aware_timestamps_stripped(self, pipeline):
        """Timezone-aware timestamps should be stored as naive UTC."""
        dates = pd.date_range(
            '2024-01-02 10:00', periods=3, freq='15min', tz='UTC'
        )
        df = pd.DataFrame({
            'open': [100.0, 100.0, 100.0],
            'high': [101.0, 101.0, 101.0],
            'low': [99.0, 99.0, 99.0],
            'close': [100.5, 100.5, 100.5],
            'volume': [1000, 1000, 1000]
        }, index=dates)

        pipeline._store_bars(df, source='historical')

        conn = sqlite3.connect(pipeline.db_path)
        cursor = conn.execute("SELECT timestamp FROM ohlcv ORDER BY timestamp")
        timestamps = [row[0] for row in cursor.fetchall()]
        conn.close()

        for ts in timestamps:
            # Should not contain timezone info
            assert '+' not in ts, f"Timestamp should be naive UTC: {ts}"
            assert 'Z' not in ts, f"Timestamp should not have Z suffix: {ts}"
            # Should be ISO format
            assert 'T' in ts, f"Timestamp should use ISO format: {ts}"

    def test_consistent_format_across_stores(self, pipeline):
        """Multiple stores should produce consistent timestamp format."""
        # First store: naive timestamps
        dates1 = pd.date_range('2024-01-02 10:00', periods=3, freq='15min')
        df1 = pd.DataFrame({
            'open': [100.0, 100.0, 100.0],
            'high': [101.0, 101.0, 101.0],
            'low': [99.0, 99.0, 99.0],
            'close': [100.5, 100.5, 100.5],
            'volume': [1000, 1000, 1000]
        }, index=dates1)

        # Second store: timezone-aware timestamps
        dates2 = pd.date_range(
            '2024-01-02 11:00', periods=3, freq='15min', tz='UTC'
        )
        df2 = pd.DataFrame({
            'open': [101.0, 101.0, 101.0],
            'high': [102.0, 102.0, 102.0],
            'low': [100.0, 100.0, 100.0],
            'close': [101.5, 101.5, 101.5],
            'volume': [2000, 2000, 2000]
        }, index=dates2)

        pipeline._store_bars(df1, source='historical')
        pipeline._store_bars(df2, source='live')

        conn = sqlite3.connect(pipeline.db_path)
        cursor = conn.execute("SELECT timestamp FROM ohlcv ORDER BY timestamp")
        timestamps = [row[0] for row in cursor.fetchall()]
        conn.close()

        # All timestamps should have consistent format
        for ts in timestamps:
            assert len(ts) == 19, f"Expected YYYY-MM-DDTHH:MM:SS format (19 chars), got: {ts}"


# =============================================================================
# Test: Internal Gap Detection
# =============================================================================

class TestInternalGapDetection:
    """Verify gap detection finds and characterizes data gaps."""

    def test_no_gaps_in_continuous_data(self, pipeline, sample_bars):
        """Continuous data should not trigger gap detection."""
        pipeline._store_bars(sample_bars, source='historical')

        # Mock the databento fetch to ensure it's NOT called
        with patch.object(pipeline, '_fetch_historical_databento') as mock_fetch:
            pipeline._detect_and_fill_internal_gaps()
            mock_fetch.assert_not_called()

    def test_weekend_gap_detected(self, pipeline, bars_with_gap):
        """Friday-to-Monday gap should be detected."""
        pipeline._store_bars(bars_with_gap, source='historical')

        # Mock the databento fetch to track if gap-fill is attempted
        with patch.object(pipeline, '_fetch_historical_databento', return_value=None) as mock_fetch:
            pipeline._detect_and_fill_internal_gaps()
            # Should have attempted to fill the weekend gap
            assert mock_fetch.called, "Should attempt to fill weekend gap"

    def test_gap_fill_stores_with_gap_fill_source(self, pipeline, bars_with_gap):
        """Gap-filled bars should be marked with source='gap-fill'."""
        pipeline._store_bars(bars_with_gap, source='historical')

        # Create gap-fill data (Sunday evening bars)
        sunday_dates = pd.date_range('2024-01-07 23:00', periods=4, freq='15min')
        gap_fill_df = pd.DataFrame({
            'open': [5000.0] * 4,
            'high': [5001.0] * 4,
            'low': [4999.0] * 4,
            'close': [5000.5] * 4,
            'volume': [1000] * 4
        }, index=sunday_dates)

        with patch.object(pipeline, '_fetch_historical_databento', return_value=gap_fill_df):
            pipeline._detect_and_fill_internal_gaps()

        conn = sqlite3.connect(pipeline.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE source='gap-fill'")
        gap_fill_count = cursor.fetchone()[0]
        conn.close()

        assert gap_fill_count > 0, "Gap-filled bars should be stored with 'gap-fill' source"


# =============================================================================
# Test: Rate Limiting (API Call Caching)
# =============================================================================

class TestRateLimiting:
    """Verify API call caching prevents excessive requests."""

    def test_min_fetch_interval_by_interval(self):
        """Different intervals should have different fetch intervals."""
        intervals = {
            '1m': 15,
            '5m': 30,
            '15m': 30,
            '30m': 60,
            '1h': 120,
            '1d': 300,
        }

        for interval, expected_seconds in intervals.items():
            with patch.object(DataPipeline, '__init__', lambda self, *a, **k: None):
                p = DataPipeline.__new__(DataPipeline)
            p.interval = interval
            result = p._get_min_fetch_interval()
            assert result == expected_seconds, \
                f"Interval '{interval}' should have {expected_seconds}s fetch interval, got {result}"

    def test_unknown_interval_defaults_to_30s(self):
        """Unknown intervals should default to 30s fetch interval."""
        with patch.object(DataPipeline, '__init__', lambda self, *a, **k: None):
            p = DataPipeline.__new__(DataPipeline)
        p.interval = '3m'  # Unusual interval
        result = p._get_min_fetch_interval()
        assert result == 30


# =============================================================================
# Test: Data Validation (_is_data_valid)
# =============================================================================

class TestDataValidation:
    """Verify data validation logic for different ticker types."""

    def test_placeholder_data_rejected_for_futures(self, pipeline):
        """All-identical OHLC bars should be rejected as placeholder data."""
        dates = pd.date_range('2024-01-02 10:00', periods=5, freq='15min')
        df = pd.DataFrame({
            'open': [5000.0] * 5,
            'high': [5000.0] * 5,
            'low': [5000.0] * 5,
            'close': [5000.0] * 5,
            'volume': [0] * 5
        }, index=dates)

        result = pipeline._is_data_valid(df, last_known_price=5000.0)
        assert result is False, "Placeholder data should be rejected"

    def test_valid_futures_data_accepted(self, pipeline, sample_bars):
        """Valid futures data should be accepted without strict validation."""
        result = pipeline._is_data_valid(sample_bars, last_known_price=5000.0)
        assert result is True

    def test_single_bar_placeholder_accepted(self, pipeline):
        """Single bar with identical OHLC should be accepted (could be doji)."""
        dates = pd.date_range('2024-01-02 10:00', periods=1, freq='15min')
        df = pd.DataFrame({
            'open': [5000.0],
            'high': [5000.0],
            'low': [5000.0],
            'close': [5000.0],
            'volume': [100]
        }, index=dates)

        result = pipeline._is_data_valid(df, last_known_price=5000.0)
        assert result is True, "Single doji bar should be accepted"

    def test_empty_dataframe_rejected(self, pipeline):
        """Empty dataframe should be rejected."""
        df = pd.DataFrame()
        result = pipeline._is_data_valid(df, last_known_price=5000.0)
        assert result is False

    def test_none_dataframe_rejected(self, pipeline):
        """None input should be rejected."""
        result = pipeline._is_data_valid(None, last_known_price=5000.0)
        assert result is False


# =============================================================================
# Test: Interval Minutes Mapping
# =============================================================================

class TestIntervalMinutes:
    """Verify interval-to-minutes mapping is correct."""

    def test_standard_intervals(self, pipeline):
        """Standard intervals should map to correct minutes."""
        expected = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '1d': 1440
        }

        for interval, minutes in expected.items():
            pipeline.interval = interval
            result = pipeline._get_interval_minutes()
            assert result == minutes, f"Interval '{interval}' should be {minutes} minutes"

    def test_unknown_interval_defaults_to_15(self, pipeline):
        """Unknown interval should default to 15 minutes."""
        pipeline.interval = '3m'
        result = pipeline._get_interval_minutes()
        assert result == 15


# =============================================================================
# Test: Backfill Logic
# =============================================================================

class TestBackfillLogic:
    """Verify backfill behavior."""

    def test_backfill_force_clears_data(self, pipeline, sample_bars):
        """Force backfill should clear existing data."""
        pipeline._store_bars(sample_bars, source='historical')

        conn = sqlite3.connect(pipeline.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM ohlcv")
        assert cursor.fetchone()[0] == len(sample_bars)
        conn.close()

        # Force backfill should clear then refetch
        with patch.object(pipeline, '_fetch_historical_databento', return_value=sample_bars.iloc[:10]):
            pipeline.backfill(days=60, force=True)

        conn = sqlite3.connect(pipeline.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM ohlcv")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 10, "Force backfill should clear old data and store new"

    def test_backfill_with_existing_data_runs_update(self, pipeline, sample_bars):
        """Backfill with existing data (and no force) should run update instead."""
        pipeline._store_bars(sample_bars, source='historical')

        with patch.object(pipeline, 'update', return_value=(True, {'new_bars': 0})) as mock_update:
            pipeline.backfill(days=60, force=False)
            mock_update.assert_called_once()


# =============================================================================
# Test: Source Tracking
# =============================================================================

class TestSourceTracking:
    """Verify data source is tracked correctly in database."""

    def test_historical_source_stored(self, pipeline, sample_bars):
        """Bars stored as 'historical' should be tracked."""
        pipeline._store_bars(sample_bars, source='historical')

        conn = sqlite3.connect(pipeline.db_path)
        cursor = conn.execute("SELECT DISTINCT source FROM ohlcv")
        sources = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert 'historical' in sources

    def test_live_source_stored(self, pipeline, sample_bars):
        """Bars stored as 'live' should be tracked."""
        pipeline._store_bars(sample_bars, source='live')

        conn = sqlite3.connect(pipeline.db_path)
        cursor = conn.execute("SELECT DISTINCT source FROM ohlcv")
        sources = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert 'live' in sources

    def test_source_updated_on_replace(self, pipeline, sample_bars):
        """INSERT OR REPLACE should update source for existing timestamps."""
        pipeline._store_bars(sample_bars, source='historical')
        pipeline._store_bars(sample_bars, source='live')

        conn = sqlite3.connect(pipeline.db_path)
        cursor = conn.execute("SELECT DISTINCT source FROM ohlcv")
        sources = {row[0] for row in cursor.fetchall()}
        conn.close()

        # All bars should now be marked as 'live' since they were replaced
        assert sources == {'live'}


# =============================================================================
# Test: Update Logic
# =============================================================================

class TestUpdateLogic:
    """Verify incremental update behavior."""

    def test_update_with_no_data_triggers_backfill(self, pipeline):
        """Update on empty database should trigger backfill."""
        with patch.object(pipeline, 'backfill', return_value=(True, {})) as mock_bf:
            pipeline.update()
            mock_bf.assert_called_once()

    def test_update_skips_when_data_current(self, pipeline, sample_bars):
        """Update should skip when data is less than 1 minute old."""
        # Store bars with the last one very recent
        recent_dates = pd.date_range(
            datetime.now(timezone.utc) - timedelta(seconds=30),
            periods=1, freq='15min'
        )
        recent_bar = pd.DataFrame({
            'open': [5000.0],
            'high': [5001.0],
            'low': [4999.0],
            'close': [5000.5],
            'volume': [1000]
        }, index=recent_dates)

        pipeline._store_bars(recent_bar, source='live')

        success, result = pipeline.update()
        assert success is True
        assert result.get('new_bars', 0) == 0
        assert 'current' in result.get('message', '').lower()


# =============================================================================
# Test: get_data Column Naming
# =============================================================================

class TestGetDataFormat:
    """Verify get_data returns properly formatted dataframes."""

    def test_column_names_capitalized(self, pipeline, sample_bars):
        """get_data should return columns with capital first letter."""
        pipeline._store_bars(sample_bars, source='historical')

        with patch.object(pipeline, '_detect_and_fill_internal_gaps'):
            with patch.object(pipeline, '_add_synthetic_bar_if_needed', side_effect=lambda df, **kw: df):
                df = pipeline.get_data(lookback_bars=10, add_synthetic_current=False)

        expected_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        assert set(df.columns) == expected_columns

    def test_index_is_datetime(self, pipeline, sample_bars):
        """get_data index should be DatetimeIndex."""
        pipeline._store_bars(sample_bars, source='historical')

        with patch.object(pipeline, '_detect_and_fill_internal_gaps'):
            with patch.object(pipeline, '_add_synthetic_bar_if_needed', side_effect=lambda df, **kw: df):
                df = pipeline.get_data(lookback_bars=10, add_synthetic_current=False)

        assert isinstance(df.index, pd.DatetimeIndex)

    def test_no_duplicate_index(self, pipeline, sample_bars):
        """get_data should not return duplicate timestamps."""
        # Store same data twice (INSERT OR REPLACE handles it in DB)
        pipeline._store_bars(sample_bars, source='historical')
        pipeline._store_bars(sample_bars, source='live')

        with patch.object(pipeline, '_detect_and_fill_internal_gaps'):
            with patch.object(pipeline, '_add_synthetic_bar_if_needed', side_effect=lambda df, **kw: df):
                df = pipeline.get_data(lookback_bars=100, add_synthetic_current=False)

        assert not df.index.duplicated().any(), "No duplicate timestamps in output"
