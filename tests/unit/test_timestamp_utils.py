"""
Unit tests for timestamp utilities and bar key functions.

These tests verify:
1. normalize_timestamp handles all formats correctly
2. compare_timestamps works reliably (not ASCII-based)
3. get_bar_key truncates correctly per precision level
4. Interval-aware bar keys (Phase 2 target)

Phase B: These tests are written BEFORE implementation changes.
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from velocity_trading.core.position_manager import (
    normalize_timestamp,
    compare_timestamps,
    get_bar_key
)


# =============================================================================
# Test: normalize_timestamp
# =============================================================================

class TestNormalizeTimestamp:
    """Verify timestamp normalization handles all input formats."""

    def test_iso_format_passthrough(self):
        """Standard ISO format should pass through."""
        result = normalize_timestamp("2024-01-02T10:15:00")
        assert result == "2024-01-02T10:15:00"

    def test_space_separator_to_iso(self):
        """Space-separated timestamp should be converted to ISO."""
        result = normalize_timestamp("2024-01-02 10:15:00")
        assert result == "2024-01-02T10:15:00"

    def test_utc_timezone_stripped(self):
        """UTC timezone should be stripped (converted to naive)."""
        result = normalize_timestamp("2024-01-02T10:15:00+00:00")
        assert result == "2024-01-02T10:15:00"

    def test_cst_offset_converted_to_utc(self):
        """CST offset (-06:00) should be converted to UTC."""
        result = normalize_timestamp("2024-01-02T04:15:00-06:00")
        assert result == "2024-01-02T10:15:00"

    def test_est_offset_converted_to_utc(self):
        """EST offset (-05:00) should be converted to UTC."""
        result = normalize_timestamp("2024-01-02T05:15:00-05:00")
        assert result == "2024-01-02T10:15:00"

    def test_date_only(self):
        """Date-only string should get midnight time."""
        result = normalize_timestamp("2024-01-02")
        assert result == "2024-01-02T00:00:00"

    def test_pandas_timestamp(self):
        """Pandas Timestamp should be handled."""
        import pandas as pd
        ts = pd.Timestamp("2024-01-02 10:15:00")
        result = normalize_timestamp(ts)
        assert result == "2024-01-02T10:15:00"

    def test_pandas_timestamp_with_tz(self):
        """Pandas Timestamp with timezone should be converted to UTC."""
        import pandas as pd
        ts = pd.Timestamp("2024-01-02 10:15:00", tz="UTC")
        result = normalize_timestamp(ts)
        assert result == "2024-01-02T10:15:00"

    def test_seconds_truncated(self):
        """Seconds should be preserved in output."""
        result = normalize_timestamp("2024-01-02T10:15:30")
        assert result == "2024-01-02T10:15:30"

    def test_consistent_output_format(self):
        """All inputs should produce consistent YYYY-MM-DDTHH:MM:SS format."""
        inputs = [
            "2024-01-02T10:15:00",
            "2024-01-02 10:15:00",
            "2024-01-02T10:15:00+00:00",
            "2024-01-02T10:15:00Z",
        ]
        expected = "2024-01-02T10:15:00"
        for inp in inputs:
            result = normalize_timestamp(inp)
            assert result == expected, f"Input '{inp}' produced '{result}', expected '{expected}'"


# =============================================================================
# Test: compare_timestamps
# =============================================================================

class TestCompareTimestamps:
    """Verify timestamp comparison is reliable (datetime-based, not ASCII)."""

    def test_equal_timestamps(self):
        """Equal timestamps should return 0."""
        assert compare_timestamps("2024-01-02T10:00:00", "2024-01-02T10:00:00") == 0

    def test_mixed_format_equal(self):
        """Same timestamp in different formats should compare equal."""
        assert compare_timestamps("2024-01-02T10:00:00", "2024-01-02 10:00:00") == 0

    def test_earlier_returns_negative(self):
        """Earlier timestamp should return -1."""
        assert compare_timestamps("2024-01-02T09:00:00", "2024-01-02T10:00:00") == -1

    def test_later_returns_positive(self):
        """Later timestamp should return 1."""
        assert compare_timestamps("2024-01-02T11:00:00", "2024-01-02T10:00:00") == 1

    def test_ascii_trap_avoided(self):
        """
        Space (ASCII 32) < 'T' (ASCII 84), so naive string comparison
        of "2024-01-02 11:15:00" vs "2024-01-02T03:45:00" would give
        wrong result. Datetime comparison should get it right.
        """
        # 11:15 is AFTER 03:45, but ASCII string comparison says otherwise
        assert compare_timestamps("2024-01-02 11:15:00", "2024-01-02T03:45:00") == 1

    def test_timezone_aware_comparison(self):
        """Timezone-aware timestamps should be compared in UTC."""
        # 10:00 UTC == 04:00 CST
        assert compare_timestamps(
            "2024-01-02T10:00:00+00:00",
            "2024-01-02T04:00:00-06:00"
        ) == 0


# =============================================================================
# Test: get_bar_key
# =============================================================================

class TestGetBarKey:
    """Verify bar key generation for same-bar detection."""

    def test_minute_precision(self):
        """Minute precision should truncate to HH:MM."""
        result = get_bar_key("2024-01-02T10:15:30", "minute")
        assert result == "2024-01-02T10:15"

    def test_hour_precision(self):
        """Hour precision should truncate to HH."""
        result = get_bar_key("2024-01-02T10:15:30", "hour")
        assert result == "2024-01-02T10"

    def test_day_precision(self):
        """Day precision should return date only."""
        result = get_bar_key("2024-01-02T10:15:30", "day")
        assert result == "2024-01-02"

    def test_same_minute_same_key(self):
        """Two timestamps in same minute should have same key."""
        key1 = get_bar_key("2024-01-02T10:15:00", "minute")
        key2 = get_bar_key("2024-01-02T10:15:45", "minute")
        assert key1 == key2

    def test_different_minute_different_key(self):
        """Two timestamps in different minutes should have different keys."""
        key1 = get_bar_key("2024-01-02T10:15:00", "minute")
        key2 = get_bar_key("2024-01-02T10:16:00", "minute")
        assert key1 != key2

    def test_timezone_stripped_for_key(self):
        """Timezone should be stripped before generating key."""
        key1 = get_bar_key("2024-01-02T10:15:00", "minute")
        key2 = get_bar_key("2024-01-02T10:15:00+00:00", "minute")
        assert key1 == key2


# =============================================================================
# Test: Interval-Aware Bar Keys (Phase 2 Target)
# =============================================================================

class TestIntervalAwareBarKeys:
    """Tests for interval-aware bar key generation."""

    def test_15m_bar_key_function_exists(self):
        """get_bar_key_for_interval should exist."""
        from velocity_trading.core.position_manager import get_bar_key_for_interval
        assert callable(get_bar_key_for_interval)

    def test_15m_bar_key_same_bar(self):
        """Two timestamps within same 15m bar should have same key."""
        from velocity_trading.core.position_manager import get_bar_key_for_interval

        # 10:00 and 10:14 are in the same 15m bar (10:00 - 10:15)
        key1 = get_bar_key_for_interval("2024-01-02T10:00:00", "15m")
        key2 = get_bar_key_for_interval("2024-01-02T10:14:00", "15m")
        assert key1 == key2

    def test_15m_bar_key_different_bar(self):
        """Two timestamps in different 15m bars should have different keys."""
        from velocity_trading.core.position_manager import get_bar_key_for_interval

        # 10:00 and 10:15 are in different 15m bars
        key1 = get_bar_key_for_interval("2024-01-02T10:00:00", "15m")
        key2 = get_bar_key_for_interval("2024-01-02T10:15:00", "15m")
        assert key1 != key2

    def test_1h_bar_key_same_bar(self):
        """Two timestamps within same 1h bar should have same key."""
        from velocity_trading.core.position_manager import get_bar_key_for_interval

        key1 = get_bar_key_for_interval("2024-01-02T10:00:00", "1h")
        key2 = get_bar_key_for_interval("2024-01-02T10:59:00", "1h")
        assert key1 == key2

    def test_1d_bar_key(self):
        """Daily bar key should use date only."""
        from velocity_trading.core.position_manager import get_bar_key_for_interval

        key1 = get_bar_key_for_interval("2024-01-02T10:00:00", "1d")
        key2 = get_bar_key_for_interval("2024-01-02T15:30:00", "1d")
        assert key1 == key2

    def test_boundary_15m(self):
        """15m boundary should separate bars."""
        from velocity_trading.core.position_manager import get_bar_key_for_interval

        # 10:14:59 -> bar starting at 10:00
        # 10:15:00 -> bar starting at 10:15
        key1 = get_bar_key_for_interval("2024-01-02T10:14:59", "15m")
        key2 = get_bar_key_for_interval("2024-01-02T10:15:00", "15m")
        assert key1 != key2

    def test_30m_bar_key(self):
        """30m intervals should round to 30-minute boundaries."""
        from velocity_trading.core.position_manager import get_bar_key_for_interval

        # 10:00 and 10:29 are in the same 30m bar
        key1 = get_bar_key_for_interval("2024-01-02T10:00:00", "30m")
        key2 = get_bar_key_for_interval("2024-01-02T10:29:00", "30m")
        assert key1 == key2

        # 10:30 is a new bar
        key3 = get_bar_key_for_interval("2024-01-02T10:30:00", "30m")
        assert key1 != key3

    def test_5m_bar_key(self):
        """5m intervals should round to 5-minute boundaries."""
        from velocity_trading.core.position_manager import get_bar_key_for_interval

        key1 = get_bar_key_for_interval("2024-01-02T10:00:00", "5m")
        key2 = get_bar_key_for_interval("2024-01-02T10:04:59", "5m")
        assert key1 == key2

        key3 = get_bar_key_for_interval("2024-01-02T10:05:00", "5m")
        assert key1 != key3

    def test_2h_bar_key(self):
        """2h intervals should round to 2-hour boundaries."""
        from velocity_trading.core.position_manager import get_bar_key_for_interval

        key1 = get_bar_key_for_interval("2024-01-02T10:00:00", "2h")
        key2 = get_bar_key_for_interval("2024-01-02T11:59:00", "2h")
        assert key1 == key2

        key3 = get_bar_key_for_interval("2024-01-02T12:00:00", "2h")
        assert key1 != key3

    def test_timezone_stripped(self):
        """Timezone should be stripped before computing bar key."""
        from velocity_trading.core.position_manager import get_bar_key_for_interval

        key1 = get_bar_key_for_interval("2024-01-02T10:07:00", "15m")
        key2 = get_bar_key_for_interval("2024-01-02T10:07:00+00:00", "15m")
        assert key1 == key2
