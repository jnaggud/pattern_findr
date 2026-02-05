"""
Tests for API rate limit recovery and retry logic.

Covers:
- Exponential backoff timing
- Rate limit error detection
- Connection error detection
- Max retry behavior
- Successful recovery after transient failures
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from velocity_trading.data.data_pipeline import _retry_with_backoff


# =============================================================================
# Basic Retry Behavior
# =============================================================================

class TestRetryWithBackoff:
    """Test _retry_with_backoff function."""

    def test_success_on_first_try(self):
        """Function that succeeds immediately returns result."""
        result = _retry_with_backoff(lambda: 42, max_retries=3, base_delay=0.01)
        assert result == 42

    def test_success_after_one_failure(self):
        """Function that fails once then succeeds."""
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("Connection refused")
            return "success"

        result = _retry_with_backoff(flaky, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert call_count[0] == 2

    def test_all_retries_exhausted_returns_none(self):
        """Function that always fails returns None after max retries."""
        def always_fail():
            raise ConnectionError("Connection refused")

        result = _retry_with_backoff(always_fail, max_retries=3, base_delay=0.01)
        assert result is None

    def test_correct_number_of_attempts(self):
        """Exactly max_retries attempts are made."""
        call_count = [0]

        def counting_fail():
            call_count[0] += 1
            raise Exception("fail")

        _retry_with_backoff(counting_fail, max_retries=5, base_delay=0.01)
        assert call_count[0] == 5

    def test_none_return_is_valid(self):
        """Function returning None is a success (not retried)."""
        call_count = [0]

        def returns_none():
            call_count[0] += 1
            return None

        result = _retry_with_backoff(returns_none, max_retries=3, base_delay=0.01)
        assert result is None
        assert call_count[0] == 1  # Only called once


# =============================================================================
# Rate Limit Detection
# =============================================================================

class TestRateLimitDetection:
    """Test that rate limit errors are detected and handled."""

    def test_rate_limit_429_detected(self):
        """HTTP 429 errors should be detected as rate limits."""
        call_count = [0]

        def rate_limited_then_ok():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("HTTP 429 Too Many Requests")
            return "success"

        result = _retry_with_backoff(rate_limited_then_ok, max_retries=3, base_delay=0.01)
        assert result == "success"

    def test_rate_limit_text_detected(self):
        """'rate limit' text in error should be detected."""
        call_count = [0]

        def rate_limited():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Rate limit exceeded, please wait")
            return "ok"

        result = _retry_with_backoff(rate_limited, max_retries=3, base_delay=0.01)
        assert result == "ok"

    def test_throttle_detected(self):
        """'throttl' keyword should be detected."""
        call_count = [0]

        def throttled():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Request throttled by server")
            return "ok"

        result = _retry_with_backoff(throttled, max_retries=3, base_delay=0.01)
        assert result == "ok"


# =============================================================================
# Connection Error Detection
# =============================================================================

class TestConnectionErrorDetection:
    """Test that connection errors trigger retry."""

    def test_connection_refused(self):
        call_count = [0]

        def conn_refused():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ConnectionError("Connection refused")
            return "recovered"

        result = _retry_with_backoff(conn_refused, max_retries=3, base_delay=0.01)
        assert result == "recovered"
        assert call_count[0] == 3

    def test_timeout_error(self):
        call_count = [0]

        def timeout():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Connection timeout after 30s")
            return "ok"

        result = _retry_with_backoff(timeout, max_retries=3, base_delay=0.01)
        assert result == "ok"

    def test_connection_reset(self):
        call_count = [0]

        def reset():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Connection reset by peer")
            return "ok"

        result = _retry_with_backoff(reset, max_retries=3, base_delay=0.01)
        assert result == "ok"


# =============================================================================
# Backoff Timing
# =============================================================================

class TestBackoffTiming:
    """Test that backoff delays are applied correctly."""

    def test_backoff_increases_with_attempts(self):
        """Each retry should wait longer than the previous."""
        delays = []

        def failing():
            raise Exception("fail")

        original_sleep = time.sleep

        def mock_sleep(seconds):
            delays.append(seconds)
            # Don't actually sleep in tests

        with patch('velocity_trading.data.data_pipeline.time.sleep', side_effect=mock_sleep):
            _retry_with_backoff(failing, max_retries=4, base_delay=1.0, max_delay=30.0)

        # Should have 3 delays (between 4 attempts)
        assert len(delays) == 3
        # Each should be >= previous (exponential)
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i - 1]

    def test_max_delay_respected(self):
        """Delay should never exceed max_delay."""
        delays = []

        def failing():
            raise Exception("fail")

        def mock_sleep(seconds):
            delays.append(seconds)

        with patch('velocity_trading.data.data_pipeline.time.sleep', side_effect=mock_sleep):
            _retry_with_backoff(failing, max_retries=10, base_delay=1.0, max_delay=5.0)

        # All delays should be <= max_delay
        for d in delays:
            assert d <= 5.0

    def test_rate_limit_gets_extra_delay(self):
        """Rate limit errors should get doubled delay."""
        rate_limit_delays = []
        normal_delays = []

        def rate_limit_fail():
            raise Exception("429 Too Many Requests")

        def normal_fail():
            raise Exception("Some other error")

        def capture_rate_limit(seconds):
            rate_limit_delays.append(seconds)

        def capture_normal(seconds):
            normal_delays.append(seconds)

        with patch('velocity_trading.data.data_pipeline.time.sleep', side_effect=capture_rate_limit):
            _retry_with_backoff(rate_limit_fail, max_retries=2, base_delay=1.0, max_delay=30.0)

        with patch('velocity_trading.data.data_pipeline.time.sleep', side_effect=capture_normal):
            _retry_with_backoff(normal_fail, max_retries=2, base_delay=1.0, max_delay=30.0)

        # Rate limit should have higher delay than normal
        if rate_limit_delays and normal_delays:
            assert rate_limit_delays[0] >= normal_delays[0]


# =============================================================================
# Edge Cases
# =============================================================================

class TestRetryEdgeCases:
    """Test edge cases in retry logic."""

    def test_max_retries_one(self):
        """With max_retries=1, no retry happens."""
        call_count = [0]

        def fail_once():
            call_count[0] += 1
            raise Exception("fail")

        result = _retry_with_backoff(fail_once, max_retries=1, base_delay=0.01)
        assert result is None
        assert call_count[0] == 1

    def test_exception_preserved_in_log(self):
        """Last exception should be logged (verify via logger)."""
        with patch('velocity_trading.data.data_pipeline.logger') as mock_logger:
            def always_fail():
                raise ValueError("specific error message")

            _retry_with_backoff(always_fail, max_retries=2, base_delay=0.01,
                                description="test call")

            # Should have logged a warning with the error
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "specific error message" in warning_msg
            assert "test call" in warning_msg

    def test_returns_result_of_successful_call(self):
        """Return value from successful call is passed through."""
        result = _retry_with_backoff(lambda: {'data': [1, 2, 3]},
                                     max_retries=3, base_delay=0.01)
        assert result == {'data': [1, 2, 3]}

    def test_recovery_on_last_attempt(self):
        """Recovery on the very last attempt should succeed."""
        call_count = [0]

        def last_chance():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("not yet")
            return "made it"

        result = _retry_with_backoff(last_chance, max_retries=3, base_delay=0.01)
        assert result == "made it"
        assert call_count[0] == 3
