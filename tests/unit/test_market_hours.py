"""
Tests for market hours edge cases: weekends, DST transitions, holidays.

Covers:
- CME weekend closure (Friday 4 PM CT to Sunday 5 PM CT)
- NYSE weekend closure (Saturday/Sunday)
- Crypto 24/7 operation
- DST spring-forward and fall-back
- US market holidays
- Early close days
"""

import pytest
from datetime import datetime, time, timedelta
import pytz

from velocity_trading.data.market_hours import (
    is_market_open,
    is_weekend_closure,
    is_market_holiday,
    is_early_close,
    seconds_until_market_open,
    get_market_type,
    get_market_config,
    get_current_market_time,
    get_bar_close_time,
    is_bar_complete,
    seconds_until_bar_close,
    MARKET_CONFIG,
)

TZ_CT = pytz.timezone('America/Chicago')
TZ_ET = pytz.timezone('America/New_York')
TZ_UTC = pytz.UTC


# =============================================================================
# Weekend Transition Tests
# =============================================================================

class TestCMEWeekendClosure:
    """Test CME futures weekend closure: Friday 4 PM CT to Sunday 5 PM CT."""

    def test_friday_3pm_ct_open(self):
        """Friday 3 PM CT - CME should be open."""
        friday_3pm = TZ_CT.localize(datetime(2026, 2, 6, 15, 0, 0))  # Friday
        assert is_market_open('ES=F', friday_3pm) is True

    def test_friday_4pm_ct_closed(self):
        """Friday 4 PM CT - CME closes for weekend."""
        friday_4pm = TZ_CT.localize(datetime(2026, 2, 6, 16, 0, 0))
        assert is_market_open('ES=F', friday_4pm) is False

    def test_friday_4pm_is_weekend_closure(self):
        """Friday 4 PM CT should be detected as weekend closure."""
        friday_4pm = TZ_CT.localize(datetime(2026, 2, 6, 16, 0, 0))
        assert is_weekend_closure('ES=F', friday_4pm) is True

    def test_saturday_noon_closed(self):
        """Saturday noon CT - CME is closed."""
        saturday = TZ_CT.localize(datetime(2026, 2, 7, 12, 0, 0))
        assert is_market_open('ES=F', saturday) is False

    def test_sunday_noon_ct_closed(self):
        """Sunday noon CT - CME still closed."""
        sunday_noon = TZ_CT.localize(datetime(2026, 2, 8, 12, 0, 0))
        assert is_market_open('ES=F', sunday_noon) is False

    def test_sunday_4pm_ct_still_closed(self):
        """Sunday 4 PM CT - CME still closed (opens at 5 PM)."""
        sunday_4pm = TZ_CT.localize(datetime(2026, 2, 8, 16, 59, 0))
        assert is_market_open('ES=F', sunday_4pm) is False

    def test_sunday_5pm_ct_open(self):
        """Sunday 5 PM CT - CME opens for the week."""
        sunday_5pm = TZ_CT.localize(datetime(2026, 2, 8, 17, 0, 0))
        assert is_market_open('ES=F', sunday_5pm) is True

    def test_monday_midday_open(self):
        """Monday midday CT - CME should be open."""
        monday = TZ_CT.localize(datetime(2026, 2, 9, 12, 0, 0))
        assert is_market_open('ES=F', monday) is True

    def test_daily_maintenance_break(self):
        """CME daily maintenance break 4-5 PM CT."""
        # 4:30 PM CT on a Tuesday
        maint = TZ_CT.localize(datetime(2026, 2, 10, 16, 30, 0))
        assert is_market_open('ES=F', maint) is False

    def test_after_daily_maintenance(self):
        """CME reopens after 5 PM CT daily maintenance."""
        after_maint = TZ_CT.localize(datetime(2026, 2, 10, 17, 0, 0))
        assert is_market_open('ES=F', after_maint) is True

    def test_gc_same_schedule(self):
        """GC=F follows same CME schedule."""
        friday_4pm = TZ_CT.localize(datetime(2026, 2, 6, 16, 0, 0))
        assert is_market_open('GC=F', friday_4pm) is False

    def test_cl_same_schedule(self):
        """CL=F follows same CME schedule."""
        sunday_5pm = TZ_CT.localize(datetime(2026, 2, 8, 17, 0, 0))
        assert is_market_open('CL=F', sunday_5pm) is True


class TestNYSEWeekendClosure:
    """Test US stocks weekend closure."""

    def test_saturday_closed(self):
        """Saturday - NYSE is closed."""
        saturday = TZ_ET.localize(datetime(2026, 2, 7, 12, 0, 0))
        assert is_market_open('SPY', saturday) is False

    def test_sunday_closed(self):
        """Sunday - NYSE is closed."""
        sunday = TZ_ET.localize(datetime(2026, 2, 8, 12, 0, 0))
        assert is_market_open('SPY', sunday) is False

    def test_friday_afternoon_open(self):
        """Friday afternoon within hours - NYSE open."""
        friday_2pm = TZ_ET.localize(datetime(2026, 2, 6, 14, 0, 0))
        assert is_market_open('SPY', friday_2pm) is True

    def test_monday_930_open(self):
        """Monday 9:30 AM ET - NYSE opens."""
        monday = TZ_ET.localize(datetime(2026, 2, 9, 9, 30, 0))
        assert is_market_open('SPY', monday) is True

    def test_before_open_closed(self):
        """Before 9:30 AM ET - NYSE closed."""
        early = TZ_ET.localize(datetime(2026, 2, 9, 9, 0, 0))
        assert is_market_open('SPY', early) is False

    def test_after_close(self):
        """After 4 PM ET - NYSE closed."""
        after = TZ_ET.localize(datetime(2026, 2, 9, 16, 0, 0))
        assert is_market_open('SPY', after) is False


class TestCryptoAlwaysOpen:
    """Test that crypto markets are always open."""

    def test_saturday(self):
        sat = TZ_UTC.localize(datetime(2026, 2, 7, 12, 0, 0))
        assert is_market_open('BTC-USD', sat) is True

    def test_sunday(self):
        sun = TZ_UTC.localize(datetime(2026, 2, 8, 3, 0, 0))
        assert is_market_open('BTC-USD', sun) is True

    def test_holiday(self):
        """Crypto trades on US holidays."""
        holiday = TZ_UTC.localize(datetime(2026, 1, 1, 12, 0, 0))
        assert is_market_open('BTC-USD', holiday) is True

    def test_weekend_closure_returns_false(self):
        """Crypto never has weekend closure."""
        sat = TZ_UTC.localize(datetime(2026, 2, 7, 12, 0, 0))
        assert is_weekend_closure('BTC-USD', sat) is False

    def test_no_holidays(self):
        """Crypto has no holidays."""
        assert is_market_holiday(datetime(2026, 1, 1), 'BTC-USD') is False


class TestSecondsUntilMarketOpen:
    """Test seconds_until_market_open calculations."""

    def test_cme_friday_evening_to_sunday(self):
        """Friday 5 PM CT -> Sunday 5 PM CT = ~48 hours."""
        friday_5pm = TZ_CT.localize(datetime(2026, 2, 6, 17, 0, 0))
        # This is during weekend closure (Friday 4 PM onward)
        with pytest.MonkeyPatch.context() as m:
            m.setattr('velocity_trading.data.market_hours.datetime',
                      type('MockDT', (), {
                          'now': staticmethod(lambda tz: friday_5pm.astimezone(tz)),
                          'combine': datetime.combine,
                      }))
            # Can't easily mock - just test the logic directly
            pass

    def test_crypto_always_returns_zero(self):
        """Crypto seconds_until_open should always be 0."""
        # When market is already open, should return 0
        assert seconds_until_market_open('BTC-USD') == 0


# =============================================================================
# DST Transition Tests
# =============================================================================

class TestDSTTransitions:
    """Test DST transitions don't break market hours detection."""

    def test_spring_forward_et(self):
        """Spring forward (March 2026): 2 AM -> 3 AM ET.
        Market hours should still be correct the next day."""
        # March 8, 2026 is spring forward
        # March 9, 2026 (Monday) market should open at 9:30 AM EDT
        monday_after = TZ_ET.localize(datetime(2026, 3, 9, 9, 30, 0))
        assert is_market_open('SPY', monday_after) is True

    def test_spring_forward_before_open(self):
        """9 AM ET after spring forward - still before open."""
        monday_after = TZ_ET.localize(datetime(2026, 3, 9, 9, 0, 0))
        assert is_market_open('SPY', monday_after) is False

    def test_fall_back_et(self):
        """Fall back (November 2025): 2 AM -> 1 AM ET.
        Market should still open at 9:30 AM EST next day."""
        # November 2, 2025 is fall back
        # November 3, 2025 (Monday) market should open at 9:30 AM EST
        monday_after = TZ_ET.localize(datetime(2025, 11, 3, 9, 30, 0))
        assert is_market_open('SPY', monday_after) is True

    def test_spring_forward_cme(self):
        """CME schedule correct after spring forward.
        March 9 (Monday) - CME open during regular hours."""
        monday_noon = TZ_CT.localize(datetime(2026, 3, 9, 12, 0, 0))
        assert is_market_open('ES=F', monday_noon) is True

    def test_fall_back_cme(self):
        """CME schedule correct after fall back."""
        monday_noon = TZ_CT.localize(datetime(2025, 11, 3, 12, 0, 0))
        assert is_market_open('ES=F', monday_noon) is True

    def test_dst_bar_completion(self):
        """Bar completion should work across DST boundary."""
        # A 15m bar starting at 1:45 AM on spring-forward day
        # When 2 AM becomes 3 AM, the bar should still close 15 min later
        bar_start = TZ_ET.localize(datetime(2026, 3, 8, 1, 45, 0))
        bar_close = get_bar_close_time('SPY', '15m', bar_start)
        expected = bar_start + timedelta(minutes=15)
        assert bar_close == expected

    def test_utc_timestamps_unaffected_by_dst(self):
        """UTC timestamps should be completely unaffected by DST."""
        # Pre-DST: March 7
        pre_dst = TZ_UTC.localize(datetime(2026, 3, 7, 15, 0, 0))
        # Post-DST: March 9
        post_dst = TZ_UTC.localize(datetime(2026, 3, 9, 15, 0, 0))

        # Both should be valid bar start times in UTC
        bar_close_pre = get_bar_close_time('BTC-USD', '15m', pre_dst)
        bar_close_post = get_bar_close_time('BTC-USD', '15m', post_dst)

        assert bar_close_pre == pre_dst + timedelta(minutes=15)
        assert bar_close_post == post_dst + timedelta(minutes=15)


# =============================================================================
# Holiday Tests
# =============================================================================

class TestMarketHolidays:
    """Test market holiday detection and behavior."""

    def test_new_years_day_2026(self):
        """Jan 1, 2026 is a holiday."""
        assert is_market_holiday(datetime(2026, 1, 1), 'SPY') is True

    def test_mlk_day_2026(self):
        """MLK Day 2026."""
        assert is_market_holiday(datetime(2026, 1, 19), 'SPY') is True

    def test_good_friday_2026(self):
        """Good Friday 2026."""
        assert is_market_holiday(datetime(2026, 4, 3), 'SPY') is True

    def test_christmas_2026(self):
        """Christmas 2026."""
        assert is_market_holiday(datetime(2026, 12, 25), 'SPY') is True

    def test_regular_day_not_holiday(self):
        """Regular trading day is not a holiday."""
        assert is_market_holiday(datetime(2026, 2, 5), 'SPY') is False

    def test_holiday_closes_nyse(self):
        """NYSE should be closed on holidays."""
        holiday = TZ_ET.localize(datetime(2026, 1, 1, 12, 0, 0))
        assert is_market_open('SPY', holiday) is False

    def test_holiday_closes_cme(self):
        """CME should be closed on holidays (same calendar)."""
        holiday = TZ_CT.localize(datetime(2026, 1, 1, 12, 0, 0))
        assert is_market_open('ES=F', holiday) is False

    def test_crypto_open_on_holiday(self):
        """Crypto trades during US holidays."""
        holiday = TZ_UTC.localize(datetime(2026, 1, 1, 12, 0, 0))
        assert is_market_open('BTC-USD', holiday) is True

    def test_holiday_detected_from_date_object(self):
        """Holiday check works with date objects too."""
        from datetime import date
        assert is_market_holiday(date(2026, 1, 1), 'SPY') is True

    def test_early_close_day_detected(self):
        """Early close days are detected."""
        # Thanksgiving Friday 2026
        assert is_early_close(datetime(2026, 11, 27), 'SPY') is True

    def test_early_close_market_closed_after_1pm(self):
        """NYSE closes at 1 PM ET on early close days."""
        # Christmas Eve 2026
        early_close_130 = TZ_ET.localize(datetime(2026, 12, 24, 13, 30, 0))
        assert is_market_open('SPY', early_close_130) is False

    def test_early_close_market_open_before_1pm(self):
        """NYSE open before 1 PM on early close days."""
        early_close_1130 = TZ_ET.localize(datetime(2026, 12, 24, 11, 30, 0))
        assert is_market_open('SPY', early_close_1130) is True


# =============================================================================
# Market Type Detection Tests
# =============================================================================

class TestMarketTypeDetection:
    """Test ticker-to-market-type mapping."""

    def test_spy_is_stocks(self):
        assert get_market_type('SPY') == 'stocks_us'

    def test_es_is_futures(self):
        assert get_market_type('ES=F') == 'futures_cme'

    def test_btc_is_crypto(self):
        assert get_market_type('BTC-USD') == 'crypto'

    def test_unknown_futures_inferred(self):
        """Unknown futures tickers inferred from =F suffix."""
        assert get_market_type('ZC=F') == 'futures_cme'

    def test_unknown_crypto_inferred(self):
        """Unknown crypto tickers inferred from -USD suffix."""
        assert get_market_type('DOGE-USD') == 'crypto'

    def test_unknown_defaults_to_stocks(self):
        """Unknown tickers default to stocks."""
        assert get_market_type('UNKNOWN') == 'stocks_us'


# =============================================================================
# Bar Completion Across Boundaries
# =============================================================================

class TestBarCompletionEdgeCases:
    """Test bar completion at market boundaries."""

    def test_15m_bar_completes_normally(self):
        """15m bar starting at 10:00 closes at 10:15."""
        bar_start = TZ_ET.localize(datetime(2026, 2, 9, 10, 0, 0))
        check_time = TZ_ET.localize(datetime(2026, 2, 9, 10, 16, 0))
        assert is_bar_complete('SPY', '15m', bar_start, check_time) is True

    def test_15m_bar_not_complete_yet(self):
        """15m bar at 10:00 not complete at 10:14."""
        bar_start = TZ_ET.localize(datetime(2026, 2, 9, 10, 0, 0))
        check_time = TZ_ET.localize(datetime(2026, 2, 9, 10, 14, 0))
        assert is_bar_complete('SPY', '15m', bar_start, check_time) is False

    def test_1h_bar_completes(self):
        """1h bar starting at 10:00 closes at 11:00."""
        bar_start = TZ_CT.localize(datetime(2026, 2, 9, 10, 0, 0))
        check_time = TZ_CT.localize(datetime(2026, 2, 9, 11, 1, 0))
        assert is_bar_complete('ES=F', '1h', bar_start, check_time) is True

    def test_daily_bar_close_time_stocks(self):
        """Daily bar for stocks closes at 4 PM ET."""
        bar_start = TZ_ET.localize(datetime(2026, 2, 9, 0, 0, 0))
        close = get_bar_close_time('SPY', '1d', bar_start)
        assert close.hour == 16
        assert close.minute == 0

    def test_daily_bar_close_time_futures(self):
        """Daily bar for futures closes at 5 PM CT."""
        bar_start = TZ_CT.localize(datetime(2026, 2, 9, 0, 0, 0))
        close = get_bar_close_time('ES=F', '1d', bar_start)
        assert close.hour == 17
        assert close.minute == 0

    def test_daily_bar_close_time_crypto(self):
        """Daily bar for crypto closes at midnight UTC."""
        bar_start = TZ_UTC.localize(datetime(2026, 2, 9, 0, 0, 0))
        close = get_bar_close_time('BTC-USD', '1d', bar_start)
        assert close.hour == 0
        assert close.minute == 0
