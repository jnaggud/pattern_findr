"""
Velocity-Based Live Trader

IMPORTANT: This is a SIGNAL-ONLY system. It does NOT execute actual trades.
It monitors price data, generates trading signals, and sends alerts to Discord.
You must manually execute trades if you choose to act on the signals.

Features:
- Interactive strategy selection at startup
- Permanent strategy storage in velocity_strategies/
- Config hot-reloading during runtime
- Scheduled status updates (market open, mid-day, close, hourly)
- Historical backtest on startup with charts
- LONG-only trading signals (no SHORT positions)
- Production-ready file locking and error handling

Usage:
    python velocity_live_trader.py [--config path/to/config.json]
"""

import json
import os
import time
import argparse
import logging
from datetime import datetime, timedelta, timezone

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system environment variables
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import io
import shutil
import pytz  # Timezone handling for market hours

# Import production utilities for safe operations
try:
    from velocity_production_utils import (
        safe_json_write, safe_json_read, get_logger,
        is_market_open, get_market_time, HeartbeatMonitor,
        load_webhook_from_env, print_signal_only_disclaimer,
        SIGNAL_ONLY_DISCLAIMER, is_market_holiday, is_early_close
    )
    PRODUCTION_UTILS_AVAILABLE = True
except ImportError:
    PRODUCTION_UTILS_AVAILABLE = False
    print("Warning: velocity_production_utils.py not found. Using basic operations.")
    def get_logger(name):
        return logging.getLogger(name)
    def is_market_open(ticker):
        return {"is_open": True, "reason": "Unknown"}
    def get_market_time():
        return datetime.now()
    def load_webhook_from_env(name):
        return None
    def print_signal_only_disclaimer():
        print("WARNING: This is a SIGNAL-ONLY system. It does NOT execute trades.")
    def is_market_holiday(date_to_check=None, ticker="SPY"):
        return False, None
    def is_early_close(date_to_check=None, ticker="SPY"):
        return False, None
    def safe_json_write(filepath, data, indent=2):
        """Fallback: atomic JSON write with file locking."""
        import tempfile
        import shutil
        import fcntl
        try:
            # Use temp file + rename for atomic writes
            dir_name = os.path.dirname(filepath) or '.'
            fd, temp_path = tempfile.mkstemp(suffix='.json', dir=dir_name)
            try:
                with os.fdopen(fd, 'w') as f:
                    # Lock the temp file while writing
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(data, f, indent=indent, default=str)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                # Atomic rename (on Unix/Mac, this is atomic)
                shutil.move(temp_path, filepath)
                return True
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise e
        except Exception as e:
            print(f"Error writing {filepath}: {e}")
            return False
    def safe_json_read(filepath, default=None):
        """Fallback: JSON read with file locking."""
        import fcntl
        try:
            with open(filepath, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                data = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return data
        except Exception:
            return default

# Import v9 regime detection (optional — graceful fallback if not available)
try:
    from velocity_trading.core.regime_detector_v9 import (
        classify_regimes, REGIME_NAMES, REGIME_UPTREND, REGIME_DOWNTREND, REGIME_CHOP,
    )
    V9_REGIME_AVAILABLE = True
except ImportError:
    V9_REGIME_AVAILABLE = False

# Import oscillator calculations directly from Streamlit source of truth
from oscillator_predictor_page import (
    create_composite_oscillator,
    calculate_rsi,
    calculate_williams_r,
    calculate_cci,
    calculate_stochastic,
    calculate_roc,
    calculate_momentum,
    calculate_bb_position,
    calculate_adx_trend,
    calculate_mfi,
)

# Import novel indicators for advanced oscillators
try:
    from novel_indicators import (
        calculate_arwo, calculate_dco, calculate_vcmo,
        calculate_ics, calculate_mji, calculate_prf, calculate_ewaf, calculate_kfif
    )
    NOVEL_INDICATORS_AVAILABLE = True
except ImportError:
    NOVEL_INDICATORS_AVAILABLE = False
    print("Warning: novel_indicators.py not found. Using standard composite oscillator only.")

# Import AlertManager for rich Discord alerts with charts
try:
    from alert_system import AlertManager
    ALERT_MANAGER_AVAILABLE = True
except ImportError:
    ALERT_MANAGER_AVAILABLE = False
    print("Warning: alert_system.py not found. Using basic Discord alerts.")

# Data source
try:
    from polygon import RESTClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    print("Warning: polygon-api-client not installed. Using yfinance fallback.")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Databento for futures data (preferred over yfinance for CME futures)
try:
    import databento as db
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False
    print("Warning: databento not installed. Using yfinance for futures. Run: pip install databento")

# Databento API key from environment (with fallback)
DATABENTO_API_KEY = os.environ.get("DATABENTO_API_KEY", "")

# Futures ticker mapping: yfinance ticker -> Databento continuous contract symbol
# Using .n.0 = nearest/front month continuous (better data coverage than .c.0)
# .n.0 provides complete OHLCV data vs .c.0 which has sparse data
FUTURES_TICKER_MAP = {
    'ES=F': 'ES.n.0',   # E-mini S&P 500 continuous
    'NQ=F': 'NQ.n.0',   # E-mini Nasdaq 100 continuous
    'YM=F': 'YM.n.0',   # E-mini Dow continuous
    'RTY=F': 'RTY.n.0', # E-mini Russell 2000 continuous
    'GC=F': 'GC.n.0',   # Gold continuous
    'SI=F': 'SI.n.0',   # Silver continuous
    'CL=F': 'CL.n.0',   # Crude Oil continuous
    'NG=F': 'NG.n.0',   # Natural Gas continuous
    'ZB=F': 'ZB.n.0',   # 30-Year Treasury Bond continuous
    'ZN=F': 'ZN.n.0',   # 10-Year Treasury Note continuous
    'ZC=F': 'ZC.n.0',   # Corn continuous
    'ZS=F': 'ZS.n.0',   # Soybeans continuous
    'ZW=F': 'ZW.n.0',   # Wheat continuous
    '6E=F': '6E.n.0',   # Euro FX continuous
    '6J=F': '6J.n.0',   # Japanese Yen continuous
}

def is_futures_ticker(ticker: str) -> bool:
    """Check if ticker is a futures contract (ends with =F)."""
    return ticker.endswith('=F') or ticker in FUTURES_TICKER_MAP

# Default Discord webhook (same as live_trader.py)
DEFAULT_DISCORD_WEBHOOK = ""

# Legal disclaimer appended to all Discord messages
LEGAL_DISCLAIMER = (
    "\n\n_⚠️ **Disclaimer:** This is not financial advice. Past performance does not "
    "guarantee future results. Trading involves substantial risk of loss. Only trade "
    "with capital you can afford to lose. For educational purposes only._"
)

# Secondary webhooks for Haus Hedge server (posts to both servers)
HAUS_HEDGE_WEBHOOKS = {
    "velocity_SPY_1y": "",
    "velocity_SPY_2y": "",
    "velocity_SPY_5y": "",
    "velocity_BTC_1y": "",
    "velocity_BTC_2y": "",
    "velocity_BTC_5y": "",
    # ES=F (E-mini S&P 500 Futures) 15-min intraday
    "velocity_ES=F_15m": "",
    "velocity_ES=F_velocity_crossover_or_zone_15min": "",
    "velocity_ES=F_any_reversal_sl.72": "",
    # BTC 15-min intraday
    "velocity_BTC-USD_15m": "",
    "velocity_BTC_15m": "",
    "velocity_BTC-USD_any_reversal_15min": "",
    "velocity_BTC-USD_velocity_crossover_or_zone_sl.97": "",
}

# Strategy storage directories
VELOCITY_STRATEGIES_DIR = "velocity_strategies"
PRODUCTION_CONFIG_PATH = "production_env/velocity_config.json"

# Sierra Chart Bridge Configuration (opt-in via environment variable)
SIERRA_BRIDGE_ENABLED = os.environ.get("SIERRA_BRIDGE_ENABLED", "false").lower() == "true"
SIERRA_SIGNAL_FILE = os.environ.get("SIERRA_SIGNAL_FILE", os.path.expanduser("~/pfr_signals.json"))

# Import Sierra Chart bridge if enabled
if SIERRA_BRIDGE_ENABLED:
    try:
        from sierra_chart_bridge import publish_signal_to_sierra, build_sierra_signal
        print(f"Sierra Chart bridge enabled. Signals will be published to: {SIERRA_SIGNAL_FILE}")
    except ImportError:
        print("Warning: sierra_chart_bridge.py not found. Sierra Chart integration disabled.")
        SIERRA_BRIDGE_ENABLED = False


def is_data_stale_for_futures(data_end_date, ticker=""):
    """
    Check if futures data is genuinely stale vs normal weekend/holiday gap.

    CME Globex schedule:
    - Opens: Sunday 6:00 PM ET
    - Closes: Friday 5:00 PM ET
    - Saturday: CLOSED
    - Holidays: Various closures (MLK Day, etc.)

    Returns: (is_stale: bool, warning_msg: str or None)
    """
    today = datetime.now().date()
    days_behind = (today - data_end_date).days
    weekday = today.weekday()  # 0=Monday, 5=Saturday, 6=Sunday

    # Data from today or yesterday is never considered stale
    if days_behind <= 1:
        return False, None

    # Check if today is a holiday - if so, data being behind is expected
    is_holiday_today, holiday_name = is_market_holiday(today, ticker)
    if is_holiday_today:
        # On a holiday, data from 2-3 days ago is normal
        if days_behind <= 3:
            return False, None

    # Check if yesterday was a holiday (day after holiday, data might be 2+ days old)
    yesterday = today - timedelta(days=1)
    is_holiday_yesterday, _ = is_market_holiday(yesterday, ticker)
    if is_holiday_yesterday:
        # Day after holiday, data being 2-3 days old is normal
        if days_behind <= 3:
            return False, None

    # On Saturday: data from Friday (1 day behind) is normal
    # On Sunday before 6PM ET: data from Friday (2 days behind) is normal
    if weekday == 5:  # Saturday
        # Friday data is expected (1-2 days behind)
        if days_behind <= 2:
            return False, None
    elif weekday == 6:  # Sunday
        # Friday data is expected (2 days behind is normal)
        if days_behind <= 2:
            return False, None
    elif weekday == 0:  # Monday
        # Friday data might still be showing early Monday (3 days behind before market opens)
        hour = datetime.now().hour
        if days_behind <= 3 and hour < 10:  # Before 10 AM
            return False, None

    # Long weekends (holiday + weekend combo like MLK Day Monday)
    # Check if there's been a recent holiday within the gap
    for i in range(days_behind):
        check_date = today - timedelta(days=i)
        is_holiday_in_gap, _ = is_market_holiday(check_date, ticker)
        if is_holiday_in_gap:
            # If there was a holiday in the gap, allow extra days
            if days_behind <= 4:
                return False, None

    # If we get here, data is genuinely stale
    return True, f"\n⚠️ _Data is {days_behind} days behind (yfinance delay)_"


# =============================================================================
# INSTRUMENT-SPECIFIC MARKET HOURS CONFIGURATION
# =============================================================================

MARKET_HOURS_CONFIG = {
    # CME Globex Futures (ES, GC, CL, NQ, etc.)
    # Open: Sunday 5:00 PM CT, Close: Friday 5:00 PM CT
    # Daily break: 4:00 PM - 5:00 PM CT (maintenance)
    # Daily bar closes at 5:00 PM CT (when new session starts)
    'futures_cme': {
        'type': 'futures',
        'timezone': 'America/Chicago',
        'weekend_close': {'day': 4, 'hour': 17},   # Friday 5 PM CT
        'weekend_open': {'day': 6, 'hour': 17},    # Sunday 5 PM CT
        'daily_break_start': {'hour': 16, 'minute': 0},   # 4:00 PM CT
        'daily_break_end': {'hour': 17, 'minute': 0},     # 5:00 PM CT
        'daily_bar_close': {'hour': 17, 'minute': 0},     # 5:00 PM CT - when daily bar is complete
    },
    # US Stock Market (SPY, QQQ, etc.)
    # Open: 9:30 AM ET, Close: 4:00 PM ET
    # Daily bar closes at 4:00 PM ET
    # Closed weekends
    'stocks_us': {
        'type': 'stocks',
        'timezone': 'America/New_York',
        'market_open': {'hour': 9, 'minute': 30},
        'market_close': {'hour': 16, 'minute': 0},
        'daily_bar_close': {'hour': 16, 'minute': 0},     # 4:00 PM ET - when daily bar is complete
        'closed_weekends': True,
    },
    # Crypto (BTC-USD, ETH-USD, etc.)
    # 24/7 trading
    # Daily bar closes at midnight UTC
    'crypto': {
        'type': 'crypto',
        'timezone': 'UTC',
        'always_open': True,
        'daily_bar_close': {'hour': 0, 'minute': 0},      # 00:00 UTC - when daily bar is complete
    },
}

# Interval to minutes mapping
INTERVAL_MINUTES = {
    '1m': 1, '2m': 2, '5m': 5, '15m': 15, '30m': 30,
    '1h': 60, '90m': 90, '4h': 240, '1d': 1440,
}

def get_ticker_market_type(ticker: str) -> str:
    """Determine the market type for a given ticker."""
    ticker_upper = ticker.upper()

    # Futures detection - use is_futures_ticker() for consistency
    # Checks ticker.endswith('=F') or ticker in FUTURES_TICKER_MAP
    if is_futures_ticker(ticker):
        return 'futures_cme'

    # Crypto detection - all crypto tickers have -USD suffix (BTC-USD, ETH-USD, etc.)
    if '-USD' in ticker_upper:
        return 'crypto'

    # Default to US stocks
    return 'stocks_us'


def get_bar_close_time(ticker: str, interval: str, bar_start: datetime = None) -> datetime:
    """
    Calculate when a bar closes for a given ticker and interval.

    For daily bars: Returns the market close time for that asset
    For intraday bars: Returns bar_start + interval duration

    Args:
        ticker: The ticker symbol
        interval: Data interval ('1d', '15m', etc.)
        bar_start: Start time of the bar (required for intraday)

    Returns:
        datetime of when the bar closes (in the asset's timezone, then converted to local)
    """
    market_type = get_ticker_market_type(ticker)
    config = MARKET_HOURS_CONFIG.get(market_type, MARKET_HOURS_CONFIG['stocks_us'])

    if interval == '1d':
        # Daily bar - closes at market close time
        tz = pytz.timezone(config['timezone'])
        now = datetime.now(tz)

        close_config = config.get('daily_bar_close', {'hour': 16, 'minute': 0})
        close_time = now.replace(
            hour=close_config['hour'],
            minute=close_config['minute'],
            second=0,
            microsecond=0
        )

        # If we're past today's close, return today's close time
        # If we're before today's close, return today's close time
        return close_time

    else:
        # Intraday bar - closes at bar_start + interval
        if bar_start is None:
            raise ValueError("bar_start required for intraday intervals")

        interval_minutes = INTERVAL_MINUTES.get(interval, 15)
        return bar_start + timedelta(minutes=interval_minutes)


def is_bar_complete(ticker: str, interval: str, bar_start: datetime = None) -> bool:
    """
    Check if a bar is complete (closed) and ready for signal evaluation.

    Signal should be generated the moment after a bar closes.

    Args:
        ticker: The ticker symbol
        interval: Data interval ('1d', '15m', etc.)
        bar_start: Start time of the bar (required for intraday)

    Returns:
        True if the bar is complete (current time > bar close time)
    """
    market_type = get_ticker_market_type(ticker)
    config = MARKET_HOURS_CONFIG.get(market_type, MARKET_HOURS_CONFIG['stocks_us'])

    if interval == '1d':
        # Daily bar - check if we're past market close
        tz = pytz.timezone(config['timezone'])
        now = datetime.now(tz)

        close_config = config.get('daily_bar_close', {'hour': 16, 'minute': 0})
        today_close = now.replace(
            hour=close_config['hour'],
            minute=close_config['minute'],
            second=0,
            microsecond=0
        )

        # Bar is complete if current time is past close time
        return now > today_close

    else:
        # Intraday bar - check if we're past bar_start + interval
        if bar_start is None:
            return False

        bar_close = get_bar_close_time(ticker, interval, bar_start)

        # Normalize bar_close to be timezone-aware if needed
        if bar_close.tzinfo is None:
            bar_close = pytz.UTC.localize(bar_close)

        now = datetime.now(pytz.UTC)
        return now > bar_close


def get_signal_check_time(ticker: str, interval: str) -> str:
    """
    Get a human-readable description of when signals should be checked.

    Args:
        ticker: The ticker symbol
        interval: Data interval ('1d', '15m', etc.)

    Returns:
        Human-readable string describing when to check for signals
    """
    market_type = get_ticker_market_type(ticker)
    config = MARKET_HOURS_CONFIG.get(market_type, MARKET_HOURS_CONFIG['stocks_us'])

    if interval == '1d':
        close_config = config.get('daily_bar_close', {'hour': 16, 'minute': 0})
        tz_name = config['timezone'].split('/')[-1].replace('_', ' ')
        return f"Daily signals checked after {close_config['hour']:02d}:{close_config['minute']:02d} {tz_name}"
    else:
        return f"{interval} signals checked after each bar closes"


def seconds_until_next_bar_close(ticker: str, interval: str) -> int:
    """
    Calculate seconds until the next bar closes.

    For smart scheduling: sleep until bar close + buffer, then check for signals.
    This minimizes delay between signal generation and notification.

    Args:
        ticker: The ticker symbol
        interval: Data interval ('1d', '15m', etc.)

    Returns:
        Seconds until next bar close (minimum 5 seconds to avoid tight loops)
    """
    market_type = get_ticker_market_type(ticker)
    config = MARKET_HOURS_CONFIG.get(market_type, MARKET_HOURS_CONFIG['stocks_us'])
    tz = pytz.timezone(config['timezone'])
    now = datetime.now(tz)

    if interval == '1d':
        # Daily bar - closes at market close time
        close_config = config.get('daily_bar_close', {'hour': 16, 'minute': 0})
        today_close = now.replace(
            hour=close_config['hour'],
            minute=close_config['minute'],
            second=0,
            microsecond=0
        )

        if now >= today_close:
            # Already past today's close, next close is tomorrow
            # For crypto (24/7), next close is in ~24h
            # For stocks/futures, need to account for weekends
            tomorrow_close = today_close + timedelta(days=1)
            seconds = (tomorrow_close - now).total_seconds()
        else:
            seconds = (today_close - now).total_seconds()

    else:
        # Intraday bar - calculate next bar boundary
        interval_minutes = INTERVAL_MINUTES.get(interval, 15)

        # Find current bar start time (round down to interval boundary)
        minutes_since_midnight = now.hour * 60 + now.minute
        current_bar_start_minutes = (minutes_since_midnight // interval_minutes) * interval_minutes

        # Next bar close = current bar start + interval
        next_close_minutes = current_bar_start_minutes + interval_minutes

        # Handle day rollover
        next_close_hour = next_close_minutes // 60
        next_close_minute = next_close_minutes % 60

        if next_close_hour >= 24:
            # Rolls over to next day
            next_close = now.replace(hour=0, minute=next_close_minute, second=0, microsecond=0) + timedelta(days=1)
        else:
            next_close = now.replace(hour=next_close_hour, minute=next_close_minute, second=0, microsecond=0)

        seconds = (next_close - now).total_seconds()

    # Minimum 5 seconds to avoid tight loops, maximum reasonable wait
    return max(5, min(int(seconds), 86400))


def is_futures_market_closed(bar_timestamp, ticker: str = "") -> bool:
    """
    Check if a bar timestamp falls during futures market closure.

    CME Globex Schedule (for ES, GC, CL, NQ, etc.):
    - CLOSED: Friday 5:00 PM CT to Sunday 5:00 PM CT
    - Daily maintenance break: 4:00 PM - 5:00 PM CT
    - DST-aware: Uses Central Time (America/Chicago) for accurate calculations

    Use this to detect and filter "phantom bars" - bars with forward-filled
    stale prices that yfinance sometimes returns during market closure.

    Args:
        bar_timestamp: The timestamp of the bar (can be tz-aware or naive)
        ticker: The ticker symbol (for future expansion to different schedules)

    Returns:
        True if the bar is during market closure (phantom bar), False otherwise
    """
    import pytz

    # Check if this is a futures ticker
    if not is_futures_ticker(ticker):
        return False  # Not a futures ticker, no closure check needed

    # Convert to pandas timestamp
    ts = pd.to_datetime(bar_timestamp)

    # Convert to Central Time (CME time) for DST-aware comparison
    ct = pytz.timezone('America/Chicago')

    if ts.tzinfo is None:
        # Assume UTC if no timezone info
        ts = pytz.utc.localize(ts)

    # Convert to Central Time
    ts_ct = ts.astimezone(ct)

    weekday_ct = ts_ct.weekday()  # 0=Monday, 5=Saturday, 6=Sunday
    hour_ct = ts_ct.hour
    minute_ct = ts_ct.minute

    # CME Globex closes Friday 5:00 PM CT and reopens Sunday 5:00 PM CT
    # CLOSED: Friday 17:00 CT to Sunday 17:00 CT

    # Saturday: Always closed (all hours in CT)
    if weekday_ct == 5:
        return True

    # Sunday before 5:00 PM CT (17:00)
    if weekday_ct == 6 and hour_ct < 17:
        return True

    # Friday at or after 5:00 PM CT (17:00)
    if weekday_ct == 4 and hour_ct >= 17:
        return True

    # Daily maintenance break: 4:00 PM - 5:00 PM CT (weekdays)
    # Note: For daily bars this doesn't matter, but for intraday it can cause phantom bars
    if weekday_ct < 5:  # Monday-Friday
        if hour_ct == 16:  # 4:00 PM - 4:59 PM CT
            return True

    return False


def is_stock_market_closed(bar_timestamp, ticker: str = "") -> bool:
    """
    Check if a bar timestamp falls outside US stock market hours.

    NYSE/NASDAQ Schedule:
    - Open: 9:30 AM ET, Close: 4:00 PM ET
    - Closed: Weekends (Saturday/Sunday)
    - DST-aware: Uses Eastern Time (America/New_York)

    Args:
        bar_timestamp: The timestamp of the bar
        ticker: The ticker symbol

    Returns:
        True if outside market hours, False if during trading hours
    """
    import pytz

    ts = pd.to_datetime(bar_timestamp)
    et = pytz.timezone('America/New_York')

    if ts.tzinfo is None:
        ts = pytz.utc.localize(ts)

    ts_et = ts.astimezone(et)
    weekday = ts_et.weekday()
    hour = ts_et.hour
    minute = ts_et.minute

    # Weekends: Closed
    if weekday >= 5:  # Saturday=5, Sunday=6
        return True

    # Before market open (9:30 AM ET)
    if hour < 9 or (hour == 9 and minute < 30):
        return True

    # At or after market close (4:00 PM ET)
    if hour >= 16:
        return True

    return False


def is_market_closed(bar_timestamp, ticker: str = "") -> bool:
    """
    Unified market hours check for any instrument type.

    Routes to the appropriate market hours check based on ticker type:
    - Futures: CME Globex schedule (with daily break)
    - Stocks: NYSE/NASDAQ schedule (9:30 AM - 4:00 PM ET)
    - Crypto: Always open (24/7)

    Args:
        bar_timestamp: The timestamp to check
        ticker: The ticker symbol

    Returns:
        True if market is closed, False if open
    """
    market_type = get_ticker_market_type(ticker)

    if market_type == 'futures_cme':
        return is_futures_market_closed(bar_timestamp, ticker)
    elif market_type == 'stocks_us':
        return is_stock_market_closed(bar_timestamp, ticker)
    elif market_type == 'crypto':
        return False  # Crypto is 24/7
    else:
        return False  # Default: assume open


def filter_phantom_bars(df: pd.DataFrame, ticker: str = "", interval: str = "") -> pd.DataFrame:
    """
    Filter out phantom bars (bars during market closure) from a DataFrame.

    Phantom bars are bars with forward-filled stale prices that yfinance
    sometimes returns during market closure. These cause false signals.

    IMPORTANT: Only applies to INTRADAY data. Daily bars should NOT be filtered
    because their timestamps (typically midnight) don't represent actual trading time.

    Uses instrument-specific market hours:
    - Futures: CME Globex schedule (weekend + daily maintenance break)
    - Stocks: NYSE/NASDAQ schedule (9:30 AM - 4:00 PM ET, weekends)
    - Crypto: No filtering (24/7)

    Args:
        df: DataFrame with datetime index
        ticker: Ticker symbol to determine market hours
        interval: Data interval (e.g., '1d', '15m'). Daily intervals skip filtering.

    Returns:
        DataFrame with phantom bars removed
    """
    if df.empty:
        return df

    # CRITICAL: Skip filtering for daily bars - their timestamps don't represent intraday times
    # Daily bars typically have midnight timestamps which would incorrectly be filtered
    if interval in ['1d', '1D', 'daily', 'd', 'D']:
        print(f"   [filter_phantom_bars] Skipping filter for daily interval: {interval}")
        return df

    # Debug: Log when filtering is NOT skipped
    print(f"   [filter_phantom_bars] interval='{interval}' (type={type(interval).__name__}) - will filter")

    # Auto-detect daily interval if not specified (check time delta between bars)
    if not interval and len(df) >= 2:
        time_delta = (df.index[1] - df.index[0]).total_seconds()
        if time_delta >= 86400:  # 24 hours or more = daily data
            return df

    market_type = get_ticker_market_type(ticker)

    # Crypto is 24/7, no filtering needed
    if market_type == 'crypto':
        return df

    # Create mask for valid bars (not during market closure)
    valid_mask = ~df.index.to_series().apply(lambda ts: is_market_closed(ts, ticker))

    filtered_df = df[valid_mask]

    removed_count = len(df) - len(filtered_df)
    if removed_count > 0:
        market_label = "market closure" if market_type == 'futures_cme' else "outside market hours"
        print(f"   ⚠️ Filtered {removed_count} phantom bars ({market_label}) from {ticker}")

    return filtered_df


# Global Chicago timezone - ALL timestamps should use this
CHICAGO_TZ = pytz.timezone('America/Chicago')


def to_chicago_time(ts):
    """Convert any timestamp to Chicago time with timezone info.

    This is the STANDARD for all timestamp storage and comparison.
    All locked_backtest, trade_state, and display times should use this.
    """
    if ts is None:
        return None
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        # Naive timestamp - assume it's UTC and convert to Chicago
        ts = pytz.UTC.localize(ts)
    # Convert to Chicago time
    return ts.astimezone(CHICAGO_TZ)


def to_chicago_str(ts):
    """Convert any timestamp to Chicago time string format.

    Returns format: '2026-01-21 11:45:00-0600' (with timezone offset)
    """
    ct = to_chicago_time(ts)
    if ct is None:
        return None
    return ct.strftime('%Y-%m-%d %H:%M:%S%z')


def normalize_tz(ts):
    """Convert timestamp to tz-naive Chicago time for comparison.

    For tz-aware timestamps, converts to Chicago time then removes timezone info.
    For tz-naive timestamps, assumes UTC and converts to Chicago time.

    NOTE: All comparisons should use Chicago time as the standard.
    """
    if ts is None:
        return None
    ts = pd.to_datetime(ts)
    if ts.tzinfo is not None:
        # Convert to Chicago time, then remove timezone info for comparison
        return ts.astimezone(CHICAGO_TZ).replace(tzinfo=None)
    else:
        # Naive timestamp - assume UTC, convert to Chicago
        utc_ts = pytz.UTC.localize(ts)
        return utc_ts.astimezone(CHICAGO_TZ).replace(tzinfo=None)


def calculate_exit_stats(exits: list, exclude_missed: bool = True) -> dict:
    """
    Centralized stats calculation from a list of exit trades.

    This is the SINGLE SOURCE OF TRUTH for calculating trading statistics.
    All stats displayed on Discord, charts, and trade logs should use this function.

    Args:
        exits: List of exit trade dicts with 'pnl', 'date', 'reason' fields
        exclude_missed: If True, filter out [SYNC]/[MISSED] trades from stats

    Returns:
        Dict with: num_trades, win_rate, total_return (compounded), profit_factor,
                   winners, losers, avg_win, avg_loss
    """
    if not exits:
        return {
            'num_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'profit_factor': 0,
            'winners': 0,
            'losers': 0,
            'avg_win': 0,
            'avg_loss': 0,
        }

    # Filter out missed/sync trades if requested
    if exclude_missed:
        filtered_exits = [
            e for e in exits
            if not e.get('missed')
            and '[SYNC]' not in str(e.get('reason', ''))
            and '[MISSED]' not in str(e.get('reason', ''))
        ]
    else:
        filtered_exits = exits

    if not filtered_exits:
        return {
            'num_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'profit_factor': 0,
            'winners': 0,
            'losers': 0,
            'avg_win': 0,
            'avg_loss': 0,
        }

    # Calculate stats
    winners = [e for e in filtered_exits if e.get('pnl', 0) > 0]
    losers = [e for e in filtered_exits if e.get('pnl', 0) <= 0]

    num_trades = len(filtered_exits)
    win_rate = (len(winners) / num_trades) * 100 if num_trades > 0 else 0

    # Compounded return (matches equity curve calculation)
    sorted_exits = sorted(filtered_exits, key=lambda x: str(x.get('date', '')))
    equity = 1.0
    for exit_trade in sorted_exits:
        equity *= (1 + exit_trade.get('pnl', 0) / 100)
    total_return = (equity - 1) * 100

    # Profit factor
    total_wins = sum(e.get('pnl', 0) for e in winners) if winners else 0
    total_losses = abs(sum(e.get('pnl', 0) for e in losers)) if losers else 0.001
    profit_factor = total_wins / total_losses if total_losses > 0 else (999.99 if total_wins > 0 else 0)

    # Averages
    avg_win = total_wins / len(winners) if winners else 0
    avg_loss = total_losses / len(losers) if losers else 0

    return {
        'num_trades': num_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'profit_factor': profit_factor,
        'winners': len(winners),
        'losers': len(losers),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
    }


def timestamps_equal(ts1, ts2) -> bool:
    """
    Safely compare two timestamps for equality.

    Handles different formats, timezones, and string representations.
    Compares by normalizing both to timezone-naive timestamps first.

    Returns True if timestamps represent the same moment in time.
    """
    if ts1 is None or ts2 is None:
        return ts1 is None and ts2 is None

    try:
        # Normalize both timestamps
        norm1 = normalize_tz(ts1)
        norm2 = normalize_tz(ts2)

        if norm1 is None or norm2 is None:
            return False

        # Compare normalized timestamps
        return norm1 == norm2
    except Exception:
        # Fallback: compare string representations (truncated to seconds)
        str1 = str(ts1)[:19]
        str2 = str(ts2)[:19]
        return str1 == str2


def ensure_strategies_dir():
    """Ensure the velocity strategies directory exists."""
    if not os.path.exists(VELOCITY_STRATEGIES_DIR):
        os.makedirs(VELOCITY_STRATEGIES_DIR)
        print(f"Created strategies directory: {VELOCITY_STRATEGIES_DIR}")


def list_saved_strategies() -> list:
    """List all saved velocity strategies."""
    ensure_strategies_dir()
    strategies = []

    for item in os.listdir(VELOCITY_STRATEGIES_DIR):
        strategy_path = os.path.join(VELOCITY_STRATEGIES_DIR, item)
        config_file = os.path.join(strategy_path, "velocity_config.json")

        if os.path.isdir(strategy_path) and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                strategies.append({
                    'name': item,
                    'path': strategy_path,
                    'config_path': config_file,
                    'config': config,
                    'ticker': config.get('ticker', 'Unknown'),
                    'strategy_name': config.get('strategy_name', item),  # For state file naming
                    'signal_type': config.get('signal_type', 'Unknown'),
                    'created': config.get('deployed_at', 'Unknown')
                })
            except Exception as e:
                print(f"Warning: Could not load {config_file}: {e}")

    # Sort by creation date (newest first)
    strategies.sort(key=lambda x: x['created'], reverse=True)
    return strategies


def save_strategy_bundle(config: dict, name: str = None) -> str:
    """
    Save a strategy config as a permanent bundle.
    Returns the path to the saved bundle.
    """
    ensure_strategies_dir()

    ticker = config.get('ticker', 'UNKNOWN')
    signal_type = config.get('signal_type', 'unknown')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if name:
        bundle_name = name
    else:
        bundle_name = f"{ticker}_{signal_type}_{timestamp}"

    bundle_path = os.path.join(VELOCITY_STRATEGIES_DIR, bundle_name)

    # Create bundle directory
    os.makedirs(bundle_path, exist_ok=True)

    # Save config
    config_path = os.path.join(bundle_path, "velocity_config.json")
    config['bundle_name'] = bundle_name
    config['saved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    safe_json_write(config_path, config, indent=4)

    print(f"✅ Strategy saved to: {bundle_path}")
    return bundle_path


def select_strategy_interactive(default_config_path: str = None) -> str:
    """
    Interactive strategy selection at startup.
    Returns the path to the selected config file.
    """
    strategies = list_saved_strategies()

    print(f"\n{'='*60}")
    print("VELOCITY STRATEGY SELECTION")
    print(f"{'='*60}")

    # Show saved strategies
    if strategies:
        print("\n📦 Saved Strategies:")
        for i, strat in enumerate(strategies):
            ticker = strat['ticker']
            sig_type = strat['signal_type']
            created = strat['created']
            print(f"  {i+1}. {strat['name']}")
            print(f"      Ticker: {ticker} | Signal: {sig_type}")
            print(f"      Created: {created}")
    else:
        print("\n📭 No saved strategies found.")

    # Show production config
    print(f"\n📍 Production Config:")
    if os.path.exists(PRODUCTION_CONFIG_PATH):
        try:
            with open(PRODUCTION_CONFIG_PATH, 'r') as f:
                prod_config = json.load(f)
            print(f"  0. {PRODUCTION_CONFIG_PATH}")
            print(f"      Ticker: {prod_config.get('ticker')} | Signal: {prod_config.get('signal_type')}")
        except Exception:  # Catch all non-system exceptions
            print(f"  0. {PRODUCTION_CONFIG_PATH} (Could not load)")
    else:
        print(f"  0. {PRODUCTION_CONFIG_PATH} (Not found)")

    print(f"\n  Enter. Use default/production config")
    print(f"{'='*60}")

    try:
        choice = input("\nSelect strategy # (or press Enter for default): ").strip()

        if not choice:
            # Use default/production
            if default_config_path and os.path.exists(default_config_path):
                return default_config_path
            elif os.path.exists(PRODUCTION_CONFIG_PATH):
                return PRODUCTION_CONFIG_PATH
            else:
                print("❌ No config found. Please deploy a strategy from the Streamlit app.")
                return None

        idx = int(choice)

        if idx == 0:
            # Production config
            if os.path.exists(PRODUCTION_CONFIG_PATH):
                return PRODUCTION_CONFIG_PATH
            else:
                print("❌ Production config not found.")
                return None

        # Saved strategy
        if 1 <= idx <= len(strategies):
            selected = strategies[idx - 1]
            print(f"\n✅ Selected: {selected['name']}")

            # Use strategy's own config file directly (NOT shared production config)
            # This allows multiple instances to run different strategies simultaneously
            strategy_config_path = selected['config_path']
            print(f"   Using config: {strategy_config_path}")

            # NOTE: Position state handling is now done during startup with the STATE MISMATCH check
            # which provides detailed context and clear Keep/Clear options. The early "Reset?" prompt
            # was removed to avoid confusing duplicate prompts and accidental state deletion.

            return strategy_config_path
        else:
            print(f"❌ Invalid selection: {idx}")
            return None

    except ValueError:
        print("❌ Invalid input. Using default.")
        if os.path.exists(PRODUCTION_CONFIG_PATH):
            return PRODUCTION_CONFIG_PATH
        return default_config_path
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        return None


def load_config(config_path: str = "production_env/velocity_config.json") -> dict:
    """Load velocity strategy configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def is_last_bar_incomplete(df: pd.DataFrame, interval: str, ticker: str) -> bool:
    """
    Check if the last bar in the DataFrame is incomplete (still forming).

    For daily intervals:
    - Stocks: Bar is incomplete if it's today AND market hasn't closed (before 4:30 PM ET with buffer)
    - Crypto: Bar is incomplete if it's today (in UTC) since daily bars close at midnight UTC

    For intraday intervals (15m, 1h, etc.):
    - Bar is incomplete if current time is still within the bar's period
    - E.g., for 15m bars: if current time is 10:07, the 10:00 bar is incomplete

    This prevents false signals from being detected on startup when the current
    bar is still forming.

    Returns True if the last bar should be excluded from backtest sync.
    """
    if df.empty:
        return False

    # Parse interval to get duration in minutes
    interval_minutes = None
    if interval.endswith('m'):
        try:
            interval_minutes = int(interval[:-1])
        except ValueError:
            pass
    elif interval.endswith('h'):
        try:
            interval_minutes = int(interval[:-1]) * 60
        except ValueError:
            pass

    # Handle intraday intervals
    if interval_minutes is not None and interval != "1d":
        from datetime import timezone
        now_utc = datetime.now(timezone.utc)

        # Get last bar's timestamp
        last_bar_time = df.index[-1]
        if hasattr(last_bar_time, 'tzinfo') and last_bar_time.tzinfo is not None:
            # Already timezone-aware, convert to UTC for comparison
            last_bar_utc = last_bar_time.tz_convert('UTC') if hasattr(last_bar_time, 'tz_convert') else last_bar_time
        else:
            # Assume UTC for timezone-naive timestamps (yfinance returns UTC for intraday)
            last_bar_utc = last_bar_time.tz_localize('UTC') if hasattr(last_bar_time, 'tz_localize') else pd.Timestamp(last_bar_time, tz='UTC')

        # Bar end time = bar start time + interval
        bar_end_utc = last_bar_utc + timedelta(minutes=interval_minutes)

        # Bar is incomplete if current time is before bar end time
        # (meaning we're still within the bar's period)
        if now_utc < bar_end_utc:
            return True
        else:
            return False

    # Only apply daily logic to "1d" intervals
    if interval != "1d":
        return False

    # Determine if this is a crypto ticker - all crypto tickers have -USD suffix
    is_crypto = '-USD' in ticker.upper()

    # Get the last bar's timestamp
    last_bar_time = df.index[-1]

    # Convert to date, handling both timezone-aware and naive timestamps
    if hasattr(last_bar_time, 'date') and callable(last_bar_time.date):
        last_bar_date = last_bar_time.date()
    else:
        last_bar_date = pd.to_datetime(last_bar_time).date()

    if is_crypto:
        # Crypto: Daily bars close at 00:00 UTC
        # Bar is incomplete if its date is today (UTC) or later
        from datetime import timezone
        utc_now = datetime.now(timezone.utc)
        today_utc = utc_now.date()

        if last_bar_date > today_utc:
            # Future date (shouldn't happen but be safe)
            return True
        elif last_bar_date == today_utc:
            # Today's bar in UTC - still forming
            return True
        elif last_bar_date == today_utc - timedelta(days=1):
            # Yesterday's bar - check if we're within 30-min buffer after midnight
            # During this buffer, data might not be fully finalized
            if utc_now.hour == 0 and utc_now.minute < 30:
                # Within buffer but bar IS complete, include it
                # (Being conservative here - if data shows yesterday, it's finalized)
                return False
            return False
        else:
            # Older bar, definitely complete
            return False
    else:
        # Stocks: Daily bars close at 4:00 PM ET
        # Use get_market_time() to get current Eastern Time
        market_time = get_market_time()
        today_et = market_time.date()

        if last_bar_date > today_et:
            # Future date (shouldn't happen but be safe)
            return True
        elif last_bar_date == today_et:
            # It's today's bar - check if market has closed
            # Market closes at 4 PM ET, add 30 min buffer for data finalization
            market_close_hour = 16
            buffer_minutes = 30

            if market_time.hour < market_close_hour:
                # Market still open, bar incomplete
                return True
            elif market_time.hour == market_close_hour and market_time.minute < buffer_minutes:
                # Within buffer period after close, be conservative
                return True
            else:
                # Market closed and buffer passed, bar is complete
                return False
        else:
            # Past date (could be yesterday, Friday if weekend, etc.)
            # Bar is complete
            return False


def fetch_price_data(ticker: str, api_key: str = None, days: int = 200, interval: str = "1d",
                     use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch historical price data using Polygon (for crypto) or yfinance with optional caching.

    Args:
        ticker: The ticker symbol
        api_key: Polygon API key (used for crypto)
        days: Number of days of data to fetch
        interval: Data interval ('1d', '1h', etc.)
        use_cache: If True, use local SQLite cache to reduce API calls
    """
    # yfinance has limits on intraday data history
    # Cap days based on interval to avoid API errors
    intraday_limits = {
        '1m': 7,      # 7 days max
        '2m': 60,     # 60 days max
        '5m': 60,     # 60 days max
        '15m': 60,    # 60 days max
        '30m': 60,    # 60 days max
        '1h': 730,    # ~2 years max
        '90m': 60,    # 60 days max
    }
    if interval in intraday_limits:
        max_days = intraday_limits[interval]
        if days > max_days:
            print(f"   ⚠️ {interval} data limited to {max_days} days (requested {days})")
            days = max_days

    # Check if this is a crypto ticker - all crypto tickers have -USD suffix
    is_crypto = '-USD' in ticker.upper()

    # For crypto, yfinance actually has more current data than Polygon (tested Jan 2026)
    # Polygon crypto daily bars have ~24hr delay, yfinance has same-day data
    # So we skip Polygon for crypto and use yfinance directly

    # Check if this is a futures ticker - use Databento (preferred) with yfinance fallback
    is_futures = is_futures_ticker(ticker)

    if is_futures and DATABENTO_AVAILABLE and DATABENTO_API_KEY:
        # Try Databento first for futures (much better intraday data than yfinance)
        try:
            df = _fetch_futures_from_databento(ticker, days=days, interval=interval)
            if not df.empty:
                # Filter phantom bars for futures (weekend/closure data with stale prices)
                df = filter_phantom_bars(df, ticker, interval=interval)

                # Check data freshness for intraday - Databento Historical has ~1 hour delay
                # Fall back to yfinance if data is too stale (>90 minutes old)
                if interval not in ['1d', '1wk', '1mo'] and len(df) > 0:
                    latest_bar = df.index[-1]
                    if latest_bar.tzinfo is not None:
                        now_utc = datetime.now(timezone.utc)
                        data_age_minutes = (now_utc - latest_bar.astimezone(timezone.utc)).total_seconds() / 60
                    else:
                        data_age_minutes = (datetime.now() - latest_bar).total_seconds() / 60

                    # Check if ES futures market is closed (weekend: Fri 4PM CT - Sun 5PM CT)
                    now_ct = datetime.now(CHICAGO_TZ)
                    weekday = now_ct.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
                    hour = now_ct.hour
                    es_market_closed = (
                        weekday == 5 or  # Saturday
                        (weekday == 6 and hour < 17) or  # Sunday before 5 PM CT
                        (weekday == 4 and hour >= 16)  # Friday after 4 PM CT
                    )

                    if data_age_minutes > 90 and not (es_market_closed and is_futures):
                        print(f"   ⚠️ Databento data is {data_age_minutes:.0f} min old, checking yfinance for fresher data")
                        # Don't return yet - fall through to yfinance check below
                    else:
                        if es_market_closed and is_futures and data_age_minutes > 90:
                            print(f"   ℹ️ ES futures market closed for weekend - data is current as of Friday close")
                        return df
                else:
                    return df
            else:
                print(f"   ⚠️ Databento returned empty, falling back to yfinance")
        except Exception as e:
            print(f"   ⚠️ Databento failed: {e}, falling back to yfinance")

    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    # Try to use cached data first (for non-crypto or if Polygon failed)
    if use_cache:
        try:
            from data_cache import fetch_and_cache
            df = fetch_and_cache(ticker, days=days, interval=interval)
            if not df.empty:
                # Filter phantom bars for futures (weekend/closure data with stale prices)
                df = filter_phantom_bars(df, ticker, interval=interval)
                return df
        except ImportError:
            print("   ⚠️ data_cache module not found, fetching directly")
        except Exception as e:
            print(f"   ⚠️ Cache error: {e}, fetching directly")

    # Fallback to direct yfinance fetch
    print(f"📊 Fetching {ticker} via yfinance ({days} days, {interval})")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Use yf.download() - exact same as Streamlit
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

    # Handle MultiIndex columns (exact same as Streamlit line 1239)
    df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns

    # Normalize column names to lowercase (exact same as Streamlit line 1242)
    df.columns = df.columns.str.lower()

    if len(df) > 0:
        print(f"   ✓ Loaded {len(df)} bars: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        # Filter phantom bars for futures (weekend/closure data with stale prices)
        df = filter_phantom_bars(df, ticker, interval=interval)
    else:
        print(f"   ⚠ Warning: No data returned for {ticker}")

    return df


def _fetch_crypto_from_polygon(ticker: str, api_key: str, days: int = 200, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch crypto data from Polygon API (faster updates than yfinance).
    Polygon crypto candles update within minutes of close vs yfinance's 12-24 hour delay.
    """
    from polygon import RESTClient

    client = RESTClient(api_key)

    # Convert ticker to Polygon format (BTC-USD -> X:BTCUSD)
    if '-USD' in ticker:
        polygon_ticker = f"X:{ticker.replace('-USD', 'USD')}"
    elif ticker in ['BTC', 'ETH']:
        polygon_ticker = f"X:{ticker}USD"
    else:
        polygon_ticker = f"X:{ticker}"

    # Map interval to Polygon timespan
    timespan_map = {'1d': 'day', '1h': 'hour', '4h': 'hour', '15m': 'minute'}
    timespan = timespan_map.get(interval, 'day')
    multiplier = 4 if interval == '4h' else (15 if interval == '15m' else 1)

    # Use tomorrow as end date to ensure we get all available data
    end_date = datetime.now() + timedelta(days=1)
    start_date = datetime.now() - timedelta(days=days + 10)  # Extra buffer

    print(f"📊 Fetching {ticker} via Polygon API ({days} days, {interval})")

    aggs = client.get_aggs(
        ticker=polygon_ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_=start_date.strftime('%Y-%m-%d'),
        to=end_date.strftime('%Y-%m-%d'),
        adjusted=False,
        sort='asc',
        limit=50000
    )

    if not aggs:
        return pd.DataFrame()

    # Convert to DataFrame - Polygon timestamps are UTC!
    # For daily crypto bars, the timestamp represents the START of the bar (00:00 UTC)
    data = []
    for bar in aggs:
        # Use UTC timestamp directly (Polygon returns UTC for crypto)
        bar_date = datetime.utcfromtimestamp(bar.timestamp / 1000)
        data.append({
            'date': bar_date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)

    return df


def _fetch_realtime_from_databento_live(ticker: str, lookback_minutes: int = 120) -> pd.DataFrame:
    """
    Fetch real-time futures data from Databento Live API.

    Uses the Live API to get the most recent bars with minimal latency.
    This supplements Historical API data which has ~15-60 min delay.

    Args:
        ticker: The yfinance-style ticker (e.g., 'ES=F', 'GC=F')
        lookback_minutes: How many minutes of history to fetch (default: 120)

    Returns:
        DataFrame with 1-minute OHLCV data, or empty DataFrame on failure
    """
    if not DATABENTO_AVAILABLE:
        return pd.DataFrame()

    if not DATABENTO_API_KEY:
        return pd.DataFrame()

    # Map ticker to Databento symbol
    db_symbol = FUTURES_TICKER_MAP.get(ticker)
    if not db_symbol:
        if ticker.endswith('=F'):
            root = ticker[:-2]
            db_symbol = f"{root}.n.0"
        else:
            return pd.DataFrame()

    try:
        client = db.Live(key=DATABENTO_API_KEY)

        # Get intraday history starting from lookback_minutes ago
        start_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)

        client.subscribe(
            dataset='GLBX.MDP3',
            schema='ohlcv-1m',
            stype_in='continuous',
            symbols=[db_symbol],
            start=start_time.strftime('%Y-%m-%dT%H:%M:%S'),
        )

        # Collect OHLCV records
        ohlcv_records = []

        def callback(record):
            if hasattr(record, 'open') and hasattr(record, 'close'):
                ohlcv_records.append({
                    'ts': record.ts_event,
                    'open': record.open / 1e9,
                    'high': record.high / 1e9,
                    'low': record.low / 1e9,
                    'close': record.close / 1e9,
                    'volume': record.volume
                })

        client.add_callback(callback)
        client.start()

        # Wait for data (with timeout)
        import time
        max_wait = 10
        waited = 0
        while waited < max_wait:
            time.sleep(0.5)
            waited += 0.5
            if len(ohlcv_records) >= lookback_minutes * 0.8:  # Got most expected bars
                break

        client.stop()

        if not ohlcv_records:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_records)
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ns', utc=True)
        df = df.set_index('timestamp').sort_index()
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = df[~df.index.duplicated(keep='last')]  # Remove duplicates

        print(f"   ⚡ Live API: Got {len(df)} bars up to {df.index[-1]}")
        return df

    except Exception as e:
        print(f"   ⚠️ Databento Live failed: {e}")
        return pd.DataFrame()


def _fetch_futures_from_databento(ticker: str, days: int = 60, interval: str = "15m") -> pd.DataFrame:
    """
    Fetch futures data from Databento API.

    Databento provides real-time and historical CME futures data with much better
    coverage than yfinance, especially for intraday bars.

    Args:
        ticker: The yfinance-style ticker (e.g., 'ES=F', 'GC=F', 'CL=F')
        days: Number of days of data to fetch
        interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')

    Returns:
        DataFrame with OHLCV data, or empty DataFrame on failure
    """
    if not DATABENTO_AVAILABLE:
        print("   ⚠️ Databento not available")
        return pd.DataFrame()

    if not DATABENTO_API_KEY:
        print("   ⚠️ DATABENTO_API_KEY not set in environment")
        return pd.DataFrame()

    # Map yfinance ticker to Databento continuous contract symbol
    db_symbol = FUTURES_TICKER_MAP.get(ticker)
    if not db_symbol:
        # Try to construct it for unknown futures
        if ticker.endswith('=F'):
            root = ticker[:-2]  # Remove '=F'
            db_symbol = f"{root}.n.0"  # Nearest/front month continuous (better data coverage)
            print(f"   ⚠️ Unknown futures ticker {ticker}, trying {db_symbol}")
        else:
            print(f"   ⚠️ Cannot map ticker {ticker} to Databento symbol")
            return pd.DataFrame()

    # Map interval to Databento schema
    # Databento supports: ohlcv-1s, ohlcv-1m, ohlcv-1h, ohlcv-1d
    schema_map = {
        '1m': 'ohlcv-1m',
        '5m': 'ohlcv-1m',   # Fetch 1m and resample
        '15m': 'ohlcv-1m',  # Fetch 1m and resample
        '30m': 'ohlcv-1m',  # Fetch 1m and resample
        '1h': 'ohlcv-1h',
        '1d': 'ohlcv-1d',
    }
    schema = schema_map.get(interval, 'ohlcv-1m')

    # Calculate date range - MUST use UTC for Databento API
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    print(f"📊 Fetching {ticker} via Databento ({days} days, {interval}) -> {db_symbol}")

    try:
        # Create historical client
        client = db.Historical(DATABENTO_API_KEY)

        # Fetch data using continuous contract symbology (ES.n.0 = nearest/front month)
        # This is equivalent to ES=F on yfinance - automatically rolls at expiration
        try:
            data = client.timeseries.get_range(
                dataset="GLBX.MDP3",  # CME Globex
                symbols=[db_symbol],
                stype_in="continuous",
                schema=schema,
                start=start_date.strftime('%Y-%m-%dT%H:%M:%S'),
                end=end_date.strftime('%Y-%m-%dT%H:%M:%S'),
            )
        except Exception as e:
            # Handle "data_end_after_available_end" error by extracting available end time
            error_str = str(e)
            if 'data_end_after_available_end' in error_str or 'available up to' in error_str:
                # Extract available end time from error message
                import re
                match = re.search(r"available up to '(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})", error_str)
                if match:
                    available_end = f"{match.group(1)}T{match.group(2)}"  # Convert to ISO format with T separator
                    print(f"   ℹ️ Databento data available up to {available_end}, retrying...")
                    data = client.timeseries.get_range(
                        dataset="GLBX.MDP3",
                        symbols=[db_symbol],
                        stype_in="continuous",
                        schema=schema,
                        start=start_date.strftime('%Y-%m-%dT%H:%M:%S'),
                        end=available_end,
                    )
                else:
                    raise
            else:
                raise

        # Convert to DataFrame
        df = data.to_df()

        if df.empty:
            print(f"   ⚠️ Databento returned empty data for {ticker}")
            return pd.DataFrame()

        # Continuous contracts return a single series - just need to extract OHLCV
        # Databento returns columns: open, high, low, close, volume, plus metadata
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in ohlcv_cols if c in df.columns]]

        # Resample if needed (Databento only has 1m, 1h, 1d native)
        if interval in ['5m', '15m', '30m']:
            resample_rule = interval.replace('m', 'min')  # 15m -> 15min
            df = df.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

        # Convert index to timezone-naive for compatibility
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        if len(df) > 0:
            print(f"   ✓ Databento Historical: Loaded {len(df)} bars: {df.index[0]} to {df.index[-1]}")

            # Always supplement intraday data with Live API for most current bars
            if interval not in ['1d', '1wk', '1mo']:
                data_age_minutes = (datetime.now() - df.index[-1]).total_seconds() / 60

                # Check if ES futures market is closed (weekend: Fri 4PM CT - Sun 5PM CT)
                now_ct = datetime.now(CHICAGO_TZ)
                weekday = now_ct.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
                hour = now_ct.hour

                # ES futures closed: Saturday all day, Sunday before 5PM CT, Friday after 4PM CT
                es_market_closed = (
                    weekday == 5 or  # Saturday
                    (weekday == 6 and hour < 17) or  # Sunday before 5 PM CT
                    (weekday == 4 and hour >= 16)  # Friday after 4 PM CT
                )

                if es_market_closed and is_futures_ticker(ticker):
                    print(f"   ℹ️ ES futures market closed for weekend (data current as of Friday {df.index[-1].strftime('%H:%M')} UTC)")
                    # Skip real-time fetch - market is closed, no new data available
                else:
                    print(f"   ℹ️ Historical data is {data_age_minutes:.0f} min old, fetching real-time supplement...")
                    live_df = _fetch_realtime_from_databento_live(ticker, lookback_minutes=max(120, int(data_age_minutes) + 30))
                    if not live_df.empty:
                        # Live data is in UTC with tz, convert to naive
                        if live_df.index.tz is not None:
                            live_df.index = live_df.index.tz_localize(None)

                        # Resample live data to match interval
                        if interval in ['5m', '15m', '30m']:
                            resample_rule = interval.replace('m', 'min')
                            live_df = live_df.resample(resample_rule).agg({
                                'open': 'first',
                                'high': 'max',
                                'low': 'min',
                                'close': 'last',
                                'volume': 'sum'
                            }).dropna()

                        # Merge: keep historical, add/update with live data
                        last_historical = df.index[-1]
                        newer_live = live_df[live_df.index > last_historical]
                        if len(newer_live) > 0:
                            df = pd.concat([df, newer_live])
                            df = df[~df.index.duplicated(keep='last')]
                            df = df.sort_index()
                            print(f"   ✓ Merged {len(newer_live)} real-time bars. Data now ends: {df.index[-1]}")
                        else:
                            print(f"   ✓ Live API confirmed data is current")
        else:
            print(f"   ⚠️ Databento: No data returned")

        return df

    except Exception as e:
        print(f"   ⚠️ Databento error: {e}")
        return pd.DataFrame()


def fetch_realtime_price(ticker: str) -> float:
    """
    Fetch real-time/current price for display purposes.
    This is separate from daily bar data - used to show actual current price.

    For futures: Uses Databento (real-time CME data)
    For crypto: Uses Coinbase API (US-friendly, real-time) with Kraken/CoinGecko fallbacks
    For stocks: Uses yfinance
    """
    is_crypto = '-USD' in ticker.upper()
    is_futures = is_futures_ticker(ticker)

    # For futures, use Databento (accurate real-time CME data)
    if is_futures and DATABENTO_AVAILABLE and DATABENTO_API_KEY:
        try:
            db_symbol = FUTURES_TICKER_MAP.get(ticker)
            if not db_symbol and ticker.endswith('=F'):
                root = ticker[:-2]
                db_symbol = f"{root}.n.0"  # Nearest/front month continuous

            if db_symbol:
                client = db.Historical(DATABENTO_API_KEY)
                # MUST use UTC for Databento API, subtract 1 min to avoid requesting unavailable data
                end_date = datetime.now(timezone.utc) - timedelta(minutes=1)
                start_date = end_date - timedelta(minutes=10)  # Last 10 minutes

                data = client.timeseries.get_range(
                    dataset="GLBX.MDP3",
                    symbols=[db_symbol],
                    stype_in="continuous",
                    schema="ohlcv-1m",
                    start=start_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    end=end_date.strftime('%Y-%m-%dT%H:%M:%S'),
                )
                df_price = data.to_df()
                if not df_price.empty:
                    price = float(df_price['close'].iloc[-1])
                    if price > 0:
                        return price
        except Exception as e:
            print(f"   ⚠ Databento real-time price failed: {e}, falling back to yfinance...")

    # For crypto, use US-friendly APIs (Coinbase, Kraken, CoinGecko)
    if is_crypto:
        # Extract base symbol (BTC-USD -> BTC)
        if '-USD' in ticker:
            base_symbol = ticker.replace('-USD', '')
        else:
            base_symbol = ticker

        # Try Coinbase first (most accurate for US users, free, no auth needed)
        try:
            url = f"https://api.coinbase.com/v2/prices/{base_symbol}-USD/spot"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                price = float(data.get('data', {}).get('amount', 0))
                if price > 0:
                    return price
        except Exception as e:
            print(f"   ⚠ Coinbase API failed: {e}, trying Kraken...")

        # Fallback to Kraken (also US-friendly)
        try:
            # Kraken uses XBT for Bitcoin
            kraken_symbol = 'XBT' if base_symbol == 'BTC' else base_symbol
            url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_symbol}USD"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                result = data.get('result', {})
                # Kraken returns different key formats (XXBTZUSD or XBTUSD)
                for key in result:
                    if 'USD' in key:
                        price = float(result[key]['c'][0])  # 'c' is last trade closed [price, lot volume]
                        if price > 0:
                            return price
        except Exception as e:
            print(f"   ⚠ Kraken API failed: {e}, trying CoinGecko...")

        # Fallback to CoinGecko
        try:
            coin_id = 'bitcoin' if base_symbol == 'BTC' else 'ethereum' if base_symbol == 'ETH' else base_symbol.lower()
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                price = data.get(coin_id, {}).get('usd', 0)
                if price > 0:
                    return float(price)
        except Exception as e:
            print(f"   ⚠ CoinGecko API failed: {e}, trying yfinance...")

    # For stocks or as final fallback, use yfinance
    try:
        t = yf.Ticker(ticker)
        # Try multiple fields in order of preference
        info = t.info
        price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
        if price:
            return float(price)
    except Exception as e:
        print(f"   ⚠ Could not fetch real-time price: {e}")

    return None


def calculate_composite_oscillator(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Calculate composite oscillator - uses the EXACT same function from Streamlit.
    Supports novel oscillator types if specified in config.
    """
    # Check for novel oscillator type in config
    oscillator_type = config.get('oscillator_type', 'composite_smooth') if config else 'composite_smooth'

    # Always calculate the base composite oscillator first
    try:
        df = create_composite_oscillator(df)
    except Exception as e:
        print(f"⚠️  Error in create_composite_oscillator: {e}")
        # Create a basic RSI-based fallback oscillator
        try:
            rsi = calculate_rsi(df['close'], period=14)
            # Normalize RSI from 0-100 to -1 to +1 range
            df['composite_smooth'] = (rsi - 50) / 50
            df['composite_oscillator'] = df['composite_smooth']
            print(f"   Using RSI-based fallback oscillator")
        except Exception as e2:
            print(f"⚠️  RSI fallback also failed: {e2}")
            # Ultimate fallback: neutral oscillator (no signals)
            df['composite_smooth'] = 0.0
            df['composite_oscillator'] = 0.0
            print(f"   Using neutral fallback oscillator (no signals will be generated)")

    # If a novel oscillator is specified and available, calculate and use it
    if oscillator_type != 'composite_smooth' and NOVEL_INDICATORS_AVAILABLE:
        try:
            print(f"   Using novel oscillator: {oscillator_type}")
            if oscillator_type == 'arwo':
                df['osc_smooth'] = calculate_arwo(df)
            elif oscillator_type == 'dco':
                df['osc_smooth'] = calculate_dco(df)
            elif oscillator_type == 'vcmo':
                df['osc_smooth'] = calculate_vcmo(df)
            elif oscillator_type == 'ics':
                df['osc_smooth'] = calculate_ics(df)
            elif oscillator_type == 'mji':
                df['osc_smooth'] = calculate_mji(df)
            elif oscillator_type == 'prf':
                df['osc_smooth'] = calculate_prf(df)
            elif oscillator_type == 'ewaf':
                df['osc_smooth'] = calculate_ewaf(df)
            elif oscillator_type == 'kfif':
                kfif_val, _, _ = calculate_kfif(df)
                df['osc_smooth'] = kfif_val
            else:
                # Unknown type, fall back to composite_smooth
                print(f"   Unknown oscillator type: {oscillator_type}, using composite_smooth")
                if 'composite_smooth' in df.columns:
                    df['osc_smooth'] = df['composite_smooth']
        except Exception as e:
            print(f"   Error calculating {oscillator_type}: {e}, using composite_smooth")
            if 'composite_smooth' in df.columns:
                df['osc_smooth'] = df['composite_smooth']
    else:
        # Use default composite_smooth
        if 'composite_smooth' in df.columns:
            df['osc_smooth'] = df['composite_smooth']
        elif 'composite_oscillator' in df.columns:
            df['osc_smooth'] = df['composite_oscillator']

    # FINAL SAFETY CHECK: Ensure osc_smooth exists
    if 'osc_smooth' not in df.columns:
        print(f"⚠️  CRITICAL: osc_smooth column missing after all calculations!")
        if 'composite_smooth' in df.columns:
            df['osc_smooth'] = df['composite_smooth']
        elif 'composite_oscillator' in df.columns:
            df['osc_smooth'] = df['composite_oscillator']
        else:
            # Ultimate fallback - neutral values
            df['osc_smooth'] = 0.0
            print(f"   Created neutral osc_smooth fallback")

    return df


def calculate_velocity_signals(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate velocity-based trading signals."""

    # Get config parameters
    signal_type = config.get('signal_type', 'velocity_crossover_and_zone')
    vel_smoothing = config.get('vel_smoothing', 3)
    extreme_zone_mult = config.get('extreme_zone_mult', 1.5)
    require_accel = config.get('require_accel', True)
    oversold_threshold = config.get('oversold_threshold', -0.3)
    overbought_threshold = config.get('overbought_threshold', 0.3)

    # Velocity and acceleration thresholds (match Streamlit)
    vel_threshold = config.get('vel_threshold', 0)
    accel_threshold = config.get('accel_threshold', 0)

    # Extra filters
    rsi_filter = config.get('rsi_filter', 'none')
    rsi_period = config.get('rsi_period', 14)
    rsi_oversold = config.get('rsi_oversold', 30)
    rsi_overbought = config.get('rsi_overbought', 70)
    use_macd_confirm = config.get('use_macd_confirm', False)
    use_bb_filter = config.get('use_bb_filter', False)

    # Use the oscillator specified in config - check if novel oscillator was already calculated
    oscillator_type = config.get('oscillator_type', 'composite_smooth')

    # If using a novel oscillator (arwo, dco, etc), osc_smooth was already set by calculate_composite_oscillator
    # Don't overwrite it with composite_smooth!
    if oscillator_type != 'composite_smooth' and 'osc_smooth' in df.columns:
        # Novel oscillator already calculated - just apply smoothing if needed
        if vel_smoothing > 1:
            df['osc_smooth'] = df['osc_smooth'].rolling(window=vel_smoothing, center=False).mean()
            df['osc_smooth'] = df['osc_smooth'].bfill()
        # osc_smooth is already set correctly, don't overwrite
    else:
        # Use composite oscillator
        osc_col = 'composite_smooth' if 'composite_smooth' in df.columns else 'composite_oscillator'
        # Apply smoothing
        if vel_smoothing > 1:
            df['osc_smooth'] = df[osc_col].rolling(window=vel_smoothing, center=False).mean()
            df['osc_smooth'] = df['osc_smooth'].bfill()
        else:
            df['osc_smooth'] = df[osc_col]

    # Apply wavelet denoising if enabled
    if config.get('use_wavelet_denoise', False):
        try:
            import pywt
            raw = df['osc_smooth'].values.copy()
            family = config.get('wavelet_family', 'db4')
            level = config.get('wavelet_level', 2)
            mode = config.get('wavelet_threshold_mode', 'hard')
            coeffs = pywt.wavedec(raw, family, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(raw)))
            denoised = [coeffs[0]] + [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
            osc_denoised = pywt.waverec(denoised, family)[:len(raw)]
            df['osc_smooth'] = pd.Series(osc_denoised, index=df.index)
        except Exception as e:
            print(f"   Wavelet denoising failed: {e}")

    # Calculate velocity (first derivative), acceleration (second derivative), and jerk (third derivative)
    df['velocity'] = df['osc_smooth'].diff()
    df['acceleration'] = df['velocity'].diff()
    df['jerk'] = df['acceleration'].diff()

    # Fill NaN values
    df['velocity'] = df['velocity'].fillna(0)
    df['acceleration'] = df['acceleration'].fillna(0)
    df['jerk'] = df['jerk'].fillna(0)

    # Detect velocity zero-crossings (slope changes)
    df['vel_cross_up'] = (df['velocity'] > 0) & (df['velocity'].shift(1) <= 0)
    df['vel_cross_down'] = (df['velocity'] < 0) & (df['velocity'].shift(1) >= 0)

    # Zone conditions
    osc_smooth = df['osc_smooth']
    velocity = df['velocity']
    acceleration = df['acceleration']

    in_oversold = osc_smooth < oversold_threshold
    in_overbought = osc_smooth > overbought_threshold
    extreme_oversold = osc_smooth < (oversold_threshold * extreme_zone_mult)
    extreme_overbought = osc_smooth > (overbought_threshold * extreme_zone_mult)

    # Strong momentum detection
    vel_std = velocity.rolling(10, min_periods=1).std().fillna(velocity.std())
    strong_momentum_up = velocity > vel_std * 1.5
    strong_momentum_down = velocity < -vel_std * 1.5

    # Build entry conditions based on signal_type
    if signal_type == 'velocity_crossover_and_zone':
        raw_buy = df['vel_cross_up'] & in_oversold
        raw_sell = df['vel_cross_down'] & in_overbought
    elif signal_type == 'velocity_crossover_or_zone':
        # Balanced: velocity crossover OR extreme zone (more trades)
        raw_buy = df['vel_cross_up'] | extreme_oversold
        raw_sell = df['vel_cross_down'] | extreme_overbought
    elif signal_type == 'zone_only':
        raw_buy = extreme_oversold & (velocity > 0)
        raw_sell = extreme_overbought & (velocity < 0)
    elif signal_type == 'momentum':
        raw_buy = strong_momentum_up & (osc_smooth < 0)
        raw_sell = strong_momentum_down & (osc_smooth > 0)
    elif signal_type == 'any_reversal':
        # Most aggressive: velocity crossover OR extreme zone OR strong momentum in zone
        raw_buy = df['vel_cross_up'] | extreme_oversold | (strong_momentum_up & in_oversold)
        raw_sell = df['vel_cross_down'] | extreme_overbought | (strong_momentum_down & in_overbought)
    elif signal_type == 'double_bottom':
        # Look for second velocity crossover up while still in oversold
        vel_cross_up_count = df['vel_cross_up'].rolling(10).sum()
        raw_buy = (vel_cross_up_count >= 2) & in_oversold
        vel_cross_down_count = df['vel_cross_down'].rolling(10).sum()
        raw_sell = (vel_cross_down_count >= 2) & in_overbought
    elif signal_type == 'divergence':
        # Bullish divergence: price lower low, oscillator higher low
        close_prices = df['close']
        price_lower_low = (close_prices < close_prices.rolling(5).min().shift(1))
        osc_higher_low = (osc_smooth > osc_smooth.rolling(5).min().shift(1))
        raw_buy = price_lower_low & osc_higher_low & in_oversold
        # Bearish divergence: price higher high, oscillator lower high
        price_higher_high = (close_prices > close_prices.rolling(5).max().shift(1))
        osc_lower_high = (osc_smooth < osc_smooth.rolling(5).max().shift(1))
        raw_sell = price_higher_high & osc_lower_high & in_overbought
    elif signal_type == 'breakout':
        # Oscillator breaks above/below threshold (entry on breakout)
        osc_breaks_above = (osc_smooth > oversold_threshold) & (osc_smooth.shift(1) <= oversold_threshold)
        osc_breaks_below = (osc_smooth < overbought_threshold) & (osc_smooth.shift(1) >= overbought_threshold)
        raw_buy = osc_breaks_above  # Buy when breaking out of oversold
        raw_sell = osc_breaks_below  # Sell when breaking into overbought
    else:
        raw_buy = df['vel_cross_up'] & in_oversold
        raw_sell = df['vel_cross_down'] & in_overbought

    # Apply velocity magnitude filter (match Streamlit)
    if vel_threshold > 0:
        raw_buy = raw_buy & (velocity.abs() >= vel_threshold)
        raw_sell = raw_sell & (velocity.abs() >= vel_threshold)

    # Apply acceleration filter (match Streamlit exactly)
    if require_accel:
        buy_accel_cond = acceleration > 0
        sell_accel_cond = acceleration < 0
        if accel_threshold > 0:
            buy_accel_cond = buy_accel_cond & (acceleration.abs() >= accel_threshold)
            sell_accel_cond = sell_accel_cond & (acceleration.abs() >= accel_threshold)
        raw_buy = raw_buy & buy_accel_cond
        raw_sell = raw_sell & sell_accel_cond

    # Apply extra indicator filters
    # RSI filter - matches Streamlit options: none, oversold_only, overbought_only, both
    if rsi_filter != 'none':
        # Calculate RSI if not already done
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / (loss + 1e-10)  # Add small epsilon to avoid division by zero
            df['rsi'] = 100 - (100 / (1 + rs))

        if rsi_filter == 'oversold_only':
            # Only filter buy signals - require RSI to be oversold
            raw_buy = raw_buy & (df['rsi'] < rsi_oversold)
        elif rsi_filter == 'overbought_only':
            # Only filter sell signals - require RSI to be overbought
            raw_sell = raw_sell & (df['rsi'] > rsi_overbought)
        elif rsi_filter == 'both':
            # Filter both buy and sell signals
            raw_buy = raw_buy & (df['rsi'] < rsi_oversold)
            raw_sell = raw_sell & (df['rsi'] > rsi_overbought)
        # Legacy options for backward compatibility
        elif rsi_filter == 'confirm':
            raw_buy = raw_buy & (df['rsi'] < rsi_oversold)
            raw_sell = raw_sell & (df['rsi'] > rsi_overbought)
        elif rsi_filter == 'divergence':
            rsi_rising = df['rsi'] > df['rsi'].shift(3)
            rsi_falling = df['rsi'] < df['rsi'].shift(3)
            raw_buy = raw_buy & rsi_rising
            raw_sell = raw_sell & rsi_falling
        elif rsi_filter == 'momentum':
            raw_buy = raw_buy & (df['rsi'] > 30) & (df['rsi'] < 50)
            raw_sell = raw_sell & (df['rsi'] < 70) & (df['rsi'] > 50)

    # MACD confirmation
    if use_macd_confirm:
        if 'macd_hist' not in df.columns:
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

        macd_bullish = df['macd_hist'] > df['macd_hist'].shift(1)
        macd_bearish = df['macd_hist'] < df['macd_hist'].shift(1)
        raw_buy = raw_buy & macd_bullish
        raw_sell = raw_sell & macd_bearish

    # Bollinger Band filter (match Streamlit exactly)
    if use_bb_filter:
        bb_sma = df['close'].rolling(20).mean()
        bb_std_val = df['close'].rolling(20).std()
        bb_upper = bb_sma + 2 * bb_std_val
        bb_lower = bb_sma - 2 * bb_std_val
        raw_buy = raw_buy & (df['close'] < bb_lower)
        raw_sell = raw_sell & (df['close'] > bb_upper)

    # News sentiment filter - blocks trades during high-risk news conditions
    use_news_filter = config.get('use_news_filter', False)
    news_risk_threshold = config.get('news_risk_threshold', 60)  # Block if risk > threshold

    if use_news_filter:
        try:
            from news_features import NewsTradeFilter
            ticker = config.get('ticker', 'SPY')
            news_filter = NewsTradeFilter()

            # Check news conditions for the most recent bar (live trading decision)
            # For historical backtest bars, we don't have historical news risk, so allow those
            filter_result = news_filter.check_entry(ticker, direction='long')

            if not filter_result['allow_entry'] and filter_result['risk_details'].get('risk_score', 0) > news_risk_threshold:
                # Only block the LAST row (live signal) - historical signals remain for backtest consistency
                if len(df) > 0:
                    last_idx = df.index[-1]
                    # Store original last signal for logging
                    original_buy = raw_buy.loc[last_idx] if isinstance(raw_buy, pd.Series) else raw_buy.iloc[-1]
                    original_sell = raw_sell.loc[last_idx] if isinstance(raw_sell, pd.Series) else raw_sell.iloc[-1]

                    if original_buy or original_sell:
                        print(f"[NEWS FILTER] Blocking signal - {filter_result['reason']} (risk: {filter_result['risk_details'].get('risk_score', 0)})")

                    # Create a copy to modify the last value
                    if isinstance(raw_buy, pd.Series):
                        raw_buy = raw_buy.copy()
                        raw_buy.loc[last_idx] = False
                    if isinstance(raw_sell, pd.Series):
                        raw_sell = raw_sell.copy()
                        raw_sell.loc[last_idx] = False
        except ImportError:
            pass  # news_features module not available
        except Exception as e:
            print(f"[NEWS FILTER] Warning: {e}")

    df['buy_signal'] = raw_buy
    df['sell_signal'] = raw_sell

    return df


def generate_trade_log_csv(locked_backtest: dict, num_trades: int = 10, trade_state: dict = None) -> io.BytesIO:
    """Generate a CSV file of the last N trades from locked backtest.

    Args:
        locked_backtest: Dict containing 'exits' list with trade data
        num_trades: Number of recent trades to include (default 10)
        trade_state: Optional current trade state - if provided, excludes trades
                     that conflict with the tracked position

    Returns:
        BytesIO buffer containing CSV data, or None if no trades
    """
    if not locked_backtest or not locked_backtest.get('exits'):
        return None

    exits = locked_backtest['exits']

    # Get tracked position entry time for filtering
    tracked_entry_dt = None
    if trade_state and trade_state.get('position'):
        tracked_entry_time = trade_state.get('entry_time') or trade_state.get('entry_signal_bar')
        if tracked_entry_time:
            try:
                tracked_entry_dt = normalize_tz(pd.to_datetime(tracked_entry_time))
            except Exception:  # Catch all non-system exceptions
                pass

    # Sort by date and take last N trades (exclude missed trades and conflicting trades)
    valid_exits = []
    for e in exits:
        # Skip missed trades
        if e.get('missed'):
            continue
        # Skip trades whose entry is after the tracked position entry
        if tracked_entry_dt:
            exit_entry_date = e.get('entry_date')
            if exit_entry_date:
                try:
                    exit_entry_dt = normalize_tz(pd.to_datetime(exit_entry_date))
                    if exit_entry_dt > tracked_entry_dt:
                        continue  # Skip this conflicting trade
                except Exception:  # Catch all non-system exceptions
                    pass
        valid_exits.append(e)

    if not valid_exits:
        return None

    sorted_exits = sorted(valid_exits, key=lambda x: str(x.get('date', '')))
    recent_trades = sorted_exits[-num_trades:]

    if not recent_trades:
        return None

    # Build CSV content
    csv_lines = ["#,Entry Time,Entry $,Exit Time,Exit $,P&L,Result"]

    for i, trade in enumerate(recent_trades, 1):
        exit_date = trade.get('date', '')
        exit_price = trade.get('price', 0)
        entry_price = trade.get('entry_price', 0)
        pnl = trade.get('pnl', 0)

        # Format entry time - show BOTH UTC and Chicago time (12-hour format)
        entry_date_raw = trade.get('entry_date', '')
        try:
            entry_dt = pd.to_datetime(entry_date_raw)
            entry_utc = entry_dt.tz_localize('UTC') if entry_dt.tzinfo is None else entry_dt.tz_convert('UTC')
            entry_ct = to_chicago_time(entry_date_raw)
            entry_utc_str = entry_utc.strftime('%Y-%m-%d %H:%M UTC')
            entry_ct_str = entry_ct.strftime('%I:%M %p CT') if entry_ct else ''
            entry_time = f"{entry_utc_str} ({entry_ct_str})" if entry_ct_str else entry_utc_str
        except Exception:
            entry_time = str(entry_date_raw)[:16].replace('T', ' ') if entry_date_raw else "N/A"

        # Format exit time - show BOTH UTC and Chicago time (12-hour format)
        try:
            exit_dt = pd.to_datetime(exit_date)
            exit_utc = exit_dt.tz_localize('UTC') if exit_dt.tzinfo is None else exit_dt.tz_convert('UTC')
            exit_ct = to_chicago_time(exit_date)
            exit_utc_str = exit_utc.strftime('%Y-%m-%d %H:%M UTC')
            exit_ct_str = exit_ct.strftime('%I:%M %p CT') if exit_ct else ''
            exit_time = f"{exit_utc_str} ({exit_ct_str})" if exit_ct_str else exit_utc_str
        except Exception:
            exit_time = str(exit_date)[:16].replace('T', ' ') if exit_date else "N/A"

        # Result indicator
        result = "WIN" if pnl > 0 else "LOSS"

        csv_lines.append(
            f"{i},{entry_time},${entry_price:,.2f},{exit_time},${exit_price:,.2f},{pnl:+.2f}%,{result}"
        )

    # Create BytesIO buffer
    csv_content = "\n".join(csv_lines)
    csv_buf = io.BytesIO(csv_content.encode('utf-8'))
    csv_buf.name = "trade_log.csv"
    csv_buf.seek(0)  # Ensure buffer is ready to read

    return csv_buf


def send_discord_alert(webhook_url: str, message: str, chart_buf: io.BytesIO = None,
                       include_disclaimer: bool = True, strategy_name: str = None,
                       csv_buf: io.BytesIO = None):
    """Send alert to Discord webhook with optional chart image and CSV attachment.

    Also sends to secondary Haus Hedge webhook if strategy_name is provided.

    Args:
        webhook_url: Primary Discord webhook URL
        message: Message content
        chart_buf: Optional chart image BytesIO buffer
        include_disclaimer: Whether to append legal disclaimer
        strategy_name: Strategy name for secondary webhook lookup
        csv_buf: Optional CSV file BytesIO buffer (e.g., trade log)
    """
    if not webhook_url:
        print(f"[ALERT] {message}")
        return False

    # Append legal disclaimer to all messages
    if include_disclaimer:
        message = message + LEGAL_DISCLAIMER

    def post_to_webhook(url: str, msg: str, chart: io.BytesIO = None, csv: io.BytesIO = None) -> bool:
        """Helper to post to a single webhook with optional file attachments."""
        try:
            files = {}

            # Add chart if provided
            if chart:
                chart.seek(0)
                files['file1'] = ('chart.png', chart, 'image/png')

            # Add CSV if provided
            if csv:
                csv.seek(0)
                files['file2'] = ('trade_log.csv', csv, 'text/csv')

            if files:
                # For multipart uploads, use payload_json to include allowed_mentions
                import json
                payload = {
                    'payload_json': json.dumps({
                        'content': msg,
                        'allowed_mentions': {'parse': ['everyone']}  # Enable @here mentions
                    })
                }
                response = requests.post(url, data=payload, files=files)
            else:
                payload = {
                    "content": msg,
                    "allowed_mentions": {"parse": ["everyone"]}  # Enable @here mentions
                }
                response = requests.post(url, json=payload)

            if response.status_code in [200, 204]:
                return True
            else:
                print(f"Discord webhook error: {response.status_code}")
                return False
        except Exception as e:
            print(f"Discord webhook error: {e}")
            return False

    # Send to primary webhook
    primary_success = post_to_webhook(webhook_url, message, chart_buf, csv_buf)

    # Send to secondary Haus Hedge webhook if strategy exists AND it's a different URL
    if strategy_name and strategy_name in HAUS_HEDGE_WEBHOOKS:
        secondary_url = HAUS_HEDGE_WEBHOOKS[strategy_name]
        # Only post to secondary if it's a DIFFERENT webhook (avoid duplicates)
        if secondary_url != webhook_url:
            # Reset buffers for second send
            if chart_buf:
                chart_buf.seek(0)
            if csv_buf:
                csv_buf.seek(0)
            secondary_success = post_to_webhook(secondary_url, message, chart_buf, csv_buf)
            if secondary_success:
                print(f"   📤 Also posted to Haus Hedge server")

    return primary_success


def generate_velocity_chart(df: pd.DataFrame, backtest: dict, config: dict, ticker: str,
                            title_suffix: str = "", trade_history: list = None,
                            current_position: dict = None, locked_backtest: dict = None,
                            full_period_backtest: dict = None, chart_type: str = "status",
                            missed_signals: dict = None) -> io.BytesIO:
    """Generate a velocity strategy chart for Discord.

    Args:
        df: DataFrame with price and indicator data
        backtest: Backtest results dict (used for stats and equity curve) - typically recent/fresh data
        config: Strategy config
        ticker: Ticker symbol
        title_suffix: Optional suffix for chart title (e.g., " - Last 180 Days")
        trade_history: Optional list of locked trades from trade_history.json (prevents repainting)
        current_position: Optional current position dict with entry_price, entry_signal_bar
        locked_backtest: Optional locked backtest dict with frozen entry/exit markers (prevents repainting)
        full_period_backtest: Optional full period backtest for showing both recent and full stats
        chart_type: "signal" for focused entry/exit charts, "status" for broader status updates
        missed_signals: Optional dict with 'entries' and 'exits' for ghost markers (signals while offline)
    """
    try:
        # Determine interval and if intraday
        interval = config.get('interval', '1d')
        is_intraday = interval in ['1m', '5m', '15m', '30m', '1h', '90m', '4h']
        is_crypto = '-USD' in ticker.upper()

        # For INTRADAY data: limit to recent bars to avoid overcrowded charts
        # Also plot by INDEX (not datetime) to eliminate market closure gaps
        if is_intraday:
            # Use different bar limits based on chart_type
            if chart_type == "signal":
                # Focused view for entry/exit signals (short timeframe)
                if is_crypto:
                    max_bars_map = {
                        '1m': 300,    # ~5 hours
                        '5m': 150,    # ~12 hours
                        '15m': 150,   # ~2.5 hours (crypto 24/7)
                        '30m': 100,   # ~50 hours
                        '1h': 48,     # 2 days
                        '90m': 32,    # 2 days
                        '4h': 24,     # 4 days
                    }
                else:
                    max_bars_map = {
                        '1m': 300,    # ~45 minutes
                        '5m': 150,    # ~2 hours
                        '15m': 100,   # ~6 hours (stocks/futures)
                        '30m': 80,    # ~6 hours
                        '1h': 48,     # 2 days
                        '90m': 32,    # 2 days
                        '4h': 24,     # 4 days
                    }
            else:
                # Broader view for status updates and startup
                if is_crypto:
                    max_bars_map = {
                        '1m': 2000,   # ~33 hours
                        '5m': 600,    # ~50 hours
                        '15m': 400,   # ~4 days (crypto 24/7)
                        '30m': 300,   # ~6 days
                        '1h': 200,    # ~8 days
                        '90m': 150,   # ~9 days
                        '4h': 100,    # ~17 days
                    }
                else:
                    max_bars_map = {
                        '1m': 2000,   # ~5 days at 390 bars/day
                        '5m': 500,    # ~6 days at 78 bars/day
                        '15m': 260,   # ~2 days (stocks/futures)
                        '30m': 200,   # ~3 days
                        '1h': 168,    # ~1 week
                        '90m': 150,   # ~7 days
                        '4h': 100,    # ~14 days
                    }
            max_bars = max_bars_map.get(interval, 300 if chart_type == "signal" else 500)
            df_plot = df.tail(max_bars).copy()

            # For intraday, use sequential bar index for x-axis (eliminates gaps)
            # Store datetime in a column for marker mapping and x-axis labels
            df_plot = df_plot.reset_index()

            # Find the datetime column
            datetime_col = None
            for col in df_plot.columns:
                if col in ['index', 'date', 'datetime', 'Date', 'Datetime']:
                    datetime_col = col
                    break
                if datetime_col is None and df_plot[col].dtype == 'datetime64[ns]':
                    datetime_col = col
                    break
            if datetime_col is None and len(df_plot.columns) > 0:
                first_col = df_plot.columns[0]
                try:
                    pd.to_datetime(df_plot[first_col].iloc[0])
                    datetime_col = first_col
                except Exception:
                    pass

            # Normalize datetime column to Chicago time (tz-naive for plotting)
            # Track if we converted to Chicago so we don't double-convert later
            datetime_already_chicago = False
            if datetime_col:
                df_plot[datetime_col] = pd.to_datetime(df_plot[datetime_col])
                if df_plot[datetime_col].dt.tz is not None:
                    df_plot[datetime_col] = df_plot[datetime_col].dt.tz_convert(CHICAGO_TZ).dt.tz_localize(None)
                    datetime_already_chicago = True
                else:
                    # tz-naive data - check if it's from Databento (UTC) or already local
                    # Databento strips tz with tz_localize(None) but values are still UTC
                    # We need to convert UTC->Chicago, which normalize_tz will do
                    datetime_already_chicago = False

            df_plot['_datetime_col'] = datetime_col  # Store for reference
            df_plot['_datetime_already_chicago'] = datetime_already_chicago  # Track conversion
            df_plot['bar_num'] = range(len(df_plot))  # Sequential bar index
            plot_by_index = True  # Use bar index for continuous lines (TradingView style)
        else:
            # For daily data, ALSO use index-based plotting (TradingView style - no weekend gaps)
            df_plot = df.copy()
            datetime_col = None  # Daily data doesn't have a datetime column, uses index

            # Ensure we have a proper datetime index first
            if not isinstance(df_plot.index, pd.DatetimeIndex):
                try:
                    df_plot.index = pd.to_datetime(df_plot.index)
                except Exception:  # Catch all non-system exceptions
                    pass

            # Store datetime for label formatting, then use bar_num for plotting
            df_plot['_datetime_for_labels'] = df_plot.index
            df_plot['bar_num'] = range(len(df_plot))
            plot_by_index = True  # TradingView style - evenly spaced bars, no gaps

        # Create figure with subplots - LARGER for Discord
        # Note: sharex=False because equity curve uses trade numbers, not dates
        fig, axes = plt.subplots(4, 1, figsize=(16, 12),
                                 gridspec_kw={'height_ratios': [3, 1.5, 1, 1.5]}, sharex=False)

        # Style
        fig.patch.set_facecolor('#1a1a2e')
        for ax in axes:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white')
            ax.grid(True, color='#333', alpha=0.5)

        # Import date formatting
        import matplotlib.dates as mdates

        # Determine appropriate date formatting based on data range
        # Check for datetime column (intraday) or _datetime_for_labels (daily)
        span_dt_col = None
        if datetime_col and datetime_col in df_plot.columns:
            span_dt_col = datetime_col
        elif '_datetime_for_labels' in df_plot.columns:
            span_dt_col = '_datetime_for_labels'

        if plot_by_index and span_dt_col:
            try:
                if datetime_already_chicago:
                    # Data is already Chicago time - use directly
                    first_dt = pd.to_datetime(df_plot[span_dt_col].iloc[0])
                    last_dt = pd.to_datetime(df_plot[span_dt_col].iloc[-1])
                else:
                    # Data needs normalization
                    first_dt = normalize_tz(pd.to_datetime(df_plot[span_dt_col].iloc[0]))
                    last_dt = normalize_tz(pd.to_datetime(df_plot[span_dt_col].iloc[-1]))
                data_span_days = (last_dt - first_dt).days if len(df_plot) > 1 else 1
            except Exception:
                data_span_days = 60 if not is_intraday else 7
        else:
            try:
                data_span_days = (normalize_tz(df_plot.index[-1]) - normalize_tz(df_plot.index[0])).days if len(df_plot) > 1 else 1
            except Exception:
                data_span_days = 7 if is_intraday else 60

        def format_xaxis(ax, show_dates=False):
            """Apply appropriate x-axis formatting with datetime labels."""
            # Determine which column has datetime info for labels
            dt_label_col = None
            if datetime_col and datetime_col in df_plot.columns:
                dt_label_col = datetime_col
            elif '_datetime_for_labels' in df_plot.columns:
                dt_label_col = '_datetime_for_labels'

            if plot_by_index and dt_label_col:
                # For index-based plotting, create custom tick labels showing datetime
                from matplotlib.ticker import FuncFormatter, MaxNLocator

                # Create bar_num to datetime lookup
                barnum_to_dt = dict(zip(df_plot['bar_num'], df_plot[dt_label_col]))

                def format_tick(x, pos):
                    if x in barnum_to_dt:
                        dt = barnum_to_dt[x]
                        try:
                            dt_parsed = pd.to_datetime(dt)
                            if pd.isna(dt_parsed):
                                return ''
                            if is_intraday:
                                return dt_parsed.strftime('%m/%d\n%H:%M')
                            else:
                                # TradingView style: "Oct", "Nov", "Dec", "2026" at year boundaries
                                return dt_parsed.strftime('%b')
                        except Exception:
                            return ''
                    return ''

                ax.xaxis.set_major_formatter(FuncFormatter(format_tick))
                # More ticks for daily data
                ax.xaxis.set_major_locator(MaxNLocator(nbins=12 if not is_intraday else 8, integer=True))
                ax.tick_params(axis='x', labelsize=9)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')  # No rotation like TradingView

            elif isinstance(df_plot.index, pd.DatetimeIndex):
                if is_intraday and data_span_days <= 3:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
                elif is_intraday and data_span_days <= 10:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                elif is_intraday and data_span_days <= 30:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                elif is_intraday or data_span_days <= 90:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                else:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.tick_params(axis='x', labelsize=8)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Determine x-axis values based on plotting mode
        datetime_col = df_plot['_datetime_col'].iloc[0] if '_datetime_col' in df_plot.columns else None
        # For daily data, use _datetime_for_labels for date lookup
        dt_lookup_col = datetime_col if datetime_col and datetime_col in df_plot.columns else '_datetime_for_labels'

        if plot_by_index:
            # Use bar number for continuous lines (no gaps during market closures)
            x_axis = df_plot['bar_num']
            # Create date-to-barnum mapping for markers
            date_to_barnum = {}
            # Check if datetime was already converted to Chicago in the preprocessing step
            # This prevents double-conversion (yfinance data gets converted once, then normalize_tz
            # would wrongly assume UTC and convert again, causing a 6-hour offset)
            datetime_already_chicago = df_plot['_datetime_already_chicago'].iloc[0] if '_datetime_already_chicago' in df_plot.columns else False
            if dt_lookup_col in df_plot.columns:
                for i, row in df_plot.iterrows():
                    try:
                        if datetime_already_chicago:
                            # Data is already in Chicago time - use directly
                            dt = pd.to_datetime(row[dt_lookup_col])
                        else:
                            # Data needs normalization (e.g., Databento naive UTC)
                            dt = normalize_tz(pd.to_datetime(row[dt_lookup_col]))
                        date_to_barnum[dt] = row['bar_num']
                        # Also add date-only key for daily data (ignore time component)
                        date_only = dt.replace(hour=0, minute=0, second=0, microsecond=0)
                        if date_only not in date_to_barnum:
                            date_to_barnum[date_only] = row['bar_num']
                    except Exception:
                        pass
        else:
            # Use datetime index directly
            x_axis = df_plot.index
            date_to_barnum = {}

        # 1. Price Chart with Entry/Exit markers
        ax1 = axes[0]

        # Detect and shade data gaps (time gaps > threshold OR stale prices for 3+ bars)
        # This helps users identify periods with unreliable data
        # NOTE: Skip for daily data - 24-hour gaps between bars are normal
        gap_regions = []
        interval = config.get('interval', '1d')
        is_daily = interval == '1d'

        if len(df_plot) > 3 and not is_daily:
            # Detect time gaps - use index if it's DatetimeIndex, else try datetime_col
            # Threshold: 2 hours for intraday (accounts for CME breaks etc)
            gap_threshold = pd.Timedelta(hours=2)

            if isinstance(df_plot.index, pd.DatetimeIndex):
                time_diff = pd.Series(df_plot.index).diff()
                time_gap_mask = time_diff > gap_threshold
                time_gap_mask = time_gap_mask.values  # Convert to array for indexing
            elif datetime_col and datetime_col in df_plot.columns:
                df_plot['_time_diff'] = pd.to_datetime(df_plot[datetime_col]).diff()
                time_gap_mask = df_plot['_time_diff'] > gap_threshold
            else:
                time_gap_mask = pd.Series([False] * len(df_plot))

            # Detect stale data (price unchanged for 3+ consecutive bars)
            df_plot['_price_change'] = df_plot['close'].diff().abs()
            stale_mask = (df_plot['_price_change'] == 0) | df_plot['_price_change'].isna()
            # Mark bars that are part of a stale sequence (3+ unchanged)
            stale_count = stale_mask.rolling(3, min_periods=3).sum()
            stale_region_mask = stale_count >= 3

            # Combine masks and find contiguous regions
            gap_mask = time_gap_mask | stale_region_mask

            # Shade gap regions
            in_gap = False
            gap_start = None
            for i, is_gap in enumerate(gap_mask):
                if is_gap and not in_gap:
                    gap_start = x_axis[i]
                    in_gap = True
                elif not is_gap and in_gap:
                    gap_end = x_axis[i]
                    ax1.axvspan(gap_start, gap_end, color='red', alpha=0.15, zorder=0)
                    gap_regions.append((gap_start, gap_end))
                    in_gap = False
            # Handle gap at end of data
            if in_gap:
                gap_end = x_axis.iloc[-1] if hasattr(x_axis, 'iloc') else x_axis[-1]
                ax1.axvspan(gap_start, gap_end, color='red', alpha=0.15, zorder=0)
                gap_regions.append((gap_start, gap_end))

            if gap_regions:
                print(f"   [Chart] Detected {len(gap_regions)} data gap region(s) - shaded in red")

        # Draw candlesticks (TradingView style - candles nearly touching)
        if plot_by_index:
            candle_width = 0.8  # Wider candles that nearly touch (TradingView style)
            wick_width = 1.0   # Thicker wicks for visibility
        else:
            # For datetime x-axis, use Timedelta for width
            candle_width = pd.Timedelta(hours=12)  # Half a day width for daily candles
            wick_width = 0.0001  # Not used for datetime

        for i in range(len(df_plot)):
            x = x_axis.iloc[i] if hasattr(x_axis, 'iloc') else x_axis[i]
            o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]

            # Determine candle color (green for bullish, red for bearish)
            if c >= o:
                body_color = '#26a69a'  # TradingView green
                wick_color = '#26a69a'
            else:
                body_color = '#ef5350'  # TradingView red
                wick_color = '#ef5350'

            # Draw wick (high-low line)
            ax1.plot([x, x], [l, h], color=wick_color, linewidth=wick_width if plot_by_index else 1, solid_capstyle='round')

            # Draw body (open-close rectangle)
            body_bottom = min(o, c)
            body_height = abs(c - o)
            if body_height < 0.001:  # Doji - draw a line
                if plot_by_index:
                    ax1.plot([x - candle_width/2, x + candle_width/2], [c, c], color=wick_color, linewidth=1)
                else:
                    ax1.plot([x - candle_width, x + candle_width], [c, c], color=wick_color, linewidth=1)
            else:
                if plot_by_index:
                    rect = plt.Rectangle((x - candle_width/2, body_bottom), candle_width, body_height,
                                         facecolor=body_color, edgecolor=wick_color, linewidth=0.5)
                else:
                    # For datetime x-axis, matplotlib handles Timedelta width
                    rect = plt.Rectangle((x - candle_width, body_bottom), candle_width * 2, body_height,
                                         facecolor=body_color, edgecolor=wick_color, linewidth=0.5)
                ax1.add_patch(rect)

        # Add current price line
        current_price = df_plot['close'].iloc[-1]
        ax1.axhline(current_price, color='#2962ff', linestyle='-', linewidth=1, alpha=0.7)

        # Plot trade markers from LOCKED backtest (prevents repainting)
        # The locked_backtest is frozen at startup and only appended to when new signals occur.
        # This ensures historical markers never shift position.
        markers_source = locked_backtest if locked_backtest else backtest

        # Get chart date range for filtering markers (normalize for tz comparison)
        # IMPORTANT: Extend chart_end by one interval to include entries at the LAST bar's close
        interval_minutes = INTERVAL_MINUTES.get(config.get('interval', '1d'), 1440)
        # Use dt_lookup_col for chart date range (handles both intraday and daily)
        # CRITICAL: Respect datetime_already_chicago flag to avoid double-conversion bug
        if plot_by_index and dt_lookup_col in df_plot.columns:
            if datetime_already_chicago:
                # Data is already Chicago time - use directly
                chart_start = pd.to_datetime(df_plot[dt_lookup_col].iloc[0])
                chart_end = pd.to_datetime(df_plot[dt_lookup_col].iloc[-1])
            else:
                # Data needs normalization (e.g., Databento naive UTC)
                chart_start = normalize_tz(pd.to_datetime(df_plot[dt_lookup_col].iloc[0]))
                chart_end = normalize_tz(pd.to_datetime(df_plot[dt_lookup_col].iloc[-1]))
            chart_end = chart_end + timedelta(minutes=interval_minutes)
        else:
            chart_start = normalize_tz(df_plot.index.min())
            chart_end = normalize_tz(df_plot.index.max())
            chart_end = chart_end + timedelta(minutes=interval_minutes)

        # Dynamic marker size based on number of trades (smaller for dense charts)
        num_entries = len(markers_source.get('entries', [])) if markers_source else 0
        if num_entries > 200:
            marker_size = 30  # Very small for 200+ trades
        elif num_entries > 100:
            marker_size = 50  # Small for 100-200 trades
        elif num_entries > 50:
            marker_size = 70  # Medium for 50-100 trades
        else:
            marker_size = 100  # Normal size for <50 trades

        # Helper function to find x-position for a given date
        def get_x_position(target_date):
            """Get x-axis position for a date. Maps to bar number for index-based plotting."""
            target_date = normalize_tz(target_date)
            if plot_by_index:
                # Look up in date_to_barnum mapping
                if target_date in date_to_barnum:
                    return date_to_barnum[target_date]

                # For DAILY charts: try date-only matching (ignore time component)
                # This fixes issues where entry dates and bar dates have different times
                if not is_intraday:
                    target_date_only = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
                    if target_date_only in date_to_barnum:
                        return date_to_barnum[target_date_only]
                    # Also try the original date's date portion
                    for dt, bar in date_to_barnum.items():
                        if hasattr(dt, 'date') and hasattr(target_date, 'date'):
                            if dt.date() == target_date.date():
                                return bar

                # Try to find closest date (within 1 interval)
                closest_bar = None
                min_diff = None
                for dt, bar in date_to_barnum.items():
                    diff = abs((dt - target_date).total_seconds())
                    if min_diff is None or diff < min_diff:
                        min_diff = diff
                        closest_bar = bar
                # Only return closest if within reasonable range (1 interval)
                if min_diff is not None and min_diff <= interval_minutes * 60:
                    return closest_bar
                return None  # Entry too far from any bar
            else:
                return target_date

        # Debug: Show marker info
        entries_in_range = 0
        entries_total = len(markers_source.get('entries', [])) if markers_source else 0
        if entries_total > 0:
            sample_entries = [normalize_tz(pd.to_datetime(e['date'])) for e in markers_source['entries'][-5:]]
            print(f"   [Chart] Date range: {chart_start} to {chart_end}")
            print(f"   [Chart] Sample entry dates (last 5): {sample_entries}")
            if plot_by_index:
                print(f"   [Chart] Date-to-barnum mapping has {len(date_to_barnum)} entries")

        # Get tracked position entry time to filter out conflicting markers
        # (entries/exits that occur after user's tracked entry shouldn't be drawn)
        tracked_entry_dt = None
        if current_position and current_position.get('position'):
            tracked_entry_time = current_position.get('entry_time') or current_position.get('entry_signal_bar')
            if tracked_entry_time:
                try:
                    tracked_entry_dt = normalize_tz(pd.to_datetime(tracked_entry_time))
                except Exception:  # Catch all non-system exceptions
                    pass
        skipped_conflicting = 0

        if markers_source and markers_source.get('entries'):
            for entry in markers_source['entries']:
                # Handle both datetime objects and strings
                entry_date = entry['date']
                if isinstance(entry_date, str):
                    try:
                        entry_date = pd.to_datetime(entry_date)
                    except Exception:  # Catch all non-system exceptions
                        continue
                entry_date = normalize_tz(entry_date)

                # SKIP markers outside the chart's date range (prevents clustering at edges)
                # For DAILY charts: use date-only comparison to handle timezone/time mismatches
                if is_intraday:
                    outside_range = entry_date < chart_start or entry_date > chart_end
                else:
                    # Daily charts: compare dates only (ignore time component)
                    entry_date_only = entry_date.date() if hasattr(entry_date, 'date') else entry_date
                    chart_start_date = chart_start.date() if hasattr(chart_start, 'date') else chart_start
                    chart_end_date = chart_end.date() if hasattr(chart_end, 'date') else chart_end
                    outside_range = entry_date_only < chart_start_date or entry_date_only > chart_end_date

                if outside_range:
                    if entries_total <= 20:  # Only log for low-volume strategies
                        print(f"   [Chart] Skipping entry {entry_date} - outside range [{chart_start} to {chart_end}]")
                    continue

                # SKIP entries that conflict with tracked position
                # (entries AT OR AFTER tracked entry time - the tracked entry has its own arrow marker)
                if tracked_entry_dt and entry_date >= tracked_entry_dt:
                    skipped_conflicting += 1
                    continue

                entries_in_range += 1

                # LONG entries: green up triangle, SHORT entries: red down triangle
                # MISSED entries: orange color
                is_long = entry.get('position', 'long') == 'long'
                is_missed = entry.get('missed', False)
                marker_shape = '^' if is_long else 'v'
                if is_missed:
                    marker_color = 'orange'
                else:
                    marker_color = '#00e676' if is_long else '#ff5252'  # Brighter green/red than candles

                # Get x-position (bar number for intraday, date for daily)
                x_pos = get_x_position(entry_date)
                if x_pos is not None:
                    # Place BUY marker BELOW the candle (standard charting convention)
                    # Up-pointing triangle below bar = "buy here"
                    bar_low = None
                    if plot_by_index and x_pos in df_plot['bar_num'].values:
                        bar_idx = df_plot[df_plot['bar_num'] == x_pos].index[0]
                        bar_low = df_plot.loc[bar_idx, 'low']
                    elif not plot_by_index:
                        # For daily charts, find bar by date in index
                        try:
                            # Try exact match first
                            if x_pos in df_plot.index:
                                bar_low = df_plot.loc[x_pos, 'low']
                            else:
                                # Try matching by date only (ignore time)
                                target_date = pd.to_datetime(x_pos).date()
                                for idx in df_plot.index:
                                    if pd.to_datetime(idx).date() == target_date:
                                        bar_low = df_plot.loc[idx, 'low']
                                        break
                        except Exception:
                            pass

                    if bar_low is not None:
                        price_range = df_plot['high'].max() - df_plot['low'].min()
                        marker_offset = price_range * 0.02  # 2% offset below candle
                        y_pos = bar_low - marker_offset
                    else:
                        y_pos = entry['price']  # Fallback to recorded price
                    ax1.scatter(x_pos, y_pos, marker=marker_shape, color=marker_color, s=marker_size, zorder=5, edgecolors='white', linewidth=0.5)

            print(f"   [Chart] Entries: {entries_in_range}/{entries_total} in chart range")

        exits_in_range = 0
        exits_total = len(markers_source.get('exits', [])) if markers_source else 0
        if markers_source and markers_source.get('exits'):
            for exit_trade in markers_source['exits']:
                # Handle both datetime objects and strings
                exit_date = exit_trade['date']
                if isinstance(exit_date, str):
                    try:
                        exit_date = pd.to_datetime(exit_date)
                    except Exception:  # Catch all non-system exceptions
                        continue
                exit_date = normalize_tz(exit_date)

                # SKIP markers outside the chart's date range (prevents clustering at edges)
                # For DAILY charts: use date-only comparison to handle timezone/time mismatches
                if is_intraday:
                    outside_range = exit_date < chart_start or exit_date > chart_end
                else:
                    # Daily charts: compare dates only (ignore time component)
                    exit_date_only = exit_date.date() if hasattr(exit_date, 'date') else exit_date
                    chart_start_date = chart_start.date() if hasattr(chart_start, 'date') else chart_start
                    chart_end_date = chart_end.date() if hasattr(chart_end, 'date') else chart_end
                    outside_range = exit_date_only < chart_start_date or exit_date_only > chart_end_date

                if outside_range:
                    continue

                # SKIP exits whose entry occurred AFTER tracked position entry
                # (these are exits from conflicting positions that shouldn't be shown)
                if tracked_entry_dt:
                    exit_entry_date = exit_trade.get('entry_date')
                    if exit_entry_date:
                        try:
                            exit_entry_dt = normalize_tz(pd.to_datetime(exit_entry_date))
                            if exit_entry_dt > tracked_entry_dt:
                                skipped_conflicting += 1
                                continue
                        except Exception:  # Catch all non-system exceptions
                            pass

                exits_in_range += 1

                # MISSED exits: orange color
                is_missed = exit_trade.get('missed', False)
                if is_missed:
                    color = 'orange'
                else:
                    # Exits are red down triangles (brighter red than candles)
                    color = '#ff5252'

                # Get x-position (bar number for intraday, date for daily)
                x_pos = get_x_position(exit_date)
                if x_pos is not None:
                    # Place SELL/EXIT marker ABOVE the candle (standard charting convention)
                    # Down-pointing triangle above bar = "sell/exit here"
                    bar_high = None
                    if plot_by_index and x_pos in df_plot['bar_num'].values:
                        bar_idx = df_plot[df_plot['bar_num'] == x_pos].index[0]
                        bar_high = df_plot.loc[bar_idx, 'high']
                    elif not plot_by_index:
                        # For daily charts, find bar by date in index
                        try:
                            if x_pos in df_plot.index:
                                bar_high = df_plot.loc[x_pos, 'high']
                            else:
                                target_date = pd.to_datetime(x_pos).date()
                                for idx in df_plot.index:
                                    if pd.to_datetime(idx).date() == target_date:
                                        bar_high = df_plot.loc[idx, 'high']
                                        break
                        except Exception:
                            pass

                    if bar_high is not None:
                        price_range = df_plot['high'].max() - df_plot['low'].min()
                        marker_offset = price_range * 0.02  # 2% offset above candle
                        y_pos = bar_high + marker_offset
                    else:
                        y_pos = exit_trade['price']  # Fallback to recorded price
                    ax1.scatter(x_pos, y_pos, marker='v', color=color, s=marker_size, zorder=5, edgecolors='white', linewidth=0.5)

            print(f"   [Chart] Exits: {exits_in_range}/{exits_total} in chart range")

        if skipped_conflicting > 0:
            print(f"   [Chart] Skipped {skipped_conflicting} markers that conflict with tracked position")

        # Plot GHOST MARKERS for missed signals (signals that occurred while bot was offline)
        # These are visually distinct: hollow markers with reduced opacity
        # They're NOT included in stats, but traders can see the strategy did generate signals
        missed_entries_plotted = 0
        missed_exits_plotted = 0
        if missed_signals:
            ghost_marker_size = marker_size * 0.8  # Slightly smaller than tracked markers
            ghost_alpha = 0.5  # 50% opacity for ghost effect

            # Plot missed entries as hollow green triangles
            for entry in missed_signals.get('entries', []):
                entry_date = entry.get('date')
                if not entry_date:
                    continue
                try:
                    entry_date = normalize_tz(pd.to_datetime(entry_date))
                except Exception:
                    continue

                # Skip if outside chart range
                if entry_date < chart_start or entry_date > chart_end:
                    continue

                x_pos = get_x_position(entry_date)
                if x_pos is not None:
                    # Place ghost BUY marker BELOW candle (standard charting convention)
                    bar_low = None
                    if plot_by_index and x_pos in df_plot['bar_num'].values:
                        bar_idx = df_plot[df_plot['bar_num'] == x_pos].index[0]
                        bar_low = df_plot.loc[bar_idx, 'low']
                    elif not plot_by_index:
                        try:
                            if x_pos in df_plot.index:
                                bar_low = df_plot.loc[x_pos, 'low']
                            else:
                                target_date = pd.to_datetime(x_pos).date()
                                for idx in df_plot.index:
                                    if pd.to_datetime(idx).date() == target_date:
                                        bar_low = df_plot.loc[idx, 'low']
                                        break
                        except Exception:
                            pass

                    if bar_low is not None:
                        price_range = df_plot['high'].max() - df_plot['low'].min()
                        y_pos = bar_low - price_range * 0.02
                    else:
                        y_pos = entry.get('price', df_plot['low'].min())

                    # Ghost entry: hollow green triangle (facecolors='none')
                    ax1.scatter(x_pos, y_pos, marker='^', s=ghost_marker_size, zorder=4,
                               facecolors='none', edgecolors='#00e676', linewidth=1.5, alpha=ghost_alpha)
                    missed_entries_plotted += 1

            # Plot missed exits as hollow red triangles
            for exit_trade in missed_signals.get('exits', []):
                exit_date = exit_trade.get('date')
                if not exit_date:
                    continue
                try:
                    exit_date = normalize_tz(pd.to_datetime(exit_date))
                except Exception:
                    continue

                # Skip if outside chart range
                if exit_date < chart_start or exit_date > chart_end:
                    continue

                x_pos = get_x_position(exit_date)
                if x_pos is not None:
                    # Place ghost EXIT marker ABOVE candle (standard charting convention)
                    bar_high = None
                    if plot_by_index and x_pos in df_plot['bar_num'].values:
                        bar_idx = df_plot[df_plot['bar_num'] == x_pos].index[0]
                        bar_high = df_plot.loc[bar_idx, 'high']
                    elif not plot_by_index:
                        try:
                            if x_pos in df_plot.index:
                                bar_high = df_plot.loc[x_pos, 'high']
                            else:
                                target_date = pd.to_datetime(x_pos).date()
                                for idx in df_plot.index:
                                    if pd.to_datetime(idx).date() == target_date:
                                        bar_high = df_plot.loc[idx, 'high']
                                        break
                        except Exception:
                            pass

                    if bar_high is not None:
                        price_range = df_plot['high'].max() - df_plot['low'].min()
                        y_pos = bar_high + price_range * 0.02
                    else:
                        y_pos = exit_trade.get('price', df_plot['high'].max())

                    # Ghost exit: hollow red triangle (facecolors='none')
                    ax1.scatter(x_pos, y_pos, marker='v', s=ghost_marker_size, zorder=4,
                               facecolors='none', edgecolors='#ff5252', linewidth=1.5, alpha=ghost_alpha)
                    missed_exits_plotted += 1

            if missed_entries_plotted + missed_exits_plotted > 0:
                print(f"   [Chart] Ghost markers: {missed_entries_plotted} entries, {missed_exits_plotted} exits (missed while offline)")
                # Add legend entry for ghost markers
                ax1.scatter([], [], marker='^', s=ghost_marker_size, facecolors='none',
                           edgecolors='gray', linewidth=1.5, alpha=ghost_alpha, label='Missed (not tracked)')

        # Mark current open position - use LOCKED trade_state (current_position) if provided
        # This ensures the entry line shows YOUR ACTUAL tracked position, not the recalculated one
        # Priority: current_position (trade_state) > locked_backtest > fresh backtest
        # IMPORTANT: If trade_state was provided but shows no position, DON'T fall through to backtest
        # (trade_state is authoritative - fresh backtest may calculate differently)
        trade_state_provided = current_position is not None
        trade_state_has_position = current_position and current_position.get('entry_price')

        if trade_state_has_position:
            ax1.axhline(current_position['entry_price'], color='cyan', linestyle='--', alpha=0.7,
                       label=f"Entry ${current_position['entry_price']:.2f}")

            # Add a STAR marker for the tracked position entry point
            # This visually distinguishes the user's ACTUAL position from historical backtest markers
            # IMPORTANT: Use entry_signal_bar (bar START time) for chart placement, NOT entry_time (bar CLOSE time)
            # For daily data especially, entry_time is 24h after the bar index, causing marker to be placed on wrong bar
            tracked_entry_time = current_position.get('entry_signal_bar') or current_position.get('entry_time')
            if tracked_entry_time:
                try:
                    tracked_entry_dt = normalize_tz(pd.to_datetime(tracked_entry_time))
                    tracked_x = get_x_position(tracked_entry_dt)
                    print(f"   [Chart] Tracked position: time={tracked_entry_time}, normalized={tracked_entry_dt}, x_pos={tracked_x}")
                    if tracked_x is not None:
                        # Place arrow BELOW the candle pointing UP (standard charting convention for LONG entry)
                        # Arrow originates below bar and points up to bar = "buy here"
                        bar_low = None
                        if plot_by_index and tracked_x in df_plot['bar_num'].values:
                            bar_idx = df_plot[df_plot['bar_num'] == tracked_x].index[0]
                            bar_low = df_plot.loc[bar_idx, 'low']
                        elif not plot_by_index:
                            try:
                                if tracked_x in df_plot.index:
                                    bar_low = df_plot.loc[tracked_x, 'low']
                                else:
                                    target_date = pd.to_datetime(tracked_x).date()
                                    for idx in df_plot.index:
                                        if pd.to_datetime(idx).date() == target_date:
                                            bar_low = df_plot.loc[idx, 'low']
                                            break
                            except Exception:
                                pass

                        price_range = df_plot['high'].max() - df_plot['low'].min()
                        if bar_low is not None:
                            arrow_head = bar_low - price_range * 0.01  # Arrow HEAD just below bar
                            arrow_tail = bar_low - price_range * 0.06  # Arrow TAIL further below
                        else:
                            arrow_head = current_position['entry_price']
                            arrow_tail = arrow_head - price_range * 0.05
                        # Green upward arrow for YOUR ENTRY (arrow draws from xytext to xy)
                        ax1.annotate('', xy=(tracked_x, arrow_head), xytext=(tracked_x, arrow_tail),
                                    arrowprops=dict(arrowstyle='->', color='#00e676', lw=2.5),
                                    zorder=15)
                        ax1.scatter(tracked_x, arrow_tail, marker='', s=0, label='YOUR ENTRY')  # For legend
                    else:
                        print(f"   [Chart] Warning: Could not find x position for tracked entry {tracked_entry_dt}")
                except Exception as e:
                    print(f"   [Chart] Warning: Could not plot tracked entry marker: {e}")

        elif not trade_state_provided and locked_backtest and locked_backtest.get('current_position'):
            # Only use locked_backtest position if no trade_state was passed
            pos = locked_backtest['current_position']
            ax1.axhline(pos['entry_price'], color='cyan', linestyle='--', alpha=0.7, label=f"Entry ${pos['entry_price']:.2f}")
        elif not trade_state_provided and backtest and backtest.get('current_position'):
            # Only use fresh backtest position if no trade_state was passed
            pos = backtest['current_position']
            ax1.axhline(pos['entry_price'], color='cyan', linestyle='--', alpha=0.7, label=f"Entry ${pos['entry_price']:.2f}")

        # Build title with date range info
        chart_title = f"{ticker} - JD Strategy{title_suffix}"
        try:
            # Determine which column has datetime info for title
            title_dt_col = None
            if datetime_col and datetime_col in df_plot.columns:
                title_dt_col = datetime_col
            elif '_datetime_for_labels' in df_plot.columns:
                title_dt_col = '_datetime_for_labels'

            if plot_by_index and title_dt_col and len(df_plot) > 0:
                start_dt = pd.to_datetime(df_plot[title_dt_col].iloc[0])
                end_dt = pd.to_datetime(df_plot[title_dt_col].iloc[-1])
                if not pd.isna(start_dt) and not pd.isna(end_dt):
                    if is_intraday:
                        date_range = f" ({start_dt.strftime('%m/%d')} - {end_dt.strftime('%m/%d %H:%M')})"
                    else:
                        date_range = f" ({start_dt.strftime('%Y-%m-%d')} - {end_dt.strftime('%Y-%m-%d')})"
                    chart_title = f"{ticker} - JD Strategy{title_suffix}{date_range}"
            elif isinstance(df_plot.index, pd.DatetimeIndex) and len(df_plot) > 0:
                start_dt = df_plot.index[0]
                end_dt = df_plot.index[-1]
                if not pd.isna(start_dt) and not pd.isna(end_dt):
                    if is_intraday:
                        date_range = f" ({start_dt.strftime('%m/%d')} - {end_dt.strftime('%m/%d %H:%M')})"
                    else:
                        date_range = f" ({start_dt.strftime('%Y-%m-%d')} - {end_dt.strftime('%Y-%m-%d')})"
                    chart_title = f"{ticker} - JD Strategy{title_suffix}{date_range}"
        except Exception:
            pass  # Keep default title if date formatting fails
        ax1.set_title(chart_title, color='white', fontsize=14, fontweight='bold')
        ax1.set_ylabel("Price ($)", color='white')
        # Only show legend if there are labeled items (avoids empty legend box)
        handles, labels = ax1.get_legend_handles_labels()
        if handles and labels:
            ax1.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white')
        format_xaxis(ax1)

        # 2. JD Oscillator with thresholds
        ax2 = axes[1]
        osc_col = 'osc_smooth' if 'osc_smooth' in df_plot.columns else 'composite_smooth'
        if osc_col in df_plot.columns:
            ax2.plot(x_axis, df_plot[osc_col], color='#e94560', linewidth=1.5, label='JD_Osc')
            ax2.axhline(config.get('oversold_threshold', -0.3), color='lime', linestyle='--', alpha=0.7, label='Oversold')
            ax2.axhline(config.get('overbought_threshold', 0.3), color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax2.fill_between(x_axis, df_plot[osc_col], 0,
                            where=(df_plot[osc_col] < config.get('oversold_threshold', -0.3)),
                            color='lime', alpha=0.3)
            ax2.fill_between(x_axis, df_plot[osc_col], 0,
                            where=(df_plot[osc_col] > config.get('overbought_threshold', 0.3)),
                            color='red', alpha=0.3)
        ax2.set_ylabel("JD_Osc", color='white')
        ax2.set_ylim(-1.2, 1.2)
        ax2.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white', fontsize='small')
        format_xaxis(ax2)

        # 3. JD Signal Indicators
        ax3 = axes[2]
        if 'velocity' in df_plot.columns:
            ax3.plot(x_axis, df_plot['velocity'], color='#00d9ff', linewidth=1.2, label='JD_Signal')
        if 'acceleration' in df_plot.columns:
            ax3.plot(x_axis, df_plot['acceleration'], color='#ffd700', linewidth=1.0, alpha=0.7, label='JD_Trend')
        ax3.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax3.set_ylabel("JD_Signal", color='white')
        ax3.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white', fontsize='small')
        format_xaxis(ax3)

        # 4. Equity Curve - Use locked_backtest (frozen historical data) to prevent repainting
        # For "signal" charts, filter exits to match df_plot date range so equity curve matches price chart
        ax4 = axes[3]
        equity_source = locked_backtest if locked_backtest and locked_backtest.get('exits') else backtest
        STARTING_CAPITAL = 100000  # $100k starting capital

        # Initialize exits_for_equity (used for both equity curve and stats)
        exits_for_equity = []

        # For signal charts, filter exits to match the visible df_plot date range
        # This ensures the equity curve shows the same period as the price chart
        if equity_source and equity_source.get('exits'):
            exits_for_equity = equity_source['exits']

            # Filter exits to df_plot date range for signal charts
            if chart_type == "signal" and len(df_plot) > 0:
                # CRITICAL: Get datetime from the correct column, not from index
                # After reset_index(), df_plot.index is RangeIndex (0,1,2...), not DatetimeIndex!
                # The datetime is stored in datetime_col (intraday) or _datetime_for_labels (daily)
                # Also respect datetime_already_chicago to avoid double-conversion bug
                dt_col_for_filter = None
                if '_datetime_col' in df_plot.columns:
                    dt_col_for_filter = df_plot['_datetime_col'].iloc[0]  # Get the column name
                if dt_col_for_filter and dt_col_for_filter in df_plot.columns:
                    if datetime_already_chicago:
                        chart_start = pd.to_datetime(df_plot[dt_col_for_filter].iloc[0])
                        chart_end = pd.to_datetime(df_plot[dt_col_for_filter].iloc[-1])
                    else:
                        chart_start = normalize_tz(pd.to_datetime(df_plot[dt_col_for_filter].iloc[0]))
                        chart_end = normalize_tz(pd.to_datetime(df_plot[dt_col_for_filter].iloc[-1]))
                elif '_datetime_for_labels' in df_plot.columns:
                    chart_start = normalize_tz(pd.to_datetime(df_plot['_datetime_for_labels'].iloc[0]))
                    chart_end = normalize_tz(pd.to_datetime(df_plot['_datetime_for_labels'].iloc[-1]))
                elif isinstance(df_plot.index, pd.DatetimeIndex):
                    chart_start = normalize_tz(df_plot.index.min())
                    chart_end = normalize_tz(df_plot.index.max())
                else:
                    # Fallback: use full exits if we can't determine date range
                    chart_start = None
                    chart_end = None

                if chart_start is not None and chart_end is not None:
                    exits_for_equity = [
                        e for e in equity_source['exits']
                        if chart_start <= normalize_tz(pd.to_datetime(e['date'])) <= chart_end
                    ]
                    # If no exits in range, show flat line (0 trades in visible range)
                    # Don't fall back to full exits - that causes misleading stats!
                else:
                    exits_for_equity = equity_source['exits']

            # Build equity curve from trades (starting with $100k, compounding full position)
            equity = [STARTING_CAPITAL]
            trade_dates = []

            # Get first entry date as starting point (normalize_tz handles tz-aware/tz-naive mix)
            entries_for_equity = equity_source.get('entries', [])
            if chart_type == "signal" and len(df_plot) > 0 and entries_for_equity:
                # Filter entries to chart range too - use same date extraction logic
                # Respect datetime_already_chicago to avoid double-conversion bug
                dt_col_for_filter = None
                if '_datetime_col' in df_plot.columns:
                    dt_col_for_filter = df_plot['_datetime_col'].iloc[0]
                if dt_col_for_filter and dt_col_for_filter in df_plot.columns:
                    if datetime_already_chicago:
                        chart_start = pd.to_datetime(df_plot[dt_col_for_filter].iloc[0])
                        chart_end = pd.to_datetime(df_plot[dt_col_for_filter].iloc[-1])
                    else:
                        chart_start = normalize_tz(pd.to_datetime(df_plot[dt_col_for_filter].iloc[0]))
                        chart_end = normalize_tz(pd.to_datetime(df_plot[dt_col_for_filter].iloc[-1]))
                elif '_datetime_for_labels' in df_plot.columns:
                    chart_start = normalize_tz(pd.to_datetime(df_plot['_datetime_for_labels'].iloc[0]))
                    chart_end = normalize_tz(pd.to_datetime(df_plot['_datetime_for_labels'].iloc[-1]))
                elif isinstance(df_plot.index, pd.DatetimeIndex):
                    chart_start = normalize_tz(df_plot.index.min())
                    chart_end = normalize_tz(df_plot.index.max())
                else:
                    chart_start = None
                    chart_end = None

                if chart_start is not None and chart_end is not None:
                    entries_for_equity = [
                        e for e in entries_for_equity
                        if chart_start <= normalize_tz(pd.to_datetime(e['date'])) <= chart_end
                    ]

            if entries_for_equity and len(entries_for_equity) > 0:
                first_date = normalize_tz(entries_for_equity[0]['date'])
                trade_dates.append(first_date)
            elif len(exits_for_equity) > 0:
                # Fallback to first exit date
                first_date = normalize_tz(exits_for_equity[0]['date'])
                trade_dates.append(first_date)

            for exit in exits_for_equity:
                equity.append(equity[-1] * (1 + exit['pnl']/100))
                trade_dates.append(normalize_tz(exit['date']))

            # Add unrealized if open position (use .get() to handle missing key)
            if equity_source.get('current_position'):
                unrealized = equity_source['current_position'].get('unrealized_pnl', 0)
                if unrealized:
                    equity.append(equity[-1] * (1 + unrealized/100))
                    # Use current time for unrealized position (already tz-naive)
                    trade_dates.append(pd.Timestamp.now())

            # Plot with dates on x-axis
            if len(trade_dates) == len(equity):
                ax4.plot(trade_dates, equity, color='#00ff88', linewidth=2)
                ax4.fill_between(trade_dates, STARTING_CAPITAL, equity, alpha=0.3,
                               color='green' if equity[-1] > STARTING_CAPITAL else 'red')
            else:
                # Fallback to trade numbers if date mismatch
                ax4.plot(range(len(equity)), equity, color='#00ff88', linewidth=2)
                ax4.fill_between(range(len(equity)), STARTING_CAPITAL, equity, alpha=0.3,
                               color='green' if equity[-1] > STARTING_CAPITAL else 'red')

            ax4.axhline(STARTING_CAPITAL, color='gray', linestyle='--', alpha=0.5)
            # Format y-axis as currency
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))

            # Format x-axis dates - auto-select based on date range
            if len(trade_dates) >= 2:
                date_range = (trade_dates[-1] - trade_dates[0]).days
                if date_range > 365:
                    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                elif date_range > 60:
                    ax4.xaxis.set_major_locator(mdates.MonthLocator())
                    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                elif date_range > 7:
                    ax4.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                else:
                    ax4.xaxis.set_major_locator(mdates.DayLocator())
                    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax4.set_ylabel("Equity ($)", color='white')
        ax4.set_xlabel("Date", color='white')

        # Stats annotation - Use locked_backtest (frozen data) to prevent repainting
        # For signal charts, calculate stats from filtered exits to match visible period
        stats_source = locked_backtest if locked_backtest and locked_backtest.get('num_trades') else backtest

        # For signal charts, ALWAYS recalculate stats from filtered exits to match equity curve
        # This ensures stats shown match the visible chart period, not full backtest
        if chart_type == "signal":
            # Use centralized stats calculation on filtered exits
            filtered_stats = calculate_exit_stats(exits_for_equity, exclude_missed=True)
            stats_source = {
                'num_trades': filtered_stats['num_trades'],
                'win_rate': filtered_stats['win_rate'],
                'total_return': filtered_stats['total_return'],
                'profit_factor': filtered_stats['profit_factor'],
            }

        if full_period_backtest and stats_source:
            # Show both: Full period on top, Recent below
            full_stats = (f"Full Period: {full_period_backtest['num_trades']} trades | "
                         f"Win: {full_period_backtest['win_rate']:.0f}% | "
                         f"Return: {full_period_backtest['total_return']:.1f}% | "
                         f"PF: {full_period_backtest['profit_factor']:.1f}")
            # Use period_days from stats_source if available, otherwise default to chart period
            period_label = f"Recent {stats_source.get('period_days', len(df_plot))}D"
            recent_stats = (f"{period_label}: {stats_source['num_trades']} trades | "
                           f"Win: {stats_source['win_rate']:.0f}% | "
                           f"Return: {stats_source['total_return']:.1f}% | "
                           f"PF: {stats_source['profit_factor']:.1f}")
            fig.text(0.5, 0.035, full_stats, ha='center', color='white', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.8))
            fig.text(0.5, 0.008, recent_stats, ha='center', color='#00ff88', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.8))
        elif stats_source:
            stats_text = (f"Trades: {stats_source['num_trades']} | "
                         f"Win: {stats_source['win_rate']:.0f}% | "
                         f"Return: {stats_source['total_return']:.1f}% | "
                         f"PF: {stats_source['profit_factor']:.1f}")
            fig.text(0.5, 0.02, stats_text, ha='center', color='white', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.8))

        plt.tight_layout()
        # Adjust bottom margin based on whether we have one or two stat lines
        bottom_margin = 0.14 if full_period_backtest else 0.12
        plt.subplots_adjust(bottom=bottom_margin)

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf

    except Exception as e:
        print(f"Error generating chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_bar_regimes(df, config):
    """Compute regime classification for each bar. Returns (regimes, is_high_vol, regime_names_map).
    Only active when config has regime_aware=True and v9 module is available."""
    if not config.get('regime_aware', False) or not V9_REGIME_AVAILABLE:
        return None, None

    rd = config.get('regime_detector', {})
    close_col = 'close' if 'close' in df.columns else 'Close'
    high_col = 'high' if 'high' in df.columns else 'High'
    low_col = 'low' if 'low' in df.columns else 'Low'

    regimes, is_high_vol = classify_regimes(
        df[high_col].values.astype(float),
        df[low_col].values.astype(float),
        df[close_col].values.astype(float),
        adx_period=rd.get('adx_period', 14),
        adx_threshold=rd.get('adx_threshold', 25.0),
        atr_period=rd.get('atr_period', 14),
        atr_high_vol_percentile=rd.get('atr_high_vol_percentile', 90.0),
        atr_high_vol_lookback=rd.get('atr_high_vol_lookback', 100),
        regime_min_bars=rd.get('regime_min_bars', 4),
    )
    return regimes, is_high_vol


def get_regime_params(config, regime_id):
    """Get the regime-specific config dict for a given regime_id.
    Falls back to top-level config if regime_aware is off."""
    if not config.get('regime_aware', False):
        return config

    regime_params = config.get('regime_params', {})
    regime_name = REGIME_NAMES.get(regime_id, 'chop')
    r_config = regime_params.get(regime_name, {})
    if r_config.get('dont_trade', False):
        return None  # Signal: don't trade this regime
    return r_config


def run_historical_backtest(df: pd.DataFrame, config: dict) -> dict:
    """
    Run a backtest on historical data to determine current state.
    Returns dict with trades, stats, and current position.

    IMPORTANT: All trade dates use BAR CLOSE time (when the signal fires and trade executes),
    not bar START time. This matches how live trading works.
    """
    # Regime-aware mode: compute regimes and per-regime signals
    is_regime_aware = config.get('regime_aware', False) and V9_REGIME_AVAILABLE
    regimes = None
    is_high_vol = None
    regime_signal_dfs = {}  # regime_name -> df with that regime's signals

    if is_regime_aware:
        regimes, is_high_vol = compute_bar_regimes(df, config)
        suppress_high_vol = config.get('regime_detector', {}).get('suppress_entries_high_vol', True)
        # Generate signals for each regime's config
        regime_params_dict = config.get('regime_params', {})
        for regime_name, r_cfg in regime_params_dict.items():
            if r_cfg.get('dont_trade', False):
                continue
            # Merge top-level wavelet/oscillator settings into regime config
            merged = {**config, **r_cfg}
            regime_signal_dfs[regime_name] = calculate_velocity_signals(df.copy(), merged)
        # Use first active regime's df for shared columns (osc_smooth, velocity, etc.)
        if regime_signal_dfs:
            first_regime_df = next(iter(regime_signal_dfs.values()))
            for col in ['osc_smooth', 'velocity', 'acceleration', 'jerk']:
                if col in first_regime_df.columns:
                    df[col] = first_regime_df[col]
        # Also generate default signals for charting (uses top-level config)
        df = calculate_velocity_signals(df, config)
    else:
        # Standard v8 path
        df = calculate_velocity_signals(df, config)

    # Get default config params (v8 or fallback)
    stop_loss_pct = config.get('stop_loss_pct', 5.0)
    take_profit_pct = config.get('take_profit_pct', 10.0)
    min_bars_between = config.get('min_bars_between', 1)
    exit_on_opposite = config.get('exit_on_opposite_signal', True)
    exit_on_midline = config.get('exit_on_midline_cross', False)

    # Acceleration Reversal Exit parameters
    use_accel_exit = config.get('use_accel_exit', False)
    accel_exit_type = config.get('accel_exit_type', 'sign_reversal')
    accel_exit_threshold = config.get('accel_exit_threshold', 0.0)
    accel_exit_min_pnl = config.get('accel_exit_min_pnl', 0.5)
    accel_exit_lookback = config.get('accel_exit_lookback', 1)
    use_jerk_confirm = config.get('use_jerk_confirm', False)
    jerk_confirm_threshold = config.get('jerk_confirm_threshold', 0.0)

    # Get interval for calculating bar close time
    # df.index contains bar START times, but trades happen at bar CLOSE
    interval_str = config.get('interval', '1d')
    interval_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '90m': 90, '4h': 240, '1d': 0}.get(interval_str, 0)
    # For daily bars, we don't add offset (close time is market close, not midnight + 1 day)
    bar_offset = pd.Timedelta(minutes=interval_minutes) if interval_minutes > 0 else pd.Timedelta(0)

    # Run simulation
    position = 0
    entry_price = None
    entry_date = None
    entry_osc = None
    entry_regime = None  # Track which regime we entered under (for v9)
    last_trade_bar = -min_bars_between
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['close']
        bar_start_time = df.index[i]
        bar_close_time = bar_start_time + bar_offset  # When signal fires and trade executes
        osc = row['osc_smooth'] if 'osc_smooth' in df.columns else 0
        bars_since = i - last_trade_bar

        # Get current bar's regime (v9) or None (v8)
        current_regime_id = int(regimes[i]) if regimes is not None else None
        current_regime_name = REGIME_NAMES.get(current_regime_id) if current_regime_id is not None else None

        # For exits, use ENTRY regime's params (v9) or top-level config (v8)
        def _exit_param(key, default):
            if is_regime_aware and entry_regime is not None:
                r_cfg = config.get('regime_params', {}).get(entry_regime, {})
                return r_cfg.get(key, config.get(key, default))
            return config.get(key, default)

        # Check exits first
        if position == 1 and entry_price:
            pnl_pct = ((price - entry_price) / entry_price) * 100
            exit_reason = None

            # Use entry regime's exit params
            e_stop_loss = _exit_param('stop_loss_pct', stop_loss_pct)
            e_take_profit = _exit_param('take_profit_pct', take_profit_pct)
            e_use_accel = _exit_param('use_accel_exit', use_accel_exit)
            e_accel_type = _exit_param('accel_exit_type', accel_exit_type)
            e_accel_threshold = _exit_param('accel_exit_threshold', accel_exit_threshold)
            e_accel_min_pnl = _exit_param('accel_exit_min_pnl', accel_exit_min_pnl)
            e_accel_lookback = _exit_param('accel_exit_lookback', accel_exit_lookback)
            e_jerk_confirm = _exit_param('use_jerk_confirm', use_jerk_confirm)
            e_jerk_threshold = _exit_param('jerk_confirm_threshold', jerk_confirm_threshold)
            e_exit_midline = _exit_param('exit_on_midline_cross', exit_on_midline)
            e_exit_opposite = _exit_param('exit_on_opposite_signal', exit_on_opposite)

            if e_stop_loss > 0 and pnl_pct <= -e_stop_loss:
                exit_reason = "Stop Loss"
            elif e_take_profit > 0 and pnl_pct >= e_take_profit:
                exit_reason = "Take Profit"
            # Acceleration Reversal Exit (early warning before stop loss)
            elif e_use_accel and 'acceleration' in df.columns and i >= e_accel_lookback:
                # Check if conditions are met
                pnl_ok = pnl_pct >= e_accel_min_pnl or pnl_pct < 0
                if pnl_ok:
                    accel_values = df['acceleration'].iloc[i-e_accel_lookback+1:i+1].values
                    current_accel = df['acceleration'].iloc[i]
                    # For LONG positions: negative acceleration is bearish
                    accel_cond = False
                    if e_accel_type == 'sign_reversal':
                        accel_cond = all(a < 0 for a in accel_values)
                    elif e_accel_type == 'magnitude':
                        accel_cond = current_accel < -e_accel_threshold
                    elif e_accel_type == 'both':
                        accel_cond = all(a < 0 for a in accel_values) and abs(current_accel) > e_accel_threshold
                    # Jerk confirmation (optional)
                    jerk_cond = True
                    if e_jerk_confirm and 'jerk' in df.columns and e_jerk_threshold > 0:
                        current_jerk = df['jerk'].iloc[i]
                        jerk_cond = current_jerk < -e_jerk_threshold
                    if accel_cond and jerk_cond:
                        exit_reason = "Accel Reversal"
            elif e_exit_midline and osc > 0:
                exit_reason = "Midline Cross"
            elif e_exit_opposite and row.get('sell_signal', False) and bars_since >= min_bars_between:
                exit_reason = "Opposite Signal"

            if exit_reason:
                trade_exit = {
                    'type': 'exit',
                    'date': bar_close_time,  # Use bar CLOSE time
                    'price': price,
                    'pnl': pnl_pct,
                    'reason': exit_reason,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                }
                if entry_regime:
                    trade_exit['regime'] = entry_regime
                trades.append(trade_exit)
                position = 0
                entry_price = None
                entry_date = None
                entry_regime = None
                last_trade_bar = i

        # Check entries
        if position == 0 and bars_since >= min_bars_between:
            # Determine if we have a buy signal
            has_buy = False

            if is_regime_aware and current_regime_name is not None:
                # v9: check regime allows trading + regime-specific buy signal
                if suppress_high_vol and is_high_vol is not None and is_high_vol[i]:
                    has_buy = False
                elif current_regime_name in regime_signal_dfs:
                    regime_df = regime_signal_dfs[current_regime_name]
                    has_buy = bool(regime_df.iloc[i].get('buy_signal', False))
                # Regimes not in regime_signal_dfs are dont_trade
            else:
                # v8: standard buy signal check
                has_buy = bool(row.get('buy_signal', False))

            if has_buy:
                position = 1
                entry_price = price
                entry_date = bar_close_time  # Use bar CLOSE time
                entry_osc = osc
                entry_regime = current_regime_name  # None for v8
                last_trade_bar = i
                trade_entry = {
                    'type': 'entry',
                    'date': bar_close_time,  # Use bar CLOSE time
                    'price': price,
                    'osc': osc,
                }
                if entry_regime:
                    trade_entry['regime'] = entry_regime
                trades.append(trade_entry)

    # Calculate stats
    exits = [t for t in trades if t['type'] == 'exit']
    entries = [t for t in trades if t['type'] == 'entry']

    if exits:
        pnls = [t['pnl'] for t in exits]
        wins = [t for t in exits if t['pnl'] > 0]
        losses = [t for t in exits if t['pnl'] <= 0]
        total_return = (np.prod([1 + p/100 for p in pnls]) - 1) * 100
        win_rate = len(wins) / len(exits) * 100
        avg_pnl = np.mean(pnls)
        gross_profit = sum([t['pnl'] for t in wins]) if wins else 0
        gross_loss = abs(sum([t['pnl'] for t in losses])) if losses else 0.01
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    else:
        total_return = 0
        win_rate = 0
        avg_pnl = 0
        profit_factor = 0
        pnls = []

    # Current position info
    current_position = None
    if position == 1 and entry_price:
        current_price = df['close'].iloc[-1]
        unrealized_pnl = ((current_price - entry_price) / entry_price) * 100
        current_position = {
            'position': 'long',
            'entry_price': entry_price,
            'entry_date': str(entry_date),
            'current_price': current_price,
            'unrealized_pnl': unrealized_pnl,
        }
        if entry_regime:
            current_position['entry_regime'] = entry_regime

    # Current regime for the latest bar (for display)
    current_regime_info = None
    if is_regime_aware and regimes is not None and len(regimes) > 0:
        last_regime_id = int(regimes[-1])
        current_regime_info = REGIME_NAMES.get(last_regime_id, 'unknown')

    return {
        'trades': trades,
        'exits': exits,
        'entries': entries,
        'total_return': total_return,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'profit_factor': profit_factor,
        'num_trades': len(exits),
        'current_position': current_position,
        'current_regime': current_regime_info,
    }


def run_backtest_for_period(df: pd.DataFrame, config: dict, days: int = None) -> tuple:
    """
    Run backtest for a specific period of days.
    Returns (filtered_df, backtest_results).

    Args:
        df: Full DataFrame with price and indicator data
        config: Strategy config
        days: Number of days to include (from end). If None, use all data.
    """
    if days is not None and len(df) > days:
        df_period = df.iloc[-days:].copy()
    else:
        df_period = df.copy()

    # Recalculate signals on the filtered data
    df_period = calculate_velocity_signals(df_period, config)

    # Run backtest on filtered data
    backtest = run_historical_backtest(df_period, config)

    return df_period, backtest


def get_state_file_path(strategy_name: str = None, ticker: str = None) -> str:
    """Get strategy-specific state file path.

    Uses strategy_name if provided (allows multiple strategies per ticker),
    falls back to ticker for backwards compatibility.
    """
    if strategy_name:
        safe_name = strategy_name.replace("/", "-").replace(":", "-").replace(" ", "_")
        return f"velocity_trade_state_{safe_name}.json"
    elif ticker:
        safe_ticker = ticker.replace("/", "-").replace(":", "-").replace(" ", "_")
        return f"velocity_trade_state_{safe_ticker}.json"
    return "velocity_trade_state.json"


def get_history_file_path(strategy_name: str = None, ticker: str = None) -> str:
    """Get strategy-specific trade history file path.

    Uses strategy_name if provided (allows multiple strategies per ticker),
    falls back to ticker for backwards compatibility.
    """
    if strategy_name:
        safe_name = strategy_name.replace("/", "-").replace(":", "-").replace(" ", "_")
        return f"velocity_trade_history_{safe_name}.json"
    elif ticker:
        safe_ticker = ticker.replace("/", "-").replace(":", "-").replace(" ", "_")
        return f"velocity_trade_history_{safe_ticker}.json"
    return "velocity_trade_history.json"


def get_locked_backtest_path(strategy_name: str = None, ticker: str = None) -> str:
    """Get strategy-specific locked backtest file path.

    The locked backtest stores frozen entry/exit signals that don't repaint.
    """
    if strategy_name:
        safe_name = strategy_name.replace("/", "-").replace(":", "-").replace(" ", "_")
        return f"velocity_locked_backtest_{safe_name}.json"
    elif ticker:
        safe_ticker = ticker.replace("/", "-").replace(":", "-").replace(" ", "_")
        return f"velocity_locked_backtest_{safe_ticker}.json"
    return "velocity_locked_backtest.json"


def save_locked_backtest(backtest: dict, strategy_name: str = None, ticker: str = None, trade_state: dict = None):
    """Save or update locked backtest snapshot (prevents repainting).

    IMPORTANT: This function now PRESERVES historical data.
    - If a locked backtest exists, it loads it and only appends NEW entries/exits
    - Stats are recalculated from all exits (historical + new)
    - This prevents losing historical trade data on each restart

    If trade_state has a tracked position, we reconcile the backtest
    with it - using the tracked position as the source of truth.
    """
    path = get_locked_backtest_path(strategy_name=strategy_name, ticker=ticker)

    # Try to load existing locked backtest to preserve historical data
    existing_locked = None
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                existing_locked = json.load(f)
            print(f"   📂 Loaded existing locked backtest: {len(existing_locked.get('entries', []))} entries, {len(existing_locked.get('exits', []))} exits")
        except Exception as e:
            print(f"   ⚠️ Could not load existing locked backtest: {e}")

    # Find the latest exit date in existing locked backtest
    # Only add trades AFTER this date (don't try to merge overlapping periods)
    latest_exit_date = None
    if existing_locked and existing_locked.get('exits'):
        for exit_trade in existing_locked['exits']:
            exit_date_str = str(exit_trade.get('date', ''))[:10]
            if exit_date_str and (latest_exit_date is None or exit_date_str > latest_exit_date):
                latest_exit_date = exit_date_str
        if latest_exit_date:
            print(f"   📅 Latest exit in locked backtest: {latest_exit_date}")

    # Start with existing data or empty
    locked = {
        "locked_at": datetime.now().isoformat(),
        "entries": existing_locked.get('entries', []) if existing_locked else [],
        "exits": existing_locked.get('exits', []) if existing_locked else [],
        "current_position": None,
        "num_trades": 0,
        "win_rate": 0,
        "total_return": 0,
        "profit_factor": 0,
    }

    # Get the tracked position's entry date (if any) for reconciliation
    tracked_entry_date = None
    if trade_state and trade_state.get('position') and trade_state.get('entry_time'):
        try:
            tracked_entry_str = str(trade_state['entry_time']).split('.')[0]
            tracked_entry_date = normalize_tz(pd.to_datetime(tracked_entry_str))
            print(f"   📍 Tracked position: {trade_state['position'].upper()} @ ${trade_state.get('entry_price', 0):.2f} on {tracked_entry_date}")
        except Exception:  # Catch all non-system exceptions
            pass

    # Add NEW entries from fresh backtest (only entries AFTER latest exit date)
    new_entries_added = 0
    skipped_overlap = 0
    skipped_timestamp_dupe = 0
    skipped_market_closed = 0  # Track entries skipped due to market closure (phantom signals)
    skipped_tracked_conflict = set()  # Track entries skipped due to tracked position conflict
    # Build set of existing entry timestamps for deduplication
    existing_entry_timestamps = {str(e.get('date', ''))[:19] for e in locked["entries"]}

    # Build map of entry timestamp -> exit info from fresh backtest
    # Used to detect entries whose exits would overlap with tracked position
    # Also used to detect stale data (0% P&L trades)
    entry_to_exit_map = {}
    entry_to_exit_pnl = {}  # For stale data detection
    for exit_t in backtest.get('exits', []):
        entry_ts = str(exit_t.get('entry_date', ''))[:19]
        exit_ts = str(exit_t.get('date', ''))[:19]
        exit_pnl = exit_t.get('pnl', 0)
        if entry_ts:
            entry_to_exit_map[entry_ts] = exit_ts
            entry_to_exit_pnl[entry_ts] = exit_pnl

    for entry in backtest.get('entries', []):
        entry_date = entry['date']
        entry_date_str = str(entry_date)[:10]
        entry_timestamp_str = str(entry_date)[:19]  # Full timestamp for dedup

        # Skip if exact timestamp already exists (prevent duplicates)
        if entry_timestamp_str in existing_entry_timestamps:
            skipped_timestamp_dupe += 1
            continue

        # Skip entries during market closure (phantom signals from data gaps)
        # This prevents adding false signals with frozen prices during CME weekend closure etc.
        if is_market_closed(entry['date'], ticker):
            skipped_market_closed += 1
            continue

        # Skip entries with 0% P&L exits (stale/frozen price data)
        # This catches phantom trades near market open where prices haven't updated
        exit_pnl = entry_to_exit_pnl.get(entry_timestamp_str)
        if exit_pnl is not None and exit_pnl == 0:
            skipped_market_closed += 1  # Count with market closed (same category - phantom signals)
            continue

        # Only add entries AFTER the latest exit in locked backtest
        # This prevents duplicates from overlapping data periods
        if latest_exit_date and entry_date_str <= latest_exit_date:
            continue

        # Parse date for comparison with tracked position
        if isinstance(entry_date, str):
            try:
                entry_date = pd.to_datetime(entry_date)
            except Exception:  # Catch all non-system exceptions
                pass
        entry_date = normalize_tz(entry_date) if hasattr(entry_date, 'tzinfo') else entry_date

        # Skip entries that conflict with tracked position
        # Case 1: Entry is at or after tracked entry time
        # Case 2: Entry's exit would be at or after tracked entry time (overlapping trade)
        if tracked_entry_date is not None and hasattr(entry_date, 'date'):
            # Case 1: Entry at or after tracked position
            if entry_date >= tracked_entry_date:
                print(f"   ⏭️  Skipping backtest entry {entry_date} (conflicts with tracked position)")
                skipped_tracked_conflict.add(str(entry_date)[:19])  # Track this entry so we skip its exit too
                continue

            # Case 2: Entry before tracked, but exit at or after tracked (overlapping trade)
            exit_ts = entry_to_exit_map.get(entry_timestamp_str)
            if exit_ts:
                try:
                    exit_dt = normalize_tz(pd.to_datetime(exit_ts))
                    if exit_dt >= tracked_entry_date:
                        print(f"   ⏭️  Skipping backtest entry {entry_date} (its exit {exit_ts} overlaps tracked position)")
                        skipped_tracked_conflict.add(entry_timestamp_str)
                        continue
                except Exception:  # Catch all non-system exceptions
                    pass

        # VALIDATION: Check if there's already an unexited entry in locked backtest
        # We can't enter a new position if we're already in one
        # Use full timestamps ([:19]) for intraday support, not just dates ([:10])
        existing_entry_timestamps = {str(e.get('date', ''))[:19] for e in locked["entries"]}
        existing_exit_entry_timestamps = {str(e.get('entry_date', ''))[:19] for e in locked["exits"]}
        unexited_entry_timestamps = existing_entry_timestamps - existing_exit_entry_timestamps

        if unexited_entry_timestamps:
            # There's at least one unexited entry - check if this new entry would overlap
            # Get the latest unexited entry timestamp
            latest_unexited = max(unexited_entry_timestamps)
            if entry_timestamp_str > latest_unexited:
                # This entry is after an unexited entry - skip it (can't enter while in position)
                skipped_overlap += 1
                continue

        locked["entries"].append({
            "date": str(entry['date']),
            "price": entry['price'],
            "position": entry.get('position', 'long')
        })
        existing_entry_timestamps.add(entry_timestamp_str)  # Prevent within-batch duplicates
        new_entries_added += 1

    if new_entries_added > 0:
        print(f"   ➕ Added {new_entries_added} new entries (after {latest_exit_date})")
    if skipped_overlap > 0:
        print(f"   ⏭️  Skipped {skipped_overlap} entries (already in position)")
    if skipped_timestamp_dupe > 0:
        print(f"   ⏭️  Skipped {skipped_timestamp_dupe} duplicate entries (same timestamp)")
    if skipped_market_closed > 0:
        print(f"   ⏭️  Skipped {skipped_market_closed} phantom entries (market was closed)")

    # Add NEW exits from fresh backtest (only exits AFTER latest exit date)
    new_exits_added = 0
    skipped_same_day = 0
    skipped_duplicate = 0
    skipped_end_of_period = 0
    skipped_no_entry = 0
    skipped_exit_market_closed = 0  # Track exits skipped due to market closure

    # Build set of entry timestamps in locked backtest for validation
    # Note: We ONLY use exact timestamps, not date-only matching (stricter validation)
    locked_entry_timestamps = {str(e.get('date', ''))[:19] for e in locked["entries"]}

    for exit_trade in backtest.get('exits', []):
        exit_date_str = str(exit_trade['date'])[:10]
        entry_date_str = str(exit_trade.get('entry_date', ''))[:10]

        # Only add exits AFTER the latest exit in locked backtest
        if latest_exit_date and exit_date_str <= latest_exit_date:
            continue

        # VALIDATION 0: Reject "end_of_period" artificial exits (not real signals)
        if exit_trade.get('reason') == 'end_of_period':
            skipped_end_of_period += 1
            continue

        # VALIDATION 0a: Skip exits during market closure (phantom signals from data gaps)
        if is_market_closed(exit_trade['date'], ticker):
            skipped_exit_market_closed += 1
            continue

        # VALIDATION 0a2: Skip exits with 0% P&L (stale/frozen price data)
        if exit_trade.get('pnl', 0) == 0:
            skipped_exit_market_closed += 1  # Count with market closed (phantom signals)
            continue

        # VALIDATION 0b: Reject exits whose entries were skipped (tracked position conflict)
        exit_entry_timestamp = str(exit_trade.get('entry_date', ''))[:19]
        if exit_entry_timestamp in skipped_tracked_conflict:
            print(f"   ⏭️  Skipping exit for entry {exit_entry_timestamp} (entry was skipped - tracked conflict)")
            continue

        # VALIDATION 1: Reject same-TIMESTAMP entry+exit (can't trade on same bar)
        # Use full timestamp for intraday support - same day is valid, same bar is not
        exit_timestamp_str = str(exit_trade['date'])[:19]
        entry_ts_str = str(exit_trade.get('entry_date', ''))[:19]
        if exit_timestamp_str == entry_ts_str:
            skipped_same_day += 1
            continue

        # VALIDATION 2: Reject duplicate exits (entry already has an exit)
        # Use full timestamp ([:19]) for intraday support
        entry_timestamp_str = str(exit_trade.get('entry_date', ''))[:19]
        entry_already_exited = any(
            str(e.get('entry_date', ''))[:19] == entry_timestamp_str and
            abs(e.get('entry_price', 0) - exit_trade.get('entry_price', 0)) < 0.01
            for e in locked["exits"]
        )
        if entry_already_exited:
            skipped_duplicate += 1
            continue

        # VALIDATION 3: Reject exits whose entries don't exist in locked backtest
        # IMPORTANT: For intraday data, we MUST match the exact timestamp, not just date
        # The date-only fallback was causing orphan exits (e.g., exit claims entry at 02:15
        # but that specific entry doesn't exist, even though other entries exist on that date)
        entry_timestamp_str = str(exit_trade.get('entry_date', ''))[:19]
        if entry_timestamp_str not in locked_entry_timestamps:
            skipped_no_entry += 1
            continue

        locked["exits"].append({
            "date": str(exit_trade['date']),
            "price": exit_trade['price'],
            "pnl": exit_trade['pnl'],
            "reason": exit_trade.get('reason', ''),
            "entry_price": exit_trade.get('entry_price', 0),
            "entry_date": str(exit_trade.get('entry_date', ''))
        })
        new_exits_added += 1

    if new_exits_added > 0:
        print(f"   ➕ Added {new_exits_added} new exits (after {latest_exit_date})")
    if skipped_same_day > 0:
        print(f"   ⚠️ Skipped {skipped_same_day} same-day entry+exit (invalid)")
    if skipped_duplicate > 0:
        print(f"   ⚠️ Skipped {skipped_duplicate} duplicate exits (entry already closed)")
    if skipped_end_of_period > 0:
        print(f"   ⏭️  Skipped {skipped_end_of_period} end_of_period exits (artificial backtest close)")
    if skipped_no_entry > 0:
        print(f"   ⚠️ Skipped {skipped_no_entry} orphan exits (no matching entry in locked backtest)")
    if skipped_exit_market_closed > 0:
        print(f"   ⏭️  Skipped {skipped_exit_market_closed} phantom exits (market was closed)")

    # Handle tracked position - add or update
    if trade_state and trade_state.get('position') and trade_state.get('entry_price'):
        # IMPORTANT: Use entry_time (bar close time) for the entry date
        # This matches how entries are created during live trading (bar close = when we enter)
        entry_time = trade_state.get('entry_time', '')
        tracked_entry_timestamp = str(entry_time)[:19]

        # Check if entry already exists (avoid duplicates)
        existing_entry = any(str(e.get('date', ''))[:19] == tracked_entry_timestamp for e in locked["entries"])

        # CRITICAL: Also check if we're already in a position (entries > exits)
        # This prevents adding a second entry when there's already an open position
        num_entries = len(locked.get("entries", []))
        num_exits = len(locked.get("exits", []))
        already_in_position = num_entries > num_exits

        if not existing_entry and not already_in_position:
            # Add the tracked position as entry (use bar close time)
            locked["entries"].append({
                "date": str(entry_time),
                "price": trade_state['entry_price'],
                "position": trade_state['position']
            })
            print(f"   ✅ Synced tracked {trade_state['position'].upper()} entry in locked backtest")
        elif already_in_position and not existing_entry:
            print(f"   ⚠️  Skipping sync entry: Already in position (entries: {num_entries}, exits: {num_exits})")
        else:
            print(f"   ✓ Tracked {trade_state['position'].upper()} entry already in locked backtest")

        # Update current_position
        locked["current_position"] = {
            "position": trade_state['position'],
            "entry_price": trade_state['entry_price'],
            "entry_date": str(entry_time)
        }

    # RECALCULATE STATS from all exits (not from fresh backtest)
    num_trades = len(locked["exits"])
    if num_trades > 0:
        wins = sum(1 for e in locked["exits"] if e.get('pnl', 0) > 0)
        win_pnl = sum(e.get('pnl', 0) for e in locked["exits"] if e.get('pnl', 0) > 0)
        loss_pnl = abs(sum(e.get('pnl', 0) for e in locked["exits"] if e.get('pnl', 0) < 0))

        # Calculate COMPOUNDED return (not simple sum)
        # Sort exits by date and compound returns
        sorted_exits = sorted(locked["exits"], key=lambda x: str(x.get('date', '')))
        equity = 1.0
        for exit_trade in sorted_exits:
            pnl_pct = exit_trade.get('pnl', 0)
            equity *= (1 + pnl_pct / 100)
        total_return = (equity - 1) * 100

        locked["num_trades"] = num_trades
        locked["win_rate"] = (wins / num_trades) * 100
        locked["total_return"] = total_return
        locked["profit_factor"] = win_pnl / loss_pnl if loss_pnl > 0 else 999.99 if win_pnl > 0 else 0  # Cap at 999.99 to avoid JSON serialization issues

    safe_json_write(path, locked, indent=2)

    print(f"   🔒 Locked backtest saved: {len(locked['entries'])} entries, {len(locked['exits'])} exits")
    print(f"   📊 Recalculated stats: {locked['num_trades']} trades, {locked['win_rate']:.1f}% win, {locked['total_return']:.1f}% return")
    return locked


def load_locked_backtest(strategy_name: str = None, ticker: str = None) -> dict:
    """Load locked backtest snapshot.

    Returns None if no locked backtest exists (will be created on startup).
    """
    path = get_locked_backtest_path(strategy_name=strategy_name, ticker=ticker)

    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"   ⚠️ Could not load locked backtest: {e}")
            return None
    return None


def validate_locked_backtest(locked_backtest: dict) -> list:
    """Validate locked backtest data integrity and return list of issues.

    Checks for:
    - Exits where entry_date == exit_date (invalid - can't trade same bar)
    - More exits than entries (data corruption)
    - Orphan exits (exit without matching entry timestamp)
    - Overlapping positions (two entries without an exit between them)

    Returns:
        List of issue descriptions (empty if no issues)
    """
    if not locked_backtest:
        return []

    issues = []

    # Check 1: Each exit should have valid entry_date (not same as exit date)
    for exit_t in locked_backtest.get('exits', []):
        entry_date = exit_t.get('entry_date')
        exit_date = exit_t.get('date')
        if entry_date and exit_date and str(entry_date)[:19] == str(exit_date)[:19]:
            issues.append(f"Invalid exit: entry_date == exit_date at {exit_date}")

    # Check 2: Entry/exit count balance
    # With open position: entries = exits + 1
    # Without open position: entries = exits
    entry_count = len(locked_backtest.get('entries', []))
    exit_count = len(locked_backtest.get('exits', []))
    has_position = locked_backtest.get('current_position') is not None

    if exit_count > entry_count:
        issues.append(f"More exits ({exit_count}) than entries ({entry_count})")
    elif has_position and entry_count > exit_count + 1:
        issues.append(f"Too many entries: {entry_count} entries, {exit_count} exits (with open position, should be {exit_count + 1})")
    elif not has_position and entry_count > exit_count:
        issues.append(f"Orphan entries: {entry_count} entries, {exit_count} exits (no position, should be equal)")

    # Check 3: Orphan exits (exit without matching entry timestamp)
    entry_timestamps = {str(e.get('date', ''))[:19] for e in locked_backtest.get('entries', [])}
    for exit_t in locked_backtest.get('exits', []):
        entry_ts = str(exit_t.get('entry_date', ''))[:19]
        if entry_ts and entry_ts not in entry_timestamps:
            # Try to find by price match instead (within 0.5%)
            entry_price = exit_t.get('entry_price', 0)
            price_match = any(
                abs(e.get('price', 0) - entry_price) / entry_price < 0.005 if entry_price > 0 else False
                for e in locked_backtest.get('entries', [])
            )
            if not price_match:
                issues.append(f"Orphan exit at {exit_t.get('date')} - no matching entry for {entry_ts}")

    # Check 4: Overlapping positions (CRITICAL for LONG-only strategy)
    # Create timeline of all events and check for consecutive entries
    events = []
    for e in locked_backtest.get('entries', []):
        events.append({'time': str(e.get('date', ''))[:19], 'type': 'entry'})
    for e in locked_backtest.get('exits', []):
        events.append({'time': str(e.get('date', ''))[:19], 'type': 'exit'})
    # Sort by time, with exits BEFORE entries when timestamps equal (reversal signals)
    # Reversal = exit old position and immediately enter new position at same bar
    events.sort(key=lambda x: (x['time'], 0 if x['type'] == 'exit' else 1))

    in_position = False
    for event in events:
        if event['type'] == 'entry':
            if in_position:
                issues.append(f"Overlapping entry at {event['time']} - already in position")
            in_position = True
        else:  # exit
            if not in_position:
                issues.append(f"Exit at {event['time']} without prior entry")
            in_position = False

    return issues


def cleanup_locked_backtest(strategy_name: str = None, ticker: str = None) -> int:
    """Clean up invalid entries from locked backtest.

    Removes:
    - Overlapping positions (entries while already in position)
    - Exits where entry_date >= exit_date (impossible - can't exit before or at entry)
    - Orphan entries (entries without a corresponding exit, except current position)
    - Exits for removed entries

    Returns:
        Number of items removed
    """
    locked = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)
    if not locked:
        return 0

    removed_count = 0

    # Step 0: Remove OVERLAPPING POSITIONS (CRITICAL)
    # Build a timeline of all events and keep only valid alternating entry/exit sequence
    entries = locked.get('entries', [])
    exits = locked.get('exits', [])

    # Create timeline events
    events = []
    for e in entries:
        events.append({
            'time': str(e.get('date', ''))[:19],
            'type': 'entry',
            'data': e
        })
    for e in exits:
        events.append({
            'time': str(e.get('date', ''))[:19],
            'type': 'exit',
            'data': e
        })
    # Sort by time, with exits BEFORE entries when timestamps equal (reversal signals)
    events.sort(key=lambda x: (x['time'], 0 if x['type'] == 'exit' else 1))

    # Process events and keep only valid sequence
    valid_entries = []
    valid_exits = []
    current_entry = None
    removed_entry_timestamps = set()

    for event in events:
        if event['type'] == 'entry':
            if current_entry is None:
                # No current position - valid entry
                current_entry = event['data']
                valid_entries.append(event['data'])
            else:
                # Already in position - OVERLAPPING ENTRY, skip it
                print(f"   🗑️ Removing overlapping entry: {event['time']} (already in position from {str(current_entry.get('date', ''))[:19]})")
                removed_entry_timestamps.add(event['time'])
                removed_count += 1
        else:  # exit
            exit_entry_date = str(event['data'].get('entry_date', ''))[:19]
            if current_entry is None:
                # No position to exit - orphan exit
                print(f"   🗑️ Removing orphan exit: {event['time']} (no position to exit)")
                removed_count += 1
            elif exit_entry_date in removed_entry_timestamps:
                # Exit for a removed overlapping entry
                print(f"   🗑️ Removing exit for removed entry: {event['time']} (entry {exit_entry_date} was removed)")
                removed_count += 1
            elif exit_entry_date == str(current_entry.get('date', ''))[:19]:
                # Matches our tracked entry - valid exit
                valid_exits.append(event['data'])
                current_entry = None
            else:
                # Exit for a different entry (shouldn't happen after overlap removal)
                print(f"   🗑️ Removing mismatched exit: {event['time']} (expected entry {str(current_entry.get('date', ''))[:19]}, got {exit_entry_date})")
                removed_count += 1

    locked['entries'] = valid_entries
    locked['exits'] = valid_exits

    # Update current_position
    if current_entry:
        locked['current_position'] = {
            'position': 'long',
            'entry_price': current_entry.get('price', 0),
            'entry_date': current_entry.get('date', '')
        }
    else:
        locked['current_position'] = None

    # Step 1: Clean invalid exits (entry_date >= exit_date)
    original_exits = locked.get('exits', [])
    cleaned_exits = []

    for exit_t in original_exits:
        entry_date = str(exit_t.get('entry_date', ''))[:19]
        exit_date = str(exit_t.get('date', ''))[:19]

        # Remove exits where entry_date >= exit_date
        if entry_date and exit_date and entry_date >= exit_date:
            print(f"   🗑️ Removing invalid exit: entry_date ({entry_date}) >= exit_date ({exit_date})")
            removed_count += 1
            continue

        cleaned_exits.append(exit_t)

    locked['exits'] = cleaned_exits

    # Step 2: Clean orphan entries (entries without corresponding exit)
    # Build set of entry timestamps that have been exited
    exited_entry_timestamps = set()
    for exit_t in cleaned_exits:
        entry_ts = str(exit_t.get('entry_date', ''))[:19]
        if entry_ts:
            exited_entry_timestamps.add(entry_ts)

    # Get current position entry timestamp (this one should NOT be cleaned)
    current_pos = locked.get('current_position')
    current_entry_ts = None
    if current_pos:
        current_entry_ts = str(current_pos.get('entry_date', ''))[:19]

    # Filter entries - keep only those that have exits OR are the current position
    original_entries = locked.get('entries', [])
    cleaned_entries = []

    for entry in original_entries:
        entry_ts = str(entry.get('date', ''))[:19]

        # Keep if this entry has a corresponding exit
        if entry_ts in exited_entry_timestamps:
            cleaned_entries.append(entry)
            continue

        # Keep if this is the current position
        if current_entry_ts and entry_ts == current_entry_ts:
            cleaned_entries.append(entry)
            continue

        # This is an orphan entry - remove it
        print(f"   🗑️ Removing orphan entry: {entry_ts} (no corresponding exit)")
        removed_count += 1

    locked['entries'] = cleaned_entries

    # Step 3: Recalculate stats and save if anything was removed
    if removed_count > 0:
        # Recalculate stats after cleanup
        tracked_exits = [e for e in cleaned_exits if not e.get('missed')]
        if tracked_exits:
            winners = [e for e in tracked_exits if e['pnl'] > 0]
            losers = [e for e in tracked_exits if e['pnl'] <= 0]
            locked["num_trades"] = len(tracked_exits)
            locked["win_rate"] = (len(winners) / len(tracked_exits)) * 100 if tracked_exits else 0
            # Recalculate compounded return
            sorted_exits = sorted(tracked_exits, key=lambda x: str(x.get('date', '')))
            equity = 1.0
            for exit_trade in sorted_exits:
                equity *= (1 + exit_trade['pnl'] / 100)
            locked["total_return"] = (equity - 1) * 100
            # Recalculate profit factor
            win_pnl = sum(e['pnl'] for e in winners) if winners else 0
            loss_pnl = abs(sum(e['pnl'] for e in losers)) if losers else 0.001
            locked["profit_factor"] = win_pnl / loss_pnl if loss_pnl > 0 else (999.99 if win_pnl > 0 else 0)

        # Write directly to file (don't use save_locked_backtest which merges with existing)
        path = get_locked_backtest_path(strategy_name=strategy_name, ticker=ticker)
        safe_json_write(path, locked, indent=2)
        print(f"   ✅ Cleaned locked backtest: removed {removed_count} invalid items")

    return removed_count


def append_to_locked_backtest(entry: dict = None, exit_trade: dict = None,
                               strategy_name: str = None, ticker: str = None,
                               missed: bool = False) -> bool:
    """Append a new entry or exit to the locked backtest.

    Called when a NEW signal is detected (at end of day). This adds
    the signal to the locked backtest so it appears on future charts.

    Args:
        entry: Entry signal dict with date, price, position
        exit_trade: Exit trade dict with date, price, pnl, reason, etc.
        strategy_name: Strategy name for file path
        ticker: Ticker symbol for file path
        missed: If True, marks the signal as missed (bot was down)

    Returns:
        True if successfully appended, False if rejected (duplicate, invalid, etc.)
        CRITICAL: Callers should check this return value before updating trade_state
        or sending Discord notifications to prevent desync.
    """
    locked = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)

    if locked is None:
        print("   ⚠️ No locked backtest to append to")
        return False

    if entry:
        # Normalize the new entry date to Chicago time for comparison
        try:
            entry_date_chicago = normalize_tz(pd.to_datetime(entry.get('date', '')))
            entry_date_str = entry_date_chicago.strftime('%Y-%m-%d %H:%M:%S') if entry_date_chicago else str(entry.get('date', ''))[:19]
        except Exception:
            entry_date_str = str(entry.get('date', ''))[:19]

        # Check for duplicate entries (same date) - normalize all to Chicago time
        existing_dates = set()
        for e in locked.get("entries", []):
            try:
                e_date = normalize_tz(pd.to_datetime(e.get('date', '')))
                existing_dates.add(e_date.strftime('%Y-%m-%d %H:%M:%S') if e_date else str(e.get('date', ''))[:19])
            except Exception:
                existing_dates.add(str(e.get('date', ''))[:19])

        if entry_date_str in existing_dates:
            print(f"   ⏭️  Skipping duplicate entry: {entry_date_str} (already exists)")
            return False  # Don't add duplicate

        # CRITICAL: Check if we're already in a position (last entry has no corresponding exit)
        # This prevents adding consecutive entries without an exit between them
        num_entries = len(locked.get("entries", []))
        num_exits = len(locked.get("exits", []))
        if num_entries > num_exits:
            last_entry = locked["entries"][-1] if locked.get("entries") else None
            print(f"   ⚠️  Rejecting duplicate entry: Already in position from {last_entry.get('date', 'unknown') if last_entry else 'unknown'}")
            print(f"      (entries: {num_entries}, exits: {num_exits} - must exit before new entry)")
            return False  # Don't add entry while already in position

        # Convert to Chicago time for consistency
        date_str = to_chicago_str(entry.get('date', ''))

        entry_record = {
            "date": date_str,
            "price": entry.get('price', 0),
            "position": entry.get('position', 'long')
        }
        if missed:
            entry_record["missed"] = True
        locked["entries"].append(entry_record)
        label = "🟠 missed" if missed else "🔒"
        print(f"   {label} Appended new entry to locked backtest: {entry.get('date')}")

    if exit_trade:
        exit_date_str = str(exit_trade.get('date', ''))[:19]  # Normalize to "YYYY-MM-DD HH:MM:SS"
        entry_date_check = str(exit_trade.get('entry_date', ''))[:19]

        # Validation: entry_date must be before exit_date
        # CRITICAL: Use proper datetime comparison with timezone normalization
        # String comparison fails when entry is UTC and exit is local time
        if entry_date_check and exit_date_str:
            try:
                # Parse and normalize both to UTC for proper comparison
                entry_dt = normalize_tz(pd.to_datetime(exit_trade.get('entry_date', '')))
                exit_dt = normalize_tz(pd.to_datetime(exit_trade.get('date', '')))
                if entry_dt >= exit_dt:
                    print(f"   ⚠️  Rejecting invalid exit: entry_date ({entry_dt}) >= exit_date ({exit_dt})")
                    return False  # Don't add invalid exit
            except Exception as e:
                # Fallback to string comparison if datetime parsing fails
                if entry_date_check >= exit_date_str:
                    print(f"   ⚠️  Rejecting invalid exit: entry_date ({entry_date_check}) >= exit_date ({exit_date_str}) [string compare, parse error: {e}]")
                    return False  # Don't add invalid exit

        # Check for duplicate exits (same date) - NORMALIZE to UTC for proper comparison
        # Raw string comparison fails when one has +00:00 and another has -0600
        existing_dates = set()
        for e in locked.get("exits", []):
            try:
                norm_dt = normalize_tz(pd.to_datetime(e.get('date', '')))
                existing_dates.add(norm_dt.strftime('%Y-%m-%d %H:%M:%S'))
            except Exception:
                existing_dates.add(str(e.get('date', ''))[:19])

        # Normalize the new exit date for comparison
        try:
            new_exit_norm = normalize_tz(pd.to_datetime(exit_trade.get('date', '')))
            new_exit_str = new_exit_norm.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            new_exit_str = exit_date_str

        if new_exit_str in existing_dates:
            print(f"   ⏭️  Skipping duplicate exit: {exit_date_str} (already exists as {new_exit_str})")
            return False  # Don't add duplicate

        # CRITICAL: Verify that the entry being closed actually EXISTS in locked_backtest
        # This prevents orphan exits from being created (exits without corresponding entries)
        # NOTE: Must normalize to Chicago time for comparison since entries are stored in Chicago time
        if entry_date_check:
            # Normalize the entry_date we're looking for to Chicago time (tz-naive for comparison)
            raw_entry_date = exit_trade.get('entry_date', '')
            try:
                entry_date_chicago = normalize_tz(pd.to_datetime(raw_entry_date))
                entry_date_chicago_str = entry_date_chicago.strftime('%Y-%m-%d %H:%M:%S') if entry_date_chicago else entry_date_check
            except Exception as e:
                print(f"   [DEBUG] Could not parse entry_date '{raw_entry_date}': {e}")
                entry_date_chicago_str = entry_date_check

            # Normalize all existing entries to Chicago time for comparison
            existing_entries_normalized = {}  # Map normalized date -> original date for debugging
            for e in locked.get("entries", []):
                try:
                    e_date = normalize_tz(pd.to_datetime(e.get('date', '')))
                    norm_str = e_date.strftime('%Y-%m-%d %H:%M:%S') if e_date else str(e.get('date', ''))[:19]
                    existing_entries_normalized[norm_str] = e.get('date', '')
                except Exception:
                    existing_entries_normalized[str(e.get('date', ''))[:19]] = e.get('date', '')

            if entry_date_chicago_str not in existing_entries_normalized:
                # Check if there's a match within 15-minute window (handles bar START vs bar CLOSE)
                found_nearby = None
                try:
                    entry_dt = pd.to_datetime(entry_date_chicago_str)
                    for norm_str in existing_entries_normalized:
                        existing_dt = pd.to_datetime(norm_str)
                        time_diff = abs((entry_dt - existing_dt).total_seconds())
                        if time_diff <= 900:  # 15 minutes = 900 seconds
                            found_nearby = (norm_str, existing_entries_normalized[norm_str], time_diff)
                            break
                except Exception:
                    pass

                if found_nearby:
                    print(f"   ⚠️  [WARN] Entry date mismatch by {found_nearby[2]:.0f}s: looking for {entry_date_chicago_str}, found {found_nearby[0]} (original: {found_nearby[1]})")
                    print(f"      Proceeding with exit (within 15-min tolerance for bar timing)")
                    # Allow the exit - this handles bar START vs bar CLOSE mismatch
                else:
                    # Show debug info for last 5 entries
                    last_5 = list(existing_entries_normalized.items())[-5:]
                    print(f"   ⚠️  Rejecting orphan exit: entry_date ({entry_date_chicago_str}) not found in locked_backtest entries")
                    print(f"      Raw entry_date from trade_state: '{raw_entry_date}'")
                    print(f"      Looking for normalized: '{entry_date_chicago_str}'")
                    print(f"      Available entries: {len(existing_entries_normalized)}")
                    print(f"      Last 5 entries (normalized -> original): {last_5}")
                    return False  # Don't add exit without corresponding entry

        # Convert to Chicago time for consistency
        date_str = to_chicago_str(exit_trade.get('date', ''))
        entry_date_str_tz = to_chicago_str(exit_trade.get('entry_date', ''))

        exit_record = {
            "date": date_str,
            "price": exit_trade.get('price', 0),
            "pnl": exit_trade.get('pnl', 0),
            "reason": exit_trade.get('reason', ''),
            "entry_price": exit_trade.get('entry_price', 0),
            "entry_date": entry_date_str_tz
        }
        if missed:
            exit_record["missed"] = True
        locked["exits"].append(exit_record)
        # Update stats (exclude missed trades from main stats)
        tracked_exits = [e for e in locked["exits"] if not e.get('missed')]
        missed_exits = [e for e in locked["exits"] if e.get('missed')]

        locked["num_trades"] = len(tracked_exits)
        locked["num_missed"] = len(missed_exits)
        if tracked_exits:
            winners = [e for e in tracked_exits if e['pnl'] > 0]
            losers = [e for e in tracked_exits if e['pnl'] <= 0]
            locked["win_rate"] = (len(winners) / len(tracked_exits)) * 100
            # Use COMPOUNDED return (consistent with other calculations)
            sorted_exits = sorted(tracked_exits, key=lambda x: str(x.get('date', '')))
            equity = 1.0
            for exit_trade in sorted_exits:
                equity *= (1 + exit_trade['pnl'] / 100)
            locked["total_return"] = (equity - 1) * 100
            # Calculate profit factor
            win_pnl = sum(e['pnl'] for e in winners) if winners else 0
            loss_pnl = abs(sum(e['pnl'] for e in losers)) if losers else 0.001
            locked["profit_factor"] = win_pnl / loss_pnl if loss_pnl > 0 else (999.99 if win_pnl > 0 else 0)  # Cap at 999.99 to avoid JSON serialization issues
        if missed_exits:
            missed_winners = [e for e in missed_exits if e['pnl'] > 0]
            locked["missed_win_rate"] = (len(missed_winners) / len(missed_exits)) * 100 if missed_exits else 0
            locked["missed_return"] = sum(e['pnl'] for e in missed_exits)

        label = "🟠 missed" if missed else "🔒"
        print(f"   {label} Appended new exit to locked backtest: {exit_trade.get('date')}")

    # Update current position status
    if exit_trade:
        locked["current_position"] = None
    elif entry:
        locked["current_position"] = {
            "position": entry.get('position', 'long'),
            "entry_price": entry.get('price', 0),
            "entry_date": str(entry.get('date', ''))
        }

    path = get_locked_backtest_path(strategy_name=strategy_name, ticker=ticker)
    safe_json_write(path, locked, indent=2)
    return True  # Successfully appended


def count_potential_missed_signals(locked_backtest: dict, fresh_backtest: dict) -> int:
    """Count signals that may have occurred while bot was down (READ-ONLY).

    This function ONLY counts potential missed signals for informational purposes.
    It does NOT modify any data files (anti-repainting policy).

    Args:
        locked_backtest: The locked backtest dict (source of truth)
        fresh_backtest: Dict with 'entries' and 'exits' from fresh backtest

    Returns: Approximate count of signals that may have been missed
    """
    missed = get_missed_signals(locked_backtest, fresh_backtest)
    return len(missed.get('entries', [])) + len(missed.get('exits', []))


def get_missed_signals(locked_backtest: dict, fresh_backtest: dict) -> dict:
    """Get signals that may have occurred while bot was down (READ-ONLY).

    Returns the actual missed signals for visualization purposes (ghost markers).
    Does NOT modify any data files (anti-repainting policy).

    Args:
        locked_backtest: The locked backtest dict (source of truth)
        fresh_backtest: Dict with 'entries' and 'exits' from fresh backtest

    Returns: Dict with 'entries' and 'exits' lists of missed signals
    """
    missed = {'entries': [], 'exits': []}

    if not locked_backtest or not fresh_backtest:
        return missed

    # Find the last signal date in locked backtest
    last_locked_date = None
    for entry in locked_backtest.get('entries', []):
        entry_date = normalize_tz(pd.to_datetime(entry['date'])) if entry.get('date') else None
        if entry_date and (last_locked_date is None or entry_date > last_locked_date):
            last_locked_date = entry_date
    for exit_t in locked_backtest.get('exits', []):
        exit_date = normalize_tz(pd.to_datetime(exit_t['date'])) if exit_t.get('date') else None
        if exit_date and (last_locked_date is None or exit_date > last_locked_date):
            last_locked_date = exit_date

    if last_locked_date is None:
        return missed

    # Get existing timestamps from locked backtest
    existing_timestamps = set()
    for entry in locked_backtest.get('entries', []):
        if entry.get('date'):
            existing_timestamps.add(str(entry['date'])[:19])
    for exit_t in locked_backtest.get('exits', []):
        if exit_t.get('date'):
            existing_timestamps.add(str(exit_t['date'])[:19])

    # Get signals in fresh backtest that are AFTER last_locked_date and NOT in existing
    for entry in fresh_backtest.get('entries', []):
        entry_date = normalize_tz(pd.to_datetime(entry['date'])) if entry.get('date') else None
        if entry_date and entry_date > last_locked_date:
            timestamp_str = str(entry['date'])[:19]
            if timestamp_str not in existing_timestamps:
                missed['entries'].append(entry)

    for exit_t in fresh_backtest.get('exits', []):
        exit_date = normalize_tz(pd.to_datetime(exit_t['date'])) if exit_t.get('date') else None
        if exit_date and exit_date > last_locked_date:
            timestamp_str = str(exit_t['date'])[:19]
            if timestamp_str not in existing_timestamps:
                missed['exits'].append(exit_t)

    return missed


def detect_and_add_missed_signals(fresh_backtest: dict, strategy_name: str = None, ticker: str = None, trade_state: dict = None, mark_as_missed: bool = False) -> int:
    """Detect signals that occurred while bot was down and add them to locked_backtest.

    Detect signals that occurred while bot was down and add them as real signals.

    Compares fresh backtest entries/exits with locked_backtest to find signals
    that occurred after the last tracked signal. Adds them with missed=True.

    IMPORTANT: If user has an active tracked position (trade_state), we skip
    adding any missed entries that would overlap/conflict with that position.
    The user's tracked position takes precedence over backtest detection.

    Args:
        fresh_backtest: Dict with 'entries' and 'exits' from fresh backtest
        strategy_name: Strategy name for loading locked backtest
        ticker: Ticker symbol
        trade_state: Current trade state dict (if user has tracked position)

    Returns: Number of missed signals detected and added.
    """
    locked = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)
    if locked is None or fresh_backtest is None:
        return 0

    # Check if user has an active tracked position - if so, skip adding missed entries
    # that would conflict with their position (user's position takes precedence)
    skip_entries_after = None
    if trade_state and trade_state.get('position'):
        tracked_entry_time = trade_state.get('entry_time') or trade_state.get('entry_signal_bar')
        if tracked_entry_time:
            skip_entries_after = normalize_tz(pd.to_datetime(tracked_entry_time))
            print(f"   📍 User has tracked position from {tracked_entry_time} - skipping conflicting missed entries")

    # Find the last tracked entry date in locked backtest
    locked_entries = locked.get('entries', [])
    locked_exits = locked.get('exits', [])

    last_locked_date = None
    for entry in locked_entries:
        entry_date = normalize_tz(pd.to_datetime(entry['date'])) if entry.get('date') else None
        if entry_date and (last_locked_date is None or entry_date > last_locked_date):
            last_locked_date = entry_date
    for exit_t in locked_exits:
        exit_date = normalize_tz(pd.to_datetime(exit_t['date'])) if exit_t.get('date') else None
        if exit_date and (last_locked_date is None or exit_date > last_locked_date):
            last_locked_date = exit_date

    if last_locked_date is None:
        return 0

    # Get set of existing entry/exit timestamps to avoid duplicates
    # CRITICAL: Normalize to UTC for comparison - raw strings fail when timezones differ
    existing_entry_timestamps = set()
    for entry in locked_entries:
        if entry.get('date'):
            try:
                norm_dt = normalize_tz(pd.to_datetime(entry['date']))
                existing_entry_timestamps.add(norm_dt.strftime('%Y-%m-%d %H:%M:%S'))
            except Exception:
                existing_entry_timestamps.add(str(entry['date'])[:19])

    existing_exit_timestamps = set()
    for exit_t in locked_exits:
        if exit_t.get('date'):
            try:
                norm_dt = normalize_tz(pd.to_datetime(exit_t['date']))
                existing_exit_timestamps.add(norm_dt.strftime('%Y-%m-%d %H:%M:%S'))
            except Exception:
                existing_exit_timestamps.add(str(exit_t['date'])[:19])

    missed_count = 0
    skipped_count = 0

    # Check fresh backtest entries for missed signals
    for entry in fresh_backtest.get('entries', []):
        entry_date = normalize_tz(pd.to_datetime(entry['date'])) if entry.get('date') else None
        if entry_date and entry_date > last_locked_date:
            # Skip entries that conflict with user's tracked position
            if skip_entries_after and entry_date >= skip_entries_after:
                skipped_count += 1
                continue

            # Normalize to UTC for comparison (matches existing_entry_timestamps normalization)
            timestamp_str = entry_date.strftime('%Y-%m-%d %H:%M:%S')  # entry_date already normalized above
            if timestamp_str not in existing_entry_timestamps:
                append_to_locked_backtest(
                    entry={'date': entry['date'], 'price': entry['price'], 'position': entry.get('position', 'long')},
                    strategy_name=strategy_name, ticker=ticker, missed=mark_as_missed
                )
                existing_entry_timestamps.add(timestamp_str)
                missed_count += 1

    # Check fresh backtest exits for missed signals
    for exit_t in fresh_backtest.get('exits', []):
        exit_date = normalize_tz(pd.to_datetime(exit_t['date'])) if exit_t.get('date') else None
        if exit_date and exit_date > last_locked_date:
            # Skip exits whose entry would be after tracked position entry
            # (these exits belong to positions we didn't add)
            exit_entry_date = exit_t.get('entry_date')
            if skip_entries_after and exit_entry_date:
                exit_entry_dt = normalize_tz(pd.to_datetime(exit_entry_date))
                if exit_entry_dt >= skip_entries_after:
                    skipped_count += 1
                    continue

            # Normalize to UTC for comparison (matches existing_exit_timestamps normalization)
            timestamp_str = exit_date.strftime('%Y-%m-%d %H:%M:%S')  # exit_date already normalized above
            if timestamp_str not in existing_exit_timestamps:
                # Use original reason from backtest, not "Missed"
                reason = exit_t.get('reason', 'Opposite Signal')
                append_to_locked_backtest(
                    exit_trade={
                        'date': exit_t['date'],
                        'price': exit_t['price'],
                        'pnl': exit_t.get('pnl', 0),
                        'reason': reason,
                        'entry_price': exit_t.get('entry_price', 0),
                        'entry_date': exit_t.get('entry_date', '')
                    },
                    strategy_name=strategy_name, ticker=ticker, missed=mark_as_missed
                )
                existing_exit_timestamps.add(timestamp_str)
                missed_count += 1

    if skipped_count > 0:
        print(f"   ⏭️  Skipped {skipped_count} signals that would conflict with tracked position")

    if missed_count > 0:
        print(f"   🟠 Detected and added {missed_count} missed signals from downtime")

    return missed_count


def load_trade_state(strategy_name: str = None, ticker: str = None, state_path: str = None) -> dict:
    """Load current trade state from file.

    Args:
        strategy_name: Strategy name (preferred - allows multiple strategies per ticker)
        ticker: Ticker symbol (fallback for backwards compatibility)
        state_path: Override path (legacy support)
    """
    if state_path is None:
        state_path = get_state_file_path(strategy_name=strategy_name, ticker=ticker)

    default_state = {
        "position": None,  # None or "long" (LONG-only strategy)
        "entry_price": None,
        "entry_time": None,
        "entry_signal_bar": None,  # Bar datetime that generated entry signal (for locked charts)
        "last_signal_time": None,
    }
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            loaded = json.load(f)
            # Merge with defaults to ensure all keys exist
            return {**default_state, **loaded}
    return default_state


def save_trade_state(state: dict, strategy_name: str = None, ticker: str = None, state_path: str = None):
    """Save trade state to file.

    Args:
        state: Trade state dict
        strategy_name: Strategy name (preferred - allows multiple strategies per ticker)
        ticker: Ticker symbol (fallback for backwards compatibility)
        state_path: Override path (legacy support)
    """
    if state_path is None:
        state_path = get_state_file_path(strategy_name=strategy_name, ticker=ticker)

    safe_json_write(state_path, state, indent=2)


def load_trade_history(strategy_name: str = None, ticker: str = None, history_path: str = None) -> list:
    """Load trade history from file.

    Args:
        strategy_name: Strategy name (preferred - allows multiple strategies per ticker)
        ticker: Ticker symbol (fallback for backwards compatibility)
        history_path: Override path (legacy support)
    """
    if history_path is None:
        history_path = get_history_file_path(strategy_name=strategy_name, ticker=ticker)

    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except Exception:  # Catch all non-system exceptions
            return []
    return []


def save_trade_history(history: list, strategy_name: str = None, ticker: str = None, history_path: str = None):
    """Save trade history to file.

    Args:
        history: Trade history list
        strategy_name: Strategy name (preferred - allows multiple strategies per ticker)
        ticker: Ticker symbol (fallback for backwards compatibility)
        history_path: Override path (legacy support)
    """
    if history_path is None:
        history_path = get_history_file_path(strategy_name=strategy_name, ticker=ticker)

    safe_json_write(history_path, history, indent=2)


def log_closed_trade(ticker: str, position_type: str, entry_price: float, exit_price: float,
                     entry_time: str, exit_time: str, exit_reason: str, pnl_pct: float,
                     strategy_name: str = None, entry_signal_bar: str = None, exit_signal_bar: str = None):
    """Log a closed trade to history and return cumulative stats.

    Args:
        entry_signal_bar: The datetime of the bar that generated the entry signal (for chart plotting)
        exit_signal_bar: The datetime of the bar that generated the exit signal (for chart plotting)
    """
    history = load_trade_history(strategy_name=strategy_name, ticker=ticker)

    # Calculate actual dollar P&L per unit
    if position_type.upper() == 'LONG':
        pnl_dollars = exit_price - entry_price
    else:  # SHORT
        pnl_dollars = entry_price - exit_price

    trade = {
        "id": len(history) + 1,
        "ticker": ticker,
        "strategy_name": strategy_name,
        "type": position_type,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_signal_bar": entry_signal_bar or entry_time,  # Fallback to entry_time if not provided
        "exit_signal_bar": exit_signal_bar or exit_time,  # Fallback to exit_time if not provided
        "exit_reason": exit_reason,
        "pnl_pct": pnl_pct,
        "pnl_dollars": pnl_dollars  # Actual dollar P&L per unit
    }

    history.append(trade)
    save_trade_history(history, strategy_name=strategy_name, ticker=ticker)

    # Calculate cumulative stats - EXCLUDE [SYNC] and [MISSED] trades
    # These are missed trades that shouldn't count toward live trading performance
    live_trades = [t for t in history if '[SYNC]' not in str(t.get('exit_reason', ''))
                   and '[MISSED]' not in str(t.get('exit_reason', ''))]
    total_trades = len(live_trades)
    winners = [t for t in live_trades if t['pnl_pct'] > 0]
    losers = [t for t in live_trades if t['pnl_pct'] < 0]
    win_rate = (len(winners) / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum(t['pnl_pct'] for t in live_trades)
    total_pnl_dollars = sum(t.get('pnl_dollars', 0) for t in live_trades)
    avg_win = sum(t['pnl_pct'] for t in winners) / len(winners) if winners else 0
    avg_loss = sum(t['pnl_pct'] for t in losers) / len(losers) if losers else 0

    return {
        "total_trades": total_trades,
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": win_rate,
        "total_pnl_pct": total_pnl,
        "total_pnl_dollars": total_pnl_dollars,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def send_status_update(webhook_url: str, df: pd.DataFrame, backtest: dict, config: dict,
                       ticker: str, title: str = "📊 Status Update", trade_state: dict = None,
                       strategy_name: str = None, locked_backtest: dict = None,
                       realtime_price: float = None):
    """Send a scheduled status update with TWO charts to Discord.

    Sends two posts:
    1. Full timeframe chart with stats over entire period
    2. Last 180 days chart with stats recalculated for that window

    Uses locked_backtest for chart markers to prevent repainting.
    Uses realtime_price (if provided) for current price display instead of daily bar close.
    """
    try:
        # Load locked trade history to prevent repainting on charts
        trade_history = load_trade_history(strategy_name=strategy_name, ticker=ticker)
        # Load locked backtest if not provided
        if locked_backtest is None:
            locked_backtest = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)
        # Helper function to build position section
        def build_position_section(current_price, trade_state_local, backtest_local, config_local):
            pos_section = "📭 **Position:** No active position"

            if trade_state_local and trade_state_local.get('position'):
                pos_type = trade_state_local.get('position', '').upper()
                entry_price = trade_state_local.get('entry_price', 0)
                entry_time_str = trade_state_local.get('entry_time', '')

                # Calculate P&L
                if pos_type == 'LONG':
                    pnl = ((current_price - entry_price) / entry_price) * 100 if entry_price else 0
                else:
                    pnl = ((entry_price - current_price) / entry_price) * 100 if entry_price else 0

                pnl_emoji = "🟢" if pnl >= 0 else "🔴"

                # Calculate hold duration
                hold_duration = "N/A"
                if entry_time_str:
                    try:
                        entry_dt = datetime.strptime(entry_time_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
                        duration = datetime.now() - entry_dt
                        days = duration.days
                        hours = duration.seconds // 3600
                        if days > 0:
                            hold_duration = f"{days}d {hours}h"
                        else:
                            hold_duration = f"{hours}h {(duration.seconds % 3600) // 60}m"
                    except Exception:  # Catch all non-system exceptions
                        hold_duration = "N/A"

                # Calculate actual dollar P&L per unit (e.g., per 1 BTC or 1 share)
                # Validate entry_price to prevent invalid calculations
                if entry_price and entry_price > 0:
                    if pos_type == 'LONG':
                        pnl_dollars = current_price - entry_price
                    else:  # SHORT
                        pnl_dollars = entry_price - current_price

                    # Stop loss and take profit levels
                    stop_loss_pct = config_local.get('stop_loss_pct', 5)
                    take_profit_pct = config_local.get('take_profit_pct', 10)

                    if pos_type == 'LONG':
                        sl_price = entry_price * (1 - stop_loss_pct/100)
                        tp_price = entry_price * (1 + take_profit_pct/100)
                    else:
                        sl_price = entry_price * (1 + stop_loss_pct/100)
                        tp_price = entry_price * (1 - take_profit_pct/100)
                else:
                    # Invalid entry_price - use safe defaults
                    pnl_dollars = 0
                    sl_price = 0
                    tp_price = 0

                pos_section = (
                    f"{pnl_emoji} **Position:** {pos_type}\n"
                    f"• Entry: ${entry_price:.2f} on {entry_time_str[:10] if entry_time_str else 'N/A'}\n"
                    f"• Current: ${current_price:.2f} | **P&L: {pnl:+.2f}%** (${pnl_dollars:+,.0f})\n"
                    f"• Duration: {hold_duration}\n"
                    f"• SL: ${sl_price:.2f} | TP: ${tp_price:.2f}"
                )
            # NOTE: Removed fallback to backtest position - only show TRACKED positions
            # This prevents confusion when backtest shows a position but we're not actually tracking it

            return pos_section

        # Create strategy label for clear identification
        opt_period = config.get('optimization_period', '')
        strategy_label = f"{ticker} {opt_period.upper()}" if opt_period else ticker
        # Use real-time price for display if available, otherwise fall back to daily bar close
        current_price = realtime_price if realtime_price else df['close'].iloc[-1]

        # Calculate date range for full data
        full_start = df.index[0].strftime('%Y-%m-%d') if hasattr(df.index[0], 'strftime') else str(df.index[0])[:10]
        full_end = df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])[:10]
        full_days = len(df)

        # --- CHART 1: Full Timeframe ---
        # Use locked_backtest for stats to prevent repainting (fresh backtest may have different trades)
        stats_source_full = locked_backtest if locked_backtest and locked_backtest.get('num_trades') else backtest
        chart_buf_full = generate_velocity_chart(df, stats_source_full, config, ticker,
                                                  title_suffix=f" (Full: {full_days} bars)",
                                                  trade_history=trade_history,
                                                  current_position=trade_state,
                                                  locked_backtest=locked_backtest,
                                                  full_period_backtest=stats_source_full)
        pos_section = build_position_section(current_price, trade_state, stats_source_full, config)

        # Check for stale data warning (accounts for weekend/futures market closure)
        data_stale_warning = ""
        try:
            data_end_date = df.index[-1].date() if hasattr(df.index[-1], 'date') else datetime.strptime(full_end, '%Y-%m-%d').date()
            is_stale, warning = is_data_stale_for_futures(data_end_date, ticker)
            if is_stale and warning:
                data_stale_warning = warning
        except Exception:  # Catch all non-system exceptions
            pass

        msg_full = (
            f"**{title}: {strategy_label}** [1/2 Full Timeframe]\n"
            f"**Period:** {full_start} to {full_end} ({full_days} bars)\n"
            f"---\n"
            f"{pos_section}\n"
            f"---\n"
            f"📊 **Full Period Stats:**\n"
            f"• Trades: {stats_source_full['num_trades']} | Win Rate: {stats_source_full['win_rate']:.0f}%\n"
            f"• Return: {stats_source_full['total_return']:.1f}% | PF: {stats_source_full['profit_factor']:.1f}\n"
            f"---\n"
            f"_Updated: {datetime.now(CHICAGO_TZ).strftime('%Y-%m-%d %H:%M CT')}_{data_stale_warning}"
        )

        send_discord_alert(webhook_url, msg_full, chart_buf_full, strategy_name=strategy_name)
        print(f"✅ Sent full timeframe update: {title}")

        # --- CHART 2: Recent Period (126 bars) ---
        # Use SUBSET of bundled data to prevent repainting (NOT fresh data)
        min_bars_for_subset = 20  # Need at least 20 bars to make a meaningful subset
        subset_bars = 126  # Number of bars to show

        # Calculate actual days based on interval (bars per day varies by timeframe)
        interval = config.get('interval', '1d')
        bars_per_day_map = {
            '1m': 390, '5m': 78, '15m': 26, '30m': 13, '1h': 7, '90m': 5, '4h': 2, '1d': 1
        }
        # Futures trade ~23 hours, Crypto trades 24/7
        is_futures = is_futures_ticker(ticker)
        is_crypto = '-USD' in ticker.upper()
        bars_per_day = bars_per_day_map.get(interval, 1)
        if is_crypto and interval not in ['1d']:
            bars_per_day = int(bars_per_day * 3.7)  # Crypto trades 24/7
        elif is_futures and interval not in ['1d']:
            bars_per_day = int(bars_per_day * 3.5)  # Futures trade longer hours

        if len(df) > min_bars_for_subset:
            if len(df) <= subset_bars:
                subset_bars = len(df) // 2

            # Use SUBSET of bundled df (NOT run_backtest_for_period which causes repainting)
            df_subset = df.iloc[-subset_bars:].copy()

            # Calculate actual days from REAL data range (not bar count estimate)
            # This handles weekends, holidays, and data gaps correctly
            if len(df_subset) > 1:
                actual_days = max(1, (df_subset.index.max() - df_subset.index.min()).days)
            else:
                actual_days = 1
            day_word = "Day" if actual_days == 1 else "Days"
            period_label = f"Last {actual_days} {day_word}"

            # Filter locked_backtest to get stats and exits for just the subset period
            # EXCLUDE missed/sync trades from stats (consistent with trade_history filtering)
            subset_start = normalize_tz(df_subset.index[0])
            # For daily charts: use date-only comparison for consistency
            is_daily = config.get('interval', '1d') == '1d'
            if is_daily:
                subset_start_date = subset_start.date() if hasattr(subset_start, 'date') else subset_start
            subset_exits = []
            if locked_backtest and locked_backtest.get('exits'):
                for exit_trade in locked_backtest['exits']:
                    # Skip missed trades (missed=True flag or [SYNC]/[MISSED] in reason)
                    if exit_trade.get('missed'):
                        continue
                    reason = str(exit_trade.get('reason', ''))
                    if '[SYNC]' in reason or '[MISSED]' in reason:
                        continue
                    exit_date = normalize_tz(pd.to_datetime(exit_trade['date']))
                    # For daily charts: use date-only comparison
                    if is_daily:
                        exit_date_only = exit_date.date() if hasattr(exit_date, 'date') else exit_date
                        if exit_date_only >= subset_start_date:
                            subset_exits.append(exit_trade)
                    else:
                        if exit_date >= subset_start:
                            subset_exits.append(exit_trade)

            # Build backtest_subset with exits for equity curve
            # Use centralized stats calculation (R6.1: single source of truth)
            stats = calculate_exit_stats(subset_exits, exclude_missed=False)  # Already filtered above
            backtest_subset = {
                'num_trades': stats['num_trades'],
                'win_rate': stats['win_rate'],
                'total_return': stats['total_return'],
                'profit_factor': stats['profit_factor'],
                'exits': subset_exits,  # Include exits for equity curve
                'current_position': locked_backtest.get('current_position') if locked_backtest else None,
                'period_days': actual_days  # Store the actual days for label
            }

            # Calculate date range for subset window
            period_start = df_subset.index[0].strftime('%Y-%m-%d') if hasattr(df_subset.index[0], 'strftime') else str(df_subset.index[0])[:10]
            period_end = df_subset.index[-1].strftime('%Y-%m-%d') if hasattr(df_subset.index[-1], 'strftime') else str(df_subset.index[-1])[:10]

            chart_buf_subset = generate_velocity_chart(df_subset, backtest_subset, config, ticker,
                                                        title_suffix=f" ({period_label})",
                                                        trade_history=trade_history,
                                                        current_position=trade_state,
                                                        locked_backtest=locked_backtest,
                                                        full_period_backtest=stats_source_full)

            # Position section stays the same (current position)
            pos_section_subset = build_position_section(current_price, trade_state, backtest_subset, config)

            # Check for stale data warning (accounts for weekend/futures market closure)
            data_stale_warning = ""
            try:
                data_end_date = df_subset.index[-1].date() if hasattr(df_subset.index[-1], 'date') else datetime.strptime(period_end, '%Y-%m-%d').date()
                is_stale, warning = is_data_stale_for_futures(data_end_date, ticker)
                if is_stale and warning:
                    data_stale_warning = warning
            except Exception:  # Catch all non-system exceptions
                pass

            msg_subset = (
                f"**{title}: {strategy_label}** [2/2 {period_label}]\n"
                f"**Period:** {period_start} to {period_end} ({len(df_subset)} bars)\n"
                f"---\n"
                f"{pos_section_subset}\n"
                f"---\n"
                f"📊 **{period_label} Stats (Tracked Only):**\n"
                f"• Trades: {backtest_subset['num_trades']} | Win Rate: {backtest_subset['win_rate']:.0f}%\n"
                f"• Return: {backtest_subset['total_return']:.1f}% | PF: {backtest_subset['profit_factor']:.1f}\n"
                f"---\n"
                f"_Updated: {datetime.now(CHICAGO_TZ).strftime('%Y-%m-%d %H:%M CT')}_{data_stale_warning}"
            )

            send_discord_alert(webhook_url, msg_subset, chart_buf_subset, strategy_name=strategy_name)
            print(f"✅ Sent {period_label} update: {title}")
        else:
            print(f"ℹ️ Skipping subset chart (only {len(df)} bars available, need >{min_bars_for_subset})")

    except Exception as e:
        print(f"⚠️ Failed to send status update: {e}")
        import traceback
        traceback.print_exc()


def run_live_trader(config_path: str = "production_env/velocity_config.json", skip_selection: bool = False):
    """Main live trading loop."""

    print(f"\n{'='*60}")
    print("VELOCITY LIVE TRADER STARTING")
    print(f"{'='*60}")

    # Interactive strategy selection (unless skipped)
    if not skip_selection:
        selected_path = select_strategy_interactive(config_path)
        if selected_path is None:
            print("No strategy selected. Exiting.")
            return
        config_path = selected_path

    # Load config
    config = load_config(config_path)

    # Track config file for hot-reloading
    last_config_mtime = os.path.getmtime(config_path) if os.path.exists(config_path) else 0

    ticker = config.get('ticker', 'BTC-USD')
    interval = config.get('interval', '1d')
    api_key = config.get('polygon_api_key')
    webhook_url = config.get('discord_webhook') or DEFAULT_DISCORD_WEBHOOK
    strategy_name = config.get('strategy_name', 'velocity_strategy')
    optimization_period = config.get('optimization_period', '')  # e.g., "1y", "2y", "5y"

    # Create a short label for Discord messages (e.g., "BTC-USD 2Y" or "SPY 5Y")
    strategy_label = f"{ticker} {optimization_period.upper()}" if optimization_period else ticker

    # ============================================================
    # PROMINENT STARTUP BANNER - Shows which strategy this terminal is running
    # ============================================================
    print(f"\n")
    print(f"╔════════════════════════════════════════════════════════════╗")
    print(f"║                                                            ║")
    print(f"║   🚀 RUNNING: {strategy_label:^42} 🚀   ║")
    print(f"║                                                            ║")
    print(f"╠════════════════════════════════════════════════════════════╣")
    print(f"║   Strategy: {strategy_name[:44]:44}   ║")
    print(f"║   Config:   {config_path[:44]:44}   ║")
    print(f"╚════════════════════════════════════════════════════════════╝")
    print(f"\n")

    # Risk parameters
    stop_loss_pct = config.get('stop_loss_pct', 5.0)
    take_profit_pct = config.get('take_profit_pct', 10.0)
    exit_on_opposite_signal = config.get('exit_on_opposite_signal', True)
    exit_on_midline_cross = config.get('exit_on_midline_cross', False)

    print(f"Strategy: {strategy_name}")
    print(f"Ticker: {ticker}")
    print(f"Interval: {interval}")
    print(f"Signal Type: {config.get('signal_type')}")
    print(f"Stop Loss: {stop_loss_pct}%")
    print(f"Take Profit: {take_profit_pct}%")
    print(f"Discord Webhook: {'Configured' if webhook_url else 'Not configured'}")
    print(f"{'='*60}\n")

    # Use data period from config if available, otherwise default to 365 days
    backtest_days = config.get('data_period_days', 365)
    print(f"📅 Data period: {backtest_days} days (from config)")

    print(f"📊 Running historical backtest...")
    try:
        # Check for bundled data from Streamlit (ensures EXACT same data)
        bundle_name = config.get('bundle_name')
        bundled_data_path = None
        if bundle_name:
            bundled_data_path = os.path.join(VELOCITY_STRATEGIES_DIR, bundle_name, "data.parquet")

        df = None
        if bundled_data_path and os.path.exists(bundled_data_path):
            # Load exact data from Streamlit bundle
            print(f"   📦 Loading bundled data from: {bundled_data_path}")
            df = pd.read_parquet(bundled_data_path)
            # Ensure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if len(df) > 0:
                print(f"   ✓ Loaded {len(df)} bars: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
                print(f"   ✓ Using EXACT same data as Streamlit!")
            else:
                print(f"   ⚠️ Warning: Bundled data is empty")
        else:
            # Fetch fresh data via yfinance (fallback)
            print(f"   Fetching fresh data via yfinance...")
            df = fetch_price_data(ticker, api_key, days=backtest_days, interval=interval)

        df = calculate_composite_oscillator(df, config)
        df = calculate_velocity_signals(df, config)  # Required for chart JD_Signal panel
        backtest = run_historical_backtest(df, config)

        print(f"\n{'='*60}")
        print("HISTORICAL BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Total Completed Trades: {backtest['num_trades']}")
        print(f"Win Rate: {backtest['win_rate']:.1f}%")
        print(f"Total Return: {backtest['total_return']:.1f}%")
        print(f"Profit Factor: {backtest['profit_factor']:.2f}")
        print(f"Avg Trade P&L: {backtest['avg_pnl']:.2f}%")

        # Show recent trades
        if backtest['exits']:
            print(f"\n📜 Last 5 Trades:")
            for t in backtest['exits'][-5:]:
                result = "✅" if t['pnl'] > 0 else "❌"
                print(f"   {result} {t['entry_date']} → {t['date']}: {t['pnl']:+.2f}% ({t['reason']})")

        # Current position from bundled data (may be stale - fresh data check below)
        if backtest['current_position']:
            pos = backtest['current_position']
            print(f"\n📊 Bundled data shows open position (will verify with fresh data below)")
            print(f"   Entry: ${pos['entry_price']:.2f} on {pos['entry_date']}")
        else:
            print(f"\n📊 Bundled data shows no open position")

        print()

    except Exception as e:
        print(f"⚠️ Could not run historical backtest: {e}")
        backtest = None

    # Load or initialize trade state from backtest (ticker-specific)
    trade_state = load_trade_state(strategy_name=strategy_name, ticker=ticker)
    print(f"📁 State file: {get_state_file_path(strategy_name=strategy_name, ticker=ticker)}")
    print(f"📁 History file: {get_history_file_path(strategy_name=strategy_name, ticker=ticker)}")

    # NOTE: DO NOT clear last_signal_time when no position!
    # last_signal_time tracks which signals have been PROCESSED (added to locked_backtest),
    # NOT which position is open. Clearing it causes re-processing of old signals.

    # FETCH FRESH DATA FOR SYNC - bundled data may be stale!
    print(f"\n🔄 Fetching FRESH data for state sync check...")
    fresh_backtest = None
    fresh_df = pd.DataFrame()  # Initialize empty to prevent NameError
    fresh_df_for_charts = pd.DataFrame()  # Full data including incomplete bar (for charts)
    excluded_incomplete_bar = False  # Track if we excluded a bar
    try:
        import time
        time.sleep(1)  # Small delay to avoid rate limiting
        fresh_df = fetch_price_data(ticker, api_key, days=200, interval=interval)
        if not fresh_df.empty:
            # Keep a copy WITH incomplete bar for charts (shows current data)
            fresh_df_for_charts = fresh_df.copy()

            # CRITICAL: Exclude incomplete current bar to prevent false signals on startup
            # For daily intervals, if the market hasn't closed, today's bar is still forming
            # and shouldn't be used for position sync (matches behavior of normal signal detection)
            if is_last_bar_incomplete(fresh_df, interval, ticker):
                excluded_bar_date = fresh_df.index[-1].strftime('%Y-%m-%d')
                fresh_df = fresh_df.iloc[:-1]
                excluded_incomplete_bar = True
                print(f"   ⏳ Excluded incomplete bar ({excluded_bar_date}) - market still open")

            if fresh_df.empty:
                print(f"   ⚠️ No completed bars available after excluding incomplete bar")
                fresh_backtest = None
            else:
                fresh_df = calculate_composite_oscillator(fresh_df, config)
                fresh_backtest = run_historical_backtest(fresh_df, config)

            # Show fresh data position vs bundled data position
            fresh_pos = fresh_backtest.get('current_position') if fresh_backtest else None
            bundled_pos = backtest.get('current_position') if backtest else None

            print(f"   Fresh data range: {fresh_df.index[0].strftime('%Y-%m-%d')} to {fresh_df.index[-1].strftime('%Y-%m-%d')}" if not fresh_df.empty else "   Fresh data: empty after exclusion")

            # Show ACTUAL current position status (from fresh data)
            if fresh_pos:
                print(f"\n{'='*60}")
                print(f"🔵 CURRENT POSITION (verified with fresh data)")
                print(f"{'='*60}")
                print(f"   LONG @ ${fresh_pos['entry_price']:.2f} on {fresh_pos['entry_date']}")
                print(f"{'='*60}")
            else:
                print(f"\n📭 No open position (verified with fresh data)")

            # Note if bundled data was stale
            if bundled_pos and not fresh_pos:
                print(f"   ℹ️  Bundled data showed position, but it has since closed.")
            elif bundled_pos and fresh_pos and abs(bundled_pos['entry_price'] - fresh_pos['entry_price']) > 1:
                print(f"   ℹ️  Bundled data was stale - fresh data has different position.")
    except Exception as e:
        print(f"   ⚠️ Could not fetch fresh data: {e}")
        fresh_backtest = backtest  # Fall back to bundled

    # Use fresh backtest for sync if available, otherwise fall back to bundled
    sync_backtest = fresh_backtest if fresh_backtest else backtest

    # Get current price for Discord messages
    sync_current_price = fresh_df.iloc[-1]['close'] if (fresh_backtest and not fresh_df.empty) else (df.iloc[-1]['close'] if not df.empty else 0)

    # CRITICAL: Validate sync_current_price to prevent calculations with 0
    if sync_current_price <= 0:
        print(f"⚠️  CRITICAL: Invalid sync_current_price ({sync_current_price}) - no valid price data available")
        print(f"   This could mean data fetch failed or market is closed. Continuing with caution...")
        # Use a fallback from trade_state entry_price if available
        if trade_state and trade_state.get('entry_price') and trade_state['entry_price'] > 0:
            sync_current_price = trade_state['entry_price']
            print(f"   Using entry_price as fallback: ${sync_current_price:.2f}")

    # SYNC STATE WITH BACKTEST - handles missed entries/exits
    backtest_position = sync_backtest.get('current_position') if sync_backtest else None
    state_position = trade_state.get('position')

    # Flag to track if user chose to keep their position despite mismatch
    # This prevents the main loop auto-sync from overriding the user's explicit choice
    user_chose_keep_position = False

    # Case 1: No state but backtest shows position - missed entry
    if state_position is None and backtest_position:
        pos = backtest_position
        trade_state = {
            'position': pos['position'],
            'entry_price': pos['entry_price'],
            'entry_time': pos['entry_date'],
            'entry_signal_bar': pos['entry_date'],  # Signal bar is the entry date from backtest
            'last_signal_time': pos['entry_date'],
        }
        save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)

        # Also sync entry to locked_backtest so charts/stats are consistent
        # Use missed=False - this IS a valid signal from fresh data, we just weren't tracking it
        # Red marker = valid signal, Orange marker = questionable signal
        append_to_locked_backtest(
            entry={'date': pos['entry_date'], 'price': pos['entry_price'], 'position': 'long'},
            strategy_name=strategy_name, ticker=ticker, missed=False
        )
        locked_backtest = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)

        print(f"✅ Initialized position from backtest: LONG @ ${pos['entry_price']:.2f}")

    # Case 2: State shows position but backtest shows none - POSSIBLE missed exit
    # For intraday strategies, backtest may not perfectly align with real-time tracking
    # ASK the user before automatically clearing the position
    elif state_position is not None and backtest_position is None:
        old_entry = trade_state.get('entry_price', 0)
        old_time = trade_state.get('entry_time', 'Unknown')
        print(f"\n⚠️  STATE MISMATCH: State={state_position.upper()} @ ${old_entry:.2f}, Backtest=None")
        print(f"   Your tracked position: {state_position.upper()} @ ${old_entry:.2f} (entered {str(old_time)[:16]})")
        print(f"   Backtest shows: No open position")
        print(f"\n   This could mean:")
        print(f"   1. The position was exited while trader was offline (missed exit)")
        print(f"   2. The backtest data doesn't perfectly align with live trading (common for intraday)")
        print(f"\n   Current price: ${sync_current_price:,.2f}")
        if old_entry > 0:
            current_pnl = ((sync_current_price - old_entry) / old_entry) * 100
            print(f"   Unrealized P&L: {current_pnl:+.2f}%")

        # Ask user what to do
        print(f"\n   Options:")
        print(f"   [K] Keep position - continue tracking your {state_position.upper()} position")
        print(f"   [C] Clear position - sync with backtest (record as missed exit)")
        user_choice = input(f"\n   Keep or Clear position? (K/c): ").strip().lower()

        if user_choice == 'c':
            print(f"   🔄 Clearing position to sync with backtest...")

            # Find and record the missed trade from backtest exits
            missed_trade = None
            if sync_backtest and sync_backtest.get('exits'):
                entry_date_str = str(old_time)[:10] if old_time else None

                for exit_trade in sync_backtest['exits']:
                    exit_entry_price = exit_trade.get('entry_price', 0)
                    # Safe division - check old_entry > 0 to avoid ZeroDivisionError
                    price_match = (old_entry > 0 and abs(exit_entry_price - old_entry) / old_entry < 0.005)
                    if price_match:
                        exit_date_str = str(exit_trade.get('date', ''))[:10]
                        if entry_date_str and exit_date_str and exit_date_str >= entry_date_str:
                            missed_trade = exit_trade
                            break
                        elif not entry_date_str:
                            missed_trade = exit_trade
                            break

            if missed_trade:
                stats = log_closed_trade(
                    ticker=ticker,
                    position_type=state_position,
                    entry_price=old_entry,
                    exit_price=missed_trade.get('price', 0),
                    entry_time=str(old_time),
                    exit_time=str(missed_trade.get('date', '')),
                    exit_reason=f"[SYNC] {missed_trade.get('reason', 'Unknown')}",
                    pnl_pct=missed_trade.get('pnl', 0),
                    strategy_name=strategy_name,
                    entry_signal_bar=trade_state.get('entry_signal_bar'),
                    exit_signal_bar=str(missed_trade.get('date', ''))
                )
                print(f"   📝 Recorded missed trade: {missed_trade.get('pnl', 0):+.2f}% ({missed_trade.get('reason', 'Unknown')})")

            trade_state = {'position': None, 'entry_price': None, 'entry_time': None, 'last_signal_time': None}
            save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)
            print(f"   ✅ State synced - position cleared")

            # Send Discord notification about missed exit
            pnl_info = f"\n**Missed Trade P&L:** {missed_trade.get('pnl', 0):+.2f}%" if missed_trade else ""
            sync_msg = (
                f"⚠️ **[{strategy_label}] State Sync - Missed Exit**\n"
                f"---\n"
                f"**Previous tracking:** {state_position.upper()} @ ${old_entry:.2f}\n"
                f"**Entry time:** {str(old_time)[:16]}{pnl_info}\n"
                f"---\n"
                f"**Backtest shows:** Position was closed\n"
                f"**Action:** State cleared, now tracking no position\n"
                f"**Current Price:** ${sync_current_price:,.2f}\n"
                f"---\n"
                f"_Trade recorded to history for accurate metrics._"
            )
            send_discord_alert(webhook_url, sync_msg, strategy_name=strategy_name)
        else:
            print(f"   ✅ Keeping tracked position: {state_position.upper()} @ ${old_entry:.2f}")
            print(f"   ℹ️  The trader will continue monitoring this position for exit signals")
            # CRITICAL: Set flag to prevent main loop auto-sync from overriding user's choice
            user_chose_keep_position = True

    # Case 3: Both show position but entries differ - missed exit AND new entry
    elif state_position is not None and backtest_position is not None:
        state_entry = trade_state.get('entry_price', 0)
        state_time = trade_state.get('entry_time', 'Unknown')
        backtest_entry = backtest_position.get('entry_price', 0)
        backtest_time = backtest_position.get('entry_date', 'Unknown')

        # Compare BOTH price AND date to detect different trades
        # Two trades with similar prices but different dates are DIFFERENT trades
        price_diff_pct = abs(state_entry - backtest_entry) / state_entry if state_entry > 0 else 1

        # Parse dates for comparison
        # NOTE: For crypto, UTC vs local time can differ by 1 day (midnight UTC = 7 PM ET previous day)
        state_date_str = str(state_time)[:10] if state_time else ''
        backtest_date_str = str(backtest_time)[:10] if backtest_time else ''

        # Calculate day difference (allowing for timezone skew)
        days_diff = 0
        try:
            if state_date_str and backtest_date_str:
                state_date = pd.to_datetime(state_date_str).date()
                backtest_date = pd.to_datetime(backtest_date_str).date()
                days_diff = abs((backtest_date - state_date).days)
        except Exception:  # Catch all non-system exceptions
            days_diff = 1 if state_date_str != backtest_date_str else 0

        # For crypto: allow 1-day tolerance if prices match closely (UTC vs local time issue)
        # For stocks: strict date matching
        is_crypto_ticker = '-USD' in ticker.upper() or ticker.upper() in ['BTC', 'ETH', 'SOL', 'DOGE']

        # Log timezone tolerance usage
        if is_crypto_ticker and days_diff == 1 and price_diff_pct <= 0.005:
            print(f"   ℹ️  Date differs by 1 day but prices match (likely UTC vs local timezone) - NOT syncing")

        dates_differ = days_diff > (1 if is_crypto_ticker and price_diff_pct <= 0.005 else 0)

        # Trigger sync if: prices differ by >0.5% OR dates differ significantly
        if state_entry > 0 and (price_diff_pct > 0.005 or dates_differ):
            print(f"⚠️  STATE MISMATCH: Different entries detected!")
            print(f"   State says: {state_position.upper()} @ ${state_entry:.2f}")
            print(f"   Backtest says: {backtest_position['position'].upper()} @ ${backtest_entry:.2f}")
            print(f"   🔄 Syncing state with backtest (missed exit + new entry)")

            # Find and record the missed trade from backtest exits
            missed_trade = None
            if sync_backtest and sync_backtest.get('exits'):
                # Parse entry time for date comparison
                entry_date_str = str(state_time)[:10] if state_time else None

                for exit_trade in sync_backtest['exits']:
                    # Match by entry price (within 0.5% tolerance)
                    if abs(exit_trade.get('entry_price', 0) - state_entry) / state_entry < 0.005:
                        # ALSO validate exit date is AFTER entry date
                        exit_date_str = str(exit_trade.get('date', ''))[:10]
                        if entry_date_str and exit_date_str and exit_date_str >= entry_date_str:
                            missed_trade = exit_trade
                            break
                        elif not entry_date_str:
                            # If we can't parse entry date, accept the match (legacy behavior)
                            missed_trade = exit_trade
                            break

            if missed_trade:
                # Record the missed trade to history
                stats = log_closed_trade(
                    ticker=ticker,
                    position_type=state_position,
                    entry_price=state_entry,
                    exit_price=missed_trade.get('price', 0),
                    entry_time=str(state_time),
                    exit_time=str(missed_trade.get('date', '')),
                    exit_reason=f"[SYNC] {missed_trade.get('reason', 'Unknown')}",
                    pnl_pct=missed_trade.get('pnl', 0),
                    strategy_name=strategy_name,
                    entry_signal_bar=trade_state.get('entry_signal_bar'),  # Use stored signal bar if available
                    exit_signal_bar=str(missed_trade.get('date', ''))  # Exit bar from backtest
                )
                print(f"   📝 Recorded missed trade: {missed_trade.get('pnl', 0):+.2f}% ({missed_trade.get('reason', 'Unknown')})")

            trade_state = {
                'position': backtest_position['position'],
                'entry_price': backtest_position['entry_price'],
                'entry_time': backtest_position['entry_date'],
                'entry_signal_bar': backtest_position['entry_date'],  # Signal bar is the entry date from backtest
                'last_signal_time': backtest_position['entry_date'],
            }
            save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)
            print(f"   ✅ State synced - now tracking: {backtest_position['position'].upper()} @ ${backtest_entry:.2f}")

            # Send Discord notification about missed exit + new entry
            pnl_info = f"\n**Missed Trade P&L:** {missed_trade.get('pnl', 0):+.2f}%" if missed_trade else ""
            unrealized_pnl = ((sync_current_price - backtest_entry) / backtest_entry * 100) if backtest_entry > 0 else 0
            pnl_emoji = "🟢" if unrealized_pnl >= 0 else "🔴"
            sync_msg = (
                f"⚠️ **[{strategy_label}] State Sync - Missed Exit + New Entry**\n"
                f"---\n"
                f"**Previous tracking:** {state_position.upper()} @ ${state_entry:.2f}\n"
                f"**Previous entry time:** {str(state_time)[:16]}{pnl_info}\n"
                f"---\n"
                f"{pnl_emoji} **Now tracking:** {backtest_position['position'].upper()} @ ${backtest_entry:.2f} ({unrealized_pnl:+.1f}%)\n"
                f"**New entry time:** {str(backtest_time)[:16]}\n"
                f"**Current Price:** ${sync_current_price:,.2f}\n"
                f"---\n"
                f"_Trade recorded to history for accurate metrics._"
            )
            send_discord_alert(webhook_url, sync_msg, strategy_name=strategy_name)

    # ============================================================
    # LOCK THE BACKTEST - Freeze entry/exit markers to prevent repainting
    # ============================================================
    # Check if locked backtest already exists
    locked_backtest_path = get_locked_backtest_path(strategy_name=strategy_name, ticker=ticker)
    existing_locked = os.path.exists(locked_backtest_path)

    locked_backtest = None
    if existing_locked:
        # CRITICAL FIX: Just LOAD the existing locked_backtest - don't merge with fresh_backtest!
        # The save_locked_backtest was incorrectly adding fresh backtest entries on every startup,
        # causing trade count to grow from 25 to 62. New signals should only be added when they
        # actually occur during live trading (via append_to_locked_backtest).
        print(f"\n🔒 Loading existing locked backtest (preserving historical data)...")
        print(f"   📁 Path: {locked_backtest_path}")
        locked_backtest = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)
        if locked_backtest:
            print(f"   ✅ Loaded: {locked_backtest.get('num_trades', 0)} trades, {locked_backtest.get('win_rate', 0):.1f}% win rate, {locked_backtest.get('total_return', 0):.1f}% return")
        else:
            print(f"   ⚠️ Failed to load locked_backtest from {locked_backtest_path}")
            # Don't create new one here - we'll fall through to the else block below
            existing_locked = False  # Force creation of new locked_backtest
    else:
        # FIRST STARTUP - seed from bundled backtest_results.json if available
        # This ensures we use EXACT same trades as shown in Streamlit UI
        bundle_name = config.get('bundle_name')
        bundled_backtest_path = os.path.join(VELOCITY_STRATEGIES_DIR, bundle_name, "backtest_results.json") if bundle_name else None

        if bundled_backtest_path and os.path.exists(bundled_backtest_path):
            # Load pre-computed backtest results from Streamlit deploy
            print(f"\n🔒 Loading backtest results from bundle: {bundled_backtest_path}")
            try:
                with open(bundled_backtest_path, 'r') as f:
                    bundled_results = json.load(f)

                # Validate bundle schema (R2.3: ensures bundle has required fields)
                required_keys = ['entries', 'exits']
                missing_keys = [k for k in required_keys if k not in bundled_results]
                if missing_keys:
                    raise ValueError(f"Invalid bundle schema: missing required keys {missing_keys}")

                # Calculate stats from exits
                exits = bundled_results.get('exits', [])
                if exits:
                    pnls = [e.get('pnl', 0) for e in exits]
                    wins = [e for e in exits if e.get('pnl', 0) > 0]
                    losses = [e for e in exits if e.get('pnl', 0) <= 0]
                    win_rate = (len(wins) / len(exits)) * 100 if exits else 0
                    total_return = (np.prod([1 + p/100 for p in pnls]) - 1) * 100 if pnls else 0
                    gross_profit = sum(e.get('pnl', 0) for e in wins) if wins else 0
                    gross_loss = abs(sum(e.get('pnl', 0) for e in losses)) if losses else 0.001
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
                else:
                    win_rate = total_return = profit_factor = 0

                locked_backtest = {
                    "locked_at": datetime.now().isoformat(),
                    "entries": bundled_results.get('entries', []),
                    "exits": exits,
                    "current_position": bundled_results.get('current_position'),
                    "num_trades": len(exits),
                    "win_rate": win_rate,
                    "total_return": total_return,
                    "profit_factor": profit_factor,
                }

                # Save as the locked_backtest file
                safe_json_write(locked_backtest_path, locked_backtest, indent=2)

                print(f"✅ Loaded {len(exits)} trades from bundled backtest_results.json")
                print(f"📁 Locked backtest file: {locked_backtest_path}")
            except Exception as e:
                print(f"⚠️ Could not load bundled backtest: {e}")
                # Fall back to regenerated backtest
                if backtest:
                    print(f"🔒 Falling back to regenerated backtest...")
                    locked_backtest = save_locked_backtest(backtest, strategy_name=strategy_name, ticker=ticker, trade_state=trade_state)
        elif backtest:
            # Fallback: seed from regenerated backtest (full historical data)
            print(f"\n🔒 Creating initial locked backtest from regenerated data...")
            locked_backtest = save_locked_backtest(backtest, strategy_name=strategy_name, ticker=ticker, trade_state=trade_state)
            print(f"📁 Locked backtest file: {locked_backtest_path}")
        elif fresh_backtest:
            print(f"\n🔒 Creating initial locked backtest from recent data...")
            locked_backtest = save_locked_backtest(fresh_backtest, strategy_name=strategy_name, ticker=ticker, trade_state=trade_state)
            print(f"📁 Locked backtest file: {locked_backtest_path}")

    # RETROACTIVE SYNC: Add missed signals to locked_backtest as real signals
    # This ensures complete signal history even if bot was offline
    # Signals are added with mark_as_missed=False so they appear as regular markers
    missed_signals = None  # No longer using ghost markers
    if locked_backtest and fresh_backtest:
        # DISABLED: Missed signals sync was causing consistent data corruption
        # The duplicate detection has timezone/timestamp format issues that cause
        # the same exit to be added multiple times with slightly different timestamps
        # TODO: Fix the root cause - timestamp normalization in get_missed_signals
        #       and detect_and_add_missed_signals needs to be more robust
        potential_missed_signals = get_missed_signals(locked_backtest, fresh_backtest)
        potential_missed = len(potential_missed_signals.get('entries', [])) + len(potential_missed_signals.get('exits', []))

        if potential_missed > 0:
            print(f"ℹ️  Detected {potential_missed} potential signal(s) while offline (sync DISABLED to prevent corruption)")

    # Validate locked backtest data integrity
    if locked_backtest:
        integrity_issues = validate_locked_backtest(locked_backtest)
        if integrity_issues:
            print(f"\n⚠️  DATA INTEGRITY ISSUES DETECTED ({len(integrity_issues)}):")
            for issue in integrity_issues[:5]:  # Show first 5
                print(f"   • {issue}")
            if len(integrity_issues) > 5:
                print(f"   ... and {len(integrity_issues) - 5} more issues")

            # DISABLED: Auto-cleanup was too aggressive and destroyed valid backtests
            # The cleanup_locked_backtest function has a bug where mismatched entry_date
            # fields cause ALL subsequent entries to be removed as "overlapping"
            # TODO: Fix cleanup_locked_backtest to be more tolerant of format mismatches
            print(f"\n⚠️  Auto-cleanup DISABLED (was destroying valid data)")
            print(f"   Please manually verify and fix data integrity issues")
        else:
            print(f"✅ Locked backtest data integrity validated")

    # Send startup notification with stats and chart
    startup_chart = None
    if backtest:
        # Get current price from fresh data if available
        current_price = fresh_df.iloc[-1]['close'] if fresh_backtest and not fresh_df.empty else df.iloc[-1]['close']

        # Use SYNCED trade_state for position display, not bundled backtest
        if trade_state.get('position'):
            pos_type = trade_state.get('position', 'long').upper()
            entry_price = trade_state.get('entry_price', 0)
            # Calculate P&L correctly for LONG vs SHORT
            if pos_type == 'LONG':
                unrealized_pnl = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            else:  # SHORT
                unrealized_pnl = ((entry_price - current_price) / entry_price * 100) if entry_price > 0 else 0
            pnl_emoji = "🟢" if unrealized_pnl >= 0 else "🔴"
            pos_status = f"{pnl_emoji} **Position:** {pos_type} @ ${entry_price:.2f} ({unrealized_pnl:+.1f}%)"
        else:
            pos_status = "⚪ **Position:** None"

        # Build stats sections - show tracked stats from locked_backtest (excludes missed trades)
        # Use locked_backtest for tracked stats if available, otherwise fall back to bundled backtest
        # DEBUG: Log which source is being used
        print(f"📊 Stats source selection:")
        print(f"   locked_backtest exists: {locked_backtest is not None}")
        if locked_backtest:
            print(f"   locked_backtest trades: {locked_backtest.get('num_trades', 'N/A')}, WR: {locked_backtest.get('win_rate', 'N/A')}")
        print(f"   backtest trades: {backtest.get('num_trades', 'N/A') if backtest else 'N/A'}")
        stats_source = locked_backtest if locked_backtest else backtest
        print(f"   USING: {'locked_backtest' if stats_source == locked_backtest else 'backtest'} ({stats_source.get('num_trades', 0)} trades)")
        num_tracked = stats_source.get('num_trades', 0)
        num_missed = stats_source.get('num_missed', 0)

        full_period_stats = (
            f"📊 **Tracked Stats ({num_tracked} trades):**\n"
            f"• Win Rate: {stats_source.get('win_rate', 0):.0f}% | Return: {stats_source.get('total_return', 0):.1f}%\n"
            f"• Profit Factor: {stats_source.get('profit_factor', 0):.1f}"
        )

        # Show missed trades stats separately if any exist
        missed_stats = ""
        if num_missed > 0:
            missed_return = stats_source.get('missed_return', 0)
            missed_win_rate = stats_source.get('missed_win_rate', 0)
            missed_stats = (
                f"\n🟠 **Missed During Downtime ({num_missed} trades):**\n"
                f"• Win Rate: {missed_win_rate:.0f}% | Return: {missed_return:.1f}%\n"
                f"• _(Not included in main stats)_"
            )

        # Calculate Recent stats from locked_backtest
        # Adjust bars shown based on interval to match chart
        recent_stats = ""
        interval = config.get('interval', '1d')

        # Calculate actual days based on bars_per_day (accounting for futures/crypto extended hours)
        # Stock market: ~6.5 hours/day, Futures: ~23 hours/day, Crypto: 24 hours/day
        bars_per_day_map = {
            '1m': 390, '5m': 78, '15m': 26, '30m': 13, '1h': 7, '90m': 5, '4h': 2, '1d': 1
        }
        is_futures = is_futures_ticker(ticker)
        is_crypto = '-USD' in ticker.upper()
        bars_per_day = bars_per_day_map.get(interval, 1)
        if is_crypto and interval not in ['1d']:
            # Crypto trades 24/7: multiply by ~3.7 (24 hours / 6.5 stock hours)
            bars_per_day = int(bars_per_day * 3.7)
        elif is_futures and interval not in ['1d']:
            bars_per_day = int(bars_per_day * 3.5)  # Futures trade ~23 hours

        if interval in ['15m', '5m', '1m']:
            recent_bars = 260
        elif interval in ['30m']:
            recent_bars = 260
        elif interval in ['1h', '90m']:
            recent_bars = 260
        else:
            recent_bars = 126

        if locked_backtest and locked_backtest.get('exits'):
            # Get the chart date range using same logic as chart generation
            chart_df = fresh_df if fresh_backtest and not fresh_df.empty else df
            if len(chart_df) > recent_bars:
                df_recent_for_stats = chart_df.iloc[-recent_bars:]
            else:
                df_recent_for_stats = chart_df
            chart_start = normalize_tz(df_recent_for_stats.index.min())
            chart_end = normalize_tz(df_recent_for_stats.index.max())

            # Calculate actual days from REAL data range (not bar count estimate)
            # This handles weekends, holidays, and data gaps correctly
            if len(df_recent_for_stats) > 1:
                actual_days = max(1, (df_recent_for_stats.index.max() - df_recent_for_stats.index.min()).days)
            else:
                actual_days = 1
            day_word = "Day" if actual_days == 1 else "Days"
            recent_label = f"{actual_days} {day_word}"

            # Filter exits to chart window (matches chart exactly)
            # IMPORTANT: Exclude 'missed' trades to match chart legend stats
            recent_exits = []
            for exit_trade in locked_backtest['exits']:
                # Skip missed trades (missed=True flag or [SYNC]/[MISSED] in reason)
                if exit_trade.get('missed'):
                    continue
                reason = str(exit_trade.get('reason', ''))
                if '[SYNC]' in reason or '[MISSED]' in reason:
                    continue
                exit_date = exit_trade.get('date', '')
                if isinstance(exit_date, str):
                    try:
                        exit_date = pd.to_datetime(exit_date)
                    except Exception:  # Catch all non-system exceptions
                        continue
                exit_date = normalize_tz(exit_date)
                if exit_date >= chart_start and exit_date <= chart_end:
                    recent_exits.append(exit_trade)

            if recent_exits:
                winners = [e for e in recent_exits if e['pnl'] > 0]
                losers = [e for e in recent_exits if e['pnl'] <= 0]
                win_rate = (len(winners) / len(recent_exits)) * 100
                # Compounded return
                sorted_exits = sorted(recent_exits, key=lambda x: str(x.get('date', '')))
                equity = 1.0
                for exit_trade in sorted_exits:
                    equity *= (1 + exit_trade['pnl'] / 100)
                total_return = (equity - 1) * 100
                total_wins = sum(e['pnl'] for e in winners) if winners else 0
                total_losses = abs(sum(e['pnl'] for e in losers)) if losers else 0.001
                profit_factor = total_wins / total_losses if total_losses > 0 else total_wins

                recent_stats = (
                    f"\n📈 **Recent {recent_label} ({len(recent_exits)} trades):**\n"
                    f"• Win Rate: {win_rate:.0f}% | Return: {total_return:.1f}%\n"
                    f"• Profit Factor: {profit_factor:.1f}"
                )

        # Calculate SL/TP dollar values
        sl_price = current_price * (1 - stop_loss_pct / 100)
        tp_price = current_price * (1 + take_profit_pct / 100)

        startup_msg = (
            f"🤖 **[{strategy_label}] Live Trader Started**\n"
            f"**Strategy:** {strategy_name}\n"
            f"**Ticker:** {ticker}\n"
            f"**Current Price:** ${current_price:,.2f}\n"
            f"**Signal Type:** {config.get('signal_type')}\n"
            f"**Risk:** SL={stop_loss_pct:.1f}% (${sl_price:,.2f}), TP={take_profit_pct:.1f}% (${tp_price:,.2f})\n"
            f"---\n"
            f"{full_period_stats}{missed_stats}{recent_stats}\n"
            f"---\n"
            f"{pos_status}\n"
            f"---\n"
            f"_Monitoring for signals..._"
        )
        # Generate charts for startup
        # Use fresh data if bundled data is stale (so current position shows on chart)
        # Markers come from locked_backtest to prevent repainting
        try:
            startup_trade_history = load_trade_history(strategy_name=strategy_name, ticker=ticker)

            # Determine which data to use for charts
            # Use fresh_df_for_charts (includes incomplete bar) to show current data
            # Otherwise fall back to bundled data
            chart_df = df  # Default to bundled
            chart_data_source = "bundled"  # Track source for accurate logging
            try:
                # Use fresh data INCLUDING incomplete bar for charts (shows current price)
                if fresh_df_for_charts is not None and len(fresh_df_for_charts) > 0:
                    # Calculate oscillators AND velocity signals for the chart data
                    fresh_chart_df = calculate_composite_oscillator(fresh_df_for_charts.copy(), config)
                    fresh_chart_df = calculate_velocity_signals(fresh_chart_df, config)
                    bundled_end = normalize_tz(df.index[-1]) if len(df) > 0 else None
                    fresh_end = normalize_tz(fresh_chart_df.index[-1])
                    if bundled_end is None or fresh_end >= bundled_end:
                        chart_df = fresh_chart_df
                        chart_data_source = "fresh"
                        print(f"📊 Using FRESH data for charts (bundled ends {bundled_end.strftime('%Y-%m-%d') if bundled_end else 'N/A'}, fresh ends {fresh_end.strftime('%Y-%m-%d')})")
                    else:
                        print(f"📊 Using bundled data for charts (up to {bundled_end.strftime('%Y-%m-%d') if bundled_end else 'unknown'})")
                else:
                    print(f"📊 Using bundled data for charts (fresh data not available)")
            except Exception as e:
                print(f"📊 Using bundled data for charts (error checking fresh: {e})")

            # FULL PERIOD chart - use locked_backtest for stats to prevent repainting
            full_period_chart = generate_velocity_chart(chart_df, locked_backtest, config, ticker,
                                                        title_suffix=" - Full Period",
                                                        trade_history=startup_trade_history,
                                                        current_position=trade_state,
                                                        locked_backtest=locked_backtest,
                                                        full_period_backtest=locked_backtest,
                                                        missed_signals=missed_signals)
            print("✅ Generated full period chart")

            # RECENT chart - subset of chart data
            # Adjust number of bars based on interval to show reasonable time period
            interval = config.get('interval', '1d')

            # Calculate actual days based on bars_per_day (accounting for futures/crypto extended hours)
            bars_per_day_map = {
                '1m': 390, '5m': 78, '15m': 26, '30m': 13, '1h': 7, '90m': 5, '4h': 2, '1d': 1
            }
            is_futures = is_futures_ticker(ticker)
            is_crypto = '-USD' in ticker.upper()
            bars_per_day = bars_per_day_map.get(interval, 1)
            if is_crypto and interval not in ['1d']:
                bars_per_day = int(bars_per_day * 3.7)  # Crypto trades 24/7
            elif is_futures and interval not in ['1d']:
                bars_per_day = int(bars_per_day * 3.5)  # Futures trade ~23 hours

            if interval in ['15m', '5m', '1m']:
                recent_bars = 260
            elif interval in ['30m']:
                recent_bars = 260
            elif interval in ['1h', '90m']:
                recent_bars = 260
            else:
                recent_bars = 126

            if len(chart_df) > recent_bars:
                df_recent = chart_df.iloc[-recent_bars:].copy()
            else:
                df_recent = chart_df.copy()

            # Calculate actual days from REAL data range (not bar count estimate)
            # This handles weekends, holidays, and data gaps correctly
            if len(df_recent) > 1:
                actual_days = max(1, (df_recent.index.max() - df_recent.index.min()).days)
            else:
                actual_days = 1

            if len(chart_df) <= recent_bars:
                recent_label = "Full Period"
            else:
                day_word = "Day" if actual_days == 1 else "Days"
                recent_label = f"{actual_days} {day_word}"

            # Calculate subset stats from locked_backtest for the recent period
            # Use BOTH start and end boundaries to match message stats exactly
            # EXCLUDE missed/sync trades from stats (consistent with trade_history filtering)
            recent_start = normalize_tz(df_recent.index.min())
            recent_end = normalize_tz(df_recent.index.max())
            recent_exits = []
            if locked_backtest and locked_backtest.get('exits'):
                for exit_trade in locked_backtest['exits']:
                    # Skip missed trades (missed=True flag or [SYNC]/[MISSED] in reason)
                    if exit_trade.get('missed'):
                        continue
                    reason = str(exit_trade.get('reason', ''))
                    if '[SYNC]' in reason or '[MISSED]' in reason:
                        continue
                    exit_date = normalize_tz(pd.to_datetime(exit_trade['date']))
                    if exit_date >= recent_start and exit_date <= recent_end:
                        recent_exits.append(exit_trade)

            # Build recent_backtest with exits for equity curve
            # Use centralized stats calculation (R6.1: single source of truth)
            stats = calculate_exit_stats(recent_exits, exclude_missed=False)  # Already filtered above
            recent_backtest = {
                'num_trades': stats['num_trades'],
                'win_rate': stats['win_rate'],
                'total_return': stats['total_return'],
                'profit_factor': stats['profit_factor'],
                'exits': recent_exits,  # Include exits for equity curve
                'current_position': locked_backtest.get('current_position') if locked_backtest else None,
                'period_days': actual_days  # Use calculated actual days, not raw bar count
            }

            recent_chart = generate_velocity_chart(df_recent, recent_backtest, config, ticker,
                                                   title_suffix=f" - Last {recent_label}",
                                                   trade_history=startup_trade_history,
                                                   current_position=trade_state,
                                                   locked_backtest=locked_backtest,
                                                   full_period_backtest=locked_backtest,
                                                   missed_signals=missed_signals)
            print(f"✅ Generated recent {recent_label} chart ({chart_data_source} data, {recent_backtest['num_trades']} trades)")

            # Use recent chart for main display (full period chart can be added later)
            startup_chart = recent_chart
        except Exception as e:
            print(f"⚠️ Could not generate startup chart: {e}")
            import traceback
            traceback.print_exc()
            startup_chart = None
    else:
        # Get current price from available data for SL/TP calculations
        fallback_price = df.iloc[-1]['close'] if not df.empty else 0
        sl_price_fallback = fallback_price * (1 - stop_loss_pct / 100) if fallback_price > 0 else 0
        tp_price_fallback = fallback_price * (1 + take_profit_pct / 100) if fallback_price > 0 else 0

        if fallback_price > 0:
            risk_line = f"**Risk:** SL={stop_loss_pct:.1f}% (${sl_price_fallback:,.2f}), TP={take_profit_pct:.1f}% (${tp_price_fallback:,.2f})"
        else:
            risk_line = f"**Risk:** SL={stop_loss_pct:.1f}%, TP={take_profit_pct:.1f}%"

        startup_msg = (
            f"🤖 **[{strategy_label}] Live Trader Started**\n"
            f"**Strategy:** {strategy_name}\n"
            f"**Ticker:** {ticker}\n"
            f"**Interval:** {interval}\n"
            f"**Signal Type:** {config.get('signal_type')}\n"
            f"{risk_line}\n"
            f"---\n"
            f"_Monitoring for signals..._"
        )

    # Generate CSV trade log attachment (last 10 trades)
    trade_log_csv = None
    if locked_backtest:
        try:
            trade_log_csv = generate_trade_log_csv(locked_backtest, num_trades=10, trade_state=trade_state)
            if trade_log_csv:
                print("✅ Generated trade log CSV (last 10 trades)")
        except Exception as e:
            print(f"⚠️ Could not generate trade log CSV: {e}")

    send_discord_alert(webhook_url, startup_msg, startup_chart, strategy_name=strategy_name, csv_buf=trade_log_csv)

    # Determine check interval based on data interval
    # Minimize delay for actionable signals - check frequently after bar closes
    if interval == "1d":
        check_interval_seconds = 120  # Check every 2 min for daily signals
    elif interval == "1h":
        check_interval_seconds = 60   # Check every 1 min for hourly signals
    elif interval in ["15m", "30m"]:
        check_interval_seconds = 30   # Check every 30 sec for 15m/30m signals
    else:
        check_interval_seconds = 30   # Default: check every 30 sec for fast signals

    # Track scheduled alerts to avoid duplicates
    # Persist to file to survive restarts and prevent duplicate alerts on same day
    def get_daily_alerts_path():
        return f"daily_alerts_{strategy_name or ticker}.json"

    def load_daily_alerts() -> set:
        """Load persisted daily alerts, filtering to today's alerts only."""
        alerts_path = get_daily_alerts_path()
        today_str = datetime.now(CHICAGO_TZ).strftime('%Y-%m-%d')  # Chicago date
        try:
            if os.path.exists(alerts_path):
                data = safe_json_read(alerts_path, default=[])
                # Filter to only today's alerts (alerts from previous days are irrelevant)
                return {a for a in data if a.startswith(today_str)}
        except Exception:
            pass
        return set()

    def save_daily_alerts(alerts: set):
        """Persist daily alerts to file."""
        alerts_path = get_daily_alerts_path()
        today_str = datetime.now(CHICAGO_TZ).strftime('%Y-%m-%d')  # Chicago date
        # Only save today's alerts
        today_alerts = [a for a in alerts if a.startswith(today_str)]
        safe_json_write(alerts_path, today_alerts)

    daily_alerts = load_daily_alerts()
    print(f"   📋 Loaded {len(daily_alerts)} persisted alerts for today")

    # Track the last completed bar we evaluated for signals (prevents re-evaluation)
    last_signal_bar_evaluated = None

    # Track the last bar we posted a status update for (prevents duplicate bar updates)
    last_bar_update_time = None

    # Determine if this is a crypto or stock ticker for signal timing
    is_crypto = '-USD' in ticker.upper()

    # Determine if this is an intraday strategy (15m, 1h, etc.) for more frequent status updates
    # Note: Signal alerts (entry/exit) are posted immediately regardless - this is for STATUS updates
    is_intraday = interval in ['1m', '5m', '15m', '30m', '1h', '90m', '4h']

    # Intraday schedule: status updates at key market times
    # ES futures trade 18:00-17:00 ET with 1hr break (17:00-18:00)
    # These are STATUS updates - entry/exit SIGNALS are posted immediately when they occur
    intraday_update_windows = [
        ("07:30", "08:00", "🌅 Pre-Market Status"),        # Before stock market open
        ("11:30", "12:00", "☀️ Mid-Day Status"),           # Late morning
        ("15:30", "16:00", "🌇 Afternoon Status"),         # Near stock close
        ("19:00", "19:30", "🌙 Evening Session Status"),   # After futures re-open
    ]

    # STARTUP CATCH-UP: If crypto and we just missed a scheduled update window, send immediately
    # This handles cases where trader started at 6:51 AM but 6:00 AM window was missed
    if is_crypto and backtest:
        startup_now = datetime.now(CHICAGO_TZ)  # Use Chicago time
        startup_hour = startup_now.hour
        startup_minute = startup_now.minute

        # Check if we're within 2 hours AFTER a scheduled crypto update time (0, 6, 12, 18)
        # e.g., if current time is 7:30, we missed the 6:00 update
        scheduled_hours = [0, 6, 12, 18]
        for sched_hour in scheduled_hours:
            # Check if current time is between sched_hour:45 and sched_hour+2:00
            # (i.e., we're past the 45-min window but within 2 hours of the scheduled time)
            hours_after = (startup_hour - sched_hour) % 24
            if hours_after == 0 and startup_minute >= 45:
                # We're in the same hour but past the 45-min window
                print(f"🔔 STARTUP CATCH-UP: Missed {sched_hour}:00 update (now {startup_hour}:{startup_minute:02d})")
                send_status_update(webhook_url, fresh_df if fresh_backtest else df, fresh_backtest or backtest, config, ticker,
                                 f"🔔 Startup Catch-up ({sched_hour}:00)", trade_state, strategy_name=strategy_name)
                day_str = startup_now.strftime('%Y-%m-%d')
                daily_alerts.add(f"{day_str}_CRYPTO_{sched_hour:02d}")
                break
            elif hours_after == 1 and startup_minute < 30:
                # We're 1 hour after (e.g., 7:15 after 6:00 update)
                print(f"🔔 STARTUP CATCH-UP: Missed {sched_hour}:00 update (now {startup_hour}:{startup_minute:02d})")
                send_status_update(webhook_url, fresh_df if fresh_backtest else df, fresh_backtest or backtest, config, ticker,
                                 f"🔔 Startup Catch-up ({sched_hour}:00)", trade_state, strategy_name=strategy_name)
                day_str = startup_now.strftime('%Y-%m-%d')
                daily_alerts.add(f"{day_str}_CRYPTO_{sched_hour:02d}")
                break

    # Track consecutive data fetch failures for backoff
    consecutive_fetch_failures = 0
    max_failures_before_alert = 3

    # AUTO-SYNC flag: Only allow auto-sync on FIRST cycle after startup
    # This prevents backtest flip-flopping from causing spurious entries/exits during live trading
    # CRITICAL: If user chose "Keep" at startup, skip auto-sync entirely to respect their choice
    auto_sync_completed = user_chose_keep_position
    if user_chose_keep_position:
        print(f"   ℹ️  Auto-sync disabled (user chose to keep tracked position)")

    while True:
        try:
            current_time_str = datetime.now(CHICAGO_TZ).strftime('%Y-%m-%d %H:%M:%S CT')
            print(f"\n[{current_time_str}] 📡 {strategy_label} | {strategy_name} | Update Cycle")

            # --- CONFIG HOT-RELOAD ---
            if os.path.exists(config_path):
                current_mtime = os.path.getmtime(config_path)
                if current_mtime > last_config_mtime:
                    print("📂 Strategy config changed on disk. Reloading...")
                    try:
                        new_config = load_config(config_path)
                        last_config_mtime = current_mtime
                        config = new_config

                        # SAFETY: Never change ticker or interval during an open trade
                        # These changes could cause catastrophic tracking issues
                        has_open_position = trade_state and trade_state.get('position')
                        new_ticker = config.get('ticker', ticker)
                        new_interval = config.get('interval', interval)

                        if has_open_position:
                            if new_ticker != ticker:
                                print(f"⚠️  BLOCKED: Cannot change ticker from {ticker} to {new_ticker} during open position!")
                                print(f"   Close your position first, then change the ticker.")
                            else:
                                ticker = new_ticker  # Same ticker, allow

                            if new_interval != interval:
                                print(f"⚠️  BLOCKED: Cannot change interval from {interval} to {new_interval} during open position!")
                                print(f"   Close your position first, then change the interval.")
                            else:
                                interval = new_interval  # Same interval, allow
                        else:
                            # No open position - safe to change ticker/interval
                            ticker = new_ticker
                            interval = new_interval

                        # These parameters can be safely changed mid-trade
                        api_key = config.get('polygon_api_key', api_key)
                        webhook_url = config.get('discord_webhook') or DEFAULT_DISCORD_WEBHOOK
                        strategy_name = config.get('strategy_name', strategy_name)
                        stop_loss_pct = config.get('stop_loss_pct', stop_loss_pct)
                        take_profit_pct = config.get('take_profit_pct', take_profit_pct)
                        exit_on_opposite_signal = config.get('exit_on_opposite_signal', exit_on_opposite_signal)
                        exit_on_midline_cross = config.get('exit_on_midline_cross', exit_on_midline_cross)

                        print(f"✅ Config reloaded: {ticker} / {config.get('signal_type')}")

                        # Notify Discord
                        send_discord_alert(webhook_url, f"🔄 **Config Reloaded**\nTicker: {ticker}\nSignal: {config.get('signal_type')}", strategy_name=strategy_name)
                    except Exception as e:
                        print(f"⚠️ Failed to reload config: {e}")

            print(f"Fetching data for {ticker}...")

            # Fetch data with proper error handling and backoff
            df = fetch_price_data(ticker, api_key, days=200, interval=interval)

            if df is None or df.empty:
                consecutive_fetch_failures += 1
                backoff_multiplier = min(consecutive_fetch_failures, 5)  # Cap at 5x
                wait_time = check_interval_seconds * backoff_multiplier

                print(f"⚠️ No data received for {ticker} (failure #{consecutive_fetch_failures})")
                print(f"   Backing off for {wait_time} seconds...")

                # Alert Discord after 3 consecutive failures
                if consecutive_fetch_failures == max_failures_before_alert:
                    try:
                        send_discord_alert(
                            webhook_url,
                            f"⚠️ **Data Fetch Alert**\n"
                            f"Ticker: {ticker}\n"
                            f"Consecutive failures: {consecutive_fetch_failures}\n"
                            f"Will retry with backoff...",
                            strategy_name=strategy_name
                        )
                    except Exception as e:
                        print(f"   Failed to send Discord alert: {e}")

                time.sleep(wait_time)
                continue

            # Reset failure counter on successful fetch
            if consecutive_fetch_failures > 0:
                print(f"   ✅ Data fetch recovered after {consecutive_fetch_failures} failures")
                consecutive_fetch_failures = 0

            # Calculate oscillator and signals
            df = calculate_composite_oscillator(df, config)
            df = calculate_velocity_signals(df, config)

            # v9 regime detection
            live_regimes, live_is_high_vol = compute_bar_regimes(df, config) if config.get('regime_aware', False) and V9_REGIME_AVAILABLE else (None, None)
            live_regime_signal_dfs = {}
            if live_regimes is not None:
                regime_params_dict = config.get('regime_params', {})
                for rname, rcfg in regime_params_dict.items():
                    if rcfg.get('dont_trade', False):
                        continue
                    merged = {**config, **rcfg}
                    live_regime_signal_dfs[rname] = calculate_velocity_signals(df.copy(), merged)
                current_regime_id = int(live_regimes[-1]) if len(live_regimes) > 0 else None
                current_regime_name = REGIME_NAMES.get(current_regime_id, 'unknown') if current_regime_id is not None else None
            else:
                current_regime_id = None
                current_regime_name = None

            # Get latest row
            latest = df.iloc[-1]
            bar_close_price = latest['close']  # Daily bar close (for signal logic)
            current_time = df.index[-1]

            # Fetch real-time price for display (separate from daily bar)
            realtime_price = fetch_realtime_price(ticker)
            current_price = realtime_price if realtime_price else bar_close_price

            print(f"Current Price: ${current_price:.2f} (real-time)" if realtime_price else f"Current Price: ${current_price:.2f} (bar close)")
            if current_regime_name is not None:
                print(f"Regime: {current_regime_name.upper()}")
            print(f"Oscillator: {latest['osc_smooth']:.4f}")
            print(f"Velocity: {latest['velocity']:.4f}")
            print(f"Acceleration: {latest['acceleration']:.4f}")
            print(f"Buy Signal: {latest['buy_signal']}")
            print(f"Sell Signal: {latest['sell_signal']}")
            print(f"Position: {trade_state.get('position', 'none')}")

            # CRITICAL: Flag to prevent re-entry on the same cycle after an exit
            # Without this, exiting on "Opposite Signal" would immediately re-enter
            # on a buy signal from a previous bar (since position is now None)
            just_exited_this_cycle = False

            # --- SCHEDULED STATUS UPDATES ---
            # Use Chicago time for all scheduling (consistent timezone)
            now = datetime.now(CHICAGO_TZ)
            day_str = now.strftime("%Y-%m-%d")
            hm = now.strftime("%H:%M")
            hour = now.hour
            minute = now.minute

            # Skip scheduled stock updates on weekends and holidays
            skip_stock_updates = False
            if not is_crypto:
                is_holiday_today, holiday_name = is_market_holiday(now.date(), ticker)
                if is_holiday_today:
                    skip_stock_updates = True
                    if minute == 0:  # Only log once per hour to avoid spam
                        print(f"   🏖️ Skipping scheduled updates - {holiday_name}")
                elif now.weekday() >= 5:  # Saturday or Sunday
                    skip_stock_updates = True
                    if minute == 0:  # Only log once per hour
                        print(f"   📅 Skipping scheduled updates - Weekend")

            # Run backtest for status updates
            status_backtest = None
            try:
                status_backtest = run_historical_backtest(df, config)
            except Exception as e:
                print(f"   ⚠️ Backtest failed: {e}")

            # Check for position sync issues - AUTO-SYNC ONLY AT STARTUP
            # CRITICAL: Do NOT auto-sync during live trading - backtests can flip-flop
            # as new bars complete, causing spurious entries/exits
            # Only run auto-sync on the FIRST cycle after startup
            # CRITICAL: Also skip auto-sync if the last bar is incomplete - the backtest
            # position could be based on a forming bar that will flip when it completes
            bar_incomplete_for_sync = is_last_bar_incomplete(df, interval, ticker)
            if bar_incomplete_for_sync:
                print(f"   ⏳ Auto-sync deferred - last bar still forming (will check when complete)")

            if not auto_sync_completed and status_backtest and status_backtest.get('current_position') and not bar_incomplete_for_sync:
                bt_pos = status_backtest['current_position']
                if not trade_state.get('position'):
                    print(f"   🔄 AUTO-SYNC (startup): Backtest shows {bt_pos.get('position', 'unknown').upper()} @ ${bt_pos.get('entry_price', 0):.2f}")
                    print(f"   🔄 trade_state has no position - syncing now...")

                    # Sync trade_state to match backtest position
                    bt_entry_date = bt_pos.get('entry_date', str(datetime.now()))
                    bt_entry_price = bt_pos.get('entry_price', 0)
                    trade_state = {
                        'position': bt_pos.get('position', 'long'),
                        'entry_price': bt_entry_price,
                        'entry_time': bt_entry_date,
                        'entry_signal_bar': bt_entry_date,
                        'last_signal_time': bt_entry_date,  # Reset to entry time to allow future signals
                    }
                    save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)

                    # Also sync the entry to locked_backtest so charts/stats are consistent
                    # Use missed=False - this IS a valid signal (bar is complete), we just weren't running
                    # Red marker = valid signal, Orange marker = questionable signal
                    append_to_locked_backtest(
                        entry={'date': bt_entry_date, 'price': bt_entry_price, 'position': 'long'},
                        strategy_name=strategy_name, ticker=ticker, missed=False
                    )
                    # Reload locked_backtest to include the new entry
                    locked_backtest = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)

                    print(f"   ✅ AUTO-SYNCED: Now tracking LONG @ ${bt_entry_price:.2f} (entered {str(bt_entry_date)[:16]})")

                    # Generate chart showing the entry
                    sync_chart = None
                    try:
                        sync_backtest = run_historical_backtest(df, config)
                        sync_chart = generate_velocity_chart(df, locked_backtest, config, ticker,
                                                            locked_backtest=locked_backtest,
                                                            full_period_backtest=locked_backtest,
                                                            current_position=trade_state,
                                                            chart_type="signal")
                        print(f"   ✅ Generated sync chart with entry marker")
                    except Exception as chart_err:
                        print(f"   ⚠️ Could not generate sync chart: {chart_err}")

                    # Send Discord notification about the sync with chart
                    sync_msg = (
                        f"🔄 **[{strategy_label}] Auto-Sync - Position Recovered**\n"
                        f"---\n"
                        f"**Position:** LONG @ ${bt_pos.get('entry_price', 0):.2f}\n"
                        f"**Entry:** {str(bt_entry_date)[:16]}\n"
                        f"**Current Price:** ${current_price:,.2f}\n"
                        f"---\n"
                        f"_State was out of sync with backtest. Position recovered automatically._"
                    )
                    try:
                        send_discord_alert(webhook_url, sync_msg, sync_chart, strategy_name=strategy_name)
                    except Exception as discord_err:
                        print(f"   ⚠️ Failed to send sync notification: {discord_err}")

                    # Skip signal detection this cycle - state just synced, wait for next bar
                    print(f"   ⏭️ Skipping signal detection this cycle - just synced state")
                    auto_sync_completed = True  # Mark sync as done
                    # Use smart scheduling to wait for next bar close
                    sync_wait = seconds_until_next_bar_close(ticker, interval) + 5
                    sync_wait = min(sync_wait, 90000 if interval == '1d' else 7200)  # Cap: 25h daily, 2h intraday
                    time.sleep(sync_wait)
                    continue

            # Reverse sync: trade_state has position but backtest shows none (missed exit)
            # CRITICAL: Only run at startup - during live trading, trust trade_state
            # CRITICAL: Also skip if last bar is incomplete - backtest position could flip
            elif not auto_sync_completed and status_backtest and not status_backtest.get('current_position') and trade_state.get('position') and not bar_incomplete_for_sync:
                old_pos = trade_state.get('position', 'long')
                old_entry = trade_state.get('entry_price', 0)
                old_time = trade_state.get('entry_time', '')
                print(f"   🔄 AUTO-SYNC: trade_state shows {old_pos.upper()} @ ${old_entry:.2f}")
                print(f"   🔄 But backtest shows NO position - syncing (missed exit)...")

                # CRITICAL: Verify the entry exists in locked_backtest before syncing exit
                # This prevents orphan exits from being recorded when trade_state is stale
                entry_time_check = str(old_time)[:19] if old_time else ''
                existing_entries = {str(e.get('date', ''))[:19] for e in locked_backtest.get("entries", [])}
                if entry_time_check and entry_time_check not in existing_entries:
                    print(f"   ⚠️  SKIPPING auto-sync: Entry at {entry_time_check} not found in locked_backtest")
                    print(f"      (trade_state may be stale - clearing without recording exit)")
                    # Clear the stale trade_state without recording an orphan exit
                    trade_state = {'position': None, 'entry_price': None, 'entry_time': None, 'last_signal_time': to_chicago_str(datetime.now())}
                    save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)
                    auto_sync_completed = True
                    continue

                # Try to find the exit in backtest to record accurate P&L
                exit_pnl = 0
                exit_reason = "[SYNC] Missed Exit"
                exit_price = current_price
                exit_time = to_chicago_str(datetime.now())

                if status_backtest.get('exits'):
                    # Find the most recent exit that matches our entry
                    for exit_trade in reversed(status_backtest['exits']):
                        if old_entry > 0:
                            exit_entry_price = exit_trade.get('entry_price', 0)
                            if abs(exit_entry_price - old_entry) / old_entry < 0.01:  # 1% tolerance
                                exit_pnl = exit_trade.get('pnl', 0)
                                exit_reason = f"[SYNC] {exit_trade.get('reason', 'Unknown')}"
                                exit_price = exit_trade.get('price', current_price)
                                exit_time = to_chicago_str(exit_trade.get('date', datetime.now()))
                                break

                # Log the closed trade
                log_closed_trade(
                    ticker=ticker,
                    position_type=old_pos,
                    entry_price=old_entry,
                    exit_price=exit_price,
                    entry_time=to_chicago_str(old_time),
                    exit_time=exit_time,
                    exit_reason=exit_reason,
                    pnl_pct=exit_pnl,
                    strategy_name=strategy_name,
                    entry_signal_bar=trade_state.get('entry_signal_bar'),
                    exit_signal_bar=exit_time
                )

                # Also sync the exit to locked_backtest so charts/stats are consistent
                # Use missed=False since this was a real signal, just synced late
                append_to_locked_backtest(
                    exit_trade={
                        'date': exit_time,
                        'price': exit_price,
                        'pnl': exit_pnl,
                        'reason': exit_reason,
                        'entry_price': old_entry,
                        'entry_date': to_chicago_str(old_time),
                    },
                    strategy_name=strategy_name, ticker=ticker, missed=False
                )
                # Reload locked_backtest
                locked_backtest = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)

                # Clear position
                trade_state = {'position': None, 'entry_price': None, 'entry_time': None, 'last_signal_time': exit_time}
                save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)
                print(f"   ✅ AUTO-SYNCED: Position closed. P&L: {exit_pnl:+.2f}%")

                # Generate chart showing the exit
                sync_chart = None
                try:
                    sync_backtest = run_historical_backtest(df, config)
                    sync_chart = generate_velocity_chart(df, locked_backtest, config, ticker,
                                                        locked_backtest=locked_backtest,
                                                        full_period_backtest=locked_backtest,
                                                        chart_type="signal")
                    print(f"   ✅ Generated sync chart with exit marker")
                except Exception as chart_err:
                    print(f"   ⚠️ Could not generate sync chart: {chart_err}")

                # Send Discord notification with chart
                sync_msg = (
                    f"🔄 **[{strategy_label}] Auto-Sync - Missed Exit Recorded**\n"
                    f"---\n"
                    f"**Previous Position:** {old_pos.upper()} @ ${old_entry:.2f}\n"
                    f"**Exit Price:** ${exit_price:.2f}\n"
                    f"**P&L:** {exit_pnl:+.2f}%\n"
                    f"**Reason:** {exit_reason}\n"
                    f"---\n"
                    f"_Position was exited while trader was offline. Trade recorded._"
                )
                try:
                    send_discord_alert(webhook_url, sync_msg, sync_chart, strategy_name=strategy_name)
                except Exception as discord_err:
                    print(f"   ⚠️ Failed to send sync notification: {discord_err}")

                # Skip signal detection this cycle - state just synced, wait for next bar
                print(f"   ⏭️ Skipping signal detection this cycle - just synced state")
                auto_sync_completed = True  # Mark sync as done
                # Use smart scheduling to wait for next bar close
                sync_wait = seconds_until_next_bar_close(ticker, interval) + 5
                sync_wait = min(sync_wait, 90000 if interval == '1d' else 7200)  # Cap: 25h daily, 2h intraday
                time.sleep(sync_wait)
                continue

            # Mark auto-sync as completed after first cycle WITH COMPLETE BAR
            # This prevents backtest flip-flopping from causing issues during live trading
            # CRITICAL: Only mark complete if bar was complete - otherwise defer to next cycle
            if not auto_sync_completed and not bar_incomplete_for_sync:
                auto_sync_completed = True
                print(f"   ✅ Auto-sync check completed (first cycle with complete bar)")

            # Market Open Update (8:30-8:45 AM ET for stocks)
            if "08:30" <= hm <= "08:45" and not is_crypto and not skip_stock_updates:
                key = f"{day_str}_OPEN"
                if key not in daily_alerts and status_backtest:
                    send_status_update(webhook_url, df, status_backtest, config, ticker,
                                     "🔔 Market Open Update", trade_state, strategy_name=strategy_name,
                                     realtime_price=realtime_price)
                    daily_alerts.add(key)

            # Mid-Day Update (11:00-11:15 AM ET for stocks)
            if "11:00" <= hm <= "11:15" and not is_crypto and not skip_stock_updates:
                key = f"{day_str}_MID"
                if key not in daily_alerts and status_backtest:
                    send_status_update(webhook_url, df, status_backtest, config, ticker,
                                     "☀️ Mid-Day Update", trade_state, strategy_name=strategy_name,
                                     realtime_price=realtime_price)
                    daily_alerts.add(key)

            # Afternoon Update (15:00-15:15 PM ET for stocks - 1 hour before close)
            if "15:00" <= hm <= "15:15" and not is_crypto and not skip_stock_updates:
                key = f"{day_str}_AFTERNOON"
                if key not in daily_alerts and status_backtest:
                    send_status_update(webhook_url, df, status_backtest, config, ticker,
                                     "🌅 Afternoon Update", trade_state, strategy_name=strategy_name,
                                     realtime_price=realtime_price)
                    daily_alerts.add(key)

            # Market Close Update (16:00-16:15 ET for stocks - actual market close time)
            if "16:00" <= hm <= "16:15" and not is_crypto and not skip_stock_updates:
                key = f"{day_str}_CLOSE"
                if key not in daily_alerts and status_backtest:
                    send_status_update(webhook_url, df, status_backtest, config, ticker,
                                     "🏁 Market Close Update", trade_state, strategy_name=strategy_name,
                                     realtime_price=realtime_price)
                    daily_alerts.add(key)

            # Hourly Updates (9:00 - 14:00 ET for stocks, except 11 which has special alert)
            # Window is first 15 minutes of each hour to ensure it gets hit
            # Excludes 15 (afternoon) and 16 (close) which have dedicated updates
            if 9 <= hour <= 14 and minute < 15 and not is_crypto and not skip_stock_updates:
                if hour != 11:
                    key = f"{day_str}_HOUR_{hour}"
                    if key not in daily_alerts and status_backtest:
                        send_status_update(webhook_url, df, status_backtest, config, ticker,
                                         f"⏱️ {hour}:00 ET Market Update", trade_state, strategy_name=strategy_name,
                                         realtime_price=realtime_price)
                        daily_alerts.add(key)

            # END OF DAY Update (16:30-17:00 ET for stocks - after market close + data finalization)
            # This is the ACTUAL end of day summary, sent after market closes and data settles
            if "16:30" <= hm <= "17:00" and not is_crypto and not skip_stock_updates:
                key = f"{day_str}_EOD"
                print(f"   📊 EOD window detected! Time={hm}, Key={key}, already_sent={key in daily_alerts}, has_backtest={status_backtest is not None}")
                if key not in daily_alerts and status_backtest:
                    send_status_update(webhook_url, df, status_backtest, config, ticker,
                                     "📊 End of Day Summary", trade_state, strategy_name=strategy_name,
                                     realtime_price=realtime_price)
                    daily_alerts.add(key)
                    print(f"   ✅ End of Day summary sent!")
                elif key in daily_alerts:
                    print(f"   ⏭️ EOD update already sent for {day_str}")
                elif not status_backtest:
                    print(f"   ⚠️ EOD update skipped - no backtest data")

            # Midnight Update (for both crypto and stocks - crypto uses local time, stocks use ET)
            if hour == 0 and minute < 5:
                key = f"{day_str}_MIDNIGHT"
                print(f"   🕛 Midnight window detected! Key: {key}, already_sent: {key in daily_alerts}, has_backtest: {status_backtest is not None}")
                if key not in daily_alerts and status_backtest:
                    send_status_update(webhook_url, df, status_backtest, config, ticker,
                                     "🌙 Midnight Update", trade_state, strategy_name=strategy_name,
                                     realtime_price=realtime_price)
                    daily_alerts.add(key)
                    print(f"   ✅ Midnight update sent!")
                elif key in daily_alerts:
                    print(f"   ⏭️ Midnight update already sent for {day_str}")
                elif not status_backtest:
                    print(f"   ⚠️ Midnight update skipped - no backtest data")

            # Crypto: 6-hourly updates (00:00, 06:00, 12:00, 18:00 local time)
            # Extended window to 45 minutes to avoid missing updates (e.g., trader started late)
            if is_crypto and hour in [0, 6, 12, 18] and minute < 45:
                key = f"{day_str}_CRYPTO_{hour:02d}"
                if key not in daily_alerts and status_backtest:
                    update_names = {0: "🌙 Midnight", 6: "🌅 Morning", 12: "☀️ Midday", 18: "🌆 Evening"}
                    send_status_update(webhook_url, df, status_backtest, config, ticker,
                                     f"{update_names[hour]} Update", trade_state, strategy_name=strategy_name,
                                     realtime_price=realtime_price)
                    daily_alerts.add(key)
                    print(f"   ✅ Crypto {hour}:00 update sent!")

            # INTRADAY (15m, 1h, etc.): Status updates at key market times
            # Entry/exit SIGNALS are posted immediately - these are STATUS updates only
            if is_intraday and not is_crypto:
                for start_time, end_time, title in intraday_update_windows:
                    if start_time <= hm <= end_time:
                        key = f"{day_str}_INTRADAY_{start_time.replace(':', '')}"
                        if key not in daily_alerts and status_backtest:
                            send_status_update(webhook_url, df, status_backtest, config, ticker,
                                             title, trade_state, strategy_name=strategy_name,
                                             realtime_price=realtime_price)
                            daily_alerts.add(key)
                            print(f"   ✅ Intraday {title} sent!")
                        break  # Only one update per cycle

            # Check for exit conditions first (if in position)
            if trade_state.get('position') == 'long':
                entry_price = trade_state.get('entry_price')
                # CRITICAL: Validate entry_price before calculations to prevent crash
                if not entry_price or entry_price <= 0:
                    print(f"⚠️  CRITICAL: Invalid entry_price ({entry_price}) in trade_state - skipping exit check")
                    print(f"   State file may be corrupted. Please check: {get_state_file_path(strategy_name=strategy_name, ticker=ticker)}")
                    continue
                pnl_pct = ((current_price - entry_price) / entry_price) * 100

                # v9: Use entry regime's params for exit decisions
                entry_regime_name = trade_state.get('entry_regime')  # None for v8
                def _exit_p(key, default_val):
                    if entry_regime_name and config.get('regime_aware', False):
                        r_cfg = config.get('regime_params', {}).get(entry_regime_name, {})
                        return r_cfg.get(key, config.get(key, default_val))
                    return config.get(key, default_val)

                live_stop_loss = _exit_p('stop_loss_pct', stop_loss_pct)
                live_take_profit = _exit_p('take_profit_pct', take_profit_pct)

                exit_reason = None
                exit_bar_close_time = None  # Will be set for signal-based exits
                signal_bar_time = None  # Will be set when checking signal-based exits
                can_check_signal_exits = False  # Will be set True when checking signal-based exits

                # Stop loss - uses real-time price (can trigger anytime)
                if pnl_pct <= -live_stop_loss:
                    exit_reason = f"Stop Loss ({pnl_pct:.2f}%)"
                # Take profit - uses real-time price (can trigger anytime)
                elif pnl_pct >= live_take_profit:
                    exit_reason = f"Take Profit ({pnl_pct:.2f}%)"
                # Signal-based exits (opposite signal, midline cross)
                # For daily strategies: use COMPLETED bar (df.iloc[-2]) to prevent repainting
                # For stocks (SPY): only check after market close (4 PM ET)
                # For crypto (BTC): only check after midnight UTC (00:00 UTC)
                else:
                    # Determine which bar to use for signal-based exits
                    if interval == "1d" and len(df) >= 2:
                        signal_bar = df.iloc[-2]  # Completed bar for daily
                        signal_bar_time = df.index[-2]

                        # Check if daily bar is complete before evaluating signal exits
                        can_check_signal_exits = True
                        if is_crypto:
                            # Crypto (BTC): Daily bar closes at 00:00 UTC
                            # Only evaluate after new bar starts (wait 30 min buffer for data)
                            utc_now = datetime.now(timezone.utc)
                            utc_hour = utc_now.hour
                            utc_minute = utc_now.minute
                            # Check between 00:30 UTC and 23:59 UTC (avoid checking right at midnight)
                            if utc_hour == 0 and utc_minute < 30:
                                can_check_signal_exits = False
                                print(f"   ⏳ [CRYPTO] Waiting for new daily bar (UTC: {utc_now.strftime('%H:%M')})")
                        else:
                            # Stocks/Futures: Use is_bar_complete() which handles different market types
                            # Stocks (SPY): closes 4 PM ET, Futures (ES=F): closes 5 PM CT
                            if not is_bar_complete(ticker, interval):
                                can_check_signal_exits = False
                    else:
                        # For intraday intervals, check if last bar is complete
                        # CRITICAL: Only use COMPLETED bars for signal-based exits to prevent repainting
                        bar_is_incomplete = is_last_bar_incomplete(df, interval, ticker)

                        if bar_is_incomplete and len(df) >= 2:
                            # Last bar still forming - use completed bar for signal exits
                            signal_bar = df.iloc[-2]
                            signal_bar_time = df.index[-2]
                            can_check_signal_exits = True
                            # Note: Stop-loss and take-profit still use real-time price (above)
                        else:
                            # Last bar is complete - use it
                            signal_bar = latest
                            signal_bar_time = df.index[-1]  # Use latest bar's time
                            can_check_signal_exits = True

                    if can_check_signal_exits:
                        # Acceleration Reversal Exit (early warning before stop loss)
                        # Check if enabled and we have acceleration data
                        # v9: use entry regime's params for exit decisions
                        live_use_accel = _exit_p('use_accel_exit', False)
                        if live_use_accel and 'acceleration' in df.columns:
                            live_accel_type = _exit_p('accel_exit_type', 'sign_reversal')
                            live_accel_threshold = _exit_p('accel_exit_threshold', 0.0)
                            live_accel_min_pnl = _exit_p('accel_exit_min_pnl', 0.5)
                            live_accel_lookback = _exit_p('accel_exit_lookback', 1)
                            live_jerk_confirm = _exit_p('use_jerk_confirm', False)
                            live_jerk_threshold = _exit_p('jerk_confirm_threshold', 0.0)

                            # Check if conditions are met
                            pnl_ok = pnl_pct >= live_accel_min_pnl or pnl_pct < 0
                            if pnl_ok:
                                # Get index of signal_bar in df
                                bar_idx = df.index.get_loc(signal_bar_time) if signal_bar_time in df.index else -1
                                if bar_idx >= live_accel_lookback:
                                    accel_values = df['acceleration'].iloc[bar_idx-live_accel_lookback+1:bar_idx+1].values
                                    current_accel = df['acceleration'].iloc[bar_idx]
                                    # For LONG positions: negative acceleration is bearish
                                    accel_cond = False
                                    if live_accel_type == 'sign_reversal':
                                        accel_cond = all(a < 0 for a in accel_values)
                                    elif live_accel_type == 'magnitude':
                                        accel_cond = current_accel < -live_accel_threshold
                                    elif live_accel_type == 'both':
                                        accel_cond = all(a < 0 for a in accel_values) and abs(current_accel) > live_accel_threshold
                                    # Jerk confirmation (optional)
                                    jerk_cond = True
                                    if live_jerk_confirm and 'jerk' in df.columns and live_jerk_threshold > 0:
                                        current_jerk = df['jerk'].iloc[bar_idx]
                                        jerk_cond = current_jerk < -live_jerk_threshold
                                    if accel_cond and jerk_cond:
                                        exit_reason = f"Accel Reversal ({pnl_pct:.2f}%)"
                                        interval_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}.get(interval, 15)
                                        exit_bar_close_time = pd.to_datetime(signal_bar_time) + pd.Timedelta(minutes=interval_minutes)

                        # Exit on opposite signal (v9: use entry regime's params)
                        live_exit_opposite = _exit_p('exit_on_opposite_signal', exit_on_opposite_signal)
                        live_exit_midline = _exit_p('exit_on_midline_cross', exit_on_midline_cross)
                        if not exit_reason and live_exit_opposite and signal_bar['sell_signal']:
                            exit_reason = f"Opposite Signal ({pnl_pct:.2f}%)"
                            # Calculate bar close time for signal-based exit
                            interval_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}.get(interval, 15)
                            exit_bar_close_time = pd.to_datetime(signal_bar_time) + pd.Timedelta(minutes=interval_minutes)
                        # Exit on midline cross (oscillator crosses above 0 = bearish for long)
                        elif not exit_reason and live_exit_midline and signal_bar['osc_smooth'] > 0:
                            exit_reason = f"Midline Cross ({pnl_pct:.2f}%)"
                            # Calculate bar close time for signal-based exit
                            interval_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}.get(interval, 15)
                            exit_bar_close_time = pd.to_datetime(signal_bar_time) + pd.Timedelta(minutes=interval_minutes)

                # For SL/TP exits, use real-time; for signal exits, use bar close time
                # Safety: check for NaT (can occur if signal_bar_time is None/invalid)
                if exit_bar_close_time is not None and not pd.isna(exit_bar_close_time):
                    exit_timestamp = exit_bar_close_time
                else:
                    exit_timestamp = current_time

                # For CHART MARKERS in locked_backtest: use signal_bar_time (bar START) not bar CLOSE
                # This is consistent with run_historical_backtest which uses bar_start for daily data
                # Chart indexes by bar START time, so exit must use bar START time for correct marker placement
                if can_check_signal_exits and signal_bar_time is not None:
                    locked_backtest_exit_date = signal_bar_time  # Bar START time for chart markers
                else:
                    locked_backtest_exit_date = exit_timestamp  # Fallback for SL/TP (real-time)

                if exit_reason:
                    # CRITICAL: Validate with append_to_locked_backtest FIRST before ANY Discord
                    # This prevents spam if the exit is rejected (e.g., orphan exit, timestamp mismatch)
                    entry_time_for_exit = trade_state.get('entry_time', '')

                    # Pre-validation: Check if this exit would be accepted
                    append_success = append_to_locked_backtest(
                        exit_trade={
                            'date': locked_backtest_exit_date,  # Bar START time for chart markers
                            'price': current_price,
                            'pnl': pnl_pct,
                            'reason': exit_reason,
                            'entry_price': entry_price,
                            'entry_date': entry_time_for_exit
                        },
                        strategy_name=strategy_name, ticker=ticker
                    )
                    if not append_success:
                        print(f"   ❌ EXIT SIGNAL REJECTED by locked_backtest - NOT sending Discord or updating trade_state")
                        print(f"   ❌ This prevents desync. trade_state still shows position: {trade_state.get('position')}")
                        continue  # Skip ALL Discord, trade_state update, and wait for next cycle

                    # Validation passed - NOW send immediate alert
                    pnl_emoji_quick = "✅" if pnl_pct > 0 else "❌"
                    quick_exit_msg = (
                        f"@here\n"
                        f"📤 **[{strategy_label}] LONG EXIT** {pnl_emoji_quick}\n"
                        f"**Reason:** {exit_reason}\n"
                        f"**Exit Price:** ${current_price:,.2f}\n"
                        f"💰 **P&L:** {pnl_pct:+.2f}%\n"
                        f"_Chart and full stats loading..._"
                    )
                    send_discord_alert(webhook_url, quick_exit_msg, strategy_name=strategy_name)
                    print(f"⚡ IMMEDIATE EXIT ALERT SENT! {exit_reason}")

                    # Generate chart and get stats for exit (can be slow)
                    exit_chart = None
                    exit_backtest = None
                    try:
                        exit_backtest = run_historical_backtest(df, config)

                        # Create a copy of locked_backtest with the current exit included
                        # so it shows on the chart (before we officially append it)
                        import copy
                        locked_backtest_with_exit = copy.deepcopy(locked_backtest) if locked_backtest else {'entries': [], 'exits': []}
                        if 'exits' not in locked_backtest_with_exit:
                            locked_backtest_with_exit['exits'] = []
                        locked_backtest_with_exit['exits'].append({
                            'date': locked_backtest_exit_date,  # Bar START time for chart markers
                            'price': current_price,
                            'pnl': pnl_pct,
                            'reason': exit_reason,
                            'entry_price': entry_price,
                            'entry_date': str(trade_state.get('entry_time', '')),
                        })
                        # Clear current_position since we're exiting
                        locked_backtest_with_exit['current_position'] = None

                        # Use locked backtest WITH current exit for markers
                        # chart_type="signal" for focused view on exit signals
                        exit_chart = generate_velocity_chart(df, locked_backtest_with_exit, config, ticker,
                                                            locked_backtest=locked_backtest_with_exit,
                                                            full_period_backtest=locked_backtest_with_exit,
                                                            chart_type="signal")
                    except Exception as e:
                        print(f"Could not generate exit chart: {e}")

                    # Build stats section - use locked_backtest_with_exit for stats (includes this exit)
                    stats_section = ""
                    stats_src = locked_backtest_with_exit if locked_backtest_with_exit and locked_backtest_with_exit.get('exits') else exit_backtest
                    if stats_src:
                        # Calculate stats from locked_backtest_with_exit exits
                        if stats_src == locked_backtest_with_exit:
                            from velocity_live_trader import calculate_exit_stats
                            calc_stats = calculate_exit_stats(stats_src.get('exits', []), exclude_missed=True)
                            stats_section = (
                                f"---\n"
                                f"📊 **Strategy Stats:**\n"
                                f"• Trades: {calc_stats['num_trades']} | Win Rate: {calc_stats['win_rate']:.0f}%\n"
                                f"• Total Return: {calc_stats['total_return']:.1f}% | PF: {calc_stats['profit_factor']:.1f}\n"
                            )
                        else:
                            stats_section = (
                                f"---\n"
                                f"📊 **Strategy Stats:**\n"
                                f"• Trades: {stats_src['num_trades']} | Win Rate: {stats_src['win_rate']:.0f}%\n"
                                f"• Total Return: {stats_src['total_return']:.1f}% | PF: {stats_src['profit_factor']:.1f}\n"
                            )

                    pnl_emoji = "✅" if pnl_pct > 0 else "❌"

                    # Calculate hold duration
                    entry_time_str = trade_state.get('entry_time', '')
                    hold_duration = "N/A"
                    if entry_time_str:
                        try:
                            entry_dt = datetime.strptime(entry_time_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
                            # Use exit_timestamp (bar close time) for duration calculation
                            exit_dt = exit_timestamp if isinstance(exit_timestamp, datetime) else current_time
                            duration = exit_dt - entry_dt
                            days = duration.days
                            hours = duration.seconds // 3600
                            if days > 0:
                                hold_duration = f"{days}d {hours}h"
                            else:
                                hold_duration = f"{hours}h {(duration.seconds % 3600) // 60}m"
                        except Exception:  # Catch all non-system exceptions
                            hold_duration = "N/A"

                    # Calculate actual dollar P&L per unit
                    pnl_dollars = current_price - entry_price  # LONG: profit when price goes up

                    # Format timestamps for display - show BOTH UTC and Chicago time (12-hour format)
                    try:
                        entry_dt = pd.to_datetime(entry_time_str)
                        entry_utc = entry_dt.tz_localize('UTC') if entry_dt.tzinfo is None else entry_dt.tz_convert('UTC')
                        entry_ct = to_chicago_time(entry_time_str)
                        entry_utc_str = entry_utc.strftime('%Y-%m-%d %H:%M UTC')
                        entry_ct_str = entry_ct.strftime('%Y-%m-%d %I:%M %p CT') if entry_ct else ''
                        entry_time_display = f"{entry_utc_str} ({entry_ct_str})" if entry_ct_str else entry_utc_str
                    except Exception:
                        entry_time_display = entry_time_str[:16] if entry_time_str else 'N/A'

                    try:
                        exit_dt = pd.to_datetime(exit_timestamp)
                        exit_utc = exit_dt.tz_localize('UTC') if exit_dt.tzinfo is None else exit_dt.tz_convert('UTC')
                        exit_ct = to_chicago_time(exit_timestamp)
                        exit_utc_str = exit_utc.strftime('%Y-%m-%d %H:%M UTC')
                        exit_ct_str = exit_ct.strftime('%Y-%m-%d %I:%M %p CT') if exit_ct else ''
                        exit_time_display = f"{exit_utc_str} ({exit_ct_str})" if exit_ct_str else exit_utc_str
                    except Exception:
                        exit_time_display = str(exit_timestamp)[:16] if exit_timestamp else 'N/A'

                    exit_msg = (
                        f"@here\n"
                        f"📤 **[{strategy_label}] LONG EXIT** {pnl_emoji}\n"
                        f"**Reason:** {exit_reason}\n"
                        f"---\n"
                        f"📅 **Entry:** {entry_time_display} @ ${entry_price:.2f}\n"
                        f"📅 **Exit:** {exit_time_display} @ ${current_price:.2f}\n"
                        f"⏱️ **Hold Duration:** {hold_duration}\n"
                        f"---\n"
                        f"💰 **P&L:** {pnl_pct:+.2f}% (${pnl_dollars:+,.2f}/unit)\n"
                        f"{stats_section}"
                    )

                    # Log closed trade and get cumulative stats
                    # Format exit timestamp for logging (Chicago time)
                    exit_ct = to_chicago_time(exit_timestamp)
                    exit_time_for_log = to_chicago_str(exit_ct) if exit_ct else str(exit_timestamp)
                    cumulative = log_closed_trade(
                        ticker=ticker,
                        position_type="LONG",
                        entry_price=entry_price,
                        exit_price=current_price,
                        entry_time=entry_time_str,
                        exit_time=exit_time_for_log,
                        exit_reason=exit_reason,
                        pnl_pct=pnl_pct,
                        strategy_name=strategy_name,
                        entry_signal_bar=trade_state.get('entry_signal_bar'),  # Locked entry signal bar
                        exit_signal_bar=str(signal_bar_time) if can_check_signal_exits and signal_bar_time else str(current_time)  # Signal bar for signal exits
                    )

                    # Append to locked_backtest already done at start of exit_reason block
                    # (validation happens BEFORE any Discord is sent to prevent spam)

                    # Reload the locked backtest with the new exit
                    locked_backtest = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)

                    # Add cumulative stats to message
                    cumulative_section = (
                        f"---\n"
                        f"📈 **Cumulative Performance:**\n"
                        f"• Trades: {cumulative['total_trades']} ({cumulative['winners']}W / {cumulative['losers']}L)\n"
                        f"• Win Rate: {cumulative['win_rate']:.0f}%\n"
                        f"• Total P&L: {cumulative['total_pnl_pct']:+.2f}% (${cumulative['total_pnl_dollars']:+,.0f})\n"
                    )

                    exit_msg += cumulative_section

                    # Generate trade log CSV (use locked_backtest_with_exit to include this trade)
                    exit_trade_log_csv = None
                    try:
                        exit_trade_log_csv = generate_trade_log_csv(locked_backtest_with_exit, num_trades=10)
                    except Exception as e:
                        print(f"⚠️ Could not generate exit trade log CSV: {e}")

                    send_discord_alert(webhook_url, exit_msg, exit_chart, strategy_name=strategy_name, csv_buf=exit_trade_log_csv)
                    print(f"EXIT LONG: {exit_reason}")

                    # Sierra Chart bridge (opt-in, fail-safe)
                    if SIERRA_BRIDGE_ENABLED:
                        try:
                            sierra_signal = build_sierra_signal(
                                signal_type="EXIT",
                                direction="LONG",
                                ticker=ticker,
                                price=current_price,
                                signal_time=current_time,
                                strategy_name=strategy_name,
                                config=config,
                                exit_data={
                                    "entry_price": entry_price,
                                    "pnl_pct": pnl_pct,
                                    "exit_reason": exit_reason
                                }
                            )
                            publish_signal_to_sierra(sierra_signal, SIERRA_SIGNAL_FILE)
                            print(f"   📊 Sierra Chart: Published EXIT signal")
                        except Exception as e:
                            print(f"   ⚠️ Sierra Chart bridge error (Discord unaffected): {e}")

                    trade_state['position'] = None
                    trade_state['entry_price'] = None
                    trade_state['entry_time'] = None
                    trade_state['entry_signal_bar'] = None  # Clear signal bar
                    trade_state['entry_regime'] = None  # Clear v9 regime
                    save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)

                    # CRITICAL: Set flag to prevent immediate re-entry on buy signal from previous bar
                    # After exiting on "Opposite Signal", we should NOT re-enter on the same cycle
                    just_exited_this_cycle = True

            # NOTE: SHORT positions are not supported in this LONG-only strategy
            # The backtest engine only tracks LONG positions (position=1 or position=0)

            # Initialize variables that may be used later (even if we skip signal checking)
            should_check_signals = False
            completed_bar_time = None

            # Check for new entry signals (only if not in position AND didn't just exit)
            # CRITICAL: just_exited_this_cycle prevents flip-flopping where exit on opposite signal
            # is immediately followed by entry on a buy signal from a previous bar
            if trade_state.get('position') is None and not just_exited_this_cycle:
                # CRITICAL: Only evaluate signals on COMPLETED bars to prevent repainting
                # Use is_bar_complete() which knows when each asset type's bar closes:
                #   - Stocks (SPY): 4:00 PM ET
                #   - Futures (ES=F): 5:00 PM CT
                #   - Crypto (BTC): 00:00 UTC

                market_time = get_market_time()  # Returns Eastern Time
                should_check_signals = False
                completed_bar = None
                completed_bar_time = None
                bar_is_incomplete = False

                if interval == "1d":
                    # DAILY BARS: Check if the bar is complete based on asset's close time
                    if len(df) >= 2:
                        completed_bar = df.iloc[-2]
                        completed_bar_time = df.index[-2]

                        # Check if we've already evaluated this bar
                        if timestamps_equal(completed_bar_time, last_signal_bar_evaluated):
                            pass  # Already evaluated this bar
                        # Check for holidays/weekends (stocks only)
                        elif not is_crypto:
                            is_holiday_today, holiday_name = is_market_holiday(market_time.date(), ticker)
                            if is_holiday_today:
                                print(f"   🏖️ [{ticker}] Market closed for {holiday_name}")
                            elif market_time.weekday() >= 5:
                                print(f"   📅 [{ticker}] Weekend - Market closed")
                            elif is_bar_complete(ticker, interval):
                                # Bar is complete - ready to generate signal
                                should_check_signals = True
                                market_type = get_ticker_market_type(ticker)
                                mh_config = MARKET_HOURS_CONFIG.get(market_type, {})
                                tz_name = mh_config.get('timezone', 'UTC').split('/')[-1]
                                print(f"   📊 [{ticker}] Daily bar complete ({tz_name}) - evaluating: {completed_bar_time}")
                            else:
                                # Bar not complete yet - waiting for close
                                market_type = get_ticker_market_type(ticker)
                                mh_config = MARKET_HOURS_CONFIG.get(market_type, {})
                                close_time = mh_config.get('daily_bar_close', {'hour': 16, 'minute': 0})
                                tz_name = mh_config.get('timezone', 'UTC').split('/')[-1]
                                print(f"   ⏳ [{ticker}] Waiting for bar close ({close_time['hour']:02d}:{close_time['minute']:02d} {tz_name})")
                        else:
                            # Crypto - check if bar is complete (after midnight UTC)
                            if is_bar_complete(ticker, interval):
                                should_check_signals = True
                                utc_now = datetime.now(timezone.utc)
                                print(f"   📊 [{ticker}] Daily bar complete (UTC: {utc_now.strftime('%H:%M')}) - evaluating: {completed_bar_time}")
                            else:
                                utc_now = datetime.now(timezone.utc)
                                print(f"   ⏳ [{ticker}] Waiting for daily bar close (00:00 UTC, now: {utc_now.strftime('%H:%M')})")
                else:
                    # INTRADAY BARS: Check if the last bar in the data is complete
                    # Use the bar's start time to calculate when it closes
                    if len(df) >= 1:
                        last_bar_time = df.index[-1]
                        bar_is_incomplete = not is_bar_complete(ticker, interval, last_bar_time)

                        if bar_is_incomplete:
                            # Last bar still forming - use second-to-last bar
                            if len(df) >= 2:
                                completed_bar = df.iloc[-2]
                                completed_bar_time = df.index[-2]
                                should_check_signals = True
                                print(f"   ⏳ [{ticker}] Current bar forming - using completed bar: {completed_bar_time}")
                            else:
                                print(f"   ⚠️ [{ticker}] Not enough bars to evaluate")
                        else:
                            # Last bar is complete - use it
                            completed_bar = df.iloc[-1]
                            completed_bar_time = df.index[-1]
                            should_check_signals = True
                            print(f"   📊 [{ticker}] Bar complete - evaluating: {completed_bar_time}")

                recent_buy_signal = None
                recent_sell_signal = None

                # v9 helper: check regime-aware buy signal for a bar at index bar_iloc
                def _has_regime_buy(bar_iloc_idx):
                    """Check if bar has a buy signal, considering regime if v9."""
                    if live_regimes is None:
                        return bar_iloc_idx >= 0 and df.iloc[bar_iloc_idx].get('buy_signal', False)
                    # v9: get regime for this bar
                    abs_idx = len(df) + bar_iloc_idx if bar_iloc_idx < 0 else bar_iloc_idx
                    if abs_idx < 0 or abs_idx >= len(live_regimes):
                        return False
                    bar_regime_id = int(live_regimes[abs_idx])
                    bar_regime_name = REGIME_NAMES.get(bar_regime_id, 'chop')
                    # Check high-vol suppression
                    suppress_hv = config.get('regime_detector', {}).get('suppress_entries_high_vol', True)
                    if suppress_hv and live_is_high_vol is not None and live_is_high_vol[abs_idx]:
                        return False
                    # Check regime-specific signal
                    if bar_regime_name not in live_regime_signal_dfs:
                        return False  # dont_trade regime
                    return bool(live_regime_signal_dfs[bar_regime_name].iloc[abs_idx].get('buy_signal', False))

                def _get_bar_regime_name(bar_iloc_idx):
                    """Get regime name for a bar index."""
                    if live_regimes is None:
                        return None
                    abs_idx = len(df) + bar_iloc_idx if bar_iloc_idx < 0 else bar_iloc_idx
                    if abs_idx < 0 or abs_idx >= len(live_regimes):
                        return None
                    return REGIME_NAMES.get(int(live_regimes[abs_idx]), 'unknown')

                if should_check_signals and interval == "1d" and len(df) >= 2:
                    # Only check the COMPLETED bar for daily strategies
                    bar = completed_bar
                    bar_time = completed_bar_time
                    last_signal_time = trade_state.get('last_signal_time')

                    # Skip if already processed this bar (use safe timestamp comparison)
                    if not (last_signal_time and timestamps_equal(bar_time, last_signal_time)):
                        if _has_regime_buy(-2) if live_regimes is not None else bar['buy_signal']:
                            recent_buy_signal = {'bar': bar, 'time': bar_time, 'index': -2,
                                                 'regime': _get_bar_regime_name(-2)}
                            print(f"   ✅ BUY signal on completed bar {bar_time}" +
                                  (f" [{recent_buy_signal['regime'].upper()}]" if recent_buy_signal.get('regime') else ""))
                        if bar['sell_signal']:
                            recent_sell_signal = {'bar': bar, 'time': bar_time, 'index': -2}
                            print(f"   ✅ SELL signal on completed bar {bar_time}")

                    # Mark this bar as evaluated
                    last_signal_bar_evaluated = str(completed_bar_time)

                elif should_check_signals and interval != "1d":
                    # Non-daily: use lookback logic starting from COMPLETED bars only
                    # CRITICAL: Skip incomplete bar to prevent repainting
                    lookback_bars = 3
                    start_offset = 2 if bar_is_incomplete else 1  # Skip incomplete bar if present

                    for i in range(start_offset, min(lookback_bars + start_offset, len(df))):
                        bar = df.iloc[-i]
                        bar_time = df.index[-i]
                        last_signal_time = trade_state.get('last_signal_time')

                        # Skip if already processed this bar (use safe timestamp comparison)
                        if last_signal_time and timestamps_equal(bar_time, last_signal_time):
                            continue

                        if recent_buy_signal is None:
                            if _has_regime_buy(-i) if live_regimes is not None else bar['buy_signal']:
                                recent_buy_signal = {'bar': bar, 'time': bar_time, 'index': -i,
                                                     'regime': _get_bar_regime_name(-i)}
                        if bar['sell_signal'] and recent_sell_signal is None:
                            recent_sell_signal = {'bar': bar, 'time': bar_time, 'index': -i}

                # Use most recent signal (prefer current bar, then recent bars)
                # CRITICAL: Only consider buy signals if NOT already in a position
                # CRITICAL: Only consider sell signals if IN a position
                already_in_position = trade_state.get('position') is not None
                effective_buy = recent_buy_signal is not None and not already_in_position
                effective_sell = recent_sell_signal is not None and already_in_position

                # Log recent signals found - ONLY if they're actionable given our position state
                # Don't warn about SELL signals when we have no position (not actionable)
                # Don't warn about BUY signals when we're already in position (not actionable)
                if recent_buy_signal and recent_buy_signal['index'] < -1 and not already_in_position:
                    print(f"⚠️  Found missed BUY signal from {recent_buy_signal['time']}")
                if recent_sell_signal and recent_sell_signal['index'] < -1 and already_in_position:
                    print(f"⚠️  Found missed SELL signal from {recent_sell_signal['time']}")

                # Avoid duplicate/stale signals
                # Skip signals that are AT OR BEFORE the last processed signal time
                last_signal_time = trade_state.get('last_signal_time')

                # FALLBACK: If no last_signal_time, use latest timestamp from locked_backtest
                # This prevents re-processing old signals that are already in locked_backtest
                if not last_signal_time and locked_backtest:
                    latest_entry = locked_backtest.get('entries', [])[-1] if locked_backtest.get('entries') else None
                    latest_exit = locked_backtest.get('exits', [])[-1] if locked_backtest.get('exits') else None
                    fallback_times = []
                    if latest_entry:
                        fallback_times.append(pd.to_datetime(latest_entry['date']))
                    if latest_exit:
                        fallback_times.append(pd.to_datetime(latest_exit['date']))
                    if fallback_times:
                        last_signal_time = str(max(fallback_times))
                        print(f"   ℹ️  Using locked_backtest fallback: last_signal_time = {last_signal_time}")

                if last_signal_time:
                    last_signal_dt = normalize_tz(pd.to_datetime(last_signal_time))
                    if recent_buy_signal:
                        buy_signal_dt = normalize_tz(pd.to_datetime(recent_buy_signal['time']))
                        # Use < instead of <= to allow processing signals at same timestamp
                        # (position-checking code prevents duplicate entries)
                        if buy_signal_dt < last_signal_dt:
                            effective_buy = False
                            print(f"   ⏭️  Skipping stale BUY signal from {recent_buy_signal['time']} (last signal was {last_signal_time})")
                    if recent_sell_signal:
                        sell_signal_dt = normalize_tz(pd.to_datetime(recent_sell_signal['time']))
                        # Use < instead of <= to allow processing signals at same timestamp
                        if sell_signal_dt < last_signal_dt:
                            effective_sell = False
                            print(f"   ⏭️  Skipping stale SELL signal from {recent_sell_signal['time']} (last signal was {last_signal_time})")

                if effective_buy:
                    signal_bar = recent_buy_signal['bar']
                    signal_time_utc = recent_buy_signal['time']  # Bar start time in UTC
                    is_delayed = recent_buy_signal['index'] < -1

                    # Calculate bar close time (when signal fires and entry happens)
                    interval_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}.get(interval, 15)
                    bar_close_time = pd.to_datetime(signal_time_utc) + pd.Timedelta(minutes=interval_minutes)

                    # Convert bar CLOSE time to display format
                    # Show BOTH UTC and Chicago time for clarity (especially for daily bars where midnight UTC = 6 PM CT)
                    try:
                        bar_close_utc = pd.to_datetime(bar_close_time).tz_localize('UTC') if bar_close_time.tzinfo is None else pd.to_datetime(bar_close_time).tz_convert('UTC')
                        signal_ct = to_chicago_time(bar_close_time)
                        utc_str = bar_close_utc.strftime('%Y-%m-%d %H:%M UTC')
                        ct_str = signal_ct.strftime('%Y-%m-%d %I:%M %p CT') if signal_ct else ''
                        signal_time_display = f"{utc_str} ({ct_str})" if ct_str else utc_str
                    except Exception:
                        signal_time_display = str(bar_close_time)[:16]

                    signal_note = f" (Signal from {signal_time_display} - entering at market)" if is_delayed else ""

                    # Append new entry to locked backtest (for future charts)
                    # IMPORTANT: Use signal_time_utc (bar START time) for chart markers, NOT bar_close_time
                    # This is consistent with run_historical_backtest which uses bar_start for daily data
                    # Chart indexes by bar START time, so entry must use bar START time for correct marker placement
                    # Live trading signals are NEVER missed - we're acting on them!
                    # CRITICAL: Check return value - if False, DON'T send Discord or update trade_state
                    # This prevents desync between locked_backtest and trade_state
                    append_success = append_to_locked_backtest(
                        entry={'date': signal_time_utc, 'price': current_price, 'position': 'long'},
                        strategy_name=strategy_name, ticker=ticker, missed=False
                    )
                    if not append_success:
                        print(f"   ❌ ENTRY SIGNAL REJECTED by locked_backtest - NOT sending Discord or updating trade_state")
                        print(f"   ❌ This prevents desync. Waiting for next cycle...")
                        continue  # Skip Discord, trade_state update, and wait for next cycle

                    # Reload the locked backtest with the new entry
                    locked_backtest = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)

                    # IMMEDIATE ALERT: Send quick text notification FIRST before slow chart generation
                    # This ensures traders get the signal within seconds of bar close
                    # v9: use entry regime's SL/TP for display
                    entry_regime_for_msg = recent_buy_signal.get('regime') if recent_buy_signal else None
                    if entry_regime_for_msg and config.get('regime_aware', False):
                        r_cfg = config.get('regime_params', {}).get(entry_regime_for_msg, {})
                        display_sl = r_cfg.get('stop_loss_pct', stop_loss_pct)
                        display_tp = r_cfg.get('take_profit_pct', take_profit_pct)
                    else:
                        display_sl = stop_loss_pct
                        display_tp = take_profit_pct
                    entry_sl_price_quick = current_price * (1 - display_sl / 100)
                    entry_tp_price_quick = current_price * (1 + display_tp / 100)

                    # Use consistent position status (respect is_delayed like the full message does)
                    quick_position_status = "🟢 **Position: LONG**" if not is_delayed else "🟡 **Position: LONG** (late entry)"
                    regime_line = f"**Regime:** {entry_regime_for_msg.upper()}\n" if entry_regime_for_msg else ""

                    quick_buy_msg = (
                        f"@here\n"
                        f"📈 **[{strategy_label}] BUY SIGNAL**{signal_note}\n"
                        f"**Signal Time:** {signal_time_display}\n"
                        f"**Entry Price:** ${current_price:,.2f}\n"
                        f"{regime_line}"
                        f"{quick_position_status}\n"
                        f"---\n"
                        f"_SL: {display_sl:.1f}% (${entry_sl_price_quick:,.2f}) | TP: {display_tp:.1f}% (${entry_tp_price_quick:,.2f})_\n"
                        f"_Chart and stats loading..._"
                    )
                    send_discord_alert(webhook_url, quick_buy_msg, strategy_name=strategy_name)
                    print(f"⚡ IMMEDIATE BUY ALERT SENT!{signal_note}")

                    # Now generate chart and stats (can be slow for large backtests)
                    signal_backtest = None
                    signal_chart = None
                    try:
                        signal_backtest = run_historical_backtest(df, config)
                        # Use locked backtest for markers and stats to prevent repainting
                        # chart_type="signal" for focused view on entry signals
                        signal_chart = generate_velocity_chart(df, locked_backtest, config, ticker,
                                                              locked_backtest=locked_backtest,
                                                              full_period_backtest=locked_backtest,
                                                              chart_type="signal")
                    except Exception as e:
                        print(f"Could not generate signal chart: {e}")

                    # Build comprehensive buy message with stats - use locked_backtest to prevent repainting
                    stats_section = ""
                    stats_src = locked_backtest if locked_backtest and locked_backtest.get('num_trades') else signal_backtest
                    if stats_src:
                        stats_section = (
                            f"---\n"
                            f"📊 **Strategy Stats:**\n"
                            f"• Trades: {stats_src['num_trades']} | Win Rate: {stats_src['win_rate']:.0f}%\n"
                            f"• Total Return: {stats_src['total_return']:.1f}% | PF: {stats_src['profit_factor']:.1f}\n"
                        )

                    # Build position status line
                    # Live trading signals are never "missed" - we're acting on them in real time
                    # is_delayed means signal came from a past bar (still valid, just entering late)
                    position_status = "🟢 **Position: LONG**" if not is_delayed else "🟡 **Position: LONG** (late entry)"

                    # Calculate SL/TP prices (v9: use regime-specific values)
                    entry_sl_price = current_price * (1 - display_sl / 100)
                    entry_tp_price = current_price * (1 + display_tp / 100)

                    buy_msg = (
                        f"@here\n"
                        f"📈 **[{strategy_label}] BUY SIGNAL**{signal_note}\n"
                        f"**Signal Time:** {signal_time_display}\n"
                        f"**Entry Price:** ${current_price:,.2f}\n"
                        f"{regime_line}"
                        f"{position_status}\n"
                        f"{stats_section}"
                        f"---\n"
                        f"_SL: {display_sl:.1f}% (${entry_sl_price:,.2f}) | TP: {display_tp:.1f}% (${entry_tp_price:,.2f})_"
                    )

                    # Generate trade log CSV (last 10 completed trades)
                    entry_trade_log_csv = None
                    try:
                        entry_trade_log_csv = generate_trade_log_csv(locked_backtest, num_trades=10)
                    except Exception as e:
                        print(f"⚠️ Could not generate entry trade log CSV: {e}")

                    send_discord_alert(webhook_url, buy_msg, signal_chart, strategy_name=strategy_name, csv_buf=entry_trade_log_csv)
                    print(f"BUY SIGNAL SENT!{signal_note}")

                    # Sierra Chart bridge (opt-in, fail-safe)
                    if SIERRA_BRIDGE_ENABLED:
                        try:
                            sierra_signal = build_sierra_signal(
                                signal_type="ENTRY",
                                direction="LONG",
                                ticker=ticker,
                                price=current_price,
                                signal_time=bar_close_time,  # Use bar close time (when signal fires)
                                strategy_name=strategy_name,
                                config=config
                            )
                            publish_signal_to_sierra(sierra_signal, SIERRA_SIGNAL_FILE)
                            print(f"   📊 Sierra Chart: Published ENTRY signal")
                        except Exception as e:
                            print(f"   ⚠️ Sierra Chart bridge error (Discord unaffected): {e}")

                    trade_state['position'] = 'long'
                    trade_state['entry_price'] = current_price
                    trade_state['entry_time'] = to_chicago_str(bar_close_time)  # Bar close time (Chicago)
                    trade_state['entry_signal_bar'] = to_chicago_str(signal_time_utc)  # Bar start time (Chicago)
                    trade_state['last_signal_time'] = to_chicago_str(bar_close_time)  # Use bar close for deduplication
                    # v9: store entry regime so exits use the correct params
                    if recent_buy_signal and recent_buy_signal.get('regime'):
                        trade_state['entry_regime'] = recent_buy_signal['regime']
                    save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)

                elif effective_sell:
                    # LONG-ONLY STRATEGY: Sell signals are ignored when not in position
                    # They are only used to EXIT existing long positions (handled above)
                    signal_time = recent_sell_signal['time']
                    print(f"   ℹ️  Sell signal on {signal_time} ignored (LONG-only strategy, no position)")
                    # Update last_signal_time to avoid re-processing
                    trade_state['last_signal_time'] = to_chicago_str(signal_time)
                    save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)

            # --- BAR CLOSE STATUS UPDATE (Intraday Only) ---
            # Post a status update every bar close for intraday strategies
            # This helps monitor the trader and catch issues early
            if is_intraday and should_check_signals and completed_bar_time is not None:
                # Check if we already posted for this bar
                completed_bar_str = str(completed_bar_time)
                if last_bar_update_time != completed_bar_str:
                    # Only post if no entry/exit happened this cycle (those have their own alerts)
                    if not effective_buy and not (trade_state.get('position') and just_exited_this_cycle):
                        try:
                            # Get oscillator zone description
                            osc_val = completed_bar.get('osc_smooth', 0) if completed_bar is not None else 0
                            vel_val = completed_bar.get('velocity', 0) if completed_bar is not None else 0

                            if osc_val < config.get('oversold_threshold', -0.3):
                                zone_desc = "OVERSOLD 📉"
                            elif osc_val > config.get('overbought_threshold', 0.3):
                                zone_desc = "OVERBOUGHT 📈"
                            else:
                                zone_desc = "Neutral"

                            # Position status
                            position_str = trade_state.get('position', 'None')
                            if position_str and position_str.lower() != 'none':
                                entry_price = trade_state.get('entry_price', 0)
                                if entry_price and current_price:
                                    unrealized_pnl = ((current_price - entry_price) / entry_price) * 100
                                    pos_status = f"LONG @ ${entry_price:.2f} ({unrealized_pnl:+.2f}%)"
                                else:
                                    pos_status = "LONG"
                            else:
                                pos_status = "Flat (no position)"

                            # Signal status
                            buy_sig = completed_bar.get('buy_signal', False) if completed_bar is not None else False
                            sell_sig = completed_bar.get('sell_signal', False) if completed_bar is not None else False
                            if buy_sig:
                                sig_status = "✅ BUY signal present"
                            elif sell_sig:
                                sig_status = "🔴 SELL signal present"
                            else:
                                sig_status = "No signal"

                            # Format bar time in Chicago timezone for display
                            try:
                                bar_ct = to_chicago_time(completed_bar_time)
                                bar_time_display = bar_ct.strftime('%H:%M CT') if bar_ct else str(completed_bar_time)[-8:-3]
                            except:
                                bar_time_display = str(completed_bar_time)[-8:-3]

                            # Build status message
                            bar_update_msg = (
                                f"📊 **[{strategy_label}] Bar Close Update** ({bar_time_display})\n"
                                f"**Price:** ${current_price:,.2f}\n"
                                f"**Oscillator:** {osc_val:+.4f} ({zone_desc})\n"
                                f"**Velocity:** {vel_val:+.5f}\n"
                                f"**Position:** {pos_status}\n"
                                f"**Signal:** {sig_status}"
                            )

                            send_discord_alert(webhook_url, bar_update_msg, strategy_name=strategy_name)
                            print(f"   📊 Bar close update sent for {completed_bar_str}")
                        except Exception as e:
                            print(f"   ⚠️ Failed to send bar close update: {e}")

                        # Update tracker regardless of success/failure to prevent spam
                        last_bar_update_time = completed_bar_str

            # Persist daily alerts to survive restarts
            save_daily_alerts(daily_alerts)

            # Smart scheduling: sleep until next bar close + buffer
            # This minimizes delay between signal generation and notification
            seconds_to_wait = seconds_until_next_bar_close(ticker, interval)
            buffer_seconds = 5  # Small buffer to ensure bar data is available
            total_wait = seconds_to_wait + buffer_seconds

            # Safety cap based on interval:
            # - Daily bars: cap at 25 hours (allows for timezone edge cases)
            # - Intraday: cap at 2 hours (in case of calculation issues)
            if interval == '1d':
                max_wait = 90000  # 25 hours
            else:
                max_wait = 7200   # 2 hours
            total_wait = min(total_wait, max_wait)

            # Display wait time in appropriate units
            if total_wait >= 3600:
                hours = total_wait // 3600
                mins = (total_wait % 3600) // 60
                print(f"⏳ [{strategy_label} | {strategy_name}] Next bar closes in {hours}h {mins}m")
            elif total_wait >= 60:
                print(f"⏳ [{strategy_label} | {strategy_name}] Next bar closes in {total_wait // 60}m {total_wait % 60}s")
            else:
                print(f"⏳ [{strategy_label} | {strategy_name}] Next bar closes in {total_wait}s")

            # CRITICAL FIX: Don't sleep for the entire duration at once!
            # For daily strategies, this could be hours - we need to wake up
            # periodically to check for scheduled status updates.
            # Sleep in chunks of check_interval_seconds to allow scheduled updates.
            time_slept = 0
            while time_slept < total_wait:
                sleep_chunk = min(check_interval_seconds, total_wait - time_slept)
                time.sleep(sleep_chunk)
                time_slept += sleep_chunk

                # Break out of wait loop to run the main cycle again
                # This allows scheduled updates to be checked
                # Only break early for daily strategies where wait is long
                if interval == '1d' and total_wait > 600:
                    break  # Go back to main loop to check scheduled updates

        except KeyboardInterrupt:
            print("\n\nShutting down...")
            shutdown_msg = "🛑 **[VELOCITY] Live Trader Stopped**\n_Manually terminated._"
            send_discord_alert(webhook_url, shutdown_msg, strategy_name=strategy_name)
            break
        except Exception as e:
            error_msg = f"⚠️ **[VELOCITY] Error in Trader**\n```{str(e)[:500]}```"
            print(f"Error: {e}")
            # Wrap Discord alert in try/except to prevent nested errors
            try:
                send_discord_alert(webhook_url, error_msg, strategy_name=strategy_name)
            except Exception as discord_err:
                print(f"   ⚠️ Failed to send error to Discord: {discord_err}")
            time.sleep(60)  # Wait 1 minute on error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Velocity-Based Live Trader")
    parser.add_argument("--config", type=str, default="production_env/velocity_config.json",
                        help="Path to velocity strategy config file")
    parser.add_argument("--skip-selection", "-s", action="store_true",
                        help="Skip interactive strategy selection and use config directly")
    parser.add_argument("--save", type=str, default=None,
                        help="Save current config as a named strategy bundle")
    args = parser.parse_args()

    # If --save is provided, save current config as a bundle and exit
    if args.save:
        try:
            config = load_config(args.config)
            save_strategy_bundle(config, args.save)
            print(f"Strategy saved as: {args.save}")
        except Exception as e:
            print(f"Failed to save strategy: {e}")
        exit(0)

    run_live_trader(args.config, skip_selection=args.skip_selection)
