"""
Market hours and trading schedule management.

Provides DST-aware market hours handling for:
- US Stocks (NYSE/NASDAQ): 9:30 AM - 4:00 PM ET
- CME Futures (ES, GC, CL, etc.): Sunday 5 PM - Friday 5 PM CT
- Crypto: 24/7

Key features:
- DST-aware timezone handling
- Bar completion checks
- Market closure detection
- Holiday calendar support
"""

from datetime import datetime, timedelta, time
from typing import Optional, Dict, Tuple
import pytz

# Timezone definitions
TZ_ET = pytz.timezone('America/New_York')  # US Eastern
TZ_CT = pytz.timezone('America/Chicago')    # US Central (CME)
TZ_UTC = pytz.UTC


# Market configuration by type
MARKET_CONFIG = {
    'stocks_us': {
        'timezone': 'America/New_York',
        'market_open': time(9, 30),
        'market_close': time(16, 0),
        'daily_bar_close': time(16, 0),
        'weekend_closed': True,
        'description': 'US Stocks (NYSE/NASDAQ)'
    },
    'futures_cme': {
        'timezone': 'America/Chicago',
        'market_open': time(17, 0),  # Sunday 5 PM CT
        'market_close': time(16, 0),  # Friday 4 PM CT
        'daily_break_start': time(16, 0),  # Daily maintenance
        'daily_break_end': time(17, 0),
        'daily_bar_close': time(17, 0),  # CME day ends 5 PM CT
        'weekend_closed': True,  # Sat 4 PM - Sun 5 PM CT
        'description': 'CME Futures (ES, GC, CL, etc.)'
    },
    'crypto': {
        'timezone': 'UTC',
        'market_open': None,  # 24/7
        'market_close': None,
        'daily_bar_close': time(0, 0),  # Midnight UTC
        'weekend_closed': False,
        'description': 'Cryptocurrency (24/7)'
    }
}

# Ticker to market type mapping
TICKER_MARKET_MAP = {
    # US Stocks/ETFs
    'SPY': 'stocks_us',
    'QQQ': 'stocks_us',
    'IWM': 'stocks_us',
    'DIA': 'stocks_us',
    'AAPL': 'stocks_us',
    'MSFT': 'stocks_us',
    'GOOGL': 'stocks_us',
    'AMZN': 'stocks_us',
    'TSLA': 'stocks_us',
    'NVDA': 'stocks_us',
    '^GSPC': 'stocks_us',
    '^DJI': 'stocks_us',
    '^IXIC': 'stocks_us',

    # CME Futures
    'ES=F': 'futures_cme',
    'NQ=F': 'futures_cme',
    'YM=F': 'futures_cme',
    'RTY=F': 'futures_cme',
    'GC=F': 'futures_cme',
    'SI=F': 'futures_cme',
    'CL=F': 'futures_cme',
    'NG=F': 'futures_cme',
    'ZB=F': 'futures_cme',
    'ZN=F': 'futures_cme',

    # Crypto
    'BTC-USD': 'crypto',
    'ETH-USD': 'crypto',
    'SOL-USD': 'crypto',
    'DOGE-USD': 'crypto',
    'XRP-USD': 'crypto',
    'ADA-USD': 'crypto',
    'AVAX-USD': 'crypto',
}

# US Market Holidays (2025-2026)
US_HOLIDAYS = {
    # 2025
    datetime(2025, 1, 1),   # New Year's Day
    datetime(2025, 1, 20),  # MLK Day
    datetime(2025, 2, 17),  # Presidents Day
    datetime(2025, 4, 18),  # Good Friday
    datetime(2025, 5, 26),  # Memorial Day
    datetime(2025, 6, 19),  # Juneteenth
    datetime(2025, 7, 4),   # Independence Day
    datetime(2025, 9, 1),   # Labor Day
    datetime(2025, 11, 27), # Thanksgiving
    datetime(2025, 12, 25), # Christmas

    # 2026
    datetime(2026, 1, 1),   # New Year's Day
    datetime(2026, 1, 19),  # MLK Day
    datetime(2026, 2, 16),  # Presidents Day
    datetime(2026, 4, 3),   # Good Friday
    datetime(2026, 5, 25),  # Memorial Day
    datetime(2026, 6, 19),  # Juneteenth
    datetime(2026, 7, 3),   # Independence Day (observed)
    datetime(2026, 9, 7),   # Labor Day
    datetime(2026, 11, 26), # Thanksgiving
    datetime(2026, 12, 25), # Christmas
}

# Early close days (1 PM ET)
US_EARLY_CLOSE = {
    datetime(2025, 7, 3),   # Day before Independence Day
    datetime(2025, 11, 28), # Day after Thanksgiving
    datetime(2025, 12, 24), # Christmas Eve
    datetime(2026, 11, 27), # Day after Thanksgiving
    datetime(2026, 12, 24), # Christmas Eve
}


def get_market_type(ticker: str) -> str:
    """
    Get the market type for a ticker.

    Returns: 'stocks_us', 'futures_cme', or 'crypto'
    """
    if ticker in TICKER_MARKET_MAP:
        return TICKER_MARKET_MAP[ticker]

    # Infer from ticker format
    if ticker.endswith('=F'):
        return 'futures_cme'
    if ticker.endswith('-USD'):
        return 'crypto'

    # Default to stocks
    return 'stocks_us'


def get_market_config(ticker: str) -> Dict:
    """Get market configuration for a ticker."""
    market_type = get_market_type(ticker)
    return MARKET_CONFIG[market_type]


def get_market_timezone(ticker: str) -> pytz.timezone:
    """Get the timezone for a ticker's market."""
    config = get_market_config(ticker)
    return pytz.timezone(config['timezone'])


def get_current_market_time(ticker: str) -> datetime:
    """Get current time in the market's timezone."""
    tz = get_market_timezone(ticker)
    return datetime.now(tz)


def is_market_open(ticker: str, check_time: datetime = None) -> bool:
    """
    Check if the market is currently open for trading.

    Args:
        ticker: Trading symbol
        check_time: Time to check (default: now)

    Returns:
        True if market is open
    """
    market_type = get_market_type(ticker)
    config = MARKET_CONFIG[market_type]

    # Crypto is always open
    if market_type == 'crypto':
        return True

    tz = pytz.timezone(config['timezone'])
    if check_time is None:
        check_time = datetime.now(tz)
    elif check_time.tzinfo is None:
        check_time = tz.localize(check_time)
    else:
        check_time = check_time.astimezone(tz)

    # Check weekend
    if config.get('weekend_closed', False):
        if is_weekend_closure(ticker, check_time):
            return False

    # Check holidays
    if is_market_holiday(check_time.date(), ticker):
        return False

    # Check market hours
    current_time = check_time.time()

    if market_type == 'stocks_us':
        # Check early close (convert date to datetime for set comparison)
        check_date_dt = datetime(check_time.year, check_time.month, check_time.day)
        if check_date_dt in US_EARLY_CLOSE:
            close_time = time(13, 0)
        else:
            close_time = config['market_close']

        return config['market_open'] <= current_time < close_time

    elif market_type == 'futures_cme':
        # CME has daily maintenance break 4-5 PM CT
        break_start = config.get('daily_break_start', time(16, 0))
        break_end = config.get('daily_break_end', time(17, 0))

        if break_start <= current_time < break_end:
            return False

        return True  # Open otherwise (except weekends already checked)

    return True


def is_weekend_closure(ticker: str, check_time: datetime = None) -> bool:
    """
    Check if market is closed for weekend.

    - Stocks: Saturday and Sunday
    - Futures: Friday 4 PM CT to Sunday 5 PM CT
    - Crypto: Never (24/7)
    """
    market_type = get_market_type(ticker)

    if market_type == 'crypto':
        return False

    tz = get_market_timezone(ticker)
    if check_time is None:
        check_time = datetime.now(tz)
    elif check_time.tzinfo is None:
        check_time = tz.localize(check_time)
    else:
        check_time = check_time.astimezone(tz)

    weekday = check_time.weekday()  # 0=Monday, 6=Sunday
    current_time = check_time.time()

    if market_type == 'stocks_us':
        return weekday >= 5  # Saturday or Sunday

    elif market_type == 'futures_cme':
        # Friday after 4 PM CT
        if weekday == 4 and current_time >= time(16, 0):
            return True
        # All day Saturday
        if weekday == 5:
            return True
        # Sunday before 5 PM CT
        if weekday == 6 and current_time < time(17, 0):
            return True

    return False


def is_market_holiday(date, ticker: str) -> bool:
    """Check if a date is a market holiday."""
    market_type = get_market_type(ticker)

    # Crypto has no holidays
    if market_type == 'crypto':
        return False

    # Convert to date if datetime
    if isinstance(date, datetime):
        date = date.date()

    check_date = datetime(date.year, date.month, date.day)
    return check_date in US_HOLIDAYS


def is_early_close(date, ticker: str) -> bool:
    """Check if a date is an early close day."""
    market_type = get_market_type(ticker)

    if market_type == 'crypto':
        return False

    if isinstance(date, datetime):
        date = date.date()

    check_date = datetime(date.year, date.month, date.day)
    return check_date in US_EARLY_CLOSE


def get_bar_close_time(ticker: str, interval: str, bar_start: datetime) -> datetime:
    """
    Calculate when a bar closes based on its start time.

    Args:
        ticker: Trading symbol
        interval: Bar interval ('1d', '15m', '1h', etc.)
        bar_start: Bar start timestamp

    Returns:
        Bar close timestamp
    """
    if interval == '1d':
        return get_daily_bar_close(ticker, bar_start)

    # Parse interval to timedelta
    if interval.endswith('m') or interval.endswith('min'):
        minutes = int(interval.rstrip('min'))
        return bar_start + timedelta(minutes=minutes)
    elif interval.endswith('h'):
        hours = int(interval.rstrip('h'))
        return bar_start + timedelta(hours=hours)

    # Default: return bar_start (already closed)
    return bar_start


def get_daily_bar_close(ticker: str, bar_date: datetime) -> datetime:
    """
    Get the close time for a daily bar.

    - Stocks: 4 PM ET
    - Futures: 5 PM CT
    - Crypto: Midnight UTC
    """
    market_type = get_market_type(ticker)
    config = MARKET_CONFIG[market_type]
    tz = pytz.timezone(config['timezone'])

    if isinstance(bar_date, datetime):
        date = bar_date.date()
    else:
        date = bar_date

    close_time = config['daily_bar_close']
    close_dt = datetime.combine(date, close_time)

    return tz.localize(close_dt)


def is_bar_complete(
    ticker: str,
    interval: str,
    bar_time: datetime,
    check_time: datetime = None
) -> bool:
    """
    Check if a bar is complete (closed).

    Args:
        ticker: Trading symbol
        interval: Bar interval
        bar_time: Bar timestamp (start time)
        check_time: Time to check against (default: now)

    Returns:
        True if bar is complete/closed
    """
    if check_time is None:
        check_time = datetime.now(pytz.UTC)

    # Ensure both times are timezone-aware
    if bar_time.tzinfo is None:
        bar_time = pytz.UTC.localize(bar_time)
    if check_time.tzinfo is None:
        check_time = pytz.UTC.localize(check_time)

    bar_close = get_bar_close_time(ticker, interval, bar_time)

    # Ensure bar_close is timezone-aware
    if bar_close.tzinfo is None:
        bar_close = pytz.UTC.localize(bar_close)

    # Convert to UTC for comparison
    bar_close_utc = bar_close.astimezone(pytz.UTC)
    check_time_utc = check_time.astimezone(pytz.UTC)

    return check_time_utc > bar_close_utc


def seconds_until_bar_close(
    ticker: str,
    interval: str,
    bar_time: datetime = None
) -> int:
    """
    Calculate seconds until the current bar closes.

    Args:
        ticker: Trading symbol
        interval: Bar interval
        bar_time: Bar start time (default: current bar)

    Returns:
        Seconds until bar close (0 if already closed)
    """
    now = datetime.now(pytz.UTC)

    if bar_time is None:
        # Estimate current bar start
        if interval == '1d':
            market_type = get_market_type(ticker)
            config = MARKET_CONFIG[market_type]
            tz = pytz.timezone(config['timezone'])
            local_now = now.astimezone(tz)
            bar_time = tz.localize(datetime.combine(local_now.date(), time(0, 0)))
        else:
            # For intraday, estimate based on interval
            if interval.endswith('m') or interval.endswith('min'):
                minutes = int(interval.rstrip('min'))
            elif interval.endswith('h'):
                minutes = int(interval.rstrip('h')) * 60
            else:
                minutes = 15  # Default

            # Round down to nearest interval
            total_minutes = now.hour * 60 + now.minute
            bar_start_minutes = (total_minutes // minutes) * minutes
            bar_time = now.replace(
                hour=bar_start_minutes // 60,
                minute=bar_start_minutes % 60,
                second=0,
                microsecond=0
            )

    bar_close = get_bar_close_time(ticker, interval, bar_time)

    if bar_close.tzinfo is None:
        bar_close = pytz.UTC.localize(bar_close)

    bar_close_utc = bar_close.astimezone(pytz.UTC)

    diff = (bar_close_utc - now).total_seconds()
    return max(0, int(diff))


def get_signal_check_window(ticker: str, interval: str) -> Tuple[int, int]:
    """
    Get the time window (in seconds) after bar close when signals should be checked.

    Returns:
        Tuple of (min_seconds, max_seconds) after bar close

    For daily bars, we wait a bit for data to finalize.
    For intraday, we can check immediately.
    """
    if interval == '1d':
        market_type = get_market_type(ticker)
        if market_type == 'stocks_us':
            return (30 * 60, 60 * 60)  # 30-60 min after 4 PM
        elif market_type == 'futures_cme':
            return (5 * 60, 30 * 60)   # 5-30 min after 5 PM CT
        else:  # crypto
            return (5 * 60, 30 * 60)   # 5-30 min after midnight
    else:
        return (5, 120)  # 5 sec to 2 min after bar close


def seconds_until_market_open(ticker: str) -> int:
    """
    Calculate seconds until market opens.

    Returns:
        Seconds until market opens (0 if already open)
    """
    if is_market_open(ticker):
        return 0

    market_type = get_market_type(ticker)
    config = MARKET_CONFIG[market_type]
    tz = pytz.timezone(config['timezone'])
    now = datetime.now(tz)

    # Crypto is always open
    if market_type == 'crypto':
        return 0

    # Check if in daily maintenance break (futures)
    if market_type == 'futures_cme':
        break_start = config.get('daily_break_start', time(16, 0))
        break_end = config.get('daily_break_end', time(17, 0))
        current_time = now.time()

        if break_start <= current_time < break_end:
            # In maintenance break - calculate seconds until 5 PM
            break_end_dt = datetime.combine(now.date(), break_end)
            break_end_dt = tz.localize(break_end_dt)
            return max(0, int((break_end_dt - now).total_seconds()))

        # Check if in weekend closure (Friday 4 PM to Sunday 5 PM)
        weekday = now.weekday()
        if weekday == 4 and current_time >= time(16, 0):  # Friday after 4 PM
            # Opens Sunday 5 PM
            days_until_sunday = 2
            open_dt = datetime.combine(now.date() + timedelta(days=days_until_sunday), time(17, 0))
            open_dt = tz.localize(open_dt)
            return max(0, int((open_dt - now).total_seconds()))
        elif weekday == 5:  # Saturday
            # Opens Sunday 5 PM
            days_until_sunday = 1
            open_dt = datetime.combine(now.date() + timedelta(days=days_until_sunday), time(17, 0))
            open_dt = tz.localize(open_dt)
            return max(0, int((open_dt - now).total_seconds()))
        elif weekday == 6 and current_time < time(17, 0):  # Sunday before 5 PM
            open_dt = datetime.combine(now.date(), time(17, 0))
            open_dt = tz.localize(open_dt)
            return max(0, int((open_dt - now).total_seconds()))

    elif market_type == 'stocks_us':
        market_open = config['market_open']
        current_time = now.time()
        weekday = now.weekday()

        # Weekend
        if weekday >= 5:
            days_until_monday = 7 - weekday
            open_dt = datetime.combine(now.date() + timedelta(days=days_until_monday), market_open)
            open_dt = tz.localize(open_dt)
            return max(0, int((open_dt - now).total_seconds()))

        # Before market open today
        if current_time < market_open:
            open_dt = datetime.combine(now.date(), market_open)
            open_dt = tz.localize(open_dt)
            return max(0, int((open_dt - now).total_seconds()))

        # After market close - next day
        open_dt = datetime.combine(now.date() + timedelta(days=1), market_open)
        # Skip weekend
        while open_dt.weekday() >= 5:
            open_dt += timedelta(days=1)
        open_dt = tz.localize(open_dt)
        return max(0, int((open_dt - now).total_seconds()))

    # Default: return 60 seconds
    return 60
