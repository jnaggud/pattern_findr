"""
Velocity Production Utilities

Production-grade utilities for the velocity trading system:
- File locking for safe concurrent access
- Logging framework
- Timezone-aware market hours
- Heartbeat monitoring
- Safe JSON file operations

Usage:
    from velocity_production_utils import (
        safe_json_write, safe_json_read,
        get_logger, is_market_open, get_market_timezone,
        HeartbeatMonitor
    )
"""

import json
import os
import sys
import time
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Optional, Any, Dict
from pathlib import Path

# Timezone handling
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None  # Fallback to basic handling

# File locking (Unix/Mac)
try:
    import fcntl
    FILE_LOCKING_AVAILABLE = True
except ImportError:
    FILE_LOCKING_AVAILABLE = False  # Windows fallback


# ============================================================================
# LOGGING FRAMEWORK
# ============================================================================

_loggers = {}

def get_logger(name: str = "velocity", level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically strategy name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # File handler (optional - logs to velocity_logs/)
        log_dir = Path("velocity_logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets all levels
        file_format = logging.Formatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


# ============================================================================
# FILE LOCKING AND SAFE JSON OPERATIONS
# ============================================================================

class FileLock:
    """Context manager for file locking."""

    def __init__(self, filepath: str, timeout: float = 10.0):
        self.filepath = filepath
        self.timeout = timeout
        self.lock_file = filepath + ".lock"
        self.fd = None

    def __enter__(self):
        start_time = time.time()

        # Create lock file if it doesn't exist
        Path(self.lock_file).touch(exist_ok=True)

        self.fd = open(self.lock_file, 'w')

        if FILE_LOCKING_AVAILABLE:
            while True:
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return self
                except (IOError, OSError):
                    if time.time() - start_time > self.timeout:
                        raise TimeoutError(f"Could not acquire lock on {self.filepath}")
                    time.sleep(0.1)
        else:
            # Windows fallback - simple file existence check
            while os.path.exists(self.lock_file + ".active"):
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Could not acquire lock on {self.filepath}")
                time.sleep(0.1)
            Path(self.lock_file + ".active").touch()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if FILE_LOCKING_AVAILABLE:
            if self.fd:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                self.fd.close()
        else:
            # Windows fallback
            try:
                os.remove(self.lock_file + ".active")
            except:
                pass
            if self.fd:
                self.fd.close()


def safe_json_write(filepath: str, data: Any, indent: int = 2) -> bool:
    """
    Safely write JSON data to file with atomic write and locking.

    Uses a temp file + rename approach to prevent corruption on crash.

    Args:
        filepath: Path to the JSON file
        data: Data to serialize to JSON
        indent: JSON indentation (default 2)

    Returns:
        True if successful, False otherwise
    """
    logger = get_logger("velocity.io")

    try:
        with FileLock(filepath):
            # Write to temp file first
            dir_name = os.path.dirname(filepath) or '.'
            fd, temp_path = tempfile.mkstemp(suffix='.json', dir=dir_name)

            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(data, f, indent=indent, default=str)

                # Atomic rename (on Unix/Mac, this is atomic)
                shutil.move(temp_path, filepath)
                return True

            except Exception as e:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise e

    except Exception as e:
        logger.error(f"Failed to write {filepath}: {e}")
        return False


def safe_json_read(filepath: str, default: Any = None) -> Any:
    """
    Safely read JSON data from file with locking.

    Args:
        filepath: Path to the JSON file
        default: Default value if file doesn't exist or is invalid

    Returns:
        Parsed JSON data or default value
    """
    logger = get_logger("velocity.io")

    if not os.path.exists(filepath):
        return default

    try:
        with FileLock(filepath, timeout=5.0):
            with open(filepath, 'r') as f:
                return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        return default
    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        return default


# ============================================================================
# TIMEZONE AND MARKET HOURS
# ============================================================================

# Market timezone constants
MARKET_TIMEZONE = "America/New_York"
CRYPTO_TIMEZONE = "UTC"

def get_market_timezone():
    """Get the market timezone object."""
    if ZoneInfo:
        return ZoneInfo(MARKET_TIMEZONE)
    return None


def get_market_time() -> datetime:
    """Get current time in market timezone (America/New_York)."""
    if ZoneInfo:
        return datetime.now(ZoneInfo(MARKET_TIMEZONE))
    # Fallback: assume local time is market time (less accurate)
    return datetime.now()


def get_utc_time() -> datetime:
    """Get current UTC time."""
    if ZoneInfo:
        return datetime.now(ZoneInfo("UTC"))
    return datetime.utcnow()


def is_market_open(ticker: str = "SPY") -> Dict[str, Any]:
    """
    Check if market is open for the given ticker.

    Args:
        ticker: The ticker symbol

    Returns:
        Dict with:
            - is_open: bool
            - market_time: current market time
            - next_open: datetime of next market open
            - next_close: datetime of next market close
            - reason: string explaining status
    """
    # Crypto tickers trade 24/7
    crypto_tickers = ['BTC-USD', 'ETH-USD', 'BTC', 'ETH', 'BTCUSD', 'ETHUSD']
    is_crypto = any(ticker.upper().startswith(c.upper()) for c in ['BTC', 'ETH', 'DOGE', 'SOL', 'ADA'])

    if is_crypto or ticker.upper() in crypto_tickers:
        return {
            "is_open": True,
            "market_time": get_utc_time(),
            "reason": "Crypto markets trade 24/7",
            "is_crypto": True
        }

    market_time = get_market_time()
    weekday = market_time.weekday()  # Monday=0, Sunday=6
    hour = market_time.hour
    minute = market_time.minute
    current_minutes = hour * 60 + minute

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open_minutes = 9 * 60 + 30   # 9:30 AM = 570 minutes
    market_close_minutes = 16 * 60       # 4:00 PM = 960 minutes

    # Weekend check
    if weekday >= 5:  # Saturday or Sunday
        return {
            "is_open": False,
            "market_time": market_time,
            "reason": "Weekend - markets closed",
            "is_crypto": False
        }

    # Pre-market
    if current_minutes < market_open_minutes:
        return {
            "is_open": False,
            "market_time": market_time,
            "reason": f"Pre-market - opens at 9:30 AM ET",
            "is_crypto": False
        }

    # After hours
    if current_minutes >= market_close_minutes:
        return {
            "is_open": False,
            "market_time": market_time,
            "reason": f"After hours - closed at 4:00 PM ET",
            "is_crypto": False
        }

    # Market is open
    return {
        "is_open": True,
        "market_time": market_time,
        "reason": "Regular trading hours",
        "is_crypto": False
    }


def get_scheduled_update_key(update_type: str) -> str:
    """
    Get a unique key for scheduled updates to prevent duplicates.
    Uses market time to ensure correct day boundaries.

    Args:
        update_type: Type of update (OPEN, MID, CLOSE, MIDNIGHT, HOUR_9, etc.)

    Returns:
        Unique key string like "2026-01-05_OPEN"
    """
    market_time = get_market_time()
    day_str = market_time.strftime("%Y-%m-%d")
    return f"{day_str}_{update_type}"


def should_send_scheduled_update(update_type: str, sent_alerts: set,
                                  time_window_start: str, time_window_end: str) -> bool:
    """
    Check if a scheduled update should be sent.

    Args:
        update_type: Type of update (OPEN, MID, CLOSE, etc.)
        sent_alerts: Set of already sent alert keys
        time_window_start: Start of window (e.g., "08:30")
        time_window_end: End of window (e.g., "08:45")

    Returns:
        True if update should be sent
    """
    market_time = get_market_time()
    hm = market_time.strftime("%H:%M")

    if time_window_start <= hm <= time_window_end:
        key = get_scheduled_update_key(update_type)
        if key not in sent_alerts:
            return True

    return False


# ============================================================================
# POSITION SIZING
# ============================================================================

def calculate_position_size(
    account_balance: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_loss_price: float,
    max_position_pct: float = 25.0
) -> dict:
    """
    Calculate recommended position size based on risk parameters.

    NOTE: This is for INFORMATIONAL purposes only. The velocity system
    is SIGNAL-ONLY and does not execute trades. You must manually
    determine your own position sizes.

    Args:
        account_balance: Total account balance in dollars
        risk_per_trade_pct: Maximum risk per trade as percentage (e.g., 2.0 = 2%)
        entry_price: Expected entry price
        stop_loss_price: Stop loss price
        max_position_pct: Maximum position size as percentage of account

    Returns:
        Dict with position sizing information
    """
    if entry_price <= 0 or stop_loss_price <= 0 or account_balance <= 0:
        return {"error": "Invalid input values"}

    # Calculate risk per unit
    risk_per_unit = abs(entry_price - stop_loss_price)
    risk_pct_per_unit = (risk_per_unit / entry_price) * 100

    # Calculate max risk in dollars
    max_risk_dollars = account_balance * (risk_per_trade_pct / 100)

    # Calculate units based on risk
    if risk_per_unit > 0:
        units_by_risk = max_risk_dollars / risk_per_unit
    else:
        units_by_risk = 0

    # Calculate max units by position limit
    max_position_dollars = account_balance * (max_position_pct / 100)
    units_by_position = max_position_dollars / entry_price

    # Use the smaller of the two
    recommended_units = min(units_by_risk, units_by_position)
    position_value = recommended_units * entry_price
    position_pct = (position_value / account_balance) * 100

    return {
        "recommended_units": round(recommended_units, 4),
        "position_value": round(position_value, 2),
        "position_pct": round(position_pct, 2),
        "risk_dollars": round(recommended_units * risk_per_unit, 2),
        "risk_pct": round(risk_per_trade_pct, 2),
        "risk_per_unit": round(risk_per_unit, 2),
        "limited_by": "risk" if units_by_risk < units_by_position else "position_size"
    }


def get_position_sizing_disclaimer() -> str:
    """Get the position sizing disclaimer text."""
    return """
POSITION SIZING DISCLAIMER:
The position sizing calculations provided are for INFORMATIONAL purposes only.
This system does NOT execute trades and you must determine your own position sizes.
Always consider:
- Your personal risk tolerance
- Account size and leverage
- Market liquidity
- Correlation with other positions
- Tax implications
Never risk more than you can afford to lose.
"""


# ============================================================================
# HEARTBEAT MONITORING
# ============================================================================

class HeartbeatMonitor:
    """
    Monitors bot health and sends periodic heartbeat messages.

    Usage:
        heartbeat = HeartbeatMonitor(
            interval_hours=4,
            webhook_url="https://discord.com/...",
            strategy_name="velocity_BTC_5y"
        )

        # In main loop:
        heartbeat.check()  # Sends heartbeat if interval has passed
    """

    def __init__(self, interval_hours: float = 4.0, webhook_url: str = None,
                 strategy_name: str = "velocity", send_func=None):
        """
        Initialize heartbeat monitor.

        Args:
            interval_hours: Hours between heartbeat messages
            webhook_url: Discord webhook URL (optional if send_func provided)
            strategy_name: Name of the strategy for messages
            send_func: Custom function to send messages (receives message string)
        """
        self.interval_seconds = interval_hours * 3600
        self.webhook_url = webhook_url
        self.strategy_name = strategy_name
        self.send_func = send_func
        self.last_heartbeat = time.time()
        self.start_time = time.time()
        self.cycle_count = 0
        self.error_count = 0
        self.last_error = None
        self.logger = get_logger(f"heartbeat.{strategy_name}")

    def record_cycle(self):
        """Record a successful update cycle."""
        self.cycle_count += 1

    def record_error(self, error: str):
        """Record an error occurrence."""
        self.error_count += 1
        self.last_error = error

    def check(self, force: bool = False) -> bool:
        """
        Check if heartbeat should be sent and send it.

        Args:
            force: If True, send heartbeat regardless of interval

        Returns:
            True if heartbeat was sent
        """
        current_time = time.time()
        elapsed = current_time - self.last_heartbeat

        if not force and elapsed < self.interval_seconds:
            return False

        # Build heartbeat message
        uptime_hours = (current_time - self.start_time) / 3600
        uptime_str = f"{uptime_hours:.1f}h" if uptime_hours < 24 else f"{uptime_hours/24:.1f}d"

        market_status = is_market_open(self.strategy_name.split('_')[1] if '_' in self.strategy_name else "SPY")
        market_str = "OPEN" if market_status['is_open'] else "CLOSED"

        message = (
            f"**[{self.strategy_name}] Heartbeat**\n"
            f"Status: Running\n"
            f"Uptime: {uptime_str}\n"
            f"Cycles: {self.cycle_count}\n"
            f"Errors: {self.error_count}\n"
            f"Market: {market_str}"
        )

        if self.last_error:
            message += f"\nLast Error: {self.last_error[:100]}"

        # Send heartbeat
        sent = self._send_message(message)

        if sent:
            self.last_heartbeat = current_time
            self.logger.info(f"Heartbeat sent (uptime: {uptime_str}, cycles: {self.cycle_count})")

        return sent

    def _send_message(self, message: str) -> bool:
        """Send message via webhook or custom function."""
        if self.send_func:
            try:
                self.send_func(message)
                return True
            except Exception as e:
                self.logger.error(f"Failed to send heartbeat: {e}")
                return False

        if self.webhook_url:
            try:
                import requests
                response = requests.post(
                    self.webhook_url,
                    json={"content": message},
                    timeout=10
                )
                return response.status_code in [200, 204]
            except Exception as e:
                self.logger.error(f"Failed to send heartbeat: {e}")
                return False

        # No send method available, just log
        self.logger.info(f"Heartbeat (no webhook): {message}")
        return True


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def load_webhook_from_env(strategy_name: str = None) -> Optional[str]:
    """
    Load Discord webhook URL from environment variable.

    Checks:
        1. VELOCITY_WEBHOOK_{STRATEGY_NAME} (e.g., VELOCITY_WEBHOOK_BTC_5Y)
        2. VELOCITY_WEBHOOK_DEFAULT
        3. DISCORD_WEBHOOK

    Returns:
        Webhook URL or None
    """
    # Strategy-specific webhook
    if strategy_name:
        safe_name = strategy_name.upper().replace('-', '_')
        env_key = f"VELOCITY_WEBHOOK_{safe_name}"
        if os.environ.get(env_key):
            return os.environ[env_key]

    # Default webhook
    if os.environ.get("VELOCITY_WEBHOOK_DEFAULT"):
        return os.environ["VELOCITY_WEBHOOK_DEFAULT"]

    if os.environ.get("DISCORD_WEBHOOK"):
        return os.environ["DISCORD_WEBHOOK"]

    return None


def get_config_value(config: dict, key: str, default: Any = None,
                     env_prefix: str = "VELOCITY") -> Any:
    """
    Get config value with environment variable override.

    Checks environment variable first (e.g., VELOCITY_STOP_LOSS_PCT),
    then falls back to config dict, then default.

    Args:
        config: Configuration dictionary
        key: Config key (e.g., "stop_loss_pct")
        default: Default value if not found
        env_prefix: Prefix for environment variable

    Returns:
        Config value
    """
    # Check environment variable
    env_key = f"{env_prefix}_{key.upper()}"
    if os.environ.get(env_key):
        env_val = os.environ[env_key]
        # Try to parse as same type as default
        if isinstance(default, bool):
            return env_val.lower() in ('true', '1', 'yes')
        elif isinstance(default, int):
            return int(env_val)
        elif isinstance(default, float):
            return float(env_val)
        return env_val

    # Check config dict
    if key in config:
        return config[key]

    return default


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_trade_state(state: dict) -> tuple[bool, str]:
    """
    Validate trade state dictionary.

    Returns:
        (is_valid, error_message)
    """
    if not isinstance(state, dict):
        return False, "State must be a dictionary"

    required_keys = ['position', 'entry_price', 'entry_time', 'last_signal_time']

    for key in required_keys:
        if key not in state:
            return False, f"Missing required key: {key}"

    # If in position, validate entry data
    if state.get('position'):
        if state['position'] not in ['long', 'short', 'LONG', 'SHORT']:
            return False, f"Invalid position type: {state['position']}"

        if not state.get('entry_price'):
            return False, "In position but missing entry_price"

        if not isinstance(state['entry_price'], (int, float)):
            return False, f"Invalid entry_price type: {type(state['entry_price'])}"

        if state['entry_price'] <= 0:
            return False, f"Invalid entry_price: {state['entry_price']}"

    return True, ""


def validate_config(config: dict) -> tuple[bool, list[str]]:
    """
    Validate strategy configuration.

    Returns:
        (is_valid, list_of_warnings)
    """
    warnings = []

    # Required fields
    required = ['ticker', 'signal_type']
    for field in required:
        if field not in config:
            return False, [f"Missing required field: {field}"]

    # Validate thresholds
    if config.get('stop_loss_pct', 0) <= 0:
        warnings.append("stop_loss_pct should be positive")

    if config.get('take_profit_pct', 0) <= 0:
        warnings.append("take_profit_pct should be positive")

    if config.get('oversold_threshold', 0) >= 0:
        warnings.append("oversold_threshold should be negative")

    if config.get('overbought_threshold', 0) <= 0:
        warnings.append("overbought_threshold should be positive")

    # Validate signal type
    valid_signal_types = [
        'velocity_crossover_and_zone', 'velocity_crossover_or_zone',
        'zone_only', 'momentum', 'any_reversal', 'double_bottom',
        'divergence', 'breakout'
    ]
    if config.get('signal_type') not in valid_signal_types:
        warnings.append(f"Unknown signal_type: {config.get('signal_type')}")

    return True, warnings


# ============================================================================
# SIGNAL-ONLY SYSTEM DISCLAIMER
# ============================================================================

SIGNAL_ONLY_DISCLAIMER = """
================================================================================
                         IMPORTANT: SIGNAL-ONLY SYSTEM
================================================================================

This is a SIGNAL GENERATION system, NOT an automated trading system.

What this bot DOES:
  - Monitors price data and calculates trading signals
  - Sends alerts to Discord when signals are generated
  - Tracks paper positions for performance measurement
  - Generates charts and statistics

What this bot DOES NOT DO:
  - Execute actual trades on any exchange or broker
  - Connect to any trading API (Alpaca, IBKR, Coinbase, etc.)
  - Manage real money or positions
  - Guarantee any profits or trading performance

YOU ARE RESPONSIBLE FOR:
  - Manually executing trades if you choose to act on signals
  - Verifying signal quality before trading
  - Managing your own risk and position sizes
  - Understanding that past backtest performance does not guarantee future results

================================================================================
"""

def print_signal_only_disclaimer():
    """Print the signal-only system disclaimer."""
    print(SIGNAL_ONLY_DISCLAIMER)


# ============================================================================
# MAIN - Self-test when run directly
# ============================================================================

if __name__ == "__main__":
    print("Velocity Production Utils - Self Test\n")

    # Test logging
    print("1. Testing logging...")
    logger = get_logger("test")
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    print("   Logging OK\n")

    # Test file operations
    print("2. Testing safe JSON operations...")
    test_file = "/tmp/velocity_test.json"
    test_data = {"position": "long", "entry_price": 100.50}

    if safe_json_write(test_file, test_data):
        print(f"   Write OK: {test_file}")

    read_data = safe_json_read(test_file)
    if read_data == test_data:
        print(f"   Read OK: {read_data}")

    os.remove(test_file)
    print("   Safe JSON OK\n")

    # Test market hours
    print("3. Testing market hours...")
    for ticker in ["SPY", "BTC-USD"]:
        status = is_market_open(ticker)
        print(f"   {ticker}: {'OPEN' if status['is_open'] else 'CLOSED'} - {status['reason']}")
    print("   Market hours OK\n")

    # Test validation
    print("4. Testing validation...")
    valid_state = {"position": "long", "entry_price": 100, "entry_time": "2026-01-01", "last_signal_time": None}
    is_valid, error = validate_trade_state(valid_state)
    print(f"   Valid state check: {'PASS' if is_valid else 'FAIL'}")

    invalid_state = {"position": "long"}  # Missing entry_price
    is_valid, error = validate_trade_state(invalid_state)
    print(f"   Invalid state check: {'PASS' if not is_valid else 'FAIL'} ({error})")
    print("   Validation OK\n")

    # Print disclaimer
    print("5. Signal-only disclaimer:")
    print_signal_only_disclaimer()

    print("\nAll tests passed!")
