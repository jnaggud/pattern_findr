"""
Velocity Core Module

Shared logic for velocity-based trading strategies.
This module contains all reusable functions for:
- Data fetching and caching
- Oscillator and signal calculation
- State management (trade state, history, locked backtests)
- Backtest engine
- Discord alerts and chart generation

IMPORTANT: This is a SIGNAL-ONLY system. It does NOT execute trades.
See velocity_production_utils.py for the full disclaimer.

Used by:
- velocity_live_trader.py (single strategy CLI)
- velocity_multi_trader.py (multi-strategy runner)
"""

import json
import os
import time
import io
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# Import production utilities for safe file operations and logging
try:
    from velocity_production_utils import (
        safe_json_write, safe_json_read, get_logger,
        is_market_open, get_market_time, validate_trade_state,
        load_webhook_from_env, HeartbeatMonitor
    )
    PRODUCTION_UTILS_AVAILABLE = True
except ImportError:
    PRODUCTION_UTILS_AVAILABLE = False
    # Fallback: define minimal versions
    def get_logger(name):
        return logging.getLogger(name)
    def safe_json_write(path, data, indent=2):
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    def safe_json_read(path, default=None):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return default

# Import oscillator calculations from Streamlit source of truth
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

# Data source
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_DISCORD_WEBHOOK = ""

LEGAL_DISCLAIMER = (
    "\n\n_Warning: This is not financial advice. Past performance does not "
    "guarantee future results. Trading involves substantial risk of loss. Only trade "
    "with capital you can afford to lose. For educational purposes only._"
)

HAUS_HEDGE_WEBHOOKS = {
    "velocity_SPY_1y": "",
    "velocity_SPY_2y": "",
    "velocity_SPY_5y": "",
    "velocity_BTC_1y": "",
    "velocity_BTC_2y": "",
    "velocity_BTC_5y": "",
}

VELOCITY_STRATEGIES_DIR = "velocity_strategies"
PRODUCTION_CONFIG_PATH = "production_env/velocity_config.json"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
                    'strategy_name': config.get('strategy_name', item),
                    'signal_type': config.get('signal_type', 'Unknown'),
                    'created': config.get('deployed_at', 'Unknown')
                })
            except Exception as e:
                print(f"Warning: Could not load {config_file}: {e}")

    strategies.sort(key=lambda x: x['created'], reverse=True)
    return strategies


def save_strategy_bundle(config: dict, name: str = None) -> str:
    """Save a strategy config as a permanent bundle."""
    ensure_strategies_dir()

    ticker = config.get('ticker', 'UNKNOWN')
    signal_type = config.get('signal_type', 'unknown')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if name:
        bundle_name = name
    else:
        bundle_name = f"{ticker}_{signal_type}_{timestamp}"

    bundle_path = os.path.join(VELOCITY_STRATEGIES_DIR, bundle_name)
    os.makedirs(bundle_path, exist_ok=True)

    config_path = os.path.join(bundle_path, "velocity_config.json")
    config['bundle_name'] = bundle_name
    config['saved_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Strategy saved to: {bundle_path}")
    return bundle_path


# ============================================================================
# CONFIG FUNCTIONS
# ============================================================================

def load_config(config_path: str = "production_env/velocity_config.json") -> dict:
    """Load velocity strategy configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


# ============================================================================
# DATA FUNCTIONS
# ============================================================================

def fetch_price_data(ticker: str, api_key: str = None, days: int = 200, interval: str = "1d",
                     use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch historical price data using yfinance with optional caching.
    """
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    # Try to use cached data first
    if use_cache:
        try:
            from data_cache import fetch_and_cache
            df = fetch_and_cache(ticker, days=days, interval=interval)
            if not df.empty:
                return df
        except ImportError:
            print("   data_cache module not found, fetching directly")
        except Exception as e:
            print(f"   Cache error: {e}, fetching directly")

    # Fallback to direct yfinance fetch
    print(f"Fetching {ticker} via yfinance ({days} days, {interval})")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

    # Handle MultiIndex columns
    df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df.columns = df.columns.str.lower()

    if len(df) > 0:
        print(f"   Loaded {len(df)} bars: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    else:
        print(f"   Warning: No data returned for {ticker}")

    return df


def fetch_realtime_price(ticker: str) -> float:
    """Fetch real-time/current price for display purposes."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
        if price:
            return float(price)
    except Exception as e:
        print(f"   Could not fetch real-time price: {e}")

    return None


# ============================================================================
# OSCILLATOR FUNCTIONS
# ============================================================================

def calculate_composite_oscillator(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Calculate composite oscillator - supports novel oscillator types."""
    oscillator_type = config.get('oscillator_type', 'composite_smooth') if config else 'composite_smooth'

    # Always calculate the base composite oscillator first
    df = create_composite_oscillator(df)

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
                print(f"   Unknown oscillator type: {oscillator_type}, using composite_smooth")
                if 'composite_smooth' in df.columns:
                    df['osc_smooth'] = df['composite_smooth']
        except Exception as e:
            print(f"   Error calculating {oscillator_type}: {e}, using composite_smooth")
            if 'composite_smooth' in df.columns:
                df['osc_smooth'] = df['composite_smooth']
    else:
        if 'composite_smooth' in df.columns:
            df['osc_smooth'] = df['composite_smooth']
        elif 'composite_oscillator' in df.columns:
            df['osc_smooth'] = df['composite_oscillator']

    return df


def calculate_velocity_signals(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate velocity-based trading signals."""
    signal_type = config.get('signal_type', 'velocity_crossover_and_zone')
    vel_smoothing = config.get('vel_smoothing', 3)
    extreme_zone_mult = config.get('extreme_zone_mult', 1.5)
    require_accel = config.get('require_accel', True)
    oversold_threshold = config.get('oversold_threshold', -0.3)
    overbought_threshold = config.get('overbought_threshold', 0.3)
    vel_threshold = config.get('vel_threshold', 0)
    accel_threshold = config.get('accel_threshold', 0)
    rsi_filter = config.get('rsi_filter', 'none')
    rsi_period = config.get('rsi_period', 14)
    rsi_oversold = config.get('rsi_oversold', 30)
    rsi_overbought = config.get('rsi_overbought', 70)
    use_macd_confirm = config.get('use_macd_confirm', False)
    use_bb_filter = config.get('use_bb_filter', False)

    osc_col = 'composite_smooth' if 'composite_smooth' in df.columns else 'composite_oscillator'

    # Apply smoothing
    if vel_smoothing > 1:
        df['osc_smooth'] = df[osc_col].rolling(window=vel_smoothing, center=False).mean()
        df['osc_smooth'] = df['osc_smooth'].bfill()
    else:
        df['osc_smooth'] = df[osc_col]

    # Calculate velocity and acceleration
    df['velocity'] = df['osc_smooth'].diff()
    df['acceleration'] = df['velocity'].diff()
    df['velocity'] = df['velocity'].fillna(0)
    df['acceleration'] = df['acceleration'].fillna(0)

    # Detect velocity zero-crossings
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
        raw_buy = df['vel_cross_up'] | extreme_oversold
        raw_sell = df['vel_cross_down'] | extreme_overbought
    elif signal_type == 'zone_only':
        raw_buy = extreme_oversold & (velocity > 0)
        raw_sell = extreme_overbought & (velocity < 0)
    elif signal_type == 'momentum':
        raw_buy = strong_momentum_up & (osc_smooth < 0)
        raw_sell = strong_momentum_down & (osc_smooth > 0)
    elif signal_type == 'any_reversal':
        raw_buy = df['vel_cross_up'] | extreme_oversold | (strong_momentum_up & in_oversold)
        raw_sell = df['vel_cross_down'] | extreme_overbought | (strong_momentum_down & in_overbought)
    elif signal_type == 'double_bottom':
        vel_cross_up_count = df['vel_cross_up'].rolling(10).sum()
        raw_buy = (vel_cross_up_count >= 2) & in_oversold
        vel_cross_down_count = df['vel_cross_down'].rolling(10).sum()
        raw_sell = (vel_cross_down_count >= 2) & in_overbought
    elif signal_type == 'divergence':
        close_prices = df['close']
        price_lower_low = (close_prices < close_prices.rolling(5).min().shift(1))
        osc_higher_low = (osc_smooth > osc_smooth.rolling(5).min().shift(1))
        raw_buy = price_lower_low & osc_higher_low & in_oversold
        price_higher_high = (close_prices > close_prices.rolling(5).max().shift(1))
        osc_lower_high = (osc_smooth < osc_smooth.rolling(5).max().shift(1))
        raw_sell = price_higher_high & osc_lower_high & in_overbought
    elif signal_type == 'breakout':
        osc_breaks_above = (osc_smooth > oversold_threshold) & (osc_smooth.shift(1) <= oversold_threshold)
        osc_breaks_below = (osc_smooth < overbought_threshold) & (osc_smooth.shift(1) >= overbought_threshold)
        raw_buy = osc_breaks_above
        raw_sell = osc_breaks_below
    else:
        raw_buy = df['vel_cross_up'] & in_oversold
        raw_sell = df['vel_cross_down'] & in_overbought

    # Apply velocity magnitude filter
    if vel_threshold > 0:
        raw_buy = raw_buy & (velocity.abs() >= vel_threshold)
        raw_sell = raw_sell & (velocity.abs() >= vel_threshold)

    # Apply acceleration filter
    if require_accel:
        buy_accel_cond = acceleration > 0
        sell_accel_cond = acceleration < 0
        if accel_threshold > 0:
            buy_accel_cond = buy_accel_cond & (acceleration.abs() >= accel_threshold)
            sell_accel_cond = sell_accel_cond & (acceleration.abs() >= accel_threshold)
        raw_buy = raw_buy & buy_accel_cond
        raw_sell = raw_sell & sell_accel_cond

    # Apply RSI filter
    if rsi_filter != 'none':
        df['rsi'] = calculate_rsi(df['close'], rsi_period)
        if rsi_filter == 'confirm':
            raw_buy = raw_buy & (df['rsi'] < rsi_oversold)
            raw_sell = raw_sell & (df['rsi'] > rsi_overbought)
        elif rsi_filter == 'divergence':
            rsi_rising = df['rsi'] > df['rsi'].shift(1)
            price_falling = df['close'] < df['close'].shift(1)
            raw_buy = raw_buy | (rsi_rising & price_falling & in_oversold)

    # Apply MACD confirmation
    if use_macd_confirm:
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        macd_bullish = macd > signal
        macd_bearish = macd < signal
        raw_buy = raw_buy & macd_bullish
        raw_sell = raw_sell & macd_bearish

    # Apply Bollinger Band filter
    if use_bb_filter:
        bb_period = 20
        bb_std = 2
        middle = df['close'].rolling(bb_period).mean()
        std = df['close'].rolling(bb_period).std()
        lower = middle - bb_std * std
        upper = middle + bb_std * std
        raw_buy = raw_buy & (df['close'] < lower)
        raw_sell = raw_sell & (df['close'] > upper)

    df['buy_signal'] = raw_buy
    df['sell_signal'] = raw_sell

    return df


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

def get_state_file_path(strategy_name: str = None, ticker: str = None) -> str:
    """Get strategy-specific state file path."""
    if strategy_name:
        safe_name = strategy_name.replace("/", "-").replace(":", "-").replace(" ", "_")
        return f"velocity_trade_state_{safe_name}.json"
    elif ticker:
        safe_ticker = ticker.replace("/", "-").replace(":", "-").replace(" ", "_")
        return f"velocity_trade_state_{safe_ticker}.json"
    return "velocity_trade_state.json"


def get_history_file_path(strategy_name: str = None, ticker: str = None) -> str:
    """Get strategy-specific trade history file path."""
    if strategy_name:
        safe_name = strategy_name.replace("/", "-").replace(":", "-").replace(" ", "_")
        return f"velocity_trade_history_{safe_name}.json"
    elif ticker:
        safe_ticker = ticker.replace("/", "-").replace(":", "-").replace(" ", "_")
        return f"velocity_trade_history_{safe_ticker}.json"
    return "velocity_trade_history.json"


def get_locked_backtest_path(strategy_name: str = None, ticker: str = None) -> str:
    """Get strategy-specific locked backtest file path."""
    if strategy_name:
        safe_name = strategy_name.replace("/", "-").replace(":", "-").replace(" ", "_")
        return f"velocity_locked_backtest_{safe_name}.json"
    elif ticker:
        safe_ticker = ticker.replace("/", "-").replace(":", "-").replace(" ", "_")
        return f"velocity_locked_backtest_{safe_ticker}.json"
    return "velocity_locked_backtest.json"


def load_trade_state(strategy_name: str = None, ticker: str = None, state_path: str = None) -> dict:
    """Load current trade state from file with safe file locking."""
    logger = get_logger("velocity.state")

    if state_path is None:
        state_path = get_state_file_path(strategy_name=strategy_name, ticker=ticker)

    default_state = {
        "position": None,  # None or "long" (LONG-only strategy)
        "entry_price": None,
        "entry_time": None,
        "entry_signal_bar": None,
        "last_signal_time": None,
    }

    state = safe_json_read(state_path, default=default_state)

    # Validate loaded state
    if PRODUCTION_UTILS_AVAILABLE:
        is_valid, error = validate_trade_state(state)
        if not is_valid:
            logger.warning(f"Invalid trade state in {state_path}: {error}")
            # Return state anyway but log the issue

    return state


def save_trade_state(state: dict, strategy_name: str = None, ticker: str = None, state_path: str = None):
    """Save trade state to file with atomic write and file locking."""
    logger = get_logger("velocity.state")

    if state_path is None:
        state_path = get_state_file_path(strategy_name=strategy_name, ticker=ticker)

    # Validate before saving
    if PRODUCTION_UTILS_AVAILABLE:
        is_valid, error = validate_trade_state(state)
        if not is_valid:
            logger.warning(f"Saving potentially invalid state to {state_path}: {error}")

    success = safe_json_write(state_path, state, indent=2)
    if not success:
        logger.error(f"Failed to save trade state to {state_path}")
    else:
        logger.debug(f"Saved trade state to {state_path}")


def load_trade_history(strategy_name: str = None, ticker: str = None, history_path: str = None) -> list:
    """Load trade history from file with safe file locking."""
    logger = get_logger("velocity.history")

    if history_path is None:
        history_path = get_history_file_path(strategy_name=strategy_name, ticker=ticker)

    history = safe_json_read(history_path, default=[])

    if not isinstance(history, list):
        logger.warning(f"Invalid history format in {history_path}, returning empty list")
        return []

    return history


def save_trade_history(history: list, strategy_name: str = None, ticker: str = None, history_path: str = None):
    """Save trade history to file with atomic write and file locking."""
    logger = get_logger("velocity.history")

    if history_path is None:
        history_path = get_history_file_path(strategy_name=strategy_name, ticker=ticker)

    success = safe_json_write(history_path, history, indent=2)
    if not success:
        logger.error(f"Failed to save trade history to {history_path}")


def log_closed_trade(ticker: str, position_type: str, entry_price: float, exit_price: float,
                     entry_time: str, exit_time: str, exit_reason: str, pnl_pct: float,
                     strategy_name: str = None, entry_signal_bar: str = None, exit_signal_bar: str = None):
    """Log a closed trade to history and return cumulative stats."""
    history = load_trade_history(strategy_name=strategy_name, ticker=ticker)

    # Calculate actual dollar P&L per unit
    if position_type.upper() == 'LONG':
        pnl_dollars = exit_price - entry_price
    else:
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
        "entry_signal_bar": entry_signal_bar or entry_time,
        "exit_signal_bar": exit_signal_bar or exit_time,
        "exit_reason": exit_reason,
        "pnl_pct": pnl_pct,
        "pnl_dollars": pnl_dollars
    }

    history.append(trade)
    save_trade_history(history, strategy_name=strategy_name, ticker=ticker)

    # Calculate cumulative stats
    total_trades = len(history)
    winners = [t for t in history if t['pnl_pct'] > 0]
    losers = [t for t in history if t['pnl_pct'] < 0]
    win_rate = (len(winners) / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum(t['pnl_pct'] for t in history)
    total_pnl_dollars = sum(t.get('pnl_dollars', 0) for t in history)
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


# ============================================================================
# LOCKED BACKTEST FUNCTIONS
# ============================================================================

def save_locked_backtest(backtest: dict, strategy_name: str = None, ticker: str = None, trade_state: dict = None):
    """Save backtest results as a locked snapshot (prevents repainting)."""
    logger = get_logger("velocity.backtest")
    path = get_locked_backtest_path(strategy_name=strategy_name, ticker=ticker)

    locked = {
        "locked_at": datetime.now().isoformat(),
        "entries": [],
        "exits": [],
        "current_position": backtest.get('current_position'),
        "num_trades": backtest.get('num_trades', 0),
        "win_rate": backtest.get('win_rate', 0),
        "total_return": backtest.get('total_return', 0),
        "profit_factor": backtest.get('profit_factor', 0),
    }

    # Get the tracked position's entry date for reconciliation
    tracked_entry_date = None
    if trade_state and trade_state.get('position') and trade_state.get('entry_time'):
        try:
            tracked_entry_str = str(trade_state['entry_time']).split('.')[0]
            tracked_entry_date = pd.to_datetime(tracked_entry_str)
            print(f"   Tracked position: {trade_state['position'].upper()} @ ${trade_state.get('entry_price', 0):.2f} on {tracked_entry_date}")
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not parse tracked entry time: {e}")

    # Convert entries to serializable format
    for entry in backtest.get('entries', []):
        entry_date = entry['date']
        if isinstance(entry_date, str):
            try:
                entry_date = pd.to_datetime(entry_date)
            except (ValueError, TypeError):
                pass  # Keep as string if parsing fails

        # Skip entries that conflict with tracked position
        if tracked_entry_date is not None and hasattr(entry_date, 'date'):
            if entry_date >= tracked_entry_date:
                print(f"   Skipping backtest entry {entry_date} (conflicts with tracked position)")
                continue

        locked["entries"].append({
            "date": str(entry['date']),
            "price": entry['price'],
            "position": entry.get('position', 'long')
        })

    # If we have a tracked position, add it as the current entry
    if trade_state and trade_state.get('position') and trade_state.get('entry_price'):
        locked["entries"].append({
            "date": str(trade_state.get('entry_time', '')),
            "price": trade_state['entry_price'],
            "position": trade_state['position']
        })
        print(f"   Added tracked {trade_state['position'].upper()} entry to locked backtest")

        locked["current_position"] = {
            "position": trade_state['position'],
            "entry_price": trade_state['entry_price'],
            "entry_date": str(trade_state.get('entry_time', ''))
        }

    # Convert exits to serializable format
    for exit_trade in backtest.get('exits', []):
        locked["exits"].append({
            "date": str(exit_trade['date']),
            "price": exit_trade['price'],
            "pnl": exit_trade['pnl'],
            "reason": exit_trade.get('reason', ''),
            "entry_price": exit_trade.get('entry_price', 0),
            "entry_date": str(exit_trade.get('entry_date', ''))
        })

    # Use safe atomic write with file locking
    success = safe_json_write(path, locked, indent=2)
    if success:
        print(f"   Locked backtest saved: {len(locked['entries'])} entries, {len(locked['exits'])} exits")
    else:
        logger = get_logger("velocity.backtest")
        logger.error(f"Failed to save locked backtest to {path}")

    return locked


def load_locked_backtest(strategy_name: str = None, ticker: str = None) -> dict:
    """Load locked backtest snapshot with safe file locking."""
    logger = get_logger("velocity.backtest")
    path = get_locked_backtest_path(strategy_name=strategy_name, ticker=ticker)

    result = safe_json_read(path, default=None)
    if result is None and os.path.exists(path):
        logger.warning(f"Could not load locked backtest from {path}")

    return result


def append_to_locked_backtest(entry: dict = None, exit_trade: dict = None,
                               strategy_name: str = None, ticker: str = None,
                               missed: bool = False):
    """Append a new entry or exit to the locked backtest."""
    locked = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)

    if locked is None:
        print("   No locked backtest to append to")
        return

    if entry:
        entry_record = {
            "date": str(entry.get('date', '')),
            "price": entry.get('price', 0),
            "position": entry.get('position', 'long')
        }
        if missed:
            entry_record["missed"] = True
        locked["entries"].append(entry_record)
        label = "missed" if missed else "locked"
        print(f"   {label} Appended new entry to locked backtest: {entry.get('date')}")

    if exit_trade:
        exit_record = {
            "date": str(exit_trade.get('date', '')),
            "price": exit_trade.get('price', 0),
            "pnl": exit_trade.get('pnl', 0),
            "reason": exit_trade.get('reason', ''),
            "entry_price": exit_trade.get('entry_price', 0),
            "entry_date": str(exit_trade.get('entry_date', ''))
        }
        if missed:
            exit_record["missed"] = True
        locked["exits"].append(exit_record)

        # Update stats
        tracked_exits = [e for e in locked["exits"] if not e.get('missed')]
        missed_exits = [e for e in locked["exits"] if e.get('missed')]

        locked["num_trades"] = len(tracked_exits)
        locked["num_missed"] = len(missed_exits)
        if tracked_exits:
            winners = [e for e in tracked_exits if e['pnl'] > 0]
            locked["win_rate"] = (len(winners) / len(tracked_exits)) * 100
            locked["total_return"] = sum(e['pnl'] for e in tracked_exits)
        if missed_exits:
            missed_winners = [e for e in missed_exits if e['pnl'] > 0]
            locked["missed_win_rate"] = (len(missed_winners) / len(missed_exits)) * 100 if missed_exits else 0
            locked["missed_return"] = sum(e['pnl'] for e in missed_exits)

        label = "missed" if missed else "locked"
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
    success = safe_json_write(path, locked, indent=2)
    if not success:
        logger = get_logger("velocity.backtest")
        logger.error(f"Failed to append to locked backtest: {path}")


def detect_and_add_missed_signals(fresh_backtest: dict, strategy_name: str = None, ticker: str = None) -> int:
    """Detect signals that occurred while bot was down and add them as missed."""
    locked = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)
    if locked is None or fresh_backtest is None:
        return 0

    locked_entries = locked.get('entries', [])
    locked_exits = locked.get('exits', [])

    last_locked_date = None
    for entry in locked_entries:
        entry_date = pd.to_datetime(entry['date']) if entry.get('date') else None
        if entry_date and (last_locked_date is None or entry_date > last_locked_date):
            last_locked_date = entry_date
    for exit_t in locked_exits:
        exit_date = pd.to_datetime(exit_t['date']) if exit_t.get('date') else None
        if exit_date and (last_locked_date is None or exit_date > last_locked_date):
            last_locked_date = exit_date

    if last_locked_date is None:
        return 0

    existing_entry_dates = set()
    for entry in locked_entries:
        if entry.get('date'):
            existing_entry_dates.add(str(entry['date'])[:10])

    existing_exit_dates = set()
    for exit_t in locked_exits:
        if exit_t.get('date'):
            existing_exit_dates.add(str(exit_t['date'])[:10])

    missed_count = 0

    for entry in fresh_backtest.get('entries', []):
        entry_date = pd.to_datetime(entry['date']) if entry.get('date') else None
        if entry_date and entry_date > last_locked_date:
            date_str = str(entry['date'])[:10]
            if date_str not in existing_entry_dates:
                append_to_locked_backtest(
                    entry={'date': entry['date'], 'price': entry['price'], 'position': entry.get('position', 'long')},
                    strategy_name=strategy_name, ticker=ticker, missed=True
                )
                existing_entry_dates.add(date_str)
                missed_count += 1

    for exit_t in fresh_backtest.get('exits', []):
        exit_date = pd.to_datetime(exit_t['date']) if exit_t.get('date') else None
        if exit_date and exit_date > last_locked_date:
            date_str = str(exit_t['date'])[:10]
            if date_str not in existing_exit_dates:
                append_to_locked_backtest(
                    exit_trade={
                        'date': exit_t['date'],
                        'price': exit_t['price'],
                        'pnl': exit_t.get('pnl', 0),
                        'reason': exit_t.get('reason', 'Missed'),
                        'entry_price': exit_t.get('entry_price', 0),
                        'entry_date': exit_t.get('entry_date', '')
                    },
                    strategy_name=strategy_name, ticker=ticker, missed=True
                )
                existing_exit_dates.add(date_str)
                missed_count += 1

    if missed_count > 0:
        print(f"   Detected and added {missed_count} missed signals from downtime")

    return missed_count


# ============================================================================
# BACKTEST FUNCTIONS
# ============================================================================

def run_historical_backtest(df: pd.DataFrame, config: dict) -> dict:
    """Run a backtest on historical data to determine current state."""
    df = calculate_velocity_signals(df, config)

    stop_loss_pct = config.get('stop_loss_pct', 5.0)
    take_profit_pct = config.get('take_profit_pct', 10.0)
    min_bars_between = config.get('min_bars_between', 1)
    exit_on_opposite = config.get('exit_on_opposite_signal', True)
    exit_on_midline = config.get('exit_on_midline_cross', False)

    position = 0
    entry_price = None
    entry_date = None
    entry_osc = None
    last_trade_bar = -min_bars_between
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = row['close']
        date = df.index[i]
        osc = row['osc_smooth']
        bars_since = i - last_trade_bar

        # Check exits first
        if position == 1 and entry_price:
            pnl_pct = ((price - entry_price) / entry_price) * 100
            exit_reason = None

            if stop_loss_pct > 0 and pnl_pct <= -stop_loss_pct:
                exit_reason = "Stop Loss"
            elif take_profit_pct > 0 and pnl_pct >= take_profit_pct:
                exit_reason = "Take Profit"
            elif exit_on_midline and osc > 0:
                exit_reason = "Midline Cross"
            elif exit_on_opposite and row['sell_signal'] and bars_since >= min_bars_between:
                exit_reason = "Opposite Signal"

            if exit_reason:
                trades.append({
                    'type': 'exit',
                    'date': date,
                    'price': price,
                    'pnl': pnl_pct,
                    'reason': exit_reason,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                })
                position = 0
                entry_price = None
                entry_date = None
                last_trade_bar = i

        # Check entries
        if position == 0 and row['buy_signal'] and bars_since >= min_bars_between:
            position = 1
            entry_price = price
            entry_date = date
            entry_osc = osc
            last_trade_bar = i
            trades.append({
                'type': 'entry',
                'date': date,
                'price': price,
                'osc': osc,
            })

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
    }


def run_backtest_for_period(df: pd.DataFrame, config: dict, days: int = None) -> tuple:
    """Run backtest for a specific period of days."""
    if days is not None and len(df) > days:
        df_period = df.iloc[-days:].copy()
    else:
        df_period = df.copy()

    df_period = calculate_velocity_signals(df_period, config)
    backtest = run_historical_backtest(df_period, config)

    return df_period, backtest


# ============================================================================
# DISCORD FUNCTIONS
# ============================================================================

def send_discord_alert(webhook_url: str, message: str, chart_buf: io.BytesIO = None,
                       include_disclaimer: bool = True, strategy_name: str = None):
    """Send alert to Discord webhook with optional chart image."""
    if not webhook_url:
        print(f"[ALERT] {message}")
        return False

    if include_disclaimer:
        message = message + LEGAL_DISCLAIMER

    def post_to_webhook(url: str, msg: str, chart: io.BytesIO = None) -> bool:
        try:
            if chart:
                chart.seek(0)
                files = {'file': ('chart.png', chart, 'image/png')}
                payload = {'content': msg}
                response = requests.post(url, data=payload, files=files)
            else:
                payload = {"content": msg}
                response = requests.post(url, json=payload)

            if response.status_code in [200, 204]:
                return True
            else:
                print(f"Discord webhook error: {response.status_code}")
                return False
        except Exception as e:
            print(f"Discord webhook error: {e}")
            return False

    primary_success = post_to_webhook(webhook_url, message, chart_buf)

    if strategy_name and strategy_name in HAUS_HEDGE_WEBHOOKS:
        secondary_url = HAUS_HEDGE_WEBHOOKS[strategy_name]
        if chart_buf:
            chart_buf.seek(0)
        secondary_success = post_to_webhook(secondary_url, message, chart_buf)
        if secondary_success:
            print(f"   Also posted to Haus Hedge server")

    return primary_success


def generate_velocity_chart(df: pd.DataFrame, backtest: dict, config: dict, ticker: str,
                            title_suffix: str = "", trade_history: list = None,
                            current_position: dict = None, locked_backtest: dict = None,
                            full_period_backtest: dict = None) -> io.BytesIO:
    """Generate a velocity strategy chart for Discord."""
    try:
        df_plot = df.copy()

        if not isinstance(df_plot.index, pd.DatetimeIndex):
            try:
                df_plot.index = pd.to_datetime(df_plot.index)
            except:
                df_plot = df_plot.reset_index(drop=True)

        fig, axes = plt.subplots(4, 1, figsize=(16, 12),
                                 gridspec_kw={'height_ratios': [3, 1.5, 1, 1.5]}, sharex=False)

        fig.patch.set_facecolor('#1a1a2e')
        for ax in axes:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white')
            ax.grid(True, color='#333', alpha=0.5)

        import matplotlib.dates as mdates

        # 1. Price Chart
        ax1 = axes[0]
        ax1.plot(df_plot.index, df_plot['close'], color='white', linewidth=1.5, label='Price')
        ax1.fill_between(df_plot.index, df_plot['low'], df_plot['high'], color='gray', alpha=0.2)

        markers_source = locked_backtest if locked_backtest else backtest
        chart_start = df_plot.index.min()
        chart_end = df_plot.index.max()

        if markers_source and markers_source.get('entries'):
            for entry in markers_source['entries']:
                entry_date = entry['date']
                if isinstance(entry_date, str):
                    try:
                        entry_date = pd.to_datetime(entry_date)
                    except:
                        continue

                if entry_date < chart_start or entry_date > chart_end:
                    continue

                is_long = entry.get('position', 'long') == 'long'
                is_missed = entry.get('missed', False)
                marker_shape = '^' if is_long else 'v'
                if is_missed:
                    marker_color = 'orange'
                else:
                    marker_color = 'lime' if is_long else 'red'

                if entry_date in df_plot.index:
                    ax1.scatter(entry_date, entry['price'], marker=marker_shape, color=marker_color, s=100, zorder=5)
                else:
                    try:
                        closest_idx = df_plot.index.get_indexer([entry_date], method='nearest')[0]
                        if 0 <= closest_idx < len(df_plot):
                            closest_date = df_plot.index[closest_idx]
                            ax1.scatter(closest_date, entry['price'], marker=marker_shape, color=marker_color, s=100, zorder=5)
                    except:
                        pass

        if markers_source and markers_source.get('exits'):
            for exit_trade in markers_source['exits']:
                exit_date = exit_trade['date']
                if isinstance(exit_date, str):
                    try:
                        exit_date = pd.to_datetime(exit_date)
                    except:
                        continue

                if exit_date < chart_start or exit_date > chart_end:
                    continue

                is_missed = exit_trade.get('missed', False)
                if is_missed:
                    color = 'orange'
                else:
                    # LONG-only: All exits are dark green (forest green) down triangles
                    color = '#228B22'  # Forest green for all long exits

                if exit_date in df_plot.index:
                    ax1.scatter(exit_date, exit_trade['price'], marker='v', color=color, s=100, zorder=5)
                else:
                    try:
                        closest_idx = df_plot.index.get_indexer([exit_date], method='nearest')[0]
                        if 0 <= closest_idx < len(df_plot):
                            closest_date = df_plot.index[closest_idx]
                            ax1.scatter(closest_date, exit_trade['price'], marker='v', color=color, s=100, zorder=5)
                    except:
                        pass

        # Entry line: Use current_position (trade_state) first, then locked_backtest, then fresh backtest
        # This ensures consistency with the locked backtest markers
        if current_position and current_position.get('entry_price'):
            ax1.axhline(current_position['entry_price'], color='cyan', linestyle='--', alpha=0.7,
                       label=f"Entry ${current_position['entry_price']:.2f}")
        elif locked_backtest and locked_backtest.get('current_position'):
            pos = locked_backtest['current_position']
            ax1.axhline(pos['entry_price'], color='cyan', linestyle='--', alpha=0.7, label=f"Entry ${pos['entry_price']:.2f}")
        elif backtest and backtest.get('current_position'):
            pos = backtest['current_position']
            ax1.axhline(pos['entry_price'], color='cyan', linestyle='--', alpha=0.7, label=f"Entry ${pos['entry_price']:.2f}")

        ax1.set_title(f"{ticker} - JD Strategy{title_suffix}", color='white', fontsize=14, fontweight='bold')
        ax1.set_ylabel("Price ($)", color='white')
        ax1.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white')
        if isinstance(df_plot.index, pd.DatetimeIndex):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.tick_params(axis='x', labelsize=8)

        # 2. JD Oscillator
        ax2 = axes[1]
        osc_col = 'osc_smooth' if 'osc_smooth' in df_plot.columns else 'composite_smooth'
        if osc_col in df_plot.columns:
            ax2.plot(df_plot.index, df_plot[osc_col], color='#e94560', linewidth=1.5, label='JD_Osc')
            ax2.axhline(config.get('oversold_threshold', -0.3), color='lime', linestyle='--', alpha=0.7, label='Oversold')
            ax2.axhline(config.get('overbought_threshold', 0.3), color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax2.fill_between(df_plot.index, df_plot[osc_col], 0,
                            where=(df_plot[osc_col] < config.get('oversold_threshold', -0.3)),
                            color='lime', alpha=0.3)
            ax2.fill_between(df_plot.index, df_plot[osc_col], 0,
                            where=(df_plot[osc_col] > config.get('overbought_threshold', 0.3)),
                            color='red', alpha=0.3)
        ax2.set_ylabel("JD_Osc", color='white')
        ax2.set_ylim(-1.2, 1.2)
        ax2.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white', fontsize='small')
        if isinstance(df_plot.index, pd.DatetimeIndex):
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator())

        # 3. JD Signal Indicators
        ax3 = axes[2]
        if 'velocity' in df_plot.columns:
            ax3.plot(df_plot.index, df_plot['velocity'], color='#00d9ff', linewidth=1.2, label='JD_Signal')
        if 'acceleration' in df_plot.columns:
            ax3.plot(df_plot.index, df_plot['acceleration'], color='#ffd700', linewidth=1.0, alpha=0.7, label='JD_Trend')
        ax3.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax3.set_ylabel("JD_Signal", color='white')
        ax3.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white', fontsize='small')
        if isinstance(df_plot.index, pd.DatetimeIndex):
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax3.xaxis.set_major_locator(mdates.MonthLocator())

        # 4. Equity Curve - Use locked_backtest for consistency with markers
        ax4 = axes[3]
        equity_source = locked_backtest if locked_backtest and locked_backtest.get('exits') else backtest
        if equity_source and equity_source.get('exits'):
            equity = [100]
            for exit in equity_source['exits']:
                equity.append(equity[-1] * (1 + exit['pnl']/100))

            if equity_source.get('current_position'):
                unrealized = equity_source['current_position'].get('unrealized_pnl', 0)
                if unrealized:
                    equity.append(equity[-1] * (1 + unrealized/100))

            ax4.plot(range(len(equity)), equity, color='#00ff88', linewidth=2)
            ax4.fill_between(range(len(equity)), 100, equity, alpha=0.3,
                           color='green' if equity[-1] > 100 else 'red')
            ax4.axhline(100, color='gray', linestyle='--', alpha=0.5)
        ax4.set_ylabel("Equity", color='white')
        ax4.set_xlabel("Trade #", color='white')

        # Stats annotation - Use locked_backtest for consistency with markers when available
        stats_source = locked_backtest if locked_backtest and locked_backtest.get('num_trades') else backtest
        if full_period_backtest and stats_source:
            full_stats = (f"Full Period: {full_period_backtest['num_trades']} trades | "
                         f"Win: {full_period_backtest['win_rate']:.0f}% | "
                         f"Return: {full_period_backtest['total_return']:.1f}% | "
                         f"PF: {full_period_backtest['profit_factor']:.1f}")
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
        bottom_margin = 0.14 if full_period_backtest else 0.12
        plt.subplots_adjust(bottom=bottom_margin)

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


def build_position_section(current_price: float, trade_state: dict, backtest: dict, config: dict) -> str:
    """Build position section text for Discord messages."""
    pos_section = "No active position"

    if trade_state and trade_state.get('position'):
        pos_type = trade_state.get('position', '').upper()
        entry_price = trade_state.get('entry_price', 0)
        entry_time_str = trade_state.get('entry_time', '')

        if pos_type == 'LONG':
            pnl = ((current_price - entry_price) / entry_price) * 100 if entry_price else 0
        else:
            pnl = ((entry_price - current_price) / entry_price) * 100 if entry_price else 0

        pnl_emoji = "+" if pnl >= 0 else ""

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
            except:
                hold_duration = "N/A"

        if pos_type == 'LONG':
            pnl_dollars = current_price - entry_price
        else:
            pnl_dollars = entry_price - current_price

        stop_loss_pct = config.get('stop_loss_pct', 5)
        take_profit_pct = config.get('take_profit_pct', 10)

        if pos_type == 'LONG':
            sl_price = entry_price * (1 - stop_loss_pct/100)
            tp_price = entry_price * (1 + take_profit_pct/100)
        else:
            sl_price = entry_price * (1 + stop_loss_pct/100)
            tp_price = entry_price * (1 - take_profit_pct/100)

        pos_section = (
            f"**Position:** {pos_type}\n"
            f"- Entry: ${entry_price:.2f} on {entry_time_str[:10] if entry_time_str else 'N/A'}\n"
            f"- Current: ${current_price:.2f} | **P&L: {pnl_emoji}{pnl:.2f}%** (${pnl_dollars:+,.0f})\n"
            f"- Duration: {hold_duration}\n"
            f"- SL: ${sl_price:.2f} | TP: ${tp_price:.2f}"
        )

    return pos_section


def build_stats_section(backtest: dict) -> str:
    """Build stats section text for Discord messages."""
    if not backtest:
        return "No backtest data available"

    return (
        f"**Stats:**\n"
        f"- Trades: {backtest.get('num_trades', 0)}\n"
        f"- Win Rate: {backtest.get('win_rate', 0):.0f}%\n"
        f"- Return: {backtest.get('total_return', 0):.1f}%\n"
        f"- Profit Factor: {backtest.get('profit_factor', 0):.1f}"
    )
