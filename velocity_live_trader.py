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
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import io
import shutil

# Import production utilities for safe operations
try:
    from velocity_production_utils import (
        safe_json_write, safe_json_read, get_logger,
        is_market_open, get_market_time, HeartbeatMonitor,
        load_webhook_from_env, print_signal_only_disclaimer,
        SIGNAL_ONLY_DISCLAIMER
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
}

# Strategy storage directories
VELOCITY_STRATEGIES_DIR = "velocity_strategies"
PRODUCTION_CONFIG_PATH = "production_env/velocity_config.json"


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

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

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
        except:
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

            # Check if there's an existing state file for this strategy and offer to reset
            ticker = selected['ticker']
            strat_name = selected.get('strategy_name', selected['name'])
            state_file = get_state_file_path(strategy_name=strat_name, ticker=ticker)
            if os.path.exists(state_file):
                try:
                    with open(state_file, 'r') as f:
                        existing_state = json.load(f)
                    if existing_state.get('position'):
                        print(f"\n⚠️  Existing position found for {strat_name}:")
                        print(f"   Position: {existing_state.get('position').upper()}")
                        print(f"   Entry: ${existing_state.get('entry_price', 0):.2f}")
                        print(f"   Time: {existing_state.get('entry_time', 'Unknown')}")
                        reset = input("\n   Reset trading state? (y/N): ").strip().lower()
                        if reset == 'y':
                            os.remove(state_file)
                            print(f"   ✅ State reset - will sync with backtest on startup")
                        else:
                            print(f"   ℹ️  Keeping existing position")
                except:
                    pass

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

    This prevents false signals from being detected on startup when the current
    day's bar is still forming.

    Edge cases handled:
    - Restart during market hours → exclude today's incomplete bar
    - Restart after market close same day → include today's completed bar
    - Restart before market open → include yesterday's completed bar
    - Restart on weekend/holiday → include last trading day's completed bar
    - Crypto restart any time during day → exclude today's incomplete bar
    - Crypto restart after midnight UTC (with 30-min buffer) → include yesterday's bar

    Returns True if the last bar should be excluded from backtest sync.
    """
    # Only apply to daily intervals
    if interval != "1d" or df.empty:
        return False

    # Determine if this is a crypto ticker
    is_crypto = '-USD' in ticker.upper() or ticker.upper() in ['BTC', 'ETH', 'BTCUSD', 'ETHUSD']

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
    # Check if this is a crypto ticker
    is_crypto = '-USD' in ticker or ticker in ['BTC', 'ETH', 'BTCUSD', 'ETHUSD']

    # For crypto, yfinance actually has more current data than Polygon (tested Jan 2026)
    # Polygon crypto daily bars have ~24hr delay, yfinance has same-day data
    # So we skip Polygon for crypto and use yfinance directly

    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    # Try to use cached data first (for non-crypto or if Polygon failed)
    if use_cache:
        try:
            from data_cache import fetch_and_cache
            df = fetch_and_cache(ticker, days=days, interval=interval)
            if not df.empty:
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


def fetch_realtime_price(ticker: str) -> float:
    """
    Fetch real-time/current price for display purposes.
    This is separate from daily bar data - used to show actual current price.

    For crypto: Uses Coinbase API (US-friendly, real-time) with Kraken/CoinGecko fallbacks
    For stocks: Uses yfinance
    """
    is_crypto = '-USD' in ticker or ticker in ['BTC', 'ETH', 'BTCUSD', 'ETHUSD']

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

    # Use smoothed oscillator
    osc_col = 'composite_smooth' if 'composite_smooth' in df.columns else 'composite_oscillator'

    # Apply smoothing
    if vel_smoothing > 1:
        df['osc_smooth'] = df[osc_col].rolling(window=vel_smoothing, center=False).mean()
        df['osc_smooth'] = df['osc_smooth'].bfill()
    else:
        df['osc_smooth'] = df[osc_col]

    # Calculate velocity (first derivative) and acceleration (second derivative)
    df['velocity'] = df['osc_smooth'].diff()
    df['acceleration'] = df['velocity'].diff()

    # Fill NaN values
    df['velocity'] = df['velocity'].fillna(0)
    df['acceleration'] = df['acceleration'].fillna(0)

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


def send_discord_alert(webhook_url: str, message: str, chart_buf: io.BytesIO = None,
                       include_disclaimer: bool = True, strategy_name: str = None):
    """Send alert to Discord webhook with optional chart image.

    Also sends to secondary Haus Hedge webhook if strategy_name is provided.
    """
    if not webhook_url:
        print(f"[ALERT] {message}")
        return False

    # Append legal disclaimer to all messages
    if include_disclaimer:
        message = message + LEGAL_DISCLAIMER

    def post_to_webhook(url: str, msg: str, chart: io.BytesIO = None) -> bool:
        """Helper to post to a single webhook."""
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

    # Send to primary webhook
    primary_success = post_to_webhook(webhook_url, message, chart_buf)

    # Send to secondary Haus Hedge webhook if strategy exists
    if strategy_name and strategy_name in HAUS_HEDGE_WEBHOOKS:
        secondary_url = HAUS_HEDGE_WEBHOOKS[strategy_name]
        # Reset chart buffer for second send
        if chart_buf:
            chart_buf.seek(0)
        secondary_success = post_to_webhook(secondary_url, message, chart_buf)
        if secondary_success:
            print(f"   📤 Also posted to Haus Hedge server")

    return primary_success


def generate_velocity_chart(df: pd.DataFrame, backtest: dict, config: dict, ticker: str,
                            title_suffix: str = "", trade_history: list = None,
                            current_position: dict = None, locked_backtest: dict = None,
                            full_period_backtest: dict = None) -> io.BytesIO:
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
    """
    try:
        # Use all available data (full test period)
        df_plot = df.copy()

        # Ensure we have a proper datetime index for plotting
        if not isinstance(df_plot.index, pd.DatetimeIndex):
            # Try to convert index to datetime if it's not already
            try:
                df_plot.index = pd.to_datetime(df_plot.index)
            except:
                # If conversion fails, reset to sequential integers
                df_plot = df_plot.reset_index(drop=True)

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

        # 1. Price Chart with Entry/Exit markers
        ax1 = axes[0]
        ax1.plot(df_plot.index, df_plot['close'], color='white', linewidth=1.5, label='Price')
        ax1.fill_between(df_plot.index, df_plot['low'], df_plot['high'], color='gray', alpha=0.2)

        # Plot trade markers from LOCKED backtest (prevents repainting)
        # The locked_backtest is frozen at startup and only appended to when new signals occur.
        # This ensures historical markers never shift position.
        markers_source = locked_backtest if locked_backtest else backtest

        # Get chart date range for filtering markers
        chart_start = df_plot.index.min()
        chart_end = df_plot.index.max()

        if markers_source and markers_source.get('entries'):
            for entry in markers_source['entries']:
                # Handle both datetime objects and strings
                entry_date = entry['date']
                if isinstance(entry_date, str):
                    try:
                        entry_date = pd.to_datetime(entry_date)
                    except:
                        continue

                # SKIP markers outside the chart's date range (prevents clustering at edges)
                if entry_date < chart_start or entry_date > chart_end:
                    continue

                # LONG entries: green up triangle, SHORT entries: red down triangle
                # MISSED entries: orange color
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
                    # Try to find the closest date within range
                    try:
                        closest_idx = df_plot.index.get_indexer([entry_date], method='nearest')[0]
                        if 0 <= closest_idx < len(df_plot):
                            closest_date = df_plot.index[closest_idx]
                            ax1.scatter(closest_date, entry['price'], marker=marker_shape, color=marker_color, s=100, zorder=5)
                    except:
                        pass

        if markers_source and markers_source.get('exits'):
            for exit_trade in markers_source['exits']:
                # Handle both datetime objects and strings
                exit_date = exit_trade['date']
                if isinstance(exit_date, str):
                    try:
                        exit_date = pd.to_datetime(exit_date)
                    except:
                        continue

                # SKIP markers outside the chart's date range (prevents clustering at edges)
                if exit_date < chart_start or exit_date > chart_end:
                    continue

                # MISSED exits: orange color
                is_missed = exit_trade.get('missed', False)
                if is_missed:
                    color = 'orange'
                else:
                    # LONG-only: All exits are dark green (forest green) down triangles
                    color = '#228B22'  # Forest green for all long exits

                if exit_date in df_plot.index:
                    ax1.scatter(exit_date, exit_trade['price'], marker='v', color=color, s=100, zorder=5)
                else:
                    # Try to find the closest date within range
                    try:
                        closest_idx = df_plot.index.get_indexer([exit_date], method='nearest')[0]
                        if 0 <= closest_idx < len(df_plot):
                            closest_date = df_plot.index[closest_idx]
                            ax1.scatter(closest_date, exit_trade['price'], marker='v', color=color, s=100, zorder=5)
                    except:
                        pass

        # Mark current open position - use LOCKED trade_state (current_position) if provided
        # This ensures the entry line shows YOUR ACTUAL tracked position, not the recalculated one
        # Priority: current_position (trade_state) > locked_backtest > fresh backtest
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
        # Format x-axis with dates
        if isinstance(df_plot.index, pd.DatetimeIndex):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.tick_params(axis='x', labelsize=8)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)

        # 2. JD Oscillator with thresholds
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
        # Format x-axis with dates
        if isinstance(df_plot.index, pd.DatetimeIndex):
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator())
            ax2.tick_params(axis='x', labelsize=8)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

        # 3. JD Signal Indicators
        ax3 = axes[2]
        if 'velocity' in df_plot.columns:
            ax3.plot(df_plot.index, df_plot['velocity'], color='#00d9ff', linewidth=1.2, label='JD_Signal')
        if 'acceleration' in df_plot.columns:
            ax3.plot(df_plot.index, df_plot['acceleration'], color='#ffd700', linewidth=1.0, alpha=0.7, label='JD_Trend')
        ax3.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax3.set_ylabel("JD_Signal", color='white')
        ax3.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white', fontsize='small')
        # Format x-axis with dates
        if isinstance(df_plot.index, pd.DatetimeIndex):
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax3.xaxis.set_major_locator(mdates.MonthLocator())
            ax3.tick_params(axis='x', labelsize=8)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

        # 4. Equity Curve - Use backtest (period-specific) for equity to match displayed timeframe
        # Markers come from locked_backtest (filtered to chart range), but equity shows period stats
        ax4 = axes[3]
        equity_source = backtest if backtest and backtest.get('exits') else locked_backtest
        if equity_source and equity_source.get('exits'):
            # Build simple equity curve from trades
            equity = [100]
            for exit in equity_source['exits']:
                equity.append(equity[-1] * (1 + exit['pnl']/100))

            # Add unrealized if open position (use .get() to handle missing key)
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

        # Stats annotation - Use backtest (period-specific) to match displayed timeframe
        # Markers come from locked_backtest (filtered), but stats should reflect visible period
        stats_source = backtest if backtest and backtest.get('num_trades') else locked_backtest
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


def run_historical_backtest(df: pd.DataFrame, config: dict) -> dict:
    """
    Run a backtest on historical data to determine current state.
    Returns dict with trades, stats, and current position.
    """
    # Calculate signals
    df = calculate_velocity_signals(df, config)

    # Get config params
    stop_loss_pct = config.get('stop_loss_pct', 5.0)
    take_profit_pct = config.get('take_profit_pct', 10.0)
    min_bars_between = config.get('min_bars_between', 1)
    exit_on_opposite = config.get('exit_on_opposite_signal', True)
    exit_on_midline = config.get('exit_on_midline_cross', False)

    # Run simulation
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
            tracked_entry_date = pd.to_datetime(tracked_entry_str)
            print(f"   📍 Tracked position: {trade_state['position'].upper()} @ ${trade_state.get('entry_price', 0):.2f} on {tracked_entry_date}")
        except:
            pass

    # Add NEW entries from fresh backtest (only entries AFTER latest exit date)
    new_entries_added = 0
    skipped_overlap = 0
    for entry in backtest.get('entries', []):
        entry_date = entry['date']
        entry_date_str = str(entry_date)[:10]

        # Only add entries AFTER the latest exit in locked backtest
        # This prevents duplicates from overlapping data periods
        if latest_exit_date and entry_date_str <= latest_exit_date:
            continue

        # Parse date for comparison with tracked position
        if isinstance(entry_date, str):
            try:
                entry_date = pd.to_datetime(entry_date)
            except:
                pass

        # Skip entries that conflict with tracked position
        if tracked_entry_date is not None and hasattr(entry_date, 'date'):
            if entry_date >= tracked_entry_date:
                print(f"   ⏭️  Skipping backtest entry {entry_date} (conflicts with tracked position)")
                continue

        # VALIDATION: Check if there's already an unexited entry in locked backtest
        # We can't enter a new position if we're already in one
        existing_entry_dates = {str(e.get('date', ''))[:10] for e in locked["entries"]}
        existing_exit_entry_dates = {str(e.get('entry_date', ''))[:10] for e in locked["exits"]}
        unexited_entries = existing_entry_dates - existing_exit_entry_dates

        if unexited_entries:
            # There's at least one unexited entry - check if this new entry would overlap
            # Get the latest unexited entry date
            latest_unexited = max(unexited_entries)
            if entry_date_str > latest_unexited:
                # This entry is after an unexited entry - skip it (can't enter while in position)
                skipped_overlap += 1
                continue

        locked["entries"].append({
            "date": str(entry['date']),
            "price": entry['price'],
            "position": entry.get('position', 'long')
        })
        new_entries_added += 1

    if new_entries_added > 0:
        print(f"   ➕ Added {new_entries_added} new entries (after {latest_exit_date})")
    if skipped_overlap > 0:
        print(f"   ⏭️  Skipped {skipped_overlap} entries (already in position)")

    # Add NEW exits from fresh backtest (only exits AFTER latest exit date)
    new_exits_added = 0
    skipped_same_day = 0
    skipped_duplicate = 0
    for exit_trade in backtest.get('exits', []):
        exit_date_str = str(exit_trade['date'])[:10]
        entry_date_str = str(exit_trade.get('entry_date', ''))[:10]

        # Only add exits AFTER the latest exit in locked backtest
        if latest_exit_date and exit_date_str <= latest_exit_date:
            continue

        # VALIDATION 1: Reject same-day entry+exit (can't trade on same bar)
        if exit_date_str == entry_date_str:
            skipped_same_day += 1
            continue

        # VALIDATION 2: Reject duplicate exits (entry already has an exit)
        entry_already_exited = any(
            str(e.get('entry_date', ''))[:10] == entry_date_str and
            abs(e.get('entry_price', 0) - exit_trade.get('entry_price', 0)) < 0.01
            for e in locked["exits"]
        )
        if entry_already_exited:
            skipped_duplicate += 1
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

    # Handle tracked position - add or update
    if trade_state and trade_state.get('position') and trade_state.get('entry_price'):
        tracked_entry_str = str(trade_state.get('entry_time', ''))[:10]

        # Remove any existing entries that match tracked position date (to avoid duplicates)
        locked["entries"] = [e for e in locked["entries"] if str(e.get('date', ''))[:10] != tracked_entry_str]

        # Add the tracked position as entry
        locked["entries"].append({
            "date": str(trade_state.get('entry_time', '')),
            "price": trade_state['entry_price'],
            "position": trade_state['position']
        })
        print(f"   ✅ Synced tracked {trade_state['position'].upper()} entry in locked backtest")

        # Update current_position
        locked["current_position"] = {
            "position": trade_state['position'],
            "entry_price": trade_state['entry_price'],
            "entry_date": str(trade_state.get('entry_time', ''))
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
        locked["profit_factor"] = win_pnl / loss_pnl if loss_pnl > 0 else float('inf') if win_pnl > 0 else 0

    with open(path, 'w') as f:
        json.dump(locked, f, indent=2, default=str)

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


def append_to_locked_backtest(entry: dict = None, exit_trade: dict = None,
                               strategy_name: str = None, ticker: str = None,
                               missed: bool = False):
    """Append a new entry or exit to the locked backtest.

    Called when a NEW signal is detected (at end of day). This adds
    the signal to the locked backtest so it appears on future charts.

    Args:
        entry: Entry signal dict with date, price, position
        exit_trade: Exit trade dict with date, price, pnl, reason, etc.
        strategy_name: Strategy name for file path
        ticker: Ticker symbol for file path
        missed: If True, marks the signal as missed (bot was down)
    """
    locked = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)

    if locked is None:
        print("   ⚠️ No locked backtest to append to")
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
        label = "🟠 missed" if missed else "🔒"
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
            locked["profit_factor"] = win_pnl / loss_pnl if loss_pnl > 0 else (float('inf') if win_pnl > 0 else 0)
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
    with open(path, 'w') as f:
        json.dump(locked, f, indent=2, default=str)


def detect_and_add_missed_signals(fresh_backtest: dict, strategy_name: str = None, ticker: str = None) -> int:
    """Detect signals that occurred while bot was down and add them as missed.

    Compares fresh backtest entries/exits with locked_backtest to find signals
    that occurred after the last tracked signal. Adds them with missed=True.

    Returns: Number of missed signals detected and added.
    """
    locked = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)
    if locked is None or fresh_backtest is None:
        return 0

    # Find the last tracked entry date in locked backtest
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

    # Get set of existing entry dates to avoid duplicates
    existing_entry_dates = set()
    for entry in locked_entries:
        if entry.get('date'):
            existing_entry_dates.add(str(entry['date'])[:10])

    existing_exit_dates = set()
    for exit_t in locked_exits:
        if exit_t.get('date'):
            existing_exit_dates.add(str(exit_t['date'])[:10])

    missed_count = 0

    # Check fresh backtest entries for missed signals
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

    # Check fresh backtest exits for missed signals
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

    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            return json.load(f)
    return {
        "position": None,  # None or "long" (LONG-only strategy)
        "entry_price": None,
        "entry_time": None,
        "entry_signal_bar": None,  # Bar datetime that generated entry signal (for locked charts)
        "last_signal_time": None,
    }


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

    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2, default=str)


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
        except:
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

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)


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
                    except:
                        hold_duration = "N/A"

                # Calculate actual dollar P&L per unit (e.g., per 1 BTC or 1 share)
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
        chart_buf_full = generate_velocity_chart(df, backtest, config, ticker,
                                                  title_suffix=f" (Full: {full_days} bars)",
                                                  trade_history=trade_history,
                                                  current_position=trade_state,
                                                  locked_backtest=locked_backtest)
        pos_section = build_position_section(current_price, trade_state, backtest, config)

        # Check for stale data warning (data > 1 day behind)
        data_stale_warning = ""
        try:
            data_end_date = df.index[-1].date() if hasattr(df.index[-1], 'date') else datetime.strptime(full_end, '%Y-%m-%d').date()
            days_behind = (datetime.now().date() - data_end_date).days
            if days_behind > 1:
                data_stale_warning = f"\n⚠️ _Data is {days_behind} days behind (yfinance delay)_"
        except:
            pass

        msg_full = (
            f"**{title}: {strategy_label}** [1/2 Full Timeframe]\n"
            f"**Period:** {full_start} to {full_end} ({full_days} bars)\n"
            f"---\n"
            f"{pos_section}\n"
            f"---\n"
            f"📊 **Full Period Stats:**\n"
            f"• Trades: {backtest['num_trades']} | Win Rate: {backtest['win_rate']:.0f}%\n"
            f"• Return: {backtest['total_return']:.1f}% | PF: {backtest['profit_factor']:.1f}\n"
            f"---\n"
            f"_Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_{data_stale_warning}"
        )

        send_discord_alert(webhook_url, msg_full, chart_buf_full, strategy_name=strategy_name)
        print(f"✅ Sent full timeframe update: {title}")

        # --- CHART 2: Recent Period (126 days = half trading year) ---
        # Use SUBSET of bundled data to prevent repainting (NOT fresh data)
        min_bars_for_subset = 20  # Need at least 20 bars to make a meaningful subset
        subset_days = 126  # Half a trading year

        if len(df) > min_bars_for_subset:
            if len(df) > subset_days:
                period_label = f"Last {subset_days} Days"
            else:
                subset_days = len(df) // 2
                period_label = f"Last {subset_days} Days"

            # Use SUBSET of bundled df (NOT run_backtest_for_period which causes repainting)
            df_subset = df.iloc[-subset_days:].copy()

            # Filter locked_backtest to get stats and exits for just the subset period
            subset_start = df_subset.index[0]
            subset_exits = []
            if locked_backtest and locked_backtest.get('exits'):
                for exit_trade in locked_backtest['exits']:
                    if not exit_trade.get('missed'):
                        exit_date = pd.to_datetime(exit_trade['date'])
                        if exit_date >= subset_start:
                            subset_exits.append(exit_trade)

            # Build backtest_subset with exits for equity curve
            backtest_subset = {
                'num_trades': len(subset_exits),
                'win_rate': 0,
                'total_return': 0,
                'profit_factor': 0,
                'exits': subset_exits,  # Include exits for equity curve
                'current_position': locked_backtest.get('current_position') if locked_backtest else None,
                'period_days': subset_days  # Store the period for label
            }
            if subset_exits:
                winners = [e for e in subset_exits if e['pnl'] > 0]
                losers = [e for e in subset_exits if e['pnl'] <= 0]
                backtest_subset['win_rate'] = (len(winners) / len(subset_exits)) * 100
                # Use COMPOUNDED return to match equity curve
                sorted_exits = sorted(subset_exits, key=lambda x: str(x.get('date', '')))
                equity = 1.0
                for exit_trade in sorted_exits:
                    equity *= (1 + exit_trade['pnl'] / 100)
                backtest_subset['total_return'] = (equity - 1) * 100
                total_wins = sum(e['pnl'] for e in winners) if winners else 0
                total_losses = abs(sum(e['pnl'] for e in losers)) if losers else 0.001
                backtest_subset['profit_factor'] = total_wins / total_losses if total_losses > 0 else total_wins

            # Calculate date range for subset window
            period_start = df_subset.index[0].strftime('%Y-%m-%d') if hasattr(df_subset.index[0], 'strftime') else str(df_subset.index[0])[:10]
            period_end = df_subset.index[-1].strftime('%Y-%m-%d') if hasattr(df_subset.index[-1], 'strftime') else str(df_subset.index[-1])[:10]

            chart_buf_subset = generate_velocity_chart(df_subset, backtest_subset, config, ticker,
                                                        title_suffix=f" ({period_label})",
                                                        trade_history=trade_history,
                                                        current_position=trade_state,
                                                        locked_backtest=locked_backtest)

            # Position section stays the same (current position)
            pos_section_subset = build_position_section(current_price, trade_state, backtest_subset, config)

            # Check for stale data warning (data > 1 day behind)
            data_stale_warning = ""
            try:
                data_end_date = df_subset.index[-1].date() if hasattr(df_subset.index[-1], 'date') else datetime.strptime(period_end, '%Y-%m-%d').date()
                days_behind = (datetime.now().date() - data_end_date).days
                if days_behind > 1:
                    data_stale_warning = f"\n⚠️ _Data is {days_behind} days behind (yfinance delay)_"
            except:
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
                f"_Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_{data_stale_warning}"
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
            print(f"   ✓ Loaded {len(df)} bars: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"   ✓ Using EXACT same data as Streamlit!")
        else:
            # Fetch fresh data via yfinance (fallback)
            print(f"   Fetching fresh data via yfinance...")
            df = fetch_price_data(ticker, api_key, days=backtest_days, interval=interval)

        df = calculate_composite_oscillator(df, config)
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

    # FETCH FRESH DATA FOR SYNC - bundled data may be stale!
    print(f"\n🔄 Fetching FRESH data for state sync check...")
    fresh_backtest = None
    fresh_df = pd.DataFrame()  # Initialize empty to prevent NameError
    excluded_incomplete_bar = False  # Track if we excluded a bar
    try:
        import time
        time.sleep(1)  # Small delay to avoid rate limiting
        fresh_df = fetch_price_data(ticker, api_key, days=200, interval=interval)
        if not fresh_df.empty:
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

    # SYNC STATE WITH BACKTEST - handles missed entries/exits
    backtest_position = sync_backtest.get('current_position') if sync_backtest else None
    state_position = trade_state.get('position')

    # Case 1: No state but backtest shows position - missed entry
    if state_position is None and backtest_position:
        pos = backtest_position
        trade_state = {
            'position': pos['position'],
            'entry_price': pos['entry_price'],
            'entry_time': pos['entry_date'],
            'last_signal_time': pos['entry_date'],
        }
        save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)
        print(f"✅ Initialized position from backtest: LONG @ ${pos['entry_price']:.2f}")

    # Case 2: State shows position but backtest shows none - missed exit
    elif state_position is not None and backtest_position is None:
        old_entry = trade_state.get('entry_price', 0)
        old_time = trade_state.get('entry_time', 'Unknown')
        print(f"⚠️  STATE MISMATCH: State={state_position.upper()} @ ${old_entry:.2f}, Backtest=None")
        print(f"   🔄 Missed exit detected - clearing state to sync with backtest")

        # Find and record the missed trade from backtest exits
        missed_trade = None
        if sync_backtest and sync_backtest.get('exits'):
            # Parse entry time for date comparison
            entry_date_str = str(old_time)[:10] if old_time else None

            for exit_trade in sync_backtest['exits']:
                # Match by entry price (within 0.5% tolerance)
                if abs(exit_trade.get('entry_price', 0) - old_entry) / old_entry < 0.005:
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
                entry_price=old_entry,
                exit_price=missed_trade.get('price', 0),
                entry_time=str(old_time),
                exit_time=str(missed_trade.get('date', '')),
                exit_reason=f"[SYNC] {missed_trade.get('reason', 'Unknown')}",
                pnl_pct=missed_trade.get('pnl', 0),
                strategy_name=strategy_name,
                entry_signal_bar=trade_state.get('entry_signal_bar'),  # Use stored signal bar if available
                exit_signal_bar=str(missed_trade.get('date', ''))  # Exit bar from backtest
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
        except:
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
        # Existing locked backtest - use fresh_backtest to add new signals only
        # (save_locked_backtest will preserve existing data and only append new)
        print(f"\n🔒 Updating locked backtest with any new signals...")
        if fresh_backtest:
            locked_backtest = save_locked_backtest(fresh_backtest, strategy_name=strategy_name, ticker=ticker, trade_state=trade_state)
        else:
            locked_backtest = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)
            print(f"🔒 Loaded existing locked backtest from previous session")
    else:
        # FIRST STARTUP - seed from bundled backtest (full historical data)
        # This ensures we capture all historical trades, not just last 200 days
        if backtest:
            print(f"\n🔒 Creating initial locked backtest from full historical data...")
            locked_backtest = save_locked_backtest(backtest, strategy_name=strategy_name, ticker=ticker, trade_state=trade_state)
            print(f"📁 Locked backtest file: {locked_backtest_path}")
        elif fresh_backtest:
            print(f"\n🔒 Creating initial locked backtest from recent data...")
            locked_backtest = save_locked_backtest(fresh_backtest, strategy_name=strategy_name, ticker=ticker, trade_state=trade_state)
            print(f"📁 Locked backtest file: {locked_backtest_path}")

    # Detect missed signals during downtime
    if locked_backtest and fresh_backtest:
        num_missed = detect_and_add_missed_signals(fresh_backtest, strategy_name=strategy_name, ticker=ticker)
        if num_missed > 0:
            print(f"🟠 Found {num_missed} missed signal(s) during downtime - marked in orange on charts")
            # Reload the locked backtest to get the updated version with missed signals
            locked_backtest = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)

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
        stats_source = locked_backtest if locked_backtest else backtest
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

        # Calculate Recent 126 Days stats (half trading year) from locked_backtest
        # This matches the chart's "Last 126 Days" period
        # IMPORTANT: Use 126 TRADING days (rows), not calendar days, to match chart
        recent_stats = ""
        recent_days = 126
        if locked_backtest and locked_backtest.get('exits'):
            # Get the chart date range using same logic as chart generation
            # Use the last 126 rows of price data (trading days) for consistency
            chart_df = fresh_df if fresh_backtest and not fresh_df.empty else df
            if len(chart_df) > recent_days:
                df_recent_for_stats = chart_df.iloc[-recent_days:]
            else:
                df_recent_for_stats = chart_df
            chart_start = df_recent_for_stats.index.min()
            chart_end = df_recent_for_stats.index.max()

            # Filter exits to 126-trading-day window (matches chart exactly)
            recent_exits = []
            for exit_trade in locked_backtest['exits']:
                exit_date = exit_trade.get('date', '')
                if isinstance(exit_date, str):
                    try:
                        exit_date = pd.to_datetime(exit_date)
                    except:
                        continue
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
                    f"\n📈 **Recent {recent_days} Days ({len(recent_exits)} trades):**\n"
                    f"• Win Rate: {win_rate:.0f}% | Return: {total_return:.1f}%\n"
                    f"• Profit Factor: {profit_factor:.1f}"
                )

        startup_msg = (
            f"🤖 **[{strategy_label}] Live Trader Started**\n"
            f"**Strategy:** {strategy_name}\n"
            f"**Ticker:** {ticker}\n"
            f"**Current Price:** ${current_price:,.2f}\n"
            f"**Signal Type:** {config.get('signal_type')}\n"
            f"**Risk:** SL={stop_loss_pct:.1f}%, TP={take_profit_pct:.1f}%\n"
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
            # If fresh data is available and more recent, use it for charts
            # Otherwise fall back to bundled data (charts may not show recent positions)
            chart_df = df  # Default to bundled
            chart_data_source = "bundled"  # Track source for accurate logging
            try:
                if fresh_backtest and fresh_df is not None and len(fresh_df) > 0:
                    bundled_end = df.index[-1] if len(df) > 0 else None
                    fresh_end = fresh_df.index[-1]
                    if bundled_end and fresh_end > bundled_end:
                        chart_df = fresh_df
                        chart_data_source = "fresh"
                        print(f"📊 Using FRESH data for charts (bundled ends {bundled_end.strftime('%Y-%m-%d')}, fresh ends {fresh_end.strftime('%Y-%m-%d')})")
                    else:
                        print(f"📊 Using bundled data for charts (up to {bundled_end.strftime('%Y-%m-%d') if bundled_end else 'unknown'})")
                else:
                    print(f"📊 Using bundled data for charts (fresh data not available)")
            except Exception as e:
                print(f"📊 Using bundled data for charts (error checking fresh: {e})")

            # FULL PERIOD chart - use all available data
            full_period_chart = generate_velocity_chart(chart_df, backtest, config, ticker,
                                                        title_suffix=" - Full Period",
                                                        trade_history=startup_trade_history,
                                                        current_position=trade_state,
                                                        locked_backtest=locked_backtest,
                                                        full_period_backtest=backtest)
            print("✅ Generated full period chart")

            # RECENT 126 days chart - subset of chart data
            recent_days = 126  # Half a trading year
            if len(chart_df) > recent_days:
                df_recent = chart_df.iloc[-recent_days:].copy()
            else:
                df_recent = chart_df.copy()

            # Calculate subset stats from locked_backtest for the recent period
            recent_start = df_recent.index.min()
            recent_exits = []
            if locked_backtest and locked_backtest.get('exits'):
                for exit_trade in locked_backtest['exits']:
                    if not exit_trade.get('missed'):
                        exit_date = pd.to_datetime(exit_trade['date'])
                        if exit_date >= recent_start:
                            recent_exits.append(exit_trade)

            # Build recent_backtest with exits for equity curve
            recent_backtest = {
                'num_trades': len(recent_exits),
                'win_rate': 0,
                'total_return': 0,
                'profit_factor': 0,
                'exits': recent_exits,  # Include exits for equity curve
                'current_position': locked_backtest.get('current_position') if locked_backtest else None,
                'period_days': recent_days  # Store the period for label
            }
            if recent_exits:
                winners = [e for e in recent_exits if e['pnl'] > 0]
                losers = [e for e in recent_exits if e['pnl'] <= 0]
                recent_backtest['win_rate'] = (len(winners) / len(recent_exits)) * 100
                # Use COMPOUNDED return to match equity curve
                sorted_exits = sorted(recent_exits, key=lambda x: str(x.get('date', '')))
                equity = 1.0
                for exit_trade in sorted_exits:
                    equity *= (1 + exit_trade['pnl'] / 100)
                recent_backtest['total_return'] = (equity - 1) * 100
                total_wins = sum(e['pnl'] for e in winners) if winners else 0
                total_losses = abs(sum(e['pnl'] for e in losers)) if losers else 0.001
                recent_backtest['profit_factor'] = total_wins / total_losses if total_losses > 0 else total_wins

            recent_chart = generate_velocity_chart(df_recent, recent_backtest, config, ticker,
                                                   title_suffix=f" - Last {recent_days} Days",
                                                   trade_history=startup_trade_history,
                                                   current_position=trade_state,
                                                   locked_backtest=locked_backtest,
                                                   full_period_backtest=locked_backtest)
            print(f"✅ Generated recent {recent_days}-day chart ({chart_data_source} data, {recent_backtest['num_trades']} trades)")

            # Use recent chart for main display (full period chart can be added later)
            startup_chart = recent_chart
        except Exception as e:
            print(f"⚠️ Could not generate startup chart: {e}")
            import traceback
            traceback.print_exc()
            startup_chart = None
    else:
        startup_msg = (
            f"🤖 **[{strategy_label}] Live Trader Started**\n"
            f"**Strategy:** {strategy_name}\n"
            f"**Ticker:** {ticker}\n"
            f"**Interval:** {interval}\n"
            f"**Signal Type:** {config.get('signal_type')}\n"
            f"**Risk:** SL={stop_loss_pct}%, TP={take_profit_pct}%\n"
            f"---\n"
            f"_Monitoring for signals..._"
        )
    send_discord_alert(webhook_url, startup_msg, startup_chart, strategy_name=strategy_name)

    # Determine check interval based on data interval
    # Check frequently enough to catch scheduled update windows
    if interval == "1d":
        check_interval_seconds = 900  # Check every 15 min for daily signals (to catch scheduled updates)
    elif interval == "1h":
        check_interval_seconds = 300  # Check every 5 min for hourly signals
    else:
        check_interval_seconds = 60  # Check every minute

    # Track scheduled alerts to avoid duplicates
    daily_alerts = set()

    # Track the last completed bar we evaluated for signals (prevents re-evaluation)
    last_signal_bar_evaluated = None

    # Determine if this is a crypto or stock ticker for signal timing
    is_crypto = ticker.upper() in ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD'] or '-USD' in ticker.upper()

    # STARTUP CATCH-UP: If crypto and we just missed a scheduled update window, send immediately
    # This handles cases where trader started at 6:51 AM but 6:00 AM window was missed
    if is_crypto and backtest:
        startup_now = datetime.now()
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

    while True:
        try:
            current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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

                        # Update key parameters
                        ticker = config.get('ticker', ticker)
                        interval = config.get('interval', interval)
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

            # Fetch data
            df = fetch_price_data(ticker, api_key, days=200, interval=interval)

            if df.empty:
                print("No data received. Waiting...")
                time.sleep(check_interval_seconds)
                continue

            # Calculate oscillator and signals
            df = calculate_composite_oscillator(df, config)
            df = calculate_velocity_signals(df, config)

            # Get latest row
            latest = df.iloc[-1]
            bar_close_price = latest['close']  # Daily bar close (for signal logic)
            current_time = df.index[-1]

            # Fetch real-time price for display (separate from daily bar)
            realtime_price = fetch_realtime_price(ticker)
            current_price = realtime_price if realtime_price else bar_close_price

            print(f"Current Price: ${current_price:.2f} (real-time)" if realtime_price else f"Current Price: ${current_price:.2f} (bar close)")
            print(f"Oscillator: {latest['osc_smooth']:.4f}")
            print(f"Velocity: {latest['velocity']:.4f}")
            print(f"Acceleration: {latest['acceleration']:.4f}")
            print(f"Buy Signal: {latest['buy_signal']}")
            print(f"Sell Signal: {latest['sell_signal']}")
            print(f"Position: {trade_state['position']}")

            # --- SCHEDULED STATUS UPDATES ---
            # Use Eastern Time for stocks (SPY market hours), UTC for crypto
            if is_crypto:
                now = datetime.now()  # Local time is fine for 24/7 crypto
            else:
                now = get_market_time()  # Eastern Time for stock market hours
            day_str = now.strftime("%Y-%m-%d")
            hm = now.strftime("%H:%M")
            hour = now.hour
            minute = now.minute

            # Run backtest for status updates
            status_backtest = None
            try:
                status_backtest = run_historical_backtest(df, config)
            except Exception as e:
                print(f"   ⚠️ Backtest failed: {e}")

            # Check for position sync issues - warn if backtest shows position but we're not tracking
            if status_backtest and status_backtest.get('current_position'):
                bt_pos = status_backtest['current_position']
                if not trade_state.get('position'):
                    print(f"   ⚠️  SYNC WARNING: Backtest shows {bt_pos.get('position', 'unknown').upper()} @ ${bt_pos.get('entry_price', 0):.2f}")
                    print(f"   ⚠️  But trade_state has no position! Consider restarting to sync.")

            # Market Open Update (8:30-8:45 AM ET for stocks)
            if "08:30" <= hm <= "08:45" and not is_crypto:
                key = f"{day_str}_OPEN"
                if key not in daily_alerts and status_backtest:
                    send_status_update(webhook_url, df, status_backtest, config, ticker,
                                     "🔔 Market Open Update", trade_state, strategy_name=strategy_name,
                                     realtime_price=realtime_price)
                    daily_alerts.add(key)

            # Mid-Day Update (11:00-11:15 AM ET for stocks)
            if "11:00" <= hm <= "11:15" and not is_crypto:
                key = f"{day_str}_MID"
                if key not in daily_alerts and status_backtest:
                    send_status_update(webhook_url, df, status_backtest, config, ticker,
                                     "☀️ Mid-Day Update", trade_state, strategy_name=strategy_name,
                                     realtime_price=realtime_price)
                    daily_alerts.add(key)

            # Afternoon Update (15:00-15:15 PM ET for stocks - 1 hour before close)
            if "15:00" <= hm <= "15:15" and not is_crypto:
                key = f"{day_str}_AFTERNOON"
                if key not in daily_alerts and status_backtest:
                    send_status_update(webhook_url, df, status_backtest, config, ticker,
                                     "🌅 Afternoon Update", trade_state, strategy_name=strategy_name,
                                     realtime_price=realtime_price)
                    daily_alerts.add(key)

            # Market Close Update (16:00-16:15 ET for stocks - actual market close time)
            if "16:00" <= hm <= "16:15" and not is_crypto:
                key = f"{day_str}_CLOSE"
                if key not in daily_alerts and status_backtest:
                    send_status_update(webhook_url, df, status_backtest, config, ticker,
                                     "🏁 Market Close Update", trade_state, strategy_name=strategy_name,
                                     realtime_price=realtime_price)
                    daily_alerts.add(key)

            # Hourly Updates (9:00 - 14:00 ET for stocks, except 11 which has special alert)
            # Window is first 15 minutes of each hour to ensure it gets hit
            # Excludes 15 (afternoon) and 16 (close) which have dedicated updates
            if 9 <= hour <= 14 and minute < 15 and not is_crypto:
                if hour != 11:
                    key = f"{day_str}_HOUR_{hour}"
                    if key not in daily_alerts and status_backtest:
                        send_status_update(webhook_url, df, status_backtest, config, ticker,
                                         f"⏱️ {hour}:00 ET Market Update", trade_state, strategy_name=strategy_name,
                                         realtime_price=realtime_price)
                        daily_alerts.add(key)

            # END OF DAY Update (16:30-17:00 ET for stocks - after market close + data finalization)
            # This is the ACTUAL end of day summary, sent after market closes and data settles
            if "16:30" <= hm <= "17:00" and not is_crypto:
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

            # Check for exit conditions first (if in position)
            if trade_state['position'] == 'long':
                entry_price = trade_state['entry_price']
                pnl_pct = ((current_price - entry_price) / entry_price) * 100

                exit_reason = None

                # Stop loss - uses real-time price (can trigger anytime)
                if pnl_pct <= -stop_loss_pct:
                    exit_reason = f"Stop Loss ({pnl_pct:.2f}%)"
                # Take profit - uses real-time price (can trigger anytime)
                elif pnl_pct >= take_profit_pct:
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
                            utc_now = datetime.utcnow()
                            utc_hour = utc_now.hour
                            utc_minute = utc_now.minute
                            # Check between 00:30 UTC and 23:59 UTC (avoid checking right at midnight)
                            if utc_hour == 0 and utc_minute < 30:
                                can_check_signal_exits = False
                                print(f"   ⏳ [CRYPTO] Waiting for new daily bar (UTC: {utc_now.strftime('%H:%M')})")
                        else:
                            # Stocks (SPY): Daily bar closes at 4 PM ET
                            market_time = get_market_time()
                            if market_time.hour < 16:
                                can_check_signal_exits = False
                    else:
                        signal_bar = latest  # Use latest for intraday
                        can_check_signal_exits = True

                    if can_check_signal_exits:
                        # Exit on opposite signal
                        if exit_on_opposite_signal and signal_bar['sell_signal']:
                            exit_reason = f"Opposite Signal ({pnl_pct:.2f}%)"
                        # Exit on midline cross (oscillator crosses above 0 = bearish for long)
                        elif exit_on_midline_cross and signal_bar['osc_smooth'] > 0:
                            exit_reason = f"Midline Cross ({pnl_pct:.2f}%)"

                if exit_reason:
                    # Generate chart and get stats for exit
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
                            'date': current_time,
                            'price': current_price,
                            'pnl': pnl_pct,
                            'reason': exit_reason,
                            'entry_price': entry_price,
                        })
                        # Clear current_position since we're exiting
                        locked_backtest_with_exit['current_position'] = None

                        # Use locked backtest WITH current exit for markers
                        exit_chart = generate_velocity_chart(df, exit_backtest, config, ticker,
                                                            locked_backtest=locked_backtest_with_exit)
                    except Exception as e:
                        print(f"Could not generate exit chart: {e}")

                    # Build stats section
                    stats_section = ""
                    if exit_backtest:
                        stats_section = (
                            f"---\n"
                            f"📊 **Strategy Stats:**\n"
                            f"• Trades: {exit_backtest['num_trades']} | Win Rate: {exit_backtest['win_rate']:.0f}%\n"
                            f"• Total Return: {exit_backtest['total_return']:.1f}% | PF: {exit_backtest['profit_factor']:.1f}\n"
                        )

                    pnl_emoji = "✅" if pnl_pct > 0 else "❌"

                    # Calculate hold duration
                    entry_time_str = trade_state.get('entry_time', '')
                    hold_duration = "N/A"
                    if entry_time_str:
                        try:
                            entry_dt = datetime.strptime(entry_time_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
                            duration = current_time - entry_dt
                            days = duration.days
                            hours = duration.seconds // 3600
                            if days > 0:
                                hold_duration = f"{days}d {hours}h"
                            else:
                                hold_duration = f"{hours}h {(duration.seconds % 3600) // 60}m"
                        except:
                            hold_duration = "N/A"

                    # Calculate actual dollar P&L per unit
                    pnl_dollars = current_price - entry_price  # LONG: profit when price goes up

                    exit_msg = (
                        f"📤 **[{strategy_label}] LONG EXIT** {pnl_emoji}\n"
                        f"**Reason:** {exit_reason}\n"
                        f"---\n"
                        f"📅 **Entry:** {entry_time_str[:16] if entry_time_str else 'N/A'} @ ${entry_price:.2f}\n"
                        f"📅 **Exit:** {current_time.strftime('%Y-%m-%d %H:%M')} @ ${current_price:.2f}\n"
                        f"⏱️ **Hold Duration:** {hold_duration}\n"
                        f"---\n"
                        f"💰 **P&L:** {pnl_pct:+.2f}% (${pnl_dollars:+,.2f}/unit)\n"
                        f"{stats_section}"
                    )

                    # Log closed trade and get cumulative stats
                    cumulative = log_closed_trade(
                        ticker=ticker,
                        position_type="LONG",
                        entry_price=entry_price,
                        exit_price=current_price,
                        entry_time=entry_time_str,
                        exit_time=current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        exit_reason=exit_reason,
                        pnl_pct=pnl_pct,
                        strategy_name=strategy_name,
                        entry_signal_bar=trade_state.get('entry_signal_bar'),  # Locked entry signal bar
                        exit_signal_bar=str(current_time)  # Exit bar
                    )

                    # Append exit to locked backtest (for future charts)
                    append_to_locked_backtest(
                        exit_trade={
                            'date': current_time,
                            'price': current_price,
                            'pnl': pnl_pct,
                            'reason': exit_reason,
                            'entry_price': entry_price,
                            'entry_date': trade_state.get('entry_time', '')
                        },
                        strategy_name=strategy_name, ticker=ticker
                    )
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

                    send_discord_alert(webhook_url, exit_msg, exit_chart, strategy_name=strategy_name)
                    print(f"EXIT LONG: {exit_reason}")

                    trade_state['position'] = None
                    trade_state['entry_price'] = None
                    trade_state['entry_time'] = None
                    trade_state['entry_signal_bar'] = None  # Clear signal bar
                    save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)

            # NOTE: SHORT positions are not supported in this LONG-only strategy
            # The backtest engine only tracks LONG positions (position=1 or position=0)

            # Check for new entry signals (only if not in position)
            if trade_state['position'] is None:
                # For daily interval, only evaluate COMPLETED bars to prevent repainting
                # The current bar (df.iloc[-1]) is still forming, use df.iloc[-2] for signals

                # Use Eastern Time for stock market hour checks (SPY closes at 4 PM ET)
                market_time = get_market_time()  # Returns Eastern Time
                should_check_signals = False

                if interval == "1d":
                    # Get the last COMPLETED bar (not the current forming bar)
                    if len(df) >= 2:
                        completed_bar = df.iloc[-2]
                        completed_bar_time = df.index[-2]

                        # Check if we've already evaluated this bar
                        if str(completed_bar_time) != str(last_signal_bar_evaluated):
                            if is_crypto:
                                # Crypto (BTC): Daily bar closes at 00:00 UTC
                                # Only evaluate after new bar starts (wait 30 min buffer for data)
                                utc_now = datetime.utcnow()
                                utc_hour = utc_now.hour
                                utc_minute = utc_now.minute
                                # Check between 00:30 UTC and 23:59 UTC (avoid checking right at midnight)
                                if utc_hour == 0 and utc_minute < 30:
                                    print(f"   ⏳ [CRYPTO] Waiting for new daily bar to finalize (UTC: {utc_now.strftime('%H:%M')})")
                                else:
                                    should_check_signals = True
                                    print(f"   📊 [CRYPTO] Evaluating completed bar: {completed_bar_time} (UTC: {utc_now.strftime('%H:%M')})")
                            else:
                                # Stocks (SPY): Only evaluate after market close (4 PM ET = 16:00)
                                # market_time is already in Eastern Time via get_market_time()
                                et_hour = market_time.hour
                                if et_hour >= 16:
                                    should_check_signals = True
                                    print(f"   📊 [STOCK] Market closed ({market_time.strftime('%H:%M')} ET) - Evaluating completed bar: {completed_bar_time}")
                                else:
                                    print(f"   ⏳ [STOCK] Market open ({market_time.strftime('%H:%M')} ET) - Waiting for 4 PM ET close to evaluate signals")
                else:
                    # For non-daily intervals, use existing logic
                    should_check_signals = True
                    completed_bar = df.iloc[-1]
                    completed_bar_time = df.index[-1]

                recent_buy_signal = None
                recent_sell_signal = None

                if should_check_signals and interval == "1d" and len(df) >= 2:
                    # Only check the COMPLETED bar for daily strategies
                    bar = completed_bar
                    bar_time = completed_bar_time
                    last_signal_time = trade_state.get('last_signal_time')

                    # Skip if already processed this bar
                    if not (last_signal_time and str(bar_time) == str(last_signal_time)):
                        if bar['buy_signal']:
                            recent_buy_signal = {'bar': bar, 'time': bar_time, 'index': -2}
                            print(f"   ✅ BUY signal on completed bar {bar_time}")
                        if bar['sell_signal']:
                            recent_sell_signal = {'bar': bar, 'time': bar_time, 'index': -2}
                            print(f"   ✅ SELL signal on completed bar {bar_time}")

                    # Mark this bar as evaluated
                    last_signal_bar_evaluated = str(completed_bar_time)

                elif should_check_signals and interval != "1d":
                    # Non-daily: use original lookback logic
                    lookback_bars = 3
                    for i in range(1, min(lookback_bars + 1, len(df))):
                        bar = df.iloc[-i]
                        bar_time = df.index[-i]
                        last_signal_time = trade_state.get('last_signal_time')

                        # Skip if already processed this bar
                        if last_signal_time and str(bar_time) == str(last_signal_time):
                            continue

                        if bar['buy_signal'] and recent_buy_signal is None:
                            recent_buy_signal = {'bar': bar, 'time': bar_time, 'index': -i}
                        if bar['sell_signal'] and recent_sell_signal is None:
                            recent_sell_signal = {'bar': bar, 'time': bar_time, 'index': -i}

                # Log recent signals found
                if recent_buy_signal and recent_buy_signal['index'] < -1:
                    print(f"⚠️  Found missed BUY signal from {recent_buy_signal['time']}")
                if recent_sell_signal and recent_sell_signal['index'] < -1:
                    print(f"⚠️  Found missed SELL signal from {recent_sell_signal['time']}")

                # Use most recent signal (prefer current bar, then recent bars)
                effective_buy = recent_buy_signal is not None
                effective_sell = recent_sell_signal is not None

                # Avoid duplicate signals
                last_signal_time = trade_state.get('last_signal_time')
                if last_signal_time and recent_buy_signal and str(recent_buy_signal['time']) == str(last_signal_time):
                    effective_buy = False
                if last_signal_time and recent_sell_signal and str(recent_sell_signal['time']) == str(last_signal_time):
                    effective_sell = False

                if effective_buy:
                    signal_bar = recent_buy_signal['bar']
                    signal_time = recent_buy_signal['time']
                    signal_note = " (MISSED - acting now)" if recent_buy_signal['index'] < -1 else ""

                    # Append new entry to locked backtest (for future charts)
                    append_to_locked_backtest(
                        entry={'date': signal_time, 'price': current_price, 'position': 'long'},
                        strategy_name=strategy_name, ticker=ticker
                    )
                    # Reload the locked backtest with the new entry
                    locked_backtest = load_locked_backtest(strategy_name=strategy_name, ticker=ticker)

                    # Run backtest to get stats for the signal alert
                    signal_backtest = None
                    signal_chart = None
                    try:
                        signal_backtest = run_historical_backtest(df, config)
                        # Use locked backtest for markers to prevent repainting
                        signal_chart = generate_velocity_chart(df, signal_backtest, config, ticker,
                                                              locked_backtest=locked_backtest)
                    except Exception as e:
                        print(f"Could not generate signal chart: {e}")

                    # Build comprehensive buy message with stats
                    stats_section = ""
                    if signal_backtest:
                        stats_section = (
                            f"---\n"
                            f"📊 **Strategy Stats:**\n"
                            f"• Trades: {signal_backtest['num_trades']} | Win Rate: {signal_backtest['win_rate']:.0f}%\n"
                            f"• Total Return: {signal_backtest['total_return']:.1f}% | PF: {signal_backtest['profit_factor']:.1f}\n"
                        )

                    buy_msg = (
                        f"📈 **[{strategy_label}] BUY SIGNAL**{signal_note}\n"
                        f"**Signal Time:** {signal_time}\n"
                        f"**Entry Price:** ${current_price:.2f}\n"
                        f"{stats_section}"
                        f"---\n"
                        f"_SL: ${current_price * (1 - stop_loss_pct/100):.2f} | TP: ${current_price * (1 + take_profit_pct/100):.2f}_"
                    )

                    send_discord_alert(webhook_url, buy_msg, signal_chart, strategy_name=strategy_name)
                    print(f"BUY SIGNAL SENT!{signal_note}")

                    trade_state['position'] = 'long'
                    trade_state['entry_price'] = current_price
                    trade_state['entry_time'] = str(current_time)
                    trade_state['entry_signal_bar'] = str(signal_time)  # Lock the signal bar for charts
                    trade_state['last_signal_time'] = str(signal_time)
                    save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)

                elif effective_sell:
                    # LONG-ONLY STRATEGY: Sell signals are ignored when not in position
                    # They are only used to EXIT existing long positions (handled above)
                    signal_time = recent_sell_signal['time']
                    print(f"   ℹ️  Sell signal on {signal_time} ignored (LONG-only strategy, no position)")
                    # Update last_signal_time to avoid re-processing
                    trade_state['last_signal_time'] = str(signal_time)
                    save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)

            print(f"⏳ [{strategy_label} | {strategy_name}] Next check in {check_interval_seconds // 60} min")
            time.sleep(check_interval_seconds)

        except KeyboardInterrupt:
            print("\n\nShutting down...")
            shutdown_msg = "🛑 **[VELOCITY] Live Trader Stopped**\n_Manually terminated._"
            send_discord_alert(webhook_url, shutdown_msg, strategy_name=strategy_name)
            break
        except Exception as e:
            error_msg = f"⚠️ **[VELOCITY] Error in Trader**\n```{str(e)}```"
            send_discord_alert(webhook_url, error_msg, strategy_name=strategy_name)
            print(f"Error: {e}")
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
