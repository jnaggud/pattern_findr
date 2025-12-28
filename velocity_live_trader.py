"""
Velocity-Based Live Trader

Real-time trading bot that monitors oscillator velocity/acceleration signals
and sends alerts to Discord. Designed for velocity strategy deployment.

Features:
- Interactive strategy selection at startup
- Permanent strategy storage in velocity_strategies/
- Config hot-reloading during runtime
- Scheduled status updates (market open, mid-day, close, hourly)
- Historical backtest on startup with charts

Usage:
    python velocity_live_trader.py [--config path/to/config.json]
"""

import json
import os
import time
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import io
import shutil

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


def fetch_price_data(ticker: str, api_key: str = None, days: int = 200, interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical price data using yfinance.
    Uses EXACT same method as Streamlit (oscillator_predictor_page.py line 1238-1242)
    """
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

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

    df['buy_signal'] = raw_buy
    df['sell_signal'] = raw_sell

    return df


def send_discord_alert(webhook_url: str, message: str, chart_buf: io.BytesIO = None,
                       include_disclaimer: bool = True):
    """Send alert to Discord webhook with optional chart image."""
    if not webhook_url:
        print(f"[ALERT] {message}")
        return False

    # Append legal disclaimer to all messages
    if include_disclaimer:
        message = message + LEGAL_DISCLAIMER

    try:
        if chart_buf:
            # Send with image attachment
            chart_buf.seek(0)
            files = {'file': ('chart.png', chart_buf, 'image/png')}
            payload = {'content': message}
            response = requests.post(webhook_url, data=payload, files=files)
        else:
            payload = {"content": message}
            response = requests.post(webhook_url, json=payload)

        if response.status_code in [200, 204]:
            return True
        else:
            print(f"Discord webhook error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Discord webhook error: {e}")
        return False


def generate_velocity_chart(df: pd.DataFrame, backtest: dict, config: dict, ticker: str,
                            title_suffix: str = "") -> io.BytesIO:
    """Generate a velocity strategy chart for Discord.

    Args:
        df: DataFrame with price and indicator data
        backtest: Backtest results dict
        config: Strategy config
        ticker: Ticker symbol
        title_suffix: Optional suffix for chart title (e.g., " - Last 180 Days")
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

        # Plot trade markers from backtest
        if backtest and backtest.get('entries'):
            for entry in backtest['entries']:
                if entry['date'] in df_plot.index:
                    ax1.scatter(entry['date'], entry['price'], marker='^', color='lime', s=100, zorder=5)

        if backtest and backtest.get('exits'):
            for exit in backtest['exits']:
                if exit['date'] in df_plot.index:
                    color = 'green' if exit['pnl'] > 0 else 'red'
                    ax1.scatter(exit['date'], exit['price'], marker='v', color=color, s=100, zorder=5)

        # Mark current open position
        if backtest and backtest.get('current_position'):
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

        # 4. Equity Curve (if we have backtest data)
        ax4 = axes[3]
        if backtest and backtest.get('exits'):
            # Build simple equity curve from trades
            equity = [100]
            for exit in backtest['exits']:
                equity.append(equity[-1] * (1 + exit['pnl']/100))

            # Add unrealized if open position
            if backtest.get('current_position'):
                equity.append(equity[-1] * (1 + backtest['current_position']['unrealized_pnl']/100))

            ax4.plot(range(len(equity)), equity, color='#00ff88', linewidth=2)
            ax4.fill_between(range(len(equity)), 100, equity, alpha=0.3,
                           color='green' if equity[-1] > 100 else 'red')
            ax4.axhline(100, color='gray', linestyle='--', alpha=0.5)
        ax4.set_ylabel("Equity", color='white')
        ax4.set_xlabel("Trade #", color='white')

        # Stats annotation
        if backtest:
            stats_text = (f"Trades: {backtest['num_trades']} | "
                         f"Win: {backtest['win_rate']:.0f}% | "
                         f"Return: {backtest['total_return']:.1f}% | "
                         f"PF: {backtest['profit_factor']:.1f}")
            fig.text(0.5, 0.02, stats_text, ha='center', color='white', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.8))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)

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
        "position": None,  # None, "long", or "short"
        "entry_price": None,
        "entry_time": None,
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
                     strategy_name: str = None):
    """Log a closed trade to history and return cumulative stats."""
    history = load_trade_history(strategy_name=strategy_name, ticker=ticker)

    trade = {
        "id": len(history) + 1,
        "ticker": ticker,
        "strategy_name": strategy_name,
        "type": position_type,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "exit_reason": exit_reason,
        "pnl_pct": pnl_pct,
        "pnl_dollars": (pnl_pct / 100) * 10000  # Assuming $10k position
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
                       ticker: str, title: str = "📊 Status Update", trade_state: dict = None):
    """Send a scheduled status update with TWO charts to Discord.

    Sends two posts:
    1. Full timeframe chart with stats over entire period
    2. Last 180 days chart with stats recalculated for that window
    """
    try:
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

                # Estimate P&L in dollars
                position_size = 10000
                pnl_dollars = (pnl / 100) * position_size

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
            elif backtest_local and backtest_local.get('current_position'):
                pos = backtest_local['current_position']
                pos_section = f"🟢 **Position:** LONG @ ${pos['entry_price']:.2f} ({pos['unrealized_pnl']:+.1f}%)"

            return pos_section

        # Create strategy label for clear identification
        opt_period = config.get('optimization_period', '')
        strategy_label = f"{ticker} {opt_period.upper()}" if opt_period else ticker
        current_price = df['close'].iloc[-1]

        # Calculate date range for full data
        full_start = df.index[0].strftime('%Y-%m-%d') if hasattr(df.index[0], 'strftime') else str(df.index[0])[:10]
        full_end = df.index[-1].strftime('%Y-%m-%d') if hasattr(df.index[-1], 'strftime') else str(df.index[-1])[:10]
        full_days = len(df)

        # --- CHART 1: Full Timeframe ---
        chart_buf_full = generate_velocity_chart(df, backtest, config, ticker,
                                                  title_suffix=f" (Full: {full_days} bars)")
        pos_section = build_position_section(current_price, trade_state, backtest, config)

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
            f"_Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_"
        )

        send_discord_alert(webhook_url, msg_full, chart_buf_full)
        print(f"✅ Sent full timeframe update: {title}")

        # --- CHART 2: Recent Period (180 days or half of data) ---
        # Determine subset size: use 180 days if available, otherwise half the data
        min_bars_for_subset = 20  # Need at least 20 bars to make a meaningful subset

        if len(df) > min_bars_for_subset:
            if len(df) > 180:
                subset_days = 180
                period_label = "Last 180 Days"
            else:
                subset_days = len(df) // 2
                period_label = f"Last {subset_days} Days"

            df_subset, backtest_subset = run_backtest_for_period(df, config, days=subset_days)

            # Calculate date range for subset window
            period_start = df_subset.index[0].strftime('%Y-%m-%d') if hasattr(df_subset.index[0], 'strftime') else str(df_subset.index[0])[:10]
            period_end = df_subset.index[-1].strftime('%Y-%m-%d') if hasattr(df_subset.index[-1], 'strftime') else str(df_subset.index[-1])[:10]

            chart_buf_subset = generate_velocity_chart(df_subset, backtest_subset, config, ticker,
                                                        title_suffix=f" ({period_label})")

            # Position section stays the same (current position)
            pos_section_subset = build_position_section(current_price, trade_state, backtest_subset, config)

            msg_subset = (
                f"**{title}: {strategy_label}** [2/2 {period_label}]\n"
                f"**Period:** {period_start} to {period_end} ({len(df_subset)} bars)\n"
                f"---\n"
                f"{pos_section_subset}\n"
                f"---\n"
                f"📊 **{period_label} Stats:**\n"
                f"• Trades: {backtest_subset['num_trades']} | Win Rate: {backtest_subset['win_rate']:.0f}%\n"
                f"• Return: {backtest_subset['total_return']:.1f}% | PF: {backtest_subset['profit_factor']:.1f}\n"
                f"---\n"
                f"_Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_"
            )

            send_discord_alert(webhook_url, msg_subset, chart_buf_subset)
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

        # Current position
        if backtest['current_position']:
            pos = backtest['current_position']
            print(f"\n{'='*60}")
            print("🔵 OPEN POSITION DETECTED")
            print(f"{'='*60}")
            print(f"Entry: ${pos['entry_price']:.2f} on {pos['entry_date']}")
            print(f"Current: ${pos['current_price']:.2f}")
            print(f"Unrealized P&L: {pos['unrealized_pnl']:+.2f}%")
            print(f"{'='*60}")
        else:
            print(f"\n📭 No open position")

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
    try:
        fresh_df = fetch_price_data(ticker, api_key, days=200, interval=interval)
        if not fresh_df.empty:
            fresh_df = calculate_composite_oscillator(fresh_df, config)
            fresh_backtest = run_historical_backtest(fresh_df, config)

            # Show fresh data position vs bundled data position
            fresh_pos = fresh_backtest.get('current_position')
            bundled_pos = backtest.get('current_position') if backtest else None

            print(f"   Fresh data range: {fresh_df.index[0].strftime('%Y-%m-%d')} to {fresh_df.index[-1].strftime('%Y-%m-%d')}")
            if fresh_pos:
                print(f"   Fresh backtest position: LONG @ ${fresh_pos['entry_price']:.2f} on {fresh_pos['entry_date']}")
            else:
                print(f"   Fresh backtest position: None")

            # Check if bundled and fresh differ
            if bundled_pos and fresh_pos:
                if abs(bundled_pos['entry_price'] - fresh_pos['entry_price']) > 1:
                    print(f"   ⚠️  Bundled data is STALE! Using fresh data for sync.")
    except Exception as e:
        print(f"   ⚠️ Could not fetch fresh data: {e}")
        fresh_backtest = backtest  # Fall back to bundled

    # Use fresh backtest for sync if available, otherwise fall back to bundled
    sync_backtest = fresh_backtest if fresh_backtest else backtest

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
        trade_state = {'position': None, 'entry_price': None, 'entry_time': None, 'last_signal_time': None}
        save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)
        print(f"   ✅ State synced - position cleared")

        # Send Discord notification about missed exit
        sync_msg = (
            f"⚠️ **[{strategy_label}] State Sync - Missed Exit**\n"
            f"---\n"
            f"**Previous tracking:** {state_position.upper()} @ ${old_entry:.2f}\n"
            f"**Entry time:** {str(old_time)[:16]}\n"
            f"---\n"
            f"**Backtest shows:** Position was closed\n"
            f"**Action:** State cleared, now tracking no position\n"
            f"---\n"
            f"_Note: The exit signal was missed. Check trade history for details._"
        )
        send_discord_alert(webhook_url, sync_msg)

    # Case 3: Both show position but entry prices differ - missed exit AND new entry
    elif state_position is not None and backtest_position is not None:
        state_entry = trade_state.get('entry_price', 0)
        state_time = trade_state.get('entry_time', 'Unknown')
        backtest_entry = backtest_position.get('entry_price', 0)
        backtest_time = backtest_position.get('entry_date', 'Unknown')

        # Check if entries differ by more than 0.5% (they should match if same trade)
        if state_entry > 0 and abs(state_entry - backtest_entry) / state_entry > 0.005:
            print(f"⚠️  STATE MISMATCH: Different entries detected!")
            print(f"   State says: {state_position.upper()} @ ${state_entry:.2f}")
            print(f"   Backtest says: {backtest_position['position'].upper()} @ ${backtest_entry:.2f}")
            print(f"   🔄 Syncing state with backtest (missed exit + new entry)")

            trade_state = {
                'position': backtest_position['position'],
                'entry_price': backtest_position['entry_price'],
                'entry_time': backtest_position['entry_date'],
                'last_signal_time': backtest_position['entry_date'],
            }
            save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)
            print(f"   ✅ State synced - now tracking: {backtest_position['position'].upper()} @ ${backtest_entry:.2f}")

            # Send Discord notification about missed exit + new entry
            sync_msg = (
                f"⚠️ **[{strategy_label}] State Sync - Missed Exit + New Entry**\n"
                f"---\n"
                f"**Previous tracking:** {state_position.upper()} @ ${state_entry:.2f}\n"
                f"**Previous entry time:** {str(state_time)[:16]}\n"
                f"---\n"
                f"**Now tracking:** {backtest_position['position'].upper()} @ ${backtest_entry:.2f}\n"
                f"**New entry time:** {str(backtest_time)[:16]}\n"
                f"---\n"
                f"_Note: Exit of previous trade + entry of new trade were missed._"
            )
            send_discord_alert(webhook_url, sync_msg)

    # Send startup notification with stats and chart
    startup_chart = None
    if backtest:
        pos_status = f"**Position:** LONG @ ${backtest['current_position']['entry_price']:.2f} ({backtest['current_position']['unrealized_pnl']:+.1f}%)" if backtest['current_position'] else "**Position:** None"
        startup_msg = (
            f"🤖 **[{strategy_label}] Live Trader Started**\n"
            f"**Strategy:** {strategy_name}\n"
            f"**Ticker:** {ticker}\n"
            f"**Signal Type:** {config.get('signal_type')}\n"
            f"**Risk:** SL={stop_loss_pct:.1f}%, TP={take_profit_pct:.1f}%\n"
            f"---\n"
            f"📊 **Historical Stats:**\n"
            f"• Trades: {backtest['num_trades']} | Win Rate: {backtest['win_rate']:.0f}%\n"
            f"• Return: {backtest['total_return']:.1f}% | PF: {backtest['profit_factor']:.1f}\n"
            f"{pos_status}\n"
            f"---\n"
            f"_Monitoring for signals..._"
        )
        # Generate chart for startup
        try:
            startup_chart = generate_velocity_chart(df, backtest, config, ticker)
            print("✅ Generated startup chart for Discord")
        except Exception as e:
            print(f"⚠️ Could not generate startup chart: {e}")
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
    send_discord_alert(webhook_url, startup_msg, startup_chart)

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
                        send_discord_alert(webhook_url, f"🔄 **Config Reloaded**\nTicker: {ticker}\nSignal: {config.get('signal_type')}")
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
            current_price = latest['close']
            current_time = df.index[-1]

            print(f"Current Price: ${current_price:.2f}")
            print(f"Oscillator: {latest['osc_smooth']:.4f}")
            print(f"Velocity: {latest['velocity']:.4f}")
            print(f"Acceleration: {latest['acceleration']:.4f}")
            print(f"Buy Signal: {latest['buy_signal']}")
            print(f"Sell Signal: {latest['sell_signal']}")
            print(f"Position: {trade_state['position']}")

            # --- SCHEDULED STATUS UPDATES ---
            now = datetime.now()
            day_str = now.strftime("%Y-%m-%d")
            hm = now.strftime("%H:%M")
            hour = now.hour
            minute = now.minute

            # Run backtest for status updates
            status_backtest = None
            try:
                status_backtest = run_historical_backtest(df, config)
            except:
                pass

            # Market Open Update (8:30-8:45 AM)
            if "08:30" <= hm <= "08:45":
                key = f"{day_str}_OPEN"
                if key not in daily_alerts and status_backtest:
                    send_status_update(webhook_url, df, status_backtest, config, ticker,
                                     "🔔 Market Open Update", trade_state)
                    daily_alerts.add(key)

            # Mid-Day Update (11:00-11:15 AM)
            if "11:00" <= hm <= "11:15":
                key = f"{day_str}_MID"
                if key not in daily_alerts and status_backtest:
                    send_status_update(webhook_url, df, status_backtest, config, ticker,
                                     "☀️ Mid-Day Update", trade_state)
                    daily_alerts.add(key)

            # Market Close Update (15:00-15:15 PM)
            if "15:00" <= hm <= "15:15":
                key = f"{day_str}_CLOSE"
                if key not in daily_alerts and status_backtest:
                    send_status_update(webhook_url, df, status_backtest, config, ticker,
                                     "🏁 Market Close Update", trade_state)
                    daily_alerts.add(key)

            # Hourly Updates (9:00 - 16:00, except 11 and 15 which have special alerts)
            # Window is first 15 minutes of each hour to ensure it gets hit
            if 9 <= hour <= 16 and minute < 15:
                if hour not in [11, 15]:
                    key = f"{day_str}_HOUR_{hour}"
                    if key not in daily_alerts and status_backtest:
                        send_status_update(webhook_url, df, status_backtest, config, ticker,
                                         f"⏱️ {hour}:00 Market Update", trade_state)
                        daily_alerts.add(key)

            # Midnight Update
            if hour == 0 and minute < 5:
                key = f"{day_str}_MIDNIGHT"
                if key not in daily_alerts and status_backtest:
                    send_status_update(webhook_url, df, status_backtest, config, ticker,
                                     "🌙 Midnight Update", trade_state)
                    daily_alerts.add(key)

            # Check for exit conditions first (if in position)
            if trade_state['position'] == 'long':
                entry_price = trade_state['entry_price']
                pnl_pct = ((current_price - entry_price) / entry_price) * 100

                exit_reason = None

                # Stop loss
                if pnl_pct <= -stop_loss_pct:
                    exit_reason = f"Stop Loss ({pnl_pct:.2f}%)"
                # Take profit
                elif pnl_pct >= take_profit_pct:
                    exit_reason = f"Take Profit ({pnl_pct:.2f}%)"
                # Exit on opposite signal
                elif exit_on_opposite_signal and latest['sell_signal']:
                    exit_reason = f"Opposite Signal ({pnl_pct:.2f}%)"
                # Exit on midline cross
                elif exit_on_midline_cross and latest['osc_smooth'] > 0:
                    exit_reason = f"Midline Cross ({pnl_pct:.2f}%)"

                if exit_reason:
                    # Generate chart and get stats for exit
                    exit_chart = None
                    exit_backtest = None
                    try:
                        exit_backtest = run_historical_backtest(df, config)
                        exit_chart = generate_velocity_chart(df, exit_backtest, config, ticker)
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

                    # Estimate P&L in dollars (assuming $10k position size)
                    position_size = 10000
                    pnl_dollars = (pnl_pct / 100) * position_size

                    exit_msg = (
                        f"📤 **[{strategy_label}] LONG EXIT** {pnl_emoji}\n"
                        f"**Reason:** {exit_reason}\n"
                        f"---\n"
                        f"📅 **Entry:** {entry_time_str[:16] if entry_time_str else 'N/A'} @ ${entry_price:.2f}\n"
                        f"📅 **Exit:** {current_time.strftime('%Y-%m-%d %H:%M')} @ ${current_price:.2f}\n"
                        f"⏱️ **Hold Duration:** {hold_duration}\n"
                        f"---\n"
                        f"💰 **P&L:** {pnl_pct:+.2f}% (${pnl_dollars:+,.0f} on $10k)\n"
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
                        strategy_name=strategy_name
                    )

                    # Add cumulative stats to message
                    cumulative_section = (
                        f"---\n"
                        f"📈 **Cumulative Performance:**\n"
                        f"• Trades: {cumulative['total_trades']} ({cumulative['winners']}W / {cumulative['losers']}L)\n"
                        f"• Win Rate: {cumulative['win_rate']:.0f}%\n"
                        f"• Total P&L: {cumulative['total_pnl_pct']:+.2f}% (${cumulative['total_pnl_dollars']:+,.0f})\n"
                    )

                    exit_msg += cumulative_section

                    send_discord_alert(webhook_url, exit_msg, exit_chart)
                    print(f"EXIT LONG: {exit_reason}")

                    trade_state['position'] = None
                    trade_state['entry_price'] = None
                    trade_state['entry_time'] = None
                    save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)

            elif trade_state['position'] == 'short':
                entry_price = trade_state['entry_price']
                pnl_pct = ((entry_price - current_price) / entry_price) * 100

                exit_reason = None

                # Stop loss
                if pnl_pct <= -stop_loss_pct:
                    exit_reason = f"Stop Loss ({pnl_pct:.2f}%)"
                # Take profit
                elif pnl_pct >= take_profit_pct:
                    exit_reason = f"Take Profit ({pnl_pct:.2f}%)"
                # Exit on opposite signal
                elif exit_on_opposite_signal and latest['buy_signal']:
                    exit_reason = f"Opposite Signal ({pnl_pct:.2f}%)"
                # Exit on midline cross
                elif exit_on_midline_cross and latest['osc_smooth'] < 0:
                    exit_reason = f"Midline Cross ({pnl_pct:.2f}%)"

                if exit_reason:
                    # Generate chart and get stats for exit
                    exit_chart = None
                    exit_backtest = None
                    try:
                        exit_backtest = run_historical_backtest(df, config)
                        exit_chart = generate_velocity_chart(df, exit_backtest, config, ticker)
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

                    # Estimate P&L in dollars (assuming $10k position size)
                    position_size = 10000
                    pnl_dollars = (pnl_pct / 100) * position_size

                    exit_msg = (
                        f"📤 **[{strategy_label}] SHORT EXIT** {pnl_emoji}\n"
                        f"**Reason:** {exit_reason}\n"
                        f"---\n"
                        f"📅 **Entry:** {entry_time_str[:16] if entry_time_str else 'N/A'} @ ${entry_price:.2f}\n"
                        f"📅 **Exit:** {current_time.strftime('%Y-%m-%d %H:%M')} @ ${current_price:.2f}\n"
                        f"⏱️ **Hold Duration:** {hold_duration}\n"
                        f"---\n"
                        f"💰 **P&L:** {pnl_pct:+.2f}% (${pnl_dollars:+,.0f} on $10k)\n"
                        f"{stats_section}"
                    )

                    # Log closed trade and get cumulative stats
                    cumulative = log_closed_trade(
                        ticker=ticker,
                        position_type="SHORT",
                        entry_price=entry_price,
                        exit_price=current_price,
                        entry_time=entry_time_str,
                        exit_time=current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        exit_reason=exit_reason,
                        pnl_pct=pnl_pct,
                        strategy_name=strategy_name
                    )

                    # Add cumulative stats to message
                    cumulative_section = (
                        f"---\n"
                        f"📈 **Cumulative Performance:**\n"
                        f"• Trades: {cumulative['total_trades']} ({cumulative['winners']}W / {cumulative['losers']}L)\n"
                        f"• Win Rate: {cumulative['win_rate']:.0f}%\n"
                        f"• Total P&L: {cumulative['total_pnl_pct']:+.2f}% (${cumulative['total_pnl_dollars']:+,.0f})\n"
                    )

                    exit_msg += cumulative_section

                    send_discord_alert(webhook_url, exit_msg, exit_chart)
                    print(f"EXIT SHORT: {exit_reason}")

                    trade_state['position'] = None
                    trade_state['entry_price'] = None
                    trade_state['entry_time'] = None
                    save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)

            # Check for new entry signals (only if not in position)
            if trade_state['position'] is None:
                # Check for recent signals we might have missed (look back up to 3 bars)
                lookback_bars = 3
                recent_buy_signal = None
                recent_sell_signal = None

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

                    # Run backtest to get stats for the signal alert
                    signal_backtest = None
                    signal_chart = None
                    try:
                        signal_backtest = run_historical_backtest(df, config)
                        signal_chart = generate_velocity_chart(df, signal_backtest, config, ticker)
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

                    send_discord_alert(webhook_url, buy_msg, signal_chart)
                    print(f"BUY SIGNAL SENT!{signal_note}")

                    trade_state['position'] = 'long'
                    trade_state['entry_price'] = current_price
                    trade_state['entry_time'] = str(current_time)
                    trade_state['last_signal_time'] = str(signal_time)
                    save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)

                elif effective_sell:
                    signal_bar = recent_sell_signal['bar']
                    signal_time = recent_sell_signal['time']
                    signal_note = " (MISSED - acting now)" if recent_sell_signal['index'] < -1 else ""

                    # Run backtest to get stats for the signal alert
                    signal_backtest = None
                    signal_chart = None
                    try:
                        signal_backtest = run_historical_backtest(df, config)
                        signal_chart = generate_velocity_chart(df, signal_backtest, config, ticker)
                    except Exception as e:
                        print(f"Could not generate signal chart: {e}")

                    # Build comprehensive sell message with stats
                    stats_section = ""
                    if signal_backtest:
                        stats_section = (
                            f"---\n"
                            f"📊 **Strategy Stats:**\n"
                            f"• Trades: {signal_backtest['num_trades']} | Win Rate: {signal_backtest['win_rate']:.0f}%\n"
                            f"• Total Return: {signal_backtest['total_return']:.1f}% | PF: {signal_backtest['profit_factor']:.1f}\n"
                        )

                    sell_msg = (
                        f"📉 **[{strategy_label}] SELL SIGNAL**{signal_note}\n"
                        f"**Signal Time:** {signal_time}\n"
                        f"**Entry Price:** ${current_price:.2f}\n"
                        f"{stats_section}"
                        f"---\n"
                        f"_SL: ${current_price * (1 + stop_loss_pct/100):.2f} | TP: ${current_price * (1 - take_profit_pct/100):.2f}_"
                    )

                    send_discord_alert(webhook_url, sell_msg, signal_chart)
                    print(f"SELL SIGNAL SENT!{signal_note}")

                    trade_state['position'] = 'short'
                    trade_state['entry_price'] = current_price
                    trade_state['entry_time'] = str(current_time)
                    trade_state['last_signal_time'] = str(signal_time)
                    save_trade_state(trade_state, strategy_name=strategy_name, ticker=ticker)

            print(f"⏳ [{strategy_label} | {strategy_name}] Next check in {check_interval_seconds // 60} min")
            time.sleep(check_interval_seconds)

        except KeyboardInterrupt:
            print("\n\nShutting down...")
            shutdown_msg = "🛑 **[VELOCITY] Live Trader Stopped**\n_Manually terminated._"
            send_discord_alert(webhook_url, shutdown_msg)
            break
        except Exception as e:
            error_msg = f"⚠️ **[VELOCITY] Error in Trader**\n```{str(e)}```"
            send_discord_alert(webhook_url, error_msg)
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
