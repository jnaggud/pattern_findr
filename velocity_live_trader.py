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

            # Copy to production config for hot-reload compatibility
            shutil.copy(selected['config_path'], PRODUCTION_CONFIG_PATH)
            print(f"   Copied to {PRODUCTION_CONFIG_PATH} for hot-reload support")

            return PRODUCTION_CONFIG_PATH
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
    """Fetch historical price data for the ticker."""

    if POLYGON_AVAILABLE and api_key:
        try:
            client = RESTClient(api_key)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Convert interval to Polygon format
            timespan = "day" if interval == "1d" else "hour" if interval == "1h" else "minute"
            multiplier = 1

            aggs = client.get_aggs(
                ticker=ticker.replace("-", ""),  # BTC-USD -> BTCUSD for Polygon
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=50000
            )

            if aggs:
                df = pd.DataFrame([{
                    'timestamp': a.timestamp,
                    'open': a.open,
                    'high': a.high,
                    'low': a.low,
                    'close': a.close,
                    'volume': a.volume
                } for a in aggs])

                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                return df
        except Exception as e:
            print(f"Polygon API error: {e}. Falling back to yfinance.")

    # Fallback to yfinance
    if YFINANCE_AVAILABLE:
        ticker_yf = yf.Ticker(ticker)
        df = ticker_yf.history(period=f"{days}d", interval=interval)
        df.columns = [c.lower() for c in df.columns]
        return df

    raise RuntimeError("No data source available. Install polygon-api-client or yfinance.")


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
        raw_buy = df['vel_cross_up'] | in_oversold
        raw_sell = df['vel_cross_down'] | in_overbought
    elif signal_type == 'zone_only':
        raw_buy = extreme_oversold & (velocity > 0)
        raw_sell = extreme_overbought & (velocity < 0)
    elif signal_type == 'momentum':
        raw_buy = strong_momentum_up & (osc_smooth < 0)
        raw_sell = strong_momentum_down & (osc_smooth > 0)
    elif signal_type == 'any_reversal':
        raw_buy = df['vel_cross_up'] | extreme_oversold
        raw_sell = df['vel_cross_down'] | extreme_overbought
    elif signal_type == 'double_bottom':
        osc_min_10 = osc_smooth.rolling(10, min_periods=1).min()
        raw_buy = (osc_smooth <= osc_min_10 * 0.95) & df['vel_cross_up']
        osc_max_10 = osc_smooth.rolling(10, min_periods=1).max()
        raw_sell = (osc_smooth >= osc_max_10 * 0.95) & df['vel_cross_down']
    elif signal_type == 'divergence':
        close_lower = df['close'] < df['close'].shift(5)
        osc_higher = osc_smooth > osc_smooth.shift(5)
        raw_buy = close_lower & osc_higher & in_oversold
        close_higher = df['close'] > df['close'].shift(5)
        osc_lower = osc_smooth < osc_smooth.shift(5)
        raw_sell = close_higher & osc_lower & in_overbought
    elif signal_type == 'breakout':
        raw_buy = (osc_smooth > oversold_threshold) & (osc_smooth.shift(1) <= oversold_threshold) & (velocity > 0)
        raw_sell = (osc_smooth < overbought_threshold) & (osc_smooth.shift(1) >= overbought_threshold) & (velocity < 0)
    else:
        raw_buy = df['vel_cross_up'] & in_oversold
        raw_sell = df['vel_cross_down'] & in_overbought

    # Apply acceleration filter
    if require_accel:
        raw_buy = raw_buy & (acceleration > 0)
        raw_sell = raw_sell & (acceleration < 0)

    # Apply extra indicator filters
    # RSI filter
    if rsi_filter != 'none':
        # Calculate RSI if not already done
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

        if rsi_filter == 'confirm':
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

    # Bollinger Band filter
    if use_bb_filter:
        bb_period = 20
        bb_std = 2
        bb_middle = df['close'].rolling(window=bb_period).mean()
        bb_std_val = df['close'].rolling(window=bb_period).std()
        df['bb_lower'] = bb_middle - (bb_std * bb_std_val)
        df['bb_upper'] = bb_middle + (bb_std * bb_std_val)

        raw_buy = raw_buy & (df['close'] < df['bb_lower'] * 1.02)
        raw_sell = raw_sell & (df['close'] > df['bb_upper'] * 0.98)

    df['buy_signal'] = raw_buy
    df['sell_signal'] = raw_sell

    return df


def send_discord_alert(webhook_url: str, message: str, chart_buf: io.BytesIO = None):
    """Send alert to Discord webhook with optional chart image."""
    if not webhook_url:
        print(f"[ALERT] {message}")
        return False

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


def generate_velocity_chart(df: pd.DataFrame, backtest: dict, config: dict, ticker: str) -> io.BytesIO:
    """Generate a velocity strategy chart for Discord."""
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

        ax1.set_title(f"{ticker} - Velocity Strategy", color='white', fontsize=14, fontweight='bold')
        ax1.set_ylabel("Price ($)", color='white')
        ax1.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white')
        # Format x-axis with dates
        if isinstance(df_plot.index, pd.DatetimeIndex):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.tick_params(axis='x', labelsize=8)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)

        # 2. Oscillator with thresholds
        ax2 = axes[1]
        osc_col = 'osc_smooth' if 'osc_smooth' in df_plot.columns else 'composite_smooth'
        if osc_col in df_plot.columns:
            ax2.plot(df_plot.index, df_plot[osc_col], color='#e94560', linewidth=1.5, label='Oscillator')
            ax2.axhline(config.get('oversold_threshold', -0.3), color='lime', linestyle='--', alpha=0.7, label='Oversold')
            ax2.axhline(config.get('overbought_threshold', 0.3), color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax2.fill_between(df_plot.index, df_plot[osc_col], 0,
                            where=(df_plot[osc_col] < config.get('oversold_threshold', -0.3)),
                            color='lime', alpha=0.3)
            ax2.fill_between(df_plot.index, df_plot[osc_col], 0,
                            where=(df_plot[osc_col] > config.get('overbought_threshold', 0.3)),
                            color='red', alpha=0.3)
        ax2.set_ylabel("Oscillator", color='white')
        ax2.set_ylim(-1.2, 1.2)
        ax2.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white', fontsize='small')
        # Format x-axis with dates
        if isinstance(df_plot.index, pd.DatetimeIndex):
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator())
            ax2.tick_params(axis='x', labelsize=8)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

        # 3. Velocity & Acceleration
        ax3 = axes[2]
        if 'velocity' in df_plot.columns:
            ax3.plot(df_plot.index, df_plot['velocity'], color='#00d9ff', linewidth=1.2, label='Velocity')
        if 'acceleration' in df_plot.columns:
            ax3.plot(df_plot.index, df_plot['acceleration'], color='#ffd700', linewidth=1.0, alpha=0.7, label='Acceleration')
        ax3.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax3.set_ylabel("Vel/Accel", color='white')
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


def load_trade_state(state_path: str = "velocity_trade_state.json") -> dict:
    """Load current trade state from file."""
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            return json.load(f)
    return {
        "position": None,  # None, "long", or "short"
        "entry_price": None,
        "entry_time": None,
        "last_signal_time": None,
    }


def save_trade_state(state: dict, state_path: str = "velocity_trade_state.json"):
    """Save trade state to file."""
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def send_status_update(webhook_url: str, df: pd.DataFrame, backtest: dict, config: dict,
                       ticker: str, title: str = "📊 Status Update", trade_state: dict = None):
    """Send a scheduled status update with chart to Discord."""
    try:
        # Generate chart
        chart_buf = generate_velocity_chart(df, backtest, config, ticker)

        # Build status message
        pos_status = "No position"
        if trade_state and trade_state.get('position') == 'long':
            entry_price = trade_state.get('entry_price', 0)
            current_price = df['close'].iloc[-1]
            pnl = ((current_price - entry_price) / entry_price) * 100 if entry_price else 0
            pos_status = f"LONG @ ${entry_price:.2f} ({pnl:+.1f}%)"
        elif backtest.get('current_position'):
            pos = backtest['current_position']
            pos_status = f"LONG @ ${pos['entry_price']:.2f} ({pos['unrealized_pnl']:+.1f}%)"

        msg = (
            f"**{title}: {ticker}**\n"
            f"**Strategy:** {config.get('strategy_name', 'velocity')}\n"
            f"**Signal Type:** {config.get('signal_type')}\n"
            f"---\n"
            f"📊 **Stats:**\n"
            f"• Trades: {backtest['num_trades']} | Win Rate: {backtest['win_rate']:.0f}%\n"
            f"• Return: {backtest['total_return']:.1f}% | PF: {backtest['profit_factor']:.1f}\n"
            f"• Position: {pos_status}\n"
            f"---\n"
            f"_Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_"
        )

        send_discord_alert(webhook_url, msg, chart_buf)
        print(f"✅ Sent status update: {title}")

    except Exception as e:
        print(f"⚠️ Failed to send status update: {e}")


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

    # Try to load saved data from Streamlit (exact match) or fetch new data
    saved_data_path = "production_env/velocity_data.parquet"
    backtest_days = 250

    print(f"📊 Running historical backtest...")
    try:
        if os.path.exists(saved_data_path):
            print(f"   Loading saved data from Streamlit: {saved_data_path}")
            df = pd.read_parquet(saved_data_path)
            print(f"   Loaded {len(df)} bars (exact Streamlit data)")
            # Data already has oscillator calculated - just ensure osc_smooth alias exists
            if 'composite_smooth' in df.columns and 'osc_smooth' not in df.columns:
                df['osc_smooth'] = df['composite_smooth']
        else:
            print(f"   Fetching {backtest_days} days of fresh data...")
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

    # Load or initialize trade state from backtest
    trade_state = load_trade_state()

    # If no saved state but backtest shows open position, initialize from backtest
    if trade_state.get('position') is None and backtest and backtest['current_position']:
        pos = backtest['current_position']
        trade_state = {
            'position': pos['position'],
            'entry_price': pos['entry_price'],
            'entry_time': pos['entry_date'],
            'last_signal_time': pos['entry_date'],
        }
        save_trade_state(trade_state)
        print(f"✅ Initialized position from historical backtest: LONG @ ${pos['entry_price']:.2f}")

    # Send startup notification with stats and chart
    startup_chart = None
    if backtest:
        pos_status = f"**Position:** LONG @ ${backtest['current_position']['entry_price']:.2f} ({backtest['current_position']['unrealized_pnl']:+.1f}%)" if backtest['current_position'] else "**Position:** None"
        startup_msg = (
            f"🤖 **[VELOCITY] Live Trader Started**\n"
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
            f"🤖 **[VELOCITY] Live Trader Started**\n"
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
            print(f"\n[{current_time_str}] Update Cycle...")

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
                    exit_msg = (
                        f"📤 **[VELOCITY] LONG EXIT - {ticker}** {pnl_emoji}\n"
                        f"**Reason:** {exit_reason}\n"
                        f"**Entry:** ${entry_price:.2f}\n"
                        f"**Exit:** ${current_price:.2f}\n"
                        f"**P&L:** {pnl_pct:+.2f}%\n"
                        f"**Time:** {current_time}\n"
                        f"{stats_section}"
                    )

                    send_discord_alert(webhook_url, exit_msg, exit_chart)
                    print(f"EXIT LONG: {exit_reason}")

                    trade_state['position'] = None
                    trade_state['entry_price'] = None
                    trade_state['entry_time'] = None
                    save_trade_state(trade_state)

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
                    exit_msg = (
                        f"📤 **[VELOCITY] SHORT EXIT - {ticker}** {pnl_emoji}\n"
                        f"**Reason:** {exit_reason}\n"
                        f"**Entry:** ${entry_price:.2f}\n"
                        f"**Exit:** ${current_price:.2f}\n"
                        f"**P&L:** {pnl_pct:+.2f}%\n"
                        f"**Time:** {current_time}\n"
                        f"{stats_section}"
                    )

                    send_discord_alert(webhook_url, exit_msg, exit_chart)
                    print(f"EXIT SHORT: {exit_reason}")

                    trade_state['position'] = None
                    trade_state['entry_price'] = None
                    trade_state['entry_time'] = None
                    save_trade_state(trade_state)

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
                        f"📈 **[VELOCITY] BUY SIGNAL - {ticker}**{signal_note}\n"
                        f"**Strategy:** {strategy_name}\n"
                        f"**Signal Type:** {config.get('signal_type')}\n"
                        f"**Signal Time:** {signal_time}\n"
                        f"**Entry Price:** ${current_price:.2f}\n"
                        f"**Oscillator:** {signal_bar['osc_smooth']:.4f}\n"
                        f"**Velocity:** {signal_bar['velocity']:.4f}\n"
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
                    save_trade_state(trade_state)

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
                        f"📉 **[VELOCITY] SELL SIGNAL - {ticker}**{signal_note}\n"
                        f"**Strategy:** {strategy_name}\n"
                        f"**Signal Type:** {config.get('signal_type')}\n"
                        f"**Signal Time:** {signal_time}\n"
                        f"**Entry Price:** ${current_price:.2f}\n"
                        f"**Oscillator:** {signal_bar['osc_smooth']:.4f}\n"
                        f"**Velocity:** {signal_bar['velocity']:.4f}\n"
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
                    save_trade_state(trade_state)

            print(f"Next check in {check_interval_seconds} seconds...")
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
