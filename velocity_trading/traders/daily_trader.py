"""
Daily trader for velocity trading.

Handles daily bar strategies for stocks (SPY) and crypto (BTC-USD).
Signals are checked after market close when the daily bar is complete.
"""

import os
import sys
import json
import glob
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from pathlib import Path
import pandas as pd
import pytz

# Add parent to path for imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Directory where Streamlit UI saves optimized strategies
VELOCITY_STRATEGIES_DIR = os.path.join(PARENT_DIR, "velocity_strategies")

from .base_trader import BaseTrader
from ..data.market_hours import (
    is_market_open, is_bar_complete, get_current_market_time,
    get_market_type, MARKET_CONFIG
)


class DailyTrader(BaseTrader):
    """
    Trader for daily bar strategies.

    Checks for signals after market close when daily bar is complete.
    Supports stocks (NYSE hours) and crypto (24/7 with UTC daily bars).
    """

    def __init__(
        self,
        strategy_name: str,
        ticker: str,
        webhook_url: str = None,
        config: Dict = None
    ):
        """
        Initialize daily trader.

        Args:
            strategy_name: Unique strategy identifier
            ticker: Trading symbol (SPY, BTC-USD, etc.)
            webhook_url: Discord webhook URL
            config: Strategy configuration dict
        """
        super().__init__(
            strategy_name=strategy_name,
            ticker=ticker,
            interval='1d',
            webhook_url=webhook_url,
            config=config
        )

        # Determine market type
        self.market_type = get_market_type(ticker)
        self.market_config = MARKET_CONFIG.get(self.market_type, MARKET_CONFIG['stocks_us'])

        # Daily-specific settings
        self.signal_window_minutes = self.config.get('signal_window_minutes', 120)  # 2 hours after close
        self.last_processed_date = None

    @property
    def check_interval_seconds(self) -> int:
        """Check every 2 minutes for daily strategies."""
        return self.config.get('check_interval_seconds', 120)

    def is_signal_window(self) -> bool:
        """
        Check if we're in the signal detection window.

        For stocks: After 4 PM ET until midnight
        For crypto: After 00:00 UTC until next bar
        """
        now = get_current_market_time(self.ticker)

        if self.market_type == 'crypto':
            # Crypto: Always in signal window (24/7)
            # But we only process once per day
            return True

        elif self.market_type == 'stocks_us':
            # Stocks: After market close (4 PM ET) until midnight
            market_close = self.market_config['market_close']
            close_hour, close_minute = market_close.hour, market_close.minute

            # Create today's close time
            close_time = now.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)

            # Signal window: close + 5 min until close + signal_window_minutes
            window_start = close_time + timedelta(minutes=5)
            window_end = close_time + timedelta(minutes=self.signal_window_minutes)

            return window_start <= now <= window_end

        else:
            # Default: Always check
            return True

    def get_completed_bar(self, df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[datetime]]:
        """
        Get the most recently completed daily bar.

        For stocks: Yesterday's bar if before today's close, today's bar if after close
        For crypto: Previous UTC day's bar

        Returns:
            Tuple of (bar_data, bar_time) or (None, None)
        """
        if df is None or df.empty:
            return None, None

        now = get_current_market_time(self.ticker)
        today = now.date()

        # Get the last bar
        last_bar = df.iloc[-1]
        last_bar_date = last_bar.name.date() if hasattr(last_bar.name, 'date') else None

        if last_bar_date is None:
            return None, None

        # Check if this bar is complete
        if self.market_type == 'crypto':
            # Crypto daily bars complete at 00:00 UTC
            # Last bar is complete if it's from a previous day
            if last_bar_date < today:
                return last_bar, last_bar.name
            else:
                # Today's bar not complete yet
                if len(df) >= 2:
                    return df.iloc[-2], df.iloc[-2].name
                return None, None

        elif self.market_type == 'stocks_us':
            # Stock daily bars complete at market close (4 PM ET)
            market_close = self.market_config['market_close']
            close_hour, close_minute = market_close.hour, market_close.minute
            close_time = now.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)

            if last_bar_date == today:
                # Today's bar - only complete if after close
                if now > close_time + timedelta(minutes=5):
                    # Check if we already processed today
                    if self.last_processed_date == today:
                        return None, None
                    return last_bar, last_bar.name
                else:
                    # Before close - use previous bar
                    if len(df) >= 2:
                        return df.iloc[-2], df.iloc[-2].name
                    return None, None
            elif last_bar_date < today:
                # Previous day's bar is complete
                return last_bar, last_bar.name
            else:
                return None, None
        else:
            # Default: Last bar is complete
            return last_bar, last_bar.name

    def _check_entry(self, df: pd.DataFrame):
        """
        Override to track processed dates and prevent duplicate signals.
        """
        if not self.is_signal_window():
            return

        completed_bar, bar_time = self.get_completed_bar(df)
        if completed_bar is None or bar_time is None:
            return

        bar_date = bar_time.date() if hasattr(bar_time, 'date') else None

        # Check if we already processed this date
        if bar_date and self.last_processed_date == bar_date:
            return

        # Check for buy signal
        if not completed_bar.get('buy_signal', False):
            # Mark as processed even if no signal (prevent re-checking)
            if bar_date:
                self.last_processed_date = bar_date
            return

        # Call parent implementation for actual entry
        super()._check_entry(df)

        # Mark date as processed after successful check
        if bar_date:
            self.last_processed_date = bar_date


def run_daily_trader(
    ticker: str,
    strategy_name: str,
    config: Dict,
    webhook_url: str = None
):
    """
    Run a daily trader.

    IMPORTANT: Config must be explicitly provided. No default values are used
    to ensure behavior matches the original Streamlit backtest.

    Args:
        ticker: Trading symbol (required)
        strategy_name: Strategy identifier (required)
        config: Strategy configuration (required) - must include all trading parameters
        webhook_url: Discord webhook URL (optional, or set DISCORD_WEBHOOK_URL env var)
    """
    if config is None:
        raise ValueError(
            "config is required - no default values allowed. "
            "Load config from bundle using load_bundle_config()."
        )

    # Validate required config parameters
    required_params = ['signal_type', 'oversold_threshold', 'overbought_threshold',
                       'stop_loss_pct', 'take_profit_pct']
    missing = [p for p in required_params if p not in config]
    if missing:
        raise ValueError(
            f"Config missing required parameters: {missing}. "
            f"All parameters must be explicitly set to ensure consistent behavior."
        )

    if webhook_url is None:
        webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')

    trader = DailyTrader(
        strategy_name=strategy_name,
        ticker=ticker,
        webhook_url=webhook_url,
        config=config
    )

    trader.run()


def get_available_bundles() -> Dict[str, Dict]:
    """
    Scan velocity_strategies/ for available strategy bundles.

    Returns dict of {strategy_name: config} for the newest bundle of each strategy.
    """
    bundles = {}

    if not os.path.exists(VELOCITY_STRATEGIES_DIR):
        return bundles

    # Find all velocity_config.json files
    pattern = os.path.join(VELOCITY_STRATEGIES_DIR, "*", "velocity_config.json")
    config_files = glob.glob(pattern)

    # Group by strategy name (remove timestamp suffix)
    import re
    for config_path in config_files:
        try:
            with open(config_path) as f:
                config = json.load(f)

            strategy_name = config.get('strategy_name')
            if not strategy_name:
                # Extract from directory name
                dir_name = os.path.basename(os.path.dirname(config_path))
                # Remove timestamp suffix like _20251222_105550
                strategy_name = re.sub(r'_\d{8}_\d{6}$', '', dir_name)

            # Keep newest (by file mtime)
            if strategy_name not in bundles:
                bundles[strategy_name] = {
                    'config': config,
                    'path': config_path,
                    'mtime': os.path.getmtime(config_path)
                }
            else:
                if os.path.getmtime(config_path) > bundles[strategy_name]['mtime']:
                    bundles[strategy_name] = {
                        'config': config,
                        'path': config_path,
                        'mtime': os.path.getmtime(config_path)
                    }
        except Exception as e:
            continue

    return bundles


def load_bundle_config(strategy_name: str) -> Optional[Dict]:
    """
    Load config for a strategy from its velocity_strategies bundle.

    Returns normalized config dict or None if not found.

    IMPORTANT: Does NOT apply default values. All parameters must be
    present in the bundle config. This ensures consistent behavior
    with the Streamlit backtest that created the bundle.
    """
    bundles = get_available_bundles()

    if strategy_name not in bundles:
        return None

    bundle = bundles[strategy_name]
    config = bundle['config']
    print(f"   Loaded config from: {bundle['path']}")

    # Validate required parameters exist in bundle
    required_params = [
        'ticker', 'signal_type', 'oversold_threshold', 'overbought_threshold',
        'stop_loss_pct', 'take_profit_pct'
    ]
    missing = [p for p in required_params if p not in config]
    if missing:
        raise ValueError(
            f"Bundle config missing required parameters: {missing}. "
            f"Strategy bundles must contain all trading parameters to ensure "
            f"consistent behavior with the original backtest."
        )

    # Map bundle config keys to our expected format - pass through without defaults
    return {
        'ticker': config['ticker'],
        'data_period': config.get('optimization_period') or config.get('data_period', '1y'),
        # Bundle identification - CRITICAL for importing trades from bundle
        'bundle_name': config.get('bundle_name'),
        'strategy_name': config.get('strategy_name'),
        'signal_type': config['signal_type'],
        'stop_loss_pct': config['stop_loss_pct'],
        'take_profit_pct': config['take_profit_pct'],
        'oversold_threshold': config['oversold_threshold'],
        'overbought_threshold': config['overbought_threshold'],
        # Oscillator type (for novel oscillators: arwo, prf, ics, etc.)
        'oscillator_type': config.get('oscillator_type', 'composite'),
        'exit_on_opposite_signal': config.get('exit_on_opposite_signal', True),
        'exit_on_midline_cross': config.get('exit_on_midline_cross', False),
        'vel_smoothing': config.get('vel_smoothing', 3),
        'extreme_zone_mult': config.get('extreme_zone_mult', 1.5),
        'webhook': config.get('discord_webhook'),
    }


# ============================================================
# STRATEGY_CONFIGS - DEPRECATED
# ============================================================
# These hardcoded fallback configs are NO LONGER USED.
# All strategies MUST load config from velocity_strategies/ bundles.
# This ensures parameters match the original Streamlit backtest.
#
# If you need to run a strategy, first create a bundle via Streamlit UI:
#   1. Run optimization in Streamlit
#   2. Save the strategy (creates velocity_strategies/{name}_*/)
#   3. Run: python -m velocity_trading.traders.daily_trader -s {name}
#
# The webhook URLs are preserved here for reference only.
STRATEGY_CONFIGS = {}  # DEPRECATED - not used

# Test webhook for all strategies during testing
TEST_WEBHOOK = ''


def select_strategy_interactive():
    """Prompt user to select a strategy from velocity_strategies bundles."""
    # Get all available bundles
    bundles = get_available_bundles()

    # Filter to daily strategies (1d interval)
    daily_bundles = {
        name: b for name, b in bundles.items()
        if b['config'].get('interval', '1d') == '1d'
    }

    if not daily_bundles:
        print("No daily strategy bundles found in velocity_strategies/")
        print("Run optimization in Streamlit UI first.")
        sys.exit(1)

    strategies = sorted(daily_bundles.keys())

    print("\n" + "=" * 70)
    print("  VELOCITY DAILY TRADER")
    print("=" * 70)
    print("\nSelect a strategy:\n")

    for i, name in enumerate(strategies, 1):
        cfg = daily_bundles[name]['config']
        # Extract full bundle directory name (includes timestamp)
        bundle_dir = os.path.basename(os.path.dirname(daily_bundles[name]['path']))
        print(f"  [{i}] {name}")
        print(f"      Bundle: {bundle_dir}")
        print(f"      {cfg.get('ticker', '?')} | {cfg.get('optimization_period', '?')} | {cfg.get('signal_type', '?')}")
        print()

    while True:
        try:
            choice = input("Enter number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                sys.exit(0)
            idx = int(choice) - 1
            if 0 <= idx < len(strategies):
                return strategies[idx]
            print(f"Please enter 1-{len(strategies)}")
        except ValueError:
            print("Please enter a number")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run daily velocity trader')
    parser.add_argument('--strategy', '-s', help='Strategy name (or omit to select interactively)')
    parser.add_argument('--test', action='store_true', help='Use test Discord channel')
    parser.add_argument('--list', action='store_true', help='List available strategies')

    # Optional overrides
    parser.add_argument('--ticker', '-t', help='Override ticker')
    parser.add_argument('--webhook', '-w', help='Override webhook URL')

    args = parser.parse_args()

    # List strategies from velocity_strategies bundles
    if args.list:
        bundles = get_available_bundles()
        print("Available daily strategies (from velocity_strategies/):")
        for name, b in sorted(bundles.items()):
            cfg = b['config']
            if cfg.get('interval', '1d') == '1d':
                print(f"  {name} ({cfg.get('ticker')} {cfg.get('optimization_period', '?')})")
        sys.exit(0)

    # Interactive selection if no strategy specified
    if args.strategy:
        strategy_name = args.strategy
    else:
        strategy_name = select_strategy_interactive()

        # Also ask about test mode if not specified via --test
        if not args.test:
            test_choice = input("\nUse TEST channel? [y/N]: ").strip().lower()
            if test_choice == 'y':
                args.test = True

    # Load strategy config from bundle
    cfg = load_bundle_config(strategy_name)
    if cfg is None:
        print(f"Strategy not found: {strategy_name}")
        print("Run optimization in Streamlit UI to create strategy bundles")
        sys.exit(1)

    # Build config dict - pass values directly from bundle WITHOUT defaults
    # This ensures behavior matches the original Streamlit backtest
    config = {
        'stop_loss_pct': cfg['stop_loss_pct'],
        'take_profit_pct': cfg['take_profit_pct'],
        'signal_type': cfg['signal_type'],
        'oversold_threshold': cfg['oversold_threshold'],
        'overbought_threshold': cfg['overbought_threshold'],
        'exit_on_opposite_signal': cfg.get('exit_on_opposite_signal', True),
        'exit_on_midline_cross': cfg.get('exit_on_midline_cross', False),
        'data_period': cfg['data_period'],
    }

    # Determine webhook
    if args.test:
        webhook = TEST_WEBHOOK
        print("*** TEST MODE - posting to test channel ***")
    elif args.webhook:
        webhook = args.webhook
    else:
        webhook = cfg.get('webhook') or os.environ.get('DISCORD_WEBHOOK_URL')

    # Allow overrides
    ticker = args.ticker or cfg['ticker']

    print(f"\nStarting {strategy_name}...")
    print(f"  Ticker: {ticker}")
    print(f"  Data Period: {cfg['data_period']}")
    print(f"  Test mode: {args.test}")
    print()

    run_daily_trader(
        ticker=ticker,
        strategy_name=strategy_name,
        webhook_url=webhook,
        config=config
    )
