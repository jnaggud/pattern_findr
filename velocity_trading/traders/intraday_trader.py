"""
Intraday trader for velocity trading.

Handles intraday bar strategies for futures (ES=F, GC=F) and stocks.
Signals are checked after each bar closes (e.g., every 15 minutes).
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
from ..core.position_manager import normalize_timestamp
from ..data.market_hours import (
    is_market_open, is_bar_complete, seconds_until_bar_close, seconds_until_market_open,
    get_current_market_time, get_market_type, MARKET_CONFIG
)


# Interval to minutes mapping
INTERVAL_MINUTES = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '2h': 120,
    '4h': 240
}


class IntradayTrader(BaseTrader):
    """
    Trader for intraday bar strategies.

    Checks for signals after each bar closes based on the interval.
    Supports futures (CME hours) and stocks (NYSE hours).
    """

    def __init__(
        self,
        strategy_name: str,
        ticker: str,
        interval: str = '15m',
        webhook_url: str = None,
        config: Dict = None
    ):
        """
        Initialize intraday trader.

        Args:
            strategy_name: Unique strategy identifier
            ticker: Trading symbol (ES=F, GC=F, SPY, etc.)
            interval: Bar interval (15m, 30m, 1h, etc.)
            webhook_url: Discord webhook URL
            config: Strategy configuration dict
        """
        super().__init__(
            strategy_name=strategy_name,
            ticker=ticker,
            interval=interval,
            webhook_url=webhook_url,
            config=config
        )

        # Determine market type
        self.market_type = get_market_type(ticker)
        self.market_config = MARKET_CONFIG.get(self.market_type, MARKET_CONFIG['futures_cme'])

        # Get interval in minutes
        self.interval_minutes = INTERVAL_MINUTES.get(interval, 15)

        # Intraday-specific settings
        self.bar_completion_buffer_seconds = self.config.get('bar_completion_buffer', 5)
        self.last_processed_bar = None
        self._bar_processed_this_cycle = False  # Track if we processed a bar in current cycle

    @property
    def check_interval_seconds(self) -> int:
        """Check every 30 seconds for intraday strategies."""
        return self.config.get('check_interval_seconds', 30)

    def is_signal_window(self) -> bool:
        """
        Check if we're in the signal detection window.

        For futures: Check if market is open (CME hours with weekend closure)
        For stocks: Check if market is open (NYSE hours)
        """
        return is_market_open(self.ticker)

    def _get_current_bar_start(self, now: datetime) -> datetime:
        """
        Calculate the start time of the current bar.

        Args:
            now: Current datetime (timezone-aware)

        Returns:
            Start time of current bar
        """
        # Get minutes since midnight
        minutes_since_midnight = now.hour * 60 + now.minute

        # Round down to bar boundary
        bar_start_minutes = (minutes_since_midnight // self.interval_minutes) * self.interval_minutes

        bar_start = now.replace(
            hour=bar_start_minutes // 60,
            minute=bar_start_minutes % 60,
            second=0,
            microsecond=0
        )

        return bar_start

    def _get_previous_bar_end(self, now: datetime) -> datetime:
        """
        Calculate the end time of the previous completed bar.

        Args:
            now: Current datetime (timezone-aware)

        Returns:
            End time of previous bar (which is start of current bar)
        """
        return self._get_current_bar_start(now)

    def get_completed_bar(self, df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[datetime]]:
        """
        Get the most recently completed intraday bar.

        An intraday bar is complete when:
        - Current time >= bar_start + interval + buffer

        Returns:
            Tuple of (bar_data, bar_time) or (None, None)
        """
        if df is None or df.empty:
            return None, None

        now = get_current_market_time(self.ticker)

        # Get the last bar in the dataframe
        last_bar = df.iloc[-1]
        last_bar_time = last_bar.name

        # CRITICAL FIX: Data from Databento is stored in UTC, but may be read back as naive.
        # Previously, naive timestamps were incorrectly localized to the market's timezone
        # (e.g., 13:15 UTC treated as 13:15 Chicago = 19:15 UTC, causing 6 hour offset!)
        # Now we correctly assume naive timestamps are UTC and convert to market timezone.
        if last_bar_time.tzinfo is None:
            # Data is naive but represents UTC (from Databento)
            last_bar_time = pytz.UTC.localize(last_bar_time)
        # Convert to market timezone for consistent comparison
        market_tz = pytz.timezone(self.market_config.get('timezone', 'America/New_York'))
        last_bar_time = last_bar_time.astimezone(market_tz)

        # Calculate when this bar ends
        bar_end_time = last_bar_time + timedelta(minutes=self.interval_minutes)

        # Add buffer for data availability
        bar_complete_time = bar_end_time + timedelta(seconds=self.bar_completion_buffer_seconds)

        # Check if the bar is complete
        if now >= bar_complete_time:
            # Check if we already processed this bar
            bar_key = str(last_bar_time)[:16]  # YYYY-MM-DD HH:MM
            if self.last_processed_bar == bar_key:
                return None, None

            return last_bar, last_bar_time
        else:
            # Current bar not complete, try previous bar
            if len(df) >= 2:
                prev_bar = df.iloc[-2]
                prev_bar_time = prev_bar.name

                # Same timezone fix for previous bar
                if prev_bar_time.tzinfo is None:
                    prev_bar_time = pytz.UTC.localize(prev_bar_time)
                prev_bar_time = prev_bar_time.astimezone(market_tz)

                bar_key = str(prev_bar_time)[:16]
                if self.last_processed_bar == bar_key:
                    return None, None

                return prev_bar, prev_bar_time

            return None, None

    def _check_entry(self, df: pd.DataFrame):
        """
        Override to track processed bars and prevent duplicate signals.
        """
        # Reset cycle flag at start of each check
        self._bar_processed_this_cycle = False

        if not self.is_signal_window():
            print("   Market closed, skipping signal check")
            return

        completed_bar, bar_time = self.get_completed_bar(df)
        if completed_bar is None or bar_time is None:
            print("   No completed bar available yet")
            return

        # Normalize timestamp to ISO format for consistent DB storage
        bar_timestamp = normalize_timestamp(bar_time)
        bar_key = bar_timestamp[:16]  # YYYY-MM-DDTHH:MM for display

        # Check if we already processed this bar (memory check for same session)
        if self.last_processed_bar == bar_timestamp:
            print(f"   Bar {bar_key} already processed, skipping")
            return

        # Mark that we're processing a bar this cycle
        self._bar_processed_this_cycle = True

        # Get signal info for logging
        buy_signal = completed_bar.get('buy_signal', False)
        sell_signal = completed_bar.get('sell_signal', False)
        osc_value = completed_bar.get('JD_Osc', completed_bar.get('osc_smooth', 0))
        velocity = completed_bar.get('velocity', 0)
        rsc = completed_bar.get('RSC', 'N/A')

        print(f"   Bar {bar_key}: Osc={osc_value:.4f} Vel={velocity:.4f} RSC={rsc} | Buy={buy_signal} Sell={sell_signal}")

        # Check for buy signal
        if not buy_signal:
            # Mark as processed even if no signal - PERSIST TO DATABASE
            # This prevents incremental updates from reprocessing this bar
            self.last_processed_bar = bar_timestamp
            self.pm._db.set_last_processed_bar(bar_timestamp)
            return

        print(f"   *** BUY SIGNAL DETECTED at {bar_key} ***")

        # Call parent implementation for actual entry
        super()._check_entry(df)

        # Mark bar as processed - PERSIST TO DATABASE
        self.last_processed_bar = bar_timestamp
        self.pm._db.set_last_processed_bar(bar_timestamp)

    def _calculate_sleep_time(self) -> int:
        """
        Calculate sleep time based on position status and market hours.

        When NOT in position: Sleep until bar closes (precision timing for signals)
        When IN position + market OPEN: Check every 5 seconds (for stop loss/take profit monitoring)
        When IN position + market CLOSED: Sleep until market opens (no point checking)
        """
        try:
            # CRITICAL: When in a position, check frequently for SL/TP
            position = self.pm.get_current_position()
            if position:
                # But only if market is open - no point checking when market is closed
                if not is_market_open(self.ticker):
                    seconds_to_open = seconds_until_market_open(self.ticker)
                    if seconds_to_open > 60:  # More than a minute until open
                        # Cap at 5 minutes to periodically re-check market status
                        sleep_seconds = min(seconds_to_open, 300)
                        hours = sleep_seconds // 3600
                        mins = (sleep_seconds % 3600) // 60
                        if hours > 0:
                            print(f"   📍 In position but market closed - sleeping {hours}h {mins}m until market opens")
                        else:
                            print(f"   📍 In position but market closed - sleeping {mins}m until market opens")
                        return sleep_seconds

                # Market is open - check every 5 seconds when in position for timely SL/TP
                sleep_seconds = self.config.get('position_check_interval', 5)
                print(f"   📍 In position - checking SL/TP every {sleep_seconds}s")
                return sleep_seconds

            # When not in position, use precision timing (sleep until bar closes)
            now = get_current_market_time(self.ticker)

            # Calculate seconds into current bar
            minutes_into_bar = (now.minute % self.interval_minutes) + (now.second / 60)
            seconds_into_bar = int(minutes_into_bar * 60)

            # Seconds until bar closes
            bar_duration_seconds = self.interval_minutes * 60
            seconds_until_close = bar_duration_seconds - seconds_into_bar

            # Add buffer for data availability - MUST match bar_completion_buffer_seconds
            # Otherwise we wake up before the bar is considered "complete" and
            # recalculate sleep as if we're at the start of the next bar
            sleep_seconds = max(1, seconds_until_close + self.bar_completion_buffer_seconds)

            # EDGE CASE: If we just woke up but the bar wasn't complete yet (timing drift),
            # we're now very early in the "next" bar and would sleep the full interval.
            # If we're within 2x the buffer window AND we didn't process a bar, retry quickly.
            # But if we DID process a bar, sleep until the next bar (normal flow).
            if seconds_into_bar <= (self.bar_completion_buffer_seconds * 2) and not self._bar_processed_this_cycle:
                retry_seconds = self.bar_completion_buffer_seconds + 1
                print(f"   Early in bar ({seconds_into_bar}s), retrying in {retry_seconds}s")
                return retry_seconds

            # Calculate when we'll wake up (for logging)
            wake_time = now + timedelta(seconds=sleep_seconds)
            print(f"   Bar closes in {seconds_until_close}s, waking at {wake_time.strftime('%H:%M:%S')}")

            return sleep_seconds

        except Exception as e:
            print(f"   Timing error: {e}, using default interval")
            return self.check_interval_seconds


def run_intraday_trader(
    ticker: str,
    interval: str,
    strategy_name: str,
    config: Dict,
    webhook_url: str = None
):
    """
    Run an intraday trader.

    CRITICAL: All required parameters must be provided - no defaults!
    Config must come from velocity_strategies bundle.

    Args:
        ticker: Trading symbol
        interval: Bar interval (15m, 30m, 1h)
        strategy_name: Strategy identifier
        config: Strategy configuration (REQUIRED - from bundle)
        webhook_url: Discord webhook URL (or set DISCORD_WEBHOOK_URL env var)
    """
    if webhook_url is None:
        webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')

    # Validate config has required parameters
    required_params = ['signal_type', 'oversold_threshold', 'overbought_threshold',
                       'stop_loss_pct', 'take_profit_pct']
    missing = [p for p in required_params if p not in config]
    if missing:
        raise ValueError(
            f"Missing required config parameters: {missing}. "
            f"Config must come from strategy bundle (velocity_strategies/{strategy_name}/)."
        )

    trader = IntradayTrader(
        strategy_name=strategy_name,
        ticker=ticker,
        interval=interval,
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

    CRITICAL: Returns config values from bundle WITHOUT adding defaults.
    Missing required parameters will cause BaseTrader to fail fast.

    Returns normalized config dict or None if not found.
    """
    bundles = get_available_bundles()

    if strategy_name not in bundles:
        return None

    bundle = bundles[strategy_name]
    config = bundle['config']
    print(f"   Loaded config from: {bundle['path']}")

    # CRITICAL: Validate required parameters exist in bundle
    required_params = ['signal_type', 'oversold_threshold', 'overbought_threshold',
                       'stop_loss_pct', 'take_profit_pct']
    missing = [p for p in required_params if p not in config]
    if missing:
        print(f"   WARNING: Bundle missing required params: {missing}")
        print(f"   Re-run optimization in Streamlit to fix the bundle.")
        return None

    # Map bundle config keys to our expected format - NO DEFAULTS for trading params
    return {
        'ticker': config.get('ticker'),
        'interval': config.get('interval', '15m'),  # Non-critical default ok
        # Bundle identification - CRITICAL for importing trades from bundle
        'bundle_name': config.get('bundle_name'),
        'strategy_name': config.get('strategy_name'),
        # CRITICAL trading parameters - pass through from bundle, NO DEFAULTS
        'signal_type': config['signal_type'],
        'stop_loss_pct': config['stop_loss_pct'],
        'take_profit_pct': config['take_profit_pct'],
        'oversold_threshold': config['oversold_threshold'],
        'overbought_threshold': config['overbought_threshold'],
        # Oscillator type (for novel oscillators: arwo, prf, ics, etc.)
        'oscillator_type': config.get('oscillator_type', 'composite'),
        # Signal generation parameters
        'vel_smoothing': config.get('vel_smoothing', 1),
        'extreme_zone_mult': config.get('extreme_zone_mult', 1.5),
        'require_accel': config.get('require_accel', False),
        # V2 Filters (Regime, Fragility, Entropy)
        'use_regime_filter': config.get('use_regime_filter', False),
        'regime_threshold': config.get('regime_threshold', -0.15),
        'use_fragility_filter': config.get('use_fragility_filter', False),
        'fragility_threshold': config.get('fragility_threshold', 0.5),
        'use_entropy_filter': config.get('use_entropy_filter', False),
        'entropy_threshold': config.get('entropy_threshold', 0.7),
        # Non-critical parameters can have safe defaults
        'exit_on_opposite_signal': config.get('exit_on_opposite_signal', True),
        'exit_on_midline_cross': config.get('exit_on_midline_cross', False),
        'webhook': config.get('discord_webhook'),
    }


# ============================================================
# DEPRECATED: STRATEGY CONFIGURATIONS
# ============================================================
# DO NOT USE - These hardcoded configs caused the parameter mismatch bugs.
# All strategies MUST load config from velocity_strategies/{name}_*/velocity_config.json
# The bundle configs are created by Streamlit UI optimization and contain the
# correct optimized parameters.
#
# REMOVED - If you see this, you're looking at legacy code that should not be used.
STRATEGY_CONFIGS = {}  # Intentionally empty - use bundle configs only

# Test webhook for all strategies during testing
TEST_WEBHOOK = ''


def select_strategy_interactive():
    """Prompt user to select a strategy from velocity_strategies bundles."""
    # Get all available bundles
    bundles = get_available_bundles()

    # Filter to intraday strategies (15m, 30m, 1h intervals)
    intraday_intervals = {'1m', '5m', '15m', '30m', '1h', '2h', '4h'}
    intraday_bundles = {
        name: b for name, b in bundles.items()
        if b['config'].get('interval', '1d') in intraday_intervals
    }

    if not intraday_bundles:
        print("No intraday strategy bundles found in velocity_strategies/")
        print("Run optimization in Streamlit UI first.")
        sys.exit(1)

    strategies = sorted(intraday_bundles.keys())

    print("\n" + "=" * 50)
    print("  VELOCITY INTRADAY TRADER")
    print("=" * 50)
    print("\nSelect a strategy:\n")

    for i, name in enumerate(strategies, 1):
        cfg = intraday_bundles[name]['config']
        print(f"  [{i}] {name}")
        print(f"      {cfg.get('ticker', '?')} | {cfg.get('interval', '?')} | {cfg.get('signal_type', '?')}")
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

    parser = argparse.ArgumentParser(description='Run intraday velocity trader')
    parser.add_argument('--strategy', '-s', help='Strategy name (or omit to select interactively)')
    parser.add_argument('--test', action='store_true', help='Use test Discord channel')
    parser.add_argument('--list', action='store_true', help='List available strategies')

    # Optional overrides
    parser.add_argument('--ticker', '-t', help='Override ticker')
    parser.add_argument('--interval', '-i', help='Override interval')
    parser.add_argument('--webhook', '-w', help='Override webhook URL')

    args = parser.parse_args()

    # List strategies from velocity_strategies bundles
    if args.list:
        bundles = get_available_bundles()
        intraday_intervals = {'1m', '5m', '15m', '30m', '1h', '2h', '4h'}
        print("Available intraday strategies (from velocity_strategies/):")
        for name, b in sorted(bundles.items()):
            cfg = b['config']
            if cfg.get('interval', '1d') in intraday_intervals:
                print(f"  {name} ({cfg.get('ticker')} {cfg.get('interval')})")
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

    # Build config dict - pass through bundle values WITHOUT adding defaults
    # Critical trading params come directly from bundle (already validated)
    config = {
        # Bundle identification - CRITICAL for importing trades from bundle
        'bundle_name': cfg.get('bundle_name'),
        'strategy_name': cfg.get('strategy_name'),
        # Trading parameters
        'stop_loss_pct': cfg['stop_loss_pct'],
        'take_profit_pct': cfg['take_profit_pct'],
        'signal_type': cfg['signal_type'],
        'oversold_threshold': cfg['oversold_threshold'],
        'overbought_threshold': cfg['overbought_threshold'],
        # Oscillator type (for novel oscillators: arwo, prf, ics, etc.)
        'oscillator_type': cfg.get('oscillator_type', 'composite'),
        # Signal generation parameters
        'vel_smoothing': cfg.get('vel_smoothing', 1),
        'extreme_zone_mult': cfg.get('extreme_zone_mult', 1.5),
        'require_accel': cfg.get('require_accel', False),
        # V2 Filters (Regime, Fragility, Entropy)
        'use_regime_filter': cfg.get('use_regime_filter', False),
        'regime_threshold': cfg.get('regime_threshold', -0.15),
        'use_fragility_filter': cfg.get('use_fragility_filter', False),
        'fragility_threshold': cfg.get('fragility_threshold', 0.5),
        'use_entropy_filter': cfg.get('use_entropy_filter', False),
        'entropy_threshold': cfg.get('entropy_threshold', 0.7),
        # Non-critical params can have safe defaults
        'exit_on_opposite_signal': cfg.get('exit_on_opposite_signal', True),
        'exit_on_midline_cross': cfg.get('exit_on_midline_cross', False),
        'check_interval_seconds': 30,
        'bar_completion_buffer': 5
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
    interval = args.interval or cfg['interval']

    print(f"\nStarting {strategy_name}...")
    print(f"  Ticker: {ticker}")
    print(f"  Interval: {interval}")
    print(f"  Test mode: {args.test}")
    print()

    run_intraday_trader(
        ticker=ticker,
        interval=interval,
        strategy_name=strategy_name,
        webhook_url=webhook,
        config=config
    )
