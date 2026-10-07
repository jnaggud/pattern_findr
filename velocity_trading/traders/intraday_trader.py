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
        self.bar_completion_buffer_seconds = self.config.get('bar_completion_buffer', 3)
        self.last_processed_bar = None
        self._bar_processed_this_cycle = False  # Track if we processed a bar in current cycle

        # min_bars_between cooldown: track last exit time to enforce cooldown
        self.min_bars_between = self.config.get('min_bars_between', 1)
        self._last_exit_bar_time = None  # Set on exit, checked on entry

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

        Scans backwards from the end of the dataframe to find the newest bar
        whose end time (bar_start + interval + buffer) has passed. This avoids
        the bug where df.iloc[-1] is an incomplete current bar, causing the
        method to return df.iloc[-2] (one bar too old) instead of the just-completed bar.

        Returns:
            Tuple of (bar_data, bar_time) or (None, None)
        """
        if df is None or df.empty:
            return None, None

        now = get_current_market_time(self.ticker)
        market_tz = pytz.timezone(self.market_config.get('timezone', 'America/New_York'))

        # Scan backwards through the last few bars to find the most recent completed one
        scan_limit = min(5, len(df))
        for offset in range(scan_limit):
            idx = len(df) - 1 - offset
            bar = df.iloc[idx]
            bar_time = bar.name

            # Normalize timezone: naive timestamps from Databento are UTC
            if bar_time.tzinfo is None:
                bar_time = pytz.UTC.localize(bar_time)
            bar_time = bar_time.astimezone(market_tz)

            # Calculate when this bar ends
            bar_end_time = bar_time + timedelta(minutes=self.interval_minutes)
            bar_complete_time = bar_end_time + timedelta(seconds=self.bar_completion_buffer_seconds)

            # Check if this bar is complete
            if now >= bar_complete_time:
                # Check if we already processed this bar
                bar_key = str(bar_time)[:16]  # YYYY-MM-DD HH:MM
                if self.last_processed_bar == bar_key:
                    return None, None

                return bar, bar_time

        return None, None

    def _get_unprocessed_bars(self, df: pd.DataFrame):
        """
        Get ALL completed but unprocessed bars since last_processed_bar.

        Unlike get_completed_bar() which returns only the most recent bar,
        this scans backwards to find ALL bars that haven't been processed yet.
        This prevents missed entries when a cycle is slow or bars are skipped.

        Returns:
            List of (bar_data, bar_time) tuples in chronological order
        """
        if df is None or df.empty:
            return []

        import pytz
        now = get_current_market_time(self.ticker)
        market_tz = pytz.timezone(self.market_config.get('timezone', 'America/New_York'))

        # Parse last_processed_bar to a comparable timestamp
        last_processed_dt = None
        if self.last_processed_bar:
            try:
                last_processed_dt = pd.to_datetime(self.last_processed_bar)
                if last_processed_dt.tzinfo is None:
                    last_processed_dt = pytz.UTC.localize(last_processed_dt)
                last_processed_dt = last_processed_dt.astimezone(market_tz)
            except Exception:
                last_processed_dt = None

        unprocessed = []
        for i in range(len(df) - 1, -1, -1):
            bar = df.iloc[i]
            bar_time = bar.name

            # Normalize timezone
            if bar_time.tzinfo is None:
                bar_time = pytz.UTC.localize(bar_time)
            bar_time = bar_time.astimezone(market_tz)

            # Check if bar is complete
            bar_end = bar_time + timedelta(minutes=self.interval_minutes)
            bar_complete = bar_end + timedelta(seconds=self.bar_completion_buffer_seconds)
            if now < bar_complete:
                continue  # Bar not yet complete

            # Check if already processed
            if last_processed_dt is not None and bar_time <= last_processed_dt:
                break  # All earlier bars are processed

            unprocessed.append((bar, bar_time))

            # Safety limit: don't scan back more than 20 bars (~5 hours for 15m)
            if len(unprocessed) >= 20:
                break

        # Return in chronological order (oldest first)
        unprocessed.reverse()
        return unprocessed

    def _check_entry(self, df: pd.DataFrame):
        """
        Override to track processed bars and prevent duplicate signals.

        CRITICAL: Scans ALL unprocessed bars (not just the latest one) to
        prevent missed entries when a cycle is slow or timing drifts.
        """
        # Reset cycle flag at start of each check
        self._bar_processed_this_cycle = False

        if not self.is_signal_window():
            print("   Market closed, skipping signal check")
            return

        # Get ALL unprocessed bars, not just the latest one
        unprocessed_bars = self._get_unprocessed_bars(df)
        if not unprocessed_bars:
            print("   No completed bar available yet")
            return

        if len(unprocessed_bars) > 1:
            print(f"   ⚠️ {len(unprocessed_bars)} unprocessed bars found, scanning all")

        for completed_bar, bar_time in unprocessed_bars:
            # Normalize timestamp to ISO format for consistent DB storage
            bar_timestamp = normalize_timestamp(bar_time)
            bar_key = bar_timestamp[:16]  # YYYY-MM-DDTHH:MM for display

            # Mark that we're processing a bar this cycle
            self._bar_processed_this_cycle = True

            # Get signal info for logging
            buy_signal = completed_bar.get('buy_signal', False)
            sell_signal = completed_bar.get('sell_signal', False)
            osc_value = completed_bar.get('JD_Osc', completed_bar.get('osc_smooth', 0))
            velocity = completed_bar.get('velocity', 0)
            rsc = completed_bar.get('RSC', 'N/A')

            # v9 regime-aware: check regime-specific buy signal
            regime_tag = ""
            regime_name = None
            if self.regime_aware and self._regimes is not None and len(self._regimes) > 0:
                from .base_trader import REGIME_NAMES
                bar_idx = df.index.get_loc(bar_time) if bar_time in df.index else len(df) - 2
                if 0 <= bar_idx < len(self._regimes):
                    regime_id = int(self._regimes[bar_idx])
                    regime_name = REGIME_NAMES.get(regime_id, 'unknown')
                    regime_tag = f" Regime={regime_name.upper()}"

                    # Override buy_signal with regime-specific signal
                    if regime_name in self._regime_signal_dfs:
                        regime_df = self._regime_signal_dfs[regime_name]
                        if bar_time in regime_df.index:
                            buy_signal = regime_df.loc[bar_time].get('buy_signal', False)
                            sell_signal = regime_df.loc[bar_time].get('sell_signal', False)
                        else:
                            buy_signal = False
                            sell_signal = False
                    else:
                        # Regime is dont_trade (e.g., chop)
                        buy_signal = False
                        sell_signal = False

            print(f"   Bar {bar_key}: Osc={osc_value:.4f} Vel={velocity:.4f} RSC={rsc}{regime_tag} | Buy={buy_signal} Sell={sell_signal}")

            # Check for buy signal
            if not buy_signal:
                # Mark as processed even if no signal - PERSIST TO DATABASE
                # This prevents incremental updates from reprocessing this bar
                self.last_processed_bar = bar_timestamp
                self.pm._db.set_last_processed_bar(bar_timestamp)
                continue

            print(f"   *** BUY SIGNAL DETECTED at {bar_key} ***")

            # min_bars_between cooldown: skip entry if not enough bars since last exit
            if self._last_exit_bar_time is not None and self.min_bars_between > 1:
                exit_ts = pd.Timestamp(self._last_exit_bar_time)
                bar_ts = pd.Timestamp(bar_time)
                # Strip timezone for comparison
                if exit_ts.tzinfo is not None:
                    exit_ts = exit_ts.tz_convert('UTC').tz_localize(None)
                if bar_ts.tzinfo is not None:
                    bar_ts = bar_ts.tz_convert('UTC').tz_localize(None)
                bars_since_exit = int((bar_ts - exit_ts).total_seconds() / (self.interval_minutes * 60))
                if bars_since_exit < self.min_bars_between:
                    print(f"   ⏳ Cooldown: {bars_since_exit}/{self.min_bars_between} bars since last exit, skipping")
                    self.last_processed_bar = bar_timestamp
                    self.pm._db.set_last_processed_bar(bar_timestamp)
                    continue

            # Novel Strategy Filters (v7+): Check GBM + XGBoost confidence
            if self.novel_filters is not None and self.novel_filters.trained:
                # v9: Set per-regime thresholds from config before checking
                if self.regime_aware and regime_tag and regime_name:
                    r_params = self.config.get('regime_params', {}).get(regime_name, {})
                    if 'gbm_conf_threshold' in r_params:
                        self.novel_filters._gbm_conf_threshold = r_params['gbm_conf_threshold']
                    if 'xgb_conf_threshold' in r_params:
                        self.novel_filters._smote_conf_threshold = r_params['xgb_conf_threshold']
                should_enter, reason = self.novel_filters.should_enter(df)
                if not should_enter:
                    print(f"   [NovelFilters] REJECTED: {reason}")
                    self.last_processed_bar = bar_timestamp
                    self.pm._db.set_last_processed_bar(bar_timestamp)
                    continue
                else:
                    print(f"   [NovelFilters] APPROVED: {reason}")

            # Execute entry directly (don't call super()._check_entry which
            # re-calls get_completed_bar and might get a different bar)
            entry_price = completed_bar['Close']
            entry_time = bar_time  # Use bar START time for intraday

            # Dedup check against last_signal_time
            signal_time_str = str(bar_time)[:19]
            if self.last_signal_time and signal_time_str <= self.last_signal_time:
                self.last_processed_bar = bar_timestamp
                self.pm._db.set_last_processed_bar(bar_timestamp)
                continue

            # Determine entry regime for v9
            entry_regime = regime_name if (self.regime_aware and regime_name) else None

            self._execute_entry(entry_price, entry_time, entry_regime=entry_regime)
            self.last_signal_time = signal_time_str

            # Mark bar as processed - PERSIST TO DATABASE
            self.last_processed_bar = bar_timestamp
            self.pm._db.set_last_processed_bar(bar_timestamp)

            # If we entered a position, stop scanning for more entries
            if self.pm.get_current_position():
                break

    def _execute_exit(self, exit_price: float, exit_reason: str, signal_bar_time=None, is_missed: bool = False):
        """Override to track last exit time for min_bars_between cooldown."""
        super()._execute_exit(exit_price, exit_reason, signal_bar_time, is_missed)
        # Record exit time for cooldown enforcement
        from ..data.market_hours import get_current_market_time
        self._last_exit_bar_time = signal_bar_time if signal_bar_time else get_current_market_time(self.ticker)

    def _calculate_sleep_time(self) -> int:
        """
        Calculate sleep time based on position status and market hours.

        When NOT in position: Sleep until bar closes (precision timing for signals)
        When IN position + market OPEN: Check every 3 seconds (for stop loss/take profit monitoring)
        When IN position + market CLOSED: Sleep until market opens (no point checking)
        """
        try:
            # RECHECK OVERRIDE: If base_trader scheduled a rapid recheck
            # (bar not yet in data near boundary), honor it instead of
            # sleeping the full bar interval.
            import time as _t
            if hasattr(self, '_bar_recheck_time') and self._bar_recheck_time > 0:
                now_ts = _t.time()
                if self._bar_recheck_time > now_ts:
                    wait = max(1, int(self._bar_recheck_time - now_ts))
                else:
                    wait = 1  # already overdue
                print(f"   📡 Recheck pending — sleeping {wait}s (not full bar)")
                return wait

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
                sleep_seconds = self.config.get('position_check_interval', 3)
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
    # For regime-aware configs, required params live inside regime_params, not at top level
    if config.get('regime_aware', False):
        regime_params = config.get('regime_params', {})
        if not regime_params:
            raise ValueError("regime_aware=True but no regime_params in config")
    else:
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
    # For regime-aware configs, required params live inside regime_params
    is_regime_aware = config.get('regime_aware', False)
    if is_regime_aware:
        if not config.get('regime_params'):
            print(f"   WARNING: regime_aware=True but no regime_params in bundle")
            return None
    else:
        required_params = ['signal_type', 'oversold_threshold', 'overbought_threshold',
                           'stop_loss_pct', 'take_profit_pct']
        missing = [p for p in required_params if p not in config]
        if missing:
            print(f"   WARNING: Bundle missing required params: {missing}")
            print(f"   Re-run optimization in Streamlit to fix the bundle.")
            return None

    # Map bundle config keys to our expected format - NO DEFAULTS for trading params
    result = {
        'ticker': config.get('ticker'),
        'interval': config.get('interval', '15m'),  # Non-critical default ok
        # Bundle identification - CRITICAL for importing trades from bundle
        'bundle_name': config.get('bundle_name'),
        'strategy_name': config.get('strategy_name'),
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
        # Volatility regime filter (ATR percentile)
        'use_vol_regime_filter': config.get('use_vol_regime_filter', False),
        'vol_regime_percentile_threshold': config.get('vol_regime_percentile_threshold', 0.25),
        # Acceleration exit parameters
        'use_accel_exit': config.get('use_accel_exit', False),
        'accel_exit_type': config.get('accel_exit_type', 'sign_reversal'),
        'accel_exit_threshold': config.get('accel_exit_threshold', 0.0),
        'accel_exit_min_pnl': config.get('accel_exit_min_pnl', 0.5),
        'accel_exit_lookback': config.get('accel_exit_lookback', 1),
        'use_jerk_confirm': config.get('use_jerk_confirm', False),
        'jerk_confirm_threshold': config.get('jerk_confirm_threshold', 0.0),
        # Trailing stop parameters (v7+)
        'use_trailing_stop': config.get('use_trailing_stop', False),
        'trailing_stop_pct': config.get('trailing_stop_pct', 1.0),
        'trailing_stop_activation_pct': config.get('trailing_stop_activation_pct', 0.3),
        # Break-even stop parameters (v7+)
        'use_breakeven_stop': config.get('use_breakeven_stop', False),
        'breakeven_trigger_pct': config.get('breakeven_trigger_pct', 0.3),
        'breakeven_offset_pct': config.get('breakeven_offset_pct', 0.05),
        # Signal parameters
        'min_hold_bars': config.get('min_hold_bars', 1),
        'min_bars_between': config.get('min_bars_between', 1),
        'vel_threshold': config.get('vel_threshold', 0.0),
        'accel_threshold': config.get('accel_threshold', 0.0),
        'velocity_std_window': config.get('velocity_std_window', 10),
        'momentum_multiplier': config.get('momentum_multiplier', 1.5),
        'double_bottom_lookback': config.get('double_bottom_lookback', 10),
        'divergence_lookback': config.get('divergence_lookback', 5),
        # Non-critical parameters can have safe defaults
        'exit_on_opposite_signal': config.get('exit_on_opposite_signal', True),
        'exit_on_midline_cross': config.get('exit_on_midline_cross', False),
        'webhook': config.get('discord_webhook'),
        # Novel strategy filters (v7+) — only present if bundle uses novel strategies
        'novel_strategies': config.get('novel_strategies'),
        # v10/v11/v12b daily retrainer params
        'daily_retrain': config.get('daily_retrain', False),
        'retrain_trials': config.get('retrain_trials', 5000),
        'retrain_train_days': config.get('retrain_train_days', 30),
        'retrain_workers': config.get('retrain_workers'),
        'retrain_pinned_params': config.get('retrain_pinned_params', {}),
        'retrain_optimize_exits': config.get('retrain_optimize_exits', False),
        'retrain_scoring': config.get('retrain_scoring', 'original'),
        'retrain_pinned_toggles': config.get('retrain_pinned_toggles', {}),
        'retrain_param_ranges': config.get('retrain_param_ranges', {}),
        'retrain_dynamic_window': config.get('retrain_dynamic_window', False),
    }

    # v9 regime-aware passthrough
    if is_regime_aware:
        result['regime_aware'] = True
        result['regime_detector'] = config.get('regime_detector', {})
        result['regime_params'] = config.get('regime_params', {})
    else:
        # CRITICAL trading parameters - pass through from bundle, NO DEFAULTS
        result['signal_type'] = config['signal_type']
        result['stop_loss_pct'] = config['stop_loss_pct']
        result['take_profit_pct'] = config['take_profit_pct']
        result['oversold_threshold'] = config['oversold_threshold']
        result['overbought_threshold'] = config['overbought_threshold']

    return result


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
TEST_WEBHOOK = os.environ.get("DISCORD_TEST_WEBHOOK_URL", "")


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
        # For regime-aware configs, signal_type lives inside regime_params
        if cfg.get('regime_aware') and cfg.get('regime_params'):
            active = [rc.get('signal_type', '?') for rn, rc in cfg['regime_params'].items()
                      if not rc.get('dont_trade', False)]
            sig_types = list(dict.fromkeys(active))  # dedupe preserving order
            sig_display = ', '.join(sig_types) if sig_types else '?'
            sig_display = f"v{cfg.get('version', '9')} regime-aware ({sig_display})"
        else:
            sig_display = cfg.get('signal_type', '?')
        print(f"  [{i}] {name}")
        print(f"      {cfg.get('ticker', '?')} | {cfg.get('interval', '?')} | {sig_display}")
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
    # Critical trading params come directly from bundle (already validated by load_bundle_config)
    config = dict(cfg)  # Start with all keys from load_bundle_config
    # Add runtime-only settings
    config['check_interval_seconds'] = 30
    config['bar_completion_buffer'] = 3
    # Wavelet denoising (v8+) — pass through from bundle
    config.setdefault('use_wavelet_denoise', False)
    config.setdefault('wavelet_family', 'db4')
    config.setdefault('wavelet_level', 2)
    config.setdefault('wavelet_threshold_mode', 'hard')

    # Determine webhook
    if args.test:
        webhook = TEST_WEBHOOK
        print("*** TEST MODE - posting to test channel ***")
    elif args.webhook:
        webhook = args.webhook
    else:
        webhook = cfg.get('discord_webhook') or cfg.get('webhook') or os.environ.get('DISCORD_WEBHOOK_URL')

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
