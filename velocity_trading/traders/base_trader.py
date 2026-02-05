"""
Base trader class for velocity trading.

Provides common functionality shared by DailyTrader and IntradayTrader:
- Position management via PositionManager
- Signal detection
- Discord notifications
- Configuration loading
- Error handling with retry
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from abc import ABC, abstractmethod
import pandas as pd

# Add parent to path for imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from ..core.position_manager import PositionManager, Position
from ..data.fetcher import fetch_price_data, fetch_realtime_price
from ..data.data_pipeline import DataPipeline
from ..data.market_hours import (
    is_market_open, is_bar_complete, seconds_until_bar_close,
    get_current_market_time, get_market_type
)
from ..indicators.oscillators import (
    calculate_composite_oscillator,
    calculate_composite_oscillator_legacy
)
from ..indicators.velocity import (
    calculate_velocity_signals,
    calculate_velocity_signals_legacy,
    find_most_recent_signal,
    check_exit_conditions
)
from ..notifications.discord import (
    send_entry_alert, send_exit_alert, send_status_update,
    send_error_alert, send_startup_notification, create_trade_log_csv_buffer
)
from ..notifications.charts import generate_chart

# Sierra Chart Bridge - native support
try:
    from sierra_chart_bridge import build_sierra_signal, publish_signal_to_sierra
    SIERRA_BRIDGE_AVAILABLE = True
except ImportError:
    SIERRA_BRIDGE_AVAILABLE = False


class BaseTrader(ABC):
    """
    Abstract base class for velocity traders.

    Provides common trading loop functionality while allowing
    subclasses to customize bar completion and signal detection.
    """

    def __init__(
        self,
        strategy_name: str,
        ticker: str,
        interval: str,
        webhook_url: str = None,
        config: Dict = None
    ):
        """
        Initialize trader.

        Args:
            strategy_name: Unique strategy identifier
            ticker: Trading symbol
            interval: Bar interval ('1d', '15m', etc.)
            webhook_url: Discord webhook URL
            config: Strategy configuration dict
        """
        self.strategy_name = strategy_name
        self.ticker = ticker
        self.interval = interval
        self.webhook_url = webhook_url
        self.config = config or {}

        # Auto-backup the strategy bundle config on startup (daily)
        self._backup_config_if_needed()

        # Initialize position manager (SQLite-backed, with daily auto-backup)
        self.pm = PositionManager(strategy_name)

        # CRITICAL: Validate required config parameters - NO DEFAULTS
        # All parameters must come from strategy bundle to ensure consistency
        required_params = ['signal_type', 'oversold_threshold', 'overbought_threshold',
                           'stop_loss_pct', 'take_profit_pct']
        missing = [p for p in required_params if p not in self.config]
        if missing:
            raise ValueError(
                f"Missing required config parameters: {missing}. "
                f"Config must come from strategy bundle (velocity_strategies/{strategy_name}/), "
                "not hardcoded defaults."
            )

        # Trading parameters from config - NO DEFAULTS
        self.stop_loss_pct = self.config['stop_loss_pct']
        self.take_profit_pct = self.config['take_profit_pct']
        self.signal_type = self.config['signal_type']
        self.oversold_threshold = self.config['oversold_threshold']
        self.overbought_threshold = self.config['overbought_threshold']

        # Oscillator type (for novel oscillators: arwo, prf, ics, etc.)
        self.oscillator_type = self.config.get('oscillator_type', 'composite')

        # Signal generation parameters
        self.vel_smoothing = self.config.get('vel_smoothing', 1)
        self.extreme_zone_mult = self.config.get('extreme_zone_mult', 1.5)
        self.require_accel = self.config.get('require_accel', False)

        # V2 Filters (Regime, Fragility, Entropy)
        self.use_regime_filter = self.config.get('use_regime_filter', False)
        self.regime_threshold = self.config.get('regime_threshold', -0.15)
        self.use_fragility_filter = self.config.get('use_fragility_filter', False)
        self.fragility_threshold = self.config.get('fragility_threshold', 0.5)
        self.use_entropy_filter = self.config.get('use_entropy_filter', False)
        self.entropy_threshold = self.config.get('entropy_threshold', 0.7)

        # Acceleration Reversal Exit parameters
        self.use_accel_exit = self.config.get('use_accel_exit', False)
        self.accel_exit_type = self.config.get('accel_exit_type', 'sign_reversal')
        self.accel_exit_threshold = self.config.get('accel_exit_threshold', 0.0)
        self.accel_exit_min_pnl = self.config.get('accel_exit_min_pnl', 0.5)
        self.accel_exit_lookback = self.config.get('accel_exit_lookback', 1)
        self.use_jerk_confirm = self.config.get('use_jerk_confirm', False)
        self.jerk_confirm_threshold = self.config.get('jerk_confirm_threshold', 0.0)

        # Non-legacy mode now produces identical results to old system
        # Legacy mode is kept as fallback but not required by default
        self.use_legacy = self.config.get('use_legacy', False)

        # State
        self.running = False
        self.last_signal_time = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5

        # Data pipeline for incremental updates (Historical + Live API)
        self.pipeline = DataPipeline(ticker, interval)
        self._pipeline_initialized = False

    # Class-level tracking of daily config backups
    _config_backups_done = {}

    def _backup_config_if_needed(self):
        """
        Backup strategy bundle config once per day on startup.

        This ensures we always have a backup of the config before any trading.
        """
        from datetime import date
        today = date.today().isoformat()
        backup_key = f"{self.strategy_name}_{today}"

        # Check if we've already backed up today
        if backup_key in BaseTrader._config_backups_done:
            return

        # Find and backup the strategy bundle config
        bundle_path = os.path.join(PARENT_DIR, 'velocity_strategies', self.strategy_name)
        config_path = os.path.join(bundle_path, 'velocity_config.json')

        if os.path.exists(config_path):
            try:
                from ..core.backup import backup_config
                backup_path = backup_config(config_path, reason=f"startup_{today}")
                if backup_path:
                    BaseTrader._config_backups_done[backup_key] = backup_path
            except Exception as e:
                print(f"   ⚠️  Config backup warning: {e}")

    @property
    @abstractmethod
    def check_interval_seconds(self) -> int:
        """Seconds between checks. Override in subclass."""
        pass

    @abstractmethod
    def is_signal_window(self) -> bool:
        """Check if we're in the signal detection window. Override in subclass."""
        pass

    @abstractmethod
    def get_completed_bar(self, df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[datetime]]:
        """Get the most recently completed bar. Override in subclass."""
        pass

    def _ensure_database(self):
        """
        Ensure database is populated with historical trades.

        Priority order:
        1. Import from bundle's backtest_results.json (ground truth - exact trades from optimization)
        2. Fall back to rebuild_from_backtest (recalculates from scratch - may differ!)

        Using bundle backtest_results ensures stats match exactly what was optimized.
        """
        trade_count = self.pm.get_trade_count()
        last_trade = self.pm.get_last_trade_date()

        print(f"📊 Database check: {trade_count} trades", end="")
        if last_trade:
            print(f", last: {last_trade[:10]}")
        else:
            print()

        if trade_count == 0:
            # Empty database - try importing from bundle first
            imported = self._try_import_from_bundle()

            if imported:
                # After bundle import, run incremental update to catch up on trades
                # since the bundle was created (bundle might be days/weeks old)
                print(f"   ℹ️  Running catch-up update for trades since bundle creation...")
                success, result = self.pm.process_new_bars(
                    ticker=self.ticker,
                    config=self.config,
                    interval=self.interval,
                    lookback_bars=500  # Enough to cover time since bundle creation
                )
                if success:
                    new_entries = result.get('new_entries', 0)
                    new_exits = result.get('new_exits', 0)
                    if new_entries > 0 or new_exits > 0:
                        print(f"   ✓ Catch-up complete: +{new_entries} entries, +{new_exits} exits")
                    else:
                        print(f"   ✓ No new trades since bundle creation")
                else:
                    print(f"   ⚠️  Catch-up failed: {result.get('error', 'Unknown error')}")
            else:
                # Fall back to rebuilding from scratch (may produce different results!)
                print(f"   ℹ️  No bundle found, rebuilding from backtest...")
                success, result = self.pm.rebuild_from_backtest(
                    ticker=self.ticker,
                    config=self.config,
                    days=self._get_rebuild_days(),
                    interval=self.interval  # CRITICAL: Pass interval for intraday strategies!
                )
                if success:
                    print(f"   ✓ Rebuilt: {result['entries']} entries, {result['exits']} exits")
                    print(f"   ✓ Win rate: {result['stats']['win_rate']:.1f}%")
                else:
                    print(f"   ⚠️  Rebuild failed: {result.get('error', 'Unknown error')}")
        else:
            # Database has data - check if we need incremental update
            # CRITICAL: Always run update if in position to catch missed exit signals
            current_position = self.pm.get_current_position()
            should_update = False
            update_reason = ""

            if last_trade:
                last_dt = pd.to_datetime(last_trade)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.tz_localize('UTC')
                else:
                    last_dt = last_dt.tz_convert('UTC')

                time_since = pd.Timestamp.now(tz='UTC') - last_dt
                days_old = time_since.days
                hours_old = time_since.total_seconds() / 3600

                if current_position:
                    # ALWAYS update if in position - we might have missed exit signals
                    should_update = True
                    update_reason = f"open position from {current_position.entry_date[:10]}"
                elif days_old > 1:
                    should_update = True
                    update_reason = f"data is {days_old} days old"
                elif self.interval in ['1m', '5m', '15m', '30m', '1h', '2h', '4h']:
                    # For intraday, ALWAYS backfill on startup to catch missed bars
                    should_update = True
                    update_reason = f"intraday startup sync ({hours_old:.1f}h since last trade)"

            if should_update:
                print(f"   ℹ️  Running incremental update ({update_reason})...")
                success, result = self.pm.update_from_latest(
                    ticker=self.ticker,
                    config=self.config,
                    interval=self.interval
                )
                if success:
                    new_entries = result.get('new_entries', 0)
                    new_exits = result.get('new_exits', 0)
                    if new_entries > 0 or new_exits > 0:
                        print(f"   ✓ Updated: +{new_entries} entries, +{new_exits} exits")
                    else:
                        print(f"   ✓ No missed trades found")
                else:
                    print(f"   ⚠️  Update failed: {result.get('error', 'Unknown error')}")
            else:
                print(f"   ✓ Database is up to date")

    def _try_import_from_bundle(self) -> bool:
        """
        Try to import trades from the bundle's backtest file.

        Checks for multiple file formats:
        1. backtest_results.json (entries/exits format)
        2. trade_log.json (trades format - used by newer bundles)

        This is the preferred method because it uses the exact trades from the
        optimization backtest, ensuring stats match exactly.

        Returns True if import succeeded, False otherwise.
        """
        import json

        # Get bundle name from config
        bundle_name = self.config.get('bundle_name')
        if not bundle_name:
            print(f"   ℹ️  No bundle_name in config, will rebuild from scratch")
            return False

        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        bundle_dir = os.path.join(parent_dir, "velocity_strategies", bundle_name)

        # Try multiple file formats
        backtest_results = None
        source_file = None

        # Format 1: backtest_results.json (entries/exits format)
        path1 = os.path.join(bundle_dir, "backtest_results.json")
        if os.path.exists(path1):
            source_file = path1
            with open(path1) as f:
                backtest_results = json.load(f)

        # Format 2: trade_log.json (trades format) - convert to entries/exits
        if backtest_results is None:
            path2 = os.path.join(bundle_dir, "trade_log.json")
            if os.path.exists(path2):
                source_file = path2
                with open(path2) as f:
                    trade_log = json.load(f)
                # Convert trade_log format to entries/exits format
                backtest_results = self._convert_trade_log_to_backtest(trade_log)

        if backtest_results is None:
            print(f"   ℹ️  No backtest file found in bundle: {bundle_name}")
            return False

        try:
            print(f"   📦 Importing from bundle: {os.path.basename(source_file)}")

            # Get the last bar date from bundle data for tracking
            data_path = os.path.join(bundle_dir, "data.parquet")
            last_bar_date = None
            if os.path.exists(data_path):
                try:
                    import pandas as pd
                    df = pd.read_parquet(data_path)
                    if len(df) > 0:
                        last_bar_date = str(df.index[-1])
                except Exception:
                    pass

            # Use import_from_locked_backtest for exact trade replication
            success, result = self.pm.import_from_locked_backtest(
                locked_backtest=backtest_results,
                ticker=self.ticker,
                clear_existing=True,
                last_bar_date=last_bar_date
            )

            if success:
                entries = result.get('entries_imported', 0)
                exits = result.get('exits_imported', 0)
                stats = result.get('stats', {})
                print(f"   ✓ Imported from bundle: {entries} entries, {exits} exits")
                print(f"   ✓ Win rate: {stats.get('win_rate', 0):.1f}% | Total return: {stats.get('total_return', 0):.1f}%")
                return True
            else:
                print(f"   ⚠️  Bundle import failed: {result.get('error', 'Unknown error')}")
                return False

        except Exception as e:
            print(f"   ⚠️  Error importing from bundle: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _convert_trade_log_to_backtest(self, trade_log: dict) -> dict:
        """
        Convert trade_log.json format to backtest_results.json format.

        trade_log format:
            {"trades": [{entry_date, exit_date, entry_price, exit_price, pnl_pct, exit_reason, ...}]}

        backtest_results format:
            {"entries": [{date, price, position}], "exits": [{date, price, pnl, reason, entry_date, entry_price}]}
        """
        trades = trade_log.get('trades', [])
        entries = []
        exits = []

        for trade in trades:
            # Create entry record
            entries.append({
                'date': trade['entry_date'],
                'price': trade['entry_price'],
                'position': 'long'  # Velocity strategies are long-only
            })

            # Create exit record
            exits.append({
                'date': trade['exit_date'],
                'price': trade['exit_price'],
                'pnl': trade.get('pnl_pct', 0),
                'reason': trade.get('exit_reason', 'signal'),
                'entry_date': trade['entry_date'],
                'entry_price': trade['entry_price']
            })

        return {'entries': entries, 'exits': exits}

    def _get_rebuild_days(self) -> int:
        """Get number of days for full rebuild. Override for longer histories."""
        # Default based on interval
        if self.interval == '1d':
            return 365 * 2  # 2 years for daily
        elif self.interval in ['15m', '30m']:
            return 60  # 60 days for intraday (API limits)
        else:
            return 365

    def run(self):
        """
        Main trading loop.

        Runs continuously, checking for signals and managing positions.
        """
        self.running = True
        print(f"\n{'='*60}")
        print(f"  VELOCITY TRADER - {self.strategy_name}")
        print(f"  {self.ticker} | {self.interval}")
        print(f"{'='*60}\n")

        # Ensure database is populated (auto-rebuild if empty, update if stale)
        self._ensure_database()

        # Send startup notification
        self._send_startup_notification()

        while self.running:
            try:
                self._trading_cycle()
                self.consecutive_errors = 0

            except KeyboardInterrupt:
                print("\n\nShutting down...")
                self.running = False
                break

            except Exception as e:
                self.consecutive_errors += 1
                print(f"\n[ERROR] Cycle failed: {e}")

                if self.consecutive_errors >= self.max_consecutive_errors:
                    self._send_error_alert(f"Too many consecutive errors: {e}")
                    print(f"Too many errors, pausing for 5 minutes...")
                    time.sleep(300)
                    self.consecutive_errors = 0
                else:
                    time.sleep(60)

    def _trading_cycle(self):
        """Single iteration of the trading loop."""
        import time as _time
        cycle_start = _time.time()
        now = datetime.now()
        print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] [{self.strategy_name}] Checking {self.ticker}...")

        # Get current position
        position = self.pm.get_current_position()

        # OPTIMIZATION: When in position, only do quick SL/TP check using realtime price
        # Full bar data fetch is only needed when:
        # 1. Not in position (looking for entry signals)
        # 2. A new bar might be complete (check for signal-based exits)
        if position and not self._should_fetch_full_data():
            self._quick_sltp_check(position)
            sleep_time = self._calculate_sleep_time()
            print(f"   ⏱️ Quick check completed, sleeping {sleep_time}s")
            time.sleep(sleep_time)
            return

        # Full data fetch and analysis path
        # Fetch price data with retry logic
        df = self._fetch_data()
        retry_count = 0
        max_retries = 30  # Max 30 seconds of retrying
        while (df is None or df.empty) and retry_count < max_retries:
            retry_count += 1
            print(f"   No data available, retrying in 1s ({retry_count}/{max_retries})...")
            time.sleep(1)
            df = self._fetch_data()

        if df is None or df.empty:
            print("   No data after retries, skipping cycle")
            time.sleep(self.check_interval_seconds)
            return

        # Record successful full fetch time (for optimization when in position)
        self._last_full_fetch_time = _time.time()

        fetch_time = _time.time() - cycle_start
        print(f"   ⏱️ Data fetch took {fetch_time:.1f}s")

        # Calculate indicators (use legacy mode for exact matching with old system)
        if self.use_legacy:
            df = calculate_composite_oscillator_legacy(df)
            df = calculate_velocity_signals_legacy(df, self.config)
        else:
            # Calculate oscillator (supports novel oscillators: arwo, prf, ics, etc.)
            df = calculate_composite_oscillator(
                df,
                oscillator_type=self.oscillator_type,
                config=self.config
            )
            # Generate signals with V2 filters (regime, fragility, entropy)
            df = calculate_velocity_signals(
                df,
                signal_type=self.signal_type,
                oversold_threshold=self.oversold_threshold,
                overbought_threshold=self.overbought_threshold,
                vel_smoothing=self.vel_smoothing,
                extreme_zone_mult=self.extreme_zone_mult,
                require_accel=self.require_accel,
                use_regime_filter=self.use_regime_filter,
                regime_threshold=self.regime_threshold,
                use_fragility_filter=self.use_fragility_filter,
                fragility_threshold=self.fragility_threshold,
                use_entropy_filter=self.use_entropy_filter,
                entropy_threshold=self.entropy_threshold
            )

        # PERFORMANCE: Cache the DataFrame with indicators for reuse in exit/entry processing
        # This avoids re-fetching and re-calculating indicators for chart generation
        self._cached_df = df.copy()

        indicators_time = _time.time() - cycle_start
        print(f"   ⏱️ Indicators calculated in {indicators_time - fetch_time:.1f}s (total: {indicators_time:.1f}s)")

        # Check for exit conditions FIRST (if in position)
        if position:
            self._check_exit(df, position)

        # Check for entry conditions (if not in position)
        if not self.pm.get_current_position():  # Re-check after potential exit
            self._check_entry(df)

        # Sleep until next check
        cycle_total = _time.time() - cycle_start
        sleep_time = self._calculate_sleep_time()
        print(f"   ⏱️ Cycle completed in {cycle_total:.1f}s, sleeping {sleep_time}s")
        time.sleep(sleep_time)

    def _should_fetch_full_data(self) -> bool:
        """
        Determine if full bar data fetch is needed.

        Returns True when:
        1. We haven't done a full fetch since the last bar closed
        2. More than interval_minutes have passed since last full fetch

        This optimizes API usage when in position - only fetch full data once per bar
        to check for signal-based exits, not every 5-second SL/TP check.
        """
        try:
            import time as _time
            now_ts = _time.time()
            interval_seconds = self._get_interval_minutes() * 60

            # Initialize last_full_fetch if not set
            if not hasattr(self, '_last_full_fetch_time'):
                self._last_full_fetch_time = 0

            # Do full fetch if it's been more than interval_minutes since last one
            time_since_last_fetch = now_ts - self._last_full_fetch_time
            if time_since_last_fetch >= interval_seconds:
                # Update timestamp (will be done when fetch actually happens)
                return True

            # Also check if we're in the first minute of a new bar and haven't fetched this bar yet
            now = get_current_market_time(self.ticker)
            interval_minutes = self._get_interval_minutes()
            minutes_into_bar = (now.minute % interval_minutes) + (now.second / 60)

            # If we're in first minute of bar AND we fetched before this bar started, fetch again
            if minutes_into_bar < 1.0:
                # Calculate when this bar started
                bar_start_minute = (now.minute // interval_minutes) * interval_minutes
                bar_start = now.replace(minute=bar_start_minute, second=0, microsecond=0)

                # If last fetch was before this bar started, do a full fetch
                last_fetch_dt = datetime.fromtimestamp(self._last_full_fetch_time)
                if last_fetch_dt < bar_start:
                    return True

            return False

        except Exception:
            # On any error, default to fetching full data
            return True

    def _quick_sltp_check(self, position):
        """
        Quick check for stop loss and take profit using only real-time price.

        This is the lightweight path when in position - no bar data fetch needed.
        """
        # Get real-time price
        current_price = fetch_realtime_price(self.ticker)
        if current_price is None:
            print("   ⚠️ Realtime price unavailable, skipping quick check")
            return

        entry_price = position.entry_price
        pnl_pct = ((current_price - entry_price) / entry_price) * 100

        # Calculate SL/TP price levels for display
        sl_price = entry_price * (1 - self.stop_loss_pct / 100)
        tp_price = entry_price * (1 + self.take_profit_pct / 100)

        print(f"   📊 Position: LONG @ ${entry_price:,.2f} | Current: ${current_price:,.2f}")
        print(f"   📊 P&L: {pnl_pct:+.2f}% | SL: ${sl_price:,.2f} | TP: ${tp_price:,.2f}")

        exit_reason = None
        exit_price = current_price

        # Stop Loss
        if pnl_pct <= -self.stop_loss_pct:
            exit_reason = f"Stop Loss ({pnl_pct:.2f}%)"
            print(f"   🛑 STOP LOSS TRIGGERED! P&L {pnl_pct:.2f}% <= -{self.stop_loss_pct}%")

        # Take Profit
        elif pnl_pct >= self.take_profit_pct:
            exit_reason = f"Take Profit ({pnl_pct:.2f}%)"
            print(f"   🎯 TAKE PROFIT TRIGGERED! P&L {pnl_pct:.2f}% >= +{self.take_profit_pct}%")

        # Execute exit if triggered
        if exit_reason:
            self._execute_exit(exit_price, exit_reason, None)

    def _fetch_data(self, for_chart: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch price data for signal generation and charts.

        CRITICAL FIX: Uses fetch_price_data (yfinance) directly to ensure
        consistency with incremental updates. The DataPipeline (Databento+yfinance)
        was producing different oscillator values causing signals to not fire.

        Args:
            for_chart: If True, include synthetic current bar for display.
                       If False (default), exclude synthetic for signal generation.
        """
        # Use fetch_price_data directly (same as incremental update) for consistency
        # The DataPipeline was causing oscillator discrepancies due to Databento data merge
        fetch_days = 60 if self.interval in ['1m', '5m', '15m', '30m', '1h', '2h', '4h'] else 365

        # Retry with exponential backoff
        for attempt in range(3):
            try:
                df = fetch_price_data(
                    self.ticker,
                    days=fetch_days,
                    interval=self.interval
                )
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                print(f"   Fetch attempt {attempt+1} failed: {e}")
                time.sleep(5 * (attempt + 1))

        return None

    def _check_exit(self, df: pd.DataFrame, position: Position):
        """Check if position should be exited."""
        # Get current price
        current_price = fetch_realtime_price(self.ticker)
        price_source = "realtime"
        if current_price is None:
            current_price = df['Close'].iloc[-1]
            price_source = "last_bar"
            print(f"   ⚠️ Realtime price unavailable, using last bar close")

        # Check stop loss and take profit
        entry_price = position.entry_price
        pnl_pct = ((current_price - entry_price) / entry_price) * 100

        # Calculate SL/TP price levels for display
        sl_price = entry_price * (1 - self.stop_loss_pct / 100)
        tp_price = entry_price * (1 + self.take_profit_pct / 100)

        # Debug logging for position monitoring
        print(f"   📊 Position: LONG @ ${entry_price:,.2f} | Current: ${current_price:,.2f} ({price_source})")
        print(f"   📊 P&L: {pnl_pct:+.2f}% | SL: -{self.stop_loss_pct}% (${sl_price:,.2f}) | TP: +{self.take_profit_pct}% (${tp_price:,.2f})")

        exit_reason = None
        exit_price = current_price
        signal_bar_time = None  # Only set for signal-based exits

        # Stop Loss
        if pnl_pct <= -self.stop_loss_pct:
            exit_reason = f"Stop Loss ({pnl_pct:.2f}%)"
            print(f"   🛑 STOP LOSS TRIGGERED! P&L {pnl_pct:.2f}% <= -{self.stop_loss_pct}%")

        # Take Profit
        elif pnl_pct >= self.take_profit_pct:
            exit_reason = f"Take Profit ({pnl_pct:.2f}%)"
            print(f"   🎯 TAKE PROFIT TRIGGERED! P&L {pnl_pct:.2f}% >= +{self.take_profit_pct}%")

        # Acceleration Reversal Exit (early warning before stop loss)
        elif self.use_accel_exit and 'acceleration' in df.columns:
            # Check if conditions are met
            pnl_ok = pnl_pct >= self.accel_exit_min_pnl or pnl_pct < 0

            if pnl_ok and len(df) >= self.accel_exit_lookback + 1:
                accel_values = df['acceleration'].iloc[-self.accel_exit_lookback:].values
                current_accel = df['acceleration'].iloc[-1]

                # For LONG positions: negative acceleration is bearish
                accel_cond = False
                if self.accel_exit_type == 'sign_reversal':
                    accel_cond = all(a < 0 for a in accel_values)
                elif self.accel_exit_type == 'magnitude':
                    accel_cond = current_accel < -self.accel_exit_threshold
                elif self.accel_exit_type == 'both':
                    accel_cond = all(a < 0 for a in accel_values) and abs(current_accel) > self.accel_exit_threshold

                # Jerk confirmation (optional)
                jerk_cond = True
                if self.use_jerk_confirm and 'jerk' in df.columns and self.jerk_confirm_threshold > 0:
                    current_jerk = df['jerk'].iloc[-1]
                    jerk_cond = current_jerk < -self.jerk_confirm_threshold

                if accel_cond and jerk_cond:
                    exit_reason = f"Accel Reversal ({pnl_pct:.2f}%)"
                    print(f"   ⚠️ ACCELERATION REVERSAL EXIT! Accel: {current_accel:.4f}, P&L: {pnl_pct:.2f}%")

        # Signal-based exits (only if no SL/TP/Accel and in signal window)
        elif self.is_signal_window():
            # CRITICAL FIX: Check ALL bars since position entry for missed exit signals
            # Previously only checked the most recent completed bar, missing signals that
            # fired during weekends, gaps, or when trader wasn't running.
            entry_date = pd.to_datetime(position.entry_date)
            # Normalize to naive UTC for comparison with df.index (which is also naive UTC)
            if entry_date.tzinfo is not None:
                entry_date = entry_date.tz_convert('UTC').tz_localize(None)

            # Find bars AFTER position entry (use > not >= to exclude entry bar)
            # Entry bar cannot have exit signal - that would be same-bar exit
            df_since_entry = df[df.index > entry_date]

            # Determine which signal to look for based on position type
            is_long = position.position_type.lower() == 'long'
            signal_col = 'sell_signal' if is_long else 'buy_signal'

            # Check for any opposite signal since entry
            if signal_col in df_since_entry.columns:
                opposite_signals = df_since_entry[df_since_entry[signal_col] == True]

                if not opposite_signals.empty:
                    # Use the FIRST opposite signal after entry (exit at earliest opportunity)
                    first_signal = opposite_signals.iloc[0]
                    first_signal_time = opposite_signals.index[0]

                    exit_price = first_signal['Close']

                    # Calculate P&L based on position type
                    if is_long:
                        bar_pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    else:
                        bar_pnl_pct = ((entry_price - exit_price) / entry_price) * 100

                    exit_reason = f"Opposite Signal ({bar_pnl_pct:.2f}%)"

                    # Use bar START time so chart plots marker on signal bar
                    # The exit_price is the signal bar's Close, so marker appears at bar's close level
                    signal_bar_time = first_signal_time

                    print(f"   📍 Found missed {signal_col} at {first_signal_time}")

        # Execute exit if triggered
        if exit_reason:
            # CRITICAL: Re-verify position still exists before exiting
            # Another process might have exited/cleared the position while we were processing
            current_pos = self.pm.get_current_position()
            if current_pos is None:
                print(f"   ⚠️ Position no longer exists in database - skipping exit")
                print(f"   ℹ️  This can happen if another process exited the position or ran a rebuild")
                return
            self._execute_exit(exit_price, exit_reason, signal_bar_time)

    def _check_entry(self, df: pd.DataFrame):
        """Check for entry signals."""
        if not self.is_signal_window():
            return

        completed_bar, bar_time = self.get_completed_bar(df)
        if completed_bar is None or bar_time is None:
            return

        # Check for buy signal
        if not completed_bar.get('buy_signal', False):
            return

        # Check if we already processed this signal
        signal_time_str = str(bar_time)[:19]
        if self.last_signal_time and signal_time_str <= self.last_signal_time:
            return

        # CRITICAL FIX: Entry happens on the NEXT bar after signal bar
        #
        # For DAILY bars:
        #   - Signal bar (Jan 29) closes at midnight UTC
        #   - Signal detected 1 second after midnight = we're now on Jan 30
        #   - Entry happens on Jan 30 at realtime price
        #   - entry_date = Jan 30 (execution day), NOT Jan 29 (signal day)
        #   - entry_price = realtime price (actual execution)
        #
        # For INTRADAY bars:
        #   - Signal fires at bar close
        #   - entry_price = signal bar's Close (for backtest consistency)
        #   - entry_date = bar close time
        #
        # NOTE: We use bar Close, not realtime price, because:
        # 1. Backtest uses Close prices for entry/exit
        # 2. Realtime price at detection is the NEXT bar's Open (not signal bar's Close)
        # 3. For stats consistency, live trading should match backtest methodology

        entry_price = completed_bar['Close']

        if self.interval in ['1d', '1wk', '1mo']:
            # Daily bars: Entry is on the NEXT day (today), not the signal bar
            # Signal bar closed at midnight, we're now on the next day
            from ..data.market_hours import get_current_market_time
            now = get_current_market_time(self.ticker)
            # Use today's date at 00:00:00 as entry_date (the execution bar)
            entry_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # Intraday bars: Use bar START time so chart plots marker on signal bar
            # The entry_price is the signal bar's Close, so marker appears at bar's top
            # Previously used bar_time + interval (close time), but chart interprets
            # that as the NEXT bar's start, causing marker to appear in wrong location
            entry_time = bar_time

        self._execute_entry(entry_price, entry_time)
        self.last_signal_time = signal_time_str

    def _execute_entry(self, entry_price: float, signal_time: datetime):
        """Execute position entry."""
        # Use the signal bar's timestamp as entry date (not current wall-clock time)
        # This ensures entries appear on the correct bar in charts
        from ..core.position_manager import normalize_timestamp
        entry_date = normalize_timestamp(signal_time)

        # CRITICAL: Use PositionManager for atomic entry
        success, result = self.pm.enter_position(
            ticker=self.ticker,
            position_type='long',
            entry_price=entry_price,
            entry_date=entry_date,
            entry_signal_bar=str(signal_time)
        )

        if success:
            print(f"   ENTRY: LONG @ ${entry_price:,.2f}")

            # Only send Discord if DB was updated (prevents desync)
            if self.webhook_url:
                import time as _time
                alert_start = _time.time()

                stats = self.pm.get_stats()

                # Get enhanced stats for Discord notification
                enhanced_stats = None
                try:
                    enhanced_stats = self.pm.get_enhanced_stats(
                        ticker=self.ticker,
                        config=self.config,
                        current_price=entry_price
                    )
                except Exception as e:
                    print(f"   Warning: Enhanced stats failed: {e}")

                # Format signal time with timezone indicator
                # Convert to UTC for display to avoid confusion
                import pytz
                if hasattr(signal_time, 'tzinfo') and signal_time.tzinfo is not None:
                    # Convert to UTC for consistent display
                    utc_time = signal_time.astimezone(pytz.UTC)
                    signal_time_str = utc_time.strftime('%Y-%m-%d %H:%M') + " UTC"
                else:
                    # Naive timestamp - just format as-is
                    signal_time_str = str(signal_time)[:16]

                # SIERRA CHART: Publish signal FIRST for fastest execution
                if SIERRA_BRIDGE_AVAILABLE:
                    try:
                        sierra_signal = build_sierra_signal(
                            signal_type="ENTRY",
                            direction="LONG",
                            ticker=self.ticker,
                            price=entry_price,
                            signal_time=signal_time,
                            strategy_name=self.strategy_name,
                            config=self.config
                        )
                        publish_signal_to_sierra(sierra_signal)
                        print(f"   📡 Sierra Chart: ENTRY signal published")
                    except Exception as e:
                        print(f"   ⚠️ Sierra Chart publish failed: {e}")

                # DISCORD ALERT: Send text alert (no chart) for notification
                stats_time = _time.time() - alert_start
                print(f"   ⏱️ Stats prepared in {stats_time:.1f}s, sending Discord alert...")

                send_entry_alert(
                    self.webhook_url,
                    self.strategy_name,
                    self.ticker,
                    entry_price,
                    signal_time_str,
                    self.stop_loss_pct,
                    self.take_profit_pct,
                    stats,
                    chart=None,  # No chart - send immediately
                    enhanced_stats=enhanced_stats
                )

                immediate_time = _time.time() - alert_start
                print(f"   ⏱️ Immediate alert sent in {immediate_time:.1f}s")

                # FOLLOW-UP: Generate and send chart separately
                chart_start = _time.time()
                chart = None
                try:
                    # ALWAYS fetch fresh data for chart to include the signal bar
                    # The cached_df was fetched BEFORE the signal bar existed
                    df = self._fetch_data(for_chart=True)
                    if df is None or df.empty:
                        # Fallback to cached data if fresh fetch fails
                        print(f"   ⚠️ Fresh fetch failed, using cached data...")
                        df = getattr(self, '_cached_df', None)
                        if df is not None and not df.empty:
                            if self.use_legacy:
                                df = calculate_composite_oscillator_legacy(df)
                                df = calculate_velocity_signals_legacy(df, self.config)
                            else:
                                df = calculate_composite_oscillator(df, oscillator_type=self.oscillator_type, config=self.config)
                                df = calculate_velocity_signals(
                                    df,
                                    signal_type=self.signal_type,
                                    oversold_threshold=self.oversold_threshold,
                                    overbought_threshold=self.overbought_threshold,
                                    vel_smoothing=self.vel_smoothing,
                                    extreme_zone_mult=self.extreme_zone_mult,
                                    require_accel=self.require_accel,
                                    use_regime_filter=self.use_regime_filter,
                                    regime_threshold=self.regime_threshold
                                )

                    if df is not None and not df.empty:
                        trade_data = self.pm.get_entries_and_exits()
                        chart = generate_chart(
                            df=df,
                            entries=trade_data.get('entries', []),
                            exits=trade_data.get('exits', []),
                            ticker=self.ticker,
                            stats=stats,
                            current_position={'position': 'long', 'entry_price': entry_price},
                            title_suffix=" - Entry",
                            chart_type="signal",
                            interval=self.interval,
                            oversold_threshold=self.oversold_threshold,
                            overbought_threshold=self.overbought_threshold,
                            signal_time=signal_time  # Show UTC + Chicago time on chart
                        )
                except Exception as e:
                    print(f"   Warning: Entry chart generation failed: {e}")

                chart_time = _time.time() - chart_start
                print(f"   ⏱️ Chart generated in {chart_time:.1f}s")

                # Send chart as follow-up message
                if chart:
                    from ..notifications.discord import send_discord_alert
                    send_discord_alert(self.webhook_url, f"📊 Chart for {self.strategy_name}", chart=chart)
                    total_time = _time.time() - alert_start
                    print(f"   ⏱️ Chart follow-up sent (total: {total_time:.1f}s)")
        else:
            print(f"   Entry rejected: {result.get('error')} - {result.get('message')}")

    def _execute_exit(self, exit_price: float, exit_reason: str, signal_bar_time=None):
        """Execute position exit."""
        from ..core.position_manager import normalize_timestamp
        from ..data.market_hours import get_current_market_time

        # Use signal bar timestamp for signal-based exits, current time for SL/TP
        if signal_bar_time is not None:
            exit_date = normalize_timestamp(signal_bar_time)
        else:
            # SL/TP exits use current time (they're triggered by price, not bar completion)
            exit_date = get_current_market_time(self.ticker).isoformat()

        # CRITICAL: Use PositionManager for atomic exit
        success, result = self.pm.exit_position(
            exit_price=exit_price,
            exit_date=exit_date,
            exit_reason=exit_reason
        )

        if success:
            pnl = result.get('pnl_pct', 0)
            print(f"   EXIT: {exit_reason} | P&L: {pnl:+.2f}%")

            # Only send Discord if DB was updated (prevents desync)
            if self.webhook_url:
                import time as _time
                alert_start = _time.time()

                stats = self.pm.get_stats()
                cumulative = {
                    'total_trades': stats.get('num_trades', 0),
                    'winners': stats.get('num_wins', 0),
                    'losers': stats.get('num_trades', 0) - stats.get('num_wins', 0),
                    'win_rate': stats.get('win_rate', 0),
                    'total_pnl_pct': stats.get('total_return', 0),
                    'total_pnl_dollars': stats.get('total_return', 0) * 100  # Approximate
                }

                # Get enhanced stats for Discord notification
                enhanced_stats = None
                csv_buf = None
                try:
                    enhanced_stats = self.pm.get_enhanced_stats(
                        ticker=self.ticker,
                        config=self.config,
                        current_price=exit_price
                    )
                    # Create CSV buffer for trade log attachment
                    trade_log_csv = enhanced_stats.get('trade_log_csv')
                    if trade_log_csv:
                        csv_buf = create_trade_log_csv_buffer(trade_log_csv)
                except Exception as e:
                    print(f"   Warning: Enhanced stats failed: {e}")

                # Format times - just use local time without confusing UTC label
                # The times shown are when the signal was actionable (bar close time)
                entry_time_str = str(result.get('entry_date', ''))[:16]
                exit_time_str = str(exit_date)[:16]

                # SIERRA CHART: Publish signal FIRST for fastest execution
                if SIERRA_BRIDGE_AVAILABLE:
                    try:
                        exit_data = {
                            "entry_price": result.get('entry_price', 0),
                            "pnl_pts": round(exit_price - result.get('entry_price', exit_price), 2),
                            "pnl_pct": pnl,
                            "exit_reason": exit_reason
                        }
                        sierra_signal = build_sierra_signal(
                            signal_type="EXIT",
                            direction="LONG",  # Current system is long-only
                            ticker=self.ticker,
                            price=exit_price,
                            signal_time=exit_date,
                            strategy_name=self.strategy_name,
                            config=self.config,
                            exit_data=exit_data
                        )
                        publish_signal_to_sierra(sierra_signal)
                        print(f"   📡 Sierra Chart: EXIT signal published")
                    except Exception as e:
                        print(f"   ⚠️ Sierra Chart publish failed: {e}")

                # DISCORD ALERT: Send text alert (no chart) for notification
                stats_time = _time.time() - alert_start
                print(f"   ⏱️ Stats prepared in {stats_time:.1f}s, sending Discord alert...")

                send_exit_alert(
                    self.webhook_url,
                    self.strategy_name,
                    self.ticker,
                    result.get('entry_price', 0),
                    exit_price,
                    pnl,
                    result.get('pnl_dollars', 0),
                    exit_reason,
                    entry_time_str,
                    exit_time_str,
                    cumulative,
                    chart=None,  # No chart - send immediately
                    csv_buf=csv_buf,
                    enhanced_stats=enhanced_stats
                )

                immediate_time = _time.time() - alert_start
                print(f"   ⏱️ Immediate alert sent in {immediate_time:.1f}s")

                # FOLLOW-UP: Generate and send chart separately
                chart_start = _time.time()
                chart = None
                try:
                    # ALWAYS fetch fresh data for chart to include the signal bar
                    # The cached_df was fetched BEFORE the signal bar existed
                    df = self._fetch_data(for_chart=True)
                    if df is None or df.empty:
                        # Fallback to cached data if fresh fetch fails
                        print(f"   ⚠️ Fresh fetch failed, using cached data...")
                        df = getattr(self, '_cached_df', None)
                        if df is not None and not df.empty:
                            if self.use_legacy:
                                df = calculate_composite_oscillator_legacy(df)
                                df = calculate_velocity_signals_legacy(df, self.config)
                            else:
                                df = calculate_composite_oscillator(df, oscillator_type=self.oscillator_type, config=self.config)
                                df = calculate_velocity_signals(
                                    df,
                                    signal_type=self.signal_type,
                                    oversold_threshold=self.oversold_threshold,
                                    overbought_threshold=self.overbought_threshold,
                                    vel_smoothing=self.vel_smoothing,
                                    extreme_zone_mult=self.extreme_zone_mult,
                                    require_accel=self.require_accel,
                                    use_regime_filter=self.use_regime_filter,
                                    regime_threshold=self.regime_threshold
                                )

                    if df is not None and not df.empty:
                        trade_data = self.pm.get_entries_and_exits()
                        chart = generate_chart(
                            df=df,
                            entries=trade_data.get('entries', []),
                            exits=trade_data.get('exits', []),
                            ticker=self.ticker,
                            stats=stats,
                            current_position=None,
                            title_suffix=" - Exit",
                            chart_type="signal",
                            interval=self.interval,
                            oversold_threshold=self.oversold_threshold,
                            overbought_threshold=self.overbought_threshold,
                            signal_time=exit_date  # Show UTC + Chicago time on chart
                        )
                except Exception as e:
                    print(f"   Warning: Exit chart generation failed: {e}")

                chart_time = _time.time() - chart_start
                print(f"   ⏱️ Chart generated in {chart_time:.1f}s")

                # Send chart as follow-up message
                if chart:
                    from ..notifications.discord import send_discord_alert
                    send_discord_alert(self.webhook_url, f"📊 Chart for {self.strategy_name}", chart=chart)
                    total_time = _time.time() - alert_start
                    print(f"   ⏱️ Chart follow-up sent (total: {total_time:.1f}s)")
        else:
            print(f"   Exit rejected: {result.get('error')} - {result.get('message')}")

    def _get_interval_minutes(self) -> int:
        """Get the bar interval in minutes."""
        interval_map = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '1d': 1440
        }
        return interval_map.get(self.interval, 15)

    def _calculate_sleep_time(self) -> int:
        """Calculate seconds until next check."""
        return self.check_interval_seconds

    def _send_startup_notification(self):
        """Send startup notification to Discord with chart and enhanced stats."""
        if not self.webhook_url:
            return

        position = self.pm.get_current_position()

        # Ensure stats are calculated (might be empty on fresh database)
        stats = self.pm.get_stats()
        if stats.get('num_trades', 0) == 0:
            # Try recalculating - the cached stats might just be stale
            stats = self.pm.recalculate_stats()
        pos_dict = position.to_dict() if position else None

        # Get current price for enhanced stats
        current_price = fetch_realtime_price(self.ticker)

        # Get enhanced stats (includes all-time, recent 2-day, trade log CSV)
        enhanced_stats = None
        csv_buf = None
        try:
            print("   📊 Preparing enhanced stats...")
            enhanced_stats = self.pm.get_enhanced_stats(
                ticker=self.ticker,
                config=self.config,
                current_price=current_price
            )

            # Create CSV buffer if trade log is available
            trade_log_csv = enhanced_stats.get('trade_log_csv')
            if trade_log_csv:
                csv_buf = create_trade_log_csv_buffer(trade_log_csv)
                print(f"   ✓ Trade log CSV prepared (last 10 trades)")

        except Exception as e:
            print(f"   Warning: Enhanced stats failed: {e}")
            import traceback
            traceback.print_exc()

        # Generate chart for startup notification
        chart = None
        try:
            # Fetch data and calculate indicators for chart
            print("   📊 Preparing startup chart...")
            df = self._fetch_data(for_chart=True)  # Include synthetic bar for display
            if df is not None and not df.empty:
                print(f"   📊 Got {len(df)} bars, calculating indicators...")
                # Calculate indicators
                if self.use_legacy:
                    df = calculate_composite_oscillator_legacy(df)
                    df = calculate_velocity_signals_legacy(df, self.config)
                else:
                    df = calculate_composite_oscillator(df, oscillator_type=self.oscillator_type, config=self.config)
                    df = calculate_velocity_signals(
                        df,
                        signal_type=self.signal_type,
                        oversold_threshold=self.oversold_threshold,
                        overbought_threshold=self.overbought_threshold,
                        vel_smoothing=self.vel_smoothing,
                        extreme_zone_mult=self.extreme_zone_mult,
                        require_accel=self.require_accel,
                        use_regime_filter=self.use_regime_filter,
                        regime_threshold=self.regime_threshold
                    )

                # Get trade history for chart markers
                print("   📊 Loading trade history...")
                trade_data = self.pm.get_entries_and_exits()
                entries = trade_data.get('entries', [])
                exits = trade_data.get('exits', [])
                print(f"   📊 Generating chart ({len(entries)} entries, {len(exits)} exits)...")

                # Determine signal time for chart display
                # If in position: show entry time
                # If flat: show last exit time
                chart_signal_time = None
                if position:
                    chart_signal_time = position.entry_signal_bar or position.entry_date
                elif exits:
                    # Get most recent exit time
                    chart_signal_time = exits[-1].get('date')

                # Generate chart
                chart = generate_chart(
                    df=df,
                    entries=entries,
                    exits=exits,
                    ticker=self.ticker,
                    stats=stats,
                    current_position=pos_dict,
                    title_suffix=" - Startup",
                    chart_type="status",
                    interval=self.interval,
                    oversold_threshold=self.oversold_threshold,
                    overbought_threshold=self.overbought_threshold,
                    signal_time=chart_signal_time  # Show UTC + Chicago time
                )
                print("   ✓ Chart generated")
        except Exception as e:
            print(f"   Warning: Chart generation failed: {e}")
            import traceback
            traceback.print_exc()

        send_startup_notification(
            self.webhook_url,
            self.strategy_name,
            self.ticker,
            self.interval,
            pos_dict,
            stats,
            chart=chart,
            enhanced_stats=enhanced_stats,
            csv_buf=csv_buf
        )

    def _send_error_alert(self, message: str):
        """Send error notification to Discord."""
        if self.webhook_url:
            send_error_alert(self.webhook_url, self.strategy_name, message)

    def stop(self):
        """Stop the trading loop."""
        self.running = False
