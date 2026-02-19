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
import numpy as np
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
    send_error_alert, send_startup_notification, create_trade_log_csv_buffer,
    send_regime_change_alert, send_regime_status_update,
)
from ..notifications.charts import generate_chart

# Sierra Chart Bridge - native support
try:
    from sierra_chart_bridge import build_sierra_signal, publish_signal_to_sierra
    SIERRA_BRIDGE_AVAILABLE = True
except ImportError:
    SIERRA_BRIDGE_AVAILABLE = False

# Novel Strategy Filters (v7+)
try:
    from ..core.novel_filters import NovelStrategyFilters
    NOVEL_FILTERS_AVAILABLE = True
except ImportError:
    NOVEL_FILTERS_AVAILABLE = False

# v9 Regime Detection
try:
    from ..core.regime_detector_v9 import (
        classify_regimes, classify_regimes_extended,
        get_regime_distribution,
        REGIME_NAMES, REGIME_UPTREND, REGIME_DOWNTREND, REGIME_CHOP,
    )
    V9_REGIME_AVAILABLE = True
except ImportError:
    V9_REGIME_AVAILABLE = False

# v10 Daily Retrainer
try:
    from ..core.daily_retrainer import DailyRetrainer
    DAILY_RETRAINER_AVAILABLE = True
except ImportError:
    DAILY_RETRAINER_AVAILABLE = False


def _bar_close_time(bar_start, interval_minutes: int):
    """Convert a bar start timestamp to its close timestamp for display.

    Financial data labels bars by start time, but the signal isn't actionable
    until bar close (start + interval). This helper adds the interval so that
    displayed timestamps match when the signal was actually available.
    """
    from datetime import timedelta
    if bar_start is None:
        return bar_start
    try:
        dt = pd.to_datetime(bar_start)
        return dt + timedelta(minutes=interval_minutes)
    except Exception:
        return bar_start


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
        self._exit_is_bar_time = False  # Tracks if last exit_date is bar-start (needs +interval for display)

        # Auto-backup the strategy bundle config on startup (daily)
        self._backup_config_if_needed()

        # Initialize position manager (SQLite-backed, with daily auto-backup)
        self.pm = PositionManager(strategy_name)

        # v9 regime-aware mode
        self.regime_aware = self.config.get('regime_aware', False) and V9_REGIME_AVAILABLE

        # CRITICAL: Validate required config parameters - NO DEFAULTS
        # For regime-aware configs, required params live inside regime_params, not at top level
        if self.regime_aware:
            regime_params = self.config.get('regime_params', {})
            if not regime_params:
                raise ValueError("regime_aware=True but no regime_params in config")
            required_regime_keys = ['signal_type', 'oversold_threshold', 'overbought_threshold',
                                    'stop_loss_pct', 'take_profit_pct']
            for rname, rcfg in regime_params.items():
                if rcfg.get('dont_trade', False):
                    continue
                missing = [p for p in required_regime_keys if p not in rcfg]
                if missing:
                    raise ValueError(f"Regime '{rname}' missing required params: {missing}")
        else:
            required_params = ['signal_type', 'oversold_threshold', 'overbought_threshold',
                               'stop_loss_pct', 'take_profit_pct']
            missing = [p for p in required_params if p not in self.config]
            if missing:
                raise ValueError(
                    f"Missing required config parameters: {missing}. "
                    f"Config must come from strategy bundle (velocity_strategies/{strategy_name}/), "
                    "not hardcoded defaults."
                )

        # Validate interval matches config (prevents running wrong timeframe)
        config_interval = self.config.get('interval')
        if config_interval and config_interval != interval:
            raise ValueError(
                f"Interval mismatch: runtime '{interval}' vs config '{config_interval}'. "
                f"Strategy {strategy_name} is configured for '{config_interval}' intervals."
            )

        # Trading parameters from config - NO DEFAULTS
        # For regime-aware configs, use first active regime as defaults for display/fallback
        if self.regime_aware:
            _first_regime = next(
                (rcfg for rcfg in self.config.get('regime_params', {}).values()
                 if not rcfg.get('dont_trade', False)),
                {}
            )
            self.stop_loss_pct = self.config.get('stop_loss_pct', _first_regime.get('stop_loss_pct', 5.0))
            self.take_profit_pct = self.config.get('take_profit_pct', _first_regime.get('take_profit_pct', 10.0))
            self.signal_type = self.config.get('signal_type', _first_regime.get('signal_type', 'velocity_crossover_or_zone'))
            self.oversold_threshold = self.config.get('oversold_threshold', _first_regime.get('oversold_threshold', -0.1))
            self.overbought_threshold = self.config.get('overbought_threshold', _first_regime.get('overbought_threshold', 0.1))
        else:
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
        self.use_vol_regime_filter = self.config.get('use_vol_regime_filter', False)
        self.vol_regime_percentile_threshold = self.config.get('vol_regime_percentile_threshold', 0.25)
        self.rsi_filter = self.config.get('rsi_filter', 'none')
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.use_macd_confirm = self.config.get('use_macd_confirm', False)
        self.use_bb_filter = self.config.get('use_bb_filter', False)

        # Acceleration Reversal Exit parameters
        self.use_accel_exit = self.config.get('use_accel_exit', False)
        self.accel_exit_type = self.config.get('accel_exit_type', 'sign_reversal')
        self.accel_exit_threshold = self.config.get('accel_exit_threshold', 0.0)
        self.accel_exit_min_pnl = self.config.get('accel_exit_min_pnl', 0.5)
        self.accel_exit_lookback = self.config.get('accel_exit_lookback', 1)
        self.use_jerk_confirm = self.config.get('use_jerk_confirm', False)
        self.jerk_confirm_threshold = self.config.get('jerk_confirm_threshold', 0.0)

        # Trailing Stop parameters (v7+)
        self.use_trailing_stop = self.config.get('use_trailing_stop', False)
        self.trailing_stop_pct = self.config.get('trailing_stop_pct', 1.0)
        self.trailing_stop_activation_pct = self.config.get('trailing_stop_activation_pct', 0.3)
        # Break-Even Stop parameters (v7+)
        self.use_breakeven_stop = self.config.get('use_breakeven_stop', False)
        self.breakeven_trigger_pct = self.config.get('breakeven_trigger_pct', 0.3)
        self.breakeven_offset_pct = self.config.get('breakeven_offset_pct', 0.05)

        # Non-legacy mode now produces identical results to old system
        # Legacy mode is kept as fallback but not required by default
        self.use_legacy = self.config.get('use_legacy', False)

        # Novel Strategy Filters (v7+) — only initialized if config has novel_strategies
        self.novel_filters = None
        novel_config = self.config.get('novel_strategies')
        if novel_config and NOVEL_FILTERS_AVAILABLE:
            self.novel_filters = NovelStrategyFilters(novel_config)
            print(f"   [NovelFilters] Initialized for strategies: {novel_config.get('strategy_ids', [])}")
        elif novel_config and not NOVEL_FILTERS_AVAILABLE:
            print(f"   [NovelFilters] WARNING: novel_strategies in config but module not available")

        # State
        self.running = False
        self.last_signal_time = None
        self._high_watermark = 0.0  # Tracks highest price since entry (for trailing/BE stops)
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5

        # Data staleness tracking for graceful degradation
        self._last_successful_fetch_time = 0
        self._stale_data_alert_sent = False
        self._last_exit_check_time = 0  # Safety net: track when _check_exit last ran

        # Data pipeline for incremental updates (Historical + Live API)
        self.pipeline = DataPipeline(ticker, interval)
        self._pipeline_initialized = False

        # v9 regime state (populated each cycle)
        self._regimes = None
        self._is_high_vol = None
        self._regime_signal_dfs = {}  # regime_name -> DataFrame with signals

        # v9 regime notification state
        self._last_regime = None
        self._last_regime_status_time = 0
        self._adx_values = None
        self._plus_di_values = None
        self._minus_di_values = None

        # v10 daily retrainer — auto-retrain numerical params after market close
        self._daily_retrainer = None
        if self.config.get('daily_retrain', False) and DAILY_RETRAINER_AVAILABLE:
            self._daily_retrainer = DailyRetrainer(
                config=self.config,
                ticker=ticker,
                interval=interval,
                n_trials=self.config.get('retrain_trials', 5000),
                n_workers=self.config.get('retrain_workers', None),
                train_days=self.config.get('retrain_train_days', 30),
                pinned_params=self.config.get('retrain_pinned_params', {}),
            )
            print(f"   [v10] Daily retrainer enabled: {self.config.get('retrain_trials', 5000)} trials, "
                  f"{self.config.get('retrain_train_days', 30)}d window")

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

    # =========================================================================
    # v9 Regime Detection Helpers
    # =========================================================================

    def _compute_regimes(self, df: pd.DataFrame):
        """Compute regime classification for each bar. Returns (regimes, is_high_vol).

        Also stores ADX/DI arrays for regime notifications.
        """
        if not self.regime_aware:
            return None, None
        rd = self.config.get('regime_detector', {})
        ext = classify_regimes_extended(
            df['High'].values.astype(float),
            df['Low'].values.astype(float),
            df['Close'].values.astype(float),
            adx_period=rd.get('adx_period', 14),
            adx_threshold=rd.get('adx_threshold', 25.0),
            atr_period=rd.get('atr_period', 14),
            atr_high_vol_percentile=rd.get('atr_high_vol_percentile', 90.0),
            atr_high_vol_lookback=rd.get('atr_high_vol_lookback', 100),
            regime_min_bars=rd.get('regime_min_bars', 4),
        )
        self._adx_values = ext['adx']
        self._plus_di_values = ext['plus_di']
        self._minus_di_values = ext['minus_di']
        return ext['regimes'], ext['is_high_vol']

    def _generate_regime_signals(self, df: pd.DataFrame) -> dict:
        """Generate per-regime signal DataFrames. Returns {regime_name: df_with_signals}."""
        regime_signal_dfs = {}
        if not self.regime_aware:
            return regime_signal_dfs
        for rname, rcfg in self.config.get('regime_params', {}).items():
            if rcfg.get('dont_trade', False):
                continue
            merged = {**self.config, **rcfg}
            regime_signal_dfs[rname] = calculate_velocity_signals(
                df.copy(),
                signal_type=merged['signal_type'],
                oversold_threshold=merged['oversold_threshold'],
                overbought_threshold=merged['overbought_threshold'],
                vel_smoothing=merged.get('vel_smoothing', 1),
                extreme_zone_mult=merged.get('extreme_zone_mult', 1.5),
                require_accel=merged.get('require_accel', False),
                use_regime_filter=merged.get('use_regime_filter', False),
                regime_threshold=merged.get('regime_threshold', -0.15),
                use_fragility_filter=merged.get('use_fragility_filter', False),
                fragility_threshold=merged.get('fragility_threshold', 0.5),
                use_entropy_filter=merged.get('use_entropy_filter', False),
                entropy_threshold=merged.get('entropy_threshold', 0.7),
                use_vol_regime_filter=merged.get('use_vol_regime_filter', False),
                vol_regime_percentile_threshold=merged.get('vol_regime_percentile_threshold', 0.25),
                rsi_filter=merged.get('rsi_filter', 'none'),
                rsi_period=merged.get('rsi_period', 14),
                rsi_oversold=merged.get('rsi_oversold', 30),
                rsi_overbought=merged.get('rsi_overbought', 70),
                use_macd_confirm=merged.get('use_macd_confirm', False),
                use_bb_filter=merged.get('use_bb_filter', False),
            )
        return regime_signal_dfs

    def _get_exit_param(self, key: str, default, position: 'Position' = None):
        """Get exit parameter from entry regime config (v9) or top-level config (v8)."""
        regime_name = position.entry_regime if position else None
        if self.regime_aware and regime_name:
            rcfg = self.config.get('regime_params', {}).get(regime_name, {})
            return rcfg.get(key, self.config.get(key, default))
        return self.config.get(key, default)

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

                # For intraday strategies: re-evaluate from last trade date on startup
                # This catches missed entries when bars were "processed" but the trader
                # wasn't entering (e.g., process died, had a bug, or was restarted).
                # The process_new_bars logic will skip bars that already have trades
                # in the DB, so this is safe to run even when no trades were missed.
                force_reeval = (
                    self.interval in ['1m', '5m', '15m', '30m', '1h', '2h', '4h']
                    and not current_position  # Only when flat (in-position handled above)
                    and hours_old >= 1.0      # At least 1 hour since last trade
                )

                success, result = self.pm.process_new_bars(
                    ticker=self.ticker,
                    config=self.config,
                    interval=self.interval,
                    force_from_last_trade=force_reeval
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
        # 3. SAFETY NET: _check_exit hasn't run in > 2x bar interval
        if position and not self._should_fetch_full_data():
            # Safety net: force full cycle if _check_exit is stale
            interval_seconds = self._get_interval_minutes() * 60
            exit_check_age = _time.time() - self._last_exit_check_time
            if exit_check_age > interval_seconds * 2:
                print(f"   ⚠️ Exit check stale ({exit_check_age:.0f}s > {interval_seconds * 2}s), forcing full cycle")
            else:
                self._quick_sltp_check(position)
                sleep_time = self._calculate_sleep_time()
                print(f"   ⏱️ Quick check completed, sleeping {sleep_time}s")
                time.sleep(sleep_time)
                return

        # Full data fetch and analysis path
        # Enable multi-source supplement near bar boundary (for both entries and exits)
        # This catches the case where yfinance hasn't finalized the latest bar
        # but Databento or Polygon already have it
        use_supplement = False
        now_market = get_current_market_time(self.ticker)
        interval_minutes = self._get_interval_minutes()
        minutes_into_bar = (now_market.minute % interval_minutes) + (now_market.second / 60)
        if minutes_into_bar < 3.0:
            use_supplement = True

        # Fetch price data with retry logic
        df = self._fetch_data(supplement_latest=use_supplement)
        retry_count = 0
        max_retries = 30  # Max 30 seconds of retrying
        while (df is None or df.empty) and retry_count < max_retries:
            retry_count += 1
            print(f"   No data available, retrying in 1s ({retry_count}/{max_retries})...")
            time.sleep(1)
            df = self._fetch_data(supplement_latest=use_supplement)

        if df is None or df.empty:
            # Graceful degradation: if in position, continue SL/TP monitoring
            # even when data fetch fails - don't abandon position monitoring
            position = self.pm.get_current_position()
            if position:
                import time as _t
                stale_seconds = _t.time() - self._last_successful_fetch_time if self._last_successful_fetch_time else 0
                print(f"   ⚠️ Data unavailable ({stale_seconds:.0f}s stale) - continuing SL/TP monitoring")
                self._quick_sltp_check(position)

                # Alert via Discord if data stale > 5 minutes
                if stale_seconds > 300 and not self._stale_data_alert_sent:
                    self._send_error_alert(
                        f"Data fetch failing for {stale_seconds:.0f}s. "
                        f"SL/TP monitoring continues via realtime price."
                    )
                    self._stale_data_alert_sent = True
            else:
                print("   No data after retries, skipping cycle (no position)")

            time.sleep(self.check_interval_seconds)
            return

        # Record successful full fetch time (for optimization when in position)
        self._last_full_fetch_time = _time.time()
        self._last_successful_fetch_time = _time.time()

        # Reset stale data alert if we recovered
        if self._stale_data_alert_sent:
            self._stale_data_alert_sent = False
            print("   ✅ Data fetch recovered")

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
            # Novel Strategy #4: OU threshold amplification (v7+)
            # Boosts oscillator values near OU-derived entry/exit levels
            if self.novel_filters is not None:
                # Train models on first cycle (or retrain daily)
                if not self.novel_filters.trained or self.novel_filters.needs_retrain():
                    print("   [NovelFilters] Training models on historical data...")
                    self.novel_filters.train(df)
                # Apply OU boost to oscillator BEFORE signal generation
                df = self.novel_filters.apply_ou_boost(df)
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
                entropy_threshold=self.entropy_threshold,
                use_vol_regime_filter=self.use_vol_regime_filter,
                vol_regime_percentile_threshold=self.vol_regime_percentile_threshold,
                rsi_filter=self.rsi_filter,
                rsi_period=self.rsi_period,
                rsi_oversold=self.rsi_oversold,
                rsi_overbought=self.rsi_overbought,
                use_macd_confirm=self.use_macd_confirm,
                use_bb_filter=self.use_bb_filter,
            )

            # Train ML entry + exit models (v7+) — needs signals computed first
            if self.novel_filters is not None and self.novel_filters.trained:
                needs_train = (
                    not self.novel_filters._entry_models_trained
                    or not self.novel_filters.exit_model.trained
                    or self.novel_filters.exit_model.needs_retrain()
                )
                if needs_train:
                    self.novel_filters.train_exit_model(
                        df, stop_loss_pct=self.stop_loss_pct
                    )

        # v9: Compute regimes and per-regime signals
        if self.regime_aware:
            self._regimes, self._is_high_vol = self._compute_regimes(df)
            self._regime_signal_dfs = self._generate_regime_signals(df)
            if self._regimes is not None and len(self._regimes) > 0:
                current_regime_name = REGIME_NAMES.get(int(self._regimes[-1]), 'unknown')
                print(f"   Regime: {current_regime_name.upper()}")
        else:
            self._regimes = None
            self._is_high_vol = None
            self._regime_signal_dfs = {}

        # v9: Regime change detection and periodic status notifications
        if self.regime_aware and self.webhook_url and self._regimes is not None and len(self._regimes) > 0:
            current_regime_name = REGIME_NAMES.get(int(self._regimes[-1]), 'unknown')

            # Regime change alert
            if self._last_regime is not None and current_regime_name != self._last_regime:
                self._send_regime_change_alert(self._last_regime, current_regime_name)
            self._last_regime = current_regime_name

            # Periodic regime status update (default every 4 hours)
            import time as _regime_time
            status_interval = self.config.get('regime_status_interval_hours', 4) * 3600
            if _regime_time.time() - self._last_regime_status_time >= status_interval:
                self._send_regime_status_update()
                self._last_regime_status_time = _regime_time.time()

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

        # YFINANCE DATA LAG FIX: If near a bar boundary and the just-closed bar
        # may not yet be in yfinance data, schedule a 1s recheck so we don't wait
        # a full interval (15min) to detect exits OR entries.
        # Applies when: (a) in position and no exit fired, or (b) flat and no entry fired.
        now_market = get_current_market_time(self.ticker)
        interval_minutes = self._get_interval_minutes()
        minutes_into_bar = (now_market.minute % interval_minutes) + (now_market.second / 60)
        if minutes_into_bar < 3.0:
            completed_bar_check, _ = self.get_completed_bar(df)
            if completed_bar_check is None:
                # Bar not yet in data — schedule rapid recheck
                import time as _t2
                self._bar_recheck_time = _t2.time() + 1
                print(f"   📡 Near bar boundary but bar not in data — recheck in 1s")

        # CRITICAL: Update last_processed_bar so process_new_bars doesn't re-process
        # bars on next restart. Without this, startup catch-up re-processes bars that
        # the live loop already handled, creating duplicate trades.
        try:
            completed_bar, bar_time = self.get_completed_bar(df)
            if bar_time is not None:
                from ..core.position_manager import normalize_timestamp
                bar_ts = normalize_timestamp(bar_time)
                self.pm._db.set_last_processed_bar(bar_ts)
        except Exception:
            pass  # Non-critical - best effort

        # Write health status periodically
        self._write_health_status()

        # v10: Daily retraining — run after market close when not in position
        if self._daily_retrainer is not None and self._daily_retrainer.needs_retrain():
            if not is_market_open(self.ticker) and not self.pm.get_current_position():
                self._run_daily_retrain(df)

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
        3. A pending recheck is due (yfinance data lag after bar close)

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

            # Initialize recheck state
            if not hasattr(self, '_bar_recheck_time'):
                self._bar_recheck_time = 0  # Unix timestamp when recheck is due

            # Check if a pending recheck is due (yfinance data lag workaround)
            if self._bar_recheck_time > 0 and now_ts >= self._bar_recheck_time:
                self._bar_recheck_time = 0  # Clear — will be re-set if still needed
                return True

            # Do full fetch if it's been more than interval_minutes since last one
            time_since_last_fetch = now_ts - self._last_full_fetch_time
            if time_since_last_fetch >= interval_seconds:
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

                # CRITICAL FIX: Use tz-aware comparison.
                # bar_start is tz-aware (from get_current_market_time), but
                # datetime.fromtimestamp() returns naive local time → TypeError.
                # Use the market timezone for both.
                market_tz = now.tzinfo
                last_fetch_dt = datetime.fromtimestamp(self._last_full_fetch_time, tz=market_tz)
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

        # Resolve SL/TP from entry regime (v9) or top-level (v8)
        sl_pct = self._get_exit_param('stop_loss_pct', self.stop_loss_pct, position)
        tp_pct = self._get_exit_param('take_profit_pct', self.take_profit_pct, position)

        # Calculate SL/TP price levels for display
        sl_price = entry_price * (1 - sl_pct / 100)
        tp_price = entry_price * (1 + tp_pct / 100)

        regime_tag = f" [{position.entry_regime}]" if position.entry_regime else ""
        print(f"   📊 Position: LONG @ ${entry_price:,.2f} | Current: ${current_price:,.2f}{regime_tag}")
        print(f"   📊 P&L: {pnl_pct:+.2f}% | SL: ${sl_price:,.2f} | TP: ${tp_price:,.2f}")

        exit_reason = None
        exit_price = current_price

        # Stop Loss
        if pnl_pct <= -sl_pct:
            exit_reason = f"Stop Loss ({pnl_pct:.2f}%)"
            print(f"   🛑 STOP LOSS TRIGGERED! P&L {pnl_pct:.2f}% <= -{sl_pct}%")

        # Take Profit
        elif pnl_pct >= tp_pct:
            exit_reason = f"Take Profit ({pnl_pct:.2f}%)"
            print(f"   🎯 TAKE PROFIT TRIGGERED! P&L {pnl_pct:.2f}% >= +{tp_pct}%")

        # Execute exit if triggered
        if exit_reason:
            self._execute_exit(exit_price, exit_reason, None)

    def _fetch_data(self, for_chart: bool = False, supplement_latest: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch price data for signal generation and charts.

        CRITICAL FIX: Uses yfinance ONLY (yfinance_only=True) to ensure
        consistent oscillator values. The DataPipeline and _fetch_futures mix
        Databento+yfinance data which produces different oscillator values,
        causing signals to not match between live trading and fresh calculations.

        Args:
            for_chart: If True, include synthetic current bar for display.
                       If False (default), exclude synthetic for signal generation.
            supplement_latest: If True, check Databento/Polygon for latest bar
                              when yfinance might be lagging (near bar boundary).
        """
        fetch_days = 60 if self.interval in ['1m', '5m', '15m', '30m', '1h', '2h', '4h'] else 365

        # Retry with exponential backoff
        for attempt in range(3):
            try:
                # CRITICAL: yfinance_only=True bypasses ALL Databento paths
                # This ensures oscillator consistency - Databento mixing was causing signal mismatches
                df = fetch_price_data(
                    self.ticker,
                    days=fetch_days,
                    interval=self.interval,
                    yfinance_only=True  # Force yfinance ONLY, no Databento at all
                )
                if df is not None and not df.empty:
                    # DATA REDUNDANCY: When near bar boundary, check if other sources
                    # have a newer bar that yfinance hasn't finalized yet.
                    # Only the latest bar is supplemented — historical data stays yfinance-only
                    # to preserve oscillator consistency.
                    if supplement_latest:
                        try:
                            from ..data.fetcher import fetch_latest_bar_multisource
                            import pytz

                            alt_df = fetch_latest_bar_multisource(
                                self.ticker, self.interval
                            )
                            if alt_df is not None and not alt_df.empty:
                                # Compare latest bar times
                                yf_last = df.index[-1]
                                alt_last = alt_df.index[-1]

                                # Normalize both to naive UTC for comparison
                                if hasattr(yf_last, 'tzinfo') and yf_last.tzinfo is not None:
                                    yf_last_cmp = yf_last.tz_convert(pytz.UTC).tz_localize(None)
                                else:
                                    yf_last_cmp = yf_last
                                if hasattr(alt_last, 'tzinfo') and alt_last.tzinfo is not None:
                                    alt_last_cmp = alt_last.tz_convert(pytz.UTC).tz_localize(None)
                                else:
                                    alt_last_cmp = alt_last

                                if alt_last_cmp > yf_last_cmp:
                                    # Alt source has a newer bar — append it
                                    new_bar = alt_df.iloc[[-1]].copy()
                                    # Match timezone of yfinance index
                                    if df.index.tz is not None and new_bar.index.tz is None:
                                        new_bar.index = new_bar.index.tz_localize('UTC').tz_convert(df.index.tz)
                                    elif df.index.tz is None and new_bar.index.tz is not None:
                                        new_bar.index = new_bar.index.tz_convert('UTC').tz_localize(None)
                                    # Only keep columns that exist in both
                                    common_cols = [c for c in df.columns if c in new_bar.columns]
                                    new_bar = new_bar[common_cols]
                                    df = pd.concat([df, new_bar])
                                    print(f"   📡 Supplemented yfinance with newer bar from alt source: {alt_last_cmp}")
                        except Exception as e:
                            print(f"   📡 Multi-source supplement failed (non-critical): {e}")

                    return df
            except Exception as e:
                print(f"   Fetch attempt {attempt+1} failed: {e}")
                time.sleep(5 * (attempt + 1))

        return None

    def _check_exit(self, df: pd.DataFrame, position: Position):
        """Check if position should be exited."""
        import time as _exit_time
        self._last_exit_check_time = _exit_time.time()
        self._exit_is_stale = False  # Track if exit is from a stale missed signal
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

        # Resolve SL/TP from entry regime (v9) or top-level (v8)
        sl_pct = self._get_exit_param('stop_loss_pct', self.stop_loss_pct, position)
        tp_pct = self._get_exit_param('take_profit_pct', self.take_profit_pct, position)

        # Calculate SL/TP price levels for display
        sl_price = entry_price * (1 - sl_pct / 100)
        tp_price = entry_price * (1 + tp_pct / 100)

        # Debug logging for position monitoring
        regime_tag = f" [{position.entry_regime}]" if position.entry_regime else ""
        print(f"   📊 Position: LONG @ ${entry_price:,.2f} | Current: ${current_price:,.2f} ({price_source}){regime_tag}")
        print(f"   📊 P&L: {pnl_pct:+.2f}% | SL: -{sl_pct}% (${sl_price:,.2f}) | TP: +{tp_pct}% (${tp_price:,.2f})")

        exit_reason = None
        exit_price = current_price
        signal_bar_time = None  # Only set for signal-based exits

        # INTRABAR SL/TP: Check completed bar's High/Low for SL/TP touches
        # The realtime price check below only sees the current instant.
        # If the bar's Low dipped below SL (or High above TP) and then recovered,
        # the realtime check would miss it. This matches WF validator behavior.
        completed_bar, completed_bar_time = self.get_completed_bar(df)
        if completed_bar is not None and exit_reason is None:
            bar_low = completed_bar.get('Low', current_price)
            bar_high = completed_bar.get('High', current_price)
            bar_low_pnl = ((bar_low - entry_price) / entry_price) * 100
            bar_high_pnl = ((bar_high - entry_price) / entry_price) * 100

            if bar_low_pnl <= -sl_pct:
                exit_price = sl_price  # Exit at exact SL level
                exit_reason = f"Stop Loss ({-sl_pct:.2f}%)"
                signal_bar_time = completed_bar_time
                print(f"   🛑 INTRABAR STOP LOSS! Bar Low ${bar_low:,.2f} hit SL ${sl_price:,.2f}")
            elif bar_high_pnl >= tp_pct:
                exit_price = tp_price  # Exit at exact TP level
                exit_reason = f"Take Profit ({tp_pct:.2f}%)"
                signal_bar_time = completed_bar_time
                print(f"   🎯 INTRABAR TAKE PROFIT! Bar High ${bar_high:,.2f} hit TP ${tp_price:,.2f}")

        # Realtime price SL/TP (fallback for current incomplete bar)
        if exit_reason is None and pnl_pct <= -sl_pct:
            exit_reason = f"Stop Loss ({pnl_pct:.2f}%)"
            print(f"   🛑 STOP LOSS TRIGGERED! P&L {pnl_pct:.2f}% <= -{sl_pct}%")

        # Take Profit
        elif exit_reason is None and pnl_pct >= tp_pct:
            exit_reason = f"Take Profit ({pnl_pct:.2f}%)"
            print(f"   🎯 TAKE PROFIT TRIGGERED! P&L {pnl_pct:.2f}% >= +{tp_pct}%")

        # Update high watermark for trailing/breakeven stops
        # Use bar high if available, otherwise current price
        bar_high_for_hwm = current_price
        if completed_bar is not None:
            bar_high_for_hwm = max(current_price, completed_bar.get('High', current_price))
        self._high_watermark = max(self._high_watermark, bar_high_for_hwm)

        # Trailing Stop (fires once activated)
        _use_trailing = self._get_exit_param('use_trailing_stop', self.use_trailing_stop, position)
        if exit_reason is None and _use_trailing and self._high_watermark > 0:
            _trail_pct = self._get_exit_param('trailing_stop_pct', self.trailing_stop_pct, position)
            _trail_act = self._get_exit_param('trailing_stop_activation_pct', self.trailing_stop_activation_pct, position)
            hwm_pnl = ((self._high_watermark - entry_price) / entry_price) * 100
            if hwm_pnl >= _trail_act:
                trail_level = self._high_watermark * (1 - _trail_pct / 100)
                # Check intrabar low if available
                check_price = current_price
                if completed_bar is not None:
                    bar_low = completed_bar.get('Low', current_price)
                    if bar_low <= trail_level:
                        exit_price = trail_level
                        exit_reason = f"Trailing Stop ({pnl_pct:.2f}%)"
                        signal_bar_time = completed_bar_time
                        print(f"   📉 TRAILING STOP! Bar Low ${bar_low:,.2f} hit trail ${trail_level:,.2f} (HWM ${self._high_watermark:,.2f})")
                if exit_reason is None and current_price <= trail_level:
                    exit_reason = f"Trailing Stop ({pnl_pct:.2f}%)"
                    print(f"   📉 TRAILING STOP! Price ${current_price:,.2f} <= trail ${trail_level:,.2f} (HWM ${self._high_watermark:,.2f})")

        # Break-Even Stop (fires once activated)
        _use_be = self._get_exit_param('use_breakeven_stop', self.use_breakeven_stop, position)
        if exit_reason is None and _use_be and self._high_watermark > 0:
            _be_trigger = self._get_exit_param('breakeven_trigger_pct', self.breakeven_trigger_pct, position)
            _be_offset = self._get_exit_param('breakeven_offset_pct', self.breakeven_offset_pct, position)
            hwm_pnl = ((self._high_watermark - entry_price) / entry_price) * 100
            if hwm_pnl >= _be_trigger:
                be_price = entry_price * (1 + _be_offset / 100)
                # Check intrabar low if available
                if completed_bar is not None:
                    bar_low = completed_bar.get('Low', current_price)
                    if bar_low <= be_price:
                        exit_price = be_price
                        exit_reason = f"Break-Even Stop ({pnl_pct:.2f}%)"
                        signal_bar_time = completed_bar_time
                        print(f"   🔒 BREAK-EVEN STOP! Bar Low ${bar_low:,.2f} hit BE ${be_price:,.2f}")
                if exit_reason is None and current_price <= be_price:
                    exit_reason = f"Break-Even Stop ({pnl_pct:.2f}%)"
                    print(f"   🔒 BREAK-EVEN STOP! Price ${current_price:,.2f} <= BE ${be_price:,.2f}")

        # Acceleration Reversal Exit (early warning before stop loss)
        # Resolve accel params from entry regime (v9) or top-level (v8)
        if exit_reason is None and self._get_exit_param('use_accel_exit', self.use_accel_exit, position) and 'acceleration' in df.columns:
            _accel_exit_type = self._get_exit_param('accel_exit_type', self.accel_exit_type, position)
            _accel_exit_threshold = self._get_exit_param('accel_exit_threshold', self.accel_exit_threshold, position)
            _accel_exit_min_pnl = self._get_exit_param('accel_exit_min_pnl', self.accel_exit_min_pnl, position)
            _accel_exit_lookback = self._get_exit_param('accel_exit_lookback', self.accel_exit_lookback, position)
            _use_jerk_confirm = self._get_exit_param('use_jerk_confirm', self.use_jerk_confirm, position)
            _jerk_confirm_threshold = self._get_exit_param('jerk_confirm_threshold', self.jerk_confirm_threshold, position)

            # Check if conditions are met
            pnl_ok = pnl_pct >= _accel_exit_min_pnl or pnl_pct < 0

            if pnl_ok and len(df) >= _accel_exit_lookback + 1:
                accel_values = df['acceleration'].iloc[-_accel_exit_lookback:].values
                current_accel = df['acceleration'].iloc[-1]

                # For LONG positions: negative acceleration is bearish
                accel_cond = False
                if _accel_exit_type == 'sign_reversal':
                    accel_cond = all(a < 0 for a in accel_values)
                elif _accel_exit_type == 'magnitude':
                    accel_cond = current_accel < -_accel_exit_threshold
                elif _accel_exit_type == 'both':
                    accel_cond = all(a < 0 for a in accel_values) and abs(current_accel) > _accel_exit_threshold

                # Jerk confirmation (optional)
                jerk_cond = True
                if _use_jerk_confirm and 'jerk' in df.columns and _jerk_confirm_threshold > 0:
                    current_jerk = df['jerk'].iloc[-1]
                    jerk_cond = current_jerk < -_jerk_confirm_threshold

                if accel_cond and jerk_cond:
                    exit_reason = f"Accel Reversal ({pnl_pct:.2f}%)"
                    print(f"   ⚠️ ACCELERATION REVERSAL EXIT! Accel: {current_accel:.4f}, P&L: {pnl_pct:.2f}%")

        # ML Exit Model (v7+): trained GBM predicts optimal exit timing
        # Checked after SL/TP/Accel but before signal-based exits
        if exit_reason is None and self.novel_filters is not None \
                and self.novel_filters.exit_model.trained \
                and self.is_signal_window():
            try:
                bar_idx = len(df) - 1

                entry_date_ml = pd.to_datetime(position.entry_date)
                if entry_date_ml.tzinfo is not None:
                    entry_date_ml = entry_date_ml.tz_convert('UTC').tz_localize(None)

                df_index_ml = df.index
                if df_index_ml.tz is not None:
                    df_index_ml = df_index_ml.tz_convert('UTC').tz_localize(None)

                entry_bar_idx = 0
                for i, idx_val in enumerate(df_index_ml):
                    if pd.Timestamp(idx_val) >= pd.Timestamp(entry_date_ml):
                        entry_bar_idx = i
                        break

                should_exit, prob, reason = self.novel_filters.exit_model.should_exit(
                    df, entry_price, entry_bar_idx, bar_idx
                )
                if should_exit:
                    exit_reason = f"ML Exit ({pnl_pct:.2f}%, p={prob:.2f})"
                    print(f"   🤖 ML EXIT TRIGGERED! P&L: {pnl_pct:.2f}%, prob: {prob:.2f}")
            except Exception as e:
                print(f"   [ExitModel] Check failed: {e}")

        # Signal-based exits (only if no SL/TP/Accel/ML and in signal window)
        if exit_reason is None and self.is_signal_window():
            # CRITICAL FIX: Check ALL bars since position entry for missed exit signals
            # Previously only checked the most recent completed bar, missing signals that
            # fired during weekends, gaps, or when trader wasn't running.
            entry_date = pd.to_datetime(position.entry_date)
            # Normalize to naive UTC for comparison with df.index (which is also naive UTC)
            if entry_date.tzinfo is not None:
                entry_date = entry_date.tz_convert('UTC').tz_localize(None)

            # Ensure df.index is also tz-naive UTC for comparison
            # CRITICAL: Must convert to UTC first, THEN strip timezone.
            # tz_localize(None) alone would leave values in local time (e.g., EST),
            # but entry_date is stored in naive UTC, causing hour offsets that prevent
            # any bars from being found after entry.
            df_index = df.index
            if df_index.tz is not None:
                df_index = df_index.tz_convert('UTC').tz_localize(None)

            # Convert entry_date to same dtype as index to avoid comparison errors
            entry_date = pd.Timestamp(entry_date)

            # v9: Use entry regime's signal DataFrame for opposite signal detection
            signal_df = df
            if self.regime_aware and position.entry_regime and position.entry_regime in self._regime_signal_dfs:
                signal_df = self._regime_signal_dfs[position.entry_regime]

            # Find bars AFTER position entry (use > not >= to exclude entry bar)
            # Entry bar cannot have exit signal - that would be same-bar exit
            signal_df_index = signal_df.index
            if signal_df_index.tz is not None:
                signal_df_index = signal_df_index.tz_convert('UTC').tz_localize(None)
            df_since_entry = signal_df[signal_df_index > entry_date]

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

                    # Check if signal is stale (from a past bar, not the most recent)
                    completed_bar_exit, completed_bar_time_exit = self.get_completed_bar(df)
                    first_signal_time_dt = pd.to_datetime(first_signal_time)
                    if first_signal_time_dt.tzinfo is not None:
                        first_signal_time_dt = first_signal_time_dt.tz_convert('UTC').tz_localize(None)
                    completed_time_dt = pd.to_datetime(completed_bar_time_exit) if completed_bar_time_exit else None
                    if completed_time_dt is not None and completed_time_dt.tzinfo is not None:
                        completed_time_dt = completed_time_dt.tz_convert('UTC').tz_localize(None)

                    is_stale_signal = (completed_time_dt is not None and
                                       first_signal_time_dt < completed_time_dt)
                    self._exit_is_stale = is_stale_signal

                    if is_stale_signal:
                        # Signal is from a past bar — use current market price
                        stale_price = fetch_realtime_price(self.ticker)
                        if stale_price and stale_price > 0:
                            exit_price = stale_price
                            print(f"   📍 Found STALE {signal_col} at {first_signal_time}, using current price ${exit_price:,.2f}")
                        else:
                            exit_price = first_signal['Close']
                            print(f"   📍 Found STALE {signal_col} at {first_signal_time}, realtime unavailable, using bar close ${exit_price:,.2f}")
                    else:
                        # Timely signal — use bar close (matches backtest)
                        exit_price = first_signal['Close']
                        print(f"   📍 Found {signal_col} at {first_signal_time}")

                    # Calculate P&L based on position type
                    if is_long:
                        bar_pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    else:
                        bar_pnl_pct = ((entry_price - exit_price) / entry_price) * 100

                    exit_reason = f"Opposite Signal ({bar_pnl_pct:.2f}%)"

                    # Use bar START time so chart plots marker on signal bar
                    signal_bar_time = first_signal_time

        # Execute exit if triggered
        if exit_reason:
            # CRITICAL: Re-verify position still exists before exiting
            # Another process might have exited/cleared the position while we were processing
            current_pos = self.pm.get_current_position()
            if current_pos is None:
                print(f"   ⚠️ Position no longer exists in database - skipping exit")
                print(f"   ℹ️  This can happen if another process exited the position or ran a rebuild")
                return
            # Mark as missed if this was a stale opposite signal recovery
            _is_missed = getattr(self, '_exit_is_stale', False)
            self._execute_exit(exit_price, exit_reason, signal_bar_time, is_missed=_is_missed)

    def _check_entry(self, df: pd.DataFrame):
        """Check for entry signals."""
        if not self.is_signal_window():
            return

        completed_bar, bar_time = self.get_completed_bar(df)
        if completed_bar is None or bar_time is None:
            return

        # v9 regime-aware: check regime-specific buy signal
        entry_regime = None
        if self.regime_aware and self._regimes is not None and len(self._regimes) > 0:
            # Get bar index for the completed bar
            bar_idx = df.index.get_loc(bar_time) if bar_time in df.index else len(df) - 2
            if bar_idx < 0 or bar_idx >= len(self._regimes):
                return
            regime_id = int(self._regimes[bar_idx])
            regime_name = REGIME_NAMES.get(regime_id, 'unknown')

            # Check high-vol suppression
            suppress = self.config.get('regime_detector', {}).get('suppress_entries_high_vol', False)
            if suppress and self._is_high_vol is not None and self._is_high_vol[bar_idx]:
                print(f"   [v9] High-vol suppressed entry (regime={regime_name})")
                return

            # Check if regime is tradeable
            if regime_name not in self._regime_signal_dfs:
                return

            # Use regime-specific signal DataFrame
            regime_df = self._regime_signal_dfs[regime_name]
            if bar_time in regime_df.index:
                regime_bar = regime_df.loc[bar_time]
                has_buy = regime_bar.get('buy_signal', False)
            else:
                has_buy = False

            if not has_buy:
                return

            entry_regime = regime_name
            print(f"   [v9] Buy signal in {regime_name.upper()} regime")
        else:
            # v8 path: use top-level buy signal
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

        self._execute_entry(entry_price, entry_time, entry_regime=entry_regime)
        self.last_signal_time = signal_time_str

    def _execute_entry(self, entry_price: float, signal_time: datetime, entry_regime: str = None):
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
            entry_signal_bar=str(signal_time),
            entry_regime=entry_regime
        )

        if success:
            # Reset high watermark for trailing/breakeven stops
            self._high_watermark = entry_price
            regime_tag = f" [{entry_regime}]" if entry_regime else ""
            print(f"   ENTRY: LONG @ ${entry_price:,.2f}{regime_tag}")

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

                # Format signal time — show bar CLOSE time (start + interval)
                # so the displayed time matches when the signal is actionable
                import pytz
                display_time = _bar_close_time(signal_time, self._get_interval_minutes())
                if hasattr(display_time, 'tzinfo') and display_time.tzinfo is not None:
                    utc_time = display_time.astimezone(pytz.UTC)
                    signal_time_str = utc_time.strftime('%Y-%m-%d %H:%M') + " UTC"
                else:
                    signal_time_str = str(display_time)[:16]

                # Add regime to signal time for Discord
                if entry_regime:
                    signal_time_str += f" | Regime: {entry_regime.upper()}"

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

                # Use regime-aware SL/TP for Discord display
                _sl = self._get_exit_param('stop_loss_pct', self.stop_loss_pct, self.pm.get_current_position()) if entry_regime else self.stop_loss_pct
                _tp = self._get_exit_param('take_profit_pct', self.take_profit_pct, self.pm.get_current_position()) if entry_regime else self.take_profit_pct
                send_entry_alert(
                    self.webhook_url,
                    self.strategy_name,
                    self.ticker,
                    entry_price,
                    signal_time_str,
                    _sl,
                    _tp,
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
                                    regime_threshold=self.regime_threshold,
                                    use_vol_regime_filter=self.use_vol_regime_filter,
                                    vol_regime_percentile_threshold=self.vol_regime_percentile_threshold,
                                    rsi_filter=self.rsi_filter,
                                    rsi_period=self.rsi_period,
                                    rsi_oversold=self.rsi_oversold,
                                    rsi_overbought=self.rsi_overbought,
                                    use_macd_confirm=self.use_macd_confirm,
                                    use_bb_filter=self.use_bb_filter,
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
                            signal_time=_bar_close_time(signal_time, self._get_interval_minutes())
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

    def _execute_exit(self, exit_price: float, exit_reason: str, signal_bar_time=None, is_missed: bool = False):
        """Execute position exit."""
        from ..core.position_manager import normalize_timestamp
        from ..data.market_hours import get_current_market_time

        # Use signal bar timestamp for signal-based exits, current time for SL/TP
        if signal_bar_time is not None:
            exit_date = normalize_timestamp(signal_bar_time)
            self._exit_is_bar_time = True  # exit_date is bar start → display needs +interval
        else:
            # SL/TP exits use current time (they're triggered by price, not bar completion)
            exit_date = get_current_market_time(self.ticker).isoformat()
            self._exit_is_bar_time = False  # exit_date is already realtime → no offset needed

        # CRITICAL: Use PositionManager for atomic exit
        success, result = self.pm.exit_position(
            exit_price=exit_price,
            exit_date=exit_date,
            exit_reason=exit_reason,
            interval=self.interval,
            is_missed=is_missed
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

                # Format times — all displayed in UTC for consistency.
                # Entry: bar start + interval = bar close time (when signal is actionable).
                # Exit: bar close time for signal exits, or realtime UTC for SL/TP/Accel.
                _int_mins = self._get_interval_minutes()
                # Entry is always bar-based → always add interval
                entry_time_str = str(_bar_close_time(result.get('entry_date', ''), _int_mins))[:16]
                # Exit: add interval for bar-start exits, convert to UTC for realtime exits
                if getattr(self, '_exit_is_bar_time', False):
                    exit_time_str = str(_bar_close_time(exit_date, _int_mins))[:16]
                else:
                    # Realtime exit (SL/TP/Accel/ML) — normalize to UTC for display
                    try:
                        _exit_dt = pd.to_datetime(exit_date)
                        if _exit_dt.tzinfo is not None:
                            _exit_dt = _exit_dt.tz_convert('UTC')
                        exit_time_str = _exit_dt.strftime('%Y-%m-%d %H:%M')
                    except Exception:
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
                                    regime_threshold=self.regime_threshold,
                                    use_vol_regime_filter=self.use_vol_regime_filter,
                                    vol_regime_percentile_threshold=self.vol_regime_percentile_threshold,
                                    rsi_filter=self.rsi_filter,
                                    rsi_period=self.rsi_period,
                                    rsi_oversold=self.rsi_oversold,
                                    rsi_overbought=self.rsi_overbought,
                                    use_macd_confirm=self.use_macd_confirm,
                                    use_bb_filter=self.use_bb_filter,
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
                            signal_time=_bar_close_time(exit_date, self._get_interval_minutes()) if getattr(self, '_exit_is_bar_time', False) else exit_date
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
                        regime_threshold=self.regime_threshold,
                        use_vol_regime_filter=self.use_vol_regime_filter,
                        vol_regime_percentile_threshold=self.vol_regime_percentile_threshold,
                        rsi_filter=self.rsi_filter,
                        rsi_period=self.rsi_period,
                        rsi_oversold=self.rsi_oversold,
                        rsi_overbought=self.rsi_overbought,
                        use_macd_confirm=self.use_macd_confirm,
                        use_bb_filter=self.use_bb_filter,
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
                _startup_is_bar_time = False
                if position:
                    chart_signal_time = position.entry_signal_bar or position.entry_date
                    _startup_is_bar_time = True  # entry dates are always bar-start
                elif exits:
                    # Get most recent exit time — could be bar-start or realtime
                    chart_signal_time = exits[-1].get('date')
                    # Check if exit reason suggests bar-based or realtime exit
                    last_exit_reason = exits[-1].get('reason', '')
                    _startup_is_bar_time = 'Stop Loss' not in last_exit_reason and 'Take Profit' not in last_exit_reason

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
                    signal_time=_bar_close_time(chart_signal_time, self._get_interval_minutes()) if _startup_is_bar_time else chart_signal_time
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

    def _send_regime_change_alert(self, old_regime: str, new_regime: str):
        """Send regime change notification to Discord."""
        try:
            rd = self.config.get('regime_detector', {})
            adx_threshold = rd.get('adx_threshold', 25.0)

            # Get latest ADX/DI values
            adx_val = float(self._adx_values[-1]) if self._adx_values is not None and len(self._adx_values) > 0 else 0.0
            pdi_val = float(self._plus_di_values[-1]) if self._plus_di_values is not None and len(self._plus_di_values) > 0 else 0.0
            mdi_val = float(self._minus_di_values[-1]) if self._minus_di_values is not None and len(self._minus_di_values) > 0 else 0.0

            # Handle NaN
            if np.isnan(adx_val):
                adx_val = 0.0
            if np.isnan(pdi_val):
                pdi_val = 0.0
            if np.isnan(mdi_val):
                mdi_val = 0.0

            # Check if new regime is tradeable
            regime_params = self.config.get('regime_params', {})
            is_tradeable = new_regime in regime_params and not regime_params[new_regime].get('dont_trade', False)

            # High vol
            is_hv = bool(self._is_high_vol[-1]) if self._is_high_vol is not None and len(self._is_high_vol) > 0 else False

            detected_at = datetime.now().strftime('%Y-%m-%d %H:%M UTC')

            send_regime_change_alert(
                webhook_url=self.webhook_url,
                strategy_name=self.strategy_name,
                old_regime=old_regime,
                new_regime=new_regime,
                adx=adx_val,
                adx_threshold=adx_threshold,
                plus_di=pdi_val,
                minus_di=mdi_val,
                is_tradeable=is_tradeable,
                is_high_vol=is_hv,
                detected_at=detected_at,
            )
            print(f"   [v9] Regime change alert sent: {old_regime} -> {new_regime}")
        except Exception as e:
            print(f"   [v9] Regime change alert failed: {e}")

    def _send_regime_status_update(self):
        """Send periodic regime status update to Discord."""
        try:
            rd = self.config.get('regime_detector', {})

            # Get latest values
            adx_val = float(self._adx_values[-1]) if self._adx_values is not None and len(self._adx_values) > 0 else 0.0
            pdi_val = float(self._plus_di_values[-1]) if self._plus_di_values is not None and len(self._plus_di_values) > 0 else 0.0
            mdi_val = float(self._minus_di_values[-1]) if self._minus_di_values is not None and len(self._minus_di_values) > 0 else 0.0

            if np.isnan(adx_val):
                adx_val = 0.0
            if np.isnan(pdi_val):
                pdi_val = 0.0
            if np.isnan(mdi_val):
                mdi_val = 0.0

            is_hv = bool(self._is_high_vol[-1]) if self._is_high_vol is not None and len(self._is_high_vol) > 0 else False

            current_regime = REGIME_NAMES.get(int(self._regimes[-1]), 'unknown') if self._regimes is not None and len(self._regimes) > 0 else 'unknown'

            # Trading status
            regime_params = self.config.get('regime_params', {})
            is_tradeable = current_regime in regime_params and not regime_params[current_regime].get('dont_trade', False)

            suppress_hv = rd.get('suppress_entries_high_vol', False)
            if not is_tradeable:
                trading_status = f"PAUSED -- {current_regime} regime (not tradeable)"
            elif suppress_hv and is_hv:
                trading_status = "PAUSED -- high volatility (entries suppressed)"
            else:
                trading_status = "ACTIVE -- accepting signals"

            # Position string
            position = self.pm.get_current_position()
            if position:
                current_price = None
                try:
                    current_price = fetch_realtime_price(self.ticker)
                except Exception:
                    pass
                if current_price:
                    pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                    position_str = f"LONG @ ${position.entry_price:,.2f} ({pnl_pct:+.1f}%)"
                else:
                    position_str = f"LONG @ ${position.entry_price:,.2f}"
            else:
                position_str = "Flat"

            # Regime distribution from last 96 bars (~24h of 15m bars)
            lookback = min(96, len(self._regimes)) if self._regimes is not None else 0
            if lookback > 0:
                recent_regimes = self._regimes[-lookback:]
                regime_dist = get_regime_distribution(recent_regimes)
            else:
                regime_dist = {}

            interval_hours = self.config.get('regime_status_interval_hours', 4)

            send_regime_status_update(
                webhook_url=self.webhook_url,
                strategy_name=self.strategy_name,
                current_regime=current_regime,
                adx=adx_val,
                plus_di=pdi_val,
                minus_di=mdi_val,
                is_high_vol=is_hv,
                trading_status=trading_status,
                position_str=position_str,
                regime_distribution=regime_dist,
                interval_hours=interval_hours,
            )
            print(f"   [v9] Regime status update sent ({current_regime})")
        except Exception as e:
            print(f"   [v9] Regime status update failed: {e}")

    def _write_health_status(self):
        """Write health status file for external monitoring."""
        import json
        health_file = f"/tmp/patternfindr_{self.strategy_name}_health.json"
        try:
            position = self.pm.get_current_position()
            status = {
                'last_update': datetime.now().isoformat(),
                'strategy': self.strategy_name,
                'ticker': self.ticker,
                'interval': self.interval,
                'pid': os.getpid(),
                'in_position': position is not None,
                'consecutive_errors': self.consecutive_errors,
                'last_successful_fetch': self._last_successful_fetch_time,
                'stale_data_alert': self._stale_data_alert_sent,
            }
            if position:
                status['entry_price'] = position.entry_price
                status['entry_date'] = position.entry_date
            with open(health_file, 'w') as f:
                json.dump(status, f, indent=2)
        except OSError:
            pass  # Non-critical

    def _run_daily_retrain(self, df):
        """
        v10: Re-optimize numerical params on trailing data after market close.
        Updates config in-place and saves to strategy bundle.
        """
        print(f"\n   {'='*50}")
        print(f"   [v10] DAILY RETRAINING — {self.strategy_name}")
        print(f"   {'='*50}")

        try:
            new_params = self._daily_retrainer.retrain(df)
            if new_params is None:
                print(f"   [v10] Retraining failed or insufficient data")
                return

            # Apply new params to live config
            old_sl = self.config.get('stop_loss_pct')
            old_tp = self.config.get('take_profit_pct')
            self.config = self._daily_retrainer.apply_params(self.config, new_params)

            # Update instance attributes that are read from config
            self.stop_loss_pct = self.config.get('stop_loss_pct', self.stop_loss_pct)
            self.take_profit_pct = self.config.get('take_profit_pct', self.take_profit_pct)
            self.oversold_threshold = self.config.get('oversold_threshold', self.oversold_threshold)
            self.overbought_threshold = self.config.get('overbought_threshold', self.overbought_threshold)
            self.vel_smoothing = self.config.get('vel_smoothing', self.vel_smoothing)
            self.extreme_zone_mult = self.config.get('extreme_zone_mult', self.extreme_zone_mult)

            print(f"   [v10] Params updated: SL {old_sl:.2f}→{self.stop_loss_pct:.2f}, "
                  f"TP {old_tp:.2f}→{self.take_profit_pct:.2f}")

            # Save to strategy bundle directory
            strategy_dir = os.path.join(PARENT_DIR, 'velocity_strategies', self.strategy_name)
            if os.path.isdir(strategy_dir):
                self._daily_retrainer.save_config(self.config, strategy_dir)

            # Discord notification
            if self.webhook_url:
                try:
                    from ..notifications.discord import send_status_update
                    param_summary = ', '.join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in sorted(new_params.items()))
                    send_status_update(
                        self.webhook_url,
                        f"[v10] Daily retrain complete — {param_summary}",
                        self.strategy_name,
                    )
                except Exception:
                    pass

            print(f"   [v10] Retraining complete")
            print(f"   {'='*50}\n")

        except Exception as e:
            print(f"   [v10] Retraining error: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """Stop the trading loop."""
        self.running = False
