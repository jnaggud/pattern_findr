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

        # Initialize position manager (SQLite-backed)
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

        - If empty: Run full rebuild from backtest (fetch data from providers)
        - If stale: Run incremental update (only fetch new data)

        This makes the new system completely independent of old JSON files.
        """
        trade_count = self.pm.get_trade_count()
        last_trade = self.pm.get_last_trade_date()

        print(f"📊 Database check: {trade_count} trades", end="")
        if last_trade:
            print(f", last: {last_trade[:10]}")
        else:
            print()

        if trade_count == 0:
            # Empty database - need full rebuild
            print(f"   ℹ️  Database empty, rebuilding from backtest...")
            success, result = self.pm.rebuild_from_backtest(
                ticker=self.ticker,
                config=self.config,
                days=self._get_rebuild_days()
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
        now = datetime.now()
        print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] [{self.strategy_name}] Checking {self.ticker}...")

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

        # Get current position
        position = self.pm.get_current_position()

        # Check for exit conditions FIRST (if in position)
        if position:
            self._check_exit(df, position)

        # Check for entry conditions (if not in position)
        if not self.pm.get_current_position():  # Re-check after potential exit
            self._check_entry(df)

        # Sleep until next check
        sleep_time = self._calculate_sleep_time()
        print(f"   [{self.strategy_name}] Next check in {sleep_time}s")
        time.sleep(sleep_time)

    def _fetch_data(self, for_chart: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch price data using incremental pipeline.

        On first call: Backfills historical data (Historical API)
        On every call: Updates with latest bars (Live API)
        Returns data from SQLite (fast, always complete)

        Args:
            for_chart: If True, include synthetic current bar for display.
                       If False (default), exclude synthetic for signal generation.
        """
        try:
            # Initialize pipeline on first run (backfill if needed)
            if not self._pipeline_initialized:
                stats = self.pipeline.get_stats()
                if stats['total_bars'] < 100:
                    print("   Backfilling historical data...")
                    days = 60 if self.interval in ['15m', '30m', '1h'] else 365
                    success, result = self.pipeline.backfill(days=days)
                    if success:
                        print(f"   Backfill complete: {result.get('bars_stored', 0)} bars")
                    else:
                        print(f"   Backfill warning: {result.get('error', 'unknown')}")
                self._pipeline_initialized = True

            # Update with latest bars from Live API (runs every cycle)
            success, result = self.pipeline.update()
            if result.get('new_bars', 0) > 0:
                print(f"   +{result['new_bars']} new bars (latest: {result.get('latest', 'N/A')[:16]})")

            # Get data from SQLite
            # Use 600 bars for intraday: ~6 days crypto (96/day), ~22 days futures (27/day)
            # Enough for chart (500 bars max) plus indicator warmup
            lookback = 600 if self.interval in ['1m', '5m', '15m', '30m', '1h', '2h', '4h'] else 500

            # For signal generation: NO synthetic bars (could trigger false signals)
            # For charts: ADD synthetic bar to show current price
            df = self.pipeline.get_data(lookback_bars=lookback, add_synthetic_current=for_chart)
            if df is not None and not df.empty:
                return df

        except Exception as e:
            print(f"   Pipeline error: {e}, falling back to direct fetch")

        # Fallback to direct fetch if pipeline fails
        # For intraday, yfinance max is 60 days; for daily, get 365 days
        fetch_days = 60 if self.interval in ['1m', '5m', '15m', '30m', '1h', '2h', '4h'] else 365
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
                time.sleep(10 * (attempt + 1))

        return None

    def _check_exit(self, df: pd.DataFrame, position: Position):
        """Check if position should be exited."""
        # Get current price
        current_price = fetch_realtime_price(self.ticker)
        if current_price is None:
            current_price = df['Close'].iloc[-1]

        # Check stop loss and take profit
        entry_price = position.entry_price
        pnl_pct = ((current_price - entry_price) / entry_price) * 100

        exit_reason = None
        exit_price = current_price
        signal_bar_time = None  # Only set for signal-based exits

        # Stop Loss
        if pnl_pct <= -self.stop_loss_pct:
            exit_reason = f"Stop Loss ({pnl_pct:.2f}%)"

        # Take Profit
        elif pnl_pct >= self.take_profit_pct:
            exit_reason = f"Take Profit ({pnl_pct:.2f}%)"

        # Signal-based exits (only if no SL/TP and in signal window)
        elif self.is_signal_window():
            completed_bar, bar_time = self.get_completed_bar(df)
            if completed_bar is not None:
                # Check for opposite signal
                if completed_bar.get('sell_signal', False):
                    exit_reason = f"Opposite Signal ({pnl_pct:.2f}%)"
                    exit_price = completed_bar['Close']
                    signal_bar_time = bar_time

        # Execute exit if triggered
        if exit_reason:
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

        # Get current price for entry
        current_price = fetch_realtime_price(self.ticker)
        if current_price is None:
            current_price = completed_bar['Close']

        # Execute entry
        self._execute_entry(current_price, bar_time)
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

                # Generate chart for entry alert
                chart = None
                try:
                    df = self._fetch_data(for_chart=True)  # Include synthetic bar for display
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
                            overbought_threshold=self.overbought_threshold
                        )
                except Exception as e:
                    print(f"   Warning: Entry chart generation failed: {e}")

                send_entry_alert(
                    self.webhook_url,
                    self.strategy_name,
                    self.ticker,
                    entry_price,
                    str(signal_time)[:16],
                    self.stop_loss_pct,
                    self.take_profit_pct,
                    stats,
                    chart=chart,
                    enhanced_stats=enhanced_stats
                )
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

                # Generate chart for exit alert
                chart = None
                try:
                    df = self._fetch_data(for_chart=True)  # Include synthetic bar for display
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
                            overbought_threshold=self.overbought_threshold
                        )
                except Exception as e:
                    print(f"   Warning: Exit chart generation failed: {e}")

                send_exit_alert(
                    self.webhook_url,
                    self.strategy_name,
                    self.ticker,
                    result.get('entry_price', 0),
                    exit_price,
                    pnl,
                    result.get('pnl_dollars', 0),
                    exit_reason,
                    result.get('entry_date', ''),
                    exit_date,
                    cumulative,
                    chart=chart,
                    csv_buf=csv_buf,
                    enhanced_stats=enhanced_stats
                )
        else:
            print(f"   Exit rejected: {result.get('error')} - {result.get('message')}")

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
                    overbought_threshold=self.overbought_threshold
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
