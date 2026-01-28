"""
Chart generation for trading notifications.

Generates TradingView-style charts for Discord webhooks.
Matches the visual style of the original velocity_live_trader.py charts.
"""

import io
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add parent directory to path for oscillator imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Try to import oscillator calculation functions from main module
try:
    from oscillator_predictor_page import create_composite_oscillator
    HAS_OSCILLATOR_MODULE = True
except ImportError:
    HAS_OSCILLATOR_MODULE = False

# Starting capital for equity curve
STARTING_CAPITAL = 100000


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance to simple column names."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns columns like ('Close', 'SPY') - extract first level
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def _ensure_oscillator_columns(df: pd.DataFrame, vel_smoothing: int = 3) -> pd.DataFrame:
    """
    Ensure the DataFrame has oscillator columns for charting.

    If osc_smooth/composite_smooth, velocity, and acceleration are missing,
    calculate them using the same method as the old system.
    """
    df = df.copy()

    # Flatten MultiIndex columns if present (from yfinance)
    df = _flatten_columns(df)

    # Check for existing oscillator columns
    has_osc = 'osc_smooth' in df.columns or 'composite_smooth' in df.columns
    has_velocity = 'velocity' in df.columns
    has_acceleration = 'acceleration' in df.columns

    if has_osc and has_velocity and has_acceleration:
        return df  # Already has all needed columns

    # Need to calculate oscillators
    if HAS_OSCILLATOR_MODULE and not has_osc:
        try:
            # Use the same function as the old system
            df = create_composite_oscillator(df)
            if 'composite_smooth' in df.columns:
                df['osc_smooth'] = df['composite_smooth']
        except Exception as e:
            print(f"   Warning: Failed to calculate composite oscillator: {e}")

    # Use composite_smooth as osc_smooth if available
    if 'composite_smooth' in df.columns and 'osc_smooth' not in df.columns:
        df['osc_smooth'] = df['composite_smooth']

    # Calculate velocity and acceleration if we have osc_smooth
    osc_col = 'osc_smooth' if 'osc_smooth' in df.columns else 'composite_smooth'
    if osc_col in df.columns:
        # Apply smoothing (same as old system)
        if vel_smoothing > 1:
            df['osc_smooth'] = df[osc_col].rolling(window=vel_smoothing, min_periods=1).mean()

        # Calculate velocity (first derivative)
        if 'velocity' not in df.columns:
            df['velocity'] = df['osc_smooth'].diff().fillna(0)

        # Calculate acceleration (second derivative)
        if 'acceleration' not in df.columns:
            df['acceleration'] = df['velocity'].diff().fillna(0)

    return df


def _format_signal_time_dual_tz(signal_time) -> Optional[str]:
    """
    Format signal time in both UTC and Chicago time for display.

    Args:
        signal_time: datetime, timestamp, or string

    Returns:
        Formatted string like "Signal: 10:15 UTC / 04:15 Chicago"
        or None if conversion fails
    """
    if signal_time is None:
        return None

    try:
        # Parse the signal time
        if isinstance(signal_time, str):
            dt = pd.to_datetime(signal_time)
        else:
            dt = pd.to_datetime(signal_time)

        # Assume UTC if no timezone
        if dt.tzinfo is None:
            if HAS_PYTZ:
                dt = pytz.UTC.localize(dt)
            else:
                # Without pytz, just show UTC
                utc_str = dt.strftime('%Y-%m-%d %H:%M')
                return f"Signal: {utc_str} UTC"

        if HAS_PYTZ:
            # Convert to UTC
            utc_dt = dt.astimezone(pytz.UTC)
            utc_str = utc_dt.strftime('%Y-%m-%d %H:%M')

            # Convert to Chicago time
            chicago_tz = pytz.timezone('America/Chicago')
            chicago_dt = dt.astimezone(chicago_tz)
            chicago_str = chicago_dt.strftime('%Y-%m-%d %H:%M')

            return f"Signal: {utc_str} UTC  /  {chicago_str} Chicago"
        else:
            # Fallback without pytz
            utc_str = dt.strftime('%Y-%m-%d %H:%M')
            return f"Signal: {utc_str} UTC"

    except Exception as e:
        print(f"   Warning: Failed to format signal time: {e}")
        return None


def generate_chart(
    df: pd.DataFrame,
    entries: List[Dict],
    exits: List[Dict],
    ticker: str,
    stats: Dict = None,
    current_position: Dict = None,
    title_suffix: str = "",
    chart_type: str = "status",
    interval: str = "1d",
    oversold_threshold: float = -0.3,
    overbought_threshold: float = 0.3,
    signal_time=None
) -> Optional[io.BytesIO]:
    """
    Generate a JD Strategy chart for Discord.

    Args:
        df: DataFrame with OHLC, osc_smooth, velocity, acceleration columns
        entries: List of entry dicts with 'date', 'price'
        exits: List of exit dicts with 'date', 'price', 'pnl'
        ticker: Trading symbol
        stats: Optional stats dict with win_rate, total_return, etc.
        current_position: Optional current position dict
        title_suffix: Optional suffix for chart title (e.g., " (Last 182 Days)")
        chart_type: "signal" for focused view, "status" for broad view
        interval: Bar interval ('1d', '15m', etc.)
        oversold_threshold: Oversold zone threshold
        overbought_threshold: Overbought zone threshold
        signal_time: Optional signal timestamp to display in UTC and Chicago time

    Returns:
        BytesIO buffer with PNG image, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        print("   Warning: matplotlib not available, skipping chart generation")
        return None

    if df is None or df.empty:
        return None

    try:
        # Determine if intraday
        is_intraday = interval in ['1m', '5m', '15m', '30m', '1h', '90m', '4h']

        # Limit data for chart
        # For intraday: balance readability vs showing enough context
        # Target: ~3 days of data for readability
        # 15m futures (27 bars/day): 260 bars = ~3-4 days
        # 15m crypto (96 bars/day): 260 bars = ~2.7 days
        if chart_type == "signal":
            max_bars = 150 if is_intraday else 90
        else:
            max_bars = 260 if is_intraday else 252  # ~3 days for 15m, ~1 year daily

        df_plot = df.tail(max_bars).copy()

        # Ensure oscillator columns exist (calculate if missing)
        df_plot = _ensure_oscillator_columns(df_plot)

        # Normalize column names (handle case variations)
        col_map = _get_column_map(df_plot)

        if 'close' not in col_map:
            print("   Warning: No close price column found")
            return None

        # Create bar numbers for x-axis (TradingView style - no gaps)
        df_plot = df_plot.reset_index()
        df_plot['bar_num'] = range(len(df_plot))

        # Find datetime column
        datetime_col = _find_datetime_column(df_plot)

        if datetime_col:
            df_plot[datetime_col] = pd.to_datetime(df_plot[datetime_col])
            if df_plot[datetime_col].dt.tz is not None:
                df_plot[datetime_col] = df_plot[datetime_col].dt.tz_localize(None)

        # Create date lookup for markers
        date_to_barnum = _build_date_lookup(df_plot, datetime_col)

        # Get chart date range (for filtering equity curve trades)
        chart_start, chart_end = _get_chart_date_range(df_plot, datetime_col)

        # For MARKERS: use all entries/exits (markers only appear if date matches a bar)
        entries_for_markers = entries
        exits_for_markers = exits

        # For EQUITY CURVE: filter to ONLY trades within the chart's visible date range
        entries_for_equity = _filter_to_date_range(entries, chart_start, chart_end, extend_days=0)
        exits_for_equity = _filter_to_date_range(exits, chart_start, chart_end, extend_days=0)

        # Calculate ALL-TIME stats (for display)
        all_time_stats = _calculate_visible_stats(exits)

        # Calculate VISIBLE PERIOD stats (for equity curve period)
        visible_stats = _calculate_visible_stats(exits_for_equity)

        print(f"      Chart: {len(df_plot)} bars, all-time: {all_time_stats['num_trades']} trades, visible: {visible_stats['num_trades']} trades")

        # Create figure with 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(14, 11),
                                 gridspec_kw={'height_ratios': [3, 1.2, 0.8, 1.2]},
                                 sharex=False)

        # Dark theme (TradingView style)
        fig.patch.set_facecolor('#1a1a2e')
        for ax in axes:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white', labelsize=8)
            ax.grid(True, color='#333', alpha=0.5, linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_color('#333')

        x_axis = df_plot['bar_num']

        # === Panel 1: Price Chart ===
        ax1 = axes[0]
        _draw_candlesticks(ax1, df_plot, col_map)
        _draw_entry_exit_markers(ax1, entries_for_markers, exits_for_markers, date_to_barnum, df_plot, col_map, is_intraday=is_intraday)

        # Mark current position entry line
        if current_position and current_position.get('entry_price'):
            entry_price = current_position['entry_price']
            ax1.axhline(y=entry_price, color='cyan', linestyle='--',
                       linewidth=1, alpha=0.7, label=f'Entry ${entry_price:,.2f}')
            handles, labels = ax1.get_legend_handles_labels()
            if handles:
                ax1.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white', fontsize=8)

        ax1.set_ylabel("Price ($)", color='white', fontsize=9)

        # Title with date range
        chart_title = _build_chart_title(ticker, title_suffix, df_plot, datetime_col, is_intraday)
        ax1.set_title(chart_title, color='white', fontsize=12, fontweight='bold', pad=10)

        # Add signal time in UTC and Chicago time (upper right corner)
        if signal_time is not None:
            signal_time_str = _format_signal_time_dual_tz(signal_time)
            if signal_time_str:
                ax1.text(0.99, 0.97, signal_time_str, transform=ax1.transAxes,
                        ha='right', va='top', color='#00ffff', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f3460', alpha=0.9, edgecolor='#00ffff'))

        _format_xaxis_dates(ax1, df_plot, datetime_col, is_intraday)

        # === Panel 2: JD_Osc (Oscillator) ===
        ax2 = axes[1]
        osc_col = col_map.get('osc_smooth') or col_map.get('composite_smooth')

        if osc_col and osc_col in df_plot.columns:
            ax2.plot(x_axis, df_plot[osc_col], color='#e94560', linewidth=1.5, label='JD_Osc')
            ax2.axhline(oversold_threshold, color='lime', linestyle='--', alpha=0.7, label='Oversold')
            ax2.axhline(overbought_threshold, color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)

            # Shade zones
            ax2.fill_between(x_axis, df_plot[osc_col], 0,
                            where=(df_plot[osc_col] < oversold_threshold),
                            color='lime', alpha=0.3)
            ax2.fill_between(x_axis, df_plot[osc_col], 0,
                            where=(df_plot[osc_col] > overbought_threshold),
                            color='red', alpha=0.3)

            ax2.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white', fontsize=7)

        ax2.set_ylabel("JD_Osc", color='white', fontsize=9)
        ax2.set_ylim(-1.2, 1.2)
        _format_xaxis_dates(ax2, df_plot, datetime_col, is_intraday)

        # === Panel 3: JD_Signal and JD_Trend (as lines, NOT bars) ===
        ax3 = axes[2]
        vel_col = col_map.get('velocity')
        acc_col = col_map.get('acceleration')

        if vel_col and vel_col in df_plot.columns:
            ax3.plot(x_axis, df_plot[vel_col], color='#00d9ff', linewidth=1.2, label='JD_Signal')

        if acc_col and acc_col in df_plot.columns:
            ax3.plot(x_axis, df_plot[acc_col], color='#ffd700', linewidth=1.0, alpha=0.7, label='JD_Trend')

        ax3.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax3.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white', fontsize=7)
        ax3.set_ylabel("JD_Signal", color='white', fontsize=9)
        _format_xaxis_dates(ax3, df_plot, datetime_col, is_intraday)

        # === Panel 4: Equity Curve (in $ with dates on x-axis) ===
        # CRITICAL: Only show trades within the chart's visible period
        ax4 = axes[3]
        _draw_equity_curve(ax4, exits_for_equity, entries_for_equity, chart_start)

        ax4.set_ylabel("Equity ($)", color='white', fontsize=9)
        ax4.set_xlabel("Date", color='white', fontsize=9)

        # Stats annotation at bottom: ALL-TIME stats + CHART PERIOD stats (including drawdown)
        # Add gap risk note for daily charts where drawdown exceeds typical stop loss
        dd_note = "*" if not is_intraday and all_time_stats['max_drawdown'] > 5 else ""
        gap_note = "  (*DD includes gap risk)" if dd_note else ""

        stats_text = (
            f"All-Time: {all_time_stats['num_trades']} trades | "
            f"{all_time_stats['win_rate']:.0f}% WR | "
            f"{all_time_stats['total_return']:.1f}% Return | "
            f"DD {all_time_stats['max_drawdown']:.1f}%{dd_note}    |    "
            f"Chart Period: {visible_stats['num_trades']} trades | "
            f"{visible_stats['win_rate']:.0f}% WR | "
            f"{visible_stats['total_return']:.1f}% Return | "
            f"DD {visible_stats['max_drawdown']:.1f}%{gap_note}"
        )
        fig.text(0.5, 0.02, stats_text, ha='center', color='white', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.8))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)

        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, facecolor=fig.get_facecolor(),
                   edgecolor='none', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf

    except Exception as e:
        print(f"   Error generating chart: {e}")
        import traceback
        traceback.print_exc()
        return None


def _get_column_map(df: pd.DataFrame) -> Dict[str, str]:
    """Build a mapping of lowercase column names to actual column names."""
    col_map = {}
    for col in df.columns:
        # Handle tuple columns (MultiIndex) - extract first element
        col_name = col[0] if isinstance(col, tuple) else col
        col_lower = str(col_name).lower()
        if col_lower == 'close':
            col_map['close'] = col
        elif col_lower == 'open':
            col_map['open'] = col
        elif col_lower == 'high':
            col_map['high'] = col
        elif col_lower == 'low':
            col_map['low'] = col
        elif col_lower == 'osc_smooth':
            col_map['osc_smooth'] = col
        elif col_lower == 'composite_smooth':
            col_map['composite_smooth'] = col
        elif col_lower == 'velocity':
            col_map['velocity'] = col
        elif col_lower == 'acceleration':
            col_map['acceleration'] = col
    return col_map


def _find_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """Find the datetime column in the dataframe."""
    for col in ['timestamp', 'Timestamp', 'index', 'date', 'datetime', 'Date', 'Datetime', 'time', 'Time']:
        if col in df.columns:
            return col
    # Also check for columns with 'date' or 'time' in name
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            return col
    return None


def _build_date_lookup(df: pd.DataFrame, datetime_col: Optional[str]) -> Dict:
    """Build a lookup from dates to bar numbers.

    CRITICAL: For intraday charts, we need minute-level precision to prevent
    multiple trades on the same day from overlapping.

    Normalizes all dates to UTC-naive for consistent matching with trade dates.
    """
    date_to_barnum = {}
    # Store first bar per date for daily fallback (don't overwrite with later bars)
    date_only_lookup = {}

    if datetime_col and datetime_col in df.columns:
        for _, row in df.iterrows():
            try:
                dt = pd.to_datetime(row[datetime_col])
                if pd.notna(dt):
                    # Normalize to UTC timezone-naive
                    dt = _normalize_to_utc_naive(dt)

                    bar_num = row['bar_num']

                    # PRIMARY: Full datetime (highest priority)
                    date_to_barnum[dt] = bar_num

                    # Without microseconds
                    dt_no_micro = dt.replace(microsecond=0)
                    date_to_barnum[dt_no_micro] = bar_num

                    # Minute-precision key (CRITICAL for intraday)
                    dt_minute = dt.replace(second=0, microsecond=0)
                    date_to_barnum[dt_minute] = bar_num

                    # String formats for fallback matching (minute precision)
                    date_to_barnum[str(dt)[:19]] = bar_num  # YYYY-MM-DD HH:MM:SS
                    date_to_barnum[str(dt)[:16]] = bar_num  # YYYY-MM-DD HH:MM

                    # ISO format variants (T separator vs space)
                    iso_str = dt.strftime('%Y-%m-%dT%H:%M:%S')
                    space_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    date_to_barnum[iso_str] = bar_num
                    date_to_barnum[space_str] = bar_num
                    date_to_barnum[iso_str[:16]] = bar_num  # YYYY-MM-DDTHH:MM
                    date_to_barnum[space_str[:16]] = bar_num  # YYYY-MM-DD HH:MM

                    # Date only - ONLY store FIRST bar of each date (for daily chart fallback)
                    # This prevents later intraday bars from overwriting earlier ones
                    date_only_str = str(dt)[:10]
                    if date_only_str not in date_only_lookup:
                        date_only_lookup[date_only_str] = bar_num
                        date_only = dt.replace(hour=0, minute=0, second=0, microsecond=0)
                        date_to_barnum[date_only] = bar_num
                        date_to_barnum[date_only_str] = bar_num
            except Exception:
                pass

    # Add date-only lookups (from first bar of each date)
    date_to_barnum.update(date_only_lookup)

    return date_to_barnum


def _get_chart_date_range(df: pd.DataFrame, datetime_col: Optional[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get the date range of the chart data."""
    if datetime_col and datetime_col in df.columns and len(df) > 0:
        try:
            start_dt = pd.to_datetime(df[datetime_col].iloc[0])
            end_dt = pd.to_datetime(df[datetime_col].iloc[-1])
            return start_dt, end_dt
        except Exception:
            pass
    return None, None


def _filter_to_date_range(trades: List[Dict], start_dt, end_dt, extend_days: int = 3) -> List[Dict]:
    """Filter trades to those within the date range.

    Normalizes all dates to UTC-naive for consistent comparison.
    Extends the start date by extend_days to include recent trades that might
    appear at the edge of the chart.

    Args:
        trades: List of trade dicts with 'date' key
        start_dt: Chart start datetime
        end_dt: Chart end datetime
        extend_days: Number of days to extend before start_dt
    """
    if start_dt is None or end_dt is None:
        return trades

    # Normalize start/end to UTC-naive
    start_dt = pd.to_datetime(start_dt)
    end_dt = pd.to_datetime(end_dt)
    start_dt = _normalize_to_utc_naive(start_dt)
    end_dt = _normalize_to_utc_naive(end_dt)

    # Extend start date to include recent trades just before chart range
    # This ensures trades near the chart boundary still show up
    extended_start = start_dt - pd.Timedelta(days=extend_days)

    filtered = []
    for trade in trades:
        try:
            trade_dt = pd.to_datetime(trade.get('date'))
            if trade_dt is None:
                continue
            # Normalize trade date to UTC-naive
            trade_dt = _normalize_to_utc_naive(trade_dt)
            if extended_start <= trade_dt <= end_dt:
                filtered.append(trade)
        except Exception:
            pass
    return filtered


def _calculate_visible_stats(exits: List[Dict]) -> Dict:
    """Calculate stats for visible exits including max drawdown."""
    if not exits:
        return {'num_trades': 0, 'win_rate': 0, 'total_return': 0, 'profit_factor': 0, 'max_drawdown': 0}

    num_trades = len(exits)
    wins = sum(1 for e in exits if e.get('pnl', 0) > 0)
    win_rate = (wins / num_trades * 100) if num_trades > 0 else 0

    # Use COMPOUND returns (matches database calculation)
    # Each pnl is a percentage (e.g., 1.5 means 1.5% gain)
    equity = 1.0
    for e in exits:
        pnl_pct = e.get('pnl', 0)
        equity *= (1 + pnl_pct / 100)
    total_return = (equity - 1) * 100

    gross_profit = sum(e.get('pnl', 0) for e in exits if e.get('pnl', 0) > 0)
    gross_loss = abs(sum(e.get('pnl', 0) for e in exits if e.get('pnl', 0) < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.99

    # Calculate max drawdown from equity curve
    max_drawdown = _calculate_max_drawdown(exits)

    return {
        'num_trades': num_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'profit_factor': min(profit_factor, 999.99),
        'max_drawdown': max_drawdown
    }


def _calculate_max_drawdown(exits: List[Dict]) -> float:
    """
    Calculate maximum drawdown percentage from a list of exits.

    Drawdown = (Peak - Trough) / Peak * 100
    """
    if not exits:
        return 0.0

    # Build equity curve
    equity = [STARTING_CAPITAL]
    for exit_trade in exits:
        pnl = exit_trade.get('pnl', 0)
        equity.append(equity[-1] * (1 + pnl / 100))

    # Calculate drawdown at each point
    peak = equity[0]
    max_dd = 0.0

    for value in equity:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100 if peak > 0 else 0
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd


def _build_chart_title(ticker: str, title_suffix: str, df: pd.DataFrame,
                       datetime_col: Optional[str], is_intraday: bool) -> str:
    """Build the chart title with date range."""
    title = f"{ticker} - JD Strategy"

    if title_suffix:
        title += title_suffix

    # Add date range
    if datetime_col and datetime_col in df.columns and len(df) > 0:
        try:
            start_dt = pd.to_datetime(df[datetime_col].iloc[0])
            end_dt = pd.to_datetime(df[datetime_col].iloc[-1])
            if not pd.isna(start_dt) and not pd.isna(end_dt):
                if is_intraday:
                    date_range = f" ({start_dt.strftime('%m/%d')} - {end_dt.strftime('%m/%d %H:%M')})"
                else:
                    date_range = f" ({start_dt.strftime('%Y-%m-%d')} - {end_dt.strftime('%Y-%m-%d')})"
                title += date_range
        except Exception:
            pass

    return title


def _draw_candlesticks(ax, df: pd.DataFrame, col_map: Dict):
    """Draw candlestick chart."""
    close_col = col_map['close']
    open_col = col_map.get('open', close_col)
    high_col = col_map.get('high', close_col)
    low_col = col_map.get('low', close_col)

    # Calculate minimum visible body height (1.5% of price range)
    # This ensures even doji candles have a clearly visible body
    # 0.3% was too small - rendered as ~1 pixel on screen
    price_range = df[high_col].max() - df[low_col].min()
    min_body_height = price_range * 0.015  # 1.5% of visible range

    for _, row in df.iterrows():
        bar_num = row['bar_num']
        o = row[open_col]
        h = row[high_col]
        l = row[low_col]
        c = row[close_col]

        color = '#26a69a' if c >= o else '#ef5350'  # Green/Red

        # Wick
        ax.plot([bar_num, bar_num], [l, h], color=color, linewidth=0.8)

        # Body - use minimum height for visibility
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < min_body_height:
            # Center the minimum body around the midpoint
            body_center = (o + c) / 2
            body_bottom = body_center - min_body_height / 2
            body_height = min_body_height

        ax.add_patch(plt.Rectangle(
            (bar_num - 0.35, body_bottom),
            0.7, body_height,
            facecolor=color, edgecolor=color, linewidth=0.5
        ))


def _draw_entry_exit_markers(ax, entries: List[Dict], exits: List[Dict], date_to_barnum: Dict,
                              df_plot: pd.DataFrame = None, col_map: Dict = None,
                              is_intraday: bool = True):
    """Draw entry and exit markers on the chart.

    Entry = green triangle UP (▲) positioned BELOW the bar low (charting convention)
    Exit = RED triangle DOWN (▼) positioned ABOVE the bar high (charting convention)

    This matches the legacy velocity_live_trader.py behavior where markers are
    placed at the candle edges, not at the trade price.

    Args:
        is_intraday: If True, don't use date-only fallback for bar matching
                     (prevents trades outside chart range from appearing on wrong bars)
    """
    # Calculate offset for markers (2% of price range)
    if df_plot is not None and col_map is not None:
        high_col = col_map.get('high', col_map.get('close'))
        low_col = col_map.get('low', col_map.get('close'))
        price_range = df_plot[high_col].max() - df_plot[low_col].min()
        marker_offset = price_range * 0.02
    else:
        marker_offset = 0

    # Build barnum to row lookup for finding highs/lows
    barnum_to_row = {}
    if df_plot is not None:
        for _, row in df_plot.iterrows():
            barnum_to_row[row['bar_num']] = row

    # Dynamic marker size based on trade density
    # Minimum size of 50 to keep markers visible even with many trades
    num_markers = len(entries) + len(exits)
    if num_markers > 500:
        marker_size = 50
    elif num_markers > 200:
        marker_size = 60
    elif num_markers > 100:
        marker_size = 80
    else:
        marker_size = 100

    # Track matched/unmatched for debugging
    matched_entries = 0
    matched_exits = 0

    # Entry markers (green triangles pointing up, BELOW bar low)
    for entry in entries:
        bar_num = _find_bar_num(entry.get('date'), date_to_barnum, is_intraday=is_intraday)
        if bar_num is not None:
            matched_entries += 1
            # Get the bar's low price to position marker below it
            if bar_num in barnum_to_row and col_map is not None:
                low_col = col_map.get('low', col_map.get('close'))
                bar_low = barnum_to_row[bar_num][low_col]
                y_pos = bar_low - marker_offset
            else:
                # Fallback to trade price if can't find bar
                y_pos = entry.get('price')

            ax.scatter(bar_num, y_pos, marker='^', s=marker_size, c='#00e676',
                      edgecolors='white', linewidths=0.5, zorder=10)

    # Exit markers (RED triangles pointing down, ABOVE bar high)
    for exit_trade in exits:
        bar_num = _find_bar_num(exit_trade.get('date'), date_to_barnum, is_intraday=is_intraday)
        if bar_num is not None:
            matched_exits += 1
            # Get the bar's high price to position marker above it
            if bar_num in barnum_to_row and col_map is not None:
                high_col = col_map.get('high', col_map.get('close'))
                bar_high = barnum_to_row[bar_num][high_col]
                y_pos = bar_high + marker_offset
            else:
                # Fallback to trade price if can't find bar
                y_pos = exit_trade.get('price')

            ax.scatter(bar_num, y_pos, marker='v', s=marker_size, c='#ff5252',
                      edgecolors='white', linewidths=0.5, zorder=10)

    # Debug: Show how many markers were matched
    if len(entries) > 0 or len(exits) > 0:
        total_entries = len(entries)
        total_exits = len(exits)
        if matched_entries < total_entries or matched_exits < total_exits:
            print(f"   Chart markers: {matched_entries}/{total_entries} entries, {matched_exits}/{total_exits} exits (some outside chart range)")


def _normalize_to_utc_naive(dt) -> pd.Timestamp:
    """Convert any datetime to UTC timezone-naive for consistent comparison.

    Trade dates come in various formats:
    - '2026-01-22 07:30:00-0600' (CST with offset)
    - '2026-01-22 07:30:00+00:00' (UTC with offset)
    - '2026-01-22 07:30:00' (naive, assumed UTC)

    Chart data is typically in UTC. We need to convert all to UTC-naive.
    """
    if dt.tzinfo is not None:
        # Convert to UTC first, then strip timezone
        dt = dt.tz_convert('UTC').tz_localize(None)
    return dt


def _find_bar_num(date_str, date_to_barnum: Dict, is_intraday: bool = True) -> Optional[int]:
    """Find bar number for a date string.

    CRITICAL: Handles mixed timestamp formats by trying multiple lookup keys.
    Order matters: try most specific (minute precision) first.

    For INTRADAY charts: do NOT use date-only fallback (causes markers from
    trades outside the chart range to appear on wrong bars).

    For DAILY charts: use date-only fallback (trade times don't match bar times).

    Handles these trade date formats:
    - '2026-01-23 11:15:00' (space separator)
    - '2026-01-23T11:15:00' (ISO with T)
    - '2026-01-23 11:15:00+00:00' (with TZ)
    - '2026-01-23T11:15:00-06:00' (CST offset)
    """
    if date_str is None:
        return None

    # Direct lookup first (handles pre-formatted keys)
    if date_str in date_to_barnum:
        return date_to_barnum[date_str]

    try:
        dt = pd.to_datetime(date_str)

        # Normalize to UTC timezone-naive for comparison
        dt = _normalize_to_utc_naive(dt)

        # Try full datetime
        if dt in date_to_barnum:
            return date_to_barnum[dt]

        # Try without microseconds
        dt_no_micro = dt.replace(microsecond=0)
        if dt_no_micro in date_to_barnum:
            return date_to_barnum[dt_no_micro]

        # Try minute precision (critical for intraday)
        dt_minute = dt.replace(second=0, microsecond=0)
        if dt_minute in date_to_barnum:
            return date_to_barnum[dt_minute]

        # Try string formats - both ISO (T) and space separators
        iso_str = dt.strftime('%Y-%m-%dT%H:%M:%S')
        space_str = dt.strftime('%Y-%m-%d %H:%M:%S')

        for fmt in [
            iso_str,                    # 2026-01-23T11:15:00
            space_str,                  # 2026-01-23 11:15:00
            iso_str[:16],               # 2026-01-23T11:15
            space_str[:16],             # 2026-01-23 11:15
            str(dt)[:19],               # pandas default format
            str(dt)[:16],               # truncated
        ]:
            if fmt in date_to_barnum:
                return date_to_barnum[fmt]

        # CRITICAL FIX: Trade dates use bar CLOSE time, but DataFrame index uses bar START time
        # For intraday, try subtracting common intervals to convert close -> start time
        # E.g., exit at 18:30 should map to bar starting at 18:15 (for 15m bars)
        if is_intraday:
            for minutes in [15, 30, 60, 5, 1]:  # Try common intervals
                dt_start = dt_minute - pd.Timedelta(minutes=minutes)
                if dt_start in date_to_barnum:
                    return date_to_barnum[dt_start]
                # Also try string formats for the adjusted time
                start_iso = dt_start.strftime('%Y-%m-%dT%H:%M:%S')
                start_space = dt_start.strftime('%Y-%m-%d %H:%M:%S')
                for fmt in [dt_start, start_iso, start_space, start_iso[:16], start_space[:16]]:
                    if fmt in date_to_barnum:
                        return date_to_barnum[fmt]

        # Date-only fallback: ONLY for daily charts
        # For intraday, this causes trades outside chart range to map to wrong bars
        if not is_intraday:
            date_only_str = dt.strftime('%Y-%m-%d')
            if date_only_str in date_to_barnum:
                return date_to_barnum[date_only_str]

            date_only = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            if date_only in date_to_barnum:
                return date_to_barnum[date_only]

    except Exception:
        pass

    return None


def _draw_equity_curve(ax, exits: List[Dict], entries: List[Dict], chart_start):
    """Draw equity curve with dates on x-axis."""
    if not exits:
        ax.text(0.5, 0.5, 'No trades in period', transform=ax.transAxes,
               ha='center', va='center', color='white', fontsize=10)
        return

    # Build equity curve
    equity = [STARTING_CAPITAL]
    trade_dates = []

    # Start date from first entry or exit
    def normalize_date(date_str):
        """Normalize date to UTC timezone-naive for matplotlib."""
        dt = pd.to_datetime(date_str)
        if dt.tzinfo is not None:
            # Convert to UTC, then strip timezone for consistent comparison
            dt = dt.tz_convert('UTC').tz_localize(None)
        return dt

    if entries:
        try:
            first_date = normalize_date(entries[0].get('date'))
            trade_dates.append(first_date)
        except Exception:
            if exits:
                trade_dates.append(normalize_date(exits[0].get('date')))
    elif exits:
        trade_dates.append(normalize_date(exits[0].get('date')))

    # Add each exit
    for exit_trade in exits:
        try:
            pnl = exit_trade.get('pnl', 0)
            equity.append(equity[-1] * (1 + pnl / 100))
            trade_dates.append(normalize_date(exit_trade.get('date')))
        except Exception:
            pass

    if len(trade_dates) != len(equity):
        # Fallback to trade numbers if date mismatch
        ax.plot(range(len(equity)), equity, color='#00ff88', linewidth=2)
        ax.fill_between(range(len(equity)), STARTING_CAPITAL, equity, alpha=0.3,
                       color='green' if equity[-1] > STARTING_CAPITAL else 'red')
    else:
        ax.plot(trade_dates, equity, color='#00ff88', linewidth=2)
        ax.fill_between(trade_dates, STARTING_CAPITAL, equity, alpha=0.3,
                       color='green' if equity[-1] > STARTING_CAPITAL else 'red')

        # Format x-axis dates
        if len(trade_dates) >= 2:
            date_range = (trade_dates[-1] - trade_dates[0]).days
            if date_range > 365:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            elif date_range > 60:
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            else:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.axhline(STARTING_CAPITAL, color='gray', linestyle='--', alpha=0.5)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))


def _format_xaxis_dates(ax, df: pd.DataFrame, datetime_col: Optional[str], is_intraday: bool):
    """Format x-axis with datetime labels."""
    if datetime_col and datetime_col in df.columns:
        barnum_to_dt = dict(zip(df['bar_num'], df[datetime_col]))

        def format_tick(x, pos):
            if x in barnum_to_dt:
                dt = barnum_to_dt[x]
                try:
                    dt_parsed = pd.to_datetime(dt)
                    if pd.isna(dt_parsed):
                        return ''
                    if is_intraday:
                        return dt_parsed.strftime('%m/%d\n%H:%M')
                    else:
                        return dt_parsed.strftime('%b')
                except Exception:
                    return ''
            return ''

        ax.xaxis.set_major_formatter(FuncFormatter(format_tick))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10 if not is_intraday else 8, integer=True))
        ax.tick_params(axis='x', labelsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')


# Alias for backward compatibility
generate_velocity_chart = generate_chart
generate_signal_chart = generate_chart
