"""
Velocity-based signal generation for velocity trading.

Generates buy/sell signals based on oscillator velocity, acceleration,
and zone transitions.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path for legacy imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Try to import old system's signal calculation for exact matching
try:
    from velocity_live_trader import calculate_velocity_signals as old_calculate_velocity_signals
    HAS_OLD_SIGNALS = True
except ImportError:
    HAS_OLD_SIGNALS = False


class SignalType(Enum):
    """Types of velocity signals."""
    ZONE_ENTRY = "zone_entry"           # Enter oversold/overbought zone
    ZONE_EXIT = "zone_exit"             # Exit zone (mean reversion)
    VELOCITY_CROSS = "velocity_cross"   # Velocity crosses zero
    REVERSAL = "reversal"               # Zone + velocity reversal
    MIDLINE_CROSS = "midline_cross"     # Cross zero line


@dataclass
class Signal:
    """A trading signal."""
    time: pd.Timestamp
    signal_type: SignalType
    direction: str  # 'buy' or 'sell'
    price: float
    oscillator_value: float
    velocity: float
    confidence: float  # 0 to 1


def calculate_velocity_signals(
    df: pd.DataFrame,
    signal_type: str = 'any_reversal',
    oversold_threshold: float = -0.3,
    overbought_threshold: float = 0.3,
    velocity_threshold: float = 0.0,
    lookback: int = 1,
    vel_smoothing: int = 3,
    extreme_zone_mult: float = 1.5,
    require_accel: bool = True,
    use_regime_filter: bool = False,
    regime_threshold: float = -0.15,
    use_fragility_filter: bool = False,
    fragility_threshold: float = 0.5,
    use_entropy_filter: bool = False,
    entropy_threshold: float = 0.7,
    use_vol_regime_filter: bool = False,
    vol_regime_percentile_threshold: float = 0.25,
    rsi_filter: str = 'none',
    rsi_period: int = 14,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    use_macd_confirm: bool = False,
    use_bb_filter: bool = False,
    use_mfv_filter: bool = False,
    mfv_mode: str = 'velocity',
    mfv_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Generate buy/sell signals based on velocity strategy.

    This matches the exact calculation from velocity_live_trader.py
    to ensure identical signals with or without legacy mode.

    Args:
        df: DataFrame with osc_smooth, velocity, acceleration columns
        signal_type: Strategy type - 'any_reversal', 'zone_exit', 'velocity_cross', etc.
        oversold_threshold: Threshold for oversold zone
        overbought_threshold: Threshold for overbought zone
        velocity_threshold: Minimum velocity for signal confirmation
        lookback: Number of bars for signal confirmation
        vel_smoothing: Smoothing period for oscillator before velocity calculation
        extreme_zone_mult: Multiplier for extreme zone thresholds
        require_accel: Whether to require acceleration confirmation
        use_regime_filter: Apply RSC regime filter
        regime_threshold: RSC threshold (signals blocked when RSC < threshold)
        use_fragility_filter: Apply MFI2 fragility filter
        fragility_threshold: MFI2 threshold (signals blocked when MFI2 > threshold)
        use_entropy_filter: Apply SEI entropy filter
        entropy_threshold: SEI threshold (signals blocked when SEI > threshold)

    Returns:
        DataFrame with additional columns:
        - buy_signal: True on buy signal bars
        - sell_signal: True on sell signal bars
        - signal_strength: Signal confidence (0-1)
    """
    result = df.copy()

    # Initialize signal columns
    result['buy_signal'] = False
    result['sell_signal'] = False
    result['signal_strength'] = 0.0

    # Get oscillator column (matching old system logic)
    osc_col = 'composite_smooth' if 'composite_smooth' in result.columns else 'osc_smooth'
    if osc_col not in result.columns:
        osc_col = 'composite_oscillator' if 'composite_oscillator' in result.columns else None
    if osc_col is None:
        return result

    # Apply smoothing (matching old system)
    if vel_smoothing > 1:
        result['osc_smooth'] = result[osc_col].rolling(window=vel_smoothing, center=False).mean()
        result['osc_smooth'] = result['osc_smooth'].bfill()
    else:
        result['osc_smooth'] = result[osc_col]

    # Calculate velocity (first derivative), acceleration (second derivative), and jerk (third derivative)
    result['velocity'] = result['osc_smooth'].diff()
    result['acceleration'] = result['velocity'].diff()
    result['jerk'] = result['acceleration'].diff()

    # Fill NaN values
    result['velocity'] = result['velocity'].fillna(0)
    result['acceleration'] = result['acceleration'].fillna(0)
    result['jerk'] = result['jerk'].fillna(0)

    osc = result['osc_smooth']
    vel = result['velocity']
    acc = result['acceleration']

    # Detect velocity zero-crossings (matching old system)
    result['vel_cross_up'] = (vel > 0) & (vel.shift(1) <= 0)
    result['vel_cross_down'] = (vel < 0) & (vel.shift(1) >= 0)

    # Zone conditions (matching old system)
    in_oversold = osc < oversold_threshold
    in_overbought = osc > overbought_threshold
    extreme_oversold = osc < (oversold_threshold * extreme_zone_mult)
    extreme_overbought = osc > (overbought_threshold * extreme_zone_mult)

    # Strong momentum detection (matching old system)
    vel_std = vel.rolling(10, min_periods=1).std().fillna(vel.std())
    strong_momentum_up = vel > vel_std * 1.5
    strong_momentum_down = vel < -vel_std * 1.5

    # Previous values for zone transitions
    prev_osc = osc.shift(1)
    was_oversold = prev_osc < oversold_threshold
    was_overbought = prev_osc > overbought_threshold

    # Generate signals based on strategy type (matching old system exactly)
    if signal_type == 'velocity_crossover_and_zone':
        result['buy_signal'] = result['vel_cross_up'] & in_oversold
        result['sell_signal'] = result['vel_cross_down'] & in_overbought

    elif signal_type == 'velocity_crossover_or_zone':
        result['buy_signal'] = result['vel_cross_up'] | extreme_oversold
        result['sell_signal'] = result['vel_cross_down'] | extreme_overbought

    elif signal_type == 'zone_only':
        result['buy_signal'] = extreme_oversold & (vel > 0)
        result['sell_signal'] = extreme_overbought & (vel < 0)

    elif signal_type == 'momentum':
        result['buy_signal'] = strong_momentum_up & (osc < 0)
        result['sell_signal'] = strong_momentum_down & (osc > 0)

    elif signal_type == 'any_reversal':
        # Most aggressive: velocity crossover OR extreme zone OR strong momentum in zone
        result['buy_signal'] = result['vel_cross_up'] | extreme_oversold | (strong_momentum_up & in_oversold)
        result['sell_signal'] = result['vel_cross_down'] | extreme_overbought | (strong_momentum_down & in_overbought)

    elif signal_type == 'double_bottom':
        vel_cross_up_count = result['vel_cross_up'].rolling(10).sum()
        result['buy_signal'] = (vel_cross_up_count >= 2) & in_oversold
        vel_cross_down_count = result['vel_cross_down'].rolling(10).sum()
        result['sell_signal'] = (vel_cross_down_count >= 2) & in_overbought

    elif signal_type == 'breakout':
        osc_breaks_above = (osc > oversold_threshold) & (prev_osc <= oversold_threshold)
        osc_breaks_below = (osc < overbought_threshold) & (prev_osc >= overbought_threshold)
        result['buy_signal'] = osc_breaks_above
        result['sell_signal'] = osc_breaks_below

    elif signal_type == 'zone_exit':
        result['buy_signal'] = was_oversold & ~in_oversold
        result['sell_signal'] = was_overbought & ~in_overbought

    elif signal_type == 'velocity_cross':
        result['buy_signal'] = result['vel_cross_up']
        result['sell_signal'] = result['vel_cross_down']

    elif signal_type == 'zone_velocity':
        result['buy_signal'] = in_oversold & (vel > velocity_threshold)
        result['sell_signal'] = in_overbought & (vel < -velocity_threshold)

    elif signal_type == 'midline_cross':
        result['buy_signal'] = (prev_osc < 0) & (osc >= 0)
        result['sell_signal'] = (prev_osc > 0) & (osc <= 0)

    else:
        # Default: velocity_crossover_and_zone (matching old system default)
        result['buy_signal'] = result['vel_cross_up'] & in_oversold
        result['sell_signal'] = result['vel_cross_down'] & in_overbought

    # Apply acceleration filter (matching old system: require_accel)
    if require_accel:
        buy_accel_cond = acc > 0
        sell_accel_cond = acc < 0
        result['buy_signal'] = result['buy_signal'] & buy_accel_cond
        result['sell_signal'] = result['sell_signal'] & sell_accel_cond

    # RSI filter (matches optuna_worker.py logic)
    if rsi_filter != 'none' and 'RSI' in result.columns:
        rsi = result['RSI']
        if rsi_filter == 'oversold_only':
            result['buy_signal'] = result['buy_signal'] & (rsi < rsi_oversold)
        elif rsi_filter == 'overbought_only':
            result['sell_signal'] = result['sell_signal'] & (rsi > rsi_overbought)
        elif rsi_filter == 'both':
            result['buy_signal'] = result['buy_signal'] & (rsi < rsi_oversold)
            result['sell_signal'] = result['sell_signal'] & (rsi > rsi_overbought)

    # MACD confirmation filter
    if use_macd_confirm and 'MACD_histogram' in result.columns:
        macd_h = result['MACD_histogram']
        macd_improving = macd_h > macd_h.shift(1)
        macd_declining = macd_h < macd_h.shift(1)
        result['buy_signal'] = result['buy_signal'] & macd_improving
        result['sell_signal'] = result['sell_signal'] & macd_declining

    # Bollinger Band filter
    if use_bb_filter and 'BB_lower' in result.columns:
        _close_col = 'close' if 'close' in result.columns else 'Close'
        result['buy_signal'] = result['buy_signal'] & (result[_close_col] < result['BB_lower'])
        result['sell_signal'] = result['sell_signal'] & (result[_close_col] > result['BB_upper'])

    # Apply V2 filters (Regime, Fragility, Entropy)
    # These filters are calculated in oscillators.py and stored in the dataframe

    # Regime filter: Block signals when market regime is unfavorable
    if use_regime_filter and 'RSC' in result.columns:
        regime_ok = result['RSC'] > regime_threshold
        result['buy_signal'] = result['buy_signal'] & regime_ok
        result['sell_signal'] = result['sell_signal'] & regime_ok

    # Fragility filter: Block signals when market is fragile (high MFI2)
    if use_fragility_filter and 'MFI2' in result.columns:
        fragility_ok = result['MFI2'] < fragility_threshold
        result['buy_signal'] = result['buy_signal'] & fragility_ok
        result['sell_signal'] = result['sell_signal'] & fragility_ok

    # Entropy filter: Block signals when market is chaotic (high SEI)
    if use_entropy_filter and 'SEI' in result.columns:
        entropy_ok = result['SEI'] < entropy_threshold
        result['buy_signal'] = result['buy_signal'] & entropy_ok
        result['sell_signal'] = result['sell_signal'] & entropy_ok

    # Volatility regime filter: Block signals in low-volatility periods
    if use_vol_regime_filter and 'VOL_REGIME' in result.columns:
        vol_ok = result['VOL_REGIME'] > vol_regime_percentile_threshold
        result['buy_signal'] = result['buy_signal'] & vol_ok

    # Money Flow Velocity filter: require money flow confirmation for signals
    if use_mfv_filter:
        if mfv_mode == 'velocity' and 'MFV_VEL' in result.columns:
            mfv_v = result['MFV_VEL']
            result['buy_signal'] = result['buy_signal'] & (mfv_v > mfv_threshold)
            result['sell_signal'] = result['sell_signal'] & (mfv_v < -mfv_threshold)
        elif mfv_mode == 'flow' and 'MFV_FLOW' in result.columns:
            mfv_f = result['MFV_FLOW']
            result['buy_signal'] = result['buy_signal'] & (mfv_f > mfv_threshold)
            result['sell_signal'] = result['sell_signal'] & (mfv_f < -mfv_threshold)
        elif mfv_mode == 'both' and 'MFV_FLOW' in result.columns and 'MFV_VEL' in result.columns:
            mfv_f = result['MFV_FLOW']
            mfv_v = result['MFV_VEL']
            result['buy_signal'] = result['buy_signal'] & (mfv_f > mfv_threshold) & (mfv_v > mfv_threshold)
            result['sell_signal'] = result['sell_signal'] & (mfv_f < -mfv_threshold) & (mfv_v < -mfv_threshold)

    # Calculate signal strength based on zone depth
    result['signal_strength'] = 0.5  # Default
    result.loc[result['buy_signal'], 'signal_strength'] = \
        (oversold_threshold - osc[result['buy_signal']]).abs().clip(0, 1)
    result.loc[result['sell_signal'], 'signal_strength'] = \
        (osc[result['sell_signal']] - overbought_threshold).abs().clip(0, 1)

    return result


def calculate_velocity_signals_legacy(
    df: pd.DataFrame,
    config: dict = None
) -> pd.DataFrame:
    """
    Calculate velocity signals using the OLD system's exact calculation.

    This ensures 100% matching with velocity_live_trader.py for verification.
    Uses calculate_velocity_signals from velocity_live_trader.py.
    """
    if not HAS_OLD_SIGNALS:
        print("Warning: Old signal calculation not available, using new calculation")
        config = config or {}
        return calculate_velocity_signals(
            df,
            signal_type=config.get('signal_type', 'any_reversal'),
            oversold_threshold=config.get('oversold_threshold', -0.3),
            overbought_threshold=config.get('overbought_threshold', 0.3)
        )

    # Use old system's calculation
    config = config or {
        'signal_type': 'any_reversal',
        'oversold_threshold': -0.3,
        'overbought_threshold': 0.3,
        'require_accel': True,
        'vel_smoothing': 3,
        'extreme_zone_mult': 1.5,
    }

    result = df.copy()
    # Ensure column names are lowercase (old system expects this)
    result.columns = result.columns.str.lower()

    result = old_calculate_velocity_signals(result, config)

    # Restore original column case for OHLCV
    result.columns = [c.title() if c in ['open', 'high', 'low', 'close', 'volume'] else c for c in result.columns]

    return result


def get_recent_signals(
    df: pd.DataFrame,
    lookback: int = 3,
    signal_type: str = 'buy'
) -> List[Dict]:
    """
    Get recent signals from a DataFrame.

    Args:
        df: DataFrame with signal columns
        lookback: Number of bars to check
        signal_type: 'buy', 'sell', or 'both'

    Returns:
        List of signal dictionaries with time, price, strength
    """
    signals = []
    recent = df.iloc[-lookback:] if len(df) >= lookback else df

    col = 'buy_signal' if signal_type == 'buy' else 'sell_signal'
    if signal_type == 'both':
        cols = ['buy_signal', 'sell_signal']
    else:
        cols = [col]

    for col in cols:
        if col not in recent.columns:
            continue

        for idx, row in recent[recent[col] == True].iterrows():
            signals.append({
                'time': idx,
                'type': 'buy' if 'buy' in col else 'sell',
                'price': row.get('Close', 0),
                'strength': row.get('signal_strength', 0.5),
                'oscillator': row.get('osc_smooth', 0),
                'velocity': row.get('velocity', 0)
            })

    return sorted(signals, key=lambda x: x['time'], reverse=True)


def find_most_recent_signal(
    df: pd.DataFrame,
    signal_type: str = 'buy',
    max_bars_ago: int = 5
) -> Optional[Dict]:
    """
    Find the most recent signal within max_bars_ago.

    Returns:
        Signal dict or None if no recent signal
    """
    signals = get_recent_signals(df, lookback=max_bars_ago, signal_type=signal_type)
    return signals[0] if signals else None


def check_acceleration_exit(
    df: pd.DataFrame,
    position_type: str,
    current_pnl: float,
    accel_exit_type: str = 'sign_reversal',
    accel_exit_threshold: float = 0.0,
    accel_exit_min_pnl: float = 0.5,
    accel_exit_lookback: int = 1,
    use_jerk_confirm: bool = False,
    jerk_confirm_threshold: float = 0.0
) -> Tuple[bool, str]:
    """
    Check if acceleration-based exit conditions are met.

    This exit triggers when momentum is reversing against the position,
    allowing early exit before stop loss is hit.

    Args:
        df: DataFrame with acceleration and jerk columns
        position_type: 'long' or 'short'
        current_pnl: Current P&L percentage
        accel_exit_type: 'sign_reversal', 'magnitude', or 'both'
        accel_exit_threshold: Min acceleration magnitude for exit
        accel_exit_min_pnl: Min P&L before accel exit allowed (or negative P&L)
        accel_exit_lookback: Bars of consecutive adverse acceleration
        use_jerk_confirm: Require jerk confirmation
        jerk_confirm_threshold: Jerk threshold for confirmation

    Returns:
        Tuple of (should_exit, reason)
    """
    if len(df) < accel_exit_lookback + 1:
        return False, ''

    # Check if we have enough positive P&L (or allow exit to prevent larger loss)
    pnl_ok = current_pnl >= accel_exit_min_pnl or current_pnl < 0

    if not pnl_ok:
        return False, ''

    # Get acceleration values
    if 'acceleration' not in df.columns:
        return False, ''

    accel_values = df['acceleration'].iloc[-accel_exit_lookback:].values
    current_accel = df['acceleration'].iloc[-1]

    # For LONG positions: negative acceleration is bearish
    # For SHORT positions: positive acceleration is bearish
    if position_type == 'long':
        adverse_accel = lambda a: a < 0
        adverse_jerk = lambda j: j < -jerk_confirm_threshold if jerk_confirm_threshold > 0 else j < 0
    else:
        adverse_accel = lambda a: a > 0
        adverse_jerk = lambda j: j > jerk_confirm_threshold if jerk_confirm_threshold > 0 else j > 0

    # Check acceleration condition
    accel_cond = False
    if accel_exit_type == 'sign_reversal':
        # Check consecutive bars of adverse acceleration
        accel_cond = all(adverse_accel(a) for a in accel_values)
    elif accel_exit_type == 'magnitude':
        accel_cond = abs(current_accel) > accel_exit_threshold and adverse_accel(current_accel)
    elif accel_exit_type == 'both':
        accel_cond = all(adverse_accel(a) for a in accel_values) and abs(current_accel) > accel_exit_threshold

    if not accel_cond:
        return False, ''

    # Jerk confirmation (optional)
    if use_jerk_confirm and 'jerk' in df.columns:
        current_jerk = df['jerk'].iloc[-1]
        if not adverse_jerk(current_jerk):
            return False, ''

    return True, f"Accel Reversal ({current_pnl:.2f}%)"


def check_exit_conditions(
    df: pd.DataFrame,
    position_type: str,
    entry_price: float,
    stop_loss_pct: float = 2.0,
    take_profit_pct: float = 5.0,
    use_opposite_signal: bool = True,
    use_midline_cross: bool = False,
    # Trailing stop parameters
    use_trailing_stop: bool = False,
    trailing_stop_pct: float = 1.0,
    trailing_stop_activation_pct: float = 0.3,
    high_watermark: float = 0.0,
    # Break-even stop parameters
    use_breakeven_stop: bool = False,
    breakeven_trigger_pct: float = 0.3,
    breakeven_offset_pct: float = 0.05,
    # Acceleration exit parameters
    use_accel_exit: bool = False,
    accel_exit_type: str = 'sign_reversal',
    accel_exit_threshold: float = 0.0,
    accel_exit_min_pnl: float = 0.5,
    accel_exit_lookback: int = 1,
    use_jerk_confirm: bool = False,
    jerk_confirm_threshold: float = 0.0
) -> Tuple[bool, str, float]:
    """
    Check if exit conditions are met.

    Exit Priority Order:
    1. Stop Loss (hard limit)
    2. Take Profit (hard limit)
    3. Trailing Stop (dynamic, tracks high watermark)
    4. Break-Even Stop (protects capital after profit reached)
    5. Acceleration Reversal Exit (early warning based on momentum)
    6. Midline Cross
    7. Opposite Signal

    Args:
        df: DataFrame with price and signal data
        position_type: 'long' or 'short'
        entry_price: Entry price
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        use_opposite_signal: Exit on opposite signal
        use_midline_cross: Exit on midline cross
        use_trailing_stop: Enable trailing stop from high watermark
        trailing_stop_pct: Trail distance from high watermark (%)
        trailing_stop_activation_pct: Min profit to activate trailing stop (%)
        high_watermark: Highest price since entry (caller must track this)
        use_breakeven_stop: Enable break-even stop
        breakeven_trigger_pct: Min profit to move SL to breakeven (%)
        breakeven_offset_pct: Buffer above entry price for breakeven (%)
        use_accel_exit: Enable acceleration-based exit
        accel_exit_type: 'sign_reversal', 'magnitude', or 'both'
        accel_exit_threshold: Min acceleration magnitude for exit
        accel_exit_min_pnl: Min P&L before accel exit allowed
        accel_exit_lookback: Bars of consecutive adverse acceleration
        use_jerk_confirm: Require jerk confirmation
        jerk_confirm_threshold: Jerk threshold for confirmation

    Returns:
        Tuple of (should_exit, reason, current_price)
    """
    if df.empty:
        return False, '', 0.0

    current = df.iloc[-1]
    current_price = current['Close']

    # Calculate P&L
    if position_type == 'long':
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
    else:
        pnl_pct = ((entry_price - current_price) / entry_price) * 100

    # 1. Stop Loss (highest priority)
    if pnl_pct <= -stop_loss_pct:
        return True, f"Stop Loss ({pnl_pct:.2f}%)", current_price

    # 2. Take Profit
    if pnl_pct >= take_profit_pct:
        return True, f"Take Profit ({pnl_pct:.2f}%)", current_price

    # 3. Trailing Stop (fires once activated, regardless of min_hold_bars)
    if use_trailing_stop and high_watermark > 0:
        hwm_pnl = ((high_watermark - entry_price) / entry_price) * 100
        if hwm_pnl >= trailing_stop_activation_pct:
            trail_level = high_watermark * (1 - trailing_stop_pct / 100)
            if current_price <= trail_level:
                return True, f"Trailing Stop ({pnl_pct:.2f}%)", current_price

    # 4. Break-Even Stop (fires once activated, regardless of min_hold_bars)
    if use_breakeven_stop and high_watermark > 0:
        hwm_pnl = ((high_watermark - entry_price) / entry_price) * 100
        if hwm_pnl >= breakeven_trigger_pct:
            be_price = entry_price * (1 + breakeven_offset_pct / 100)
            if current_price <= be_price:
                return True, f"Break-Even Stop ({pnl_pct:.2f}%)", current_price

    # 5. Acceleration Reversal Exit (early warning)
    if use_accel_exit:
        should_exit, reason = check_acceleration_exit(
            df=df,
            position_type=position_type,
            current_pnl=pnl_pct,
            accel_exit_type=accel_exit_type,
            accel_exit_threshold=accel_exit_threshold,
            accel_exit_min_pnl=accel_exit_min_pnl,
            accel_exit_lookback=accel_exit_lookback,
            use_jerk_confirm=use_jerk_confirm,
            jerk_confirm_threshold=jerk_confirm_threshold
        )
        if should_exit:
            return True, reason, current_price

    # 6. Midline Cross
    if use_midline_cross:
        osc = current.get('osc_smooth', 0)
        prev_osc = df.iloc[-2].get('osc_smooth', 0) if len(df) >= 2 else 0

        if position_type == 'long' and prev_osc > 0 and osc <= 0:
            return True, f"Midline Cross ({pnl_pct:.2f}%)", current_price
        if position_type == 'short' and prev_osc < 0 and osc >= 0:
            return True, f"Midline Cross ({pnl_pct:.2f}%)", current_price

    # 7. Opposite Signal
    if use_opposite_signal:
        if position_type == 'long' and current.get('sell_signal', False):
            return True, f"Opposite Signal ({pnl_pct:.2f}%)", current_price
        if position_type == 'short' and current.get('buy_signal', False):
            return True, f"Opposite Signal ({pnl_pct:.2f}%)", current_price

    return False, '', current_price
