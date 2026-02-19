"""
Composite oscillator calculation for velocity trading.

This module wraps the existing oscillator calculation infrastructure
from oscillator_predictor_page.py, providing a clean interface for
the new modular trading system.
"""

import os
import sys
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

# Add parent directory to path to import existing modules
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Try to import existing oscillator functions
try:
    from oscillator_predictor_page import (
        calculate_rsi,
        calculate_stochastic,
        calculate_williams_r,
        calculate_cci,
        calculate_momentum,
        calculate_roc,
        calculate_macd_histogram,
        calculate_bb_position,
        calculate_mfi,
        calculate_adx_trend,
        create_composite_oscillator as old_create_composite_oscillator
    )
    HAS_OSCILLATOR_MODULE = True
    HAS_OLD_COMPOSITE = True
except ImportError:
    HAS_OSCILLATOR_MODULE = False
    HAS_OLD_COMPOSITE = False
    print("Warning: oscillator_predictor_page not found, using built-in calculations")

# Import novel oscillators (ARWO, PRF, ICS, etc.)
try:
    from novel_indicators import (
        calculate_arwo,
        calculate_prf,
        calculate_ics,
        calculate_dco,
        calculate_vcmo,
        calculate_mji,
        calculate_ewaf,
        calculate_kfif
    )
    HAS_NOVEL_OSCILLATORS = True
except ImportError:
    HAS_NOVEL_OSCILLATORS = False
    print("Warning: novel_indicators not found, novel oscillators unavailable")

# Import V2 filters (Regime, Fragility, Entropy)
try:
    from novel_indicators_v2 import (
        calculate_rsc,   # Regime State Classifier
        calculate_mfi2,  # Market Fragility Index v2
        calculate_sei,   # Shannon Entropy Index
        calculate_mfv,   # Money Flow Velocity
    )
    HAS_V2_FILTERS = True
except ImportError:
    HAS_V2_FILTERS = False
    print("Warning: novel_indicators_v2 not found, V2 filters unavailable")


def calculate_composite_oscillator(
    df: pd.DataFrame,
    indicators: List[str] = None,
    weights: Dict[str, float] = None,
    oscillator_type: str = 'composite',
    config: Dict = None
) -> pd.DataFrame:
    """
    Calculate composite oscillator from multiple indicators.

    Supports both the standard composite oscillator and novel oscillators
    (ARWO, PRF, ICS, etc.) for the velocity trading system.

    Args:
        df: DataFrame with OHLCV data
        indicators: List of indicators to include (default: all matching old system)
        weights: Dict of indicator weights (default: equal weight)
        oscillator_type: Type of oscillator to calculate:
            - 'composite': Standard multi-indicator composite (default)
            - 'arwo': Adaptive Regime-Weighted Oscillator
            - 'prf': Price Regime Filter
            - 'ics': Integrated Cycle Strength
            - 'dco': Dynamic Cycle Oscillator
            - 'vcmo': Volume-Confirmed Momentum Oscillator
            - 'mji': Market Jitter Index
            - 'ewaf': Exponentially Weighted Adaptive Filter
            - 'kfif': Kalman Filter Integrated Forecast
        config: Strategy configuration dict (for advanced parameters)

    Returns:
        DataFrame with additional columns:
        - osc_smooth (or JD_Osc): Smoothed oscillator (-1 to 1)
        - velocity: Rate of change of oscillator
        - acceleration: Rate of change of velocity
    """
    # Handle novel oscillators
    if oscillator_type and oscillator_type.lower() != 'composite':
        return calculate_novel_oscillator(df, oscillator_type, config)
    result = df.copy()

    # Ensure consistent column names (case-insensitive access)
    close = result.get('Close', result.get('close', pd.Series()))
    high = result.get('High', result.get('high', pd.Series()))
    low = result.get('Low', result.get('low', pd.Series()))
    volume = result.get('Volume', result.get('volume', pd.Series(0, index=result.index)))

    if close.empty:
        result['osc_smooth'] = 0.0
        result['velocity'] = 0.0
        result['acceleration'] = 0.0
        return result

    # Calculate all component oscillators (matching old system exactly)
    components = {}

    if HAS_OSCILLATOR_MODULE:
        # RSI at multiple periods
        components['rsi_norm'] = calculate_rsi(close, 14)
        components['rsi_norm_7'] = calculate_rsi(close, 7)
        components['rsi_norm_21'] = calculate_rsi(close, 21)

        # Williams %R
        components['willr_norm'] = calculate_williams_r(high, low, close, 14)

        # CCI
        components['cci_norm'] = calculate_cci(high, low, close, 20)

        # Stochastic
        stoch_k, stoch_d = calculate_stochastic(high, low, close)
        components['stoch_k_norm'] = stoch_k
        components['stoch_d_norm'] = stoch_d

        # ROC at multiple periods
        components['roc_norm'] = calculate_roc(close, 10)
        components['roc_norm_5'] = calculate_roc(close, 5)

        # Momentum
        components['momentum_norm'] = calculate_momentum(close, 10)

        # Bollinger Band Position
        components['bb_position'] = calculate_bb_position(close, 20)

        # ADX Trend Direction
        components['adx_trend'] = calculate_adx_trend(high, low, close, 14)

        # MFI (if volume available)
        if volume.sum() > 0:
            components['mfi_norm'] = calculate_mfi(high, low, close, volume, 14)

            # Volume Velocity (matching old system)
            volume_sma = volume.rolling(20).mean()
            volume_rel = volume / volume_sma
            volume_velocity_raw = volume_rel.rolling(5).mean().diff()
            vol_vel_std = volume_velocity_raw.rolling(60).std()
            components['volume_velocity_norm'] = (volume_velocity_raw / (vol_vel_std + 0.01)).clip(-3, 3) / 3

            # Volume Momentum
            volume_momentum_raw = volume.pct_change(5)
            vol_mom_std = volume_momentum_raw.rolling(60).std()
            components['volume_momentum_norm'] = (volume_momentum_raw / (vol_mom_std + 0.01)).clip(-3, 3) / 3
    else:
        # Fallback to built-in calculations
        components['rsi_norm'] = _normalize_0_100(_builtin_rsi(close, 14))
        components['stoch_k_norm'] = _normalize_0_100(_builtin_stochastic(high, low, close))
        components['momentum_norm'] = _normalize_unbounded(close.pct_change(10))
        components['roc_norm'] = _normalize_unbounded(close.pct_change(12) * 100)

    # Add components to dataframe
    for name, series in components.items():
        result[name] = series

    # Use equal weights (matching old system default)
    if weights is None:
        weights = {k: 1.0 for k in components.keys()}

    # Calculate weighted composite
    composite = pd.Series(0.0, index=result.index)
    total_weight = 0

    for name, weight in weights.items():
        if name in result.columns:
            composite += result[name].fillna(0) * weight
            total_weight += weight

    result['composite_oscillator'] = composite / total_weight if total_weight > 0 else composite

    # Smooth the composite with trailing rolling mean (center=False for forward-only stability)
    result['composite_smooth'] = result['composite_oscillator'].rolling(window=3, center=False).mean()
    result['composite_smooth'] = result['composite_smooth'].fillna(result['composite_oscillator'])

    # Alias for new system compatibility
    result['osc_smooth'] = result['composite_smooth']

    # Apply wavelet denoising if enabled
    if config and config.get('use_wavelet_denoise', False):
        try:
            import pywt
            raw = result['osc_smooth'].values.copy()
            family = config.get('wavelet_family', 'db4')
            level = config.get('wavelet_level', 2)
            mode = config.get('wavelet_threshold_mode', 'hard')
            coeffs = pywt.wavedec(raw, family, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(raw)))
            denoised = [coeffs[0]] + [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
            osc_denoised = pywt.waverec(denoised, family)[:len(raw)]
            result['osc_smooth'] = pd.Series(osc_denoised, index=result.index)
            result['composite_smooth'] = result['osc_smooth']
        except ImportError:
            pass  # pywt not installed, use original
        except Exception:
            pass  # Denoising failed, use original

    # Calculate velocity (first derivative)
    result['velocity'] = result['osc_smooth'].diff()

    # Calculate acceleration (second derivative)
    result['acceleration'] = result['velocity'].diff()

    # Fill NaN values
    result['osc_smooth'] = result['osc_smooth'].fillna(0)
    result['velocity'] = result['velocity'].fillna(0)
    result['acceleration'] = result['acceleration'].fillna(0)

    # Compute supplementary indicators for backtest engine filters
    close_vals = result.get('Close', result.get('close', pd.Series()))
    if not close_vals.empty:
        # RSI (0-100 scale)
        if 'RSI' not in result.columns:
            try:
                _rsi_period = config.get('rsi_period', 14) if config else 14
                delta = close_vals.diff()
                gain = delta.where(delta > 0, 0.0).rolling(_rsi_period, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(_rsi_period, min_periods=1).mean()
                rs = gain / (loss + 1e-10)
                result['RSI'] = 100 - (100 / (1 + rs))
            except Exception:
                result['RSI'] = 50.0

        # MACD histogram
        if 'MACD_histogram' not in result.columns:
            try:
                ema12 = close_vals.ewm(span=12, adjust=False).mean()
                ema26 = close_vals.ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                result['MACD_histogram'] = macd_line - signal_line
            except Exception:
                result['MACD_histogram'] = 0.0

        # Bollinger Bands
        if 'BB_lower' not in result.columns:
            try:
                bb_sma = close_vals.rolling(20, min_periods=1).mean()
                bb_std = close_vals.rolling(20, min_periods=1).std().fillna(0)
                result['BB_upper'] = bb_sma + 2 * bb_std
                result['BB_lower'] = bb_sma - 2 * bb_std
            except Exception:
                result['BB_upper'] = close_vals
                result['BB_lower'] = close_vals

    return result


def calculate_novel_oscillator(
    df: pd.DataFrame,
    oscillator_type: str,
    config: Dict = None
) -> pd.DataFrame:
    """
    Calculate a novel oscillator (ARWO, PRF, ICS, etc.).

    Args:
        df: DataFrame with OHLCV data
        oscillator_type: Type of oscillator ('arwo', 'prf', 'ics', etc.)
        config: Strategy configuration dict

    Returns:
        DataFrame with JD_Osc, velocity, acceleration columns
    """
    if not HAS_NOVEL_OSCILLATORS:
        print(f"Warning: Novel oscillators not available, falling back to composite")
        return calculate_composite_oscillator(df, oscillator_type='composite')

    result = df.copy()
    config = config or {}

    # Ensure lowercase column names for novel_indicators compatibility
    col_map = {c: c.lower() for c in result.columns}
    result.columns = result.columns.str.lower()

    # Calculate the specified oscillator
    osc_type = oscillator_type.lower()

    try:
        if osc_type == 'arwo':
            result['JD_Osc'] = calculate_arwo(result)
        elif osc_type == 'prf':
            result['JD_Osc'] = calculate_prf(result)
        elif osc_type == 'ics':
            result['JD_Osc'] = calculate_ics(result)
        elif osc_type == 'dco':
            result['JD_Osc'] = calculate_dco(result)
        elif osc_type == 'vcmo':
            result['JD_Osc'] = calculate_vcmo(result)
        elif osc_type == 'mji':
            result['JD_Osc'] = calculate_mji(result)
        elif osc_type == 'ewaf':
            result['JD_Osc'] = calculate_ewaf(result)
        elif osc_type == 'kfif':
            result['JD_Osc'] = calculate_kfif(result)
        elif osc_type in ('composite', 'composite_smooth'):
            # Use composite oscillator from oscillator_indicators
            try:
                from oscillator_indicators import create_composite_oscillator_features
                osc_df = create_composite_oscillator_features(result)
                # Find the oscillator column
                for col in ['osc_composite_smooth', 'composite_oscillator', 'osc_composite']:
                    if col in osc_df.columns:
                        result['JD_Osc'] = osc_df[col]
                        break
                else:
                    # Fallback if no composite column found
                    result['JD_Osc'] = calculate_arwo(result)
            except ImportError:
                print("Warning: oscillator_indicators not available, using ARWO")
                result['JD_Osc'] = calculate_arwo(result)
        else:
            print(f"Warning: Unknown oscillator type '{oscillator_type}', using ARWO")
            result['JD_Osc'] = calculate_arwo(result)
    except Exception as e:
        print(f"Error calculating {oscillator_type}: {e}, falling back to ARWO")
        result['JD_Osc'] = calculate_arwo(result)

    # Compute supplementary indicators for filters (RSI, MACD, BB)
    # These are needed by backtest_engine.py's prepare_backtest_arrays() filters
    close_col = 'close' if 'close' in result.columns else 'Close'
    close_vals = result[close_col]

    # RSI (0-100 scale, used by rsi_filter)
    if 'RSI' not in result.columns:
        try:
            rsi_period = config.get('rsi_period', 14)
            delta = close_vals.diff()
            gain = delta.where(delta > 0, 0.0).rolling(rsi_period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_period, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            result['RSI'] = 100 - (100 / (1 + rs))
        except Exception:
            result['RSI'] = 50.0

    # MACD histogram (used by use_macd_confirm)
    if 'MACD_histogram' not in result.columns:
        try:
            ema12 = close_vals.ewm(span=12, adjust=False).mean()
            ema26 = close_vals.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            result['MACD_histogram'] = macd_line - signal_line
        except Exception:
            result['MACD_histogram'] = 0.0

    # Bollinger Bands (used by use_bb_filter)
    if 'BB_lower' not in result.columns:
        try:
            bb_sma = close_vals.rolling(20, min_periods=1).mean()
            bb_std = close_vals.rolling(20, min_periods=1).std().fillna(0)
            result['BB_upper'] = bb_sma + 2 * bb_std
            result['BB_lower'] = bb_sma - 2 * bb_std
        except Exception:
            result['BB_upper'] = close_vals
            result['BB_lower'] = close_vals

    # Calculate V2 filters if enabled in config
    if config.get('use_regime_filter') and HAS_V2_FILTERS:
        try:
            result['RSC'] = calculate_rsc(result)
        except Exception as e:
            print(f"Warning: RSC calculation failed: {e}")
            result['RSC'] = 0.0

    if config.get('use_fragility_filter') and HAS_V2_FILTERS:
        try:
            result['MFI2'] = calculate_mfi2(result)
        except Exception as e:
            print(f"Warning: MFI2 calculation failed: {e}")
            result['MFI2'] = 0.0

    if config.get('use_entropy_filter') and HAS_V2_FILTERS:
        try:
            result['SEI'] = calculate_sei(result)
        except Exception as e:
            print(f"Warning: SEI calculation failed: {e}")
            result['SEI'] = 0.0

    # Volatility regime filter (ATR percentile)
    if config.get('use_vol_regime_filter', False):
        try:
            result['VOL_REGIME'] = calculate_vol_regime(result)
        except Exception as e:
            print(f"Warning: VOL_REGIME calculation failed: {e}")
            result['VOL_REGIME'] = 0.5

    # Money Flow Velocity filter
    if config.get('use_mfv_filter') and HAS_V2_FILTERS:
        try:
            _interval = config.get('interval', '15m')
            mfv_flow, mfv_vel, mfv_acc = calculate_mfv(result, interval=_interval)
            result['MFV_FLOW'] = mfv_flow
            result['MFV_VEL'] = mfv_vel
            result['MFV_ACC'] = mfv_acc
        except Exception as e:
            print(f"Warning: MFV calculation failed: {e}")
            result['MFV_FLOW'] = 0.0
            result['MFV_VEL'] = 0.0
            result['MFV_ACC'] = 0.0

    # Alias for compatibility with signal detection
    result['osc_smooth'] = result['JD_Osc']
    result['composite_smooth'] = result['JD_Osc']

    # Calculate velocity and acceleration
    vel_smoothing = config.get('vel_smoothing', 1)
    if vel_smoothing > 1:
        smoothed = result['JD_Osc'].rolling(window=vel_smoothing, center=False).mean()
        smoothed = smoothed.bfill()
        result['velocity'] = smoothed.diff()
    else:
        result['velocity'] = result['JD_Osc'].diff()

    result['acceleration'] = result['velocity'].diff()

    # Fill NaN values
    result['JD_Osc'] = result['JD_Osc'].fillna(0)
    result['osc_smooth'] = result['osc_smooth'].fillna(0)
    result['velocity'] = result['velocity'].fillna(0)
    result['acceleration'] = result['acceleration'].fillna(0)

    # Restore original column case for OHLCV
    result.columns = [c.title() if c in ['open', 'high', 'low', 'close', 'volume'] else c for c in result.columns]

    return result


def calculate_composite_oscillator_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate composite oscillator using the OLD system's exact calculation.

    This ensures 100% matching with velocity_live_trader.py for verification.
    Uses create_composite_oscillator from oscillator_predictor_page.py.
    """
    if not HAS_OLD_COMPOSITE:
        print("Warning: Old composite oscillator not available, using new calculation")
        return calculate_composite_oscillator(df)

    result = df.copy()

    # Use old system's calculation (it expects lowercase column names)
    result.columns = result.columns.str.lower()
    result = old_create_composite_oscillator(result)

    # Rename columns to match expected format
    if 'composite_smooth' in result.columns:
        result['osc_smooth'] = result['composite_smooth']
    if 'composite_velocity' in result.columns:
        result['velocity'] = result['composite_velocity']
    if 'composite_acceleration' in result.columns:
        result['acceleration'] = result['composite_acceleration']

    # Ensure required columns exist
    if 'osc_smooth' not in result.columns:
        result['osc_smooth'] = 0.0
    if 'velocity' not in result.columns:
        result['velocity'] = result['osc_smooth'].diff().fillna(0)
    if 'acceleration' not in result.columns:
        result['acceleration'] = result['velocity'].diff().fillna(0)

    # Restore original column case for OHLCV
    result.columns = [c.title() if c in ['open', 'high', 'low', 'close', 'volume'] else c for c in result.columns]

    return result


def _calculate_single_oscillator(df: pd.DataFrame, indicator: str) -> Optional[pd.Series]:
    """
    Calculate a single oscillator and normalize to -1 to 1 range.

    Note: Functions from oscillator_predictor_page.py already return
    normalized values (-1 to +1), so we don't need to re-normalize them.
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df.get('Volume', pd.Series(0, index=df.index))

    if HAS_OSCILLATOR_MODULE:
        # Use existing functions from oscillator_predictor_page
        # All these functions already return values normalized to -1 to +1
        if indicator == 'RSI':
            return calculate_rsi(close)  # Already -1 to +1
        elif indicator == 'Stochastic':
            # Returns tuple (stoch_k, stoch_d), both already -1 to +1
            stoch_k, stoch_d = calculate_stochastic(high, low, close)
            return stoch_k  # Use %K
        elif indicator == 'Williams_R':
            return calculate_williams_r(high, low, close)  # Already -1 to +1
        elif indicator == 'CCI':
            return calculate_cci(high, low, close)  # Already -1 to +1
        elif indicator == 'Momentum':
            return calculate_momentum(close)  # Already -1 to +1
        elif indicator == 'ROC':
            return calculate_roc(close)  # Already -1 to +1
        elif indicator == 'MACD':
            return calculate_macd_histogram(close)  # Already -1 to +1
        elif indicator == 'BB':
            return calculate_bb_position(close)  # Already -1 to +1
        elif indicator == 'MFI':
            if volume.sum() > 0:
                return calculate_mfi(high, low, close, volume)  # Already -1 to +1
            return None

    # Built-in fallback calculations
    return _calculate_builtin_oscillator(df, indicator)


def _calculate_builtin_oscillator(df: pd.DataFrame, indicator: str) -> Optional[pd.Series]:
    """
    Built-in oscillator calculations (fallback if main module not available).
    """
    close = df['Close']
    high = df['High']
    low = df['Low']

    if indicator == 'RSI':
        return _normalize_0_100(_builtin_rsi(close))
    elif indicator == 'Stochastic':
        return _normalize_0_100(_builtin_stochastic(high, low, close))
    elif indicator == 'Momentum':
        return _normalize_unbounded(close.pct_change(10))
    elif indicator == 'ROC':
        return _normalize_unbounded(close.pct_change(12) * 100)

    return None


def _builtin_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Built-in RSI calculation."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 0.001)
    return 100 - (100 / (1 + rs))


def _builtin_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                        k_period: int = 14) -> pd.Series:
    """Built-in Stochastic calculation."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    return 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, 0.001)


def _normalize_0_100(series: pd.Series) -> pd.Series:
    """Normalize 0-100 range to -1 to 1."""
    return (series - 50) / 50


def _normalize_unbounded(series: pd.Series, clip_at: float = 3.0) -> pd.Series:
    """Normalize unbounded series using z-score and clip."""
    mean = series.rolling(window=50, min_periods=10).mean()
    std = series.rolling(window=50, min_periods=10).std().replace(0, 1)
    z = (series - mean) / std
    return z.clip(-clip_at, clip_at) / clip_at


def get_oscillator_zone(value: float, oversold: float = -0.3, overbought: float = 0.3) -> str:
    """
    Get the zone description for an oscillator value.

    Returns: 'oversold', 'overbought', or 'neutral'
    """
    if value < oversold:
        return 'oversold'
    elif value > overbought:
        return 'overbought'
    return 'neutral'


def detect_crossings(
    df: pd.DataFrame,
    column: str = 'osc_smooth',
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Detect threshold crossings in an oscillator.

    Args:
        df: DataFrame with oscillator column
        column: Column name to analyze
        threshold: Crossing threshold

    Returns:
        DataFrame with additional columns:
        - cross_up: True on bars where value crosses above threshold
        - cross_down: True on bars where value crosses below threshold
    """
    result = df.copy()
    values = result[column]

    # Previous values
    prev_values = values.shift(1)

    # Cross up: was below, now above
    result['cross_up'] = (prev_values < threshold) & (values >= threshold)

    # Cross down: was above, now below
    result['cross_down'] = (prev_values > threshold) & (values <= threshold)

    return result


def calculate_vol_regime(df, atr_period=14, lookback=50):
    """Calculate ATR-based volatility regime as a rolling percentile (0-1).

    Returns a Series where 0 = lowest volatility, 1 = highest volatility.
    Low-volatility bars (< 0.25) correspond to choppy, low-movement periods
    where mean-reversion signals tend to be noise.

    Args:
        df: DataFrame with 'close', 'high', 'low' columns (lowercase)
        atr_period: Period for ATR calculation (default 14)
        lookback: Rolling window for percentile rank (default 50)
    """
    close = df['close'] if 'close' in df.columns else df['Close']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(atr_period, min_periods=1).mean()
    atr_pct = atr / close * 100  # Normalize as % of price

    vol_regime = atr_pct.rolling(lookback, min_periods=10).rank(pct=True).fillna(0.5)
    return vol_regime
