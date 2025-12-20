"""
Oscillator-Based Indicators Module

This module contains all the oscillator-based indicators developed in the Oscillator Predictor page,
now made available to the Strategy Optimization page and other parts of the application.

Includes:
1. Normalized Component Oscillators (RSI, Williams %R, CCI, Stochastic, ROC, MFI, etc.)
2. Composite Oscillator (weighted average of components)
3. Oscillator Derivatives (velocity, acceleration, jerk)
4. Rolling Statistics (mean, std, min, max, range, position)
5. Novel Oscillators (ARWO, DCO, VCMO, ICS, MJI, PRF, EWAF, KFIF)
6. Consensus/Dispersion Features
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

# Try to import novel indicators
try:
    from novel_indicators import (
        calculate_arwo, calculate_dco, calculate_vcmo, calculate_ics,
        calculate_mji, calculate_prf, calculate_ewaf, calculate_kfif
    )
    NOVEL_INDICATORS_AVAILABLE = True
except ImportError:
    NOVEL_INDICATORS_AVAILABLE = False
    print("Note: Novel indicators not available in oscillator_indicators.py")


# =============================================================================
# NORMALIZED COMPONENT OSCILLATORS
# All normalized to -1 to +1 range for consistency
# =============================================================================

def normalize_to_range(series: pd.Series, min_val: float = -1, max_val: float = 1) -> pd.Series:
    """Normalize a series to a specified range."""
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series(0, index=series.index)
    return min_val + (series - s_min) * (max_val - min_val) / (s_max - s_min)


def calculate_rsi_normalized(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI normalized to -1 to +1 range."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Normalize: RSI 0-100 -> -1 to +1
    return (rsi - 50) / 50


def calculate_williams_r_normalized(high: pd.Series, low: pd.Series,
                                     close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Williams %R normalized to -1 to +1 range."""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    willr = -100 * ((highest_high - close) / (highest_high - lowest_low + 1e-10))
    # Normalize: -100 to 0 -> -1 to +1
    return (willr + 50) / 50


def calculate_cci_normalized(high: pd.Series, low: pd.Series,
                              close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate CCI normalized to -1 to +1 range."""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = (typical_price - sma).abs().rolling(window=period).mean()
    cci = (typical_price - sma) / (0.015 * mean_deviation + 1e-10)
    # Normalize CCI (typically -200 to +200) to -1 to +1
    return np.clip(cci / 200, -1, 1)


def calculate_stochastic_normalized(high: pd.Series, low: pd.Series, close: pd.Series,
                                     k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic K and D normalized to -1 to +1 range."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
    stoch_d = stoch_k.rolling(window=d_period).mean()
    # Normalize: 0-100 -> -1 to +1
    return (stoch_k - 50) / 50, (stoch_d - 50) / 50


def calculate_roc_normalized(close: pd.Series, period: int = 10) -> pd.Series:
    """Calculate Rate of Change normalized to -1 to +1 range."""
    roc = ((close - close.shift(period)) / (close.shift(period) + 1e-10)) * 100
    # Clip to reasonable range and normalize
    roc_clipped = np.clip(roc, -20, 20)
    return roc_clipped / 20


def calculate_momentum_normalized(close: pd.Series, period: int = 10) -> pd.Series:
    """Calculate Momentum normalized to -1 to +1 range."""
    mom = close - close.shift(period)
    # Normalize using rolling std
    mom_std = mom.rolling(window=50, min_periods=10).std()
    normalized = mom / (mom_std * 3 + 1e-10)
    return np.clip(normalized, -1, 1)


def calculate_bb_position(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Calculate position within Bollinger Bands (-1 at lower band, +1 at upper band)."""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper_band = sma + std_dev * std
    lower_band = sma - std_dev * std
    # Position: -1 at lower, +1 at upper
    position = 2 * ((close - lower_band) / (upper_band - lower_band + 1e-10)) - 1
    return np.clip(position, -1, 1)


def calculate_adx_trend(high: pd.Series, low: pd.Series, close: pd.Series,
                         period: int = 14) -> pd.Series:
    """Calculate ADX-based trend direction indicator (-1 to +1)."""
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Calculate +DM and -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

    # Smooth DM
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))

    # Trend direction: +DI - -DI, normalized
    trend = (plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return np.clip(trend, -1, 1)


def calculate_mfi_normalized(high: pd.Series, low: pd.Series, close: pd.Series,
                              volume: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index normalized to -1 to +1 range."""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    # Positive and negative money flow
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
    # Normalize: 0-100 -> -1 to +1
    return (mfi - 50) / 50


def calculate_macd_histogram_normalized(close: pd.Series, fast: int = 12,
                                         slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD Histogram normalized to -1 to +1 range."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line

    # Normalize using rolling std
    hist_std = histogram.rolling(window=50, min_periods=10).std()
    normalized = histogram / (hist_std * 3 + 1e-10)
    return np.clip(normalized, -1, 1)


def calculate_tsi_normalized(close: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
    """Calculate True Strength Index normalized to -1 to +1 range."""
    momentum = close.diff(1)

    # Double smoothed momentum
    smooth1 = momentum.ewm(span=r, adjust=False).mean()
    double_smooth = smooth1.ewm(span=s, adjust=False).mean()

    # Double smoothed absolute momentum
    abs_smooth1 = momentum.abs().ewm(span=r, adjust=False).mean()
    abs_double_smooth = abs_smooth1.ewm(span=s, adjust=False).mean()

    tsi = double_smooth / (abs_double_smooth + 1e-10)
    return np.clip(tsi, -1, 1)


def calculate_ultimate_oscillator_normalized(high: pd.Series, low: pd.Series,
                                              close: pd.Series) -> pd.Series:
    """Calculate Ultimate Oscillator normalized to -1 to +1 range."""
    bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    tr = pd.concat([high, close.shift(1)], axis=1).max(axis=1) - pd.concat([low, close.shift(1)], axis=1).min(axis=1)

    avg7 = bp.rolling(7).sum() / (tr.rolling(7).sum() + 1e-10)
    avg14 = bp.rolling(14).sum() / (tr.rolling(14).sum() + 1e-10)
    avg28 = bp.rolling(28).sum() / (tr.rolling(28).sum() + 1e-10)

    uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
    # Normalize: 0-100 -> -1 to +1
    return (uo - 50) / 50


# =============================================================================
# COMPOSITE OSCILLATOR AND DERIVATIVES
# =============================================================================

def create_composite_oscillator_features(data: pd.DataFrame,
                                          weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Create composite oscillator and all derivative features.

    Returns DataFrame with all oscillator features added.
    """
    df = data.copy()

    # Ensure column names are lowercase
    df.columns = df.columns.str.lower()

    # Calculate all component oscillators
    components = {}

    # RSI variants
    components['osc_rsi_norm'] = calculate_rsi_normalized(df['close'], 14)
    components['osc_rsi_norm_7'] = calculate_rsi_normalized(df['close'], 7)
    components['osc_rsi_norm_21'] = calculate_rsi_normalized(df['close'], 21)

    # Williams %R
    components['osc_willr_norm'] = calculate_williams_r_normalized(df['high'], df['low'], df['close'], 14)

    # CCI
    components['osc_cci_norm'] = calculate_cci_normalized(df['high'], df['low'], df['close'], 20)

    # Stochastic
    stoch_k, stoch_d = calculate_stochastic_normalized(df['high'], df['low'], df['close'])
    components['osc_stoch_k_norm'] = stoch_k
    components['osc_stoch_d_norm'] = stoch_d

    # ROC variants
    components['osc_roc_norm'] = calculate_roc_normalized(df['close'], 10)
    components['osc_roc_norm_5'] = calculate_roc_normalized(df['close'], 5)

    # Momentum
    components['osc_momentum_norm'] = calculate_momentum_normalized(df['close'], 10)

    # Bollinger Band Position
    components['osc_bb_position'] = calculate_bb_position(df['close'], 20)

    # ADX Trend
    components['osc_adx_trend'] = calculate_adx_trend(df['high'], df['low'], df['close'], 14)

    # MFI (if volume available)
    if 'volume' in df.columns and df['volume'].sum() > 0:
        components['osc_mfi_norm'] = calculate_mfi_normalized(df['high'], df['low'], df['close'], df['volume'], 14)

    # MACD Histogram
    components['osc_macd_hist_norm'] = calculate_macd_histogram_normalized(df['close'])

    # TSI
    components['osc_tsi_norm'] = calculate_tsi_normalized(df['close'])

    # Ultimate Oscillator
    components['osc_uo_norm'] = calculate_ultimate_oscillator_normalized(df['high'], df['low'], df['close'])

    # Add all components to dataframe
    for name, series in components.items():
        df[name] = series

    # Default weights if not specified
    if weights is None:
        weights = {k: 1.0 for k in components.keys()}

    # Calculate weighted composite
    composite = pd.Series(0.0, index=df.index)
    total_weight = 0

    for name, weight in weights.items():
        if name in df.columns:
            composite += df[name].fillna(0) * weight
            total_weight += weight

    if total_weight > 0:
        df['osc_composite'] = composite / total_weight
    else:
        df['osc_composite'] = composite

    # Smooth composite
    df['osc_composite_smooth'] = df['osc_composite'].rolling(window=3, center=True).mean()
    df['osc_composite_smooth'] = df['osc_composite_smooth'].fillna(df['osc_composite'])

    return df


def add_oscillator_derivatives(df: pd.DataFrame,
                                oscillator_col: str = 'osc_composite_smooth') -> pd.DataFrame:
    """
    Add velocity, acceleration, and jerk features for the oscillator.
    """
    df = df.copy()
    osc = df[oscillator_col]

    # Derivatives
    df['osc_velocity'] = osc.diff(1)
    df['osc_acceleration'] = df['osc_velocity'].diff(1)
    df['osc_jerk'] = df['osc_acceleration'].diff(1)

    # Lagged values
    for lag in [1, 2, 3, 5, 7, 10]:
        df[f'osc_lag_{lag}'] = osc.shift(lag)
        df[f'osc_velocity_lag_{lag}'] = df['osc_velocity'].shift(lag)

    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'osc_mean_{window}'] = osc.rolling(window).mean()
        df[f'osc_std_{window}'] = osc.rolling(window).std()
        df[f'osc_min_{window}'] = osc.rolling(window).min()
        df[f'osc_max_{window}'] = osc.rolling(window).max()
        df[f'osc_range_{window}'] = df[f'osc_max_{window}'] - df[f'osc_min_{window}']
        df[f'osc_position_{window}'] = (osc - df[f'osc_min_{window}']) / (df[f'osc_range_{window}'] + 1e-10)

    # Distance from extremes
    df['osc_dist_from_1'] = 1 - osc
    df['osc_dist_from_neg1'] = osc - (-1)

    # Momentum features
    df['osc_momentum_3'] = osc - osc.shift(3)
    df['osc_momentum_5'] = osc - osc.shift(5)
    df['osc_momentum_10'] = osc - osc.shift(10)

    # Trend features
    df['osc_above_mean_10'] = (osc > df['osc_mean_10']).astype(int)
    df['osc_above_mean_20'] = (osc > df['osc_mean_20']).astype(int)

    # Zero crossing detection
    df['osc_crossed_zero'] = ((osc.shift(1) < 0) & (osc >= 0)) | ((osc.shift(1) > 0) & (osc <= 0))
    df['osc_crossed_zero'] = df['osc_crossed_zero'].astype(int)

    # Volatility
    df['osc_volatility_10'] = osc.rolling(10).std()
    df['osc_volatility_20'] = osc.rolling(20).std()

    # Velocity of component oscillators
    component_cols = [c for c in df.columns if c.startswith('osc_') and c.endswith('_norm')]
    for col in component_cols[:5]:
        df[f'{col}_velocity'] = df[col].diff(1)

    return df


def add_novel_oscillator_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all 8 novel oscillators and their derivative features.
    """
    if not NOVEL_INDICATORS_AVAILABLE:
        return df

    df = df.copy()

    try:
        # Calculate all 8 novel oscillators
        df['novel_arwo'] = calculate_arwo(df)
        df['novel_dco'] = calculate_dco(df)
        df['novel_vcmo'] = calculate_vcmo(df)
        df['novel_ics'] = calculate_ics(df)
        df['novel_mji'] = calculate_mji(df)
        df['novel_prf'] = calculate_prf(df)
        df['novel_ewaf'] = calculate_ewaf(df)

        kfif_val, kfif_upper, kfif_lower = calculate_kfif(df)
        df['novel_kfif'] = kfif_val
        df['novel_kfif_width'] = kfif_upper - kfif_lower  # Uncertainty measure

        # Add velocities
        novel_cols = ['novel_arwo', 'novel_dco', 'novel_vcmo', 'novel_ics',
                      'novel_mji', 'novel_prf', 'novel_ewaf', 'novel_kfif']
        for col in novel_cols:
            if col in df.columns:
                df[f'{col}_velocity'] = df[col].diff(1)

        # Consensus/dispersion features
        existing_novel = [c for c in novel_cols if c in df.columns]
        if len(existing_novel) >= 3:
            df['novel_consensus'] = df[existing_novel].mean(axis=1)
            df['novel_dispersion'] = df[existing_novel].std(axis=1)
            df['novel_direction_agreement'] = (df[existing_novel] > 0).sum(axis=1) / len(existing_novel)

            # Extreme readings
            df['novel_all_positive'] = (df[existing_novel] > 0.3).all(axis=1).astype(int)
            df['novel_all_negative'] = (df[existing_novel] < -0.3).all(axis=1).astype(int)

    except Exception as e:
        print(f"Warning: Could not add novel oscillator features: {e}")

    return df


# =============================================================================
# MAIN INTEGRATION FUNCTION
# =============================================================================

def integrate_oscillator_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to integrate ALL oscillator-based indicators.

    This adds:
    1. 15+ normalized component oscillators
    2. Composite oscillator (weighted average)
    3. Oscillator derivatives (velocity, acceleration, jerk)
    4. Rolling statistics (mean, std, min, max, range, position)
    5. Lagged values
    6. Momentum and trend features
    7. All 8 novel oscillators with derivatives
    8. Consensus/dispersion features

    Returns:
        DataFrame with all oscillator indicators added
    """
    df = data.copy()

    # Step 1: Create composite oscillator and components
    df = create_composite_oscillator_features(df)

    # Step 2: Add derivatives and rolling features
    df = add_oscillator_derivatives(df)

    # Step 3: Add novel oscillator features
    df = add_novel_oscillator_features(df)

    return df


# List of all oscillator indicator columns for reference
OSCILLATOR_INDICATOR_LIST = [
    # Component oscillators
    'osc_rsi_norm', 'osc_rsi_norm_7', 'osc_rsi_norm_21',
    'osc_willr_norm', 'osc_cci_norm', 'osc_stoch_k_norm', 'osc_stoch_d_norm',
    'osc_roc_norm', 'osc_roc_norm_5', 'osc_momentum_norm',
    'osc_bb_position', 'osc_adx_trend', 'osc_mfi_norm',
    'osc_macd_hist_norm', 'osc_tsi_norm', 'osc_uo_norm',
    # Composite
    'osc_composite', 'osc_composite_smooth',
    # Derivatives
    'osc_velocity', 'osc_acceleration', 'osc_jerk',
    # Lagged
    'osc_lag_1', 'osc_lag_2', 'osc_lag_3', 'osc_lag_5', 'osc_lag_7', 'osc_lag_10',
    'osc_velocity_lag_1', 'osc_velocity_lag_2', 'osc_velocity_lag_3',
    # Rolling stats
    'osc_mean_5', 'osc_mean_10', 'osc_mean_20',
    'osc_std_5', 'osc_std_10', 'osc_std_20',
    'osc_min_5', 'osc_min_10', 'osc_min_20',
    'osc_max_5', 'osc_max_10', 'osc_max_20',
    'osc_range_5', 'osc_range_10', 'osc_range_20',
    'osc_position_5', 'osc_position_10', 'osc_position_20',
    # Distances and momentum
    'osc_dist_from_1', 'osc_dist_from_neg1',
    'osc_momentum_3', 'osc_momentum_5', 'osc_momentum_10',
    'osc_above_mean_10', 'osc_above_mean_20', 'osc_crossed_zero',
    'osc_volatility_10', 'osc_volatility_20',
    # Novel oscillators
    'novel_arwo', 'novel_dco', 'novel_vcmo', 'novel_ics',
    'novel_mji', 'novel_prf', 'novel_ewaf', 'novel_kfif', 'novel_kfif_width',
    # Novel velocities
    'novel_arwo_velocity', 'novel_dco_velocity', 'novel_vcmo_velocity',
    'novel_ics_velocity', 'novel_mji_velocity', 'novel_prf_velocity',
    'novel_ewaf_velocity', 'novel_kfif_velocity',
    # Novel consensus
    'novel_consensus', 'novel_dispersion', 'novel_direction_agreement',
    'novel_all_positive', 'novel_all_negative',
]
