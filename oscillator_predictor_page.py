"""
Composite Oscillator ML Predictor
=================================
A simple, clean approach to trading signal generation:
1. Combine multiple indicators into a single composite oscillator (-1 to +1)
2. Find peaks/valleys in the oscillator using scipy
3. Train ML model to PREDICT oscillator peaks/valleys (not price!)
4. Generate trading signals from predicted oscillator extremes

This is fundamentally more tractable than predicting price because:
- Oscillators are bounded and mean-reverting
- Peaks/valleys are well-defined mathematical concepts
- The target is smoother and more predictable
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# SMOTE for class imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# For parallel processing
import os
import json
import time
import multiprocessing
from joblib import Parallel, delayed

# Price prediction module
try:
    from price_prediction import PriceRangePredictor, PriceTargetCalculator, ExitTimingPredictor
    PRICE_PREDICTION_AVAILABLE = True
except ImportError:
    PRICE_PREDICTION_AVAILABLE = False

# Polygon API for options data
try:
    from polygon_manager import PolygonManager
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

# Market hours utilities (for data refresh logic)
try:
    from velocity_production_utils import is_market_open, get_market_time
    MARKET_UTILS_AVAILABLE = True
except ImportError:
    MARKET_UTILS_AVAILABLE = False

# Optimization timing tracking
OPTIMIZATION_TIMING_FILE = "optimization_timing.json"

def load_optimization_timing() -> dict:
    """Load optimization timing history."""
    if os.path.exists(OPTIMIZATION_TIMING_FILE):
        try:
            with open(OPTIMIZATION_TIMING_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"runs": [], "avg_trials_per_minute": None}

def save_optimization_timing(trials: int, duration_seconds: float, n_workers: int):
    """Save optimization timing data and update average."""
    timing_data = load_optimization_timing()

    trials_per_minute = (trials / duration_seconds) * 60 if duration_seconds > 0 else 0

    # Add this run
    timing_data["runs"].append({
        "timestamp": datetime.now().isoformat(),
        "trials": trials,
        "duration_seconds": round(duration_seconds, 1),
        "n_workers": n_workers,
        "trials_per_minute": round(trials_per_minute, 0)
    })

    # Keep only last 20 runs
    timing_data["runs"] = timing_data["runs"][-20:]

    # Calculate weighted average (more recent runs weighted higher)
    if timing_data["runs"]:
        # Weight recent runs more heavily
        weights = [i + 1 for i in range(len(timing_data["runs"]))]
        weighted_sum = sum(r["trials_per_minute"] * w for r, w in zip(timing_data["runs"], weights))
        total_weight = sum(weights)
        timing_data["avg_trials_per_minute"] = round(weighted_sum / total_weight, 0)

    with open(OPTIMIZATION_TIMING_FILE, 'w') as f:
        json.dump(timing_data, f, indent=2)

    return trials_per_minute

def estimate_optimization_time(trials: int, n_workers: int) -> str:
    """Estimate optimization time based on historical data."""
    timing_data = load_optimization_timing()

    if not timing_data.get("avg_trials_per_minute"):
        return "No timing data yet - run an optimization first"

    avg_rate = timing_data["avg_trials_per_minute"]

    # Rough adjustment for worker count (not perfectly linear)
    # Use the most recent run's worker count as baseline
    if timing_data["runs"]:
        last_workers = timing_data["runs"][-1].get("n_workers", 8)
        # Scaling factor: more workers = faster, but diminishing returns
        if last_workers != n_workers:
            # Approximate scaling: sqrt relationship for diminishing returns
            scale_factor = (n_workers / last_workers) ** 0.7
            avg_rate = avg_rate * scale_factor

    estimated_minutes = trials / avg_rate

    if estimated_minutes < 1:
        return f"~{int(estimated_minutes * 60)} seconds"
    elif estimated_minutes < 60:
        return f"~{int(estimated_minutes)} minutes"
    else:
        hours = int(estimated_minutes // 60)
        mins = int(estimated_minutes % 60)
        return f"~{hours}h {mins}m"

# Force spawn method to avoid TensorFlow re-import issues in worker processes
# Must be done before any multiprocessing happens
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

N_CORES = os.cpu_count() or 4
N_JOBS_OPTUNA = max(1, N_CORES - 1)  # Leave 1 core free for system

# Import lightweight Optuna worker module (avoids TensorFlow re-import in worker processes)
import optuna_worker

# Novel indicators module (advanced composite oscillators)
try:
    from novel_indicators import (
        calculate_all_novel_indicators,
        calculate_arwo, calculate_dco, calculate_vcmo,
        calculate_ics, calculate_mji, calculate_prf, calculate_ewaf, calculate_kfif,
        NOVEL_INDICATORS
    )
    NOVEL_INDICATORS_AVAILABLE = True
except ImportError:
    NOVEL_INDICATORS_AVAILABLE = False
    print("Warning: novel_indicators module not available")

# XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ============================================================================
# STEP 1: INDICATOR CALCULATIONS
# ============================================================================

def normalize_to_range(series: pd.Series, target_min: float = -1, target_max: float = 1) -> pd.Series:
    """Normalize any series to a target range (default -1 to +1)"""
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series(0, index=series.index)
    normalized = (series - s_min) / (s_max - s_min)  # 0 to 1
    return normalized * (target_max - target_min) + target_min  # target range


def normalize_oscillator(series: pd.Series, low: float, high: float) -> pd.Series:
    """Normalize an oscillator with known bounds to -1 to +1"""
    # Map [low, high] -> [-1, +1]
    midpoint = (low + high) / 2
    half_range = (high - low) / 2
    return (series - midpoint) / half_range


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI (0-100) and normalize to -1 to +1"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Normalize: RSI 0-100 -> -1 to +1 (RSI 50 = 0)
    return (rsi - 50) / 50


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Williams %R (-100 to 0) and normalize to -1 to +1"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Normalize: -100 to 0 -> -1 to +1
    return (willr + 50) / 50


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate CCI (unbounded but typically -200 to +200) and normalize"""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma) / (0.015 * mad)
    # Clip to reasonable range and normalize
    return (cci / 200).clip(-1, 1)


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                         k_period: int = 14, d_period: int = 3) -> tuple:
    """Calculate Stochastic %K and %D, normalize to -1 to +1"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    # Normalize: 0-100 -> -1 to +1
    return (stoch_k - 50) / 50, (stoch_d - 50) / 50


def calculate_roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Calculate Rate of Change and normalize"""
    roc = ((close - close.shift(period)) / close.shift(period)) * 100
    # Clip to reasonable range and normalize
    return (roc / 10).clip(-1, 1)


def calculate_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """Calculate Momentum and normalize"""
    mom = close - close.shift(period)
    # Normalize using rolling window
    rolling_max = mom.rolling(window=50).max()
    rolling_min = mom.rolling(window=50).min()
    normalized = 2 * (mom - rolling_min) / (rolling_max - rolling_min) - 1
    return normalized.clip(-1, 1)


def calculate_macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD Histogram and normalize"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - macd_signal
    # Normalize using rolling percentile
    return normalize_to_range(histogram.rolling(window=100).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    ) * 2 - 1)


def calculate_bb_position(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Calculate position within Bollinger Bands (-1 to +1)"""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    # Position: -1 at lower band, +1 at upper band
    position = 2 * (close - lower) / (upper - lower) - 1
    return position.clip(-1, 1)


def calculate_adx_trend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ADX-based trend strength, normalize to -1 to +1"""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    # DI Difference normalized (positive = bullish trend, negative = bearish)
    di_diff = (plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return di_diff.clip(-1, 1)


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index and normalize"""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    money_flow_positive = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    money_flow_negative = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_sum = money_flow_positive.rolling(window=period).sum()
    negative_sum = money_flow_negative.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + positive_sum / (negative_sum + 1e-10)))
    # Normalize: 0-100 -> -1 to +1
    return (mfi - 50) / 50


# ============================================================================
# STEP 2: COMPOSITE OSCILLATOR CREATION
# ============================================================================

def create_composite_oscillator(data: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """
    Create a composite oscillator from multiple normalized indicators.
    All indicators are normalized to -1 to +1 range.
    """
    df = data.copy()

    # Ensure column names are lowercase
    df.columns = df.columns.str.lower()

    # Calculate all component oscillators (all normalized to -1 to +1)
    components = {}

    # RSI
    components['rsi_norm'] = calculate_rsi(df['close'], 14)
    components['rsi_norm_7'] = calculate_rsi(df['close'], 7)
    components['rsi_norm_21'] = calculate_rsi(df['close'], 21)

    # Williams %R
    components['willr_norm'] = calculate_williams_r(df['high'], df['low'], df['close'], 14)

    # CCI
    components['cci_norm'] = calculate_cci(df['high'], df['low'], df['close'], 20)

    # Stochastic
    stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
    components['stoch_k_norm'] = stoch_k
    components['stoch_d_norm'] = stoch_d

    # ROC
    components['roc_norm'] = calculate_roc(df['close'], 10)
    components['roc_norm_5'] = calculate_roc(df['close'], 5)

    # Momentum
    components['momentum_norm'] = calculate_momentum(df['close'], 10)

    # Bollinger Band Position
    components['bb_position'] = calculate_bb_position(df['close'], 20)

    # ADX Trend Direction
    components['adx_trend'] = calculate_adx_trend(df['high'], df['low'], df['close'], 14)

    # MFI (if volume available)
    if 'volume' in df.columns and df['volume'].sum() > 0:
        components['mfi_norm'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'], 14)

        # Volume Velocity (1st derivative) - normalized to -1 to +1
        # These showed #2 and #5 importance in walk-forward analysis
        volume = df['volume']
        volume_sma = volume.rolling(20).mean()
        volume_rel = volume / volume_sma  # Relative volume

        # Volume velocity (5-day smoothed)
        volume_velocity_raw = volume_rel.rolling(5).mean().diff()
        vol_vel_std = volume_velocity_raw.rolling(60).std()
        # Normalize to roughly -1 to +1 using z-score and clip
        components['volume_velocity_norm'] = (volume_velocity_raw / (vol_vel_std + 0.01)).clip(-3, 3) / 3

        # Volume momentum (5-day rate of change)
        volume_momentum_raw = volume.pct_change(5)
        vol_mom_std = volume_momentum_raw.rolling(60).std()
        # Normalize to roughly -1 to +1
        components['volume_momentum_norm'] = (volume_momentum_raw / (vol_mom_std + 0.01)).clip(-3, 3) / 3

    # Add all components to dataframe
    for name, series in components.items():
        df[name] = series

    # Default equal weights if not specified
    if weights is None:
        weights = {k: 1.0 for k in components.keys()}

    # Calculate weighted composite
    composite = pd.Series(0.0, index=df.index)
    total_weight = 0

    for name, weight in weights.items():
        if name in df.columns:
            composite += df[name].fillna(0) * weight
            total_weight += weight

    df['composite_oscillator'] = composite / total_weight

    # Smooth the composite slightly to reduce noise
    df['composite_smooth'] = df['composite_oscillator'].rolling(window=3, center=True).mean()
    df['composite_smooth'] = df['composite_smooth'].fillna(df['composite_oscillator'])

    return df


# ============================================================================
# STEP 3: PEAK/VALLEY DETECTION ON OSCILLATOR
# ============================================================================

def detect_oscillator_peaks_valleys(oscillator: pd.Series, prominence: float = 0.1,
                                    distance: int = 5) -> tuple:
    """
    Detect peaks and valleys in the oscillator using scipy.

    Args:
        oscillator: The composite oscillator series
        prominence: Minimum prominence of peaks (0-1 scale)
        distance: Minimum distance between peaks

    Returns:
        peak_indices, valley_indices
    """
    values = oscillator.values

    # Find peaks (local maxima)
    peak_indices, peak_props = find_peaks(values, prominence=prominence, distance=distance)

    # Find valleys (local minima) by inverting
    valley_indices, valley_props = find_peaks(-values, prominence=prominence, distance=distance)

    return peak_indices, valley_indices


def create_peak_valley_labels(df: pd.DataFrame, oscillator_col: str = 'composite_smooth',
                               prominence: float = 0.15, distance: int = 5,
                               lookahead: int = 5) -> pd.DataFrame:
    """
    Create labels for predicting upcoming peaks/valleys.

    Label encoding (CORRECTED - intuitive convention):
        +1 = Valley approaching (BUY signal) - oscillator oversold, price low, BUY opportunity
        0 = Neither (HOLD)
        -1 = Peak approaching (SELL signal) - oscillator overbought, price high, SELL opportunity

    Trading Logic:
        - Oscillator VALLEY (low, oversold) → Price is LOW → BUY (+1)
        - Oscillator PEAK (high, overbought) → Price is HIGH → SELL (-1)
    """
    df = df.copy()
    oscillator = df[oscillator_col]

    # Detect peaks and valleys
    peak_idx, valley_idx = detect_oscillator_peaks_valleys(
        oscillator, prominence=prominence, distance=distance
    )

    # Initialize labels
    df['label'] = 0

    # Mark valleys (and N bars before) as BUY opportunities
    # Valley = oversold = price low = BUY = +1
    for idx in valley_idx:
        start = max(0, idx - lookahead)
        df.iloc[start:idx+1, df.columns.get_loc('label')] = 1  # +1 = Valley = BUY

    # Mark peaks (and N bars before) as SELL opportunities
    # Peak = overbought = price high = SELL = -1
    for idx in peak_idx:
        start = max(0, idx - lookahead)
        df.iloc[start:idx+1, df.columns.get_loc('label')] = -1  # -1 = Peak = SELL

    # Store peak/valley locations for visualization
    df['is_peak'] = 0
    df['is_valley'] = 0
    df.iloc[peak_idx, df.columns.get_loc('is_peak')] = 1
    df.iloc[valley_idx, df.columns.get_loc('is_valley')] = 1

    return df


# ============================================================================
# STEP 4: FEATURE ENGINEERING FOR ML
# ============================================================================

def create_ml_features(df: pd.DataFrame, oscillator_col: str = 'composite_smooth') -> pd.DataFrame:
    """
    Create features for ML model to predict oscillator peaks/valleys.
    """
    df = df.copy()
    osc = df[oscillator_col]

    # Oscillator-based features
    df['osc_value'] = osc
    df['osc_velocity'] = osc.diff(1)  # First derivative (speed)
    df['osc_acceleration'] = osc.diff(1).diff(1)  # Second derivative
    df['osc_jerk'] = osc.diff(1).diff(1).diff(1)  # Third derivative

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
        # Position within rolling range
        df[f'osc_position_{window}'] = (osc - df[f'osc_min_{window}']) / (df[f'osc_range_{window}'] + 1e-10)

    # Distance from extremes
    df['dist_from_1'] = 1 - osc  # Distance from +1 (overbought)
    df['dist_from_neg1'] = osc - (-1)  # Distance from -1 (oversold)

    # Momentum features
    df['osc_momentum_3'] = osc - osc.shift(3)
    df['osc_momentum_5'] = osc - osc.shift(5)
    df['osc_momentum_10'] = osc - osc.shift(10)

    # Trend features
    df['osc_above_mean_10'] = (osc > df['osc_mean_10']).astype(int)
    df['osc_above_mean_20'] = (osc > df['osc_mean_20']).astype(int)

    # Zero crossing detection
    df['crossed_zero'] = ((osc.shift(1) < 0) & (osc >= 0)) | ((osc.shift(1) > 0) & (osc <= 0))
    df['crossed_zero'] = df['crossed_zero'].astype(int)

    # Volatility of oscillator
    df['osc_volatility_10'] = osc.rolling(10).std()
    df['osc_volatility_20'] = osc.rolling(20).std()

    # Component indicators (already in df from create_composite_oscillator)
    component_cols = [c for c in df.columns if c.endswith('_norm') or c in ['bb_position', 'adx_trend']]

    # Component velocity
    for col in component_cols[:5]:  # Limit to avoid too many features
        df[f'{col}_velocity'] = df[col].diff(1)

    # Add novel indicators as features (if available)
    if NOVEL_INDICATORS_AVAILABLE:
        try:
            # Add all 8 novel oscillators as features
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

            # Add velocity (first derivative) of novel oscillators
            for novel_col in ['novel_arwo', 'novel_dco', 'novel_vcmo', 'novel_ics',
                              'novel_mji', 'novel_prf', 'novel_ewaf', 'novel_kfif']:
                df[f'{novel_col}_velocity'] = df[novel_col].diff(1)

            # Add cross-oscillator agreement features
            novel_cols = ['novel_arwo', 'novel_dco', 'novel_vcmo', 'novel_ics',
                          'novel_mji', 'novel_prf', 'novel_ewaf', 'novel_kfif']
            existing_novel = [c for c in novel_cols if c in df.columns]
            if len(existing_novel) >= 3:
                # Average of novel oscillators
                df['novel_consensus'] = df[existing_novel].mean(axis=1)
                # Standard deviation (disagreement)
                df['novel_dispersion'] = df[existing_novel].std(axis=1)
                # How many agree on direction (positive vs negative)
                df['novel_direction_agreement'] = (df[existing_novel] > 0).sum(axis=1) / len(existing_novel)

        except Exception as e:
            print(f"Warning: Could not add novel indicators to ML features: {e}")

    return df


# ============================================================================
# STEP 5: ML PIPELINE
# ============================================================================

def prepare_ml_data(df: pd.DataFrame, feature_cols: list = None,
                    test_size: float = 0.2, use_smote: bool = False) -> tuple:
    """
    Prepare data for ML with proper train/test split.
    Uses time-based split (no shuffling for time series).
    Optionally uses SMOTE for class imbalance.
    """
    df = df.copy()

    # Only drop rows where feature columns have NaN (not all columns)
    if feature_cols is not None:
        df = df.dropna(subset=feature_cols + ['label'])
    else:
        df = df.dropna()

    # Safety check
    if len(df) < 10:
        raise ValueError(f"Not enough data after cleaning: only {len(df)} rows")

    # Define feature columns
    if feature_cols is None:
        exclude_cols = ['label', 'is_peak', 'is_valley', 'open', 'high', 'low', 'close',
                        'volume', 'date', 'Date', 'composite_oscillator', 'composite_smooth']
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]

    X = df[feature_cols]
    y = df['label']

    # Time-based split (no shuffling!)
    split_idx = int(len(df) * (1 - test_size))

    # Ensure we have at least 1 sample in each set
    split_idx = max(1, min(split_idx, len(df) - 1))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Safety check for empty splits
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(f"Train/test split resulted in empty set. Train: {len(X_train)}, Test: {len(X_test)}")

    # Scale features BEFORE SMOTE
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE to training data only (after scaling, before converting back to DataFrame)
    smote_applied = False
    if use_smote and SMOTE_AVAILABLE:
        try:
            # Check if we have enough samples of each class
            class_counts = y_train.value_counts()
            min_class_count = class_counts.min()

            if min_class_count >= 2:  # SMOTE needs at least 2 samples per class
                smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_count - 1))
                X_train_scaled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
                y_train = pd.Series(y_train_resampled)
                smote_applied = True
        except Exception as e:
            print(f"SMOTE failed: {e}")

    # Convert back to DataFrames
    # Note: After SMOTE, we lose the original index for training data
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols, smote_applied


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                model_type: str = 'random_forest') -> object:
    """
    Train classification model.
    """
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=20,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, label_encoder=None) -> dict:
    """
    Evaluate model performance with multiple metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels (original format: -1, 0, 1)
        label_encoder: Optional LabelEncoder for XGBoost (decodes predictions back to -1, 0, 1)
    """
    y_pred = model.predict(X_test)

    # If label_encoder provided (XGBoost), decode predictions back to original labels
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred)

    # Handle case where model has predict_proba
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    return metrics


# ============================================================================
# STEP 5b: OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================================
# NOTE: The objective function is in optuna_worker.py - a lightweight module
# that avoids importing TensorFlow and other heavy dependencies in worker processes.


def optimize_model_with_optuna(X_train, y_train, X_val, y_val,
                                model_type: str, n_trials: int = 50,
                                progress_container=None) -> dict:
    """
    Optimize hyperparameters for a single model using Optuna.

    Uses optuna_worker module for proper multiprocessing without TensorFlow re-imports.
    Returns dict with best_params, best_score, study, and trained model.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not available. Install: pip install optuna")

    # Create study with in-memory storage
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    # Save shared data to temp file for parallel workers to access
    # This file-based approach works with multiprocessing (global vars don't)
    data_path = optuna_worker.set_shared_data(X_train, y_train, X_val, y_val, model_type)

    # Create picklable objective with data path embedded
    objective = optuna_worker.create_objective(data_path)

    # Show status before starting
    if progress_container:
        status = progress_container.empty()
        status.info(f"Running {n_trials} trials with {N_JOBS_OPTUNA} parallel workers...")

    # Progress tracking for console output
    import time
    last_print_time = [time.time()]  # Use list to allow modification in closure
    print_interval = 5  # Print every 5 seconds

    def progress_callback(study, trial):
        """Callback to print progress to console after each trial."""
        current_time = time.time()
        # Only print every few seconds to avoid flooding console
        if current_time - last_print_time[0] >= print_interval:
            last_print_time[0] = current_time

            completed = len([t for t in study.trials if t.state.name == 'COMPLETE'])
            failed = len([t for t in study.trials if t.state.name == 'FAIL'])
            remaining = n_trials - completed - failed

            best_val = study.best_value if study.best_trial else 0.0
            best_params_str = ""
            if study.best_trial:
                # Show key params
                bp = study.best_params
                if 'n_estimators' in bp:
                    best_params_str = f" | n_est={bp.get('n_estimators', '?')}, depth={bp.get('max_depth', '?')}"

            print(f"  [{model_type}] Trial {completed}/{n_trials} ({remaining} remaining) | "
                  f"Best F1: {best_val:.4f}{best_params_str}")

    try:
        # Suppress Optuna's verbose logging during optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(f"\n{'='*60}")
        print(f"Starting {model_type} optimization: {n_trials} trials, {N_JOBS_OPTUNA} workers")
        print(f"{'='*60}")

        # Use all available cores (N_JOBS_OPTUNA = N_CORES - 1)
        # The objective is a picklable class instance with the data path
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=N_JOBS_OPTUNA,
            show_progress_bar=False,  # Don't show in terminal, we have Streamlit UI
            catch=(Exception,),  # Catch exceptions to continue other trials
            callbacks=[progress_callback]  # Add progress callback
        )

        # Final summary
        completed = len([t for t in study.trials if t.state.name == 'COMPLETE'])
        failed = len([t for t in study.trials if t.state.name == 'FAIL'])
        best_val = study.best_value if study.best_trial else 0.0
        print(f"\n[{model_type}] COMPLETE: {completed} trials, {failed} failed | Best F1: {best_val:.4f}")
        if study.best_trial:
            print(f"[{model_type}] Best params: {study.best_params}")
        print(f"{'='*60}\n")

        optuna.logging.set_verbosity(optuna.logging.INFO)

        # Clear shared data after optimization
        optuna_worker.clear_shared_data()

    except Exception as e:
        # Clear shared data on error too
        optuna_worker.clear_shared_data()
        raise RuntimeError(f"Optimization failed: {e}")

    # Update status after completion
    if progress_container:
        completed = len([t for t in study.trials if t.state.name == 'COMPLETE'])
        best_val = study.best_value if study.best_trial else 0.0
        status.success(f"Completed {completed}/{n_trials} trials | Best F1: {best_val:.4f}")

    # Check if any trials completed
    completed_trials = [t for t in study.trials if t.state.name == 'COMPLETE' or t.value is not None]
    if not completed_trials:
        trial_states = {}
        for t in study.trials:
            state = t.state.name
            trial_states[state] = trial_states.get(state, 0) + 1
        raise RuntimeError(f"No trials completed. Trial states: {trial_states}")

    # Train final model with best parameters
    best_params = study.best_params.copy()
    label_encoder = None  # For XGBoost label encoding

    if model_type == 'random_forest':
        best_params['class_weight'] = 'balanced'
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        final_model = RandomForestClassifier(**best_params)
        final_model.fit(X_train, y_train)
    elif model_type == 'gradient_boosting':
        best_params['random_state'] = 42
        final_model = GradientBoostingClassifier(**best_params)
        final_model.fit(X_train, y_train)
    elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
        best_params['random_state'] = 42
        best_params['use_label_encoder'] = False
        best_params['eval_metric'] = 'mlogloss'
        best_params['verbosity'] = 0
        best_params['n_jobs'] = -1  # Use all cores for final model
        final_model = XGBClassifier(**best_params)

        # XGBoost requires 0-indexed labels, encode [-1, 0, 1] -> [0, 1, 2]
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        final_model.fit(X_train, y_train_encoded)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return {
        'model_type': model_type,
        'best_params': best_params,
        'best_score': study.best_value,
        'study': study,
        'model': final_model,
        'label_encoder': label_encoder,  # Store for decoding predictions
        'n_trials': n_trials
    }


def train_multiple_models_with_optuna(X_train, y_train, X_val, y_val, X_test, y_test,
                                       model_types: list, n_trials: int = 50,
                                       progress_placeholder=None) -> dict:
    """
    Train multiple model types with Optuna optimization.

    Returns dict with results for each model type.
    """
    results = {}

    # Console header
    print(f"\n{'#'*70}")
    print(f"# ML MODEL OPTIMIZATION - {len(model_types)} models x {n_trials} trials each")
    print(f"# Models: {', '.join(model_types)}")
    print(f"# Workers: {N_JOBS_OPTUNA} parallel")
    print(f"{'#'*70}\n")

    for i, model_type in enumerate(model_types):
        # Create a container for this model's progress
        if progress_placeholder:
            model_header = progress_placeholder.empty()
            model_header.info(f"**Optimizing {model_type}** ({i+1}/{len(model_types)}) - {n_trials} trials with {N_JOBS_OPTUNA} workers...")
            progress_container = progress_placeholder.container()
        else:
            progress_container = None

        print(f"\n>>> MODEL {i+1}/{len(model_types)}: {model_type.upper()} <<<")

        try:
            # Optimize (runs synchronously with full CPU utilization)
            result = optimize_model_with_optuna(
                X_train, y_train, X_val, y_val,
                model_type, n_trials, progress_container
            )

            # Evaluate on test set
            model = result['model']
            label_encoder = result.get('label_encoder')  # For XGBoost
            metrics = evaluate_model(model, X_test, y_test, label_encoder)
            result['test_metrics'] = metrics

            results[model_type] = result

            # Console output for completion
            print(f">>> {model_type.upper()} TEST RESULTS: F1={metrics['f1_macro']:.4f}, Acc={metrics['accuracy']:.4f}")

            # Show completion message
            if progress_placeholder:
                model_header.success(f"**{model_type}** complete! Val F1: {result['best_score']:.4f} | Test F1: {metrics['f1_macro']:.4f}")

        except Exception as e:
            results[model_type] = {'error': str(e)}
            print(f">>> {model_type.upper()} FAILED: {str(e)}")
            if progress_placeholder:
                model_header.error(f"**{model_type}** failed: {str(e)}")

    # Final summary
    print(f"\n{'#'*70}")
    print(f"# OPTIMIZATION COMPLETE - SUMMARY")
    print(f"{'#'*70}")
    best_model = None
    best_f1 = 0
    for model_name, result in results.items():
        if 'test_metrics' in result:
            f1 = result['test_metrics']['f1_macro']
            acc = result['test_metrics']['accuracy']
            val_f1 = result['best_score']
            print(f"  {model_name:20s}: Val F1={val_f1:.4f} | Test F1={f1:.4f} | Acc={acc:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_model = model_name
        else:
            print(f"  {model_name:20s}: FAILED - {result.get('error', 'unknown')}")

    if best_model:
        print(f"\n  BEST MODEL: {best_model} (Test F1: {best_f1:.4f})")
    print(f"{'#'*70}\n")

    return results


# ============================================================================
# STEP 5c: DEEP LEARNING FEATURES (Optional Enhancement)
# ============================================================================

def create_dl_sequences(df: pd.DataFrame, sequence_length: int = 20,
                        feature_cols: list = None) -> tuple:
    """
    Create sequences for LSTM/deep learning models.
    Returns X (sequences) and indices for alignment.
    """
    if feature_cols is None:
        feature_cols = ['composite_smooth', 'osc_velocity', 'osc_acceleration']
        feature_cols = [c for c in feature_cols if c in df.columns]

    data = df[feature_cols].values
    sequences = []
    indices = []

    for i in range(sequence_length, len(data)):
        sequences.append(data[i-sequence_length:i])
        indices.append(df.index[i])

    return np.array(sequences), indices


def create_lstm_features(df: pd.DataFrame, sequence_length: int = 20) -> pd.DataFrame:
    """
    Create LSTM-based features using a simple autoencoder approach.
    Falls back gracefully if TensorFlow is not available.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector
        tf.get_logger().setLevel('ERROR')
    except ImportError:
        return df  # Return unchanged if TF not available

    df = df.copy()

    # Select features for LSTM
    osc_cols = ['composite_smooth', 'osc_velocity', 'osc_acceleration',
                'osc_momentum_5', 'osc_position_10']
    osc_cols = [c for c in osc_cols if c in df.columns]

    if len(osc_cols) < 2:
        return df

    # Prepare sequences
    sequences, indices = create_dl_sequences(df, sequence_length, osc_cols)

    if len(sequences) < 100:
        return df  # Not enough data

    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    flat_data = sequences.reshape(-1, len(osc_cols))
    flat_scaled = scaler.fit_transform(flat_data)
    sequences_scaled = flat_scaled.reshape(sequences.shape)

    # Build simple LSTM autoencoder
    n_features = len(osc_cols)
    encoding_dim = 4

    # Encoder
    inputs = Input(shape=(sequence_length, n_features))
    encoded = LSTM(8, activation='tanh', return_sequences=False)(inputs)
    encoded = Dense(encoding_dim, activation='tanh')(encoded)

    # Decoder
    decoded = RepeatVector(sequence_length)(encoded)
    decoded = LSTM(8, activation='tanh', return_sequences=True)(decoded)
    decoded = Dense(n_features)(decoded)

    # Models
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    # Train (quick, just for feature extraction)
    autoencoder.fit(sequences_scaled, sequences_scaled,
                    epochs=10, batch_size=32, verbose=0, shuffle=False)

    # Extract embeddings
    embeddings = encoder.predict(sequences_scaled, verbose=0)

    # Add embeddings to dataframe
    embedding_df = pd.DataFrame(
        embeddings,
        index=indices,
        columns=[f'lstm_embed_{i}' for i in range(encoding_dim)]
    )

    # Merge back
    df = df.join(embedding_df, how='left')

    # Fill NaN values for the first sequence_length rows using backfill then forward fill
    for col in embedding_df.columns:
        # Use bfill() then ffill() to handle all NaN cases
        df[col] = df[col].bfill().ffill()
        # If still NaN (edge case), fill with 0
        df[col] = df[col].fillna(0)

    return df


# ============================================================================
# CHART UTILITIES FOR INTRADAY DATA
# ============================================================================

class ChartHelper:
    """
    Helper class for creating gap-free financial charts.

    For daily data: Uses datetime index directly
    For intraday data: Uses sequential integers with datetime labels

    Usage:
        helper = ChartHelper(df, interval)
        x_vals = helper.get_x(df.index)  # For dataframe series
        x_trade = helper.get_x(trade_date)  # For single date
        helper.apply_formatting(fig)  # Apply axis formatting
    """

    def __init__(self, df: pd.DataFrame, interval: str):
        self.df = df
        self.interval = interval
        self.is_intraday = interval != "1d"

        if self.is_intraday:
            # Create datetime to index mapping
            self.date_to_idx = {dt: i for i, dt in enumerate(df.index)}
        else:
            self.date_to_idx = None

    def get_x(self, dates):
        """
        Convert datetime(s) to appropriate x-axis values.

        Args:
            dates: Single datetime, list of datetimes, or DatetimeIndex

        Returns:
            Appropriate x values for plotting
        """
        if not self.is_intraday:
            return dates

        # Handle different input types
        if hasattr(dates, '__iter__') and not isinstance(dates, str):
            if hasattr(dates, 'tolist'):  # DatetimeIndex or Series
                return [self.date_to_idx.get(d, None) for d in dates]
            else:  # List
                return [self._lookup_date(d) for d in dates]
        else:
            # Single date
            return self._lookup_date(dates)

    def _lookup_date(self, dt):
        """Look up a date in the mapping, with fallback for close matches."""
        if dt in self.date_to_idx:
            return self.date_to_idx[dt]
        # Try to find closest date
        for ref_dt, idx in self.date_to_idx.items():
            if abs((ref_dt - dt).total_seconds()) < 3600:  # Within 1 hour
                return idx
        return None

    def apply_formatting(self, fig):
        """Apply axis formatting to the figure."""
        if not self.is_intraday:
            return

        # Set up tick labels showing dates/times
        n_ticks = min(12, len(self.df))
        step = max(1, len(self.df) // n_ticks)
        tickvals = list(range(0, len(self.df), step))
        ticktext = []
        for i in tickvals:
            if i < len(self.df):
                ticktext.append(self.df.index[i].strftime('%b %d\n%H:%M'))
        tickvals = tickvals[:len(ticktext)]

        fig.update_xaxes(
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext
        )


# Legacy function for backward compatibility
def apply_rangebreaks_to_figure(fig, interval: str, xaxis_name: str = "xaxis"):
    """Legacy function - now a no-op since we use sequential indexing."""
    pass


# ============================================================================
# STEP 6: STREAMLIT PAGE
# ============================================================================

def render_oscillator_predictor_page():
    """Main Streamlit page for Composite Oscillator ML Predictor"""

    st.title("Composite Oscillator ML Predictor")
    st.markdown("""
    **A simpler, more scientific approach:**
    1. Combine multiple indicators into a single composite oscillator (-1 to +1)
    2. Detect peaks/valleys in the oscillator using scipy
    3. Train ML to predict oscillator peaks/valleys (not price!)
    4. Generate trading signals from predictions
    """)

    # ========== SIDEBAR: DATA & PARAMETERS ==========
    st.sidebar.header("Settings")

    # Data settings
    ticker = st.sidebar.text_input("Ticker Symbol", value="SPY")
    st.session_state['ticker'] = ticker  # Store for access in sub-functions

    # Starting capital
    starting_capital = st.sidebar.number_input(
        "Starting Capital ($)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=1000,
        help="Initial capital for backtest calculations"
    )

    # Store in session state for access throughout
    st.session_state['starting_capital'] = starting_capital

    # Data period settings
    st.sidebar.subheader("Data Period")
    years = st.sidebar.slider("Years of Data", 1, 10, 5)

    # Bar interval/timeframe selector
    interval_options = {
        "1 Day": "1d",
        "12 Hours": "12h",  # Note: yfinance doesn't support 12h, we'll handle this
        "4 Hours": "4h",    # Note: yfinance doesn't support 4h directly
        "1 Hour": "1h",
        "15 Minutes": "15m"
    }
    interval_display = st.sidebar.selectbox(
        "Bar Length",
        list(interval_options.keys()),
        index=0,
        help="Timeframe for each bar. Note: Intraday data (< 1 day) is limited to 60 days for free data."
    )
    interval = interval_options[interval_display]

    # Store interval in session state
    st.session_state['data_interval'] = interval

    # Warning for intraday data limitations
    if interval != "1d":
        st.sidebar.warning(f"⚠️ Intraday data ({interval_display}) limited to ~60 days history with free yfinance API.")

    # Peak detection settings
    st.sidebar.subheader("Peak Detection")
    prominence = st.sidebar.slider("Prominence", 0.05, 0.5, 0.15, 0.05,
                                   help="Minimum prominence for peak/valley detection")
    distance = st.sidebar.slider("Min Distance", 3, 20, 5,
                                 help="Minimum bars between peaks/valleys")
    lookahead = st.sidebar.slider("Lookahead Labels", 1, 10, 5,
                                  help="Bars before peak/valley to label as signal")

    # Model settings
    st.sidebar.subheader("Model Settings")
    test_size = st.sidebar.slider("Test Size", 0.1, 0.7, 0.2, 0.05)

    # Check if key settings changed - if so, clear model state to prevent mismatch
    current_settings = f"{ticker}_{years}_{interval}_{prominence}_{distance}_{lookahead}_{test_size}"
    if 'osc_settings_hash' not in st.session_state:
        st.session_state['osc_settings_hash'] = current_settings
    elif st.session_state['osc_settings_hash'] != current_settings:
        # Settings changed - clear model state
        keys_to_clear = ['osc_model_trained', 'osc_best_model_name', 'osc_X_test',
                         'osc_y_test', 'osc_training_results', 'osc_df']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state['osc_settings_hash'] = current_settings

    # Optuna Hyperparameter Optimization
    st.sidebar.subheader("Hyperparameter Optimization")

    if not OPTUNA_AVAILABLE:
        st.sidebar.warning("Optuna not available. Install: pip install optuna")
        use_optuna = False
        n_trials = 50
        model_types_to_train = ['random_forest']
    else:
        use_optuna = st.sidebar.checkbox("Enable Optuna Optimization",
                                          value=True,
                                          help="Use Optuna to find best hyperparameters")

        if use_optuna:
            n_trials = st.sidebar.slider("Optimization Trials", 10, 1000, 50, 10,
                                          help="More trials = better params but slower")

            # Show parallelization info
            st.sidebar.caption(f"Using {N_JOBS_OPTUNA} parallel workers ({N_CORES} cores detected)")

            # Model selection for multi-model training
            available_models = ['random_forest', 'gradient_boosting']
            if XGBOOST_AVAILABLE:
                available_models.append('xgboost')

            model_types_to_train = st.sidebar.multiselect(
                "Models to Train",
                available_models,
                default=['random_forest', 'gradient_boosting'],
                help="Select models to optimize (trains each with Optuna)"
            )

            if not model_types_to_train:
                model_types_to_train = ['random_forest']
                st.sidebar.warning("At least one model required. Defaulting to Random Forest.")
        else:
            n_trials = 50
            model_type = st.sidebar.selectbox("Model Type",
                                               ["random_forest", "gradient_boosting"])
            model_types_to_train = [model_type]

    # Class imbalance handling
    st.sidebar.subheader("Class Imbalance")
    use_smote = st.sidebar.checkbox("Enable SMOTE",
                                     value=True,
                                     help="Use SMOTE to oversample minority classes (BUY/SELL signals)")
    if not SMOTE_AVAILABLE:
        st.sidebar.warning("SMOTE not available. Install: pip install imbalanced-learn")
        use_smote = False

    # Deep Learning settings
    st.sidebar.subheader("Deep Learning")
    use_dl_features = st.sidebar.checkbox("Enable LSTM Features",
                                           value=False,
                                           help="Add LSTM autoencoder embeddings (requires TensorFlow)")

    # ========== MAIN CONTENT ==========

    # Step 1: Load Data
    st.header("Step 1: Load Data")

    @st.cache_data(ttl=3600)
    def load_data(ticker: str, years: int, interval: str = "1d", include_today: bool = False):
        """
        Load data with specified interval.

        Args:
            include_today: If True, extends end_date to include today's data.
                           Should be True when market is closed for daily data.

        yfinance limitations:
        - 1d: unlimited history
        - 1h: up to 730 days
        - 15m, 30m: up to 60 days
        - 4h, 12h: not directly supported, resample from 1h
        """
        # yfinance end is EXCLUSIVE, add 1 day to include today when market is closed
        if include_today:
            end_date = datetime.now() + timedelta(days=1)
        else:
            end_date = datetime.now()

        # Adjust history based on interval limitations
        if interval in ["15m", "30m"]:
            # Max 60 days for minute data
            max_days = min(years * 365, 59)
            start_date = datetime.now() - timedelta(days=max_days)
            yf_interval = interval
        elif interval in ["1h"]:
            # Max 730 days for hourly data
            max_days = min(years * 365, 729)
            start_date = datetime.now() - timedelta(days=max_days)
            yf_interval = "1h"
        elif interval in ["4h", "12h"]:
            # Not directly supported - fetch 1h and resample
            max_days = min(years * 365, 729)
            start_date = datetime.now() - timedelta(days=max_days)
            yf_interval = "1h"
        else:
            # Daily data - unlimited
            start_date = datetime.now() - timedelta(days=years * 365)
            yf_interval = "1d"

        df = yf.download(ticker, start=start_date, end=end_date, interval=yf_interval, progress=False)
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns

        # Normalize column names to lowercase for consistency
        df.columns = df.columns.str.lower()

        # Resample if needed for 4h or 12h
        if interval == "4h" and not df.empty:
            df = df.resample('4h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        elif interval == "12h" and not df.empty:
            df = df.resample('12h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

        return df

    # Check market status to determine if today's bar is complete
    include_today = False
    market_status_str = ""
    if MARKET_UTILS_AVAILABLE:
        market_status = is_market_open(ticker)
        include_today = not market_status.get('is_open', True)  # Include today if market closed
        market_status_str = "CLOSED" if include_today else "OPEN"

    with st.spinner(f"Loading {ticker} data ({interval_display})..."):
        raw_data = load_data(ticker, years, interval, include_today=include_today)

    if raw_data.empty:
        st.error("Failed to load data. Check ticker symbol.")
        return

    # Display appropriate date/time info based on interval
    if interval == "1d":
        date_range = f"{raw_data.index[0].date()} to {raw_data.index[-1].date()}"
    else:
        date_range = f"{raw_data.index[0]} to {raw_data.index[-1]}"

    market_info = f" | Market: {market_status_str}" if market_status_str else ""
    st.success(f"Loaded {len(raw_data)} bars ({interval_display}) from {date_range}{market_info}")

    # Step 2: Create Composite Oscillator
    st.header("Step 2: Create Composite Oscillator")

    with st.spinner("Calculating indicators and composite oscillator..."):
        df = create_composite_oscillator(raw_data)

    # Store full dataframe in session state for save buttons to access
    st.session_state['df'] = df

    # Show component indicators
    component_cols = [c for c in df.columns if c.endswith('_norm') or c in ['bb_position', 'adx_trend']]

    with st.expander("View Component Indicators", expanded=False):
        st.write(f"**{len(component_cols)} normalized indicators:**")
        for col in component_cols:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                st.write(f"- {col}: range [{valid_data.min():.2f}, {valid_data.max():.2f}]")

    # Plot composite oscillator
    chart_osc = ChartHelper(df, interval)
    fig_osc = go.Figure()
    fig_osc.add_trace(go.Scatter(x=chart_osc.get_x(df.index), y=df['composite_smooth'],
                                  mode='lines', name='Composite Oscillator'))
    fig_osc.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_osc.add_hline(y=-0.5, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_osc.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_osc.update_layout(title="Composite Oscillator", height=300,
                          yaxis_title="Value (-1 to +1)", xaxis_title="Date")
    chart_osc.apply_formatting(fig_osc)
    st.plotly_chart(fig_osc, use_container_width=True)

    # Step 3: Detect Peaks & Valleys
    st.header("Step 3: Detect Peaks & Valleys in Oscillator")

    df = create_peak_valley_labels(df, 'composite_smooth', prominence, distance, lookahead)

    # Count peaks/valleys
    n_peaks = df['is_peak'].sum()
    n_valleys = df['is_valley'].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Peaks Detected", n_peaks)
    col2.metric("Valleys Detected", n_valleys)
    col3.metric("Total Signals", n_peaks + n_valleys)

    # Show label distribution (CORRECTED labels)
    label_counts = df['label'].value_counts().sort_index()
    st.write("**Label Distribution:**")
    st.write(f"- SELL (-1 = approaching peak, overbought): {label_counts.get(-1, 0)} ({label_counts.get(-1, 0)/len(df)*100:.1f}%)")
    st.write(f"- HOLD (0 = neutral): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
    st.write(f"- BUY (+1 = approaching valley, oversold): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")

    # ============================================================
    # PERFECT SIGNAL TRADING STATISTICS
    # What if we followed every oscillator signal exactly?
    # ============================================================
    st.subheader("Perfect Signal Trading Statistics")
    st.caption("If we bought at every valley and sold at every peak...")

    # Get valley and peak indices
    valley_indices = df.index[df['is_valley'] == 1].tolist()
    peak_indices = df.index[df['is_peak'] == 1].tolist()

    # Simulate perfect trading: alternate between valleys (buy) and peaks (sell)
    perfect_trades = []
    all_signals = []

    for idx in valley_indices:
        all_signals.append(('valley', idx, df.loc[idx, 'close']))
    for idx in peak_indices:
        all_signals.append(('peak', idx, df.loc[idx, 'close']))

    # Sort by date
    all_signals.sort(key=lambda x: x[1])

    # Execute trades
    position = None
    entry_price = None
    entry_date = None

    for signal_type, date, price in all_signals:
        if signal_type == 'valley' and position is None:
            # BUY at valley
            position = 'long'
            entry_price = price
            entry_date = date
        elif signal_type == 'peak' and position == 'long':
            # SELL at peak
            pnl_pct = ((price - entry_price) / entry_price) * 100
            hold_days = (date - entry_date).days
            perfect_trades.append({
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': date,
                'exit_price': price,
                'pnl_pct': pnl_pct,
                'hold_days': hold_days
            })
            position = None
            entry_price = None
            entry_date = None

    if perfect_trades:
        # Calculate statistics
        total_trades = len(perfect_trades)
        wins = [t for t in perfect_trades if t['pnl_pct'] > 0]
        losses = [t for t in perfect_trades if t['pnl_pct'] <= 0]
        win_rate = len(wins) / total_trades * 100

        all_pnls = [t['pnl_pct'] for t in perfect_trades]
        avg_pnl = np.mean(all_pnls)
        total_return = np.prod([1 + p/100 for p in all_pnls]) - 1
        total_return_pct = total_return * 100

        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        max_win = max(all_pnls)
        max_loss = min(all_pnls)
        avg_hold = np.mean([t['hold_days'] for t in perfect_trades])

        # Profit factor
        gross_profit = sum([t['pnl_pct'] for t in wins]) if wins else 0
        gross_loss = abs(sum([t['pnl_pct'] for t in losses])) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", total_trades)
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        col3.metric("Avg Trade", f"{avg_pnl:.2f}%")
        col4.metric("Total Return", f"{total_return_pct:.1f}%")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Avg Win", f"{avg_win:.2f}%")
        col6.metric("Avg Loss", f"{avg_loss:.2f}%")
        col7.metric("Profit Factor", f"{profit_factor:.2f}")
        col8.metric("Avg Hold (days)", f"{avg_hold:.1f}")

        col9, col10, col11, col12 = st.columns(4)
        col9.metric("Best Trade", f"{max_win:.2f}%")
        col10.metric("Worst Trade", f"{max_loss:.2f}%")
        col11.metric("Wins", len(wins))
        col12.metric("Losses", len(losses))

        # Buy & Hold comparison
        bh_return = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        data_years = (df.index[-1] - df.index[0]).days / 365
        st.write(f"**Buy & Hold Return:** {bh_return:.1f}% over {data_years:.1f} years")
        st.write(f"**Strategy vs B&H:** {total_return_pct - bh_return:+.1f}%")

        # Capital metrics row
        st.markdown("---")
        st.markdown("##### 💰 Capital Performance (Theoretical Perfect)")
        perf_ending_capital = starting_capital * (1 + total_return_pct / 100)
        perf_profit_loss = perf_ending_capital - starting_capital
        perf_bh_ending_capital = starting_capital * (1 + bh_return / 100)

        col13, col14, col15, col16 = st.columns(4)
        col13.metric("Starting Capital", f"${starting_capital:,.0f}")
        col14.metric("Ending Capital", f"${perf_ending_capital:,.0f}",
                    delta=f"${perf_profit_loss:+,.0f}")
        col15.metric("Buy & Hold Capital", f"${perf_bh_ending_capital:,.0f}")
        col16.metric("Strategy Advantage", f"${perf_ending_capital - perf_bh_ending_capital:+,.0f}")

        # Trade table
        with st.expander("View All Perfect Trades", expanded=False):
            trade_df = pd.DataFrame(perfect_trades)
            trade_df['entry_price'] = trade_df['entry_price'].apply(lambda x: f"${x:.2f}")
            trade_df['exit_price'] = trade_df['exit_price'].apply(lambda x: f"${x:.2f}")
            trade_df['pnl_pct'] = trade_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")
            trade_df['result'] = trade_df['pnl_pct'].apply(lambda x: '✅' if float(x.replace('%','')) > 0 else '❌')
            st.dataframe(trade_df, use_container_width=True)
    else:
        st.warning("No complete trades found (need alternating valleys and peaks)")

    st.markdown("---")

    # Visualize peaks and valleys
    chart_pv = ChartHelper(df, interval)
    x_vals_pv = chart_pv.get_x(df.index)

    fig_pv = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Price with peak/valley markers
    fig_pv.add_trace(go.Scatter(x=x_vals_pv, y=df['close'], mode='lines',
                                 name='Price', line=dict(color='blue')), row=1, col=1)

    # Mark peaks (sell points)
    peak_mask = df['is_peak'] == 1
    fig_pv.add_trace(go.Scatter(x=chart_pv.get_x(df.index[peak_mask]), y=df['close'][peak_mask],
                                 mode='markers', name='Oscillator Peak (SELL)',
                                 marker=dict(symbol='triangle-down', size=12, color='red')),
                     row=1, col=1)

    # Mark valleys (buy points)
    valley_mask = df['is_valley'] == 1
    fig_pv.add_trace(go.Scatter(x=chart_pv.get_x(df.index[valley_mask]), y=df['close'][valley_mask],
                                 mode='markers', name='Oscillator Valley (BUY)',
                                 marker=dict(symbol='triangle-up', size=12, color='green')),
                     row=1, col=1)

    # Oscillator
    fig_pv.add_trace(go.Scatter(x=x_vals_pv, y=df['composite_smooth'], mode='lines',
                                 name='Oscillator', line=dict(color='purple')), row=2, col=1)
    fig_pv.add_hline(y=0.5, line_dash="dash", line_color="red", row=2, col=1)
    fig_pv.add_hline(y=-0.5, line_dash="dash", line_color="green", row=2, col=1)

    fig_pv.update_layout(height=600, title="Price with Oscillator Peaks/Valleys")
    chart_pv.apply_formatting(fig_pv)
    st.plotly_chart(fig_pv, use_container_width=True)

    # Step 4: Create ML Features
    st.header("Step 4: Create ML Features")

    df = create_ml_features(df, 'composite_smooth')

    # Optional: Add Deep Learning features
    if use_dl_features:
        with st.spinner("Creating LSTM embeddings (this may take a moment)..."):
            try:
                rows_before = len(df)
                df = create_lstm_features(df, sequence_length=20)
                lstm_cols = [c for c in df.columns if c.startswith('lstm_embed')]
                if lstm_cols:
                    st.success(f"Added {len(lstm_cols)} LSTM embedding features")
                else:
                    st.warning("LSTM features not created (TensorFlow may not be available)")
            except Exception as e:
                st.warning(f"Could not create LSTM features: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Show data size before dropping NaN
    rows_before_dropna = len(df)
    nan_counts = df.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]

    df = df.dropna()
    rows_after_dropna = len(df)

    if rows_after_dropna < rows_before_dropna:
        st.caption(f"Dropped {rows_before_dropna - rows_after_dropna} rows with NaN values ({rows_after_dropna} remaining)")

    # Safety check
    if len(df) < 50:
        st.error(f"Not enough data after cleaning: only {len(df)} rows. Need at least 50.")
        st.write("Columns with NaN values (before drop):")
        st.write(cols_with_nan.head(10).to_dict())
        return

    # Get feature columns
    exclude_cols = ['label', 'is_peak', 'is_valley', 'open', 'high', 'low', 'close',
                    'volume', 'date', 'Date', 'composite_oscillator', 'composite_smooth']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    st.success(f"Created {len(feature_cols)} total features")

    with st.expander("View Features", expanded=False):
        st.write(feature_cols)

    # Step 5: Train/Test Split
    st.header("Step 5: Train/Test Split")

    try:
        X_train, X_test, y_train, y_test, scaler, feature_cols, smote_applied = prepare_ml_data(
            df, feature_cols, test_size, use_smote=use_smote
        )
    except ValueError as e:
        st.error(f"Error preparing data: {e}")
        st.write("**Debug info:**")
        st.write(f"- DataFrame shape: {df.shape}")
        st.write(f"- Feature columns: {len(feature_cols)}")
        st.write(f"- NaN in features: {df[feature_cols].isna().sum().sum()}")
        return
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Training Samples", len(X_train))
    col2.metric("Test Samples", len(X_test))
    col3.metric("SMOTE Applied", "Yes" if smote_applied else "No")

    if smote_applied:
        st.success("SMOTE applied to balance training classes")

    # Show class distribution in train/test (CORRECTED labels: -1=SELL, 0=HOLD, +1=BUY)
    train_dist = y_train.value_counts().sort_index()
    test_dist = y_test.value_counts().sort_index()

    st.write("**Class Distribution:**")
    dist_df = pd.DataFrame({
        'Train': train_dist,
        'Test': test_dist
    }).T
    dist_df.columns = ['SELL (-1)', 'HOLD (0)', 'BUY (+1)']
    st.dataframe(dist_df)

    # Store df and test indices for scipy backtest
    st.session_state['osc_df'] = df
    st.session_state['osc_X_test'] = X_test
    st.session_state['osc_y_test'] = y_test
    st.session_state['osc_scaler'] = scaler
    st.session_state['osc_features'] = feature_cols

    # ============================================================
    # SCIPY PEAKS BACKTEST (Always visible - no ML required)
    # ============================================================
    st.header("Step 5b: Scipy Peaks Backtest (No ML Required)")
    st.caption("This shows trading results using raw scipy peak/valley detection - no machine learning needed")

    # Get test period data for scipy backtest
    test_df_scipy = df.loc[X_test.index].copy()
    test_df_scipy['returns'] = test_df_scipy['close'].pct_change()

    # Run scipy peaks trading simulation
    scipy_position = 0
    scipy_positions = []
    scipy_trades = []
    scipy_entry_price = None
    scipy_entry_date = None

    for i in range(len(test_df_scipy)):
        date = test_df_scipy.index[i]
        price = test_df_scipy['close'].iloc[i]
        is_valley = test_df_scipy['is_valley'].iloc[i] == 1
        is_peak = test_df_scipy['is_peak'].iloc[i] == 1

        # BUY at valley
        if is_valley and scipy_position == 0:
            scipy_position = 1
            scipy_entry_price = price
            scipy_entry_date = date
            scipy_trades.append({'type': 'entry', 'date': date, 'price': price, 'signal': 'BUY'})
        # SELL at peak
        elif is_peak and scipy_position == 1:
            scipy_position = 0
            pnl = ((price - scipy_entry_price) / scipy_entry_price) * 100 if scipy_entry_price else 0
            scipy_trades.append({
                'type': 'exit', 'date': date, 'price': price, 'signal': 'SELL',
                'pnl': pnl, 'entry_date': scipy_entry_date, 'entry_price': scipy_entry_price
            })
            scipy_entry_price = None
            scipy_entry_date = None

        scipy_positions.append(scipy_position)

    test_df_scipy['position'] = scipy_positions
    test_df_scipy['strategy_returns'] = test_df_scipy['position'].shift(1) * test_df_scipy['returns']
    test_df_scipy['cum_market'] = (1 + test_df_scipy['returns']).cumprod()
    test_df_scipy['cum_strategy'] = (1 + test_df_scipy['strategy_returns'].fillna(0)).cumprod()

    scipy_exits = [t for t in scipy_trades if t['type'] == 'exit']

    if scipy_exits:
        scipy_pnls = [t.get('pnl', 0) for t in scipy_exits]
        scipy_wins = [t for t in scipy_exits if t.get('pnl', 0) > 0]
        scipy_losses = [t for t in scipy_exits if t.get('pnl', 0) <= 0]
        scipy_win_rate = len(scipy_wins) / len(scipy_exits) * 100
        scipy_avg_pnl = np.mean(scipy_pnls)
        scipy_total_return = (test_df_scipy['cum_strategy'].iloc[-1] - 1) * 100
        scipy_market_return = (test_df_scipy['cum_market'].iloc[-1] - 1) * 100

        # Profit factor
        scipy_gross_profit = sum([t['pnl'] for t in scipy_wins]) if scipy_wins else 0
        scipy_gross_loss = abs(sum([t['pnl'] for t in scipy_losses])) if scipy_losses else 1
        scipy_profit_factor = scipy_gross_profit / scipy_gross_loss if scipy_gross_loss > 0 else float('inf')

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", len(scipy_exits))
        col2.metric("Win Rate", f"{scipy_win_rate:.1f}%")
        col3.metric("Avg Trade", f"{scipy_avg_pnl:.2f}%")
        col4.metric("Profit Factor", f"{scipy_profit_factor:.2f}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Strategy Return", f"{scipy_total_return:.1f}%")
        col6.metric("Market Return", f"{scipy_market_return:.1f}%")
        col7.metric("Outperformance", f"{scipy_total_return - scipy_market_return:+.1f}%")
        col8.metric("Wins / Losses", f"{len(scipy_wins)} / {len(scipy_losses)}")

        # Capital metrics row
        st.markdown("---")
        st.markdown("##### 💰 Capital Performance")
        scipy_ending_capital = starting_capital * (1 + scipy_total_return / 100)
        scipy_profit_loss = scipy_ending_capital - starting_capital
        scipy_bh_ending_capital = starting_capital * (1 + scipy_market_return / 100)

        col9, col10, col11, col12 = st.columns(4)
        col9.metric("Starting Capital", f"${starting_capital:,.0f}")
        col10.metric("Ending Capital", f"${scipy_ending_capital:,.0f}",
                    delta=f"${scipy_profit_loss:+,.0f}")
        col11.metric("Buy & Hold Capital", f"${scipy_bh_ending_capital:,.0f}")
        col12.metric("Strategy Advantage", f"${scipy_ending_capital - scipy_bh_ending_capital:+,.0f}")

        # ============================================================
        # SCIPY PEAKS CANDLESTICK CHART
        # ============================================================
        st.subheader("Scipy Peaks Trading Chart")

        # Create chart helper for gap-free display
        chart = ChartHelper(test_df_scipy, interval)
        x_vals = chart.get_x(test_df_scipy.index)

        fig_scipy = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Price Action with Scipy Entries/Exits', 'Composite Oscillator with Peaks/Valleys', 'Cumulative Returns')
        )

        # Row 1: Candlestick Chart
        fig_scipy.add_trace(
            go.Candlestick(
                x=x_vals,
                open=test_df_scipy['open'],
                high=test_df_scipy['high'],
                low=test_df_scipy['low'],
                close=test_df_scipy['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )

        # Add entry markers (BUY at valleys)
        scipy_entries = [t for t in scipy_trades if t['type'] == 'entry']
        if scipy_entries:
            fig_scipy.add_trace(
                go.Scatter(
                    x=chart.get_x([t['date'] for t in scipy_entries]),
                    y=[t['price'] for t in scipy_entries],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=15, color='lime', line=dict(width=2, color='darkgreen')),
                    name='BUY (Valley)',
                    hovertemplate='BUY<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )

        # Add exit markers (SELL at peaks)
        if scipy_exits:
            fig_scipy.add_trace(
                go.Scatter(
                    x=chart.get_x([t['date'] for t in scipy_exits]),
                    y=[t['price'] for t in scipy_exits],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=15, color='red', line=dict(width=2, color='darkred')),
                    name='SELL (Peak)',
                    hovertemplate='SELL<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )

        # Draw trade lines connecting entries to exits
        for trade in scipy_exits:
            if 'entry_date' in trade and 'entry_price' in trade:
                color = 'rgba(0,255,0,0.3)' if trade.get('pnl', 0) > 0 else 'rgba(255,0,0,0.3)'
                fig_scipy.add_trace(
                    go.Scatter(
                        x=chart.get_x([trade['entry_date'], trade['date']]),
                        y=[trade['entry_price'], trade['price']],
                        mode='lines',
                        line=dict(color=color, width=2, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )

        # Row 2: Composite Oscillator with peaks/valleys
        osc_col = 'composite_smooth' if 'composite_smooth' in test_df_scipy.columns else 'composite_oscillator'
        fig_scipy.add_trace(
            go.Scatter(
                x=x_vals,
                y=test_df_scipy[osc_col],
                mode='lines',
                name='Oscillator',
                line=dict(color='purple', width=1.5)
            ),
            row=2, col=1
        )

        # Add threshold lines
        fig_scipy.add_hline(y=0.5, line_dash="dash", line_color="red", row=2, col=1)
        fig_scipy.add_hline(y=-0.5, line_dash="dash", line_color="green", row=2, col=1)
        fig_scipy.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

        # Mark peaks on oscillator
        peaks_df = test_df_scipy[test_df_scipy['is_peak'] == 1]
        if len(peaks_df) > 0:
            fig_scipy.add_trace(
                go.Scatter(
                    x=chart.get_x(peaks_df.index),
                    y=peaks_df[osc_col],
                    mode='markers',
                    marker=dict(symbol='circle', size=10, color='red'),
                    name='Peak (SELL)',
                    showlegend=True
                ),
                row=2, col=1
            )

        # Mark valleys on oscillator
        valleys_df = test_df_scipy[test_df_scipy['is_valley'] == 1]
        if len(valleys_df) > 0:
            fig_scipy.add_trace(
                go.Scatter(
                    x=chart.get_x(valleys_df.index),
                    y=valleys_df[osc_col],
                    mode='markers',
                    marker=dict(symbol='circle', size=10, color='lime'),
                    name='Valley (BUY)',
                    showlegend=True
                ),
                row=2, col=1
            )

        # Row 3: Cumulative Returns
        fig_scipy.add_trace(
            go.Scatter(
                x=x_vals,
                y=test_df_scipy['cum_market'],
                mode='lines',
                name='Buy & Hold',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
        fig_scipy.add_trace(
            go.Scatter(
                x=x_vals,
                y=test_df_scipy['cum_strategy'],
                mode='lines',
                name='Scipy Strategy',
                line=dict(color='orange', width=2)
            ),
            row=3, col=1
        )

        # Update layout
        fig_scipy.update_layout(
            height=900,
            title_text="Scipy Peaks Trading Analysis",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False
        )

        fig_scipy.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_scipy.update_yaxes(title_text="Oscillator", row=2, col=1, range=[-1.2, 1.2])
        fig_scipy.update_yaxes(title_text="Cum. Return", row=3, col=1)

        # Apply formatting for proper date labels (especially for intraday)
        chart.apply_formatting(fig_scipy)
        st.plotly_chart(fig_scipy, use_container_width=True)

        # Trade log expander
        with st.expander("View Scipy Peaks Trade Log", expanded=False):
            scipy_trade_summary = []
            for i, t in enumerate(scipy_exits):
                hold_d = (t['date'] - t['entry_date']).days if t.get('entry_date') else 0
                scipy_trade_summary.append({
                    '#': i + 1,
                    'Entry': t.get('entry_date', 'N/A'),
                    'Entry $': f"${t.get('entry_price', 0):.2f}",
                    'Exit': t['date'],
                    'Exit $': f"${t['price']:.2f}",
                    'P&L': f"{t.get('pnl', 0):.2f}%",
                    'Days': hold_d,
                    'Result': '✅' if t.get('pnl', 0) > 0 else '❌'
                })
            st.dataframe(pd.DataFrame(scipy_trade_summary), use_container_width=True)
    else:
        st.warning("No completed scipy trades in test period")

    # ============================================================
    # STEP 5c: VELOCITY-BASED TRADING (Real-Time Capable)
    # ============================================================
    st.markdown("---")
    st.header("Step 5c: Velocity-Based Trading (Live Trading Capable)")
    st.caption("Uses oscillator velocity & acceleration to detect reversals in REAL-TIME - no look-ahead bias")

    # Auto-apply best parameters if flagged (MUST be before sliders render)
    if st.session_state.get('vel_apply_best', False):
        best_params = st.session_state.get('vel_best_params', {})
        if best_params:
            # Map best params to slider session state keys
            signal_type_options = ['velocity_crossover_and_zone', 'velocity_crossover_or_zone', 'zone_only',
                                   'momentum', 'any_reversal', 'double_bottom', 'divergence', 'breakout']
            if best_params.get('signal_type') in signal_type_options:
                st.session_state['signal_type'] = best_params['signal_type']
            if 'extreme_zone_mult' in best_params:
                st.session_state['extreme_mult'] = float(best_params['extreme_zone_mult'])
            if 'vel_smoothing' in best_params:
                st.session_state['vel_smooth'] = int(best_params['vel_smoothing'])
            if 'min_bars_between' in best_params:
                st.session_state['vel_min_bars'] = int(best_params['min_bars_between'])
            if 'require_accel' in best_params:
                st.session_state['vel_accel'] = bool(best_params['require_accel'])
            if 'oversold_threshold' in best_params:
                st.session_state['vel_oversold'] = float(best_params['oversold_threshold'])
            if 'overbought_threshold' in best_params:
                st.session_state['vel_overbought'] = float(best_params['overbought_threshold'])
            if 'stop_loss_pct' in best_params:
                st.session_state['vel_sl'] = float(best_params['stop_loss_pct'])
            if 'take_profit_pct' in best_params:
                st.session_state['vel_tp'] = float(best_params['take_profit_pct'])
            # Exit strategies
            if 'exit_on_opposite_signal' in best_params:
                st.session_state['vel_exit_opposite'] = bool(best_params['exit_on_opposite_signal'])
            if 'exit_on_midline_cross' in best_params:
                st.session_state['vel_exit_midline'] = bool(best_params['exit_on_midline_cross'])
            # RSI settings
            if 'rsi_filter' in best_params:
                st.session_state['vel_rsi_filter'] = best_params['rsi_filter']
            if 'rsi_period' in best_params:
                st.session_state['vel_rsi_period'] = int(best_params['rsi_period'])
            if 'rsi_oversold' in best_params:
                st.session_state['vel_rsi_os'] = int(best_params['rsi_oversold'])
            if 'rsi_overbought' in best_params:
                st.session_state['vel_rsi_ob'] = int(best_params['rsi_overbought'])
            # MACD and BB
            if 'use_macd_confirm' in best_params:
                st.session_state['vel_macd'] = bool(best_params['use_macd_confirm'])
            if 'use_bb_filter' in best_params:
                st.session_state['vel_bb'] = bool(best_params['use_bb_filter'])
        # Clear the flag
        st.session_state['vel_apply_best'] = False

    # Velocity trading parameters - organized in expanders
    with st.expander("Trading Parameters", expanded=True):
        # Oscillator Type Selection (Novel indicators)
        st.markdown("**Oscillator Selection**")
        osc_type_col1, osc_type_col2 = st.columns([2, 1])
        with osc_type_col1:
            oscillator_options = ['composite_smooth (Default)']
            if NOVEL_INDICATORS_AVAILABLE:
                oscillator_options.extend([
                    'ARWO (Adaptive Regime-Weighted)',
                    'DCO (Divergence Consensus)',
                    'VCMO (Volume-Confirmed Momentum)',
                    'ICS (Indicator Convergence Score)',
                    'MJI (Momentum Jerk)',
                    'PRF (Percentile Rank Fusion)',
                    'EWAF (Entropy-Weighted Adaptive Fusion)',
                    'KFIF (Kalman-Filtered Indicator Fusion)'
                ])
            selected_oscillator = st.selectbox(
                "Oscillator Type",
                options=oscillator_options,
                index=0,
                key="vel_oscillator_type",
                help="Select which oscillator to use for signal generation. Novel oscillators may capture different market dynamics."
            )
        with osc_type_col2:
            if not NOVEL_INDICATORS_AVAILABLE:
                st.warning("Novel indicators not available")
            else:
                st.success(f"8 novel oscillators available")

        st.markdown("---")
        # Row 0: Signal Logic Type (NEW - critical for generating more trades)
        st.markdown("**Entry Signal Logic**")
        signal_logic_col1, signal_logic_col2 = st.columns(2)
        with signal_logic_col1:
            signal_type = st.selectbox(
                "Entry Signal Type",
                options=[
                    'velocity_crossover_and_zone',
                    'velocity_crossover_or_zone',
                    'zone_only',
                    'momentum',
                    'any_reversal',
                    'double_bottom',
                    'divergence',
                    'breakout'
                ],
                index=0,
                format_func=lambda x: {
                    'velocity_crossover_and_zone': 'Velocity Crossover AND Zone (Strictest)',
                    'velocity_crossover_or_zone': 'Velocity Crossover OR Extreme Zone (Balanced)',
                    'zone_only': 'Extreme Zone Only (Mean Reversion)',
                    'momentum': 'Strong Momentum (Trend Following)',
                    'any_reversal': 'Any Reversal Signal (Most Aggressive)',
                    'double_bottom': 'Double Bottom/Top (Pattern)',
                    'divergence': 'Price/Oscillator Divergence',
                    'breakout': 'Threshold Breakout'
                }.get(x, x),
                key="signal_type",
                help="Controls how entry signals are generated. Less strict = more trades."
            )
        with signal_logic_col2:
            extreme_zone_mult = st.slider("Extreme Zone Multiplier", 1.1, 2.5, 1.5, 0.1, key="extreme_mult",
                                          help="For zone_only: multiply threshold for 'extreme' zone (e.g., 1.5x = -0.45 if threshold is -0.3)")

        # Row 1: Basic settings
        vel_row1_col1, vel_row1_col2, vel_row1_col3, vel_row1_col4 = st.columns(4)
        with vel_row1_col1:
            vel_smoothing = st.slider("Velocity Smoothing", 1, 15, 3, key="vel_smooth",
                                      help="Smooth velocity to reduce noise")
        with vel_row1_col2:
            min_bars_between = st.slider("Min Bars Between Trades", 1, 15, 5, key="vel_min_bars",
                                         help="Minimum bars between consecutive trades")
        with vel_row1_col3:
            require_accel = st.checkbox("Require Acceleration Confirmation", value=False, key="vel_accel",
                                        help="Only trade when acceleration confirms the reversal")
        with vel_row1_col4:
            use_trailing_stop = st.checkbox("Use Trailing Stop", value=False, key="vel_trail",
                                            help="Use trailing stop instead of fixed stop loss")

        # Row 2: Asymmetric oscillator thresholds
        st.markdown("**Oscillator Thresholds (Asymmetric)**")
        vel_row2_col1, vel_row2_col2 = st.columns(2)
        with vel_row2_col1:
            oversold_threshold = st.slider("Oversold Threshold (BUY zone)", -0.6, -0.02, -0.2, 0.02,
                                           key="vel_oversold",
                                           help="BUY when oscillator below this level (more negative = more oversold)")
        with vel_row2_col2:
            overbought_threshold = st.slider("Overbought Threshold (SELL zone)", 0.02, 0.6, 0.2, 0.02,
                                             key="vel_overbought",
                                             help="SELL when oscillator above this level")

        # Row 3: Stop loss and take profit
        st.markdown("**Risk Management**")
        vel_row3_col1, vel_row3_col2, vel_row3_col3 = st.columns(3)
        with vel_row3_col1:
            stop_loss_pct = st.slider("Stop Loss %", 0.0, 10.0, 2.0, 0.1, key="vel_sl",
                                      help="Exit if price drops this % below entry (0 = disabled)")
        with vel_row3_col2:
            take_profit_pct = st.slider("Take Profit %", 0.0, 20.0, 0.0, 0.1, key="vel_tp",
                                        help="Exit if price rises this % above entry (0 = disabled)")
        with vel_row3_col3:
            trailing_stop_pct = st.slider("Trailing Stop %", 0.5, 10.0, 2.0, 0.1, key="vel_trail_pct",
                                          help="Trail stop this % below highest price (only if trailing stop enabled)")

        # Row 4: Advanced filters
        st.markdown("**Signal Filters**")
        vel_row4_col1, vel_row4_col2 = st.columns(2)
        with vel_row4_col1:
            vel_threshold = st.slider("Min Velocity Magnitude", 0.0, 0.1, 0.0, 0.005, key="vel_mag",
                                      help="Minimum velocity magnitude to trigger signal (0 = any)")
        with vel_row4_col2:
            accel_threshold = st.slider("Min Acceleration Magnitude", 0.0, 0.05, 0.0, 0.005, key="vel_accel_mag",
                                        help="Minimum acceleration magnitude (0 = any)")

        # Row 5: Exit Strategies
        st.markdown("**Exit Strategies**")
        vel_row5_col1, vel_row5_col2 = st.columns(2)
        with vel_row5_col1:
            exit_on_opposite_signal = st.checkbox("Exit on Opposite Signal", value=True, key="vel_exit_opposite",
                                                   help="Exit long when sell signal triggers")
        with vel_row5_col2:
            exit_on_midline_cross = st.checkbox("Exit on Midline Cross", value=False, key="vel_exit_midline",
                                                 help="Exit when oscillator crosses above 0 (for longs)")

        # Row 6: Extra Indicators (RSI, MACD, BB)
        st.markdown("**Extra Indicator Filters**")
        vel_row6_col1, vel_row6_col2, vel_row6_col3, vel_row6_col4 = st.columns(4)
        with vel_row6_col1:
            rsi_filter = st.selectbox("RSI Filter", options=['none', 'oversold_only', 'overbought_only', 'both'],
                                      index=0, key="vel_rsi_filter",
                                      format_func=lambda x: {'none': 'None', 'oversold_only': 'Oversold Only',
                                                            'overbought_only': 'Overbought Only', 'both': 'Both'}.get(x, x))
        with vel_row6_col2:
            rsi_period = st.slider("RSI Period", 5, 30, 14, 1, key="vel_rsi_period")
        with vel_row6_col3:
            rsi_oversold = st.slider("RSI Oversold", 15, 40, 30, 1, key="vel_rsi_os")
        with vel_row6_col4:
            rsi_overbought = st.slider("RSI Overbought", 60, 85, 70, 1, key="vel_rsi_ob")

        vel_row7_col1, vel_row7_col2 = st.columns(2)
        with vel_row7_col1:
            use_macd_confirm = st.checkbox("MACD Confirmation", value=False, key="vel_macd",
                                           help="Require MACD histogram improving for buys")
        with vel_row7_col2:
            use_bb_filter = st.checkbox("Bollinger Band Filter", value=False, key="vel_bb",
                                        help="Require price below lower band for buys")

    # Optuna-Based Smart Optimization (Bayesian + Parallel)
    with st.expander("Smart Parameter Optimization (Optuna)", expanded=False):
        st.markdown("**Bayesian optimization (TPE) - much smarter than random search!**")

        grid_col1, grid_col2, grid_col3, grid_col4 = st.columns(4)
        with grid_col1:
            grid_metric = st.selectbox("Optimize For", ["Total Return", "Win Rate", "Profit Factor", "Sharpe Ratio", "Risk-Adjusted Return"],
                                       key="grid_metric")
        with grid_col2:
            grid_iterations = st.slider("Trials to Run", 1000, 500000, 10000, 1000, key="grid_iter",
                                        help="TPE finds great solutions quickly - 5k-20k usually enough")
        with grid_col3:
            n_workers = st.slider("Parallel Studies", 1, 32, min(N_CORES, 16), key="vel_n_jobs",
                                  help=f"Run independent studies in parallel ({N_CORES} CPU cores)")
        with grid_col4:
            use_extra_indicators = st.checkbox("Extra Indicators", value=True, key="use_extra_ind",
                                               help="RSI, MACD, Bollinger Bands")

        # Show time estimate based on historical data
        time_estimate = estimate_optimization_time(grid_iterations, n_workers)
        timing_data = load_optimization_timing()
        if timing_data.get("avg_trials_per_minute"):
            st.caption(f"⏱️ **Estimated time: {time_estimate}** ({int(timing_data['avg_trials_per_minute']):,} trials/min avg)")
        else:
            st.caption("⏱️ Run an optimization to get time estimates")

        if st.button("🚀 Run Smart Optimization", type="primary", key="run_grid"):
          try:
            import optuna

            # Suppress Optuna logs
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            # Prepare test data once
            test_df_grid = df.loc[X_test.index].copy()
            osc_col = 'composite_smooth' if 'composite_smooth' in test_df_grid.columns else 'composite_oscillator'
            close_prices = test_df_grid['close'].values
            osc_values = test_df_grid[osc_col].values

            # Pre-calculate ALL oscillator types for optimization search
            all_oscillators = {'composite_smooth': osc_values}

            if NOVEL_INDICATORS_AVAILABLE:
                st.info("🔬 Pre-calculating novel oscillators for optimization search...")
                try:
                    all_oscillators['arwo'] = calculate_arwo(test_df_grid).values
                    all_oscillators['dco'] = calculate_dco(test_df_grid).values
                    all_oscillators['vcmo'] = calculate_vcmo(test_df_grid).values
                    all_oscillators['ics'] = calculate_ics(test_df_grid).values
                    all_oscillators['mji'] = calculate_mji(test_df_grid).values
                    all_oscillators['prf'] = calculate_prf(test_df_grid).values
                    all_oscillators['ewaf'] = calculate_ewaf(test_df_grid).values
                    kfif_val, _, _ = calculate_kfif(test_df_grid)
                    all_oscillators['kfif'] = kfif_val.values
                    st.success(f"✅ Pre-calculated {len(all_oscillators)} oscillator types for search")
                except Exception as e:
                    st.warning(f"Could not calculate some novel oscillators: {e}")

            # Pre-calculate indicators
            def calc_rsi_np(prices, period):
                delta = np.diff(prices, prepend=prices[0])
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                avg_gain = pd.Series(gain).rolling(period).mean().values
                avg_loss = pd.Series(loss).rolling(period).mean().values
                rs = avg_gain / (avg_loss + 1e-10)
                return 100 - (100 / (1 + rs))

            rsi_cache = {p: calc_rsi_np(close_prices, p) for p in [5, 7, 10, 14, 21, 30]}

            # MACD
            close_series = pd.Series(close_prices)
            ema12 = close_series.ewm(span=12, adjust=False).mean().values
            ema26 = close_series.ewm(span=26, adjust=False).mean().values
            macd_histogram = (ema12 - ema26) - pd.Series(ema12 - ema26).ewm(span=9, adjust=False).mean().values

            # Bollinger Bands
            bb_sma = pd.Series(close_prices).rolling(20).mean().values
            bb_std = pd.Series(close_prices).rolling(20).std().values
            bb_upper = bb_sma + 2 * bb_std
            bb_lower = bb_sma - 2 * bb_std

            # Metric mapping
            metric_map = {
                "Total Return": "total_return",
                "Win Rate": "win_rate",
                "Profit Factor": "profit_factor",
                "Sharpe Ratio": "sharpe_ratio",
                "Risk-Adjusted Return": "risk_adjusted"
            }
            optimize_metric = metric_map.get(grid_metric, "total_return")

            # Save shared data for parallel workers (file-based sharing)
            data_path = optuna_worker.set_velocity_shared_data(
                close_prices, osc_values, rsi_cache, macd_histogram, bb_upper, bb_lower,
                optimize_metric=optimize_metric, use_extra_indicators=use_extra_indicators,
                all_oscillators=all_oscillators  # Pass all oscillator types for search
            )

            # Progress display
            progress_bar = st.progress(0)
            status_text = st.empty()

            st.info(f"🚀 Running {grid_iterations:,} trials across {n_workers} parallel studies...")

            try:
                from joblib import Parallel, delayed

                # Divide trials among workers
                trials_per_worker = grid_iterations // n_workers

                # Console output
                opt_start_date = test_df_grid.index[0].strftime('%Y-%m-%d')
                opt_end_date = test_df_grid.index[-1].strftime('%Y-%m-%d')
                opt_period_days = (test_df_grid.index[-1] - test_df_grid.index[0]).days
                print(f"\n{'='*60}", flush=True)
                print(f"🚀 VELOCITY OPTIMIZATION STARTING", flush=True)
                print(f"   Total trials: {grid_iterations:,} ({trials_per_worker:,} per worker)", flush=True)
                print(f"   Parallel workers: {n_workers}", flush=True)
                print(f"   Optimizing for: {optimize_metric}", flush=True)
                print(f"   Date range: {opt_start_date} to {opt_end_date} ({opt_period_days} days, {len(test_df_grid)} bars)", flush=True)
                print(f"{'='*60}", flush=True)

                # Run parallel studies using joblib with loky backend (uses spawn)
                status_text.text(f"Starting {n_workers} parallel Optuna studies (see terminal for progress)...")

                # Start timing
                optimization_start_time = time.time()

                # Each worker gets a different random seed for diversity
                results_lists = Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
                    delayed(optuna_worker.run_velocity_study)(
                        data_path, trials_per_worker, seed=42 + i, optimize_metric=optimize_metric, worker_id=i
                    )
                    for i in range(n_workers)
                )

                # End timing
                optimization_duration = time.time() - optimization_start_time

                # Merge all results
                all_results = []
                for result_list in results_lists:
                    all_results.extend(result_list)

                progress_bar.progress(1.0)
                status_text.text(f"Completed! Tested {len(all_results):,} parameter combinations.")

                # Save timing data
                trials_per_min = save_optimization_timing(len(all_results), optimization_duration, n_workers)

                # Console completion summary
                print(f"\n{'='*60}", flush=True)
                print(f"✅ OPTIMIZATION COMPLETE", flush=True)
                print(f"   Total results: {len(all_results):,}", flush=True)
                # Show date period for the optimization
                start_date = test_df_grid.index[0].strftime('%Y-%m-%d')
                end_date = test_df_grid.index[-1].strftime('%Y-%m-%d')
                period_days = (test_df_grid.index[-1] - test_df_grid.index[0]).days
                print(f"   Date range: {start_date} to {end_date} ({period_days} days, {len(test_df_grid)} bars)", flush=True)
                # Show timing info
                print(f"   Duration: {optimization_duration:.1f}s ({trials_per_min:.0f} trials/min)", flush=True)
                if all_results:
                    best = max(all_results, key=lambda x: x.get(optimize_metric, 0))
                    print(f"   Best {optimize_metric}: {best.get(optimize_metric, 0):.2f}", flush=True)
                print(f"{'='*60}\n", flush=True)

            finally:
                # Always clean up shared data
                optuna_worker.clear_velocity_shared_data()

            if all_results:
                results_df = pd.DataFrame(all_results)

                # Sort by selected metric
                results_df = results_df.sort_values(optimize_metric, ascending=False)

                # Store in session state
                st.session_state['vel_grid_results'] = results_df

                # Show summary stats
                best = results_df.iloc[0]
                st.success(f"✅ Found {len(results_df):,} valid combinations! Best: {best['total_return']:.1f}% return, {int(best['num_trades'])} trades, {best['win_rate']:.0f}% win rate")

                # Show optimization insights
                st.markdown("**Optimization Insights:**")
                st.write(f"- Optuna explored the search space using Bayesian optimization (TPE)")
                st.write(f"- {n_workers} parallel studies, {trials_per_worker:,} trials each")
                st.write(f"- Top signal types: {results_df['signal_type'].value_counts().head(3).to_dict()}")
                if 'oscillator_type' in results_df.columns:
                    st.write(f"- Top oscillator types: {results_df['oscillator_type'].value_counts().head(3).to_dict()}")
            else:
                st.warning("No valid results found. Try adjusting parameters or running more trials.")

          except Exception as e:
            st.error(f"Optimization error: {e}")
            import traceback
            st.code(traceback.format_exc())

        # Display grid search results
        if 'vel_grid_results' in st.session_state:
            results_df = st.session_state['vel_grid_results']

            st.subheader(f"Top 10 of {len(results_df):,} Parameter Combinations")
            # Core columns first, then extra indicators
            display_cols = [
                'oscillator_type', 'signal_type', 'vel_smoothing', 'oversold_threshold', 'overbought_threshold',
                'stop_loss_pct', 'take_profit_pct', 'min_bars_between',
                'total_return', 'win_rate', 'profit_factor', 'num_trades', 'risk_adjusted'
            ]
            # Add extra indicator columns if they exist
            extra_cols = ['rsi_filter', 'rsi_period', 'use_macd_confirm', 'use_bb_filter',
                         'exit_on_opposite_signal', 'exit_on_midline_cross', 'extreme_zone_mult']
            display_cols.extend([c for c in extra_cols if c in results_df.columns])
            # Filter to columns that exist
            display_cols = [c for c in display_cols if c in results_df.columns]

            # Format the dataframe for better display
            display_df = results_df[display_cols].head(10).copy()
            if 'total_return' in display_df.columns:
                display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x:.1f}%")
            if 'win_rate' in display_df.columns:
                display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.0f}%")
            if 'profit_factor' in display_df.columns:
                display_df['profit_factor'] = display_df['profit_factor'].apply(lambda x: f"{x:.2f}")

            st.dataframe(display_df, use_container_width=True)

            # Buttons to apply parameters
            apply_col1, apply_col2, apply_col3 = st.columns([2, 2, 3])

            # Helper function to extract all params from a row
            def extract_all_params(row):
                return {
                    'signal_type': row.get('signal_type', 'velocity_crossover_and_zone'),
                    'vel_smoothing': int(row['vel_smoothing']),
                    'oversold_threshold': row['oversold_threshold'],
                    'overbought_threshold': row['overbought_threshold'],
                    'stop_loss_pct': row['stop_loss_pct'],
                    'take_profit_pct': row['take_profit_pct'],
                    'min_bars_between': int(row['min_bars_between']),
                    'require_accel': row.get('require_accel', False),
                    'extreme_zone_mult': row.get('extreme_zone_mult', 1.5),
                    # Exit strategies
                    'exit_on_opposite_signal': row.get('exit_on_opposite_signal', True),
                    'exit_on_midline_cross': row.get('exit_on_midline_cross', False),
                    # RSI settings
                    'rsi_filter': row.get('rsi_filter', 'none'),
                    'rsi_period': int(row.get('rsi_period', 14)),
                    'rsi_oversold': int(row.get('rsi_oversold', 30)),
                    'rsi_overbought': int(row.get('rsi_overbought', 70)),
                    # MACD and BB
                    'use_macd_confirm': row.get('use_macd_confirm', False),
                    'use_bb_filter': row.get('use_bb_filter', False),
                    # Novel oscillator type
                    'oscillator_type': row.get('oscillator_type', st.session_state.get('selected_oscillator_type', 'composite_smooth')),
                }

            with apply_col1:
                if st.button("🎯 Auto-Apply #1 (Best)", type="primary", key="apply_best"):
                    best = results_df.iloc[0]
                    st.session_state['vel_best_params'] = extract_all_params(best)
                    st.session_state['vel_apply_best'] = True  # Flag to trigger auto-apply
                    st.rerun()

            with apply_col2:
                # Allow selecting which rank to apply
                apply_rank = st.selectbox("Or select rank to apply:",
                                         options=list(range(1, min(11, len(results_df) + 1))),
                                         format_func=lambda x: f"#{x} - {results_df.iloc[x-1]['total_return']:.1f}% return, {int(results_df.iloc[x-1]['num_trades'])} trades",
                                         key="apply_rank_select")

            with apply_col3:
                if st.button(f"Apply #{apply_rank} Settings", key="apply_selected"):
                    selected = results_df.iloc[apply_rank - 1]
                    st.session_state['vel_best_params'] = extract_all_params(selected)
                    st.session_state['vel_apply_best'] = True  # Flag to trigger auto-apply
                    st.rerun()

            # Show currently applied params if any
            if 'vel_best_params' in st.session_state and st.session_state['vel_best_params']:
                bp = st.session_state['vel_best_params']
                extra_info = []
                if bp.get('rsi_filter', 'none') != 'none':
                    extra_info.append(f"RSI={bp.get('rsi_filter')}")
                if bp.get('use_macd_confirm'):
                    extra_info.append("MACD")
                if bp.get('use_bb_filter'):
                    extra_info.append("BB")
                extra_str = f" | Filters: {', '.join(extra_info)}" if extra_info else ""
                st.success(f"✅ Applied: Signal={bp.get('signal_type', 'N/A')}, "
                          f"Thresholds={bp.get('oversold_threshold', 'N/A')}/{bp.get('overbought_threshold', 'N/A')}, "
                          f"SL={bp.get('stop_loss_pct', 'N/A')}%, TP={bp.get('take_profit_pct', 'N/A')}%{extra_str}")

            # === SAVE & DEPLOY SECTION ===
            st.markdown("---")
            st.subheader("💾 Save & Deploy Strategy")

            save_col1, save_col2, save_col3 = st.columns([2, 2, 2])

            with save_col1:
                strategy_name = st.text_input("Strategy Name", value=f"velocity_{ticker}_{signal_type}", key="vel_strat_name")

            with save_col2:
                # Default webhook (same as live_trader.py)
                default_webhook = ""
                discord_webhook = st.text_input("Discord Webhook URL", value=default_webhook, key="vel_discord_webhook", type="password")

            with save_col3:
                st.write("")  # Spacer
                st.write("")  # Spacer for alignment

            deploy_col1, deploy_col2 = st.columns(2)

            with deploy_col1:
                if st.button("🚀 Deploy to Discord Bot", type="primary", key="deploy_vel_strat"):
                    if not discord_webhook:
                        st.error("Please enter a Discord Webhook URL to deploy")
                    else:
                        # Get currently applied params or use current UI settings
                        if 'vel_best_params' in st.session_state and st.session_state['vel_best_params']:
                            params = st.session_state['vel_best_params']
                        else:
                            params = {
                                'signal_type': signal_type,
                                'extreme_zone_mult': extreme_zone_mult,
                                'vel_smoothing': vel_smoothing,
                                'min_bars_between': min_bars_between,
                                'require_accel': require_accel,
                                'vel_threshold': vel_threshold,
                                'accel_threshold': accel_threshold,
                                'oversold_threshold': oversold_threshold,
                                'overbought_threshold': overbought_threshold,
                                'stop_loss_pct': stop_loss_pct,
                                'take_profit_pct': take_profit_pct,
                                'exit_on_opposite_signal': exit_on_opposite_signal,
                                'exit_on_midline_cross': exit_on_midline_cross,
                                'rsi_filter': rsi_filter,
                                'rsi_period': rsi_period,
                                'rsi_oversold': rsi_oversold,
                                'rsi_overbought': rsi_overbought,
                                'use_macd_confirm': use_macd_confirm,
                                'use_bb_filter': use_bb_filter,
                            }

                        # Helper function to convert numpy types to native Python types
                        def to_native(val):
                            if hasattr(val, 'item'):  # numpy scalar
                                return val.item()
                            elif isinstance(val, (np.bool_, np.integer, np.floating)):
                                return val.item()
                            return val

                        # Create production config for live trader with native Python types
                        production_config = {
                            "strategy_type": "velocity",
                            "strategy_name": str(strategy_name),
                            "ticker": str(ticker),
                            "interval": str(interval),
                            "optimization_period": f"{years}y",
                            "optimization_bars": len(st.session_state.get('df', [])) if 'df' in st.session_state else None,
                            "polygon_api_key": "",

                            # Core velocity parameters
                            "signal_type": str(params.get('signal_type', 'velocity_crossover_and_zone')),
                            "vel_smoothing": int(to_native(params.get('vel_smoothing', 3))),
                            "extreme_zone_mult": float(to_native(params.get('extreme_zone_mult', 1.5))),
                            "min_bars_between": int(to_native(params.get('min_bars_between', 1))),
                            "require_accel": bool(to_native(params.get('require_accel', True))),

                            # Threshold parameters
                            "oversold_threshold": float(to_native(params.get('oversold_threshold', -0.3))),
                            "overbought_threshold": float(to_native(params.get('overbought_threshold', 0.3))),

                            # Risk management
                            "stop_loss_pct": float(to_native(params.get('stop_loss_pct', 5.0))),
                            "take_profit_pct": float(to_native(params.get('take_profit_pct', 10.0))),

                            # Exit strategies
                            "exit_on_opposite_signal": bool(to_native(params.get('exit_on_opposite_signal', True))),
                            "exit_on_midline_cross": bool(to_native(params.get('exit_on_midline_cross', False))),

                            # Extra indicator filters
                            "rsi_filter": str(params.get('rsi_filter', 'none')),
                            "rsi_period": int(to_native(params.get('rsi_period', 14))),
                            "rsi_oversold": int(to_native(params.get('rsi_oversold', 30))),
                            "rsi_overbought": int(to_native(params.get('rsi_overbought', 70))),
                            "use_macd_confirm": bool(to_native(params.get('use_macd_confirm', False))),
                            "use_bb_filter": bool(to_native(params.get('use_bb_filter', False))),

                            # Novel oscillator type (if using advanced oscillators)
                            "oscillator_type": str(params.get('oscillator_type', st.session_state.get('selected_oscillator_type', 'composite_smooth'))),

                            # Discord webhook for alerts
                            "discord_webhook": discord_webhook,

                            "deployed_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }

                        # Save production config
                        prod_dir = "production_env"
                        os.makedirs(prod_dir, exist_ok=True)

                        prod_config_path = os.path.join(prod_dir, "velocity_config.json")
                        with open(prod_config_path, 'w') as f:
                            json.dump(production_config, f, indent=4)

                        # Save the FULL DataFrame for exact matching in live trader
                        # IMPORTANT: Save 'df' (full data), NOT 'test_df_vel' (which is only test portion ~20%)
                        try:
                            if 'df' in st.session_state and len(st.session_state['df']) > 0:
                                data_path = os.path.join(prod_dir, "velocity_data.parquet")
                                st.session_state['df'].to_parquet(data_path)
                                df_start = st.session_state['df'].index[0].strftime('%Y-%m-%d')
                                df_end = st.session_state['df'].index[-1].strftime('%Y-%m-%d')
                                st.info(f"📊 Saved {len(st.session_state['df'])} bars ({df_start} to {df_end})")
                            else:
                                st.warning("No data in session - load data first")
                        except Exception as e:
                            st.warning(f"Could not save data: {e}")

                        st.success(f"✅ Strategy deployed! Config saved to: {prod_config_path}")
                        st.info(f"🤖 Run `python velocity_live_trader.py` to start live trading with Discord alerts")

                        # Send test notification to Discord
                        try:
                            import requests
                            test_msg = {
                                "content": f"🚀 **Velocity Strategy Deployed**\n"
                                          f"**Strategy:** {strategy_name}\n"
                                          f"**Ticker:** {ticker}\n"
                                          f"**Signal Type:** {params.get('signal_type')}\n"
                                          f"**Thresholds:** OS={params.get('oversold_threshold')}, OB={params.get('overbought_threshold')}\n"
                                          f"**Risk:** SL={params.get('stop_loss_pct')}%, TP={params.get('take_profit_pct')}%\n"
                                          f"---\n"
                                          f"_Bot will alert on signals. Run velocity_live_trader.py to start._"
                            }
                            response = requests.post(discord_webhook, json=test_msg)
                            if response.status_code == 204:
                                st.success("📨 Test notification sent to Discord!")
                            else:
                                st.warning(f"Discord webhook response: {response.status_code}")
                        except Exception as e:
                            st.warning(f"Could not send test notification: {e}")

            with deploy_col2:
                if st.button("💾 Save Strategy", key="save_vel_permanent"):
                    # Get currently applied params or use current UI settings
                    if 'vel_best_params' in st.session_state and st.session_state['vel_best_params']:
                        params = st.session_state['vel_best_params']
                    else:
                        params = {
                            'signal_type': signal_type,
                            'extreme_zone_mult': extreme_zone_mult,
                            'vel_smoothing': vel_smoothing,
                            'min_bars_between': min_bars_between,
                            'require_accel': require_accel,
                            'vel_threshold': vel_threshold,
                            'accel_threshold': accel_threshold,
                            'oversold_threshold': oversold_threshold,
                            'overbought_threshold': overbought_threshold,
                            'stop_loss_pct': stop_loss_pct,
                            'take_profit_pct': take_profit_pct,
                            'exit_on_opposite_signal': exit_on_opposite_signal,
                            'exit_on_midline_cross': exit_on_midline_cross,
                            'rsi_filter': rsi_filter,
                            'rsi_period': rsi_period,
                            'rsi_oversold': rsi_oversold,
                            'rsi_overbought': rsi_overbought,
                            'use_macd_confirm': use_macd_confirm,
                            'use_bb_filter': use_bb_filter,
                        }

                    # Helper function to convert numpy types to native Python types
                    def to_native(val):
                        if hasattr(val, 'item'):
                            return val.item()
                        elif isinstance(val, (np.bool_, np.integer, np.floating)):
                            return val.item()
                        return val

                    # Create bundle directory name - USE the user's strategy_name
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    # Use user's strategy name, sanitized for filesystem
                    safe_name = strategy_name.replace(" ", "_").replace("/", "-").replace(":", "-")
                    bundle_name = f"{safe_name}_{timestamp}"
                    bundle_dir = os.path.join("velocity_strategies", bundle_name)
                    os.makedirs(bundle_dir, exist_ok=True)

                    # Calculate actual data period in days from the dataframe
                    data_period_days = 365 * years  # Default based on years
                    if 'df' in st.session_state and len(st.session_state['df']) > 0:
                        df_temp = st.session_state['df']
                        data_period_days = (df_temp.index[-1] - df_temp.index[0]).days
                    elif 'test_df_vel' in st.session_state and len(st.session_state['test_df_vel']) > 0:
                        df_temp = st.session_state['test_df_vel']
                        data_period_days = (df_temp.index[-1] - df_temp.index[0]).days

                    # Create full config for bundle
                    bundle_config = {
                        "strategy_type": "velocity",
                        "strategy_name": str(strategy_name),
                        "bundle_name": bundle_name,
                        "ticker": str(ticker),
                        "interval": str(interval),
                        "optimization_period": f"{years}y",
                        "data_period_days": int(data_period_days),
                        "optimization_bars": len(st.session_state.get('df', [])) if 'df' in st.session_state else None,
                        "polygon_api_key": "",

                        # Core velocity parameters
                        "signal_type": str(params.get('signal_type', 'velocity_crossover_and_zone')),
                        "vel_smoothing": int(to_native(params.get('vel_smoothing', 3))),
                        "extreme_zone_mult": float(to_native(params.get('extreme_zone_mult', 1.5))),
                        "min_bars_between": int(to_native(params.get('min_bars_between', 1))),
                        "require_accel": bool(to_native(params.get('require_accel', True))),
                        "vel_threshold": float(to_native(params.get('vel_threshold', 0))),
                        "accel_threshold": float(to_native(params.get('accel_threshold', 0))),

                        # Threshold parameters
                        "oversold_threshold": float(to_native(params.get('oversold_threshold', -0.3))),
                        "overbought_threshold": float(to_native(params.get('overbought_threshold', 0.3))),

                        # Risk management
                        "stop_loss_pct": float(to_native(params.get('stop_loss_pct', 5.0))),
                        "take_profit_pct": float(to_native(params.get('take_profit_pct', 10.0))),

                        # Exit strategies
                        "exit_on_opposite_signal": bool(to_native(params.get('exit_on_opposite_signal', True))),
                        "exit_on_midline_cross": bool(to_native(params.get('exit_on_midline_cross', False))),

                        # Extra indicator filters
                        "rsi_filter": str(params.get('rsi_filter', 'none')),
                        "rsi_period": int(to_native(params.get('rsi_period', 14))),
                        "rsi_oversold": int(to_native(params.get('rsi_oversold', 30))),
                        "rsi_overbought": int(to_native(params.get('rsi_overbought', 70))),
                        "use_macd_confirm": bool(to_native(params.get('use_macd_confirm', False))),
                        "use_bb_filter": bool(to_native(params.get('use_bb_filter', False))),

                        # Discord webhook
                        "discord_webhook": discord_webhook if discord_webhook else None,

                        "deployed_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    # Save to bundle directory
                    config_path = os.path.join(bundle_dir, "velocity_config.json")
                    with open(config_path, 'w') as f:
                        json.dump(bundle_config, f, indent=4)

                    # Save the FULL dataframe so live trader uses identical data
                    # IMPORTANT: Save 'df' (full data), NOT 'test_df_vel' (which is only test portion ~20%)
                    if 'df' in st.session_state and len(st.session_state['df']) > 0:
                        data_path = os.path.join(bundle_dir, "data.parquet")
                        st.session_state['df'].to_parquet(data_path)
                        df_start = st.session_state['df'].index[0].strftime('%Y-%m-%d')
                        df_end = st.session_state['df'].index[-1].strftime('%Y-%m-%d')
                        st.info(f"📊 Saved {len(st.session_state['df'])} bars of data ({df_start} to {df_end})")
                    else:
                        st.warning("No data in session - load data first")

                    st.success(f"✅ Strategy saved permanently to: {bundle_dir}")
                    st.info("📦 This strategy will appear in the live trader's strategy selection menu")

            # === BOT CONTROL SECTION ===
            st.markdown("---")
            st.subheader("🤖 Bot Control")

            import subprocess
            import signal

            # Initialize bot process tracking in session state
            if 'velocity_bot_process' not in st.session_state:
                st.session_state['velocity_bot_process'] = None
            if 'velocity_bot_pid' not in st.session_state:
                st.session_state['velocity_bot_pid'] = None

            # Check if bot is running (check if PID exists and process is alive)
            bot_running = False
            if st.session_state['velocity_bot_pid']:
                try:
                    # Check if process is still running
                    os.kill(st.session_state['velocity_bot_pid'], 0)
                    bot_running = True
                except (OSError, ProcessLookupError):
                    # Process not running
                    st.session_state['velocity_bot_pid'] = None
                    st.session_state['velocity_bot_process'] = None

            bot_col1, bot_col2, bot_col3 = st.columns([2, 2, 2])

            with bot_col1:
                if bot_running:
                    st.success(f"🟢 Bot Running (PID: {st.session_state['velocity_bot_pid']})")
                else:
                    st.info("🔴 Bot Stopped")

            with bot_col2:
                if not bot_running:
                    if st.button("▶️ Start Bot", type="primary", key="start_vel_bot"):
                        config_path = "production_env/velocity_config.json"
                        if os.path.exists(config_path):
                            try:
                                # Start the bot as a background process
                                log_file = open("velocity_bot.log", "w")
                                process = subprocess.Popen(
                                    ["python", "velocity_live_trader.py", "--config", config_path],
                                    stdout=log_file,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=True
                                )
                                st.session_state['velocity_bot_process'] = process
                                st.session_state['velocity_bot_pid'] = process.pid
                                st.success(f"✅ Bot started! PID: {process.pid}")
                                st.info("📄 Logs: velocity_bot.log")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to start bot: {e}")
                        else:
                            st.error("Please deploy a strategy first (config not found)")
                else:
                    st.button("▶️ Start Bot", disabled=True, key="start_vel_bot_disabled")

            with bot_col3:
                if bot_running:
                    if st.button("⏹️ Stop Bot", type="secondary", key="stop_vel_bot"):
                        try:
                            pid = st.session_state['velocity_bot_pid']
                            os.kill(pid, signal.SIGTERM)
                            st.session_state['velocity_bot_pid'] = None
                            st.session_state['velocity_bot_process'] = None
                            st.success("✅ Bot stopped!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to stop bot: {e}")
                            st.session_state['velocity_bot_pid'] = None
                else:
                    st.button("⏹️ Stop Bot", disabled=True, key="stop_vel_bot_disabled")

            # Show recent log output (using checkbox instead of expander to avoid nesting)
            if bot_running:
                show_logs = st.checkbox("📄 Show Bot Logs (last 20 lines)", key="show_vel_logs")
                if show_logs:
                    try:
                        if os.path.exists("velocity_bot.log"):
                            with open("velocity_bot.log", "r") as f:
                                lines = f.readlines()
                                last_lines = lines[-20:] if len(lines) > 20 else lines
                                st.code("".join(last_lines), language="text")
                                if st.button("🔄 Refresh Logs", key="refresh_vel_logs"):
                                    st.rerun()
                        else:
                            st.info("No log file yet")
                    except Exception as e:
                        st.warning(f"Could not read logs: {e}")

            # Show terminal command option (using checkbox instead of expander)
            show_cmd = st.checkbox("💻 Show Manual Terminal Command", key="show_vel_cmd")
            if show_cmd:
                st.code("python velocity_live_trader.py", language="bash")
                st.caption("Or run in background with logs:")
                st.code("nohup python velocity_live_trader.py > velocity_bot.log 2>&1 &", language="bash")

    # Calculate velocity and acceleration on test data
    test_df_vel = df.loc[X_test.index].copy()

    # Handle oscillator type selection - calculate novel oscillator if selected
    osc_type_map = {
        'composite_smooth (Default)': 'composite_smooth',
        'ARWO (Adaptive Regime-Weighted)': 'arwo',
        'DCO (Divergence Consensus)': 'dco',
        'VCMO (Volume-Confirmed Momentum)': 'vcmo',
        'ICS (Indicator Convergence Score)': 'ics',
        'MJI (Momentum Jerk)': 'mji',
        'PRF (Percentile Rank Fusion)': 'prf',
        'EWAF (Entropy-Weighted Adaptive Fusion)': 'ewaf',
        'KFIF (Kalman-Filtered Indicator Fusion)': 'kfif'
    }
    selected_osc_key = osc_type_map.get(selected_oscillator, 'composite_smooth')

    # Calculate novel oscillator if selected and available
    if selected_osc_key != 'composite_smooth' and NOVEL_INDICATORS_AVAILABLE:
        with st.spinner(f"Calculating {selected_oscillator}..."):
            try:
                if selected_osc_key == 'arwo':
                    test_df_vel['novel_osc'] = calculate_arwo(test_df_vel)
                elif selected_osc_key == 'dco':
                    test_df_vel['novel_osc'] = calculate_dco(test_df_vel)
                elif selected_osc_key == 'vcmo':
                    test_df_vel['novel_osc'] = calculate_vcmo(test_df_vel)
                elif selected_osc_key == 'ics':
                    test_df_vel['novel_osc'] = calculate_ics(test_df_vel)
                elif selected_osc_key == 'mji':
                    test_df_vel['novel_osc'] = calculate_mji(test_df_vel)
                elif selected_osc_key == 'prf':
                    test_df_vel['novel_osc'] = calculate_prf(test_df_vel)
                elif selected_osc_key == 'ewaf':
                    test_df_vel['novel_osc'] = calculate_ewaf(test_df_vel)
                elif selected_osc_key == 'kfif':
                    kfif_val, _, _ = calculate_kfif(test_df_vel)
                    test_df_vel['novel_osc'] = kfif_val
                osc_col = 'novel_osc'
                st.info(f"Using {selected_oscillator} oscillator")
            except Exception as e:
                st.warning(f"Error calculating {selected_oscillator}: {e}. Falling back to composite_smooth.")
                osc_col = 'composite_smooth' if 'composite_smooth' in test_df_vel.columns else 'composite_oscillator'
    else:
        osc_col = 'composite_smooth' if 'composite_smooth' in test_df_vel.columns else 'composite_oscillator'

    # Store in session state for deploy button access
    st.session_state['test_df_vel'] = test_df_vel
    st.session_state['selected_oscillator_type'] = selected_osc_key

    # Smooth the oscillator if needed
    if vel_smoothing > 1:
        test_df_vel['osc_smooth'] = test_df_vel[osc_col].rolling(window=vel_smoothing, center=False).mean()
        test_df_vel['osc_smooth'] = test_df_vel['osc_smooth'].bfill()
    else:
        test_df_vel['osc_smooth'] = test_df_vel[osc_col]

    # Calculate velocity (first derivative) and acceleration (second derivative)
    test_df_vel['velocity'] = test_df_vel['osc_smooth'].diff()
    test_df_vel['acceleration'] = test_df_vel['velocity'].diff()

    # Fill NaN values
    test_df_vel['velocity'] = test_df_vel['velocity'].fillna(0)
    test_df_vel['acceleration'] = test_df_vel['acceleration'].fillna(0)

    # Detect velocity zero-crossings (slope changes)
    test_df_vel['vel_cross_up'] = (test_df_vel['velocity'] > 0) & (test_df_vel['velocity'].shift(1) <= 0)
    test_df_vel['vel_cross_down'] = (test_df_vel['velocity'] < 0) & (test_df_vel['velocity'].shift(1) >= 0)

    # Build entry conditions based on signal_type (NEW - supports multiple signal generation methods)
    osc_smooth = test_df_vel['osc_smooth']
    velocity = test_df_vel['velocity']
    acceleration = test_df_vel['acceleration']

    in_oversold = osc_smooth < oversold_threshold
    in_overbought = osc_smooth > overbought_threshold
    extreme_oversold = osc_smooth < (oversold_threshold * extreme_zone_mult)
    extreme_overbought = osc_smooth > (overbought_threshold * extreme_zone_mult)

    # Strong momentum detection
    vel_std = velocity.rolling(10, min_periods=1).std().fillna(velocity.std())
    strong_momentum_up = velocity > vel_std * 1.5
    strong_momentum_down = velocity < -vel_std * 1.5

    # Get close prices for new signal types
    close_prices_vel = test_df_vel['close']

    if signal_type == 'velocity_crossover_and_zone':
        # Original strict logic: BOTH velocity crossover AND in zone (fewest trades)
        buy_condition = test_df_vel['vel_cross_up'] & in_oversold
        sell_condition = test_df_vel['vel_cross_down'] & in_overbought
    elif signal_type == 'velocity_crossover_or_zone':
        # Balanced: velocity crossover OR extreme zone (more trades)
        buy_condition = test_df_vel['vel_cross_up'] | extreme_oversold
        sell_condition = test_df_vel['vel_cross_down'] | extreme_overbought
    elif signal_type == 'zone_only':
        # Zone-based: extreme zones + velocity turning (mean reversion)
        buy_condition = extreme_oversold & (velocity > 0)
        sell_condition = extreme_overbought & (velocity < 0)
    elif signal_type == 'momentum':
        # Trend following: strong velocity in one direction
        buy_condition = strong_momentum_up & (osc_smooth < 0)
        sell_condition = strong_momentum_down & (osc_smooth > 0)
    elif signal_type == 'any_reversal':
        # Most aggressive: any of the above (most trades)
        buy_condition = test_df_vel['vel_cross_up'] | extreme_oversold | (strong_momentum_up & in_oversold)
        sell_condition = test_df_vel['vel_cross_down'] | extreme_overbought | (strong_momentum_down & in_overbought)
    elif signal_type == 'double_bottom':
        # Look for second velocity crossover up while still in oversold
        vel_cross_up_count = test_df_vel['vel_cross_up'].rolling(10).sum()
        buy_condition = (vel_cross_up_count >= 2) & in_oversold
        vel_cross_down_count = test_df_vel['vel_cross_down'].rolling(10).sum()
        sell_condition = (vel_cross_down_count >= 2) & in_overbought
    elif signal_type == 'divergence':
        # Bullish divergence: price lower low, oscillator higher low
        price_lower_low = (close_prices_vel < close_prices_vel.rolling(5).min().shift(1))
        osc_higher_low = (osc_smooth > osc_smooth.rolling(5).min().shift(1))
        buy_condition = price_lower_low & osc_higher_low & in_oversold
        # Bearish divergence
        price_higher_high = (close_prices_vel > close_prices_vel.rolling(5).max().shift(1))
        osc_lower_high = (osc_smooth < osc_smooth.rolling(5).max().shift(1))
        sell_condition = price_higher_high & osc_lower_high & in_overbought
    elif signal_type == 'breakout':
        # Oscillator breaks above/below threshold (entry on breakout)
        osc_breaks_above = (osc_smooth > oversold_threshold) & (osc_smooth.shift(1) <= oversold_threshold)
        osc_breaks_below = (osc_smooth < overbought_threshold) & (osc_smooth.shift(1) >= overbought_threshold)
        buy_condition = osc_breaks_above  # Buy when breaking out of oversold
        sell_condition = osc_breaks_below  # Sell when breaking into overbought
    else:
        # Default to original strict logic
        buy_condition = test_df_vel['vel_cross_up'] & in_oversold
        sell_condition = test_df_vel['vel_cross_down'] & in_overbought

    # Apply velocity magnitude filter
    if vel_threshold > 0:
        buy_condition = buy_condition & (velocity.abs() >= vel_threshold)
        sell_condition = sell_condition & (velocity.abs() >= vel_threshold)

    # Apply acceleration filter
    if require_accel:
        buy_accel_cond = acceleration > 0
        sell_accel_cond = acceleration < 0
        if accel_threshold > 0:
            buy_accel_cond = buy_accel_cond & (acceleration.abs() >= accel_threshold)
            sell_accel_cond = sell_accel_cond & (acceleration.abs() >= accel_threshold)
        buy_condition = buy_condition & buy_accel_cond
        sell_condition = sell_condition & sell_accel_cond

    # Apply extra indicator filters (RSI, MACD, BB)
    if rsi_filter != 'none' or use_macd_confirm or use_bb_filter:
        close_prices_main = test_df_vel['close']

        # Calculate RSI
        if rsi_filter != 'none':
            delta = close_prices_main.diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / (loss + 1e-10)
            rsi_values = 100 - (100 / (1 + rs))

            if rsi_filter == 'oversold_only':
                buy_condition = buy_condition & (rsi_values < rsi_oversold)
            elif rsi_filter == 'overbought_only':
                sell_condition = sell_condition & (rsi_values > rsi_overbought)
            elif rsi_filter == 'both':
                buy_condition = buy_condition & (rsi_values < rsi_oversold)
                sell_condition = sell_condition & (rsi_values > rsi_overbought)

        # Calculate MACD
        if use_macd_confirm:
            ema12 = close_prices_main.ewm(span=12, adjust=False).mean()
            ema26 = close_prices_main.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - macd_signal_line
            buy_condition = buy_condition & (macd_hist > macd_hist.shift(1))
            sell_condition = sell_condition & (macd_hist < macd_hist.shift(1))

        # Calculate Bollinger Bands
        if use_bb_filter:
            bb_sma = close_prices_main.rolling(20).mean()
            bb_std = close_prices_main.rolling(20).std()
            bb_upper = bb_sma + 2 * bb_std
            bb_lower = bb_sma - 2 * bb_std
            buy_condition = buy_condition & (close_prices_main < bb_lower)
            sell_condition = sell_condition & (close_prices_main > bb_upper)

    test_df_vel['buy_signal'] = buy_condition
    test_df_vel['sell_signal'] = sell_condition

    # Run velocity-based trading simulation with STOP LOSS, TAKE PROFIT, and TRAILING STOP
    vel_position = 0
    vel_positions = []
    vel_trades = []
    vel_entry_price = None
    vel_entry_date = None
    vel_highest_price = None  # For trailing stop
    last_trade_bar = -min_bars_between  # Allow first trade immediately

    for i in range(len(test_df_vel)):
        date = test_df_vel.index[i]
        price = test_df_vel['close'].iloc[i]

        # Check minimum bars between trades
        bars_since_last = i - last_trade_bar

        # If in position, check exit conditions first
        if vel_position == 1 and vel_entry_price is not None:
            # Update highest price for trailing stop
            if vel_highest_price is None:
                vel_highest_price = price
            else:
                vel_highest_price = max(vel_highest_price, price)

            exit_triggered = False
            exit_reason = None

            # Check STOP LOSS (fixed or trailing)
            if use_trailing_stop and trailing_stop_pct > 0:
                # Trailing stop: exit if price drops trailing_stop_pct below highest
                trail_stop_price = vel_highest_price * (1 - trailing_stop_pct / 100)
                if price <= trail_stop_price:
                    exit_triggered = True
                    exit_reason = 'trailing_stop'
            elif stop_loss_pct > 0:
                # Fixed stop loss
                sl_price = vel_entry_price * (1 - stop_loss_pct / 100)
                if price <= sl_price:
                    exit_triggered = True
                    exit_reason = 'stop_loss'

            # Check TAKE PROFIT
            if not exit_triggered and take_profit_pct > 0:
                tp_price = vel_entry_price * (1 + take_profit_pct / 100)
                if price >= tp_price:
                    exit_triggered = True
                    exit_reason = 'take_profit'

            # Execute stop/take profit exit
            if exit_triggered:
                vel_position = 0
                pnl = ((price - vel_entry_price) / vel_entry_price) * 100
                last_trade_bar = i

                time_str = date.strftime('%Y-%m-%d %H:%M') if hasattr(date, 'strftime') else str(date)
                entry_time_str = vel_entry_date.strftime('%Y-%m-%d %H:%M') if hasattr(vel_entry_date, 'strftime') else str(vel_entry_date)

                vel_trades.append({
                    'type': 'exit',
                    'date': date,
                    'time': time_str,
                    'price': price,
                    'signal': exit_reason.upper(),
                    'pnl': pnl,
                    'entry_date': vel_entry_date,
                    'entry_time': entry_time_str,
                    'entry_price': vel_entry_price,
                    'osc_value': test_df_vel['osc_smooth'].iloc[i],
                    'velocity': test_df_vel['velocity'].iloc[i],
                    'acceleration': test_df_vel['acceleration'].iloc[i]
                })
                vel_entry_price = None
                vel_entry_date = None
                vel_highest_price = None
                vel_positions.append(vel_position)
                continue

        # BUY signal (entry)
        if test_df_vel['buy_signal'].iloc[i] and vel_position == 0 and bars_since_last >= min_bars_between:
            vel_position = 1
            vel_entry_price = price
            vel_entry_date = date
            vel_highest_price = price
            last_trade_bar = i

            time_str = date.strftime('%Y-%m-%d %H:%M') if hasattr(date, 'strftime') else str(date)

            vel_trades.append({
                'type': 'entry',
                'date': date,
                'time': time_str,
                'price': price,
                'signal': 'BUY',
                'osc_value': test_df_vel['osc_smooth'].iloc[i],
                'velocity': test_df_vel['velocity'].iloc[i],
                'acceleration': test_df_vel['acceleration'].iloc[i]
            })

        # Check for exit conditions (SELL signal or midline cross)
        exit_now = False
        exit_signal_name = None

        # Exit on opposite signal (sell signal)
        if exit_on_opposite_signal and test_df_vel['sell_signal'].iloc[i] and vel_position == 1 and bars_since_last >= min_bars_between:
            exit_now = True
            exit_signal_name = 'SELL'

        # Exit on midline cross (oscillator crosses above 0)
        if exit_on_midline_cross and vel_position == 1 and i > 0:
            current_osc = test_df_vel['osc_smooth'].iloc[i]
            prev_osc = test_df_vel['osc_smooth'].iloc[i-1]
            if current_osc > 0 and prev_osc <= 0:
                exit_now = True
                exit_signal_name = 'MIDLINE_CROSS'

        if exit_now and vel_position == 1:
            vel_position = 0
            pnl = ((price - vel_entry_price) / vel_entry_price) * 100 if vel_entry_price else 0
            last_trade_bar = i

            time_str = date.strftime('%Y-%m-%d %H:%M') if hasattr(date, 'strftime') else str(date)
            entry_time_str = vel_entry_date.strftime('%Y-%m-%d %H:%M') if hasattr(vel_entry_date, 'strftime') else str(vel_entry_date)

            vel_trades.append({
                'type': 'exit',
                'date': date,
                'time': time_str,
                'price': price,
                'signal': exit_signal_name,
                'pnl': pnl,
                'entry_date': vel_entry_date,
                'entry_time': entry_time_str,
                'entry_price': vel_entry_price,
                'osc_value': test_df_vel['osc_smooth'].iloc[i],
                'velocity': test_df_vel['velocity'].iloc[i],
                'acceleration': test_df_vel['acceleration'].iloc[i]
            })
            vel_entry_price = None
            vel_entry_date = None
            vel_highest_price = None

        vel_positions.append(vel_position)

    test_df_vel['position'] = vel_positions
    test_df_vel['returns'] = test_df_vel['close'].pct_change()
    test_df_vel['strategy_returns'] = test_df_vel['position'].shift(1) * test_df_vel['returns']
    test_df_vel['cum_market'] = (1 + test_df_vel['returns']).cumprod()
    test_df_vel['cum_strategy'] = (1 + test_df_vel['strategy_returns'].fillna(0)).cumprod()

    # Update session state with FINAL calculated DataFrame (for deploy button)
    st.session_state['test_df_vel'] = test_df_vel

    vel_exits = [t for t in vel_trades if t['type'] == 'exit']
    vel_entries = [t for t in vel_trades if t['type'] == 'entry']

    # Track open position (if still in trade at end of data)
    vel_open_position = None
    if vel_position == 1 and vel_entry_price is not None:
        latest_date = test_df_vel.index[-1]
        latest_price = test_df_vel['close'].iloc[-1]
        open_pnl = ((latest_price - vel_entry_price) / vel_entry_price) * 100
        vel_open_position = {
            'type': 'open',
            'entry_date': vel_entry_date,
            'entry_time': vel_entry_date.strftime('%Y-%m-%d %H:%M') if hasattr(vel_entry_date, 'strftime') else str(vel_entry_date),
            'entry_price': vel_entry_price,
            'current_date': latest_date,
            'current_price': latest_price,
            'unrealized_pnl': open_pnl,
            'osc_value': test_df_vel['osc_smooth'].iloc[-1],
        }

    # Display velocity trading stats
    if vel_exits or vel_open_position:
        vel_pnls = [t.get('pnl', 0) for t in vel_exits] if vel_exits else []
        vel_wins = [t for t in vel_exits if t.get('pnl', 0) > 0]
        vel_losses = [t for t in vel_exits if t.get('pnl', 0) <= 0]
        vel_win_rate = len(vel_wins) / len(vel_exits) * 100 if vel_exits else 0
        vel_avg_pnl = np.mean(vel_pnls) if vel_pnls else 0
        vel_total_return = (test_df_vel['cum_strategy'].iloc[-1] - 1) * 100
        vel_market_return = (test_df_vel['cum_market'].iloc[-1] - 1) * 100

        # Profit factor
        vel_gross_profit = sum([t['pnl'] for t in vel_wins]) if vel_wins else 0
        vel_gross_loss = abs(sum([t['pnl'] for t in vel_losses])) if vel_losses else 1
        vel_profit_factor = vel_gross_profit / vel_gross_loss if vel_gross_loss > 0 else float('inf')

        # Avg win/loss
        vel_avg_win = np.mean([t['pnl'] for t in vel_wins]) if vel_wins else 0
        vel_avg_loss = np.mean([t['pnl'] for t in vel_losses]) if vel_losses else 0

        st.subheader("Velocity Trading Statistics")

        # Show open position alert if exists
        if vel_open_position:
            pnl_color = "green" if vel_open_position['unrealized_pnl'] > 0 else "red"
            st.success(f"📈 **OPEN POSITION** | Entry: ${vel_open_position['entry_price']:.2f} on {vel_open_position['entry_time']} | "
                      f"Current: ${vel_open_position['current_price']:.2f} | "
                      f"Unrealized P&L: {vel_open_position['unrealized_pnl']:+.2f}%")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", len(vel_exits))
        col2.metric("Win Rate", f"{vel_win_rate:.1f}%")
        col3.metric("Avg Trade", f"{vel_avg_pnl:.2f}%")
        col4.metric("Profit Factor", f"{vel_profit_factor:.2f}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Strategy Return", f"{vel_total_return:.1f}%")
        col6.metric("Market Return", f"{vel_market_return:.1f}%")
        col7.metric("Outperformance", f"{vel_total_return - vel_market_return:+.1f}%")
        col8.metric("Wins / Losses", f"{len(vel_wins)} / {len(vel_losses)}")

        col9, col10, col11, col12 = st.columns(4)
        col9.metric("Avg Win", f"{vel_avg_win:.2f}%")
        col10.metric("Avg Loss", f"{vel_avg_loss:.2f}%")
        col11.metric("Best Trade", f"{max(vel_pnls):.2f}%" if vel_pnls else "N/A")
        col12.metric("Worst Trade", f"{min(vel_pnls):.2f}%" if vel_pnls else "N/A")

        # Capital metrics row
        st.markdown("---")
        st.markdown("##### 💰 Capital Performance")
        vel_ending_capital = starting_capital * (1 + vel_total_return / 100)
        vel_profit_loss = vel_ending_capital - starting_capital
        bh_ending_capital = starting_capital * (1 + vel_market_return / 100)

        col13, col14, col15, col16 = st.columns(4)
        col13.metric("Starting Capital", f"${starting_capital:,.0f}")
        col14.metric("Ending Capital", f"${vel_ending_capital:,.0f}",
                    delta=f"${vel_profit_loss:+,.0f}")
        col15.metric("Buy & Hold Capital", f"${bh_ending_capital:,.0f}")
        col16.metric("Strategy Advantage", f"${vel_ending_capital - bh_ending_capital:+,.0f}")

        # ============================================================
        # VELOCITY TRADING CHART
        # ============================================================
        st.subheader("Velocity Trading Chart")

        # Create ChartHelper for gap-free intraday charts
        chart_vel = ChartHelper(test_df_vel, interval)
        x_vals_vel = chart_vel.get_x(test_df_vel.index)

        fig_vel = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            subplot_titles=(
                'Price Action with Velocity-Based Entries/Exits',
                'Composite Oscillator',
                'Velocity (First Derivative) & Acceleration',
                'Cumulative Returns'
            )
        )

        # Row 1: Candlestick Chart
        fig_vel.add_trace(
            go.Candlestick(
                x=x_vals_vel,
                open=test_df_vel['open'],
                high=test_df_vel['high'],
                low=test_df_vel['low'],
                close=test_df_vel['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )

        # Add entry markers
        if vel_entries:
            fig_vel.add_trace(
                go.Scatter(
                    x=chart_vel.get_x([t['date'] for t in vel_entries]),
                    y=[t['price'] for t in vel_entries],
                    mode='markers+text',
                    marker=dict(symbol='triangle-up', size=15, color='lime', line=dict(width=2, color='darkgreen')),
                    name='BUY (Vel Cross Up)',
                    text=[t['time'].split(' ')[1] if ' ' in t['time'] else '' for t in vel_entries],
                    textposition='bottom center',
                    textfont=dict(size=8),
                    hovertemplate='BUY<br>Time: %{text}<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )

        # Add exit markers
        if vel_exits:
            fig_vel.add_trace(
                go.Scatter(
                    x=chart_vel.get_x([t['date'] for t in vel_exits]),
                    y=[t['price'] for t in vel_exits],
                    mode='markers+text',
                    marker=dict(symbol='triangle-down', size=15, color='red', line=dict(width=2, color='darkred')),
                    name='SELL (Vel Cross Down)',
                    text=[t['time'].split(' ')[1] if ' ' in t['time'] else '' for t in vel_exits],
                    textposition='top center',
                    textfont=dict(size=8),
                    hovertemplate='SELL<br>Time: %{text}<br>Price: $%{y:.2f}<br>PnL: ' +
                                  '<br>'.join([f"{t.get('pnl', 0):.1f}%" for t in vel_exits]) + '<extra></extra>'
                ),
                row=1, col=1
            )

        # Draw trade lines
        for trade in vel_exits:
            if 'entry_date' in trade and 'entry_price' in trade:
                color = 'rgba(0,255,0,0.3)' if trade.get('pnl', 0) > 0 else 'rgba(255,0,0,0.3)'
                fig_vel.add_trace(
                    go.Scatter(
                        x=chart_vel.get_x([trade['entry_date'], trade['date']]),
                        y=[trade['entry_price'], trade['price']],
                        mode='lines',
                        line=dict(color=color, width=2, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )

        # Row 2: Oscillator
        fig_vel.add_trace(
            go.Scatter(
                x=x_vals_vel,
                y=test_df_vel['osc_smooth'],
                mode='lines',
                name='Oscillator',
                line=dict(color='purple', width=1.5)
            ),
            row=2, col=1
        )
        fig_vel.add_hline(y=overbought_threshold, line_dash="dash", line_color="red", row=2, col=1)
        fig_vel.add_hline(y=oversold_threshold, line_dash="dash", line_color="green", row=2, col=1)
        fig_vel.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

        # Mark buy/sell signals on oscillator
        buy_signals = test_df_vel[test_df_vel['buy_signal']]
        sell_signals = test_df_vel[test_df_vel['sell_signal']]

        if len(buy_signals) > 0:
            fig_vel.add_trace(
                go.Scatter(
                    x=chart_vel.get_x(buy_signals.index),
                    y=buy_signals['osc_smooth'],
                    mode='markers',
                    marker=dict(symbol='circle', size=10, color='lime'),
                    name='Buy Signal',
                    showlegend=True
                ),
                row=2, col=1
            )

        if len(sell_signals) > 0:
            fig_vel.add_trace(
                go.Scatter(
                    x=chart_vel.get_x(sell_signals.index),
                    y=sell_signals['osc_smooth'],
                    mode='markers',
                    marker=dict(symbol='circle', size=10, color='red'),
                    name='Sell Signal',
                    showlegend=True
                ),
                row=2, col=1
            )

        # Row 3: Velocity and Acceleration
        fig_vel.add_trace(
            go.Scatter(
                x=x_vals_vel,
                y=test_df_vel['velocity'],
                mode='lines',
                name='Velocity',
                line=dict(color='blue', width=1.5)
            ),
            row=3, col=1
        )
        fig_vel.add_trace(
            go.Scatter(
                x=x_vals_vel,
                y=test_df_vel['acceleration'],
                mode='lines',
                name='Acceleration',
                line=dict(color='orange', width=1)
            ),
            row=3, col=1
        )
        fig_vel.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)

        # Row 4: Cumulative Returns
        fig_vel.add_trace(
            go.Scatter(
                x=x_vals_vel,
                y=test_df_vel['cum_market'],
                mode='lines',
                name='Buy & Hold',
                line=dict(color='blue', width=2)
            ),
            row=4, col=1
        )
        fig_vel.add_trace(
            go.Scatter(
                x=x_vals_vel,
                y=test_df_vel['cum_strategy'],
                mode='lines',
                name='Velocity Strategy',
                line=dict(color='orange', width=2)
            ),
            row=4, col=1
        )

        # Update layout
        fig_vel.update_layout(
            height=1000,
            title_text="Velocity-Based Trading Analysis (Live Trading Capable)",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False
        )

        fig_vel.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_vel.update_yaxes(title_text="Oscillator", row=2, col=1, range=[-1.2, 1.2])
        fig_vel.update_yaxes(title_text="Vel / Accel", row=3, col=1)
        fig_vel.update_yaxes(title_text="Cum. Return", row=4, col=1)

        # Apply formatting for intraday charts
        chart_vel.apply_formatting(fig_vel)
        st.plotly_chart(fig_vel, use_container_width=True)

        # Trade log with times
        st.subheader("Velocity Trade Log (with Times)")
        vel_trade_summary = []
        for i, t in enumerate(vel_exits):
            hold_d = (t['date'] - t['entry_date']).days if t.get('entry_date') else 0
            vel_trade_summary.append({
                '#': i + 1,
                'Entry Time': t.get('entry_time', 'N/A'),
                'Entry $': f"${t.get('entry_price', 0):.2f}",
                'Exit Time': t.get('time', 'N/A'),
                'Exit $': f"${t['price']:.2f}",
                'P&L': f"{t.get('pnl', 0):.2f}%",
                'Days': hold_d,
                'Entry Osc': f"{t.get('osc_value', 0):.3f}",
                'Result': '✅' if t.get('pnl', 0) > 0 else '❌'
            })

        # Add open position to trade log if exists
        if vel_open_position:
            hold_d = (vel_open_position['current_date'] - vel_open_position['entry_date']).days
            vel_trade_summary.append({
                '#': len(vel_exits) + 1,
                'Entry Time': vel_open_position['entry_time'],
                'Entry $': f"${vel_open_position['entry_price']:.2f}",
                'Exit Time': '🔴 OPEN',
                'Exit $': f"${vel_open_position['current_price']:.2f}",
                'P&L': f"{vel_open_position['unrealized_pnl']:+.2f}%",
                'Days': hold_d,
                'Entry Osc': f"{vel_open_position['osc_value']:.3f}",
                'Result': '🔵 OPEN'
            })

        st.dataframe(pd.DataFrame(vel_trade_summary), use_container_width=True)

        # Comparison with Scipy
        st.subheader("Comparison: Velocity vs Scipy Peaks")
        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown("**Scipy Peaks (Look-Ahead)**")
            if scipy_exits:
                st.write(f"- Trades: {len(scipy_exits)}")
                st.write(f"- Win Rate: {scipy_win_rate:.1f}%")
                st.write(f"- Return: {scipy_total_return:.1f}%")
                st.write(f"- Profit Factor: {scipy_profit_factor:.2f}")
            else:
                st.write("No trades")

        with comp_col2:
            st.markdown("**Velocity (Real-Time)**")
            st.write(f"- Trades: {len(vel_exits)}")
            st.write(f"- Win Rate: {vel_win_rate:.1f}%")
            st.write(f"- Return: {vel_total_return:.1f}%")
            st.write(f"- Profit Factor: {vel_profit_factor:.2f}")

        # Live trading note
        st.info("""
        **Live Trading Capability:** This velocity-based approach can be used for live trading because:
        - Signals are generated using ONLY past data (no look-ahead)
        - Velocity zero-crossing is detectable in real-time
        - Acceleration confirmation adds reliability
        - Trades would execute at the NEXT bar's open after signal
        """)

    else:
        st.warning("No completed velocity-based trades in test period. Try adjusting parameters.")

    st.markdown("---")

    # Step 5d: Strategy Discovery Engine
    render_strategy_discovery_section(df)

    st.markdown("---")

    # Step 6: Train Model
    st.header("Step 6: Train ML Model (Optional)")

    if use_optuna:
        st.write(f"**Optuna Optimization:** {n_trials} trials per model")
        st.write(f"**Models to train:** {', '.join(model_types_to_train)}")

        # Create validation split from training data for Optuna
        val_split = 0.2
        val_size = int(len(X_train) * val_split)
        X_train_opt = X_train.iloc[:-val_size]
        X_val_opt = X_train.iloc[-val_size:]
        y_train_opt = y_train.iloc[:-val_size] if hasattr(y_train, 'iloc') else pd.Series(y_train[:-val_size])
        y_val_opt = y_train.iloc[-val_size:] if hasattr(y_train, 'iloc') else pd.Series(y_train[-val_size:])

        st.caption(f"Using {len(X_train_opt)} samples for training, {len(X_val_opt)} for validation, {len(X_test)} for testing")

        if st.button("Start Optuna Optimization", type="primary"):
            progress_placeholder = st.container()
            progress_placeholder.info("Starting hyperparameter optimization...")

            # Train all selected models with Optuna
            optuna_results = train_multiple_models_with_optuna(
                X_train_opt, y_train_opt, X_val_opt, y_val_opt, X_test, y_test,
                model_types_to_train, n_trials, progress_placeholder
            )

            # Find best model
            best_model_name = None
            best_f1 = 0
            for model_name, result in optuna_results.items():
                if 'test_metrics' in result:
                    f1 = result['test_metrics']['f1_macro']
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model_name = model_name

            # Store all results and best model in session state
            st.session_state['osc_optuna_results'] = optuna_results
            st.session_state['osc_best_model_name'] = best_model_name

            if best_model_name:
                best_result = optuna_results[best_model_name]
                st.session_state['osc_model'] = best_result['model']
                st.session_state['osc_metrics'] = best_result['test_metrics']
                st.session_state['osc_label_encoder'] = best_result.get('label_encoder')  # For XGBoost
                st.session_state['osc_scaler'] = scaler
                st.session_state['osc_features'] = feature_cols
                st.session_state['osc_df'] = df
                st.session_state['osc_X_test'] = X_test
                st.session_state['osc_y_test'] = y_test

            progress_placeholder.empty()
            st.success(f"Optimization complete! Best model: {best_model_name} (F1: {best_f1:.4f})")

        # Display Optuna results if available
        if 'osc_optuna_results' in st.session_state:
            st.subheader("Optimization Results")

            optuna_results = st.session_state['osc_optuna_results']
            best_model_name = st.session_state.get('osc_best_model_name')

            # Summary table
            summary_data = []
            for model_name, result in optuna_results.items():
                if 'error' in result:
                    summary_data.append({
                        'Model': model_name,
                        'Val F1 (Best)': 'Error',
                        'Test F1': 'Error',
                        'Test Accuracy': 'Error',
                        'Trials': 'N/A',
                        'Status': f"Error: {result['error']}"
                    })
                else:
                    test_metrics = result.get('test_metrics', {})
                    is_best = " (Best)" if model_name == best_model_name else ""
                    summary_data.append({
                        'Model': f"{model_name}{is_best}",
                        'Val F1 (Best)': f"{result['best_score']:.4f}",
                        'Test F1': f"{test_metrics.get('f1_macro', 0):.4f}",
                        'Test Accuracy': f"{test_metrics.get('accuracy', 0):.1%}",
                        'Trials': result['n_trials'],
                        'Status': 'Success'
                    })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

            # Detailed results per model
            with st.expander("View Best Parameters Per Model", expanded=False):
                for model_name, result in optuna_results.items():
                    if 'best_params' in result:
                        st.write(f"**{model_name}:**")
                        st.json(result['best_params'])

            # Model selector to use for backtesting
            model_options = [name for name, r in optuna_results.items() if 'model' in r]
            if model_options:
                selected_model = st.selectbox(
                    "Select model for backtesting:",
                    model_options,
                    index=model_options.index(best_model_name) if best_model_name in model_options else 0
                )

                if st.button("Use Selected Model"):
                    result = optuna_results[selected_model]
                    st.session_state['osc_model'] = result['model']
                    st.session_state['osc_metrics'] = result['test_metrics']
                    st.session_state['osc_label_encoder'] = result.get('label_encoder')  # For XGBoost
                    st.session_state['osc_best_model_name'] = selected_model
                    st.success(f"Now using {selected_model} for backtesting")
                    st.rerun()

    else:
        # Non-Optuna training (original behavior)
        model_type = model_types_to_train[0] if model_types_to_train else 'random_forest'

        if st.button("Train Model", type="primary"):
            with st.spinner(f"Training {model_type}..."):
                model = train_model(X_train, y_train, model_type)
                metrics = evaluate_model(model, X_test, y_test)

            # Store in session state
            st.session_state['osc_model'] = model
            st.session_state['osc_metrics'] = metrics
            st.session_state['osc_scaler'] = scaler
            st.session_state['osc_features'] = feature_cols
            st.session_state['osc_df'] = df
            st.session_state['osc_X_test'] = X_test
            st.session_state['osc_y_test'] = y_test
            st.session_state['osc_best_model_name'] = model_type

            st.success("Model trained!")

    # Step 7: Evaluate Model
    if 'osc_model' in st.session_state:
        model_name = st.session_state.get('osc_best_model_name', 'unknown')
        st.header(f"Step 7: Model Evaluation ({model_name})")

        metrics = st.session_state['osc_metrics']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        col2.metric("Precision (macro)", f"{metrics['precision_macro']:.1%}")
        col3.metric("Recall (macro)", f"{metrics['recall_macro']:.1%}")
        col4.metric("F1 Score (macro)", f"{metrics['f1_macro']:.1%}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = metrics['confusion_matrix']
        # Labels are sorted: -1 (SELL), 0 (HOLD), +1 (BUY)
        cm_df = pd.DataFrame(cm,
                             index=['Actual: SELL (-1)', 'Actual: HOLD (0)', 'Actual: BUY (+1)'],
                             columns=['Pred: SELL (-1)', 'Pred: HOLD (0)', 'Pred: BUY (+1)'])
        st.dataframe(cm_df)

        # Classification Report
        with st.expander("Full Classification Report"):
            st.text(metrics['classification_report'])

        # Feature Importance
        st.subheader("Feature Importance (Top 20)")
        model = st.session_state['osc_model']
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': st.session_state['osc_features'],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(20)

            fig_imp = go.Figure(go.Bar(x=importance_df['Importance'],
                                        y=importance_df['Feature'],
                                        orientation='h'))
            fig_imp.update_layout(height=500, title="Feature Importance",
                                  yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)

        # Step 8: Backtest
        st.header("Step 8: Backtest on Test Set")

        df = st.session_state['osc_df']
        X_test = st.session_state['osc_X_test']
        y_pred = metrics['y_pred']

        # Get test period data - ensure indices align
        # y_pred corresponds to X_test rows, so use X_test.index
        test_indices = X_test.index

        # Validate lengths match before assignment
        if len(y_pred) != len(test_indices):
            st.error(f"Data mismatch: predictions ({len(y_pred)}) vs test data ({len(test_indices)}). Please re-train the model.")
            st.stop()

        test_df = df.loc[test_indices].copy()
        test_df['prediction'] = y_pred

        # ============================================================
        # PROMINENT SIGNAL SOURCE TOGGLE
        # ============================================================
        st.markdown("---")
        toggle_col1, toggle_col2, toggle_col3 = st.columns([1, 2, 1])

        with toggle_col2:
            st.markdown("### Signal Source")
            use_scipy = st.toggle(
                "Use Scipy Peaks?",
                value=False,
                help="Toggle ON for Scipy peaks/valleys, OFF for ML predictions"
            )

            if use_scipy:
                signal_mode = "SCIPY PEAKS"
                signal_color = "#00ff00"  # Green
                st.success("**SCIPY PEAKS MODE**: Trading on detected peaks/valleys (green/red dots)")
            else:
                signal_mode = "ML PREDICTIONS"
                signal_color = "#ff6600"  # Orange
                st.info(f"**ML MODE ({model_name})**: Trading on model predictions")

        st.markdown("---")

        use_direct_signals = use_scipy

        # Calculate returns
        test_df['returns'] = test_df['close'].pct_change()

        # Strategy: BUY on valley (oversold), SELL on peak (overbought)
        # Track actual trades (entries and exits)
        test_df['position'] = 0
        position = 0
        positions = []
        trades = []
        entry_price = None
        entry_date = None

        for i in range(len(test_df)):
            date = test_df.index[i]
            price = test_df['close'].iloc[i]

            # Determine signal based on source
            if use_direct_signals:
                # Use actual oscillator peaks/valleys
                is_valley = test_df['is_valley'].iloc[i] == 1
                is_peak = test_df['is_peak'].iloc[i] == 1
                buy_signal = is_valley
                sell_signal = is_peak
            else:
                # Use ML predictions: +1 = BUY (valley), -1 = SELL (peak)
                signal = test_df['prediction'].iloc[i]
                buy_signal = (signal == 1)
                sell_signal = (signal == -1)

            # BUY at valley (oversold)
            if buy_signal and position == 0:
                position = 1
                entry_price = price
                entry_date = date
                trades.append({
                    'type': 'entry',
                    'date': date,
                    'price': price,
                    'signal': 'BUY'
                })
            # SELL at peak (overbought)
            elif sell_signal and position == 1:
                position = 0
                pnl = ((price - entry_price) / entry_price) * 100 if entry_price else 0
                trades.append({
                    'type': 'exit',
                    'date': date,
                    'price': price,
                    'signal': 'SELL',
                    'pnl': pnl,
                    'entry_date': entry_date,
                    'entry_price': entry_price
                })
                entry_price = None
                entry_date = None

            positions.append(position)

        test_df['position'] = positions
        test_df['strategy_returns'] = test_df['position'].shift(1) * test_df['returns']

        # Calculate cumulative returns
        test_df['cum_market'] = (1 + test_df['returns']).cumprod()
        test_df['cum_strategy'] = (1 + test_df['strategy_returns'].fillna(0)).cumprod()

        # Performance metrics
        market_return = (test_df['cum_market'].iloc[-1] - 1) * 100
        strategy_return = (test_df['cum_strategy'].iloc[-1] - 1) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Market Return", f"{market_return:.1f}%")
        col2.metric("Strategy Return", f"{strategy_return:.1f}%")
        col3.metric("Outperformance", f"{strategy_return - market_return:.1f}%")

        # ============================================================
        # COMPREHENSIVE CANDLESTICK CHART WITH SIGNALS
        # ============================================================
        st.subheader(f"Trading Chart - {signal_mode}")

        # Create ChartHelper for gap-free intraday charts
        chart_ml = ChartHelper(test_df, interval)
        x_vals_ml = chart_ml.get_x(test_df.index)

        # Dynamic subplot titles based on mode
        signal_subplot_title = 'Scipy Peak/Valley Signals' if use_scipy else 'ML Predictions'

        # Create subplots: Candlestick, Oscillator, Signals
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.2, 0.15, 0.15],
            subplot_titles=(f'Price Action - {signal_mode}', 'Composite Oscillator',
                           signal_subplot_title, 'Cumulative Returns')
        )

        # Row 1: Candlestick Chart
        fig.add_trace(
            go.Candlestick(
                x=x_vals_ml,
                open=test_df['open'],
                high=test_df['high'],
                low=test_df['low'],
                close=test_df['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )

        # Add entry markers (BUY)
        entries = [t for t in trades if t['type'] == 'entry']
        if entries:
            entry_dates = [t['date'] for t in entries]
            entry_prices = [t['price'] for t in entries]
            fig.add_trace(
                go.Scatter(
                    x=chart_ml.get_x(entry_dates),
                    y=entry_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='lime',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='BUY Entry',
                    hovertemplate='BUY<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )

        # Add exit markers (SELL)
        exits = [t for t in trades if t['type'] == 'exit']
        if exits:
            exit_dates = [t['date'] for t in exits]
            exit_prices = [t['price'] for t in exits]
            exit_pnls = [t.get('pnl', 0) for t in exits]
            fig.add_trace(
                go.Scatter(
                    x=chart_ml.get_x(exit_dates),
                    y=exit_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='SELL Exit',
                    hovertemplate='SELL<br>Date: %{x}<br>Price: $%{y:.2f}<br>PnL: ' +
                                  '<br>'.join([f'{p:.1f}%' for p in exit_pnls]) + '<extra></extra>'
                ),
                row=1, col=1
            )

        # Draw trade lines connecting entries to exits
        for trade in exits:
            if 'entry_date' in trade and 'entry_price' in trade:
                color = 'rgba(0,255,0,0.3)' if trade.get('pnl', 0) > 0 else 'rgba(255,0,0,0.3)'
                fig.add_trace(
                    go.Scatter(
                        x=chart_ml.get_x([trade['entry_date'], trade['date']]),
                        y=[trade['entry_price'], trade['price']],
                        mode='lines',
                        line=dict(color=color, width=2, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )

        # Row 2: Composite Oscillator
        osc_col = 'composite_smooth' if 'composite_smooth' in test_df.columns else 'composite_oscillator'
        if osc_col in test_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_vals_ml,
                    y=test_df[osc_col],
                    mode='lines',
                    name='Oscillator',
                    line=dict(color='purple', width=1.5)
                ),
                row=2, col=1
            )
            # Add threshold lines
            fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                         annotation_text="Overbought", row=2, col=1)
            fig.add_hline(y=-0.5, line_dash="dash", line_color="green",
                         annotation_text="Oversold", row=2, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

            # Mark oscillator peaks/valleys
            if 'is_peak' in test_df.columns:
                peaks = test_df[test_df['is_peak'] == 1]
                if len(peaks) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=chart_ml.get_x(peaks.index),
                            y=peaks[osc_col],
                            mode='markers',
                            marker=dict(symbol='circle', size=8, color='red'),
                            name='Osc Peak',
                            showlegend=True
                        ),
                        row=2, col=1
                    )
            if 'is_valley' in test_df.columns:
                valleys = test_df[test_df['is_valley'] == 1]
                if len(valleys) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=chart_ml.get_x(valleys.index),
                            y=valleys[osc_col],
                            mode='markers',
                            marker=dict(symbol='circle', size=8, color='green'),
                            name='Osc Valley',
                            showlegend=True
                        ),
                        row=2, col=1
                    )

        # Row 3: Signal bars (different based on mode)
        if use_scipy:
            # Create scipy signal series: +1 for valley (BUY), -1 for peak (SELL), 0 otherwise
            scipy_signals = pd.Series(0, index=test_df.index)
            scipy_signals[test_df['is_valley'] == 1] = 1   # Valley = BUY
            scipy_signals[test_df['is_peak'] == 1] = -1    # Peak = SELL
            signal_colors = scipy_signals.map({-1: 'red', 0: 'gray', 1: 'lime'})
            fig.add_trace(
                go.Bar(
                    x=x_vals_ml,
                    y=scipy_signals,
                    marker_color=signal_colors,
                    name='Scipy Signal',
                    hovertemplate='Signal: %{y}<br>Date: %{x}<extra></extra>'
                ),
                row=3, col=1
            )
        else:
            # ML Predictions as colored bars
            colors = test_df['prediction'].map({-1: 'red', 0: 'gray', 1: 'lime'})
            fig.add_trace(
                go.Bar(
                    x=x_vals_ml,
                    y=test_df['prediction'],
                    marker_color=colors,
                    name='ML Signal',
                    hovertemplate='Signal: %{y}<br>Date: %{x}<extra></extra>'
                ),
                row=3, col=1
            )

        # Row 4: Cumulative Returns
        fig.add_trace(
            go.Scatter(
                x=x_vals_ml,
                y=test_df['cum_market'],
                mode='lines',
                name='Buy & Hold',
                line=dict(color='blue', width=2)
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals_ml,
                y=test_df['cum_strategy'],
                mode='lines',
                name='Strategy',
                line=dict(color='orange', width=2)
            ),
            row=4, col=1
        )

        # Update layout
        fig.update_layout(
            height=1000,
            title_text=f"Trading Analysis - {signal_mode}",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False
        )

        # Update y-axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Oscillator", row=2, col=1, range=[-1.2, 1.2])
        fig.update_yaxes(title_text="Signal", row=3, col=1, tickvals=[-1, 0, 1],
                        ticktext=['BUY', 'HOLD', 'SELL'])
        fig.update_yaxes(title_text="Cum. Return", row=4, col=1)

        # Apply formatting for intraday charts
        chart_ml.apply_formatting(fig)
        st.plotly_chart(fig, use_container_width=True)

        # ============================================================
        # PREDICTION ALIGNMENT ANALYSIS (ML Mode Only)
        # ============================================================
        if not use_scipy:
            st.subheader("Prediction Alignment Analysis")
            st.caption("Comparing ML predictions to actual peaks/valleys detected by scipy")

            # Get lookahead from session state or sidebar
            lookahead_val = lookahead if 'lookahead' in dir() else 3

            # Calculate alignment metrics
            predictions = test_df['prediction'].values
            actual_peaks = test_df['is_peak'].values
            actual_valleys = test_df['is_valley'].values

            # For each prediction, check if there's an actual peak/valley within lookahead window
            correct_buy = 0  # Predicted BUY (+1) near actual valley
            correct_sell = 0  # Predicted SELL (-1) near actual peak
            false_buy = 0  # Predicted BUY but no valley nearby
            false_sell = 0  # Predicted SELL but no peak nearby
            missed_valleys = 0  # Actual valley but no BUY prediction nearby
            missed_peaks = 0  # Actual peak but no SELL prediction nearby
            correct_hold = 0  # Predicted HOLD (0) correctly

            n = len(predictions)

            # Track which peaks/valleys were caught
            valley_caught = [False] * n
            peak_caught = [False] * n

            # Analyze each prediction
            alignment_status = []  # For visualization
            for i in range(n):
                pred = predictions[i]

                if pred == 1:  # BUY prediction
                    # Check if there's a valley within lookahead window ahead
                    found_valley = False
                    for j in range(i, min(i + lookahead_val + 1, n)):
                        if actual_valleys[j] == 1:
                            found_valley = True
                            valley_caught[j] = True
                            break
                    if found_valley:
                        correct_buy += 1
                        alignment_status.append('correct_buy')
                    else:
                        false_buy += 1
                        alignment_status.append('false_buy')

                elif pred == -1:  # SELL prediction
                    # Check if there's a peak within lookahead window ahead
                    found_peak = False
                    for j in range(i, min(i + lookahead_val + 1, n)):
                        if actual_peaks[j] == 1:
                            found_peak = True
                            peak_caught[j] = True
                            break
                    if found_peak:
                        correct_sell += 1
                        alignment_status.append('correct_sell')
                    else:
                        false_sell += 1
                        alignment_status.append('false_sell')
                else:  # HOLD prediction
                    # Check if there's NO peak or valley in the lookahead window
                    has_signal = False
                    for j in range(i, min(i + lookahead_val + 1, n)):
                        if actual_peaks[j] == 1 or actual_valleys[j] == 1:
                            has_signal = True
                            break
                    if not has_signal:
                        correct_hold += 1
                        alignment_status.append('correct_hold')
                    else:
                        alignment_status.append('missed')

            # Count missed peaks/valleys
            for i in range(n):
                if actual_valleys[i] == 1 and not valley_caught[i]:
                    missed_valleys += 1
                if actual_peaks[i] == 1 and not peak_caught[i]:
                    missed_peaks += 1

            test_df['alignment'] = alignment_status

            # Calculate summary statistics
            total_buy_preds = correct_buy + false_buy
            total_sell_preds = correct_sell + false_sell
            total_valleys = int(actual_valleys.sum())
            total_peaks = int(actual_peaks.sum())

            buy_precision = (correct_buy / total_buy_preds * 100) if total_buy_preds > 0 else 0
            sell_precision = (correct_sell / total_sell_preds * 100) if total_sell_preds > 0 else 0
            valley_recall = ((total_valleys - missed_valleys) / total_valleys * 100) if total_valleys > 0 else 0
            peak_recall = ((total_peaks - missed_peaks) / total_peaks * 100) if total_peaks > 0 else 0

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("BUY Precision", f"{buy_precision:.1f}%",
                         help=f"Of {total_buy_preds} BUY predictions, {correct_buy} were near actual valleys")
            with col2:
                st.metric("SELL Precision", f"{sell_precision:.1f}%",
                         help=f"Of {total_sell_preds} SELL predictions, {correct_sell} were near actual peaks")
            with col3:
                st.metric("Valley Recall", f"{valley_recall:.1f}%",
                         help=f"Caught {total_valleys - missed_valleys} of {total_valleys} valleys")
            with col4:
                st.metric("Peak Recall", f"{peak_recall:.1f}%",
                         help=f"Caught {total_peaks - missed_peaks} of {total_peaks} peaks")

            # Create alignment visualization
            # Use existing chart_ml helper for gap-free intraday charts
            fig_align = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=('Oscillator with Prediction Quality', 'Prediction Alignment'),
                row_heights=[0.6, 0.4]
            )

            # Row 1: Oscillator with colored markers for prediction quality
            fig_align.add_trace(
                go.Scatter(
                    x=x_vals_ml,
                    y=test_df['composite_smooth'],
                    mode='lines',
                    name='Oscillator',
                    line=dict(color='purple', width=1.5)
                ),
                row=1, col=1
            )

            # Add actual peaks (red diamonds)
            peak_mask = test_df['is_peak'] == 1
            fig_align.add_trace(
                go.Scatter(
                    x=chart_ml.get_x(test_df.index[peak_mask]),
                    y=test_df.loc[peak_mask, 'composite_smooth'],
                    mode='markers',
                    name='Actual Peak',
                    marker=dict(symbol='diamond', size=12, color='red', line=dict(width=2, color='darkred'))
                ),
                row=1, col=1
            )

            # Add actual valleys (green diamonds)
            valley_mask = test_df['is_valley'] == 1
            fig_align.add_trace(
                go.Scatter(
                    x=chart_ml.get_x(test_df.index[valley_mask]),
                    y=test_df.loc[valley_mask, 'composite_smooth'],
                    mode='markers',
                    name='Actual Valley',
                    marker=dict(symbol='diamond', size=12, color='lime', line=dict(width=2, color='darkgreen'))
                ),
                row=1, col=1
            )

            # Add correct BUY predictions (green circles)
            correct_buy_mask = test_df['alignment'] == 'correct_buy'
            if correct_buy_mask.sum() > 0:
                fig_align.add_trace(
                    go.Scatter(
                        x=chart_ml.get_x(test_df.index[correct_buy_mask]),
                        y=test_df.loc[correct_buy_mask, 'composite_smooth'],
                        mode='markers',
                        name=f'Correct BUY ({correct_buy})',
                        marker=dict(symbol='triangle-up', size=10, color='green', opacity=0.7)
                    ),
                    row=1, col=1
                )

            # Add false BUY predictions (orange circles)
            false_buy_mask = test_df['alignment'] == 'false_buy'
            if false_buy_mask.sum() > 0:
                fig_align.add_trace(
                    go.Scatter(
                        x=chart_ml.get_x(test_df.index[false_buy_mask]),
                        y=test_df.loc[false_buy_mask, 'composite_smooth'],
                        mode='markers',
                        name=f'False BUY ({false_buy})',
                        marker=dict(symbol='triangle-up', size=8, color='orange', opacity=0.6)
                    ),
                    row=1, col=1
                )

            # Add correct SELL predictions (red triangles down)
            correct_sell_mask = test_df['alignment'] == 'correct_sell'
            if correct_sell_mask.sum() > 0:
                fig_align.add_trace(
                    go.Scatter(
                        x=chart_ml.get_x(test_df.index[correct_sell_mask]),
                        y=test_df.loc[correct_sell_mask, 'composite_smooth'],
                        mode='markers',
                        name=f'Correct SELL ({correct_sell})',
                        marker=dict(symbol='triangle-down', size=10, color='red', opacity=0.7)
                    ),
                    row=1, col=1
                )

            # Add false SELL predictions (pink triangles down)
            false_sell_mask = test_df['alignment'] == 'false_sell'
            if false_sell_mask.sum() > 0:
                fig_align.add_trace(
                    go.Scatter(
                        x=chart_ml.get_x(test_df.index[false_sell_mask]),
                        y=test_df.loc[false_sell_mask, 'composite_smooth'],
                        mode='markers',
                        name=f'False SELL ({false_sell})',
                        marker=dict(symbol='triangle-down', size=8, color='pink', opacity=0.6)
                    ),
                    row=1, col=1
                )

            # Row 2: Bar chart of alignment categories
            alignment_colors = {
                'correct_buy': 'green',
                'false_buy': 'orange',
                'correct_sell': 'red',
                'false_sell': 'pink',
                'correct_hold': 'lightgray',
                'missed': 'yellow'
            }
            bar_colors = [alignment_colors.get(s, 'gray') for s in alignment_status]
            bar_values = [1 if s.startswith('correct') else -1 if 'false' in s else 0 for s in alignment_status]

            fig_align.add_trace(
                go.Bar(
                    x=x_vals_ml,
                    y=bar_values,
                    marker_color=bar_colors,
                    name='Alignment',
                    hovertemplate='%{x}<br>Status: ' + pd.Series(alignment_status) + '<extra></extra>'
                ),
                row=2, col=1
            )

            # Add overbought/oversold lines
            fig_align.add_hline(y=0.5, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
            fig_align.add_hline(y=-0.5, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)

            fig_align.update_layout(
                height=500,
                title_text="ML Prediction Alignment with Actual Peaks/Valleys",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_align.update_yaxes(title_text="Oscillator", row=1, col=1)
            fig_align.update_yaxes(title_text="Quality", row=2, col=1,
                                   tickvals=[-1, 0, 1], ticktext=['False', 'Hold', 'Correct'])

            # Apply formatting for intraday charts
            chart_ml.apply_formatting(fig_align)
            st.plotly_chart(fig_align, use_container_width=True)

            # Detailed breakdown
            with st.expander("Detailed Alignment Breakdown"):
                st.markdown(f"""
                **Prediction Summary (Lookahead Window: {lookahead_val} bars)**

                | Category | Count | Description |
                |----------|-------|-------------|
                | Correct BUY | {correct_buy} | BUY prediction within {lookahead_val} bars before actual valley |
                | False BUY | {false_buy} | BUY prediction with no valley in window |
                | Correct SELL | {correct_sell} | SELL prediction within {lookahead_val} bars before actual peak |
                | False SELL | {false_sell} | SELL prediction with no peak in window |
                | Correct HOLD | {correct_hold} | HOLD prediction when no signal expected |
                | Missed Valleys | {missed_valleys} | Valleys with no nearby BUY prediction |
                | Missed Peaks | {missed_peaks} | Peaks with no nearby SELL prediction |

                **Interpretation:**
                - **Precision** measures how many of your predictions were correct
                - **Recall** measures how many actual peaks/valleys you caught
                - Low precision = too many false signals (noisy)
                - Low recall = missing opportunities
                """)

        # ============================================================
        # COMPREHENSIVE TRADING STATISTICS
        # ============================================================
        stats_title = "Scipy Peaks Trading Statistics" if use_scipy else f"ML Model Trading Statistics ({model_name})"
        st.subheader(stats_title)

        if exits:
            # Calculate all statistics
            all_pnls = [t.get('pnl', 0) for t in exits]
            wins = [t for t in exits if t.get('pnl', 0) > 0]
            losses = [t for t in exits if t.get('pnl', 0) <= 0]
            total_trades = len(exits)
            win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0

            avg_pnl = np.mean(all_pnls) if all_pnls else 0
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
            max_win = max(all_pnls) if all_pnls else 0
            max_loss = min(all_pnls) if all_pnls else 0

            # Compounded return
            compounded_return = np.prod([1 + p/100 for p in all_pnls]) - 1
            compounded_return_pct = compounded_return * 100

            # Profit Factor
            gross_profit = sum([t['pnl'] for t in wins]) if wins else 0
            gross_loss = abs(sum([t['pnl'] for t in losses])) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Calculate hold times
            hold_times = []
            for t in exits:
                if t.get('entry_date') and t.get('date'):
                    hold_days = (t['date'] - t['entry_date']).days
                    hold_times.append(hold_days)
            avg_hold = np.mean(hold_times) if hold_times else 0

            # Expectancy
            expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

            # Consecutive wins/losses
            win_loss_streak = [1 if t.get('pnl', 0) > 0 else 0 for t in exits]
            max_consec_wins = 0
            max_consec_losses = 0
            current_wins = 0
            current_losses = 0
            for wl in win_loss_streak:
                if wl == 1:
                    current_wins += 1
                    current_losses = 0
                    max_consec_wins = max(max_consec_wins, current_wins)
                else:
                    current_losses += 1
                    current_wins = 0
                    max_consec_losses = max(max_consec_losses, current_losses)

            # Display metrics - Row 1: Core Stats
            st.write("**Core Performance:**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", total_trades)
            col2.metric("Win Rate", f"{win_rate:.1f}%")
            col3.metric("Avg Trade", f"{avg_pnl:.2f}%")
            col4.metric("Compounded Return", f"{compounded_return_pct:.1f}%")

            # Row 2: Win/Loss Details
            st.write("**Win/Loss Analysis:**")
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Avg Win", f"{avg_win:.2f}%")
            col6.metric("Avg Loss", f"{avg_loss:.2f}%")
            col7.metric("Best Trade", f"{max_win:.2f}%")
            col8.metric("Worst Trade", f"{max_loss:.2f}%")

            # Row 3: Advanced Metrics
            st.write("**Advanced Metrics:**")
            col9, col10, col11, col12 = st.columns(4)
            col9.metric("Profit Factor", f"{profit_factor:.2f}")
            col10.metric("Expectancy", f"{expectancy:.2f}%")
            col11.metric("Avg Hold (days)", f"{avg_hold:.1f}")
            col12.metric("Wins / Losses", f"{len(wins)} / {len(losses)}")

            # Row 4: Streaks
            st.write("**Streaks:**")
            col13, col14, col15, col16 = st.columns(4)
            col13.metric("Max Consec. Wins", max_consec_wins)
            col14.metric("Max Consec. Losses", max_consec_losses)
            col15.metric("Gross Profit", f"{gross_profit:.1f}%")
            col16.metric("Gross Loss", f"{gross_loss:.1f}%")

            # Comparison with Buy & Hold
            st.markdown("---")
            st.write("**Strategy vs Buy & Hold:**")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Strategy Return", f"{strategy_return:.1f}%")
            col_b.metric("Buy & Hold Return", f"{market_return:.1f}%")
            outperf = strategy_return - market_return
            col_c.metric("Outperformance", f"{outperf:+.1f}%",
                        delta=f"{'Better' if outperf > 0 else 'Worse'}")

            # Trade table
            st.markdown("---")
            st.subheader("Trade Log")
            trade_summary = []
            for i, exit_trade in enumerate(exits):
                hold_d = (exit_trade['date'] - exit_trade['entry_date']).days if exit_trade.get('entry_date') else 0
                trade_summary.append({
                    '#': i + 1,
                    'Entry Date': exit_trade.get('entry_date', 'N/A'),
                    'Entry $': f"${exit_trade.get('entry_price', 0):.2f}",
                    'Exit Date': exit_trade['date'],
                    'Exit $': f"${exit_trade['price']:.2f}",
                    'P&L': f"{exit_trade.get('pnl', 0):.2f}%",
                    'Days': hold_d,
                    'Result': '✅' if exit_trade.get('pnl', 0) > 0 else '❌'
                })

            trade_df = pd.DataFrame(trade_summary)
            st.dataframe(trade_df, use_container_width=True)

        else:
            st.info("No completed trades in test period")

        # Signal distribution in test period
        st.markdown("---")
        signal_dist_title = "Scipy Signal Distribution" if use_scipy else "ML Signal Distribution"
        st.subheader(f"{signal_dist_title} in Test Period")

        if use_scipy:
            # Count scipy peaks/valleys
            n_peaks_test = (test_df['is_peak'] == 1).sum()
            n_valleys_test = (test_df['is_valley'] == 1).sum()
            n_neutral = len(test_df) - n_peaks_test - n_valleys_test

            col1, col2, col3 = st.columns(3)
            col1.metric("SELL Signals (Peaks)", n_peaks_test)
            col2.metric("HOLD (Neutral)", n_neutral)
            col3.metric("BUY Signals (Valleys)", n_valleys_test)

            total_signals = len(test_df)
            st.write(f"Signal rates: SELL {n_peaks_test/total_signals*100:.1f}% | "
                    f"HOLD {n_neutral/total_signals*100:.1f}% | "
                    f"BUY {n_valleys_test/total_signals*100:.1f}%")
        else:
            pred_counts = pd.Series(y_pred).value_counts().sort_index()

            col1, col2, col3 = st.columns(3)
            col1.metric("SELL Signals (-1)", pred_counts.get(-1, 0))
            col2.metric("HOLD Signals (0)", pred_counts.get(0, 0))
            col3.metric("BUY Signals (+1)", pred_counts.get(1, 0))

            # Signal percentages
            total_signals = len(y_pred)
            st.write(f"Signal rates: SELL {pred_counts.get(-1, 0)/total_signals*100:.1f}% | "
                    f"HOLD {pred_counts.get(0, 0)/total_signals*100:.1f}% | "
                    f"BUY {pred_counts.get(1, 0)/total_signals*100:.1f}%")


# ============================================================================
# STEP 5d: STRATEGY DISCOVERY ENGINE
# ============================================================================

def render_strategy_discovery_section(df: pd.DataFrame):
    """Render the Strategy Discovery Engine section"""

    st.header("Step 5d: Strategy Discovery Engine")
    st.markdown("""
    🚀 **Systematic Strategy Discovery** - This engine:
    - Generates **100+ technical indicators** using pandas_ta
    - Creates **1000+ trading conditions** (slopes, crossovers, divergences, combinations)
    - Tests **thousands of strategy combinations** via grid search
    - Uses **ML feature importance** to identify what matters
    - **Surfaces winning strategies** with interpretable rules
    """)

    with st.expander("📊 Configuration", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            max_strategies = st.slider(
                "Max Strategies to Test",
                min_value=100,
                max_value=5000,
                value=500,
                step=100,
                help="More strategies = better coverage but longer runtime"
            )

        with col2:
            n_jobs = st.slider(
                "Parallel Workers",
                min_value=1,
                max_value=os.cpu_count() or 4,
                value=max(1, (os.cpu_count() or 4) - 1),
                help="Number of CPU cores to use (for fast mode)"
            )

        with col3:
            output_dir = st.text_input(
                "Output Directory",
                value="analysis2",
                help="Directory for CSV outputs"
            )

        with col4:
            show_progress = st.checkbox(
                "Show Progress",
                value=True,
                help="Show detailed progress (slower) vs fast parallel mode"
            )

        col5, col6 = st.columns(2)
        with col5:
            sort_by = st.selectbox(
                "Sort Results By",
                options=['total_return', 'sharpe_ratio', 'win_rate', 'profit_factor', 'composite'],
                index=0,
                format_func=lambda x: {
                    'total_return': 'Total Return (Highest First)',
                    'sharpe_ratio': 'Sharpe Ratio (Best Risk-Adjusted)',
                    'win_rate': 'Win Rate (Most Consistent)',
                    'profit_factor': 'Profit Factor (Best Edge)',
                    'composite': 'Composite Score (Balanced)'
                }.get(x, x),
                help="How to rank discovered strategies"
            )

    # Check if we have enough data
    if len(df) < 100:
        st.warning("Need at least 100 data points for strategy discovery.")
        return

    if st.button("🚀 Run Strategy Discovery", type="primary", key="run_discovery"):
        try:
            from strategy_discovery_engine import StrategyDiscoveryEngine

            # Create progress display elements
            progress_container = st.container()

            with progress_container:
                progress_bar = st.progress(0, text="Initializing...")
                status_text = st.empty()
                detail_text = st.empty()
                log_expander = st.expander("📋 Discovery Log", expanded=True)
                log_container = log_expander.container()
                log_messages = []

                def update_streamlit_progress(step, total_steps, message, detail=""):
                    """Callback to update Streamlit UI with progress"""
                    progress_pct = step / total_steps if total_steps > 0 else 0
                    progress_bar.progress(progress_pct, text=message)
                    status_text.markdown(f"**{message}**")
                    if detail:
                        detail_text.text(detail)

                    # Add to log
                    log_messages.append(f"{message}")
                    if detail:
                        log_messages.append(f"  → {detail}")

                    # Update log display (show last 15 messages)
                    log_container.code("\n".join(log_messages[-15:]), language=None)

                # Initialize engine
                engine = StrategyDiscoveryEngine(df, output_dir)

                # Run discovery with optional progress callback
                if show_progress:
                    results = engine.run_discovery(
                        max_strategies=max_strategies,
                        n_jobs=n_jobs,
                        verbose=True,
                        progress_callback=update_streamlit_progress,
                        sort_by=sort_by
                    )
                else:
                    # Fast mode - parallel execution, less detailed progress
                    update_streamlit_progress(0, 6, "🚀 Running in FAST mode...", "Using parallel execution (no detailed backtest progress)")
                    results = engine.run_discovery(
                        max_strategies=max_strategies,
                        n_jobs=n_jobs,
                        verbose=True,
                        progress_callback=None,  # No callback = parallel mode
                        sort_by=sort_by
                    )

                # Final progress update
                progress_bar.progress(1.0, text="🎉 Discovery Complete!")

                # Store results in session state
                st.session_state['discovery_results'] = results
                st.session_state['discovery_engine'] = engine
                st.session_state['discovery_output_dir'] = output_dir

                if results:
                    st.success(f"✅ Discovery complete! Found {len(results)} valid strategies. Top return: {results[0].total_return:.1%}")
                else:
                    st.warning("Discovery complete but no valid strategies found. Try adjusting parameters.")

        except ImportError as e:
            st.error(f"Could not import strategy_discovery_engine: {e}")
            st.info("Make sure strategy_discovery_engine.py is in the same directory.")
        except Exception as e:
            st.error(f"Error running discovery: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Display results if available
    if 'discovery_results' in st.session_state:
        results = st.session_state['discovery_results']
        output_dir = st.session_state.get('discovery_output_dir', 'analysis2')

        st.markdown("---")
        st.subheader("🏆 Top Discovered Strategies")

        # Re-sort option
        resort_col1, resort_col2 = st.columns([3, 1])
        with resort_col1:
            resort_by = st.selectbox(
                "Re-sort results by:",
                options=['total_return', 'outperformance', 'sharpe_ratio', 'win_rate', 'profit_factor', 'num_trades'],
                index=0,
                format_func=lambda x: {
                    'total_return': 'Total Return',
                    'outperformance': 'Outperformance vs Buy & Hold',
                    'sharpe_ratio': 'Sharpe Ratio',
                    'win_rate': 'Win Rate',
                    'profit_factor': 'Profit Factor',
                    'num_trades': 'Number of Trades'
                }.get(x, x),
                key="resort_selector"
            )

        # Re-sort the results based on selection
        if resort_by == 'total_return':
            results = sorted(results, key=lambda r: r.total_return, reverse=True)
        elif resort_by == 'outperformance':
            results = sorted(results, key=lambda r: getattr(r, 'outperformance', r.total_return), reverse=True)
        elif resort_by == 'sharpe_ratio':
            results = sorted(results, key=lambda r: r.sharpe_ratio, reverse=True)
        elif resort_by == 'win_rate':
            results = sorted(results, key=lambda r: r.win_rate, reverse=True)
        elif resort_by == 'profit_factor':
            results = sorted(results, key=lambda r: r.profit_factor if r.profit_factor < 999 else 0, reverse=True)
        elif resort_by == 'num_trades':
            results = sorted(results, key=lambda r: r.num_trades, reverse=True)

        if results:
            # Show buy & hold baseline prominently
            buy_hold = results[0].buy_hold_return if hasattr(results[0], 'buy_hold_return') else 0
            st.info(f"📊 **Buy & Hold Baseline Return: {buy_hold:.1%}** (over the backtested period)")

            # Create summary dataframe
            top_n = min(20, len(results))

            summary_data = []
            for i, r in enumerate(results[:top_n]):
                outperf = getattr(r, 'outperformance', r.total_return - buy_hold)
                outperf_str = f"+{outperf:.1%}" if outperf >= 0 else f"{outperf:.1%}"
                summary_data.append({
                    'Rank': i + 1,
                    'Return': f"{r.total_return:.1%}",
                    'vs B&H': outperf_str,
                    'Sharpe': f"{r.sharpe_ratio:.2f}",
                    'Win Rate': f"{r.win_rate:.0%}",
                    'Trades': r.num_trades,
                    'Profit Factor': f"{r.profit_factor:.2f}",
                    'Max DD': f"{r.max_drawdown:.1%}",
                    'Avg Trade': f"{r.avg_trade:.2%}"
                })

            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

            # Detailed view of top strategies
            st.subheader("📋 Strategy Details")

            for i, r in enumerate(results[:5]):
                outperf = getattr(r, 'outperformance', r.total_return - buy_hold)
                outperf_label = f"+{outperf:.1%}" if outperf >= 0 else f"{outperf:.1%}"
                with st.expander(f"#{i+1}: {r.total_return:.1%} Return ({outperf_label} vs B&H) | {r.sharpe_ratio:.2f} Sharpe", expanded=(i==0)):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Performance Metrics:**")
                        st.write(f"- Total Return: {r.total_return:.2%}")
                        st.write(f"- Buy & Hold Return: {buy_hold:.2%}")
                        outperf_color = "green" if outperf >= 0 else "red"
                        st.markdown(f"- <span style='color:{outperf_color}'>**Outperformance: {outperf_label}**</span>", unsafe_allow_html=True)
                        st.write(f"- Sharpe Ratio: {r.sharpe_ratio:.2f}")
                        st.write(f"- Win Rate: {r.win_rate:.1%}")
                        st.write(f"- Number of Trades: {r.num_trades}")

                    with col2:
                        st.markdown("**Risk Metrics:**")
                        st.write(f"- Profit Factor: {r.profit_factor:.2f}")
                        st.write(f"- Max Drawdown: {r.max_drawdown:.2%}")
                        st.write(f"- Avg Trade: {r.avg_trade:.2%}")

                    st.markdown("**Trading Rules:**")
                    config = r.strategy_config

                    if config.get('entry_long'):
                        st.success(f"📈 **Entry Long:** {', '.join(config['entry_long'])}")

                    if config.get('exit_long'):
                        st.error(f"📉 **Exit Long:** {', '.join(config['exit_long'])}")

                    if config.get('entry_short'):
                        st.info(f"📉 **Entry Short:** {', '.join(config['entry_short'])}")

                    if config.get('exit_short'):
                        st.warning(f"📈 **Exit Short:** {', '.join(config['exit_short'])}")

                    if config.get('filters'):
                        st.write(f"🔍 **Filters:** {', '.join(config['filters'])}")

            # ============================================================
            # STRATEGY VISUALIZATION
            # ============================================================
            st.markdown("---")
            st.subheader("📈 Strategy Visualization")

            # Strategy selector
            strategy_options = [
                f"#{i+1}: {r.total_return:.1%} Return | {r.num_trades} trades"
                for i, r in enumerate(results[:20])
            ]

            selected_idx = st.selectbox(
                "Select a strategy to visualize:",
                range(len(strategy_options)),
                format_func=lambda x: strategy_options[x],
                key="strategy_viz_selector"
            )

            if selected_idx is not None and 'discovery_engine' in st.session_state:
                selected_strategy = results[selected_idx]
                engine = st.session_state['discovery_engine']

                # Get the strategy config
                config = selected_strategy.strategy_config
                entry_long = config.get('entry_long', [])
                exit_long = config.get('exit_long', [])
                entry_short = config.get('entry_short', [])
                exit_short = config.get('exit_short', [])
                filters = config.get('filters', [])

                # Re-run backtest to get detailed trade information
                from strategy_discovery_engine import StrategyBacktester

                backtester = StrategyBacktester(
                    engine.data,
                    engine.indicators,
                    engine.conditions
                )

                # Get entry/exit signals
                def combine_conditions(condition_names, conditions_dict, logic='AND'):
                    if not condition_names:
                        return pd.Series(False, index=engine.data.index)
                    valid = [conditions_dict[n] for n in condition_names if n in conditions_dict]
                    if not valid:
                        return pd.Series(False, index=engine.data.index)
                    if logic == 'AND':
                        result = valid[0]
                        for c in valid[1:]:
                            result = result & c
                    else:
                        result = valid[0]
                        for c in valid[1:]:
                            result = result | c
                    return result.fillna(False)

                entry_long_signal = combine_conditions(entry_long, engine.conditions, 'AND')
                exit_long_signal = combine_conditions(exit_long, engine.conditions, 'OR')

                if entry_short:
                    entry_short_signal = combine_conditions(entry_short, engine.conditions, 'AND')
                else:
                    entry_short_signal = pd.Series(False, index=engine.data.index)

                if exit_short:
                    exit_short_signal = combine_conditions(exit_short, engine.conditions, 'OR')
                else:
                    exit_short_signal = pd.Series(False, index=engine.data.index)

                if filters:
                    filter_signal = combine_conditions(filters, engine.conditions, 'AND')
                    entry_long_signal = entry_long_signal & filter_signal
                    entry_short_signal = entry_short_signal & filter_signal

                # Simulate trades
                trades = []
                position = None
                entry_price = 0
                entry_idx = 0
                capital = 100000
                equity_curve = [capital]

                close = engine.data['close'].values

                for i in range(1, len(engine.data)):
                    current_price = close[i]
                    current_date = engine.data.index[i]

                    # Check exits
                    if position == 'long' and exit_long_signal.iloc[i]:
                        pnl = (current_price - entry_price) / entry_price
                        capital *= (1 + pnl)
                        trades.append({
                            'type': 'long',
                            'entry_idx': entry_idx,
                            'entry_date': engine.data.index[entry_idx],
                            'entry_price': entry_price,
                            'exit_idx': i,
                            'exit_date': current_date,
                            'exit_price': current_price,
                            'pnl': pnl
                        })
                        position = None

                    elif position == 'short' and exit_short_signal.iloc[i]:
                        pnl = (entry_price - current_price) / entry_price
                        capital *= (1 + pnl)
                        trades.append({
                            'type': 'short',
                            'entry_idx': entry_idx,
                            'entry_date': engine.data.index[entry_idx],
                            'entry_price': entry_price,
                            'exit_idx': i,
                            'exit_date': current_date,
                            'exit_price': current_price,
                            'pnl': pnl
                        })
                        position = None

                    # Check entries
                    if position is None:
                        if entry_long_signal.iloc[i]:
                            position = 'long'
                            entry_price = current_price
                            entry_idx = i
                        elif entry_short_signal.iloc[i]:
                            position = 'short'
                            entry_price = current_price
                            entry_idx = i

                    equity_curve.append(capital)

                # Close open position
                if position == 'long':
                    pnl = (close[-1] - entry_price) / entry_price
                    trades.append({
                        'type': 'long',
                        'entry_idx': entry_idx,
                        'entry_date': engine.data.index[entry_idx],
                        'entry_price': entry_price,
                        'exit_idx': len(engine.data) - 1,
                        'exit_date': engine.data.index[-1],
                        'exit_price': close[-1],
                        'pnl': pnl,
                        'open': True
                    })
                elif position == 'short':
                    pnl = (entry_price - close[-1]) / entry_price
                    trades.append({
                        'type': 'short',
                        'entry_idx': entry_idx,
                        'entry_date': engine.data.index[entry_idx],
                        'entry_price': entry_price,
                        'exit_idx': len(engine.data) - 1,
                        'exit_date': engine.data.index[-1],
                        'exit_price': close[-1],
                        'pnl': pnl,
                        'open': True
                    })

                # Extract key indicators from strategy conditions
                def extract_indicator_name(condition_name):
                    """Extract the base indicator name from a condition"""
                    # Common patterns
                    parts = condition_name.split('_')
                    # Try to find known indicator prefixes
                    for prefix in ['rsi', 'macd', 'cci', 'stoch', 'adx', 'mfi', 'bb', 'ema', 'sma', 'willr', 'obv', 'cmf']:
                        if prefix in condition_name.lower():
                            # Find columns matching this indicator
                            for col in engine.indicators.columns:
                                if col.lower().startswith(prefix) and 'slope' not in col and 'dist' not in col:
                                    return col
                    return None

                # Get unique indicators from all conditions
                all_conditions = (entry_long or []) + (exit_long or []) + (entry_short or []) + (exit_short or [])
                key_indicators = []
                seen = set()
                for cond in all_conditions:
                    ind = extract_indicator_name(cond)
                    if ind and ind not in seen and ind in engine.indicators.columns:
                        key_indicators.append(ind)
                        seen.add(ind)

                # Limit to top 3 indicators for display
                key_indicators = key_indicators[:3]

                # Create visualization
                from plotly.subplots import make_subplots
                import plotly.graph_objects as go

                # Create ChartHelper for gap-free intraday charts
                chart_disc = ChartHelper(engine.data, interval)
                x_vals_disc = chart_disc.get_x(engine.data.index)

                n_subplots = 2 + len(key_indicators)  # Price, Equity, + indicators
                row_heights = [0.4] + [0.15] * (len(key_indicators)) + [0.2]

                subplot_titles = ['Price Action with Trades'] + key_indicators + ['Equity Curve']

                fig = make_subplots(
                    rows=n_subplots,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=row_heights,
                    subplot_titles=subplot_titles
                )

                # Row 1: Candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=x_vals_disc,
                        open=engine.data['open'],
                        high=engine.data['high'],
                        low=engine.data['low'],
                        close=engine.data['close'],
                        name='Price',
                        increasing_line_color='#26a69a',
                        decreasing_line_color='#ef5350'
                    ),
                    row=1, col=1
                )

                # Add trade markers
                if trades:
                    # Long entries
                    long_entries = [t for t in trades if t['type'] == 'long']
                    if long_entries:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_disc.get_x([t['entry_date'] for t in long_entries]),
                                y=[t['entry_price'] for t in long_entries],
                                mode='markers',
                                marker=dict(symbol='triangle-up', size=14, color='lime', line=dict(width=2, color='darkgreen')),
                                name='Long Entry',
                                hovertemplate='LONG ENTRY<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=chart_disc.get_x([t['exit_date'] for t in long_entries]),
                                y=[t['exit_price'] for t in long_entries],
                                mode='markers',
                                marker=dict(symbol='triangle-down', size=14, color='red', line=dict(width=2, color='darkred')),
                                name='Long Exit',
                                hovertemplate='LONG EXIT<br>Date: %{x}<br>Price: $%{y:.2f}<br>PnL: %{text}<extra></extra>',
                                text=[f"{t['pnl']:.1%}" for t in long_entries]
                            ),
                            row=1, col=1
                        )

                    # Short entries
                    short_entries = [t for t in trades if t['type'] == 'short']
                    if short_entries:
                        fig.add_trace(
                            go.Scatter(
                                x=chart_disc.get_x([t['entry_date'] for t in short_entries]),
                                y=[t['entry_price'] for t in short_entries],
                                mode='markers',
                                marker=dict(symbol='triangle-down', size=14, color='orange', line=dict(width=2, color='darkorange')),
                                name='Short Entry',
                                hovertemplate='SHORT ENTRY<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=chart_disc.get_x([t['exit_date'] for t in short_entries]),
                                y=[t['exit_price'] for t in short_entries],
                                mode='markers',
                                marker=dict(symbol='triangle-up', size=14, color='cyan', line=dict(width=2, color='darkcyan')),
                                name='Short Exit',
                                hovertemplate='SHORT EXIT<br>Date: %{x}<br>Price: $%{y:.2f}<br>PnL: %{text}<extra></extra>',
                                text=[f"{t['pnl']:.1%}" for t in short_entries]
                            ),
                            row=1, col=1
                        )

                    # Draw trade lines
                    for t in trades:
                        color = 'rgba(0,255,0,0.3)' if t['pnl'] > 0 else 'rgba(255,0,0,0.3)'
                        fig.add_trace(
                            go.Scatter(
                                x=chart_disc.get_x([t['entry_date'], t['exit_date']]),
                                y=[t['entry_price'], t['exit_price']],
                                mode='lines',
                                line=dict(color=color, width=2, dash='dot'),
                                showlegend=False,
                                hoverinfo='skip'
                            ),
                            row=1, col=1
                        )

                # Add indicator subplots
                for idx, ind_name in enumerate(key_indicators):
                    row_num = 2 + idx
                    ind_data = engine.indicators[ind_name]

                    # Determine color based on indicator type
                    if 'rsi' in ind_name.lower() or 'stoch' in ind_name.lower() or 'mfi' in ind_name.lower():
                        color = 'purple'
                        # Add overbought/oversold lines
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=row_num, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=row_num, col=1)
                    elif 'macd' in ind_name.lower():
                        color = 'blue'
                        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=row_num, col=1)
                    elif 'cci' in ind_name.lower():
                        color = 'orange'
                        fig.add_hline(y=100, line_dash="dash", line_color="red", row=row_num, col=1)
                        fig.add_hline(y=-100, line_dash="dash", line_color="green", row=row_num, col=1)
                    else:
                        color = 'teal'

                    fig.add_trace(
                        go.Scatter(
                            x=chart_disc.get_x(engine.indicators.index),
                            y=ind_data,
                            mode='lines',
                            name=ind_name,
                            line=dict(color=color, width=1.5)
                        ),
                        row=row_num, col=1
                    )

                # Add equity curve
                equity_df = pd.DataFrame({
                    'equity': equity_curve
                }, index=engine.data.index[:len(equity_curve)])

                fig.add_trace(
                    go.Scatter(
                        x=chart_disc.get_x(equity_df.index),
                        y=equity_df['equity'],
                        mode='lines',
                        name='Equity',
                        line=dict(color='gold', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(255,215,0,0.1)'
                    ),
                    row=n_subplots, col=1
                )

                # Update layout
                fig.update_layout(
                    title=f"Strategy #{selected_idx + 1}: {selected_strategy.total_return:.1%} Return | {len(trades)} Trades",
                    height=200 + 150 * n_subplots,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis_rangeslider_visible=False
                )

                # Apply formatting for intraday charts
                chart_disc.apply_formatting(fig)
                st.plotly_chart(fig, use_container_width=True)

                # Trade log
                if trades:
                    st.subheader("📋 Trade Log")
                    trade_df = pd.DataFrame([
                        {
                            '#': i + 1,
                            'Type': t['type'].upper(),
                            'Entry Date': t['entry_date'].strftime('%Y-%m-%d %H:%M') if hasattr(t['entry_date'], 'strftime') else str(t['entry_date']),
                            'Entry $': f"${t['entry_price']:.2f}",
                            'Exit Date': t['exit_date'].strftime('%Y-%m-%d %H:%M') if hasattr(t['exit_date'], 'strftime') else str(t['exit_date']),
                            'Exit $': f"${t['exit_price']:.2f}",
                            'PnL': f"{t['pnl']:.2%}",
                            'Result': '✅' if t['pnl'] > 0 else '❌',
                            'Status': '🔓 Open' if t.get('open') else '🔒 Closed'
                        }
                        for i, t in enumerate(trades)
                    ])
                    st.dataframe(trade_df, use_container_width=True)

                    # Trade summary stats
                    st.markdown("**Trade Statistics:**")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Trades", len(trades))
                    col2.metric("Winners", sum(1 for t in trades if t['pnl'] > 0))
                    col3.metric("Avg Win", f"{np.mean([t['pnl'] for t in trades if t['pnl'] > 0]):.2%}" if any(t['pnl'] > 0 for t in trades) else "N/A")
                    col4.metric("Avg Loss", f"{np.mean([t['pnl'] for t in trades if t['pnl'] < 0]):.2%}" if any(t['pnl'] < 0 for t in trades) else "N/A")

            # CSV download links
            st.markdown("---")
            st.subheader("📁 Output Files")

            csv_files = [
                ('indicators_raw.csv', 'All generated indicators'),
                ('conditions_signals.csv', 'All condition signals'),
                ('conditions_summary.csv', 'Condition frequency summary'),
                ('top_strategies.csv', 'Top performing strategies'),
                ('top_strategies.json', 'Detailed strategy configurations'),
                ('feature_importance.csv', 'ML feature importance rankings')
            ]

            cols = st.columns(3)
            for i, (filename, description) in enumerate(csv_files):
                filepath = f"{output_dir}/{filename}"
                if os.path.exists(filepath):
                    with cols[i % 3]:
                        st.write(f"✅ {filename}")
                        st.caption(description)
                else:
                    with cols[i % 3]:
                        st.write(f"⏳ {filename}")

            st.info(f"📂 All files saved to: `{output_dir}/`")

        else:
            st.warning("No valid strategies found. Try adjusting parameters or data.")

    # Feature importance section
    if 'discovery_engine' in st.session_state:
        engine = st.session_state['discovery_engine']
        output_dir = st.session_state.get('discovery_output_dir', 'analysis2')

        importance_file = f"{output_dir}/feature_importance.csv"
        if os.path.exists(importance_file):
            st.markdown("---")
            st.subheader("🤖 ML Feature Importance")

            importance_df = pd.read_csv(importance_file)

            # Top 20 features chart
            top_features = importance_df.head(20)

            import plotly.express as px
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title='Top 20 Most Predictive Features',
                labels={'importance': 'Importance Score', 'feature': 'Feature'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Insights
            st.markdown("**Key Insights:**")
            top_3 = importance_df.head(3)['feature'].tolist()
            st.write(f"- Most important features: **{', '.join(top_3)}**")

            slope_features = [f for f in importance_df.head(20)['feature'] if 'slope' in f.lower()]
            if slope_features:
                st.write(f"- Key slope features in top 20: **{', '.join(slope_features[:3])}**")

            momentum_features = [f for f in importance_df.head(20)['feature'] if any(x in f.lower() for x in ['rsi', 'macd', 'mom', 'cci'])]
            if momentum_features:
                st.write(f"- Key momentum features: **{', '.join(momentum_features[:3])}**")

    # ============================================================================
    # PRICE PREDICTION SUITE
    # ============================================================================

    if PRICE_PREDICTION_AVAILABLE:
        st.markdown("---")
        st.header("Price Prediction Suite")

        # --- Strategy Selector ---
        def list_saved_strategies_for_predictions():
            """List saved strategies from strategies/ and velocity_strategies/ directories."""
            strategies = []

            # Check both strategy directories
            strategy_dirs = ["strategies", "velocity_strategies"]

            for strategies_dir in strategy_dirs:
                if not os.path.exists(strategies_dir):
                    continue

                for item in os.listdir(strategies_dir):
                    strategy_path = os.path.join(strategies_dir, item)

                    if not os.path.isdir(strategy_path):
                        continue

                    # Check for strategy_config.json or velocity_config.json
                    config_file = None
                    for config_name in ["strategy_config.json", "velocity_config.json"]:
                        potential_config = os.path.join(strategy_path, config_name)
                        if os.path.exists(potential_config):
                            config_file = potential_config
                            break

                    if config_file:
                        try:
                            with open(config_file, 'r') as f:
                                config = json.load(f)

                            # Get strategy type indicator
                            strategy_type = "ML" if strategies_dir == "strategies" else "Velocity"

                            strategies.append({
                                'name': item,
                                'display_name': f"[{strategy_type}] {config.get('ticker', 'Unknown')} - {config.get('strategy_name', item)}",
                                'path': strategy_path,
                                'config_path': config_file,
                                'config': config,
                                'ticker': config.get('ticker', 'Unknown'),
                                'interval': config.get('interval', '1d'),
                                'strategy_name': config.get('strategy_name', item),
                                'created': config.get('created_at', config.get('deployed_at', config.get('exported_at', 'Unknown'))),
                                'polygon_api_key': config.get('polygon_api_key', ''),
                                'strategy_type': strategy_type
                            })
                        except Exception as e:
                            pass

            # Sort by creation date (newest first)
            strategies.sort(key=lambda x: x['created'], reverse=True)
            return strategies

        # Get saved strategies
        saved_strategies = list_saved_strategies_for_predictions()

        # Strategy selection UI
        st.markdown("### Data Source")
        strategy_options = ["Use Current Session Data"]
        strategy_map = {}

        for strat in saved_strategies:
            display = f"{strat['display_name']} ({strat['interval']})"
            strategy_options.append(display)
            strategy_map[display] = strat

        selected_strategy_option = st.selectbox(
            "Select Data Source",
            strategy_options,
            key="pred_strategy_select",
            help="Use current session data or load data from a saved strategy"
        )

        # Determine data source
        strategy_polygon_key = ''  # Initialize for both cases

        if selected_strategy_option == "Use Current Session Data":
            if 'df' not in st.session_state:
                st.warning("No data in current session. Please load data in Step 1 above, or select a saved strategy.")
                st.stop()

            pred_ticker = st.session_state.get('ticker', 'SPY')
            pred_interval = st.session_state.get('data_interval', '1d')

            # Refresh button for current session data
            refresh_col1, refresh_col2 = st.columns([4, 1])
            with refresh_col1:
                st.markdown(f"**Data Source:** {pred_ticker} ({pred_interval})")
            with refresh_col2:
                refresh_clicked = st.button("🔄 Refresh", key="refresh_session_data", help="Re-fetch latest price data")

            if refresh_clicked:
                # Re-fetch data for the ticker
                import yfinance as yf
                with st.spinner(f"Fetching fresh data for {pred_ticker}..."):
                    # Check market status to determine if today's bar is complete
                    # yfinance end is EXCLUSIVE, so we add 1 day to include that date
                    market_closed = False
                    if MARKET_UTILS_AVAILABLE:
                        market_status = is_market_open(pred_ticker)
                        market_closed = not market_status.get('is_open', True)

                    if market_closed:
                        # Market closed - today's bar is complete, include it
                        end_date = datetime.now() + timedelta(days=1)
                        data_msg = "today's close (market closed)"
                    else:
                        # Market open - today's bar is incomplete, exclude it
                        end_date = datetime.now()
                        data_msg = "yesterday's close (market open)"

                    # Use the same years setting from session state (default 5 years for range prediction)
                    years_setting = st.session_state.get('years', 5)
                    days_to_fetch = years_setting * 365
                    start_date = datetime.now() - timedelta(days=days_to_fetch)
                    fresh_df = yf.download(pred_ticker, start=start_date, end=end_date, interval=pred_interval, progress=False)

                    if not fresh_df.empty:
                        fresh_df.columns = fresh_df.columns.get_level_values(0) if isinstance(fresh_df.columns, pd.MultiIndex) else fresh_df.columns
                        fresh_df.columns = fresh_df.columns.str.lower()
                        st.session_state['df'] = fresh_df
                        st.success(f"Refreshed with {data_msg}! {len(fresh_df)} bars, last close: ${fresh_df['close'].iloc[-1]:,.2f}")
                        st.rerun()
                    else:
                        st.error("Failed to fetch fresh data")

            pred_df = st.session_state['df'].copy()

            # Show data info with timestamp and market status
            last_bar_time = pred_df.index[-1]
            last_close = pred_df['close'].iloc[-1]

            # Check current market status
            market_status_str = ""
            if MARKET_UTILS_AVAILABLE:
                market_status = is_market_open(pred_ticker)
                if market_status.get('is_open', False):
                    market_status_str = "OPEN"
                else:
                    market_status_str = "CLOSED"

            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            with info_col1:
                st.metric("Bars", f"{len(pred_df)}")
            with info_col2:
                st.metric("Last Bar", f"{last_bar_time.strftime('%Y-%m-%d') if hasattr(last_bar_time, 'strftime') else last_bar_time}")
            with info_col3:
                st.metric("Last Close", f"${last_close:,.2f}")
            with info_col4:
                if market_status_str:
                    st.metric("Market", market_status_str)
        else:
            # Load data for selected strategy
            selected_strat = strategy_map[selected_strategy_option]
            pred_ticker = selected_strat['ticker']
            pred_interval = selected_strat['interval']
            # Get Polygon API key from strategy config
            strategy_polygon_key = selected_strat.get('polygon_api_key', '')

            st.info(f"Loading data for **{pred_ticker}** ({pred_interval}) from saved strategy...")

            # Fetch fresh data using yfinance
            @st.cache_data(ttl=60)  # Cache for 1 minute (reduced from 5)
            def fetch_strategy_data(ticker: str, interval: str, days: int = 1825, include_today: bool = False):
                """Fetch fresh data for a strategy.

                Args:
                    days: Number of days of data to fetch (default 1825 = 5 years for daily)
                    include_today: If True, extends end_date to include today's data.
                                   Should be True when market is closed.
                """
                import yfinance as yf
                # yfinance end is EXCLUSIVE, add 1 day to include today when market is closed
                if include_today:
                    end_date = datetime.now() + timedelta(days=1)
                else:
                    end_date = datetime.now()
                fetch_time = datetime.now()  # Track when data was fetched

                # Adjust days based on interval limitations
                if interval in ["15m", "30m"]:
                    max_days = min(days, 59)
                elif interval in ["1h", "4h", "12h"]:
                    max_days = min(days, 729)
                else:
                    max_days = days

                start_date = datetime.now() - timedelta(days=max_days)

                # Map interval to yfinance format
                yf_interval = interval
                if interval in ["4h", "12h"]:
                    yf_interval = "1h"

                df = yf.download(ticker, start=start_date, end=end_date, interval=yf_interval, progress=False)

                if df.empty:
                    return pd.DataFrame(), None

                df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
                df.columns = df.columns.str.lower()

                # Resample if needed
                if interval == "4h" and not df.empty:
                    df = df.resample('4h').agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                    }).dropna()
                elif interval == "12h" and not df.empty:
                    df = df.resample('12h').agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                    }).dropna()

                return df, fetch_time

            # Check market status to determine if today's bar is complete
            include_today = False
            market_status_str = ""
            if MARKET_UTILS_AVAILABLE:
                market_status = is_market_open(pred_ticker)
                include_today = not market_status.get('is_open', True)  # Include today if market closed
                market_status_str = "CLOSED" if include_today else "OPEN"

            with st.spinner(f"Fetching {pred_ticker} data..."):
                result = fetch_strategy_data(pred_ticker, pred_interval, include_today=include_today)
                if isinstance(result, tuple):
                    pred_df, fetch_time = result
                else:
                    pred_df = result
                    fetch_time = None

            if pred_df.empty:
                st.error(f"Failed to load data for {pred_ticker}")
                st.stop()

            # Calculate composite oscillator for exit timing
            pred_df = create_composite_oscillator(pred_df)

            # Show data info with timestamp and market status
            last_bar_time = pred_df.index[-1]
            last_close = pred_df['close'].iloc[-1]

            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            with info_col1:
                st.success(f"**{len(pred_df)}** bars loaded")
            with info_col2:
                st.info(f"Last bar: {last_bar_time.strftime('%Y-%m-%d %H:%M') if hasattr(last_bar_time, 'strftime') else last_bar_time}")
            with info_col3:
                st.info(f"Last close: **${last_close:,.2f}**")
            with info_col4:
                if market_status_str:
                    st.info(f"Market: **{market_status_str}**")

        # Initialize Polygon manager if API key available
        # Priority: 1) Strategy config, 2) Environment variable, 3) Hardcoded default
        polygon_api_key = ''
        if selected_strategy_option != "Use Current Session Data":
            polygon_api_key = strategy_polygon_key
        if not polygon_api_key:
            polygon_api_key = os.environ.get('POLYGON_API_KEY', '')
        if not polygon_api_key:
            # Default key from user's strategies
            polygon_api_key = ''

        # Reuse PolygonManager instance across Streamlit reruns (preserves in-memory cache)
        polygon = None
        if POLYGON_AVAILABLE and polygon_api_key:
            cache_key = f"polygon_manager_{polygon_api_key[-8:]}"
            if cache_key not in st.session_state:
                st.session_state[cache_key] = PolygonManager(polygon_api_key)
                st.success(f"Polygon API connected (key: ...{polygon_api_key[-8:]})")
            polygon = st.session_state[cache_key]

        st.markdown("---")

        pred_tab1, pred_tab2, pred_tab3, pred_tab4, pred_tab5 = st.tabs([
            "Daily Range",
            "Price Targets",
            "Exit Timing",
            "Meta Prediction",
            "Walk Forward Analysis"
        ])

        # ==================== TAB 1: DAILY RANGE PREDICTION ====================
        with pred_tab1:
            st.subheader("Daily Range Prediction")
            st.markdown("Predict tomorrow's expected trading range (high/low) using ML and options data.")

            # Current Stats Row
            current_close = pred_df['close'].iloc[-1]
            today_high = pred_df['high'].iloc[-1]
            today_low = pred_df['low'].iloc[-1]
            today_range = today_high - today_low
            today_range_pct = (today_range / current_close) * 100

            # Calculate ATR
            tr = pd.concat([
                pred_df['high'] - pred_df['low'],
                abs(pred_df['high'] - pred_df['close'].shift(1)),
                abs(pred_df['low'] - pred_df['close'].shift(1))
            ], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean().iloc[-1]
            atr_14_pct = (atr_14 / current_close) * 100

            # Calculate historical volatility as fallback for crypto
            returns = pred_df['close'].pct_change().dropna()
            hist_vol_20d = returns.rolling(20).std().iloc[-1] * np.sqrt(365) * 100  # Annualized
            hist_vol_5d = returns.rolling(5).std().iloc[-1] * np.sqrt(365) * 100
            vol_trend = "Expanding" if hist_vol_5d > hist_vol_20d else "Contracting"

            # Check if this is crypto (no options data available)
            is_crypto = "USD" in pred_ticker or "BTC" in pred_ticker or "ETH" in pred_ticker

            # Display current stats
            stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
            with stat_col1:
                st.metric("Current Price", f"${current_close:,.2f}")
            with stat_col2:
                st.metric("Today's Range", f"${today_range:,.2f}", f"{today_range_pct:.2f}%")
            with stat_col3:
                st.metric("ATR(14)", f"${atr_14:,.2f}", f"{atr_14_pct:.2f}%")
            with stat_col4:
                if is_crypto:
                    # For crypto, show historical volatility trend
                    st.metric("Vol Trend", vol_trend, f"5d vs 20d")
                else:
                    # Options PCR if available
                    if polygon:
                        try:
                            sentiment = polygon.get_options_sentiment(pred_ticker)
                            if sentiment.get('status') == 'ok':
                                pcr = sentiment.get('pcr_volume', 1.0)
                                st.metric("Put/Call Ratio", f"{pcr:.2f}", sentiment.get('sentiment', 'N/A'))
                            else:
                                st.metric("Put/Call Ratio", "N/A", "No data")
                        except:
                            st.metric("Put/Call Ratio", "N/A", "API Error")
                    else:
                        st.metric("Put/Call Ratio", "N/A", "No API Key")
            with stat_col5:
                if is_crypto:
                    # For crypto, show historical volatility
                    st.metric("Hist Vol (20d)", f"{hist_vol_20d:.1f}%", vol_trend)
                else:
                    # Implied Volatility if available
                    if polygon:
                        try:
                            iv_data = polygon.calculate_aggregate_iv(pred_ticker, current_close)
                            if iv_data.get('available'):
                                iv = iv_data.get('iv_weighted', 0)
                                iv_skew = iv_data.get('iv_skew', 0)
                                skew_label = "Puts higher" if iv_skew > 0 else "Calls higher"
                                st.metric("Implied Vol", f"{iv:.1f}%", skew_label if abs(iv_skew) > 1 else "Neutral skew")
                            else:
                                st.metric("Implied Vol", "N/A", "No data")
                        except:
                            st.metric("Implied Vol", "N/A", "API Error")
                    else:
                        st.metric("Implied Vol", "N/A", "No API Key")

            st.markdown("---")

            # Range Predictor Training Section
            range_predictor_key = f'range_predictor_{pred_ticker}'

            col_train, col_predict = st.columns([1, 2])

            with col_train:
                st.markdown("**Model Training**")

                # Trials input with presets
                trial_preset = st.selectbox(
                    "Trial Presets",
                    ["Quick (100)", "Standard (500)", "Thorough (1,000)", "Deep (5,000)", "Extreme (10,000)", "Maximum (25,000)", "Ultra (50,000)", "Custom"],
                    index=1,
                    key="range_trial_preset"
                )

                preset_values = {
                    "Quick (100)": 100,
                    "Standard (500)": 500,
                    "Thorough (1,000)": 1000,
                    "Deep (5,000)": 5000,
                    "Extreme (10,000)": 10000,
                    "Maximum (25,000)": 25000,
                    "Ultra (50,000)": 50000
                }

                if trial_preset == "Custom":
                    n_trials_range = st.number_input(
                        "Custom Trials",
                        min_value=10,
                        max_value=50000,
                        value=500,
                        step=100,
                        key="range_trials_custom"
                    )
                else:
                    n_trials_range = preset_values[trial_preset]
                    st.caption(f"Trials: {n_trials_range:,}")

                # Time estimate
                est_time = estimate_optimization_time(n_trials_range, N_JOBS_OPTUNA)
                st.caption(f"Est. time: {est_time} ({N_JOBS_OPTUNA} workers)")

                if st.button("Train Range Model", key="train_range_btn", type="primary"):
                    # Use container pattern like working Optuna code (lines 3526-3562)
                    progress_container = st.container()
                    progress_container.info(f"Training model ({n_trials_range:,} trials, {N_JOBS_OPTUNA} workers)... Check terminal for progress.")

                    try:
                        start_time = time.time()
                        print(f"[RANGE MODEL] Starting training at {datetime.now()}")

                        # Initialize predictor
                        range_predictor = PriceRangePredictor(polygon)

                        # Get options features (or use historical vol for crypto)
                        if is_crypto:
                            options_features = {
                                'pcr_volume': 1.0,
                                'pcr_oi': 1.0,
                                'sentiment': 'NEUTRAL',
                                'iv_weighted': hist_vol_20d,
                                'iv_call': hist_vol_20d,
                                'iv_put': hist_vol_20d,
                                'iv_skew': 0,
                                'max_pain': None,
                                'high_call_strike': None,
                                'high_put_strike': None
                            }
                            print(f"[RANGE MODEL] Using crypto fallback: hist_vol_20d={hist_vol_20d:.2f}%")
                        else:
                            options_features = range_predictor.get_options_features(pred_ticker, current_close) if polygon else None
                            print(f"[RANGE MODEL] Options features loaded")

                        # Train model with parallel workers
                        # IMPORTANT: Don't pass options_features to training!
                        # Options data is point-in-time, not historical.
                        # Use volatility_20d as IV proxy during training.
                        result = range_predictor.train_range_model(
                            pred_df,
                            options_features=None,  # Don't use options for training!
                            n_trials=n_trials_range,
                            n_workers=N_JOBS_OPTUNA
                        )
                        print(f"[RANGE MODEL] Training complete, result keys: {result.keys() if result else 'None'}")

                        # Make prediction immediately after training
                        prediction = range_predictor.predict_daily_range(
                            pred_df,
                            options_features=options_features,
                            confidence_level=0.9
                        )
                        print(f"[RANGE MODEL] Prediction made: high=${prediction.get('predicted_high', 0):.2f}, low=${prediction.get('predicted_low', 0):.2f}")

                        duration = time.time() - start_time
                        save_optimization_timing(n_trials_range, duration, N_JOBS_OPTUNA)

                        # Store RESULTS in session state (CRITICAL for display)
                        st.session_state['range_prediction'] = prediction
                        st.session_state['range_model_metrics'] = result.get('metrics', {})
                        st.session_state['range_model_trained'] = True
                        st.session_state['range_training_duration'] = duration
                        st.session_state['range_predictor'] = range_predictor  # Store predictor for later use
                        print(f"[RANGE MODEL] Stored in session_state, triggering rerun...")

                        # Clear progress message and show success
                        progress_container.empty()
                        st.success(f"Training complete in {duration:.1f}s! Model R²: {result.get('metrics', {}).get('r2', 0):.4f}")

                        # Rerun to update col_predict display
                        st.rerun()

                    except Exception as e:
                        progress_container.empty()
                        st.error(f"Training failed: {str(e)}")
                        print(f"[RANGE MODEL] ERROR: {e}")
                        import traceback
                        traceback.print_exc()

                # Display model metrics if trained (outside button block - shows after rerun)
                if st.session_state.get('range_model_trained'):
                    metrics = st.session_state.get('range_model_metrics', {})
                    duration = st.session_state.get('range_training_duration', 0)
                    st.success(f"Model trained (R²: {metrics.get('r2', 0):.4f}, {duration:.1f}s)")

            with col_predict:
                st.markdown("**Range Prediction**")

                # Check if prediction exists in session state
                prediction = st.session_state.get('range_prediction')
                print(f"[RANGE MODEL] col_predict checking session_state: prediction={'exists' if prediction else 'None'}")

                if prediction is not None and prediction.get('predicted_high', 0) > 0:
                    # Display prediction card with beautiful styling
                    st.markdown("### Tomorrow's Predicted Range")

                    # Main prediction display
                    pred_high = prediction.get('predicted_high', 0)
                    high_unc = prediction.get('high_uncertainty', 0)
                    pred_low = prediction.get('predicted_low', 0)
                    low_unc = prediction.get('low_uncertainty', 0)
                    pred_range = prediction.get('predicted_range_dollars', 0)
                    pred_range_pct = prediction.get('predicted_range', 0) * 100

                    # Beautiful prediction cards
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                                padding: 20px; border-radius: 15px; margin-bottom: 15px;
                                border: 1px solid #0f3460;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="text-align: center; flex: 1;">
                                <p style="color: #888; margin: 0; font-size: 12px;">PREDICTED HIGH</p>
                                <p style="color: #00ff88; margin: 5px 0; font-size: 28px; font-weight: bold;">${pred_high:,.2f}</p>
                                <p style="color: #666; margin: 0; font-size: 11px;">+/- ${high_unc:,.2f}</p>
                            </div>
                            <div style="text-align: center; flex: 1; border-left: 1px solid #333; border-right: 1px solid #333; padding: 0 20px;">
                                <p style="color: #888; margin: 0; font-size: 12px;">EXPECTED RANGE</p>
                                <p style="color: #00d4ff; margin: 5px 0; font-size: 28px; font-weight: bold;">${pred_range:,.2f}</p>
                                <p style="color: #666; margin: 0; font-size: 11px;">{pred_range_pct:.2f}% of price</p>
                            </div>
                            <div style="text-align: center; flex: 1;">
                                <p style="color: #888; margin: 0; font-size: 12px;">PREDICTED LOW</p>
                                <p style="color: #ff6b6b; margin: 5px 0; font-size: 28px; font-weight: bold;">${pred_low:,.2f}</p>
                                <p style="color: #666; margin: 0; font-size: 11px;">+/- ${low_unc:,.2f}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Additional metrics row
                    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                    with met_col1:
                        r2 = prediction.get('model_r2', 0)
                        r2_color = "normal" if r2 > 0 else "off"
                        st.metric("Model R²", f"{r2:.4f}", help="Model fit quality. >0 means better than baseline, <0 means worse than just using average.")
                    with met_col2:
                        # Calculate simple backtest accuracy: how often was actual range within predicted bounds?
                        historical_ranges = (pred_df['high'] - pred_df['low']).tail(20)
                        predicted_range_val = pred_range
                        range_tolerance = predicted_range_val * 0.3  # 30% tolerance
                        within_range = ((historical_ranges >= predicted_range_val - range_tolerance) &
                                       (historical_ranges <= predicted_range_val + range_tolerance)).mean() * 100
                        st.metric("Backtest Acc", f"{within_range:.0f}%", help="% of last 20 days where actual range was within 30% of today's predicted range")
                    with met_col3:
                        atr_multiple = pred_range / atr_14 if atr_14 > 0 else 0
                        st.metric("Range vs ATR", f"{atr_multiple:.2f}x", help="Predicted range as multiple of 14-day ATR")
                    with met_col4:
                        conf_level = prediction.get('confidence_level', 0.9)
                        st.metric("CI Level", f"{conf_level*100:.0f}%", help="Confidence Interval level for prediction bounds (not model accuracy)")

                    # Save prediction with custom name
                    save_dr_col1, save_dr_col2, save_dr_col3 = st.columns([2, 2, 2])
                    with save_dr_col1:
                        # Generate default name with timestamp
                        from datetime import datetime as dt_save
                        default_name = f"{pred_ticker}_daily_range_{dt_save.now().strftime('%Y%m%d_%H%M')}"
                        dr_save_name = st.text_input("Prediction Name", value=default_name, key="dr_save_name")

                    with save_dr_col2:
                        if st.button("Save for Options Builder", key="save_daily_range_btn"):
                            try:
                                # Get stored predictor from session state
                                stored_predictor = st.session_state.get('range_predictor')
                                # Use custom name for save path
                                custom_path = f"predictions/{dr_save_name}.json"
                                if stored_predictor:
                                    save_path = stored_predictor.save_predictions(prediction, pred_ticker, path=custom_path)
                                    st.success(f"Saved to {save_path}!")
                                else:
                                    # Fallback: create new predictor
                                    new_predictor = PriceRangePredictor()
                                    new_predictor.model_metrics = st.session_state.get('range_model_metrics', {})
                                    save_path = new_predictor.save_predictions(prediction, pred_ticker, path=custom_path)
                                    st.success(f"Saved to {save_path}!")
                            except Exception as save_err:
                                st.error(f"Save failed: {save_err}")

                    with save_dr_col3:
                        # Show existing predictions for this ticker
                        import glob as glob_dr
                        existing_preds = glob_dr.glob(f"predictions/{pred_ticker}*.json")
                        if existing_preds:
                            st.caption(f"{len(existing_preds)} saved predictions for {pred_ticker}")

                    # Chart with predicted range
                    st.markdown("---")

                    import plotly.graph_objects as go  # Local import for chart
                    chart_df = pred_df.tail(30).copy()  # Last 30 bars for cleaner view

                    fig_range = go.Figure()

                    # Candlestick with better colors
                    fig_range.add_trace(go.Candlestick(
                        x=chart_df.index,
                        open=chart_df['open'],
                        high=chart_df['high'],
                        low=chart_df['low'],
                        close=chart_df['close'],
                        name='Price',
                        increasing_line_color='#26a69a',
                        decreasing_line_color='#ef5350',
                        increasing_fillcolor='#26a69a',
                        decreasing_fillcolor='#ef5350'
                    ))

                    # Current price line
                    current_price = chart_df['close'].iloc[-1]
                    fig_range.add_hline(
                        y=current_price,
                        line_dash="dot",
                        line_color="#ffffff",
                        line_width=1,
                        annotation_text=f"Current: ${current_price:,.2f}",
                        annotation_position="left"
                    )

                    # Predicted High band (green zone)
                    fig_range.add_hline(
                        y=pred_high,
                        line_dash="dash",
                        line_color="#00ff88",
                        line_width=2,
                        annotation_text=f"HIGH: ${pred_high:,.2f}",
                        annotation_position="right",
                        annotation_font_size=14,
                        annotation_font_color="#00ff88"
                    )

                    # Predicted Low band (red zone)
                    fig_range.add_hline(
                        y=pred_low,
                        line_dash="dash",
                        line_color="#ff6b6b",
                        line_width=2,
                        annotation_text=f"LOW: ${pred_low:,.2f}",
                        annotation_position="right",
                        annotation_font_size=14,
                        annotation_font_color="#ff6b6b"
                    )

                    # Shaded prediction zone (subtle)
                    fig_range.add_hrect(
                        y0=pred_low,
                        y1=pred_high,
                        fillcolor="rgba(100, 100, 100, 0.1)",
                        layer="below",
                        line_width=0
                    )

                    fig_range.update_layout(
                        title=dict(
                            text=f"<b>{pred_ticker}</b> - Tomorrow's Predicted Range",
                            font=dict(size=18, color='white'),
                            x=0.5
                        ),
                        yaxis_title="Price ($)",
                        xaxis_title="",
                        height=500,
                        showlegend=False,
                        xaxis_rangeslider_visible=False,
                        plot_bgcolor='#1a1a2e',
                        paper_bgcolor='#1a1a2e',
                        font=dict(color='#888'),
                        yaxis=dict(
                            gridcolor='rgba(255,255,255,0.1)',
                            tickformat='$,.0f'
                        ),
                        xaxis=dict(
                            gridcolor='rgba(255,255,255,0.05)'
                        ),
                        margin=dict(l=60, r=120, t=60, b=40)
                    )

                    st.plotly_chart(fig_range, use_container_width=True)

                else:
                    st.info("Click 'Train Range Model' to generate predictions.")
                    st.caption("The model will predict tomorrow's expected high/low range based on historical patterns, volatility, and options data (if available).")

        # ==================== TAB 2: PRICE TARGETS ====================
        with pred_tab2:
            st.subheader("Price Targets Calculator")
            st.markdown("Calculate optimal take-profit levels using ATR, Fibonacci, and Support/Resistance.")

            # Initialize target calculator
            target_calculator = PriceTargetCalculator(polygon)

            # Position input
            target_col1, target_col2, target_col3 = st.columns(3)

            with target_col1:
                position_direction = st.selectbox(
                    "Position Direction",
                    ["long", "short"],
                    key="target_direction"
                )

            with target_col2:
                entry_price = st.number_input(
                    "Entry Price ($)",
                    min_value=0.01,
                    value=float(current_close),
                    step=0.01,
                    key="target_entry"
                )

            with target_col3:
                stop_loss = st.number_input(
                    "Stop Loss ($)",
                    min_value=0.01,
                    value=float(current_close * (0.98 if position_direction == 'long' else 1.02)),
                    step=0.01,
                    key="target_stop"
                )

            st.markdown("---")

            if st.button("Calculate Targets", key="calc_targets_btn"):
                with st.spinner("Calculating price targets..."):
                    try:
                        # Get optimal targets
                        targets = target_calculator.get_optimal_targets(
                            pred_df,
                            entry_price,
                            position_direction,
                            pred_ticker
                        )

                        # Display target levels
                        st.markdown("### Target Levels")

                        recommendations = targets.get('recommendations', {})

                        tgt_col1, tgt_col2, tgt_col3 = st.columns(3)

                        # Conservative target
                        with tgt_col1:
                            if 'conservative' in recommendations and recommendations['conservative']:
                                cons = recommendations['conservative']
                                cons_price = cons.get('price', 0)
                                cons_pct = cons.get('pct_gain', 0)
                                risk = abs(entry_price - stop_loss)
                                reward = abs(cons_price - entry_price)
                                rr_ratio = reward / risk if risk > 0 else 0

                                st.markdown("**Conservative (1x ATR)**")
                                st.metric("Target 1", f"${cons_price:.2f}", f"+{cons_pct:.2f}%")
                                st.caption(f"R:R = 1:{rr_ratio:.1f}")

                        # Moderate target
                        with tgt_col2:
                            if 'moderate' in recommendations and recommendations['moderate']:
                                mod = recommendations['moderate']
                                mod_price = mod.get('price', 0)
                                mod_pct = mod.get('pct_gain', 0)
                                risk = abs(entry_price - stop_loss)
                                reward = abs(mod_price - entry_price)
                                rr_ratio = reward / risk if risk > 0 else 0

                                st.markdown("**Moderate (1.5x ATR)**")
                                st.metric("Target 2", f"${mod_price:.2f}", f"+{mod_pct:.2f}%")
                                st.caption(f"R:R = 1:{rr_ratio:.1f}")

                        # Aggressive target
                        with tgt_col3:
                            if 'aggressive' in recommendations and recommendations['aggressive']:
                                agg = recommendations['aggressive']
                                agg_price = agg.get('price', 0)
                                agg_pct = agg.get('pct_gain', 0)
                                risk = abs(entry_price - stop_loss)
                                reward = abs(agg_price - entry_price)
                                rr_ratio = reward / risk if risk > 0 else 0

                                st.markdown("**Aggressive (2x ATR)**")
                                st.metric("Target 3", f"${agg_price:.2f}", f"+{agg_pct:.2f}%")
                                st.caption(f"R:R = 1:{rr_ratio:.1f}")

                        # Support/Resistance levels
                        st.markdown("---")
                        st.markdown("### Support & Resistance Levels")

                        sr_levels = targets.get('support_resistance', {})

                        sr_col1, sr_col2 = st.columns(2)

                        with sr_col1:
                            st.markdown("**Resistance Levels**")
                            for i in range(1, 4):
                                r_key = f'resistance_{i}'
                                if sr_levels.get(r_key):
                                    r_price = sr_levels[r_key]
                                    pct_away = ((r_price - current_close) / current_close) * 100
                                    st.write(f"R{i}: ${r_price:.2f} ({pct_away:+.2f}%)")

                        with sr_col2:
                            st.markdown("**Support Levels**")
                            for i in range(1, 4):
                                s_key = f'support_{i}'
                                if sr_levels.get(s_key):
                                    s_price = sr_levels[s_key]
                                    pct_away = ((s_price - current_close) / current_close) * 100
                                    st.write(f"S{i}: ${s_price:.2f} ({pct_away:+.2f}%)")

                        # Options-based levels (if available)
                        options_data = targets.get('options_data', {})
                        if options_data.get('available'):
                            st.markdown("---")
                            st.markdown("### Options-Based Levels")

                            opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)

                            with opt_col1:
                                pcr = options_data.get('pcr_volume', 1.0)
                                sentiment = options_data.get('sentiment', 'NEUTRAL')
                                st.metric("Put/Call Ratio", f"{pcr:.2f}", sentiment)

                            with opt_col2:
                                max_pain = options_data.get('max_pain')
                                if max_pain:
                                    pct_from_price = ((max_pain - current_close) / current_close) * 100
                                    st.metric("Max Pain", f"${max_pain:.2f}", f"{pct_from_price:+.2f}%")
                                else:
                                    st.metric("Max Pain", "N/A")

                            with opt_col3:
                                high_call = options_data.get('high_call_oi_strike')
                                if high_call:
                                    pct_from_price = ((high_call - current_close) / current_close) * 100
                                    st.metric("High Call OI", f"${high_call:.2f}", f"{pct_from_price:+.2f}%")
                                else:
                                    st.metric("High Call OI", "N/A")

                            with opt_col4:
                                high_put = options_data.get('high_put_oi_strike')
                                if high_put:
                                    pct_from_price = ((high_put - current_close) / current_close) * 100
                                    st.metric("High Put OI", f"${high_put:.2f}", f"{pct_from_price:+.2f}%")
                                else:
                                    st.metric("High Put OI", "N/A")

                        # Chart with targets
                        st.markdown("---")
                        st.markdown("### Price Chart with Targets")

                        chart_df = pred_df.tail(60).copy()

                        fig_targets = go.Figure()

                        # Candlestick
                        fig_targets.add_trace(go.Candlestick(
                            x=chart_df.index,
                            open=chart_df['open'],
                            high=chart_df['high'],
                            low=chart_df['low'],
                            close=chart_df['close'],
                            name='Price'
                        ))

                        # Entry line
                        fig_targets.add_hline(
                            y=entry_price,
                            line_dash="solid",
                            line_color="blue",
                            annotation_text=f"Entry: ${entry_price:.2f}"
                        )

                        # Stop loss line
                        fig_targets.add_hline(
                            y=stop_loss,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Stop: ${stop_loss:.2f}"
                        )

                        # Target lines
                        colors = ["lightgreen", "green", "darkgreen"]
                        for i, (level, data) in enumerate(recommendations.items()):
                            if data:
                                fig_targets.add_hline(
                                    y=data['price'],
                                    line_dash="dash",
                                    line_color=colors[i] if i < len(colors) else "green",
                                    annotation_text=f"T{i+1}: ${data['price']:.2f}"
                                )

                        fig_targets.update_layout(
                            title=f"{pred_ticker} - Price Targets ({position_direction.upper()})",
                            yaxis_title="Price",
                            xaxis_title="Date",
                            height=400,
                            showlegend=False,
                            xaxis_rangeslider_visible=False
                        )

                        st.plotly_chart(fig_targets, use_container_width=True)

                    except Exception as e:
                        st.error(f"Target calculation error: {e}")

        # ==================== TAB 3: EXIT TIMING ====================
        with pred_tab3:
            st.subheader("Exit Timing Prediction")
            st.markdown("Predict optimal exit timing based on oscillator momentum analysis.")

            # Initialize exit predictor
            exit_predictor = ExitTimingPredictor()

            # Position input
            exit_col1, exit_col2 = st.columns(2)

            with exit_col1:
                exit_position = st.selectbox(
                    "Current Position",
                    ["long", "short"],
                    key="exit_position"
                )

            with exit_col2:
                osc_column = st.selectbox(
                    "Oscillator to Use",
                    ['composite_smooth', 'composite_oscillator'],
                    key="exit_osc_col"
                )

            st.markdown("---")

            # Check if oscillator column exists
            if osc_column in pred_df.columns:
                try:
                    # Get exit timing prediction
                    exit_timing = exit_predictor.predict_exit_timing(
                        pred_df,
                        position=exit_position,
                        osc_column=osc_column
                    )

                    # Display exit timing card
                    st.markdown("### Exit Timing Analysis")

                    urgency = exit_timing.get('urgency', {})
                    timing = exit_timing.get('timing', {})
                    indicators = exit_timing.get('indicators', {})

                    # Urgency display
                    urgency_col1, urgency_col2, urgency_col3 = st.columns(3)

                    with urgency_col1:
                        urgency_score = urgency.get('urgency_score', 0)
                        urgency_level = urgency.get('urgency_level', 'LOW')

                        # Color based on urgency
                        if urgency_level == 'HIGH':
                            st.error(f"Exit Urgency: {urgency_level}")
                        elif urgency_level == 'MEDIUM':
                            st.warning(f"Exit Urgency: {urgency_level}")
                        else:
                            st.success(f"Exit Urgency: {urgency_level}")

                        st.progress(urgency_score / 100, text=f"Score: {urgency_score}/100")

                    with urgency_col2:
                        est_bars = timing.get('estimated_bars', 0)
                        bars_range = timing.get('bars_range', (0, 0))
                        confidence = timing.get('confidence', 0)

                        st.metric(
                            "Predicted Exit Window",
                            f"{est_bars} bars",
                            f"Range: {bars_range[0]}-{bars_range[1]} bars"
                        )
                        st.caption(f"Confidence: {confidence}%")

                    with urgency_col3:
                        recommendation = exit_timing.get('recommendation', 'HOLD')

                        if recommendation == 'EXIT NOW':
                            st.error(f"Action: {recommendation}")
                        elif recommendation == 'PREPARE EXIT':
                            st.warning(f"Action: {recommendation}")
                        else:
                            st.success(f"Action: {recommendation}")

                        st.caption(timing.get('note', ''))

                    # Reasons for urgency
                    reasons = urgency.get('reasons', [])
                    if reasons:
                        st.markdown("---")
                        st.markdown("**Exit Urgency Factors:**")
                        for reason in reasons:
                            st.write(f"- {reason}")

                    # Momentum Indicators
                    st.markdown("---")
                    st.markdown("### Momentum Indicators")

                    ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)

                    with ind_col1:
                        osc_val = indicators.get('oscillator', 0)
                        st.metric("Oscillator", f"{osc_val:.3f}")

                    with ind_col2:
                        velocity = indicators.get('velocity', 0)
                        vel_direction = "bullish" if velocity > 0 else "bearish"
                        st.metric("Velocity", f"{velocity:.4f}", vel_direction)

                    with ind_col3:
                        accel = indicators.get('acceleration', 0)
                        accel_direction = "increasing" if accel > 0 else "decreasing"
                        st.metric("Acceleration", f"{accel:.4f}", accel_direction)

                    with ind_col4:
                        bars_in_zone = indicators.get('bars_in_zone', 0)
                        st.metric("Bars in Zone", f"{int(bars_in_zone)}")

                    # Oscillator Chart with Momentum
                    st.markdown("---")
                    st.markdown("### Oscillator with Momentum Indicators")

                    chart_df = pred_df.tail(100).copy()

                    # Calculate momentum features for chart
                    momentum_df = exit_predictor.calculate_momentum_features(chart_df, osc_column)

                    fig_exit = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=['Price', 'Oscillator', 'Velocity']
                    )

                    # Price chart
                    fig_exit.add_trace(go.Candlestick(
                        x=chart_df.index,
                        open=chart_df['open'],
                        high=chart_df['high'],
                        low=chart_df['low'],
                        close=chart_df['close'],
                        name='Price'
                    ), row=1, col=1)

                    # Oscillator
                    fig_exit.add_trace(go.Scatter(
                        x=chart_df.index,
                        y=chart_df[osc_column],
                        name='Oscillator',
                        line=dict(color='purple')
                    ), row=2, col=1)

                    # Overbought/oversold zones
                    fig_exit.add_hline(y=0.6, line_dash="dash", line_color="red", row=2, col=1)
                    fig_exit.add_hline(y=-0.6, line_dash="dash", line_color="green", row=2, col=1)
                    fig_exit.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1)

                    # Velocity
                    if 'velocity_smooth' in momentum_df.columns:
                        vel_colors = ['green' if v > 0 else 'red' for v in momentum_df['velocity_smooth'].fillna(0)]
                        fig_exit.add_trace(go.Bar(
                            x=momentum_df.index,
                            y=momentum_df['velocity_smooth'],
                            name='Velocity',
                            marker_color=vel_colors
                        ), row=3, col=1)

                    fig_exit.update_layout(
                        title=f"{pred_ticker} - Exit Timing Analysis ({exit_position.upper()})",
                        height=600,
                        showlegend=False,
                        xaxis_rangeslider_visible=False
                    )

                    st.plotly_chart(fig_exit, use_container_width=True)

                except Exception as e:
                    st.error(f"Exit timing error: {e}")
            else:
                st.warning(f"Oscillator column '{osc_column}' not found in data. Please ensure the oscillator is calculated first.")

        # ==================== TAB 4: META PREDICTION ====================
        with pred_tab4:
            import plotly.graph_objects as go  # Local import for charts

            st.subheader("Meta Prediction - Combined Analysis")
            st.markdown("Unified forecast with baseline comparisons and deep model analysis.")

            # Check what predictions are available
            range_prediction = st.session_state.get('range_prediction')
            range_metrics = st.session_state.get('range_model_metrics', {})
            has_range = range_prediction is not None

            # Get current price info
            current_close = pred_df['close'].iloc[-1]
            today_high = pred_df['high'].iloc[-1]
            today_low = pred_df['low'].iloc[-1]
            today_range = today_high - today_low

            # Calculate ATR for baseline
            tr = pd.concat([
                pred_df['high'] - pred_df['low'],
                abs(pred_df['high'] - pred_df['close'].shift(1)),
                abs(pred_df['low'] - pred_df['close'].shift(1))
            ], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean().iloc[-1]
            atr_7 = tr.rolling(7).mean().iloc[-1]

            # Historical volatility
            returns = pred_df['close'].pct_change().dropna()
            hist_vol_20d = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100

            if has_range:
                # Get range predictions
                pred_high = range_prediction.get('predicted_high', 0)
                pred_low = range_prediction.get('predicted_low', 0)
                pred_range = range_prediction.get('predicted_range_dollars', 0)
                high_unc = range_prediction.get('high_uncertainty', 0)
                low_unc = range_prediction.get('low_uncertainty', 0)
                model_r2 = range_prediction.get('model_r2', 0)

                # Get confidence bounds (new)
                high_lower = range_prediction.get('high_lower', pred_high - high_unc)
                high_upper = range_prediction.get('high_upper', pred_high + high_unc)
                low_lower = range_prediction.get('low_lower', pred_low - low_unc)
                low_upper = range_prediction.get('low_upper', pred_low + low_unc)
                conf_method = range_prediction.get('confidence_method', 'rmse')
                conf_level = range_prediction.get('confidence_level', 0.9)

                # Calculate percentage moves
                high_pct = ((pred_high - current_close) / current_close) * 100
                low_pct = ((current_close - pred_low) / current_close) * 100

                # ============================================================
                # SECTION 1: FORECAST SUMMARY
                # ============================================================
                st.markdown("### Tomorrow's Forecast")

                # Use Streamlit metrics instead of complex HTML
                forecast_col1, forecast_col2, forecast_col3 = st.columns(3)

                with forecast_col1:
                    st.metric(
                        "Predicted High",
                        f"${pred_high:,.2f}",
                        f"+{high_pct:.2f}%",
                        delta_color="normal"
                    )
                    st.caption(f"CI: ${high_lower:,.2f} - ${high_upper:,.2f}")

                with forecast_col2:
                    st.metric(
                        "Current Close",
                        f"${current_close:,.2f}",
                        f"Range: ${pred_range:,.2f}"
                    )
                    # Show confidence method
                    if conf_method == 'conformal':
                        st.caption(f"{conf_level*100:.0f}% Conformal CI")
                    else:
                        st.caption(f"{conf_level*100:.0f}% CI (RMSE)")

                with forecast_col3:
                    st.metric(
                        "Predicted Low",
                        f"${pred_low:,.2f}",
                        f"-{low_pct:.2f}%",
                        delta_color="inverse"
                    )
                    st.caption(f"CI: ${low_lower:,.2f} - ${low_upper:,.2f}")

                # ============================================================
                # SECTION 2: BASELINE COMPARISONS
                # ============================================================
                st.markdown("---")
                st.markdown("### Model vs Baseline Comparisons")
                st.caption("Compare ML model against simple baselines to understand if ML adds value")

                # Calculate baselines
                # Baseline 1: Simple ATR
                atr_baseline_high = current_close + (atr_14 * 0.55)
                atr_baseline_low = current_close - (atr_14 * 0.45)
                atr_baseline_range = atr_14

                # Baseline 2: IV-weighted ATR (if IV available)
                iv_weighted = None
                if not is_crypto and polygon:
                    try:
                        iv_data = polygon.calculate_aggregate_iv(pred_ticker, current_close)
                        if iv_data.get('available'):
                            iv_weighted = iv_data.get('iv_weighted', 0) / 100  # Convert from % to decimal
                    except:
                        pass

                if iv_weighted is None:
                    # Use historical vol as proxy
                    iv_weighted = hist_vol_20d / 100

                # IV-weighted adjustment: scale ATR by IV/HistVol ratio
                iv_ratio = iv_weighted / (hist_vol_20d / 100) if hist_vol_20d > 0 else 1.0
                iv_adjusted_range = atr_14 * iv_ratio
                iv_baseline_high = current_close + (iv_adjusted_range * 0.55)
                iv_baseline_low = current_close - (iv_adjusted_range * 0.45)

                # Baseline 3: Recent range average
                recent_ranges = (pred_df['high'] - pred_df['low']).tail(10)
                avg_recent_range = recent_ranges.mean()
                recent_baseline_high = current_close + (avg_recent_range * 0.55)
                recent_baseline_low = current_close - (avg_recent_range * 0.45)

                # Display comparison table
                baseline_data = {
                    'Model': ['ML Model', 'ATR(14) Baseline', 'IV-Weighted ATR', 'Recent Avg Range'],
                    'Pred High': [f"${pred_high:,.2f}", f"${atr_baseline_high:,.2f}", f"${iv_baseline_high:,.2f}", f"${recent_baseline_high:,.2f}"],
                    'Pred Low': [f"${pred_low:,.2f}", f"${atr_baseline_low:,.2f}", f"${iv_baseline_low:,.2f}", f"${recent_baseline_low:,.2f}"],
                    'Pred Range': [f"${pred_range:,.2f}", f"${atr_baseline_range:,.2f}", f"${iv_adjusted_range:,.2f}", f"${avg_recent_range:,.2f}"],
                    'Range/ATR': [f"{pred_range/atr_14:.2f}x", "1.00x", f"{iv_ratio:.2f}x", f"{avg_recent_range/atr_14:.2f}x"]
                }
                baseline_df = pd.DataFrame(baseline_data)
                st.dataframe(baseline_df, use_container_width=True, hide_index=True)

                # ============================================================
                # SECTION 3: DEEP MODEL ANALYSIS
                # ============================================================
                st.markdown("---")
                st.markdown("### Deep Model Analysis")

                analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
                    "Model Metrics",
                    "Feature Importance",
                    "Backtesting"
                ])

                with analysis_tab1:
                    st.markdown("#### Model Performance Metrics")

                    met_col1, met_col2, met_col3, met_col4 = st.columns(4)

                    with met_col1:
                        r2_val = range_metrics.get('r2', 0)
                        r2_status = "POOR" if r2_val < 0 else "WEAK" if r2_val < 0.3 else "MODERATE" if r2_val < 0.6 else "GOOD"
                        st.metric("R² Score", f"{r2_val:.4f}")
                        st.caption(f"Status: {r2_status}")

                    with met_col2:
                        rmse = range_metrics.get('rmse', 0)
                        st.metric("RMSE", f"{rmse:.6f}")
                        st.caption(f"~{rmse*current_close:.2f}$ error")

                    with met_col3:
                        mae = range_metrics.get('mae', 0)
                        st.metric("MAE", f"{mae:.6f}")
                        st.caption(f"~{mae*current_close:.2f}$ avg error")

                    with met_col4:
                        train_samples = range_metrics.get('train_samples', 0)
                        test_samples = range_metrics.get('test_samples', 0)
                        st.metric("Data Split", f"{train_samples}/{test_samples}")
                        st.caption("Train/Test samples")

                    # R² Interpretation
                    st.markdown("#### R² Score Interpretation")
                    if r2_val < 0:
                        st.error(f"""
                        **R² = {r2_val:.4f} (NEGATIVE)** - Model performs WORSE than predicting the mean.

                        This means:
                        - The model's predictions have higher error than simply using the average range
                        - Features may not be predictive of future range
                        - Model may be overfitting to noise

                        **Recommendations:**
                        1. Try classification instead (High/Normal/Low volatility)
                        2. Use simpler baseline (ATR or IV-weighted ATR)
                        3. Add more predictive features (VIX changes, earnings dates)
                        4. Use longer training history
                        """)
                    elif r2_val < 0.3:
                        st.warning(f"""
                        **R² = {r2_val:.4f} (WEAK)** - Model explains {r2_val*100:.1f}% of variance.

                        The model captures some signal but predictions have high uncertainty.
                        Consider using baselines for comparison.
                        """)
                    else:
                        st.success(f"""
                        **R² = {r2_val:.4f} (ACCEPTABLE)** - Model explains {r2_val*100:.1f}% of variance.
                        """)

                    # Best params
                    if 'best_params' in range_metrics:
                        with st.expander("View Best Hyperparameters"):
                            st.json(range_metrics['best_params'])

                with analysis_tab2:
                    st.markdown("#### Feature Importance Analysis")

                    # Get feature importance from stored predictor
                    range_predictor = st.session_state.get('range_predictor')
                    if range_predictor and hasattr(range_predictor, 'range_model') and range_predictor.range_model is not None:
                        # Use feature_names to match model's feature_importances_
                        feature_names_for_importance = range_predictor.feature_names
                        importance_df = pd.DataFrame({
                            'Feature': feature_names_for_importance,
                            'Importance': range_predictor.range_model.feature_importances_
                        }).sort_values('Importance', ascending=False)

                        # Top 15 features
                        top_features = importance_df.head(15)

                        # Bar chart
                        fig_imp = go.Figure(go.Bar(
                            x=top_features['Importance'],
                            y=top_features['Feature'],
                            orientation='h',
                            marker_color='#00d4ff'
                        ))
                        fig_imp.update_layout(
                            title="Top 15 Features by Importance",
                            xaxis_title="Importance Score",
                            yaxis_title="",
                            height=450,
                            yaxis=dict(autorange="reversed"),
                            plot_bgcolor='#1a1a2e',
                            paper_bgcolor='#1a1a2e',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

                        # Analysis of top features
                        st.markdown("#### Feature Analysis")
                        top_3 = top_features.head(3)['Feature'].tolist()

                        st.write(f"**Top 3 Features:** {', '.join(top_3)}")

                        # Check if options features are being used
                        options_features = ['pcr_volume', 'pcr_oi', 'iv_weighted', 'iv_skew', 'max_pain_distance']
                        used_options = [f for f in options_features if f in top_features.head(10)['Feature'].tolist()]

                        if used_options:
                            st.success(f"Options features being used: {', '.join(used_options)}")
                        else:
                            st.warning("Options features are not in top 10. IV/PCR may not be adding value.")

                        # Full table
                        with st.expander("View All Feature Importances"):
                            st.dataframe(importance_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("Feature importance not available. Retrain the model.")

                with analysis_tab3:
                    st.markdown("#### Historical Backtest Analysis")
                    st.caption("How well would the model have predicted past ranges?")

                    # Calculate historical accuracy
                    historical_ranges = (pred_df['high'] - pred_df['low']).tail(30)
                    historical_range_pct = historical_ranges / pred_df['close'].tail(30) * 100

                    # Compare predicted range to historical distribution
                    pred_range_pct = pred_range / current_close * 100
                    percentile = (historical_range_pct < pred_range_pct).mean() * 100

                    st.write(f"**Predicted Range:** {pred_range_pct:.2f}% of price")
                    st.write(f"**Historical Percentile:** {percentile:.0f}th percentile (last 30 days)")

                    if percentile > 80:
                        st.warning("Model predicts unusually HIGH volatility compared to recent history")
                    elif percentile < 20:
                        st.warning("Model predicts unusually LOW volatility compared to recent history")
                    else:
                        st.success("Prediction is within normal historical range")

                    # Historical range distribution
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=historical_range_pct,
                        nbinsx=20,
                        name='Historical Ranges',
                        marker_color='#00d4ff',
                        opacity=0.7
                    ))
                    fig_hist.add_vline(
                        x=pred_range_pct,
                        line_dash="dash",
                        line_color="red",
                        line_width=2,
                        annotation_text=f"Predicted: {pred_range_pct:.2f}%"
                    )
                    fig_hist.update_layout(
                        title="Historical Range Distribution vs Prediction",
                        xaxis_title="Daily Range (% of close)",
                        yaxis_title="Frequency",
                        height=350,
                        plot_bgcolor='#1a1a2e',
                        paper_bgcolor='#1a1a2e',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    # ATR baseline accuracy (would ATR have been better?)
                    st.markdown("#### ATR Baseline Comparison")

                    # Calculate how well ATR predicted historical ranges
                    atr_predictions = tr.rolling(14).mean().shift(1).tail(30)  # Previous day's ATR
                    actual_ranges = (pred_df['high'] - pred_df['low']).tail(30)

                    # Remove NaN
                    valid_mask = ~(atr_predictions.isna() | actual_ranges.isna())
                    atr_pred_valid = atr_predictions[valid_mask]
                    actual_valid = actual_ranges[valid_mask]

                    if len(atr_pred_valid) > 5:
                        from sklearn.metrics import mean_squared_error, r2_score
                        atr_rmse = np.sqrt(mean_squared_error(actual_valid, atr_pred_valid))
                        atr_r2 = r2_score(actual_valid, atr_pred_valid)

                        compare_col1, compare_col2 = st.columns(2)
                        with compare_col1:
                            st.metric("ATR Baseline R²", f"{atr_r2:.4f}")
                        with compare_col2:
                            st.metric("ATR Baseline RMSE", f"${atr_rmse:.2f}")

                        model_r2_val = range_metrics.get('r2', 0)
                        if model_r2_val > atr_r2:
                            st.success(f"ML Model ({model_r2_val:.4f}) beats ATR Baseline ({atr_r2:.4f})")
                        else:
                            st.error(f"ATR Baseline ({atr_r2:.4f}) beats ML Model ({model_r2_val:.4f}). Consider using ATR instead.")

                # ============================================================
                # SECTION 4: VISUAL FORECAST
                # ============================================================
                st.markdown("---")
                st.markdown("### Visual Forecast")

                chart_df = pred_df.tail(30).copy()

                # Calculate tomorrow's date
                last_date = chart_df.index[-1]
                if hasattr(last_date, 'date'):
                    tomorrow = last_date + pd.Timedelta(days=1)
                    # Skip weekends for stocks
                    while tomorrow.weekday() >= 5:  # 5=Sat, 6=Sun
                        tomorrow += pd.Timedelta(days=1)
                else:
                    tomorrow = last_date

                fig_meta = go.Figure()

                # Candlestick for historical data
                fig_meta.add_trace(go.Candlestick(
                    x=chart_df.index,
                    open=chart_df['open'],
                    high=chart_df['high'],
                    low=chart_df['low'],
                    close=chart_df['close'],
                    name='Price',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ))

                # Get confidence bounds if available
                high_lower = prediction.get('high_lower', pred_high - 1)
                high_upper = prediction.get('high_upper', pred_high + 1)
                low_lower = prediction.get('low_lower', pred_low - 1)
                low_upper = prediction.get('low_upper', pred_low + 1)
                conf_method = prediction.get('confidence_method', 'rmse')
                conf_level = prediction.get('confidence_level', 0.9)

                # Add TOMORROW's predicted candle as a box/bar
                # ML Prediction - show as a vertical bar at tomorrow's position
                fig_meta.add_trace(go.Candlestick(
                    x=[tomorrow],
                    open=[current_close],
                    high=[pred_high],
                    low=[pred_low],
                    close=[(pred_high + pred_low) / 2],  # Midpoint
                    name='ML Prediction',
                    increasing_line_color='#00d4ff',
                    increasing_fillcolor='rgba(0, 212, 255, 0.5)',
                    decreasing_line_color='#00d4ff',
                    decreasing_fillcolor='rgba(0, 212, 255, 0.5)',
                ))

                # Confidence bounds as error bars / shaded region at tomorrow
                fig_meta.add_trace(go.Scatter(
                    x=[tomorrow, tomorrow, tomorrow, tomorrow, tomorrow],
                    y=[high_upper, high_lower, None, low_upper, low_lower],
                    mode='lines',
                    line=dict(color='rgba(0, 212, 255, 0.3)', width=8),
                    name=f'{conf_level*100:.0f}% CI ({conf_method})',
                    showlegend=True
                ))

                # ATR Baseline prediction bar at tomorrow
                fig_meta.add_trace(go.Candlestick(
                    x=[tomorrow],
                    open=[current_close],
                    high=[atr_baseline_high],
                    low=[atr_baseline_low],
                    close=[(atr_baseline_high + atr_baseline_low) / 2],
                    name='ATR Baseline',
                    increasing_line_color='#ffc107',
                    increasing_fillcolor='rgba(255, 193, 7, 0.3)',
                    decreasing_line_color='#ffc107',
                    decreasing_fillcolor='rgba(255, 193, 7, 0.3)',
                ))

                # Current close line
                fig_meta.add_hline(y=current_close, line_dash="dot", line_color="white", line_width=1,
                                   annotation_text=f"Current: ${current_close:,.2f}", annotation_position="right")

                # Add annotations for predictions
                fig_meta.add_annotation(
                    x=tomorrow, y=pred_high,
                    text=f"ML High: ${pred_high:,.2f}",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowcolor="#00ff88",
                    font=dict(color="#00ff88", size=11),
                    ax=40, ay=-20
                )
                fig_meta.add_annotation(
                    x=tomorrow, y=pred_low,
                    text=f"ML Low: ${pred_low:,.2f}",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowcolor="#ff6b6b",
                    font=dict(color="#ff6b6b", size=11),
                    ax=40, ay=20
                )

                fig_meta.update_layout(
                    title=dict(text=f"<b>{pred_ticker}</b> - Tomorrow's Forecast", font=dict(size=18, color='white'), x=0.5),
                    yaxis_title="Price ($)",
                    height=500,
                    showlegend=True,
                    legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)', font=dict(size=10)),
                    xaxis_rangeslider_visible=False,
                    plot_bgcolor='#1a1a2e',
                    paper_bgcolor='#1a1a2e',
                    font=dict(color='#888'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickformat='$,.0f'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    margin=dict(l=60, r=100, t=60, b=40)
                )

                st.plotly_chart(fig_meta, use_container_width=True)

                # Caption with confidence info
                if conf_method == 'conformal':
                    st.caption(f"Blue bar = ML prediction | Yellow bar = ATR baseline | Shaded = {conf_level*100:.0f}% Conformal Prediction Interval")
                else:
                    st.caption(f"Blue bar = ML prediction | Yellow bar = ATR baseline | Shaded = {conf_level*100:.0f}% CI (RMSE-based)")

            else:
                # No predictions available yet
                st.markdown("""
                <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                            border-radius: 20px; margin: 20px 0; border: 1px dashed #0f3460;">
                    <h2 style="color: #666; margin-bottom: 20px;">No Predictions Available Yet</h2>
                    <p style="color: #888; font-size: 16px; max-width: 500px; margin: 0 auto;">
                        Train the Range Model in the "Daily Range" tab first to see the combined meta prediction.
                    </p>
                    <div style="margin-top: 30px;">
                        <p style="color: #555; font-size: 14px;">Steps:</p>
                        <p style="color: #666; font-size: 13px;">
                            1. Go to "Daily Range" tab<br>
                            2. Select trial preset (Standard recommended)<br>
                            3. Click "Train Range Model"<br>
                            4. Return here for combined analysis
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ==================== TAB 5: WALK FORWARD ANALYSIS ====================
        with pred_tab5:
            st.subheader("Walk Forward Analysis")
            st.markdown("Robust out-of-sample testing: train on historical data, predict each day, compare to actual.")

            # User inputs for test range
            wf_col1, wf_col2, wf_col3 = st.columns([1, 1, 1])

            with wf_col1:
                wf_test_days = st.number_input(
                    "Test Period (days)",
                    min_value=10,
                    max_value=252,
                    value=60,
                    help="Number of recent days to test. Model trains on all data before each test day."
                )

            with wf_col2:
                wf_min_train_days = st.number_input(
                    "Minimum Training Days",
                    min_value=60,
                    max_value=500,
                    value=180,
                    help="Minimum historical days required to train the model."
                )

            with wf_col3:
                wf_retrain_freq = st.selectbox(
                    "Retrain Frequency",
                    options=["Every Day", "Weekly", "Monthly"],
                    index=1,
                    help="How often to retrain the model. Less frequent = faster but less adaptive."
                )

            # Optuna settings
            wf_col4, wf_col5 = st.columns(2)
            with wf_col4:
                wf_n_trials = st.number_input(
                    "Optuna Trials per Training",
                    min_value=10,
                    max_value=50000,
                    value=100,
                    help="More trials = better hyperparameters but slower. No hard limit, but diminishing returns after ~1000."
                )
            with wf_col5:
                wf_n_workers = st.number_input(
                    "Parallel Workers",
                    min_value=1,
                    max_value=32,
                    value=min(16, max(1, (os.cpu_count() or 4) - 1)),
                    help="CPU cores for parallel Optuna optimization."
                )

            st.markdown("---")

            # Run button
            if st.button("Run Walk Forward Analysis", type="primary", use_container_width=True):
                if len(pred_df) < wf_min_train_days + wf_test_days:
                    st.error(f"Insufficient data. Need at least {wf_min_train_days + wf_test_days} days, have {len(pred_df)}.")
                else:
                    # Calculate test range
                    test_start_idx = len(pred_df) - wf_test_days
                    test_dates = pred_df.index[test_start_idx:]

                    # Determine retrain schedule
                    if wf_retrain_freq == "Every Day":
                        retrain_interval = 1
                    elif wf_retrain_freq == "Weekly":
                        retrain_interval = 5
                    else:  # Monthly
                        retrain_interval = 21

                    st.info(f"Testing {wf_test_days} days from {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")

                    # Storage for results
                    wf_results = []
                    wf_feature_importances = []  # Track feature importance from each training
                    wf_feature_names = None  # Store feature names
                    current_model = None
                    current_scaler = None
                    last_train_idx = -999  # Force initial training

                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    metrics_placeholder = st.empty()

                    # Walk forward loop
                    for i, test_idx in enumerate(range(test_start_idx, len(pred_df))):
                        test_date = pred_df.index[test_idx]
                        progress = (i + 1) / wf_test_days

                        # Check if we need to retrain
                        days_since_train = test_idx - last_train_idx
                        need_retrain = (current_model is None) or (days_since_train >= retrain_interval)

                        if need_retrain:
                            status_text.text(f"Training model for {test_date.strftime('%Y-%m-%d')}... ({i+1}/{wf_test_days})")

                            # Get training data (ROLLING window - only last N days before test)
                            # This matches comprehensive_walkforward_test.py and keeps model focused on recent data
                            train_start_idx = max(0, test_idx - wf_min_train_days)
                            train_df = pred_df.iloc[train_start_idx:test_idx].copy()

                            if len(train_df) >= wf_min_train_days:
                                try:
                                    # Create fresh predictor and train
                                    wf_predictor = PriceRangePredictor(polygon if polygon_api_key else None)

                                    # NOTE: Options features are NOT used for training (point-in-time data)
                                    # VIX data is now cached at CLASS level, so no redundant fetches

                                    # Train with reduced output
                                    # IMPORTANT: Do NOT pass options_features to training!
                                    # Options data is point-in-time, not historical.
                                    # Passing it makes all IV features constant, breaking the model.
                                    with st.spinner(f"Training on {len(train_df)} days..."):
                                        train_result = wf_predictor.train_range_model(
                                            train_df,
                                            options_features=None,  # Don't use options for training!
                                            n_trials=wf_n_trials,
                                            n_workers=wf_n_workers
                                        )

                                    current_model = wf_predictor
                                    last_train_idx = test_idx

                                    # Store feature importances for analysis
                                    if train_result and 'feature_importance' in train_result:
                                        imp_df = train_result['feature_importance'].copy()
                                        imp_df['train_date'] = test_date
                                        imp_df['train_idx'] = test_idx
                                        wf_feature_importances.append(imp_df)
                                        if wf_feature_names is None:
                                            wf_feature_names = train_result.get('feature_names', [])

                                except Exception as train_err:
                                    st.warning(f"Training failed at {test_date}: {train_err}")
                                    continue
                        else:
                            status_text.text(f"Predicting {test_date.strftime('%Y-%m-%d')}... ({i+1}/{wf_test_days})")

                        # Make prediction for this day (using data up to previous day)
                        if current_model is not None:
                            try:
                                # Use data up to the day BEFORE test_date for prediction
                                pred_input_df = pred_df.iloc[:test_idx].copy()

                                # Get prediction
                                prediction = current_model.predict_daily_range(pred_input_df)

                                # Get actual values for test_date
                                actual_high = pred_df['high'].iloc[test_idx]
                                actual_low = pred_df['low'].iloc[test_idx]
                                actual_close = pred_df['close'].iloc[test_idx]
                                actual_open = pred_df['open'].iloc[test_idx]
                                actual_range = actual_high - actual_low

                                # Store result
                                wf_results.append({
                                    'date': test_date,
                                    'predicted_high': prediction['predicted_high'],
                                    'predicted_low': prediction['predicted_low'],
                                    'predicted_range': prediction['predicted_range_dollars'],
                                    'high_lower': prediction['high_lower'],
                                    'high_upper': prediction['high_upper'],
                                    'low_lower': prediction['low_lower'],
                                    'low_upper': prediction['low_upper'],
                                    'actual_high': actual_high,
                                    'actual_low': actual_low,
                                    'actual_open': actual_open,
                                    'actual_close': actual_close,
                                    'actual_range': actual_range,
                                    'model_r2': prediction.get('model_r2', 0),
                                    'retrained': need_retrain
                                })

                            except Exception as pred_err:
                                st.warning(f"Prediction failed at {test_date}: {pred_err}")

                        progress_bar.progress(progress)

                        # Update live metrics every 10 days
                        if len(wf_results) > 0 and len(wf_results) % 10 == 0:
                            temp_df = pd.DataFrame(wf_results)
                            temp_high_in_range = ((temp_df['actual_high'] >= temp_df['high_lower']) &
                                                  (temp_df['actual_high'] <= temp_df['high_upper'])).mean() * 100
                            temp_low_in_range = ((temp_df['actual_low'] >= temp_df['low_lower']) &
                                                 (temp_df['actual_low'] <= temp_df['low_upper'])).mean() * 100
                            metrics_placeholder.markdown(f"**Running Metrics:** High in range: {temp_high_in_range:.1f}% | Low in range: {temp_low_in_range:.1f}%")

                    progress_bar.progress(1.0)
                    status_text.text("Walk forward analysis complete!")

                    # Store results in session state
                    if wf_results:
                        st.session_state['wf_results'] = pd.DataFrame(wf_results)
                        st.session_state['wf_feature_importances'] = wf_feature_importances
                        st.session_state['wf_feature_names'] = wf_feature_names
                        st.session_state['wf_pred_df'] = pred_df  # Store for feature analysis
                        st.session_state['wf_current_model'] = current_model  # Store trained model for predictions
                        st.success(f"Completed {len(wf_results)} predictions!")
                    else:
                        st.error("No predictions were generated.")

            st.markdown("---")

            # Display results if available
            if 'wf_results' in st.session_state and len(st.session_state['wf_results']) > 0:
                wf_df = st.session_state['wf_results']

                # Calculate metrics
                st.subheader("Performance Metrics")

                # High/Low containment
                high_in_range = ((wf_df['actual_high'] >= wf_df['high_lower']) &
                                 (wf_df['actual_high'] <= wf_df['high_upper'])).mean() * 100
                low_in_range = ((wf_df['actual_low'] >= wf_df['low_lower']) &
                                (wf_df['actual_low'] <= wf_df['low_upper'])).mean() * 100

                # Directional accuracy (did price stay within predicted range?)
                price_contained = ((wf_df['actual_high'] <= wf_df['high_upper']) &
                                   (wf_df['actual_low'] >= wf_df['low_lower'])).mean() * 100

                # Range prediction accuracy
                range_mae = np.abs(wf_df['predicted_range'] - wf_df['actual_range']).mean()
                range_mape = (np.abs(wf_df['predicted_range'] - wf_df['actual_range']) / wf_df['actual_range']).mean() * 100

                # High/Low prediction errors
                high_error = (wf_df['predicted_high'] - wf_df['actual_high']).mean()
                low_error = (wf_df['predicted_low'] - wf_df['actual_low']).mean()
                high_mae = np.abs(wf_df['predicted_high'] - wf_df['actual_high']).mean()
                low_mae = np.abs(wf_df['predicted_low'] - wf_df['actual_low']).mean()

                # Display metrics in cards
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)

                with met_col1:
                    st.metric("High in Confidence Range", f"{high_in_range:.1f}%",
                              help="% of days where actual high fell within predicted confidence interval")
                with met_col2:
                    st.metric("Low in Confidence Range", f"{low_in_range:.1f}%",
                              help="% of days where actual low fell within predicted confidence interval")
                with met_col3:
                    st.metric("Full Containment", f"{price_contained:.1f}%",
                              help="% of days where entire candle was within predicted bounds")
                with met_col4:
                    st.metric("Range MAPE", f"{range_mape:.1f}%",
                              help="Mean Absolute Percentage Error of range prediction")

                met_col5, met_col6, met_col7, met_col8 = st.columns(4)

                with met_col5:
                    st.metric("High MAE", f"${high_mae:.2f}",
                              help="Mean Absolute Error of high prediction")
                with met_col6:
                    st.metric("Low MAE", f"${low_mae:.2f}",
                              help="Mean Absolute Error of low prediction")
                with met_col7:
                    st.metric("High Bias", f"${high_error:+.2f}",
                              help="Average prediction bias (+ = overpredict)")
                with met_col8:
                    st.metric("Low Bias", f"${low_error:+.2f}",
                              help="Average prediction bias (+ = overpredict)")

                # Save predictions button for Options Builder
                st.markdown("---")
                save_col1, save_col2, save_col3 = st.columns([2, 2, 2])
                with save_col1:
                    # Custom prediction name
                    from datetime import datetime as dt_wf
                    wf_default_name = f"{pred_ticker}_walkforward_{dt_wf.now().strftime('%Y%m%d_%H%M')}"
                    wf_save_name = st.text_input("Prediction Name", value=wf_default_name, key="wf_save_name")

                with save_col2:
                    if st.button("Save Predictions", key="save_wf_predictions_btn", type="primary"):
                        # Get the latest prediction from walk forward
                        latest_row = wf_df.iloc[-1]
                        current_model = st.session_state.get('wf_current_model')

                        # Build prediction dict matching PriceRangePredictor.predict_daily_range() output
                        save_data = {
                            'current_close': float(latest_row['actual_close']),
                            'predicted_high': float(latest_row['predicted_high']),
                            'predicted_low': float(latest_row['predicted_low']),
                            'predicted_range': float(latest_row['predicted_range']),
                            'predicted_range_dollars': float(latest_row['predicted_range']),
                            'high_upper': float(latest_row['high_upper']),
                            'high_lower': float(latest_row['high_lower']),
                            'low_upper': float(latest_row['low_upper']),
                            'low_lower': float(latest_row['low_lower']),
                            'high_uncertainty': float((latest_row['high_upper'] - latest_row['high_lower']) / 2),
                            'low_uncertainty': float((latest_row['low_upper'] - latest_row['low_lower']) / 2),
                            'confidence_level': 0.9,
                            'model_r2': current_model.model_metrics.get('r2', 0) if current_model else 0,
                        }

                        # Save using PriceRangePredictor with custom path
                        try:
                            predictor = PriceRangePredictor()
                            predictor.model_metrics = current_model.model_metrics if current_model else {}
                            predictor.feature_names = current_model.feature_names if current_model else []
                            custom_path = f"predictions/{wf_save_name}.json"
                            save_path = predictor.save_predictions(save_data, pred_ticker, path=custom_path)
                            st.success(f"Saved to {save_path}!")
                        except Exception as save_err:
                            st.error(f"Save failed: {save_err}")

                with save_col3:
                    # Show count of existing predictions
                    import glob as glob_wf
                    existing_wf_preds = glob_wf.glob(f"predictions/{pred_ticker}*.json")
                    if existing_wf_preds:
                        st.caption(f"{len(existing_wf_preds)} saved predictions for {pred_ticker}")

                st.markdown("---")

                # Candlestick chart with predicted bands
                st.subheader("Predictions vs Actuals")

                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("Price with Predicted Range Bands", "Prediction Errors")
                )

                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=wf_df['date'],
                        open=wf_df['actual_open'],
                        high=wf_df['actual_high'],
                        low=wf_df['actual_low'],
                        close=wf_df['actual_close'],
                        name='Actual Price',
                        increasing_line_color='#26a69a',
                        decreasing_line_color='#ef5350'
                    ),
                    row=1, col=1
                )

                # Predicted high band (confidence interval)
                fig.add_trace(
                    go.Scatter(
                        x=wf_df['date'],
                        y=wf_df['high_upper'],
                        mode='lines',
                        line=dict(color='rgba(255, 152, 0, 0.3)', width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=wf_df['date'],
                        y=wf_df['high_lower'],
                        mode='lines',
                        line=dict(color='rgba(255, 152, 0, 0.3)', width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 152, 0, 0.15)',
                        name='Predicted High Range'
                    ),
                    row=1, col=1
                )

                # Predicted low band (confidence interval)
                fig.add_trace(
                    go.Scatter(
                        x=wf_df['date'],
                        y=wf_df['low_upper'],
                        mode='lines',
                        line=dict(color='rgba(33, 150, 243, 0.3)', width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=wf_df['date'],
                        y=wf_df['low_lower'],
                        mode='lines',
                        line=dict(color='rgba(33, 150, 243, 0.3)', width=0),
                        fill='tonexty',
                        fillcolor='rgba(33, 150, 243, 0.15)',
                        name='Predicted Low Range'
                    ),
                    row=1, col=1
                )

                # Predicted high/low lines (point estimates)
                fig.add_trace(
                    go.Scatter(
                        x=wf_df['date'],
                        y=wf_df['predicted_high'],
                        mode='lines',
                        line=dict(color='#ff9800', width=1.5, dash='dash'),
                        name='Predicted High'
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=wf_df['date'],
                        y=wf_df['predicted_low'],
                        mode='lines',
                        line=dict(color='#2196f3', width=1.5, dash='dash'),
                        name='Predicted Low'
                    ),
                    row=1, col=1
                )

                # Mark retrain days
                retrain_days = wf_df[wf_df['retrained']]
                if len(retrain_days) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=retrain_days['date'],
                            y=retrain_days['actual_high'] * 1.002,
                            mode='markers',
                            marker=dict(symbol='triangle-down', size=8, color='purple'),
                            name='Model Retrained'
                        ),
                        row=1, col=1
                    )

                # Error subplot
                high_errors = wf_df['predicted_high'] - wf_df['actual_high']
                low_errors = wf_df['predicted_low'] - wf_df['actual_low']

                fig.add_trace(
                    go.Bar(
                        x=wf_df['date'],
                        y=high_errors,
                        name='High Error',
                        marker_color=['#ef5350' if e > 0 else '#26a69a' for e in high_errors],
                        opacity=0.6
                    ),
                    row=2, col=1
                )

                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

                fig.update_layout(
                    height=700,
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=50, r=50, t=80, b=50)
                )

                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Error ($)", row=2, col=1)

                st.plotly_chart(fig, use_container_width=True)

                # Detailed results table
                with st.expander("View Detailed Results"):
                    display_df = wf_df.copy()
                    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                    display_df['high_error'] = display_df['predicted_high'] - display_df['actual_high']
                    display_df['low_error'] = display_df['predicted_low'] - display_df['actual_low']
                    display_df['high_in_range'] = ((display_df['actual_high'] >= display_df['high_lower']) &
                                                    (display_df['actual_high'] <= display_df['high_upper']))
                    display_df['low_in_range'] = ((display_df['actual_low'] >= display_df['low_lower']) &
                                                   (display_df['actual_low'] <= display_df['low_upper']))

                    st.dataframe(
                        display_df[['date', 'predicted_high', 'actual_high', 'high_error', 'high_in_range',
                                   'predicted_low', 'actual_low', 'low_error', 'low_in_range', 'retrained']].round(2),
                        use_container_width=True,
                        height=400
                    )

                    # Download button
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        "Download Results CSV",
                        csv,
                        file_name=f"walk_forward_{st.session_state.get('ticker', 'SPY')}_{wf_df['date'].iloc[0].strftime('%Y%m%d')}_{wf_df['date'].iloc[-1].strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

                # ==================== PREDICT TOMORROW'S RANGE ====================
                st.markdown("---")
                st.subheader("Predict Tomorrow's Range")

                # Check if we have a trained model from the walk-forward analysis
                wf_current_model = st.session_state.get('wf_current_model')

                if wf_current_model is not None:
                    pred_col1, pred_col2 = st.columns([3, 1])

                    with pred_col2:
                        predict_tomorrow_btn = st.button(
                            "🔮 Predict Tomorrow's Range",
                            type="primary",
                            use_container_width=True,
                            help="Use the trained model to predict tomorrow's high/low range"
                        )

                    if predict_tomorrow_btn or st.session_state.get('wf_tomorrow_prediction') is not None:
                        with st.spinner("Generating prediction..."):
                            try:
                                # Get options features for enhanced prediction
                                wf_options = None
                                if polygon:
                                    try:
                                        wf_options = wf_current_model.get_options_features(
                                            st.session_state.get('ticker', 'SPY'),
                                            pred_df['close'].iloc[-1]
                                        )
                                    except:
                                        pass

                                # Make prediction using the trained model
                                if predict_tomorrow_btn:
                                    tomorrow_pred = wf_current_model.predict_daily_range(
                                        pred_df,
                                        options_features=wf_options
                                    )
                                    st.session_state['wf_tomorrow_prediction'] = tomorrow_pred
                                else:
                                    tomorrow_pred = st.session_state['wf_tomorrow_prediction']

                                # Display prediction stats
                                with pred_col1:
                                    st.markdown("**Tomorrow's Predicted Range**")

                                pred_stat_col1, pred_stat_col2, pred_stat_col3, pred_stat_col4 = st.columns(4)

                                current_close = pred_df['close'].iloc[-1]
                                pred_high = tomorrow_pred['predicted_high']
                                pred_low = tomorrow_pred['predicted_low']
                                pred_range = tomorrow_pred['predicted_range_dollars']
                                pred_range_pct = tomorrow_pred['predicted_range'] * 100

                                with pred_stat_col1:
                                    high_change = ((pred_high - current_close) / current_close) * 100
                                    st.metric("Predicted High", f"${pred_high:.2f}", f"{high_change:+.2f}%")
                                with pred_stat_col2:
                                    low_change = ((pred_low - current_close) / current_close) * 100
                                    st.metric("Predicted Low", f"${pred_low:.2f}", f"{low_change:+.2f}%")
                                with pred_stat_col3:
                                    st.metric("Predicted Range", f"${pred_range:.2f}", f"{pred_range_pct:.2f}%")
                                with pred_stat_col4:
                                    confidence = tomorrow_pred.get('confidence_level', 0.9) * 100
                                    st.metric("CI Level", f"{confidence:.0f}%", help="Confidence Interval level")

                                # Confidence bounds
                                st.markdown("**Prediction Bounds (90% CI)**")
                                bounds_col1, bounds_col2, bounds_col3, bounds_col4 = st.columns(4)

                                with bounds_col1:
                                    st.metric("High Lower", f"${tomorrow_pred['high_lower']:.2f}")
                                with bounds_col2:
                                    st.metric("High Upper", f"${tomorrow_pred['high_upper']:.2f}")
                                with bounds_col3:
                                    st.metric("Low Lower", f"${tomorrow_pred['low_lower']:.2f}")
                                with bounds_col4:
                                    st.metric("Low Upper", f"${tomorrow_pred['low_upper']:.2f}")

                                # Create candlestick chart with prediction
                                st.markdown("---")
                                st.markdown("**Chart: Recent Price Action + Tomorrow's Predicted Range**")

                                # Get last 30 days of data for the chart
                                chart_df = pred_df.tail(30).copy()

                                import plotly.graph_objects as go
                                from plotly.subplots import make_subplots

                                fig_pred = make_subplots(
                                    rows=1, cols=1,
                                    subplot_titles=("Price with Tomorrow's Predicted Range",)
                                )

                                # Candlestick for historical data
                                fig_pred.add_trace(
                                    go.Candlestick(
                                        x=chart_df.index,
                                        open=chart_df['open'],
                                        high=chart_df['high'],
                                        low=chart_df['low'],
                                        close=chart_df['close'],
                                        name='Price',
                                        increasing_line_color='#26a69a',
                                        decreasing_line_color='#ef5350'
                                    )
                                )

                                # Calculate next trading day
                                last_date = chart_df.index[-1]
                                if hasattr(last_date, 'date'):
                                    # Add 1 day, skip weekends
                                    next_date = last_date + pd.Timedelta(days=1)
                                    while next_date.weekday() >= 5:  # Saturday=5, Sunday=6
                                        next_date += pd.Timedelta(days=1)
                                else:
                                    next_date = last_date

                                # Prediction candle (using predicted values)
                                fig_pred.add_trace(
                                    go.Candlestick(
                                        x=[next_date],
                                        open=[current_close],
                                        high=[pred_high],
                                        low=[pred_low],
                                        close=[(pred_high + pred_low) / 2],  # Midpoint as "close"
                                        name='Predicted',
                                        increasing_line_color='#00d4ff',
                                        decreasing_line_color='#00d4ff'
                                    )
                                )

                                # High confidence band (shaded area)
                                fig_pred.add_trace(
                                    go.Scatter(
                                        x=[next_date, next_date],
                                        y=[tomorrow_pred['high_lower'], tomorrow_pred['high_upper']],
                                        mode='lines',
                                        line=dict(color='rgba(255, 152, 0, 0.8)', width=3),
                                        name='High Range (90%)'
                                    )
                                )

                                # Low confidence band
                                fig_pred.add_trace(
                                    go.Scatter(
                                        x=[next_date, next_date],
                                        y=[tomorrow_pred['low_lower'], tomorrow_pred['low_upper']],
                                        mode='lines',
                                        line=dict(color='rgba(33, 150, 243, 0.8)', width=3),
                                        name='Low Range (90%)'
                                    )
                                )

                                # Add horizontal lines for predicted high/low extending from last bar
                                fig_pred.add_hline(
                                    y=pred_high,
                                    line_dash="dash",
                                    line_color="orange",
                                    annotation_text=f"Pred High: ${pred_high:.2f}",
                                    annotation_position="right"
                                )
                                fig_pred.add_hline(
                                    y=pred_low,
                                    line_dash="dash",
                                    line_color="blue",
                                    annotation_text=f"Pred Low: ${pred_low:.2f}",
                                    annotation_position="right"
                                )

                                fig_pred.update_layout(
                                    height=500,
                                    xaxis_rangeslider_visible=False,
                                    showlegend=True,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                    yaxis_title="Price ($)"
                                )

                                st.plotly_chart(fig_pred, use_container_width=True)

                                # Model info
                                st.markdown("---")
                                model_r2 = tomorrow_pred.get('model_r2', 0)
                                st.caption(f"Model R²: {model_r2:.4f} | Method: {tomorrow_pred.get('confidence_method', 'rmse')} | "
                                          f"Last trained on walk-forward data")

                            except Exception as pred_err:
                                st.error(f"Prediction failed: {pred_err}")
                                import traceback
                                st.code(traceback.format_exc())
                else:
                    st.info("Run walk-forward analysis first to train a model for predictions.")

                # ==================== FEATURE ANALYSIS SECTION ====================
                st.markdown("---")
                st.subheader("Feature Analysis & Discovery")

                # Check if we have feature importances
                wf_importances = st.session_state.get('wf_feature_importances', [])
                wf_feat_names = st.session_state.get('wf_feature_names', [])
                wf_analysis_df = st.session_state.get('wf_pred_df', pred_df)

                if wf_importances and len(wf_importances) > 0:
                    # Aggregate feature importances across all trainings
                    all_imp = pd.concat(wf_importances, ignore_index=True)

                    # Calculate mean and std importance per feature
                    agg_importance = all_imp.groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
                    agg_importance.columns = ['feature', 'mean_importance', 'std_importance', 'train_count']
                    agg_importance['consistency'] = agg_importance['mean_importance'] / (agg_importance['std_importance'] + 1e-10)
                    agg_importance = agg_importance.sort_values('mean_importance', ascending=False)

                    # Feature analysis tabs
                    feat_tab1, feat_tab2, feat_tab3, feat_tab4 = st.tabs([
                        "Importance Ranking",
                        "Correlation Analysis",
                        "SHAP Analysis",
                        "Feature Discovery"
                    ])

                    # === TAB 1: IMPORTANCE RANKING ===
                    with feat_tab1:
                        st.markdown("**Aggregated Feature Importance** (across all walk-forward trainings)")

                        # Top features
                        top_n = min(20, len(agg_importance))
                        top_features = agg_importance.head(top_n)

                        # Bar chart
                        import plotly.express as px
                        fig_imp = px.bar(
                            top_features,
                            x='mean_importance',
                            y='feature',
                            orientation='h',
                            error_x='std_importance',
                            title=f'Top {top_n} Features by Mean Importance',
                            labels={'mean_importance': 'Mean Importance', 'feature': 'Feature'}
                        )
                        fig_imp.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_imp, use_container_width=True)

                        # Dead features (consistently low importance)
                        importance_threshold = agg_importance['mean_importance'].quantile(0.25)
                        dead_features = agg_importance[agg_importance['mean_importance'] < importance_threshold]

                        col_top, col_dead = st.columns(2)

                        with col_top:
                            st.markdown("**Top 10 Most Important Features:**")
                            for i, row in agg_importance.head(10).iterrows():
                                st.markdown(f"- `{row['feature']}`: {row['mean_importance']:.4f} (±{row['std_importance']:.4f})")

                        with col_dead:
                            st.markdown(f"**Dead Features** (below {importance_threshold:.4f}):")
                            dead_list = dead_features['feature'].tolist()[:15]
                            if dead_list:
                                for f in dead_list:
                                    st.markdown(f"- `{f}`")
                                if len(dead_features) > 15:
                                    st.markdown(f"*...and {len(dead_features) - 15} more*")
                            else:
                                st.markdown("*No consistently dead features found*")

                    # === TAB 2: CORRELATION ANALYSIS ===
                    with feat_tab2:
                        st.markdown("**Feature-Target Correlation Analysis**")

                        # Create features and target for correlation
                        try:
                            # Use PriceRangePredictor to create features
                            corr_predictor = PriceRangePredictor()
                            corr_features = corr_predictor.create_range_features(wf_analysis_df)
                            corr_targets = corr_predictor.create_targets(wf_analysis_df)

                            if 'next_range' in corr_targets.columns:
                                # Calculate correlations
                                feature_cols = [c for c in corr_features.columns if c in corr_predictor.feature_names]
                                correlations = []

                                for col in feature_cols:
                                    valid_idx = corr_features[col].dropna().index.intersection(corr_targets['next_range'].dropna().index)
                                    if len(valid_idx) > 30:
                                        corr = corr_features.loc[valid_idx, col].corr(corr_targets.loc[valid_idx, 'next_range'])
                                        correlations.append({'feature': col, 'correlation': corr, 'abs_corr': abs(corr)})

                                corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)

                                # Correlation bar chart
                                top_corr = corr_df.head(25)
                                fig_corr = px.bar(
                                    top_corr,
                                    x='correlation',
                                    y='feature',
                                    orientation='h',
                                    title='Top 25 Features by Correlation with Next Day Range',
                                    color='correlation',
                                    color_continuous_scale='RdBu_r',
                                    range_color=[-0.5, 0.5]
                                )
                                fig_corr.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                                st.plotly_chart(fig_corr, use_container_width=True)

                                # Show high correlation features
                                col_pos, col_neg = st.columns(2)
                                with col_pos:
                                    st.markdown("**Positive Correlations** (higher value → larger range):")
                                    pos_corr = corr_df[corr_df['correlation'] > 0.1].head(10)
                                    for _, row in pos_corr.iterrows():
                                        st.markdown(f"- `{row['feature']}`: {row['correlation']:.3f}")

                                with col_neg:
                                    st.markdown("**Negative Correlations** (higher value → smaller range):")
                                    neg_corr = corr_df[corr_df['correlation'] < -0.1].head(10)
                                    for _, row in neg_corr.iterrows():
                                        st.markdown(f"- `{row['feature']}`: {row['correlation']:.3f}")

                                # Correlation vs Importance comparison
                                st.markdown("---")
                                st.markdown("**Correlation vs Importance Comparison**")
                                merged = agg_importance.merge(corr_df[['feature', 'correlation', 'abs_corr']], on='feature', how='inner')
                                if len(merged) > 0:
                                    fig_compare = px.scatter(
                                        merged,
                                        x='abs_corr',
                                        y='mean_importance',
                                        hover_data=['feature'],
                                        title='Feature Correlation vs Model Importance',
                                        labels={'abs_corr': 'Absolute Correlation', 'mean_importance': 'Model Importance'}
                                    )
                                    fig_compare.update_layout(height=400)
                                    st.plotly_chart(fig_compare, use_container_width=True)

                                    # Undervalued features (high correlation, low importance)
                                    merged['corr_rank'] = merged['abs_corr'].rank(ascending=False)
                                    merged['imp_rank'] = merged['mean_importance'].rank(ascending=False)
                                    merged['rank_diff'] = merged['corr_rank'] - merged['imp_rank']
                                    undervalued = merged[merged['rank_diff'] < -10].sort_values('rank_diff')

                                    if len(undervalued) > 0:
                                        st.markdown("**Potentially Undervalued Features** (high correlation but low importance):")
                                        for _, row in undervalued.head(5).iterrows():
                                            st.markdown(f"- `{row['feature']}`: corr={row['correlation']:.3f}, imp={row['mean_importance']:.4f}")

                        except Exception as corr_err:
                            st.warning(f"Could not compute correlations: {corr_err}")

                    # === TAB 3: SHAP ANALYSIS ===
                    with feat_tab3:
                        st.markdown("**SHAP Value Analysis**")
                        st.info("SHAP analysis shows HOW each feature affects predictions, not just importance.")

                        try:
                            import shap
                            import matplotlib.pyplot as plt
                            SHAP_AVAILABLE = True
                        except ImportError:
                            SHAP_AVAILABLE = False
                            st.warning("SHAP library not installed. Run: `pip install shap`")

                        if SHAP_AVAILABLE:
                            if st.button("Run SHAP Analysis", help="This may take a minute to compute"):
                                with st.spinner("Computing SHAP values..."):
                                    try:
                                        # Train a model on full data for SHAP
                                        shap_predictor = PriceRangePredictor()
                                        shap_features = shap_predictor.create_range_features(wf_analysis_df)
                                        shap_targets = shap_predictor.create_targets(wf_analysis_df)

                                        X_shap = shap_features[shap_predictor.feature_names].dropna()
                                        y_shap = shap_targets.loc[X_shap.index, 'next_range'].dropna()
                                        common_idx = X_shap.index.intersection(y_shap.index)
                                        X_shap = X_shap.loc[common_idx]
                                        y_shap = y_shap.loc[common_idx]

                                        # Scale and train
                                        from sklearn.preprocessing import StandardScaler
                                        scaler = StandardScaler()
                                        X_scaled = scaler.fit_transform(X_shap)

                                        # Train simple XGBoost
                                        import xgboost as xgb
                                        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, verbosity=0)
                                        model.fit(X_scaled, y_shap)

                                        # Compute SHAP values (use sample for speed)
                                        sample_size = min(500, len(X_scaled))
                                        X_sample = X_scaled[:sample_size]

                                        explainer = shap.TreeExplainer(model)
                                        shap_values = explainer.shap_values(X_sample)

                                        # SHAP summary plot
                                        st.markdown("**SHAP Summary Plot**")
                                        fig_shap, ax = plt.subplots(figsize=(10, 8))
                                        shap.summary_plot(shap_values, X_shap.iloc[:sample_size], feature_names=shap_predictor.feature_names, show=False, max_display=20)
                                        st.pyplot(fig_shap)
                                        plt.close()

                                        # Mean absolute SHAP values
                                        mean_shap = np.abs(shap_values).mean(axis=0)
                                        shap_importance = pd.DataFrame({
                                            'feature': shap_predictor.feature_names,
                                            'shap_importance': mean_shap
                                        }).sort_values('shap_importance', ascending=False)

                                        st.session_state['shap_importance'] = shap_importance

                                        st.markdown("**Top SHAP Features:**")
                                        for i, row in shap_importance.head(10).iterrows():
                                            st.markdown(f"- `{row['feature']}`: {row['shap_importance']:.4f}")

                                    except Exception as shap_err:
                                        st.error(f"SHAP analysis failed: {shap_err}")

                            # Show stored SHAP results if available
                            if 'shap_importance' in st.session_state:
                                shap_imp = st.session_state['shap_importance']
                                fig_shap_bar = px.bar(
                                    shap_imp.head(20),
                                    x='shap_importance',
                                    y='feature',
                                    orientation='h',
                                    title='Top 20 Features by Mean |SHAP|'
                                )
                                fig_shap_bar.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                                st.plotly_chart(fig_shap_bar, use_container_width=True)

                    # === TAB 4: FEATURE DISCOVERY ===
                    with feat_tab4:
                        st.markdown("**Feature Discovery Tools**")
                        st.markdown("Generate and test new features based on top performers.")

                        # Get top features for discovery
                        top_10_features = agg_importance.head(10)['feature'].tolist()

                        discovery_tab1, discovery_tab2, discovery_tab3, discovery_tab4, discovery_tab5 = st.tabs([
                            "Lag Analysis",
                            "Interaction Terms",
                            "Rolling Windows",
                            "Regime Features",
                            "Volume Analysis"
                        ])

                        # --- LAG ANALYSIS ---
                        with discovery_tab1:
                            st.markdown("**Lag Analysis** - Test different lag periods for top features")

                            lag_feature = st.selectbox("Select feature to analyze lags:", top_10_features[:5] if len(top_10_features) >= 5 else top_10_features)
                            max_lag = st.slider("Maximum lag (days):", 1, 20, 10)

                            if st.button("Run Lag Analysis"):
                                with st.spinner("Analyzing lags..."):
                                    try:
                                        lag_predictor = PriceRangePredictor()
                                        lag_features = lag_predictor.create_range_features(wf_analysis_df)
                                        lag_targets = lag_predictor.create_targets(wf_analysis_df)

                                        if lag_feature in lag_features.columns and 'next_range' in lag_targets.columns:
                                            lag_results = []
                                            base_series = lag_features[lag_feature]
                                            target_series = lag_targets['next_range']

                                            for lag in range(0, max_lag + 1):
                                                lagged = base_series.shift(lag)
                                                valid_idx = lagged.dropna().index.intersection(target_series.dropna().index)
                                                if len(valid_idx) > 30:
                                                    corr = lagged.loc[valid_idx].corr(target_series.loc[valid_idx])
                                                    lag_results.append({'lag': lag, 'correlation': corr, 'abs_corr': abs(corr)})

                                            lag_df = pd.DataFrame(lag_results)
                                            best_lag = lag_df.loc[lag_df['abs_corr'].idxmax()]

                                            fig_lag = px.line(lag_df, x='lag', y='correlation', markers=True,
                                                            title=f'Correlation of {lag_feature} at Different Lags')
                                            fig_lag.add_hline(y=0, line_dash="dash", line_color="gray")
                                            st.plotly_chart(fig_lag, use_container_width=True)

                                            st.success(f"**Best lag: {int(best_lag['lag'])} days** (correlation: {best_lag['correlation']:.3f})")

                                            if best_lag['lag'] > 0:
                                                st.markdown(f"**Suggestion:** Add `{lag_feature}_lag{int(best_lag['lag'])}` to your features")

                                    except Exception as lag_err:
                                        st.error(f"Lag analysis failed: {lag_err}")

                        # --- INTERACTION TERMS ---
                        with discovery_tab2:
                            st.markdown("**Interaction Terms** - Combine top features")

                            col1, col2 = st.columns(2)
                            with col1:
                                feat1 = st.selectbox("Feature 1:", top_10_features, key="int_feat1")
                            with col2:
                                feat2 = st.selectbox("Feature 2:", [f for f in top_10_features if f != feat1], key="int_feat2")

                            interaction_type = st.selectbox("Interaction type:", ["Multiply", "Divide", "Add", "Subtract"])

                            if st.button("Test Interaction"):
                                with st.spinner("Testing interaction..."):
                                    try:
                                        int_predictor = PriceRangePredictor()
                                        int_features = int_predictor.create_range_features(wf_analysis_df)
                                        int_targets = int_predictor.create_targets(wf_analysis_df)

                                        if feat1 in int_features.columns and feat2 in int_features.columns:
                                            s1 = int_features[feat1]
                                            s2 = int_features[feat2]

                                            if interaction_type == "Multiply":
                                                interaction = s1 * s2
                                                name = f"{feat1}_x_{feat2}"
                                            elif interaction_type == "Divide":
                                                interaction = s1 / (s2 + 1e-10)
                                                name = f"{feat1}_div_{feat2}"
                                            elif interaction_type == "Add":
                                                interaction = s1 + s2
                                                name = f"{feat1}_plus_{feat2}"
                                            else:
                                                interaction = s1 - s2
                                                name = f"{feat1}_minus_{feat2}"

                                            # Calculate correlation
                                            valid_idx = interaction.dropna().index.intersection(int_targets['next_range'].dropna().index)
                                            if len(valid_idx) > 30:
                                                int_corr = interaction.loc[valid_idx].corr(int_targets.loc[valid_idx, 'next_range'])

                                                # Compare to individual features
                                                corr1 = s1.loc[valid_idx].corr(int_targets.loc[valid_idx, 'next_range'])
                                                corr2 = s2.loc[valid_idx].corr(int_targets.loc[valid_idx, 'next_range'])

                                                st.markdown(f"**Results:**")
                                                st.markdown(f"- `{feat1}` correlation: {corr1:.3f}")
                                                st.markdown(f"- `{feat2}` correlation: {corr2:.3f}")
                                                st.markdown(f"- **`{name}` correlation: {int_corr:.3f}**")

                                                if abs(int_corr) > max(abs(corr1), abs(corr2)):
                                                    st.success(f"Interaction `{name}` is better than individual features!")
                                                else:
                                                    st.info("Interaction doesn't improve over individual features.")

                                    except Exception as int_err:
                                        st.error(f"Interaction test failed: {int_err}")

                        # --- ROLLING WINDOWS ---
                        with discovery_tab3:
                            st.markdown("**Rolling Window Analysis** - Test different lookback periods")

                            base_metric = st.selectbox("Base metric:", ["range_pct", "volatility_5", "atr_14", "returns_1d"])
                            windows_to_test = st.multiselect("Window sizes to test:", [3, 5, 7, 10, 14, 21, 30, 50], default=[5, 10, 21])

                            if st.button("Test Rolling Windows") and windows_to_test:
                                with st.spinner("Testing windows..."):
                                    try:
                                        roll_predictor = PriceRangePredictor()
                                        roll_features = roll_predictor.create_range_features(wf_analysis_df)
                                        roll_targets = roll_predictor.create_targets(wf_analysis_df)

                                        # Get base data
                                        if 'range_pct' in wf_analysis_df.columns:
                                            base_data = wf_analysis_df['range_pct']
                                        else:
                                            base_data = (wf_analysis_df['high'] - wf_analysis_df['low']) / wf_analysis_df['close']

                                        window_results = []
                                        for w in windows_to_test:
                                            # Mean
                                            rolled_mean = base_data.rolling(w).mean()
                                            valid_idx = rolled_mean.dropna().index.intersection(roll_targets['next_range'].dropna().index)
                                            if len(valid_idx) > 30:
                                                corr_mean = rolled_mean.loc[valid_idx].corr(roll_targets.loc[valid_idx, 'next_range'])
                                                window_results.append({'window': w, 'aggregation': 'mean', 'correlation': corr_mean})

                                            # Std
                                            rolled_std = base_data.rolling(w).std()
                                            valid_idx = rolled_std.dropna().index.intersection(roll_targets['next_range'].dropna().index)
                                            if len(valid_idx) > 30:
                                                corr_std = rolled_std.loc[valid_idx].corr(roll_targets.loc[valid_idx, 'next_range'])
                                                window_results.append({'window': w, 'aggregation': 'std', 'correlation': corr_std})

                                        window_df = pd.DataFrame(window_results)

                                        fig_window = px.bar(
                                            window_df,
                                            x='window',
                                            y='correlation',
                                            color='aggregation',
                                            barmode='group',
                                            title='Correlation by Rolling Window Size'
                                        )
                                        st.plotly_chart(fig_window, use_container_width=True)

                                        best_window = window_df.loc[window_df['correlation'].abs().idxmax()]
                                        st.success(f"**Best window:** {int(best_window['window'])} days ({best_window['aggregation']}) with correlation {best_window['correlation']:.3f}")

                                    except Exception as roll_err:
                                        st.error(f"Rolling window test failed: {roll_err}")

                        # --- REGIME FEATURES ---
                        with discovery_tab4:
                            st.markdown("**Regime-Specific Analysis** - Feature importance in different market conditions")

                            regime_metric = st.selectbox("Regime split metric:", ["volatility", "trend", "range_size"])

                            if st.button("Analyze Regime Features"):
                                with st.spinner("Analyzing regimes..."):
                                    try:
                                        reg_predictor = PriceRangePredictor()
                                        reg_features = reg_predictor.create_range_features(wf_analysis_df)
                                        reg_targets = reg_predictor.create_targets(wf_analysis_df)

                                        # Define regime
                                        if regime_metric == "volatility":
                                            if 'volatility_20' in reg_features.columns:
                                                regime_series = reg_features['volatility_20']
                                            else:
                                                regime_series = wf_analysis_df['close'].pct_change().rolling(20).std()
                                        elif regime_metric == "trend":
                                            regime_series = wf_analysis_df['close'].pct_change(20)
                                        else:  # range_size
                                            regime_series = (wf_analysis_df['high'] - wf_analysis_df['low']) / wf_analysis_df['close']

                                        median_val = regime_series.median()
                                        high_regime_idx = regime_series[regime_series > median_val].index
                                        low_regime_idx = regime_series[regime_series <= median_val].index

                                        # Calculate correlations in each regime
                                        feature_cols = [c for c in reg_features.columns if c in reg_predictor.feature_names][:30]

                                        regime_results = []
                                        for col in feature_cols:
                                            # High regime
                                            high_idx = reg_features[col].dropna().index.intersection(reg_targets['next_range'].dropna().index).intersection(high_regime_idx)
                                            if len(high_idx) > 20:
                                                corr_high = reg_features.loc[high_idx, col].corr(reg_targets.loc[high_idx, 'next_range'])
                                            else:
                                                corr_high = 0

                                            # Low regime
                                            low_idx = reg_features[col].dropna().index.intersection(reg_targets['next_range'].dropna().index).intersection(low_regime_idx)
                                            if len(low_idx) > 20:
                                                corr_low = reg_features.loc[low_idx, col].corr(reg_targets.loc[low_idx, 'next_range'])
                                            else:
                                                corr_low = 0

                                            regime_results.append({
                                                'feature': col,
                                                f'high_{regime_metric}': corr_high,
                                                f'low_{regime_metric}': corr_low,
                                                'diff': abs(corr_high) - abs(corr_low)
                                            })

                                        regime_df = pd.DataFrame(regime_results)

                                        # Features better in high regime
                                        high_better = regime_df.sort_values('diff', ascending=False).head(10)
                                        low_better = regime_df.sort_values('diff', ascending=True).head(10)

                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown(f"**Better in High {regime_metric.title()}:**")
                                            for _, row in high_better.iterrows():
                                                st.markdown(f"- `{row['feature']}`: high={row[f'high_{regime_metric}']:.3f}, low={row[f'low_{regime_metric}']:.3f}")

                                        with col2:
                                            st.markdown(f"**Better in Low {regime_metric.title()}:**")
                                            for _, row in low_better.iterrows():
                                                st.markdown(f"- `{row['feature']}`: high={row[f'high_{regime_metric}']:.3f}, low={row[f'low_{regime_metric}']:.3f}")

                                        # Scatter plot
                                        fig_regime = px.scatter(
                                            regime_df,
                                            x=f'low_{regime_metric}',
                                            y=f'high_{regime_metric}',
                                            hover_data=['feature'],
                                            title=f'Feature Correlation: High vs Low {regime_metric.title()} Regime'
                                        )
                                        fig_regime.add_shape(type="line", x0=-0.5, y0=-0.5, x1=0.5, y1=0.5, line=dict(dash="dash", color="gray"))
                                        st.plotly_chart(fig_regime, use_container_width=True)

                                        st.markdown("*Points above the diagonal work better in high regime, below work better in low regime*")

                                    except Exception as reg_err:
                                        st.error(f"Regime analysis failed: {reg_err}")

                        # --- VOLUME ANALYSIS ---
                        with discovery_tab5:
                            st.markdown("**Volume Derivatives Analysis**")
                            st.markdown("Test volume and its derivatives (velocity, acceleration) as predictive features.")

                            vol_windows = st.multiselect(
                                "Smoothing windows for derivatives:",
                                [1, 3, 5, 7, 10, 14, 21],
                                default=[1, 5, 14],
                                help="Test different smoothing windows for volume derivatives"
                            )

                            if st.button("Analyze Volume Features"):
                                with st.spinner("Analyzing volume features..."):
                                    try:
                                        vol_predictor = PriceRangePredictor()
                                        vol_features = vol_predictor.create_range_features(wf_analysis_df)
                                        vol_targets = vol_predictor.create_targets(wf_analysis_df)

                                        # Get volume data
                                        volume = wf_analysis_df['volume'].copy()

                                        # Normalize volume (relative to 20-day average)
                                        volume_norm = volume / volume.rolling(20).mean()

                                        vol_results = []

                                        for window in vol_windows:
                                            # Smooth volume
                                            if window > 1:
                                                vol_smooth = volume.rolling(window).mean()
                                                vol_norm_smooth = volume_norm.rolling(window).mean()
                                            else:
                                                vol_smooth = volume
                                                vol_norm_smooth = volume_norm

                                            # Volume (raw and normalized)
                                            valid_idx = vol_smooth.dropna().index.intersection(vol_targets['next_range'].dropna().index)
                                            if len(valid_idx) > 30:
                                                corr_vol = vol_smooth.loc[valid_idx].corr(vol_targets.loc[valid_idx, 'next_range'])
                                                vol_results.append({
                                                    'feature': f'volume_smooth{window}',
                                                    'type': 'Volume',
                                                    'window': window,
                                                    'correlation': corr_vol
                                                })

                                            valid_idx = vol_norm_smooth.dropna().index.intersection(vol_targets['next_range'].dropna().index)
                                            if len(valid_idx) > 30:
                                                corr_vol_norm = vol_norm_smooth.loc[valid_idx].corr(vol_targets.loc[valid_idx, 'next_range'])
                                                vol_results.append({
                                                    'feature': f'volume_rel_smooth{window}',
                                                    'type': 'Volume (Relative)',
                                                    'window': window,
                                                    'correlation': corr_vol_norm
                                                })

                                            # Velocity (1st derivative)
                                            vol_velocity = vol_smooth.diff()
                                            valid_idx = vol_velocity.dropna().index.intersection(vol_targets['next_range'].dropna().index)
                                            if len(valid_idx) > 30:
                                                corr_vel = vol_velocity.loc[valid_idx].corr(vol_targets.loc[valid_idx, 'next_range'])
                                                vol_results.append({
                                                    'feature': f'volume_velocity_w{window}',
                                                    'type': 'Velocity (dV/dt)',
                                                    'window': window,
                                                    'correlation': corr_vel
                                                })

                                            # Acceleration (2nd derivative)
                                            vol_accel = vol_velocity.diff()
                                            valid_idx = vol_accel.dropna().index.intersection(vol_targets['next_range'].dropna().index)
                                            if len(valid_idx) > 30:
                                                corr_acc = vol_accel.loc[valid_idx].corr(vol_targets.loc[valid_idx, 'next_range'])
                                                vol_results.append({
                                                    'feature': f'volume_accel_w{window}',
                                                    'type': 'Acceleration (d²V/dt²)',
                                                    'window': window,
                                                    'correlation': corr_acc
                                                })

                                            # Normalized velocity
                                            vol_norm_velocity = vol_norm_smooth.diff()
                                            valid_idx = vol_norm_velocity.dropna().index.intersection(vol_targets['next_range'].dropna().index)
                                            if len(valid_idx) > 30:
                                                corr_norm_vel = vol_norm_velocity.loc[valid_idx].corr(vol_targets.loc[valid_idx, 'next_range'])
                                                vol_results.append({
                                                    'feature': f'volume_rel_velocity_w{window}',
                                                    'type': 'Rel. Velocity',
                                                    'window': window,
                                                    'correlation': corr_norm_vel
                                                })

                                        # Volume-Price divergence
                                        price_change = wf_analysis_df['close'].pct_change(5)
                                        vol_change = volume.pct_change(5)
                                        divergence = price_change - vol_change  # If price up but volume down = bearish divergence
                                        valid_idx = divergence.dropna().index.intersection(vol_targets['next_range'].dropna().index)
                                        if len(valid_idx) > 30:
                                            corr_div = divergence.loc[valid_idx].corr(vol_targets.loc[valid_idx, 'next_range'])
                                            vol_results.append({
                                                'feature': 'price_volume_divergence_5d',
                                                'type': 'Divergence',
                                                'window': 5,
                                                'correlation': corr_div
                                            })

                                        vol_df = pd.DataFrame(vol_results)
                                        vol_df['abs_corr'] = vol_df['correlation'].abs()
                                        vol_df = vol_df.sort_values('abs_corr', ascending=False)

                                        # Display results
                                        fig_vol = px.bar(
                                            vol_df,
                                            x='correlation',
                                            y='feature',
                                            color='type',
                                            orientation='h',
                                            title='Volume Feature Correlations with Next Day Range',
                                            color_discrete_sequence=px.colors.qualitative.Set2
                                        )
                                        fig_vol.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                                        st.plotly_chart(fig_vol, use_container_width=True)

                                        # Best features
                                        st.markdown("**Top Volume Features:**")
                                        for _, row in vol_df.head(5).iterrows():
                                            direction = "+" if row['correlation'] > 0 else ""
                                            st.markdown(f"- `{row['feature']}`: {direction}{row['correlation']:.3f}")

                                        # Suggestions
                                        best_vol = vol_df.iloc[0]
                                        st.success(f"**Best volume feature:** `{best_vol['feature']}` with correlation {best_vol['correlation']:.3f}")

                                        if best_vol['correlation'] > 0.05:
                                            st.markdown("**Suggestion:** Add these volume derivatives to your feature set in `price_prediction.py`:")
                                            st.code(f"""
# In create_range_features():
volume = df['volume']
volume_norm = volume / volume.rolling(20).mean()

# Volume velocity (1st derivative)
features['volume_velocity'] = volume.rolling({best_vol['window']}).mean().diff()

# Volume acceleration (2nd derivative)
features['volume_accel'] = features['volume_velocity'].diff()

# Relative volume velocity
features['volume_rel_velocity'] = volume_norm.diff()
""", language='python')

                                    except Exception as vol_err:
                                        st.error(f"Volume analysis failed: {vol_err}")

                else:
                    st.info("Run walk forward analysis first to enable feature analysis.")

            else:
                st.info("Run walk forward analysis to see results here.")

    elif not PRICE_PREDICTION_AVAILABLE:
        st.markdown("---")
        st.info("Price Prediction Suite not available. Ensure price_prediction.py is in the project directory.")

    # ============================================================================
    # OPTIONS TRADING BUILDER SECTION
    # ============================================================================
    try:
        from options_builder import (
            OptionsStrategyEngine, TradeTracker,
            format_recommendation_for_display, create_payoff_data
        )
        OPTIONS_BUILDER_AVAILABLE = True
    except ImportError:
        OPTIONS_BUILDER_AVAILABLE = False

    if OPTIONS_BUILDER_AVAILABLE:
        st.markdown("---")
        st.header("Options Trading Builder")

        # Show data requirements status
        st.markdown("""
        <div style="background: #1e1e2e; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #f39c12;">
            <p style="margin: 0; font-size: 14px; color: #ccc;">
                <strong style="color: #f39c12;">Required Data Sources:</strong><br>
                <span style="color: #888;">1. <strong>Velocity Strategy</strong> (Step 5c) → Direction, TP/SL targets, Hold time</span><br>
                <span style="color: #888;">2. <strong>Range Prediction</strong> (Daily Range or Walk Forward) → Expected move, High/Low</span><br>
                <span style="color: #888;">Both are needed for optimal options trade recommendations.</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Initialize session state for options builder
        if 'options_recommendations' not in st.session_state:
            st.session_state.options_recommendations = []
        if 'options_trade_tracker' not in st.session_state:
            st.session_state.options_trade_tracker = TradeTracker()

        # Create tabs
        opt_tab1, opt_tab2, opt_tab3, opt_tab4, opt_tab5 = st.tabs([
            "Strategy Builder",
            "Trade Recommendations",
            "Position Tracker",
            "Performance",
            "Backtest"
        ])

        # ======================
        # TAB 1: STRATEGY BUILDER
        # ======================
        with opt_tab1:
            st.subheader("Build Options Strategy")

            # Get all velocity strategies with metadata (same logic as Backtest tab)
            import glob
            all_opt_strategy_dirs = sorted(glob.glob("velocity_strategies/*"), reverse=True)
            all_opt_strategy_dirs = [d for d in all_opt_strategy_dirs if os.path.isdir(d)]

            opt_strategy_info = []
            for d in all_opt_strategy_dirs:
                dir_name = os.path.basename(d)
                config_path = os.path.join(d, 'velocity_config.json')
                ticker = "Unknown"
                signal = "unknown"
                created = ""

                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            cfg = json.load(f)
                        ticker = cfg.get('ticker', 'Unknown')
                        signal = cfg.get('signal_mode', cfg.get('signal_type', 'unknown'))
                    except:
                        pass

                # Extract date from directory name
                parts = dir_name.split('_')
                for i, part in enumerate(parts):
                    if len(part) == 8 and part.isdigit():
                        try:
                            date_str = part
                            time_str = parts[i+1] if i+1 < len(parts) and len(parts[i+1]) == 6 and parts[i+1].isdigit() else "000000"
                            created = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}"
                        except:
                            pass
                        break

                opt_strategy_info.append({
                    'dir': dir_name,
                    'ticker': ticker,
                    'signal': signal,
                    'created': created,
                    'display': f"{dir_name} | {ticker} | {created}" if created else dir_name
                })

            # Input row
            col1, col2 = st.columns([1, 1])
            with col1:
                num_contracts = st.number_input("Contracts", min_value=1, max_value=100, value=1, key="opt_contracts")
            with col2:
                # All strategies dropdown with metadata
                opt_strat_options = ["None"] + [s['display'] for s in opt_strategy_info]
                opt_strat_idx = st.selectbox("Load Velocity Strategy", range(len(opt_strat_options)),
                                             format_func=lambda i: opt_strat_options[i], key="opt_strategy_select")

                if opt_strat_idx > 0:
                    selected_strategy = opt_strategy_info[opt_strat_idx - 1]['dir']
                    opt_ticker = opt_strategy_info[opt_strat_idx - 1]['ticker']
                    st.caption(f"Ticker: {opt_ticker} | Signal: {opt_strategy_info[opt_strat_idx - 1]['signal']}")
                else:
                    selected_strategy = "None"
                    opt_ticker = "SPY"

            # Second row for range prediction selection
            pred_col1, pred_col2 = st.columns([3, 2])
            with pred_col1:
                # Find all range predictions (not filtered by ticker)
                prediction_files = sorted(glob.glob("predictions/*.json"), reverse=True)
                prediction_options = ["None"] + [os.path.basename(f) for f in prediction_files]
                selected_prediction = st.selectbox("Load Range Prediction", prediction_options, key="opt_prediction_select")
            with pred_col2:
                st.caption("Save predictions from Daily Range or Walk Forward Analysis tabs above")

            # Data Readiness Status
            has_strategy = selected_strategy != "None"
            has_prediction = selected_prediction != "None" and len(prediction_files) > 0
            prediction_path = f"predictions/{selected_prediction}" if has_prediction else None

            status_col1, status_col2, status_col3 = st.columns([1, 1, 2])
            with status_col1:
                if has_strategy:
                    st.success("Velocity Strategy: Ready")
                else:
                    st.warning("Velocity Strategy: Missing")
                    if len(all_opt_strategy_dirs) == 0:
                        st.caption("No saved strategies. Go to Step 5c to create one.")
            with status_col2:
                if has_prediction:
                    st.success("Range Prediction: Ready")
                else:
                    st.warning("Range Prediction: Missing")
                    st.caption("Run Daily Range or Walk Forward, then save.")
            with status_col3:
                if has_strategy and has_prediction:
                    st.info("All data ready! Generate recommendations below.")
                elif has_strategy or has_prediction:
                    st.warning("Partial data - recommendations will use defaults for missing values.")
                else:
                    st.error("No data loaded - complete Step 5c and Price Prediction Suite first.")

            st.markdown("---")

            # Load saved data
            col1, col2 = st.columns(2)

            velocity_strategy = None
            range_prediction = None

            with col1:
                st.markdown("**Velocity Strategy**")
                if selected_strategy != "None":
                    strategy_path = f"velocity_strategies/{selected_strategy}"
                    engine = OptionsStrategyEngine()
                    velocity_strategy = engine.load_velocity_strategy(strategy_path)

                    if velocity_strategy:
                        st.success(f"Loaded: {velocity_strategy['strategy_id']}")
                        params = velocity_strategy.get('parameters', {})
                        direction = velocity_strategy.get('direction', 'neutral')

                        st.metric("Direction", direction.upper())

                        m1, m2 = st.columns(2)
                        with m1:
                            tp = params.get('take_profit_pct', 0)
                            st.metric("Take Profit", f"{tp:.1f}%")
                        with m2:
                            sl = params.get('stop_loss_pct', 0)
                            st.metric("Stop Loss", f"{sl:.1f}%")

                        avg_hold = velocity_strategy.get('avg_hold_days', 10)
                        st.metric("Avg Hold", f"~{avg_hold:.0f} days")
                    else:
                        st.warning("Could not load strategy config")
                else:
                    st.info("Select a saved strategy to load")

            with col2:
                st.markdown("**Range Prediction**")
                # Use selected prediction from dropdown (prediction_path set earlier)
                if prediction_path and os.path.exists(prediction_path):
                    # Use module-level import (don't import locally - causes scoping issues)
                    range_prediction = PriceRangePredictor.load_predictions(prediction_path)

                    if range_prediction:
                        preds = range_prediction.get('predictions', {})
                        st.success(f"Loaded: {range_prediction.get('timestamp', '')[:10]}")

                        m1, m2 = st.columns(2)
                        with m1:
                            pred_high = preds.get('predicted_high', 0)
                            st.metric("Pred High", f"${pred_high:.2f}")
                        with m2:
                            pred_low = preds.get('predicted_low', 0)
                            st.metric("Pred Low", f"${pred_low:.2f}")

                        current = preds.get('current_close', 0)
                        if current > 0 and pred_high > 0:
                            exp_move = ((pred_high - pred_low) / current) * 100
                            st.metric("Expected Move", f"{exp_move:.2f}%")

                        r2 = range_prediction.get('model_r2', 0)
                        st.metric("Model R²", f"{r2:.4f}")
                else:
                    st.info("Select a saved prediction from dropdown above")
                    st.caption("Or run Walk-Forward Analysis and save predictions first")

            st.markdown("---")

            # Market Context (get current price)
            st.markdown("**Market Context**")

            # Get current price
            try:
                import yfinance as yf
                ticker_data = yf.Ticker(opt_ticker)
                current_price = ticker_data.info.get('regularMarketPrice') or ticker_data.info.get('previousClose', 0)
                if current_price == 0:
                    hist = ticker_data.history(period="1d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
            except:
                current_price = 0

            if current_price > 0:
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with mc2:
                    # Try to get IV from Polygon
                    iv_rank = "N/A"
                    if 'polygon_manager' in dir() and polygon_manager:
                        try:
                            iv_data = polygon_manager.get_atm_iv(opt_ticker)
                            if iv_data:
                                iv_rank = f"{iv_data.get('iv_rank', 0):.0f}%"
                        except:
                            pass
                    st.metric("IV Rank", iv_rank)
                with mc3:
                    st.metric("IV %ile", "N/A")
                with mc4:
                    regime = "Unknown"
                    st.metric("Regime", regime)
            else:
                st.warning(f"Could not fetch current price for {opt_ticker}")

            st.markdown("---")

            # DTE Selection
            st.markdown("**DTE Selection**")

            # Calculate suggested DTE based on strategy's avg hold time
            suggested_dte = 21  # Default
            dte_reasoning = "Default: No strategy loaded"

            if velocity_strategy:
                avg_hold = velocity_strategy.get('avg_hold_days', 10)
                # DTE = avg_hold * 1.5 (buffer for theta decay), clamped to 7-45
                suggested_dte = min(45, max(7, int(avg_hold * 1.5)))
                dte_reasoning = f"Based on {avg_hold:.1f} day avg hold time × 1.5 buffer"

                # Show prominent suggestion
                dte_col1, dte_col2 = st.columns([1, 2])
                with dte_col1:
                    st.metric("Suggested DTE", f"{suggested_dte} days", delta=f"{avg_hold:.0f}d hold × 1.5")
                with dte_col2:
                    st.info(f"**Reasoning:** {dte_reasoning}")
                    st.caption("Formula: hold_days × 1.5 = buffer for theta decay (range: 7-45 DTE)")
            else:
                st.warning("Load a velocity strategy to get DTE suggestion based on historical hold times")

            # DTE options with suggested value highlighted
            dte_options = [7, 14, 21, 30, 45, 60]

            # Find closest match in options or add suggested
            if suggested_dte not in dte_options:
                dte_options = sorted(dte_options + [suggested_dte])

            default_idx = dte_options.index(suggested_dte) if suggested_dte in dte_options else 2

            selected_dte = st.selectbox(
                "Select DTE",
                dte_options,
                index=default_idx,
                key="opt_dte_select",
                format_func=lambda x: f"{x} DTE {'⭐ (Suggested)' if x == suggested_dte else ''}"
            )

            custom_dte = st.number_input("Or enter custom DTE", min_value=1, max_value=365, value=selected_dte, key="opt_custom_dte")
            if custom_dte != selected_dte:
                selected_dte = custom_dte

            # Strategy Type Selection
            st.markdown("---")
            st.markdown("**Strategy Type**")

            # Get strategy recommendation
            recommended_strategy = "vertical_spread"
            recommendation_reason = ""

            if velocity_strategy or range_prediction:
                engine = OptionsStrategyEngine()
                context = {
                    'ticker': opt_ticker,
                    'current_price': current_price,
                    'iv_rank': 50,  # Default if not available
                }
                recommended_strategy, recommendation_reason = engine.select_optimal_strategy(
                    context, velocity_strategy, range_prediction
                )

            if recommendation_reason:
                st.info(f"Recommended: **{recommended_strategy.replace('_', ' ').title()}** - {recommendation_reason}")

            strategy_options = {
                'Single Leg (Call/Put)': 'single_leg',
                'Vertical Spread': 'vertical_spread',
                'Straddle': 'straddle',
                'Strangle': 'strangle',
                'Calendar Spread': 'calendar_spread'
            }

            # Find default index
            default_idx = list(strategy_options.values()).index(recommended_strategy) if recommended_strategy in strategy_options.values() else 1

            selected_strategy_type = st.selectbox(
                "Strategy Type",
                list(strategy_options.keys()),
                index=default_idx,
                key="opt_strategy_type"
            )
            strategy_type = strategy_options[selected_strategy_type]

            # Target/Stop inputs
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                # Calculate target from strategy or prediction
                default_target = current_price * 1.02  # 2% default
                if velocity_strategy:
                    tp_pct = velocity_strategy.get('parameters', {}).get('take_profit_pct', 2)
                    direction = velocity_strategy.get('direction', 'long')
                    if direction in ['long', 'bullish']:
                        default_target = current_price * (1 + tp_pct/100)
                    else:
                        default_target = current_price * (1 - tp_pct/100)

                target_price = st.number_input("Target Price", value=float(default_target), format="%.2f", key="opt_target")

            with col2:
                # Calculate stop from strategy
                default_stop = current_price * 0.98  # 2% default
                if velocity_strategy:
                    sl_pct = velocity_strategy.get('parameters', {}).get('stop_loss_pct', 5)
                    direction = velocity_strategy.get('direction', 'long')
                    if direction in ['long', 'bullish']:
                        default_stop = current_price * (1 - sl_pct/100)
                    else:
                        default_stop = current_price * (1 + sl_pct/100)

                stop_price = st.number_input("Stop Price", value=float(default_stop), format="%.2f", key="opt_stop")

            # Generate button
            st.markdown("---")
            if st.button("Generate Recommendations", type="primary", key="generate_options_btn"):
                if current_price <= 0:
                    st.error("Cannot generate recommendations without current price")
                else:
                    engine = OptionsStrategyEngine()
                    direction = 'long'
                    if velocity_strategy:
                        direction = velocity_strategy.get('direction', 'long')
                    elif target_price < current_price:
                        direction = 'short'

                    # Get expected hold days
                    expected_hold = 10
                    if velocity_strategy:
                        expected_hold = velocity_strategy.get('avg_hold_days', 10)

                    recommendations = engine.generate_recommendations(
                        ticker=opt_ticker,
                        strategy_type=strategy_type,
                        direction=direction,
                        current_price=current_price,
                        target_price=target_price,
                        stop_price=stop_price,
                        dte=selected_dte,
                        num_contracts=num_contracts,
                        expected_hold_days=expected_hold
                    )

                    if recommendations:
                        st.session_state.options_recommendations = recommendations
                        st.success(f"Generated {len(recommendations)} recommendation(s). See 'Trade Recommendations' tab.")
                    else:
                        st.warning("No recommendations generated. Try different parameters.")

        # ==============================
        # TAB 2: TRADE RECOMMENDATIONS
        # ==============================
        with opt_tab2:
            st.subheader("Trade Recommendations")

            if not st.session_state.options_recommendations:
                st.info("No recommendations yet. Use 'Strategy Builder' tab to generate recommendations.")
            else:
                recommendations = st.session_state.options_recommendations

                for i, rec in enumerate(recommendations):
                    is_alternative = rec.get('alternative', False)
                    label = "TOP PICK" if i == 0 and not is_alternative else "ALTERNATIVE"

                    with st.expander(f"{label}: {rec.get('strategy_name', 'Unknown')}", expanded=(i == 0)):
                        # Strategy details
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown(f"**{rec.get('strategy_name', '')}**")
                            st.markdown(f"Ticker: {rec.get('ticker', '')} | Direction: {rec.get('direction', '').upper()}")

                            # Legs
                            st.markdown("**Legs:**")
                            for leg in rec.get('legs', []):
                                action = leg.get('action', '').upper()
                                opt_type = leg.get('type', '').upper()
                                strike = leg.get('strike', 0)
                                exp = leg.get('expiration', 'N/A')
                                contracts = leg.get('contracts', 1)
                                premium = leg.get('premium', 0)
                                st.markdown(f"- {action} {contracts}x ${strike} {opt_type} @ ${premium:.2f} (Exp: {exp})")

                        with col2:
                            # Key metrics
                            entry_cost = rec.get('entry_cost', 0)
                            if entry_cost < 0:
                                st.metric("Credit Received", f"${abs(entry_cost):.2f}")
                            else:
                                st.metric("Entry Cost", f"${entry_cost:.2f}")

                            max_profit = rec.get('max_profit', 'N/A')
                            if isinstance(max_profit, (int, float)):
                                st.metric("Max Profit", f"${max_profit:.2f}")
                            else:
                                st.metric("Max Profit", str(max_profit))

                            max_loss = rec.get('max_loss_dollars', rec.get('max_loss', 'N/A'))
                            if isinstance(max_loss, (int, float)):
                                st.metric("Max Loss", f"${max_loss:.2f}")
                            else:
                                st.metric("Max Loss", str(max_loss))

                        # Additional metrics row
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            if 'break_even' in rec:
                                st.metric("Break-even", f"${rec['break_even']:.2f}")
                            elif 'break_even_up' in rec:
                                st.metric("BE Up", f"${rec['break_even_up']:.2f}")
                        with m2:
                            if 'break_even_down' in rec:
                                st.metric("BE Down", f"${rec['break_even_down']:.2f}")
                            elif 'prob_profit' in rec:
                                st.metric("Prob Profit", f"{rec['prob_profit']:.1f}%")
                        with m3:
                            if 'risk_reward' in rec:
                                st.metric("R:R Ratio", f"1:{rec['risk_reward']:.2f}")
                            elif 'spread_width' in rec:
                                st.metric("Spread Width", f"${rec['spread_width']:.0f}")
                        with m4:
                            st.metric("DTE", f"{rec.get('dte', 'N/A')}")

                        # Notes
                        if rec.get('notes'):
                            st.info(rec['notes'])

                        # Risk warning for unlimited risk strategies
                        if rec.get('risk_warning'):
                            st.error(rec['risk_warning'])

                        # Payoff diagram
                        st.markdown("**Payoff at Expiration:**")
                        try:
                            payoff_df = create_payoff_data(rec)
                            if not payoff_df.empty:
                                import plotly.graph_objects as go
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=payoff_df['price'],
                                    y=payoff_df['payoff'],
                                    mode='lines',
                                    name='P&L',
                                    line=dict(color='blue', width=2)
                                ))
                                fig.add_hline(y=0, line_dash="dash", line_color="gray")

                                # Add break-even lines
                                if 'break_even' in rec:
                                    fig.add_vline(x=rec['break_even'], line_dash="dot", line_color="orange",
                                                 annotation_text="BE")
                                if 'break_even_up' in rec:
                                    fig.add_vline(x=rec['break_even_up'], line_dash="dot", line_color="orange",
                                                 annotation_text="BE Up")
                                if 'break_even_down' in rec:
                                    fig.add_vline(x=rec['break_even_down'], line_dash="dot", line_color="orange",
                                                 annotation_text="BE Down")

                                fig.update_layout(
                                    title="Payoff Diagram",
                                    xaxis_title="Stock Price at Expiration",
                                    yaxis_title="Profit/Loss ($)",
                                    height=300,
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate payoff diagram: {e}")

                        # Action buttons
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button(f"Save Trade", key=f"save_trade_{i}"):
                                tracker = st.session_state.options_trade_tracker
                                trade_id = tracker.save_recommendation(rec)
                                st.success(f"Saved! Trade ID: {trade_id}")
                        with col2:
                            # Copy to clipboard (as text)
                            trade_text = format_recommendation_for_display(rec)
                            st.code(trade_text, language=None)
                        with col3:
                            pass  # Placeholder for future actions

        # ==========================
        # TAB 3: POSITION TRACKER
        # ==========================
        with opt_tab3:
            st.subheader("Position Tracker")

            tracker = st.session_state.options_trade_tracker

            # Open Positions
            st.markdown("### Open Positions")
            open_positions = tracker.get_open_positions()

            if not open_positions:
                st.info("No open positions. Execute a saved recommendation to track it.")
            else:
                for pos in open_positions:
                    with st.expander(f"{pos.get('ticker', '')} - {pos.get('strategy_name', '')}", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            exec_info = pos.get('execution', {})
                            st.metric("Entry Price", f"${exec_info.get('fill_price', pos.get('entry_cost', 0)):.2f}")
                        with col2:
                            # Would need live price to calculate current P&L
                            st.metric("Status", "OPEN")
                        with col3:
                            st.metric("DTE Remaining", f"{pos.get('dte', 'N/A')}")

                        # Close position
                        close_price = st.number_input(f"Close Price", value=0.0, format="%.2f",
                                                      key=f"close_price_{pos['trade_id']}")
                        if st.button(f"Mark Closed", key=f"close_{pos['trade_id']}"):
                            if close_price > 0:
                                tracker.mark_closed(pos['trade_id'], {
                                    'close_price': close_price,
                                    'underlying_price': 0,
                                    'notes': 'Closed via Position Tracker'
                                })
                                st.success("Position closed!")
                                st.rerun()
                            else:
                                st.warning("Enter close price first")

            # Saved Recommendations (not executed)
            st.markdown("---")
            st.markdown("### Saved Recommendations")
            saved_recs = tracker.get_saved_recommendations()

            if not saved_recs:
                st.info("No saved recommendations. Save trades from the 'Trade Recommendations' tab.")
            else:
                for rec in saved_recs:
                    with st.expander(f"{rec.get('ticker', '')} - {rec.get('strategy_name', '')} ({rec['trade_id']})"):
                        st.markdown(f"**Created:** {rec.get('created_at', '')[:16]}")

                        # Show legs summary
                        for leg in rec.get('legs', []):
                            st.markdown(f"- {leg.get('action', '').upper()} {leg.get('contracts', 1)}x ${leg.get('strike', 0)} {leg.get('type', '').upper()}")

                        col1, col2 = st.columns(2)
                        with col1:
                            fill_price = st.number_input("Fill Price", value=float(rec.get('entry_cost', 0)),
                                                         format="%.2f", key=f"fill_{rec['trade_id']}")
                            if st.button("Execute", key=f"exec_{rec['trade_id']}"):
                                tracker.mark_executed(rec['trade_id'], {
                                    'fill_price': fill_price,
                                    'underlying_price': 0,
                                    'notes': ''
                                })
                                st.success("Marked as executed!")
                                st.rerun()
                        with col2:
                            if st.button("Delete", key=f"del_{rec['trade_id']}"):
                                tracker.delete_trade(rec['trade_id'])
                                st.success("Deleted!")
                                st.rerun()

            # Manual Trade Entry
            st.markdown("---")
            st.markdown("### Add Manual Trade")
            with st.expander("Enter a trade executed outside the system"):
                m_ticker = st.text_input("Ticker", key="manual_ticker")
                m_type = st.selectbox("Strategy Type", ["single_leg", "vertical_spread", "straddle", "strangle", "calendar_spread"], key="manual_type")
                m_direction = st.selectbox("Direction", ["bullish", "bearish", "neutral"], key="manual_direction")
                m_entry = st.number_input("Entry Cost/Credit", value=0.0, format="%.2f", key="manual_entry")
                m_contracts = st.number_input("Contracts", min_value=1, value=1, key="manual_contracts")
                m_notes = st.text_area("Notes", key="manual_notes")

                if st.button("Add Manual Trade", key="add_manual"):
                    if m_ticker:
                        manual_trade = {
                            'ticker': m_ticker.upper(),
                            'strategy_type': m_type,
                            'strategy_name': f"Manual {m_type.replace('_', ' ').title()}",
                            'direction': m_direction,
                            'entry_cost': m_entry,
                            'legs': [{'contracts': m_contracts}],
                            'notes': m_notes
                        }
                        trade_id = tracker.save_recommendation(manual_trade)
                        tracker.mark_executed(trade_id, {'fill_price': m_entry})
                        st.success(f"Added trade {trade_id}")
                        st.rerun()
                    else:
                        st.warning("Enter ticker symbol")

        # ======================
        # TAB 4: PERFORMANCE
        # ======================
        with opt_tab4:
            st.subheader("Performance Analytics")

            tracker = st.session_state.options_trade_tracker
            stats = tracker.get_performance_stats()

            if stats['total_trades'] == 0:
                st.info("No closed trades yet. Close some positions to see performance analytics.")
            else:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Trades", stats['total_trades'])
                with col2:
                    st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
                with col3:
                    st.metric("Avg P&L", f"${stats['avg_pnl']:.2f}")
                with col4:
                    total_color = "green" if stats['total_pnl'] >= 0 else "red"
                    st.metric("Total P&L", f"${stats['total_pnl']:.2f}")

                # Best/Worst
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Best Trade", f"${stats['best_trade']:.2f}")
                with col2:
                    st.metric("Worst Trade", f"${stats['worst_trade']:.2f}")

                # Performance by Strategy Type
                st.markdown("---")
                st.markdown("### Performance by Strategy Type")

                by_strategy = stats.get('by_strategy', {})
                if by_strategy:
                    strat_data = []
                    for strat, data in by_strategy.items():
                        win_rate = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
                        strat_data.append({
                            'Strategy': strat.replace('_', ' ').title(),
                            'Trades': data['trades'],
                            'Win Rate': f"{win_rate:.1f}%",
                            'Total P&L': f"${data['pnl']:.2f}"
                        })
                    st.dataframe(pd.DataFrame(strat_data), use_container_width=True)

                # Performance by Ticker
                st.markdown("### Performance by Ticker")

                by_ticker = stats.get('by_ticker', {})
                if by_ticker:
                    ticker_data = []
                    for tick, data in by_ticker.items():
                        win_rate = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
                        ticker_data.append({
                            'Ticker': tick,
                            'Trades': data['trades'],
                            'Win Rate': f"{win_rate:.1f}%",
                            'Total P&L': f"${data['pnl']:.2f}"
                        })
                    st.dataframe(pd.DataFrame(ticker_data), use_container_width=True)

                # Equity Curve
                st.markdown("---")
                st.markdown("### Equity Curve")

                closed_trades = tracker.get_closed_trades()
                if closed_trades:
                    # Build equity curve
                    equity = [0]
                    dates = ['Start']
                    for trade in sorted(closed_trades, key=lambda x: x.get('close', {}).get('date', '')):
                        pnl = trade.get('pnl', {}).get('total', 0)
                        equity.append(equity[-1] + pnl)
                        close_date = trade.get('close', {}).get('date', '')[:10]
                        dates.append(close_date)

                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(equity))),
                        y=equity,
                        mode='lines+markers',
                        name='Equity',
                        line=dict(color='green' if equity[-1] >= 0 else 'red', width=2)
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.update_layout(
                        title="Cumulative P&L",
                        xaxis_title="Trade #",
                        yaxis_title="Cumulative P&L ($)",
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Trade History Table
                st.markdown("---")
                st.markdown("### Trade History")

                if closed_trades:
                    history_data = []
                    for trade in reversed(closed_trades):  # Most recent first
                        pnl = trade.get('pnl', {})
                        history_data.append({
                            'Trade ID': trade['trade_id'],
                            'Ticker': trade.get('ticker', ''),
                            'Strategy': trade.get('strategy_type', '').replace('_', ' ').title(),
                            'Direction': trade.get('direction', ''),
                            'Entry': f"${trade.get('execution', {}).get('fill_price', trade.get('entry_cost', 0)):.2f}",
                            'Exit': f"${trade.get('close', {}).get('close_price', 0):.2f}",
                            'P&L': f"${pnl.get('total', 0):.2f}",
                            'Return': f"{pnl.get('pct_return', 0):.1f}%",
                            'Closed': trade.get('close', {}).get('date', '')[:10]
                        })
                    st.dataframe(pd.DataFrame(history_data), use_container_width=True)

        # ======================
        # TAB 5: BACKTEST
        # ======================
        with opt_tab5:
            st.subheader("Options Strategy Backtest")
            st.caption("Simulate historical options trades based on velocity signals")

            # Input selectors - Find ALL velocity strategies with metadata
            import glob
            from datetime import datetime as dt

            # Get all strategy directories
            all_strategy_dirs = sorted(glob.glob("velocity_strategies/*"), reverse=True)
            all_strategy_dirs = [d for d in all_strategy_dirs if os.path.isdir(d)]

            # Build strategy info list with ticker, signal, and date
            strategy_info = []
            for d in all_strategy_dirs:
                dir_name = os.path.basename(d)
                config_path = os.path.join(d, 'velocity_config.json')
                ticker = "Unknown"
                signal = "unknown"
                created = ""

                # Try to load config for metadata
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            cfg = json.load(f)
                        ticker = cfg.get('ticker', 'Unknown')
                        signal = cfg.get('signal_mode', cfg.get('signal_type', 'unknown'))
                    except:
                        pass

                # Extract date from directory name (format: *_YYYYMMDD_HHMMSS)
                parts = dir_name.split('_')
                for i, part in enumerate(parts):
                    if len(part) == 8 and part.isdigit():  # Date part
                        try:
                            date_str = part
                            time_str = parts[i+1] if i+1 < len(parts) and len(parts[i+1]) == 6 and parts[i+1].isdigit() else "000000"
                            created = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}"
                        except:
                            pass
                        break

                strategy_info.append({
                    'dir': dir_name,
                    'ticker': ticker,
                    'signal': signal,
                    'created': created,
                    'display': f"{dir_name} | {ticker} | {created}" if created else dir_name
                })

            bt_col1, bt_col2 = st.columns(2)

            with bt_col1:
                # Strategy selector showing all strategies with metadata
                bt_strategy_options = ["Select Strategy..."] + [s['display'] for s in strategy_info]
                bt_selected_idx = st.selectbox("Velocity Strategy", range(len(bt_strategy_options)),
                                               format_func=lambda i: bt_strategy_options[i], key="bt_strategy_idx")

                # Get actual directory name from selection
                if bt_selected_idx > 0:
                    bt_selected_strategy = strategy_info[bt_selected_idx - 1]['dir']
                    bt_ticker = strategy_info[bt_selected_idx - 1]['ticker']
                    st.caption(f"Ticker: {bt_ticker} | Signal: {strategy_info[bt_selected_idx - 1]['signal']}")
                else:
                    bt_selected_strategy = "Select Strategy..."
                    bt_ticker = "SPY"

            with bt_col2:
                # Find range predictions (show all, not filtered by ticker)
                bt_prediction_files = sorted(glob.glob("predictions/*.json"), reverse=True)
                bt_prediction_options = ["Select Prediction..."] + [os.path.basename(f) for f in bt_prediction_files]
                bt_selected_prediction = st.selectbox("Range Prediction", bt_prediction_options, key="bt_prediction")

            # Load velocity config for DTE suggestion
            suggested_dte = 21  # Default
            avg_hold_days = None
            if bt_selected_idx > 0:
                strat_config_path = f"velocity_strategies/{bt_selected_strategy}/velocity_config.json"
                if os.path.exists(strat_config_path):
                    try:
                        with open(strat_config_path, 'r') as f:
                            strat_cfg = json.load(f)
                        # Get avg hold days from backtest metrics or calculate from TP/SL
                        avg_hold_days = strat_cfg.get('backtest_metrics', {}).get('avg_hold_days')
                        if not avg_hold_days:
                            avg_hold_days = strat_cfg.get('avg_hold_days')
                        if avg_hold_days:
                            # Suggested DTE = avg_hold * 1.5 (buffer for theta), min 14, max 45
                            suggested_dte = min(45, max(14, int(avg_hold_days * 1.5)))
                    except:
                        pass

            # Parameters row
            bt_param_col1, bt_param_col2, bt_param_col3 = st.columns(3)
            with bt_param_col1:
                bt_contracts = st.number_input("Contracts per Trade", min_value=1, max_value=10, value=1, key="bt_contracts")
            with bt_param_col2:
                st.markdown("**DTE Selection**")
                st.caption("Dynamic per trade: hold_days × 1.5")
                st.caption("Range: 7-45 DTE based on actual hold time")
            with bt_param_col3:
                bt_capital = st.number_input("Starting Capital ($)", min_value=1000, max_value=100000, value=10000, key="bt_capital")

            # Strategy types to evaluate
            ALL_STRATEGY_TYPES = ["vertical_spread", "single_leg", "straddle", "strangle"]
            st.caption("System evaluates all strategy types and recommends the best one")

            st.markdown("---")

            # Run Backtest button
            if st.button("Run Options Backtest", type="primary", key="run_bt"):
                if bt_selected_strategy == "Select Strategy..." :
                    st.error("Please select a velocity strategy")
                else:
                    with st.spinner("Running options backtest..."):
                        # Also check for strategies directory config
                        strategy_config_path = f"velocity_strategies/{bt_selected_strategy}/velocity_config.json"
                        locked_backtest = None
                        velocity_config = None

                        # Load strategy config FIRST to get strategy_name for locked backtest lookup
                        if os.path.exists(strategy_config_path):
                            try:
                                with open(strategy_config_path, 'r') as f:
                                    velocity_config = json.load(f)
                                st.success(f"Loaded velocity config: TP={velocity_config.get('take_profit_pct', 0):.1f}%, SL={velocity_config.get('stop_loss_pct', 0):.1f}%")
                            except Exception as e:
                                st.warning(f"Could not load velocity config: {e}")

                        # Get the strategy_name from config (used for locked backtest filename)
                        # Locked backtests use short names like "velocity_BTC_5y" not full dir names
                        strategy_name = None
                        if velocity_config:
                            strategy_name = velocity_config.get('strategy_name') or velocity_config.get('bundle_name')

                        # Build list of possible locked backtest paths
                        possible_paths = []

                        # 1. If strategy_name from config (most accurate)
                        if strategy_name:
                            possible_paths.append(f"velocity_locked_backtest_{strategy_name}.json")

                        # 2. Try extracting base name by removing timestamp (YYYYMMDD_HHMMSS)
                        import re
                        base_name_match = re.match(r'^(.*?)_\d{8}_\d{6}$', bt_selected_strategy)
                        if base_name_match:
                            base_name = base_name_match.group(1)
                            possible_paths.append(f"velocity_locked_backtest_{base_name}.json")

                        # 3. Try full directory name
                        possible_paths.append(f"velocity_locked_backtest_{bt_selected_strategy}.json")

                        # 4. Try with velocity_ prefix variations
                        if bt_ticker:
                            # Pattern like: velocity_locked_backtest_velocity_BTC_5y.json
                            for timeframe in ['1y', '2y', '5y', '1d', '4h']:
                                possible_paths.append(f"velocity_locked_backtest_velocity_{bt_ticker}_{timeframe}.json")
                                # Also try without "velocity_" prefix
                                possible_paths.append(f"velocity_locked_backtest_{bt_ticker}_{timeframe}.json")

                        # Try to find locked backtest
                        for path in possible_paths:
                            if path and os.path.exists(path):
                                try:
                                    with open(path, 'r') as f:
                                        locked_backtest = json.load(f)
                                    st.info(f"Loaded locked backtest: {len(locked_backtest.get('exits', []))} historical trades")
                                    break
                                except:
                                    pass

                        # Also try loading from the strategy bundle data
                        bundle_data_path = f"velocity_strategies/{bt_selected_strategy}/data.parquet"
                        if os.path.exists(bundle_data_path):
                            try:
                                bt_df = pd.read_parquet(bundle_data_path)
                                st.info(f"Loaded price data: {len(bt_df)} bars")
                            except:
                                bt_df = None
                        else:
                            bt_df = None

                        # Load range prediction if selected
                        range_pred = None
                        if bt_selected_prediction != "Select Prediction...":
                            pred_path = f"predictions/{bt_selected_prediction}"
                            if os.path.exists(pred_path):
                                try:
                                    with open(pred_path, 'r') as f:
                                        range_pred = json.load(f)
                                    st.success(f"Loaded range prediction (R²={range_pred.get('model_r2', 0):.3f})")
                                except:
                                    pass

                        # If no locked backtest, run fresh backtest on bundled data
                        if not locked_backtest or not locked_backtest.get('exits'):
                            if bt_df is not None and velocity_config:
                                st.info("No locked backtest found. Running fresh backtest on bundled data...")
                                try:
                                    # Import backtest function from testing page
                                    from oscillator_predictor_testing_page import run_velocity_backtest

                                    # Build params dict from velocity config
                                    backtest_params = {
                                        'signal_type': velocity_config.get('signal_type', 'any_reversal'),
                                        'vel_smoothing': velocity_config.get('vel_smoothing', 4),
                                        'oversold_threshold': velocity_config.get('oversold_threshold', -0.1),
                                        'overbought_threshold': velocity_config.get('overbought_threshold', 0.1),
                                        'stop_loss_pct': velocity_config.get('stop_loss_pct', 5.0),
                                        'take_profit_pct': velocity_config.get('take_profit_pct', 10.0),
                                        'min_bars_between': velocity_config.get('min_bars_between', 2),
                                        'extreme_zone_mult': velocity_config.get('extreme_zone_mult', 2.0),
                                        'exit_on_opposite_signal': velocity_config.get('exit_on_opposite_signal', True),
                                        'exit_on_midline_cross': velocity_config.get('exit_on_midline_cross', False),
                                        'rsi_filter': velocity_config.get('rsi_filter', 'none'),
                                        'rsi_period': velocity_config.get('rsi_period', 14),
                                        'rsi_oversold': velocity_config.get('rsi_oversold', 30),
                                        'rsi_overbought': velocity_config.get('rsi_overbought', 70),
                                        'use_macd_confirm': velocity_config.get('use_macd_confirm', False),
                                        'use_bb_filter': velocity_config.get('use_bb_filter', False),
                                        'require_accel': velocity_config.get('require_accel', False),
                                    }

                                    # Run backtest
                                    backtest_result = run_velocity_backtest(bt_df, backtest_params)

                                    # Convert trades format to exits format (matching locked backtest schema)
                                    if backtest_result and backtest_result.get('trades'):
                                        exits = []
                                        for trade in backtest_result['trades']:
                                            exits.append({
                                                'date': str(trade.get('exit_date', '')),
                                                'price': trade.get('exit_price', 0),
                                                'pnl': trade.get('pnl', 0),
                                                'reason': trade.get('exit_reason', 'Unknown'),
                                                'entry_price': trade.get('entry_price', 0),
                                                'entry_date': str(trade.get('entry_date', ''))
                                            })
                                        locked_backtest = {
                                            'exits': exits,
                                            'num_trades': len(exits),
                                            'win_rate': backtest_result.get('win_rate', 0),
                                            'total_return': backtest_result.get('total_return', 0),
                                            'profit_factor': backtest_result.get('profit_factor', 0)
                                        }
                                        st.success(f"Generated backtest: {len(exits)} trades from bundled data")
                                    else:
                                        st.warning("Backtest generated no trades with current parameters")
                                except Exception as e:
                                    st.error(f"Could not run backtest: {e}")

                        # Run the backtest simulation for ALL strategy types
                        if locked_backtest and locked_backtest.get('exits'):
                            exits = locked_backtest['exits']
                            tp_pct = velocity_config.get('take_profit_pct', 2.0) if velocity_config else 2.0
                            sl_pct = velocity_config.get('stop_loss_pct', 5.0) if velocity_config else 5.0

                            # Function to simulate a single strategy type
                            def simulate_strategy(strategy_type, exits, bt_contracts):
                                from datetime import datetime
                                results = []
                                total_pnl = 0
                                wins = 0
                                losses = 0

                                for exit_trade in exits:
                                    entry_price = exit_trade.get('entry_price', 0)
                                    exit_price = exit_trade.get('price', 0)
                                    underlying_pnl_pct = exit_trade.get('pnl', 0)
                                    entry_date = exit_trade.get('entry_date', '')
                                    exit_date = exit_trade.get('date', '')
                                    exit_reason = exit_trade.get('reason', '')

                                    if entry_price <= 0:
                                        continue

                                    # Calculate dynamic DTE based on actual hold time
                                    # DTE = hold_days * 1.5 (buffer for theta), min 7, max 45
                                    try:
                                        entry_dt = datetime.strptime(str(entry_date)[:10], '%Y-%m-%d')
                                        exit_dt = datetime.strptime(str(exit_date)[:10], '%Y-%m-%d')
                                        hold_days = max(1, (exit_dt - entry_dt).days)
                                        # DTE should be hold_days * 1.5 to give buffer, but at least 7 days
                                        trade_dte = min(45, max(7, int(hold_days * 1.5)))
                                    except:
                                        hold_days = 5
                                        trade_dte = 14  # Default fallback

                                    direction = 'bullish'
                                    atm_premium_pct = 0.025 * (trade_dte / 21) ** 0.5
                                    atm_premium = entry_price * atm_premium_pct

                                    if strategy_type == 'single_leg':
                                        underlying_move = exit_price - entry_price
                                        delta = 0.50
                                        if underlying_pnl_pct > 2:
                                            delta = 0.65
                                        elif underlying_pnl_pct < -2:
                                            delta = 0.35
                                        option_value_change = delta * underlying_move * 100 * bt_contracts
                                        option_cost = atm_premium * 100 * bt_contracts
                                        option_pnl = max(-option_cost, option_value_change - (option_cost * 0.3))
                                        max_loss = option_cost

                                    elif strategy_type == 'vertical_spread':
                                        spread_width = 5.0
                                        debit = spread_width * 0.45
                                        short_strike = entry_price + spread_width
                                        if exit_price >= short_strike:
                                            option_pnl = (spread_width - debit) * 100 * bt_contracts
                                        elif exit_price <= entry_price:
                                            option_pnl = -debit * 100 * bt_contracts
                                        else:
                                            intrinsic = exit_price - entry_price
                                            option_pnl = (intrinsic - debit) * 100 * bt_contracts
                                        max_loss = debit * 100 * bt_contracts

                                    elif strategy_type == 'straddle':
                                        total_premium = entry_price * 0.045 * (trade_dte / 21) ** 0.5
                                        abs_move = abs(exit_price - entry_price)
                                        if abs_move > total_premium:
                                            option_pnl = (abs_move - total_premium) * 100 * bt_contracts
                                        else:
                                            option_pnl = max(-(total_premium - abs_move) * 100 * bt_contracts, -total_premium * 100 * bt_contracts)
                                        max_loss = total_premium * 100 * bt_contracts

                                    elif strategy_type == 'strangle':
                                        total_premium = entry_price * 0.030 * (trade_dte / 21) ** 0.5
                                        abs_move = abs(exit_price - entry_price)
                                        if abs_move > total_premium:
                                            option_pnl = (abs_move - total_premium) * 100 * bt_contracts
                                        else:
                                            option_pnl = max(-(total_premium - abs_move) * 100 * bt_contracts, -total_premium * 100 * bt_contracts)
                                        max_loss = total_premium * 100 * bt_contracts
                                    else:
                                        option_pnl = 0
                                        max_loss = 0

                                    total_pnl += option_pnl
                                    if option_pnl > 0:
                                        wins += 1
                                    else:
                                        losses += 1

                                    results.append({
                                        'entry_date': entry_date[:10] if entry_date else '',
                                        'exit_date': exit_date[:10] if exit_date else '',
                                        'hold_days': hold_days,
                                        'dte_used': trade_dte,
                                        'entry_price': entry_price,
                                        'exit_price': exit_price,
                                        'underlying_pnl': underlying_pnl_pct,
                                        'option_pnl': option_pnl,
                                        'max_loss': max_loss,
                                        'exit_reason': exit_reason,
                                        'strategy': strategy_type
                                    })

                                return {
                                    'results': results,
                                    'total_pnl': total_pnl,
                                    'wins': wins,
                                    'losses': losses,
                                    'win_rate': (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0,
                                    'num_trades': wins + losses
                                }

                            # Run backtest for ALL strategy types
                            all_results = {}
                            for strat_type in ALL_STRATEGY_TYPES:
                                all_results[strat_type] = simulate_strategy(strat_type, exits, bt_contracts)

                            # Find the best strategy (highest total P&L)
                            best_strategy = max(all_results.keys(), key=lambda k: all_results[k]['total_pnl'])
                            best_by_winrate = max(all_results.keys(), key=lambda k: all_results[k]['win_rate'])
                            best_by_risk_adj = max(all_results.keys(), key=lambda k: all_results[k]['total_pnl'] / max(1, sum(r['max_loss'] for r in all_results[k]['results'])))

                            # Store all results in session state
                            st.session_state.bt_all_results = all_results
                            st.session_state.bt_best_strategy = best_strategy
                            st.session_state.bt_results = all_results[best_strategy]['results']
                            st.session_state.bt_total_pnl = all_results[best_strategy]['total_pnl']
                            st.session_state.bt_wins = all_results[best_strategy]['wins']
                            st.session_state.bt_losses = all_results[best_strategy]['losses']
                            st.session_state.bt_starting_capital = bt_capital

                            st.success(f"Backtest complete! Evaluated {len(ALL_STRATEGY_TYPES)} strategy types across {len(exits)} trades.")
                        else:
                            st.error("No historical trade data found. Run velocity_live_trader first to generate trade history.")

            # Display results if available
            if 'bt_results' in st.session_state and st.session_state.bt_results:
                results = st.session_state.bt_results
                total_pnl = st.session_state.bt_total_pnl
                wins = st.session_state.bt_wins
                losses = st.session_state.bt_losses
                starting_capital = st.session_state.get('bt_starting_capital', 10000)

                st.markdown("---")

                # STRATEGY COMPARISON TABLE (if all results available)
                if 'bt_all_results' in st.session_state:
                    all_results = st.session_state.bt_all_results
                    best_strategy = st.session_state.get('bt_best_strategy', 'vertical_spread')

                    st.markdown("### Strategy Comparison")
                    st.caption("All 4 strategy types evaluated on the same historical trades")

                    # Build comparison dataframe
                    comparison_data = []
                    for strat, data in all_results.items():
                        pnl_return = (data['total_pnl'] / starting_capital * 100) if starting_capital > 0 else 0
                        avg_pnl = data['total_pnl'] / data['num_trades'] if data['num_trades'] > 0 else 0
                        is_best = strat == best_strategy
                        comparison_data.append({
                            'Strategy': f"{'⭐ ' if is_best else ''}{strat.replace('_', ' ').title()}",
                            'Total P&L': f"${data['total_pnl']:,.2f}",
                            'Return': f"{pnl_return:.1f}%",
                            'Win Rate': f"{data['win_rate']:.1f}%",
                            'Trades': data['num_trades'],
                            'Wins': data['wins'],
                            'Losses': data['losses'],
                            'Avg P&L': f"${avg_pnl:.2f}",
                            '_pnl_sort': data['total_pnl']  # Hidden sort column
                        })

                    comp_df = pd.DataFrame(comparison_data)
                    comp_df = comp_df.sort_values('_pnl_sort', ascending=False).drop('_pnl_sort', axis=1)
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)

                    # Recommendation box
                    best_data = all_results[best_strategy]
                    best_return = (best_data['total_pnl'] / starting_capital * 100) if starting_capital > 0 else 0

                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%); padding: 20px; border-radius: 10px; margin: 15px 0; border-left: 4px solid #00d4aa;">
                        <h4 style="color: #00d4aa; margin: 0 0 10px 0;">⭐ Recommended: {best_strategy.replace('_', ' ').title()}</h4>
                        <p style="color: #ccc; margin: 0;">
                            Highest Total P&L: <strong style="color: #00d4aa;">${best_data['total_pnl']:,.2f}</strong> ({best_return:.1f}% return)<br>
                            Win Rate: <strong>{best_data['win_rate']:.1f}%</strong> |
                            {best_data['wins']} wins, {best_data['losses']} losses
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Strategy selector for detailed view
                    st.markdown("---")
                    selected_view_strat = st.selectbox(
                        "View detailed results for:",
                        list(all_results.keys()),
                        index=list(all_results.keys()).index(best_strategy),
                        format_func=lambda x: f"{'⭐ ' if x == best_strategy else ''}{x.replace('_', ' ').title()}",
                        key="bt_view_strategy"
                    )

                    # Update displayed results based on selection
                    results = all_results[selected_view_strat]['results']
                    total_pnl = all_results[selected_view_strat]['total_pnl']
                    wins = all_results[selected_view_strat]['wins']
                    losses = all_results[selected_view_strat]['losses']

                st.markdown(f"### Detailed Results: {selected_view_strat.replace('_', ' ').title() if 'bt_all_results' in st.session_state else 'Selected Strategy'}")

                # Summary metrics
                total_trades = wins + losses
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                total_return = (total_pnl / starting_capital * 100) if starting_capital > 0 else 0
                avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Total Trades", total_trades)
                with metric_col2:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                with metric_col3:
                    st.metric("Total P&L", f"${total_pnl:,.2f}", delta=f"{total_return:.1f}%")
                with metric_col4:
                    st.metric("Avg Trade P&L", f"${avg_pnl:.2f}")

                # Additional stats
                if results:
                    winners = [r['option_pnl'] for r in results if r['option_pnl'] > 0]
                    losers = [r['option_pnl'] for r in results if r['option_pnl'] <= 0]

                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    with stat_col1:
                        avg_win = sum(winners) / len(winners) if winners else 0
                        st.metric("Avg Winner", f"${avg_win:.2f}")
                    with stat_col2:
                        avg_loss = sum(losers) / len(losers) if losers else 0
                        st.metric("Avg Loser", f"${avg_loss:.2f}")
                    with stat_col3:
                        best = max([r['option_pnl'] for r in results]) if results else 0
                        st.metric("Best Trade", f"${best:.2f}")
                    with stat_col4:
                        worst = min([r['option_pnl'] for r in results]) if results else 0
                        st.metric("Worst Trade", f"${worst:.2f}")

                # Equity Curve
                st.markdown("---")
                st.markdown("### Equity Curve")

                equity = [starting_capital]
                dates = ['Start']
                for r in results:
                    equity.append(equity[-1] + r['option_pnl'])
                    dates.append(r['exit_date'])

                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(equity))),
                    y=equity,
                    mode='lines+markers',
                    name='Equity',
                    line=dict(color='#00d4aa' if equity[-1] >= starting_capital else '#ff6b6b', width=2),
                    hovertemplate='Trade %{x}<br>Equity: $%{y:,.2f}<extra></extra>'
                ))
                fig.add_hline(y=starting_capital, line_dash="dash", line_color="gray",
                             annotation_text="Starting Capital")
                fig.update_layout(
                    title=f"Options Backtest Equity Curve ({selected_view_strat.replace('_', ' ').title()})",
                    xaxis_title="Trade #",
                    yaxis_title="Account Value ($)",
                    height=400,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

                # P&L Distribution
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### P&L Distribution")
                    pnls = [r['option_pnl'] for r in results]
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=pnls,
                        nbinsx=20,
                        marker_color=['#00d4aa' if p > 0 else '#ff6b6b' for p in sorted(pnls)]
                    ))
                    fig_hist.add_vline(x=0, line_dash="dash", line_color="white")
                    fig_hist.update_layout(
                        xaxis_title="P&L ($)",
                        yaxis_title="Count",
                        height=300,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col2:
                    st.markdown("### Underlying vs Options P&L")
                    underlying_pnls = [r['underlying_pnl'] for r in results]
                    option_pnls = [r['option_pnl'] for r in results]

                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(
                        x=underlying_pnls,
                        y=option_pnls,
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=['#00d4aa' if p > 0 else '#ff6b6b' for p in option_pnls]
                        ),
                        hovertemplate='Underlying: %{x:.2f}%<br>Options: $%{y:.2f}<extra></extra>'
                    ))
                    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
                    fig_scatter.update_layout(
                        xaxis_title="Underlying P&L (%)",
                        yaxis_title="Options P&L ($)",
                        height=300,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                # Trade Details Table
                st.markdown("---")
                st.markdown("### Trade Details")

                trade_df = pd.DataFrame(results)
                trade_df['option_pnl'] = trade_df['option_pnl'].apply(lambda x: f"${x:,.2f}")
                trade_df['underlying_pnl'] = trade_df['underlying_pnl'].apply(lambda x: f"{x:.2f}%")
                trade_df['entry_price'] = trade_df['entry_price'].apply(lambda x: f"${x:.2f}")
                trade_df['exit_price'] = trade_df['exit_price'].apply(lambda x: f"${x:.2f}")
                trade_df['max_loss'] = trade_df['max_loss'].apply(lambda x: f"${x:.2f}")

                trade_df = trade_df.rename(columns={
                    'entry_date': 'Entry Date',
                    'exit_date': 'Exit Date',
                    'hold_days': 'Hold',
                    'dte_used': 'DTE',
                    'entry_price': 'Entry $',
                    'exit_price': 'Exit $',
                    'underlying_pnl': 'Stock P&L',
                    'option_pnl': 'Options P&L',
                    'max_loss': 'Max Risk',
                    'exit_reason': 'Exit Reason',
                    'strategy': 'Strategy'
                })

                # Reorder columns to show Hold and DTE prominently
                col_order = ['Entry Date', 'Exit Date', 'Hold', 'DTE', 'Entry $', 'Exit $', 'Stock P&L', 'Options P&L', 'Max Risk', 'Exit Reason', 'Strategy']
                trade_df = trade_df[[c for c in col_order if c in trade_df.columns]]

                st.dataframe(trade_df, use_container_width=True, hide_index=True)

                # Comparison to underlying
                st.markdown("---")
                st.markdown("### Options vs Stock Comparison")

                underlying_total = sum([r['underlying_pnl'] for r in results])
                underlying_capital_result = starting_capital * (1 + underlying_total / 100)

                comp_col1, comp_col2, comp_col3 = st.columns(3)
                with comp_col1:
                    st.metric(
                        "Stock-Only Return",
                        f"${underlying_capital_result - starting_capital:,.2f}",
                        f"{underlying_total:.1f}%"
                    )
                with comp_col2:
                    st.metric(
                        "Options Return",
                        f"${total_pnl:,.2f}",
                        f"{total_return:.1f}%"
                    )
                with comp_col3:
                    leverage = total_return / underlying_total if underlying_total != 0 else 0
                    st.metric(
                        "Effective Leverage",
                        f"{leverage:.2f}x",
                        "vs buying stock"
                    )

    else:
        st.markdown("---")
        st.info("Options Trading Builder not available. Ensure options_builder.py is in the project directory.")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="Oscillator Predictor", layout="wide")
    render_oscillator_predictor_page()
