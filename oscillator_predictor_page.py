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
import multiprocessing
from joblib import Parallel, delayed

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

    try:
        # Suppress Optuna's verbose logging during optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Use all available cores (N_JOBS_OPTUNA = N_CORES - 1)
        # The objective is a picklable class instance with the data path
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=N_JOBS_OPTUNA,
            show_progress_bar=False,  # Don't show in terminal, we have Streamlit UI
            catch=(Exception,)  # Catch exceptions to continue other trials
        )

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

    for i, model_type in enumerate(model_types):
        # Create a container for this model's progress
        if progress_placeholder:
            model_header = progress_placeholder.empty()
            model_header.info(f"**Optimizing {model_type}** ({i+1}/{len(model_types)}) - {n_trials} trials with {N_JOBS_OPTUNA} workers...")
            progress_container = progress_placeholder.container()
        else:
            progress_container = None

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

            # Show completion message
            if progress_placeholder:
                model_header.success(f"**{model_type}** complete! Val F1: {result['best_score']:.4f} | Test F1: {metrics['f1_macro']:.4f}")

        except Exception as e:
            results[model_type] = {'error': str(e)}
            if progress_placeholder:
                model_header.error(f"**{model_type}** failed: {str(e)}")

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
    years = st.sidebar.slider("Years of Data", 1, 10, 5)

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
    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

    # Check if key settings changed - if so, clear model state to prevent mismatch
    current_settings = f"{ticker}_{years}_{prominence}_{distance}_{lookahead}_{test_size}"
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
    def load_data(ticker: str, years: int):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        return df

    with st.spinner(f"Loading {ticker} data..."):
        raw_data = load_data(ticker, years)

    if raw_data.empty:
        st.error("Failed to load data. Check ticker symbol.")
        return

    st.success(f"Loaded {len(raw_data)} bars from {raw_data.index[0].date()} to {raw_data.index[-1].date()}")

    # Step 2: Create Composite Oscillator
    st.header("Step 2: Create Composite Oscillator")

    with st.spinner("Calculating indicators and composite oscillator..."):
        df = create_composite_oscillator(raw_data)

    # Show component indicators
    component_cols = [c for c in df.columns if c.endswith('_norm') or c in ['bb_position', 'adx_trend']]

    with st.expander("View Component Indicators", expanded=False):
        st.write(f"**{len(component_cols)} normalized indicators:**")
        for col in component_cols:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                st.write(f"- {col}: range [{valid_data.min():.2f}, {valid_data.max():.2f}]")

    # Plot composite oscillator
    fig_osc = go.Figure()
    fig_osc.add_trace(go.Scatter(x=df.index, y=df['composite_smooth'],
                                  mode='lines', name='Composite Oscillator'))
    fig_osc.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_osc.add_hline(y=-0.5, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_osc.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_osc.update_layout(title="Composite Oscillator", height=300,
                          yaxis_title="Value (-1 to +1)", xaxis_title="Date")
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
        years = (df.index[-1] - df.index[0]).days / 365
        st.write(f"**Buy & Hold Return:** {bh_return:.1f}% over {years:.1f} years")
        st.write(f"**Strategy vs B&H:** {total_return_pct - bh_return:+.1f}%")

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
    fig_pv = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Price with peak/valley markers
    fig_pv.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines',
                                 name='Price', line=dict(color='blue')), row=1, col=1)

    # Mark peaks (sell points)
    peak_mask = df['is_peak'] == 1
    fig_pv.add_trace(go.Scatter(x=df.index[peak_mask], y=df['close'][peak_mask],
                                 mode='markers', name='Oscillator Peak (SELL)',
                                 marker=dict(symbol='triangle-down', size=12, color='red')),
                     row=1, col=1)

    # Mark valleys (buy points)
    valley_mask = df['is_valley'] == 1
    fig_pv.add_trace(go.Scatter(x=df.index[valley_mask], y=df['close'][valley_mask],
                                 mode='markers', name='Oscillator Valley (BUY)',
                                 marker=dict(symbol='triangle-up', size=12, color='green')),
                     row=1, col=1)

    # Oscillator
    fig_pv.add_trace(go.Scatter(x=df.index, y=df['composite_smooth'], mode='lines',
                                 name='Oscillator', line=dict(color='purple')), row=2, col=1)
    fig_pv.add_hline(y=0.5, line_dash="dash", line_color="red", row=2, col=1)
    fig_pv.add_hline(y=-0.5, line_dash="dash", line_color="green", row=2, col=1)

    fig_pv.update_layout(height=600, title="Price with Oscillator Peaks/Valleys")
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

        # ============================================================
        # SCIPY PEAKS CANDLESTICK CHART
        # ============================================================
        st.subheader("Scipy Peaks Trading Chart")

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
                x=test_df_scipy.index,
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
                    x=[t['date'] for t in scipy_entries],
                    y=[t['price'] for t in scipy_entries],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=15, color='lime', line=dict(width=2, color='darkgreen')),
                    name='BUY (Valley)',
                    hovertemplate='BUY<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )

        # Add exit markers (SELL at peaks)
        if scipy_exits:
            fig_scipy.add_trace(
                go.Scatter(
                    x=[t['date'] for t in scipy_exits],
                    y=[t['price'] for t in scipy_exits],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=15, color='red', line=dict(width=2, color='darkred')),
                    name='SELL (Peak)',
                    hovertemplate='SELL<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )

        # Draw trade lines connecting entries to exits
        for trade in scipy_exits:
            if 'entry_date' in trade and 'entry_price' in trade:
                color = 'rgba(0,255,0,0.3)' if trade.get('pnl', 0) > 0 else 'rgba(255,0,0,0.3)'
                fig_scipy.add_trace(
                    go.Scatter(
                        x=[trade['entry_date'], trade['date']],
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
                x=test_df_scipy.index,
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
                    x=peaks_df.index,
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
                    x=valleys_df.index,
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
                x=test_df_scipy.index,
                y=test_df_scipy['cum_market'],
                mode='lines',
                name='Buy & Hold',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
        fig_scipy.add_trace(
            go.Scatter(
                x=test_df_scipy.index,
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
                print(f"\n{'='*60}", flush=True)
                print(f"🚀 VELOCITY OPTIMIZATION STARTING", flush=True)
                print(f"   Total trials: {grid_iterations:,} ({trials_per_worker:,} per worker)", flush=True)
                print(f"   Parallel workers: {n_workers}", flush=True)
                print(f"   Optimizing for: {optimize_metric}", flush=True)
                print(f"{'='*60}", flush=True)

                # Run parallel studies using joblib with loky backend (uses spawn)
                status_text.text(f"Starting {n_workers} parallel Optuna studies (see terminal for progress)...")

                # Each worker gets a different random seed for diversity
                results_lists = Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
                    delayed(optuna_worker.run_velocity_study)(
                        data_path, trials_per_worker, seed=42 + i, optimize_metric=optimize_metric, worker_id=i
                    )
                    for i in range(n_workers)
                )

                # Merge all results
                all_results = []
                for result_list in results_lists:
                    all_results.extend(result_list)

                progress_bar.progress(1.0)
                status_text.text(f"Completed! Tested {len(all_results):,} parameter combinations.")

                # Console completion summary
                print(f"\n{'='*60}", flush=True)
                print(f"✅ OPTIMIZATION COMPLETE", flush=True)
                print(f"   Total results: {len(all_results):,}", flush=True)
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
                                         format_func=lambda x: f"#{x} - {results_df.iloc[x-1]['total_return']:.1%} return, {int(results_df.iloc[x-1]['num_trades'])} trades",
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

            deploy_col1, deploy_col2, deploy_col3 = st.columns(3)

            with deploy_col1:
                if st.button("💾 Save Strategy Config", key="save_vel_strat"):
                    # Get currently applied params or use current UI settings
                    if 'vel_best_params' in st.session_state and st.session_state['vel_best_params']:
                        params = st.session_state['vel_best_params']
                    else:
                        # Use current slider settings
                        params = {
                            'signal_type': signal_type,
                            'extreme_zone_mult': extreme_zone_mult,
                            'vel_smoothing': vel_smoothing,
                            'min_bars_between': min_bars_between,
                            'require_accel': require_accel_confirm,
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

                    # Create velocity strategy config with native Python types
                    velocity_config = {
                        "strategy_type": "velocity",
                        "strategy_name": str(strategy_name),
                        "ticker": str(ticker),
                        "interval": "1d",
                        "created_at": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),

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

                        # Discord webhook (if provided)
                        "discord_webhook": discord_webhook if discord_webhook else None,
                    }

                    # Save to strategies folder
                    strategies_dir = "velocity_strategies"
                    os.makedirs(strategies_dir, exist_ok=True)

                    config_path = os.path.join(strategies_dir, f"{strategy_name}.json")

                    with open(config_path, 'w') as f:
                        json.dump(velocity_config, f, indent=4)

                    st.success(f"✅ Strategy saved to: {config_path}")
                    st.session_state['vel_saved_config_path'] = config_path

            with deploy_col2:
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
                                'require_accel': require_accel_confirm,
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
                            "interval": "1d",
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

                        # Save the calculated DataFrame for exact matching in live trader
                        try:
                            if 'test_df_vel' in st.session_state:
                                data_path = os.path.join(prod_dir, "velocity_data.parquet")
                                st.session_state['test_df_vel'].to_parquet(data_path)
                                st.info(f"📊 Saved {len(st.session_state['test_df_vel'])} bars to: {data_path}")
                            else:
                                st.warning("No velocity data in session - run backtest first")
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

            with deploy_col3:
                if st.button("📦 Save Permanently", key="save_vel_permanent"):
                    # Get currently applied params or use current UI settings
                    if 'vel_best_params' in st.session_state and st.session_state['vel_best_params']:
                        params = st.session_state['vel_best_params']
                    else:
                        params = {
                            'signal_type': signal_type,
                            'extreme_zone_mult': extreme_zone_mult,
                            'vel_smoothing': vel_smoothing,
                            'min_bars_between': min_bars_between,
                            'require_accel': require_accel_confirm,
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

                    # Create bundle directory name
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    bundle_name = f"{ticker}_{params.get('signal_type', 'velocity')}_{timestamp}"
                    bundle_dir = os.path.join("velocity_strategies", bundle_name)
                    os.makedirs(bundle_dir, exist_ok=True)

                    # Create full config for bundle
                    bundle_config = {
                        "strategy_type": "velocity",
                        "strategy_name": str(strategy_name),
                        "bundle_name": bundle_name,
                        "ticker": str(ticker),
                        "interval": "1d",
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

                        # Discord webhook
                        "discord_webhook": discord_webhook if discord_webhook else None,

                        "deployed_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    # Save to bundle directory
                    config_path = os.path.join(bundle_dir, "velocity_config.json")
                    with open(config_path, 'w') as f:
                        json.dump(bundle_config, f, indent=4)

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

        # ============================================================
        # VELOCITY TRADING CHART
        # ============================================================
        st.subheader("Velocity Trading Chart")

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
                x=test_df_vel.index,
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
                    x=[t['date'] for t in vel_entries],
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
                    x=[t['date'] for t in vel_exits],
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
                        x=[trade['entry_date'], trade['date']],
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
                x=test_df_vel.index,
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
                    x=buy_signals.index,
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
                    x=sell_signals.index,
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
                x=test_df_vel.index,
                y=test_df_vel['velocity'],
                mode='lines',
                name='Velocity',
                line=dict(color='blue', width=1.5)
            ),
            row=3, col=1
        )
        fig_vel.add_trace(
            go.Scatter(
                x=test_df_vel.index,
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
                x=test_df_vel.index,
                y=test_df_vel['cum_market'],
                mode='lines',
                name='Buy & Hold',
                line=dict(color='blue', width=2)
            ),
            row=4, col=1
        )
        fig_vel.add_trace(
            go.Scatter(
                x=test_df_vel.index,
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
                x=test_df.index,
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
                    x=entry_dates,
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
                    x=exit_dates,
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
                        x=[trade['entry_date'], trade['date']],
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
                    x=test_df.index,
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
                            x=peaks.index,
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
                            x=valleys.index,
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
                    x=test_df.index,
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
                    x=test_df.index,
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
                x=test_df.index,
                y=test_df['cum_market'],
                mode='lines',
                name='Buy & Hold',
                line=dict(color='blue', width=2)
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=test_df.index,
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
                    x=test_df.index,
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
                    x=test_df.index[peak_mask],
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
                    x=test_df.index[valley_mask],
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
                        x=test_df.index[correct_buy_mask],
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
                        x=test_df.index[false_buy_mask],
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
                        x=test_df.index[correct_sell_mask],
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
                        x=test_df.index[false_sell_mask],
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
                    x=test_df.index,
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
                        x=engine.data.index,
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
                                x=[t['entry_date'] for t in long_entries],
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
                                x=[t['exit_date'] for t in long_entries],
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
                                x=[t['entry_date'] for t in short_entries],
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
                                x=[t['exit_date'] for t in short_entries],
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
                                x=[t['entry_date'], t['exit_date']],
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
                            x=engine.indicators.index,
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
                        x=equity_df.index,
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
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="Oscillator Predictor", layout="wide")
    render_oscillator_predictor_page()
