"""
Peak/Valley ML Trading System v2 - Enhanced Clean Architecture
===============================================================

FEATURES:
- SMOTE balancing for imbalanced peak/valley labels
- Optuna hyperparameter optimization
- Candlestick charts with trade markers
- Equity curves and performance analysis
- Feature importance visualization
- Proper model/feature bundling

ARCHITECTURE:
- Clean session state management
- No redundant imports or code execution
- Consistent feature pipeline between training and production
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('optuna').setLevel(logging.WARNING)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    defaults = {
        'v2_data': None,
        'v2_labels': None, 
        'v2_features': None,
        'v2_model': None,
        'v2_model_path': None,
        'v2_ticker': 'BTC-USD',
        'v2_period': '5y',
        'v2_trained': False,
        'v2_backtest': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_price_data(ticker: str, period: str) -> pd.DataFrame:
    """Load OHLCV data from yfinance"""
    import yfinance as yf
    data = yf.Ticker(ticker).history(period=period)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data.columns = [c.lower() for c in data.columns]
    return data[['open', 'high', 'low', 'close', 'volume']]


def detect_peaks_valleys(data: pd.DataFrame, order: int = 5) -> tuple:
    """
    Detect peaks and valleys with PREDICTIVE labeling.
    Returns labels and detection info.
    """
    from scipy.signal import argrelextrema
    
    highs = data['high'].values
    lows = data['low'].values
    
    peak_indices = argrelextrema(highs, np.greater, order=order)[0]
    valley_indices = argrelextrema(lows, np.less, order=order)[0]
    
    # Create labels - signal 1 day BEFORE event
    labels = pd.Series(0, index=data.index, name='label')
    
    for idx in valley_indices:
        if idx > 0:
            labels.iloc[idx - 1] = 1  # BUY before valley
    
    for idx in peak_indices:
        if idx > 0:
            labels.iloc[idx - 1] = -1  # SELL before peak
    
    info = {
        'num_peaks': len(peak_indices),
        'num_valleys': len(valley_indices),
        'peak_dates': data.index[peak_indices].tolist(),
        'valley_dates': data.index[valley_indices].tolist(),
    }
    
    return labels, info


def generate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive technical indicator features"""
    import pandas_ta as ta
    
    df = data.copy()
    
    # Price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_10'] = df['returns'].rolling(10).std()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_open_range'] = (df['close'] - df['open']) / df['open']
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = ta.sma(df['close'], length=period)
        df[f'ema_{period}'] = ta.ema(df['close'], length=period)
        
        # OPTIMIZATION: Remove short-term mean reversion to force trend learning
        # We skip 5 and 10 to stop the model from overfitting to noise
        if period >= 20:
            df[f'close_to_sma{period}'] = df['close'] / df[f'sma_{period}'] - 1
    
    # Momentum indicators
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    df['rsi_7'] = ta.rsi(df['close'], length=7)
    df['rsi_21'] = ta.rsi(df['close'], length=21)
    
    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
    
    # Bollinger Bands
    bbands = ta.bbands(df['close'], length=20, std=2)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)
        # BB position
        if 'BBU_20_2.0' in df.columns and 'BBL_20_2.0' in df.columns:
            df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
    
    # Stochastic
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    if stoch is not None:
        df = pd.concat([df, stoch], axis=1)
    
    # ADX
    adx = ta.adx(df['high'], df['low'], df['close'])
    if adx is not None:
        df = pd.concat([df, adx], axis=1)
    
    # ATR
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_pct'] = df['atr_14'] / df['close'] * 100
    
    # Williams %R
    df['willr_14'] = ta.willr(df['high'], df['low'], df['close'], length=14)
    
    # CCI
    df['cci_14'] = ta.cci(df['high'], df['low'], df['close'], length=14)
    df['cci_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    
    # ROC
    df['roc_5'] = ta.roc(df['close'], length=5)
    df['roc_10'] = ta.roc(df['close'], length=10)
    df['roc_20'] = ta.roc(df['close'], length=20)
    
    # MFI
    df['mfi_14'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
    
    # OBV
    df['obv'] = ta.obv(df['close'], df['volume'])
    df['obv_sma'] = ta.sma(df['obv'], length=20)
    
    # Volume features
    df['volume_sma_10'] = ta.sma(df['volume'], length=10)
    df['volume_sma_20'] = ta.sma(df['volume'], length=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # ==============================================================================
    # NEW: Composite Oscillator & ROC
    # ==============================================================================
    # Normalize key oscillators to -1 to 1 range
    rsi_norm = (df['rsi_14'] - 50) / 50
    willr_norm = (df['willr_14'] + 50) / 50
    cci_norm = (df['cci_14'] / 100).clip(-1, 1)
    roc_norm = (df['roc_10'] / 5).clip(-1, 1)
    
    # Calculate composite (average of available normalized oscillators)
    df['composite_oscillator'] = (rsi_norm + willr_norm + cci_norm + roc_norm) / 4
    
    # Calculate ROC of the composite oscillator (how fast is momentum changing?)
    df['composite_roc_5'] = df['composite_oscillator'].diff(5)
    df['composite_roc_10'] = df['composite_oscillator'].diff(10)
    
    # ==============================================================================
    # NEW: Breakout & Trend Features (Catch "Black Swans")
    # ==============================================================================
    # Long-term momentum
    df['roc_50'] = ta.roc(df['close'], length=50)
    
    # Distance from SMA50 (Trend Strength)
    if 'sma_50' in df.columns:
        df['dist_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    
    # Breakout Signal: Close > 20-day High (Donchian Channel Breakout)
    # Shift 1 to compare Today's Close vs Previous 20 days High
    df['high_20d'] = df['high'].rolling(20).max().shift(1)
    df['breakout_20d'] = (df['close'] > df['high_20d']).astype(int)
    
    # ADX Trend Strength
    # pandas_ta returns columns like ADX_14, DMP_14, DMN_14
    # We concatenated them earlier, so check columns
    adx_col = [c for c in df.columns if c.startswith('ADX_')]
    if adx_col:
        df['adx_trend'] = (df[adx_col[0]] > 25).astype(int)
    
    # Lagged features
    # OPTIMIZATION: Removed 'close_lag' to prevent overfitting to simple price drops (mean reversion).
    # Added composite_slope to detect when the indicator turns up/down.
    df['composite_slope'] = df['composite_oscillator'].diff(1)
    
    # EXPLICIT COMPOSITE SIGNALS (User Observation: Low = Bottom, High = Top)
    df['comp_oversold'] = (df['composite_oscillator'] < -0.6).astype(int)
    df['comp_overbought'] = (df['composite_oscillator'] > 0.6).astype(int)
    
    # Crossing signals (leaving the extreme zone)
    df['comp_cross_low'] = ((df['composite_oscillator'].shift(1) < -0.6) & (df['composite_oscillator'] > -0.6)).astype(int)
    df['comp_cross_high'] = ((df['composite_oscillator'].shift(1) > 0.6) & (df['composite_oscillator'] < 0.6)).astype(int)
    
    # Turning points (Extreme value + Change in direction)
    df['comp_bottom_turn'] = (df['comp_oversold'] & (df['composite_slope'] > 0)).astype(int)
    df['comp_top_turn'] = (df['comp_overbought'] & (df['composite_slope'] < 0)).astype(int)
    
    for lag in [1, 2, 3, 5]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)
        # df[f'close_lag_{lag}'] = df['close'].pct_change(lag) # REMOVED
        df[f'composite_lag_{lag}'] = df['composite_oscillator'].shift(lag)
    
    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
        df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
        df[f'rsi_mean_{window}'] = df['rsi_14'].rolling(window).mean()
        df[f'composite_mean_{window}'] = df['composite_oscillator'].rolling(window).mean()
        df[f'composite_std_{window}'] = df['composite_oscillator'].rolling(window).std()
    
    # ==============================================================================
    # NEW: Interaction Features (Trend * Momentum)
    # ==============================================================================
    # Combine Trend Strength (dist_sma50) with Momentum (rsi)
    if 'dist_sma50' in df.columns and 'rsi_14' in df.columns:
        df['trend_momentum'] = df['dist_sma50'] * (df['rsi_14'] - 50)
        
    # Combine Volatility with Breakout
    if 'volatility_20' in df.columns and 'breakout_20d' in df.columns:
        df['vol_breakout'] = df['volatility_20'] * df['breakout_20d']
        
    # Relative Volume * Price Change (Volume Force)
    if 'volume_ratio' in df.columns:
        df['volume_force'] = df['volume_ratio'] * df['returns']

    # Drop NaN and select numeric features
    df = df.dropna()
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    features = df[feature_cols].select_dtypes(include=[np.number])
    
    # Remove constant columns
    features = features.loc[:, features.std() > 0]
    
    return features


def train_model_with_optuna(features: pd.DataFrame, labels: pd.Series, 
                            model_type: str = 'xgboost', use_smote: bool = True,
                            n_trials: int = 20, progress_callback=None,
                            optimize_metric: str = 'f1_weighted', n_cv_splits: int = 5,
                            class_weight_ratio: float = 1.0):
    """
    Train model with Optuna optimization and SMOTE balancing.
    Uses proper time series cross-validation (no future data leakage).
    
    Args:
        optimize_metric: 'f1_weighted', 'accuracy', or 'f1_macro'
        n_cv_splits: Number of TimeSeriesSplit folds (default 5)
        class_weight_ratio: Multiplier for BUY/SELL class weights (default 1.0)
    """
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.utils.class_weight import compute_sample_weight
    import optuna
    
    # Console output header
    print("\n" + "="*70)
    print(f"🚀 TRAINING {model_type.upper()} MODEL")
    print("="*70)
    
    # Align data
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]
    
    print(f"📊 Dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"📈 Label distribution: {dict(y.value_counts())}")
    
    # Time-based split (80/20) - CRITICAL: no shuffling for time series
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"📅 Train period: {X_train.index[0].strftime('%Y-%m-%d')} to {X_train.index[-1].strftime('%Y-%m-%d')} ({len(X_train)} samples)")
    print(f"📅 Test period:  {X_test.index[0].strftime('%Y-%m-%d')} to {X_test.index[-1].strftime('%Y-%m-%d')} ({len(X_test)} samples)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE balancing (applied to training data only)
    if use_smote:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"⚖️  SMOTE: {len(X_train_scaled)} → {len(X_train_balanced)} samples (balanced)")
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train.values
        print(f"⚖️  SMOTE: Disabled")
    
    # Label encoding for XGBoost
    label_encoder = None
    if model_type == 'xgboost':
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_balanced)
        y_test_encoded = label_encoder.transform(y_test)
        print(f"🏷️  Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    else:
        y_train_encoded = y_train_balanced
        y_test_encoded = y_test.values
    
    # Calculate sample weights if ratio > 1.0
    sample_weights = None
    if class_weight_ratio > 1.0:
        # Identify classes: 0=HOLD, -1=SELL, 1=BUY
        # If label encoded, map accordingly
        if label_encoder:
            # Map original labels to encoded
            hold_val = label_encoder.transform([0])[0]
            buy_val = label_encoder.transform([1])[0]
            sell_val = label_encoder.transform([-1])[0]
        else:
            hold_val, buy_val, sell_val = 0, 1, -1
            
        weights = np.ones(len(y_train_encoded))
        weights[y_train_encoded == buy_val] = class_weight_ratio
        weights[y_train_encoded == sell_val] = class_weight_ratio
        sample_weights = weights
        print(f"⚖️  Applied class weights: BUY/SELL={class_weight_ratio}x, HOLD=1.0x")
    
    print(f"\n🎯 Optimization target: {optimize_metric}")
    print(f"📊 Time Series CV: {n_cv_splits} folds (forward-chaining, no look-ahead)")
    print("-"*70)
    
    # Thread-safe tracking for parallel execution
    import threading
    lock = threading.Lock()
    best_score_so_far = [0.0]
    best_trial_num = [0]
    completed_trials = [0]
    
    # Optuna optimization with console logging (thread-safe)
    def objective(trial):
        if model_type == 'xgboost':
            from xgboost import XGBClassifier
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss',
                'verbosity': 0,
                'n_jobs': 1  # Single thread per model (parallelism at trial level)
            }
            model = XGBClassifier(**params)
        else:
            from sklearn.ensemble import RandomForestClassifier
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42,
                'n_jobs': 1  # Single thread per model (parallelism at trial level)
            }
            
            # For Random Forest, we can use class_weight param directly
            if class_weight_ratio > 1.0:
                # Map: 0=HOLD, 1=BUY, -1=SELL (if not encoded)
                # If encoded, we need to know the integer mapping.
                # Let's assume balanced/encoded labels.
                # Easier to rely on sample_weight in fit_params for consistency with XGBoost
                pass
                
            model = RandomForestClassifier(**params)
        
        # Time Series Cross-Validation (forward-chaining)
        # Manual loop to handle sample_weights correctly and avoid sklearn version issues
        tscv = TimeSeriesSplit(n_splits=n_cv_splits)
        scores = []
        
        for train_index, val_index in tscv.split(X_train_balanced):
            X_tr, X_val = X_train_balanced[train_index], X_train_balanced[val_index]
            y_tr, y_val = y_train_encoded[train_index], y_train_encoded[val_index]
            
            # Split weights if they exist
            if sample_weights is not None:
                w_tr = sample_weights[train_index]
                model.fit(X_tr, y_tr, sample_weight=w_tr)
            else:
                model.fit(X_tr, y_tr)
            
            y_pred = model.predict(X_val)
            
            if optimize_metric == 'f1_weighted':
                score = f1_score(y_val, y_pred, average='weighted')
            elif optimize_metric == 'accuracy':
                score = accuracy_score(y_val, y_pred)
            elif optimize_metric == 'f1_macro':
                score = f1_score(y_val, y_pred, average='macro')
            else:
                score = f1_score(y_val, y_pred, average='weighted') # Default
                
            scores.append(score)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Thread-safe console output
        with lock:
            completed_trials[0] += 1
            trial_num = completed_trials[0]
            is_best = mean_score > best_score_so_far[0]
            if is_best:
                best_score_so_far[0] = mean_score
                best_trial_num[0] = trial.number + 1
                marker = "⭐ NEW BEST"
            else:
                marker = ""
            
            print(f"Trial {trial_num:3d}/{n_trials} | CV Score: {mean_score:.4f} (±{std_score:.4f}) | Best: {best_score_so_far[0]:.4f} {marker}")
        
        return mean_score
    
    # Determine number of parallel jobs (all cores except 1)
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    n_jobs_optuna = max(1, n_cores - 1)  # Leave 1 core free for system
    
    # Run optimization with parallel trials
    # NOTE: Cannot use Streamlit progress callback with parallel execution (NoSessionContext error)
    # Progress is shown via console output instead
    print(f"\n🔍 Starting Optuna optimization ({n_trials} trials)...")
    print(f"🖥️  Using {n_jobs_optuna} parallel workers (of {n_cores} cores)")
    print("-"*70)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=n_jobs_optuna)
    
    # Update UI progress after optimization completes
    if progress_callback:
        progress_callback(n_trials, n_trials)
    
    best_params = study.best_params
    
    print("-"*70)
    print(f"✅ Optimization complete! Best trial: #{best_trial_num[0]} with CV score: {study.best_value:.4f}")
    print(f"📋 Best parameters: {best_params}")
    
    # Train final model with best params
    print(f"\n🏋️ Training final model with best parameters...")
    
    if model_type == 'xgboost':
        from xgboost import XGBClassifier
        best_params.update({'random_state': 42, 'use_label_encoder': False, 
                           'eval_metric': 'mlogloss', 'verbosity': 0})
        model = XGBClassifier(**best_params)
    else:
        from sklearn.ensemble import RandomForestClassifier
        best_params.update({'random_state': 42, 'n_jobs': -1})
        model = RandomForestClassifier(**best_params)
    
    # Apply sample weights if available
    if sample_weights is not None:
        model.fit(X_train_balanced, y_train_encoded, sample_weight=sample_weights)
    else:
        model.fit(X_train_balanced, y_train_encoded)
    
    # Predictions on held-out test set
    y_pred = model.predict(X_test_scaled)
    
    # Decode predictions
    if label_encoder:
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        y_test_decoded = y_test.values
    else:
        y_pred_decoded = y_pred
        y_test_decoded = y_test.values
    
    # Final metrics on TEST set (out-of-sample)
    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    f1 = f1_score(y_test_decoded, y_pred_decoded, average='weighted')
    f1_macro = f1_score(y_test_decoded, y_pred_decoded, average='macro')
    
    # Print final results
    print("\n" + "="*70)
    print("📊 FINAL MODEL PERFORMANCE (Out-of-Sample Test Set)")
    print("="*70)
    print(f"   Accuracy:     {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"   F1 Weighted:  {f1:.4f} ({f1*100:.1f}%)")
    print(f"   F1 Macro:     {f1_macro:.4f} ({f1_macro*100:.1f}%)")
    print(f"   CV Score:     {study.best_value:.4f} ({study.best_value*100:.1f}%)")
    print("="*70)
    
    # Classification report
    print("\n📋 Classification Report (Test Set):")
    print(classification_report(y_test_decoded, y_pred_decoded, 
                               target_names=['SELL (-1)', 'HOLD (0)', 'BUY (1)']))
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': list(X.columns),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        importance_df = None
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': list(X.columns),
        'model_type': model_type,
        'best_params': best_params,
        'accuracy': accuracy,
        'f1_score': f1,
        'cv_score': study.best_value,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'label_distribution': dict(y.value_counts()),
        'split_date': X_train.index[-1],
        'feature_importance': importance_df,
        'use_smote': use_smote,
    }


def save_model(model_data: dict, ticker: str, notes: str = "") -> str:
    """Save model with all required data"""
    save_dir = "saved_models_v2"
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_data['model_type']}_{ticker}_{timestamp}.joblib"
    filepath = os.path.join(save_dir, filename)
    
    payload = {
        'model': model_data['model'],
        'scaler': model_data['scaler'],
        'label_encoder': model_data['label_encoder'],
        'feature_names': model_data['feature_names'],
        'model_type': model_data['model_type'],
        'best_params': model_data['best_params'],
        'ticker': ticker,
        'timestamp': timestamp,
        'notes': notes,
        'metrics': {
            'accuracy': model_data['accuracy'],
            'f1_score': model_data['f1_score'],
            'cv_score': model_data['cv_score'],
        }
    }
    
    joblib.dump(payload, filepath)
    return filepath


def load_model(filepath: str) -> dict:
    """Load a saved model"""
    return joblib.load(filepath)


def generate_signals(model_data: dict, features: pd.DataFrame, threshold: float = 0.5) -> tuple[pd.Series, pd.DataFrame]:
    """
    Generate signals using the trained model with custom threshold.
    Returns: (signals, probabilities)
    """
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    feature_names = model_data['feature_names']
    
    # Align features
    common_features = [f for f in feature_names if f in features.columns]
    if len(common_features) < len(feature_names):
        st.warning(f"Missing {len(feature_names) - len(common_features)} features in current data")
    
    X = features[feature_names]
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Get probabilities
    probs = model.predict_proba(X_scaled)
    
    # Determine classes based on threshold
    # Default classes: 0, 1, 2 (mapped from -1, 0, 1 usually)
    # We need to know which column corresponds to which class
    classes = model.classes_
    
    # Map encoded classes back to -1, 0, 1
    if label_encoder:
        original_classes = label_encoder.inverse_transform(classes)
    else:
        original_classes = classes
        
    # Find column indices for each class
    try:
        buy_idx = np.where(original_classes == 1)[0][0]
        sell_idx = np.where(original_classes == -1)[0][0]
        hold_idx = np.where(original_classes == 0)[0][0]
    except IndexError:
        # Fallback if some classes are missing (unlikely with proper training)
        return pd.Series(0, index=features.index), pd.DataFrame(probs, index=features.index, columns=original_classes)
    
    # Apply threshold logic
    # If prob(BUY) > threshold -> BUY
    # If prob(SELL) > threshold -> SELL
    # Else -> HOLD
    
    signals = np.zeros(len(X))
    
    # Vectorized threshold application
    buy_mask = probs[:, buy_idx] > threshold
    sell_mask = probs[:, sell_idx] > threshold
    
    # Conflict resolution: if both > threshold, pick higher prob
    conflict_mask = buy_mask & sell_mask
    if conflict_mask.any():
        buy_higher = probs[:, buy_idx] > probs[:, sell_idx]
        buy_mask[conflict_mask] = buy_higher[conflict_mask]
        sell_mask[conflict_mask] = ~buy_higher[conflict_mask]
        
    signals[buy_mask] = 1
    signals[sell_mask] = -1
    
    # Create DataFrame for probabilities
    prob_df = pd.DataFrame(probs, index=features.index, columns=[f"prob_{c}" for c in original_classes])
    
    # Add confidence score (prob of predicted class)
    prob_df['confidence'] = probs.max(axis=1)
    prob_df['predicted_signal'] = signals
    
    return pd.Series(signals, index=features.index), prob_df


def run_backtest(data: pd.DataFrame, signals: pd.Series, initial_capital: float = 100000, limit_pct: float = 0.0) -> dict:
    """
    Run backtest with detailed trade tracking.
    If limit_pct > 0, attempts to enter at Close * (1 - limit_pct) on the NEXT day.
    """
    common_idx = data.index.intersection(signals.index)
    df = data.loc[common_idx].copy()
    df['signal'] = signals.loc[common_idx]
    
    capital = initial_capital
    position = 0
    shares = 0
    trades = []
    equity_curve = []
    
    # Pre-calculate next day low for limit checks
    if limit_pct > 0:
        df['next_low'] = df['low'].shift(-1)
    
    for i in range(len(df)):
        date = df.index[i]
        row = df.iloc[i]
        price = row['close']
        signal = row['signal']
        
        # Track equity
        current_equity = shares * price if position == 1 else capital
        equity_curve.append({'date': date, 'equity': current_equity, 'price': price})
        
        if signal == 1 and position == 0:  # BUY
            entry_price = price
            filled = True
            fill_date = date
            
            if limit_pct > 0:
                # Limit Logic: Try to fill next day
                # Note: Since we are iterating, 'next_low' is the Low of i+1 (tomorrow)
                # If we are at the last day, we can't fill tomorrow
                if i < len(df) - 1:
                    limit_price = price * (1 - limit_pct)
                    next_low = row['next_low']
                    
                    if next_low <= limit_price:
                        # Filled!
                        entry_price = limit_price
                        # We fill "tomorrow", but for simplicity in the log we keep signal date
                        # or we could shift. Let's keep signal date but adjust price.
                        filled = True
                    else:
                        filled = False # Missed trade
                else:
                    filled = False
            
            if filled:
                shares = capital / entry_price
                entry_capital = capital
                position = 1
                trades.append({
                    'type': 'BUY',
                    'entry_date': fill_date,
                    'entry_price': entry_price,
                    'shares': shares,
                    'entry_capital': entry_capital
                })
            
        elif signal == -1 and position == 1:  # SELL
            exit_value = shares * price
            profit = exit_value - trades[-1]['entry_capital']
            profit_pct = profit / trades[-1]['entry_capital'] * 100
            trades[-1].update({
                'exit_date': date,
                'exit_price': price,
                'exit_value': exit_value,
                'profit': profit,
                'profit_pct': profit_pct,
                'status': 'closed'
            })
            capital = exit_value
            position = 0
            shares = 0
    
    # Handle open position
    if position == 1:
        final_price = df['close'].iloc[-1]
        exit_value = shares * final_price
        profit = exit_value - trades[-1]['entry_capital']
        profit_pct = profit / trades[-1]['entry_capital'] * 100
        
        trades[-1].update({
            'exit_date': df.index[-1],
            'exit_price': final_price,
            'exit_value': exit_value,
            'profit': profit,
            'profit_pct': profit_pct,
            'status': 'open'
        })
        capital = exit_value
        
    # Calculate metrics
    total_return = (capital - initial_capital) / initial_capital * 100
    
    trades_count = len(trades)
    win_rate = 0
    if trades:
        profits = [t.get('profit_pct', 0) for t in trades]
        wins = [p for p in profits if p > 0]
        win_rate = len(wins) / len(profits) * 100
        avg_trade_return = sum(profits) / len(profits)
    else:
        avg_trade_return = 0
        
    # Create equity curve df
    equity_df = pd.DataFrame(equity_curve)
    
    # Calculate Max Drawdown
    if not equity_df.empty:
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
    else:
        max_drawdown = 0
    
    # Calculate Buy & Hold
    initial_close = df['close'].iloc[0]
    final_close = df['close'].iloc[-1]
    bh_return = (final_close - initial_close) / initial_close * 100
    
    return {
        'total_return': total_return,
        'final_capital': capital,
        'final_equity': capital, # Alias for UI
        'num_trades': trades_count,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'avg_trade_return': avg_trade_return,
        'trades': trades,
        'equity_curve': equity_df,
        'buy_hold_return': bh_return,
        'initial_capital': initial_capital
    }


def calculate_theoretical_return(data: pd.DataFrame, peak_dates: list, valley_dates: list, 
                                  initial_capital: float = 100000) -> dict:
    """
    Calculate theoretical perfect trading return - buying at every valley and selling at every peak.
    
    Returns dict with:
        - total_return: Percentage return from perfect trading
        - num_trades: Number of completed round-trip trades
        - avg_trade_return: Average return per trade
        - trades: List of individual trade details
    """
    # Create events list with buy at valleys, sell at peaks
    events = []
    for d in valley_dates:
        if d in data.index:
            events.append(('buy', d, data.loc[d, 'close']))
    for d in peak_dates:
        if d in data.index:
            events.append(('sell', d, data.loc[d, 'close']))
    
    # Sort by date
    events.sort(key=lambda x: x[1])
    
    capital = initial_capital
    position = 0
    shares = 0
    entry_price = 0
    entry_date = None
    trades = []
    
    for action, date, price in events:
        if action == 'buy' and position == 0:
            # Enter long position at valley
            shares = capital / price
            entry_price = price
            entry_date = date
            position = 1
            
        elif action == 'sell' and position == 1:
            # Exit at peak
            exit_value = shares * price
            trade_return = (price - entry_price) / entry_price * 100
            trades.append({
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': date,
                'exit_price': price,
                'return_pct': trade_return
            })
            capital = exit_value
            position = 0
            shares = 0
    
    # Close any open position at last price
    if position == 1:
        final_price = data['close'].iloc[-1]
        exit_value = shares * final_price
        trade_return = (final_price - entry_price) / entry_price * 100
        trades.append({
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': data.index[-1],
            'exit_price': final_price,
            'return_pct': trade_return,
            'status': 'open'
        })
        capital = exit_value
    
    total_return = (capital - initial_capital) / initial_capital * 100
    avg_trade_return = np.mean([t['return_pct'] for t in trades]) if trades else 0
    
    return {
        'total_return': total_return,
        'final_capital': capital,
        'num_trades': len(trades),
        'avg_trade_return': avg_trade_return,
        'trades': trades
    }


def run_comprehensive_analysis(analysis_dir: str = "analysis") -> str:
    """
    Run deep analysis on the latest output file.
    Returns the report as a string.
    """
    if not os.path.exists(analysis_dir):
        return "No analysis directory found."
    
    # Find latest file
    files = [f for f in os.listdir(analysis_dir) if f.endswith('.csv')]
    if not files:
        return "No analysis files found."
    
    latest_file = sorted(files)[-1]
    filepath = os.path.join(analysis_dir, latest_file)
    
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    except Exception as e:
        return f"Error reading file: {e}"
    
    report = []
    report.append("="*60)
    report.append(f"🔍 COMPREHENSIVE ANALYSIS REPORT")
    report.append(f"File: {latest_file}")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*60)
    
    # 1. Signal Distribution
    report.append("\n📊 SIGNAL DISTRIBUTION")
    sig_counts = df['signal'].value_counts()
    total_sigs = len(df)
    for sig, count in sig_counts.items():
        label = "BUY (1)" if sig == 1 else "SELL (-1)" if sig == -1 else "HOLD (0)"
        report.append(f"   {label}: {count} ({count/total_sigs*100:.1f}%)")
    
    # 2. Confidence Analysis
    if 'confidence' in df.columns:
        report.append("\n🧠 MODEL CONFIDENCE")
        avg_conf = df['confidence'].mean()
        report.append(f"   Average Confidence: {avg_conf:.1%}")
        
        # Confidence by signal type
        for sig in [1, -1, 0]:
            mask = df['signal'] == sig
            if mask.any():
                sig_conf = df.loc[mask, 'confidence'].mean()
                label = "BUY" if sig == 1 else "SELL" if sig == -1 else "HOLD"
                report.append(f"   Avg {label} Confidence: {sig_conf:.1%}")
    
    # 3. Signal Quality (vs Future Returns)
    report.append("\n🎯 SIGNAL QUALITY (Forward Returns)")
    
    # Calculate forward returns if not present (simple close-to-close)
    for period in [1, 3, 5, 10]:
        df[f'fwd_ret_{period}d'] = df['close'].shift(-period) / df['close'] - 1
    
    for sig in [1, -1]:
        mask = df['signal'] == sig
        if not mask.any():
            continue
            
        label = "BUY" if sig == 1 else "SELL"
        report.append(f"\n   {label} Signals ({mask.sum()}):")
        
        for period in [1, 3, 5, 10]:
            avg_ret = df.loc[mask, f'fwd_ret_{period}d'].mean() * 100
            win_rate = (df.loc[mask, f'fwd_ret_{period}d'] > 0).mean() * 100 if sig == 1 else (df.loc[mask, f'fwd_ret_{period}d'] < 0).mean() * 100
            report.append(f"      {period}d Forward Return: {avg_ret:+.2f}% | Win Rate: {win_rate:.1f}%")
    
    # 4. Missed Opportunities (Big Moves not captured)
    report.append("\n📉 MISSED OPPORTUNITIES")
    # Find days with > 3% move where signal was HOLD
    big_move_mask = (df['close'].pct_change().abs() > 0.03) & (df['signal'] == 0)
    missed_days = big_move_mask.sum()
    report.append(f"   Big moves (>3%) missed: {missed_days}")
    
    if missed_days > 0:
        report.append("   Top 3 Missed Days:")
        missed = df[big_move_mask].copy()
        missed['abs_ret'] = missed['close'].pct_change().abs()
        top_missed = missed.sort_values('abs_ret', ascending=False).head(3)
        for date, row in top_missed.iterrows():
            pct = row['close'] / df['close'].shift(1).loc[date] - 1
            report.append(f"      {date.strftime('%Y-%m-%d')}: {pct*100:+.1f}% (Conf: {row.get('confidence', 0):.1%})")
    
    # 5. Composite Indicator Correlation
    if 'composite_indicator' in df.columns:
        report.append("\n🔗 COMPOSITE INDICATOR CORRELATION")
        corr = df['composite_indicator'].corr(df['close'].pct_change())
        report.append(f"   Correlation with Price Returns: {corr:+.3f}")
        
        # Avg composite value per signal
        report.append("   Avg Composite Value per Signal:")
        for sig in [1, -1, 0]:
            mask = df['signal'] == sig
            if mask.any():
                avg_val = df.loc[mask, 'composite_indicator'].mean()
                label = "BUY" if sig == 1 else "SELL" if sig == -1 else "HOLD"
                report.append(f"      {label}: {avg_val:+.3f}")

    # 6. Market Regime Analysis (Bull vs Bear)
    # Define Bull as Price > SMA200 (approx 200 days)
    # If we don't have SMA200, we can calculate it or use a proxy
    report.append("\n🐂 MARKET REGIME ANALYSIS (Bull vs Bear)")
    try:
        # Calculate SMA 200 for context
        sma200 = df['close'].rolling(200).mean()
        is_bull = df['close'] > sma200
        
        for regime_name, regime_mask in [("BULL (Price > SMA200)", is_bull), ("BEAR (Price < SMA200)", ~is_bull)]:
            if regime_mask.sum() < 10:
                continue
                
            regime_df = df[regime_mask]
            buys = regime_df[regime_df['signal'] == 1]
            sells = regime_df[regime_df['signal'] == -1]
            
            if len(buys) > 0:
                buy_ret = buys['fwd_ret_5d'].mean() * 100
                report.append(f"   {regime_name}:")
                report.append(f"      BUY Signals: {len(buys)} | Avg 5d Ret: {buy_ret:+.2f}%")
            
            if len(sells) > 0:
                sell_ret = sells['fwd_ret_5d'].mean() * 100
                report.append(f"      SELL Signals: {len(sells)} | Avg 5d Ret: {sell_ret:+.2f}%")
                
    except Exception as e:
        report.append(f"   Could not calculate regime metrics: {str(e)}")

    # 7. Confidence Calibration (Buckets)
    if 'confidence' in df.columns:
        report.append("\n🎚️ CONFIDENCE CALIBRATION (BUY Signals)")
        buy_df = df[df['signal'] == 1].copy()
        if not buy_df.empty:
            buy_df['conf_bucket'] = pd.cut(buy_df['confidence'], bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            grouped = buy_df.groupby('conf_bucket')['fwd_ret_5d'].agg(['count', 'mean'])
            
            for bucket, row in grouped.iterrows():
                if row['count'] > 0:
                    report.append(f"   {bucket}: {int(row['count'])} trades | Avg 5d Ret: {row['mean']*100:+.2f}%")

    # 8. Composite Indicator Analysis (Validation of User Hypothesis)
    if 'composite_indicator' in df.columns:
        report.append("\n🧬 COMPOSITE INDICATOR EFFICIENCY (Manual Strategy Proxy)")
        report.append("   Hypothesis: Low Composite (< -0.6) = Bottom, High (> 0.6) = Top")
        
        # Create bins
        try:
            # pd.cut might fail if data is weird, wrap in try
            df['comp_bin'] = pd.cut(df['composite_indicator'], bins=[-1.5, -0.6, -0.2, 0.2, 0.6, 1.5], 
                                  labels=["Oversold (<-0.6)", "Bearish", "Neutral", "Bullish", "Overbought (>0.6)"])
            
            grouped = df.groupby('comp_bin')['fwd_ret_5d'].agg(['count', 'mean'])
            
            for interval, row in grouped.iterrows():
                if row['count'] > 0:
                    report.append(f"   {interval}: {int(row['count']):4d} days | Avg 5d Ret: {row['mean']*100:+.2f}%")
                    
            # Check win rate for Oversold
            low_comp = df[df['composite_indicator'] < -0.6]
            if not low_comp.empty:
                win_rate = (low_comp['fwd_ret_5d'] > 0).mean()
                report.append(f"\n   OVERSOLD SIGNAL QUALITY (Comp < -0.6):")
                report.append(f"   Win Rate (5d > 0): {win_rate*100:.1f}%")
                
        except Exception as e:
            report.append(f"   Could not calculate composite stats: {str(e)}")

    return "\n".join(report)


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.title("📈 Peak/Valley ML v2 - Enhanced")
st.caption("Clean architecture with SMOTE, Optuna, and comprehensive analysis")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker Symbol", value=st.session_state.v2_ticker)
    period = st.selectbox("Training Period", ['1y', '2y', '3y', '5y'], index=3)
    
    st.markdown("---")
    st.subheader("🎯 Training Options")
    model_type = st.selectbox("Model Type", ['xgboost', 'random_forest'])
    use_smote = st.checkbox("Use SMOTE Balancing", value=True)
    use_optuna = st.checkbox("Use Optuna Optimization", value=True)
    n_trials = st.slider("Optuna Trials", 10, 100, 30) if use_optuna else 10
    
    # Optimization metric selection
    optimize_metric = st.selectbox("Optimize For", 
        ['f1_weighted', 'accuracy', 'f1_macro'],
        help="f1_weighted: Best for imbalanced classes, accuracy: Overall correctness, f1_macro: Equal weight to all classes")
    
    n_cv_splits = st.slider("CV Folds", 3, 10, 5, 
                           help="Number of TimeSeriesSplit folds for cross-validation")
    
    # Advanced Model Config
    with st.expander("⚙️ Advanced Model Config"):
        st.caption("Fine-tune how aggressive the model is")
        class_weight_ratio = st.slider("Class Weight Ratio", 1.0, 10.0, 1.0, 0.5,
                                      help="Higher = penalize missing BUY/SELL signals more heavily. 1.0 = Equal weights.")
        decision_threshold = st.slider("Decision Threshold", 0.3, 0.9, 0.5, 0.05,
                                      help="Lower = more aggressive signals. Higher = higher confidence required.")
    
    detection_order = st.slider("Peak/Valley Sensitivity", 3, 10, 5,
                               help="Lower = more sensitive")
    
    st.markdown("---")
    st.subheader("📊 Backtest Period")
    backtest_mode = st.selectbox("Backtest Mode", 
        ["Full Training Period", "Test Period Only", "Recent Days", "Custom Date Range"],
        help="Select which period to backtest over")
    
    # Default values
    prod_days = 180
    custom_start = datetime.now() - timedelta(days=365)
    custom_end = datetime.now()
    
    if backtest_mode == "Recent Days":
        prod_days = st.slider("Days", 30, 365, 180)
    elif backtest_mode == "Custom Date Range":
        col1, col2 = st.columns(2)
        with col1:
            custom_start = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        with col2:
            custom_end = st.date_input("End Date", value=datetime.now())

# Main tabs
tab1, tab2, tab3 = st.tabs(["🎯 Train Model", "🚀 Production", "📊 Analysis"])

# =============================================================================
# TAB 1: TRAINING
# =============================================================================
with tab1:
    st.header("Model Training")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Ticker:** {ticker}")
    with col2:
        st.info(f"**Period:** {period}")
    with col3:
        st.info(f"**Model:** {model_type}")
    
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        progress = st.progress(0, text="Starting...")
        
        # Load data
        progress.progress(10, text="Loading price data...")
        data = load_price_data(ticker, period)
        st.session_state.v2_data = data
        st.session_state.v2_ticker = ticker
        
        # Detect peaks/valleys
        progress.progress(20, text="Detecting peaks and valleys...")
        labels, detection_info = detect_peaks_valleys(data, order=detection_order)
        st.session_state.v2_labels = labels
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", len(data))
        with col2:
            st.metric("Peaks Detected", detection_info['num_peaks'])
        with col3:
            st.metric("Valleys Detected", detection_info['num_valleys'])
        
        # Calculate theoretical return (perfect trading at every peak/valley)
        theoretical = calculate_theoretical_return(data, detection_info['peak_dates'], 
                                                   detection_info['valley_dates'])
        st.session_state.v2_theoretical = theoretical  # Store for efficiency calculation
        
        # Display theoretical performance
        st.markdown("### 🎯 Theoretical Maximum (Perfect Trading)")
        st.caption("*Compounded returns from buying at every valley and selling at every peak with 100% capital*")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Perfect Return", f"{theoretical['total_return']:,.1f}%")
        with col2:
            st.metric("Perfect Trades", theoretical['num_trades'])
        with col3:
            st.metric("Avg Trade Return", f"{theoretical['avg_trade_return']:.1f}%")
        with col4:
            # Buy and hold for comparison
            bh_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100
            st.metric("Buy & Hold", f"{bh_return:.1f}%")
        
        # Generate features
        progress.progress(30, text="Generating features...")
        features = generate_features(data)
        st.session_state.v2_features = features
        st.info(f"Generated **{len(features.columns)}** features")
        
        # Label distribution
        label_counts = labels.value_counts()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BUY Labels", label_counts.get(1, 0))
        with col2:
            st.metric("HOLD Labels", label_counts.get(0, 0))
        with col3:
            st.metric("SELL Labels", label_counts.get(-1, 0))
        
        # Train model
        progress.progress(40, text=f"Training {model_type} with {'Optuna' if use_optuna else 'defaults'}...")
        st.info(f"🎯 Optimizing for: **{optimize_metric}** | CV Folds: **{n_cv_splits}** (TimeSeriesSplit)")
        
        def update_progress(current, total):
            pct = 40 + int(50 * current / total)
            progress.progress(pct, text=f"Optuna trial {current}/{total}...")
        
        model_data = train_model_with_optuna(
            features, labels, model_type, use_smote,
            n_trials=n_trials if use_optuna else 1,
            progress_callback=None, # Console only now
            optimize_metric=optimize_metric,
            n_cv_splits=n_cv_splits,
            class_weight_ratio=class_weight_ratio
        )
        
        st.session_state.v2_model = model_data
        st.session_state.v2_trained = True
        
        # Show results
        progress.progress(95, text="Saving model...")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{model_data['accuracy']:.1%}")
        with col2:
            st.metric("F1 Score", f"{model_data['f1_score']:.1%}")
        with col3:
            st.metric("CV Score", f"{model_data['cv_score']:.1%}")
        with col4:
            st.metric("Features", len(model_data['feature_names']))
        
        # Best params
        with st.expander("🔧 Best Hyperparameters"):
            st.json(model_data['best_params'])
        
        # Save model
        filepath = save_model(model_data, ticker)
        st.session_state.v2_model_path = filepath
        
        progress.progress(100, text="Complete!")
        st.success(f"✅ Model saved: `{filepath}`")
        st.balloons()

# =============================================================================
# TAB 2: PRODUCTION
# =============================================================================
with tab2:
    st.header("Production Trading")
    
    # Model selection
    model_dir = "saved_models_v2"
    if os.path.exists(model_dir):
        model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.joblib')], reverse=True)
    else:
        model_files = []
    
    if not model_files:
        st.warning("⚠️ No trained models found. Please train a model first.")
        st.stop()
    
    selected_model = st.selectbox("Select Model", model_files)
    model_path = os.path.join(model_dir, selected_model)
    
    # Show selected backtest mode info
    st.info(f"📅 **Backtest Mode:** {backtest_mode}")
    
    # Execution Settings
    limit_entry_pct = st.slider("Entry Limit Offset %", 0.0, 3.0, 0.0, 0.1, 
                               help="Try to buy lower than signal price (e.g. 1.0% lower). 0 = Market Order.") / 100
    
    if st.button("📊 Generate Signals & Backtest", type="primary", use_container_width=True):
        with st.spinner("Loading model and generating signals..."):
            # Load model
            model_data = load_model(model_path)
            model_ticker = model_data.get('ticker', ticker)
            
            # Determine date range based on backtest mode
            import yfinance as yf
            
            if backtest_mode == "Full Training Period":
                # Use the same period as training (e.g., 5y)
                data = yf.Ticker(model_ticker).history(period=period)
                period_label = f"Full {period} Training Period"
                
            elif backtest_mode == "Test Period Only":
                # Load full data, then use only the test portion (last 20%)
                data = yf.Ticker(model_ticker).history(period=period)
                split_idx = int(len(data) * 0.8)
                data = data.iloc[split_idx:]
                period_label = "Test Period (Last 20%)"
                
            elif backtest_mode == "Recent Days":
                end_date = datetime.now()
                start_date = end_date - timedelta(days=prod_days + 100)
                data = yf.Ticker(model_ticker).history(start=start_date, end=end_date)
                period_label = f"Recent {prod_days} Days"
                
            else:  # Custom Date Range
                data = yf.Ticker(model_ticker).history(start=custom_start, end=custom_end)
                period_label = f"Custom: {custom_start} to {custom_end}"
            
            data.index = pd.to_datetime(data.index).tz_localize(None)
            data.columns = [c.lower() for c in data.columns]
            data = data[['open', 'high', 'low', 'close', 'volume']]
            
            # Generate features
            features = generate_features(data)
            
            # Generate signals with custom threshold
            signals, prob_df = generate_signals(model_data, features, threshold=decision_threshold)
            
            # Calculate Composite Technical Indicator (Average of normalized oscillators)
            # Use features already generated
            oscillators = pd.DataFrame(index=features.index)
            
            # RSI (normalized to -1 to 1)
            if 'rsi_14' in features.columns:
                oscillators['rsi_norm'] = (features['rsi_14'] - 50) / 50
            
            # Williams %R (already -100 to 0, normalize to -1 to 1)
            if 'willr_14' in features.columns:
                oscillators['willr_norm'] = (features['willr_14'] + 50) / 50
            
            # CCI (normalize by dividing by 100, clip to -1 to 1)
            if 'cci_14' in features.columns:
                oscillators['cci_norm'] = (features['cci_14'] / 100).clip(-1, 1)
            
            # ROC (normalize, soft clip)
            if 'roc_10' in features.columns:
                oscillators['roc_norm'] = (features['roc_10'] / 5).clip(-1, 1)
                
            # Calculate composite
            if not oscillators.empty:
                composite_indicator = oscillators.mean(axis=1)
            else:
                composite_indicator = pd.Series(0, index=features.index)
            
            # Store composite and confidence in session state for plotting
            st.session_state.v2_composite = composite_indicator
            st.session_state.v2_probs = prob_df
            
            # Save detailed analysis output
            analysis_dir = "analysis"
            os.makedirs(analysis_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save signals and probs
            analysis_df = pd.concat([
                data[['open', 'high', 'low', 'close', 'volume']],
                signals.rename('signal'),
                prob_df,
                composite_indicator.rename('composite_indicator')
            ], axis=1)
            
            analysis_path = os.path.join(analysis_dir, f"analysis_{ticker}_{timestamp}.csv")
            analysis_df.to_csv(analysis_path)
            
            # For "Recent Days" mode, trim to exact days requested
            if backtest_mode == "Recent Days":
                signals = signals.iloc[-prod_days:]
                data = data.loc[signals.index]
                prob_df = prob_df.loc[signals.index]
                composite_indicator = composite_indicator.loc[signals.index]
            else:
                # Align data with signals (features drop some rows due to NaN)
                data = data.loc[signals.index]
                prob_df = prob_df.loc[signals.index]
                composite_indicator = composite_indicator.loc[signals.index]
            
            # Run backtest
            backtest = run_backtest(data, signals, limit_pct=limit_entry_pct)
            st.session_state.v2_backtest = backtest
            
            # Calculate theoretical maximum for this period
            labels_for_period, detection_info = detect_peaks_valleys(data, order=detection_order)
            theoretical_for_period = calculate_theoretical_return(
                data, detection_info['peak_dates'], detection_info['valley_dates']
            )
            backtest['theoretical_return'] = theoretical_for_period['total_return']
            backtest['theoretical_trades'] = theoretical_for_period['num_trades']
            
            # Calculate efficiency (how much of theoretical max was captured)
            if theoretical_for_period['total_return'] > 0:
                efficiency = (backtest['total_return'] / theoretical_for_period['total_return']) * 100
            else:
                efficiency = 0
            backtest['efficiency'] = efficiency
            
            # Store period info for display
            backtest['period_label'] = period_label
            backtest['start_date'] = data.index[0]
            backtest['end_date'] = data.index[-1]
            backtest['total_days'] = len(data)
        
        # Period info
        st.success(f"📅 **{backtest['period_label']}** | {backtest['start_date'].strftime('%Y-%m-%d')} to {backtest['end_date'].strftime('%Y-%m-%d')} ({backtest['total_days']} trading days)")
        
        # Efficiency banner
        efficiency = backtest.get('efficiency', 0)
        theoretical_ret = backtest.get('theoretical_return', 0)
        if efficiency > 0:
            if efficiency >= 50:
                st.success(f"🎯 **Efficiency: {efficiency:.1f}%** of theoretical maximum ({theoretical_ret:,.1f}% perfect return)")
            elif efficiency >= 25:
                st.warning(f"🎯 **Efficiency: {efficiency:.1f}%** of theoretical maximum ({theoretical_ret:,.1f}% perfect return)")
            else:
                st.error(f"🎯 **Efficiency: {efficiency:.1f}%** of theoretical maximum ({theoretical_ret:,.1f}% perfect return)")
        
        # Performance metrics
        st.subheader("📈 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delta = backtest['total_return'] - backtest['buy_hold_return']
            st.metric("ML Strategy Return", f"{backtest['total_return']:.1f}%", 
                     delta=f"{delta:+.1f}% vs B&H")
        with col2:
            st.metric("Buy & Hold Return", f"{backtest['buy_hold_return']:.1f}%")
        with col3:
            st.metric("Win Rate", f"{backtest['win_rate']:.1f}%")
        with col4:
            st.metric("Max Drawdown", f"{backtest['max_drawdown']:.1f}%")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Trades", backtest['num_trades'])
        with col2:
            st.metric("Final Equity", f"${backtest['final_equity']:,.0f}")
        with col3:
            signal_counts = signals.value_counts()
            st.metric("Signals", f"{signal_counts.get(1, 0)} BUY / {signal_counts.get(-1, 0)} SELL")
        with col4:
            # Annualized return
            years = backtest['total_days'] / 252
            if years > 0:
                annualized = ((1 + backtest['total_return']/100) ** (1/years) - 1) * 100
                st.metric("Annualized Return", f"{annualized:.1f}%")
        
        # Candlestick chart with signals
        st.subheader("📊 Price Chart with Signals")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.7, 0.3],
                           subplot_titles=('Price & Signals', 'Equity Curve'))
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ), row=1, col=1)
        
        # Buy signals
        buy_mask = signals == 1
        if buy_mask.any():
            buy_dates = signals[buy_mask].index
            buy_prices = data.loc[buy_dates, 'low'] * 0.98
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='lime', 
                           line=dict(width=2, color='darkgreen')),
                name='BUY Signal'
            ), row=1, col=1)
        
        # Sell signals
        sell_mask = signals == -1
        if sell_mask.any():
            sell_dates = signals[sell_mask].index
            sell_prices = data.loc[sell_dates, 'high'] * 1.02
            fig.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                marker=dict(symbol='triangle-down', size=15, color='red',
                           line=dict(width=2, color='darkred')),
                name='SELL Signal'
            ), row=1, col=1)
        
        # Trade markers (actual executions)
        for trade in backtest['trades']:
            # Entry
            fig.add_trace(go.Scatter(
                x=[trade['entry_date']],
                y=[trade['entry_price']],
                mode='markers',
                marker=dict(symbol='circle', size=12, color='blue',
                           line=dict(width=2, color='white')),
                name='Entry',
                showlegend=False
            ), row=1, col=1)
            
            # Exit
            if 'exit_date' in trade:
                color = 'green' if trade.get('profit', 0) > 0 else 'red'
                fig.add_trace(go.Scatter(
                    x=[trade['exit_date']],
                    y=[trade['exit_price']],
                    mode='markers',
                    marker=dict(symbol='x', size=12, color=color,
                               line=dict(width=2, color='white')),
                    name='Exit',
                    showlegend=False
                ), row=1, col=1)
        
        # Equity curve
        equity_df = backtest['equity_curve']
        fig.add_trace(go.Scatter(
            x=equity_df['date'],
            y=equity_df['equity'],
            mode='lines',
            name='ML Strategy',
            line=dict(color='blue', width=2)
        ), row=2, col=1)
        
        # Buy and hold equity
        initial = backtest['initial_capital']
        bh_equity = initial * (data['close'] / data['close'].iloc[0])
        fig.add_trace(go.Scatter(
            x=data.index,
            y=bh_equity,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='gray', width=1, dash='dash')
        ), row=2, col=1)
        
        fig.update_layout(
            title=f"{model_ticker} - ML Trading Signals ({selected_model})",
            height=800,
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # =====================================================================
        # SECOND CHART: Last 180 Days Detail View
        # =====================================================================
        st.subheader("📊 Last 180 Days - Detailed View")
        
        # Get last 180 days of data
        days_to_show = 180
        if len(data) > days_to_show:
            recent_data = data.iloc[-days_to_show:]
            recent_signals = signals.iloc[-days_to_show:]
        else:
            recent_data = data
            recent_signals = signals
        
        # Filter trades to recent period
        recent_trades = [t for t in backtest['trades'] 
                        if t['entry_date'] >= recent_data.index[0]]
        
        # Create recent chart with 3 subplots (Price, Equity, Technicals)
        fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=('Recent Price Action & Signals', 'Recent Equity', 'Technicals (Confidence & Composite)'))
        
        # Candlestick for recent period
        fig2.add_trace(go.Candlestick(
            x=recent_data.index,
            open=recent_data['open'],
            high=recent_data['high'],
            low=recent_data['low'],
            close=recent_data['close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ), row=1, col=1)
        
        # Recent signals
        recent_buy_mask = recent_signals == 1
        recent_sell_mask = recent_signals == -1
        
        if recent_buy_mask.any():
            recent_buy_dates = recent_signals[recent_buy_mask].index
            recent_buy_prices = recent_data.loc[recent_buy_dates, 'low'] * 0.98
            fig2.add_trace(go.Scatter(
                x=recent_buy_dates,
                y=recent_buy_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', size=14, color='lime',
                           line=dict(width=2, color='darkgreen')),
                name='BUY Signal'
            ), row=1, col=1)
        
        if recent_sell_mask.any():
            recent_sell_dates = recent_signals[recent_sell_mask].index
            recent_sell_prices = recent_data.loc[recent_sell_dates, 'high'] * 1.02
            fig2.add_trace(go.Scatter(
                x=recent_sell_dates,
                y=recent_sell_prices,
                mode='markers',
                marker=dict(symbol='triangle-down', size=14, color='red',
                           line=dict(width=2, color='darkred')),
                name='SELL Signal'
            ), row=1, col=1)
        
        # Recent trade markers
        for trade in recent_trades:
            # Entry marker
            fig2.add_trace(go.Scatter(
                x=[trade['entry_date']],
                y=[trade['entry_price']],
                mode='markers+text',
                marker=dict(symbol='circle', size=12, color='blue',
                           line=dict(width=2, color='white')),
                text=['▶'],
                textposition='middle left',
                name='Entry',
                showlegend=False
            ), row=1, col=1)
            
            # Exit marker
            if 'exit_date' in trade and trade['exit_date'] >= recent_data.index[0]:
                profit_color = 'green' if trade.get('profit', 0) > 0 else 'red'
                fig2.add_trace(go.Scatter(
                    x=[trade['exit_date']],
                    y=[trade['exit_price']],
                    mode='markers',
                    marker=dict(symbol='x', size=12, color=profit_color,
                               line=dict(width=3, color='white')),
                    name='Exit',
                    showlegend=False
                ), row=1, col=1)
                
                # Draw line connecting entry to exit
                fig2.add_trace(go.Scatter(
                    x=[trade['entry_date'], trade['exit_date']],
                    y=[trade['entry_price'], trade['exit_price']],
                    mode='lines',
                    line=dict(color=profit_color, width=1, dash='dot'),
                    showlegend=False
                ), row=1, col=1)
        
        # Recent equity curve
        recent_equity = backtest['equity_curve'][backtest['equity_curve']['date'] >= recent_data.index[0]]
        if len(recent_equity) > 0:
            fig2.add_trace(go.Scatter(
                x=recent_equity['date'],
                y=recent_equity['equity'],
                mode='lines',
                name='ML Strategy',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,100,255,0.1)'
            ), row=2, col=1)
            
            # Buy and hold for recent period
            recent_bh_start = backtest['initial_capital'] * (recent_data['close'].iloc[0] / data['close'].iloc[0])
            recent_bh = recent_bh_start * (recent_data['close'] / recent_data['close'].iloc[0])
            fig2.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_bh,
                mode='lines',
                name='Buy & Hold',
                line=dict(color='gray', width=1, dash='dash')
            ), row=2, col=1)
        
        # ROW 3: TECHNICALS
        # 1. Model Confidence (Blue)
        # Use prob of predicted class * signal sign (-1 or 1)
        recent_probs = prob_df[prob_df.index >= recent_data.index[0]]
        recent_composite = composite_indicator[composite_indicator.index >= recent_data.index[0]]
        
        if not recent_probs.empty:
            # Confidence signal: +confidence for BUY, -confidence for SELL, 0 for HOLD
            conf_signal = recent_probs['predicted_signal'] * recent_probs['confidence']
            
            # Plot confidence line
            fig2.add_trace(go.Scatter(
                x=recent_probs.index,
                y=conf_signal,
                mode='lines',
                name='Model Confidence',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,0,255,0.1)'
            ), row=3, col=1)
        
        # 2. Composite Indicator (Orange)
        if not recent_composite.empty:
            fig2.add_trace(go.Scatter(
                x=recent_composite.index,
                y=recent_composite,
                mode='lines',
                name='Composite Tech',
                line=dict(color='orange', width=1, dash='dot')
            ), row=3, col=1)
        
        # Add threshold lines
        fig2.add_shape(type="line", x0=recent_data.index[0], x1=recent_data.index[-1], y0=0, y1=0,
                      line=dict(color="gray", width=1, dash="dot"), row=3, col=1)
        fig2.add_shape(type="line", x0=recent_data.index[0], x1=recent_data.index[-1], y0=decision_threshold, y1=decision_threshold,
                      line=dict(color="green", width=1, dash="dot"), row=3, col=1)
        fig2.add_shape(type="line", x0=recent_data.index[0], x1=recent_data.index[-1], y0=-decision_threshold, y1=-decision_threshold,
                      line=dict(color="red", width=1, dash="dot"), row=3, col=1)
        
        fig2.update_layout(
            title=f"Last {len(recent_data)} Trading Days - {model_ticker}",
            height=900,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
        )
        fig2.update_yaxes(title_text="Price", row=1, col=1)
        fig2.update_yaxes(title_text="Equity ($)", row=2, col=1)
        fig2.update_yaxes(title_text="Signal", row=3, col=1, range=[-1.1, 1.1])
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Recent period stats
        if len(recent_trades) > 0:
            recent_profits = [t.get('profit', 0) for t in recent_trades if 'profit' in t]
            recent_wins = sum(1 for p in recent_profits if p > 0)
            recent_total_profit = sum(recent_profits)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Recent Trades", len(recent_trades))
            with col2:
                st.metric("Recent Win Rate", f"{100*recent_wins/len(recent_trades):.0f}%" if recent_trades else "N/A")
            with col3:
                st.metric("Recent P/L", f"${recent_total_profit:,.0f}")
            with col4:
                # Recent signals count
                recent_buys = (recent_signals == 1).sum()
                recent_sells = (recent_signals == -1).sum()
                st.metric("Recent Signals", f"{recent_buys} BUY / {recent_sells} SELL")
        
        # Trade log
        st.subheader("📋 Trade Log")
        if backtest['trades']:
            trades_df = pd.DataFrame(backtest['trades'])
            display_cols = ['type', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 'profit', 'profit_pct', 'status']
            available_cols = [c for c in display_cols if c in trades_df.columns]
            
            # Format
            trades_display = trades_df[available_cols].copy()
            if 'entry_price' in trades_display.columns:
                trades_display['entry_price'] = trades_display['entry_price'].apply(lambda x: f"${x:,.2f}")
            if 'exit_price' in trades_display.columns:
                trades_display['exit_price'] = trades_display['exit_price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
            if 'profit' in trades_display.columns:
                trades_display['profit'] = trades_display['profit'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
            if 'profit_pct' in trades_display.columns:
                trades_display['profit_pct'] = trades_display['profit_pct'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
            
            st.dataframe(trades_display, use_container_width=True)
        else:
            st.info("No trades executed in this period")

# =============================================================================
# TAB 3: ANALYSIS
# =============================================================================
with tab3:
    st.header("Model Analysis")
    
    # Model selection - current session or saved models
    model_dir = "saved_models_v2"
    saved_models = []
    if os.path.exists(model_dir):
        saved_models = sorted([f for f in os.listdir(model_dir) if f.endswith('.joblib')], reverse=True)
    
    # Build selection options
    model_options = []
    if st.session_state.v2_trained and st.session_state.v2_model is not None:
        model_options.append("📍 Current Session Model")
    model_options.extend([f"💾 {f}" for f in saved_models])
    
    if not model_options:
        st.info("👈 Train a model or load a saved model to see analysis")
        st.stop()
    
    selected_analysis_model = st.selectbox("Select Model to Analyze", model_options, key="analysis_model_select")
    
    # Deep Analysis Button
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔬 Run Deep Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing latest results..."):
                report = run_comprehensive_analysis()
                st.session_state.v2_analysis_report = report
    with col2:
        if 'v2_analysis_report' in st.session_state:
            st.download_button("📥 Download Report", 
                             st.session_state.v2_analysis_report, 
                             file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    if 'v2_analysis_report' in st.session_state:
        with st.expander("📄 Comprehensive Analysis Report", expanded=True):
            st.text(st.session_state.v2_analysis_report)
    st.markdown("---")
    
    # Load the selected model data
    if selected_analysis_model == "📍 Current Session Model":
        model_data = st.session_state.v2_model
        model_source = "Current Session"
    else:
        # Load from file
        model_filename = selected_analysis_model.replace("💾 ", "")
        model_path = os.path.join(model_dir, model_filename)
        loaded_model = load_model(model_path)
        
        # Convert loaded model format to analysis format
        model_data = {
            'model': loaded_model['model'],
            'scaler': loaded_model['scaler'],
            'label_encoder': loaded_model.get('label_encoder'),
            'feature_names': loaded_model['feature_names'],
            'model_type': loaded_model['model_type'],
            'best_params': loaded_model.get('best_params', {}),
            'accuracy': loaded_model.get('metrics', {}).get('accuracy', 0),
            'f1_score': loaded_model.get('metrics', {}).get('f1_score', 0),
            'cv_score': loaded_model.get('metrics', {}).get('cv_score', 0),
            'train_size': 0,  # Not stored in saved model
            'test_size': 0,
            'label_distribution': {},
            'split_date': None,
            'use_smote': True,  # Assume true for saved models
            'feature_importance': None,
        }
        
        # Try to get feature importance from loaded model
        if hasattr(loaded_model['model'], 'feature_importances_'):
            model_data['feature_importance'] = pd.DataFrame({
                'feature': loaded_model['feature_names'],
                'importance': loaded_model['model'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        model_source = model_filename
    
    st.success(f"📊 Analyzing: **{model_source}**")
    
    # Model metrics
    st.subheader("📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        acc = model_data.get('accuracy', 0)
        st.metric("Accuracy", f"{acc:.1%}" if acc else "N/A")
    with col2:
        f1 = model_data.get('f1_score', 0)
        st.metric("F1 Score", f"{f1:.1%}" if f1 else "N/A")
    with col3:
        cv = model_data.get('cv_score', 0)
        st.metric("CV Score", f"{cv:.1%}" if cv else "N/A")
    with col4:
        train = model_data.get('train_size', 0)
        test = model_data.get('test_size', 0)
        st.metric("Train/Test", f"{train}/{test}" if train else "N/A")
    
    # Label distribution (only for current session)
    dist = model_data.get('label_distribution', {})
    if dist:
        st.subheader("📈 Label Distribution")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BUY (1)", dist.get(1, 0))
        with col2:
            st.metric("HOLD (0)", dist.get(0, 0))
        with col3:
            st.metric("SELL (-1)", dist.get(-1, 0))
    
    # Training settings
    st.subheader("⚙️ Model Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Model Type:** {model_data.get('model_type', 'Unknown')}")
        st.info(f"**SMOTE:** {'Enabled' if model_data.get('use_smote') else 'Disabled'}")
    with col2:
        st.info(f"**Features:** {len(model_data.get('feature_names', []))}")
        split_date = model_data.get('split_date')
        st.info(f"**Split Date:** {split_date.strftime('%Y-%m-%d') if split_date else 'N/A'}")
    
    # Feature list
    with st.expander("📋 All Features Used"):
        feature_names = model_data.get('feature_names', [])
        if feature_names:
            # Display in columns
            cols = st.columns(3)
            for i, feat in enumerate(feature_names):
                cols[i % 3].write(f"• {feat}")
        else:
            st.write("No feature list available")
    
    # Feature importance
    if model_data.get('feature_importance') is not None:
        st.subheader("🎯 Top 20 Features")
        
        importance_df = model_data['feature_importance'].head(20)
        
        import plotly.express as px
        fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                    title="Feature Importance", color='importance',
                    color_continuous_scale='viridis')
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Best parameters
    best_params = model_data.get('best_params', {})
    if best_params:
        st.subheader("🔧 Optimized Hyperparameters")
        st.json(best_params)
    
    # Model metadata (for saved models)
    if selected_analysis_model != "📍 Current Session Model":
        st.subheader("📁 Model Metadata")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Ticker:** {loaded_model.get('ticker', 'Unknown')}")
            st.info(f"**Saved:** {loaded_model.get('timestamp', 'Unknown')}")
        with col2:
            notes = loaded_model.get('notes', '')
            st.info(f"**Notes:** {notes if notes else 'None'}")
