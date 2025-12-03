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
    
    # Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)
        df[f'close_lag_{lag}'] = df['close'].pct_change(lag)
    
    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
        df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
        df[f'rsi_mean_{window}'] = df['rsi_14'].rolling(window).mean()
    
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
                            n_trials: int = 20, progress_callback=None):
    """
    Train model with Optuna optimization and SMOTE balancing.
    """
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Align data
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]
    
    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE balancing
    if use_smote:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train.values
    
    # Label encoding for XGBoost
    label_encoder = None
    if model_type == 'xgboost':
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_balanced)
        y_test_encoded = label_encoder.transform(y_test)
    else:
        y_train_encoded = y_train_balanced
        y_test_encoded = y_test.values
    
    # Optuna optimization
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
                'verbosity': 0
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
                'n_jobs': -1
            }
            model = RandomForestClassifier(**params)
        
        # Time series CV
        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X_train_balanced, y_train_encoded, 
                                cv=tscv, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()
    
    # Run optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                  callbacks=[lambda study, trial: progress_callback(trial.number + 1, n_trials) if progress_callback else None])
    
    best_params = study.best_params
    
    # Train final model with best params
    if model_type == 'xgboost':
        from xgboost import XGBClassifier
        best_params.update({'random_state': 42, 'use_label_encoder': False, 
                           'eval_metric': 'mlogloss', 'verbosity': 0})
        model = XGBClassifier(**best_params)
    else:
        from sklearn.ensemble import RandomForestClassifier
        best_params.update({'random_state': 42, 'n_jobs': -1})
        model = RandomForestClassifier(**best_params)
    
    model.fit(X_train_balanced, y_train_encoded)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Decode predictions
    if label_encoder:
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        y_test_decoded = y_test.values
    else:
        y_pred_decoded = y_pred
        y_test_decoded = y_test.values
    
    # Metrics
    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    f1 = f1_score(y_test_decoded, y_pred_decoded, average='weighted')
    
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


def generate_signals(model_data: dict, features: pd.DataFrame) -> pd.Series:
    """Generate trading signals using the model"""
    # Match features
    required = model_data['feature_names']
    available = [f for f in required if f in features.columns]
    
    if len(available) < len(required) * 0.8:
        raise ValueError(f"Feature mismatch: need {len(required)}, have {len(available)}")
    
    # Fill missing features with 0
    X = features.reindex(columns=required, fill_value=0)
    
    # Scale
    X_scaled = model_data['scaler'].transform(X)
    
    # Predict
    raw_pred = model_data['model'].predict(X_scaled)
    
    # Decode
    if model_data.get('label_encoder'):
        predictions = model_data['label_encoder'].inverse_transform(raw_pred)
    else:
        predictions = raw_pred
    
    return pd.Series(predictions, index=features.index, name='signal')


def run_backtest(data: pd.DataFrame, signals: pd.Series, initial_capital: float = 100000) -> dict:
    """Run backtest with detailed trade tracking"""
    common_idx = data.index.intersection(signals.index)
    prices = data.loc[common_idx, 'close']
    sigs = signals.loc[common_idx]
    
    capital = initial_capital
    position = 0
    shares = 0
    trades = []
    equity_curve = []
    
    for i, (date, price) in enumerate(prices.items()):
        signal = sigs.iloc[i]
        
        # Track equity
        current_equity = shares * price if position == 1 else capital
        equity_curve.append({'date': date, 'equity': current_equity, 'price': price})
        
        if signal == 1 and position == 0:  # BUY
            shares = capital / price
            entry_capital = capital
            position = 1
            trades.append({
                'type': 'BUY',
                'entry_date': date,
                'entry_price': price,
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
        final_price = prices.iloc[-1]
        exit_value = shares * final_price
        profit = exit_value - trades[-1]['entry_capital']
        profit_pct = profit / trades[-1]['entry_capital'] * 100
        trades[-1].update({
            'exit_date': prices.index[-1],
            'exit_price': final_price,
            'exit_value': exit_value,
            'profit': profit,
            'profit_pct': profit_pct,
            'status': 'open'
        })
        capital = exit_value
    
    # Calculate metrics
    closed_trades = [t for t in trades if t.get('status') == 'closed']
    winning_trades = [t for t in closed_trades if t.get('profit', 0) > 0]
    
    total_return = (capital - initial_capital) / initial_capital * 100
    win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
    
    # Buy and hold comparison
    buy_hold_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
    
    # Max drawdown
    equity_df = pd.DataFrame(equity_curve)
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
    max_drawdown = equity_df['drawdown'].min()
    
    return {
        'trades': trades,
        'equity_curve': equity_df,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'win_rate': win_rate,
        'num_trades': len(closed_trades),
        'max_drawdown': max_drawdown,
        'final_equity': capital,
        'initial_capital': initial_capital,
    }


def calculate_theoretical_return(data: pd.DataFrame, peak_dates: list, valley_dates: list) -> float:
    """Calculate theoretical perfect trading return"""
    events = []
    for d in valley_dates:
        events.append(('buy', d))
    for d in peak_dates:
        events.append(('sell', d))
    events.sort(key=lambda x: x[1])
    
    capital = 100000
    position = 0
    shares = 0
    
    for action, date in events:
        if date not in data.index:
            continue
        price = data.loc[date, 'close']
        
        if action == 'buy' and position == 0:
            shares = capital / price
            position = 1
        elif action == 'sell' and position == 1:
            capital = shares * price
            position = 0
            shares = 0
    
    if position == 1:
        capital = shares * data['close'].iloc[-1]
    
    return (capital - 100000) / 100000 * 100


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
        
        # Calculate theoretical return
        theoretical = calculate_theoretical_return(data, detection_info['peak_dates'], 
                                                   detection_info['valley_dates'])
        st.success(f"🎯 Theoretical Perfect Trading Return: **{theoretical:.1f}%**")
        
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
        
        def update_progress(current, total):
            pct = 40 + int(50 * current / total)
            progress.progress(pct, text=f"Optuna trial {current}/{total}...")
        
        model_data = train_model_with_optuna(
            features, labels, model_type, use_smote,
            n_trials=n_trials if use_optuna else 1,
            progress_callback=update_progress if use_optuna else None
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
            
            # Generate signals
            signals = generate_signals(model_data, features)
            
            # For "Recent Days" mode, trim to exact days requested
            if backtest_mode == "Recent Days":
                signals = signals.iloc[-prod_days:]
                data = data.loc[signals.index]
            else:
                # Align data with signals (features drop some rows due to NaN)
                data = data.loc[signals.index]
            
            # Run backtest
            backtest = run_backtest(data, signals)
            st.session_state.v2_backtest = backtest
            
            # Store period info for display
            backtest['period_label'] = period_label
            backtest['start_date'] = data.index[0]
            backtest['end_date'] = data.index[-1]
            backtest['total_days'] = len(data)
        
        # Period info
        st.success(f"📅 **{backtest['period_label']}** | {backtest['start_date'].strftime('%Y-%m-%d')} to {backtest['end_date'].strftime('%Y-%m-%d')} ({backtest['total_days']} trading days)")
        
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
    
    if not st.session_state.v2_trained or st.session_state.v2_model is None:
        st.info("👈 Train a model first to see analysis")
        st.stop()
    
    model_data = st.session_state.v2_model
    
    # Model metrics
    st.subheader("📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{model_data['accuracy']:.1%}")
    with col2:
        st.metric("F1 Score", f"{model_data['f1_score']:.1%}")
    with col3:
        st.metric("CV Score", f"{model_data['cv_score']:.1%}")
    with col4:
        st.metric("Train/Test", f"{model_data['train_size']}/{model_data['test_size']}")
    
    # Label distribution
    st.subheader("📈 Label Distribution")
    dist = model_data['label_distribution']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("BUY (1)", dist.get(1, 0))
    with col2:
        st.metric("HOLD (0)", dist.get(0, 0))
    with col3:
        st.metric("SELL (-1)", dist.get(-1, 0))
    
    # Training settings
    st.subheader("⚙️ Training Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Model Type:** {model_data['model_type']}")
        st.info(f"**SMOTE:** {'Enabled' if model_data['use_smote'] else 'Disabled'}")
    with col2:
        st.info(f"**Features:** {len(model_data['feature_names'])}")
        st.info(f"**Split Date:** {model_data['split_date'].strftime('%Y-%m-%d')}")
    
    # Feature importance
    if model_data['feature_importance'] is not None:
        st.subheader("🎯 Top 20 Features")
        
        importance_df = model_data['feature_importance'].head(20)
        
        import plotly.express as px
        fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                    title="Feature Importance", color='importance',
                    color_continuous_scale='viridis')
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Best parameters
    st.subheader("🔧 Optimized Hyperparameters")
    st.json(model_data['best_params'])
