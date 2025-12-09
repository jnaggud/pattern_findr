import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
from scipy.signal import argrelextrema
import pandas_ta as ta

# Try to import DL Feature Extractor
try:
    from dl_feature_extractor import DLFeatureExtractor
except ImportError:
    DLFeatureExtractor = None

# =============================================================================
# DATA LOADING & PROCESSING
# =============================================================================

def load_price_data(ticker: str, period: str, interval: str = '1d') -> pd.DataFrame:
    """Load OHLCV data from yfinance"""
    import yfinance as yf
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data.columns = [c.lower() for c in data.columns]
    return data[['open', 'high', 'low', 'close', 'volume']]

def detect_peaks_valleys(data: pd.DataFrame, order: int = 5) -> tuple:
    """
    Detect peaks and valleys with PREDICTIVE labeling.
    Returns labels and detection info.
    """
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

def calculate_theoretical_return(data, peak_dates, valley_dates, initial_capital=10000.0):
    """Calculate theoretical maximum return if trading perfectly"""
    events = []
    for d in peak_dates: events.append((d, 'SELL'))
    for d in valley_dates: events.append((d, 'BUY'))
    
    events.sort(key=lambda x: x[0])
    
    capital = initial_capital
    position = 0
    trades = 0
    
    for date, action in events:
        price = data.loc[date, 'close']
        
        if action == 'BUY' and position == 0:
            position = capital / price
            capital = 0
            trades += 1
        elif action == 'SELL' and position > 0:
            capital = position * price
            position = 0
            trades += 1
            
    # Close final position
    if position > 0:
        capital = position * data.iloc[-1]['close']
        
    return {
        'total_return': (capital - initial_capital) / initial_capital * 100,
        'final_capital': capital,
        'num_trades': trades,
        'avg_trade_return': ((capital/initial_capital)**(1/max(1, trades)) - 1) * 100
    }

def calculate_metrics_from_signals(data: pd.DataFrame, signals: pd.Series, initial_capital=10000.0) -> dict:
    """
    Calculate full suite of metrics from signals series.
    Returns dict with metrics and equity curve.
    """
    capital = initial_capital
    position = 0
    trades_list = [] 
    equity_vals = []
    
    entry_price = 0
    entry_date = None
    
    # Align signals to data
    # Assuming same index
    
    for i in range(len(data)):
        price = data['close'].iloc[i]
        date = data.index[i]
        
        # Check signal (Signal is usually generated at CLOSE of bar, to be executed NEXT OPEN? 
        # For simplicity in this estimate, we execute at CLOSE of signal bar)
        sig = signals.iloc[i] if i < len(signals) else 0
        
        # Execute Trade
        if sig == 1 and position == 0: # Buy
            position = capital / price
            capital = 0
            entry_price = price
            entry_date = date
            
        elif sig == -1 and position > 0: # Sell
            revenue = position * price
            pnl = (revenue - (position * entry_price))
            pnl_pct = (price - entry_price) / entry_price
            
            capital = revenue
            position = 0
            
            trades_list.append({
                'entry_date': entry_date,
                'exit_date': date,
                'entry_price': entry_price,
                'exit_price': price,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
        
        # Update Equity
        curr_val = capital
        if position > 0:
            curr_val = position * price
            
        equity_vals.append(curr_val)
        
    equity_series = pd.Series(equity_vals, index=data.index)
    
    # Metrics
    if len(trades_list) > 0:
        wins = len([t for t in trades_list if t['pnl'] > 0])
        win_rate = (wins / len(trades_list)) * 100
        avg_return = np.mean([t['pnl_pct'] for t in trades_list]) * 100
    else:
        win_rate = 0.0
        avg_return = 0.0
        
    # Max DD
    cum_max = equity_series.cummax()
    drawdown = (equity_series - cum_max) / cum_max
    max_dd = drawdown.min() * 100
    
    total_return = (equity_series.iloc[-1] - initial_capital) / initial_capital * 100
    
    # Annualized (Approximate)
    days = (data.index[-1] - data.index[0]).days
    if days > 0:
        ann_return = ((1 + total_return/100) ** (365/days) - 1) * 100
    else:
        ann_return = 0.0
        
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'max_drawdown': max_dd,
        'annualized_return': ann_return,
        'num_trades': len(trades_list),
        'equity_curve': equity_series,
        'recent_trades': trades_list[-5:] # Last 5 trades
    }

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def generate_features(data: pd.DataFrame, dl_config: dict = None) -> pd.DataFrame:
    """Generate comprehensive technical indicator features"""
    df = data.copy()
    
    # Efficiency Ratio (Kaufman) - Dynamic Filter
    # High ER = Strong Trend, Low ER = Choppy/Noise
    if 'close' in df.columns:
        try:
            df['efficiency_ratio'] = ta.er(df['close'], length=10)
        except:
            pass # pandas_ta version compatibility
            
    # Deep Learning Features (CNN/LSTM Autoencoder)
    if dl_config and DLFeatureExtractor is None:
        # We can't use st.error here easily as we want to be UI agnostic
        # But we can print to stderr
        print("❌ DLFeatureExtractor could not be imported! Deep Learning features disabled. Check if 'tensorflow' is installed.", file=sys.stderr)

    if dl_config and DLFeatureExtractor:
        try:
            is_training = dl_config.get('train', False)
            model_path = dl_config.get('model_path', 'dl_model.h5')
            
            # Initialize Extractor (30-day sequence, 8 latent features)
            extractor = DLFeatureExtractor(sequence_length=30, encoding_dim=8)
            
            # Load model if not training
            model_loaded = True
            if not is_training:
                if os.path.exists(model_path):
                    extractor.load_model(model_path)
                else:
                    print(f"⚠️ DL Model not found at {os.path.abspath(model_path)}. Skipping DL features.", file=sys.stderr)
                    model_loaded = False
            
            if model_loaded:
                dl_features = extractor.generate_embeddings(df, epochs=10, verbose=0, train=is_training)
                
                if is_training:
                    extractor.save_model(model_path)
                    
                # Merge DL features
                df = pd.concat([df, dl_features], axis=1)
                
        except Exception as e:
            print(f"❌ Error generating DL features: {e}", file=sys.stderr)
    
    # Price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_10'] = df['returns'].rolling(10).std()
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = ta.sma(df['close'], length=period)
        df[f'ema_{period}'] = ta.ema(df['close'], length=period)
        
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
        if 'BBU_20_2.0' in df.columns and 'BBL_20_2.0' in df.columns:
            df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
            
    # VWAP (Intraday)
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            # Try/Except because VWAP requires datetime index
            vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'], anchor='D')
            if vwap is not None:
                df['vwap'] = vwap
                df['dist_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    except Exception:
        pass
    
    # Stochastic
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    if stoch is not None:
        df = pd.concat([df, stoch], axis=1)
    
    # ADX
    adx = ta.adx(df['high'], df['low'], df['close'])
    if adx is not None:
        df = pd.concat([df, adx], axis=1)
        adx_col = [c for c in df.columns if c.startswith('ADX_')]
        if adx_col:
            df['adx_slope'] = df[adx_col[0]].diff(3)
    
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
    
    # Composite Oscillator
    rsi_norm = (df['rsi_14'] - 50) / 50
    willr_norm = (df['willr_14'] + 50) / 50
    cci_norm = (df['cci_14'] / 100).clip(-1, 1)
    roc_norm = (df['roc_10'] / 5).clip(-1, 1)
    
    df['composite_oscillator'] = (rsi_norm + willr_norm + cci_norm + roc_norm) / 4
    
    df['composite_roc_5'] = df['composite_oscillator'].diff(5)
    df['composite_roc_10'] = df['composite_oscillator'].diff(10)
    
    # Long-term momentum
    df['roc_50'] = ta.roc(df['close'], length=50)
    
    # Distance from SMA50 (Trend Strength)
    if 'sma_50' in df.columns:
        df['dist_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    
    # Breakout Signal
    df['high_20d'] = df['high'].rolling(20).max().shift(1)
    df['breakout_20d'] = (df['close'] > df['high_20d']).astype(int)
    
    # ADX Trend
    adx_col = [c for c in df.columns if c.startswith('ADX_')]
    if adx_col:
        df['adx_trend'] = (df[adx_col[0]] > 25).astype(int)
    
    # Lagged features
    df['composite_slope'] = df['composite_oscillator'].diff(1)
    df['comp_oversold'] = (df['composite_oscillator'] < -0.6).astype(int)
    df['comp_overbought'] = (df['composite_oscillator'] > 0.6).astype(int)
    df['comp_cross_low'] = ((df['composite_oscillator'].shift(1) < -0.6) & (df['composite_oscillator'] > -0.6)).astype(int)
    df['comp_cross_high'] = ((df['composite_oscillator'].shift(1) > 0.6) & (df['composite_oscillator'] < 0.6)).astype(int)
    df['comp_bottom_turn'] = (df['comp_oversold'] & (df['composite_slope'] > 0)).astype(int)
    df['comp_top_turn'] = (df['comp_overbought'] & (df['composite_slope'] < 0)).astype(int)
    
    for lag in [1, 2, 3, 5]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)
        df[f'composite_lag_{lag}'] = df['composite_oscillator'].shift(lag)
    
    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
        df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
        df[f'rsi_mean_{window}'] = df['rsi_14'].rolling(window).mean()
        df[f'composite_mean_{window}'] = df['composite_oscillator'].rolling(window).mean()
        df[f'composite_std_{window}'] = df['composite_oscillator'].rolling(window).std()
    
    # Interaction Features
    if 'dist_sma50' in df.columns and 'rsi_14' in df.columns:
        df['trend_momentum'] = df['dist_sma50'] * (df['rsi_14'] - 50)
        
    if 'volatility_20' in df.columns and 'breakout_20d' in df.columns:
        df['vol_breakout'] = df['volatility_20'] * df['breakout_20d']
        
    if 'volume_ratio' in df.columns:
        df['volume_force'] = df['volume_ratio'] * df['returns']

    # Candlestick Pattern Recognition
    try:
        df.ta.cdl_pattern(name="all", append=True)
    except Exception as e:
        print(f"Warning: Could not generate candle patterns: {e}", file=sys.stderr)

    df = df.dropna()
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    features = df[feature_cols].select_dtypes(include=[np.number])
    
    # NOTE: Constant column removal DISABLED for production safety
    
    return features

# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

def save_model(model_data: dict, ticker: str, notes: str = "") -> str:
    """Save model with all required data"""
    save_dir = "saved_models_v2"
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add Profit to Filename if available
    return_suffix = ""
    if 'total_return' in model_data:
        r = model_data['total_return']
        sign = "+" if r >= 0 else ""
        return_suffix = f"_Ret{sign}{r:.1f}pct"
        
    filename = f"{model_data['model_type']}_{ticker}_{timestamp}{return_suffix}.joblib"
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
        'use_dl': model_data.get('use_dl', False),
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

# =============================================================================
# INFERENCE
# =============================================================================

def generate_signals(model_data: dict, features: pd.DataFrame, threshold: float = 0.5, 
                     use_slope_signals: bool = False,
                     slope_buy_thresh: float = 0.0,
                     slope_sell_thresh: float = 0.0,
                     use_ml_confirm: bool = False,
                     ml_confirm_thresh: float = 0.3,
                     slope_roc_thresh: float = 0.0,
                     use_mom_zone: bool = False,
                     buy_threshold: float = None,
                     sell_threshold: float = None) -> tuple[pd.Series, pd.DataFrame]:
    """
    Generate signals using the trained model with custom thresholds.
    Returns: (signals, probabilities)
    """
    if buy_threshold is None: buy_threshold = threshold
    if sell_threshold is None: sell_threshold = threshold

    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    label_encoder = model_data.get('label_encoder')
    
    # Align features
    missing = [f for f in feature_names if f not in features.columns]
    if missing:
        print(f"Warning: Missing {len(missing)} features. Filling with 0.0. (Example: {missing[0]})", file=sys.stderr)
        for col in missing:
            features[col] = 0.0
    
    X = features[feature_names]
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Get probabilities
    probs = model.predict_proba(X_scaled)
    
    # Get class indices (Handle Label Encoding)
    classes = model.classes_
    
    buy_val = 1
    sell_val = -1
    
    if label_encoder:
        # If encoded, transform 1/-1 to encoded integer
        try:
            buy_val = label_encoder.transform([1])[0]
        except:
            buy_val = None # 1 not in training labels
            
        try:
            sell_val = label_encoder.transform([-1])[0]
        except:
            sell_val = None # -1 not in training labels

    # Safely get indices
    buy_indices = np.where(classes == buy_val)[0] if buy_val is not None else []
    sell_indices = np.where(classes == sell_val)[0] if sell_val is not None else []
    
    if len(buy_indices) > 0:
        buy_idx = buy_indices[0]
        prob_buy = probs[:, buy_idx]
    else:
        prob_buy = np.zeros(len(features))
        
    if len(sell_indices) > 0:
        sell_idx = sell_indices[0]
        prob_sell = probs[:, sell_idx]
    else:
        prob_sell = np.zeros(len(features))
    
    prob_df = pd.DataFrame({
        'prob_buy': prob_buy,
        'prob_sell': prob_sell
    }, index=features.index)
    
    # Basic signals
    signals = pd.Series(0, index=features.index)
    signals[prob_buy > buy_threshold] = 1
    signals[prob_sell > sell_threshold] = -1
    
    # CONFIDENCE BOOSTING
    # If High Confidence (>0.8), ignore trend/chop filters
    high_conf_buy = prob_buy > 0.8
    high_conf_sell = prob_sell > 0.8
    signals[high_conf_buy] = 1
    signals[high_conf_sell] = -1

    # Add predicted_signal and confidence to prob_df (Required by UI)
    prob_df['predicted_signal'] = signals
    prob_df['confidence'] = probs.max(axis=1)

    
    # SLOPE TURN SIGNALS
    if use_slope_signals and 'composite_oscillator' in features.columns:
        comp = features['composite_oscillator']
        slope = comp.diff(1)
        slope_roc = np.diff(slope, prepend=0) # Acceleration
        
        # Turn UP (Buy)
        # Condition: Oscillator is Low AND Slope turns Positive
        cond_low = comp < slope_buy_thresh
        cond_turn_up = (slope.shift(1) < 0) & (slope > 0)
        
        # Momentum Zone logic (Buy Breakouts)
        valid_buy_zone = cond_low
        if use_mom_zone:
            # Allow buying in "Momentum Zone" (0.2 to 0.4) if Slope is accelerating
            cond_mom = (comp > 0.2) & (comp < 0.4) & (slope > 0.05)
            valid_buy_zone = cond_low | cond_mom
            
        # Acceleration logic
        valid_accel = True
        if slope_roc_thresh > 0:
            valid_accel = pd.Series(slope_roc, index=features.index) > slope_roc_thresh
            
        turn_up_mask = valid_buy_zone & cond_turn_up & valid_accel
        
        # Turn DOWN (Sell)
        cond_high = comp > slope_sell_thresh
        cond_turn_down = (slope.shift(1) > 0) & (slope < 0)
        turn_down_mask = cond_high & cond_turn_down
        
        # ML Confirmation
        if use_ml_confirm:
            turn_up_mask = turn_up_mask & (prob_buy > ml_confirm_thresh)
            turn_down_mask = turn_down_mask & (prob_sell > ml_confirm_thresh)
            
        # Apply Override
        signals[turn_up_mask] = 1
        signals[turn_down_mask] = -1
        
    return signals, prob_df

def apply_trend_filter(signals: pd.Series, data: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """
    Apply Trend Filter (SMA 200 Rule).
    Matches logic in Streamlit App.
    
    Logic:
    - Calculate SMA 200
    - If Price > SMA 200 (Bull Market):
      - ALLOW all BUY signals
      - REJECT SELL signals UNLESS Composite Oscillator > 0.6 (Extreme Overbought)
    - If Price < SMA 200 (Bear Market):
      - ALLOW all signals (Buy dips, Sell rallies)
    """
    if 'close' not in data.columns:
        return signals
        
    filtered_signals = signals.copy()
    
    # 1. Calc SMA 200
    # We need enough data. If data < 200, we can't filter effectively.
    if len(data) < 200:
        return signals
        
    sma200 = data['close'].rolling(200).mean()
    is_bull = data['close'] > sma200
    
    # 2. Re-Calc Composite (To ensure we have it)
    # Similar logic to peak_valley_ml_v2.py
    oscillators = pd.DataFrame(index=features.index)
    if 'rsi_14' in features.columns: oscillators['rsi'] = (features['rsi_14'] - 50) / 50
    if 'willr_14' in features.columns: oscillators['willr'] = (features['willr_14'] + 50) / 50
    if 'cci_14' in features.columns: oscillators['cci'] = (features['cci_14'] / 100).clip(-1, 1)
    if 'roc_10' in features.columns: oscillators['roc'] = (features['roc_10'] / 5).clip(-1, 1)
    
    if not oscillators.empty:
        composite = oscillators.mean(axis=1)
    else:
        # Fallback if features missing
        composite = pd.Series(0, index=features.index)
    
    # 3. Apply Logic
    # Align indices
    common_idx = signals.index.intersection(is_bull.index).intersection(composite.index)
    
    if common_idx.empty:
        return signals
        
    bull_mask = is_bull.loc[common_idx]
    comp_overbought = composite.loc[common_idx] > 0.6
    
    # Filter SELLs (-1)
    # Condition to REJECT: Is Sell AND Is Bull AND Not Overbought
    # i.e. Don't sell in a bull market unless it's a blow-off top
    
    mask_reject_sell = (signals.loc[common_idx] == -1) & bull_mask & (~comp_overbought)
    
    # Apply rejection
    # We must operate on the original series index
    reject_indices = mask_reject_sell[mask_reject_sell].index
    filtered_signals.loc[reject_indices] = 0
    
    return filtered_signals

# =============================================================================
# META MODEL (TRADE FILTERING)
# =============================================================================

def train_meta_model(base_model_data: dict, features: pd.DataFrame, labels: pd.Series = None) -> dict:
    """
    Train a Production-Grade Meta-Model (XGBoost) to filter base model signals.
    Uses Cross-Validation (OOF) predictions to prevent leakage.
    """
    from sklearn.model_selection import cross_val_predict, TimeSeriesSplit
    from sklearn.metrics import precision_score, roc_auc_score
    from xgboost import XGBClassifier
    from sklearn.base import clone
    
    # 1. Prepare Data for OOF Predictions
    # We need to recreate the training scenario of the base model.
    # If labels are None (simulated training in app), we try to infer them or fail.
    # In production, we assume 'features' contains the target or we can re-derive it.
    
    # Deriving target (same as main training: 5d forward return > 0? or labeled?)
    # For OOF, we need the original 'y' used to train base model.
    # If not provided, we can't do strict OOF. 
    # Fallback: If labels missing, use simple inference (Leaky but necessary if data not passed)
    # BUT the user called it with labels=None in the UI. 
    # I should update the UI to pass labels if possible, or re-calculate them here.
    
    if labels is None:
        # Re-calculate labels based on typical strategy (e.g. 5d return > 0)
        # This assumes the base model was trained on similar labels.
        if 'close' in features.columns:
            fwd_ret = features['close'].pct_change(5).shift(-5)
            # Standard labeling: Buy (1) if ret > 1%, Sell (-1) if ret < -1%, else 0
            # Simplify to Binary for OOF generation? No, Base model is multiclass usually.
            # Let's try to match base model logic.
            y_derived = pd.Series(0, index=features.index)
            y_derived[fwd_ret > 0.01] = 1
            y_derived[fwd_ret < -0.01] = -1
            labels = y_derived
        else:
            return {'status': 'failed', 'reason': 'No labels or close price for OOF generation'}

    # 2. Generate OOF Predictions (The "Input" for Meta Model)
    # This shows us what the Base Model predicts when it hasn't seen the data.
    print("   ⟳ Generating Out-of-Fold predictions for Meta-Model...")
    
    try:
        # Re-instantiate base model with best params
        model_type = base_model_data.get('model_type', 'xgboost')
        best_params = base_model_data.get('best_params', {})
        
        if model_type == 'xgboost':
            from xgboost import XGBClassifier
            base_clf = XGBClassifier(**best_params)
        else:
            from sklearn.ensemble import RandomForestClassifier
            base_clf = RandomForestClassifier(**best_params)
            
        # Prepare X and y for Base Model
        # Need to align with features used in base model
        feature_names = base_model_data['feature_names']
        
        # Ensure features exist (fill missing with 0)
        X_base = features.reindex(columns=feature_names, fill_value=0)
        
        # Scale if scaler exists
        scaler = base_model_data['scaler']
        X_base_scaled = scaler.transform(X_base)
        
        # CV Strategy
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Get Probabilities (OOF)
        # cross_val_predict with method='predict_proba'
        oof_probs = cross_val_predict(base_clf, X_base_scaled, labels, cv=tscv, method='predict_proba', n_jobs=-1)
        
        # Pad the beginning (TimeSeriesSplit leaves first fold empty)
        # We'll just fill with 0.5 (uncertainty) or drop.
        # cross_val_predict returns array of size equal to input? 
        # For TimeSeriesSplit, it usually returns predictions only for test indices. 
        # Actually sklearn cross_val_predict doesn't support TimeSeriesSplit nicely for size matching in older versions.
        # Let's use KFold if shuffling allowed? No, Time Series.
        # Manual Loop for safety and correctness.
        
        oof_prob_buy = np.zeros(len(features))
        oof_prob_sell = np.zeros(len(features))
        
        # If classes are [0, 1, 2] (Hold, Buy, Sell) or [-1, 0, 1] mapped.
        # We need to map probability columns to Buy/Sell correctly.
        # Label Encoder?
        le = base_model_data.get('label_encoder')
        buy_idx = 1
        sell_idx = 2 # Guessing
        
        if le:
            classes = le.classes_
            # find index of 1 and -1
            if 1 in classes: buy_idx = np.where(classes == 1)[0][0]
            if -1 in classes: sell_idx = np.where(classes == -1)[0][0]
        
        # Since we can't easily do OOF for the *first* fold, we will define the Meta-Model training set
        # as the data *after* the first fold of the base model.
        
        valid_indices = []
        
        for train_ix, test_ix in tscv.split(X_base_scaled):
            X_tr, y_tr = X_base_scaled[train_ix], labels.iloc[train_ix]
            X_te = X_base_scaled[test_ix]
            
            # Fit Base on Past
            base_clf.fit(X_tr, y_tr)
            
            # Predict on Future
            probs = base_clf.predict_proba(X_te)
            
            # Store
            # Check shape of probs (might miss classes if not in training fold)
            # Handle variable class count...
            fold_classes = base_clf.classes_
            
            for i, idx in enumerate(test_ix):
                # Map probs to Buy/Sell
                p_buy = 0.0
                p_sell = 0.0
                
                # Check if Buy class exists in this fold
                if 1 in fold_classes:
                    c_idx = np.where(fold_classes == 1)[0][0]
                    p_buy = probs[i, c_idx]
                    
                if -1 in fold_classes:
                    c_idx = np.where(fold_classes == -1)[0][0]
                    p_sell = probs[i, c_idx]
                    
                oof_prob_buy[idx] = p_buy
                oof_prob_sell[idx] = p_sell
                valid_indices.append(idx)
                
    except Exception as e:
        print(f"   ⚠️ OOF Generation failed ({e}). Falling back to simple inference (Leaky).")
        # Fallback: Predict on all data using pre-trained model
        _, prob_df = generate_signals(base_model_data, features, threshold=0.0)
        oof_prob_buy = prob_df['prob_buy'].values
        oof_prob_sell = prob_df['prob_sell'].values
        valid_indices = range(len(features))

    # 3. Construct Meta-Features (X_meta)
    meta_df = pd.DataFrame(index=features.index)
    meta_df['base_conf_buy'] = oof_prob_buy
    meta_df['base_conf_sell'] = oof_prob_sell
    
    # Interaction: Confidence * Volatility (High conf in high vol = Risky or Good?)
    if 'volatility_20' in features.columns:
        meta_df['conf_vol_interaction'] = meta_df['base_conf_buy'] * features['volatility_20']
        
    # Context Features
    context_cols = ['rsi_14', 'adx_14', 'volatility_20', 'composite_oscillator', 'dist_sma50', 'volume_ratio']
    for col in context_cols:
        if col in features.columns:
            meta_df[col] = features[col]
            
    # 4. Define Meta-Target (Success = Profit)
    # We want to filter BUY signals.
    if 'close' in features.columns:
        # Target: > 1.5% return in 5 days (Aggressive success)
        fwd_ret = features['close'].pct_change(5).shift(-5) * 100
    else:
        fwd_ret = pd.Series(0, index=features.index)
        
    meta_target = (fwd_ret > 1.5).astype(int)
    
    # Only train on valid OOF indices (skip the start where we have no OOF prediction)
    # And only train on rows where Base Model has SOME interest (e.g. Prob > 0.3)
    # Training on "Easy Holds" (Prob=0.01) is useless. We need to classify the "Maybe/Yes" trades.
    
    mask_interesting = (meta_df['base_conf_buy'] > 0.3) & (meta_df.index.isin(features.index[valid_indices]))
    
    train_data = meta_df.loc[mask_interesting].join(meta_target.rename('target')).dropna()
    
    if len(train_data) < 20:
         return {'status': 'failed', 'reason': f'Not enough interesting signals to train Meta Model (n={len(train_data)})'}
         
    X_meta = train_data.drop('target', axis=1)
    y_meta = train_data['target']
    
    # 5. Train SOTA Meta-Model (XGBoost)
    # Monotonic constraints? No.
    # Scale_pos_weight? Yes, unbalanced wins.
    ratio = float(len(y_meta[y_meta==0])) / len(y_meta[y_meta==1]) if len(y_meta[y_meta==1]) > 0 else 1.0
    
    meta_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=ratio,
        eval_metric='logloss',
        random_state=42
    )
    
    meta_model.fit(X_meta, y_meta)
    
    # Calc Score (AUC/Precision)
    y_pred = meta_model.predict(X_meta)
    score = precision_score(y_meta, y_pred, zero_division=0)
    
    return {
        'status': 'success',
        'model': meta_model,
        'score': score,
        'features': X_meta.columns.tolist(),
        'type': 'XGBoost_SOTA_Meta'
    }

def apply_meta_filter(meta_model_data: dict, base_signals: pd.Series, base_probs: pd.DataFrame, features: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """
    Filter base signals using the Meta-Model.
    Returns: Filtered Signals (0 where Meta-Model says "Don't Trade")
    """
    if meta_model_data.get('status') != 'success':
        return base_signals
        
    model = meta_model_data['model']
    req_cols = meta_model_data['features']
    
    # Reconstruct Meta Features
    meta_df = pd.DataFrame(index=features.index)
    meta_df['base_conf_buy'] = base_probs['prob_buy']
    meta_df['base_conf_sell'] = base_probs['prob_sell']
    
    for col in req_cols:
        if col in features.columns and col not in meta_df.columns:
            meta_df[col] = features[col]
            
    # Align cols
    # Handle missing cols (fill 0)
    for col in req_cols:
        if col not in meta_df.columns:
            meta_df[col] = 0.0
            
    X_meta = meta_df[req_cols].fillna(0)
    
    # Predict Probability of Success
    meta_probs = model.predict_proba(X_meta)[:, 1] # Prob of Class 1 (Success)
    
    # Filter
    # If Meta-Prob < threshold, reject trade
    # We only filter the 1s (Buys). Sells (-1) pass through or need separate meta model.
    
    filtered_signals = base_signals.copy()
    
    # Identify Buys that Meta Model dislikes
    # Buy Signal AND Low Meta Confidence
    bad_buys = (base_signals == 1) & (meta_probs < threshold)
    
    filtered_signals[bad_buys] = 0 # Kill the signal
    
    return filtered_signals

def train_regressor(features: pd.DataFrame, target: pd.Series, n_trials: int = 20, progress_callback=None) -> dict:
    """Train XGBRegressor with Optuna tuning"""
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    import optuna
    
    # Align
    common_idx = features.index.intersection(target.index)
    X = features.loc[common_idx]
    y = target.loc[common_idx]
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'n_jobs': -1,
            'objective': 'reg:squarederror',
            'verbosity': 0
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_t, y_t)
            preds = model.predict(X_v)
            scores.append(np.sqrt(mean_squared_error(y_v, preds))) # RMSE
            
        return np.mean(scores)
    
    # Callback for progress updates
    def optuna_callback(study, trial):
        if progress_callback:
            progress_callback(trial.number, n_trials)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, callbacks=[optuna_callback])
    
    best_params = study.best_params
    best_params['n_jobs'] = -1
    best_params['objective'] = 'reg:squarederror'
    
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
    preds_test = final_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds_test))
    r2 = r2_score(y_test, preds_test)
    
    return {
        'model': final_model,
        'rmse': rmse,
        'r2': r2,
        'best_params': best_params
    }
