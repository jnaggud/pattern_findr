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
import joblib
import os
import json
from datetime import datetime
from data_manager import DataManager, StrategyFileManager
from ml_utils import (
    load_price_data, 
    detect_peaks_valleys, 
    generate_features, 
    calculate_theoretical_return,
    save_model,
    load_model,
    generate_signals,
    train_meta_model,
    apply_meta_filter,
    train_regressor,
    optimize_full_strategy
)

# Try to import Conformal Wrapper
try:
    from conformal_utils import ConformalPredictionWrapper
except ImportError:
    ConformalPredictionWrapper = None

# Import Polygon Manager
try:
    from polygon_manager import PolygonManager
except ImportError:
    PolygonManager = None

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
    
    # Load persisted optimization results from file if not in session state
    if 'v2_last_optimization' not in st.session_state:
        import json
        opt_cache_path = os.path.join("saved_models_v2", "last_optimization.json")
        if os.path.exists(opt_cache_path):
            try:
                with open(opt_cache_path, 'r') as f:
                    st.session_state.v2_last_optimization = json.load(f)
            except Exception:
                st.session_state.v2_last_optimization = None
        else:
            st.session_state.v2_last_optimization = None

init_session_state()

# =============================================================================
# CORE FUNCTIONS MOVED TO ML_UTILS.PY
# =============================================================================
# Functions imported above.




def train_model_with_optuna(features: pd.DataFrame, labels: pd.Series, 
                            model_type: str = 'xgboost', use_smote: bool = True,
                            n_trials: int = 20, progress_callback=None,
                            optimize_metric: str = 'f1_weighted', n_cv_splits: int = 5,
                            class_weight_ratio: float = 1.0, optimize_training_params: bool = False):
    """
    Train model with Optuna optimization.
    Supports optimizing Class Weights and SMOTE if optimize_training_params=True.
    """
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score
    from sklearn.utils.class_weight import compute_sample_weight
    from imblearn.over_sampling import SMOTE
    import optuna
    
    # Console output header
    print("\n" + "="*70)
    print(f"🚀 TRAINING {model_type.upper()} MODEL (Optuna)")
    print("="*70)
    
    # Align data
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]
    
    # Time-based split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Pre-compute Datasets (for speed)
    # 1. Raw
    data_raw = (X_train_scaled, y_train.values)
    
    # 2. SMOTE (Calculate once if possible, or just instantiate object)
    # If we optimize SMOTE, we need to have the SMOTE data ready.
    print("⏳ Pre-computing SMOTE dataset...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    data_smote = (X_train_smote, y_train_smote)
    print(f"   Raw: {len(X_train_scaled)} | SMOTE: {len(X_train_smote)}")
    
    # Label Encoding
    label_encoder = None
    if model_type == 'xgboost':
        label_encoder = LabelEncoder()
        # Fit on full labels to ensure all classes covered
        label_encoder.fit(y) 
    
    # Helper to get encoded y
    def get_encoded_y(y_in):
        if label_encoder:
            return label_encoder.transform(y_in)
        return y_in
        
    # Helper to get weights
    def get_weights(y_in, ratio):
        if ratio <= 1.0: return None
        if label_encoder:
            buy_val = label_encoder.transform([1])[0]
            sell_val = label_encoder.transform([-1])[0]
        else:
            buy_val, sell_val = 1, -1
            
        w = np.ones(len(y_in))
        w[y_in == buy_val] = ratio
        w[y_in == sell_val] = ratio
        return w

    # Thread-safe tracking
    import threading
    lock = threading.Lock()
    best_score_so_far = [0.0]
    completed_trials = [0]
    best_trial_num = [0]
    
    def objective(trial):
        # 1. Suggest Data & Config
        if optimize_training_params:
            use_smote_trial = trial.suggest_categorical('use_smote', [True, False])
            cw_ratio_trial = trial.suggest_float('class_weight_ratio', 1.0, 10.0, step=0.5)
        else:
            use_smote_trial = use_smote
            cw_ratio_trial = class_weight_ratio
            
        # Select Data
        X_t, y_t_raw = data_smote if use_smote_trial else data_raw
        y_t = get_encoded_y(y_t_raw)
        
        # Calculate Weights
        sample_weights_t = get_weights(y_t, cw_ratio_trial)
        
        # 2. Suggest Hyperparams
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
                'n_jobs': 1
            }
            model = XGBClassifier(**params)
        else:
            from sklearn.ensemble import RandomForestClassifier
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42,
                'n_jobs': 1
            }
            model = RandomForestClassifier(**params)
            
        # 3. Cross-Validation
        tscv = TimeSeriesSplit(n_splits=n_cv_splits)
        scores = []
        
        # We need to handle weights in CV. 
        # Sklearn cross_val_score supports fit_params but manual loop is clearer for weights.
        # Actually, sample_weight must be aligned with split.
        
        for train_idx, val_idx in tscv.split(X_t):
            X_fold_train, X_fold_val = X_t[train_idx], X_t[val_idx]
            y_fold_train, y_fold_val = y_t[train_idx], y_t[val_idx]
            
            w_fold_train = sample_weights_t[train_idx] if sample_weights_t is not None else None
            
            model.fit(X_fold_train, y_fold_train, sample_weight=w_fold_train)
            y_pred = model.predict(X_fold_val)
            
            if optimize_metric == 'f1_weighted':
                score = f1_score(y_fold_val, y_pred, average='weighted')
            elif optimize_metric == 'f1_macro':
                score = f1_score(y_fold_val, y_pred, average='macro')
            elif optimize_metric == 'recall':
                # Optimize for BUY Recall (Class 1)
                pos_label = 1
                if label_encoder:
                    try: pos_label = label_encoder.transform([1])[0]
                    except: pos_label = None
                
                if pos_label is not None:
                    # Calculate recall for specific label
                    # average=None returns array of recall for each class in labels
                    recalls = recall_score(y_fold_val, y_pred, average=None, labels=[pos_label], zero_division=0)
                    score = recalls[0] if len(recalls) > 0 else 0.0
                else:
                    score = 0.0
            else:
                score = accuracy_score(y_fold_val, y_pred)
            scores.append(score)
            
        avg_score = np.mean(scores)
        
        # Logging
        with lock:
            completed_trials[0] += 1
            is_best = False
            if avg_score > best_score_so_far[0]:
                best_score_so_far[0] = avg_score
                best_trial_num[0] = trial.number
                is_best = True
            
            if is_best:
                print(f"⭐ New Best (Trial {trial.number}): {avg_score:.4f} | SMOTE={use_smote_trial}, CW={cw_ratio_trial:.1f}")
            else:
                print(f"   Trial {trial.number}: {avg_score:.4f} | SMOTE={use_smote_trial}, CW={cw_ratio_trial:.1f}")
            
            if progress_callback:
                progress_callback(completed_trials[0])
                
        return avg_score
    
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
    
    best_params = study.best_params.copy()
    
    print("-"*70)
    print(f"✅ Optimization complete! Best trial: #{best_trial_num[0]} with CV score: {study.best_value:.4f}")
    print(f"📋 Best parameters: {best_params}")
    
    # Extract Training Config from best_params
    final_use_smote = best_params.pop('use_smote', use_smote)
    final_cw_ratio = best_params.pop('class_weight_ratio', class_weight_ratio)
    
    # Train final model with best params
    print(f"\n🏋️ Training final model with best parameters...")
    print(f"   SMOTE: {final_use_smote} | Class Weight: {final_cw_ratio:.2f}x")
    
    if model_type == 'xgboost':
        from xgboost import XGBClassifier
        best_params.update({'random_state': 42, 'use_label_encoder': False, 
                           'eval_metric': 'mlogloss', 'verbosity': 0})
        model = XGBClassifier(**best_params)
    else:
        from sklearn.ensemble import RandomForestClassifier
        best_params.update({'random_state': 42, 'n_jobs': -1})
        model = RandomForestClassifier(**best_params)
    
    # Prepare Final Training Data
    X_final, y_final_raw = data_smote if final_use_smote else data_raw
    y_final = get_encoded_y(y_final_raw)
    
    # Prepare Final Weights
    final_weights = get_weights(y_final, final_cw_ratio)
    
    # Fit
    model.fit(X_final, y_final, sample_weight=final_weights)
    
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


def run_backtest(data: pd.DataFrame, signals: pd.Series, initial_capital: float = 100000, limit_pct: float = 0.0, stop_loss_pct: float = 0.0, take_profit_pct: float = 0.0, verbose: bool = True) -> dict:
    """
    Run backtest with detailed trade tracking.
    If limit_pct > 0, attempts to enter at Close * (1 - limit_pct) on the NEXT day.
    Supports Stop Loss and Take Profit (pct as decimal, e.g. 0.05 for 5%).
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
        
        # Check Stop Loss / Take Profit for EXISTING positions
        if position == 1 and trades:
            entry_price = trades[-1]['entry_price']
            
            # Stop Loss Logic
            if stop_loss_pct > 0:
                sl_price = entry_price * (1 - stop_loss_pct)
                if row['low'] <= sl_price:
                    # SL Triggered
                    exit_price = sl_price
                    if row['open'] < sl_price: exit_price = row['open'] # Gap down
                    
                    exit_value = shares * exit_price
                    profit = exit_value - trades[-1]['entry_capital']
                    profit_pct = profit / trades[-1]['entry_capital'] * 100
                    
                    trades[-1].update({
                        'exit_date': date,
                        'exit_price': exit_price,
                        'exit_value': exit_value,
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'status': 'closed (SL)'
                    })
                    capital = exit_value
                    position = 0
                    shares = 0
                    
                    # Track equity after close
                    equity_curve.append({'date': date, 'equity': capital, 'price': price})
                    continue # Trade closed, skip to next bar

            # Take Profit Logic
            if take_profit_pct > 0 and position == 1:
                tp_price = entry_price * (1 + take_profit_pct)
                if row['high'] >= tp_price:
                    # TP Triggered
                    exit_price = tp_price
                    if row['open'] > tp_price: exit_price = row['open'] # Gap up
                    
                    exit_value = shares * exit_price
                    profit = exit_value - trades[-1]['entry_capital']
                    profit_pct = profit / trades[-1]['entry_capital'] * 100
                    
                    trades[-1].update({
                        'exit_date': date,
                        'exit_price': exit_price,
                        'exit_value': exit_value,
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'status': 'closed (TP)'
                    })
                    capital = exit_value
                    position = 0
                    shares = 0
                    
                    # Track equity after close
                    equity_curve.append({'date': date, 'equity': capital, 'price': price})
                    continue # Trade closed, skip to next bar

        # Track equity
        current_equity = shares * price if position == 1 else capital
        equity_curve.append({'date': date, 'equity': current_equity, 'price': price})
        
        if signal == 1 and position == 0:  # BUY
            entry_price = price
            filled = True
            fill_date = date
            
            if limit_pct > 0.001:  # Treat < 0.1% as Market Order (0.0) to avoid float errors
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




def optimize_strategy(data, model_data, features, n_trials=100, progress_callback=None):
    """
    Optimize Strategy Sliders using Optuna (Fast Post-Processing).
    Does NOT re-train the model. Optimizes filters only.
    """
    import optuna
    # Suppress console output to avoid confusion with model training
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print("--- STARTING STRATEGY OPTIMIZATION ---")
    
    def objective(trial):
        # Update progress if callback provided
        if progress_callback:
            progress_callback(trial.number, n_trials)

        # Suggest parameters (Strategy Sliders)
        s_buy = trial.suggest_float('slope_buy_thresh', -1.0, 0.0, step=0.1)
        s_sell = trial.suggest_float('slope_sell_thresh', 0.0, 1.0, step=0.1)
        roc_thresh = trial.suggest_float('slope_roc_thresh', 0.0, 0.5, step=0.05)
        mom_zone = trial.suggest_categorical('use_mom_zone', [True, False])
        
        # Trend Filter
        use_trend = trial.suggest_categorical('use_trend_filter', [True, False])
        
        # Volatility Filter (ADX)
        use_adx = trial.suggest_categorical('use_adx_filter', [True, False])
        adx_thresh = trial.suggest_int('adx_threshold', 15, 30)
        
        # ML Thresholds (Instead of fixed 0.5)
        # Allow asymmetry (e.g. easy entry, strict exit or vice versa)
        ml_buy = trial.suggest_float('ml_buy_threshold', 0.2, 0.8, step=0.05)
        ml_sell = trial.suggest_float('ml_sell_threshold', 0.2, 0.8, step=0.05)
        
        # ML Confirmation params (for slope signals)
        use_ml = trial.suggest_categorical('use_ml_confirm', [True, False])
        ml_conf_thresh = trial.suggest_float('ml_confirm_thresh', 0.0, 0.5, step=0.05)
        
        # Generate signals
        signals, _ = generate_signals(
            model_data, features,
            use_slope_signals=True,
            slope_buy_thresh=s_buy,
            slope_sell_thresh=s_sell,
            use_ml_confirm=use_ml,
            ml_confirm_thresh=ml_conf_thresh,
            slope_roc_thresh=roc_thresh,
            use_mom_zone=mom_zone,
            buy_threshold=ml_buy,
            sell_threshold=ml_sell
        )
        
        # Apply Trend Filter (if selected)
        if use_trend:
            # Calculate SMA200 (using full available data to be safe)
            aligned_data = data.loc[signals.index]
            if len(aligned_data) > 200:
                sma200 = aligned_data['close'].rolling(200).mean()
                is_bull = aligned_data['close'] > sma200
                
                bull_mask = is_bull
                # Filter SELLs (-1) in Bull Market
                # Keep BUYs (1)
                sell_mask = (signals == -1) & bull_mask
                signals.loc[sell_mask] = 0
                
        # Apply Volatility Filter (ADX) (if selected)
        if use_adx:
            adx_cols = [c for c in features.columns if c.startswith('ADX_')]
            if adx_cols:
                # Get ADX values and align
                adx_vals = features[adx_cols[0]].loc[signals.index]
                
                # Identify "Chop" zones (ADX < Threshold)
                chop_mask = adx_vals < adx_thresh
                
                # Block ALL new signals in chop
                # (We set signal to 0, which means HOLD)
                signals.loc[chop_mask] = 0
        
        # Run Backtest
        # Pure Signal-Based: No SL/TP overrides
        res = run_backtest(data, signals, verbose=False)
        
        ret = res['total_return']
        
        # Console progress (optional, can be spammy)
        # print(f"Trial {trial.number}: {ret:.2f}%")
        
        # Objective: Maximize Total Return
        # Penalize very few trades
        if res['num_trades'] < 5: 
            return -100.0 
            
        return ret

    study = optuna.create_study(direction='maximize')
    
    # Use all cores except 1
    import os
    n_jobs = max(1, os.cpu_count() - 1)
    if progress_callback:
        n_jobs = 1 # Must be single-threaded to update Streamlit UI safely
        
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    
    return study.best_params, study.best_value



def analyze_slope_signals(file_path: str):
    """
    Analyze Slope Turn Signals from a saved analysis CSV.
    Generates a comprehensive report including multi-timeframe return matrix, histograms, and scatter plots.
    """
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        if 'composite_indicator' not in df.columns:
            st.error("Composite Indicator not found in analysis file.")
            return
        
        # 1. Calculations
        df['composite_slope'] = df['composite_indicator'].diff()
        # Rate of Change of Slope (Acceleration)
        df['slope_roc'] = df['composite_slope'].diff()
        
        slope = df['composite_slope'].values
        slope_prev = np.roll(slope, 1)
        slope_prev[0] = 0
        
        turn_up = (slope_prev < 0) & (slope > 0)
        turn_down = (slope_prev > 0) & (slope < 0)
        
        # Returns (1d, 3d, 5d, 10d, 20d)
        horizons = [1, 3, 5, 10, 20]
        for d in horizons:
            df[f'ret_{d}d'] = df['close'].pct_change(d).shift(-d) * 100
            
        buys = df[turn_up].copy()
        sells = df[turn_down].copy()
        
        # --- GENERATE REPORT ---
        report = []
        report.append("============================================================")
        report.append("📉 SLOPE SIGNAL DEEP DIVE REPORT (ENHANCED V3)")
        report.append("============================================================")
        report.append(f"Total Potential BUYS: {len(buys)}")
        report.append(f"Total Potential SELLS: {len(sells)}")
        report.append("")
        
        # 1. COMPOSITE BIN PERFORMANCE MATRIX
        report.append("📊 1. COMPOSITE BIN PERFORMANCE MATRIX (Avg Return %)")
        # Bins of 0.1
        bins = np.arange(-1.0, 1.1, 0.1)
        buys['comp_bin'] = pd.cut(buys['composite_indicator'], bins=bins)
        
        # Pivot table for Buys
        buy_matrix = buys.groupby('comp_bin')[[f'ret_{d}d' for d in horizons]].mean()
        buy_counts = buys.groupby('comp_bin')['ret_5d'].count()
        buy_matrix.insert(0, 'Count', buy_counts)
        
        report.append("\n🟢 BUY SIGNAL RETURNS BY HORIZON:")
        report.append(buy_matrix.to_string())
        
        report_text = "\n".join(report)
        
        # Display Report Text
        st.text_area("📋 Copy-Paste Analysis Report", report_text, height=400)
        
        # --- VISUALIZATIONS ---
        import plotly.express as px
        
        st.subheader("📊 Signal Analysis Plots")
        
        tab_v1, tab_v2, tab_v3 = st.tabs(["Scatter Analysis", "Histograms", "ROC Analysis"])
        
        with tab_v1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Buy Signals: Composite Value vs 5d Return**")
                fig1 = px.scatter(buys, x='composite_indicator', y='ret_5d', 
                                 color='ret_5d', color_continuous_scale='RdYlGn',
                                 hover_data=['close'])
                fig1.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig1, use_container_width=True)
                
            with col2:
                st.markdown("**Buy Signals: Slope Magnitude vs 5d Return**")
                # Slope Magnitude (how sharp was the turn?)
                # Actually composite_slope is the change.
                fig2 = px.scatter(buys, x='composite_slope', y='ret_5d',
                                 color='ret_5d', color_continuous_scale='RdYlGn')
                fig2.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig2, use_container_width=True)
                
        with tab_v2:
            st.markdown("**Distribution of 5-Day Returns (Buy Signals)**")
            fig3 = px.histogram(buys, x='ret_5d', nbins=50, 
                               color_discrete_sequence=['green'])
            fig3.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig3, use_container_width=True)
            
        with tab_v3:
            st.markdown("**Rate of Change Analysis**")
            # Is acceleration relevant?
            # Plot Acceleration vs Return
            fig4 = px.scatter(buys, x='slope_roc', y='ret_5d', 
                             title="Slope Acceleration vs 5d Return",
                             color='ret_5d', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig4, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")


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

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
import threading

try:
    from dl_feature_extractor import DLFeatureExtractor
except ImportError:
    DLFeatureExtractor = None

st.title("📈 Peak/Valley ML v2 - Enhanced")
st.caption("Clean architecture with SMOTE, Optuna, and comprehensive analysis")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker Symbol", value=st.session_state.v2_ticker)
    
    # Load API Key from user_settings.json (Primary) or strategy_config.json (Fallback)
    if 'polygon_api_key_input' not in st.session_state:
        st.session_state.polygon_api_key_input = ""
        
        # Try user_settings.json
        if os.path.exists("user_settings.json"):
            try:
                import json
                with open("user_settings.json", "r") as f:
                    u_conf = json.load(f)
                    if u_conf.get("polygon_api_key"):
                        st.session_state.polygon_api_key_input = u_conf.get("polygon_api_key")
            except: pass
            
        # Fallback to strategy_config.json
        elif os.path.exists("strategy_config.json"):
            try:
                import json
                with open("strategy_config.json", "r") as f:
                    saved_conf = json.load(f)
                    if saved_conf.get("polygon_api_key"):
                        st.session_state.polygon_api_key_input = saved_conf.get("polygon_api_key")
            except: pass

    polygon_api_key = st.text_input("🔑 Polygon.io API Key (Optional)", type="password", help="Enable to compare Polygon vs Yahoo Data.", key="polygon_api_key_input")
    period = st.selectbox("Training Period", ['1mo', '3mo', '6mo', '1y', '2y', '3y', '5y'], index=6)
    interval = st.selectbox("Interval", ['1d', '60m', '30m', '15m', '5m'], index=0, help="Bar size. Intraday (min/hour) has limited history (e.g. 1m=7d, 1h=730d).")
    
    if st.button("💾 Save Sidebar Settings"):
        import json
        with open("user_settings.json", "w") as f:
            json.dump({"polygon_api_key": polygon_api_key, "ticker": ticker}, f)
        st.success("Settings Saved!")
    
    st.markdown("---")
    st.subheader("🎯 Training Options")
    model_type = st.selectbox("Model Type", ['xgboost', 'random_forest'])
    use_smote = st.checkbox("Use SMOTE Balancing", value=True)
    use_optuna = st.checkbox("Use Optuna Optimization", value=True)
    n_trials = st.slider("Optuna Trials", 10, 5000, 30) if use_optuna else 10
    
    # Optimization metric selection
    optimize_metric = st.selectbox("Optimize For", 
        ['f1_weighted', 'recall', 'accuracy', 'f1_macro'],
        help="recall: Maximize finding ALL Buy signals (good for not missing moves). f1_weighted: Balanced approach.")
    
    n_cv_splits = st.slider("CV Folds", 3, 10, 5, 
                           help="Number of TimeSeriesSplit folds for cross-validation")
    
    # Advanced Model Config
    with st.expander("⚙️ Advanced Model Config"):
        st.caption("Fine-tune how aggressive the model is")
        class_weight_ratio = st.slider("Class Weight Ratio", 1.0, 10.0, 1.0, 0.5,
                                      help="Higher = penalize missing BUY/SELL signals more heavily. 1.0 = Equal weights.")
        decision_threshold = st.slider("Decision Threshold", 0.3, 0.9, 0.5, 0.05,
                                      help="Lower = more aggressive signals. Higher = higher confidence required.")
        use_regime_detection = st.checkbox("🧠 Use Market Regime Detection", value=True, 
                                          help="Detects Bull/Bear/Sideways markets using Gaussian Mixture Models.")
    
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

# -----------------------------------------------------------------------------
# GLOBAL DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_global_data(t, p, i, poly_key):
    if poly_key and PolygonManager:
         try:
             pm = PolygonManager(poly_key)
             df = pm.get_historical_data(t, period=p, interval=i)
             if not df.empty: return df
         except: pass
    return load_price_data(t, p, i)

data = load_global_data(ticker, period, interval, polygon_api_key if polygon_api_key else None)
features = pd.DataFrame()

if not data.empty:
    # Generate base features globally
    features = generate_features(data, use_regime=use_regime_detection)
else:
    st.error(f"No data found for {ticker}")

# Main tabs
tab_optimized, tab_simple, tab_opt, tab1, tab2, tab3, tab4 = st.tabs(["🔬 Optimized", "⚡ Simple", "🧪 Master Opt.", "🎯 Manual Train", "🚀 Production", "📊 Analysis", "🧪 Alpha Lab"])

# =============================================================================
# TAB OPTIMIZED: OPTIMIZED STRATEGY (Optuna + Proper Train/Test)
# =============================================================================
with tab_optimized:
    try:
        from optimized_strategy import render_optimized_strategy_page
        render_optimized_strategy_page(data, features)
    except ImportError as e:
        st.error(f"Optimized Strategy module not available: {e}")
    except Exception as e:
        st.error(f"Error loading Optimized Strategy: {e}")
        import traceback
        st.code(traceback.format_exc())

# =============================================================================
# TAB SIMPLE: SIMPLE STRATEGY (Rule-Based + ML Enhancement)
# =============================================================================
with tab_simple:
    st.header("⚡ Simple Strategy")
    st.info("**Rule-based signals with ML enhancement** • Proven to work • No overfitting • Regime-aware")
    
    # Import the simple strategy module
    try:
        from simple_strategy import (
            generate_rule_signals, 
            generate_hybrid_signals,
            walk_forward_backtest,
            analyze_signal_quality,
            calculate_efficiency
        )
        SIMPLE_AVAILABLE = True
    except ImportError as e:
        SIMPLE_AVAILABLE = False
        st.error(f"Simple Strategy module not available: {e}")
    
    if SIMPLE_AVAILABLE and not data.empty:
        # Configuration columns
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        
        with col_cfg1:
            st.subheader("📈 BUY Conditions")
            simple_buy_rsi = st.slider("RSI Oversold Threshold", 20, 45, 35, key="simple_buy_rsi",
                                       help="Buy when RSI drops below this level")
            simple_buy_comp = st.slider("Composite Oversold", -0.8, 0.0, -0.3, 0.1, key="simple_buy_comp",
                                        help="Buy when composite oscillator drops below this")
            simple_require_multi = st.checkbox("Require 2+ oversold indicators", value=True, key="simple_multi",
                                               help="More conservative: requires multiple confirmations")
        
        with col_cfg2:
            st.subheader("📉 SELL Conditions")
            simple_sell_rsi = st.slider("RSI Overbought Threshold", 60, 85, 70, key="simple_sell_rsi",
                                        help="Sell when RSI rises above this level")
            simple_sell_comp = st.slider("Composite Overbought", 0.2, 0.8, 0.5, 0.1, key="simple_sell_comp",
                                         help="Sell when composite oscillator rises above this")
            simple_regime_aware = st.checkbox("Only sell in Bear markets", value=True, key="simple_regime",
                                              help="In bull markets, only sell on extreme overbought (recommended)")
        
        with col_cfg3:
            st.subheader("🎛️ Strategy Mode")
            simple_mode = st.radio("Signal Generation", 
                                   ["Rule-Based (Recommended)", "Hybrid (Rules + ML)"],
                                   key="simple_mode",
                                   help="Rule-based uses proven indicators. Hybrid adds ML confirmation.")
            
            if "Hybrid" in simple_mode:
                simple_ml_confirm = st.slider("ML Confirmation Threshold", 0.1, 0.7, 0.3, 0.1, key="simple_ml_conf",
                                              help="Minimum ML confidence to confirm rule-based signal")
            else:
                simple_ml_confirm = 0.3
        
        # Build config
        simple_config = {
            'buy_rsi_thresh': simple_buy_rsi,
            'buy_composite_thresh': simple_buy_comp,
            'sell_rsi_thresh': simple_sell_rsi,
            'sell_composite_thresh': simple_sell_comp,
            'require_multiple_oversold': simple_require_multi,
            'regime_aware': simple_regime_aware,
            'ml_buy_confirm': simple_ml_confirm,
            'ml_sell_confirm': simple_ml_confirm,
        }
        
        # Run button
        if st.button("🚀 Generate Signals & Backtest", type="primary", key="simple_run"):
            with st.spinner("Generating signals..."):
                # Generate signals
                simple_signals = generate_rule_signals(data, features, simple_config)
                
                # Signal summary
                buy_count = (simple_signals == 1).sum()
                sell_count = (simple_signals == -1).sum()
                
                st.success(f"✅ Generated {buy_count} BUY and {sell_count} SELL signals")
            
            # Signal Quality Analysis
            with st.spinner("Analyzing signal quality..."):
                quality = analyze_signal_quality(data, simple_signals)
            
            st.markdown("---")
            st.subheader("🎯 Signal Quality (Forward Returns)")
            
            col_q1, col_q2 = st.columns(2)
            
            with col_q1:
                st.markdown("**BUY Signals**")
                if quality['buy_stats']:
                    bs = quality['buy_stats']
                    st.metric("5-Day Avg Return", f"{bs['5d_return']:.2f}%")
                    st.metric("5-Day Win Rate", f"{bs['5d_win_rate']:.1f}%")
                    st.caption(f"1d: {bs['1d_return']:.2f}% | 3d: {bs['3d_return']:.2f}% | 10d: {bs['10d_return']:.2f}%")
                else:
                    st.warning("No BUY signals generated")
            
            with col_q2:
                st.markdown("**SELL Signals**")
                if quality['sell_stats']:
                    ss = quality['sell_stats']
                    st.metric("5-Day Avg Return", f"{ss['5d_return']:.2f}%")
                    st.metric("5-Day Win Rate", f"{ss['5d_win_rate']:.1f}%")
                    st.caption(f"1d: {ss['1d_return']:.2f}% | 3d: {ss['3d_return']:.2f}% | 10d: {ss['10d_return']:.2f}%")
                else:
                    st.warning("No SELL signals generated")
            
            # Run Backtest
            with st.spinner("Running walk-forward backtest..."):
                simple_backtest = walk_forward_backtest(data, simple_signals)
            
            st.markdown("---")
            st.subheader("📈 Backtest Results")
            
            col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
            
            col_b1.metric("Strategy Return", f"{simple_backtest['total_return']:.1f}%",
                         f"{simple_backtest['alpha']:+.1f}% vs B&H")
            col_b2.metric("Buy & Hold", f"{simple_backtest['buy_hold_return']:.1f}%")
            col_b3.metric("Win Rate", f"{simple_backtest['win_rate']:.1f}%")
            col_b4.metric("Max Drawdown", f"{simple_backtest['max_drawdown']:.1f}%")
            col_b5.metric("Trades", simple_backtest['num_trades'])
            
            # Efficiency
            efficiency = calculate_efficiency(data, simple_signals)
            st.metric("⚡ Efficiency", f"{efficiency['efficiency']:.1f}%",
                     help=f"Capturing {efficiency['efficiency']:.1f}% of theoretical maximum ({efficiency['theoretical_max']:.0f}%)")
            
            # Chart
            st.subheader("📉 Price Chart with Signals")
            
            fig_simple = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Candlestick
            fig_simple.add_trace(go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ), row=1, col=1)
            
            # BUY signals
            buy_dates = simple_signals[simple_signals == 1].index
            if len(buy_dates) > 0:
                buy_prices = data.loc[buy_dates, 'low'] * 0.98
                fig_simple.add_trace(go.Scatter(
                    x=buy_dates, y=buy_prices, mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='lime',
                               line=dict(width=1, color='darkgreen')),
                    name='BUY Signal'
                ), row=1, col=1)
            
            # SELL signals
            sell_dates = simple_signals[simple_signals == -1].index
            if len(sell_dates) > 0:
                sell_prices = data.loc[sell_dates, 'high'] * 1.02
                fig_simple.add_trace(go.Scatter(
                    x=sell_dates, y=sell_prices, mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red',
                               line=dict(width=1, color='darkred')),
                    name='SELL Signal'
                ), row=1, col=1)
            
            # Equity curve
            if len(simple_backtest['equity_curve']) > 0:
                eq = simple_backtest['equity_curve']
                fig_simple.add_trace(go.Scatter(
                    x=eq['date'], y=eq['equity'],
                    mode='lines', name='Strategy Equity',
                    line=dict(color='cyan', width=2)
                ), row=2, col=1)
                
                # Buy & Hold line
                bh_equity = 100000 * (data['close'] / data['close'].iloc[0])
                fig_simple.add_trace(go.Scatter(
                    x=data.index, y=bh_equity,
                    mode='lines', name='Buy & Hold',
                    line=dict(color='gray', width=1, dash='dash')
                ), row=2, col=1)
            
            fig_simple.update_layout(
                height=700,
                template='plotly_dark',
                xaxis_rangeslider_visible=False,
                showlegend=True,
                title=f"Simple Strategy - {ticker}"
            )
            fig_simple.update_yaxes(title_text="Price", row=1, col=1)
            fig_simple.update_yaxes(title_text="Equity ($)", row=2, col=1)
            
            st.plotly_chart(fig_simple, use_container_width=True)
            
            # Trade List
            with st.expander("📋 Trade Details"):
                if simple_backtest['trades']:
                    trades_df = pd.DataFrame(simple_backtest['trades'])
                    st.dataframe(trades_df, use_container_width=True)
                else:
                    st.write("No trades executed")
            
            # Save to session state
            st.session_state['simple_signals'] = simple_signals
            st.session_state['simple_backtest'] = simple_backtest
            st.session_state['simple_config'] = simple_config
        
        # Show comparison with current ML model if available
        if 'v2_backtest' in st.session_state and st.session_state.v2_backtest is not None:
            st.markdown("---")
            st.subheader("📊 Comparison: Simple vs ML Model")
            
            ml_bt = st.session_state.v2_backtest
            
            col_cmp1, col_cmp2 = st.columns(2)
            
            with col_cmp1:
                st.markdown("**Current ML Model**")
                if isinstance(ml_bt, dict):
                    st.write(f"Return: {ml_bt.get('total_return', 0):.1f}%")
                    st.write(f"Trades: {ml_bt.get('num_trades', 0)}")
                    st.write(f"Win Rate: {ml_bt.get('win_rate', 0):.1f}%")
                else:
                    st.write("ML backtest data not available")
            
            with col_cmp2:
                st.markdown("**Simple Strategy**")
                if 'simple_backtest' in st.session_state and st.session_state.simple_backtest is not None:
                    sbt = st.session_state.simple_backtest
                    st.write(f"Return: {sbt['total_return']:.1f}%")
                    st.write(f"Trades: {sbt['num_trades']}")
                    st.write(f"Win Rate: {sbt['win_rate']:.1f}%")
                else:
                    st.write("Run Simple Strategy first")

# =============================================================================
# TAB OPT: MASTER OPTIMIZATION
# =============================================================================
with tab_opt:
    st.header("🧪 Master Strategy Optimization")
    st.info("Automated Strategy Finder: Optimizes Model + Signals + Filters in one go.")
    
    col_opt1, col_opt2 = st.columns([1, 2])
    
    with col_opt1:
        st.subheader("Settings")
        n_trials_opt = st.slider("Optimization Trials", 10, 5000, 100, help="More trials = Better results but slower.")
        
        with st.expander("📋 Optimization Scope (What Gets Tuned)", expanded=True):
            st.markdown("#### 🤖 XGBoost Model Hyperparameters (9 params)")
            st.markdown("""
            | Parameter | Range | Description |
            |-----------|-------|-------------|
            | `n_estimators` | 50 - 2000 | Number of boosting rounds |
            | `max_depth` | 2 - 15 | Tree depth (complexity) |
            | `learning_rate` | 0.001 - 0.5 | Step size (log scale) |
            | `subsample` | 0.5 - 1.0 | Row sampling ratio |
            | `colsample_bytree` | 0.5 - 1.0 | Feature sampling ratio |
            | `gamma` | 0 - 5 | Min loss reduction for split |
            | `min_child_weight` | 1 - 10 | Min samples per leaf |
            | `reg_alpha` | 0.001 - 10 | L1 regularization (log) |
            | `reg_lambda` | 0.001 - 10 | L2 regularization (log) |
            """)
            
            st.markdown("#### 🎛️ Training Settings (5 params)")
            st.markdown("""
            | Parameter | Range | Description |
            |-----------|-------|-------------|
            | `peak_valley_order` | 3 - 10 | Peak/Valley detection sensitivity |
            | `class_weight_ratio` | 1.0 - 5.0 | Penalize missing BUY/SELL |
            | `decision_threshold` | 0.3 - 0.7 | Confidence required for signals |
            | `use_smote` | True/False | SMOTE class balancing |
            | `use_regime_detection` | True/False | Market regime features |
            """)
            
            st.markdown("#### 📊 ML Signal Thresholds (2 params)")
            st.markdown("""
            | Parameter | Range | Description |
            |-----------|-------|-------------|
            | `ml_buy_thresh` | 0.20 - 0.90 | ML confidence for BUY |
            | `ml_sell_thresh` | 0.20 - 0.90 | ML confidence for SELL |
            """)
            
            st.markdown("#### 📈 Slope/Composite Signals (5 params)")
            st.markdown("""
            | Parameter | Range | Description |
            |-----------|-------|-------------|
            | `use_slope_signals` | True/False | Enable slope-based signals |
            | `slope_buy` | -0.9 to -0.1 | Composite buy trigger |
            | `slope_sell` | 0.1 - 0.9 | Composite sell trigger |
            | `slope_roc_thresh` | 0.0 - 0.3 | Min slope acceleration |
            | `use_mom_zone` | True/False | Allow momentum zone buys |
            """)
            
            st.markdown("#### 🧠 ML Confirmation (2 params)")
            st.markdown("""
            | Parameter | Range | Description |
            |-----------|-------|-------------|
            | `use_ml_confirm` | True/False | Require ML confirmation |
            | `ml_confirm_thresh` | 0.10 - 0.50 | ML prob threshold |
            """)
            
            st.markdown("#### 🛡️ Market Filters (3 params)")
            st.markdown("""
            | Parameter | Range | Description |
            |-----------|-------|-------------|
            | `use_adx_filter` | True/False | Block signals in chop |
            | `adx_threshold` | 15 - 35 | Min ADX for trending |
            | `use_trend_filter` | True/False | Only long in uptrend |
            """)
            
            st.success("**Total: 27 parameters optimized simultaneously!**")
            
            st.markdown("#### ⚠️ Fixed Parameters (Not Optimized)")
            st.markdown("""
            - Conformal Prediction alpha (set in Production tab)
            - Stop Loss / Take Profit levels
            - Deep Learning feature extraction settings
            """)
        
        # Display last optimization results if available (persisted in session state)
        if 'v2_last_optimization' in st.session_state and st.session_state.v2_last_optimization:
            st.divider()
            st.subheader("📊 Last Optimization Results")
            last_opt = st.session_state.v2_last_optimization
            
            st.metric("Best Return", f"{last_opt.get('best_return', 0):.2%}")
            st.caption(f"Optimized at: {last_opt.get('timestamp', 'Unknown')}")
            
            # Display full model filename
            model_path = last_opt.get('model_path', '')
            if model_path:
                model_filename = os.path.basename(model_path)
                st.info(f"**Model:** `{model_filename}`")
                st.code(model_path, language=None)
            
            with st.expander("View Best Parameters", expanded=False):
                st.json(last_opt.get('best_params', {}))
            
            if st.button("🔄 Clear Saved Results"):
                del st.session_state.v2_last_optimization
                st.rerun()
        
    with col_opt2:
        if st.button("🚀 Run Full Strategy Optimization", type="primary", use_container_width=True):
             progress_bar = st.progress(0)
             status_text = st.empty()
             
             try:
                 # 1. Prepare Data
                 status_text.text("Preparing Data & Features...")
                 
                 # Recalculate labels with current sensitivity
                 labels, info = detect_peaks_valleys(data, order=detection_order)
                 
                 # Robust Alignment
                 common_idx = data.index.intersection(features.index).intersection(labels.index)
                 
                 # Align Features
                 f_aligned = features.loc[common_idx]
                 l_aligned = labels.loc[common_idx]
                 d_aligned = data.loc[common_idx]
                 
                 # Drop NaN (Important for ML)
                 valid_mask = f_aligned.notna().all(axis=1) & l_aligned.notna()
                 valid_idx = f_aligned.index[valid_mask]
                 
                 X_opt = f_aligned.loc[valid_idx]
                 y_opt = l_aligned.loc[valid_idx]
                 d_opt = d_aligned.loc[valid_idx]
                 
                 if len(X_opt) < 100:
                     st.error("Not enough valid data points for optimization. Try a longer period.")
                     st.stop()
                 
                 # 2. Run Optimization
                 status_text.text(f"Running Optuna Study ({n_trials_opt} trials)...")
                 
                 def prog_cb(i, n):
                     progress_bar.progress(i / n)
                     status_text.text(f"Optimization Progress: Trial {i}/{n}")
                     
                 best_params = optimize_full_strategy(d_opt, X_opt, y_opt, n_trials=n_trials_opt, progress_callback=prog_cb, raw_data=data)
                 
                 progress_bar.progress(1.0)
                 status_text.text("Optimization Complete!")
                 
                 # 3. Save & Apply
                 st.success("✅ Best Strategy Found!")
                 st.json(best_params)
                 
                 # Train Final Model with Best Params
                 status_text.text("Training Final Model...")
                 model_params = {k:v for k,v in best_params.items() if k in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'n_jobs']}
                 
                 # Prepare Labels for XGBoost (Must be 0, 1, 2...)
                 from sklearn.preprocessing import LabelEncoder, StandardScaler
                 le = LabelEncoder()
                 y_opt_encoded = le.fit_transform(y_opt)
                 
                 # Add Scaling (Required by downstream app logic)
                 scaler = StandardScaler()
                 X_opt_scaled = scaler.fit_transform(X_opt)
                 
                 model_params['objective'] = 'multi:softprob'
                 model_params['num_class'] = len(le.classes_)
                 
                 import xgboost as xgb
                 final_model = xgb.XGBClassifier(**model_params)
                 final_model.fit(X_opt_scaled, y_opt_encoded)
                 
                 # Save
                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                 model_filename = f"xgboost_{ticker}_{timestamp}_OPTIMIZED.joblib"
                 save_path = os.path.join("saved_models_v2", model_filename)
                 
                 # Metadata
                 meta = {
                    'model': final_model,
                    'scaler': scaler,
                    'label_encoder': le,
                    'feature_names': X_opt.columns.tolist(),
                    'ticker': ticker,
                    'best_params': best_params, # Contains thresholds too!
                    'model_type': 'xgboost',
                    'detection_order': detection_order,
                    'label_classes': le.classes_ 
                 }
                 
                 # Also calculate performance metrics for display
                 from sklearn.metrics import accuracy_score, f1_score
                 y_pred = final_model.predict(X_opt_scaled)
                 meta['accuracy'] = accuracy_score(y_opt_encoded, y_pred)
                 meta['f1_score'] = f1_score(y_opt_encoded, y_pred, average='weighted')
                 meta['cv_score'] = best_params.get('best_value', 0.0) # We don't have this easily, skip or approx
                 
                 joblib.dump(meta, save_path)
                 
                 # Calculate Theoretical Stats for UI
                 theoretical_for_period = calculate_theoretical_return(
                    data, info['peak_dates'], info['valley_dates']
                 )
                 
                 # Update Session State FULLY
                 st.session_state.v2_model_path = save_path
                 st.session_state.v2_model = meta
                 st.session_state.v2_data = data
                 st.session_state.v2_features = features
                 st.session_state.v2_labels = labels
                 st.session_state.v2_detection_info = info
                 st.session_state.v2_theoretical = theoretical_for_period
                 st.session_state.v2_trained = True
                 
                 # Update Sidebar Thresholds in Session State (Apply Callbacks)
                 if 'slope_buy' in best_params: st.session_state.s_buy = float(best_params['slope_buy'])
                 if 'slope_sell' in best_params: st.session_state.s_sell = float(best_params['slope_sell'])
                 if 'ml_confirm' in best_params: 
                     conf = float(best_params['ml_confirm'])
                     
                     # 1. Update Session State Variables
                     st.session_state.s_ml_conf_thresh = conf
                     st.session_state.s_buy = float(best_params.get('slope_buy', -0.5))
                     st.session_state.s_sell = float(best_params.get('slope_sell', 0.5))
                     
                     # 2. CRITICAL: Update the Primary ML Thresholds (used by UI Backtest)
                     st.session_state.s_ml_buy = conf
                     st.session_state.s_ml_sell = conf
                     
                     # 3. Update Widget Keys directly (to force Sidebar update)
                     st.session_state["s_ml_buy"] = conf
                     st.session_state["s_ml_sell"] = conf
                     st.session_state["decision_threshold_slider"] = conf # Legacy
                 
                 # Save optimization results to session state for persistence across tab navigation
                 st.session_state.v2_last_optimization = {
                     'best_params': best_params,
                     'best_return': best_params.get('best_value', 0.0) if isinstance(best_params, dict) else 0.0,
                     'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     'model_path': save_path,
                     'model_filename': model_filename,
                     'ticker': ticker,
                     'n_trials': n_trials_opt
                 }
                 
                 # PERSIST to file so it survives browser refresh
                 import json
                 opt_cache_path = os.path.join("saved_models_v2", "last_optimization.json")
                 try:
                     with open(opt_cache_path, 'w') as f:
                         json.dump(st.session_state.v2_last_optimization, f, indent=2)
                 except Exception as e:
                     print(f"Warning: Could not save optimization cache: {e}")
                 
                 st.success(f"💾 Optimized Model Saved: `{model_filename}`")
                 st.success(f"✅ Thresholds updated to {best_params.get('ml_confirm', 0.5):.2f}. Go to 'Production' to Bundle!")
                 st.info(f"📁 Full path: `{save_path}`")
                 st.balloons()
                 
                 # NOTE: Removed st.rerun() - session state is saved, user can manually refresh if needed
                 
             except Exception as e:
                 st.error(f"Optimization Failed: {e}")

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
    
    use_dl_features = st.checkbox("🧠 Use Deep Learning Features (CNN/LSTM)", value=False, 
                                 help="Train a Multi-Scale CNN Autoencoder to extract latent pattern features. Slower but captures complex shapes.")

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        progress = st.progress(0, text="Starting...")
        
        try:
            # 1. Load data
            progress.progress(10, text="Loading price data...")
            data = load_price_data(ticker, period, interval=interval)
            st.session_state.v2_data = data
            st.session_state.v2_ticker = ticker
            
            # 2. Detect peaks/valleys
            progress.progress(20, text="Detecting peaks and valleys...")
            labels, detection_info = detect_peaks_valleys(data, order=detection_order)
            st.session_state.v2_labels = labels
            st.session_state.v2_detection_info = detection_info
            
            # 3. Calculate theoretical return
            theoretical = calculate_theoretical_return(data, detection_info['peak_dates'], 
                                                       detection_info['valley_dates'])
            st.session_state.v2_theoretical = theoretical
            
            # 4. Generate features
            progress.progress(30, text="Generating features...")
            dl_config = None
            if use_dl_features:
                os.makedirs("saved_models_v2", exist_ok=True)
                dl_config = {'train': True, 'model_path': f"saved_models_v2/dl_{ticker}.h5"}
                
            features = generate_features(data, dl_config=dl_config, use_regime=use_regime_detection)
            st.session_state.v2_features = features
            
            # 5. Train model
            progress.progress(40, text=f"Training {model_type} with {'Optuna' if use_optuna else 'defaults'}...")
            
            model_data = train_model_with_optuna(
                features, labels, model_type, use_smote,
                n_trials=n_trials if use_optuna else 1,
                progress_callback=None,
                optimize_metric=optimize_metric,
                n_cv_splits=n_cv_splits,
                class_weight_ratio=class_weight_ratio,
                optimize_training_params=use_optuna
            )
            
            model_data['use_dl'] = use_dl_features
            
            # 6. Quick Evaluation Backtest (Test Set)
            try:
                split_idx = int(len(features) * 0.8)
                test_features = features.iloc[split_idx:]
                test_data = data.loc[test_features.index]
                q_sigs, _ = generate_signals(model_data, test_features, threshold=decision_threshold)
                q_bt = run_backtest(test_data, q_sigs, verbose=False)
                model_data['total_return'] = q_bt['total_return']
            except Exception as e:
                print(f"Warning: Quick backtest failed: {e}")
            
            st.session_state.v2_model = model_data
            
            # 7. Save Model
            progress.progress(95, text="Saving model...")
            filepath = save_model(model_data, ticker)
            st.session_state.v2_model_path = filepath
            
            st.session_state.v2_trained = True
            progress.progress(100, text="Complete!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Training Failed: {e}")
            
    # PERSISTENT DISPLAY BLOCK
    if st.session_state.get('v2_trained'):
        # Retrieve State
        data = st.session_state.v2_data
        detection_info = st.session_state.v2_detection_info
        theoretical = st.session_state.v2_theoretical
        features = st.session_state.v2_features
        labels = st.session_state.v2_labels
        model_data = st.session_state.v2_model
        
        st.success(f"✅ Model Trained & Saved: `{st.session_state.get('v2_model_path', '')}`")
        
        # Display Stats
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Data Points", len(data))
        with col2: st.metric("Peaks Detected", detection_info['num_peaks'])
        with col3: st.metric("Valleys Detected", detection_info['num_valleys'])
        
        st.markdown("### 🎯 Theoretical Maximum")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Perfect Return", f"{theoretical['total_return']:,.1f}%")
        with col2: st.metric("Perfect Trades", theoretical['num_trades'])
        with col3: st.metric("Avg Trade Return", f"{theoretical['avg_trade_return']:.1f}%")
        with col4:
            bh_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100
            st.metric("Buy & Hold", f"{bh_return:.1f}%")
            
        st.info(f"Generated **{len(features.columns)}** features")
        
        label_counts = labels.value_counts()
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("BUY Labels", label_counts.get(1, 0))
        with col2: st.metric("HOLD Labels", label_counts.get(0, 0))
        with col3: st.metric("SELL Labels", label_counts.get(-1, 0))
        
        st.markdown("### 📊 Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Accuracy", f"{model_data['accuracy']:.1%}")
        with col2: st.metric("F1 Score", f"{model_data['f1_score']:.1%}")
        with col3: st.metric("CV Score", f"{model_data['cv_score']:.1%}")
        with col4: st.metric("Features", len(model_data['feature_names']))
        
        with st.expander("🔧 Best Hyperparameters"):
            st.json(model_data['best_params'])

# =============================================================================
# TAB 2: PRODUCTION
# =============================================================================
with tab2:
    st.header("Production Trading")
    
    # 1. Select Base Model
    model_dir = "saved_models_v2"
    strategies_dir = "strategies"
    
    # Force refresh button for model list
    col_refresh, col_spacer = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Refresh Models", help="Refresh the model list to see newly trained models"):
            st.rerun()
    
    # Collect Raw Models (re-scan directory each time)
    raw_models = []
    if os.path.exists(model_dir):
        raw_models = sorted([f"raw/{f}" for f in os.listdir(model_dir) if f.endswith('.joblib')], reverse=True)
        
    # Collect Bundles
    bundles = []
    if os.path.exists(strategies_dir):
        # List directories in strategies/
        bundles = sorted([f"bundle/{d}" for d in os.listdir(strategies_dir) if os.path.isdir(os.path.join(strategies_dir, d))], reverse=True)
    
    all_options = bundles + raw_models
    
    if not all_options:
        st.warning("⚠️ No models or bundles found.")
        st.stop()
        
    selected_option = st.selectbox("Select Strategy / Model", all_options)
    
    # Resolve Path
    if selected_option.startswith("bundle/"):
        bundle_name = selected_option.replace("bundle/", "")
        # Find the model file inside the bundle
        bundle_path = os.path.join(strategies_dir, bundle_name)
        # Look for .joblib in bundle
        model_files = [f for f in os.listdir(bundle_path) if f.endswith('.joblib') and 'alpha' not in f]
        if model_files:
            model_path = os.path.join(bundle_path, model_files[0])
        else:
            st.error(f"No model found in bundle {bundle_name}")
            st.stop()
    else:
        # Raw model
        model_name = selected_option.replace("raw/", "")
        model_path = os.path.join(model_dir, model_name)
    
    # Show selected backtest mode info
    st.info(f"📅 **Backtest Mode:** {backtest_mode}")
    
    # =========================================================================
    # APPLY MASTER OPT PARAMS BUTTON (Prominent Location)
    # =========================================================================
    if 'v2_last_optimization' in st.session_state and st.session_state.v2_last_optimization:
        last_opt = st.session_state.v2_last_optimization
        with st.container():
            col_opt_a, col_opt_b = st.columns([3, 1])
            with col_opt_a:
                st.success(f"📊 **Master Opt Available:** {last_opt.get('timestamp', 'Unknown')} | Best Return: {last_opt.get('best_return', 0):.2%} | Model: `{last_opt.get('model_filename', 'N/A')}`")
            with col_opt_b:
                if st.button("⚡ Apply All Params", type="primary", use_container_width=True, help="Apply ALL best parameters found in Master Optimization"):
                    params = last_opt.get('best_params', {})
                    applied_count = 0
                    
                    # ML THRESHOLDS
                    if 'ml_buy_thresh' in params:
                        st.session_state.s_ml_buy = float(params['ml_buy_thresh'])
                        applied_count += 1
                    if 'ml_sell_thresh' in params:
                        st.session_state.s_ml_sell = float(params['ml_sell_thresh'])
                        applied_count += 1
                    
                    # SLOPE SIGNALS
                    if 'use_slope_signals' in params:
                        st.session_state.s_slope_bool = params['use_slope_signals']
                        applied_count += 1
                    if 'slope_buy' in params:
                        st.session_state.s_buy = float(params['slope_buy'])
                        applied_count += 1
                    if 'slope_sell' in params:
                        st.session_state.s_sell = float(params['slope_sell'])
                        applied_count += 1
                    if 'slope_roc_thresh' in params:
                        st.session_state.s_roc = float(params['slope_roc_thresh'])
                        applied_count += 1
                    if 'use_mom_zone' in params:
                        st.session_state.s_mom = params['use_mom_zone']
                        applied_count += 1
                    
                    # ML CONFIRMATION
                    if 'use_ml_confirm' in params:
                        st.session_state.s_ml_conf_bool = params['use_ml_confirm']
                        applied_count += 1
                    if 'ml_confirm_thresh' in params:
                        st.session_state.s_ml_conf_thresh = float(params['ml_confirm_thresh'])
                        applied_count += 1
                    
                    # MARKET FILTERS
                    if 'use_adx_filter' in params:
                        st.session_state.s_adx_bool = params['use_adx_filter']
                        applied_count += 1
                    if 'adx_threshold' in params:
                        st.session_state.s_adx_thresh = int(params['adx_threshold'])
                        applied_count += 1
                    if 'use_trend_filter' in params:
                        st.session_state.s_trend_bool = params['use_trend_filter']
                        applied_count += 1
                    
                    # TRAINING SETTINGS
                    if 'peak_valley_order' in params:
                        st.session_state.s_pv_order = int(params['peak_valley_order'])
                        applied_count += 1
                    if 'use_smote' in params:
                        st.session_state.s_use_smote = params['use_smote']
                        applied_count += 1
                    if 'use_regime_detection' in params:
                        st.session_state.s_use_regime = params['use_regime_detection']
                        applied_count += 1
                    
                    # CONFORMAL PREDICTION
                    if 'use_conformal' in params:
                        st.session_state.s_cp_bool = params['use_conformal']
                        applied_count += 1
                    if 'cp_alpha' in params:
                        st.session_state.s_cp_alpha = float(params['cp_alpha'])
                        applied_count += 1
                    
                    st.success(f"✅ Applied {applied_count} optimized parameters!")
                    st.rerun()
    else:
        st.warning("💡 **Tip:** Run Master Optimization first to get optimized parameters, then click 'Apply All Params' here.")
    
    # Execution Settings
    st.markdown("### ⚙️ Signal Filters & Sensitivity")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1. Market Filters**")
        
        use_trend_filter = st.checkbox("✅ Use Trend Filter (SMA200)", value=False, key="s_trend",
                                      help="Block SELLS in Bull Market unless extremely overbought.")
        
        use_adx_filter = st.checkbox("✅ Use Volatility Filter (ADX)", value=False, key="s_adx_bool",
                                    help="Block ALL TRADES in Choppy/Sideways Market (Low ADX).")
        
        use_conformal = st.checkbox("🔬 Use Conformal Prediction (Uncertainty)", value=False, key="s_cp_bool",
                                   help="Rigorous mathematical filter. Blocks trades if the model is 'Uncertain' (Set Size > 1).")
        
        adx_threshold = 20
        if use_adx_filter:
             adx_threshold = st.slider("Min ADX Threshold", 10, 40, 20, 1, key="s_adx_thresh")
             
        cp_alpha = 0.1
        if use_conformal:
            cp_alpha = st.slider("Significance Level (Alpha)", 0.01, 0.20, 0.10, 0.01, key="s_cp_alpha",
                                help="0.10 = 90% Confidence. Lower = Stricter.")
             
        limit_entry_pct = st.slider("Entry Limit Offset %", 0.0, 3.0, 0.0, 0.1, key="s_limit",
                                   help="Try to buy lower than signal price. 0 = Market Order.") / 100

    with col2:
        st.markdown("**2. AI Confidence**")
        # ML Thresholds
        ml_buy_thresh = st.slider("ML Buy Confidence", 0.2, 0.95, 0.5, 0.05, key="s_ml_buy", help="Minimum Probability to Trigger BUY")
        ml_sell_thresh = st.slider("ML Sell Confidence", 0.2, 0.95, 0.5, 0.05, key="s_ml_sell", help="Minimum Probability to Trigger SELL")

    # Slope Signals (Advanced/Legacy)
    with st.expander("Advanced: Composite Slope Signals"):
        use_slope_signals = st.checkbox("Enable Slope Turn Signals", value=False, key="s_slope_bool",
                                      help="Force Buy on Composite Turn Up (Valley), Force Sell on Turn Down (Peak). Overrides ML.")
        
        slope_buy_thresh = 0.0
        slope_sell_thresh = 0.0
        use_ml_confirm = False
        ml_confirm_thresh = 0.3
        slope_roc_thresh = 0.0
        use_mom_zone = False
        
        if use_slope_signals:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                slope_buy_thresh = st.slider("Buy Thresh (Comp < X)", -1.0, 0.0, -0.5, 0.1, key="s_buy")
            with col_s2:
                slope_sell_thresh = st.slider("Sell Thresh (Comp > X)", 0.0, 1.0, 0.5, 0.1, key="s_sell")
            
            use_mom_zone = st.checkbox("🚀 Allow Momentum Zone (0.2 - 0.3)", value=False, key="s_mom", help="Buy high-probability breakouts.")
            slope_roc_thresh = st.slider("Min Slope Accel (ROC)", 0.0, 0.5, 0.0, 0.05, key="s_roc", help="Filter out lazy turns.")
            
            use_ml_confirm = st.checkbox("🧠 Use ML Confirmation", value=False, key="s_ml_conf_bool", help="Only take Slope Signal if ML Probability > Threshold")
            if use_ml_confirm:
                ml_confirm_thresh = st.slider("ML Confirm Prob", 0.0, 0.5, 0.10, 0.01, key="s_ml_conf_thresh")
    
    # Strategy Auto-Tuner
    with st.expander("🤖 Auto-Tune Strategy Parameters"):
        st.info("Find the optimal slider settings (Buy/Sell Thresholds, Momentum, ML Confirm) for the selected period.")
        tune_trials = st.slider("Optimization Trials", 10, 5000, 30)
        if st.button("Start Optimization"):
            # Progress bar
            prog_bar = st.progress(0, text="Initializing optimization...")
            
            def update_prog(current, total):
                pct = min(100, int(100 * current / total))
                prog_bar.progress(pct, text=f"Strategy Optimization: Trial {current}/{total}")

            with st.spinner("Optimizing strategy parameters..."):
                # Load model
                model_data = load_model(model_path)
                model_ticker = model_data.get('ticker', ticker)
                
                import yfinance as yf
                
                # Determine date range
                if backtest_mode == "Full Training Period":
                    data_opt = yf.Ticker(model_ticker).history(period=period, interval=interval)
                elif backtest_mode == "Test Period Only":
                    data_opt = yf.Ticker(model_ticker).history(period=period, interval=interval)
                    split_idx = int(len(data_opt) * 0.8)
                    data_opt = data_opt.iloc[split_idx:]
                elif backtest_mode == "Recent Days":
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=prod_days + 100)
                    data_opt = yf.Ticker(model_ticker).history(start=start_date, end=end_date, interval=interval)
                else:
                    data_opt = yf.Ticker(model_ticker).history(start=custom_start, end=custom_end, interval=interval)
                
                data_opt.index = pd.to_datetime(data_opt.index).tz_localize(None)
                data_opt.columns = [c.lower() for c in data_opt.columns]
                data_opt = data_opt[['open', 'high', 'low', 'close', 'volume']]
                
                # Check if model uses DL
                use_dl = model_data.get('use_dl', False)
                # Fallback: If metadata missing, check feature names
                if not use_dl and 'feature_names' in model_data:
                    if any('DL_pred_' in f for f in model_data['feature_names']):
                        use_dl = True
                
                dl_config = {'train': False, 'model_path': f"saved_models_v2/dl_{model_ticker}.h5"} if use_dl else None
                
                # Generate Features
                features_opt = generate_features(data_opt, dl_config=dl_config)
                
                # Optimize
                best_params, best_return = optimize_strategy(data_opt, model_data, features_opt, n_trials=tune_trials, progress_callback=update_prog)
                
                # Complete progress
                prog_bar.progress(100, text="Optimization Complete!")
                
                # Store in session state
                st.session_state.v2_opt_params = best_params
                st.session_state.v2_opt_return = best_return
        
        # Display results (Persistent)
        if 'v2_opt_params' in st.session_state:
            st.success(f"✅ Optimization Complete! Best Return: {st.session_state.v2_opt_return:,.2f}%")
            st.markdown("### 🏆 Recommended Settings")
            st.json(st.session_state.v2_opt_params)
            
            # Callback to update state before rerun
            def apply_callback():
                params = st.session_state.v2_opt_params
                
                # Update Session State Keys
                if 'use_trend_filter' in params: st.session_state.s_trend = params['use_trend_filter']
                if 'use_adx_filter' in params: st.session_state.s_adx_bool = params['use_adx_filter']
                if 'adx_threshold' in params: st.session_state.s_adx_thresh = int(params['adx_threshold'])
                
                if 'ml_buy_threshold' in params: st.session_state.s_ml_buy = float(params['ml_buy_threshold'])
                if 'ml_sell_threshold' in params: st.session_state.s_ml_sell = float(params['ml_sell_threshold'])
                
                # Force Slope Signals ON (since optimization uses them)
                st.session_state.s_slope_bool = True
                
                if 'slope_buy_thresh' in params: st.session_state.s_buy = float(params['slope_buy_thresh'])
                if 'slope_sell_thresh' in params: st.session_state.s_sell = float(params['slope_sell_thresh'])
                if 'slope_roc_thresh' in params: st.session_state.s_roc = float(params['slope_roc_thresh'])
                if 'use_mom_zone' in params: st.session_state.s_mom = params['use_mom_zone']
                if 'use_ml_confirm' in params: st.session_state.s_ml_conf_bool = params['use_ml_confirm']
                if 'ml_confirm_thresh' in params: st.session_state.s_ml_conf_thresh = float(params['ml_confirm_thresh'])
            
            st.button("✅ Apply Best Settings to Sliders", on_click=apply_callback)


    # Meta-Model Filter Control
    st.markdown("### 🧠 Meta-Model Filter (Experimental)")
    use_meta = st.checkbox("Enable Meta-Model Filter", help="Filters out low-probability trades using a secondary model.")
    meta_threshold = 0.5
    
    if use_meta:
        meta_threshold = st.slider("Meta-Model Threshold", 0.1, 0.9, 0.5, 0.05, help="Lower = More Trades (Less Strict). Higher = Fewer Trades (More Strict).")
        
        if 'v2_meta_model' not in st.session_state:
            st.warning("Meta-Model not trained yet.")
            if st.button(f"Train Meta-Model (Uses {period} Data)"):
                with st.spinner("Training Meta-Model..."):
                    # Load data for training
                    import yfinance as yf
                    # Use the selected period from sidebar
                    mm_data = yf.Ticker(ticker).history(period=period, interval=interval) 
                    mm_data.index = pd.to_datetime(mm_data.index).tz_localize(None)
                    mm_data.columns = [c.lower() for c in mm_data.columns]
                    
                    # Generate features
                    # Load base model to get config
                    base_model = load_model(model_path)
                    
                    # Check DL
                    mm_dl_config = None
                    if base_model.get('use_dl', False):
                        mm_dl_config = {'train': False, 'model_path': f"saved_models_v2/dl_{ticker}.h5"}
                        
                    mm_features = generate_features(mm_data, dl_config=mm_dl_config, use_regime=use_regime_detection)
                    
                    # CRITICAL: Re-attach CLOSE price for OOF Target Generation
                    # generate_features drops raw prices, but we need 'close' to calc returns
                    mm_features['close'] = mm_data.loc[mm_features.index]['close']
                    
                    # Train
                    mm_result = train_meta_model(base_model, mm_features, labels=None) # labels not needed for simulated training
                    
                    if mm_result['status'] == 'success':
                        st.session_state.v2_meta_model = mm_result
                        st.success(f"Meta-Model Trained! Precision Score: {mm_result['score']:.2f}")
                        st.rerun()
                    else:
                        st.error(f"Training Failed: {mm_result.get('reason')}")
        else:
            st.success(f"Meta-Model Active (Score: {st.session_state.v2_meta_model.get('score', 0):.2f})")
            if st.button("Reset Meta-Model"):
                del st.session_state.v2_meta_model
                st.rerun()

    with st.expander("🚀 Strategy Bundling & Live Setup", expanded=False):
        st.info("Package your entire strategy (Base Model, DL, Meta, Alpha, Config) into a versioned bundle.")
        
        strat_name = st.text_input("Strategy Name (Optional)", value=f"{ticker}_Strategy")
        
        if st.button("📦 Bundle & Save Strategy"):
            # Check Meta Model Export
            meta_path = None
            if use_meta and 'v2_meta_model' in st.session_state:
                 meta_path = f"saved_models_v2/meta_model_{ticker}_latest.joblib"
                 joblib.dump(st.session_state.v2_meta_model, meta_path)
            
            # 1. Gather Active Models
            active_models = {}
            active_models['base'] = model_path
            
            # DL Check
            dl_path = f"saved_models_v2/dl_{ticker}.h5"
            if os.path.exists(dl_path):
                active_models['dl'] = dl_path
            
            # Meta Check
            if meta_path:
                active_models['meta'] = meta_path
            
            # Alpha Check
            alpha_path_disk = f"saved_models_v2/alpha_regressors_{ticker}.joblib"
            if os.path.exists(alpha_path_disk):
                active_models['alpha'] = alpha_path_disk
                
            # 2. Config Dictionary
            config = {
                "ticker": ticker,
                "interval": interval,
                "polygon_api_key": polygon_api_key,
                "model_path": model_path,
                "buy_threshold": ml_buy_thresh,
                "sell_threshold": ml_sell_thresh,
                "use_trend_filter": use_trend_filter,
                "use_adx_filter": use_adx_filter,
                "adx_threshold": adx_threshold,
                "limit_entry_pct": limit_entry_pct,
                "use_slope_signals": use_slope_signals,
                "slope_buy_thresh": slope_buy_thresh,
                "slope_sell_thresh": slope_sell_thresh,
                "slope_roc_thresh": slope_roc_thresh,
                "use_mom_zone": use_mom_zone,
                "use_ml_confirm": use_ml_confirm,
                "ml_confirm_thresh": ml_confirm_thresh,
                "use_meta_filter": use_meta,
                "meta_threshold": meta_threshold,
                "use_regime_detection": use_regime_detection,
                "use_conformal": use_conformal,
                "cp_alpha": cp_alpha,
                # "meta_model_path": handled by bundle
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 3. Create Bundle
            sfm = StrategyFileManager()
            bundle_path = sfm.save_strategy_bundle(strat_name, config, active_models)
            
            # 4. Update Root Config
            folder_name = os.path.basename(bundle_path)
            root_config = config.copy()
            
            # Point to the bundled files
            if 'base' in active_models: root_config['model_path'] = f"strategies/{folder_name}/{os.path.basename(active_models['base'])}"
            if 'dl' in active_models: root_config['dl_model_path'] = f"strategies/{folder_name}/{os.path.basename(active_models['dl'])}"
            if 'meta' in active_models: root_config['meta_model_path'] = f"strategies/{folder_name}/{os.path.basename(active_models['meta'])}"
            if 'alpha' in active_models: root_config['alpha_model_path'] = f"strategies/{folder_name}/{os.path.basename(active_models['alpha'])}"
            
            with open("strategy_config.json", "w") as f:
                json.dump(root_config, f, indent=4)
                
            st.success(f"✅ Strategy Bundled to: {bundle_path}")
            st.success("✅ Live Trader Config Updated to use this bundle!")

    if st.button("📊 Generate Signals & Backtest", type="primary", use_container_width=True):
        with st.spinner("Loading model and generating signals..."):
            # Load model
            model_data = load_model(model_path)
            model_ticker = model_data.get('ticker', ticker)
            
            # Determine date range based on backtest mode
            import yfinance as yf
            
            if backtest_mode == "Full Training Period":
                # Use the same period as training (e.g., 5y)
                data = yf.Ticker(model_ticker).history(period=period, interval=interval)
                period_label = f"Full {period} Training Period"
                
            elif backtest_mode == "Test Period Only":
                # Load full data, then use only the test portion (last 20%)
                data = yf.Ticker(model_ticker).history(period=period, interval=interval)
                split_idx = int(len(data) * 0.8)
                data = data.iloc[split_idx:]
                period_label = "Test Period (Last 20%)"
                
            elif backtest_mode == "Recent Days":
                end_date = datetime.now()
                start_date = end_date - timedelta(days=prod_days + 100)
                data = yf.Ticker(model_ticker).history(start=start_date, end=end_date, interval=interval)
                period_label = f"Last {prod_days} Days"
                
            else:
                data = yf.Ticker(model_ticker).history(start=custom_start, end=custom_end, interval=interval)
                period_label = f"{custom_start} to {custom_end}"
            
            data.index = pd.to_datetime(data.index).tz_localize(None)
            data.columns = [c.lower() for c in data.columns]
            data = data[['open', 'high', 'low', 'close', 'volume']]
            
            # Check if model uses DL
            use_dl = model_data.get('use_dl', False)
            # Fallback
            if not use_dl and 'feature_names' in model_data:
                if any('DL_pred_' in f for f in model_data['feature_names']):
                    use_dl = True
            
            dl_config = {'train': False, 'model_path': f"saved_models_v2/dl_{model_ticker}.h5"} if use_dl else None
            
            # Generate features
            features = generate_features(data, dl_config=dl_config, use_regime=use_regime_detection)
            
            # Generate signals with custom threshold
            signals, prob_df = generate_signals(model_data, features, 
                                              buy_threshold=ml_buy_thresh,
                                              sell_threshold=ml_sell_thresh,
                                              use_slope_signals=use_slope_signals,
                                              slope_buy_thresh=slope_buy_thresh,
                                              slope_sell_thresh=slope_sell_thresh,
                                              use_ml_confirm=use_ml_confirm,
                                              ml_confirm_thresh=ml_confirm_thresh,
                                              slope_roc_thresh=slope_roc_thresh,
                                              use_mom_zone=use_mom_zone)
                                              
            # Apply Volatility Filter (ADX)
            if use_adx_filter:
                adx_cols = [c for c in features.columns if c.startswith('ADX_')]
                if adx_cols:
                    adx_vals = features[adx_cols[0]].loc[signals.index]
                    chop_mask = adx_vals < adx_threshold
                    signals.loc[chop_mask] = 0
            
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
            
            # APPLY TREND FILTER (If Enabled)
            if use_trend_filter:
                # Calculate SMA200 on the FULL data
                sma200 = data['close'].rolling(200).mean()
                is_bull = data['close'] > sma200
                
                # Filter signals based on regime
                filtered_signals = signals.copy()
                
                # Vectorized masking for speed
                # Align indices first
                common_idx = signals.index.intersection(is_bull.index).intersection(composite_indicator.index)
                
                if not common_idx.empty:
                    # Create masks
                    bull_mask = is_bull.loc[common_idx]
                    bear_mask = ~bull_mask
                    comp_overbought = composite_indicator.loc[common_idx] > 0.6
                    
                    # Filter BUYs (1): ALWAYS ALLOWED
                    # Analysis showed the model is excellent at buying dips even in Bear markets (+3.58% return)
                    # So we removed the restriction on Buys.
                    
                    # Filter SELLs (-1): Allowed if Bear OR (Bull AND Extreme Overbought)
                    # If Bull AND Not Overbought -> Filter out (Don't short the trend unless it's a top)
                    sell_filter_mask = (signals.loc[common_idx] == -1) & bull_mask & (~comp_overbought)
                    filtered_signals.loc[sell_filter_mask[sell_filter_mask].index] = 0
                    
                signals = filtered_signals
            
            # Store raw for comparison
            signals_raw = signals.copy()
            
            # APPLY META FILTER (If Enabled)
            if use_meta and 'v2_meta_model' in st.session_state:
                signals = apply_meta_filter(st.session_state.v2_meta_model, signals, prob_df, features, threshold=meta_threshold)
                st.caption(f"✨ Meta-Filter Applied (Thresh: {meta_threshold}). Active signals reduced to: {(signals!=0).sum()}")

            # COMPUTE CONFORMAL PREDICTION CONFIDENCE (Always, for hover display)
            # Store conformal info for chart hover even if filtering is disabled
            cp_confidence = pd.Series(index=signals.index, dtype=float)
            cp_set_size = pd.Series(index=signals.index, dtype=int)
            cp_confidence[:] = np.nan
            cp_set_size[:] = 0
            
            if ConformalPredictionWrapper:
                try:
                    with st.spinner(f"🔬 Computing Conformal Prediction Confidence..."):
                        base_estimator = model_data['model']
                        cp = ConformalPredictionWrapper(base_estimator, method="score", cv=5)
                        
                        # Calibrate
                        cal_labels, _ = detect_peaks_valleys(data, order=detection_order)
                        common_idx = features.index.intersection(cal_labels.index)
                        X_cal = features.loc[common_idx]
                        y_cal = cal_labels.loc[common_idx]
                        
                        valid_mask = X_cal.notna().all(axis=1) & y_cal.notna()
                        X_cal = X_cal[valid_mask]
                        y_cal = y_cal[valid_mask]
                        
                        if not X_cal.empty:
                            cp.fit(X_cal, y_cal)
                            
                            # Get prediction sets for ALL signal points
                            X_pred = features.loc[signals.index].fillna(0)
                            _, y_pis = cp.predict(X_pred, alpha=cp_alpha)
                            
                            # Compute set sizes and confidence for each signal
                            classes = cp.mapie.classes_
                            class_map = {c: i for i, c in enumerate(classes)}
                            
                            for i, (idx, sig) in enumerate(signals.items()):
                                set_size = int(np.sum(y_pis[i]))
                                cp_set_size.at[idx] = set_size
                                
                                # Confidence = 1 if set size is 1 and correct class in set
                                # Otherwise compute as inverse of set size
                                if set_size == 1:
                                    cp_confidence.at[idx] = 1.0 - cp_alpha  # e.g., 90% for alpha=0.1
                                elif set_size > 1:
                                    cp_confidence.at[idx] = (1.0 - cp_alpha) / set_size  # Diluted confidence
                                else:
                                    cp_confidence.at[idx] = 0.0  # Empty set = no confidence
                            
                            # APPLY FILTER if enabled
                            if use_conformal:
                                signals = cp.filter_signals(signals, X_pred, alpha=cp_alpha)
                                st.caption(f"🔬 Conformal Filter Applied (Alpha: {cp_alpha}). Active signals reduced to: {(signals!=0).sum()}")
                except Exception as e:
                    st.warning(f"Conformal prediction failed: {e}")
            
            # Store for chart hover
            st.session_state.v2_cp_confidence = cp_confidence
            st.session_state.v2_cp_set_size = cp_set_size

            # Store composite and confidence in session state for plotting
            st.session_state.v2_composite = composite_indicator
            st.session_state.v2_probs = prob_df
            
            # Save detailed analysis output
            analysis_dir = "analysis"
            os.makedirs(analysis_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save signals, probs, and ALL features
            analysis_df = pd.concat([
                data[['open', 'high', 'low', 'close', 'volume']],
                signals.rename('signal'),
                prob_df,
                features # Include all technical indicators and features
            ], axis=1)
            
            # Remove duplicate columns (e.g. if 'close' is in features too)
            analysis_df = analysis_df.loc[:, ~analysis_df.columns.duplicated()]
            
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
            
            # Save results to session state for persistence
            st.session_state.v2_results = {
                'data': data,
                'signals': signals,
                'signals_raw': signals_raw,
                'backtest': backtest,
                'features': features,
                'prob_df': prob_df,
                'composite_indicator': composite_indicator,
                'model_ticker': model_ticker,
                'model_path': model_path,
                'config': {
                    'limit_entry_pct': limit_entry_pct,
                    'slope_buy_thresh': slope_buy_thresh,
                    'slope_sell_thresh': slope_sell_thresh,
                    'use_ml_confirm': use_ml_confirm,
                    'ml_confirm_thresh': ml_confirm_thresh,
                    'use_trend_filter': use_trend_filter,
                    'use_adx_filter': use_adx_filter,
                    'adx_threshold': adx_threshold,
                    'ml_buy_thresh': ml_buy_thresh,
                    'ml_sell_thresh': ml_sell_thresh
                }
            }
        
    # Persistent Display Block
    if 'v2_results' in st.session_state:
        res = st.session_state.v2_results
        data = res['data']
        signals = res['signals']
        signals_raw = res.get('signals_raw', signals)
        backtest = res['backtest']
        features = res['features']
        prob_df = res['prob_df']
        composite_indicator = res['composite_indicator']
        model_ticker = res['model_ticker']
        model_path = res['model_path']
        
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
        st.info("ℹ️ **Chart Legend**: Triangles are **Signals** (Limit Orders placed). Blue Dots are **Trades** (Limit Orders filled). If you see a Triangle but no Blue Dot, the price didn't reach your limit entry.")
        
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
        
        # Rejected Signals (Meta-Model Filtered)
        if not signals_raw.equals(signals):
            # Rejected Buys
            rej_buy_mask = (signals_raw == 1) & (signals != 1)
            if rej_buy_mask.any():
                rej_dates = signals_raw[rej_buy_mask].index
                rej_prices = data.loc[rej_dates, 'low'] * 0.98
                fig.add_trace(go.Scatter(
                    x=rej_dates, y=rej_prices, mode='markers',
                    marker=dict(symbol='triangle-up-open', size=12, color='gray', line=dict(width=2)),
                    name='Rejected BUY (Meta)', opacity=0.7
                ), row=1, col=1)

            # Rejected Sells
            rej_sell_mask = (signals_raw == -1) & (signals != -1)
            if rej_sell_mask.any():
                rej_dates = signals_raw[rej_sell_mask].index
                rej_prices = data.loc[rej_dates, 'high'] * 1.02
                fig.add_trace(go.Scatter(
                    x=rej_dates, y=rej_prices, mode='markers',
                    marker=dict(symbol='triangle-down-open', size=12, color='gray', line=dict(width=2)),
                    name='Rejected SELL (Meta)', opacity=0.7
                ), row=1, col=1)

        # Get conformal prediction data for hover
        cp_conf = st.session_state.get('v2_cp_confidence', pd.Series(dtype=float))
        cp_sets = st.session_state.get('v2_cp_set_size', pd.Series(dtype=int))
        prob_df_hover = st.session_state.get('v2_probs', pd.DataFrame())
        
        # Buy signals
        buy_mask = signals == 1
        if buy_mask.any():
            buy_dates = signals[buy_mask].index
            buy_prices = data.loc[buy_dates, 'low'] * 0.98
            
            # Build hover text with ML confidence and conformal prediction
            hover_texts = []
            for dt in buy_dates:
                price = data.loc[dt, 'close']
                ml_conf = prob_df_hover.loc[dt, 'prob_buy'] * 100 if dt in prob_df_hover.index and 'prob_buy' in prob_df_hover.columns else 0
                cp_c = cp_conf.get(dt, np.nan) * 100 if dt in cp_conf.index else np.nan
                cp_s = cp_sets.get(dt, 0) if dt in cp_sets.index else 0
                
                hover = f"<b>BUY SIGNAL</b><br>"
                hover += f"Date: {dt.strftime('%Y-%m-%d')}<br>"
                hover += f"Price: ${price:,.2f}<br>"
                hover += f"ML Confidence: {ml_conf:.1f}%<br>"
                if not np.isnan(cp_c):
                    hover += f"<b>Conformal Conf: {cp_c:.1f}%</b><br>"
                    hover += f"Prediction Set Size: {cp_s}"
                else:
                    hover += "Conformal: N/A"
                hover_texts.append(hover)
            
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='lime', 
                           line=dict(width=2, color='darkgreen')),
                name='BUY Signal (Intent)',
                hovertemplate='%{text}<extra></extra>',
                text=hover_texts
            ), row=1, col=1)
        
        # Sell signals
        sell_mask = signals == -1
        if sell_mask.any():
            sell_dates = signals[sell_mask].index
            sell_prices = data.loc[sell_dates, 'high'] * 1.02
            
            # Build hover text with ML confidence and conformal prediction
            hover_texts = []
            for dt in sell_dates:
                price = data.loc[dt, 'close']
                ml_conf = prob_df_hover.loc[dt, 'prob_sell'] * 100 if dt in prob_df_hover.index and 'prob_sell' in prob_df_hover.columns else 0
                cp_c = cp_conf.get(dt, np.nan) * 100 if dt in cp_conf.index else np.nan
                cp_s = cp_sets.get(dt, 0) if dt in cp_sets.index else 0
                
                hover = f"<b>SELL SIGNAL</b><br>"
                hover += f"Date: {dt.strftime('%Y-%m-%d')}<br>"
                hover += f"Price: ${price:,.2f}<br>"
                hover += f"ML Confidence: {ml_conf:.1f}%<br>"
                if not np.isnan(cp_c):
                    hover += f"<b>Conformal Conf: {cp_c:.1f}%</b><br>"
                    hover += f"Prediction Set Size: {cp_s}"
                else:
                    hover += "Conformal: N/A"
                hover_texts.append(hover)
            
            fig.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                marker=dict(symbol='triangle-down', size=15, color='red',
                           line=dict(width=2, color='darkred')),
                name='SELL Signal (Intent)',
                hovertemplate='%{text}<extra></extra>',
                text=hover_texts
            ), row=1, col=1)
        
        # Trade markers (actual executions)
        first_trade = True
        for trade in backtest['trades']:
            # Entry
            fig.add_trace(go.Scatter(
                x=[trade['entry_date']],
                y=[trade['entry_price']],
                mode='markers',
                marker=dict(symbol='circle', size=12, color='blue',
                           line=dict(width=2, color='white')),
                name='Trade Entry (Filled)',
                showlegend=first_trade
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
                    name='Trade Exit',
                    showlegend=first_trade
                ), row=1, col=1)
            
            first_trade = False
        
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
            title=f"{model_ticker} - ML Trading Signals ({selected_option})",
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
        
        # Meta-Model Analysis Chart (Rejected Trades)
        if use_meta and 'signals_raw' in locals():
            st.markdown("### 🧠 Meta-Model Decisions")
            
            # Filter to recent period
            rec_sig_raw = signals_raw[signals_raw.index >= recent_data.index[0]]
            # recent_signals is already the filtered version
            
            rej_buys = (rec_sig_raw == 1) & (recent_signals == 0)
            rej_sells = (rec_sig_raw == -1) & (recent_signals == 0)
            
            total_rej = rej_buys.sum() + rej_sells.sum()
            
            if total_rej > 0:
                st.caption(f"The Meta-Model rejected **{total_rej}** signals in this period.")
                fig_rej = go.Figure()
                
                # Price
                fig_rej.add_trace(go.Candlestick(
                    x=recent_data.index, open=recent_data['open'], high=recent_data['high'],
                    low=recent_data['low'], close=recent_data['close'], name='Price'
                ))
                
                # --- ACCEPTED SIGNALS (Green/Red Triangles) ---
                recent_buy_mask = recent_signals == 1
                recent_sell_mask = recent_signals == -1
                
                if recent_buy_mask.any():
                    recent_buy_dates = recent_signals[recent_buy_mask].index
                    recent_buy_prices = recent_data.loc[recent_buy_dates, 'low'] * 0.98
                    fig_rej.add_trace(go.Scatter(
                        x=recent_buy_dates, y=recent_buy_prices, mode='markers',
                        marker=dict(symbol='triangle-up', size=14, color='lime', line=dict(width=2, color='darkgreen')),
                        name='Accepted BUY'
                    ))
                
                if recent_sell_mask.any():
                    recent_sell_dates = recent_signals[recent_sell_mask].index
                    recent_sell_prices = recent_data.loc[recent_sell_dates, 'high'] * 1.02
                    fig_rej.add_trace(go.Scatter(
                        x=recent_sell_dates, y=recent_sell_prices, mode='markers',
                        marker=dict(symbol='triangle-down', size=14, color='red', line=dict(width=2, color='darkred')),
                        name='Accepted SELL'
                    ))
                    
                # --- TRADES (Entries/Exits) ---
                for trade in recent_trades:
                    fig_rej.add_trace(go.Scatter(
                        x=[trade['entry_date']], y=[trade['entry_price']], mode='markers+text',
                        marker=dict(symbol='circle', size=10, color='blue', line=dict(width=1, color='white')),
                        text=['▶'], textposition='middle left', name='Entry', showlegend=False
                    ))
                    if 'exit_date' in trade and trade['exit_date'] >= recent_data.index[0]:
                        profit_color = 'green' if trade.get('profit', 0) > 0 else 'red'
                        fig_rej.add_trace(go.Scatter(
                            x=[trade['exit_date']], y=[trade['exit_price']], mode='markers',
                            marker=dict(symbol='x', size=10, color=profit_color, line=dict(width=2, color='white')),
                            name='Exit', showlegend=False
                        ))
                
                # --- REJECTED SIGNALS (Red/Orange Xs) ---
                if rej_buys.sum() > 0:
                    fig_rej.add_trace(go.Scatter(
                        x=recent_data.loc[rej_buys].index, y=recent_data.loc[rej_buys]['low']*0.98,
                        mode='markers', marker=dict(symbol='x-thin', size=12, color='red', line_width=3), 
                        name='Rejected Buy'
                    ))
                
                if rej_sells.sum() > 0:
                    fig_rej.add_trace(go.Scatter(
                        x=recent_data.loc[rej_sells].index, y=recent_data.loc[rej_sells]['high']*1.02,
                        mode='markers', marker=dict(symbol='x-thin', size=12, color='orange', line_width=3), 
                        name='Rejected Sell'
                    ))
                    
                fig_rej.update_layout(title="Meta-Model Impact: Accepted vs Rejected", height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_rej, use_container_width=True)
        
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

        # ==========================================================================
        # 💎 Polygon.io Comparison (High Fidelity Data)
        # ==========================================================================
        # 1. Calculation (Only runs when Generating Signals)
        # Relaxed check: If dl_config is missing, assume None (No DL)
        if polygon_api_key and PolygonManager:
            # Ensure we are in a generation context (model_data exists)
            if 'model_data' in locals():
                dl_config = locals().get('dl_config', None)
                
                with st.spinner("Fetching Polygon Data (180 Days) & Re-Calculating..."):
                    poly_mgr = PolygonManager(polygon_api_key)
                    
                    # Exact 180 Day Window
                    poly_end = datetime.now().strftime("%Y-%m-%d")
                    poly_start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
                    
                    poly_data = poly_mgr.get_price_data(ticker, start_date=poly_start, end_date=poly_end, limit=50000)
                    
                    if not poly_data.empty:
                        # Generate Features
                        poly_features = generate_features(poly_data, dl_config=dl_config)
                        
                        # Generate Signals
                        poly_signals, poly_probs = generate_signals(
                            model_data, poly_features, 
                            threshold=decision_threshold,
                            use_slope_signals=use_slope_signals,
                            slope_buy_thresh=slope_buy_thresh,
                            slope_sell_thresh=slope_sell_thresh,
                            use_ml_confirm=use_ml_confirm,
                            ml_confirm_thresh=ml_confirm_thresh,
                            slope_roc_thresh=slope_roc_thresh,
                            use_mom_zone=use_mom_zone,
                            buy_threshold=ml_buy_thresh,
                            sell_threshold=ml_sell_thresh
                        )
                        
                        # Backtest
                        poly_backtest = run_backtest(poly_data, poly_signals, limit_pct=limit_entry_pct)
                        
                        # Save to Session State for Persistence
                        st.session_state.v2_polygon_results = {
                            'data': poly_data,
                            'signals': poly_signals,
                            'backtest': poly_backtest,
                            'ticker': ticker
                        }
                    else:
                        if 'v2_polygon_results' in st.session_state:
                            del st.session_state.v2_polygon_results
                        st.warning("Could not fetch data from Polygon. Check your API Key and Ticker.")

        # 2. Display (Runs on every render if results exist)
        if 'v2_polygon_results' in st.session_state:
            p_res = st.session_state.v2_polygon_results
            poly_data = p_res['data']
            poly_signals = p_res['signals']
            poly_backtest = p_res['backtest']
            
            st.markdown("---")
            st.header("💎 Polygon.io Comparison")
            st.caption("Comparing 'What Would Have Happened' with High-Fidelity Polygon Data vs Yahoo Data (Last 180 Days)")

            # Metrics
            col_p1, col_p2, col_p3 = st.columns(3)
            
            # Get Standard Return for Delta (from session state or local)
            std_return = backtest['total_return'] if 'backtest' in locals() else st.session_state.get('v2_backtest', {}).get('total_return', 0.0)
            
            with col_p1:
                st.metric("Polygon Total Return", f"{poly_backtest['total_return']:.1f}%", 
                          delta=f"{poly_backtest['total_return'] - std_return:.1f}% vs Standard")
            with col_p2:
                st.metric("Polygon Win Rate", f"{poly_backtest['win_rate']:.1f}%")
            with col_p3:
                st.metric("Polygon Trades", poly_backtest['num_trades'])

            # Duplicate Chart (Simplified Plotly)
            fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Candles
            fig_p.add_trace(go.Candlestick(
                x=poly_data.index, open=poly_data['open'], high=poly_data['high'],
                low=poly_data['low'], close=poly_data['close'], name='Polygon Price'
            ), row=1, col=1)
            
            # Buy Signals
            p_buys = poly_signals[poly_signals == 1]
            if not p_buys.empty:
                fig_p.add_trace(go.Scatter(
                    x=p_buys.index, y=poly_data.loc[p_buys.index, 'low']*0.98,
                    mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'),
                    name='Buy Signal'
                ), row=1, col=1)

            # Sell Signals
            p_sells = poly_signals[poly_signals == -1]
            if not p_sells.empty:
                fig_p.add_trace(go.Scatter(
                    x=p_sells.index, y=poly_data.loc[p_sells.index, 'high']*1.02,
                    mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
                    name='Sell Signal'
                ), row=1, col=1)
                
            # Equity Curve
            fig_p.add_trace(go.Scatter(
                x=poly_backtest['equity_curve'].index, y=poly_backtest['equity_curve']['equity'],
                line=dict(color='cyan', width=2), name='Equity', fill='tozeroy'
            ), row=2, col=1)
            
            fig_p.update_layout(title=f"Polygon Data Analysis ({p_res['ticker']}) - High Fidelity", height=600, template="plotly_dark")
            fig_p.update_yaxes(title_text="Price", row=1, col=1)
            fig_p.update_yaxes(title_text="Equity ($)", row=2, col=1)
            st.plotly_chart(fig_p, use_container_width=True)

        # SAVE STRATEGY STATE BUTTON
        st.divider()
        if st.button("💾 Save Strategy State (Signals + Config)"):
            import json
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = "strategy_states"
            os.makedirs(save_dir, exist_ok=True)
            
            # 1. Save Signals CSV
            signals_path = os.path.join(save_dir, f"signals_{timestamp}.csv")
            
            # Reconstruct necessary data for CSV
            output_df = pd.DataFrame(index=data.index)
            output_df['close'] = data['close']
            output_df['signal'] = signals
            if 'composite_oscillator' in features.columns:
                output_df['composite'] = features['composite_oscillator']
                
            # Add probabilities if available (need to match index)
            # prob_df has same index as data?
            if not prob_df.empty and len(prob_df) == len(data):
                 # Find columns
                 buy_col = [c for c in prob_df.columns if str(c) == '1']
                 sell_col = [c for c in prob_df.columns if str(c) == '-1']
                 if buy_col: output_df['ml_buy_prob'] = prob_df[buy_col[0]]
                 if sell_col: output_df['ml_sell_prob'] = prob_df[sell_col[0]]
            
            output_df.to_csv(signals_path)
            
            # 2. Save Config JSON
            config = {
                'model_path': model_path,
                'slope_buy_thresh': slope_buy_thresh,
                'slope_sell_thresh': slope_sell_thresh,
                'use_ml_confirm': use_ml_confirm,
                'ml_confirm_thresh': ml_confirm_thresh,
                'win_rate': backtest['win_rate'],
                'total_return': backtest['total_return']
            }
            config_path = os.path.join(save_dir, f"config_{timestamp}.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
                
            st.success(f"Strategy State Saved to {save_dir}/")

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
            'scaler': loaded_model.get('scaler'), # Safe get
            'label_encoder': loaded_model.get('label_encoder'),
            'feature_names': loaded_model['feature_names'],
            'model_type': loaded_model['model_type'],
            'best_params': loaded_model.get('best_params', {}),
            'accuracy': loaded_model.get('metrics', {}).get('accuracy', 0),
            'f1_score': loaded_model.get('metrics', {}).get('f1_score', 0),
            'cv_score': loaded_model.get('metrics', {}).get('cv_score', 0),
            'train_size': 0,
            'test_size': 0,
            'label_distribution': {},
            'split_date': None,
            'use_smote': True,
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

    # ==========================================================================
    # MARKET REGIME ANALYSIS (New Section)
    # ==========================================================================
    st.markdown("---")
    st.subheader("🌍 Market Regime Analysis")
    
    if st.checkbox("Show Regime Analysis", value=False):
        # 1. Get Data & Generate Regimes
        # We need to regenerate features to get the latest regime classification
        reg_data = st.session_state.get('v2_data')
        if reg_data is not None:
            with st.spinner("Classifying Market Regimes..."):
                # Force regime generation
                reg_features = generate_features(reg_data, use_regime=True)
                
                if 'regime' in reg_features.columns:
                    # Plot Price with Regime Colors
                    st.markdown("#### Market States (Bull / Bear / Chop)")
                    
                    # Create Plot
                    fig_reg = go.Figure()
                    
                    # Price Line
                    fig_reg.add_trace(go.Scatter(
                        x=reg_data.index, y=reg_data['close'],
                        mode='lines', name='Price', line=dict(color='black', width=1)
                    ))
                    
                    # Use FIXED regime mapping from MarketRegimeDetector
                    # 0 = Bull Trend, 1 = Sideways/Chop, 2 = Bear/Stress
                    bull_label = 0
                    chop_label = 1
                    bear_label = 2
                    
                    colors = {
                        bull_label: 'rgba(0, 255, 0, 0.2)',      # Green - Bull
                        chop_label: 'rgba(128, 128, 128, 0.2)',  # Gray - Chop
                        bear_label: 'rgba(255, 0, 0, 0.2)'       # Red - Bear
                    }
                    
                    names = {
                        bull_label: 'Bull Trend', 
                        chop_label: 'Chop/Sideways', 
                        bear_label: 'Bear/Stress'
                    }
                    
                    # Add colored regions
                    # We iterate through regime changes
                    regime_changes = reg_features['regime'].diff().fillna(0) != 0
                    segments = reg_features[regime_changes].index.tolist()
                    segments.append(reg_features.index[-1]) # End
                    
                    # This is slow for many segments. Better to use Heatmap or Bar?
                    # Let's use Scatter markers with fill? No.
                    # Let's use VRects (Vertical Rectangles)
                    
                    # Optimized Segment Loop
                    # (Simplified for performance: just last 365 days)
                    recent_feat = reg_features.iloc[-365:]
                    recent_data = reg_data.loc[recent_feat.index]
                    
                    # Re-plot for recent only
                    fig_reg = go.Figure()
                    fig_reg.add_trace(go.Scatter(
                        x=recent_data.index, y=recent_data['close'],
                        mode='lines', name='Price', line=dict(color='black', width=1.5)
                    ))
                    
                    curr_reg = recent_feat['regime'].iloc[0]
                    start_date = recent_feat.index[0]
                    
                    for date, row in recent_feat.iterrows():
                        r = row['regime']
                        if r != curr_reg:
                            # End segment
                            fig_reg.add_vrect(
                                x0=start_date, x1=date,
                                fillcolor=colors[curr_reg], opacity=1,
                                layer="below", line_width=0,
                                annotation_text=names[curr_reg] if (date - start_date).days > 30 else None,
                                annotation_position="top left"
                            )
                            curr_reg = r
                            start_date = date
                            
                    # Last segment
                    fig_reg.add_vrect(
                        x0=start_date, x1=recent_feat.index[-1],
                        fillcolor=colors[curr_reg], opacity=1,
                        layer="below", line_width=0
                    )
                    
                    fig_reg.update_layout(title="Market Regimes (Last 365 Days)", height=500)
                    st.plotly_chart(fig_reg, use_container_width=True)
                    
                    st.info(f"**Legend:** 🟢 {names[bull_label]} | 🔴 {names[bear_label]} | ⚪ {names[chop_label]}")
                    
                else:
                    st.warning("Regime data not found in features. Ensure 'Use Market Regime Detection' is enabled.")
        else:
            st.info("Load data in Manual Train or Production first.")

    # ==========================================================================
    # SLOPE SIGNAL ANALYSIS (New Section)
    # ==========================================================================
    st.markdown("---")
    st.subheader("📉 Slope Signal Analysis (Deep Dive)")
    st.caption("Analyze the quality of 'Slope Turn Signals' to find the best filter threshold.")
    
    analysis_dir = "analysis"
    if os.path.exists(analysis_dir):
        analysis_files = sorted([f for f in os.listdir(analysis_dir) if f.startswith("analysis_")], reverse=True)
        if analysis_files:
            latest_file = analysis_files[0]
            if st.button(f"🔬 Analyze Slope Signals from {latest_file}"):
                analyze_slope_signals(os.path.join(analysis_dir, latest_file))
        else:
            st.info("No analysis files found. Run a backtest in Production tab first.")
    else:
        st.info("No analysis directory found.")

# =============================================================================
# TAB 4: ALPHA LAB
# =============================================================================
with tab4:
    st.header("🧪 Alpha Lab: Execution Optimization")
    st.info("Improve your entries by predicting intraday Highs/Lows and using Limit Orders instead of Market Orders.")
    
    # 1. Select Base Model
    model_dir = "saved_models_v2"
    strategies_dir = "strategies"
    
    # Collect Raw Models
    raw_models = []
    if os.path.exists(model_dir):
        raw_models = sorted([f"raw/{f}" for f in os.listdir(model_dir) if f.endswith('.joblib') and 'meta' not in f], reverse=True)
        
    # Collect Bundles
    bundles = []
    if os.path.exists(strategies_dir):
        # List directories in strategies/
        bundles = sorted([f"bundle/{d}" for d in os.listdir(strategies_dir) if os.path.isdir(os.path.join(strategies_dir, d))], reverse=True)
    
    all_options = bundles + raw_models
    
    if not all_options:
        st.warning("⚠️ No models or bundles found.")
        st.stop()
        
    alpha_model_select = st.selectbox("Select Strategy / Model", all_options, key="alpha_model_select")
    
    # Resolve Path
    if alpha_model_select.startswith("bundle/"):
        bundle_name = alpha_model_select.replace("bundle/", "")
        bundle_path = os.path.join(strategies_dir, bundle_name)
        # Find model inside bundle
        model_files = [f for f in os.listdir(bundle_path) if f.endswith('.joblib') and 'alpha' not in f]
        if model_files:
            alpha_model_path = os.path.join(bundle_path, model_files[0])
        else:
            st.error(f"No model found in bundle {bundle_name}")
            st.stop()
    else:
        # Raw model
        model_name = alpha_model_select.replace("raw/", "")
        alpha_model_path = os.path.join(model_dir, model_name)
    
    # 2. Train Range Predictor
    st.markdown("### 1. Range Predictor (Daily High/Low)")
    st.caption("Train a Regression Model to predict how far the price will move from the Open.")
    
    if st.button("🧠 Train Range Predictor"):
        # Progress Bar Setup
        prog_bar = st.progress(0, text="Initializing...")
        status_text = st.empty()
        
        try:
            # Load Daily Data for Regression
            status_text.text("Loading Daily Data for Regression...")
            prog_bar.progress(5, text="Loading Data...")
            
            import yfinance as yf
            # We use Daily data for this because we predict Daily High/Low
            reg_data = yf.Ticker(ticker).history(period=period, interval='1d') 
            reg_data.index = pd.to_datetime(reg_data.index).tz_localize(None)
            reg_data.columns = [c.lower() for c in reg_data.columns]
            
            # Generate Features
            status_text.text("Generating Technical Features (Daily)...")
            prog_bar.progress(15, text="Generating Features...")
            reg_features = generate_features(reg_data)
            
            # Targets
            # High Target: % move from Open to High
            target_high = (reg_data['high'] - reg_data['open']) / reg_data['open']
            # Low Target: % move from Open to Low
            target_low = (reg_data['low'] - reg_data['open']) / reg_data['open']
            
            # Align
            common = reg_features.index.intersection(target_high.index)
            reg_features = reg_features.loc[common]
            target_high = target_high.loc[common]
            target_low = target_low.loc[common]
            
            # Train High Predictor
            status_text.text("Optimizing High Predictor (XGBoost Regressor)...")
            def update_high(current, total):
                # 20% to 55%
                pct = 20 + int(35 * (current + 1) / total)
                prog_bar.progress(pct, text=f"Optimizing High Predictor: Trial {current+1}/{total}")
                
            res_high = train_regressor(reg_features, target_high, progress_callback=update_high)
            
            # Train Low Predictor
            status_text.text("Optimizing Low Predictor (XGBoost Regressor)...")
            def update_low(current, total):
                # 55% to 90%
                pct = 55 + int(35 * (current + 1) / total)
                prog_bar.progress(pct, text=f"Optimizing Low Predictor: Trial {current+1}/{total}")
                
            res_low = train_regressor(reg_features, target_low, progress_callback=update_low)
            
            prog_bar.progress(100, text="Training Complete!")
            status_text.success("✅ Training Complete!")
            
            st.session_state.alpha_regressors = {
                'high': res_high,
                'low': res_low,
                'data': reg_data
            }
            
            # Save to disk
            alpha_path = f"saved_models_v2/alpha_regressors_{ticker}.joblib"
            joblib.dump(st.session_state.alpha_regressors, alpha_path)
            st.info(f"💾 Alpha Models saved to {alpha_path}")
            
            st.success(f"✅ Regressors Trained! Low RMSE: {res_low['rmse']:.4f}, High RMSE: {res_high['rmse']:.4f}")
            
        except Exception as e:
            st.error(f"Error during training: {e}")
            
    if 'alpha_regressors' in st.session_state:
        regs = st.session_state.alpha_regressors
        
        # Visualization
        st.markdown("#### Predicted vs Actual Range (Last 50 Days)")
        
        # Predict on recent data
        data = regs['data']
        features = generate_features(data)
        
        # Load models
        model_h = regs['high']['model']
        model_l = regs['low']['model']
        
        # Predict
        pred_h = model_h.predict(features)
        pred_l = model_l.predict(features)
        
        # Create DF
        df_res = pd.DataFrame({
            'Actual High': (data['high'] - data['open']) / data['open'],
            'Actual Low': (data['low'] - data['open']) / data['open'],
            'Pred High': pred_h,
            'Pred Low': pred_l
        }, index=features.index)
        
        # Plot
        st.line_chart(df_res.iloc[-50:][['Actual Low', 'Pred Low']])
        
        # Execution Simulation
        st.markdown("### 2. Sniper Strategy Simulation (Entry & Exit)")
        st.caption("Strategy: Buy Limit @ Predicted Low. Sell Limit @ Predicted High. Exit at Close if Target not hit.")
        
        if st.button("⚔️ Run Strategy Simulation"):
            # Load Base Model Signals
            base_model = load_model(alpha_model_path)
            # Generate signals on this data
            sigs, _ = generate_signals(base_model, features)
            
            # Filter for BUYS
            buy_days = sigs[sigs == 1].index
            
            results = []
            
            for date in buy_days:
                if date not in df_res.index: continue
                
                row = data.loc[date]
                preds = df_res.loc[date]
                
                # Standard: Buy at Open, Sell at Close
                entry_std = row['open']
                exit_std = row['close']
                ret_std = (exit_std - entry_std) / entry_std
                
                # Sniper Strategy
                # 1. Place Buy Limit at Predicted Low
                buy_limit = row['open'] * (1 + preds['Pred Low'])
                # 2. Place Sell Limit at Predicted High
                sell_limit = row['open'] * (1 + preds['Pred High'])
                
                # Check Entry Fill
                if row['low'] <= buy_limit:
                    entry_sniper = buy_limit
                    status = 'FILLED_ENTRY'
                    
                    # Check Exit Fill (Target)
                    # Note: With daily data, we assume High > Sell_Limit is a fill.
                    # Risk: High might have happened BEFORE Low. 
                    # Mitigation: We are conservative.
                    if row['high'] >= sell_limit:
                        exit_sniper = sell_limit
                        status = 'TARGET_HIT 🎯'
                    else:
                        # Exit at Close
                        exit_sniper = row['close']
                        status = 'CLOSE_EXIT'
                        
                    ret_sniper = (exit_sniper - entry_sniper) / entry_sniper
                else:
                    # Missed Entry
                    entry_sniper = row['close'] # No trade
                    status = 'MISSED'
                    ret_sniper = 0.0
                
                results.append({
                    'date': date,
                    'std_return': ret_std,
                    'sniper_return': ret_sniper,
                    'status': status,
                    'buy_limit': buy_limit,
                    'sell_limit': sell_limit
                })
                
            res_df = pd.DataFrame(results)
            if not res_df.empty:
                st.metric("Total Trades", len(res_df))
                
                fill_rate = (res_df['status'] != 'MISSED').mean()
                target_rate = (res_df['status'] == 'TARGET_HIT 🎯').mean()
                
                c1, c2 = st.columns(2)
                c1.metric("Entry Fill Rate", f"{fill_rate:.1%}")
                c2.metric("Target Hit Rate", f"{target_rate:.1%}")
                
                tot_std = res_df['std_return'].sum() * 100
                tot_snip = res_df['sniper_return'].sum() * 100
                
                col1, col2 = st.columns(2)
                col1.metric("Standard Return", f"{tot_std:.1f}%")
                col2.metric("Sniper Strategy Return", f"{tot_snip:.1f}%", delta=f"{tot_snip - tot_std:.1f}%")
                
                st.dataframe(res_df)
                
                # --- VISUALIZATION ---
                st.subheader("📊 Sniper Strategy Simulation Chart")
                
                # Use the full range of the prediction dataframe (df_res) for context
                # This ensures we show candles even on days without trades
                if not df_res.empty:
                    sim_start = df_res.index[0]
                    sim_end = df_res.index[-1]
                    sim_data = data.loc[sim_start:sim_end]
                else:
                    sim_data = data.iloc[-50:] # Fallback
                
                fig_sim = go.Figure()
                
                # 1. Candlestick
                fig_sim.add_trace(go.Candlestick(
                    x=sim_data.index,
                    open=sim_data['open'], high=sim_data['high'],
                    low=sim_data['low'], close=sim_data['close'],
                    name='Price'
                ))
                
                # 2. Entries (Filled)
                filled_mask = res_df['status'] != 'MISSED'
                if filled_mask.any():
                    filled_df = res_df[filled_mask]
                    fig_sim.add_trace(go.Scatter(
                        x=filled_df['date'], y=filled_df['buy_limit'],
                        mode='markers', marker=dict(symbol='triangle-up', size=10, color='lime'),
                        name='Limit Entry Filled'
                    ))
                    
                # 3. Exits (Target Hit)
                target_mask = res_df['status'] == 'TARGET_HIT 🎯'
                if target_mask.any():
                    target_df = res_df[target_mask]
                    fig_sim.add_trace(go.Scatter(
                        x=target_df['date'], y=target_df['sell_limit'],
                        mode='markers', marker=dict(symbol='star', size=12, color='gold', line=dict(width=1, color='black')),
                        name='Target Hit'
                    ))
                    
                # 4. Exits (Close)
                close_mask = res_df['status'] == 'CLOSE_EXIT'
                if close_mask.any():
                    close_df = res_df[close_mask]
                    close_prices = sim_data.loc[close_df['date']]['close']
                    fig_sim.add_trace(go.Scatter(
                        x=close_df['date'], y=close_prices,
                        mode='markers', marker=dict(symbol='x', size=8, color='orange'),
                        name='Close Exit'
                    ))

                fig_sim.update_layout(height=600, template="plotly_dark", title="Sniper Strategy Execution", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_sim, use_container_width=True)

        # Real-Time Signals
        st.markdown("### 3. 🔮 Real-Time Signals (Next Session)")
        st.info("Use these levels for your Limit Orders in the next trading session.")
        
        # Get latest data point
        last_date = data.index[-1]
        last_row = data.iloc[-1]
        last_preds = df_res.loc[last_date] # Predictions for the last known day
        
        # We need predictions for TOMORROW (or Today if live).
        # The regression model uses LAGGED features. So inputting today's features predicts TODAY's range.
        # So we actually need to look at the prediction for the LATEST row.
        
        # Base Signal
        base_model = load_model(alpha_model_path)
        # We need to run inference on the last row
        last_feat = features.iloc[[-1]] # Keep as DF
        last_sig, _ = generate_signals(base_model, last_feat)
        signal_val = last_sig.iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if signal_val == 1:
                st.success("## 🟢 BUY Signal")
            elif signal_val == -1:
                st.error("## 🔴 SELL Signal")
            else:
                st.warning("## ⚪ HOLD")
                
        with col2:
            st.metric("📅 Date", last_date.strftime('%Y-%m-%d'))
            st.metric("Open Price", f"${last_row['open']:,.2f}")
            
        with col3:
            # Calculate Limits
            buy_limit = last_row['open'] * (1 + last_preds['Pred Low'])
            sell_limit = last_row['open'] * (1 + last_preds['Pred High'])
            
            st.metric("📉 Buy Limit Order", f"${buy_limit:,.2f}", delta=f"{last_preds['Pred Low']*100:.2f}% from Open")
            st.metric("📈 Sell Limit Order", f"${sell_limit:,.2f}", delta=f"{last_preds['Pred High']*100:.2f}% from Open")
            
        st.caption("Note: These limits are calculated relative to the Open price. If trading tomorrow, update 'Open Price' with tomorrow's open.")
