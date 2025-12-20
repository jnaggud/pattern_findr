"""
Optimized Strategy Page - Pattern_FindR
========================================
Combines the best of all approaches:
1. Optuna hyperparameter optimization (from Master Opt)
2. Proper train/test split (70/30) - NO data leakage
3. Walk-forward validation for realistic performance
4. Rule-based + ML hybrid signals
5. Regime-aware trading

Key insight: Optimize on TRAINING data, validate on TEST data (never seen).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import json

# Optuna for optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# XGBoost for ML
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def split_data(data: pd.DataFrame, features: pd.DataFrame, 
               train_ratio: float = 0.7) -> dict:
    """
    Split data into train/test sets chronologically.
    NO SHUFFLING - respects time series nature.
    """
    n = len(data)
    split_idx = int(n * train_ratio)
    
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    # Align features
    train_features = features.loc[features.index.isin(train_data.index)].copy()
    test_features = features.loc[features.index.isin(test_data.index)].copy()
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'train_features': train_features,
        'test_features': test_features,
        'split_idx': split_idx,
        'split_date': data.index[split_idx]
    }


def generate_labels(data: pd.DataFrame, order: int = 5) -> pd.Series:
    """
    Generate peak/valley labels for ML training.
    Uses scipy's argrelextrema to find local peaks and valleys.
    """
    from scipy.signal import argrelextrema
    
    labels = pd.Series(0, index=data.index)
    
    # Find peaks (local maxima in highs)
    peaks = argrelextrema(data['high'].values, np.greater_equal, order=order)[0]
    
    # Find valleys (local minima in lows)
    valleys = argrelextrema(data['low'].values, np.less_equal, order=order)[0]
    
    # Label: 1 = SELL (before peak), -1 = BUY (before valley)
    for peak_idx in peaks:
        if peak_idx > 0:
            labels.iloc[peak_idx - 1] = 1  # SELL signal day before peak
    
    for valley_idx in valleys:
        if valley_idx > 0:
            labels.iloc[valley_idx - 1] = -1  # BUY signal day before valley
    
    return labels


def calculate_backtest_return(data: pd.DataFrame, signals: pd.Series, 
                              initial_capital: float = 100000) -> dict:
    """
    Calculate backtest returns with proper position tracking.
    Long-only strategy: BUY enters, SELL exits to cash.
    """
    common_idx = data.index.intersection(signals.index)
    df = data.loc[common_idx].copy()
    df['signal'] = signals.loc[common_idx]
    
    capital = initial_capital
    position = 0
    shares = 0
    entry_price = 0
    trades = []
    equity = [initial_capital]
    
    for i in range(len(df)):
        price = df['close'].iloc[i]
        signal = df['signal'].iloc[i]
        
        if signal == 1 and position == 0:  # BUY
            shares = capital / price
            entry_price = price
            position = 1
            trades.append({'type': 'BUY', 'price': price, 'date': df.index[i]})
            
        elif signal == -1 and position == 1:  # SELL
            capital = shares * price
            pnl_pct = (price / entry_price - 1) * 100
            trades[-1]['exit_price'] = price
            trades[-1]['pnl_pct'] = pnl_pct
            position = 0
            shares = 0
        
        # Track equity
        if position == 1:
            equity.append(shares * price)
        else:
            equity.append(capital)
    
    # Close open position
    if position == 1:
        capital = shares * df['close'].iloc[-1]
        if trades:
            trades[-1]['exit_price'] = df['close'].iloc[-1]
            trades[-1]['pnl_pct'] = (df['close'].iloc[-1] / entry_price - 1) * 100
    
    total_return = (capital / initial_capital - 1) * 100
    bh_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    
    # Calculate max drawdown
    equity_series = pd.Series(equity)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min() * 100
    
    # Win rate
    closed_trades = [t for t in trades if 'pnl_pct' in t]
    wins = [t for t in closed_trades if t['pnl_pct'] > 0]
    win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0
    
    return {
        'total_return': total_return,
        'buy_hold_return': bh_return,
        'alpha': total_return - bh_return,
        'num_trades': len(closed_trades),
        'win_rate': win_rate,
        'max_drawdown': max_dd,
        'final_capital': capital,
        'trades': trades,
        'equity': equity
    }


def generate_signals_from_params(data: pd.DataFrame, features: pd.DataFrame, 
                                  labels: pd.Series, params: dict) -> pd.Series:
    """
    Generate signals using optimized parameters.
    Combines ML predictions with rule-based filters.
    """
    # Extract parameters
    ml_buy_thresh = params.get('ml_buy_thresh', 0.5)
    ml_sell_thresh = params.get('ml_sell_thresh', 0.5)
    rsi_buy_thresh = params.get('rsi_buy_thresh', 35)
    rsi_sell_thresh = params.get('rsi_sell_thresh', 70)
    use_rsi_filter = params.get('use_rsi_filter', True)
    use_regime_filter = params.get('use_regime_filter', True)
    
    # XGBoost parameters
    xgb_params = {
        'n_estimators': params.get('n_estimators', 100),
        'max_depth': params.get('max_depth', 3),
        'learning_rate': params.get('learning_rate', 0.1),
        'subsample': params.get('subsample', 0.8),
        'colsample_bytree': params.get('colsample_bytree', 0.8),
        'gamma': params.get('gamma', 0),
        'min_child_weight': params.get('min_child_weight', 1),
        'reg_alpha': params.get('reg_alpha', 0),
        'reg_lambda': params.get('reg_lambda', 1),
    }
    
    # Prepare data
    common_idx = features.index.intersection(labels.index).intersection(data.index)
    X = features.loc[common_idx].copy()
    y = labels.loc[common_idx].copy()
    
    # Handle NaN
    X = X.fillna(method='ffill').fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        **xgb_params,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_scaled, y_encoded)
    
    # Get predictions
    probs = model.predict_proba(X_scaled)
    
    # Map class indices
    classes = le.classes_
    buy_idx = np.where(classes == -1)[0][0] if -1 in classes else None
    sell_idx = np.where(classes == 1)[0][0] if 1 in classes else None
    
    # Generate signals
    signals = pd.Series(0, index=common_idx)
    
    if buy_idx is not None:
        buy_mask = probs[:, buy_idx] > ml_buy_thresh
        signals.loc[common_idx[buy_mask]] = 1
    
    if sell_idx is not None:
        sell_mask = probs[:, sell_idx] > ml_sell_thresh
        signals.loc[common_idx[sell_mask]] = -1
    
    # Apply RSI filter
    if use_rsi_filter and 'rsi_14' in features.columns:
        rsi = features.loc[common_idx, 'rsi_14']
        # Only BUY when RSI is oversold
        buy_signals = signals == 1
        signals.loc[buy_signals & (rsi > rsi_buy_thresh)] = 0
        # Only SELL when RSI is overbought
        sell_signals = signals == -1
        signals.loc[sell_signals & (rsi < rsi_sell_thresh)] = 0
    
    # Apply regime filter (only sell in bear markets)
    if use_regime_filter:
        sma200 = data.loc[common_idx, 'close'].rolling(200, min_periods=50).mean()
        is_bull = data.loc[common_idx, 'close'] > sma200
        # In bull markets, only sell on extreme conditions
        sell_signals = signals == -1
        if 'rsi_14' in features.columns:
            rsi = features.loc[common_idx, 'rsi_14']
            extreme_overbought = rsi > 80
            signals.loc[sell_signals & is_bull & ~extreme_overbought] = 0
    
    return signals, model, scaler, le


def optimize_strategy(train_data: pd.DataFrame, train_features: pd.DataFrame,
                      n_trials: int = 100, progress_callback=None) -> dict:
    """
    Use Optuna to find optimal parameters on TRAINING data only.
    """
    # Generate labels for training data
    train_labels = generate_labels(train_data, order=5)
    
    def objective(trial):
        params = {
            # XGBoost hyperparameters
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, log=True),
            
            # Signal thresholds
            'ml_buy_thresh': trial.suggest_float('ml_buy_thresh', 0.3, 0.8),
            'ml_sell_thresh': trial.suggest_float('ml_sell_thresh', 0.3, 0.8),
            
            # Rule-based filters
            'rsi_buy_thresh': trial.suggest_int('rsi_buy_thresh', 25, 45),
            'rsi_sell_thresh': trial.suggest_int('rsi_sell_thresh', 60, 85),
            'use_rsi_filter': trial.suggest_categorical('use_rsi_filter', [True, False]),
            'use_regime_filter': trial.suggest_categorical('use_regime_filter', [True, False]),
            
            # Peak/valley sensitivity
            'peak_valley_order': trial.suggest_int('peak_valley_order', 3, 10),
        }
        
        # Use cross-validation on training data
        tscv = TimeSeriesSplit(n_splits=3)
        returns = []
        total_trades = 0
        max_dd = 0
        
        for train_idx, val_idx in tscv.split(train_data):
            cv_train_data = train_data.iloc[train_idx]
            cv_val_data = train_data.iloc[val_idx]
            cv_train_features = train_features.iloc[train_idx]
            cv_val_features = train_features.iloc[val_idx]
            
            # Generate labels with trial's order
            cv_labels = generate_labels(cv_train_data, order=params['peak_valley_order'])
            
            try:
                # Train model on CV train data
                signals, model, scaler, le = generate_signals_from_params(
                    cv_train_data, cv_train_features, cv_labels, params
                )
                
                # Apply trained model to validation data (OUT OF SAMPLE)
                cv_val_features_clean = cv_val_features.fillna(method='ffill').fillna(0)
                val_scaled = scaler.transform(cv_val_features_clean)
                val_probs = model.predict_proba(val_scaled)
                
                # Generate validation signals
                classes = le.classes_
                buy_idx = np.where(classes == -1)[0][0] if -1 in classes else None
                sell_idx = np.where(classes == 1)[0][0] if 1 in classes else None
                
                val_signals = pd.Series(0, index=cv_val_data.index)
                
                if buy_idx is not None:
                    buy_mask = val_probs[:, buy_idx] > params['ml_buy_thresh']
                    val_signals.iloc[buy_mask] = 1
                
                if sell_idx is not None:
                    sell_mask = val_probs[:, sell_idx] > params['ml_sell_thresh']
                    val_signals.iloc[sell_mask] = -1
                
                # Apply RSI filter to validation signals
                if params.get('use_rsi_filter', True) and 'rsi_14' in cv_val_features.columns:
                    rsi = cv_val_features['rsi_14']
                    val_signals.loc[(val_signals == 1) & (rsi > params['rsi_buy_thresh'])] = 0
                    val_signals.loc[(val_signals == -1) & (rsi < params['rsi_sell_thresh'])] = 0
                
                backtest = calculate_backtest_return(cv_val_data, val_signals)
                returns.append(backtest['total_return'])
                total_trades += backtest['num_trades']
                max_dd = min(max_dd, backtest['max_drawdown'])
            except Exception as e:
                returns.append(-100)
        
        avg_return = np.mean(returns)
        
        # Penalize too few trades (across all folds)
        if total_trades < 3:
            avg_return -= 30
        
        # Penalize excessive drawdown
        if max_dd < -60:
            avg_return -= 20
        
        if progress_callback:
            progress_callback(trial.number, n_trials, avg_return)
        
        return avg_return
    
    # Run optimization with parallel trials
    import os
    n_jobs = os.cpu_count() or 4  # Use all available CPU cores
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)
    
    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'study': study
    }


def _optimize_single_window(args):
    """
    Helper function to optimize a single window (for parallel processing).
    """
    i, train_data, test_data, train_features, test_features, n_trials_per_window, n = args
    
    if len(train_data) < 100 or len(test_data) < 20:
        return None
    
    # Optimize on training window (uses parallel trials internally)
    opt_result = optimize_strategy(train_data, train_features, 
                                   n_trials=n_trials_per_window)
    
    # Generate signals for test window using optimized params
    train_labels = generate_labels(train_data, 
                                   order=opt_result['best_params'].get('peak_valley_order', 5))
    
    signals, model, scaler, le = generate_signals_from_params(
        train_data, train_features, train_labels, opt_result['best_params']
    )
    
    # Apply model to test data
    test_features_clean = test_features.fillna(method='ffill').fillna(0)
    test_scaled = scaler.transform(test_features_clean)
    test_probs = model.predict_proba(test_scaled)
    
    # Generate test signals
    classes = le.classes_
    buy_idx = np.where(classes == -1)[0][0] if -1 in classes else None
    sell_idx = np.where(classes == 1)[0][0] if 1 in classes else None
    
    test_signals = pd.Series(0, index=test_data.index)
    ml_buy_thresh = opt_result['best_params']['ml_buy_thresh']
    ml_sell_thresh = opt_result['best_params']['ml_sell_thresh']
    
    if buy_idx is not None:
        buy_mask = test_probs[:, buy_idx] > ml_buy_thresh
        test_signals.iloc[buy_mask] = 1
    
    if sell_idx is not None:
        sell_mask = test_probs[:, sell_idx] > ml_sell_thresh
        test_signals.iloc[sell_mask] = -1
    
    # Apply filters
    if opt_result['best_params'].get('use_rsi_filter', True) and 'rsi_14' in test_features.columns:
        rsi = test_features['rsi_14']
        rsi_buy = opt_result['best_params']['rsi_buy_thresh']
        rsi_sell = opt_result['best_params']['rsi_sell_thresh']
        test_signals.loc[(test_signals == 1) & (rsi > rsi_buy)] = 0
        test_signals.loc[(test_signals == -1) & (rsi < rsi_sell)] = 0
    
    # Calculate test performance
    test_backtest = calculate_backtest_return(test_data, test_signals)
    
    return {
        'window': i,
        'train_period': f"{train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}",
        'test_period': f"{test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}",
        'best_params': opt_result['best_params'],
        'test_return': test_backtest['total_return'],
        'test_bh': test_backtest['buy_hold_return'],
        'test_alpha': test_backtest['alpha'],
        'test_trades': test_backtest['num_trades'],
        'test_win_rate': test_backtest['win_rate'],
        'test_signals': test_signals,
        'test_data_index': test_data.index
    }


def walk_forward_optimize(data: pd.DataFrame, features: pd.DataFrame,
                          n_windows: int = 5, train_pct: float = 0.6,
                          n_trials_per_window: int = 50,
                          progress_callback=None,
                          parallel_windows: bool = False) -> dict:
    """
    Walk-forward optimization: Train on window, test on next period, roll forward.
    This is the GOLD STANDARD for realistic backtesting.
    
    Args:
        parallel_windows: If True, optimize windows in parallel (uses more memory but faster)
                         If False, optimize sequentially but each window uses parallel trials
    """
    n = len(data)
    window_size = n // n_windows
    
    all_signals = pd.Series(0, index=data.index)
    all_results = []
    
    # Prepare window data
    window_args = []
    for i in range(n_windows - 1):
        train_start = i * window_size
        train_end = train_start + int(window_size * train_pct)
        test_start = train_end
        test_end = (i + 1) * window_size if i < n_windows - 2 else n
        
        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]
        train_features = features.iloc[train_start:train_end]
        test_features = features.iloc[test_start:test_end]
        
        window_args.append((i, train_data, test_data, train_features, test_features, n_trials_per_window, n))
    
    if parallel_windows:
        # Parallel window optimization (faster but uses more memory)
        from joblib import Parallel, delayed
        import os
        n_jobs = min(os.cpu_count() or 4, len(window_args))
        
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_optimize_single_window)(args) for args in window_args
        )
        
        for result in results:
            if result is not None:
                all_signals.loc[result['test_data_index']] = result['test_signals']
                del result['test_signals']
                del result['test_data_index']
                all_results.append(result)
    else:
        # Sequential window optimization (each window uses parallel trials)
        for idx, args in enumerate(window_args):
            result = _optimize_single_window(args)
            if result is not None:
                all_signals.loc[result['test_data_index']] = result['test_signals']
                del result['test_signals']
                del result['test_data_index']
                all_results.append(result)
            
            if progress_callback:
                progress_callback(idx + 1, len(window_args))
    
    # Calculate overall performance
    final_backtest = calculate_backtest_return(data, all_signals)
    
    return {
        'signals': all_signals,
        'window_results': all_results,
        'overall_return': final_backtest['total_return'],
        'overall_bh': final_backtest['buy_hold_return'],
        'overall_alpha': final_backtest['alpha'],
        'overall_trades': final_backtest['num_trades'],
        'overall_win_rate': final_backtest['win_rate'],
        'overall_max_dd': final_backtest['max_drawdown'],
        'backtest': final_backtest
    }


# =============================================================================
# STREAMLIT PAGE
# =============================================================================

def render_optimized_strategy_page(data: pd.DataFrame, features: pd.DataFrame):
    """Main page renderer for Optimized Strategy."""
    
    st.header("🔬 Optimized Strategy")
    st.info("**Optuna optimization + Proper train/test split + Walk-forward validation** = Realistic performance")
    
    st.caption(f"📊 Using data from sidebar: {len(data)} days loaded")
    
    if not OPTUNA_AVAILABLE:
        st.error("Optuna not available. Please install: `pip install optuna`")
        return
    
    if not XGB_AVAILABLE:
        st.error("XGBoost not available. Please install: `pip install xgboost`")
        return
    
    if data.empty:
        st.warning("No data loaded. Please configure ticker in sidebar.")
        return
    
    # Configuration - all settings are specific to this page with unique keys
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Data Split")
        opt_train_ratio = st.slider("Training Data %", 50, 80, 70, 5, key="optimized_train_ratio",
                               help="Percentage of data for training. Rest is for testing.")
        
        split_info = split_data(data, features, opt_train_ratio / 100)
        st.caption(f"Train: {len(split_info['train_data'])} days")
        st.caption(f"Test: {len(split_info['test_data'])} days")
        st.caption(f"Split date: {split_info['split_date'].strftime('%Y-%m-%d')}")
    
    with col2:
        st.subheader("🎯 Optimization")
        opt_mode = st.radio("Mode", ["Simple (Train/Test)", "Walk-Forward (Best)"],
                           key="optimized_mode",
                           help="Walk-forward is more realistic but slower")
        
        if "Simple" in opt_mode:
            opt_n_trials = st.slider("Optuna Trials", 20, 500, 100, 10, key="optimized_n_trials")
        else:
            opt_n_windows = st.slider("Walk-Forward Windows", 3, 10, 5, key="optimized_n_windows")
            opt_n_trials_per = st.slider("Trials per Window", 10, 100, 30, key="optimized_trials_per")
            opt_parallel_windows = st.checkbox("Parallel Windows", value=False, key="optimized_parallel",
                                          help="Optimize all windows in parallel (faster but uses more memory)")
    
    with col3:
        st.subheader("⚙️ Settings")
        opt_show_details = st.checkbox("Show optimization details", value=True, key="optimized_details")
        opt_save_results = st.checkbox("Save best parameters", value=True, key="optimized_save")
    
    # Run Optimization
    if st.button("🚀 Run Optimization", type="primary", key="opt_run"):
        
        if "Simple" in opt_mode:
            # Simple train/test split
            st.markdown("---")
            st.subheader("📈 Simple Train/Test Optimization")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(trial_num, total, value):
                progress_bar.progress(trial_num / total)
                status_text.text(f"Trial {trial_num}/{total} | Best: {value:.1f}%")
            
            with st.spinner("Optimizing on training data..."):
                opt_result = optimize_strategy(
                    split_info['train_data'], 
                    split_info['train_features'],
                    n_trials=opt_n_trials,
                    progress_callback=progress_callback
                )
            
            progress_bar.progress(1.0)
            status_text.text("Optimization complete!")
            
            # Show best parameters
            if opt_show_details:
                with st.expander("🎛️ Best Parameters", expanded=True):
                    params_df = pd.DataFrame([opt_result['best_params']]).T
                    params_df.columns = ['Value']
                    st.dataframe(params_df)
            
            # Generate signals for BOTH train and test
            train_labels = generate_labels(split_info['train_data'], 
                                          order=opt_result['best_params'].get('peak_valley_order', 5))
            
            train_signals, model, scaler, le = generate_signals_from_params(
                split_info['train_data'],
                split_info['train_features'],
                train_labels,
                opt_result['best_params']
            )
            
            # Apply to test data (OUT OF SAMPLE)
            test_features_clean = split_info['test_features'].fillna(method='ffill').fillna(0)
            test_scaled = scaler.transform(test_features_clean)
            test_probs = model.predict_proba(test_scaled)
            
            classes = le.classes_
            buy_idx = np.where(classes == -1)[0][0] if -1 in classes else None
            sell_idx = np.where(classes == 1)[0][0] if 1 in classes else None
            
            test_signals = pd.Series(0, index=split_info['test_data'].index)
            ml_buy_thresh = opt_result['best_params']['ml_buy_thresh']
            ml_sell_thresh = opt_result['best_params']['ml_sell_thresh']
            
            if buy_idx is not None:
                buy_mask = test_probs[:, buy_idx] > ml_buy_thresh
                test_signals.iloc[buy_mask] = 1
            
            if sell_idx is not None:
                sell_mask = test_probs[:, sell_idx] > ml_sell_thresh
                test_signals.iloc[sell_mask] = -1
            
            # Apply filters to test signals
            if opt_result['best_params'].get('use_rsi_filter', True) and 'rsi_14' in split_info['test_features'].columns:
                rsi = split_info['test_features']['rsi_14']
                rsi_buy = opt_result['best_params']['rsi_buy_thresh']
                rsi_sell = opt_result['best_params']['rsi_sell_thresh']
                test_signals.loc[(test_signals == 1) & (rsi > rsi_buy)] = 0
                test_signals.loc[(test_signals == -1) & (rsi < rsi_sell)] = 0
            
            # Calculate performance
            train_backtest = calculate_backtest_return(split_info['train_data'], train_signals)
            test_backtest = calculate_backtest_return(split_info['test_data'], test_signals)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Results")
            
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                st.markdown("### 🎓 Training Performance (In-Sample)")
                st.metric("Return", f"{train_backtest['total_return']:.1f}%",
                         f"{train_backtest['alpha']:+.1f}% vs B&H")
                st.metric("Win Rate", f"{train_backtest['win_rate']:.1f}%")
                st.metric("Trades", train_backtest['num_trades'])
                st.caption(f"Buy & Hold: {train_backtest['buy_hold_return']:.1f}%")
            
            with col_r2:
                st.markdown("### 🧪 Test Performance (OUT-OF-SAMPLE)")
                st.metric("Return", f"{test_backtest['total_return']:.1f}%",
                         f"{test_backtest['alpha']:+.1f}% vs B&H",
                         delta_color="normal")
                st.metric("Win Rate", f"{test_backtest['win_rate']:.1f}%")
                st.metric("Trades", test_backtest['num_trades'])
                st.caption(f"Buy & Hold: {test_backtest['buy_hold_return']:.1f}%")
            
            # Key insight
            if test_backtest['total_return'] > test_backtest['buy_hold_return']:
                st.success("✅ Strategy BEATS Buy & Hold on unseen test data!")
            else:
                st.warning("⚠️ Strategy underperforms Buy & Hold on test data. May need more optimization or different approach.")
            
            # Combine signals for chart
            all_signals = pd.concat([train_signals, test_signals])
            
        else:
            # Walk-forward optimization
            st.markdown("---")
            st.subheader("📈 Walk-Forward Optimization")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def wf_progress(window, total):
                progress_bar.progress(window / total)
                status_text.text(f"Window {window}/{total}")
            
            # Show CPU info
            import os
            cpu_count = os.cpu_count() or 4
            st.info(f"🖥️ Using {cpu_count} CPU cores for parallel optimization")
            
            with st.spinner("Running walk-forward optimization..."):
                wf_result = walk_forward_optimize(
                    data, features,
                    n_windows=opt_n_windows,
                    n_trials_per_window=opt_n_trials_per,
                    progress_callback=wf_progress if not opt_parallel_windows else None,
                    parallel_windows=opt_parallel_windows
                )
            
            progress_bar.progress(1.0)
            status_text.text("Walk-forward complete!")
            
            # Show window results
            if opt_show_details:
                with st.expander("📋 Window Results", expanded=True):
                    for wr in wf_result['window_results']:
                        st.markdown(f"**Window {wr['window'] + 1}**")
                        st.caption(f"Train: {wr['train_period']}")
                        st.caption(f"Test: {wr['test_period']}")
                        st.write(f"Test Return: {wr['test_return']:.1f}% | B&H: {wr['test_bh']:.1f}% | Alpha: {wr['test_alpha']:+.1f}%")
                        st.markdown("---")
            
            # Overall results
            st.subheader("📊 Overall Walk-Forward Results")
            
            col_wf1, col_wf2, col_wf3, col_wf4, col_wf5 = st.columns(5)
            
            col_wf1.metric("Total Return", f"{wf_result['overall_return']:.1f}%",
                          f"{wf_result['overall_alpha']:+.1f}% vs B&H")
            col_wf2.metric("Buy & Hold", f"{wf_result['overall_bh']:.1f}%")
            col_wf3.metric("Win Rate", f"{wf_result['overall_win_rate']:.1f}%")
            col_wf4.metric("Max Drawdown", f"{wf_result['overall_max_dd']:.1f}%")
            col_wf5.metric("Trades", wf_result['overall_trades'])
            
            all_signals = wf_result['signals']
            test_backtest = wf_result['backtest']
        
        # Chart
        st.subheader("📉 Price Chart with Signals")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
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
        
        # Train/Test split line (for simple mode)
        if "Simple" in opt_mode:
            fig.add_vline(x=split_info['split_date'], line_dash="dash", 
                         line_color="yellow", annotation_text="Train/Test Split")
        
        # BUY signals
        buy_dates = all_signals[all_signals == 1].index
        if len(buy_dates) > 0:
            buy_prices = data.loc[buy_dates, 'low'] * 0.98
            fig.add_trace(go.Scatter(
                x=buy_dates, y=buy_prices, mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='lime',
                           line=dict(width=1, color='darkgreen')),
                name='BUY Signal'
            ), row=1, col=1)
        
        # SELL signals
        sell_dates = all_signals[all_signals == -1].index
        if len(sell_dates) > 0:
            sell_prices = data.loc[sell_dates, 'high'] * 1.02
            fig.add_trace(go.Scatter(
                x=sell_dates, y=sell_prices, mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red',
                           line=dict(width=1, color='darkred')),
                name='SELL Signal'
            ), row=1, col=1)
        
        # Equity curve
        equity = test_backtest['equity']
        equity_dates = data.index[:len(equity)]
        fig.add_trace(go.Scatter(
            x=equity_dates, y=equity,
            mode='lines', name='Strategy Equity',
            line=dict(color='cyan', width=2)
        ), row=2, col=1)
        
        # Buy & Hold
        bh_equity = 100000 * (data['close'] / data['close'].iloc[0])
        fig.add_trace(go.Scatter(
            x=data.index, y=bh_equity,
            mode='lines', name='Buy & Hold',
            line=dict(color='gray', width=1, dash='dash')
        ), row=2, col=1)
        
        fig.update_layout(
            height=700,
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Save results
        if opt_save_results:
            save_path = os.path.join("saved_models_v2", "optimized_strategy_params.json")
            os.makedirs("saved_models_v2", exist_ok=True)
            
            if "Simple" in opt_mode:
                save_data = {
                    'params': opt_result['best_params'],
                    'train_return': train_backtest['total_return'],
                    'test_return': test_backtest['total_return'],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                save_data = {
                    'overall_return': wf_result['overall_return'],
                    'overall_alpha': wf_result['overall_alpha'],
                    'window_results': wf_result['window_results'],
                    'timestamp': datetime.now().isoformat()
                }
            
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            st.success(f"✅ Results saved to {save_path}")
        
        # Store in session state
        st.session_state['opt_signals'] = all_signals
        st.session_state['opt_backtest'] = test_backtest


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="Optimized Strategy", layout="wide")
    st.warning("Run this from the main app (peak_valley_ml_v2.py)")
