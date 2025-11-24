"""
Fixed version of optimization.py with proper error handling and dynamic trade frequency parameters
"""

import numpy as np
import pandas as pd
import optuna
import itertools
import warnings
warnings.filterwarnings('ignore')

from backtester import Backtester
import streamlit as st
import tensorflow as tf
import logging

# Suppress optuna and tensorflow verbosity
optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

def universal_strategy(data, params, debug=False):
    """
    Universal strategy that works with any combination of indicators.
    Uses a scoring system where each indicator contributes to buy/sell scores.
    """
    debug_count = getattr(universal_strategy, '_debug_count', 0)
    if debug and debug_count < 2:
        print(f"\n🎯 UNIVERSAL STRATEGY DEBUG (Call #{debug_count + 1})")
        print(f"   Data shape: {data.shape}")
        print(f"   Params keys: {list(params.keys())}")
        universal_strategy._debug_count = debug_count + 1
    
    if len(data) == 0:
        return pd.Series(0, index=data.index)
    
    # Initialize scores
    buy_scores = pd.Series(0, index=data.index)
    sell_scores = pd.Series(0, index=data.index) 
    
    # Process active indicators
    for key, value in params.items():
        if key.startswith('use_') and value:
            indicator_name = key[4:]  # Remove 'use_' prefix
            
            if indicator_name in data.columns:
                if data[indicator_name].dtype == 'bool':
                    # Boolean indicator: True = buy signal, False = sell signal
                    buy_scores += data[indicator_name].astype(int)
                    sell_scores += (~data[indicator_name]).astype(int)
                else:
                    # Numerical indicator: use thresholds
                    buy_threshold_key = f'{indicator_name}_buy_threshold'
                    sell_threshold_key = f'{indicator_name}_sell_threshold'
                    
                    if buy_threshold_key in params and sell_threshold_key in params:
                        buy_threshold = params[buy_threshold_key]
                        sell_threshold = params[sell_threshold_key]
                        
                        # Buy when indicator is low (oversold), sell when high (overbought)
                        buy_signals = data[indicator_name] <= buy_threshold
                        sell_signals = data[indicator_name] >= sell_threshold
                        
                        buy_scores += buy_signals.astype(int)
                        sell_scores += sell_signals.astype(int)
    
    # Convert scores to signals using thresholds
    buy_threshold = params.get('buy_score_threshold', 1)
    sell_threshold = params.get('sell_score_threshold', 1)
    
    # Generate raw signals
    raw_signals = pd.Series(0, index=data.index)
    raw_signals[buy_scores >= buy_threshold] = 1   # Buy
    raw_signals[sell_scores >= sell_threshold] = -1  # Sell
    
    # Apply anti-whipsaw filters
    filtered_signals = raw_signals.copy()
    
    # 1. Minimum hold days filter
    min_hold_days = params.get('min_hold_days', 1)
    if min_hold_days > 1:
        for i in range(1, len(filtered_signals)):
            if filtered_signals.iloc[i-1] != 0:  # Previous day had a signal
                # Block opposite signals for min_hold_days
                lookback = min(min_hold_days, i)
                if any(filtered_signals.iloc[i-lookback:i] == -filtered_signals.iloc[i]):
                    filtered_signals.iloc[i] = 0
    
    # 2. Confirmation requirement
    if params.get('require_confirmation', False):
        confirmed_signals = pd.Series(0, index=data.index)
        for i in range(1, len(filtered_signals)):
            if (filtered_signals.iloc[i] == filtered_signals.iloc[i-1] and 
                filtered_signals.iloc[i] != 0):
                confirmed_signals.iloc[i] = filtered_signals.iloc[i]
        filtered_signals = confirmed_signals
    
    # 3. Trend strength filter - only trade in strong trends
    if params.get('use_trend_filter', False):
        trend_strength = 0
        trend_count = 0
        
        # Method 1: ADX-based trend detection
        if 'ADX_14' in data.columns:
            adx_threshold = params.get('adx_threshold', 20)
            adx_strong_trend = data['ADX_14'] > adx_threshold
            trend_count += 1
        else:
            adx_strong_trend = pd.Series(True, index=data.index)
        
        # Method 2: Price-based trend detection
        ma_trend = pd.Series(True, index=data.index)
        if 'SMA_50' in data.columns and 'close' in data.columns:
            price_above_ma = data['close'] > data['SMA_50']
            ma_trend = ma_trend & price_above_ma
        
        # Combine both methods
        strong_trend = adx_strong_trend | ma_trend
        
        # Check Supertrend if available
        if any('SUPERT' in col for col in data.columns):
            st_cols = [col for col in data.columns if 'SUPERT_' in col and 'SUPERTd' in col]
            if st_cols:
                st_trend = data[st_cols[0]]
                filtered_signals = filtered_signals.where(
                    (filtered_signals != 1) | (st_trend == 1),
                    0
                )
        
        # Only trade during strong trends
        filtered_signals = filtered_signals.where(strong_trend, 0)
    
    return filtered_signals

def objective(trial, data, trade_preference=0.5):
    """
    Objective function for Optuna with proper error handling and dynamic parameters.
    """
    try:
        # Validate input data
        if data is None or data.empty:
            return float('-inf')
        
        # Check for required columns
        required_cols = ['close', 'open', 'high', 'low', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return float('-inf')
        
        numerical_indicators = [col for col in data.columns if data[col].dtype != 'bool' and col not in required_cols]
        boolean_indicators = [col for col in data.columns if data[col].dtype == 'bool']
        
        # Debug output for first few trials
        if trial.number < 2:
            print(f"\n🎯 OBJECTIVE FUNCTION - TRIAL {trial.number}")
            print(f"   Data shape: {data.shape}")
            print(f"   Trade Preference: {trade_preference:.2f}")
            print(f"   Numerical indicators: {len(numerical_indicators)}")
            print(f"   Boolean indicators: {len(boolean_indicators)}")

        params = {}
        active_indicators = []

        # Add flexible scoring thresholds
        params['buy_score_threshold'] = trial.suggest_int('buy_score_threshold', 1, 5)
        params['sell_score_threshold'] = trial.suggest_int('sell_score_threshold', 1, 5)
        
        # === DYNAMIC ANTI-WHIPSAW PARAMETERS ===
        # 1. min_hold_days: Conservative = longer holds, Aggressive = shorter holds
        if trade_preference < 0.4:  # Conservative: 3-10 days
            min_hold_min, min_hold_max = 3, 10
        elif trade_preference < 0.7:  # Balanced: 1-7 days  
            min_hold_min, min_hold_max = 1, 7
        else:  # Aggressive: 1-3 days (quick trades)
            min_hold_min, min_hold_max = 1, 3
        params['min_hold_days'] = trial.suggest_int('min_hold_days', min_hold_min, min_hold_max)
        
        # 2. require_confirmation: Conservative = more likely to require, Aggressive = more likely to skip
        if trade_preference < 0.4:  # Conservative: 80% chance of requiring confirmation
            confirmation_choices = [True, True, True, True, False]
        elif trade_preference < 0.7:  # Balanced: 50/50
            confirmation_choices = [True, False]
        else:  # Aggressive: 20% chance of requiring confirmation
            confirmation_choices = [True, False, False, False, False]
        params['require_confirmation'] = trial.suggest_categorical('require_confirmation', confirmation_choices)
        
        # 3. use_trend_filter: Conservative = more likely to use filter, Aggressive = more likely to skip
        if trade_preference < 0.4:  # Conservative: 80% chance of using trend filter
            trend_filter_choices = [True, True, True, True, False]
        elif trade_preference < 0.7:  # Balanced: 50/50
            trend_filter_choices = [True, False]
        else:  # Aggressive: 20% chance of using trend filter
            trend_filter_choices = [True, False, False, False, False]
        params['use_trend_filter'] = trial.suggest_categorical('use_trend_filter', trend_filter_choices)
        
        # 4. adx_threshold: Conservative = higher threshold, Aggressive = lower threshold
        if trade_preference < 0.4:  # Conservative: 22-30 (strong trends only)
            adx_min, adx_max = 22, 30
        elif trade_preference < 0.7:  # Balanced: 18-26
            adx_min, adx_max = 18, 26
        else:  # Aggressive: 15-22 (weaker trends allowed)
            adx_min, adx_max = 15, 22
        params['adx_threshold'] = trial.suggest_int('adx_threshold', adx_min, adx_max)
        
        # Process numerical indicators
        for indicator in numerical_indicators:
            use_indicator = trial.suggest_categorical(f'use_{indicator}', [True, False])
            if use_indicator:
                min_val, max_val = data[indicator].min(), data[indicator].max()
                
                if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
                    continue
                
                range_size = max_val - min_val
                buffer = range_size * 0.01
                midpoint = min_val + range_size * 0.5
                
                buy_max = midpoint - buffer
                sell_min = midpoint + buffer
                
                if buy_max <= min_val:
                    buy_max = min_val + range_size * 0.25
                if sell_min >= max_val:
                    sell_min = max_val - range_size * 0.25
                
                params[f'{indicator}_buy_threshold'] = trial.suggest_float(
                    f'{indicator}_buy_threshold', min_val, buy_max
                )
                params[f'{indicator}_sell_threshold'] = trial.suggest_float(
                    f'{indicator}_sell_threshold', sell_min, max_val
                )
                active_indicators.append(indicator)
        
        # Process boolean indicators
        for indicator in boolean_indicators:
            use_indicator = trial.suggest_categorical(f'use_{indicator}', [True, False])
            if use_indicator:
                active_indicators.append(indicator)
        
        # Ensure we have at least one active indicator
        if len(active_indicators) == 0:
            return float('-inf')
        
        # Run backtest
        try:
            backtester = Backtester(data, f"Trial_{trial.number}", universal_strategy, params, 10000)
            backtester.run()
            trade_log, summary = backtester.get_results()
            
            if summary is None or 'total_return_pct' not in summary:
                return float('-inf')
            
            total_return = summary['total_return_pct']
            num_trades = len(trade_log)
            win_rate = summary.get('win_rate', 0) / 100.0
            profit_factor = summary.get('profit_factor', 1)
            max_drawdown = summary.get('max_drawdown_pct', 0)
            
            if num_trades == 0 or total_return <= -50:
                return float('-inf')
            
            # Dynamic scoring based on trade preference
            very_high_threshold = 50 + (trade_preference * 30)  # 50-80 trades
            high_threshold = 30 + (trade_preference * 20)       # 30-50 trades
            low_threshold = 20 + (trade_preference * 10)        # 20-30 trades
            very_low_threshold = 10 + (trade_preference * 10)   # 10-20 trades
            
            penalty_strength = 1.0 - (trade_preference * 0.3)
            bonus_strength = 1.0 + (0.5 - trade_preference) * 0.4
            
            # Trade frequency scoring
            if num_trades > very_high_threshold:
                trade_penalty = 0.5 + (0.2 * penalty_strength)
            elif num_trades > high_threshold:
                trade_penalty = 0.7 + (0.1 * penalty_strength)
            elif num_trades < very_low_threshold:
                trade_penalty = 1.1 * bonus_strength
            else:
                trade_penalty = 1.0
            
            # Win rate bonus
            win_rate_bonus = 1.0 + (win_rate - 0.5) * 0.5
            
            # Profit factor bonus
            if profit_factor > 2.0:
                pf_bonus = 1.2
            elif profit_factor > 1.5:
                pf_bonus = 1.1
            elif profit_factor < 1.1:
                pf_bonus = 0.9
            else:
                pf_bonus = 1.0
            
            # Drawdown penalty
            if max_drawdown > 30:
                dd_penalty = 0.7
            elif max_drawdown > 20:
                dd_penalty = 0.85
            elif max_drawdown < 10:
                dd_penalty = 1.15
            else:
                dd_penalty = 1.0
            
            # Calculate quality score
            quality_score = (
                total_return 
                * trade_penalty
                * win_rate_bonus
                * pf_bonus
                * dd_penalty
            )
            
            # Debug output for significant trials
            if trial.number <= 3 or total_return > 50:
                print(f"\n  📊 TRIAL {trial.number} SCORING:")
                print(f"    Return: {total_return:.1f}%, Trades: {num_trades}")
                print(f"    Quality Score: {quality_score:.1f}")
            
            return quality_score
            
        except Exception as e:
            return float('-inf')
            
    except Exception as e:
        return float('-inf')

def run_optimization(data, n_trials=5000, n_jobs=None, progress_callback=None, trade_preference=0.5):
    """Run optimization with proper error handling"""
    try:
        print(f"🚀 Starting optimization...")
        print(f"🎯 Trials: {n_trials:,}")
        print(f"💰 Trade Preference: {trade_preference:.2f}")
        
        study = optuna.create_study(direction='maximize')
        
        if n_jobs == 1:
            study.optimize(
                lambda trial: objective(trial, data, trade_preference), 
                n_trials=n_trials, 
                show_progress_bar=True
            )
        else:
            study.optimize(
                lambda trial: objective(trial, data, trade_preference), 
                n_trials=n_trials, 
                n_jobs=n_jobs,
                show_progress_bar=True
            )
        
        # Process results
        successful_trials = [trial for trial in study.trials if trial.value is not None and trial.value > 0]
        
        if len(successful_trials) == 0:
            print("❌ No successful trials found!")
            return []
        
        # Sort by value and return top 3
        successful_trials.sort(key=lambda x: x.value, reverse=True)
        top_trials = successful_trials[:3]
        
        print(f"✅ Found {len(successful_trials)} successful strategies")
        return top_trials
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return []
