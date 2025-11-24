import optuna
import pandas as pd
import numpy as np
import itertools
from backtester import Backtester
import joblib
from joblib import Parallel, delayed

# Import signal optimization enhancements
try:
    from signal_optimization import (
        analyze_signal_distribution, 
        enhance_signal_scoring,
        optimize_signal_parameters
    )
    SIGNAL_OPTIMIZATION_AVAILABLE = True
except ImportError:
    SIGNAL_OPTIMIZATION_AVAILABLE = False
    print("⚠️  Signal optimization module not available - using basic optimization")

# Import Enhancement #2: Position Sizing & Risk Management
try:
    from enhanced_backtester import EnhancedBacktester, create_enhanced_backtester
    from position_sizing import PositionSizer, optimize_position_sizing_parameters
    POSITION_SIZING_AVAILABLE = True
    print("✅ Enhancement #2: Position Sizing & Risk Management available")
except ImportError as e:
    POSITION_SIZING_AVAILABLE = False
    print(f"⚠️  Enhancement #2 not available: {e}")
except Exception as e:
    POSITION_SIZING_AVAILABLE = False
    print(f"❌ Enhancement #2 import error: {e}")

# Import Detailed Logging System
try:
    from detailed_logger import create_comprehensive_strategy_log
    DETAILED_LOGGING_AVAILABLE = True
    print("✅ Detailed CSV logging system available")
except ImportError as e:
    DETAILED_LOGGING_AVAILABLE = False
    print(f"⚠️  Detailed logging not available: {e}")
except Exception as e:
    DETAILED_LOGGING_AVAILABLE = False
    print(f"❌ Detailed logging import error: {e}")

# Import Enhancement #3: Market Regime Detection
try:
    from market_regime_detector import MarketRegimeDetector, create_regime_aware_strategy_params
    MARKET_REGIME_AVAILABLE = True
    print("✅ Enhancement #3: Market Regime Detection available")
except ImportError as e:
    MARKET_REGIME_AVAILABLE = False
    print(f"⚠️  Enhancement #3 not available: {e}")
except Exception as e:
    MARKET_REGIME_AVAILABLE = False
    print(f"❌ Enhancement #3 import error: {e}")

def universal_strategy(data, params):
    """
    A universal strategy that uses a scoring system to combine signals.
    """
    buy_score = pd.Series(0, index=data.index)
    sell_score = pd.Series(0, index=data.index)

    # Extract active indicators from the flat params structure
    # EXCLUDE meta-flags like trend_filter, dynamic sizing, weighted scoring, regime detection, etc.
    meta_flags = {
        'use_trend_filter',
        'use_dynamic_position_sizing', 
        'use_weighted_scoring',
        'use_regime_detection',
        'use_indicators_ready',
        'use_volatility_sizing',        # Removed feature
        'use_dynamic_thresholds',       # Removed feature
        'volatility_multiplier',
        'volatility_threshold',
        'high_vol_threshold_reduction',
        'volatility_percentile_threshold',
        'buy_pct_min',
        'buy_pct_max', 
        'sell_pct_min',
        # Enhancement #3: Market Regime Detection
        'enable_market_regime',
        # Enhancement #2: Position Sizing & Risk Management
        'enable_position_sizing',
        'max_position_pct',
        'kelly_optimization',
        'volatility_adjustment',
        'position_confidence_weighting',
        'max_total_exposure',
        'aggressive_vol_scaling',
        'confidence_risk_multiplier',
        'sell_pct_max',
        # Old trade preference parameters
        'min_hold_days',
        'require_confirmation',
        'confirmation_weight',
        'trend_filter_weight',
        # APPROACH A: New optimization parameters with unique names
        'trend_adx_threshold',
        'trend_rsi_oversold',
        'trend_willr_threshold',
        'trend_stoch_threshold',
        'trend_sma_period',
        'signal_persistence_days',
        'buy_score_threshold',
        'sell_score_threshold',
    }

    active_indicators = [
        param.replace('use_', '')
        for param in params.keys()
        if param.startswith('use_')
        and params[param] is True
        and param not in meta_flags
    ]
    
    # Debug output for first few calls only
    debug_mode = True  # Temporarily enabled for debugging
    # Reset counter and always print to force debug output
    if not hasattr(universal_strategy, 'call_count'):
        universal_strategy.call_count = 0
    universal_strategy.call_count += 1
    
    # Reduce debug output for performance
    if universal_strategy.call_count <= 2:  # Only first 2 calls
        print(f"\n🔍 UNIVERSAL_STRATEGY CALL #{universal_strategy.call_count}")
        print(f"   Active indicators: {len(active_indicators)} | Thresholds: buy={params.get('buy_score_threshold')}, sell={params.get('sell_score_threshold')}")
    
    # Baseline: each active indicator contributes +1 to the relevant score when
    # its condition is met. No composite indicators or weighting.
    indicators_processed = 0

    # SIGNAL PERSISTENCE SYSTEM - Allow signals to "stack up" over time
    # This catches opportunities when indicators trigger at different times
    signal_persistence_days = params.get('signal_persistence_days', 2)  # Now optimized by Optuna!
    
    # Create persistent buy/sell scores that decay over time
    buy_score_persistent = pd.Series(0.0, index=data.index)
    sell_score_persistent = pd.Series(0.0, index=data.index)

    for indicator in active_indicators:
        if indicator not in data.columns:
            print(f"MISSING COLUMN: {indicator}")
            continue

        if indicator.startswith('pattern_') or indicator.startswith('dl_signal_'):
            # Boolean indicators: match on True/False flags
            buy_param = params.get(f'{indicator}_buy')
            sell_param = params.get(f'{indicator}_sell')

            if buy_param is not None and data[indicator].any():
                # Add 1.0 to buy score when condition is met, then decay over next days
                raw_signals = (data[indicator] == buy_param).astype(float)
                buy_score_persistent += raw_signals.rolling(window=signal_persistence_days, min_periods=1).sum()
                indicators_processed += 1
            if sell_param is not None and data[indicator].any():
                raw_signals = (data[indicator] == sell_param).astype(float)
                sell_score_persistent += raw_signals.rolling(window=signal_persistence_days, min_periods=1).sum()
        else:
            # Numerical indicators: compare against thresholds
            buy_threshold = params.get(f'{indicator}_buy')
            sell_threshold = params.get(f'{indicator}_sell')

            if buy_threshold is not None:
                # Add 1.0 to buy score when condition is met, then decay over next days
                raw_signals = (data[indicator] < buy_threshold).astype(float)
                buy_score_persistent += raw_signals.rolling(window=signal_persistence_days, min_periods=1).sum()
                indicators_processed += 1
            if sell_threshold is not None:
                raw_signals = (data[indicator] > sell_threshold).astype(float)
                sell_score_persistent += raw_signals.rolling(window=signal_persistence_days, min_periods=1).sum()

    # Convert persistent scores to signals with base thresholds
    buy_signals = buy_score_persistent >= params['buy_score_threshold']
    sell_signals = sell_score_persistent >= params['sell_score_threshold']

    # === APPLY TREND FILTER ===
    # If enabled, we only allow BUY signals when the trend filter is positive.
    # This acts as a safety net, allowing us to use more sensitive indicator thresholds
    # (e.g., buying dips) without catching falling knives in a crash.
    if params.get('use_trend_filter', False):
        # === APPROACH A: RECALCULATE TREND FILTER WITH OPTIMIZED PARAMETERS ===
        # Instead of using static trend filter, recalculate with trial-specific thresholds
        
        # Get optimized thresholds with new parameter names
        adx_threshold = params.get('trend_adx_threshold', 20)
        rsi_oversold_threshold = params.get('trend_rsi_oversold', 35)
        willr_threshold = params.get('trend_willr_threshold', -80)
        stoch_threshold = params.get('trend_stoch_threshold', 15)
        sma_period = params.get('trend_sma_period', 50)
        
        # Recalculate trend filter with optimized parameters
        dynamic_trend_filter = pd.Series(True, index=data.index)
        
        if 'ADX_14' in data.columns:
            adx_trend = data['ADX_14'] > adx_threshold
            dynamic_trend_filter = dynamic_trend_filter & adx_trend
        
        # Use optimized SMA period
        sma_column = f'SMA_{sma_period}'
        if sma_column in data.columns and 'close' in data.columns:
            price_trend = data['close'] > data[sma_column]
            dynamic_trend_filter = dynamic_trend_filter | price_trend
        elif 'SMA_50' in data.columns and 'close' in data.columns:
            # Fallback to SMA_50 if optimized period not available
            price_trend = data['close'] > data['SMA_50']
            dynamic_trend_filter = dynamic_trend_filter | price_trend

        if 'RSI_14' in data.columns:
            oversold_bailout = data['RSI_14'] < rsi_oversold_threshold
            dynamic_trend_filter = dynamic_trend_filter | oversold_bailout
        
        if 'WILLR_14' in data.columns:
            willr_oversold = data['WILLR_14'] < willr_threshold
            dynamic_trend_filter = dynamic_trend_filter | willr_oversold
            
        if 'STOCHk_14_3_3' in data.columns:
            stoch_oversold = data['STOCHk_14_3_3'] < stoch_threshold
            dynamic_trend_filter = dynamic_trend_filter | stoch_oversold
        
        # Apply the dynamically calculated trend filter
        buy_signals = buy_signals & dynamic_trend_filter

    # CRITICAL FIX: Make signals mutually exclusive to prevent backtester conflicts
    # When both buy and sell trigger, choose the stronger signal DETERMINISTICALLY
    conflicting_bars = buy_signals & sell_signals
    if conflicting_bars.any():
        # On conflicting bars, choose based on absolute score difference (deterministic)
        buy_excess = buy_score_persistent - params['buy_score_threshold']
        sell_excess = sell_score_persistent - params['sell_score_threshold']
        
        # Where buy excess is greater, keep buy signal and remove sell
        stronger_buy = conflicting_bars & (buy_excess > sell_excess)
        sell_signals.loc[stronger_buy] = False
        
        # Where sell excess is greater or equal, keep sell signal and remove buy  
        stronger_sell = conflicting_bars & (sell_excess >= buy_excess)
        buy_signals.loc[stronger_sell] = False

    # Optional debug output
    if debug_mode and universal_strategy.call_count <= 3:
        print(f"  PROCESSED: {indicators_processed} indicators")
        print(f"  FINAL SCORES: max_buy={buy_score_persistent.max():.1f}, max_sell={sell_score_persistent.max():.1f}")
        print(f"  SIGNALS: {buy_signals.sum()} buy, {sell_signals.sum()} sell")
        if conflicting_bars.any():
            print(f"  CONFLICTS RESOLVED: {conflicting_bars.sum()} conflicting bars fixed")

    signals = pd.Series(0, index=data.index)
    signals.loc[buy_signals] = 1
    signals.loc[sell_signals] = -1
    
    return signals

def enhanced_universal_strategy(data, params, use_signal_optimization=True):
    """
    Enhanced universal strategy with signal optimization features from Enhancement #1
    """
    if not use_signal_optimization or not SIGNAL_OPTIMIZATION_AVAILABLE:
        print(f"🔄 FALLBACK: Using original strategy (opt={use_signal_optimization}, avail={SIGNAL_OPTIMIZATION_AVAILABLE})")
        return universal_strategy(data, params)
    
    print(f"🚀 ENHANCED: Using enhanced strategy with signal optimization")
    
    # Enhancement #1: Use adaptive thresholds and confidence weighting
    enhanced_params = enhance_signal_scoring(data, params)
    
    # Get base signal scores using the enhanced parameters
    buy_score = pd.Series(0.0, index=data.index)
    sell_score = pd.Series(0.0, index=data.index)
    
    # Extract active indicators (same logic as original)
    meta_flags = {
        'buy_score_threshold', 'sell_score_threshold', 'signal_persistence_days',
        'min_hold_days', 'require_confirmation', 'confirmation_weight',
        'use_weighted_scoring', 'use_volatility_sizing', 'use_dynamic_thresholds',
        'use_regime_detection', 'use_dynamic_position_sizing', 'use_trend_filter',
        'trend_adx_threshold', 'trend_rsi_oversold', 'trend_willr_threshold',
        'trend_stoch_threshold', 'trend_sma_period', 'trend_filter_weight',
        'volatility_threshold', 'volatility_multiplier', 'volatility_percentile_threshold',
        'high_vol_threshold_reduction', 'buy_pct_min', 'buy_pct_max',
        'sell_pct_min', 'sell_pct_max', 'use_indicators_ready',
        # New enhancement parameters
        'adaptive_buy_threshold', 'adaptive_sell_threshold', 
        'buy_confidence_weights', 'sell_confidence_weights', 'market_volatility',
        # Enhancement #2: Position Sizing & Risk Management parameters
        'enable_position_sizing', 'max_position_pct', 'kelly_optimization',
        'volatility_adjustment', 'position_confidence_weighting', 'max_total_exposure'
    }
    
    active_indicators = [
        param.replace('use_', '')
        for param in enhanced_params.keys()
        if param.startswith('use_')
        and enhanced_params[param] is True
        and param not in meta_flags
    ]
    
    # Get confidence weights and adaptive thresholds
    buy_confidence = enhanced_params.get('buy_confidence_weights', pd.Series(1.0, index=data.index))
    sell_confidence = enhanced_params.get('sell_confidence_weights', pd.Series(1.0, index=data.index))
    adaptive_buy_thresh = enhanced_params.get('adaptive_buy_threshold', enhanced_params['buy_score_threshold'])
    adaptive_sell_thresh = enhanced_params.get('adaptive_sell_threshold', enhanced_params['sell_score_threshold'])
    
    # Signal persistence with enhancement
    signal_persistence_days = enhanced_params.get('signal_persistence_days', 2)
    
    # Initialize enhanced persistent scores  
    buy_score_persistent = pd.Series(0.0, index=data.index)
    sell_score_persistent = pd.Series(0.0, index=data.index)
    
    # Process indicators with confidence weighting
    indicators_processed = 0
    
    for indicator in active_indicators:
        if indicator not in data.columns:
            continue
            
        if indicator.startswith('pattern_') or indicator.startswith('dl_signal_'):
            # Boolean indicators with confidence weighting
            buy_param = enhanced_params.get(f'{indicator}_buy')
            sell_param = enhanced_params.get(f'{indicator}_sell')
            
            if buy_param is not None and data[indicator].any():
                raw_signals = (data[indicator] == buy_param).astype(float)
                # Apply confidence weighting
                weighted_signals = raw_signals * buy_confidence
                buy_score_persistent += weighted_signals.rolling(window=signal_persistence_days, min_periods=1).sum()
                indicators_processed += 1
                
            if sell_param is not None and data[indicator].any():
                raw_signals = (data[indicator] == sell_param).astype(float)
                weighted_signals = raw_signals * sell_confidence
                sell_score_persistent += weighted_signals.rolling(window=signal_persistence_days, min_periods=1).sum()
        else:
            # Numerical indicators with confidence weighting
            buy_threshold = enhanced_params.get(f'{indicator}_buy')
            sell_threshold = enhanced_params.get(f'{indicator}_sell')
            
            if buy_threshold is not None:
                raw_signals = (data[indicator] < buy_threshold).astype(float)
                weighted_signals = raw_signals * buy_confidence
                buy_score_persistent += weighted_signals.rolling(window=signal_persistence_days, min_periods=1).sum()
                indicators_processed += 1
                
            if sell_threshold is not None:
                raw_signals = (data[indicator] > sell_threshold).astype(float)
                weighted_signals = raw_signals * sell_confidence
                sell_score_persistent += weighted_signals.rolling(window=signal_persistence_days, min_periods=1).sum()
    
    # Apply adaptive thresholds instead of static ones
    if isinstance(adaptive_buy_thresh, pd.Series):
        buy_signals = buy_score_persistent >= adaptive_buy_thresh
    else:
        buy_signals = buy_score_persistent >= adaptive_buy_thresh
        
    if isinstance(adaptive_sell_thresh, pd.Series):
        sell_signals = sell_score_persistent >= adaptive_sell_thresh
    else:
        sell_signals = sell_score_persistent >= adaptive_sell_thresh
    
    # Apply trend filter (same logic as original)
    if enhanced_params.get('use_trend_filter', False):
        # Use the same trend filter logic as the original function
        adx_threshold = enhanced_params.get('trend_adx_threshold', 20)
        rsi_oversold_threshold = enhanced_params.get('trend_rsi_oversold', 35)
        willr_threshold = enhanced_params.get('trend_willr_threshold', -80)
        stoch_threshold = enhanced_params.get('trend_stoch_threshold', 15)
        sma_period = enhanced_params.get('trend_sma_period', 50)
        
        dynamic_trend_filter = pd.Series(True, index=data.index)
        
        if 'ADX_14' in data.columns:
            adx_trend = data['ADX_14'] > adx_threshold
            dynamic_trend_filter = dynamic_trend_filter & adx_trend
        
        sma_column = f'SMA_{sma_period}'
        if sma_column in data.columns and 'close' in data.columns:
            price_trend = data['close'] > data[sma_column]
            dynamic_trend_filter = dynamic_trend_filter | price_trend
        elif 'SMA_50' in data.columns and 'close' in data.columns:
            price_trend = data['close'] > data['SMA_50']
            dynamic_trend_filter = dynamic_trend_filter | price_trend

        if 'RSI_14' in data.columns:
            oversold_bailout = data['RSI_14'] < rsi_oversold_threshold
            dynamic_trend_filter = dynamic_trend_filter | oversold_bailout
        
        if 'WILLR_14' in data.columns:
            willr_oversold = data['WILLR_14'] < willr_threshold
            dynamic_trend_filter = dynamic_trend_filter | willr_oversold
            
        if 'STOCHk_14_3_3' in data.columns:
            stoch_oversold = data['STOCHk_14_3_3'] < stoch_threshold
            dynamic_trend_filter = dynamic_trend_filter | stoch_oversold
        
        buy_signals = buy_signals & dynamic_trend_filter
    
    # Handle conflicts with enhanced logic
    conflicting_bars = buy_signals & sell_signals
    if conflicting_bars.any():
        # Use confidence-weighted excess for conflict resolution
        buy_excess = (buy_score_persistent * buy_confidence) - (adaptive_buy_thresh if isinstance(adaptive_buy_thresh, (int, float)) else adaptive_buy_thresh.fillna(enhanced_params['buy_score_threshold']))
        sell_excess = (sell_score_persistent * sell_confidence) - (adaptive_sell_thresh if isinstance(adaptive_sell_thresh, (int, float)) else adaptive_sell_thresh.fillna(enhanced_params['sell_score_threshold']))
        
        stronger_buy = conflicting_bars & (buy_excess > sell_excess)
        sell_signals.loc[stronger_buy] = False
        
        stronger_sell = conflicting_bars & (sell_excess >= buy_excess)
        buy_signals.loc[stronger_sell] = False
    
    # Debug output for enhanced strategy
    print(f"📈 ENHANCED STRATEGY: {indicators_processed} indicators, {buy_signals.sum()} buy, {sell_signals.sum()} sell signals")
    if isinstance(adaptive_buy_thresh, pd.Series):
        print(f"   📊 Adaptive thresholds: buy={adaptive_buy_thresh.mean():.1f}±{adaptive_buy_thresh.std():.1f}, sell={adaptive_sell_thresh.mean():.1f}±{adaptive_sell_thresh.std():.1f}")
    else:
        print(f"   📊 Adaptive thresholds: buy={adaptive_buy_thresh}, sell={adaptive_sell_thresh}")
    
    signals = pd.Series(0, index=data.index)
    signals.loc[buy_signals] = 1
    signals.loc[sell_signals] = -1
    
    return signals

def single_trial_optimization(n_trials_worker, data, trade_preference, numerical_indicators, boolean_indicators, enable_position_sizing):
    """Run optimization trials in an isolated worker process"""
    import uuid
    import os
    
    # Create unique study for this worker
    study_name = f"worker_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    worker_study = optuna.create_study(direction='maximize', study_name=study_name)
    
    # Run trials for this worker
    worker_study.optimize(
        lambda trial: objective(trial, data, trade_preference, numerical_indicators, boolean_indicators, enable_position_sizing),
        n_trials=n_trials_worker,
        show_progress_bar=False
    )
    
    return worker_study.trials

def objective(trial, data, trade_preference=0.5, numerical_indicators=None, boolean_indicators=None, enable_position_sizing=True):
    """
    Objective function for Optuna to dynamically build and test hybrid strategies.
    Args:
        trial: Optuna trial object
        data: Market data
        trade_preference: 0.0 = Conservative (fewer trades), 1.0 = Aggressive (more trades)
        numerical_indicators: Pre-determined list of numerical indicators (STATIC)
        boolean_indicators: Pre-determined list of boolean indicators (STATIC)
    """
    # Add explicit debug output for first few trials
    if trial.number <= 3:
        print(f"\n🎯 OBJECTIVE FUNCTION - TRIAL {trial.number}")
        print(f"   Data shape: {data.shape}")
        print(f"   Columns: {len(data.columns)}")
    
    # Use pre-determined STATIC indicator lists to ensure consistent parameter space
    if numerical_indicators is None or boolean_indicators is None:
        raise ValueError("Must provide static indicator lists to maintain consistent parameter space!")

    if trial.number <= 3:
        print(f"   Numerical indicators: {len(numerical_indicators)} (first 3: {numerical_indicators[:3]})")
        print(f"   Boolean indicators: {len(boolean_indicators)} (first 3: {boolean_indicators[:3]})")

    params = {}
    active_indicators = []

    # === PLACEHOLDER THRESHOLDS (Will be calculated properly after we know active indicators) ===
    # Don't suggest anything yet - we'll set these later based on actual active indicator count
    
    # === DYNAMIC ANTI-WHIPSAW PARAMETERS (Based on trade_preference) ===
    # 1. min_hold_days: Conservative = longer holds, Aggressive = shorter holds
    if trade_preference < 0.4:  # Conservative: 3-10 days
        min_hold_min, min_hold_max = 3, 10
    elif trade_preference < 0.7:  # Balanced: 1-7 days  
        min_hold_min, min_hold_max = 1, 7
    else:  # Aggressive: 1-2 days (EXTREMELY quick trades for max frequency)
        min_hold_min, min_hold_max = 1, 2
    params['min_hold_days'] = trial.suggest_int('min_hold_days', min_hold_min, min_hold_max)
    
    # 2. require_confirmation: FIXED choices to maintain consistent parameter space
    # Always use same choices, but suggest weighted parameter based on trade_preference
    confirmation_raw = trial.suggest_categorical('require_confirmation', [True, False])
    
    # Apply trade preference weighting (bias the result)
    if trade_preference < 0.4:  # Conservative: 80% chance of requiring confirmation
        # If raw suggests False, override to True 80% of the time
        confirmation_weight = trial.suggest_float('confirmation_weight', 0.0, 1.0)
        params['require_confirmation'] = confirmation_raw if confirmation_raw else (confirmation_weight > 0.2)
    elif trade_preference < 0.7:  # Balanced: 50/50 - use raw suggestion
        params['require_confirmation'] = confirmation_raw
    else:  # Aggressive: 20% chance of requiring confirmation
        # If raw suggests True, override to False 80% of the time
        confirmation_weight = trial.suggest_float('confirmation_weight', 0.0, 1.0)
        params['require_confirmation'] = confirmation_raw if not confirmation_raw else (confirmation_weight < 0.2)
    
    # === TRUE OPTIMIZATION (APPROACH A) ===
    # Let Optuna search the space of:
    # - Which standard indicators to use (via use_* flags)
    # - Optimal buy/sell thresholds for each indicator
    # - Global buy/sell score thresholds (calculated later)
    # No composite indicators, regime detection, or dynamic sizing here – this
    # restores the earlier, simpler and high-performing behaviour.
    # === DEFINE THEORETICAL RANGES FOR BOUNDED INDICATORS ===
    # This ensures we search the full logical space (e.g. RSI 0-100) rather than just observed data limits
    THEORETICAL_RANGES = {
        'RSI': (2.0, 98.0),
        'STOCH': (2.0, 98.0), 'K_': (2.0, 98.0), 'D_': (2.0, 98.0),
        'MFI': (2.0, 98.0),
        'ADX': (5.0, 60.0),
        'AROON': (5.0, 100.0),
        'UO': (5.0, 95.0),
        'WILLR': (-98.0, -2.0),
        'CCI': (-300.0, 300.0),
        'CMO': (-90.0, 90.0),
        'BOP': (-0.9, 0.9),
        'PGO': (-5.0, 5.0),
        'BIAS': (-10.0, 10.0),
        'ROC': (-20.0, 20.0),  # Soft bounds for unbounded
        'PPO': (-5.0, 5.0),    # Soft bounds
        'TSI': (-100.0, 100.0)
    }

    # Debug the indicator detection logic
    if trial.number <= 2:  # Only for first few trials
        print(f"\n🔧 OBJECTIVE FUNCTION DEBUG - TRIAL {trial.number}")
        print(f"   Numerical indicators count: {len(numerical_indicators)}")
        print(f"   First 5 numerical: {numerical_indicators[:5] if numerical_indicators else 'NONE'}")
        print(f"   Boolean indicators count: {len(boolean_indicators)}")
        print(f"   First 5 boolean: {boolean_indicators[:5] if boolean_indicators else 'NONE'}")

    for indicator in numerical_indicators:
        # === APPROACH A: ALWAYS include all numerical indicators ===
        # Use trial.suggest_categorical with single True value to ensure it's stored in trial.params
        params[f'use_{indicator}'] = trial.suggest_categorical(f'use_{indicator}', [True])
        active_indicators.append(indicator)
        
        # Debug for first trial
        if trial.number <= 1 and len(active_indicators) <= 5:
            print(f"   ✅ Suggesting use_{indicator} = True")
        
        # Determine range: Use theoretical if available, otherwise data-driven
        range_min, range_max = None, None
        
        # Check if known bounded indicator
        for key, (t_min, t_max) in THEORETICAL_RANGES.items():
            if key in indicator.upper():
                range_min, range_max = t_min, t_max
                break
        
        # Fallback to data-driven range with expansion
        if range_min is None:
            data_min, data_max = data[indicator].min(), data[indicator].max()
            range_size = data_max - data_min
            # Suggest indicator-specific buy/sell thresholds within CRASH-AWARE ranges
            if indicator in universal_strategy_thresholds:
                min_val, max_val = universal_strategy_thresholds[indicator]
                params[f'{indicator}_buy'] = trial.suggest_float(f'{indicator}_buy', min_val, max_val)
                params[f'{indicator}_sell'] = trial.suggest_float(f'{indicator}_sell', min_val, max_val)
            else:
                # CRASH-AWARE ranges that can catch market bottoms like April 2025
                if 'RSI' in indicator or 'MFI' in indicator:
                    # RSI: Must be able to catch RSI 21.6 (April bottom), so use reasonable range
                    params[f'{indicator}_buy'] = trial.suggest_float(f'{indicator}_buy', 20, 40)  # Was 10-40
                    params[f'{indicator}_sell'] = trial.suggest_float(f'{indicator}_sell', 60, 80)  # Was 60-90
                elif 'STOCH' in indicator:
                    params[f'{indicator}_buy'] = trial.suggest_float(f'{indicator}_buy', 10, 30)
                    params[f'{indicator}_sell'] = trial.suggest_float(f'{indicator}_sell', 70, 90)
                elif 'WILLR' in indicator:
                    # Williams %R: Must catch -84.5 (April bottom)
                    params[f'{indicator}_buy'] = trial.suggest_float(f'{indicator}_buy', -90, -50)  # Was -90 to -60
                    params[f'{indicator}_sell'] = trial.suggest_float(f'{indicator}_sell', -40, -10)
                elif 'CCI' in indicator:
                    params[f'{indicator}_buy'] = trial.suggest_float(f'{indicator}_buy', -200, -50)  # More oversold range
                    params[f'{indicator}_sell'] = trial.suggest_float(f'{indicator}_sell', 50, 150)
                elif 'MACD' in indicator and not indicator.endswith('s'):
                    # MACD: Must work with bearish MACD during crashes (-16.975 at April bottom)
                    params[f'{indicator}_buy'] = trial.suggest_float(f'{indicator}_buy', -10, 2)  # Was -5 to 2 
                    params[f'{indicator}_sell'] = trial.suggest_float(f'{indicator}_sell', -2, 10)  # Was -2 to 5
                else:
                    # Generic ranges for unknown indicators
                    params[f'{indicator}_buy'] = trial.suggest_float(f'{indicator}_buy', -100, 50)
                    params[f'{indicator}_sell'] = trial.suggest_float(f'{indicator}_sell', -50, 100)  # === DIRECT OPTUNA OPTIMIZATION OF SIGNAL THRESHOLDS ===
    # Let Optuna directly optimize the exact number of indicators needed
    # This removes all percentage-based constraints and artificial minimums
    
    actual_active = len(active_indicators)
    if actual_active == 0:
        if trial.number <= 10:
            print(f"  SKIPPING TRIAL {trial.number}: No valid indicators")
        return -1e9
    
    # CRASH-SENSITIVE thresholds to catch major market bottoms
    # April 2025 analysis: Only 2 indicators (RSI + WillR) were oversold at the bottom
    # Need to ensure buy thresholds can be low enough (1-3) to catch these opportunities
    max_reasonable_threshold = min(8, actual_active)  # Cap at 8 for crash sensitivity
    
    params['buy_score_threshold'] = trial.suggest_int('buy_score_threshold', 1, max_reasonable_threshold)
    params['sell_score_threshold'] = trial.suggest_int('sell_score_threshold', 1, max_reasonable_threshold)
    
    # === APPROACH A: OPTIMIZE ALL TREND FILTER PARAMETERS ===
    # In APPROACH A, trend filter is always enabled (already suggested above)
    # Use unique parameter names to avoid Optuna conflicts
    
    if params.get('use_trend_filter', True):  # Default to True for APPROACH A
        # Optimize ADX threshold with unique name
        params['trend_adx_threshold'] = trial.suggest_float('trend_adx_threshold', 15, 35)
        
        # Optimize RSI oversold threshold with unique name
        params['trend_rsi_oversold'] = trial.suggest_float('trend_rsi_oversold', 20, 45)
        
        # Optimize Williams %R threshold with unique name
        params['trend_willr_threshold'] = trial.suggest_float('trend_willr_threshold', -95, -65)
        
        # Optimize Stochastic threshold with unique name
        params['trend_stoch_threshold'] = trial.suggest_float('trend_stoch_threshold', 8, 25)
        
        # Optimize SMA period with unique name
        params['trend_sma_period'] = trial.suggest_int('trend_sma_period', 20, 100)
    else:
        # Set default values when trend filter is disabled
        params['trend_adx_threshold'] = 20
        params['trend_rsi_oversold'] = 35
        params['trend_willr_threshold'] = -80
        params['trend_stoch_threshold'] = 15
        params['trend_sma_period'] = 50
        
    # === APPROACH A: OPTIMIZE SIGNAL PERSISTENCE ===
    # Instead of static 2-day persistence, let Optuna find optimal window
    params['signal_persistence_days'] = trial.suggest_int('signal_persistence_days', 1, 5)
    
    # === ENHANCEMENT #2: POSITION SIZING & RISK MANAGEMENT ===
    if POSITION_SIZING_AVAILABLE:
        # Use the setting from Streamlit app (user can toggle on/off)
        params['enable_position_sizing'] = enable_position_sizing
        
        if params['enable_position_sizing']:
            # MAXIMUM AGGRESSIVE: Optimize maximum position size (50% to 100% of capital per position)
            params['max_position_pct'] = trial.suggest_float('max_position_pct', 0.50, 1.00)
            
            # Optimize risk management features
            params['kelly_optimization'] = trial.suggest_categorical('kelly_optimization', [True, False])
            params['volatility_adjustment'] = trial.suggest_categorical('volatility_adjustment', [True, False])
            
            # MAXIMUM AGGRESSIVE: Maximum total exposure (70% to 100% of capital)
            params['max_total_exposure'] = trial.suggest_float('max_total_exposure', 0.70, 1.00)
            
            # NEW: Aggressive volatility scaling
            params['aggressive_vol_scaling'] = trial.suggest_categorical('aggressive_vol_scaling', [True, False])
            
            # NEW: Risk multiplier for high-confidence signals
            params['confidence_risk_multiplier'] = trial.suggest_float('confidence_risk_multiplier', 1.0, 2.0)
        else:
            # Default values when position sizing is disabled
            params['max_position_pct'] = 1.00  # Full capital deployment (original behavior)
            params['kelly_optimization'] = False
            params['volatility_adjustment'] = False
            params['max_total_exposure'] = 1.00
    else:
        # Position sizing not available - use defaults
        params['enable_position_sizing'] = False
        params['max_position_pct'] = 1.00
        params['kelly_optimization'] = False
        params['volatility_adjustment'] = False
    
    # === ENHANCEMENT #3: MARKET REGIME DETECTION ===
    if MARKET_REGIME_AVAILABLE:
        params['enable_market_regime'] = trial.suggest_categorical('enable_market_regime', [True, False])
        
        if params['enable_market_regime']:
            # When regime detection is enabled, it will override thresholds dynamically
            print(f"   🎯 Market regime adaptation enabled - thresholds will adapt to market conditions")
        else:
            print(f"   📊 Using static thresholds for all market conditions")
    else:
        params['enable_market_regime'] = False
        params['max_total_exposure'] = 1.00
    
    # Keep it simple - core strategy only
    params['use_weighted_scoring'] = False

    # Debug output for first few trials
    if trial.number <= 3:
        actual_buy_pct = params['buy_score_threshold'] / actual_active * 100
        actual_sell_pct = params['sell_score_threshold'] / actual_active * 100
        print(f"TRIAL {trial.number}: {actual_active} active indicators (ALL INDICATORS APPROACH)")
        print(f"   📊 OPTUNA THRESHOLDS: buy={params['buy_score_threshold']} ({actual_buy_pct:.1f}%), sell={params['sell_score_threshold']} ({actual_sell_pct:.1f}%)")
        print(f"   📊 Threshold range: 1-{max_reasonable_threshold} (capped at 25 for all 85 indicators)")
        print(f"   📊 SIGNAL PERSISTENCE: {params.get('signal_persistence_days', 2)} days")
        if params.get('use_trend_filter', False):
            print(f"   📊 TREND FILTER: RSI<{params.get('trend_rsi_oversold', 35):.1f}, ADX>{params.get('trend_adx_threshold', 20):.1f}, WILLR<{params.get('trend_willr_threshold', -80):.1f}")
        else:
            print(f"   📊 TREND FILTER: Disabled")
    
    try:
        strategy_name = ' + '.join(active_indicators)
        
        # Test the strategy function first - use original strategy for consistency
        print(f"🔍 TRIAL {trial.number}: Using original strategy for consistent results")
        signals = universal_strategy(data, params)
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        
        if buy_signals + sell_signals == 0:
            print(f"DEBUG: No trades generated for trial {trial.number}. Signals: {signals}")
            return -1e9
            
        # Enhancement #2: Use enhanced backtester with position sizing if available and enabled
        if POSITION_SIZING_AVAILABLE and params.get('enable_position_sizing', False):
            backtester = EnhancedBacktester(
                data=data,
                strategy_name=strategy_name,
                strategy_func=universal_strategy,
                params=params,
                starting_capital=100000,
                enable_position_sizing=True,
                max_position_pct=params.get('max_position_pct', 0.20)
            )
            
            if trial.number <= 2:
                print(f"   🚀 Using Enhanced Backtester with {params.get('max_position_pct', 0.20):.1%} max position size")
                print(f"   📊 Aggressive settings: exposure={params.get('max_total_exposure', 0.6):.1%}, "
                      f"vol_scaling={params.get('aggressive_vol_scaling', False)}, "
                      f"risk_mult={params.get('confidence_risk_multiplier', 1.0):.1f}x")
        else:
            backtester = Backtester(data, strategy_name, universal_strategy, params)
            if trial.number <= 2:
                print(f"   📊 Using Standard Backtester (100% position sizing)")
        
        backtester.run()
        _, summary = backtester.get_results()
        
        # Heavily penalize strategies that don't trade
        if summary['total_trades'] == 0:
            if trial.number <= 3:
                print(f"  TRIAL {trial.number}: No trades generated (buy signals: {buy_signals}, sell signals: {sell_signals})")
                # Debug: Show actual signal distribution
                print(f"    Signal values: {signals.value_counts().to_dict()}")
                print(f"    First 10 signals: {signals.head(10).tolist()}")
            return -1e9
        
        # ALWAYS compute objective value directly from capital to guarantee consistency
        start_cap = summary.get('starting_capital', 100000)
        end_cap = summary['ending_capital']
        objective_return = (end_cap - start_cap) / start_cap * 100

        # Keep summary in sync in case it is used elsewhere
        summary['total_return_pct'] = round(objective_return, 2)
        
        # === QUALITY ADJUSTMENTS ===
        # To find "best trading metrics" (high win rate, low drawdown), we adjust the
        # objective value passed to Optuna. This acts as a "soft" constraint/guide.
        # The actual return is preserved in the summary, but Optuna sees a penalized score
        # if the strategy is "unhealthy" (too few trades, too much risk).
        
        score = objective_return
        
        # 1. Penalty for low trade count (prevents lucky 1-trade outliers)
        # We want at least ~10 trades to consider it statistically relevant
        if summary['total_trades'] < 10:
            # Penalty grows as trades decrease. 1 trade = -9% penalty.
            trade_penalty = (10 - summary['total_trades']) * 1.0
            score -= trade_penalty
            
        # 2. Penalty for high drawdown (risk aversion)
        # We tolerate up to 20% drawdown, then penalize
        if summary['max_drawdown_pct'] > 20.0:
            # Penalty: 0.5% for every 1% drawdown above 20%
            drawdown_excess = summary['max_drawdown_pct'] - 20.0
            dd_penalty = drawdown_excess * 0.5
            score -= dd_penalty
            
        # 3. Bonus for high Win Rate (encouragement)
        # Only applied if we have enough trades to trust the win rate
        if summary['total_trades'] >= 10 and summary['win_rate'] > 60.0:
            # Bonus: 0.1% for every 1% win rate above 60%
            wr_bonus = (summary['win_rate'] - 60.0) * 0.1
            score += wr_bonus

        # Optional debug for first few trials
        if trial.number <= 3:
            print(f"  OBJECTIVE SCORING: Return={objective_return:.2f}%, Trades={summary['total_trades']}, DD={summary['max_drawdown_pct']:.2f}% -> Score={score:.2f}")

        return score
    except Exception as e:
        return -1e9

# Removed old batch function - now using joblib-based individual trial optimization

def run_optimization(data, n_trials=1000, n_jobs=None, progress_callback=None, trade_preference=0.5, enable_position_sizing=True, ticker="SPY"):
    """
    Runs the new dynamic optimization and returns top 3 strategies.
    
    Args:
        data: Market data for optimization
        n_trials: Number of optimization trials
        n_jobs: Number of parallel jobs (None = auto-detect cores, 1 = single-threaded)
        enable_position_sizing: Whether to use enhanced position sizing (True) or standard 100% capital (False)
    """
    import multiprocessing as mp
    
    # Auto-detect CPU cores if not specified
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)  # Leave 1 core free
    
    print(f"\n{'='*40}")
    print("🚀 STANDARD OPTIMIZATION STARTING")
    print(f"Data shape: {data.shape}")
    print(f"Columns count: {len(data.columns)}")
    print(f"First 10 columns: {list(data.columns)[:10]}")
    print(f"Trials: {n_trials}")
    print(f"Workers: {n_jobs} ({'SINGLE-THREADED' if n_jobs == 1 else 'MULTI-THREADED'})")
    print(f"Method: STANDARD (debugging enabled)")
    print(f"Trade Preference: {trade_preference:.2f} ({'Conservative' if trade_preference < 0.4 else 'Aggressive' if trade_preference > 0.6 else 'Balanced'})")
    
    # Reset universal_strategy call counter for this run
    if hasattr(universal_strategy, 'call_count'):
        delattr(universal_strategy, 'call_count')
    
    print(f"{'='*40}\n")
    
    # PRE-DETERMINE STATIC INDICATOR LISTS (crucial for consistent parameter space)
    print("📋 Pre-determining indicator lists for consistent parameter space...")
    
    # More inclusive numerical indicator detection
    # Include all numeric-like columns that aren't OHLCV or metadata
    exclude_cols = {'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close'}
    
    numerical_indicators = []
    boolean_indicators = []
    
    for col in data.columns:
        if col.lower() in exclude_cols:
            continue
        elif data[col].dtype == 'bool':
            boolean_indicators.append(col)
        else:
            # Include all other columns as numerical indicators
            # This includes float64, int64, object (that contain numbers), etc.
            numerical_indicators.append(col)
    
    print(f"   📊 Numerical indicators: {len(numerical_indicators)} (first 5: {numerical_indicators[:5]})")
    print(f"   🔘 Boolean indicators: {len(boolean_indicators)} (first 5: {boolean_indicators[:5]})")
    
    # BRING BACK JOBLIB PARALLELIZATION with proper isolation
    if n_jobs == 1:
        # Single-threaded execution
        print("⚙️  SINGLE-THREADED EXECUTION")
        import uuid
        study_name = f"main_study_{uuid.uuid4().hex[:8]}"
        study = optuna.create_study(direction='maximize', study_name=study_name)
        study.optimize(lambda trial: objective(trial, data, trade_preference, numerical_indicators, boolean_indicators, enable_position_sizing), 
                      n_trials=n_trials, show_progress_bar=True)
        all_trials = study.trials
    else:
        # Multi-threaded execution with joblib (RESTORED!)
        print(f"⚙️  MULTI-THREADED EXECUTION with {n_jobs} workers")
        print("   Using joblib with isolated studies per trial")
        
        # Create trials distributed across workers
        trials_per_worker = max(1, n_trials // n_jobs)
        worker_trials = [trials_per_worker] * (n_jobs - 1)
        worker_trials.append(n_trials - sum(worker_trials))  # Remaining trials for last worker
        
        print(f"   Trial distribution: {worker_trials}")
        
        # Run parallel optimization with isolated studies
        from joblib import Parallel, delayed
        all_worker_trials = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(single_trial_optimization)(worker_trials[i], data, trade_preference, numerical_indicators, boolean_indicators, enable_position_sizing) 
            for i in range(n_jobs)
        )
        
        # Flatten results from all workers
        all_trials = []
        for worker_trials_result in all_worker_trials:
            all_trials.extend(worker_trials_result)
    
    # Get top 3 unique trials (best performing strategies)
    # Filter out failed trials and sort by performance  
    successful_trials = [t for t in all_trials if t.value is not None and t.value > -1e9]
    top_trials = sorted(successful_trials, key=lambda t: t.value, reverse=True)[:3]
    
    print(f"\n{'='*40}")
    print("TOP 3 STRATEGIES FOUND:")
    
    # Debug: Show what we found
    print(f"DEBUG: Total trials run: {len(all_trials)}")
    print(f"DEBUG: Successful trials: {len(successful_trials)}")
    if len(all_trials) > 0:
        trial_values = [t.value for t in all_trials if t.value is not None]
        if trial_values:
            print(f"DEBUG: Best trial value: {max(trial_values):.2f}")
            print(f"DEBUG: Worst trial value: {min(trial_values):.2f}")
            print(f"DEBUG: Trials with -1e9: {sum(1 for v in trial_values if v <= -1e9)}")
        else:
            print("DEBUG: No trial values found!")
    
    for i, trial in enumerate(top_trials, 1):
        active_indicators = [param.replace('use_', '') for param in trial.params.keys() 
                           if param.startswith('use_') and trial.params[param] == True]
        strategy_name = ' + '.join(active_indicators[:3]) + ('...' if len(active_indicators) > 3 else '')
        buy_thresh = trial.params.get('buy_score_threshold', 1)
        sell_thresh = trial.params.get('sell_score_threshold', 1)
        print(f"  {i}. Trial {trial.number}: {strategy_name}: {trial.value:.2f}% (Thresholds: {buy_thresh}/{sell_thresh})")
    print(f"{'='*40}\n")
    
    # Check if we have enough unique strategies
    if len(successful_trials) < 3:
        print(f"⚠️  Only found {len(successful_trials)} successful strategies out of {n_trials} trials")
        if len(successful_trials) == 0:
            print("🔍 All trials returned -1e9 penalty. Possible issues:")
            print("   - All indicator threshold combinations invalid")
            print("   - All strategies generate no trades")
            print("   - Backtesting errors in all trials")
    else:
        print(f"✅ Found {len(successful_trials)} successful strategies out of {n_trials} trials")

    # FINAL SANITY CHECK: recompute returns for the top trials using a fresh backtest
    # This guarantees that trial.value (used in the UI header) matches actual capital-based returns
    sanitized_top_trials = []
    for t in top_trials:
        params = t.params.copy()
        strategy_name = ' + '.join([
            p.replace('use_', '') for p in params.keys()
            if p.startswith('use_') and params[p] is True
        ])

        # Enhancement #2: Use enhanced backtester for final sanity check if position sizing was enabled
        # Use the same setting from the Streamlit app to ensure consistency
        if POSITION_SIZING_AVAILABLE and enable_position_sizing:
            max_pos = params.get('max_position_pct', 0.20)
            print(f"🚀 FINAL SANITY CHECK: Using EnhancedBacktester with {max_pos:.1%} max position")
            bt = EnhancedBacktester(
                data=data,
                strategy_name=strategy_name or "Top Trial",
                strategy_func=universal_strategy,
                params=params,
                starting_capital=100000,
                enable_position_sizing=True,
                max_position_pct=max_pos
            )
        else:
            print(f"📊 FINAL SANITY CHECK: Using Standard Backtester (100% capital)")
            bt = Backtester(data, strategy_name or "Top Trial", universal_strategy, params)
        
        bt.run()
        trade_log, summary = bt.get_results()

        start_cap = summary.get('starting_capital', 100000)
        end_cap = summary['ending_capital']
        true_return = (end_cap - start_cap) / start_cap * 100
        
        # Generate comprehensive CSV log for analysis
        if DETAILED_LOGGING_AVAILABLE and len(sanitized_top_trials) < 3:  # Only log top 3 strategies
            try:
                # Generate signals and scores for logging
                signals_for_log = universal_strategy(data, params)
                
                # Create buy/sell score series (simplified for logging)
                buy_scores_log = pd.Series(0, index=data.index)
                sell_scores_log = pd.Series(0, index=data.index)
                
                # Use the ticker parameter passed to run_optimization
                
                # Create comprehensive log
                log_filepath = create_comprehensive_strategy_log(
                    ticker=ticker,
                    data=data,
                    strategy_params=params,
                    signals=signals_for_log,
                    buy_scores_series=buy_scores_log,
                    sell_scores_series=sell_scores_log,
                    trades_df=trade_log,
                    trial_number=getattr(t, 'number', len(sanitized_top_trials) + 1),
                    return_pct=true_return
                )
                
                print(f"📊 Created detailed log for strategy #{len(sanitized_top_trials) + 1}")
                
            except Exception as e:
                print(f"⚠️  Failed to create detailed log: {e}")
                # Continue without failing the optimization

        # Debug: Track return calculation discrepancy
        original_return = getattr(t, 'value', 0)
        if abs(true_return - original_return) > 1.0:  # More than 1% difference
            print(f"🔍 RETURN DISCREPANCY DETECTED:")
            print(f"   Original optimization return: {original_return:.2f}%")
            print(f"   Final sanity check return: {true_return:.2f}%")
            print(f"   Difference: {true_return - original_return:.2f}%")
            print(f"   Strategy: {strategy_name[:50]}...")

        # Overwrite trial.value so the UI header uses the true return
        try:
            t.value = true_return
        except Exception:
            # If assignment fails, we still append; UI will use existing value
            pass

        sanitized_top_trials.append(t)

    return sanitized_top_trials
