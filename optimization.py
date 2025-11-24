import optuna
import pandas as pd
import numpy as np
import itertools
from backtester import Backtester
import joblib
from joblib import Parallel, delayed

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
    }

    active_indicators = [
        param.replace('use_', '')
        for param in params.keys()
        if param.startswith('use_')
        and params[param] is True
        and param not in meta_flags
    ]
    
    # Debug output for first few calls only - ALWAYS print for debugging
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

    for indicator in active_indicators:
        if indicator not in data.columns:
            print(f"MISSING COLUMN: {indicator}")
            continue

        if indicator.startswith('pattern_') or indicator.startswith('dl_signal_'):
            # Boolean indicators: match on True/False flags
            buy_param = params.get(f'{indicator}_buy')
            sell_param = params.get(f'{indicator}_sell')

            if buy_param is not None and data[indicator].any():
                buy_score[data[indicator] == buy_param] += 1
                indicators_processed += 1
            if sell_param is not None and data[indicator].any():
                sell_score[data[indicator] == sell_param] += 1
        else:
            # Numerical indicators: compare against thresholds
            buy_threshold = params.get(f'{indicator}_buy')
            sell_threshold = params.get(f'{indicator}_sell')

            if buy_threshold is not None:
                buy_score[data[indicator] < buy_threshold] += 1
                indicators_processed += 1
            if sell_threshold is not None:
                sell_score[data[indicator] > sell_threshold] += 1

    buy_signals = buy_score >= params['buy_score_threshold']
    sell_signals = sell_score >= params['sell_score_threshold']

    # === APPLY TREND FILTER ===
    # If enabled, we only allow BUY signals when the trend filter is positive.
    # This acts as a safety net, allowing us to use more sensitive indicator thresholds
    # (e.g., buying dips) without catching falling knives in a crash.
    if params.get('use_trend_filter', False):
        if 'trend_filter' in data.columns:
            # Only allow buys when trend_filter is True
            buy_signals = buy_signals & data['trend_filter']
        else:
            # Should not happen given indicators.py, but good for safety
            pass

    # CRITICAL FIX: Make signals mutually exclusive to prevent backtester conflicts
    # When both buy and sell trigger, choose the stronger signal DETERMINISTICALLY
    conflicting_bars = buy_signals & sell_signals
    if conflicting_bars.any():
        # On conflicting bars, choose based on absolute score difference (deterministic)
        buy_excess = buy_score - params['buy_score_threshold']
        sell_excess = sell_score - params['sell_score_threshold']
        
        # Where buy excess is greater, keep buy signal and remove sell
        stronger_buy = conflicting_bars & (buy_excess > sell_excess)
        sell_signals.loc[stronger_buy] = False
        
        # Where sell excess is greater or equal, keep sell signal and remove buy  
        stronger_sell = conflicting_bars & (sell_excess >= buy_excess)
        buy_signals.loc[stronger_sell] = False

    # Optional debug output
    if debug_mode and universal_strategy.call_count <= 3:
        print(f"  PROCESSED: {indicators_processed} indicators")
        print(f"  FINAL SCORES: max_buy={buy_score.max()}, max_sell={sell_score.max()}")
        print(f"  SIGNALS: {buy_signals.sum()} buy, {sell_signals.sum()} sell")
        if conflicting_bars.any():
            print(f"  CONFLICTS RESOLVED: {conflicting_bars.sum()} conflicting bars fixed")

    signals = pd.Series(0, index=data.index)
    signals.loc[buy_signals] = 1
    signals.loc[sell_signals] = -1
    
    return signals

def single_trial_optimization(n_trials_worker, data, trade_preference, numerical_indicators, boolean_indicators):
    """Run optimization trials in an isolated worker process"""
    import uuid
    import os
    
    # Create unique study for this worker
    study_name = f"worker_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    worker_study = optuna.create_study(direction='maximize', study_name=study_name)
    
    # Run trials for this worker
    worker_study.optimize(
        lambda trial: objective(trial, data, trade_preference, numerical_indicators, boolean_indicators),
        n_trials=n_trials_worker,
        show_progress_bar=False
    )
    
    return worker_study.trials

def objective(trial, data, trade_preference=0.5, numerical_indicators=None, boolean_indicators=None):
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
    
    # 3. use_trend_filter: FIXED choices to maintain consistent parameter space
    # Always use same choices, but suggest weighted parameter based on trade_preference
    trend_filter_raw = trial.suggest_categorical('use_trend_filter', [True, False])
    
    # Apply trade preference weighting (bias the result)
    if trade_preference < 0.4:  # Conservative: 80% chance of using trend filter
        # If raw suggests False, override to True 80% of the time
        trend_weight = trial.suggest_float('trend_filter_weight', 0.0, 1.0)
        params['use_trend_filter'] = trend_filter_raw if trend_filter_raw else (trend_weight > 0.2)
    elif trade_preference < 0.7:  # Balanced: 50/50 - use raw suggestion
        params['use_trend_filter'] = trend_filter_raw
    else:  # Aggressive: 20% chance of using trend filter  
        # If raw suggests True, override to False 80% of the time
        trend_weight = trial.suggest_float('trend_filter_weight', 0.0, 1.0)
        params['use_trend_filter'] = trend_filter_raw if not trend_filter_raw else (trend_weight < 0.2)
    
    # 4. adx_threshold: Conservative = higher threshold, Aggressive = lower threshold
    if trade_preference < 0.4:  # Conservative: 22-30 (strong trends only)
        adx_min, adx_max = 22, 30
    elif trade_preference < 0.7:  # Balanced: 18-26
        adx_min, adx_max = 18, 26
    else:  # Aggressive: 10-18 (VERY weak trends allowed for max trades)
        adx_min, adx_max = 10, 18
    params['adx_threshold'] = trial.suggest_int('adx_threshold', adx_min, adx_max)
    
    # Debug output for dynamic parameters (moved to after threshold recalculation)
    
    # === TRUE OPTIMIZATION (BASELINE) ===
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

    for indicator in numerical_indicators:
        # Always suggest whether to use this indicator
        use_indicator = trial.suggest_categorical(f'use_{indicator}', [True, False])
        
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
            if range_size == 0: range_size = 1.0 # Prevent zero range
            # Expand by 50% to catch extremes
            range_min = data_min - (range_size * 0.5)
            range_max = data_max + (range_size * 0.5)

        # Suggest thresholds using the determined wide range
        buy_threshold = trial.suggest_float(f'{indicator}_buy', range_min, range_max)
        sell_threshold = trial.suggest_float(f'{indicator}_sell', range_min, range_max)
        
        # Ensure buy < sell for logical consistency
        if buy_threshold >= sell_threshold:
             # Swap and force a small gap
            range_span = range_max - range_min
            buy_threshold, sell_threshold = sell_threshold, buy_threshold
            if buy_threshold == sell_threshold:
                 buy_threshold -= range_span * 0.01

        if use_indicator:
            params[f'use_{indicator}'] = True
            active_indicators.append(indicator)
        else:
            params[f'use_{indicator}'] = False
        
        params[f'{indicator}_buy'] = buy_threshold
        params[f'{indicator}_sell'] = sell_threshold

    # Decide which boolean indicators to use (ALWAYS suggest ALL parameters for consistent parameter space)
    for indicator in boolean_indicators:
        use_indicator = trial.suggest_categorical(f'use_{indicator}', [True, False])
        # ALWAYS suggest buy/sell parameters to maintain consistent parameter space
        buy_param = trial.suggest_categorical(f'{indicator}_buy', [True, False])
        sell_param = trial.suggest_categorical(f'{indicator}_sell', [True, False])
        
        if use_indicator:
            params[f'use_{indicator}'] = True
            active_indicators.append(indicator)
            params[f'{indicator}_buy'] = buy_param
            params[f'{indicator}_sell'] = sell_param
        else:
            # Still set parameters even when not used to maintain consistent parameter space
            params[f'use_{indicator}'] = False
            params[f'{indicator}_buy'] = buy_param  # Dummy value
            params[f'{indicator}_sell'] = sell_param  # Dummy value

    # === RECALCULATE THRESHOLDS BASED ON ACTUAL ACTIVE INDICATORS ===
    actual_active = len(active_indicators)
    if actual_active == 0:
        if trial.number <= 10:
            print(f"  SKIPPING TRIAL {trial.number}: No valid indicators")
        return -1e9
    
    # Apply trade preference to calculate percentage-based thresholds
    if trade_preference < 0.4:  # Conservative: 40-70% agreement needed
        buy_pct_min, buy_pct_max = 0.4, 0.7
        sell_pct_min, sell_pct_max = 0.4, 0.7
    elif trade_preference < 0.7:  # Balanced: 25-50% agreement needed
        buy_pct_min, buy_pct_max = 0.25, 0.5
        sell_pct_min, sell_pct_max = 0.25, 0.5
    else:  # Aggressive: 1-8% agreement needed (EXTREME trading)
        buy_pct_min, buy_pct_max = 0.01, 0.08
        sell_pct_min, sell_pct_max = 0.01, 0.08
    
    # Calculate proper thresholds based on actual active indicators  
    # REMOVE CAPS for maximum performance - let Optuna find the best thresholds
    proper_buy_min = max(1, int(actual_active * buy_pct_min))
    proper_buy_max = max(2, int(actual_active * buy_pct_max))  # No artificial cap
    proper_sell_min = max(1, int(actual_active * sell_pct_min))  
    proper_sell_max = max(2, int(actual_active * sell_pct_max))  # No artificial cap
    
    # NOW suggest the thresholds with proper ranges (Optuna will use these correctly)
    params['buy_score_threshold'] = trial.suggest_int('buy_score_threshold', proper_buy_min, proper_buy_max)
    params['sell_score_threshold'] = trial.suggest_int('sell_score_threshold', proper_sell_min, proper_sell_max)
    
    # Baseline scoring: no weighted scoring or meta-weights – each indicator
    # contributes equally to the buy/sell score.
    params['use_weighted_scoring'] = False

    # Debug output for first few trials
    if trial.number <= 3:
        actual_buy_pct = params['buy_score_threshold'] / actual_active * 100
        actual_sell_pct = params['sell_score_threshold'] / actual_active * 100
        print(f"TRIAL {trial.number}: {actual_active} active indicators: {active_indicators[:3]}")
        print(f"   📊 CORRECTED THRESHOLDS: buy={params['buy_score_threshold']} ({actual_buy_pct:.1f}%), sell={params['sell_score_threshold']} ({actual_sell_pct:.1f}%)")
        print(f"   📊 Proper ranges: buy={proper_buy_min}-{proper_buy_max}, sell={proper_sell_min}-{proper_sell_max}")
    
    try:
        strategy_name = ' + '.join(active_indicators)
        
        # Test the strategy function first
        signals = universal_strategy(data, params)
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        
        if buy_signals == 0 and sell_signals == 0:
            print(f"DEBUG: No trades generated for trial {trial.number}. Signals: {signals}")
            return -1e9
            
        backtester = Backtester(data, strategy_name, universal_strategy, params)
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

def run_optimization(data, n_trials=1000, n_jobs=None, progress_callback=None, trade_preference=0.5):
    """
    Runs the new dynamic optimization and returns top 3 strategies.
    
    Args:
        data: Market data for optimization
        n_trials: Number of optimization trials
        n_jobs: Number of parallel jobs (None = auto-detect cores, 1 = single-threaded)
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
    numerical_indicators = [col for col in data.columns if data[col].dtype != 'bool' and col not in ['open', 'high', 'low', 'close', 'volume']]
    boolean_indicators = [col for col in data.columns if data[col].dtype == 'bool']
    print(f"   📊 Numerical indicators: {len(numerical_indicators)}")
    print(f"   🔘 Boolean indicators: {len(boolean_indicators)}")
    
    # BRING BACK JOBLIB PARALLELIZATION with proper isolation
    if n_jobs == 1:
        # Single-threaded execution
        print("⚙️  SINGLE-THREADED EXECUTION")
        import uuid
        study_name = f"main_study_{uuid.uuid4().hex[:8]}"
        study = optuna.create_study(direction='maximize', study_name=study_name)
        study.optimize(lambda trial: objective(trial, data, trade_preference, numerical_indicators, boolean_indicators), 
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
            delayed(single_trial_optimization)(worker_trials[i], data, trade_preference, numerical_indicators, boolean_indicators) 
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

        bt = Backtester(data, strategy_name or "Top Trial", universal_strategy, params)
        bt.run()
        _, summary = bt.get_results()

        start_cap = summary.get('starting_capital', 100000)
        end_cap = summary['ending_capital']
        true_return = (end_cap - start_cap) / start_cap * 100

        # Overwrite trial.value so the UI header uses the true return
        try:
            t.value = true_return
        except Exception:
            # If assignment fails, we still append; UI will use existing value
            pass

        sanitized_top_trials.append(t)

    return sanitized_top_trials
