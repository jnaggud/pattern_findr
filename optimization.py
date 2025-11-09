import optuna
import pandas as pd
import numpy as np
import itertools
from backtester import Backtester

def universal_strategy(data, params):
    """
    A universal strategy that uses a scoring system to combine signals.
    """
    buy_score = pd.Series(0, index=data.index)
    sell_score = pd.Series(0, index=data.index)

    # Extract active indicators from the flat params structure
    active_indicators = [param.replace('use_', '') for param in params.keys() 
                        if param.startswith('use_') and params[param] == True]
    
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
    
    # Process all indicators (simplified to focus on the core issue)
    indicators_processed = 0
    for indicator in active_indicators:
        if indicator not in data.columns:
            print(f"MISSING COLUMN: {indicator}")
            continue
            
        if indicator.startswith('pattern_') or indicator.startswith('dl_signal_'):
            # Boolean indicators
            true_count = data[indicator].sum() if hasattr(data[indicator], 'sum') else 0
            buy_param = params.get(f'{indicator}_buy')
            sell_param = params.get(f'{indicator}_sell')
            
            if buy_param:
                buy_score[data[indicator]] += 1
                indicators_processed += 1
            if sell_param:
                sell_score[data[indicator]] += 1
        else:
            # Numerical indicators
            buy_threshold = params.get(f'{indicator}_buy')
            sell_threshold = params.get(f'{indicator}_sell')
            
            if buy_threshold is not None:
                buy_score[data[indicator] < buy_threshold] += 1
                indicators_processed += 1
            if sell_threshold is not None:
                sell_score[data[indicator] > sell_threshold] += 1

    buy_signals = buy_score >= params['buy_score_threshold']
    sell_signals = sell_score >= params['sell_score_threshold']

    # Optional debug output
    if debug_mode and universal_strategy.call_count <= 3:
        print(f"  PROCESSED: {indicators_processed} indicators")
        print(f"  FINAL SCORES: max_buy={buy_score.max()}, max_sell={sell_score.max()}")
        print(f"  SIGNALS: {buy_signals.sum()} buy, {sell_signals.sum()} sell")

    signals = pd.Series(0, index=data.index)
    signals.loc[buy_signals] = 1
    signals.loc[sell_signals] = -1  # Allow sell signals regardless of buy signals
    
    return signals

def objective(trial, data):
    """
    Objective function for Optuna to dynamically build and test hybrid strategies.
    """
    # Add explicit debug output for first few trials
    if trial.number <= 3:
        print(f"\n🎯 OBJECTIVE FUNCTION - TRIAL {trial.number}")
        print(f"   Data shape: {data.shape}")
        print(f"   Columns: {len(data.columns)}")
    
    numerical_indicators = [col for col in data.columns if data[col].dtype != 'bool' and col not in ['open', 'high', 'low', 'close', 'volume']]
    boolean_indicators = [col for col in data.columns if data[col].dtype == 'bool']

    if trial.number <= 3:
        print(f"   Numerical indicators: {len(numerical_indicators)} (first 3: {numerical_indicators[:3]})")
        print(f"   Boolean indicators: {len(boolean_indicators)} (first 3: {boolean_indicators[:3]})")

    params = {}
    active_indicators = []

    # Add flexible scoring thresholds to expand search space
    params['buy_score_threshold'] = trial.suggest_int('buy_score_threshold', 1, 5)
    params['sell_score_threshold'] = trial.suggest_int('sell_score_threshold', 1, 5)
    
    # Decide which numerical indicators to use and define their params
    for indicator in numerical_indicators:
        use_indicator = trial.suggest_categorical(f'use_{indicator}', [True, False])
        if use_indicator:
            # For most indicators: buy when low (oversold), sell when high (overbought)
            min_val, max_val = data[indicator].min(), data[indicator].max()
            
            # Generate valid thresholds with guaranteed separation
            range_size = max_val - min_val
            
            # Add small buffer to ensure no overlap (1% of range)
            buffer = range_size * 0.01
            midpoint = min_val + range_size * 0.5
            
            # Buy threshold: lower half minus buffer
            # Sell threshold: upper half plus buffer  
            buy_max = midpoint - buffer
            sell_min = midpoint + buffer
            
            # Ensure ranges are valid
            if buy_max <= min_val or sell_min >= max_val or range_size <= 0:
                # Range too small, skip this indicator
                if trial.number <= 3:
                    print(f"  SKIPPING {indicator}: Invalid range (min={min_val:.2f}, max={max_val:.2f}, range={range_size:.2f})")
                continue
                
            buy_threshold = trial.suggest_float(f'{indicator}_buy', min_val, buy_max)
            sell_threshold = trial.suggest_float(f'{indicator}_sell', sell_min, max_val)

            # Double-check and skip if still invalid (should be rare now)
            if buy_threshold >= sell_threshold:
                if trial.number <= 3:
                    print(f"  SKIPPING {indicator}: buy_threshold {buy_threshold:.2f} >= sell_threshold {sell_threshold:.2f}")
                continue

            # Only add valid indicators to params
            params[f'use_{indicator}'] = True
            active_indicators.append(indicator)
            params[f'{indicator}_buy'] = buy_threshold
            params[f'{indicator}_sell'] = sell_threshold

    # Decide which boolean indicators to use
    for indicator in boolean_indicators:
        use_indicator = trial.suggest_categorical(f'use_{indicator}', [True, False])
        if use_indicator:
            params[f'use_{indicator}'] = True
            active_indicators.append(indicator)
            params[f'{indicator}_buy'] = trial.suggest_categorical(f'{indicator}_buy', [True, False])
            params[f'{indicator}_sell'] = trial.suggest_categorical(f'{indicator}_sell', [True, False])

    # Skip strategies with too few indicators (temporarily lowered for debugging)
    if len(active_indicators) < 1:
        if trial.number <= 10:
            print(f"  SKIPPING TRIAL {trial.number}: Only {len(active_indicators)} valid indicators")
        return -1e9

    # Debug output for first few trials
    if trial.number <= 3:
        print(f"TRIAL {trial.number}: {len(active_indicators)} valid indicators: {active_indicators[:3]}")
    
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
            
        return summary['total_return_pct']
    except Exception as e:
        return -1e9

def run_optimization_batch(args):
    """Run a batch of trials in a separate process"""
    batch_trials, data_copy = args
    local_study = optuna.create_study(direction='maximize')
    local_study.optimize(
        lambda trial: objective(trial, data_copy), 
        n_trials=batch_trials, 
        show_progress_bar=False
    )
    return local_study.trials

def run_optimization(data, n_trials=1000, n_jobs=None):
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
    
    # Reset universal_strategy call counter for this run
    if hasattr(universal_strategy, 'call_count'):
        delattr(universal_strategy, 'call_count')
    
    print(f"{'='*40}\n")
    
    # Create study with parallel execution
    study = optuna.create_study(direction='maximize')
    
    if n_jobs == 1:
        # Single-threaded execution
        study.optimize(lambda trial: objective(trial, data), n_trials=n_trials, show_progress_bar=True)
    else:
        # Use a different approach for better parallelization
        print(f"🚀 Using multiprocessing with {n_jobs} parallel workers")
        
        # Split trials among workers more efficiently
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm
        
        # Split trials into batches for each worker
        trials_per_worker = n_trials // n_jobs
        remainder = n_trials % n_jobs
        
        batch_sizes = [trials_per_worker] * n_jobs
        for i in range(remainder):
            batch_sizes[i] += 1
        
        print(f"Running {n_trials} trials across {n_jobs} workers: {batch_sizes}")
        
        # Execute batches in parallel
        all_trials = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(run_optimization_batch, (batch_size, data)) for batch_size in batch_sizes]
            
            with tqdm(total=n_trials, desc="Parallel Trials") as pbar:
                for future in as_completed(futures):
                    batch_trials = future.result()
                    all_trials.extend(batch_trials)
                    pbar.update(len(batch_trials))
        
        # Add all trials to the main study
        for trial in all_trials:
            study.add_trial(trial)
    
    # Get top 3 unique trials (best performing strategies)
    # Filter out failed trials and sort by performance
    successful_trials = [t for t in study.trials if t.value is not None and t.value > -1e9]
    top_trials = sorted(successful_trials, key=lambda t: t.value, reverse=True)[:3]
    
    print(f"\n{'='*40}")
    print("TOP 3 STRATEGIES FOUND:")
    
    # Debug: Show what we found
    print(f"DEBUG: Total trials run: {len(study.trials)}")
    print(f"DEBUG: Successful trials: {len(successful_trials)}")
    if len(study.trials) > 0:
        trial_values = [t.value for t in study.trials if t.value is not None]
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
    
    return top_trials
