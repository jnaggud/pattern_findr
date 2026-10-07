import optuna
import pandas as pd
import numpy as np
import itertools
from backtester import Backtester
import warnings
import logging

# Suppress common Streamlit warnings that clutter console during optimization
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*streamlit.*")
warnings.filterwarnings("ignore", message=".*session state.*")
warnings.filterwarnings("ignore", message=".*MemoryCacheStorageManager.*")
warnings.filterwarnings("ignore", message=".*No runtime found.*")

# Suppress verbose logging from external libraries during optimization
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.ERROR)

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
    
    # === ANTI-WHIPSAW FILTERS ===
    # Prevent rapid entry/exit cycles
    
    # 1. Minimum holding period filter (configurable via params)
    min_hold_days = params.get('min_hold_days', 3)  # Default 3 days minimum hold
    
    filtered_signals = signals.copy()
    last_entry_idx = None
    
    for i in range(len(signals)):
        if signals.iloc[i] == 1:  # Buy signal
            last_entry_idx = i
        elif signals.iloc[i] == -1 and last_entry_idx is not None:  # Sell signal
            # Check if minimum holding period met
            days_held = i - last_entry_idx
            if days_held < min_hold_days:
                # Cancel this sell signal (too soon)
                filtered_signals.iloc[i] = 0
    
    # 2. Signal confirmation filter - require 2 consecutive days of same signal
    # (Only if params enable it)
    if params.get('require_confirmation', False):
        confirmed_signals = pd.Series(0, index=data.index)
        for i in range(1, len(filtered_signals)):
            # Confirm if current and previous day agree
            if filtered_signals.iloc[i] == filtered_signals.iloc[i-1] and filtered_signals.iloc[i] != 0:
                confirmed_signals.iloc[i] = filtered_signals.iloc[i]
        filtered_signals = confirmed_signals
    
    # 3. Trend strength filter - only trade in strong trends
    # Check if we have trend indicators available
    if params.get('use_trend_filter', False):
        trend_strength = 0
        trend_count = 0
        
        # Method 1: ADX-based trend detection (traditional but can lag)
        if 'ADX_14' in data.columns:
            adx_threshold = params.get('adx_threshold', 20)  # Configurable, default 20
            adx_strong_trend = data['ADX_14'] > adx_threshold
            trend_count += 1
        else:
            adx_strong_trend = pd.Series(True, index=data.index)
        
        # Method 2: Price-based trend detection (more responsive to actual price action)
        # Check if price is above key moving averages
        ma_trend = pd.Series(True, index=data.index)  # Default to True if no MAs
        if 'SMA_50' in data.columns and 'close' in data.columns:
            # Uptrend = price above 50-day MA
            price_above_ma = data['close'] > data['SMA_50']
            ma_trend = ma_trend & price_above_ma
        
        # Combine both methods: ADX confirms strength OR price confirms direction
        # This prevents missing obvious uptrends when ADX temporarily dips
        strong_trend = adx_strong_trend | ma_trend  # OR logic: trade if EITHER confirms trend
        
        # Check Supertrend if available (directional filter)
        if any('SUPERT' in col for col in data.columns):
            # Find supertrend columns
            st_cols = [col for col in data.columns if 'SUPERT_' in col and 'SUPERTd' in col]
            if st_cols:
                # Supertrend direction: 1 = uptrend, -1 = downtrend
                st_trend = data[st_cols[0]]
                # Only allow buys in uptrend, sells in downtrend
                filtered_signals = filtered_signals.where(
                    (filtered_signals != 1) | (st_trend == 1),  # Buy only in uptrend
                    0
                )
        
        # Only trade during strong trends (using combined ADX + MA logic)
        filtered_signals = filtered_signals.where(strong_trend, 0)
    
    return filtered_signals

def objective(trial, data, trade_preference=0.5):
    """
    Objective function for Optuna to dynamically build and test hybrid strategies.
    
    Args:
        trial: Optuna trial object
        data: Market data
        trade_preference: 0.0 = Very Conservative (fewest trades), 1.0 = Very Aggressive (most trades)
                         0.5 = Balanced (default)
    """
    try:
        # Validate input data
        if data is None or data.empty:
            print(f"⚠️  Trial {trial.number}: Empty data")
            return float('-inf')
        
        # Check for required columns
        required_cols = ['close', 'open', 'high', 'low', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"⚠️  Trial {trial.number}: Missing columns: {missing_cols}")
            return float('-inf')
        
        numerical_indicators = [col for col in data.columns if data[col].dtype != 'bool' and col not in ['open', 'high', 'low', 'close', 'volume']]
        boolean_indicators = [col for col in data.columns if data[col].dtype == 'bool']
        
        # Add explicit debug output for first few trials
        if trial.number < 2:
            print(f"\n🎯 OBJECTIVE FUNCTION - TRIAL {trial.number}")
            print(f"   Data shape: {data.shape}")
            print(f"   Columns: {len(data.columns)}")
            print(f"   Trade Preference: {trade_preference:.2f} ({'Conservative' if trade_preference < 0.4 else 'Aggressive' if trade_preference > 0.6 else 'Balanced'})")
            print(f"   Numerical indicators: {len(numerical_indicators)} (first 3: {numerical_indicators[:3]})")
            print(f"   Boolean indicators: {len(boolean_indicators)} (first 3: {boolean_indicators[:3]})")

        params = {}
        active_indicators = []

        # Add flexible scoring thresholds to expand search space
        params['buy_score_threshold'] = trial.suggest_int('buy_score_threshold', 1, 5)
        params['sell_score_threshold'] = trial.suggest_int('sell_score_threshold', 1, 5)
        
        # === ANTI-WHIPSAW PARAMETERS (Dynamic based on trade preference) ===
        # Trade preference: 0.0 = Conservative (fewer trades), 1.0 = Aggressive (more trades)
        
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
        
        # 4. adx_threshold: Conservative = higher threshold (stronger trends only), Aggressive = lower threshold (more trades)
        if trade_preference < 0.4:  # Conservative: 22-30 (strong trends only)
            adx_min, adx_max = 22, 30
        elif trade_preference < 0.7:  # Balanced: 18-26
            adx_min, adx_max = 18, 26
        else:  # Aggressive: 15-22 (weaker trends allowed)
            adx_min, adx_max = 15, 22
        params['adx_threshold'] = trial.suggest_int('adx_threshold', adx_min, adx_max)
        
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
        
        # === IMPROVED OBJECTIVE FUNCTION ===
        # Rewards fewer, high-quality trades instead of overtrading
        
        total_return = summary['total_return_pct']
        num_trades = summary['total_trades']
        win_rate = summary['win_rate'] / 100  # Convert to 0-1
        profit_factor = summary['profit_factor']
        max_drawdown = abs(summary['max_drawdown_pct'])
        
        # 1. Trade Efficiency Score (reward fewer trades with same return)
        # Adjust penalties/bonuses based on user's trade preference
        # trade_preference: 0.0 = Conservative (fewer trades), 0.5 = Balanced, 1.0 = Aggressive (more trades)
        
        # Scale the thresholds based on preference
        # Conservative (0.0): Penalize >30 trades heavily, reward <10
        # Balanced (0.5): Penalize >50 trades, reward <20 (original behavior)
        # Aggressive (1.0): Penalize >80 trades, reward <30
        
        very_high_threshold = 50 + (trade_preference * 30)  # 50-80 trades
        high_threshold = 30 + (trade_preference * 20)       # 30-50 trades
        low_threshold = 20 + (trade_preference * 10)        # 20-30 trades
        very_low_threshold = 10 + (trade_preference * 10)   # 10-20 trades
        
        # Adjust penalty/bonus strength based on preference
        # More conservative = stronger penalties for overtrading
        penalty_strength = 1.0 - (trade_preference * 0.3)   # 1.0 at conservative, 0.7 at aggressive
        bonus_strength = 1.0 + (0.5 - trade_preference) * 0.4  # 1.2 at conservative, 1.0 at aggressive
        
        if num_trades > very_high_threshold:
            trade_penalty = 0.5 + (0.2 * penalty_strength)  # 0.5-0.7 penalty
        elif num_trades > high_threshold:
            trade_penalty = 0.7 + (0.15 * penalty_strength)  # 0.7-0.85 penalty
        elif num_trades < very_low_threshold:
            trade_penalty = 1.0 + (0.2 * bonus_strength)  # 1.0-1.2 bonus
        elif num_trades < low_threshold:
            trade_penalty = 1.0 + (0.1 * bonus_strength)  # 1.0-1.1 bonus
        else:
            trade_penalty = 1.0  # Neutral
        
        # 2. Win Rate Quality (reward high win rates)
        if win_rate > 0.6:
            win_rate_bonus = 1.2  # 20% bonus
        elif win_rate > 0.5:
            win_rate_bonus = 1.1  # 10% bonus
        elif win_rate < 0.4:
            win_rate_bonus = 0.8  # 20% penalty
        else:
            win_rate_bonus = 1.0
        
        # 3. Profit Factor (reward catching big moves)
        if profit_factor > 2.0:
            pf_bonus = 1.3  # 30% bonus for excellent profit factor
        elif profit_factor > 1.5:
            pf_bonus = 1.15  # 15% bonus
        elif profit_factor < 1.2:
            pf_bonus = 0.9  # 10% penalty for barely profitable
        else:
            pf_bonus = 1.0
        
        # 4. Drawdown Penalty (reward lower drawdowns)
        if max_drawdown > 30:
            dd_penalty = 0.7  # Big penalty for large drawdowns
        elif max_drawdown > 20:
            dd_penalty = 0.85
        elif max_drawdown < 10:
            dd_penalty = 1.15  # Bonus for low drawdown
        else:
            dd_penalty = 1.0
        
        # 5. Average Profit Per Trade (reward bigger wins)
        avg_profit_per_trade = total_return / num_trades
        if avg_profit_per_trade > 5:  # >5% per trade
            apt_bonus = 1.2
        elif avg_profit_per_trade > 3:  # >3% per trade
            apt_bonus = 1.1
        elif avg_profit_per_trade < 1:  # <1% per trade (overtrading)
            apt_bonus = 0.8
        else:
            apt_bonus = 1.0
        
        # Calculate quality-adjusted score
        quality_score = (
            total_return 
            * trade_penalty      # Penalize overtrading
            * win_rate_bonus     # Reward high win rate
            * pf_bonus           # Reward catching big moves
            * dd_penalty         # Penalize high drawdown
            * apt_bonus          # Reward profitable trades
        )
        
        # Debug output for top trials
        if trial.number <= 3 or total_return > 50:
            print(f"\n  📊 TRIAL {trial.number} SCORING:")
            print(f"    Base Return: {total_return:.1f}%")
            print(f"    Trades: {num_trades} (penalty: {trade_penalty:.2f}x)")
            print(f"    Win Rate: {win_rate*100:.1f}% (bonus: {win_rate_bonus:.2f}x)")
            print(f"    Profit Factor: {profit_factor:.2f} (bonus: {pf_bonus:.2f}x)")
            print(f"    Max DD: {max_drawdown:.1f}% (penalty: {dd_penalty:.2f}x)")
            print(f"    Avg/Trade: {avg_profit_per_trade:.2f}% (bonus: {apt_bonus:.2f}x)")
            print(f"    → Quality Score: {quality_score:.1f}")
            
        return quality_score
    except Exception as e:
        return -1e9

def run_optimization_batch(args):
    """Run a batch of trials in a separate process"""
    batch_trials, data_copy, trade_preference = args
    local_study = optuna.create_study(direction='maximize')
    local_study.optimize(
        lambda trial: objective(trial, data_copy, trade_preference), 
        n_trials=batch_trials, 
        show_progress_bar=False
    )
    return local_study.trials

def run_optimization(data, n_trials=5000, n_jobs=None, progress_callback=None, trade_preference=0.5):
    """
    Runs the new dynamic optimization and returns top 3 strategies.
    
    Args:
        data: Market data for optimization
        n_trials: Number of optimization trials
        n_jobs: Number of parallel jobs (None = auto-detect cores, 1 = single-threaded)
        progress_callback: Optional callback function(current_trial, total_trials) for progress updates
        trade_preference: 0.0 = Very Conservative (fewest trades), 1.0 = Very Aggressive (most trades)
                         0.5 = Balanced (default)
    """
    import multiprocessing as mp
    
    # Auto-detect CPU cores if not specified
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)  # Leave 1 core free
    
    print(f"\n{'='*60}")
    print("🚀 PATTERN_FINDR OPTIMIZATION STARTING")
    print(f"{'='*60}")
    print(f"📊 Data shape: {data.shape}")
    print(f"📈 Indicators: {len(data.columns)} total columns")
    print(f"🎯 Trials: {n_trials:,}")
    print(f"⚙️  Workers: {n_jobs} ({'SINGLE-THREADED' if n_jobs == 1 else 'MULTI-THREADED'})")
    print(f"📋 Method: STANDARD (debugging enabled)")
    print(f"💰 Trade Preference: {trade_preference:.2f} ({'Very Conservative' if trade_preference < 0.3 else 'Conservative' if trade_preference < 0.4 else 'Balanced' if trade_preference < 0.6 else 'Aggressive' if trade_preference < 0.8 else 'Very Aggressive'})")
    
    # Count indicator types for better visibility
    adv_indicators = [col for col in data.columns if col.startswith('adv_')]
    enh_indicators = [col for col in data.columns if col.startswith('enh_')]
    pattern_indicators = [col for col in data.columns if col.startswith('pattern_')]
    
    print(f"🧠 Advanced indicators: {len(adv_indicators)}")
    print(f"🔍 Enhanced patterns: {len(enh_indicators)}")
    print(f"📊 Chart patterns: {len(pattern_indicators)}")
    
    # Reset universal_strategy call counter for this run
    if hasattr(universal_strategy, 'call_count'):
        delattr(universal_strategy, 'call_count')
    
    print(f"{'='*60}")
    print("🔄 STARTING OPTIMIZATION TRIALS...")
    print(f"{'='*60}\n")
    
    # Create study with parallel execution
    study = optuna.create_study(direction='maximize')
    
    if n_jobs == 1:
        # Single-threaded execution with progress callback
        if progress_callback:
            # Create Optuna callback for progress updates
            def optuna_callback(study, trial):
                current = len([t for t in study.trials if t.state.is_finished()])
                
                # Console progress updates every 10 trials
                if current % 10 == 0 or current == 1:
                    percent = (current / n_trials) * 100
                    best_value = study.best_value if study.best_trial else 0
                    print(f"🔄 Trial {current:,}/{n_trials:,} ({percent:.1f}%) | Best: {best_value:.3f}")
                
                progress_callback(current, n_trials)
            
            study.optimize(
                lambda trial: objective(trial, data, trade_preference), 
                n_trials=n_trials, 
                callbacks=[optuna_callback],
                show_progress_bar=True
            )
        else:
            study.optimize(lambda trial: objective(trial, data, trade_preference), n_trials=n_trials, show_progress_bar=True)
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
        
        # Initial progress update
        if progress_callback:
            progress_callback(0, n_trials)
        
        # Execute batches in parallel
        all_trials = []
        completed_trials = 0
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(run_optimization_batch, (batch_size, data, trade_preference)) for batch_size in batch_sizes]
            
            with tqdm(total=n_trials, desc="Parallel Trials") as pbar:
                for future in as_completed(futures):
                    batch_trials = future.result()
                    all_trials.extend(batch_trials)
                    completed_trials += len(batch_trials)
                    pbar.update(len(batch_trials))
                    
                    # Update Streamlit progress
                    if progress_callback:
                        progress_callback(completed_trials, n_trials)
        
        # Add all trials to the main study
        for trial in all_trials:
            study.add_trial(trial)
    
    # Get top 3 unique trials (best performing strategies)
    # Filter out failed trials and sort by performance
    successful_trials = [t for t in study.trials if t.value is not None and t.value > -1e9]
    top_trials = sorted(successful_trials, key=lambda t: t.value, reverse=True)[:3]
    
    print(f"\n{'='*60}")
    print("🏆 OPTIMIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 Total trials run: {len(study.trials):,}")
    print(f"✅ Successful trials: {len(successful_trials):,}")
    
    if len(study.trials) > 0:
        trial_values = [t.value for t in study.trials if t.value is not None]
        if trial_values:
            print(f"🎯 Best performance: {max(trial_values):.2f}%")
            print(f"📉 Worst performance: {min(trial_values):.2f}%")
            failed_trials = sum(1 for v in trial_values if v <= -1e9)
            if failed_trials > 0:
                print(f"❌ Failed trials: {failed_trials:,}")
    
    print(f"\n🏅 TOP 3 STRATEGIES FOUND:")
    print(f"{'='*60}")
    
    for i, trial in enumerate(top_trials, 1):
        active_indicators = [param.replace('use_', '') for param in trial.params.keys() 
                           if param.startswith('use_') and trial.params[param] == True]
        strategy_name = ' + '.join(active_indicators[:3]) + ('...' if len(active_indicators) > 3 else '')
        buy_thresh = trial.params.get('buy_score_threshold', 1)
        sell_thresh = trial.params.get('sell_score_threshold', 1)
        print(f"  {i}. 📈 {strategy_name}")
        print(f"     💰 Return: {trial.value:.2f}% | Thresholds: Buy={buy_thresh}, Sell={sell_thresh}")
        print()
    
    # Check if we have enough unique strategies
    if len(successful_trials) < 3:
        print(f"⚠️  Only found {len(successful_trials)} successful strategies out of {n_trials:,} trials")
        if len(successful_trials) == 0:
            print("🔍 All trials returned penalty values. Possible issues:")
            print("   - No profitable indicator combinations found")
            print("   - All strategies generated too few trades")
            print("   - Consider adjusting trade preference or parameters")
    else:
        print(f"🎉 Successfully found {len(successful_trials):,} profitable strategies!")
    
    print(f"{'='*60}\n")
    
    return top_trials
