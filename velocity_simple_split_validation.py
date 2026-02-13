#!/usr/bin/env python3
"""
Simple Train/Test Split Validation for Velocity Strategies

This script implements a simple 80/20 train/test split (NOT walk-forward):
1. Train period (80%): First portion - optimize parameters here
2. Test period (20%): Last portion - validate parameters here

This is different from walk-forward validation which uses rolling windows.
Use this script alongside walk-forward to compare results and detect overfitting.

Usage:
    python velocity_simple_split_validation.py --ticker ES=F --interval 15m --period 60d --n-trials 50000 --n-jobs 32
    python velocity_simple_split_validation.py --ticker BTC-USD --interval 1d --period 2y --n-trials 50000 --n-jobs 32
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optuna_worker import VelocityOptunaObjective, run_velocity_study
from oscillator_indicators import create_composite_oscillator_features


def fetch_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    print(f"Fetching {ticker} {interval} data for {period}...")

    yf_ticker = yf.Ticker(ticker)
    df = yf_ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    print(f"  Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def prepare_velocity_data(df: pd.DataFrame, use_extra_indicators: bool = True):
    """Prepare data for velocity optimization including oscillator calculation."""

    # Calculate composite oscillator
    print("Calculating composite oscillator...")
    osc_df = create_composite_oscillator_features(df)

    # Find the oscillator column
    osc_col = None
    for col in ['osc_composite_smooth', 'composite_oscillator', 'osc_composite']:
        if col in osc_df.columns:
            osc_col = col
            break

    if osc_col is None:
        raise ValueError(f"No oscillator column found. Available: {osc_df.columns.tolist()}")

    close_col = 'close' if 'close' in df.columns else 'Close'

    # Align data
    aligned_df = osc_df.loc[df.index].dropna()
    close_prices = df.loc[aligned_df.index, close_col].values
    osc_values = aligned_df[osc_col].values

    # RSI cache (multiple periods)
    rsi_cache = {}
    for period in [5, 14, 21, 30]:
        delta = pd.Series(close_prices).diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_cache[period] = rsi.fillna(50).values

    # MACD histogram
    ema12 = pd.Series(close_prices).ewm(span=12).mean()
    ema26 = pd.Series(close_prices).ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_histogram = (macd_line - signal_line).values

    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close_prices).rolling(bb_period).mean()
    std = pd.Series(close_prices).rolling(bb_period).std()
    bb_upper = (sma + bb_std * std).fillna(method='bfill').values
    bb_lower = (sma - bb_std * std).fillna(method='bfill').values

    return {
        'close_prices': close_prices,
        'osc_values': osc_values,
        'rsi_cache': rsi_cache,
        'macd_histogram': macd_histogram,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'dates': aligned_df.index,
        'use_extra_indicators': use_extra_indicators,
        'all_oscillators': {osc_col: osc_values}
    }


def run_backtest_with_params(params: dict, data: dict) -> dict:
    """Run a single backtest with given parameters."""

    close_prices = data['close_prices']
    osc_values = data['osc_values']

    # Apply smoothing
    if params.get('vel_smoothing', 1) > 1:
        osc_smooth = pd.Series(osc_values).rolling(window=params['vel_smoothing']).mean().bfill().values
    else:
        osc_smooth = osc_values

    velocity = np.diff(osc_smooth, prepend=osc_smooth[0])
    acceleration = np.diff(velocity, prepend=velocity[0])
    jerk = np.diff(acceleration, prepend=acceleration[0])

    # Build conditions
    vel_cross_up = (velocity > 0) & (np.roll(velocity, 1) <= 0)
    vel_cross_down = (velocity < 0) & (np.roll(velocity, 1) >= 0)
    in_oversold = osc_smooth < params.get('oversold_threshold', -0.2)
    in_overbought = osc_smooth > params.get('overbought_threshold', 0.2)
    extreme_mult = params.get('extreme_zone_mult', 2.0)
    extreme_oversold = osc_smooth < (params.get('oversold_threshold', -0.2) * extreme_mult)
    extreme_overbought = osc_smooth > (params.get('overbought_threshold', 0.2) * extreme_mult)

    vel_std_window = params.get('velocity_std_window', 10)
    vel_std = pd.Series(velocity).rolling(vel_std_window, min_periods=1).std().fillna(np.std(velocity)).values
    momentum_mult = params.get('momentum_multiplier', 1.5)
    strong_momentum_up = velocity > vel_std * momentum_mult
    strong_momentum_down = velocity < -vel_std * momentum_mult

    # Signal type conditions
    sig_type = params.get('signal_type', 'any_reversal')
    if sig_type == 'velocity_crossover_and_zone':
        buy_cond = vel_cross_up & in_oversold
        sell_cond = vel_cross_down & in_overbought
    elif sig_type == 'velocity_crossover_or_zone':
        buy_cond = vel_cross_up | extreme_oversold
        sell_cond = vel_cross_down | extreme_overbought
    elif sig_type == 'zone_only':
        buy_cond = extreme_oversold & (velocity > 0)
        sell_cond = extreme_overbought & (velocity < 0)
    elif sig_type == 'momentum':
        buy_cond = strong_momentum_up & (osc_smooth < 0)
        sell_cond = strong_momentum_down & (osc_smooth > 0)
    elif sig_type == 'any_reversal':
        buy_cond = vel_cross_up | extreme_oversold | (strong_momentum_up & in_oversold)
        sell_cond = vel_cross_down | extreme_overbought | (strong_momentum_down & in_overbought)
    else:
        buy_cond = vel_cross_up | extreme_oversold
        sell_cond = vel_cross_down | extreme_overbought

    # Acceleration requirement
    if params.get('require_accel', False):
        buy_cond = buy_cond & (acceleration > 0)
        sell_cond = sell_cond & (acceleration < 0)

    # Trading simulation
    position = 0  # 0 = flat, 1 = long
    entry_price = 0
    entry_bar = 0
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_drawdown = 0.0

    min_bars_between = params.get('min_bars_between', 1)
    min_hold_bars = params.get('min_hold_bars', 1)
    stop_loss_pct = params.get('stop_loss_pct', 5.0)
    take_profit_pct = params.get('take_profit_pct', 10.0)
    exit_opposite = params.get('exit_on_opposite_signal', True)
    exit_midline = params.get('exit_on_midline_cross', False)

    # Acceleration exit params
    use_accel_exit = params.get('use_accel_exit', False)
    accel_exit_type = params.get('accel_exit_type', 'sign_reversal')
    accel_exit_threshold = params.get('accel_exit_threshold', 0.0)
    accel_exit_min_pnl = params.get('accel_exit_min_pnl', 0.5)
    use_jerk_confirm = params.get('use_jerk_confirm', False)
    jerk_confirm_threshold = params.get('jerk_confirm_threshold', 0.0)

    last_trade_bar = -min_bars_between

    for i in range(1, len(close_prices)):
        price = close_prices[i]

        if position == 0:
            # Check for entry
            if buy_cond[i] and (i - last_trade_bar) >= min_bars_between:
                position = 1
                entry_price = price
                entry_bar = i
        else:
            # Check for exit
            current_pnl = (price - entry_price) / entry_price * 100
            bars_held = i - entry_bar

            exit_reason = None

            # Stop loss
            if current_pnl <= -stop_loss_pct:
                exit_reason = 'stop_loss'
            # Take profit
            elif current_pnl >= take_profit_pct:
                exit_reason = 'take_profit'
            # Acceleration exit (for longs: exit when acceleration turns negative)
            elif use_accel_exit and current_pnl >= accel_exit_min_pnl and bars_held >= min_hold_bars:
                accel_exit_triggered = False
                if accel_exit_type == 'sign_reversal':
                    accel_exit_triggered = acceleration[i] < 0
                elif accel_exit_type == 'magnitude':
                    accel_exit_triggered = acceleration[i] < -accel_exit_threshold
                elif accel_exit_type == 'both':
                    accel_exit_triggered = acceleration[i] < 0 and abs(acceleration[i]) > accel_exit_threshold

                if accel_exit_triggered:
                    if not use_jerk_confirm or jerk[i] < -jerk_confirm_threshold:
                        exit_reason = 'accel_exit'
            # Opposite signal
            elif exit_opposite and sell_cond[i] and bars_held >= min_hold_bars:
                exit_reason = 'opposite_signal'
            # Midline cross
            elif exit_midline and osc_smooth[i] > 0 and bars_held >= min_hold_bars:
                exit_reason = 'midline_cross'

            if exit_reason:
                pnl = (price - entry_price) / entry_price * 100
                equity *= (1 + pnl / 100)
                peak_equity = max(peak_equity, equity)
                drawdown = (peak_equity - equity) / peak_equity * 100
                max_drawdown = max(max_drawdown, drawdown)

                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })

                position = 0
                last_trade_bar = i

    # Calculate metrics
    if not trades:
        return {
            'total_return': 0,
            'n_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'profit_factor': 0
        }

    total_return = equity - 100
    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if t['pnl'] < 0]

    return {
        'total_return': total_return,
        'n_trades': len(trades),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
        'max_drawdown': max_drawdown,
        'profit_factor': sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf'),
        'trades': trades
    }


def run_simple_split_validation(
    ticker: str,
    interval: str,
    period: str,
    n_trials: int,
    n_jobs: int,
    train_ratio: float = 0.8,
    optimize_metric: str = 'total_return'
):
    """
    Run simple train/test split validation for velocity strategy.

    Unlike walk-forward validation, this uses a single split:
    1. Split data into train (80%) and test (20%)
    2. Optimize on train period
    3. Validate on test period
    4. Report both results for comparison
    """

    print("=" * 80)
    print("SIMPLE TRAIN/TEST SPLIT VALIDATION FOR VELOCITY STRATEGY")
    print("=" * 80)
    print(f"Ticker: {ticker}")
    print(f"Interval: {interval}")
    print(f"Period: {period}")
    print(f"Train/Test Split: {train_ratio*100:.0f}%/{(1-train_ratio)*100:.0f}%")
    print(f"Optimization Trials: {n_trials:,}")
    print(f"Parallel Workers: {n_jobs}")
    print("=" * 80)
    print("\nNOTE: This is a SIMPLE split validation, NOT walk-forward.")
    print("Compare results with walk-forward validation to detect overfitting.")
    print("=" * 80)

    # Fetch full data
    df = fetch_data(ticker, interval, period)

    # Calculate split point
    n_bars = len(df)
    train_end_idx = int(n_bars * train_ratio)

    train_df = df.iloc[:train_end_idx]
    test_df = df.iloc[train_end_idx:]

    print(f"\nData Split:")
    print(f"  Total bars: {n_bars}")
    print(f"  Train period: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} bars)")
    print(f"  Test period: {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} bars)")

    # Prepare train data for optimization
    print("\n" + "=" * 80)
    print("PHASE 1: OPTIMIZATION ON TRAINING DATA")
    print("=" * 80)

    train_data = prepare_velocity_data(train_df)

    # Save data to temp file for parallel workers
    fd, data_path = tempfile.mkstemp(suffix='.joblib', prefix='velocity_split_')
    os.close(fd)

    train_data_for_optuna = {
        'close_prices': train_data['close_prices'],
        'osc_values': train_data['osc_values'],
        'rsi_cache': train_data['rsi_cache'],
        'macd_histogram': train_data['macd_histogram'],
        'bb_upper': train_data['bb_upper'],
        'bb_lower': train_data['bb_lower'],
        'optimize_metric': optimize_metric,
        'use_extra_indicators': True,
        'all_oscillators': train_data['all_oscillators'],
        'force_midline_exit': False,
        'force_opposite_exit': False,
        'sl_range': (0.5, 15.0),
        'tp_range': (1.0, 30.0),
        'use_drawdown_penalty': True,
        'max_drawdown_threshold': 20.0,
        'drawdown_penalty_weight': 0.3
    }

    joblib.dump(train_data_for_optuna, data_path)

    # Run parallel optimization
    print(f"\nStarting {n_jobs} parallel workers with {n_trials // n_jobs} trials each...")
    start_time = time.time()

    trials_per_worker = n_trials // n_jobs
    seeds = [42 + i for i in range(n_jobs)]

    worker_results = joblib.Parallel(n_jobs=n_jobs, verbose=10)(
        joblib.delayed(run_velocity_study)(
            data_path, trials_per_worker, seed, optimize_metric, worker_id
        )
        for worker_id, seed in enumerate(seeds)
    )

    elapsed = time.time() - start_time
    print(f"\nOptimization completed in {elapsed:.1f}s ({n_trials / elapsed:.1f} trials/sec)")

    # Clean up temp file
    os.remove(data_path)

    # Collect and sort all results
    all_results = []
    for results in worker_results:
        all_results.extend(results)

    # Sort by total return (or optimize metric)
    all_results.sort(key=lambda x: x.get(optimize_metric, 0), reverse=True)

    if not all_results:
        print("ERROR: No valid results from optimization!")
        return None

    best_result = all_results[0]
    # Params are spread into result dict, extract them
    param_keys = [
        'signal_type', 'vel_smoothing', 'extreme_zone_mult', 'min_bars_between',
        'require_accel', 'oversold_threshold', 'overbought_threshold',
        'stop_loss_pct', 'take_profit_pct', 'exit_on_opposite_signal', 'exit_on_midline_cross',
        'min_hold_bars', 'velocity_std_window', 'momentum_multiplier',
        'double_bottom_lookback', 'divergence_lookback',
        'use_accel_exit', 'accel_exit_type', 'accel_exit_threshold',
        'accel_exit_min_pnl', 'accel_exit_lookback', 'use_jerk_confirm', 'jerk_confirm_threshold',
        'rsi_filter', 'rsi_period', 'rsi_oversold', 'rsi_overbought',
        'use_macd_confirm', 'use_bb_filter', 'oscillator_type'
    ]
    best_params = {k: best_result[k] for k in param_keys if k in best_result}

    print(f"\n{'=' * 80}")
    print("BEST PARAMETERS FROM TRAINING (Top 3)")
    print("=" * 80)

    for i, result in enumerate(all_results[:3]):
        n_trades = result.get('num_trades', result.get('n_trades', 0))
        print(f"\n#{i+1}: Return={result['total_return']:.2f}%, WR={result['win_rate']:.1f}%, "
              f"Trades={n_trades}, DD={result['max_drawdown']:.1f}%")
        print(f"    signal_type: {result.get('signal_type')}")
        print(f"    use_accel_exit: {result.get('use_accel_exit')}")
        if result.get('use_accel_exit'):
            print(f"    accel_exit_type: {result.get('accel_exit_type')}")
            print(f"    accel_exit_threshold: {result.get('accel_exit_threshold', 0):.4f}")
            print(f"    accel_exit_min_pnl: {result.get('accel_exit_min_pnl', 0):.2f}%")

    # Prepare test data
    print(f"\n{'=' * 80}")
    print("PHASE 2: OUT-OF-SAMPLE VALIDATION ON TEST DATA")
    print("=" * 80)

    test_data = prepare_velocity_data(test_df)

    # Run backtest on test data with best params
    test_result = run_backtest_with_params(best_params, test_data)

    print(f"\nTest Period Results ({test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}):")
    print(f"  Total Return: {test_result['total_return']:.2f}%")
    print(f"  Win Rate: {test_result['win_rate']:.1f}%")
    print(f"  Trades: {test_result['n_trades']}")
    print(f"  Max Drawdown: {test_result['max_drawdown']:.1f}%")
    print(f"  Profit Factor: {test_result['profit_factor']:.2f}")

    # Compare train vs test
    print(f"\n{'=' * 80}")
    print("TRAIN vs TEST COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<20} {'Train':>15} {'Test':>15} {'Diff':>15}")
    print("-" * 65)

    train_return = best_result['total_return']
    test_return = test_result['total_return']
    print(f"{'Total Return':<20} {train_return:>14.2f}% {test_return:>14.2f}% {test_return - train_return:>+14.2f}%")

    train_wr = best_result['win_rate']
    test_wr = test_result['win_rate']
    print(f"{'Win Rate':<20} {train_wr:>14.1f}% {test_wr:>14.1f}% {test_wr - train_wr:>+14.1f}%")

    train_dd = best_result['max_drawdown']
    test_dd = test_result['max_drawdown']
    print(f"{'Max Drawdown':<20} {train_dd:>14.1f}% {test_dd:>14.1f}% {test_dd - train_dd:>+14.1f}%")

    # Validation assessment
    print(f"\n{'=' * 80}")
    print("VALIDATION ASSESSMENT")
    print("=" * 80)

    warnings = []
    is_valid = True

    # Check for overfitting indicators
    if test_return < train_return * 0.3:
        warnings.append("WARNING: Test return is less than 30% of train return - possible overfitting")
        is_valid = False

    if test_wr < train_wr - 15:
        warnings.append("WARNING: Win rate dropped significantly - parameters may not generalize")
        is_valid = False

    if test_result['n_trades'] < 10:
        warnings.append("WARNING: Very few trades in test period - results may not be statistically significant")

    if test_return < 0:
        warnings.append("CRITICAL: Negative test return - strategy loses money out-of-sample")
        is_valid = False

    if warnings:
        for w in warnings:
            print(f"  {w}")
    else:
        print("  PASSED: Strategy shows reasonable out-of-sample performance")
        print(f"  Test return ({test_return:.2f}%) is {test_return/train_return*100:.1f}% of training return")

    # Build results dict
    results = {
        'validation_type': 'simple_split',
        'ticker': ticker,
        'interval': interval,
        'period': period,
        'train_ratio': train_ratio,
        'n_trials': n_trials,
        'n_jobs': n_jobs,
        'optimize_metric': optimize_metric,
        'train_period': f"{train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')}",
        'test_period': f"{test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}",
        'train_bars': len(train_df),
        'test_bars': len(test_df),
        'best_params': best_params,
        'train_result': {
            'total_return': best_result['total_return'],
            'win_rate': best_result['win_rate'],
            'n_trades': best_result.get('num_trades', 0),
            'max_drawdown': best_result['max_drawdown'],
            'profit_factor': best_result.get('profit_factor', 0)
        },
        'test_result': {
            'total_return': test_result['total_return'],
            'win_rate': test_result['win_rate'],
            'n_trades': test_result['n_trades'],
            'max_drawdown': test_result['max_drawdown'],
            'profit_factor': test_result['profit_factor']
        },
        'is_valid': is_valid,
        'warnings': warnings,
        'optimization_method': {
            'script': 'velocity_simple_split_validation.py',
            'n_trials': n_trials,
            'n_jobs': n_jobs,
            'metric': optimize_metric,
            'ticker': ticker,
            'interval': interval,
            'period': period,
            'train_ratio': train_ratio,
            'optimized_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }

    return results


def update_strategy_config(config_path: str, new_params: dict, validation_results: dict,
                          optimization_info: dict = None):
    """Update strategy config with new optimized parameters."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Update parameters
    params_to_update = [
        'signal_type', 'vel_smoothing', 'extreme_zone_mult', 'min_bars_between',
        'require_accel', 'oversold_threshold', 'overbought_threshold',
        'stop_loss_pct', 'take_profit_pct', 'exit_on_opposite_signal', 'exit_on_midline_cross',
        # Acceleration exit params
        'use_accel_exit', 'accel_exit_type', 'accel_exit_threshold',
        'accel_exit_min_pnl', 'accel_exit_lookback', 'use_jerk_confirm', 'jerk_confirm_threshold'
    ]

    for param in params_to_update:
        if param in new_params:
            config[param] = new_params[param]

    # Add validation metadata (different prefix than walk-forward)
    config['split_validated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config['split_train_return'] = validation_results['train_result']['total_return']
    config['split_test_return'] = validation_results['test_result']['total_return']
    config['split_train_period'] = validation_results['train_period']
    config['split_test_period'] = validation_results['test_period']

    # Add optimization method tracking
    if optimization_info:
        config['optimization_method'] = optimization_info
    else:
        config['optimization_method'] = {
            'script': 'velocity_simple_split_validation.py',
            'optimized_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    # Backup original
    backup_path = config_path + '.before_split'
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy(config_path, backup_path)

    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\nConfig updated: {config_path}")
    return config


def main():
    parser = argparse.ArgumentParser(description='Simple Train/Test Split Validation for Velocity Strategies')
    parser.add_argument('--ticker', type=str, default='BTC-USD', help='Ticker symbol')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval')
    parser.add_argument('--period', type=str, default='2y', help='Data period')
    parser.add_argument('--n-trials', type=int, default=50000, help='Number of optimization trials')
    parser.add_argument('--n-jobs', type=int, default=32, help='Number of parallel workers')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/test split ratio')
    parser.add_argument('--config-path', type=str, help='Path to strategy config to update')
    parser.add_argument('--metric', type=str, default='total_return', help='Optimization metric')

    args = parser.parse_args()

    # Run simple split validation
    results = run_simple_split_validation(
        ticker=args.ticker,
        interval=args.interval,
        period=args.period,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        train_ratio=args.train_ratio,
        optimize_metric=args.metric
    )

    if results is None:
        print("Simple split validation failed!")
        sys.exit(1)

    # Save results with optimization tracking
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"velocity_split_results_{args.ticker}_{args.period}_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=str)

    print(f"\nResults saved to: {results_file}")

    # Update config if path provided
    if args.config_path and os.path.exists(args.config_path):
        if results['is_valid']:
            optimization_info = results['optimization_method']
            update_strategy_config(args.config_path, results['best_params'], results, optimization_info)
        else:
            print("\nWARNING: Validation failed - config NOT updated")
            print("Review warnings above and consider re-running with different parameters")

    # Print final summary
    print(f"\n{'=' * 80}")
    print("OPTIMIZED PARAMETERS FOR DEPLOYMENT")
    print("=" * 80)

    params = results['best_params']
    print(json.dumps({
        'signal_type': params.get('signal_type'),
        'use_accel_exit': params.get('use_accel_exit'),
        'accel_exit_type': params.get('accel_exit_type'),
        'accel_exit_threshold': params.get('accel_exit_threshold'),
        'accel_exit_min_pnl': params.get('accel_exit_min_pnl'),
        'use_jerk_confirm': params.get('use_jerk_confirm'),
        'stop_loss_pct': params.get('stop_loss_pct'),
        'take_profit_pct': params.get('take_profit_pct'),
        'oversold_threshold': params.get('oversold_threshold'),
        'overbought_threshold': params.get('overbought_threshold')
    }, indent=4))

    # Print comparison guidance
    print(f"\n{'=' * 80}")
    print("NEXT STEPS: COMPARE WITH WALK-FORWARD VALIDATION")
    print("=" * 80)
    print(f"\nRun walk-forward validation for comparison:")
    print(f"  python velocity_walkforward_validation.py --ticker {args.ticker} --interval {args.interval} --period {args.period} --n-trials {args.n_trials} --n-jobs {args.n_jobs}")
    print(f"\nCompare the results:")
    print(f"  - If both methods find similar parameters: Higher confidence")
    print(f"  - If parameters differ significantly: Investigate further")
    print(f"  - If walk-forward passes but simple split fails (or vice versa): Use caution")


if __name__ == '__main__':
    main()
