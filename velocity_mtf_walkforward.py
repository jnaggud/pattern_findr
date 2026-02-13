#!/usr/bin/env python3
"""
Multi-Timeframe Walk-Forward Optimization for Velocity Strategies

Computes oscillators on sub-interval data (e.g., 5m) then samples at primary
interval boundaries (e.g., 15m) for trade decisions. This gives smoother
oscillators without lookahead bias, replicating the benefit that center=True
provided illegitimately.

Usage:
    python velocity_mtf_walkforward.py \
      --ticker ES=F \
      --interval 15m \
      --sub-interval 5m \
      --period 60d \
      --n-trials 100000 \
      --n-jobs 32 \
      --metric profit_factor \
      --train-ratio 0.8
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optuna_worker import run_velocity_study
from velocity_walkforward_validation import run_backtest_with_params
from oscillator_indicators import create_composite_oscillator_features
from novel_indicators import (
    calculate_arwo, calculate_dco, calculate_vcmo, calculate_ics,
    calculate_mji, calculate_prf, calculate_ewaf, calculate_kfif
)
from novel_indicators_v2 import calculate_rsc, calculate_mfi2, calculate_sei


# --- Interval validation and conversion ---

# yfinance period limits by sub-interval
YFINANCE_LIMITS = {
    '1m': '7d',
    '2m': '60d',
    '5m': '60d',
    '15m': '60d',
    '30m': '60d',
    '60m': '730d',
    '1h': '730d',
    '90m': '60d',
}

# Mapping from interval string to minutes
INTERVAL_MINUTES = {
    '1m': 1, '2m': 2, '5m': 5, '15m': 15, '30m': 30,
    '60m': 60, '1h': 60, '90m': 90,
}


def _interval_to_resample_rule(interval: str) -> str:
    """Convert interval string to pandas resample rule.

    '15m' -> '15min', '1h' -> '1h' (already valid), '60m' -> '60min'
    """
    if interval.endswith('m') and not interval.endswith('min'):
        return interval.replace('m', 'min')
    return interval


def _validate_intervals(primary_interval: str, sub_interval: str):
    """Validate that sub_interval evenly divides primary_interval."""
    primary_min = INTERVAL_MINUTES.get(primary_interval)
    sub_min = INTERVAL_MINUTES.get(sub_interval)

    if primary_min is None:
        raise ValueError(f"Unsupported primary interval: {primary_interval}")
    if sub_min is None:
        raise ValueError(f"Unsupported sub-interval: {sub_interval}")
    if sub_min >= primary_min:
        raise ValueError(
            f"Sub-interval ({sub_interval}={sub_min}min) must be smaller than "
            f"primary interval ({primary_interval}={primary_min}min)"
        )
    if primary_min % sub_min != 0:
        raise ValueError(
            f"Sub-interval ({sub_interval}={sub_min}min) must evenly divide "
            f"primary interval ({primary_interval}={primary_min}min). "
            f"Remainder: {primary_min % sub_min}"
        )


def _validate_period(sub_interval: str, period: str):
    """Warn if period exceeds yfinance limits for the sub-interval."""
    limit = YFINANCE_LIMITS.get(sub_interval)
    if limit is None:
        return
    # Simple heuristic: extract numeric days
    if sub_interval == '1m' and period != '7d':
        print(f"  WARNING: 1m data limited to 7d by yfinance. Requested {period}.")
        print(f"  Consider using 5m sub-interval for periods > 7d.")


# --- Data fetching ---

def fetch_sub_interval_data(ticker: str, sub_interval: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance at the sub-interval resolution."""
    print(f"Fetching {ticker} {sub_interval} data for {period}...")

    _validate_period(sub_interval, period)

    yf_ticker = yf.Ticker(ticker)
    df = yf_ticker.history(period=period, interval=sub_interval)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} at {sub_interval}/{period}")

    # Standardize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Drop non-OHLCV columns if present (dividends, stock splits)
    keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    df = df[keep_cols]

    print(f"  Fetched {len(df)} sub-interval bars from {df.index[0]} to {df.index[-1]}")
    return df


# --- Resampling ---

def resample_ohlcv(df_sub: pd.DataFrame, primary_interval: str) -> pd.DataFrame:
    """Aggregate sub-interval OHLCV to primary interval bars.

    open=first, high=max, low=min, close=last, volume=sum.
    Drops incomplete bars (NaN rows).
    """
    rule = _interval_to_resample_rule(primary_interval)

    # Strip timezone before resample to avoid DST edge cases, reattach after
    original_tz = df_sub.index.tz
    if original_tz is not None:
        df_work = df_sub.copy()
        df_work.index = df_work.index.tz_localize(None)
    else:
        df_work = df_sub

    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }
    if 'volume' in df_work.columns:
        agg_dict['volume'] = 'sum'

    df_primary = df_work.resample(rule).agg(agg_dict).dropna()

    # Reattach timezone
    if original_tz is not None:
        df_primary.index = df_primary.index.tz_localize(original_tz)

    return df_primary


def resample_indicator(series_sub: pd.Series, primary_interval: str) -> pd.Series:
    """Resample a scalar indicator series to primary interval by taking LAST value.

    The last sub-interval value within each primary window uses only past data,
    ensuring no lookahead bias.
    """
    rule = _interval_to_resample_rule(primary_interval)

    # Strip timezone
    original_tz = series_sub.index.tz
    if original_tz is not None:
        s = series_sub.copy()
        s.index = s.index.tz_localize(None)
    else:
        s = series_sub

    resampled = s.resample(rule).last().dropna()

    if original_tz is not None:
        resampled.index = resampled.index.tz_localize(original_tz)

    return resampled


# --- Core MTF data preparation ---

def prepare_mtf_velocity_data(df_sub: pd.DataFrame, primary_interval: str,
                              use_extra_indicators: bool = True) -> dict:
    """Prepare velocity data using multi-timeframe approach.

    1. Compute all indicators on sub-interval (e.g., 5m) data
    2. Resample indicators to primary interval (e.g., 15m) using last()
    3. Resample OHLCV to primary interval using standard aggregation
    4. Align all arrays and return data dict for run_backtest_with_params()
    """
    print("Computing indicators on sub-interval data...")

    # --- Step 1: Compute all oscillators on sub-interval data ---

    df_lower = df_sub.copy()
    df_lower.columns = [c.lower() for c in df_lower.columns]

    # Composite oscillator
    print("  - Composite oscillator...")
    osc_df = create_composite_oscillator_features(df_sub)

    composite_col = None
    for col in ['osc_composite_smooth', 'composite_oscillator', 'osc_composite']:
        if col in osc_df.columns:
            composite_col = col
            break
    if composite_col is None:
        raise ValueError(f"No composite oscillator column found. Available: {osc_df.columns.tolist()}")

    # Get composite on sub-interval index
    composite_sub = osc_df[composite_col].dropna()

    # Novel oscillators on sub-interval data
    novel_oscillators_sub = {}

    for name, calc_fn, unpack in [
        ('ics', calculate_ics, False),
        ('arwo', calculate_arwo, False),
        ('dco', calculate_dco, False),
        ('vcmo', calculate_vcmo, False),
        ('mji', calculate_mji, False),
        ('prf', calculate_prf, False),
        ('ewaf', calculate_ewaf, False),
        ('kfif', calculate_kfif, True),  # Returns tuple
    ]:
        print(f"  - {name.upper()}...")
        try:
            result = calc_fn(df_lower)
            if unpack:
                novel_oscillators_sub[name] = result[0]  # First element is the main series
            else:
                novel_oscillators_sub[name] = result
        except Exception as e:
            print(f"    Warning: {name.upper()} calculation failed: {e}")

    # --- Step 2: Compute RSI, MACD, BB on sub-interval close ---

    close_sub = df_lower['close']

    # RSI (multiple periods)
    rsi_sub = {}
    for period in [5, 14, 21, 30]:
        delta = close_sub.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_sub[period] = (100 - (100 / (1 + rs))).fillna(50)

    # MACD histogram
    ema12 = close_sub.ewm(span=12).mean()
    ema26 = close_sub.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist_sub = macd_line - signal_line

    # Bollinger Bands
    bb_period = 20
    bb_std_mult = 2
    sma = close_sub.rolling(bb_period).mean()
    std = close_sub.rolling(bb_period).std()
    bb_upper_sub = sma + bb_std_mult * std
    bb_lower_sub = sma - bb_std_mult * std

    # --- Step 2b: Compute V2 indicators (RSC, MFI2, SEI) on sub-interval data ---

    v2_indicators_sub = {}
    for name, calc_fn in [
        ('rsc', calculate_rsc),
        ('mfi2', calculate_mfi2),
        ('sei', calculate_sei),
    ]:
        print(f"  - V2: {name.upper()}...")
        try:
            v2_indicators_sub[name] = calc_fn(df_lower)
        except Exception as e:
            print(f"    Warning: V2 {name.upper()} calculation failed: {e}")

    # --- Step 3: Resample everything to primary interval ---

    print(f"Resampling to {primary_interval} boundaries...")

    # OHLCV aggregation
    df_primary = resample_ohlcv(df_sub, primary_interval)

    # Resample composite oscillator
    composite_primary = resample_indicator(composite_sub, primary_interval)

    # Resample novel oscillators
    novel_oscillators_primary = {}
    for name, series in novel_oscillators_sub.items():
        novel_oscillators_primary[name] = resample_indicator(series, primary_interval)

    # Resample RSI
    rsi_primary = {}
    for period, series in rsi_sub.items():
        rsi_primary[period] = resample_indicator(series, primary_interval)

    # Resample MACD, BB
    macd_hist_primary = resample_indicator(macd_hist_sub, primary_interval)
    bb_upper_primary = resample_indicator(bb_upper_sub, primary_interval)
    bb_lower_primary = resample_indicator(bb_lower_sub, primary_interval)

    # Resample V2 indicators
    v2_indicators_primary = {}
    for name, series in v2_indicators_sub.items():
        v2_indicators_primary[name] = resample_indicator(series, primary_interval)

    # --- Step 4: Align all arrays to common valid index ---

    # Start with OHLCV index as the base
    valid_idx = df_primary.index

    # Intersect with composite oscillator
    valid_idx = valid_idx.intersection(composite_primary.index)

    # Intersect with novel oscillators
    for name, series in novel_oscillators_primary.items():
        valid_idx = valid_idx.intersection(series.index)

    # Intersect with RSI
    for period, series in rsi_primary.items():
        valid_idx = valid_idx.intersection(series.index)

    # Intersect with MACD, BB
    valid_idx = valid_idx.intersection(macd_hist_primary.index)
    valid_idx = valid_idx.intersection(bb_upper_primary.index)
    valid_idx = valid_idx.intersection(bb_lower_primary.index)

    # Intersect with V2 indicators
    for name, series in v2_indicators_primary.items():
        valid_idx = valid_idx.intersection(series.index)

    valid_idx = valid_idx.sort_values()

    # Build all_oscillators dict (aligned to valid_idx)
    all_oscillators = {
        'composite': composite_primary.loc[valid_idx].values,
    }
    for name, series in novel_oscillators_primary.items():
        all_oscillators[name] = series.loc[valid_idx].values

    # Default osc_values = composite
    osc_values = all_oscillators['composite']

    # Aligned OHLCV
    close_prices = df_primary.loc[valid_idx, 'close'].values
    low_prices = df_primary.loc[valid_idx, 'low'].values
    high_prices = df_primary.loc[valid_idx, 'high'].values

    # Aligned RSI cache
    rsi_cache = {}
    for period, series in rsi_primary.items():
        rsi_cache[period] = series.loc[valid_idx].values

    # Aligned MACD, BB
    macd_histogram = macd_hist_primary.loc[valid_idx].values
    bb_upper = bb_upper_primary.loc[valid_idx].fillna(method='bfill').values
    bb_lower = bb_lower_primary.loc[valid_idx].fillna(method='bfill').values

    # Build V2 indicators dict (aligned to valid_idx)
    v2_indicators = {}
    for name, series in v2_indicators_primary.items():
        v2_indicators[name] = series.loc[valid_idx].values

    print(f"  Calculated {len(all_oscillators)} oscillator types: {list(all_oscillators.keys())}")
    print(f"  Calculated {len(v2_indicators)} V2 filters: {list(v2_indicators.keys())}")

    # Data integrity check
    n = len(valid_idx)
    ratio = len(df_sub) / n if n > 0 else 0
    print(f"\n  Data integrity:")
    print(f"    Sub-interval bars: {len(df_sub)}")
    print(f"    Primary bars:      {n}")
    print(f"    Ratio:             {ratio:.1f}x")

    # Verify all arrays have same length
    assert len(close_prices) == n, f"close_prices length {len(close_prices)} != {n}"
    assert len(low_prices) == n, f"low_prices length {len(low_prices)} != {n}"
    assert len(high_prices) == n, f"high_prices length {len(high_prices)} != {n}"
    assert len(osc_values) == n, f"osc_values length {len(osc_values)} != {n}"
    assert len(macd_histogram) == n, f"macd_histogram length {len(macd_histogram)} != {n}"
    for osc_name, osc_arr in all_oscillators.items():
        assert len(osc_arr) == n, f"{osc_name} length {len(osc_arr)} != {n}"
    for period, rsi_arr in rsi_cache.items():
        assert len(rsi_arr) == n, f"RSI({period}) length {len(rsi_arr)} != {n}"
    for v2_name, v2_arr in v2_indicators.items():
        assert len(v2_arr) == n, f"V2 {v2_name} length {len(v2_arr)} != {n}"

    return {
        'close_prices': close_prices,
        'low_prices': low_prices,
        'high_prices': high_prices,
        'osc_values': osc_values,
        'rsi_cache': rsi_cache,
        'macd_histogram': macd_histogram,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'dates': valid_idx,
        'use_extra_indicators': use_extra_indicators,
        'all_oscillators': all_oscillators,
        'v2_indicators': v2_indicators,
    }


# --- Walk-forward orchestrator ---

def run_mtf_walk_forward(
    ticker: str,
    primary_interval: str,
    sub_interval: str,
    period: str,
    n_trials: int,
    n_jobs: int,
    train_ratio: float = 0.8,
    optimize_metric: str = 'total_return',
):
    """Run multi-timeframe walk-forward optimization.

    1. Fetch sub-interval data
    2. Split by date into train/test
    3. Compute indicators on sub-interval, resample to primary for each split
    4. Optuna optimization on train
    5. Validate top 20 on OOS test, pick best valid
    6. Print B&H comparison and save JSON
    """
    _validate_intervals(primary_interval, sub_interval)

    print("=" * 80)
    print("MULTI-TIMEFRAME WALK-FORWARD VALIDATION")
    print("=" * 80)
    print(f"Ticker:         {ticker}")
    print(f"Primary:        {primary_interval}")
    print(f"Sub-interval:   {sub_interval}")
    print(f"Period:          {period}")
    print(f"Train/Test:      {train_ratio*100:.0f}%/{(1-train_ratio)*100:.0f}%")
    print(f"Trials:          {n_trials:,}")
    print(f"Workers:         {n_jobs}")
    print(f"Metric:          {optimize_metric}")
    print("=" * 80)

    # --- Fetch full sub-interval data ---
    df_sub = fetch_sub_interval_data(ticker, sub_interval, period)

    # --- Split by date ---
    n_bars = len(df_sub)
    train_end_idx = int(n_bars * train_ratio)

    train_sub = df_sub.iloc[:train_end_idx]
    test_sub = df_sub.iloc[train_end_idx:]

    print(f"\nData Split (sub-interval bars):")
    print(f"  Total:  {n_bars}")
    print(f"  Train:  {train_sub.index[0]} to {train_sub.index[-1]} ({len(train_sub)} bars)")
    print(f"  Test:   {test_sub.index[0]} to {test_sub.index[-1]} ({len(test_sub)} bars)")

    # --- Prepare MTF data for train ---
    print(f"\n{'=' * 80}")
    print("PHASE 1: COMPUTING MTF INDICATORS ON TRAINING DATA")
    print("=" * 80)

    train_data = prepare_mtf_velocity_data(train_sub, primary_interval)

    # --- Save data for parallel Optuna workers ---
    fd, data_path = tempfile.mkstemp(suffix='.joblib', prefix='velocity_mtf_')
    os.close(fd)

    train_bars = len(train_data['close_prices'])
    min_trades = max(50, train_bars // 100)

    print(f"\n  Primary bars for training: {train_bars}")
    print(f"  Minimum trade count filter: {min_trades}")

    # V2 filter settings: enable all for Optuna to search over
    v2_filter_settings = {
        'use_regime_filter': 'rsc' in train_data.get('v2_indicators', {}),
        'use_fragility_filter': 'mfi2' in train_data.get('v2_indicators', {}),
        'use_entropy_filter': 'sei' in train_data.get('v2_indicators', {}),
    }
    enabled_v2 = [k for k, v in v2_filter_settings.items() if v]
    print(f"  V2 filter search enabled: {enabled_v2 or 'none'}")

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
        'v2_indicators': train_data.get('v2_indicators', {}),
        'v2_filter_settings': v2_filter_settings,
        'force_midline_exit': False,
        'force_opposite_exit': False,
        'sl_range': (2.0, 10.0),
        'tp_range': (2.0, 20.0),
        'use_drawdown_penalty': True,
        'max_drawdown_threshold': 10.0,
        'drawdown_penalty_weight': 0.5,
        'min_trades': min_trades,
        'trade_count_bonus_weight': 0.3 if optimize_metric in ('total_return', 'risk_adjusted') else 0.0,
    }

    joblib.dump(train_data_for_optuna, data_path)

    # --- Run parallel Optuna optimization ---
    print(f"\n{'=' * 80}")
    print("PHASE 2: OPTUNA OPTIMIZATION ON TRAINING DATA")
    print("=" * 80)

    trials_per_worker = n_trials // n_jobs
    seeds = [42 + i for i in range(n_jobs)]

    print(f"Starting {n_jobs} parallel workers with {trials_per_worker} trials each...")
    start_time = time.time()

    worker_results = joblib.Parallel(n_jobs=n_jobs, verbose=10)(
        joblib.delayed(run_velocity_study)(
            data_path, trials_per_worker, seed, optimize_metric, worker_id
        )
        for worker_id, seed in enumerate(seeds)
    )

    elapsed = time.time() - start_time
    print(f"\nOptimization completed in {elapsed:.1f}s ({n_trials / elapsed:.1f} trials/sec)")

    # Clean up temp file
    try:
        os.remove(data_path)
    except OSError:
        pass

    # Collect and sort results
    all_results = []
    for results in worker_results:
        all_results.extend(results)

    all_results.sort(key=lambda x: x.get(optimize_metric, 0), reverse=True)

    if not all_results:
        print("ERROR: No valid results from optimization!")
        return None

    # --- Show top training results ---
    param_keys = [
        'oscillator_type', 'signal_type', 'vel_smoothing', 'extreme_zone_mult',
        'min_bars_between', 'require_accel', 'oversold_threshold', 'overbought_threshold',
        'stop_loss_pct', 'take_profit_pct', 'exit_on_opposite_signal', 'exit_on_midline_cross',
        'min_hold_bars', 'velocity_std_window', 'momentum_multiplier',
        'double_bottom_lookback', 'divergence_lookback',
        'use_accel_exit', 'accel_exit_type', 'accel_exit_threshold',
        'accel_exit_min_pnl', 'accel_exit_lookback', 'use_jerk_confirm', 'jerk_confirm_threshold',
        'rsi_filter', 'rsi_period', 'rsi_oversold', 'rsi_overbought',
        'use_macd_confirm', 'use_bb_filter', 'use_regime_filter', 'use_fragility_filter',
        'use_entropy_filter',
    ]

    best_params = {k: all_results[0][k] for k in param_keys if k in all_results[0]}

    print(f"\n{'=' * 80}")
    print("BEST PARAMETERS FROM TRAINING (Top 3)")
    print("=" * 80)

    for i, result in enumerate(all_results[:3]):
        n_trades = result.get('num_trades', result.get('n_trades', 0))
        print(f"\n#{i+1}: Return={result['total_return']:.2f}%, WR={result['win_rate']:.1f}%, "
              f"Trades={n_trades}, DD={result['max_drawdown']:.1f}%")
        print(f"    oscillator_type: {result.get('oscillator_type', 'composite')}")
        print(f"    signal_type: {result.get('signal_type')}")
        if result.get('use_accel_exit'):
            print(f"    accel_exit: {result.get('accel_exit_type')} "
                  f"(thresh={result.get('accel_exit_threshold', 0):.4f}, "
                  f"min_pnl={result.get('accel_exit_min_pnl', 0):.2f}%)")

    # --- Prepare test data ---
    print(f"\n{'=' * 80}")
    print("PHASE 3: OUT-OF-SAMPLE VALIDATION ON TEST DATA")
    print("=" * 80)

    test_data = prepare_mtf_velocity_data(test_sub, primary_interval)

    # Test top N on OOS
    top_n = min(20, len(all_results))
    print(f"\nTesting top {top_n} training configs on out-of-sample data...")

    oos_candidates = []
    for i, train_result in enumerate(all_results[:top_n]):
        candidate_params = {k: train_result[k] for k in param_keys if k in train_result}
        oos_result = run_backtest_with_params(candidate_params, test_data)
        oos_candidates.append({
            'rank': i + 1,
            'params': candidate_params,
            'train_result': train_result,
            'test_result': oos_result,
        })
        n_trades_train = train_result.get('num_trades', train_result.get('n_trades', 0))
        n_trades_test = oos_result.get('n_trades', 0)
        print(f"  #{i+1}: Train={train_result['total_return']:.2f}%/{n_trades_train}t "
              f"-> Test={oos_result['total_return']:.2f}%/{n_trades_test}t "
              f"WR={oos_result['win_rate']:.1f}% PF={oos_result['profit_factor']:.2f}")

    # Pick best valid OOS result
    valid_oos = [c for c in oos_candidates
                 if c['test_result']['total_return'] > 0
                 and c['test_result']['n_trades'] >= 5]

    if valid_oos:
        valid_oos.sort(key=lambda x: x['test_result']['total_return'], reverse=True)
        best_candidate = valid_oos[0]
        best_params = best_candidate['params']
        best_result = best_candidate['train_result']
        test_result = best_candidate['test_result']
        is_valid = True
        print(f"\n  Best OOS config: Training rank #{best_candidate['rank']}")
    else:
        print(f"\n  No config passed OOS validation (positive return + >=5 trades)")
        print(f"  Using best training config as fallback")
        best_result = all_results[0]
        best_params = {k: best_result[k] for k in param_keys if k in best_result}
        test_result = run_backtest_with_params(best_params, test_data)
        is_valid = False

    # --- Buy & Hold comparison ---
    print(f"\n{'=' * 80}")
    print("STRATEGY vs BUY & HOLD COMPARISON")
    print("=" * 80)

    # B&H on train
    train_close = train_data['close_prices']
    train_bh_return = (train_close[-1] / train_close[0] - 1) * 100

    # B&H on test
    test_close = test_data['close_prices']
    test_bh_return = (test_close[-1] / test_close[0] - 1) * 100

    # Calculate days from the dates index
    train_dates = train_data['dates']
    test_dates = test_data['dates']

    train_days = max((train_dates[-1] - train_dates[0]).days, 1)
    test_days = max((test_dates[-1] - test_dates[0]).days, 1)

    train_return = best_result['total_return']
    test_return = test_result['total_return']

    train_daily = train_return / train_days
    test_daily = test_return / test_days
    train_bh_daily = train_bh_return / train_days
    test_bh_daily = test_bh_return / test_days

    print(f"\nTRAIN PERIOD ({train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')}, {train_days}d)")
    print(f"  Strategy:     {train_return:+.2f}%  ({train_daily:+.3f}%/day)")
    print(f"  Buy & Hold:   {train_bh_return:+.2f}%  ({train_bh_daily:+.3f}%/day)")
    print(f"  Alpha:        {train_return - train_bh_return:+.2f}%")

    print(f"\nTEST PERIOD ({test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}, {test_days}d)")
    print(f"  Strategy:     {test_return:+.2f}%  ({test_daily:+.3f}%/day)")
    print(f"  Buy & Hold:   {test_bh_return:+.2f}%  ({test_bh_daily:+.3f}%/day)")
    print(f"  Alpha:        {test_return - test_bh_return:+.2f}%")

    # --- Train vs Test comparison ---
    print(f"\n{'=' * 80}")
    print("TRAIN vs TEST COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<20} {'Train':>15} {'Test':>15} {'Diff':>15}")
    print("-" * 65)

    print(f"{'Total Return':<20} {train_return:>14.2f}% {test_return:>14.2f}% {test_return - train_return:>+14.2f}%")
    print(f"{'Avg Return/Day':<20} {train_daily:>14.3f}% {test_daily:>14.3f}% {test_daily - train_daily:>+14.3f}%")
    print(f"{'Days in Period':<20} {train_days:>15} {test_days:>15}")

    train_wr = best_result['win_rate']
    test_wr = test_result['win_rate']
    print(f"{'Win Rate':<20} {train_wr:>14.1f}% {test_wr:>14.1f}% {test_wr - train_wr:>+14.1f}%")

    train_dd = best_result['max_drawdown']
    test_dd = test_result['max_drawdown']
    print(f"{'Max Drawdown':<20} {train_dd:>14.1f}% {test_dd:>14.1f}% {test_dd - train_dd:>+14.1f}%")

    train_avg_mae = best_result.get('avg_mae', 0)
    test_avg_mae = test_result.get('avg_mae', 0)
    print(f"{'Avg MAE':<20} {train_avg_mae:>14.2f}% {test_avg_mae:>14.2f}% {test_avg_mae - train_avg_mae:>+14.2f}%")

    # --- Validation assessment ---
    print(f"\n{'=' * 80}")
    print("VALIDATION ASSESSMENT")
    print("=" * 80)

    warnings = []
    if test_return < 0:
        warnings.append("WARNING: Negative return on test data - possible overfitting")
        is_valid = False
    elif test_return < train_return * 0.3:
        warnings.append("WARNING: Test return < 30% of train return - possible overfitting")

    if test_wr < 50 and test_result['n_trades'] >= 5:
        warnings.append("WARNING: Win rate below 50% on test data")

    if test_dd > 30:
        warnings.append("WARNING: Max drawdown exceeds 30% on test data")

    if test_result['n_trades'] < 3:
        warnings.append("WARNING: Less than 3 trades on test data - insufficient validation")

    if warnings:
        for w in warnings:
            print(f"  {w}")
    else:
        print("  Strategy passed validation checks")
        print("  Consistent performance between train and test periods")

    # --- Build output ---
    output = {
        'ticker': ticker,
        'interval': primary_interval,
        'sub_interval': sub_interval,
        'period': period,
        'train_period': f"{train_dates[0].strftime('%Y-%m-%d %H:%M')} to {train_dates[-1].strftime('%Y-%m-%d %H:%M')}",
        'test_period': f"{test_dates[0].strftime('%Y-%m-%d %H:%M')} to {test_dates[-1].strftime('%Y-%m-%d %H:%M')}",
        'best_params': best_params,
        'train_result': {
            'total_return': train_return,
            'win_rate': train_wr,
            'n_trades': best_result.get('num_trades', best_result.get('n_trades', 0)),
            'max_drawdown': train_dd,
            'profit_factor': best_result.get('profit_factor', 0),
            'avg_mae': train_avg_mae,
            'max_mae': best_result.get('max_mae', 0),
        },
        'test_result': {
            'total_return': test_return,
            'win_rate': test_wr,
            'n_trades': test_result['n_trades'],
            'max_drawdown': test_dd,
            'profit_factor': test_result['profit_factor'],
            'avg_mae': test_avg_mae,
            'max_mae': test_result.get('max_mae', 0),
        },
        'buy_and_hold': {
            'train_return': round(train_bh_return, 4),
            'test_return': round(test_bh_return, 4),
            'train_daily_return': round(train_bh_daily, 6),
            'test_daily_return': round(test_bh_daily, 6),
        },
        'is_valid': is_valid,
        'warnings': warnings,
        'optimization_method': {
            'script': 'velocity_mtf_walkforward.py',
            'sub_interval': sub_interval,
            'n_trials': n_trials,
            'n_jobs': n_jobs,
            'metric': optimize_metric,
            'train_ratio': train_ratio,
            'optimized_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
    }

    return output


# --- CLI entry point ---

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Timeframe Walk-Forward Optimization for Velocity Strategies'
    )
    parser.add_argument('--ticker', type=str, required=True, help='Trading symbol (e.g., ES=F, BTC-USD)')
    parser.add_argument('--interval', type=str, default='15m', help='Primary interval for trade decisions (default: 15m)')
    parser.add_argument('--sub-interval', type=str, default='5m', help='Sub-interval for indicator computation (default: 5m)')
    parser.add_argument('--period', type=str, default='60d', help='Data period (default: 60d)')
    parser.add_argument('--n-trials', type=int, default=50000, help='Optuna optimization trials (default: 50000)')
    parser.add_argument('--n-jobs', type=int, default=32, help='Parallel workers (default: 32)')
    parser.add_argument('--metric', type=str, default='profit_factor', help='Optimization metric (default: profit_factor)')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/test split ratio (default: 0.8)')

    args = parser.parse_args()

    results = run_mtf_walk_forward(
        ticker=args.ticker,
        primary_interval=args.interval,
        sub_interval=args.sub_interval,
        period=args.period,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        train_ratio=args.train_ratio,
        optimize_metric=args.metric,
    )

    if results is None:
        print("Multi-timeframe walk-forward optimization failed!")
        sys.exit(1)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"velocity_mtf_wf_{args.ticker}_{args.period}_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=str)

    print(f"\nResults saved to: {results_file}")

    # Print optimized params summary
    print(f"\n{'=' * 80}")
    print("OPTIMIZED PARAMETERS FOR DEPLOYMENT")
    print("=" * 80)

    params = results['best_params']
    print(json.dumps({
        'oscillator_type': params.get('oscillator_type', 'composite'),
        'signal_type': params.get('signal_type'),
        'use_accel_exit': params.get('use_accel_exit'),
        'accel_exit_type': params.get('accel_exit_type'),
        'accel_exit_threshold': params.get('accel_exit_threshold'),
        'accel_exit_min_pnl': params.get('accel_exit_min_pnl'),
        'use_jerk_confirm': params.get('use_jerk_confirm'),
        'stop_loss_pct': params.get('stop_loss_pct'),
        'take_profit_pct': params.get('take_profit_pct'),
        'oversold_threshold': params.get('oversold_threshold'),
        'overbought_threshold': params.get('overbought_threshold'),
    }, indent=4))

    return results


if __name__ == '__main__':
    main()
