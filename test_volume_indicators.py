"""
Volume Indicators A/B Test

Proper comparison using the SAME optimization logic as Step 5c in Streamlit.
Tests whether adding volume_velocity_norm and volume_momentum_norm to the
composite oscillator improves performance.

Usage:
    python test_volume_indicators.py --ticker SPY --years 5 --trials 10000
    python test_volume_indicators.py --ticker SPY --years 5 --trials 50000 --workers 32
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser(description='Volume Indicators A/B Test')
parser.add_argument('--ticker', default='SPY', help='Ticker to test')
parser.add_argument('--years', type=int, default=5, help='Years of historical data')
parser.add_argument('--test-pct', type=float, default=0.2, help='Test set percentage (default 20%)')
parser.add_argument('--trials', type=int, default=10000, help='Optuna trials per test')
parser.add_argument('--workers', type=int, default=-1, help='Parallel workers (-1 = all cores)')
args = parser.parse_args()

if args.workers == -1:
    args.workers = os.cpu_count() or 8

print("=" * 80)
print("VOLUME INDICATORS A/B TEST")
print("=" * 80)
print(f"Ticker: {args.ticker}")
print(f"Historical Data: {args.years} years")
print(f"Test Set: {args.test_pct*100:.0f}%")
print(f"Trials per test: {args.trials:,}")
print(f"Workers: {args.workers}")
print("=" * 80)

# Import yfinance
import yfinance as yf

# Fetch data
print("\n" + "-" * 80)
print("FETCHING DATA...")
print("-" * 80)

end_date = datetime.now()
start_date = end_date - timedelta(days=args.years * 365)

print(f"Fetching {args.ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
df = yf.download(args.ticker, start=start_date, end=end_date, progress=False)
df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
df.columns = [c.lower() for c in df.columns]

print(f"Total bars: {len(df)}")
print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

# Split into train/test
test_size = int(len(df) * args.test_pct)
train_df = df.iloc[:-test_size].copy()
test_df = df.iloc[-test_size:].copy()

print(f"Train: {len(train_df)} bars ({train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')})")
print(f"Test: {len(test_df)} bars ({test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')})")

# ============================================================================
# COMPOSITE OSCILLATOR CREATION (matches oscillator_predictor_page.py)
# ============================================================================

def create_composite_oscillator(df_input, include_volume=True):
    """Create composite oscillator with optional volume indicators."""
    df = df_input.copy()
    components = {}

    # RSI variants (normalized to -1 to +1)
    for period in [14, 7, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        suffix = '' if period == 14 else f'_{period}'
        components[f'rsi_norm{suffix}'] = (rsi - 50) / 50

    # Williams %R (already -100 to 0, normalize to -1 to +1)
    period = 14
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    willr = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-10)
    components['willr_norm'] = (willr + 50) / 50

    # CCI (typically -100 to +100, clip and normalize)
    period = 20
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mean_dev = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma_tp) / (0.015 * mean_dev + 1e-10)
    components['cci_norm'] = cci.clip(-200, 200) / 200

    # Stochastic K and D
    period = 14
    lowest_low = df['low'].rolling(window=period).min()
    highest_high = df['high'].rolling(window=period).max()
    stoch_k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
    stoch_d = stoch_k.rolling(window=3).mean()
    components['stoch_k_norm'] = (stoch_k - 50) / 50
    components['stoch_d_norm'] = (stoch_d - 50) / 50

    # ROC variants
    for period in [10, 5]:
        roc = df['close'].pct_change(periods=period) * 100
        suffix = '' if period == 10 else f'_{period}'
        components[f'roc_norm{suffix}'] = roc.clip(-10, 10) / 10

    # Momentum
    period = 10
    mom = df['close'] - df['close'].shift(period)
    mom_std = mom.rolling(60).std()
    components['momentum_norm'] = (mom / (mom_std + 0.01)).clip(-3, 3) / 3

    # Bollinger Band Position
    period = 20
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    bb_pos = (df['close'] - lower) / (upper - lower + 1e-10)
    components['bb_position_norm'] = (bb_pos - 0.5) * 2

    # MFI
    if 'volume' in df.columns and df['volume'].sum() > 0:
        period = 14
        tp = (df['high'] + df['low'] + df['close']) / 3
        raw_mf = tp * df['volume']
        pos_mf = raw_mf.where(tp > tp.shift(1), 0).rolling(window=period).sum()
        neg_mf = raw_mf.where(tp < tp.shift(1), 0).rolling(window=period).sum()
        mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
        components['mfi_norm'] = (mfi - 50) / 50

    # ADX Trend (normalized)
    period = 14
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    plus_dm = df['high'].diff().where(lambda x: (x > 0) & (x > -df['low'].diff()), 0)
    minus_dm = (-df['low'].diff()).where(lambda x: (x > 0) & (x > df['high'].diff()), 0)
    plus_di = 100 * plus_dm.rolling(window=period).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.rolling(window=period).mean() / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    adx_direction = (plus_di - minus_di) / 50
    components['adx_trend_norm'] = (adx_direction * (adx / 100)).clip(-1, 1)

    # VOLUME INDICATORS (optional)
    if include_volume and 'volume' in df.columns and df['volume'].sum() > 0:
        volume = df['volume']
        volume_sma = volume.rolling(20).mean()
        volume_rel = volume / volume_sma

        # Volume Velocity (1st derivative)
        volume_velocity_raw = volume_rel.rolling(5).mean().diff()
        vol_vel_std = volume_velocity_raw.rolling(60).std()
        components['volume_velocity_norm'] = (volume_velocity_raw / (vol_vel_std + 0.01)).clip(-3, 3) / 3

        # Volume Momentum (rate of change)
        volume_momentum_raw = volume.pct_change(5)
        vol_mom_std = volume_momentum_raw.rolling(60).std()
        components['volume_momentum_norm'] = (volume_momentum_raw / (vol_mom_std + 0.01)).clip(-3, 3) / 3

    # Add components to df
    for name, values in components.items():
        df[name] = values

    # Create composite (equal-weighted mean of all normalized components)
    norm_cols = [col for col in df.columns if col.endswith('_norm')]
    df['composite_oscillator'] = df[norm_cols].mean(axis=1)
    df['composite_smooth'] = df['composite_oscillator'].rolling(window=5, center=False).mean()
    df['composite_smooth'] = df['composite_smooth'].bfill()

    return df, norm_cols

# ============================================================================
# BACKTEST FUNCTION (matches velocity_live_trader.py logic)
# ============================================================================

def run_velocity_backtest(df_input, params):
    """Run backtest with full velocity parameters (matches production)."""
    df = df_input.copy()

    # Parameters
    signal_type = params.get('signal_type', 'velocity_crossover_and_zone')
    vel_smoothing = params.get('vel_smoothing', 3)
    oversold_threshold = params.get('oversold_threshold', -0.3)
    overbought_threshold = params.get('overbought_threshold', 0.3)
    stop_loss_pct = params.get('stop_loss_pct', 0)
    take_profit_pct = params.get('take_profit_pct', 0)
    min_bars_between = params.get('min_bars_between', 1)
    require_accel = params.get('require_accel', True)
    extreme_zone_mult = params.get('extreme_zone_mult', 1.5)
    exit_on_opposite = params.get('exit_on_opposite_signal', True)
    exit_on_midline = params.get('exit_on_midline_cross', False)

    # Oscillator column
    osc_col = 'composite_smooth' if 'composite_smooth' in df.columns else 'composite_oscillator'

    # Apply smoothing
    if vel_smoothing > 1:
        df['osc_smooth'] = df[osc_col].rolling(window=vel_smoothing, center=False).mean()
        df['osc_smooth'] = df['osc_smooth'].bfill()
    else:
        df['osc_smooth'] = df[osc_col]

    # Velocity and acceleration
    df['velocity'] = df['osc_smooth'].diff().fillna(0)
    df['acceleration'] = df['velocity'].diff().fillna(0)

    # Velocity crossings
    df['vel_cross_up'] = (df['velocity'] > 0) & (df['velocity'].shift(1) <= 0)
    df['vel_cross_down'] = (df['velocity'] < 0) & (df['velocity'].shift(1) >= 0)

    # Zone conditions
    osc_smooth = df['osc_smooth']
    velocity = df['velocity']
    acceleration = df['acceleration']

    in_oversold = osc_smooth < oversold_threshold
    in_overbought = osc_smooth > overbought_threshold
    extreme_oversold = osc_smooth < (oversold_threshold * extreme_zone_mult)
    extreme_overbought = osc_smooth > (overbought_threshold * extreme_zone_mult)

    # Strong momentum
    vel_std = velocity.rolling(10, min_periods=1).std().fillna(velocity.std())
    strong_momentum_up = velocity > vel_std * 1.5
    strong_momentum_down = velocity < -vel_std * 1.5

    # Signal types
    if signal_type == 'velocity_crossover_and_zone':
        raw_buy = df['vel_cross_up'] & in_oversold
        raw_sell = df['vel_cross_down'] & in_overbought
    elif signal_type == 'velocity_crossover_or_zone':
        raw_buy = df['vel_cross_up'] | extreme_oversold
        raw_sell = df['vel_cross_down'] | extreme_overbought
    elif signal_type == 'zone_only':
        raw_buy = extreme_oversold & (velocity > 0)
        raw_sell = extreme_overbought & (velocity < 0)
    elif signal_type == 'momentum':
        raw_buy = strong_momentum_up & (osc_smooth < 0)
        raw_sell = strong_momentum_down & (osc_smooth > 0)
    else:
        raw_buy = df['vel_cross_up'] & in_oversold
        raw_sell = df['vel_cross_down'] & in_overbought

    # Acceleration filter
    if require_accel:
        raw_buy = raw_buy & (acceleration > 0)
        raw_sell = raw_sell & (acceleration < 0)

    df['buy_signal'] = raw_buy
    df['sell_signal'] = raw_sell

    # Run backtest
    position = 0
    entry_price = 0
    entry_bar = 0
    trades = []
    equity = [100000]
    capital = 100000

    for i in range(len(df)):
        price = df['close'].iloc[i]
        osc = df['osc_smooth'].iloc[i]
        bars_since = i - entry_bar

        # Exit logic
        if position == 1 and entry_price > 0:
            pnl_pct = ((price - entry_price) / entry_price) * 100
            exit_reason = None

            if stop_loss_pct > 0 and pnl_pct <= -stop_loss_pct:
                exit_reason = "Stop Loss"
            elif take_profit_pct > 0 and pnl_pct >= take_profit_pct:
                exit_reason = "Take Profit"
            elif exit_on_midline and osc > 0:
                exit_reason = "Midline Cross"
            elif exit_on_opposite and df['sell_signal'].iloc[i] and bars_since >= min_bars_between:
                exit_reason = "Opposite Signal"

            if exit_reason:
                capital *= (1 + pnl_pct / 100)
                trades.append({'pnl': pnl_pct, 'reason': exit_reason})
                position = 0
                entry_price = 0

        # Entry logic
        if position == 0 and df['buy_signal'].iloc[i] and (i - entry_bar) >= min_bars_between:
            position = 1
            entry_price = price
            entry_bar = i

        # Track equity
        if position == 1:
            unrealized = capital * ((price - entry_price) / entry_price)
            equity.append(capital + unrealized)
        else:
            equity.append(capital)

    # Calculate metrics
    if not trades:
        return {'total_return': 0, 'win_rate': 0, 'profit_factor': 0, 'sharpe_ratio': 0, 'num_trades': 0}

    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if t['pnl'] <= 0]

    total_return = (capital - 100000) / 100000 * 100
    win_rate = len(wins) / len(trades) * 100

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

    # Sharpe
    equity_series = pd.Series(equity)
    returns = equity_series.pct_change().dropna()
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10) if len(returns) > 1 else 0

    # Max drawdown
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades)
    }

# ============================================================================
# OPTUNA OPTIMIZATION (using joblib for true parallelization)
# ============================================================================

import tempfile
import joblib as jl
from joblib import Parallel, delayed


def _run_single_study(data_path, n_trials, seed, worker_id):
    """
    Run a single Optuna study (called by joblib worker).
    Each worker runs independently with its own study.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Load data from disk (shared across workers)
    data = jl.load(data_path)
    df_prepared = data['df_prepared']

    def objective(trial):
        params = {
            'signal_type': trial.suggest_categorical('signal_type', [
                'velocity_crossover_and_zone', 'velocity_crossover_or_zone', 'zone_only', 'momentum'
            ]),
            'vel_smoothing': trial.suggest_int('vel_smoothing', 1, 15),
            'oversold_threshold': trial.suggest_float('oversold_threshold', -0.6, -0.02),
            'overbought_threshold': trial.suggest_float('overbought_threshold', 0.02, 0.6),
            'stop_loss_pct': trial.suggest_float('stop_loss_pct', 0.0, 10.0),
            'take_profit_pct': trial.suggest_float('take_profit_pct', 0.0, 20.0),
            'min_bars_between': trial.suggest_int('min_bars_between', 1, 15),
            'require_accel': trial.suggest_categorical('require_accel', [True, False]),
            'extreme_zone_mult': trial.suggest_float('extreme_zone_mult', 1.1, 2.5),
            'exit_on_opposite_signal': trial.suggest_categorical('exit_on_opposite_signal', [True, False]),
            'exit_on_midline_cross': trial.suggest_categorical('exit_on_midline_cross', [True, False]),
        }

        result = run_velocity_backtest(df_prepared, params)

        if result['num_trades'] < 2:
            return float('-inf')

        # Composite score (same as Step 5c)
        score = result['total_return']
        if result['profit_factor'] > 1:
            score += result['profit_factor'] * 2
        if result['sharpe_ratio'] > 0:
            score += result['sharpe_ratio'] * 5

        # Store result for later retrieval
        trial.set_user_attr('result', {**result, 'params': params})

        return score

    # Create independent study with unique seed
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(50, n_trials // 5))
    )

    # Progress logging
    log_interval = max(50, n_trials // 10)

    def progress_callback(study, trial):
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if n_complete % log_interval == 0 or n_complete == n_trials:
            best_val = study.best_value if study.best_trial else 0
            print(f"[Worker {worker_id:2d}] {n_complete:,}/{n_trials:,} trials | Best score: {best_val:.2f}", flush=True)

    print(f"[Worker {worker_id:2d}] Starting {n_trials:,} trials (seed={seed})...", flush=True)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1, callbacks=[progress_callback])

    # Extract results
    results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value != float('-inf'):
            result = trial.user_attrs.get('result')
            if result:
                results.append(result)

    best_val = study.best_value if study.best_trial else 0
    print(f"[Worker {worker_id:2d}] ✓ Done! {len(results):,} valid results, best score: {best_val:.2f}", flush=True)

    return results


def run_optimization(test_df, include_volume, n_trials, n_workers, label):
    """Run Optuna optimization with true multi-process parallelization."""

    # Prepare data
    df_prepared, norm_cols = create_composite_oscillator(test_df, include_volume=include_volume)

    print(f"\n{'-'*60}")
    print(f"{label}")
    print(f"{'-'*60}")
    print(f"Oscillator components: {len(norm_cols)}")
    if include_volume:
        print("   (includes volume_velocity_norm, volume_momentum_norm)")

    # Save data to temp file for worker processes
    fd, data_path = tempfile.mkstemp(suffix='.joblib', prefix='volume_test_')
    os.close(fd)
    jl.dump({'df_prepared': df_prepared}, data_path)

    # Divide trials among workers
    trials_per_worker = n_trials // n_workers
    remainder = n_trials % n_workers

    print(f"\nRunning {n_trials:,} total trials:")
    print(f"   {n_workers} parallel workers x {trials_per_worker:,} trials each")
    if remainder > 0:
        print(f"   (+{remainder} extra trials distributed)")

    start_time = time.time()

    try:
        # Run parallel studies using joblib with loky backend
        results_lists = Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
            delayed(_run_single_study)(
                data_path,
                trials_per_worker + (1 if i < remainder else 0),  # distribute remainder
                seed=42 + i,
                worker_id=i
            )
            for i in range(n_workers)
        )

        # Merge all results
        all_results = []
        for result_list in results_lists:
            all_results.extend(result_list)

    finally:
        # Clean up temp file
        if os.path.exists(data_path):
            os.remove(data_path)

    elapsed = time.time() - start_time

    if not all_results:
        print(f"\n⚠ No valid results found!")
        return {'total_return': 0, 'win_rate': 0, 'profit_factor': 0, 'sharpe_ratio': 0, 'num_trades': 0}, {}

    # Find best result
    best = max(all_results, key=lambda x: x['total_return'])
    best_params = best['params']
    best_result = {k: v for k, v in best.items() if k != 'params'}

    print(f"\n{'='*60}")
    print(f"Completed in {elapsed:.1f}s ({n_trials/elapsed:.0f} trials/sec)")
    print(f"Valid results: {len(all_results):,}")
    print(f"\nBest Parameters:")
    print(f"   signal_type: {best_params['signal_type']}")
    print(f"   vel_smoothing: {best_params['vel_smoothing']}")
    print(f"   oversold: {best_params['oversold_threshold']:.3f}")
    print(f"   overbought: {best_params['overbought_threshold']:.3f}")
    print(f"   stop_loss: {best_params['stop_loss_pct']:.1f}%")
    print(f"   take_profit: {best_params['take_profit_pct']:.1f}%")
    print(f"   require_accel: {best_params['require_accel']}")

    print(f"\nResults:")
    print(f"   Total Return: {best_result['total_return']:.2f}%")
    print(f"   Win Rate: {best_result['win_rate']:.2f}%")
    print(f"   Profit Factor: {best_result['profit_factor']:.2f}")
    print(f"   Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {best_result['max_drawdown']:.2f}%")
    print(f"   Trades: {best_result['num_trades']}")

    return best_result, best_params

# ============================================================================
# RUN A/B TEST
# ============================================================================

print("\n" + "=" * 80)
print("RUNNING A/B TEST: WITHOUT vs WITH VOLUME INDICATORS")
print("=" * 80)

# Test WITHOUT volume indicators
result_without, params_without = run_optimization(
    test_df, include_volume=False, n_trials=args.trials, n_workers=args.workers,
    label="TEST A: WITHOUT VOLUME INDICATORS"
)

# Test WITH volume indicators
result_with, params_with = run_optimization(
    test_df, include_volume=True, n_trials=args.trials, n_workers=args.workers,
    label="TEST B: WITH VOLUME INDICATORS"
)

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("A/B TEST RESULTS")
print("=" * 80)

print(f"\n{'Metric':<20} {'WITHOUT Volume':>15} {'WITH Volume':>15} {'Change':>15}")
print("-" * 70)

metrics = [
    ('total_return', 'Total Return (%)', True),
    ('win_rate', 'Win Rate (%)', True),
    ('profit_factor', 'Profit Factor', True),
    ('sharpe_ratio', 'Sharpe Ratio', True),
    ('max_drawdown', 'Max Drawdown (%)', 'less_negative'),
    ('num_trades', 'Trades', None),
]

improvements = 0
regressions = 0

for key, label, higher_better in metrics:
    val_without = result_without.get(key, 0)
    val_with = result_with.get(key, 0)
    change = val_with - val_without

    if higher_better is None:
        indicator = ""
    elif higher_better == 'less_negative':
        indicator = " ✓" if val_with > val_without else (" ✗" if val_with < val_without else "")
        if val_with > val_without: improvements += 1
        elif val_with < val_without: regressions += 1
    elif higher_better:
        indicator = " ✓" if change > 0 else (" ✗" if change < 0 else "")
        if change > 0: improvements += 1
        elif change < 0: regressions += 1

    print(f"{label:<20} {val_without:>15.2f} {val_with:>15.2f} {change:>+14.2f}{indicator}")

print("-" * 70)
print(f"\nSummary: {improvements} improvements, {regressions} regressions")

if improvements > regressions:
    print("\n✓ VOLUME INDICATORS ADD VALUE - Consider adding to production")
elif regressions > improvements:
    print("\n✗ VOLUME INDICATORS DO NOT HELP - Do not add to production")
else:
    print("\n⚠ MIXED RESULTS - Further testing recommended")

# Save results
results_file = f"volume_ab_test_{args.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(results_file, 'w') as f:
    json.dump({
        'ticker': args.ticker,
        'test_period': f"{test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}",
        'trials_per_test': args.trials,
        'without_volume': {**result_without, 'params': params_without},
        'with_volume': {**result_with, 'params': params_with},
        'improvements': improvements,
        'regressions': regressions
    }, f, indent=2, default=str)

print(f"\n✓ Results saved to {results_file}")
print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
