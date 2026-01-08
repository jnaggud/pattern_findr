"""
Automated Oscillator Performance Test

Tests the composite oscillator + velocity-based trading system.
Run this BEFORE and AFTER making changes to compare metrics.

Usage:
    python test_oscillator_performance.py
    python test_oscillator_performance.py --ticker SPY --years 5
    python test_oscillator_performance.py --save-baseline  # Save results as baseline
    python test_oscillator_performance.py --compare        # Compare to saved baseline
    python test_oscillator_performance.py --optimize       # Run Optuna optimization
    python test_oscillator_performance.py --optimize --trials 100  # More trials
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser(description='Oscillator Performance Test')
parser.add_argument('--ticker', default='SPY', help='Ticker to test')
parser.add_argument('--years', type=int, default=5, help='Years of historical data')
parser.add_argument('--save-baseline', action='store_true', help='Save results as baseline')
parser.add_argument('--compare', action='store_true', help='Compare to saved baseline')
parser.add_argument('--test-days', type=int, default=252, help='Days to test (default 1 year)')
parser.add_argument('--optimize', action='store_true', help='Run Optuna optimization for velocity params')
parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
parser.add_argument('--workers', type=int, default=-1, help='Parallel workers for Optuna (-1 = all cores)')
parser.add_argument('--no-volume', action='store_true', help='Disable volume indicators in composite oscillator')
args = parser.parse_args()

# Set workers to all cores if -1
if args.workers == -1:
    args.workers = os.cpu_count() or 8

BASELINE_FILE = 'oscillator_baseline_results.json'

print("=" * 80)
print("OSCILLATOR PERFORMANCE TEST")
print("=" * 80)
print(f"Ticker: {args.ticker}")
print(f"Historical Data: {args.years} years")
print(f"Test Period: {args.test_days} trading days")
print("=" * 80)

# Import from oscillator_predictor_page
try:
    from oscillator_predictor_page import (
        create_composite_oscillator,
        calculate_rsi,
        calculate_williams_r,
        calculate_cci,
        calculate_stochastic,
        calculate_roc,
        calculate_momentum,
        calculate_bb_position,
        calculate_adx_trend,
        calculate_mfi,
    )
    print("✓ Imported oscillator functions")
except ImportError as e:
    print(f"✗ Failed to import oscillator functions: {e}")
    exit(1)

import yfinance as yf

# Fetch data
print("\n" + "-" * 80)
print("FETCHING DATA...")
print("-" * 80)

end_date = datetime.now()
start_date = end_date - timedelta(days=args.years * 365)

print(f"Fetching {args.ticker} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
df = yf.download(args.ticker, start=start_date, end=end_date, progress=False)
df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
df.columns = [c.lower() for c in df.columns]

print(f"Total bars: {len(df)}")
print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
print(f"Current price: ${df['close'].iloc[-1]:.2f}")

# Create composite oscillator
print("\n" + "-" * 80)
print("CREATING COMPOSITE OSCILLATOR...")
print("-" * 80)

df = create_composite_oscillator(df)

# Remove volume indicators if --no-volume flag is set
if args.no_volume:
    print("⚠ Volume indicators DISABLED (--no-volume flag)")
    volume_cols = ['volume_velocity_norm', 'volume_momentum_norm']
    for col in volume_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Recalculate composite without volume indicators
    norm_cols = [col for col in df.columns if col.endswith('_norm') and col not in volume_cols]
    if norm_cols:
        df['composite_oscillator'] = df[norm_cols].mean(axis=1)
        df['composite_smooth'] = df['composite_oscillator'].rolling(window=5, center=False).mean()
        df['composite_smooth'] = df['composite_smooth'].bfill()
else:
    print("✓ Volume indicators ENABLED")

# Check what columns were created
osc_columns = [col for col in df.columns if 'norm' in col or 'composite' in col or 'osc' in col]
print(f"Oscillator columns created: {len(osc_columns)}")
for col in osc_columns[:10]:
    print(f"   - {col}")
if len(osc_columns) > 10:
    print(f"   ... and {len(osc_columns) - 10} more")

# ============================================================================
# BACKTEST FUNCTION (used for both normal run and Optuna optimization)
# ============================================================================

def run_backtest(df_input, vel_smoothing, oversold_threshold, overbought_threshold,
                 extreme_zone_mult, require_accel, test_days, initial_capital=100000):
    """
    Run backtest with given velocity parameters.
    Returns dict with metrics or None if no trades.
    """
    df_test = df_input.copy()

    # Use smoothed oscillator
    osc_col = 'composite_smooth' if 'composite_smooth' in df_test.columns else 'composite_oscillator'

    # Apply smoothing
    if vel_smoothing > 1:
        df_test['osc_smooth'] = df_test[osc_col].rolling(window=int(vel_smoothing), center=False).mean()
        df_test['osc_smooth'] = df_test['osc_smooth'].bfill()
    else:
        df_test['osc_smooth'] = df_test[osc_col]

    # Calculate velocity and acceleration
    df_test['velocity'] = df_test['osc_smooth'].diff()
    df_test['acceleration'] = df_test['velocity'].diff()
    df_test['velocity'] = df_test['velocity'].fillna(0)
    df_test['acceleration'] = df_test['acceleration'].fillna(0)

    # Detect velocity zero-crossings
    df_test['vel_cross_up'] = (df_test['velocity'] > 0) & (df_test['velocity'].shift(1) <= 0)
    df_test['vel_cross_down'] = (df_test['velocity'] < 0) & (df_test['velocity'].shift(1) >= 0)

    # Zone conditions
    osc_smooth = df_test['osc_smooth']
    acceleration = df_test['acceleration']

    in_oversold = osc_smooth < oversold_threshold
    in_overbought = osc_smooth > overbought_threshold

    # Generate signals (velocity crossover + zone)
    df_test['entry_long'] = (
        df_test['vel_cross_up'] &
        in_oversold &
        (acceleration > 0 if require_accel else True)
    )

    df_test['exit_long'] = (
        df_test['vel_cross_down'] &
        in_overbought
    )

    # Use last N days for testing
    test_start_idx = max(60, len(df_test) - test_days)
    test_df = df_test.iloc[test_start_idx:].copy()

    # Run backtest
    capital = initial_capital
    position = None
    entry_price = 0
    entry_date = None
    trades = []
    equity_curve = [initial_capital]

    for i in range(1, len(test_df)):
        current_price = test_df['close'].iloc[i]
        current_date = test_df.index[i]

        # Check for entry
        if position is None and test_df['entry_long'].iloc[i]:
            position = 'long'
            entry_price = current_price
            entry_date = current_date

        # Check for exit
        elif position == 'long' and test_df['exit_long'].iloc[i]:
            exit_price = current_price
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            pnl_dollar = capital * (exit_price - entry_price) / entry_price
            capital += pnl_dollar

            trades.append({
                'entry_date': entry_date,
                'exit_date': current_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'pnl_dollar': pnl_dollar,
                'hold_days': (current_date - entry_date).days
            })

            position = None
            entry_price = 0
            entry_date = None

        # Track equity
        if position == 'long':
            unrealized = capital * (current_price - entry_price) / entry_price
            equity_curve.append(capital + unrealized)
        else:
            equity_curve.append(capital)

    # Calculate metrics
    if not trades:
        return None

    trades_df = pd.DataFrame(trades)

    # Win/Loss metrics
    winning_trades = trades_df[trades_df['pnl_pct'] > 0]
    losing_trades = trades_df[trades_df['pnl_pct'] <= 0]

    win_rate = len(winning_trades) / len(trades_df) * 100
    avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0

    # Profit factor
    gross_profit = winning_trades['pnl_pct'].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades['pnl_pct'].sum()) if len(losing_trades) > 0 else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

    # Total return
    total_return = (capital - initial_capital) / initial_capital * 100

    # Max drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    # Sharpe ratio (annualized)
    if len(trades_df) > 1 and trades_df['pnl_pct'].std() > 0:
        returns = trades_df['pnl_pct'].values
        sharpe = np.sqrt(252 / max(trades_df['hold_days'].mean(), 1)) * (returns.mean() / returns.std())
    else:
        sharpe = 0

    return {
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'profit_factor': profit_factor,
        'total_return_pct': total_return,
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe,
        'avg_hold_days': trades_df['hold_days'].mean(),
        'final_capital': capital,
        'equity_curve': equity_curve,
        'trades': trades
    }

# ============================================================================
# OPTUNA OPTIMIZATION
# ============================================================================

if args.optimize:
    import optuna
    from optuna.samplers import TPESampler

    print("\n" + "=" * 80)
    print("OPTUNA OPTIMIZATION")
    print("=" * 80)
    print(f"Trials: {args.trials}")
    print(f"Workers: {args.workers}")
    print("=" * 80)

    def objective(trial):
        """Optuna objective function - maximize risk-adjusted return."""
        # Sample velocity parameters
        vel_smoothing = trial.suggest_int('vel_smoothing', 1, 10)
        oversold_threshold = trial.suggest_float('oversold_threshold', -0.6, -0.1)
        overbought_threshold = trial.suggest_float('overbought_threshold', 0.1, 0.6)
        extreme_zone_mult = trial.suggest_float('extreme_zone_mult', 1.0, 2.5)
        require_accel = trial.suggest_categorical('require_accel', [True, False])

        # Run backtest
        result = run_backtest(
            df, vel_smoothing, oversold_threshold, overbought_threshold,
            extreme_zone_mult, require_accel, args.test_days
        )

        if result is None or result['total_trades'] < 2:
            return -999  # Penalty for no trades

        # Objective: maximize combined score
        # Sharpe ratio + profit factor + penalize large drawdowns
        score = (
            result['sharpe_ratio'] * 0.4 +
            result['profit_factor'] * 0.3 +
            result['total_return_pct'] * 0.2 +
            (result['max_drawdown_pct'] + 30) * 0.1  # Less negative DD is better
        )

        # Bonus for more trades (statistical significance)
        if result['total_trades'] >= 5:
            score += 0.5
        if result['total_trades'] >= 10:
            score += 0.5

        return score

    # Create study
    sampler = TPESampler(seed=42, n_startup_trials=10)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='velocity_optimization'
    )

    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Run optimization
    print("\nOptimizing velocity parameters...")
    study.optimize(objective, n_trials=args.trials, n_jobs=args.workers, show_progress_bar=True)

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    print("\n" + "-" * 80)
    print("OPTIMIZATION RESULTS")
    print("-" * 80)
    print(f"\nBest Score: {best_value:.4f}")
    print(f"\nBest Parameters:")
    print(f"   vel_smoothing: {best_params['vel_smoothing']}")
    print(f"   oversold_threshold: {best_params['oversold_threshold']:.3f}")
    print(f"   overbought_threshold: {best_params['overbought_threshold']:.3f}")
    print(f"   extreme_zone_mult: {best_params['extreme_zone_mult']:.3f}")
    print(f"   require_accel: {best_params['require_accel']}")

    # Run final backtest with best params
    print("\n" + "-" * 80)
    print("RESULTS WITH OPTIMIZED PARAMETERS")
    print("-" * 80)

    best_result = run_backtest(
        df,
        best_params['vel_smoothing'],
        best_params['oversold_threshold'],
        best_params['overbought_threshold'],
        best_params['extreme_zone_mult'],
        best_params['require_accel'],
        args.test_days
    )

    if best_result:
        print(f"\nTrading Metrics:")
        print(f"   Total Trades: {best_result['total_trades']}")
        print(f"   Win Rate: {best_result['win_rate']:.2f}%")
        print(f"   Avg Win: {best_result['avg_win_pct']:.2f}%")
        print(f"   Avg Loss: {best_result['avg_loss_pct']:.2f}%")
        print(f"   Profit Factor: {best_result['profit_factor']:.2f}")

        print(f"\nPerformance Metrics:")
        print(f"   Total Return: {best_result['total_return_pct']:.2f}%")
        print(f"   Max Drawdown: {best_result['max_drawdown_pct']:.2f}%")
        print(f"   Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
        print(f"   Avg Hold Time: {best_result['avg_hold_days']:.1f} days")
        print(f"   Final Capital: ${best_result['final_capital']:,.2f}")

        # Save optimized params
        optimized_results = {
            'ticker': args.ticker,
            'test_period': f"{df.index[-args.test_days].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
            'test_days': args.test_days,
            'total_trades': best_result['total_trades'],
            'winning_trades': int(best_result['total_trades'] * best_result['win_rate'] / 100),
            'losing_trades': int(best_result['total_trades'] * (100 - best_result['win_rate']) / 100),
            'win_rate': round(best_result['win_rate'], 2),
            'avg_win_pct': round(best_result['avg_win_pct'], 2),
            'avg_loss_pct': round(best_result['avg_loss_pct'], 2),
            'profit_factor': round(best_result['profit_factor'], 2),
            'total_return_pct': round(best_result['total_return_pct'], 2),
            'max_drawdown_pct': round(best_result['max_drawdown_pct'], 2),
            'sharpe_ratio': round(best_result['sharpe_ratio'], 2),
            'avg_hold_days': round(best_result['avg_hold_days'], 1),
            'final_capital': round(best_result['final_capital'], 2),
            'optimized_params': best_params,
            'optuna_trials': args.trials,
            'optuna_best_score': round(best_value, 4),
            'timestamp': datetime.now().isoformat()
        }

        # Save to file
        optimized_file = 'oscillator_optimized_results.json'
        with open(optimized_file, 'w') as f:
            json.dump(optimized_results, f, indent=2)
        print(f"\n✓ Optimized results saved to {optimized_file}")

        # Compare to baseline if exists
        if os.path.exists(BASELINE_FILE):
            print("\n" + "-" * 80)
            print("COMPARISON: DEFAULT vs OPTIMIZED")
            print("-" * 80)

            with open(BASELINE_FILE, 'r') as f:
                baseline = json.load(f)

            print(f"\n{'Metric':<25} {'Default':>12} {'Optimized':>12} {'Change':>12}")
            print("-" * 65)

            compare_metrics = [
                ('win_rate', 'Win Rate (%)', True),
                ('profit_factor', 'Profit Factor', True),
                ('total_return_pct', 'Total Return (%)', True),
                ('max_drawdown_pct', 'Max Drawdown (%)', 'less_negative'),
                ('sharpe_ratio', 'Sharpe Ratio', True),
                ('total_trades', 'Total Trades', None),
            ]

            for key, label, higher_better in compare_metrics:
                baseline_val = baseline.get(key, 0)
                current_val = optimized_results.get(key, 0)
                change = current_val - baseline_val

                if higher_better is None:
                    change_str = f"{change:+.2f}"
                elif higher_better == 'less_negative':
                    if current_val > baseline_val:
                        change_str = f"{change:+.2f} ✓"
                    elif current_val < baseline_val:
                        change_str = f"{change:+.2f} ✗"
                    else:
                        change_str = f"{change:+.2f}"
                elif higher_better:
                    if change > 0:
                        change_str = f"{change:+.2f} ✓"
                    elif change < 0:
                        change_str = f"{change:+.2f} ✗"
                    else:
                        change_str = f"{change:+.2f}"
                else:
                    change_str = f"{change:+.2f}"

                print(f"{label:<25} {baseline_val:>12.2f} {current_val:>12.2f} {change_str:>12}")

            print("-" * 65)

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    exit(0)

# ============================================================================
# NORMAL RUN (non-optimized)
# ============================================================================

# Calculate velocity and acceleration
print("\n" + "-" * 80)
print("CALCULATING VELOCITY SIGNALS...")
print("-" * 80)

# Velocity parameters (default values)
vel_smoothing = 3
oversold_threshold = -0.3
overbought_threshold = 0.3
extreme_zone_mult = 1.5
require_accel = True

# Use smoothed oscillator
osc_col = 'composite_smooth' if 'composite_smooth' in df.columns else 'composite_oscillator'
print(f"Using oscillator column: {osc_col}")

# Apply smoothing
if vel_smoothing > 1:
    df['osc_smooth'] = df[osc_col].rolling(window=vel_smoothing, center=False).mean()
    df['osc_smooth'] = df['osc_smooth'].bfill()
else:
    df['osc_smooth'] = df[osc_col]

# Calculate velocity and acceleration
df['velocity'] = df['osc_smooth'].diff()
df['acceleration'] = df['velocity'].diff()
df['velocity'] = df['velocity'].fillna(0)
df['acceleration'] = df['acceleration'].fillna(0)

# Detect velocity zero-crossings
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

# Generate signals (velocity crossover + zone)
df['entry_long'] = (
    df['vel_cross_up'] &
    in_oversold &
    (acceleration > 0 if require_accel else True)
)

df['exit_long'] = (
    df['vel_cross_down'] &
    in_overbought
)

print(f"Entry signals generated: {df['entry_long'].sum()}")
print(f"Exit signals generated: {df['exit_long'].sum()}")

# Run backtest
print("\n" + "-" * 80)
print("RUNNING BACKTEST...")
print("-" * 80)

# Use last N days for testing
test_start_idx = max(60, len(df) - args.test_days)  # Need some warmup
test_df = df.iloc[test_start_idx:].copy()

print(f"Test period: {test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}")
print(f"Test bars: {len(test_df)}")

# Simple backtest
initial_capital = 100000
capital = initial_capital
position = None
entry_price = 0
entry_date = None
trades = []
equity_curve = [initial_capital]

for i in range(1, len(test_df)):
    current_price = test_df['close'].iloc[i]
    current_date = test_df.index[i]

    # Check for entry
    if position is None and test_df['entry_long'].iloc[i]:
        position = 'long'
        entry_price = current_price
        entry_date = current_date

    # Check for exit
    elif position == 'long' and test_df['exit_long'].iloc[i]:
        exit_price = current_price
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        pnl_dollar = capital * (exit_price - entry_price) / entry_price
        capital += pnl_dollar

        trades.append({
            'entry_date': entry_date,
            'exit_date': current_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_dollar,
            'hold_days': (current_date - entry_date).days
        })

        position = None
        entry_price = 0
        entry_date = None

    # Track equity
    if position == 'long':
        unrealized = capital * (current_price - entry_price) / entry_price
        equity_curve.append(capital + unrealized)
    else:
        equity_curve.append(capital)

# Calculate metrics
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

if trades:
    trades_df = pd.DataFrame(trades)

    # Win/Loss metrics
    winning_trades = trades_df[trades_df['pnl_pct'] > 0]
    losing_trades = trades_df[trades_df['pnl_pct'] <= 0]

    win_rate = len(winning_trades) / len(trades_df) * 100
    avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0

    # Profit factor
    gross_profit = winning_trades['pnl_pct'].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades['pnl_pct'].sum()) if len(losing_trades) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

    # Total return
    total_return = (capital - initial_capital) / initial_capital * 100

    # Max drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    # Sharpe ratio (annualized)
    if len(trades_df) > 1:
        returns = trades_df['pnl_pct'].values
        sharpe = np.sqrt(252 / trades_df['hold_days'].mean()) * (returns.mean() / returns.std()) if returns.std() > 0 else 0
    else:
        sharpe = 0

    # Average hold time
    avg_hold_days = trades_df['hold_days'].mean()

    results = {
        'ticker': args.ticker,
        'test_period': f"{test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}",
        'test_days': len(test_df),
        'total_trades': len(trades_df),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': round(win_rate, 2),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2),
        'total_return_pct': round(total_return, 2),
        'max_drawdown_pct': round(max_drawdown, 2),
        'sharpe_ratio': round(sharpe, 2),
        'avg_hold_days': round(avg_hold_days, 1),
        'final_capital': round(capital, 2),
        'timestamp': datetime.now().isoformat()
    }

    print(f"\nTrading Metrics:")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Win Rate: {results['win_rate']}%")
    print(f"   Avg Win: {results['avg_win_pct']}%")
    print(f"   Avg Loss: {results['avg_loss_pct']}%")
    print(f"   Profit Factor: {results['profit_factor']}")

    print(f"\nPerformance Metrics:")
    print(f"   Total Return: {results['total_return_pct']}%")
    print(f"   Max Drawdown: {results['max_drawdown_pct']}%")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']}")
    print(f"   Avg Hold Time: {results['avg_hold_days']} days")
    print(f"   Final Capital: ${results['final_capital']:,.2f}")

    # Save baseline if requested
    if args.save_baseline:
        with open(BASELINE_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Baseline saved to {BASELINE_FILE}")

    # Compare to baseline if requested
    if args.compare and os.path.exists(BASELINE_FILE):
        print("\n" + "-" * 80)
        print("COMPARISON TO BASELINE")
        print("-" * 80)

        with open(BASELINE_FILE, 'r') as f:
            baseline = json.load(f)

        print(f"\nBaseline from: {baseline.get('timestamp', 'Unknown')}")
        print(f"\n{'Metric':<25} {'Baseline':>12} {'Current':>12} {'Change':>12}")
        print("-" * 65)

        compare_metrics = [
            ('win_rate', 'Win Rate (%)', True),
            ('profit_factor', 'Profit Factor', True),
            ('total_return_pct', 'Total Return (%)', True),
            ('max_drawdown_pct', 'Max Drawdown (%)', 'less_negative'),  # Less negative is better
            ('sharpe_ratio', 'Sharpe Ratio', True),
            ('total_trades', 'Total Trades', None),  # Neutral
        ]

        improvements = 0
        regressions = 0

        for key, label, higher_better in compare_metrics:
            baseline_val = baseline.get(key, 0)
            current_val = results.get(key, 0)
            change = current_val - baseline_val

            if higher_better is None:
                change_str = f"{change:+.2f}"
            elif higher_better == 'less_negative':
                # For drawdown: less negative (closer to 0) is better
                # e.g., -13% is better than -16%
                if current_val > baseline_val:  # Less negative = improvement
                    change_str = f"{change:+.2f} ✓"
                    improvements += 1
                elif current_val < baseline_val:  # More negative = regression
                    change_str = f"{change:+.2f} ✗"
                    regressions += 1
                else:
                    change_str = f"{change:+.2f}"
            elif higher_better:
                if change > 0:
                    change_str = f"{change:+.2f} ✓"
                    improvements += 1
                elif change < 0:
                    change_str = f"{change:+.2f} ✗"
                    regressions += 1
                else:
                    change_str = f"{change:+.2f}"
            else:  # Lower is better (absolute value)
                if change < 0:
                    change_str = f"{change:+.2f} ✓"
                    improvements += 1
                elif change > 0:
                    change_str = f"{change:+.2f} ✗"
                    regressions += 1
                else:
                    change_str = f"{change:+.2f}"

            print(f"{label:<25} {baseline_val:>12.2f} {current_val:>12.2f} {change_str:>12}")

        print("-" * 65)
        print(f"\nSummary: {improvements} improvements, {regressions} regressions")

        if regressions == 0 and improvements > 0:
            print("✓ PASS - Changes improve the system without regressions")
        elif regressions > improvements:
            print("✗ FAIL - More regressions than improvements")
        else:
            print("⚠ MIXED - Some improvements, some regressions")

    elif args.compare:
        print(f"\n⚠ No baseline file found. Run with --save-baseline first.")

else:
    print("No trades generated during test period")
    results = {'error': 'No trades generated'}

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
