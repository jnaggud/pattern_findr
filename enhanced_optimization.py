"""
Enhanced Optimization with Novel V2 Indicator Filters

Tests combinations of signal types, thresholds, and V2 indicator filters
to find the best performing configuration.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_cache import fetch_and_cache
from oscillator_indicators import integrate_oscillator_indicators
from oscillator_predictor_testing_page import create_composite_oscillator, run_velocity_backtest


def run_enhanced_optimization(df: pd.DataFrame, n_trials: int = 200) -> dict:
    """
    Run enhanced optimization with V2 indicator filters.
    """
    # Define parameter grid
    signal_types = [
        'velocity_crossover_and_zone',
        'velocity_crossover_or_zone',
        'zone_only',
        'momentum',
        'any_reversal',
        'double_bottom',
        'divergence',
        'breakout'
    ]

    # Oversold/overbought threshold pairs (wider for intraday)
    threshold_pairs = [
        (-0.3, 0.3),
        (-0.4, 0.4),
        (-0.5, 0.5),
        (-0.6, 0.6),
        (-0.2, 0.4),  # asymmetric
        (-0.4, 0.2),  # asymmetric
    ]

    vel_smoothings = [1, 2, 3, 5]
    extreme_mults = [1.3, 1.5, 1.7, 2.0]

    # V2 filter combinations
    v2_filter_configs = [
        {'use_regime_filter': False, 'use_fragility_filter': False, 'use_entropy_filter': False},
        {'use_regime_filter': True, 'regime_threshold': 0.0, 'use_fragility_filter': False, 'use_entropy_filter': False},
        {'use_regime_filter': True, 'regime_threshold': -0.2, 'use_fragility_filter': False, 'use_entropy_filter': False},
        {'use_regime_filter': False, 'use_fragility_filter': True, 'fragility_threshold': 0.5, 'use_entropy_filter': False},
        {'use_regime_filter': False, 'use_fragility_filter': True, 'fragility_threshold': 0.3, 'use_entropy_filter': False},
        {'use_regime_filter': False, 'use_fragility_filter': False, 'use_entropy_filter': True, 'entropy_threshold': 0.7},
        {'use_regime_filter': False, 'use_fragility_filter': False, 'use_entropy_filter': True, 'entropy_threshold': 0.5},
        {'use_regime_filter': True, 'regime_threshold': 0.0, 'use_fragility_filter': True, 'fragility_threshold': 0.5, 'use_entropy_filter': False},
        {'use_regime_filter': True, 'regime_threshold': 0.0, 'use_fragility_filter': False, 'use_entropy_filter': True, 'entropy_threshold': 0.7},
    ]

    # Risk management
    stop_losses = [3.0, 5.0, 8.0]
    take_profits = [0.5, 1.0, 1.5, 2.0]

    results = []
    tested = 0

    print(f"\nRunning enhanced optimization ({n_trials} random samples from parameter space)...")
    print("=" * 80)

    # Generate random combinations
    np.random.seed(42)

    for _ in range(n_trials):
        sig_type = np.random.choice(signal_types)
        os_thresh, ob_thresh = threshold_pairs[np.random.randint(len(threshold_pairs))]
        vel_smooth = np.random.choice(vel_smoothings)
        ext_mult = np.random.choice(extreme_mults)
        v2_config = v2_filter_configs[np.random.randint(len(v2_filter_configs))]
        sl = np.random.choice(stop_losses)
        tp = np.random.choice(take_profits)

        params = {
            'signal_type': sig_type,
            'vel_smoothing': vel_smooth,
            'oversold_threshold': os_thresh,
            'overbought_threshold': ob_thresh,
            'extreme_zone_mult': ext_mult,
            'min_bars_between': 1,
            'require_accel': False,
            'stop_loss_pct': sl,
            'take_profit_pct': tp,
            'exit_on_opposite_signal': True,
            'exit_on_midline_cross': False,
            'rsi_filter': 'none',
            'use_macd_confirm': False,
            'use_bb_filter': False,
            'velocity_std_window': 10,
            'momentum_multiplier': 1.5,
            'double_bottom_lookback': 10,
            'divergence_lookback': 5,
        }
        params.update(v2_config)

        try:
            result = run_velocity_backtest(df, params)

            num_trades = result.get('num_trades', 0)
            if num_trades < 5:  # Skip configs with too few trades
                continue

            total_return = result.get('total_return', 0)
            win_rate = result.get('win_rate', 0)
            profit_factor = result.get('profit_factor', 0)
            sharpe = result.get('sharpe_ratio', 0)

            # Calculate risk-adjusted score
            risk_adjusted = total_return * (win_rate / 100) * min(profit_factor, 10) / 10

            results.append({
                'params': params.copy(),
                'num_trades': num_trades,
                'total_return': total_return,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe': sharpe,
                'risk_adjusted': risk_adjusted
            })

            tested += 1
            if tested % 20 == 0:
                print(f"  Tested {tested} valid configs...")

        except Exception as e:
            continue

    print(f"\n  Completed! {len(results)} valid configurations found.\n")

    # Sort by risk-adjusted score
    results.sort(key=lambda x: x['risk_adjusted'], reverse=True)

    return results


def format_results(results: list, top_n: int = 10) -> str:
    """Format top results as table."""
    lines = []
    lines.append("=" * 120)
    lines.append("TOP CONFIGURATIONS (sorted by risk-adjusted score)")
    lines.append("=" * 120)
    lines.append(f"{'#':<3} {'Signal Type':<28} {'Thresh':<12} {'Trades':>7} {'Win%':>7} {'Return%':>9} {'PF':>7} {'V2 Filters':<20}")
    lines.append("-" * 120)

    for i, r in enumerate(results[:top_n]):
        p = r['params']
        thresh = f"{p['oversold_threshold']:.1f}/{p['overbought_threshold']:.1f}"

        # V2 filter summary
        v2 = []
        if p.get('use_regime_filter'): v2.append('RSC')
        if p.get('use_fragility_filter'): v2.append('MFI2')
        if p.get('use_entropy_filter'): v2.append('SEI')
        v2_str = '+'.join(v2) if v2 else 'none'

        pf = r['profit_factor']
        pf_str = f"{pf:.2f}" if pf < 100 else "inf"

        lines.append(
            f"{i+1:<3} {p['signal_type']:<28} {thresh:<12} "
            f"{r['num_trades']:>7} {r['win_rate']:>6.1f}% {r['total_return']:>8.2f}% "
            f"{pf_str:>7} {v2_str:<20}"
        )

    lines.append("=" * 120)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Enhanced optimization with V2 filters')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--interval', type=str, default='15m', help='Data interval')
    parser.add_argument('--trials', type=int, default=200, help='Number of random trials')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Enhanced Optimization with V2 Indicator Filters")
    print(f"{'='*60}")
    print(f"Ticker: {args.ticker}")
    print(f"Interval: {args.interval}")
    print(f"Trials: {args.trials}")
    print(f"{'='*60}")

    # Load data
    print("\nLoading data...")
    days = 60 if args.interval in ['5m', '15m', '30m', '1h'] else 365
    df = fetch_and_cache(args.ticker, days=days, interval=args.interval)
    print(f"  Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Calculate indicators
    print("\nCalculating indicators (including Novel V2)...")
    df = integrate_oscillator_indicators(df, interval=args.interval)
    df = create_composite_oscillator(df)

    # Check V2 indicators (lowercase names from novel_indicators_v2.py)
    v2_cols = {'rsc_regime': 'RSC', 'mfi2': 'MFI2', 'sei': 'SEI'}
    found = [v2_cols[c] for c in v2_cols if c in df.columns]
    print(f"  V2 indicators available: {found}")

    # Run optimization
    results = run_enhanced_optimization(df, n_trials=args.trials)

    # Display results
    print(format_results(results, top_n=15))

    # Show best config details
    if results:
        best = results[0]
        print("\nBEST CONFIGURATION DETAILS:")
        print("-" * 60)
        for k, v in best['params'].items():
            print(f"  {k}: {v}")
        print("-" * 60)
        print(f"  Total Return: {best['total_return']:.2f}%")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  Profit Factor: {best['profit_factor']:.2f}")
        print(f"  Num Trades: {best['num_trades']}")
        print("-" * 60)

    # Save results
    output_file = args.output or f"enhanced_opt_{args.ticker}_{args.interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'ticker': args.ticker,
                'interval': args.interval,
                'trials': args.trials,
                'bars': len(df),
                'run_at': datetime.now().isoformat()
            },
            'top_results': results[:20]
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
