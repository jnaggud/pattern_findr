#!/usr/bin/env python3
"""
Create Strategy Bundle from Validation Results

This script creates a deployable strategy bundle from validation results
(either walk-forward or simple split validation).

Usage:
    python create_strategy_bundle.py --results velocity_split_results_ES=F_60d_20260130_125826.json
    python create_strategy_bundle.py --results velocity_wf_results_BTC-USD_2y_20260130_084112.json --name my_btc_strategy
    python create_strategy_bundle.py --results results.json --webhook https://discord.com/api/webhooks/...
"""

import argparse
import json
import os
import sys
from datetime import datetime


def create_bundle_from_results(results_path: str, bundle_name: str = None,
                                discord_webhook: str = None) -> str:
    """
    Create a strategy bundle from validation results.

    Args:
        results_path: Path to validation results JSON file
        bundle_name: Optional custom bundle name
        discord_webhook: Optional Discord webhook URL for alerts

    Returns:
        Path to created bundle directory
    """

    # Load validation results
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Extract key info
    ticker = results.get('ticker', 'UNKNOWN')
    interval = results.get('interval', '1d')
    period = results.get('period', 'unknown')
    validation_type = results.get('validation_type', 'walk_forward')
    best_params = results.get('best_params', {})
    train_result = results.get('train_result', {})
    test_result = results.get('test_result', {})
    optimization_method = results.get('optimization_method', {})

    # Generate bundle name if not provided
    if not bundle_name:
        signal_type = best_params.get('signal_type', 'velocity')
        # Clean up ticker for filename
        ticker_clean = ticker.replace('=', '').replace('-', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bundle_name = f"velocity_{ticker_clean}_{interval}_{signal_type}_{timestamp}"

    # Build the strategy config
    config = {
        # Core identifiers
        'strategy_type': 'velocity',
        'strategy_name': bundle_name,
        'bundle_name': bundle_name,
        'ticker': ticker,
        'interval': interval,
        'optimization_period': period,
        'data_period_days': results.get('train_bars', 0) + results.get('test_bars', 0),
        'optimization_bars': results.get('train_bars', 0) + results.get('test_bars', 0),

        # Signal parameters
        'signal_type': best_params.get('signal_type', 'velocity_crossover_or_zone'),
        'vel_smoothing': best_params.get('vel_smoothing', 1),
        'extreme_zone_mult': best_params.get('extreme_zone_mult', 2.0),
        'min_bars_between': best_params.get('min_bars_between', 1),
        'require_accel': best_params.get('require_accel', False),
        'vel_threshold': best_params.get('vel_threshold', 0.0),
        'accel_threshold': best_params.get('accel_threshold', 0.0),
        'velocity_std_window': best_params.get('velocity_std_window', 10),
        'momentum_multiplier': best_params.get('momentum_multiplier', 1.5),
        'double_bottom_lookback': best_params.get('double_bottom_lookback', 10),
        'divergence_lookback': best_params.get('divergence_lookback', 5),

        # Thresholds
        'oversold_threshold': best_params.get('oversold_threshold', -0.3),
        'overbought_threshold': best_params.get('overbought_threshold', 0.3),

        # Exit parameters
        'stop_loss_pct': best_params.get('stop_loss_pct', 5.0),
        'take_profit_pct': best_params.get('take_profit_pct', 10.0),
        'exit_on_opposite_signal': best_params.get('exit_on_opposite_signal', True),
        'exit_on_midline_cross': best_params.get('exit_on_midline_cross', False),
        'min_hold_bars': best_params.get('min_hold_bars', 1),

        # Acceleration exit parameters
        'use_accel_exit': best_params.get('use_accel_exit', False),
        'accel_exit_type': best_params.get('accel_exit_type', 'sign_reversal'),
        'accel_exit_threshold': best_params.get('accel_exit_threshold', 0.0),
        'accel_exit_min_pnl': best_params.get('accel_exit_min_pnl', 0.5),
        'accel_exit_lookback': best_params.get('accel_exit_lookback', 1),
        'use_jerk_confirm': best_params.get('use_jerk_confirm', False),
        'jerk_confirm_threshold': best_params.get('jerk_confirm_threshold', 0.0),

        # Trailing stop parameters (v7+)
        'use_trailing_stop': best_params.get('use_trailing_stop', False),
        'trailing_stop_pct': best_params.get('trailing_stop_pct', 1.0),
        'trailing_stop_activation_pct': best_params.get('trailing_stop_activation_pct', 0.3),
        # Break-even stop parameters (v7+)
        'use_breakeven_stop': best_params.get('use_breakeven_stop', False),
        'breakeven_trigger_pct': best_params.get('breakeven_trigger_pct', 0.3),
        'breakeven_offset_pct': best_params.get('breakeven_offset_pct', 0.05),

        # Filter parameters
        'rsi_filter': best_params.get('rsi_filter', 'none'),
        'rsi_period': best_params.get('rsi_period', 14),
        'rsi_oversold': best_params.get('rsi_oversold', 30),
        'rsi_overbought': best_params.get('rsi_overbought', 70),
        'use_macd_confirm': best_params.get('use_macd_confirm', False),
        'use_bb_filter': best_params.get('use_bb_filter', False),

        # Regime filters (if present)
        'use_regime_filter': best_params.get('use_regime_filter', False),
        'regime_threshold': best_params.get('regime_threshold', 0.0),
        'use_fragility_filter': best_params.get('use_fragility_filter', False),
        'fragility_threshold': best_params.get('fragility_threshold', 0.5),
        'use_entropy_filter': best_params.get('use_entropy_filter', False),
        'entropy_threshold': best_params.get('entropy_threshold', 0.7),
        'use_vol_regime_filter': best_params.get('use_vol_regime_filter', False),
        'vol_regime_percentile_threshold': best_params.get('vol_regime_percentile_threshold', 0.25),

        # Oscillator
        'oscillator_type': best_params.get('oscillator_type', 'composite'),

        # Discord webhook (optional)
        'discord_webhook': discord_webhook or '',

        # Timestamps
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'deployed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Add validation results based on type
    if validation_type == 'simple_split':
        config.update({
            'split_validated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'split_train_return': train_result.get('total_return', 0),
            'split_test_return': test_result.get('total_return', 0),
            'split_train_win_rate': train_result.get('win_rate', 0),
            'split_test_win_rate': test_result.get('win_rate', 0),
            'split_train_trades': train_result.get('n_trades', 0),
            'split_test_trades': test_result.get('n_trades', 0),
            'split_train_max_dd': train_result.get('max_drawdown', 0),
            'split_test_max_dd': test_result.get('max_drawdown', 0),
            'split_train_profit_factor': train_result.get('profit_factor', 0),
            'split_test_profit_factor': test_result.get('profit_factor', 0),
            'split_train_period': results.get('train_period', ''),
            'split_test_period': results.get('test_period', ''),
        })
    else:
        # Walk-forward validation format
        config.update({
            'wf_validated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'wf_train_return': train_result.get('total_return', 0),
            'wf_test_return': test_result.get('total_return', 0),
            'wf_train_win_rate': train_result.get('win_rate', 0),
            'wf_test_win_rate': test_result.get('win_rate', 0),
            'wf_train_trades': train_result.get('n_trades', 0),
            'wf_test_trades': test_result.get('n_trades', 0),
            'wf_train_max_dd': train_result.get('max_drawdown', 0),
            'wf_test_max_dd': test_result.get('max_drawdown', 0),
            'wf_train_profit_factor': train_result.get('profit_factor', 0),
            'wf_test_profit_factor': test_result.get('profit_factor', 0),
            'wf_train_period': results.get('train_period', ''),
            'wf_test_period': results.get('test_period', ''),
        })

    # Add optimization method tracking
    config['optimization_method'] = {
        'script': optimization_method.get('script', 'unknown'),
        'n_trials': optimization_method.get('n_trials', 0),
        'n_jobs': optimization_method.get('n_jobs', 0),
        'metric': optimization_method.get('metric', 'total_return'),
        'train_ratio': optimization_method.get('train_ratio', 0.8),
        'optimized_at': optimization_method.get('optimized_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        'source_file': os.path.basename(results_path)
    }

    # Create bundle directory
    bundle_path = os.path.join('velocity_strategies', bundle_name)

    # AUTOMATIC BACKUP: If bundle already exists, backup before overwriting
    if os.path.exists(bundle_path):
        try:
            from velocity_trading.core.backup import backup_strategy_bundle
            print(f"\n   [Auto-Backup] Existing bundle found, creating backup...")
            backup_strategy_bundle(bundle_path, reason="pre_update")
        except Exception as e:
            print(f"   ⚠️  Backup warning: {e}")

    os.makedirs(bundle_path, exist_ok=True)

    # Save config
    config_path = os.path.join(bundle_path, 'velocity_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Save backtest_results.json from OOS test trades (if available)
    # This allows the intraday trader to import exact trades instead of
    # re-running a fresh backtest that produces different stats.
    test_trades = results.get('test_trades', [])
    if test_trades:
        entries = []
        exits = []
        for t in test_trades:
            entries.append({
                'date': t['entry_date'],
                'price': t['entry_price'],
                'position': 'long'
            })
            exits.append({
                'date': t['exit_date'],
                'price': t['exit_price'],
                'pnl': t['pnl_pct'],
                'reason': t.get('exit_reason', 'signal'),
                'entry_date': t['entry_date'],
                'entry_price': t['entry_price']
            })

        backtest_results_path = os.path.join(bundle_path, 'backtest_results.json')
        with open(backtest_results_path, 'w') as f:
            json.dump({'entries': entries, 'exits': exits}, f, indent=4)
        print(f"   Saved {len(exits)} OOS trades to backtest_results.json")

    return bundle_path, config


def print_bundle_summary(config: dict, bundle_path: str):
    """Print a summary of the created bundle."""

    print("\n" + "=" * 70)
    print("STRATEGY BUNDLE CREATED")
    print("=" * 70)
    print(f"\nBundle Path: {bundle_path}")
    print(f"Config File: {bundle_path}/velocity_config.json")

    print(f"\n{'─' * 70}")
    print("STRATEGY DETAILS")
    print(f"{'─' * 70}")
    print(f"  Ticker:        {config['ticker']}")
    print(f"  Interval:      {config['interval']}")
    print(f"  Signal Type:   {config['signal_type']}")
    print(f"  Oscillator:    {config.get('oscillator_type', 'composite')}")

    print(f"\n{'─' * 70}")
    print("KEY PARAMETERS")
    print(f"{'─' * 70}")
    print(f"  Stop Loss:     {config['stop_loss_pct']:.2f}%")
    print(f"  Take Profit:   {config['take_profit_pct']:.2f}%")
    print(f"  Oversold:      {config['oversold_threshold']:.4f}")
    print(f"  Overbought:    {config['overbought_threshold']:.4f}")
    print(f"  Accel Exit:    {config['use_accel_exit']}")
    if config['use_accel_exit']:
        print(f"    Type:        {config['accel_exit_type']}")
        print(f"    Threshold:   {config['accel_exit_threshold']:.4f}")
        print(f"    Min PnL:     {config['accel_exit_min_pnl']:.2f}%")
        print(f"    Jerk Confirm: {config['use_jerk_confirm']}")
    print(f"  Trailing Stop: {config.get('use_trailing_stop', False)}")
    if config.get('use_trailing_stop', False):
        print(f"    Trail Pct:   {config['trailing_stop_pct']:.2f}%")
        print(f"    Activation:  {config['trailing_stop_activation_pct']:.2f}%")
    print(f"  Break-Even:    {config.get('use_breakeven_stop', False)}")
    if config.get('use_breakeven_stop', False):
        print(f"    Trigger:     {config['breakeven_trigger_pct']:.2f}%")
        print(f"    Offset:      {config['breakeven_offset_pct']:.2f}%")

    print(f"\n{'─' * 70}")
    print("VALIDATION RESULTS")
    print(f"{'─' * 70}")

    # Check which validation type
    if 'split_train_return' in config:
        prefix = 'split'
        val_type = 'Simple Split'
    else:
        prefix = 'wf'
        val_type = 'Walk-Forward'

    print(f"  Validation Type: {val_type}")
    print(f"\n  {'Metric':<20} {'Train':>12} {'Test':>12}")
    print(f"  {'-' * 44}")
    print(f"  {'Total Return':<20} {config.get(f'{prefix}_train_return', 0):>11.2f}% {config.get(f'{prefix}_test_return', 0):>11.2f}%")
    print(f"  {'Win Rate':<20} {config.get(f'{prefix}_train_win_rate', 0):>11.1f}% {config.get(f'{prefix}_test_win_rate', 0):>11.1f}%")
    print(f"  {'Trades':<20} {config.get(f'{prefix}_train_trades', 0):>12} {config.get(f'{prefix}_test_trades', 0):>12}")
    print(f"  {'Max Drawdown':<20} {config.get(f'{prefix}_train_max_dd', 0):>11.2f}% {config.get(f'{prefix}_test_max_dd', 0):>11.2f}%")
    print(f"  {'Profit Factor':<20} {config.get(f'{prefix}_train_profit_factor', 0):>12.2f} {config.get(f'{prefix}_test_profit_factor', 0):>12.2f}")

    print(f"\n{'─' * 70}")
    print("OPTIMIZATION METHOD")
    print(f"{'─' * 70}")
    opt = config.get('optimization_method', {})
    print(f"  Script:      {opt.get('script', 'unknown')}")
    print(f"  Trials:      {opt.get('n_trials', 0):,}")
    print(f"  Workers:     {opt.get('n_jobs', 0)}")
    print(f"  Metric:      {opt.get('metric', 'total_return')}")
    print(f"  Optimized:   {opt.get('optimized_at', 'unknown')}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"\n1. Deploy the strategy:")
    print(f"   python velocity_live_trader.py --config {bundle_path}/velocity_config.json")
    print(f"\n2. Or add to multi-trader:")
    print(f"   Edit velocity_multi_trader.py to include this bundle")
    print(f"\n3. Set up Discord alerts (if not already set):")
    print(f"   Add webhook URL to: {bundle_path}/velocity_config.json")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Create Strategy Bundle from Validation Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create bundle from simple split results
  python create_strategy_bundle.py --results velocity_split_results_ES=F_60d_20260130_125826.json

  # Create bundle with custom name
  python create_strategy_bundle.py --results results.json --name my_es_strategy

  # Create bundle with Discord webhook
  python create_strategy_bundle.py --results results.json --webhook https://discord.com/api/webhooks/...

  # List available result files
  python create_strategy_bundle.py --list
        """
    )

    parser.add_argument('--results', '-r', type=str,
                        help='Path to validation results JSON file')
    parser.add_argument('--name', '-n', type=str,
                        help='Custom bundle name (auto-generated if not provided)')
    parser.add_argument('--webhook', '-w', type=str,
                        help='Discord webhook URL for alerts')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available validation result files')

    args = parser.parse_args()

    # List mode
    if args.list:
        print("\nAvailable validation result files:")
        print("-" * 50)

        import glob
        patterns = [
            'velocity_split_results_*.json',
            'velocity_wf_results_*.json'
        ]

        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))

        if not files:
            print("  No validation result files found.")
        else:
            for f in sorted(files):
                # Get file size and modification time
                stat = os.stat(f)
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
                size = stat.st_size / 1024  # KB
                print(f"  {f} ({size:.1f} KB, {mtime})")

        print()
        return

    # Validate arguments
    if not args.results:
        parser.error("--results is required (use --list to see available files)")

    if not os.path.exists(args.results):
        print(f"ERROR: Results file not found: {args.results}")
        sys.exit(1)

    # Create the bundle
    try:
        bundle_path, config = create_bundle_from_results(
            results_path=args.results,
            bundle_name=args.name,
            discord_webhook=args.webhook
        )

        print_bundle_summary(config, bundle_path)

    except Exception as e:
        print(f"ERROR: Failed to create bundle: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
