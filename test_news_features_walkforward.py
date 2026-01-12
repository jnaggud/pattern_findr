"""
Walk-Forward Analysis: News Features A/B Test (PARALLELIZED)

Proper walk-forward test that predicts next day's HIGH and LOW.
Compares performance WITH vs WITHOUT news features.

Uses run_parallel_walk_forward for maximum speed.

Usage:
    python test_news_features_walkforward.py --ticker SPY --test-days 60 --train-days 252
"""

import json
import os
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def fetch_data(ticker: str, days: int = 365) -> pd.DataFrame:
    """Fetch price data from local database or yfinance."""
    try:
        from market_data_db import MarketDataDB
        db = MarketDataDB()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = db.get_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if df is not None and len(df) > 100:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            print(f"  Loaded {len(df)} bars from local database")
            return df
    except Exception as e:
        print(f"  Local DB failed: {e}")

    try:
        import yfinance as yf
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'), progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = df.columns.str.lower()
        print(f"  Loaded {len(df)} bars from yfinance")
        return df
    except Exception as e:
        print(f"  yfinance failed: {e}")
        return pd.DataFrame()


def run_parallel_walkforward(df: pd.DataFrame, ticker: str, test_days: int, train_window: int,
                              n_trials: int, n_workers: int, use_news: bool) -> Dict:
    """
    Run PARALLEL walk-forward test using run_parallel_walk_forward from price_prediction.py
    """
    from price_prediction import run_parallel_walk_forward, FEATURE_CONFIG

    total_bars = len(df)
    test_start_idx = total_bars - test_days

    if test_start_idx < train_window:
        print(f"  Error: Not enough data. Need {train_window + test_days} bars, have {total_bars}")
        return {'predictions': [], 'aggregate': {}}

    mode_str = "WITH NEWS" if use_news else "NO NEWS"
    print(f"\n  Parallel Walk-Forward ({mode_str}):")
    print(f"    Test days: {test_days}")
    print(f"    Train window: {train_window}")
    print(f"    Workers: {n_workers}")

    # Toggle news features via FEATURE_CONFIG
    original_news_setting = FEATURE_CONFIG.get('news_sentiment', True)
    FEATURE_CONFIG['news_sentiment'] = use_news
    print(f"    News features: {'ENABLED' if use_news else 'DISABLED'}")

    try:
        # Run parallel walk-forward
        # Pass ticker to enable news V2 features (only works when news_sentiment=True)
        results_list = run_parallel_walk_forward(
            pred_df=df,
            test_start_idx=test_start_idx,
            train_window=train_window,
            n_trials=n_trials,
            n_workers=n_workers,
            ci_level=0.90,
            polygon_api_key=None,
            ticker=ticker if use_news else None  # KEY: Pass ticker to enable V2 news features
        )
    finally:
        # Restore original setting
        FEATURE_CONFIG['news_sentiment'] = original_news_setting

    # Convert to our format
    predictions = []
    for r in results_list:
        predictions.append({
            'date': r['date'],
            'predicted_high': r['predicted_high'],
            'predicted_low': r['predicted_low'],
            'predicted_range': r.get('predicted_range_dollars', r['predicted_high'] - r['predicted_low']),
            'high_lower': r['high_lower'],
            'high_upper': r['high_upper'],
            'low_lower': r['low_lower'],
            'low_upper': r['low_upper'],
            'actual_high': r['actual_high'],
            'actual_low': r['actual_low'],
            'actual_open': r.get('actual_open', 0),
            'actual_close': r.get('actual_close', 0),
            'actual_range': r['actual_high'] - r['actual_low'],
        })

    result = {
        'ticker': ticker,
        'use_news': use_news,
        'timestamp': datetime.now().isoformat(),
        'predictions': predictions,
    }

    # Calculate aggregate metrics
    if predictions:
        wf_df = pd.DataFrame(predictions)

        high_in_range = ((wf_df['actual_high'] >= wf_df['high_lower']) &
                         (wf_df['actual_high'] <= wf_df['high_upper'])).mean() * 100
        low_in_range = ((wf_df['actual_low'] >= wf_df['low_lower']) &
                        (wf_df['actual_low'] <= wf_df['low_upper'])).mean() * 100
        price_contained = ((wf_df['actual_high'] <= wf_df['high_upper']) &
                           (wf_df['actual_low'] >= wf_df['low_lower'])).mean() * 100

        range_mae = np.abs(wf_df['predicted_range'] - wf_df['actual_range']).mean()
        range_mape = (np.abs(wf_df['predicted_range'] - wf_df['actual_range']) /
                      wf_df['actual_range']).mean() * 100

        high_mae = np.abs(wf_df['predicted_high'] - wf_df['actual_high']).mean()
        low_mae = np.abs(wf_df['predicted_low'] - wf_df['actual_low']).mean()
        high_bias = (wf_df['predicted_high'] - wf_df['actual_high']).mean()
        low_bias = (wf_df['predicted_low'] - wf_df['actual_low']).mean()

        # Calculate R² for HIGH and LOW predictions
        from sklearn.metrics import r2_score
        r2_high = r2_score(wf_df['actual_high'], wf_df['predicted_high'])
        r2_low = r2_score(wf_df['actual_low'], wf_df['predicted_low'])
        r2_range = r2_score(wf_df['actual_range'], wf_df['predicted_range'])

        result['aggregate'] = {
            'high_containment': float(high_in_range),
            'low_containment': float(low_in_range),
            'full_containment': float(price_contained),
            'range_mae': float(range_mae),
            'range_mape': float(range_mape),
            'high_mae': float(high_mae),
            'low_mae': float(low_mae),
            'high_bias': float(high_bias),
            'low_bias': float(low_bias),
            'r2_high': float(r2_high),
            'r2_low': float(r2_low),
            'r2_range': float(r2_range),
            'n_predictions': len(predictions),
        }

    return result


def print_results(results: Dict, label: str):
    """Print walk-forward results."""
    if 'aggregate' not in results or not results['aggregate']:
        print(f"  {label}: No results")
        return

    agg = results['aggregate']
    print(f"\n  {label}:")
    print(f"    High Containment:  {agg['high_containment']:.1f}%")
    print(f"    Low Containment:   {agg['low_containment']:.1f}%")
    print(f"    Full Containment:  {agg['full_containment']:.1f}%")
    print(f"    Range MAPE:        {agg['range_mape']:.1f}%")
    print(f"    High MAE:          ${agg['high_mae']:.2f}")
    print(f"    Low MAE:           ${agg['low_mae']:.2f}")
    print(f"    High Bias:         ${agg['high_bias']:+.2f}")
    print(f"    Low Bias:          ${agg['low_bias']:+.2f}")
    print(f"    R² High:           {agg['r2_high']:.4f}")
    print(f"    R² Low:            {agg['r2_low']:.4f}")
    print(f"    R² Range:          {agg['r2_range']:.4f}")


def compare_results(baseline: Dict, with_news: Dict):
    """Compare baseline vs news results."""
    print_header("A/B TEST COMPARISON")

    if 'aggregate' not in baseline or 'aggregate' not in with_news:
        print("  Cannot compare - missing results")
        return

    if not baseline['aggregate'] or not with_news['aggregate']:
        print("  Cannot compare - empty results")
        return

    b = baseline['aggregate']
    n = with_news['aggregate']

    print(f"\n  {'Metric':<25} {'Baseline':>12} {'With News':>12} {'Change':>12}")
    print("  " + "-" * 65)

    # Higher is better
    for metric in ['high_containment', 'low_containment', 'full_containment']:
        diff = n[metric] - b[metric]
        label = metric.replace('_', ' ').title()
        better = "+" if diff > 0 else ""
        print(f"  {label:<25} {b[metric]:>11.1f}% {n[metric]:>11.1f}% {better}{diff:>+10.1f}pp")

    # Lower is better
    print()
    for metric in ['range_mape']:
        diff = b[metric] - n[metric]
        label = metric.replace('_', ' ').upper()
        better = "+" if diff > 0 else ""
        print(f"  {label:<25} {b[metric]:>11.1f}% {n[metric]:>11.1f}% {better}{diff:>+10.1f}pp")

    for metric in ['high_mae', 'low_mae']:
        diff = b[metric] - n[metric]
        label = metric.replace('_', ' ').upper()
        better = "+" if diff > 0 else ""
        print(f"  {label:<25} ${b[metric]:>10.2f} ${n[metric]:>10.2f} {better}${diff:>+9.2f}")

    # Bias
    print()
    for metric in ['high_bias', 'low_bias']:
        label = metric.replace('_', ' ').title()
        print(f"  {label:<25} ${b[metric]:>+10.2f} ${n[metric]:>+10.2f}")

    # R² metrics
    print()
    for metric in ['r2_high', 'r2_low', 'r2_range']:
        diff = n[metric] - b[metric]
        label = metric.replace('_', ' ').upper()
        better = "+" if diff > 0 else ""
        print(f"  {label:<25} {b[metric]:>12.4f} {n[metric]:>12.4f} {better}{diff:>+11.4f}")

    # Summary
    print_header("SUMMARY")
    high_diff = n['high_containment'] - b['high_containment']
    low_diff = n['low_containment'] - b['low_containment']

    if high_diff > 0 and low_diff > 0:
        print(f"\n  NEWS FEATURES IMPROVED both High (+{high_diff:.1f}pp) and Low (+{low_diff:.1f}pp) containment")
    elif high_diff > 0 or low_diff > 0:
        print(f"\n  NEWS FEATURES IMPROVED some metrics:")
        if high_diff > 0:
            print(f"    - High containment: +{high_diff:.1f}pp")
        if low_diff > 0:
            print(f"    - Low containment: +{low_diff:.1f}pp")
    else:
        print(f"\n  NEWS FEATURES did not improve containment")
        print(f"    - High: {high_diff:+.1f}pp")
        print(f"    - Low: {low_diff:+.1f}pp")


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward A/B Test: News Features (PARALLEL)')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--test-days', type=int, default=60, help='Days to test on')
    parser.add_argument('--train-days', type=int, default=252, help='Rolling training window')
    parser.add_argument('--trials', type=int, default=300, help='Optuna trials per training')
    parser.add_argument('--workers', type=int, default=32, help='Parallel workers')
    parser.add_argument('--data-days', type=int, default=600, help='Total days of data to fetch')
    args = parser.parse_args()

    print_header("WALK-FORWARD A/B TEST: NEWS FEATURES (PARALLEL)")
    print(f"\n  Ticker: {args.ticker}")
    print(f"  Test Period: {args.test_days} days")
    print(f"  Training Window: {args.train_days} days (rolling)")
    print(f"  Optuna Trials: {args.trials}")
    print(f"  Parallel Workers: {args.workers}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Fetch data
    print(f"\n  Fetching {args.data_days} days of data...")
    df = fetch_data(args.ticker, args.data_days)

    if df.empty or len(df) < args.train_days + args.test_days:
        print(f"  Error: Insufficient data (need {args.train_days + args.test_days}, have {len(df)})")
        return

    print(f"  Data range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total bars: {len(df)}")

    # Run BASELINE (no news features)
    print_header("BASELINE: No News Features")
    baseline_results = run_parallel_walkforward(
        df.copy(), args.ticker, args.test_days, args.train_days,
        args.trials, args.workers, use_news=False
    )
    print_results(baseline_results, "Baseline Results")

    # Run WITH NEWS features
    print_header("TEST: With News Features")
    news_results = run_parallel_walkforward(
        df.copy(), args.ticker, args.test_days, args.train_days,
        args.trials, args.workers, use_news=True
    )
    print_results(news_results, "With News Results")

    # Compare
    compare_results(baseline_results, news_results)

    # Save results
    output = {
        'ticker': args.ticker,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'test_days': args.test_days,
            'train_days': args.train_days,
            'trials': args.trials,
            'workers': args.workers
        },
        'baseline': baseline_results,
        'with_news': news_results
    }

    output_path = f"news_walkforward_test_{args.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
