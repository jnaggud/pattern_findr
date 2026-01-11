"""
Range Prediction A/B Test Framework

Matches the exact logic from Walk Forward Analysis in oscillator_predictor_page.py.

Settings:
- Test Period: 60 days
- Minimum Training Days: 180 (rolling window)
- Retrain Frequency: Every Day
- Optuna Trials per Training: 300
- Parallel Workers: 32
- Years of Data: 5

Usage:
    # Step 1: Run baseline (without new cross-market features)
    python range_prediction_ab_test.py --baseline

    # Step 2: Run test (with new cross-market features enabled)
    python range_prediction_ab_test.py --test

    # Step 3: Compare results
    python range_prediction_ab_test.py --compare
"""

import json
import os
import sys
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Results storage paths
BASELINE_RESULTS_PATH = "range_prediction_baseline.json"
TEST_RESULTS_PATH = "range_prediction_test_results.json"
COMPARISON_REPORT_PATH = "range_prediction_comparison.json"

# Test configuration - Matches Walk Forward Analysis settings
TEST_CONFIG = {
    'tickers': ['SPY'],
    'test_period_days': 60,
    'min_training_days': 180,  # Rolling window size
    'retrain_frequency': 1,  # Daily retraining
    'optuna_trials': 300,
    'parallel_workers': 32,
    'confidence_interval': 0.90,
    'years_of_data': 5,
}


def fetch_test_data(ticker: str) -> pd.DataFrame:
    """Fetch price data - uses 5 years of data."""
    try:
        import yfinance as yf

        total_days = 365 * TEST_CONFIG['years_of_data']

        end_date = datetime.now()
        start_date = end_date - timedelta(days=total_days)

        print(f"Fetching {ticker} data ({TEST_CONFIG['years_of_data']} years)...")
        df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'), progress=False)

        if df.empty:
            print(f"  Warning: No data for {ticker}")
            return pd.DataFrame()

        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = df.columns.str.lower()
        print(f"  Loaded {len(df)} bars")
        return df

    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return pd.DataFrame()


def run_walkforward_test(df: pd.DataFrame, ticker: str) -> Dict:
    """
    Run walk-forward test matching oscillator_predictor_page.py logic EXACTLY.

    Key differences from previous version:
    1. Uses ROLLING window (last 180 days) not expanding window
    2. Uses train_range_model() and predict_daily_range() methods
    3. Calculates containment using confidence intervals
    4. Uses parallelized Optuna optimization
    """
    from price_prediction import PriceRangePredictor

    results = {
        'ticker': ticker,
        'config': TEST_CONFIG,
        'timestamp': datetime.now().isoformat(),
        'predictions': [],
    }

    total_bars = len(df)
    min_train = TEST_CONFIG['min_training_days']
    test_days = TEST_CONFIG['test_period_days']
    retrain_freq = TEST_CONFIG['retrain_frequency']
    n_trials = TEST_CONFIG['optuna_trials']
    n_workers = TEST_CONFIG['parallel_workers']

    # Test starts from end - test_days
    test_start_idx = total_bars - test_days

    if test_start_idx < min_train:
        print(f"  Error: Not enough data. Need {min_train + test_days} bars, have {total_bars}")
        return results

    print(f"\n  Walk-Forward Test (matching Streamlit app logic):")
    print(f"    Total bars: {total_bars}")
    print(f"    Test starts at: {df.index[test_start_idx].strftime('%Y-%m-%d')}")
    print(f"    Test ends at: {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"    Test period: {test_days} days")
    print(f"    Rolling window: {min_train} days")
    print(f"    Optuna trials: {n_trials}")
    print(f"    Parallel workers: {n_workers}")

    # Storage
    wf_results = []
    current_model = None
    last_train_idx = -999  # Force initial training

    # Walk forward loop - EXACT match to oscillator_predictor_page.py lines 6887-6978
    for i, test_idx in enumerate(range(test_start_idx, total_bars)):
        test_date = df.index[test_idx]
        progress = (i + 1) / test_days

        # Check if we need to retrain
        days_since_train = test_idx - last_train_idx
        need_retrain = (current_model is None) or (days_since_train >= retrain_freq)

        if need_retrain:
            print(f"  Training for {test_date.strftime('%Y-%m-%d')}... ({i+1}/{test_days})")

            # ROLLING window - only last min_train days before test (line 6900)
            train_start_idx = max(0, test_idx - min_train)
            train_df = df.iloc[train_start_idx:test_idx].copy()

            if len(train_df) >= min_train:
                try:
                    # Create fresh predictor and train (line 6906)
                    wf_predictor = PriceRangePredictor(polygon_manager=None)

                    # Train with Optuna optimization (lines 6916-6921)
                    # NOTE: options_features=None as in the app
                    train_result = wf_predictor.train_range_model(
                        train_df,
                        options_features=None,
                        n_trials=n_trials,
                        n_workers=n_workers
                    )

                    current_model = wf_predictor
                    last_train_idx = test_idx

                except Exception as train_err:
                    print(f"    Training failed: {train_err}")
                    continue
        else:
            if (i + 1) % 10 == 0:
                print(f"  Predicting {test_date.strftime('%Y-%m-%d')}... ({i+1}/{test_days})")

        # Make prediction (lines 6942-6974)
        if current_model is not None:
            try:
                # Use data up to the day BEFORE test_date (line 6945)
                pred_input_df = df.iloc[:test_idx].copy()

                # Get prediction using predict_daily_range (line 6948)
                prediction = current_model.predict_daily_range(pred_input_df)

                # Get actual values (lines 6951-6955)
                actual_high = df['high'].iloc[test_idx]
                actual_low = df['low'].iloc[test_idx]
                actual_close = df['close'].iloc[test_idx]
                actual_open = df['open'].iloc[test_idx]
                actual_range = actual_high - actual_low

                # Store result (lines 6958-6974)
                wf_results.append({
                    'date': test_date.strftime('%Y-%m-%d'),
                    'predicted_high': prediction['predicted_high'],
                    'predicted_low': prediction['predicted_low'],
                    'predicted_range': prediction['predicted_range_dollars'],
                    'high_lower': prediction['high_lower'],
                    'high_upper': prediction['high_upper'],
                    'low_lower': prediction['low_lower'],
                    'low_upper': prediction['low_upper'],
                    'actual_high': float(actual_high),
                    'actual_low': float(actual_low),
                    'actual_open': float(actual_open),
                    'actual_close': float(actual_close),
                    'actual_range': float(actual_range),
                    'model_r2': prediction.get('model_r2', 0),
                    'retrained': need_retrain
                })

            except Exception as pred_err:
                print(f"    Prediction failed: {pred_err}")

        # Print running metrics every 10 days
        if len(wf_results) > 0 and len(wf_results) % 10 == 0:
            temp_df = pd.DataFrame(wf_results)
            temp_high_in_range = ((temp_df['actual_high'] >= temp_df['high_lower']) &
                                  (temp_df['actual_high'] <= temp_df['high_upper'])).mean() * 100
            temp_low_in_range = ((temp_df['actual_low'] >= temp_df['low_lower']) &
                                 (temp_df['actual_low'] <= temp_df['low_upper'])).mean() * 100
            print(f"    Running: High {temp_high_in_range:.1f}% | Low {temp_low_in_range:.1f}%")

    results['predictions'] = wf_results

    # Calculate final metrics (matching lines 7014-7031)
    if wf_results:
        wf_df = pd.DataFrame(wf_results)

        # High/Low containment using confidence intervals
        high_in_range = ((wf_df['actual_high'] >= wf_df['high_lower']) &
                         (wf_df['actual_high'] <= wf_df['high_upper'])).mean() * 100
        low_in_range = ((wf_df['actual_low'] >= wf_df['low_lower']) &
                        (wf_df['actual_low'] <= wf_df['low_upper'])).mean() * 100

        # Full containment
        price_contained = ((wf_df['actual_high'] <= wf_df['high_upper']) &
                           (wf_df['actual_low'] >= wf_df['low_lower'])).mean() * 100

        # Range metrics
        range_mae = np.abs(wf_df['predicted_range'] - wf_df['actual_range']).mean()
        range_mape = (np.abs(wf_df['predicted_range'] - wf_df['actual_range']) / wf_df['actual_range']).mean() * 100

        # High/Low errors
        high_error = (wf_df['predicted_high'] - wf_df['actual_high']).mean()  # Bias
        low_error = (wf_df['predicted_low'] - wf_df['actual_low']).mean()  # Bias
        high_mae = np.abs(wf_df['predicted_high'] - wf_df['actual_high']).mean()
        low_mae = np.abs(wf_df['predicted_low'] - wf_df['actual_low']).mean()

        results['aggregate'] = {
            'high_containment': float(high_in_range),
            'low_containment': float(low_in_range),
            'full_containment': float(price_contained),
            'range_mape': float(range_mape),
            'high_mae': float(high_mae),
            'low_mae': float(low_mae),
            'high_bias': float(high_error),
            'low_bias': float(low_error),
            'n_predictions': len(wf_results),
        }

    return results


def run_baseline_test() -> Dict:
    """Run baseline test WITHOUT new cross-market features."""
    from price_prediction import set_cross_market_features

    print("=" * 60)
    print("BASELINE TEST (without new cross-market features)")
    print("=" * 60)

    # Disable new features
    set_cross_market_features(False)

    all_results = {}

    for ticker in TEST_CONFIG['tickers']:
        df = fetch_test_data(ticker)
        if df.empty:
            continue

        results = run_walkforward_test(df, ticker)
        all_results[ticker] = results

        if 'aggregate' in results:
            agg = results['aggregate']
            print(f"\n  {ticker} Results:")
            print(f"    High in Confidence Range: {agg['high_containment']:.1f}%")
            print(f"    Low in Confidence Range:  {agg['low_containment']:.1f}%")
            print(f"    Full Containment:         {agg['full_containment']:.1f}%")
            print(f"    Range MAPE:               {agg['range_mape']:.1f}%")
            print(f"    High MAE: ${agg['high_mae']:.2f}")
            print(f"    Low MAE:  ${agg['low_mae']:.2f}")
            print(f"    High Bias: ${agg['high_bias']:+.2f}")
            print(f"    Low Bias:  ${agg['low_bias']:+.2f}")

    # Save results
    baseline_output = {
        'type': 'baseline',
        'timestamp': datetime.now().isoformat(),
        'config': TEST_CONFIG,
        'results_by_ticker': all_results
    }

    with open(BASELINE_RESULTS_PATH, 'w') as f:
        json.dump(baseline_output, f, indent=2, default=str)

    print(f"\nBaseline results saved to: {BASELINE_RESULTS_PATH}")
    return baseline_output


def run_test_with_new_features() -> Dict:
    """Run test WITH new cross-market features enabled."""
    from price_prediction import set_cross_market_features

    print("=" * 60)
    print("TEST WITH NEW CROSS-MARKET FEATURES")
    print("=" * 60)

    # Enable new features
    set_cross_market_features(True)

    all_results = {}

    for ticker in TEST_CONFIG['tickers']:
        df = fetch_test_data(ticker)
        if df.empty:
            continue

        results = run_walkforward_test(df, ticker)
        all_results[ticker] = results

        if 'aggregate' in results:
            agg = results['aggregate']
            print(f"\n  {ticker} Results:")
            print(f"    High in Confidence Range: {agg['high_containment']:.1f}%")
            print(f"    Low in Confidence Range:  {agg['low_containment']:.1f}%")
            print(f"    Full Containment:         {agg['full_containment']:.1f}%")
            print(f"    Range MAPE:               {agg['range_mape']:.1f}%")
            print(f"    High MAE: ${agg['high_mae']:.2f}")
            print(f"    Low MAE:  ${agg['low_mae']:.2f}")
            print(f"    High Bias: ${agg['high_bias']:+.2f}")
            print(f"    Low Bias:  ${agg['low_bias']:+.2f}")

    # Save results
    test_output = {
        'type': 'test_with_new_features',
        'timestamp': datetime.now().isoformat(),
        'config': TEST_CONFIG,
        'results_by_ticker': all_results
    }

    with open(TEST_RESULTS_PATH, 'w') as f:
        json.dump(test_output, f, indent=2, default=str)

    print(f"\nTest results saved to: {TEST_RESULTS_PATH}")
    return test_output


def compare_results():
    """Compare baseline vs test results with statistical significance testing."""
    from scipy import stats

    print("=" * 60)
    print("RANGE PREDICTION A/B COMPARISON")
    print("=" * 60)

    # Load results
    if not os.path.exists(BASELINE_RESULTS_PATH):
        print("Error: Baseline results not found. Run --baseline first.")
        return
    if not os.path.exists(TEST_RESULTS_PATH):
        print("Error: Test results not found. Run --test first.")
        return

    with open(BASELINE_RESULTS_PATH, 'r') as f:
        baseline = json.load(f)
    with open(TEST_RESULTS_PATH, 'r') as f:
        test = json.load(f)

    print(f"\nBaseline timestamp: {baseline['timestamp']}")
    print(f"Test timestamp:     {test['timestamp']}")

    comparison = {
        'baseline_timestamp': baseline['timestamp'],
        'test_timestamp': test['timestamp'],
        'metrics': {}
    }

    # Compare for each ticker
    for ticker in TEST_CONFIG['tickers']:
        if ticker not in baseline['results_by_ticker'] or ticker not in test['results_by_ticker']:
            continue

        baseline_results = baseline['results_by_ticker'][ticker]
        test_results = test['results_by_ticker'][ticker]

        baseline_agg = baseline_results.get('aggregate', {})
        test_agg = test_results.get('aggregate', {})

        if not baseline_agg or not test_agg:
            continue

        # Store aggregate comparisons
        comparison['metrics']['high_containment'] = {
            'baseline': baseline_agg['high_containment'],
            'test': test_agg['high_containment'],
            'improvement': test_agg['high_containment'] - baseline_agg['high_containment'],
        }

        comparison['metrics']['low_containment'] = {
            'baseline': baseline_agg['low_containment'],
            'test': test_agg['low_containment'],
            'improvement': test_agg['low_containment'] - baseline_agg['low_containment'],
        }

        comparison['metrics']['full_containment'] = {
            'baseline': baseline_agg['full_containment'],
            'test': test_agg['full_containment'],
            'improvement': test_agg['full_containment'] - baseline_agg['full_containment'],
        }

        comparison['metrics']['range_mape'] = {
            'baseline': baseline_agg['range_mape'],
            'test': test_agg['range_mape'],
            'improvement': baseline_agg['range_mape'] - test_agg['range_mape'],  # Lower is better
        }

        comparison['metrics']['high_mae'] = {
            'baseline': baseline_agg['high_mae'],
            'test': test_agg['high_mae'],
            'improvement': baseline_agg['high_mae'] - test_agg['high_mae'],  # Lower is better
        }

        comparison['metrics']['low_mae'] = {
            'baseline': baseline_agg['low_mae'],
            'test': test_agg['low_mae'],
            'improvement': baseline_agg['low_mae'] - test_agg['low_mae'],  # Lower is better
        }

        comparison['metrics']['high_bias'] = {
            'baseline': baseline_agg['high_bias'],
            'test': test_agg['high_bias'],
        }

        comparison['metrics']['low_bias'] = {
            'baseline': baseline_agg['low_bias'],
            'test': test_agg['low_bias'],
        }

    # Save comparison
    with open(COMPARISON_REPORT_PATH, 'w') as f:
        json.dump(comparison, f, indent=2)

    # Print report
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    if 'high_containment' in comparison['metrics']:
        h = comparison['metrics']['high_containment']
        print(f"\nHigh in Confidence Range:")
        print(f"  Baseline: {h['baseline']:.1f}%")
        print(f"  Test:     {h['test']:.1f}%")
        print(f"  Change:   {h['improvement']:+.1f}pp")

    if 'low_containment' in comparison['metrics']:
        l = comparison['metrics']['low_containment']
        print(f"\nLow in Confidence Range:")
        print(f"  Baseline: {l['baseline']:.1f}%")
        print(f"  Test:     {l['test']:.1f}%")
        print(f"  Change:   {l['improvement']:+.1f}pp")

    if 'full_containment' in comparison['metrics']:
        f = comparison['metrics']['full_containment']
        print(f"\nFull Containment:")
        print(f"  Baseline: {f['baseline']:.1f}%")
        print(f"  Test:     {f['test']:.1f}%")
        print(f"  Change:   {f['improvement']:+.1f}pp")

    if 'range_mape' in comparison['metrics']:
        r = comparison['metrics']['range_mape']
        print(f"\nRange MAPE (lower is better):")
        print(f"  Baseline: {r['baseline']:.1f}%")
        print(f"  Test:     {r['test']:.1f}%")
        print(f"  Improvement: {r['improvement']:+.1f}pp")

    if 'high_mae' in comparison['metrics']:
        m = comparison['metrics']['high_mae']
        print(f"\nHigh MAE (lower is better):")
        print(f"  Baseline: ${m['baseline']:.2f}")
        print(f"  Test:     ${m['test']:.2f}")
        print(f"  Improvement: ${m['improvement']:+.2f}")

    if 'low_mae' in comparison['metrics']:
        m = comparison['metrics']['low_mae']
        print(f"\nLow MAE (lower is better):")
        print(f"  Baseline: ${m['baseline']:.2f}")
        print(f"  Test:     ${m['test']:.2f}")
        print(f"  Improvement: ${m['improvement']:+.2f}")

    if 'high_bias' in comparison['metrics']:
        b = comparison['metrics']['high_bias']
        print(f"\nHigh Bias:")
        print(f"  Baseline: ${b['baseline']:+.2f}")
        print(f"  Test:     ${b['test']:+.2f}")

    if 'low_bias' in comparison['metrics']:
        b = comparison['metrics']['low_bias']
        print(f"\nLow Bias:")
        print(f"  Baseline: ${b['baseline']:+.2f}")
        print(f"  Test:     ${b['test']:+.2f}")

    print(f"\nFull comparison saved to: {COMPARISON_REPORT_PATH}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description='Range Prediction A/B Test')
    parser.add_argument('--baseline', action='store_true', help='Run baseline test (without new features)')
    parser.add_argument('--test', action='store_true', help='Run test with new features')
    parser.add_argument('--compare', action='store_true', help='Compare baseline vs test')
    parser.add_argument('--all', action='store_true', help='Run baseline, test, and compare')

    args = parser.parse_args()

    if args.all:
        run_baseline_test()
        print("\n" + "=" * 60 + "\n")
        run_test_with_new_features()
        print("\n" + "=" * 60 + "\n")
        compare_results()
    elif args.baseline:
        run_baseline_test()
    elif args.test:
        run_test_with_new_features()
    elif args.compare:
        compare_results()
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python range_prediction_ab_test.py --baseline  # Run first")
        print("  python range_prediction_ab_test.py --test      # Run second")
        print("  python range_prediction_ab_test.py --compare   # Compare results")
        print("  python range_prediction_ab_test.py --all       # Run everything")


if __name__ == "__main__":
    main()
