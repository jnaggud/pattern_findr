"""
Comprehensive Walk-Forward Test for Range Predictor

Parameters:
- 5 years of historical data
- 60 day test period
- 180 day training window
- Retrain daily (rolling window)
- 100 Optuna trials per training
- 32 parallel workers

This is a rigorous test that simulates real trading conditions.
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser(description='Comprehensive Walk-Forward Test')
parser.add_argument('--ticker', default='SPY', help='Ticker to test')
parser.add_argument('--years', type=int, default=5, help='Years of historical data')
parser.add_argument('--test-days', type=int, default=60, help='Days to test predictions')
parser.add_argument('--train-days', type=int, default=180, help='Training window size')
parser.add_argument('--trials', type=int, default=100, help='Optuna trials per training')
parser.add_argument('--workers', type=int, default=32, help='Parallel workers')
parser.add_argument('--retrain-freq', type=int, default=1, help='Retrain every N days (1=daily)')
args = parser.parse_args()

print("=" * 80)
print("COMPREHENSIVE WALK-FORWARD TEST")
print("=" * 80)
print(f"Ticker: {args.ticker}")
print(f"Historical Data: {args.years} years")
print(f"Test Period: {args.test_days} days")
print(f"Training Window: {args.train_days} days")
print(f"Retrain Frequency: Every {args.retrain_freq} day(s)")
print(f"Optuna Trials: {args.trials} per training")
print(f"Workers: {args.workers}")
print("=" * 80)

# Import modules
from price_prediction import PriceRangePredictor
try:
    from polygon_manager import PolygonManager
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

import yfinance as yf

# Fetch data
print("\n" + "-" * 80)
print("FETCHING DATA...")
print("-" * 80)

end_date = datetime.now()
start_date = end_date - timedelta(days=args.years * 365)

print(f"Fetching {args.ticker} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
df = yf.Ticker(args.ticker).history(start=start_date, end=end_date)
df.columns = [c.lower() for c in df.columns]

print(f"Total bars: {len(df)}")
print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
print(f"Current price: ${df['close'].iloc[-1]:.2f}")

# DATA SOURCE STRATEGY:
# - Price data (OHLCV): yfinance (5 years historical available)
# - Options data: Polygon (point-in-time only, used for PREDICTION not training)
#
# NOTE: Historical options data is not available. Options features (IV, PCR, Greeks)
# are only used when making predictions, not during training. The model learns from
# price/volatility patterns and uses current options data to enhance predictions.

print("\nData Source Strategy:")
print("  - Price data: yfinance (historical)")
print("  - Options data: Polygon (current-day only, for predictions)")

# We don't need Polygon during walk-forward since we're testing historical predictions
# Options data would only be available for the current day anyway
polygon = None
options_features = None
print("  - Note: Walk-forward test uses NO options data (historical test)")
print("          Options features only help for LIVE predictions")

# Walk-forward test setup
print("\n" + "-" * 80)
print("WALK-FORWARD TEST")
print("-" * 80)

test_start_idx = len(df) - args.test_days
min_train_idx = test_start_idx - args.train_days

if min_train_idx < 60:  # Need at least 60 bars for features
    print(f"ERROR: Not enough data. Need at least {args.train_days + args.test_days + 60} bars.")
    exit(1)

print(f"Test period: {df.index[test_start_idx].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
print(f"Training uses {args.train_days} days before each test day")

results = []
model = None
last_train_idx = -999  # Force initial training

total_start_time = time.time()

for i in range(args.test_days - 1):  # -1 because we need next day's actual values
    test_idx = test_start_idx + i
    test_date = df.index[test_idx]

    # Determine if we need to retrain
    need_retrain = (test_idx - last_train_idx) >= args.retrain_freq or model is None

    if need_retrain:
        # Get training data (rolling window)
        train_end_idx = test_idx
        train_start_idx = max(0, train_end_idx - args.train_days)
        df_train = df.iloc[train_start_idx:train_end_idx].copy()

        print(f"\n[Day {i+1}/{args.test_days-1}] Training on {len(df_train)} bars ({df_train.index[0].strftime('%Y-%m-%d')} to {df_train.index[-1].strftime('%Y-%m-%d')})...")

        # Train new model
        model = PriceRangePredictor(polygon)
        train_start = time.time()

        try:
            # Suppress verbose output during training
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            train_results = model.train_range_model(
                df_train,
                options_features=None,  # Don't use options for training (point-in-time)
                n_trials=args.trials,
                n_workers=args.workers
            )

            sys.stdout = old_stdout

            train_time = time.time() - train_start
            r2 = train_results['metrics']['r2']
            rmse_pct = train_results['metrics'].get('rmse_pct', train_results['metrics']['rmse'] * 100)

            print(f"   Trained in {train_time:.1f}s | R²={r2:.4f} | RMSE={rmse_pct:.3f}%")
            last_train_idx = test_idx

        except Exception as e:
            print(f"   Training failed: {e}")
            continue

    # Make prediction
    df_up_to_now = df.iloc[:test_idx + 1].copy()

    try:
        # Suppress verbose output during prediction
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        pred = model.predict_daily_range(df_up_to_now, options_features=options_features)

        sys.stdout = old_stdout

        # Get actual next day values
        actual_high = df['high'].iloc[test_idx + 1]
        actual_low = df['low'].iloc[test_idx + 1]
        actual_close = df['close'].iloc[test_idx]

        # Check containment
        high_contained = pred['high_lower'] <= actual_high <= pred['high_upper']
        low_contained = pred['low_lower'] <= actual_low <= pred['low_upper']

        results.append({
            'date': test_date,
            'actual_high': actual_high,
            'actual_low': actual_low,
            'pred_high': pred['predicted_high'],
            'pred_low': pred['predicted_low'],
            'pred_range_pct': pred['predicted_range'] * 100,
            'high_lower': pred['high_lower'],
            'high_upper': pred['high_upper'],
            'low_lower': pred['low_lower'],
            'low_upper': pred['low_upper'],
            'high_error': actual_high - pred['predicted_high'],
            'low_error': actual_low - pred['predicted_low'],
            'high_contained': high_contained,
            'low_contained': low_contained
        })

        # Progress update every 10 days
        if (i + 1) % 10 == 0:
            elapsed = time.time() - total_start_time
            remaining = elapsed / (i + 1) * (args.test_days - 1 - i - 1)
            curr_high_rate = sum(r['high_contained'] for r in results) / len(results) * 100
            curr_low_rate = sum(r['low_contained'] for r in results) / len(results) * 100
            print(f"   Progress: {i+1}/{args.test_days-1} | High: {curr_high_rate:.1f}% | Low: {curr_low_rate:.1f}% | ETA: {remaining/60:.1f}min")

    except Exception as e:
        print(f"   Prediction failed on {test_date.strftime('%Y-%m-%d')}: {e}")

total_time = time.time() - total_start_time

# Calculate final metrics
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

if results:
    results_df = pd.DataFrame(results)

    # Containment rates
    high_containment = results_df['high_contained'].mean() * 100
    low_containment = results_df['low_contained'].mean() * 100
    both_contained = (results_df['high_contained'] & results_df['low_contained']).mean() * 100

    # Error metrics
    high_mae = results_df['high_error'].abs().mean()
    low_mae = results_df['low_error'].abs().mean()
    high_bias = results_df['high_error'].mean()
    low_bias = results_df['low_error'].mean()

    # Range prediction accuracy
    pred_range_mean = results_df['pred_range_pct'].mean()
    pred_range_std = results_df['pred_range_pct'].std()

    print(f"\nContainment Rates (90% confidence):")
    print(f"   High in range: {high_containment:.1f}%")
    print(f"   Low in range:  {low_containment:.1f}%")
    print(f"   Both in range: {both_contained:.1f}%")

    print(f"\nError Metrics:")
    print(f"   High MAE: ${high_mae:.2f}")
    print(f"   Low MAE:  ${low_mae:.2f}")
    print(f"   High Bias: ${high_bias:+.2f} ({'over' if high_bias > 0 else 'under'}predicting)")
    print(f"   Low Bias:  ${low_bias:+.2f} ({'over' if low_bias > 0 else 'under'}predicting)")

    print(f"\nPredicted Range Stats:")
    print(f"   Mean: {pred_range_mean:.2f}%")
    print(f"   Std:  {pred_range_std:.2f}%")
    print(f"   Min:  {results_df['pred_range_pct'].min():.2f}%")
    print(f"   Max:  {results_df['pred_range_pct'].max():.2f}%")

    print(f"\nPerformance Assessment:")
    if high_containment >= 66.7:
        print(f"   ✓ High containment PASSED (target: 66.7%, got: {high_containment:.1f}%)")
    else:
        print(f"   ✗ High containment BELOW target (target: 66.7%, got: {high_containment:.1f}%)")

    if low_containment >= 50:
        print(f"   ✓ Low containment IMPROVED (baseline: 38.3%, got: {low_containment:.1f}%)")
    else:
        print(f"   → Low containment needs work (baseline: 38.3%, got: {low_containment:.1f}%)")

    print(f"\nTest Statistics:")
    print(f"   Total predictions: {len(results)}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Avg time per day: {total_time/len(results):.1f} seconds")

else:
    print("No results to analyze")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
