"""
Quick Test Script for Enhanced Range Predictor

Tests:
1. Basic PriceRangePredictor with enhanced features
2. RegimeSwitchingRangePredictor
3. Polygon data retrieval (if API key available)
4. Feature generation and IV debug output

Usage:
    python test_range_predictor.py
    python test_range_predictor.py --ticker AAPL --days 365
    python test_range_predictor.py --regime  # Test regime-switching model
"""

import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ticker', default='SPY', help='Ticker to test')
parser.add_argument('--days', type=int, default=500, help='Days of historical data')
parser.add_argument('--regime', action='store_true', help='Test regime-switching model')
parser.add_argument('--trials', type=int, default=30, help='Optuna trials')
parser.add_argument('--test-days', type=int, default=30, help='Days to test predictions')
args = parser.parse_args()

print("=" * 70)
print(f"ENHANCED RANGE PREDICTOR TEST")
print(f"Ticker: {args.ticker} | History: {args.days} days | Test: {args.test_days} days")
print("=" * 70)

# Import modules
try:
    from price_prediction import PriceRangePredictor, RegimeSwitchingRangePredictor
    print("✓ Imported price_prediction module")
except ImportError as e:
    print(f"✗ Failed to import price_prediction: {e}")
    exit(1)

try:
    from polygon_manager import PolygonManager
    POLYGON_AVAILABLE = True
    print("✓ Imported polygon_manager module")
except ImportError:
    POLYGON_AVAILABLE = False
    print("✗ polygon_manager not available")

try:
    import yfinance as yf
    YF_AVAILABLE = True
    print("✓ Imported yfinance")
except ImportError:
    YF_AVAILABLE = False
    print("✗ yfinance not available")

# Fetch data
print("\n" + "-" * 70)
print("FETCHING DATA...")
print("-" * 70)

polygon = None
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")

if POLYGON_AVAILABLE:
    polygon = PolygonManager(POLYGON_API_KEY)
    print(f"✓ Polygon manager initialized")

    # Try to get data from Polygon first
    df = polygon.get_price_data(args.ticker, limit=args.days)
    if not df.empty:
        print(f"✓ Got {len(df)} bars from Polygon")
    else:
        df = None

if df is None or df.empty:
    if YF_AVAILABLE:
        print("Falling back to yfinance...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        ticker_obj = yf.Ticker(args.ticker)
        df = ticker_obj.history(start=start_date, end=end_date)
        df.columns = [c.lower() for c in df.columns]
        print(f"✓ Got {len(df)} bars from yfinance")
    else:
        print("✗ No data source available")
        exit(1)

print(f"Data range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
print(f"Current price: ${df['close'].iloc[-1]:.2f}")

# Split into train/test
test_start = len(df) - args.test_days
df_train = df.iloc[:test_start].copy()
df_test = df.iloc[test_start:].copy()
print(f"Training samples: {len(df_train)}, Test samples: {len(df_test)}")

# Get options features
print("\n" + "-" * 70)
print("FETCHING OPTIONS DATA...")
print("-" * 70)

options_features = None
if polygon and 'USD' not in args.ticker:
    predictor_temp = PriceRangePredictor(polygon)
    options_features = predictor_temp.get_options_features(args.ticker, df['close'].iloc[-1])

    if options_features.get('iv_weighted'):
        print(f"✓ Options data retrieved:")
        print(f"   PCR Volume: {options_features.get('pcr_volume', 'N/A')}")
        print(f"   IV Weighted: {options_features.get('iv_weighted', 'N/A')}%")
        print(f"   ATM IV: {options_features.get('atm_iv', 'N/A')}%")
        print(f"   Max Pain: ${options_features.get('max_pain', 'N/A')}")
        print(f"   Term Structure: {options_features.get('term_structure_type', 'N/A')}")
        print(f"   Activity Bias: {options_features.get('activity_bias', 'N/A')}")
    else:
        print("✗ No options data available (will use volatility fallback)")
else:
    print("Skipping options data (crypto or no Polygon)")

# Test feature generation
print("\n" + "-" * 70)
print("TESTING FEATURE GENERATION...")
print("-" * 70)

predictor = PriceRangePredictor(polygon)
features = predictor.create_range_features(df_train, options_features)
print(f"✓ Generated {len(features.columns)} features")

# Show some key new features
new_features = [
    'volume_zscore', 'volume_trend_5d', 'vol_price_divergence', 'volume_spike',
    'range_momentum', 'vol_regime_interaction', 'atr_breakout', 'trend_dislocation',
    'is_downtrend', 'is_uptrend', 'downtrend_volume', 'downtrend_atr',
    'atm_iv', 'term_structure_slope', 'net_gamma_normalized'
]

print("\nNew enhanced features (sample values from last row):")
for feat in new_features:
    if feat in features.columns:
        val = features[feat].iloc[-1]
        if pd.notna(val):
            print(f"   {feat}: {val:.4f}")
        else:
            print(f"   {feat}: NaN")
    else:
        print(f"   {feat}: NOT FOUND")

# Train model
print("\n" + "-" * 70)
if args.regime:
    print("TRAINING REGIME-SWITCHING MODEL...")
else:
    print("TRAINING STANDARD MODEL...")
print("-" * 70)

# IMPORTANT: Don't pass options_features to training!
# Options data is point-in-time (today's values), not historical.
# Passing it would make all IV features constant (same value for all rows).
# The model should learn from volatility_20d (which varies daily) as IV proxy.
print("NOTE: Training WITHOUT options_features (uses volatility fallback)")
print("      Options data will only be used during prediction.\n")

if args.regime:
    model = RegimeSwitchingRangePredictor(polygon)
    train_results = model.train(df_train, options_features=None, n_trials=args.trials)
    print(f"\nTraining Results:")
    print(f"   High-Vol R²: {train_results.get('high_vol_r2', 0):.4f}")
    print(f"   Low-Vol R²: {train_results.get('low_vol_r2', 0):.4f}")
    print(f"   Vol Threshold: {train_results.get('vol_threshold', 0):.2f}%")
else:
    model = PriceRangePredictor(polygon)
    train_results = model.train_range_model(df_train, options_features=None, n_trials=args.trials)
    print(f"\nTraining Results:")
    print(f"   R²: {train_results.get('r2', 0):.4f}")
    print(f"   RMSE: {train_results.get('rmse', 0):.4f}")

# Walk-forward test
print("\n" + "-" * 70)
print("WALK-FORWARD TESTING...")
print("-" * 70)

results = []
for i in range(len(df_test)):
    # Get data up to this point
    test_date = df_test.index[i]
    df_up_to_now = df[df.index <= test_date].copy()

    # Get actual next day values (if available)
    if i < len(df_test) - 1:
        actual_high = df_test['high'].iloc[i + 1]
        actual_low = df_test['low'].iloc[i + 1]
        actual_close = df_test['close'].iloc[i]

        try:
            # Make prediction
            pred = model.predict_daily_range(df_up_to_now, options_features=options_features)

            pred_high = pred['predicted_high']
            pred_low = pred['predicted_low']
            high_upper = pred['high_upper']
            high_lower = pred['high_lower']
            low_upper = pred['low_upper']
            low_lower = pred['low_lower']

            # Check containment
            high_contained = high_lower <= actual_high <= high_upper
            low_contained = low_lower <= actual_low <= low_upper

            results.append({
                'date': test_date,
                'actual_high': actual_high,
                'actual_low': actual_low,
                'pred_high': pred_high,
                'pred_low': pred_low,
                'high_error': actual_high - pred_high,
                'low_error': actual_low - pred_low,
                'high_contained': high_contained,
                'low_contained': low_contained,
                'regime': pred.get('regime_used', 'standard')
            })

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(df_test) - 1} days...")

        except Exception as e:
            print(f"   Error on {test_date}: {e}")

# Calculate metrics
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

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

    print(f"\nContainment Rates (90% confidence):")
    print(f"   High in range: {high_containment:.1f}%")
    print(f"   Low in range:  {low_containment:.1f}%")
    print(f"   Both in range: {both_contained:.1f}%")

    print(f"\nError Metrics:")
    print(f"   High MAE: ${high_mae:.2f}")
    print(f"   Low MAE:  ${low_mae:.2f}")
    print(f"   High Bias: ${high_bias:+.2f} ({'over' if high_bias > 0 else 'under'}predicting)")
    print(f"   Low Bias:  ${low_bias:+.2f} ({'over' if low_bias > 0 else 'under'}predicting)")

    # Regime breakdown (if using regime model)
    if args.regime and 'regime' in results_df.columns:
        print(f"\nRegime Breakdown:")
        for regime in results_df['regime'].unique():
            regime_df = results_df[results_df['regime'] == regime]
            print(f"   {regime}: {len(regime_df)} predictions, High={regime_df['high_contained'].mean()*100:.1f}%, Low={regime_df['low_contained'].mean()*100:.1f}%")

    # Comparison to target
    print(f"\nPerformance Assessment:")
    if high_containment >= 66:
        print(f"   ✓ High containment IMPROVED (target: 66.7%, got: {high_containment:.1f}%)")
    else:
        print(f"   ✗ High containment BELOW target (target: 66.7%, got: {high_containment:.1f}%)")

    if low_containment >= 50:
        print(f"   ✓ Low containment IMPROVED (baseline: 38.3%, got: {low_containment:.1f}%)")
    else:
        print(f"   → Low containment still needs work (baseline: 38.3%, got: {low_containment:.1f}%)")

else:
    print("No results to analyze")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
