"""
Test script to verify price prediction features are scale-independent across different tickers.
Tests SPY (~$690), ES=F (~$6000), and BTC-USD (~$100,000) to ensure features have similar magnitudes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the price prediction module
from price_prediction import PriceRangePredictor

def test_feature_scale_independence():
    """Test that features have similar magnitudes across different price scales."""

    tickers = ['SPY', 'ES=F', 'BTC-USD']
    results = {}

    print("=" * 70)
    print("TESTING FEATURE SCALE INDEPENDENCE")
    print("=" * 70)

    for ticker in tickers:
        print(f"\n{'='*70}")
        print(f"Testing {ticker}...")
        print("=" * 70)

        try:
            # Create predictor and generate features
            predictor = PriceRangePredictor()

            # Fetch data
            import yfinance as yf
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                print(f"  ERROR: No data for {ticker}")
                continue

            # Flatten columns if multi-index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = df.columns.str.lower()

            print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            print(f"  Current price: ${df['close'].iloc[-1]:.2f}")

            # Create features
            features = predictor.create_range_features(df, ticker=ticker)

            # Get feature statistics
            feature_stats = {}
            for col in predictor.feature_names[:30]:  # Check first 30 features
                if col in features.columns:
                    values = features[col].dropna()
                    if len(values) > 0:
                        feature_stats[col] = {
                            'mean': values.mean(),
                            'std': values.std(),
                            'min': values.min(),
                            'max': values.max()
                        }

            results[ticker] = {
                'price': df['close'].iloc[-1],
                'feature_stats': feature_stats,
                'feature_names': predictor.feature_names
            }

            # Print key feature statistics
            print(f"\n  Key Feature Statistics (should be similar across tickers):")
            key_features = ['atr_14_pct', 'daily_range', 'volatility_20d', 'rsi_14', 'volume_rel']
            for feat in key_features:
                if feat in feature_stats:
                    s = feature_stats[feat]
                    print(f"    {feat}: mean={s['mean']:.4f}, std={s['std']:.4f}, range=[{s['min']:.4f}, {s['max']:.4f}]")

            # Verify no absolute-scale features are included
            excluded = ['atr_7', 'atr_14', 'atr_21', 'volume_velocity', 'max_pain']
            found_excluded = [f for f in excluded if f in predictor.feature_names]
            if found_excluded:
                print(f"\n  WARNING: Found absolute-scale features: {found_excluded}")
            else:
                print(f"\n  OK: No absolute-scale features found in feature list")

            print(f"  Total features: {len(predictor.feature_names)}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Compare feature ranges across tickers
    if len(results) >= 2:
        print("\n" + "=" * 70)
        print("CROSS-TICKER COMPARISON")
        print("=" * 70)

        compare_features = ['atr_14_pct', 'daily_range', 'volatility_20d', 'rsi_14', 'volume_rel',
                           'range_mean_5d', 'atr_ratio_7_21', 'range_zscore']

        print(f"\n{'Feature':<25} ", end="")
        for ticker in results.keys():
            print(f"{ticker:>15} ", end="")
        print()
        print("-" * 70)

        for feat in compare_features:
            print(f"{feat:<25} ", end="")
            for ticker in results.keys():
                if feat in results[ticker]['feature_stats']:
                    mean = results[ticker]['feature_stats'][feat]['mean']
                    print(f"{mean:>15.4f} ", end="")
                else:
                    print(f"{'N/A':>15} ", end="")
            print()

        # Check if feature means are within reasonable range of each other
        print("\n" + "=" * 70)
        print("SCALE INDEPENDENCE CHECK")
        print("=" * 70)

        all_pass = True
        for feat in compare_features:
            means = []
            for ticker in results.keys():
                if feat in results[ticker]['feature_stats']:
                    means.append(results[ticker]['feature_stats'][feat]['mean'])

            if len(means) >= 2:
                # Check if max/min ratio is within 10x (should be much closer for truly scale-independent features)
                if min(means) != 0:
                    ratio = max(means) / min(means) if min(means) > 0 else float('inf')
                else:
                    ratio = float('inf') if max(means) != 0 else 1.0

                # For percentage features, ratio should be < 5x typically
                status = "PASS" if abs(ratio) < 10 else "FAIL"
                if status == "FAIL":
                    all_pass = False
                print(f"  {feat:<25}: ratio={ratio:.2f}x  [{status}]")

        print("\n" + "=" * 70)
        if all_pass:
            print("SUCCESS: All features appear scale-independent!")
        else:
            print("WARNING: Some features may have scale issues - review above")
        print("=" * 70)

    return results


def test_quick_training():
    """Test that training works without errors after the changes."""
    print("\n" + "=" * 70)
    print("QUICK TRAINING TEST (SPY)")
    print("=" * 70)

    try:
        predictor = PriceRangePredictor()

        # Use shorter period for quick test
        import yfinance as yf
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 months

        df = yf.download('SPY', start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = df.columns.str.lower()

        print(f"  Training on {len(df)} bars...")

        # Train with minimal optimization for speed
        result = predictor.train_range_model(df, ticker='SPY', optimize=False)

        print(f"\n  Training Results:")
        print(f"    R²: {result.get('r2', 'N/A'):.4f}")
        print(f"    RMSE: {result.get('rmse_pct', 'N/A'):.4f}%")
        print(f"    Features used: {len(predictor.feature_names)}")

        # Make a prediction
        prediction = predictor.predict(df)
        if prediction:
            print(f"\n  Sample Prediction:")
            print(f"    Predicted High: ${prediction.get('predicted_high', 0):.2f}")
            print(f"    Predicted Low: ${prediction.get('predicted_low', 0):.2f}")
            print(f"    Predicted Range: {prediction.get('predicted_range', 0):.2f}%")

        print("\n  TRAINING TEST: PASSED")
        return True

    except Exception as e:
        print(f"\n  TRAINING TEST: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test feature scale independence
    results = test_feature_scale_independence()

    # Quick training test
    test_quick_training()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
