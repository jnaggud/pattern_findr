"""Quick test for R² + Composite objective optimization and Ensemble models."""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

# Test the optimization by running the price prediction with optimization enabled
from price_prediction import PriceRangePredictor

def test_r2_optimization():
    """Test that the R² + Composite optimization and Ensemble runs without errors."""
    print("=" * 60)
    print("Testing R² + Composite Optimization + Ensemble Models")
    print("=" * 60)

    # Get some test data
    print("\n1. Downloading SPY data...")
    df = yf.download("SPY", period="2y", progress=False)
    if df.empty:
        print("ERROR: Could not download data")
        return False

    # Handle yfinance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    print(f"   Downloaded {len(df)} bars")

    # Create predictor
    print("\n2. Creating PriceRangePredictor...")
    predictor = PriceRangePredictor()

    # Run training with optimization (small number of trials for quick test)
    print("\n3. Training with R² + Composite optimization (20 trials, 2 workers)...")
    print("   This should show 'Best Composite Score' instead of 'Best MSE'")
    print("-" * 60)

    try:
        predictor.train_range_model(df, n_trials=20, n_workers=2)
        print("-" * 60)
        print("\n4. Checking model metrics...")
        print(f"   R²: {predictor.model_metrics['r2']:.4f}")
        print(f"   RMSE: {predictor.model_metrics['rmse_pct']:.3f}%")
        print(f"   MAE: {predictor.model_metrics['mae_pct']:.3f}%")

        # Check ensemble models
        print("\n4b. Checking ensemble models...")
        if hasattr(predictor, 'ensemble_models') and len(predictor.ensemble_models) > 0:
            print(f"   Ensemble size: {len(predictor.ensemble_models)} models")
            print("   Ensemble status: OK")
        else:
            print("   WARNING: No ensemble models found!")

        print("\n5. Testing prediction...")
        result = predictor.predict_daily_range(df.tail(2))
        if result is not None:
            # Get the actual keys from the result
            print(f"   Result keys: {list(result.keys())}")
            # Use the actual keys based on the returned dict
            if 'predicted_high' in result:
                print(f"   Predicted High: ${result['predicted_high']:.2f}")
            if 'predicted_low' in result:
                print(f"   Predicted Low: ${result['predicted_low']:.2f}")
            if 'predicted_range_pct' in result:
                print(f"   Predicted Range: {result['predicted_range_pct']:.2f}%")

        print("\n" + "=" * 60)
        print("SUCCESS: R² + Composite optimization working correctly!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_r2_optimization()
    sys.exit(0 if success else 1)
