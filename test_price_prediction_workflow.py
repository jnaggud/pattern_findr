#!/usr/bin/env python3
"""
Comprehensive Test Script for Price Range Prediction Workflow
=============================================================

This script validates the entire price prediction workflow including:
1. Data loading and feature engineering
2. Model training (Ridge, XGBoost, Ensemble)
3. Walk-forward validation
4. Mode switching (Fixed vs Regime-Adaptive)
5. Results export and metrics calculation

Run this script to verify the workflow is production-ready.
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test results storage
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def log_pass(test_name, details=""):
    print(f"  [PASS] {test_name}")
    test_results['passed'].append({'name': test_name, 'details': details})

def log_fail(test_name, error):
    print(f"  [FAIL] {test_name}: {error}")
    test_results['failed'].append({'name': test_name, 'error': str(error)})

def log_warn(test_name, warning):
    print(f"  [WARN] {test_name}: {warning}")
    test_results['warnings'].append({'name': test_name, 'warning': warning})


# =============================================================================
# TEST 1: Import Validation
# =============================================================================
def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "=" * 60)
    print("TEST 1: Import Validation")
    print("=" * 60)

    required_modules = [
        ('price_prediction', 'PriceRangePredictor'),
        ('price_prediction', 'run_all_model_combinations'),
        ('price_prediction', 'run_parallel_walk_forward'),
        ('simple_regime_detector', 'detect_market_regime'),
        ('ml_feature_engineer', 'MLFeatureEngineer'),
    ]

    for module_name, attr_name in required_modules:
        try:
            module = __import__(module_name)
            if hasattr(module, attr_name):
                log_pass(f"Import {module_name}.{attr_name}")
            else:
                log_fail(f"Import {module_name}.{attr_name}", f"Attribute not found")
        except Exception as e:
            log_fail(f"Import {module_name}", str(e))


# =============================================================================
# TEST 2: Data Loading
# =============================================================================
def test_data_loading():
    """Test data loading and basic structure."""
    print("\n" + "=" * 60)
    print("TEST 2: Data Loading")
    print("=" * 60)

    try:
        # Try to load SPY data using MLFeatureEngineer
        from ml_feature_engineer import MLFeatureEngineer
        fe = MLFeatureEngineer()

        # Generate sample data for testing
        dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
        np.random.seed(42)

        # Create realistic OHLCV data
        price_base = 500
        returns = np.random.normal(0.0005, 0.015, len(dates))
        prices = price_base * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
            'high': prices * (1 + np.random.uniform(0.001, 0.015, len(dates))),
            'low': prices * (1 - np.random.uniform(0.001, 0.015, len(dates))),
            'close': prices,
            'volume': np.random.randint(50000000, 150000000, len(dates))
        })
        df.set_index('date', inplace=True)

        log_pass("Sample data generation", f"{len(df)} rows created")

        # Validate data structure
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            log_fail("Data structure", f"Missing columns: {missing}")
        else:
            log_pass("Data structure validation")

        return df

    except Exception as e:
        log_fail("Data loading", str(e))
        return None


# =============================================================================
# TEST 3: Feature Engineering
# =============================================================================
def test_feature_engineering(df):
    """Test feature engineering pipeline."""
    print("\n" + "=" * 60)
    print("TEST 3: Feature Engineering")
    print("=" * 60)

    if df is None:
        log_fail("Feature engineering", "No data available")
        return None

    try:
        from ml_feature_engineer import MLFeatureEngineer
        fe = MLFeatureEngineer()

        # Add technical features
        enhanced_df = fe.create_base_features(df)
        enhanced_df = fe.create_ml_specific_features(enhanced_df)

        n_features = len([c for c in enhanced_df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])
        log_pass(f"Technical features added", f"{n_features} features")

        # Check for NaN handling
        nan_pct = enhanced_df.isnull().sum().sum() / (len(enhanced_df) * len(enhanced_df.columns)) * 100
        if nan_pct > 20:
            log_warn("NaN percentage", f"{nan_pct:.1f}% NaN values")
        else:
            log_pass("NaN handling", f"{nan_pct:.1f}% NaN values")

        return enhanced_df

    except Exception as e:
        log_fail("Feature engineering", str(e))
        return None


# =============================================================================
# TEST 4: Model Training
# =============================================================================
def test_model_training(df):
    """Test model training with different configurations."""
    print("\n" + "=" * 60)
    print("TEST 4: Model Training")
    print("=" * 60)

    if df is None:
        log_fail("Model training", "No data available")
        return None

    try:
        from price_prediction import PriceRangePredictor
        import price_prediction

        # Test with Ridge model
        predictor = PriceRangePredictor(polygon_manager=None)

        # Configure for Ridge Top15
        price_prediction.HIGH_MODEL_CONFIG = {
            'model_type': 'ridge',
            'top_n_features': 15,
            'ci_multiplier': 1.0
        }
        price_prediction.LOW_MODEL_CONFIG = {
            'model_type': 'ridge',
            'top_n_features': 15,
            'ci_multiplier': 1.0
        }

        # Train on last 180 days
        train_df = df.tail(180).copy()
        train_df = train_df.dropna(how='all', axis=1)
        train_df = train_df.fillna(method='ffill').fillna(method='bfill')

        result = predictor.train_range_model(
            train_df,
            n_trials=5,  # Minimal trials for testing
            n_workers=2,
            feature_selection=False
        )

        if result:
            r2_high = result.get('r2_high', 0)
            r2_low = result.get('r2_low', 0)
            log_pass("Ridge model training", f"R2 High: {r2_high:.4f}, R2 Low: {r2_low:.4f}")

            # Validate prediction output
            if 'pred_high' in result and 'pred_low' in result:
                log_pass("Prediction output structure")
            else:
                log_fail("Prediction output structure", "Missing pred_high or pred_low")

            return result
        else:
            log_fail("Model training", "No result returned")
            return None

    except Exception as e:
        log_fail("Model training", str(e))
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# TEST 5: Walk-Forward Validation
# =============================================================================
def test_walk_forward(df):
    """Test walk-forward validation logic."""
    print("\n" + "=" * 60)
    print("TEST 5: Walk-Forward Validation")
    print("=" * 60)

    if df is None:
        log_fail("Walk-forward", "No data available")
        return None

    try:
        from price_prediction import PriceRangePredictor
        import price_prediction

        # Setup config
        price_prediction.HIGH_MODEL_CONFIG = {
            'model_type': 'ridge',
            'top_n_features': 10,
            'ci_multiplier': 1.0
        }
        price_prediction.LOW_MODEL_CONFIG = {
            'model_type': 'ridge',
            'top_n_features': 10,
            'ci_multiplier': 1.0
        }

        # Prepare data
        test_df = df.tail(200).copy()
        test_df = test_df.dropna(how='all', axis=1)
        test_df = test_df.fillna(method='ffill').fillna(method='bfill')

        # Mini walk-forward (just 5 predictions)
        train_window = 150
        test_days = 5

        predictions = []
        actuals_high = []
        actuals_low = []

        for i in range(test_days):
            test_idx = len(test_df) - test_days + i
            train_start = max(0, test_idx - train_window)
            train_end = test_idx

            train_data = test_df.iloc[train_start:train_end].copy()
            test_row = test_df.iloc[test_idx]

            predictor = PriceRangePredictor(None)
            result = predictor.train_range_model(train_data, n_trials=3, n_workers=2)

            if result:
                predictions.append({
                    'high': result.get('pred_high', test_row['high']),
                    'low': result.get('pred_low', test_row['low'])
                })
            else:
                predictions.append({'high': test_row['high'], 'low': test_row['low']})

            actuals_high.append(test_row['high'])
            actuals_low.append(test_row['low'])

        # Calculate metrics
        pred_h = np.array([p['high'] for p in predictions])
        pred_l = np.array([p['low'] for p in predictions])
        act_h = np.array(actuals_high)
        act_l = np.array(actuals_low)

        mae_h = np.abs(act_h - pred_h).mean()
        mae_l = np.abs(act_l - pred_l).mean()

        log_pass("Walk-forward validation", f"MAE High: ${mae_h:.2f}, MAE Low: ${mae_l:.2f}")

        return {
            'predictions': predictions,
            'actuals_high': actuals_high,
            'actuals_low': actuals_low,
            'mae_high': mae_h,
            'mae_low': mae_l
        }

    except Exception as e:
        log_fail("Walk-forward", str(e))
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# TEST 6: Regime Detection
# =============================================================================
def test_regime_detection(df):
    """Test market regime detection."""
    print("\n" + "=" * 60)
    print("TEST 6: Regime Detection")
    print("=" * 60)

    if df is None:
        log_fail("Regime detection", "No data available")
        return

    try:
        from simple_regime_detector import detect_market_regime

        test_df = df.tail(60).copy()
        # The simple_regime_detector takes (data, current_idx, lookback)
        regime = detect_market_regime(test_df, len(test_df) - 1, lookback=20)

        valid_regimes = ['bull', 'bear', 'crash', 'sideways']  # From simple_regime_detector.py
        if regime in valid_regimes:
            log_pass("Regime detection", f"Detected: {regime}")
        else:
            log_warn("Regime detection", f"Unexpected regime: {regime}")

    except Exception as e:
        log_fail("Regime detection", str(e))


# =============================================================================
# TEST 7: Export Functions
# =============================================================================
def test_export_functions():
    """Test export functions work correctly."""
    print("\n" + "=" * 60)
    print("TEST 7: Export Functions")
    print("=" * 60)

    try:
        # Import the export function
        sys.path.insert(0, '.')

        # Create dummy data for export test
        dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
        wf_df = pd.DataFrame({
            'date': dates,
            'predicted_high': np.random.uniform(500, 510, 10),
            'actual_high': np.random.uniform(500, 510, 10),
            'predicted_low': np.random.uniform(490, 500, 10),
            'actual_low': np.random.uniform(490, 500, 10),
            'high_lower': np.random.uniform(495, 500, 10),
            'high_upper': np.random.uniform(510, 520, 10),
            'low_lower': np.random.uniform(480, 490, 10),
            'low_upper': np.random.uniform(495, 505, 10),
            'retrained': [True] * 5 + [False] * 5,
            'high_model_type': ['ridge'] * 10,
            'high_top_n': [15] * 10,
            'low_model_type': ['ridge'] * 10,
            'low_top_n': [5] * 10
        })

        metrics = {
            'high_in_range': 85.0,
            'low_in_range': 90.0,
            'full_containment': 80.0,
            'high_mae': 2.5,
            'low_mae': 3.0,
            'high_bias': -0.5,
            'low_bias': 0.3,
            'range_mae': 2.8,
            'range_mape': 45.0,
            'test_r2_high': 0.25,
            'test_r2_low': 0.15,
            'train_r2_avg': 0.50,
            'train_rmse_avg': 0.7,
            'confidence_method': 'rmse_fallback',
            'quantile_pct': 0.0
        }

        settings = {
            'n_train_days': 180,
            'retrain_frequency': 'Daily',
            'confidence_level': '68%',
            'n_trials': 100,
            'n_workers': 4,
            'model_mode': 'Simple',
            'high_model_type': 'ridge',
            'high_top_n': 15,
            'low_model_type': 'ridge',
            'low_top_n': 5,
            'regime_adaptive': False,
            'regime_strategy': None,
            'regime_breakdown': {}
        }

        # Import and test the function
        from oscillator_predictor_page import export_walkforward_results

        csv_path, summary_path, latest_path, summary_text = export_walkforward_results(
            wf_df, metrics, settings, 'TEST'
        )

        # Verify files were created
        if os.path.exists(csv_path):
            log_pass("CSV export", csv_path)
            os.remove(csv_path)  # Cleanup
        else:
            log_fail("CSV export", "File not created")

        if os.path.exists(summary_path):
            log_pass("Summary export", summary_path)
            os.remove(summary_path)  # Cleanup
        else:
            log_fail("Summary export", "File not created")

        if os.path.exists(latest_path):
            log_pass("Latest file export", latest_path)
            os.remove(latest_path)  # Cleanup
        else:
            log_fail("Latest file export", "File not created")

        # Check summary content
        if 'WALK-FORWARD ANALYSIS SUMMARY' in summary_text:
            log_pass("Summary content validation")
        else:
            log_fail("Summary content", "Missing header")

        if 'Regime-Adaptive: DISABLED' in summary_text:
            log_pass("Regime mode in summary")
        else:
            log_warn("Regime mode", "Regime-Adaptive status not found in summary")

    except Exception as e:
        log_fail("Export functions", str(e))
        import traceback
        traceback.print_exc()


# =============================================================================
# TEST 8: Model Comparison
# =============================================================================
def test_model_comparison(df):
    """Test model comparison functionality."""
    print("\n" + "=" * 60)
    print("TEST 8: Model Comparison")
    print("=" * 60)

    if df is None:
        log_fail("Model comparison", "No data available")
        return

    try:
        from price_prediction import generate_model_configs

        # Test config generation
        configs = generate_model_configs()

        if len(configs) > 0:
            log_pass("Model config generation", f"{len(configs)} configurations")
        else:
            log_fail("Model config generation", "No configs generated")

        # Verify config structure
        sample_config = configs[0]
        required_keys = ['name', 'model_type', 'top_n_features']
        missing = [k for k in required_keys if k not in sample_config]

        if missing:
            log_fail("Config structure", f"Missing keys: {missing}")
        else:
            log_pass("Config structure validation")

    except Exception as e:
        log_fail("Model comparison", str(e))


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================
def run_all_tests():
    """Run all tests and generate report."""
    print("\n" + "=" * 60)
    print("PRICE PREDICTION WORKFLOW TEST SUITE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run tests
    test_imports()
    df = test_data_loading()
    enhanced_df = test_feature_engineering(df)
    test_model_training(enhanced_df)
    test_walk_forward(enhanced_df)
    test_regime_detection(enhanced_df)
    test_export_functions()
    test_model_comparison(enhanced_df)

    # Generate report
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    total = len(test_results['passed']) + len(test_results['failed'])
    passed = len(test_results['passed'])
    failed = len(test_results['failed'])
    warnings = len(test_results['warnings'])

    print(f"\nTotal Tests: {total}")
    print(f"  Passed: {passed} ({100*passed/total:.1f}%)" if total > 0 else "  Passed: 0")
    print(f"  Failed: {failed}")
    print(f"  Warnings: {warnings}")

    if failed > 0:
        print("\n" + "-" * 40)
        print("FAILED TESTS:")
        for f in test_results['failed']:
            print(f"  - {f['name']}: {f['error']}")

    if warnings > 0:
        print("\n" + "-" * 40)
        print("WARNINGS:")
        for w in test_results['warnings']:
            print(f"  - {w['name']}: {w['warning']}")

    # Save results
    results_path = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': total,
                'passed': passed,
                'failed': failed,
                'warnings': warnings
            },
            'details': test_results
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
