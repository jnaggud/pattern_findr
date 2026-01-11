"""
Test Script: Model Hypotheses for Range Prediction

This script tests three key hypotheses:
1. Super-features only: Does training ONLY on composite super-features work better?
2. Ridge/Lasso vs XGBoost: Do linear models handle collinearity better?
3. Top-N correlation features: Does limiting to top correlated features help?

Run with: python test_model_hypotheses.py
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

# Local imports
from polygon_manager import PolygonManager
from price_prediction import PriceRangePredictor


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def evaluate_model(y_true, y_pred, model_name):
    """Calculate and return metrics for a model."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {
        'model': model_name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }


def test_super_features_only(X_train, X_test, y_train, y_test, feature_names):
    """
    Test Hypothesis 1: Super-features only model

    Theory: If super_master has 0.71 correlation but low importance,
    training ONLY on super-features should reveal if XGBoost can use them.
    """
    print_header("HYPOTHESIS 1: Super-Features Only")

    # Identify super-features
    super_feature_cols = [f for f in feature_names if f.startswith('super_')]
    print(f"Super-features found: {super_feature_cols}")

    if not super_feature_cols:
        print("ERROR: No super-features found in data!")
        return None

    # Get indices of super-features
    super_indices = [feature_names.index(f) for f in super_feature_cols if f in feature_names]

    if len(super_indices) == 0:
        print("ERROR: Could not find super-feature indices!")
        return None

    # Extract super-features only
    X_train_super = X_train[:, super_indices]
    X_test_super = X_test[:, super_indices]

    print(f"Training shape: {X_train_super.shape} (reduced from {X_train.shape})")

    results = []

    # Test XGBoost on super-features only
    print("\nTraining XGBoost on super-features only...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train_super, y_train)
    y_pred_xgb = xgb_model.predict(X_test_super)
    results.append(evaluate_model(y_test, y_pred_xgb, 'XGBoost (super only)'))

    # Feature importance for super-features
    print("\nSuper-feature importances:")
    for i, (name, imp) in enumerate(zip(super_feature_cols, xgb_model.feature_importances_)):
        print(f"  {name}: {imp:.4f}")

    # Test Ridge on super-features
    print("\nTraining Ridge on super-features only...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_super, y_train)
    y_pred_ridge = ridge_model.predict(X_test_super)
    results.append(evaluate_model(y_test, y_pred_ridge, 'Ridge (super only)'))

    return results


def test_linear_vs_xgboost(X_train, X_test, y_train, y_test, feature_names):
    """
    Test Hypothesis 2: Ridge/Lasso vs XGBoost

    Theory: Linear models with regularization handle collinear features better
    because they can assign weights to multiple correlated features.
    """
    print_header("HYPOTHESIS 2: Linear Models vs XGBoost")

    results = []

    # Test XGBoost (baseline)
    print("\nTraining XGBoost (all features)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    results.append(evaluate_model(y_test, y_pred_xgb, 'XGBoost'))

    # Test Ridge Regression
    print("Training Ridge Regression...")
    for alpha in [0.1, 1.0, 10.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        results.append(evaluate_model(y_test, y_pred, f'Ridge (alpha={alpha})'))

    # Test Lasso Regression
    print("Training Lasso Regression...")
    for alpha in [0.01, 0.1, 1.0]:
        lasso = Lasso(alpha=alpha, max_iter=5000)
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)
        n_nonzero = np.sum(lasso.coef_ != 0)
        results.append(evaluate_model(y_test, y_pred, f'Lasso (alpha={alpha}, {n_nonzero} features)'))

    # Test ElasticNet
    print("Training ElasticNet...")
    enet = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
    enet.fit(X_train, y_train)
    y_pred = enet.predict(X_test)
    n_nonzero = np.sum(enet.coef_ != 0)
    results.append(evaluate_model(y_test, y_pred, f'ElasticNet ({n_nonzero} features)'))

    # Print coefficients for highly correlated features (Ridge)
    ridge_best = Ridge(alpha=1.0)
    ridge_best.fit(X_train, y_train)

    print("\nRidge coefficients for super-features:")
    for f in feature_names:
        if f.startswith('super_'):
            idx = feature_names.index(f)
            print(f"  {f}: {ridge_best.coef_[idx]:.4f}")

    return results


def test_top_correlation_features(X_train, X_test, y_train, y_test, feature_names, correlations):
    """
    Test Hypothesis 3: Top-N correlation features only

    Theory: Using only the top N most correlated features might reduce noise
    and give XGBoost clearer signals.
    """
    print_header("HYPOTHESIS 3: Top Correlation Features Only")

    results = []

    # Sort features by correlation
    sorted_features = correlations.abs().sort_values(ascending=False)

    print("Top 20 features by correlation:")
    for i, (f, corr) in enumerate(sorted_features.head(20).items(), 1):
        print(f"  {i}. {f}: {corr:.4f}")

    # Test different N values
    for n_features in [5, 10, 15, 20, 30, 50]:
        top_n_features = sorted_features.head(n_features).index.tolist()

        # Get indices
        top_indices = [feature_names.index(f) for f in top_n_features if f in feature_names]

        if len(top_indices) < n_features:
            print(f"Warning: Only found {len(top_indices)} of {n_features} features")

        X_train_top = X_train[:, top_indices]
        X_test_top = X_test[:, top_indices]

        # Train XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=min(4, n_features // 2),  # Limit depth for small feature sets
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train_top, y_train)
        y_pred = xgb_model.predict(X_test_top)
        results.append(evaluate_model(y_test, y_pred, f'XGBoost (top {len(top_indices)} corr)'))

        # Train Ridge for comparison
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_top, y_train)
        y_pred_ridge = ridge.predict(X_test_top)
        results.append(evaluate_model(y_test, y_pred_ridge, f'Ridge (top {len(top_indices)} corr)'))

    return results


def run_all_tests():
    """Run all hypothesis tests."""
    print_header("MODEL HYPOTHESIS TESTING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load API key from user_settings.json
    try:
        with open('user_settings.json', 'r') as f:
            settings = json.load(f)
            api_key = settings.get('polygon_api_key', '')
    except:
        api_key = ''

    if not api_key:
        print("ERROR: No Polygon API key found in user_settings.json")
        return []

    # Load data
    print("\nLoading SPY data...")
    pm = PolygonManager(api_key)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    df = pm.get_price_data('SPY', limit=365, start_date=start_date, end_date=end_date)
    print(f"Data loaded: {len(df)} bars")

    # Create predictor and features
    print("\nCreating features...")
    predictor = PriceRangePredictor(pm)
    features = predictor.create_range_features(df)
    targets = predictor.create_targets(df)

    feature_names = predictor.feature_names
    print(f"Features created: {len(feature_names)}")

    # Prepare data
    X = features[feature_names].copy()
    y = targets['next_range_pct'].copy()

    # Remove NaN
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    print(f"Valid samples: {len(X)}")

    # Time series split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train_df, X_test_df = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx].values, y.iloc[split_idx:].values

    print(f"Train: {len(X_train_df)}, Test: {len(X_test_df)}")

    # Compute correlations BEFORE scaling (for interpretability)
    correlations = X_train_df.corrwith(pd.Series(y_train, index=X_train_df.index))

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    # Run all tests
    all_results = []

    # Hypothesis 1: Super-features only
    results1 = test_super_features_only(X_train, X_test, y_train, y_test, feature_names)
    if results1:
        all_results.extend(results1)

    # Hypothesis 2: Linear vs XGBoost
    results2 = test_linear_vs_xgboost(X_train, X_test, y_train, y_test, feature_names)
    if results2:
        all_results.extend(results2)

    # Hypothesis 3: Top correlation features
    results3 = test_top_correlation_features(X_train, X_test, y_train, y_test, feature_names, correlations)
    if results3:
        all_results.extend(results3)

    # Print summary
    print_header("RESULTS SUMMARY")

    # Sort by R²
    all_results_sorted = sorted(all_results, key=lambda x: x['r2'], reverse=True)

    print(f"\n{'Model':<45} {'R²':>10} {'RMSE':>10} {'MAE':>10}")
    print("-" * 75)
    for r in all_results_sorted:
        print(f"{r['model']:<45} {r['r2']:>10.4f} {r['rmse']:>10.4f} {r['mae']:>10.4f}")

    # Best results by category
    print("\n" + "-"*75)
    print("BEST BY CATEGORY:")

    # Best overall
    best = all_results_sorted[0]
    print(f"  Overall Best: {best['model']} (R²={best['r2']:.4f})")

    # Best super-features
    super_results = [r for r in all_results if 'super' in r['model'].lower()]
    if super_results:
        best_super = max(super_results, key=lambda x: x['r2'])
        print(f"  Best Super-Features: {best_super['model']} (R²={best_super['r2']:.4f})")

    # Best linear
    linear_results = [r for r in all_results if any(m in r['model'] for m in ['Ridge', 'Lasso', 'Elastic'])]
    if linear_results:
        best_linear = max(linear_results, key=lambda x: x['r2'])
        print(f"  Best Linear: {best_linear['model']} (R²={best_linear['r2']:.4f})")

    # Best top-N correlation
    top_n_results = [r for r in all_results if 'top' in r['model'].lower() and 'corr' in r['model'].lower()]
    if top_n_results:
        best_top_n = max(top_n_results, key=lambda x: x['r2'])
        print(f"  Best Top-N Corr: {best_top_n['model']} (R²={best_top_n['r2']:.4f})")

    # Save results
    results_df = pd.DataFrame(all_results_sorted)
    output_file = f"hypothesis_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    print_header("TEST COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_results_sorted


if __name__ == "__main__":
    results = run_all_tests()
