"""
Test script to compare model performance with different feature counts.

This tests whether our feature selection settings are optimal.
Parallelized using joblib (same as walk-forward analysis).
"""
import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
import time
import io
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# Set environment variables before importing anything else
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['QUIET_WORKERS'] = '1'  # Suppress worker output for cleaner test output


class SuppressOutput:
    """Context manager to suppress stdout/stderr."""
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def train_with_custom_selection(df, corr_formula, imp_formula, name, n_trials=50):
    """
    Train model with custom feature selection formulas.
    Designed for parallel execution.
    """
    # Import inside function for multiprocessing
    from price_prediction import PriceRangePredictor
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    start_time = time.time()

    with SuppressOutput():
        predictor = PriceRangePredictor()
        features = predictor.create_range_features(df)
        targets = predictor.create_targets(df)

    X = features[predictor.feature_names].copy()
    y = targets['next_range_pct'].copy()

    # Remove NaN
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # Time series split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    n_train = len(X_train)
    total_features = len(predictor.feature_names)

    # Stage 1: Correlation selection with different formulas
    correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)

    if corr_formula == 'old_aggressive':
        max_corr = min(50, max(10, n_train // 10))
    elif corr_formula == 'current':
        max_corr = min(total_features, max(80, n_train // 3))
    elif corr_formula == 'medium':
        max_corr = min(total_features, max(100, n_train // 2))
    elif corr_formula == 'all':
        max_corr = total_features
    else:
        max_corr = total_features

    # Filter by correlation threshold (>= 0.05)
    if corr_formula != 'all':
        significant = correlations[correlations >= 0.05]
        top_features = significant.head(max_corr).index.tolist()
    else:
        top_features = predictor.feature_names

    features_after_corr = len(top_features)

    X_train_corr = X_train[top_features]
    X_test_corr = X_test[top_features]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_corr)
    X_test_scaled = scaler.transform(X_test_corr)

    # Train with Optuna
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'n_jobs': 1,
            'verbosity': 0,
            'objective': 'reg:squarederror'
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_train_scaled, y_train.values)
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test.values, y_pred)
        mape = np.mean(np.abs((y_test.values - y_pred) / (y_test.values + 1e-8)))
        mape_component = max(0, 1 - mape)
        composite = 0.6 * r2 + 0.4 * mape_component

        return -composite

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params['n_jobs'] = 1
    best_params['objective'] = 'reg:squarederror'
    best_params['verbosity'] = 0

    # Train with best params
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train_scaled, y_train.values)

    # Stage 2: Importance-based selection
    if imp_formula == 'all':
        final_features = features_after_corr
        X_test_final = X_test_scaled
    else:
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': top_features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        cumulative_importance = importance_df['importance'].cumsum() / importance_df['importance'].sum()

        if imp_formula == 'old_aggressive':
            n_features_target = (cumulative_importance < 0.90).sum() + 1
            n_selected = max(30, min(n_features_target, 60))
        elif imp_formula == 'current':
            n_features_target = (cumulative_importance < 0.95).sum() + 1
            n_selected = max(60, min(n_features_target, 150))
        elif imp_formula == 'medium':
            n_features_target = (cumulative_importance < 0.97).sum() + 1
            n_selected = max(80, min(n_features_target, 180))
        else:
            n_selected = features_after_corr

        nonzero = (importance_df['importance'] > 0).sum()
        n_selected = min(n_selected, nonzero, features_after_corr)

        selected_features = importance_df.head(n_selected)['feature'].tolist()
        selected_indices = [top_features.index(f) for f in selected_features]

        X_train_selected = X_train_scaled[:, selected_indices]
        X_test_final = X_test_scaled[:, selected_indices]

        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train_selected, y_train.values)

        final_features = n_selected

    # Evaluate
    y_pred = model.predict(X_test_final)

    r2 = r2_score(y_test.values, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))
    mae = mean_absolute_error(y_test.values, y_pred)

    elapsed = time.time() - start_time

    return {
        'name': name,
        'corr_formula': corr_formula,
        'imp_formula': imp_formula,
        'total_features': total_features,
        'after_corr': features_after_corr,
        'final_features': final_features,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'elapsed': elapsed
    }


def main():
    n_cores = os.cpu_count() or 4
    print(f"FEATURE COUNT COMPARISON - {n_cores} CPU cores detected")
    print("Loading data...")

    # Get data
    df = yf.download("SPY", period="2y", progress=False)
    if df.empty:
        print("ERROR: Could not download data")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    print(f"Data: {len(df)} bars\n")

    # Test configurations
    configs = [
        ('old_aggressive', 'old_aggressive', "OLD_AGGRESSIVE"),
        ('current', 'current', "CURRENT"),
        ('medium', 'medium', "MEDIUM"),
        ('all', 'all', "ALL_FEATURES"),
    ]

    # Run all configs in parallel using joblib
    n_parallel = len(configs)
    print(f"Running {n_parallel} configs in parallel using joblib...")
    start_total = time.time()

    results = Parallel(n_jobs=n_parallel, backend='loky', verbose=0)(
        delayed(train_with_custom_selection)(df, corr, imp, name, 50)
        for corr, imp, name in configs
    )

    # Filter out None results
    results = [r for r in results if r is not None]

    total_time = time.time() - start_total
    print(f"\nTotal wall time: {total_time:.0f}s (parallel)\n")

    # Sort by config order
    config_order = {name: i for i, (_, _, name) in enumerate(configs)}
    results.sort(key=lambda x: config_order.get(x['name'], 999))

    # Summary
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Config':<18} {'Features':<22} {'R²':<10} {'RMSE':<10}")
    print("-" * 80)

    for r in results:
        feat_str = f"{r['total_features']}→{r['after_corr']}→{r['final_features']}"
        print(f"{r['name']:<18} {feat_str:<22} {r['r2']:<10.4f} {r['rmse']:.4f}%")

    # Find best
    if results:
        best_r2 = max(results, key=lambda x: x['r2'])
        worst_r2 = min(results, key=lambda x: x['r2'])

        print(f"\n{'='*80}")
        print(f"BEST:  {best_r2['name']} with R²={best_r2['r2']:.4f} ({best_r2['final_features']} features)")
        print(f"WORST: {worst_r2['name']} with R²={worst_r2['r2']:.4f} ({worst_r2['final_features']} features)")

        # Quick recommendation
        print(f"\n{'='*80}")
        print("RECOMMENDATION")
        print("=" * 80)

        old = next((r for r in results if r['name'] == 'OLD_AGGRESSIVE'), None)
        all_feat = next((r for r in results if r['name'] == 'ALL_FEATURES'), None)

        if old and all_feat:
            improvement = all_feat['r2'] - old['r2']
            if improvement > 0:
                print(f"\nALL features improves R² by +{improvement:.4f} over OLD aggressive selection")
            else:
                print(f"\nFeature selection helps - ALL features R² is {improvement:.4f} worse")

    return results


if __name__ == "__main__":
    results = main()
