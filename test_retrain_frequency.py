"""
Test: Does daily retraining help or hurt prediction performance?

Compares:
1. Daily retraining (current default in walk-forward)
2. Weekly retraining (every 5 days)
3. Monthly retraining (every 21 days)
4. No retraining (train once, predict all)

Parallelized using joblib (same as walk-forward analysis).
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import r2_score, mean_absolute_error
import time
import warnings
import sys
import io
import os
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


def run_walkforward(df, retrain_interval, n_test_days=60, n_train_days=180, n_trials=30, n_workers=4):
    """Run walk-forward with specified retrain interval."""
    # Import inside function for multiprocessing
    from price_prediction import PriceRangePredictor
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results = []
    train_times = []
    current_model = None
    last_train_idx = -999

    test_start_idx = len(df) - n_test_days

    for i, test_idx in enumerate(range(test_start_idx, len(df))):
        days_since_train = test_idx - last_train_idx
        need_retrain = (current_model is None) or (days_since_train >= retrain_interval)

        if need_retrain:
            train_start = max(0, test_idx - n_train_days)
            train_df = df.iloc[train_start:test_idx].copy()

            if len(train_df) >= 100:
                start_time = time.time()

                with SuppressOutput():
                    predictor = PriceRangePredictor()
                    predictor.train_range_model(train_df, n_trials=n_trials, n_workers=n_workers)

                train_time = time.time() - start_time
                train_times.append(train_time)

                current_model = predictor
                last_train_idx = test_idx

        if current_model is not None:
            pred_df = df.iloc[:test_idx].copy()
            try:
                with SuppressOutput():
                    pred = current_model.predict_daily_range(pred_df, confidence_level=0.9)

                actual_high = df['high'].iloc[test_idx]
                actual_low = df['low'].iloc[test_idx]

                results.append({
                    'date': df.index[test_idx],
                    'pred_high': pred['predicted_high'],
                    'pred_low': pred['predicted_low'],
                    'actual_high': actual_high,
                    'actual_low': actual_low,
                    'retrained': need_retrain
                })
            except Exception:
                pass

    return pd.DataFrame(results), train_times


def analyze_results(results_df, train_times, name):
    """Analyze walk-forward results."""
    if len(results_df) == 0:
        return None

    high_errors = results_df['pred_high'] - results_df['actual_high']
    low_errors = results_df['pred_low'] - results_df['actual_low']

    high_mae = np.abs(high_errors).mean()
    low_mae = np.abs(low_errors).mean()

    prev_close = results_df['actual_high'].shift(1).fillna(results_df['actual_high'].iloc[0])
    actual_high_dev = (results_df['actual_high'] - prev_close) / prev_close * 100
    pred_high_dev = (results_df['pred_high'] - prev_close) / prev_close * 100

    high_r2 = r2_score(actual_high_dev, pred_high_dev) if actual_high_dev.std() > 0 else 0

    pred_high_changes = results_df['pred_high'].diff().abs()
    actual_high_changes = results_df['actual_high'].diff().abs()
    stability_ratio = pred_high_changes.mean() / (actual_high_changes.mean() + 0.001)

    n_retrains = results_df['retrained'].sum()
    total_train_time = sum(train_times)

    return {
        'name': name,
        'high_mae': high_mae,
        'low_mae': low_mae,
        'high_r2': high_r2,
        'stability_ratio': stability_ratio,
        'n_retrains': n_retrains,
        'total_train_time': total_train_time,
        'n_predictions': len(results_df)
    }


def run_single_config(df, interval, name, n_test_days, n_train_days, n_trials, n_workers):
    """Run a single config - for parallel execution."""
    start = time.time()
    results_df, train_times = run_walkforward(
        df, interval, n_test_days, n_train_days, n_trials, n_workers
    )
    analysis = analyze_results(results_df, train_times, name)
    elapsed = time.time() - start

    if analysis:
        analysis['elapsed'] = elapsed
    return analysis


def main():
    # Detect CPU cores
    n_cores = os.cpu_count() or 4
    print(f"RETRAIN FREQUENCY TEST - {n_cores} CPU cores detected")
    print("Loading data...")

    # Download data
    df = yf.download("SPY", period="2y", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    print(f"Data: {len(df)} bars, testing 60 days out-of-sample\n")

    # Test configurations
    configs = [
        (1, "DAILY"),
        (5, "WEEKLY"),
        (21, "MONTHLY"),
        (999, "NEVER"),
    ]

    # Calculate workers per config to fully utilize all cores
    # Each config runs in parallel, and each config's Optuna uses workers_per_config
    n_parallel_configs = len(configs)
    workers_per_config = max(1, n_cores // n_parallel_configs)

    print(f"Running {n_parallel_configs} configs in parallel using joblib")
    print(f"Each config uses {workers_per_config} Optuna workers")
    print(f"Total core utilization: {n_parallel_configs * workers_per_config}/{n_cores} cores\n")

    # Run all configs in parallel using joblib (same as walk-forward)
    start_total = time.time()

    all_results = Parallel(n_jobs=n_parallel_configs, backend='loky', verbose=0)(
        delayed(run_single_config)(
            df, interval, name, 60, 180, 30, workers_per_config
        )
        for interval, name in configs
    )

    # Filter out None results
    all_results = [r for r in all_results if r is not None]

    total_time = time.time() - start_total
    print(f"\nTotal wall time: {total_time:.0f}s (parallel)\n")

    # Sort by config order
    config_order = {name: i for i, (_, name) in enumerate(configs)}
    all_results.sort(key=lambda x: config_order.get(x['name'], 999))

    # Summary table
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Config':<12} {'R²':<10} {'MAE ($)':<10} {'Stability':<12} {'Retrains':<10} {'Time':<10}")
    print("-" * 80)

    for r in all_results:
        print(f"{r['name']:<12} {r['high_r2']:<10.4f} {r['high_mae']:<10.2f} {r['stability_ratio']:<12.2f} {r['n_retrains']:<10} {r['total_train_time']:<10.0f}s")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if len(all_results) >= 2:
        best_r2 = max(all_results, key=lambda x: x['high_r2'])
        best_stability = min(all_results, key=lambda x: x['stability_ratio'])
        fastest = min(all_results, key=lambda x: x['total_train_time'])

        print(f"\nBest Accuracy (R²):    {best_r2['name']} = {best_r2['high_r2']:.4f}")
        print(f"Best Stability:        {best_stability['name']} = {best_stability['stability_ratio']:.2f}")
        print(f"Fastest:               {fastest['name']} = {fastest['total_train_time']:.0f}s")

        daily = next((r for r in all_results if r['name'] == 'DAILY'), None)
        weekly = next((r for r in all_results if r['name'] == 'WEEKLY'), None)

        if daily and weekly:
            r2_diff = weekly['high_r2'] - daily['high_r2']
            speedup = daily['total_train_time'] / weekly['total_train_time'] if weekly['total_train_time'] > 0 else 0

            print(f"\nWEEKLY vs DAILY:")
            print(f"  R² change:    {r2_diff:+.4f} ({'better' if r2_diff > 0 else 'worse'})")
            print(f"  Speedup:      {speedup:.1f}x faster")
            print(f"  Stability:    {weekly['stability_ratio']:.2f} vs {daily['stability_ratio']:.2f}")

        # Recommendation
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)

        if daily and weekly and weekly['high_r2'] >= daily['high_r2'] - 0.02:
            print(f"\nUse WEEKLY retraining - similar accuracy, {speedup:.1f}x faster")
        elif best_r2['name'] == 'DAILY':
            print(f"\nUse DAILY retraining - best accuracy ({best_r2['high_r2']:.4f})")
        else:
            print(f"\nUse {best_r2['name']} retraining - best accuracy ({best_r2['high_r2']:.4f})")


if __name__ == "__main__":
    main()
