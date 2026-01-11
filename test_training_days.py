"""
Test: How many training days produces the best walk-forward performance?

Compares different training window sizes to find optimal balance between
having enough data and using recent/relevant data.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import r2_score
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
os.environ['QUIET_WORKERS'] = '1'


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


def run_walkforward(df, n_train_days, n_test_days=60, n_trials=50, n_workers=4):
    """Run walk-forward with specified training window size."""
    from price_prediction import PriceRangePredictor
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results = []
    current_model = None
    last_train_idx = -999

    test_start_idx = len(df) - n_test_days

    for test_idx in range(test_start_idx, len(df)):
        days_since_train = test_idx - last_train_idx
        need_retrain = (current_model is None) or (days_since_train >= 1)  # Daily retrain

        if need_retrain:
            train_start = max(0, test_idx - n_train_days)
            train_df = df.iloc[train_start:test_idx].copy()

            if len(train_df) >= 100:
                with SuppressOutput():
                    predictor = PriceRangePredictor()
                    predictor.train_range_model(train_df, n_trials=n_trials, n_workers=n_workers)

                current_model = predictor
                last_train_idx = test_idx

        if current_model is not None:
            pred_df = df.iloc[:test_idx].copy()
            try:
                with SuppressOutput():
                    pred = current_model.predict_daily_range(pred_df, confidence_level=0.9)

                actual_high = df['high'].iloc[test_idx]
                actual_low = df['low'].iloc[test_idx]
                actual_close = df['close'].iloc[test_idx - 1] if test_idx > 0 else df['close'].iloc[0]

                results.append({
                    'date': df.index[test_idx],
                    'pred_high': pred['predicted_high'],
                    'pred_low': pred['predicted_low'],
                    'actual_high': actual_high,
                    'actual_low': actual_low,
                    'actual_close': actual_close,
                    'high_upper': pred.get('ci_high_upper', pred['predicted_high']),
                    'high_lower': pred.get('ci_high_lower', pred['predicted_high']),
                    'low_upper': pred.get('ci_low_upper', pred['predicted_low']),
                    'low_lower': pred.get('ci_low_lower', pred['predicted_low']),
                })
            except Exception:
                pass

    return pd.DataFrame(results)


def analyze_results(results_df, name, n_train_days):
    """Analyze walk-forward results."""
    if len(results_df) == 0:
        return None

    # Calculate R² on deviations (same as Streamlit)
    prev_close = results_df['actual_close']

    actual_high_dev = (results_df['actual_high'] - prev_close) / prev_close * 100
    actual_low_dev = (prev_close - results_df['actual_low']) / prev_close * 100

    pred_high_dev = (results_df['pred_high'] - prev_close) / prev_close * 100
    pred_low_dev = (prev_close - results_df['pred_low']) / prev_close * 100

    # R² calculation
    ss_res_high = ((actual_high_dev - pred_high_dev) ** 2).sum()
    ss_tot_high = ((actual_high_dev - actual_high_dev.mean()) ** 2).sum()
    test_r2_high = 1 - (ss_res_high / ss_tot_high) if ss_tot_high > 0 else 0

    ss_res_low = ((actual_low_dev - pred_low_dev) ** 2).sum()
    ss_tot_low = ((actual_low_dev - actual_low_dev.mean()) ** 2).sum()
    test_r2_low = 1 - (ss_res_low / ss_tot_low) if ss_tot_low > 0 else 0

    # MAE
    high_mae = np.abs(results_df['pred_high'] - results_df['actual_high']).mean()
    low_mae = np.abs(results_df['pred_low'] - results_df['actual_low']).mean()

    # Containment (how often actual falls within confidence bands)
    high_in_range = ((results_df['actual_high'] >= results_df['high_lower']) &
                     (results_df['actual_high'] <= results_df['high_upper'])).mean() * 100
    low_in_range = ((results_df['actual_low'] >= results_df['low_lower']) &
                    (results_df['actual_low'] <= results_df['low_upper'])).mean() * 100

    return {
        'name': name,
        'n_train_days': n_train_days,
        'test_r2_high': test_r2_high,
        'test_r2_low': test_r2_low,
        'high_mae': high_mae,
        'low_mae': low_mae,
        'high_containment': high_in_range,
        'low_containment': low_in_range,
        'n_predictions': len(results_df)
    }


def run_single_config(df, n_train_days, name, n_test_days, n_trials, n_workers):
    """Run a single config - for parallel execution."""
    start = time.time()
    results_df = run_walkforward(df, n_train_days, n_test_days, n_trials, n_workers)
    analysis = analyze_results(results_df, name, n_train_days)
    elapsed = time.time() - start

    if analysis:
        analysis['elapsed'] = elapsed
    return analysis


def main():
    n_cores = os.cpu_count() or 4
    print(f"TRAINING DAYS COMPARISON - {n_cores} CPU cores detected")
    print("Loading data...")

    # Download data - need enough for longest training window + test period
    df = yf.download("SPY", period="3y", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    print(f"Data: {len(df)} bars, testing 60 days out-of-sample\n")

    # Test configurations - different training window sizes
    configs = [
        (120, "120_DAYS"),
        (180, "180_DAYS"),
        (250, "250_DAYS"),
        (365, "365_DAYS"),
        (500, "500_DAYS"),
    ]

    n_parallel_configs = len(configs)
    workers_per_config = max(1, n_cores // n_parallel_configs)

    print(f"Running {n_parallel_configs} configs in parallel")
    print(f"Each config uses {workers_per_config} Optuna workers\n")

    start_total = time.time()

    all_results = Parallel(n_jobs=n_parallel_configs, backend='loky', verbose=0)(
        delayed(run_single_config)(
            df, n_train_days, name, 60, 50, workers_per_config
        )
        for n_train_days, name in configs
    )

    all_results = [r for r in all_results if r is not None]

    total_time = time.time() - start_total
    print(f"Total wall time: {total_time:.0f}s\n")

    # Sort by config order
    config_order = {name: i for i, (_, name) in enumerate(configs)}
    all_results.sort(key=lambda x: config_order.get(x['name'], 999))

    # Summary table
    print("=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Config':<12} {'R² High':<10} {'R² Low':<10} {'MAE High':<10} {'MAE Low':<10} {'High %':<10} {'Low %':<10}")
    print("-" * 90)

    for r in all_results:
        print(f"{r['name']:<12} {r['test_r2_high']:<10.4f} {r['test_r2_low']:<10.4f} "
              f"${r['high_mae']:<9.2f} ${r['low_mae']:<9.2f} "
              f"{r['high_containment']:<10.1f} {r['low_containment']:<10.1f}")

    # Key findings
    print("\n" + "=" * 90)
    print("KEY FINDINGS")
    print("=" * 90)

    if len(all_results) >= 2:
        best_high_r2 = max(all_results, key=lambda x: x['test_r2_high'])
        best_low_r2 = max(all_results, key=lambda x: x['test_r2_low'])
        best_containment = max(all_results, key=lambda x: (x['high_containment'] + x['low_containment']) / 2)

        print(f"\nBest High R²:      {best_high_r2['name']} = {best_high_r2['test_r2_high']:.4f}")
        print(f"Best Low R²:       {best_low_r2['name']} = {best_low_r2['test_r2_low']:.4f}")
        print(f"Best Containment:  {best_containment['name']} = {(best_containment['high_containment'] + best_containment['low_containment'])/2:.1f}%")

        # Compare 180 vs 500 specifically
        d180 = next((r for r in all_results if r['name'] == '180_DAYS'), None)
        d500 = next((r for r in all_results if r['name'] == '500_DAYS'), None)

        if d180 and d500:
            print(f"\n180 DAYS vs 500 DAYS:")
            print(f"  High R² diff: {d180['test_r2_high'] - d500['test_r2_high']:+.4f} ({'180 better' if d180['test_r2_high'] > d500['test_r2_high'] else '500 better'})")
            print(f"  Low R² diff:  {d180['test_r2_low'] - d500['test_r2_low']:+.4f} ({'180 better' if d180['test_r2_low'] > d500['test_r2_low'] else '500 better'})")
            print(f"  Containment:  {(d180['high_containment']+d180['low_containment'])/2:.1f}% vs {(d500['high_containment']+d500['low_containment'])/2:.1f}%")

        # Recommendation
        print("\n" + "=" * 90)
        print("RECOMMENDATION")
        print("=" * 90)

        # Pick best overall (weighted toward R² high since that's more important)
        best_overall = max(all_results, key=lambda x: x['test_r2_high'] * 0.6 + x['test_r2_low'] * 0.4)
        print(f"\nUse {best_overall['name']} training window")
        print(f"  High R²: {best_overall['test_r2_high']:.4f}")
        print(f"  Low R²: {best_overall['test_r2_low']:.4f}")
        print(f"  Containment: {(best_overall['high_containment'] + best_overall['low_containment'])/2:.1f}%")


if __name__ == "__main__":
    main()
