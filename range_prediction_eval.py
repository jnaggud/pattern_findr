"""
Range Prediction Evaluation Script

PURPOSE: Establish baseline performance and measure improvements.

METRICS TRACKED:
1. Range Prediction:
   - Range MAE (Mean Absolute Error)
   - Range MAPE (Mean Absolute Percentage Error)
   - Range R² (coefficient of determination)

2. High/Low Prediction:
   - High MAE, Low MAE
   - High Bias, Low Bias (systematic over/under prediction)
   - Directional accuracy (did we get the direction right?)

3. Confidence Band Quality:
   - Containment Rate: % of actual values within predicted bands
   - Band Width: Average width of confidence bands
   - Calibration: Is 90% CI actually containing 90%?

4. Practical Trading Metrics:
   - Useful Range %: How often was predicted range actionable?
   - Gap Handling: Performance on gap up/down days

USAGE:
    python range_prediction_eval.py                    # Run full evaluation
    python range_prediction_eval.py --quick            # Quick test (20 days)
    python range_prediction_eval.py --ticker QQQ       # Different ticker
    python range_prediction_eval.py --save baseline_v1 # Save results with name
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from price_prediction import PriceRangePredictor
from market_data_db import MarketDataDB


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    'ticker': 'SPY',
    'test_days': 60,              # Days to test on
    'min_train_days': 252,        # Minimum training window (1 year)
    'retrain_frequency': 5,       # Retrain every N days
    'confidence_level': 0.90,     # Confidence level for bands
    'optuna_trials': 50,          # Trials for hyperparameter tuning
    'years_of_data': 3,           # Years of historical data
}

RESULTS_DIR = 'range_eval_results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_metrics(predictions: List[Dict], actuals: List[Dict]) -> Dict:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        predictions: List of prediction dicts with keys:
            - predicted_high, predicted_low, predicted_range
            - high_lower, high_upper, low_lower, low_upper
            - current_close
        actuals: List of actual dicts with keys:
            - actual_high, actual_low, actual_range
            - actual_open (for gap analysis)

    Returns:
        Dict with all metrics
    """
    n = len(predictions)
    if n == 0:
        return {'error': 'No predictions to evaluate'}

    # Extract arrays for vectorized calculations
    pred_high = np.array([p['predicted_high'] for p in predictions])
    pred_low = np.array([p['predicted_low'] for p in predictions])
    pred_range = np.array([p['predicted_range'] for p in predictions])

    actual_high = np.array([a['actual_high'] for a in actuals])
    actual_low = np.array([a['actual_low'] for a in actuals])
    actual_range = np.array([a['actual_range'] for a in actuals])

    current_close = np.array([p['current_close'] for p in predictions])

    # Confidence bands
    high_lower = np.array([p.get('high_lower', p['predicted_high']) for p in predictions])
    high_upper = np.array([p.get('high_upper', p['predicted_high']) for p in predictions])
    low_lower = np.array([p.get('low_lower', p['predicted_low']) for p in predictions])
    low_upper = np.array([p.get('low_upper', p['predicted_low']) for p in predictions])

    metrics = {}

    # =========================================================================
    # 1. RANGE PREDICTION METRICS
    # =========================================================================

    # Convert to percentage for interpretability
    pred_range_pct = pred_range * 100
    actual_range_pct = actual_range * 100

    # MAE (in percentage points)
    range_errors = pred_range_pct - actual_range_pct
    metrics['range_mae'] = np.mean(np.abs(range_errors))
    metrics['range_mape'] = np.mean(np.abs(range_errors) / (actual_range_pct + 1e-6)) * 100

    # R² for range
    ss_res = np.sum((actual_range_pct - pred_range_pct) ** 2)
    ss_tot = np.sum((actual_range_pct - np.mean(actual_range_pct)) ** 2)
    metrics['range_r2'] = 1 - (ss_res / (ss_tot + 1e-6))

    # Range bias (positive = overestimate, negative = underestimate)
    metrics['range_bias'] = np.mean(range_errors)

    # =========================================================================
    # 2. HIGH PREDICTION METRICS
    # =========================================================================

    high_errors = pred_high - actual_high
    high_errors_pct = (high_errors / current_close) * 100

    metrics['high_mae'] = np.mean(np.abs(high_errors))
    metrics['high_mae_pct'] = np.mean(np.abs(high_errors_pct))
    metrics['high_bias'] = np.mean(high_errors)  # Positive = predicting too high
    metrics['high_bias_pct'] = np.mean(high_errors_pct)

    # R² for high
    ss_res_high = np.sum((actual_high - pred_high) ** 2)
    ss_tot_high = np.sum((actual_high - np.mean(actual_high)) ** 2)
    metrics['high_r2'] = 1 - (ss_res_high / (ss_tot_high + 1e-6))

    # =========================================================================
    # 3. LOW PREDICTION METRICS
    # =========================================================================

    low_errors = pred_low - actual_low
    low_errors_pct = (low_errors / current_close) * 100

    metrics['low_mae'] = np.mean(np.abs(low_errors))
    metrics['low_mae_pct'] = np.mean(np.abs(low_errors_pct))
    metrics['low_bias'] = np.mean(low_errors)  # Positive = predicting too high (missing lows)
    metrics['low_bias_pct'] = np.mean(low_errors_pct)

    # R² for low
    ss_res_low = np.sum((actual_low - pred_low) ** 2)
    ss_tot_low = np.sum((actual_low - np.mean(actual_low)) ** 2)
    metrics['low_r2'] = 1 - (ss_res_low / (ss_tot_low + 1e-6))

    # =========================================================================
    # 4. CONFIDENCE BAND METRICS
    # =========================================================================

    # High containment: actual_high within [high_lower, high_upper]
    high_contained = (actual_high >= high_lower) & (actual_high <= high_upper)
    metrics['high_containment_rate'] = np.mean(high_contained) * 100

    # Low containment: actual_low within [low_lower, low_upper]
    low_contained = (actual_low >= low_lower) & (actual_low <= low_upper)
    metrics['low_containment_rate'] = np.mean(low_contained) * 100

    # Full containment: both high and low within bounds
    full_contained = high_contained & low_contained
    metrics['full_containment_rate'] = np.mean(full_contained) * 100

    # Band width (as % of price)
    high_band_width = (high_upper - high_lower) / current_close * 100
    low_band_width = (low_upper - low_lower) / current_close * 100
    metrics['high_band_width_pct'] = np.mean(high_band_width)
    metrics['low_band_width_pct'] = np.mean(low_band_width)

    # Calibration error: difference from target containment rate
    target_rate = predictions[0].get('confidence_level', 0.90) * 100
    metrics['high_calibration_error'] = metrics['high_containment_rate'] - target_rate
    metrics['low_calibration_error'] = metrics['low_containment_rate'] - target_rate

    # =========================================================================
    # 5. PRACTICAL TRADING METRICS
    # =========================================================================

    # Directional accuracy: Did actual move in predicted direction?
    # If pred_high > close and actual_high > close, correct
    pred_up = pred_high > current_close
    actual_up = actual_high > current_close
    pred_down = pred_low < current_close
    actual_down = actual_low < current_close

    metrics['upside_direction_accuracy'] = np.mean(pred_up == actual_up) * 100
    metrics['downside_direction_accuracy'] = np.mean(pred_down == actual_down) * 100

    # Useful predictions: Range prediction within 50% of actual
    range_within_50pct = np.abs(range_errors) <= (actual_range_pct * 0.5)
    metrics['useful_range_pct'] = np.mean(range_within_50pct) * 100

    # Actual high/low split ratio (for comparison with 55/45 assumption)
    actual_high_move = (actual_high - current_close) / current_close
    actual_low_move = (current_close - actual_low) / current_close
    actual_total_move = actual_high_move + actual_low_move

    # Avoid division by zero
    valid_moves = actual_total_move > 0.0001
    if np.sum(valid_moves) > 0:
        high_ratios = actual_high_move[valid_moves] / actual_total_move[valid_moves]
        metrics['actual_high_ratio_mean'] = np.mean(high_ratios)
        metrics['actual_high_ratio_std'] = np.std(high_ratios)
    else:
        metrics['actual_high_ratio_mean'] = 0.5
        metrics['actual_high_ratio_std'] = 0.0

    # =========================================================================
    # 6. GAP ANALYSIS (if open data available)
    # =========================================================================

    if 'actual_open' in actuals[0]:
        actual_open = np.array([a['actual_open'] for a in actuals])

        # Gap: open vs previous close
        gap_pct = (actual_open - current_close) / current_close * 100

        # Gap up days (> 0.5% gap)
        gap_up_mask = gap_pct > 0.5
        gap_down_mask = gap_pct < -0.5
        no_gap_mask = ~gap_up_mask & ~gap_down_mask

        if np.sum(gap_up_mask) > 0:
            metrics['gap_up_high_mae_pct'] = np.mean(np.abs(high_errors_pct[gap_up_mask]))
            metrics['gap_up_low_mae_pct'] = np.mean(np.abs(low_errors_pct[gap_up_mask]))
            metrics['gap_up_count'] = int(np.sum(gap_up_mask))

        if np.sum(gap_down_mask) > 0:
            metrics['gap_down_high_mae_pct'] = np.mean(np.abs(high_errors_pct[gap_down_mask]))
            metrics['gap_down_low_mae_pct'] = np.mean(np.abs(low_errors_pct[gap_down_mask]))
            metrics['gap_down_count'] = int(np.sum(gap_down_mask))

        if np.sum(no_gap_mask) > 0:
            metrics['no_gap_high_mae_pct'] = np.mean(np.abs(high_errors_pct[no_gap_mask]))
            metrics['no_gap_low_mae_pct'] = np.mean(np.abs(low_errors_pct[no_gap_mask]))
            metrics['no_gap_count'] = int(np.sum(no_gap_mask))

    # =========================================================================
    # 7. SUMMARY STATISTICS
    # =========================================================================

    metrics['n_predictions'] = n
    metrics['avg_actual_range_pct'] = np.mean(actual_range_pct)
    metrics['avg_predicted_range_pct'] = np.mean(pred_range_pct)

    return metrics


# =============================================================================
# WALK-FORWARD EVALUATION
# =============================================================================

def run_walk_forward_evaluation(
    ticker: str,
    test_days: int,
    min_train_days: int,
    retrain_frequency: int,
    confidence_level: float,
    optuna_trials: int,
    years_of_data: int,
    verbose: bool = True
) -> Tuple[Dict, List[Dict], List[Dict]]:
    """
    Run walk-forward evaluation of range predictions.

    Returns:
        Tuple of (metrics_dict, predictions_list, actuals_list)
    """
    print(f"\n{'='*70}")
    print(f"RANGE PREDICTION EVALUATION - {ticker}")
    print(f"{'='*70}")
    print(f"Test Period: {test_days} days")
    print(f"Min Training: {min_train_days} days")
    print(f"Retrain Frequency: Every {retrain_frequency} days")
    print(f"Confidence Level: {confidence_level*100:.0f}%")
    print(f"{'='*70}\n")

    # Load data
    db = MarketDataDB()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_of_data * 365)

    print(f"Loading {ticker} data from {start_date.date()} to {end_date.date()}...")
    df = db.get_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if df is None or len(df) < min_train_days + test_days:
        raise ValueError(f"Insufficient data: got {len(df) if df is not None else 0} days, need {min_train_days + test_days}")

    # Ensure datetime index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    print(f"Loaded {len(df)} days of data")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

    # Split into train/test
    test_start_idx = len(df) - test_days

    predictions = []
    actuals = []
    predictor = None
    last_train_idx = -999  # Force initial training

    print(f"\nRunning walk-forward test on {test_days} days...")
    print("-" * 70)

    for i in range(test_days):
        current_idx = test_start_idx + i

        # Current day's data (what we know at prediction time)
        train_end_idx = current_idx  # Train up to but not including test day
        train_df = df.iloc[:train_end_idx].copy()

        # Next day's actual values (what we're trying to predict)
        if current_idx + 1 >= len(df):
            break  # No more data to evaluate

        next_day = df.iloc[current_idx + 1]
        current_day = df.iloc[current_idx]

        # Retrain if needed
        if i - last_train_idx >= retrain_frequency or predictor is None:
            if verbose:
                print(f"\n[Day {i+1}/{test_days}] Retraining model on {len(train_df)} days...")

            predictor = PriceRangePredictor()

            try:
                predictor.train_range_model(
                    train_df,
                    ticker=ticker,
                    optuna_trials=optuna_trials,
                    train_high_low_models=True,
                    train_quantile_models=True,
                    confidence_levels=[confidence_level]
                )
                last_train_idx = i

                if verbose:
                    r2 = predictor.model_metrics.get('r2', 0)
                    print(f"   Model trained. R² = {r2:.4f}")

            except Exception as e:
                print(f"   Training failed: {e}")
                continue

        # Make prediction
        try:
            # Use data up to current day for prediction
            pred_df = df.iloc[:current_idx + 1].copy()

            pred = predictor.predict_daily_range(
                pred_df,
                confidence_level=confidence_level
            )

            if pred is None:
                if verbose:
                    print(f"   [Day {i+1}] Prediction failed")
                continue

            # Store prediction
            predictions.append({
                'date': df.index[current_idx],
                'predicted_high': pred['predicted_high'],
                'predicted_low': pred['predicted_low'],
                'predicted_range': pred['predicted_range'],
                'current_close': pred['current_close'],
                'high_lower': pred.get('high_lower', pred['predicted_high']),
                'high_upper': pred.get('high_upper', pred['predicted_high']),
                'low_lower': pred.get('low_lower', pred['predicted_low']),
                'low_upper': pred.get('low_upper', pred['predicted_low']),
                'confidence_level': confidence_level,
                'model_r2': pred.get('model_r2', 0)
            })

            # Store actual values
            actual_range = (next_day['high'] - next_day['low']) / current_day['close']
            actuals.append({
                'date': df.index[current_idx + 1],
                'actual_high': next_day['high'],
                'actual_low': next_day['low'],
                'actual_range': actual_range,
                'actual_open': next_day['open']
            })

            if verbose and (i + 1) % 10 == 0:
                print(f"   [Day {i+1}/{test_days}] Pred: ${pred['predicted_low']:.2f}-${pred['predicted_high']:.2f}, "
                      f"Actual: ${next_day['low']:.2f}-${next_day['high']:.2f}")

        except Exception as e:
            if verbose:
                print(f"   [Day {i+1}] Prediction error: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"Completed {len(predictions)} predictions out of {test_days} days")
    print(f"{'='*70}\n")

    # Calculate metrics
    if len(predictions) > 0:
        metrics = calculate_metrics(predictions, actuals)
        metrics['ticker'] = ticker
        metrics['test_days'] = test_days
        metrics['successful_predictions'] = len(predictions)
        metrics['evaluation_date'] = datetime.now().isoformat()
    else:
        metrics = {'error': 'No successful predictions'}

    return metrics, predictions, actuals


# =============================================================================
# REPORTING
# =============================================================================

def print_metrics_report(metrics: Dict):
    """Print a formatted metrics report."""

    print("\n" + "=" * 70)
    print("RANGE PREDICTION EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nTicker: {metrics.get('ticker', 'N/A')}")
    print(f"Predictions: {metrics.get('n_predictions', 0)} / {metrics.get('test_days', 0)} days")
    print(f"Evaluation Date: {metrics.get('evaluation_date', 'N/A')[:10]}")

    print("\n" + "-" * 70)
    print("1. RANGE PREDICTION ACCURACY")
    print("-" * 70)
    print(f"   Range R²:           {metrics.get('range_r2', 0):.4f}")
    print(f"   Range MAE:          {metrics.get('range_mae', 0):.3f}% (avg actual: {metrics.get('avg_actual_range_pct', 0):.3f}%)")
    print(f"   Range MAPE:         {metrics.get('range_mape', 0):.1f}%")
    print(f"   Range Bias:         {metrics.get('range_bias', 0):+.3f}% ({'overestimate' if metrics.get('range_bias', 0) > 0 else 'underestimate'})")

    print("\n" + "-" * 70)
    print("2. HIGH/LOW PREDICTION ACCURACY")
    print("-" * 70)
    print(f"   HIGH R²:            {metrics.get('high_r2', 0):.4f}")
    print(f"   HIGH MAE:           ${metrics.get('high_mae', 0):.2f} ({metrics.get('high_mae_pct', 0):.3f}%)")
    print(f"   HIGH Bias:          {metrics.get('high_bias_pct', 0):+.3f}% ({'too high' if metrics.get('high_bias_pct', 0) > 0 else 'too low'})")
    print()
    print(f"   LOW R²:             {metrics.get('low_r2', 0):.4f}")
    print(f"   LOW MAE:            ${metrics.get('low_mae', 0):.2f} ({metrics.get('low_mae_pct', 0):.3f}%)")
    print(f"   LOW Bias:           {metrics.get('low_bias_pct', 0):+.3f}% ({'missing lows' if metrics.get('low_bias_pct', 0) > 0 else 'overshooting lows'})")

    print("\n" + "-" * 70)
    print("3. CONFIDENCE BAND QUALITY (Target: 90%)")
    print("-" * 70)
    print(f"   HIGH Containment:   {metrics.get('high_containment_rate', 0):.1f}% (calibration error: {metrics.get('high_calibration_error', 0):+.1f}%)")
    print(f"   LOW Containment:    {metrics.get('low_containment_rate', 0):.1f}% (calibration error: {metrics.get('low_calibration_error', 0):+.1f}%)")
    print(f"   FULL Containment:   {metrics.get('full_containment_rate', 0):.1f}%")
    print(f"   HIGH Band Width:    {metrics.get('high_band_width_pct', 0):.2f}%")
    print(f"   LOW Band Width:     {metrics.get('low_band_width_pct', 0):.2f}%")

    print("\n" + "-" * 70)
    print("4. PRACTICAL TRADING METRICS")
    print("-" * 70)
    print(f"   Useful Range %:     {metrics.get('useful_range_pct', 0):.1f}% (within 50% of actual)")
    print(f"   Upside Direction:   {metrics.get('upside_direction_accuracy', 0):.1f}% correct")
    print(f"   Downside Direction: {metrics.get('downside_direction_accuracy', 0):.1f}% correct")

    # Actual split ratio analysis
    actual_ratio = metrics.get('actual_high_ratio_mean', 0.5)
    print(f"\n   Actual High/Low Split: {actual_ratio*100:.1f}% / {(1-actual_ratio)*100:.1f}%")
    print(f"   (Model assumes 55% / 45%)")
    if abs(actual_ratio - 0.55) > 0.05:
        print(f"   ⚠️  Actual ratio differs from assumption by {abs(actual_ratio - 0.55)*100:.1f}%")

    # Gap analysis if available
    if 'gap_up_count' in metrics:
        print("\n" + "-" * 70)
        print("5. GAP DAY PERFORMANCE")
        print("-" * 70)
        if 'gap_up_count' in metrics:
            print(f"   Gap Up Days ({metrics.get('gap_up_count', 0)}):   High MAE {metrics.get('gap_up_high_mae_pct', 0):.3f}%, Low MAE {metrics.get('gap_up_low_mae_pct', 0):.3f}%")
        if 'gap_down_count' in metrics:
            print(f"   Gap Down Days ({metrics.get('gap_down_count', 0)}): High MAE {metrics.get('gap_down_high_mae_pct', 0):.3f}%, Low MAE {metrics.get('gap_down_low_mae_pct', 0):.3f}%")
        if 'no_gap_count' in metrics:
            print(f"   No Gap Days ({metrics.get('no_gap_count', 0)}):   High MAE {metrics.get('no_gap_high_mae_pct', 0):.3f}%, Low MAE {metrics.get('no_gap_low_mae_pct', 0):.3f}%")

    print("\n" + "=" * 70)

    # Overall assessment
    print("\nOVERALL ASSESSMENT:")

    issues = []
    if metrics.get('range_r2', 0) < 0.3:
        issues.append(f"- Low Range R² ({metrics.get('range_r2', 0):.3f}) - model explains <30% of variance")
    if metrics.get('high_containment_rate', 0) < 85:
        issues.append(f"- HIGH bands under-coverage ({metrics.get('high_containment_rate', 0):.1f}% vs 90% target)")
    if metrics.get('low_containment_rate', 0) < 85:
        issues.append(f"- LOW bands under-coverage ({metrics.get('low_containment_rate', 0):.1f}% vs 90% target)")
    if abs(metrics.get('range_bias', 0)) > 0.2:
        issues.append(f"- Systematic range bias ({metrics.get('range_bias', 0):+.3f}%)")
    if metrics.get('low_r2', 0) < metrics.get('high_r2', 0) * 0.8:
        issues.append(f"- LOW model underperforms HIGH (R² {metrics.get('low_r2', 0):.3f} vs {metrics.get('high_r2', 0):.3f})")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ All metrics within acceptable ranges")

    print("=" * 70 + "\n")


def save_results(metrics: Dict, predictions: List[Dict], actuals: List[Dict], name: str = None):
    """Save evaluation results to JSON."""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ticker = metrics.get('ticker', 'UNK')

    if name:
        filename = f"{RESULTS_DIR}/{ticker}_eval_{name}_{timestamp}.json"
    else:
        filename = f"{RESULTS_DIR}/{ticker}_eval_{timestamp}.json"

    results = {
        'metrics': metrics,
        'predictions': [{**p, 'date': str(p['date'])} for p in predictions],
        'actuals': [{**a, 'date': str(a['date'])} for a in actuals],
        'config': DEFAULT_CONFIG
    }

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {filename}")

    # Also save latest as symlink-like file
    latest_file = f"{RESULTS_DIR}/{ticker}_eval_LATEST.json"
    with open(latest_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return filename


def compare_results(file1: str, file2: str):
    """Compare two evaluation results."""

    with open(file1) as f:
        r1 = json.load(f)
    with open(file2) as f:
        r2 = json.load(f)

    m1, m2 = r1['metrics'], r2['metrics']

    print("\n" + "=" * 70)
    print("COMPARISON REPORT")
    print("=" * 70)
    print(f"Baseline: {os.path.basename(file1)}")
    print(f"New:      {os.path.basename(file2)}")
    print("-" * 70)

    key_metrics = [
        ('range_r2', 'Range R²', True),
        ('range_mae', 'Range MAE', False),
        ('high_r2', 'HIGH R²', True),
        ('low_r2', 'LOW R²', True),
        ('high_containment_rate', 'HIGH Containment %', True),
        ('low_containment_rate', 'LOW Containment %', True),
        ('useful_range_pct', 'Useful Range %', True),
    ]

    print(f"\n{'Metric':<25} {'Baseline':>12} {'New':>12} {'Change':>12} {'Better?':>10}")
    print("-" * 70)

    improvements = 0
    regressions = 0

    for key, label, higher_is_better in key_metrics:
        v1 = m1.get(key, 0)
        v2 = m2.get(key, 0)
        change = v2 - v1

        if higher_is_better:
            better = "✓ Yes" if change > 0.01 else ("✗ No" if change < -0.01 else "~")
            if change > 0.01:
                improvements += 1
            elif change < -0.01:
                regressions += 1
        else:
            better = "✓ Yes" if change < -0.01 else ("✗ No" if change > 0.01 else "~")
            if change < -0.01:
                improvements += 1
            elif change > 0.01:
                regressions += 1

        print(f"{label:<25} {v1:>12.4f} {v2:>12.4f} {change:>+12.4f} {better:>10}")

    print("-" * 70)
    print(f"\nSummary: {improvements} improvements, {regressions} regressions")
    print("=" * 70 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Range Prediction Evaluation')
    parser.add_argument('--ticker', type=str, default=DEFAULT_CONFIG['ticker'],
                        help=f"Ticker symbol (default: {DEFAULT_CONFIG['ticker']})")
    parser.add_argument('--test-days', type=int, default=DEFAULT_CONFIG['test_days'],
                        help=f"Number of test days (default: {DEFAULT_CONFIG['test_days']})")
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 20 days and fewer trials')
    parser.add_argument('--save', type=str, default=None,
                        help='Save results with this name')
    parser.add_argument('--compare', nargs=2, metavar=('FILE1', 'FILE2'),
                        help='Compare two result files')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    # Configure based on args
    config = DEFAULT_CONFIG.copy()
    config['ticker'] = args.ticker
    config['test_days'] = args.test_days

    if args.quick:
        config['test_days'] = 20
        config['optuna_trials'] = 20
        config['retrain_frequency'] = 10
        print("Running in QUICK mode (20 days, fewer trials)")

    verbose = args.verbose and not args.quiet

    # Run evaluation
    metrics, predictions, actuals = run_walk_forward_evaluation(
        ticker=config['ticker'],
        test_days=config['test_days'],
        min_train_days=config['min_train_days'],
        retrain_frequency=config['retrain_frequency'],
        confidence_level=config['confidence_level'],
        optuna_trials=config['optuna_trials'],
        years_of_data=config['years_of_data'],
        verbose=verbose
    )

    # Print report
    print_metrics_report(metrics)

    # Save results
    if args.save or True:  # Always save for now
        save_results(metrics, predictions, actuals, args.save or 'baseline')


if __name__ == '__main__':
    main()
