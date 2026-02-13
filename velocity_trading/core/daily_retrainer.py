"""
Daily parameter retrainer for velocity strategies.

After each trading day, re-optimizes numerical params on a trailing
N-day window using Optuna. This is the key insight: the walk-forward
backtest achieves Sharpe 3.89 because it adapts params daily. Fixed
params go stale within days as market conditions shift.

Usage (standalone):
    python -m velocity_trading.core.daily_retrainer --strategy velocity_ES=F_15m_v10

Usage (from BaseTrader):
    retrainer = DailyRetrainer(config, ticker, interval)
    new_params = retrainer.retrain(df)  # Returns optimized numerical params
"""

import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime
from typing import Dict, Optional

import joblib
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Add parent for imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from velocity_trading.core.backtest_engine import prepare_backtest_arrays, run_backtest


# Which params are numerical and can be re-optimized
OPTIMIZABLE_PARAMS = {
    'stop_loss_pct':        {'type': 'float', 'low': 0.3, 'high': 5.0},
    'take_profit_pct':      {'type': 'float', 'low': 0.3, 'high': 8.0},
    'oversold_threshold':   {'type': 'float', 'low': -0.6, 'high': -0.08},
    'overbought_threshold': {'type': 'float', 'low': 0.08, 'high': 0.6},
    'vel_smoothing':        {'type': 'int',   'low': 1, 'high': 8},
    'extreme_zone_mult':    {'type': 'float', 'low': 1.1, 'high': 2.5},
    'min_hold_bars':        {'type': 'int',   'low': 1, 'high': 4},
    'min_bars_between':     {'type': 'int',   'low': 1, 'high': 6},
}

# Exit toggle params — optimized when optimize_exits=True in config
EXIT_TOGGLE_PARAMS = {
    'exit_on_opposite_signal': [True, False],
    'exit_on_midline_cross':   [True, False],
    'use_accel_exit':          [True, False],
}
ACCEL_EXIT_PARAMS = {
    'accel_exit_type':      {'type': 'categorical', 'choices': ['sign_reversal', 'magnitude', 'both']},
    'accel_exit_threshold': {'type': 'float', 'low': 0.0, 'high': 0.1},
    'accel_exit_min_pnl':   {'type': 'float', 'low': 0.1, 'high': 2.0},
    'accel_exit_lookback':  {'type': 'int',   'low': 1, 'high': 4},
}

# Structural params that never change during retraining
STRUCTURAL_PARAMS = [
    'signal_type', 'require_accel', 'use_wavelet_denoise',
    'use_jerk_confirm', 'oscillator_type',
]
# Exit toggles are structural UNLESS optimize_exits is enabled
EXIT_STRUCTURAL_PARAMS = [
    'exit_on_opposite_signal', 'exit_on_midline_cross', 'use_accel_exit',
]

# Scoring functions for the objective
SCORING_FUNCTIONS = {
    'original': 'return / dd_penalty * sqrt(trades)',
    'pure_return': 'total return only',
}


def _score_backtest(result, scoring='original'):
    """Score a backtest result using the specified scoring function."""
    n_trades = result.get('n_trades', 0)
    if n_trades < 3:
        return float('-inf')
    total_return = result.get('total_return', 0)
    max_dd = result.get('max_drawdown', 100)

    if scoring == 'pure_return':
        return total_return
    # Default: original
    dd_penalty = 1 + max_dd / 100
    trade_bonus = math.sqrt(max(n_trades, 1))
    return total_return / dd_penalty * trade_bonus


def _worker(data_path, worker_id, trials_per_worker, fixed_config, seed_base,
            optimize_exits=False, scoring='original'):
    """Run independent Optuna study in a separate process."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from velocity_trading.core.backtest_engine import prepare_backtest_arrays, run_backtest

    train_df = joblib.load(data_path)

    def objective(trial):
        config = dict(fixed_config)
        for param_name, spec in OPTIMIZABLE_PARAMS.items():
            if param_name in config.get('_pinned', {}):
                config[param_name] = config['_pinned'][param_name]
                continue
            if spec['type'] == 'float':
                config[param_name] = trial.suggest_float(param_name, spec['low'], spec['high'])
            else:
                config[param_name] = trial.suggest_int(param_name, spec['low'], spec['high'])

        # Optimize exit toggles if enabled
        if optimize_exits:
            for toggle_name, choices in EXIT_TOGGLE_PARAMS.items():
                config[toggle_name] = trial.suggest_categorical(toggle_name, choices)
            # If accel exit chosen, also optimize its sub-params
            if config.get('use_accel_exit', False):
                for p_name, p_spec in ACCEL_EXIT_PARAMS.items():
                    if p_spec['type'] == 'categorical':
                        config[p_name] = trial.suggest_categorical(p_name, p_spec['choices'])
                    elif p_spec['type'] == 'float':
                        config[p_name] = trial.suggest_float(p_name, p_spec['low'], p_spec['high'])
                    else:
                        config[p_name] = trial.suggest_int(p_name, p_spec['low'], p_spec['high'])

        try:
            arrays = prepare_backtest_arrays(train_df, config)
            result = run_backtest(
                **arrays,
                stop_loss_pct=config.get('stop_loss_pct', 5.0),
                take_profit_pct=config.get('take_profit_pct', 10.0),
                min_hold_bars=config.get('min_hold_bars', 1),
                min_bars_between=config.get('min_bars_between', 1),
                exit_on_opposite=config.get('exit_on_opposite_signal', True),
                exit_on_midline=config.get('exit_on_midline_cross', False),
                use_accel_exit=config.get('use_accel_exit', False),
                accel_exit_type=config.get('accel_exit_type', 'sign_reversal'),
                accel_exit_threshold=config.get('accel_exit_threshold', 0.0),
                accel_exit_min_pnl=config.get('accel_exit_min_pnl', 0.5),
                accel_exit_lookback=config.get('accel_exit_lookback', 1),
                use_jerk_confirm=config.get('use_jerk_confirm', False),
                jerk_confirm_threshold=config.get('jerk_confirm_threshold', 0.0),
            )
        except Exception:
            return float('-inf')

        return _score_backtest(result, scoring)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed_base + worker_id))
    study.optimize(objective, n_trials=trials_per_worker, n_jobs=1)

    if study.best_trial is None or study.best_value == float('-inf'):
        return {'best_value': float('-inf'), 'best_params': {}, 'worker_id': worker_id}

    return {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'worker_id': worker_id,
    }


class DailyRetrainer:
    """
    Handles nightly parameter re-optimization for a velocity strategy.

    The retrainer keeps the structural params (signal_type, oscillator_type,
    exit toggles) fixed and only re-optimizes numerical params (SL/TP,
    thresholds, smoothing, etc.) on a trailing window of data.
    """

    def __init__(
        self,
        config: Dict,
        ticker: str,
        interval: str,
        n_trials: int = 5000,
        n_workers: int = None,
        train_days: int = 30,
        pinned_params: Dict = None,
    ):
        """
        Args:
            config: Current strategy config dict
            ticker: Trading symbol
            interval: Bar interval
            n_trials: Total Optuna trials across all workers
            n_workers: Parallel processes (default: all CPU cores)
            train_days: Trailing training window in calendar days
            pinned_params: Numerical params to NOT optimize (kept fixed)
        """
        self.config = config
        self.ticker = ticker
        self.interval = interval
        self.n_trials = n_trials
        self.n_workers = n_workers or os.cpu_count() or 8
        self.train_days = train_days
        self.pinned_params = pinned_params or {}
        self.optimize_exits = config.get('retrain_optimize_exits', False)
        self.scoring = config.get('retrain_scoring', 'original')

        # Extract structural config (never changes)
        self.fixed_config = {}
        for key in STRUCTURAL_PARAMS:
            if key in config:
                self.fixed_config[key] = config[key]

        # Exit toggles are structural UNLESS optimize_exits is enabled
        if not self.optimize_exits:
            for key in EXIT_STRUCTURAL_PARAMS:
                if key in config:
                    self.fixed_config[key] = config[key]
            # Also carry over accel exit params if accel exit is enabled
            if config.get('use_accel_exit', False):
                for key in ['accel_exit_type', 'accel_exit_threshold',
                            'accel_exit_min_pnl', 'accel_exit_lookback']:
                    if key in config:
                        self.fixed_config[key] = config[key]

        if config.get('use_jerk_confirm', False):
            if 'jerk_confirm_threshold' in config:
                self.fixed_config['jerk_confirm_threshold'] = config['jerk_confirm_threshold']

        # Store pinned params
        self.fixed_config['_pinned'] = self.pinned_params

        self._last_retrain_date = None
        self._last_best_params = None

    def needs_retrain(self) -> bool:
        """Check if we should retrain (once per trading day)."""
        today = datetime.now().date()
        if self._last_retrain_date is None:
            return True
        return today > self._last_retrain_date

    def retrain(self, df: 'pd.DataFrame') -> Optional[Dict]:
        """
        Re-optimize numerical params on trailing training window.

        Args:
            df: Full DataFrame with composite_smooth column already calculated.
                Must have enough history for the training window.

        Returns:
            Dict of optimized params (only the numerical ones), or None if failed.
        """
        if df is None or len(df) < 200:
            print(f"   [Retrainer] Not enough data ({len(df) if df is not None else 0} bars)")
            return None

        # Use trailing N days as training data
        bars_per_day = 23 * (60 // int(self.interval.replace('m', '').replace('h', '')))
        train_bars = self.train_days * bars_per_day
        train_df = df.iloc[-train_bars:] if len(df) > train_bars else df

        if len(train_df) < 100:
            print(f"   [Retrainer] Training data too small ({len(train_df)} bars)")
            return None

        trials_per_worker = max(1, self.n_trials // self.n_workers)

        print(f"   [Retrainer] Optimizing {self.n_trials} trials across {self.n_workers} cores "
              f"on {len(train_df)} bars ({self.train_days}d window)...")

        # Save training data for workers
        tmp = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        joblib.dump(train_df, tmp.name)

        try:
            t0 = time.time()
            seed_base = int(datetime.now().timestamp()) % 100000
            tasks = [
                joblib.delayed(_worker)(
                    tmp.name, i, trials_per_worker, self.fixed_config, seed_base,
                    optimize_exits=self.optimize_exits, scoring=self.scoring
                )
                for i in range(self.n_workers)
            ]
            results = joblib.Parallel(
                n_jobs=self.n_workers, backend='loky', verbose=0
            )(tasks)
            elapsed = time.time() - t0
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

        # Find overall best
        valid = [r for r in results if r['best_value'] > float('-inf')]
        if not valid:
            print(f"   [Retrainer] All workers failed ({elapsed:.1f}s)")
            return None

        best = max(valid, key=lambda x: x['best_value'])
        self._last_best_params = best['best_params']
        self._last_retrain_date = datetime.now().date()

        print(f"   [Retrainer] Done in {elapsed:.1f}s (score={best['best_value']:.2f})")
        for k, v in sorted(best['best_params'].items()):
            print(f"      {k}: {v}")

        return best['best_params']

    def apply_params(self, config: Dict, new_params: Dict) -> Dict:
        """Apply new numerical params to config, preserving structural params."""
        updated = dict(config)
        for k, v in new_params.items():
            updated[k] = v
        updated['last_retrained'] = datetime.now().isoformat()
        return updated

    def save_config(self, config: Dict, strategy_dir: str):
        """Save updated config to strategy bundle directory."""
        config_path = os.path.join(strategy_dir, 'velocity_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"   [Retrainer] Config saved to {config_path}")
