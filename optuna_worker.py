"""
Lightweight Optuna Worker Module

This module contains only the objective function and minimal imports needed
for Optuna parallel optimization. By keeping this separate from the main
Streamlit page, we avoid re-importing TensorFlow and other heavy dependencies
in each worker process.

Uses file-based data sharing for proper multiprocessing support.
"""

import numpy as np
import os
import tempfile
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

# Try XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Path to shared data file (set by main process, read by workers)
_DATA_FILE_PATH = None
_CACHED_DATA = None  # Cache loaded data to avoid repeated file reads


def _get_shared_data():
    """Load shared data from file (cached after first load)."""
    global _CACHED_DATA

    if _CACHED_DATA is not None:
        return _CACHED_DATA

    if _DATA_FILE_PATH is None or not os.path.exists(_DATA_FILE_PATH):
        raise RuntimeError(f"Shared data file not found: {_DATA_FILE_PATH}")

    _CACHED_DATA = joblib.load(_DATA_FILE_PATH)
    return _CACHED_DATA


def optuna_objective(trial):
    """
    Module-level objective function for Optuna that can be pickled for parallel execution.
    Loads training data from shared file on first access.
    Returns F1-macro score as the optimization metric.
    """
    # Load shared data (cached after first load in each worker)
    data = _get_shared_data()

    X_train_arr = data['X_train']
    y_train_arr = data['y_train']
    X_val_arr = data['X_val']
    y_val_arr = data['y_val']
    model_type = data['model_type']

    try:
        if model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': 1  # Single thread per trial (Optuna parallelizes trials)
            }
            model = RandomForestClassifier(**params)

        elif model_type == 'gradient_boosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': 42
            }
            model = GradientBoostingClassifier(**params)

        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 2, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss',
                'verbosity': 0,
                'n_jobs': 1  # Single thread per trial (Optuna parallelizes trials)
            }
            model = XGBClassifier(**params)

            # XGBoost requires 0-indexed labels, encode [-1, 0, 1] -> [0, 1, 2]
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train_arr)
            y_val_encoded = le.transform(y_val_arr)

            model.fit(X_train_arr, y_train_encoded)
            y_pred = model.predict(X_val_arr)
            y_pred = le.inverse_transform(y_pred)  # Convert back to original labels

            f1 = f1_score(y_val_arr, y_pred, average='macro', zero_division=0)
            return f1
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train and evaluate using numpy arrays (for non-XGBoost models)
        model.fit(X_train_arr, y_train_arr)
        y_pred = model.predict(X_val_arr)

        # Optimize for F1-macro (balanced across all classes)
        f1 = f1_score(y_val_arr, y_pred, average='macro', zero_division=0)

        return f1

    except Exception as e:
        # Log error and return a poor score so optimization continues
        print(f"Trial failed: {e}")
        return 0.0


def set_shared_data(X_train, y_train, X_val, y_val, model_type):
    """
    Save shared data to a temp file for parallel workers to access.
    Returns the path to the temp file.
    """
    global _DATA_FILE_PATH, _CACHED_DATA

    # Clear any cached data
    _CACHED_DATA = None

    # Create temp file
    fd, path = tempfile.mkstemp(suffix='.joblib', prefix='optuna_data_')
    os.close(fd)

    # Save data
    data = {
        'X_train': X_train.values if hasattr(X_train, 'values') else X_train,
        'y_train': y_train.values if hasattr(y_train, 'values') else y_train,
        'X_val': X_val.values if hasattr(X_val, 'values') else X_val,
        'y_val': y_val.values if hasattr(y_val, 'values') else y_val,
        'model_type': model_type
    }
    joblib.dump(data, path)

    _DATA_FILE_PATH = path
    return path


def set_data_path(path):
    """Set the path to shared data file (called by workers)."""
    global _DATA_FILE_PATH, _CACHED_DATA
    _DATA_FILE_PATH = path
    _CACHED_DATA = None  # Clear cache when path changes


def clear_shared_data():
    """Clean up the shared data file."""
    global _DATA_FILE_PATH, _CACHED_DATA

    if _DATA_FILE_PATH and os.path.exists(_DATA_FILE_PATH):
        try:
            os.remove(_DATA_FILE_PATH)
        except:
            pass

    _DATA_FILE_PATH = None
    _CACHED_DATA = None


def get_data_path():
    """Get the current data file path."""
    return _DATA_FILE_PATH


class OptunaObjective:
    """
    Callable class for Optuna objective that can be pickled with the data path.
    This allows proper multiprocessing where each worker knows where to find the data.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self._cached_data = None

    def _load_data(self):
        """Load data from file (cached after first load)."""
        if self._cached_data is None:
            self._cached_data = joblib.load(self.data_path)
        return self._cached_data

    def __call__(self, trial):
        """Optuna objective function."""
        data = self._load_data()

        X_train_arr = data['X_train']
        y_train_arr = data['y_train']
        X_val_arr = data['X_val']
        y_val_arr = data['y_val']
        model_type = data['model_type']

        try:
            if model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': 1
                }
                model = RandomForestClassifier(**params)

            elif model_type == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 2, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': 42
                }
                model = GradientBoostingClassifier(**params)

            elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 2, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'random_state': 42,
                    'use_label_encoder': False,
                    'eval_metric': 'mlogloss',
                    'verbosity': 0,
                    'n_jobs': 1
                }
                model = XGBClassifier(**params)

                # XGBoost requires 0-indexed labels, encode [-1, 0, 1] -> [0, 1, 2]
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(y_train_arr)

                model.fit(X_train_arr, y_train_encoded)
                y_pred = model.predict(X_val_arr)
                y_pred = le.inverse_transform(y_pred)  # Convert back to original labels

                f1 = f1_score(y_val_arr, y_pred, average='macro', zero_division=0)
                return f1
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.fit(X_train_arr, y_train_arr)
            y_pred = model.predict(X_val_arr)
            f1 = f1_score(y_val_arr, y_pred, average='macro', zero_division=0)

            return f1

        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0


def create_objective(data_path):
    """Create a picklable objective function with the data path embedded."""
    return OptunaObjective(data_path)


# ============================================================================
# VELOCITY OPTIMIZATION (for oscillator velocity-based trading strategies)
# ============================================================================

_VELOCITY_DATA_PATH = None
_VELOCITY_CACHED_DATA = None


def set_velocity_shared_data(close_prices, osc_values, rsi_cache, macd_histogram, bb_upper, bb_lower,
                              optimize_metric='total_return', use_extra_indicators=True,
                              all_oscillators=None, v2_indicators=None,
                              # MTF oscillators (computed on sub-interval, resampled to primary)
                              all_oscillators_mtf=None, v2_indicators_mtf=None,
                              # Exit strategy constraints
                              force_midline_exit=False, force_opposite_exit=False,
                              sl_range=None, tp_range=None,
                              # Drawdown penalty settings
                              use_drawdown_penalty=False, max_drawdown_threshold=15.0,
                              drawdown_penalty_weight=0.3,
                              # V2 filter constraints (from UI checkboxes)
                              v2_filter_settings=None,
                              # Additional OHLCV data for signal improvements
                              high_prices=None, low_prices=None, volume=None,
                              # Volatility regime (ATR percentile) for vol regime filter
                              vol_regime=None):
    """
    Save velocity optimization data to a temp file for parallel workers.
    Returns the path to the temp file.

    Args:
        all_oscillators: Dict mapping oscillator names to their values (standard single-timeframe)
        v2_indicators: Dict with 'rsc', 'mfi2', 'sei' arrays (standard single-timeframe)
        all_oscillators_mtf: Dict mapping oscillator names to MTF values (sub-interval resampled)
        v2_indicators_mtf: Dict with 'rsc', 'mfi2', 'sei' arrays (MTF sub-interval resampled)
        force_midline_exit: If True, always use exit_on_midline_cross=True
        force_opposite_exit: If True, always use exit_on_opposite_signal=True
        sl_range: Tuple (min_pct, max_pct) for stop loss search range
        tp_range: Tuple (min_pct, max_pct) for take profit search range
        use_drawdown_penalty: If True, penalize strategies with high max drawdown
        max_drawdown_threshold: Drawdown % above which penalty starts
        drawdown_penalty_weight: Weight of drawdown penalty in objective (0-1)
    """
    global _VELOCITY_DATA_PATH, _VELOCITY_CACHED_DATA

    _VELOCITY_CACHED_DATA = None

    # Use project directory for temp file instead of system temp
    # macOS aggressively cleans /var/folders which causes race conditions with parallel workers
    project_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(project_dir, '.optuna_temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Create unique filename
    import uuid
    filename = f'velocity_optuna_{uuid.uuid4().hex[:8]}.joblib'
    path = os.path.join(temp_dir, filename)

    # Process all oscillators if provided
    processed_oscillators = {}
    if all_oscillators:
        for name, values in all_oscillators.items():
            processed_oscillators[name] = np.array(values) if hasattr(values, 'values') else values

    # Process MTF oscillators if provided
    processed_oscillators_mtf = {}
    if all_oscillators_mtf:
        for name, values in all_oscillators_mtf.items():
            processed_oscillators_mtf[name] = np.array(values) if hasattr(values, 'values') else values

    # Process V2 indicators if provided
    processed_v2 = {}
    if v2_indicators:
        for name, values in v2_indicators.items():
            if values is not None:
                processed_v2[name] = np.array(values) if hasattr(values, 'values') else values

    # Process MTF V2 indicators if provided
    processed_v2_mtf = {}
    if v2_indicators_mtf:
        for name, values in v2_indicators_mtf.items():
            if values is not None:
                processed_v2_mtf[name] = np.array(values) if hasattr(values, 'values') else values

    # Default SL/TP ranges if not specified
    if sl_range is None:
        sl_range = (0.0, 10.0)
    if tp_range is None:
        tp_range = (0.0, 20.0)

    data = {
        'close_prices': np.array(close_prices) if hasattr(close_prices, 'values') else close_prices,
        'osc_values': np.array(osc_values) if hasattr(osc_values, 'values') else osc_values,
        'rsi_cache': {k: np.array(v) if hasattr(v, 'values') else v for k, v in rsi_cache.items()},
        'macd_histogram': np.array(macd_histogram) if hasattr(macd_histogram, 'values') else macd_histogram,
        'bb_upper': np.array(bb_upper) if hasattr(bb_upper, 'values') else bb_upper,
        'bb_lower': np.array(bb_lower) if hasattr(bb_lower, 'values') else bb_lower,
        'optimize_metric': optimize_metric,
        'use_extra_indicators': use_extra_indicators,
        'all_oscillators': processed_oscillators,  # Standard oscillator types for search
        'all_oscillators_mtf': processed_oscillators_mtf,  # MTF oscillator types for search
        'v2_indicators': processed_v2,  # Standard V2 indicators for filter optimization
        'v2_indicators_mtf': processed_v2_mtf,  # MTF V2 indicators for filter optimization
        # Exit strategy constraints
        'force_midline_exit': force_midline_exit,
        'force_opposite_exit': force_opposite_exit,
        'sl_range': sl_range,
        'tp_range': tp_range,
        # Drawdown penalty settings
        'use_drawdown_penalty': use_drawdown_penalty,
        'max_drawdown_threshold': max_drawdown_threshold,
        'drawdown_penalty_weight': drawdown_penalty_weight,
        # V2 filter constraints from UI
        'v2_filter_settings': v2_filter_settings or {},
        # Additional OHLCV data for signal improvements
        'high_prices': np.array(high_prices) if high_prices is not None else None,
        'low_prices': np.array(low_prices) if low_prices is not None else None,
        'volume': np.array(volume) if volume is not None else None,
        'vol_regime': np.array(vol_regime) if vol_regime is not None else None,
    }
    joblib.dump(data, path)

    _VELOCITY_DATA_PATH = path
    return path


def clear_velocity_shared_data():
    """Clean up velocity shared data file."""
    global _VELOCITY_DATA_PATH, _VELOCITY_CACHED_DATA

    if _VELOCITY_DATA_PATH and os.path.exists(_VELOCITY_DATA_PATH):
        try:
            os.remove(_VELOCITY_DATA_PATH)
        except:
            pass

    _VELOCITY_DATA_PATH = None
    _VELOCITY_CACHED_DATA = None


class VelocityOptunaObjective:
    """
    Picklable objective class for velocity-based strategy optimization.
    Each worker loads data from file and runs backtest.
    """

    def __init__(self, data_path, all_results_list=None):
        self.data_path = data_path
        self._cached_data = None
        self.all_results = all_results_list  # Shared list for collecting results

    def _load_data(self):
        if self._cached_data is None:
            self._cached_data = joblib.load(self.data_path)
        return self._cached_data

    def __call__(self, trial):
        data = self._load_data()

        close_prices = data['close_prices']
        osc_values = data['osc_values']  # Default/fallback oscillator
        rsi_cache = data['rsi_cache']
        macd_histogram = data['macd_histogram']
        bb_upper = data['bb_upper']
        bb_lower = data['bb_lower']
        optimize_metric = data['optimize_metric']
        use_extra_indicators = data['use_extra_indicators']
        all_oscillators = data.get('all_oscillators', {})
        all_oscillators_mtf = data.get('all_oscillators_mtf', {})

        # Exit strategy constraints from UI
        force_midline_exit = data.get('force_midline_exit', False)
        force_opposite_exit = data.get('force_opposite_exit', False)
        force_signal_type = data.get('force_signal_type', None)  # Force specific signal type
        sl_range = data.get('sl_range', (0.0, 10.0))
        tp_range = data.get('tp_range', (0.0, 20.0))

        # Drawdown penalty settings
        use_drawdown_penalty = data.get('use_drawdown_penalty', False)
        max_drawdown_threshold = data.get('max_drawdown_threshold', 15.0)
        drawdown_penalty_weight = data.get('drawdown_penalty_weight', 0.3)

        # MTF search: if MTF oscillators available, let Optuna choose
        mtf_available = bool(all_oscillators_mtf)
        if mtf_available:
            use_mtf = trial.suggest_categorical('use_mtf', [True, False])
        else:
            use_mtf = False

        # Build a FIXED oscillator type list (union of standard + MTF keys)
        # Optuna requires the same categorical choices across all trials
        all_osc_keys = set(all_oscillators.keys())
        if all_oscillators_mtf:
            all_osc_keys |= set(all_oscillators_mtf.keys())
        oscillator_types = sorted(all_osc_keys) if len(all_osc_keys) > 1 else ['composite_smooth']

        # Select oscillator source based on MTF choice
        active_oscillators = all_oscillators_mtf if use_mtf else all_oscillators

        # Suggest parameters - including oscillator type if multiple available
        params = {}
        params['use_mtf'] = use_mtf

        # Add oscillator_type as a searchable parameter (fixed choice set)
        if len(oscillator_types) > 1:
            params['oscillator_type'] = trial.suggest_categorical('oscillator_type', oscillator_types)
        else:
            params['oscillator_type'] = 'composite_smooth'

        # SL/TP with constrained ranges from UI
        sl_min, sl_max = sl_range
        tp_min, tp_max = tp_range

        # Exit conditions - either forced or searchable
        if force_opposite_exit:
            exit_opposite = True
        else:
            exit_opposite = trial.suggest_categorical('exit_on_opposite_signal', [True, False])

        if force_midline_exit:
            exit_midline = True
        else:
            exit_midline = trial.suggest_categorical('exit_on_midline_cross', [True, False])

        # Signal type: forced or searchable
        if force_signal_type:
            _signal_type = force_signal_type
        else:
            _signal_type = trial.suggest_categorical('signal_type', [
                'velocity_crossover_and_zone', 'velocity_crossover_or_zone', 'zone_only',
                'momentum', 'any_reversal', 'double_bottom', 'divergence', 'breakout'
            ])

        params.update({
            'signal_type': _signal_type,
            'vel_smoothing': trial.suggest_int('vel_smoothing', 1, 15),
            'oversold_threshold': trial.suggest_float('oversold_threshold', -0.6, -0.02),
            'overbought_threshold': trial.suggest_float('overbought_threshold', 0.02, 0.6),
            'stop_loss_pct': trial.suggest_float('stop_loss_pct', sl_min, sl_max),
            'take_profit_pct': trial.suggest_float('take_profit_pct', tp_min, tp_max),
            'min_hold_bars': trial.suggest_int('min_hold_bars', 1, 10),
            'min_bars_between': trial.suggest_int('min_bars_between', 1, 15),
            'require_accel': trial.suggest_categorical('require_accel', [True, False]),
            'extreme_zone_mult': trial.suggest_float('extreme_zone_mult', 1.1, 2.5),
            'exit_on_opposite_signal': exit_opposite,
            'exit_on_midline_cross': exit_midline,
            # Previously hardcoded parameters - now searchable for exhaustive optimization
            'velocity_std_window': trial.suggest_int('velocity_std_window', 5, 20),
            'momentum_multiplier': trial.suggest_float('momentum_multiplier', 1.0, 3.0),
            'double_bottom_lookback': trial.suggest_int('double_bottom_lookback', 5, 20),
            'divergence_lookback': trial.suggest_int('divergence_lookback', 3, 10),
            # Acceleration Reversal Exit parameters - exit based on acceleration/jerk
            'use_accel_exit': trial.suggest_categorical('use_accel_exit', [True, False]),
            'accel_exit_type': trial.suggest_categorical('accel_exit_type', ['sign_reversal', 'magnitude', 'both']),
            'accel_exit_threshold': trial.suggest_float('accel_exit_threshold', 0.0, 0.1),
            'accel_exit_min_pnl': trial.suggest_float('accel_exit_min_pnl', 0.5, 5.0),
            'accel_exit_lookback': trial.suggest_int('accel_exit_lookback', 1, 5),
            'use_jerk_confirm': trial.suggest_categorical('use_jerk_confirm', [True, False]),
            'jerk_confirm_threshold': trial.suggest_float('jerk_confirm_threshold', 0.0, 0.05),
            # Trailing stop - replaces fixed stop loss with dynamic trailing stop
            'use_trailing_stop': trial.suggest_categorical('use_trailing_stop', [True, False]),
            'trailing_stop_pct': trial.suggest_float('trailing_stop_pct', 0.2, 3.0),
            'trailing_stop_activation_pct': trial.suggest_float('trailing_stop_activation_pct', 0.1, 2.0),
            # Break-even stop - move SL to entry after profit reached
            'use_breakeven_stop': trial.suggest_categorical('use_breakeven_stop', [True, False]),
            'breakeven_trigger_pct': trial.suggest_float('breakeven_trigger_pct', 0.1, 2.0),
            'breakeven_offset_pct': trial.suggest_float('breakeven_offset_pct', 0.01, 0.2),
            # === SIGNAL IMPROVEMENTS (1-8) ===
            # #1: Adaptive smoothing - reduce oscillator noise
            'smoothing_type': trial.suggest_categorical('smoothing_type', ['sma', 'ema', 'adaptive']),
            'adaptive_fast_alpha': trial.suggest_float('adaptive_fast_alpha', 0.1, 0.5),
            'adaptive_slow_alpha': trial.suggest_float('adaptive_slow_alpha', 0.01, 0.15),
            # #2: Multi-oscillator consensus voting
            'use_consensus': trial.suggest_categorical('use_consensus', [True, False]),
            'consensus_count': trial.suggest_int('consensus_count', 2, 4),
            # #3: ATR-based trailing stop
            'trailing_stop_type': trial.suggest_categorical('trailing_stop_type', ['fixed_pct', 'atr']),
            'trailing_atr_mult': trial.suggest_float('trailing_atr_mult', 1.0, 4.0),
            'trailing_atr_period': trial.suggest_int('trailing_atr_period', 10, 20),
            # #4: Volume confirmation
            'use_volume_confirm': trial.suggest_categorical('use_volume_confirm', [True, False]),
            'volume_ratio_threshold': trial.suggest_float('volume_ratio_threshold', 0.5, 2.5),
            # #5: Price range position filter
            'use_range_filter': trial.suggest_categorical('use_range_filter', [True, False]),
            'range_lookback': trial.suggest_int('range_lookback', 10, 30),
            'range_max_position': trial.suggest_float('range_max_position', 0.3, 0.7),
            # #6: Regime-adaptive (trend filter)
            'use_trend_filter': trial.suggest_categorical('use_trend_filter', [True, False]),
            'trend_sma_period': trial.suggest_int('trend_sma_period', 30, 100),
            'trend_strict_mult': trial.suggest_float('trend_strict_mult', 1.0, 2.0),
            # #7: Higher-timeframe momentum alignment
            'use_htf_filter': trial.suggest_categorical('use_htf_filter', [True, False]),
            'htf_slow_window': trial.suggest_int('htf_slow_window', 10, 50),
            'htf_threshold': trial.suggest_float('htf_threshold', -0.5, 0.0),
            # #8: Midline exit delay
            'midline_exit_bars': trial.suggest_int('midline_exit_bars', 1, 5),
        })

        if use_extra_indicators:
            params['rsi_filter'] = trial.suggest_categorical('rsi_filter', ['none', 'oversold_only', 'overbought_only', 'both'])
            params['rsi_period'] = trial.suggest_int('rsi_period', 5, 30)
            params['rsi_oversold'] = trial.suggest_int('rsi_oversold', 15, 40)
            params['rsi_overbought'] = trial.suggest_int('rsi_overbought', 60, 85)
            params['use_macd_confirm'] = trial.suggest_categorical('use_macd_confirm', [True, False])
            params['use_bb_filter'] = trial.suggest_categorical('use_bb_filter', [True, False])

        # V2 indicator filters (Regime, Fragility, Entropy)
        # Select V2 indicators based on MTF choice
        v2_indicators_std = data.get('v2_indicators', {})
        v2_indicators_mtf = data.get('v2_indicators_mtf', {})
        v2_indicators = v2_indicators_mtf if (use_mtf and v2_indicators_mtf) else v2_indicators_std

        if v2_indicators:
            # Regime filter: searchable if indicator available
            if 'rsc' in v2_indicators:
                params['use_regime_filter'] = trial.suggest_categorical('use_regime_filter', [True, False])
                params['regime_threshold'] = trial.suggest_float('regime_threshold', -0.5, 0.5) if params['use_regime_filter'] else 0.0
            else:
                params['use_regime_filter'] = False
                params['regime_threshold'] = 0.0

            # Fragility filter: searchable if indicator available
            if 'mfi2' in v2_indicators:
                params['use_fragility_filter'] = trial.suggest_categorical('use_fragility_filter', [True, False])
                params['fragility_threshold'] = trial.suggest_float('fragility_threshold', 0.2, 0.8) if params['use_fragility_filter'] else 0.5
            else:
                params['use_fragility_filter'] = False
                params['fragility_threshold'] = 0.5

            # Entropy filter: searchable if indicator available
            if 'sei' in v2_indicators:
                params['use_entropy_filter'] = trial.suggest_categorical('use_entropy_filter', [True, False])
                params['entropy_threshold'] = trial.suggest_float('entropy_threshold', 0.3, 0.9) if params['use_entropy_filter'] else 0.7
            else:
                params['use_entropy_filter'] = False
                params['entropy_threshold'] = 0.7
        else:
            params['use_regime_filter'] = False
            params['use_fragility_filter'] = False
            params['use_entropy_filter'] = False

        # Volatility regime filter (ATR percentile) — independent of v2_indicators
        vol_regime = data.get('vol_regime')
        if vol_regime is not None:
            params['use_vol_regime_filter'] = trial.suggest_categorical('use_vol_regime_filter', [True, False])
            if params['use_vol_regime_filter']:
                params['vol_regime_percentile_threshold'] = trial.suggest_float('vol_regime_percentile_threshold', 0.1, 0.5)
            else:
                params['vol_regime_percentile_threshold'] = 0.25
        else:
            params['use_vol_regime_filter'] = False
            params['vol_regime_percentile_threshold'] = 0.25

        # Money Flow Velocity filter — independent of v2_indicators
        mfv_flow = data.get('mfv_flow')
        if mfv_flow is not None:
            params['use_mfv_filter'] = trial.suggest_categorical('use_mfv_filter', [True, False])
            if params['use_mfv_filter']:
                params['mfv_mode'] = trial.suggest_categorical('mfv_mode', ['velocity', 'flow', 'both'])
                params['mfv_threshold'] = trial.suggest_float('mfv_threshold', 0.0, 0.5)
            else:
                params['mfv_mode'] = 'velocity'
                params['mfv_threshold'] = 0.0
        else:
            params['use_mfv_filter'] = False
            params['mfv_mode'] = 'velocity'
            params['mfv_threshold'] = 0.0

        # Select the oscillator values based on the chosen type and MTF setting
        selected_osc_type = params['oscillator_type']
        if selected_osc_type in active_oscillators:
            selected_osc_values = active_oscillators[selected_osc_type]
        else:
            selected_osc_values = osc_values  # Fallback to default

        # Inject fixed wavelet params from data (not searched by Optuna)
        if data.get('use_wavelet_denoise', False):
            params['use_wavelet_denoise'] = True
            params['wavelet_family'] = data.get('wavelet_family', 'db4')
            params['wavelet_level'] = data.get('wavelet_level', 2)
            params['wavelet_threshold_mode'] = data.get('wavelet_threshold_mode', 'hard')

        # Run backtest with selected oscillator and V2 indicators
        result = self._run_backtest(params, close_prices, selected_osc_values, rsi_cache,
                                     macd_histogram, bb_upper, bb_lower, use_extra_indicators,
                                     v2_indicators,
                                     high_prices=data.get('high_prices'),
                                     low_prices=data.get('low_prices'),
                                     volume=data.get('volume'),
                                     all_oscillators=active_oscillators,
                                     vol_regime=data.get('vol_regime'))

        if result is None:
            return float('-inf')

        # Minimum trade count filter: reject strategies with too few trades
        min_trades = data.get('min_trades', 0)
        n_trades = result.get('n_trades', result.get('num_trades', 0))
        if min_trades > 0 and n_trades < min_trades:
            return float('-inf')

        # Get the base score from the optimization metric
        base_score = result[optimize_metric]

        # Apply trade count bonus: reward statistical significance
        # sqrt(n_trades) scaling gives diminishing returns for more trades
        # This prevents the optimizer from favoring rare lucky trades
        trade_count_bonus_weight = data.get('trade_count_bonus_weight', 0.0)
        if trade_count_bonus_weight > 0 and n_trades > 0 and base_score > 0:
            import math
            # Normalize: sqrt(n_trades) / sqrt(100) so 100 trades = 1.0x bonus
            trade_bonus = math.sqrt(n_trades) / math.sqrt(100)
            base_score = base_score * (1.0 + trade_count_bonus_weight * (trade_bonus - 1.0))

        # Apply max drawdown penalty if enabled
        if use_drawdown_penalty and 'max_drawdown' in result:
            max_dd = result['max_drawdown']
            if max_dd > max_drawdown_threshold:
                # Calculate penalty: linear reduction based on how much DD exceeds threshold
                excess_dd = max_dd - max_drawdown_threshold
                # Penalty scales from 0 to drawdown_penalty_weight as excess increases
                # At 2x threshold excess, penalty is full weight
                penalty_factor = min(1.0, excess_dd / max_drawdown_threshold)
                penalty = penalty_factor * drawdown_penalty_weight

                # Apply penalty to score (reduce by penalty percentage)
                if base_score > 0:
                    base_score = base_score * (1 - penalty)
                else:
                    # For negative scores, make them more negative
                    base_score = base_score * (1 + penalty)

        # Store result for later retrieval
        trial.set_user_attr('result', result)

        return base_score

    def _run_backtest(self, params, close_prices, osc_values, rsi_cache,
                      macd_histogram, bb_upper, bb_lower, use_extra_indicators,
                      v2_indicators=None, high_prices=None, low_prices=None,
                      volume=None, all_oscillators=None, vol_regime=None):
        """Fast vectorized backtest with V2 indicator filters and signal improvements."""
        import pandas as pd

        # === IMPROVEMENT #1: Adaptive smoothing ===
        smoothing_type = params.get('smoothing_type', 'sma')
        vel_smoothing = params['vel_smoothing']

        if vel_smoothing <= 1:
            osc_smooth = osc_values
        elif smoothing_type == 'ema':
            alpha = 2.0 / (vel_smoothing + 1)
            osc_smooth = pd.Series(osc_values).ewm(alpha=alpha, adjust=False).mean().values
        elif smoothing_type == 'adaptive':
            # Adaptive EMA: alpha varies with local oscillator volatility
            fast_alpha = params.get('adaptive_fast_alpha', 0.3)
            slow_alpha = params.get('adaptive_slow_alpha', 0.05)
            osc_changes = np.abs(np.diff(osc_values, prepend=osc_values[0]))
            vol = pd.Series(osc_changes).rolling(vel_smoothing, min_periods=1).mean().values
            vol_norm = vol / (np.max(vol) + 1e-10)  # 0-1 normalized
            alphas = slow_alpha + (fast_alpha - slow_alpha) * vol_norm
            osc_smooth = np.zeros_like(osc_values, dtype=float)
            osc_smooth[0] = osc_values[0]
            for idx in range(1, len(osc_values)):
                osc_smooth[idx] = alphas[idx] * osc_values[idx] + (1 - alphas[idx]) * osc_smooth[idx - 1]
        else:  # 'sma' (default)
            osc_smooth = pd.Series(osc_values).rolling(window=vel_smoothing).mean().bfill().values

        # Apply wavelet denoising if enabled
        if params.get('use_wavelet_denoise', False):
            try:
                import pywt
                family = params.get('wavelet_family', 'db4')
                level = params.get('wavelet_level', 2)
                mode = params.get('wavelet_threshold_mode', 'hard')
                coeffs = pywt.wavedec(osc_smooth, family, level=level)
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(osc_smooth)))
                denoised = [coeffs[0]] + [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
                osc_smooth = pywt.waverec(denoised, family)[:len(osc_smooth)]
            except Exception:
                pass

        velocity = np.diff(osc_smooth, prepend=osc_smooth[0])
        acceleration = np.diff(velocity, prepend=velocity[0])
        jerk = np.diff(acceleration, prepend=acceleration[0])  # Third derivative for accel exit

        # Build conditions
        vel_cross_up = (velocity > 0) & (np.roll(velocity, 1) <= 0)
        vel_cross_down = (velocity < 0) & (np.roll(velocity, 1) >= 0)
        in_oversold = osc_smooth < params['oversold_threshold']
        in_overbought = osc_smooth > params['overbought_threshold']
        extreme_oversold = osc_smooth < (params['oversold_threshold'] * params['extreme_zone_mult'])
        extreme_overbought = osc_smooth > (params['overbought_threshold'] * params['extreme_zone_mult'])

        vel_std_window = params.get('velocity_std_window', 10)
        vel_std = pd.Series(velocity).rolling(vel_std_window, min_periods=1).std().fillna(np.std(velocity)).values
        momentum_mult = params.get('momentum_multiplier', 1.5)
        strong_momentum_up = velocity > vel_std * momentum_mult
        strong_momentum_down = velocity < -vel_std * momentum_mult

        # Signal type conditions
        sig_type = params['signal_type']
        if sig_type == 'velocity_crossover_and_zone':
            buy_cond = vel_cross_up & in_oversold
            sell_cond = vel_cross_down & in_overbought
        elif sig_type == 'velocity_crossover_or_zone':
            buy_cond = vel_cross_up | extreme_oversold
            sell_cond = vel_cross_down | extreme_overbought
        elif sig_type == 'zone_only':
            buy_cond = extreme_oversold & (velocity > 0)
            sell_cond = extreme_overbought & (velocity < 0)
        elif sig_type == 'momentum':
            buy_cond = strong_momentum_up & (osc_smooth < 0)
            sell_cond = strong_momentum_down & (osc_smooth > 0)
        elif sig_type == 'any_reversal':
            buy_cond = vel_cross_up | extreme_oversold | (strong_momentum_up & in_oversold)
            sell_cond = vel_cross_down | extreme_overbought | (strong_momentum_down & in_overbought)
        elif sig_type == 'double_bottom':
            db_lookback = params.get('double_bottom_lookback', 10)
            vel_cross_up_count = pd.Series(vel_cross_up.astype(int)).rolling(db_lookback).sum().values
            buy_cond = (vel_cross_up_count >= 2) & in_oversold
            vel_cross_down_count = pd.Series(vel_cross_down.astype(int)).rolling(db_lookback).sum().values
            sell_cond = (vel_cross_down_count >= 2) & in_overbought
        elif sig_type == 'divergence':
            div_lookback = params.get('divergence_lookback', 5)
            price_series = pd.Series(close_prices)
            osc_series = pd.Series(osc_smooth)
            price_lower_low = (close_prices < price_series.rolling(div_lookback).min().shift(1).values)
            osc_higher_low = (osc_smooth > osc_series.rolling(div_lookback).min().shift(1).values)
            buy_cond = price_lower_low & osc_higher_low & in_oversold
            price_higher_high = (close_prices > price_series.rolling(div_lookback).max().shift(1).values)
            osc_lower_high = (osc_smooth < osc_series.rolling(div_lookback).max().shift(1).values)
            sell_cond = price_higher_high & osc_lower_high & in_overbought
        elif sig_type == 'breakout':
            osc_breaks_above = (osc_smooth > params['oversold_threshold']) & (np.roll(osc_smooth, 1) <= params['oversold_threshold'])
            osc_breaks_below = (osc_smooth < params['overbought_threshold']) & (np.roll(osc_smooth, 1) >= params['overbought_threshold'])
            buy_cond = osc_breaks_above
            sell_cond = osc_breaks_below
        else:
            buy_cond = vel_cross_up & in_oversold
            sell_cond = vel_cross_down & in_overbought

        if params['require_accel']:
            buy_cond = buy_cond & (acceleration > 0)
            sell_cond = sell_cond & (acceleration < 0)

        # Extra indicators
        if use_extra_indicators and params.get('rsi_filter', 'none') != 'none':
            rsi_period = params.get('rsi_period', 14)
            rsi = rsi_cache.get(rsi_period, rsi_cache.get(14, np.zeros_like(close_prices)))
            rsi_os = params.get('rsi_oversold', 30)
            rsi_ob = params.get('rsi_overbought', 70)
            if params['rsi_filter'] == 'oversold_only':
                buy_cond = buy_cond & (rsi < rsi_os)
            elif params['rsi_filter'] == 'overbought_only':
                sell_cond = sell_cond & (rsi > rsi_ob)
            elif params['rsi_filter'] == 'both':
                buy_cond = buy_cond & (rsi < rsi_os)
                sell_cond = sell_cond & (rsi > rsi_ob)

        if use_extra_indicators and params.get('use_macd_confirm', False):
            macd_improving = macd_histogram > np.roll(macd_histogram, 1)
            macd_declining = macd_histogram < np.roll(macd_histogram, 1)
            buy_cond = buy_cond & macd_improving
            sell_cond = sell_cond & macd_declining

        if use_extra_indicators and params.get('use_bb_filter', False):
            buy_cond = buy_cond & (close_prices < bb_lower)
            sell_cond = sell_cond & (close_prices > bb_upper)

        # === IMPROVEMENT #2: Multi-oscillator consensus voting ===
        if params.get('use_consensus', False) and all_oscillators and len(all_oscillators) >= 3:
            consensus_count = params.get('consensus_count', 2)
            # Count how many oscillators agree on buy/sell at each bar
            buy_votes = np.zeros(len(close_prices), dtype=int)
            sell_votes = np.zeros(len(close_prices), dtype=int)
            for osc_name, osc_vals in all_oscillators.items():
                if len(osc_vals) != len(close_prices):
                    continue
                # Compute velocity for this oscillator
                o_smooth = osc_vals
                if vel_smoothing > 1 and smoothing_type == 'sma':
                    o_smooth = pd.Series(osc_vals).rolling(window=vel_smoothing).mean().bfill().values
                o_vel = np.diff(o_smooth, prepend=o_smooth[0])
                o_cross_up = (o_vel > 0) & (np.roll(o_vel, 1) <= 0)
                o_cross_down = (o_vel < 0) & (np.roll(o_vel, 1) >= 0)
                o_oversold = o_smooth < params.get('oversold_threshold', -0.2)
                o_overbought = o_smooth > params.get('overbought_threshold', 0.2)
                buy_votes += (o_cross_up | o_oversold).astype(int)
                sell_votes += (o_cross_down | o_overbought).astype(int)
            buy_cond = buy_cond & (buy_votes >= consensus_count)
            sell_cond = sell_cond & (sell_votes >= consensus_count)

        # === IMPROVEMENT #4: Volume confirmation ===
        if params.get('use_volume_confirm', False) and volume is not None and len(volume) == len(close_prices):
            vol_threshold = params.get('volume_ratio_threshold', 1.0)
            vol_sma = pd.Series(volume.astype(float)).rolling(20, min_periods=1).mean().values
            vol_ratio = volume / (vol_sma + 1e-10)
            buy_cond = buy_cond & (vol_ratio > vol_threshold)
            sell_cond = sell_cond & (vol_ratio > vol_threshold)

        # === IMPROVEMENT #5: Price range position filter ===
        if params.get('use_range_filter', False):
            range_lb = params.get('range_lookback', 20)
            range_max = params.get('range_max_position', 0.5)
            price_series = pd.Series(close_prices)
            rolling_min = price_series.rolling(range_lb, min_periods=1).min().values
            rolling_max = price_series.rolling(range_lb, min_periods=1).max().values
            price_range = rolling_max - rolling_min
            position_in_range = np.where(price_range > 0, (close_prices - rolling_min) / price_range, 0.5)
            buy_cond = buy_cond & (position_in_range < range_max)
            sell_cond = sell_cond & (position_in_range > (1.0 - range_max))

        # === IMPROVEMENT #6: Regime-adaptive trend filter ===
        if params.get('use_trend_filter', False):
            trend_period = params.get('trend_sma_period', 50)
            trend_mult = params.get('trend_strict_mult', 1.5)
            trend_sma = pd.Series(close_prices).rolling(trend_period, min_periods=1).mean().values
            in_uptrend = close_prices > trend_sma
            # In strong uptrend, require deeper oversold for buys (tighten threshold)
            tightened_oversold = osc_smooth < (params.get('oversold_threshold', -0.2) * trend_mult)
            # Override buy_cond in uptrend: must be deeper oversold
            buy_cond = np.where(in_uptrend, buy_cond & tightened_oversold, buy_cond)

        # === IMPROVEMENT #7: Higher-timeframe momentum alignment ===
        if params.get('use_htf_filter', False):
            htf_window = params.get('htf_slow_window', 20)
            htf_thresh = params.get('htf_threshold', -0.2)
            # Compute slow velocity as proxy for higher timeframe momentum
            slow_osc = pd.Series(osc_smooth).rolling(htf_window, min_periods=1).mean().values
            slow_velocity = np.diff(slow_osc, prepend=slow_osc[0])
            # Don't buy when slow velocity is strongly negative
            buy_cond = buy_cond & (slow_velocity > htf_thresh)
            # Don't sell when slow velocity is strongly positive
            sell_cond = sell_cond & (slow_velocity < -htf_thresh)

        # V2 indicator filters (Regime, Fragility, Entropy)
        if v2_indicators:
            if params.get('use_regime_filter', False) and 'rsc' in v2_indicators:
                rsc = v2_indicators['rsc']
                regime_thresh = params.get('regime_threshold', 0.0)
                buy_cond = buy_cond & (rsc > regime_thresh)

            if params.get('use_fragility_filter', False) and 'mfi2' in v2_indicators:
                mfi2 = v2_indicators['mfi2']
                frag_thresh = params.get('fragility_threshold', 0.5)
                buy_cond = buy_cond & (mfi2 < frag_thresh)

            if params.get('use_entropy_filter', False) and 'sei' in v2_indicators:
                sei = v2_indicators['sei']
                entropy_thresh = params.get('entropy_threshold', 0.7)
                buy_cond = buy_cond & (sei < entropy_thresh)

        # Volatility regime filter (ATR percentile)
        if params.get('use_vol_regime_filter', False) and vol_regime is not None:
            vol_thresh = params.get('vol_regime_percentile_threshold', 0.25)
            buy_cond = buy_cond & (vol_regime > vol_thresh)

        # === IMPROVEMENT #3: Pre-compute ATR for dynamic trailing stop ===
        trailing_stop_type = params.get('trailing_stop_type', 'fixed_pct')
        atr_values = None
        atr_mult = params.get('trailing_atr_mult', 2.0)
        if trailing_stop_type == 'atr' and high_prices is not None and low_prices is not None:
            atr_period = params.get('trailing_atr_period', 14)
            tr = np.maximum(
                high_prices - low_prices,
                np.maximum(
                    np.abs(high_prices - np.roll(close_prices, 1)),
                    np.abs(low_prices - np.roll(close_prices, 1))
                )
            )
            tr[0] = high_prices[0] - low_prices[0]
            atr_values = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values

        # === IMPROVEMENT #8: Pre-compute consecutive bars above midline ===
        midline_exit_bars_required = params.get('midline_exit_bars', 1)
        if midline_exit_bars_required > 1:
            above_midline = (osc_smooth > 0).astype(int)
            # Count consecutive bars above midline
            consecutive_above = np.zeros(len(osc_smooth), dtype=int)
            for idx in range(1, len(osc_smooth)):
                if above_midline[idx]:
                    consecutive_above[idx] = consecutive_above[idx - 1] + 1
                else:
                    consecutive_above[idx] = 0

        # Trading simulation with equity curve tracking for max drawdown
        position = 0
        entry_price = 0.0
        entry_bar_idx = 0  # Track entry bar for min_hold_bars
        highest_price = 0.0  # Track highest price for trailing stop
        last_trade_bar = -params['min_bars_between']
        trades = []
        exit_on_opposite = params.get('exit_on_opposite_signal', True)
        exit_on_midline = params.get('exit_on_midline_cross', False)
        min_hold_bars = params.get('min_hold_bars', 1)
        use_trailing_stop = params.get('use_trailing_stop', False)
        trailing_stop_pct = params.get('trailing_stop_pct', 2.0)
        trailing_stop_activation_pct = params.get('trailing_stop_activation_pct', 0.3)
        use_breakeven_stop = params.get('use_breakeven_stop', False)
        breakeven_trigger_pct = params.get('breakeven_trigger_pct', 0.3)
        breakeven_offset_pct = params.get('breakeven_offset_pct', 0.05)

        # Track equity curve for max drawdown calculation
        equity = 100.0  # Start with $100
        equity_curve = [equity]
        peak_equity = equity

        for i in range(1, len(close_prices)):
            price = close_prices[i]
            bars_since = i - last_trade_bar
            bars_held = i - entry_bar_idx
            can_exit = bars_held >= min_hold_bars

            # Update highest price for trailing stop (use intrabar high)
            if position == 1:
                bar_high_ts = high_prices[i] if high_prices is not None else price
                highest_price = max(highest_price, bar_high_ts)

            if position == 1:
                exited = False
                # Priority 1: Fixed Stop Loss (ALWAYS fires)
                if params['stop_loss_pct'] > 0:
                    sl_price_level = entry_price * (1 - params['stop_loss_pct'] / 100)
                    bar_low = low_prices[i] if low_prices is not None else price
                    if bar_low <= sl_price_level:
                        pnl_pct = -params['stop_loss_pct']
                        trades.append(pnl_pct)
                        equity *= (1 + pnl_pct / 100)
                        equity_curve.append(equity)
                        peak_equity = max(peak_equity, equity)
                        position = 0
                        last_trade_bar = i
                        exited = True
                # Priority 2: Take Profit (ALWAYS fires)
                if not exited and params['take_profit_pct'] > 0:
                    tp_price_level = entry_price * (1 + params['take_profit_pct'] / 100)
                    bar_high = high_prices[i] if high_prices is not None else price
                    if bar_high >= tp_price_level:
                        pnl_pct = params['take_profit_pct']
                        trades.append(pnl_pct)
                        equity *= (1 + pnl_pct / 100)
                        equity_curve.append(equity)
                        peak_equity = max(peak_equity, equity)
                        position = 0
                        last_trade_bar = i
                        exited = True
                # Priority 3: Trailing Stop (fires once activated, ignores min_hold_bars)
                if not exited and use_trailing_stop and trailing_stop_pct > 0:
                    hwm_pnl = (highest_price - entry_price) / entry_price * 100
                    if hwm_pnl >= trailing_stop_activation_pct:
                        if trailing_stop_type == 'atr' and atr_values is not None:
                            trail_stop_price = highest_price - atr_values[i] * atr_mult
                        else:
                            trail_stop_price = highest_price * (1 - trailing_stop_pct / 100)
                        bar_low_ts = low_prices[i] if low_prices is not None else price
                        if bar_low_ts <= trail_stop_price:
                            pnl_pct = (trail_stop_price - entry_price) / entry_price * 100
                            trades.append(pnl_pct)
                            equity *= (1 + pnl_pct / 100)
                            equity_curve.append(equity)
                            peak_equity = max(peak_equity, equity)
                            position = 0
                            last_trade_bar = i
                            exited = True
                # Priority 4: Break-Even Stop (fires once activated, ignores min_hold_bars)
                if not exited and use_breakeven_stop:
                    hwm_pnl = (highest_price - entry_price) / entry_price * 100
                    if hwm_pnl >= breakeven_trigger_pct:
                        be_price = entry_price * (1 + breakeven_offset_pct / 100)
                        bar_low_be = low_prices[i] if low_prices is not None else price
                        if bar_low_be <= be_price:
                            pnl_pct = (be_price - entry_price) / entry_price * 100
                            trades.append(pnl_pct)
                            equity *= (1 + pnl_pct / 100)
                            equity_curve.append(equity)
                            peak_equity = max(peak_equity, equity)
                            position = 0
                            last_trade_bar = i
                            exited = True
                # Remaining exits require min_hold_bars
                if not exited and can_exit and params.get('use_accel_exit', False):
                    current_pnl = (price - entry_price) / entry_price * 100
                    accel_min_pnl = params.get('accel_exit_min_pnl', 0.5)
                    accel_type = params.get('accel_exit_type', 'sign_reversal')
                    accel_thresh = params.get('accel_exit_threshold', 0.0)
                    accel_lookback = params.get('accel_exit_lookback', 1)
                    use_jerk = params.get('use_jerk_confirm', False)
                    jerk_thresh = params.get('jerk_confirm_threshold', 0.0)

                    # Check if we have enough positive P&L (or allow exit to prevent larger loss)
                    pnl_ok = current_pnl >= accel_min_pnl or current_pnl < 0

                    # Check acceleration condition for LONG position (negative accel = bearish)
                    accel_cond = False
                    if accel_type == 'sign_reversal':
                        # Check consecutive bars of adverse acceleration
                        if i >= accel_lookback:
                            accel_cond = all(acceleration[i-j] < 0 for j in range(accel_lookback))
                        else:
                            accel_cond = acceleration[i] < 0
                    elif accel_type == 'magnitude':
                        accel_cond = acceleration[i] < -accel_thresh
                    elif accel_type == 'both':
                        if i >= accel_lookback:
                            accel_cond = all(acceleration[i-j] < 0 for j in range(accel_lookback)) and abs(acceleration[i]) > accel_thresh
                        else:
                            accel_cond = acceleration[i] < 0 and abs(acceleration[i]) > accel_thresh

                    # Jerk confirmation (optional)
                    jerk_cond = True
                    if use_jerk and jerk_thresh > 0:
                        jerk_cond = jerk[i] < -jerk_thresh

                    if pnl_ok and accel_cond and jerk_cond:
                        pnl_pct = current_pnl
                        trades.append(pnl_pct)
                        equity *= (1 + pnl_pct / 100)
                        equity_curve.append(equity)
                        peak_equity = max(peak_equity, equity)
                        position = 0
                        last_trade_bar = i
                        exited = True
                # === IMPROVEMENT #8: Midline exit delay ===
                midline_met = False
                if not exited and can_exit and exit_on_midline:
                    if midline_exit_bars_required > 1:
                        midline_met = consecutive_above[i] >= midline_exit_bars_required
                    else:
                        midline_met = osc_smooth[i] > 0 and osc_smooth[i-1] <= 0
                if not exited and midline_met:
                    pnl_pct = (price - entry_price) / entry_price * 100
                    trades.append(pnl_pct)
                    equity *= (1 + pnl_pct / 100)
                    equity_curve.append(equity)
                    peak_equity = max(peak_equity, equity)
                    position = 0
                    last_trade_bar = i
                    exited = True
                if not exited and can_exit and exit_on_opposite and sell_cond[i] and bars_since >= params['min_bars_between']:
                    pnl_pct = (price - entry_price) / entry_price * 100
                    trades.append(pnl_pct)
                    equity *= (1 + pnl_pct / 100)
                    equity_curve.append(equity)
                    peak_equity = max(peak_equity, equity)
                    position = 0
                    last_trade_bar = i
                    exited = True
                if exited:
                    continue

            if buy_cond[i] and position == 0 and bars_since >= params['min_bars_between']:
                position = 1
                entry_price = price
                highest_price = price  # Reset trailing stop tracker
                entry_bar_idx = i  # Track entry bar for min_hold_bars
                last_trade_bar = i

        if position == 1:
            pnl_pct = (close_prices[-1] - entry_price) / entry_price * 100
            trades.append(pnl_pct)
            equity *= (1 + pnl_pct / 100)
            equity_curve.append(equity)

        if not trades:
            return None

        pnls = trades
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_return = (np.prod([1 + p/100 for p in pnls]) - 1) * 100
        win_rate = len(wins) / len(trades) * 100
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.01
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        sharpe = np.mean(pnls) / np.std(pnls) if len(pnls) > 1 and np.std(pnls) > 0 else 0
        risk_adjusted = total_return * (win_rate / 100) * np.sqrt(max(1, len(trades)))

        # Calculate max drawdown from equity curve
        equity_arr = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (running_max - equity_arr) / running_max * 100
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        # Calculate Risk/Reward ratio (TP% / SL%)
        sl_pct = params.get('stop_loss_pct', 0)
        tp_pct = params.get('take_profit_pct', 0)
        risk_reward_ratio = tp_pct / sl_pct if sl_pct > 0 else 0.0

        return {
            **params,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'risk_adjusted': risk_adjusted,
            'num_trades': len(trades),
            'max_drawdown': max_drawdown,
            'risk_reward_ratio': risk_reward_ratio
        }


def create_velocity_objective(data_path):
    """Create a picklable velocity objective function."""
    return VelocityOptunaObjective(data_path)


def run_velocity_study(data_path, n_trials, seed, optimize_metric='total_return', worker_id=0):
    """
    Run a single Optuna study with velocity objective.
    Designed to be called from joblib for parallel execution.
    Returns list of results from completed trials.
    """
    import optuna
    import sys
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Check for quiet mode (set by test scripts)
    quiet_mode = os.environ.get('QUIET_WORKERS', '0') == '1'

    objective = VelocityOptunaObjective(data_path)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(50, n_trials // 5))
    )

    # Progress callback
    log_interval = max(100, n_trials // 20)  # Log every 5% or 100 trials

    def progress_callback(study, trial):
        if quiet_mode:
            return
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if n_complete % log_interval == 0 or n_complete == n_trials:
            best_val = study.best_value if study.best_trial else 0
            print(f"[Worker {worker_id}] Trial {n_complete}/{n_trials} | Best {optimize_metric}: {best_val:.2f}", flush=True)

    if not quiet_mode:
        print(f"[Worker {worker_id}] Starting {n_trials} trials (seed={seed})...", flush=True)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1, callbacks=[progress_callback])

    # Extract results from completed trials
    results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value != float('-inf'):
            result = trial.user_attrs.get('result')
            if result:
                results.append(result)

    best_val = study.best_value if study.best_trial else 0
    if not quiet_mode:
        print(f"[Worker {worker_id}] ✓ Completed! {len(results)} valid results, best: {best_val:.2f}", flush=True)

    return results


# ============================================================================
# RANGE MODEL OPTIMIZATION (for daily range prediction)
# SIMPLE VERSION - no feature group selection, just XGBoost hyperparameters
# ============================================================================

_RANGE_DATA_PATH = None
_RANGE_CACHED_DATA = None


def set_range_shared_data(X_train_scaled, y_train, feature_names=None):
    """Save range model data to temp file for parallel workers."""
    global _RANGE_DATA_PATH, _RANGE_CACHED_DATA
    _RANGE_CACHED_DATA = None

    fd, path = tempfile.mkstemp(suffix='.joblib', prefix='range_optuna_')
    os.close(fd)

    data = {
        'X_train_scaled': X_train_scaled,
        'y_train': y_train,
        'feature_names': feature_names
    }
    joblib.dump(data, path)
    _RANGE_DATA_PATH = path
    return path


def clear_range_shared_data():
    """Clean up range model shared data file."""
    global _RANGE_DATA_PATH, _RANGE_CACHED_DATA
    if _RANGE_DATA_PATH and os.path.exists(_RANGE_DATA_PATH):
        try:
            os.remove(_RANGE_DATA_PATH)
        except:
            pass
    _RANGE_DATA_PATH = None
    _RANGE_CACHED_DATA = None


class RangeModelObjective:
    """
    Optuna objective for range model XGBoost optimization.
    Uses composite objective: R² + MAPE for better trading performance.

    Composite Score = 0.6 * R² + 0.4 * (1 - MAPE/100)

    Where:
    - R² measures how well the model explains variance (0-1, higher is better)
    - MAPE measures percentage error (lower is better)

    We MAXIMIZE this score (return negative for Optuna minimization).
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self._cached_data = None

    def _load_data(self):
        if self._cached_data is None:
            self._cached_data = joblib.load(self.data_path)
        return self._cached_data

    def __call__(self, trial):
        from xgboost import XGBRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import r2_score, mean_absolute_percentage_error

        data = self._load_data()
        X = data['X_train_scaled']
        y = data['y_train']

        # XGBoost hyperparameters with CONSERVATIVE regularization ranges
        # NOTE: High regularization (reg_alpha/lambda > 1, min_child_weight > 5, gamma > 0.3)
        # can cause model collapse with small target values (~0.01-0.03 range)
        # NOTE: max_depth < 5 with many features can also cause constant predictions
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 5, 8),  # Increased min from 3 to 5
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 0.1, log=True),  # Tighter range
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 2.0, log=True),  # Higher min to prevent overfitting
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 3),  # Reduced max for better splits
            'gamma': trial.suggest_float('gamma', 0, 0.1),  # Reduced max
            'n_jobs': 1,
            'objective': 'reg:squarederror',
            'verbosity': 0
        }

        model = XGBRegressor(**params)

        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=3)
        r2_scores = []
        mape_scores = []

        for train_idx, val_idx in tscv.split(X):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])

            # R² score (can be negative for bad models)
            r2 = r2_score(y[val_idx], pred)
            r2_scores.append(r2)

            # MAPE (Mean Absolute Percentage Error)
            # Clip to avoid division by zero issues
            y_val_safe = np.clip(y[val_idx], 0.01, None)  # Minimum 0.01% range
            mape = np.mean(np.abs((y[val_idx] - pred) / y_val_safe)) * 100
            mape_scores.append(min(mape, 100))  # Cap at 100% for extreme cases

        avg_r2 = np.mean(r2_scores)
        avg_mape = np.mean(mape_scores)

        # Composite score: higher is better
        # R² component: 0-1 scale (can be negative for bad models)
        # MAPE component: convert to 0-1 scale (1 = perfect, 0 = 100% error)
        r2_component = max(0, avg_r2)  # Clip negative R² to 0
        mape_component = max(0, 1 - avg_mape / 100)  # Convert MAPE to 0-1 scale

        composite_score = 0.6 * r2_component + 0.4 * mape_component

        # Return negative because Optuna minimizes by default
        return -composite_score


def run_range_study(data_path, n_trials, seed, worker_id=0):
    """Run Optuna study for range model optimization.

    Uses composite objective: R² + MAPE (higher is better).
    Returns negative score (for minimization), so best_value closer to -1.0 is better.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Check for quiet mode (set by test scripts)
    quiet_mode = os.environ.get('QUIET_WORKERS', '0') == '1'

    objective = RangeModelObjective(data_path)

    study = optuna.create_study(
        direction='minimize',  # Minimize negative composite score (= maximize score)
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(10, n_trials // 5))
    )

    log_interval = max(10, n_trials // 10)

    def progress_callback(study, trial):
        if quiet_mode:
            return
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if n_complete % log_interval == 0 or n_complete == n_trials:
            if study.best_trial:
                # Show positive composite score (better = higher = closer to 1.0)
                best_score = -study.best_value  # Convert back to positive
                print(f"[Worker {worker_id}] Trial {n_complete}/{n_trials} | Best Score: {best_score:.4f}", flush=True)
            else:
                print(f"[Worker {worker_id}] Trial {n_complete}/{n_trials} | No valid trials yet", flush=True)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1, callbacks=[progress_callback])

    best_val = study.best_value if study.best_trial else float('inf')
    best_params = study.best_params if study.best_trial else {}
    best_score = -best_val if best_val != float('inf') else 0.0
    if not quiet_mode:
        print(f"[Worker {worker_id}] Done! Best Composite Score: {best_score:.4f}", flush=True)

    # Get top N trials for ensemble (sorted by value, ascending = best first)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value)
    top_n = 5  # Return top 5 so we can pick best 3 globally across workers

    top_trials = []
    for t in sorted_trials[:top_n]:
        top_trials.append({
            'value': t.value,  # Negative composite score
            'score': -t.value,  # Positive composite score
            'params': t.params
        })

    return {
        'best_value': best_val,  # Negative composite score (for consistency with minimize)
        'best_score': best_score,  # Positive composite score (for display)
        'best_params': best_params,
        'top_trials': top_trials,  # Top N trials for ensemble
        'n_trials': len(completed_trials)
    }


# ============================================================================
# HIGH/LOW MODEL OPTIMIZATION (separate optimization for high and low)
# This directly optimizes for predicting next_high_pct and next_low_pct
# instead of next_range_pct, which should improve R² for those targets
# ============================================================================

_HIGHLOW_DATA_PATH = None


def set_highlow_shared_data(X_train_scaled, y_high_train, y_low_train, feature_names=None):
    """Save high/low model data to temp file for parallel workers."""
    global _HIGHLOW_DATA_PATH

    data = {
        'X_train_scaled': X_train_scaled,
        'y_high_train': y_high_train,
        'y_low_train': y_low_train,
        'feature_names': feature_names
    }

    fd, path = tempfile.mkstemp(suffix='.joblib')
    os.close(fd)
    joblib.dump(data, path)
    _HIGHLOW_DATA_PATH = path
    return path


def get_highlow_data_path():
    """Get path to shared high/low data."""
    return _HIGHLOW_DATA_PATH


class HighLowModelObjective:
    """
    Optuna objective for HIGH or LOW prediction.

    Optimizes XGBoost hyperparameters directly for predicting
    next_high_pct or next_low_pct (not range).
    """

    def __init__(self, data_path, target='high'):
        self.data_path = data_path
        self.target = target  # 'high' or 'low'
        self._cached_data = None

    def _load_data(self):
        if self._cached_data is None:
            self._cached_data = joblib.load(self.data_path)
        return self._cached_data

    def __call__(self, trial):
        from xgboost import XGBRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import r2_score

        data = self._load_data()
        X = data['X_train_scaled']
        y = data['y_high_train'] if self.target == 'high' else data['y_low_train']

        # XGBoost hyperparameters - optimized for high/low prediction
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 5.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'gamma': trial.suggest_float('gamma', 0, 0.3),
            'n_jobs': 1,
            'objective': 'reg:squarederror',
            'verbosity': 0
        }

        model = XGBRegressor(**params)

        # Time Series Cross-Validation with more splits for better evaluation
        tscv = TimeSeriesSplit(n_splits=4)
        r2_scores = []
        mae_scores = []

        for train_idx, val_idx in tscv.split(X):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])

            r2 = r2_score(y[val_idx], pred)
            r2_scores.append(r2)

            mae = np.mean(np.abs(y[val_idx] - pred))
            mae_scores.append(mae)

        avg_r2 = np.mean(r2_scores)
        avg_mae = np.mean(mae_scores)

        # Composite score focused on R² (what we want to improve)
        # MAE component scaled to ~0-1 range assuming typical MAE is 0.5-2%
        r2_component = max(-0.5, avg_r2)  # Allow slightly negative R²
        mae_component = max(0, 1 - avg_mae / 2)  # 0% MAE = 1, 2% MAE = 0

        # Weight R² more heavily since that's our target metric
        composite_score = 0.7 * r2_component + 0.3 * mae_component

        return -composite_score  # Minimize negative = maximize positive


def run_highlow_study(data_path, n_trials, seed, target='high', worker_id=0):
    """
    Run Optuna study for HIGH or LOW model optimization.
    Designed to be called from joblib for parallel execution (like run_range_study).

    target: 'high' for next_high_pct, 'low' for next_low_pct
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    quiet_mode = os.environ.get('QUIET_WORKERS', '0') == '1'

    objective = HighLowModelObjective(data_path, target=target)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(5, n_trials // 3))
    )

    log_interval = max(5, n_trials // 3)

    def progress_callback(study, trial):
        if quiet_mode:
            return
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if n_complete % log_interval == 0 or n_complete == n_trials:
            if study.best_trial:
                best_score = -study.best_value
                print(f"[W{worker_id}:{target.upper()}] Trial {n_complete}/{n_trials} | Best: {best_score:.4f}", flush=True)

    # n_jobs=1 - parallelism comes from joblib spawning multiple processes
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1, callbacks=[progress_callback])

    best_val = study.best_value if study.best_trial else float('inf')
    best_params = study.best_params if study.best_trial else {}
    best_score = -best_val if best_val != float('inf') else 0.0

    if not quiet_mode:
        print(f"[Worker {worker_id}] {target.upper()} Done! Best Score: {best_score:.4f}", flush=True)

    return {
        'target': target,
        'best_value': best_val,
        'best_score': best_score,
        'best_params': best_params,
        'n_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    }
