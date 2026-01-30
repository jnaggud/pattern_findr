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
                              # Exit strategy constraints
                              force_midline_exit=False, force_opposite_exit=True,
                              sl_range=None, tp_range=None,
                              # Drawdown penalty settings
                              use_drawdown_penalty=False, max_drawdown_threshold=15.0,
                              drawdown_penalty_weight=0.3,
                              # V2 filter constraints (from UI checkboxes)
                              v2_filter_settings=None):
    """
    Save velocity optimization data to a temp file for parallel workers.
    Returns the path to the temp file.

    Args:
        all_oscillators: Dict mapping oscillator names to their values (for searching across oscillator types)
        v2_indicators: Dict with 'rsc', 'mfi2', 'sei' arrays for V2 filter optimization
        force_midline_exit: If True, always use exit_on_midline_cross=True
        force_opposite_exit: If True, always use exit_on_opposite_signal=True
        sl_range: Tuple (min_pct, max_pct) for stop loss search range
        tp_range: Tuple (min_pct, max_pct) for take profit search range
        use_drawdown_penalty: If True, penalize strategies with high max drawdown
        v2_filter_settings: Dict with filter constraints from UI checkboxes:
            {'use_regime_filter': True/False, 'regime_threshold': float,
             'use_fragility_filter': True/False, 'fragility_threshold': float,
             'use_entropy_filter': True/False, 'entropy_threshold': float}
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

    # Process V2 indicators if provided
    processed_v2 = {}
    if v2_indicators:
        for name, values in v2_indicators.items():
            if values is not None:
                processed_v2[name] = np.array(values) if hasattr(values, 'values') else values

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
        'all_oscillators': processed_oscillators,  # All oscillator types for search
        'v2_indicators': processed_v2,  # V2 indicators for filter optimization
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

        # Exit strategy constraints from UI
        force_midline_exit = data.get('force_midline_exit', False)
        force_opposite_exit = data.get('force_opposite_exit', True)
        sl_range = data.get('sl_range', (0.0, 10.0))
        tp_range = data.get('tp_range', (0.0, 20.0))

        # Drawdown penalty settings
        use_drawdown_penalty = data.get('use_drawdown_penalty', False)
        max_drawdown_threshold = data.get('max_drawdown_threshold', 15.0)
        drawdown_penalty_weight = data.get('drawdown_penalty_weight', 0.3)

        # Determine available oscillator types
        if all_oscillators and len(all_oscillators) > 1:
            oscillator_types = list(all_oscillators.keys())
        else:
            oscillator_types = ['composite_smooth']

        # Suggest parameters - including oscillator type if multiple available
        params = {}

        # Add oscillator_type as a searchable parameter if we have multiple oscillators
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

        params.update({
            'signal_type': trial.suggest_categorical('signal_type', [
                'velocity_crossover_and_zone', 'velocity_crossover_or_zone', 'zone_only',
                'momentum', 'any_reversal', 'double_bottom', 'divergence', 'breakout'
            ]),
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
        })

        if use_extra_indicators:
            params['rsi_filter'] = trial.suggest_categorical('rsi_filter', ['none', 'oversold_only', 'overbought_only', 'both'])
            params['rsi_period'] = trial.suggest_int('rsi_period', 5, 30)
            params['rsi_oversold'] = trial.suggest_int('rsi_oversold', 15, 40)
            params['rsi_overbought'] = trial.suggest_int('rsi_overbought', 60, 85)
            params['use_macd_confirm'] = trial.suggest_categorical('use_macd_confirm', [True, False])
            params['use_bb_filter'] = trial.suggest_categorical('use_bb_filter', [True, False])

        # V2 indicator filters (Regime, Fragility, Entropy)
        # RESPECT UI CHECKBOX SETTINGS: if user checked a filter, force it ON for all trials
        v2_indicators = data.get('v2_indicators', {})
        v2_settings = data.get('v2_filter_settings', {})

        if v2_indicators:
            # Regime filter: use UI setting if specified, otherwise don't use it
            if v2_settings.get('use_regime_filter', False) and 'rsc' in v2_indicators:
                params['use_regime_filter'] = True
                # Search for optimal threshold
                params['regime_threshold'] = trial.suggest_float('regime_threshold', -0.5, 0.5)
            else:
                params['use_regime_filter'] = False
                params['regime_threshold'] = 0.0

            # Fragility filter: use UI setting if specified
            if v2_settings.get('use_fragility_filter', False) and 'mfi2' in v2_indicators:
                params['use_fragility_filter'] = True
                params['fragility_threshold'] = trial.suggest_float('fragility_threshold', 0.2, 0.8)
            else:
                params['use_fragility_filter'] = False
                params['fragility_threshold'] = 0.5

            # Entropy filter: use UI setting if specified
            if v2_settings.get('use_entropy_filter', False) and 'sei' in v2_indicators:
                params['use_entropy_filter'] = True
                params['entropy_threshold'] = trial.suggest_float('entropy_threshold', 0.3, 0.9)
            else:
                params['use_entropy_filter'] = False
                params['entropy_threshold'] = 0.7
        else:
            params['use_regime_filter'] = False
            params['use_fragility_filter'] = False
            params['use_entropy_filter'] = False

        # Select the oscillator values based on the chosen type
        selected_osc_type = params['oscillator_type']
        if selected_osc_type in all_oscillators:
            selected_osc_values = all_oscillators[selected_osc_type]
        else:
            selected_osc_values = osc_values  # Fallback to default

        # Run backtest with selected oscillator and V2 indicators
        result = self._run_backtest(params, close_prices, selected_osc_values, rsi_cache,
                                     macd_histogram, bb_upper, bb_lower, use_extra_indicators,
                                     v2_indicators)

        if result is None:
            return float('-inf')

        # Get the base score from the optimization metric
        base_score = result[optimize_metric]

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
                      v2_indicators=None):
        """Fast vectorized backtest with V2 indicator filters."""
        import pandas as pd

        # Apply smoothing
        if params['vel_smoothing'] > 1:
            osc_smooth = pd.Series(osc_values).rolling(window=params['vel_smoothing']).mean().bfill().values
        else:
            osc_smooth = osc_values

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

        # Trading simulation with equity curve tracking for max drawdown
        position = 0
        entry_price = 0.0
        entry_bar_idx = 0  # Track entry bar for min_hold_bars
        last_trade_bar = -params['min_bars_between']
        trades = []
        exit_on_opposite = params.get('exit_on_opposite_signal', True)
        exit_on_midline = params.get('exit_on_midline_cross', False)
        min_hold_bars = params.get('min_hold_bars', 1)

        # Track equity curve for max drawdown calculation
        equity = 100.0  # Start with $100
        equity_curve = [equity]
        peak_equity = equity

        for i in range(1, len(close_prices)):
            price = close_prices[i]
            bars_since = i - last_trade_bar
            bars_held = i - entry_bar_idx
            can_exit = bars_held >= min_hold_bars

            if position == 1 and can_exit:
                if params['stop_loss_pct'] > 0 and price <= entry_price * (1 - params['stop_loss_pct'] / 100):
                    pnl_pct = (price - entry_price) / entry_price * 100
                    trades.append(pnl_pct)
                    equity *= (1 + pnl_pct / 100)
                    equity_curve.append(equity)
                    peak_equity = max(peak_equity, equity)
                    position = 0
                    last_trade_bar = i
                    continue
                if params['take_profit_pct'] > 0 and price >= entry_price * (1 + params['take_profit_pct'] / 100):
                    pnl_pct = (price - entry_price) / entry_price * 100
                    trades.append(pnl_pct)
                    equity *= (1 + pnl_pct / 100)
                    equity_curve.append(equity)
                    peak_equity = max(peak_equity, equity)
                    position = 0
                    last_trade_bar = i
                    continue
                # Acceleration Reversal Exit - exit on momentum reversal before stop loss
                if params.get('use_accel_exit', False):
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
                        continue
                if exit_on_midline and osc_smooth[i] > 0 and osc_smooth[i-1] <= 0:
                    pnl_pct = (price - entry_price) / entry_price * 100
                    trades.append(pnl_pct)
                    equity *= (1 + pnl_pct / 100)
                    equity_curve.append(equity)
                    peak_equity = max(peak_equity, equity)
                    position = 0
                    last_trade_bar = i
                    continue
                if exit_on_opposite and sell_cond[i] and bars_since >= params['min_bars_between']:
                    pnl_pct = (price - entry_price) / entry_price * 100
                    trades.append(pnl_pct)
                    equity *= (1 + pnl_pct / 100)
                    equity_curve.append(equity)
                    peak_equity = max(peak_equity, equity)
                    position = 0
                    last_trade_bar = i
                    continue

            if buy_cond[i] and position == 0 and bars_since >= params['min_bars_between']:
                position = 1
                entry_price = price
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
