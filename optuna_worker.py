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
                              all_oscillators=None):
    """
    Save velocity optimization data to a temp file for parallel workers.
    Returns the path to the temp file.

    Args:
        all_oscillators: Dict mapping oscillator names to their values (for searching across oscillator types)
    """
    global _VELOCITY_DATA_PATH, _VELOCITY_CACHED_DATA

    _VELOCITY_CACHED_DATA = None

    fd, path = tempfile.mkstemp(suffix='.joblib', prefix='velocity_optuna_')
    os.close(fd)

    # Process all oscillators if provided
    processed_oscillators = {}
    if all_oscillators:
        for name, values in all_oscillators.items():
            processed_oscillators[name] = np.array(values) if hasattr(values, 'values') else values

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

        params.update({
            'signal_type': trial.suggest_categorical('signal_type', [
                'velocity_crossover_and_zone', 'velocity_crossover_or_zone', 'zone_only',
                'momentum', 'any_reversal', 'double_bottom', 'divergence', 'breakout'
            ]),
            'vel_smoothing': trial.suggest_int('vel_smoothing', 1, 15),
            'oversold_threshold': trial.suggest_float('oversold_threshold', -0.6, -0.02),
            'overbought_threshold': trial.suggest_float('overbought_threshold', 0.02, 0.6),
            'stop_loss_pct': trial.suggest_float('stop_loss_pct', 0.0, 10.0),
            'take_profit_pct': trial.suggest_float('take_profit_pct', 0.0, 20.0),
            'min_bars_between': trial.suggest_int('min_bars_between', 1, 15),
            'require_accel': trial.suggest_categorical('require_accel', [True, False]),
            'extreme_zone_mult': trial.suggest_float('extreme_zone_mult', 1.1, 2.5),
            'exit_on_opposite_signal': trial.suggest_categorical('exit_on_opposite_signal', [True, False]),
            'exit_on_midline_cross': trial.suggest_categorical('exit_on_midline_cross', [True, False]),
        })

        if use_extra_indicators:
            params['rsi_filter'] = trial.suggest_categorical('rsi_filter', ['none', 'oversold_only', 'overbought_only', 'both'])
            params['rsi_period'] = trial.suggest_int('rsi_period', 5, 30)
            params['rsi_oversold'] = trial.suggest_int('rsi_oversold', 15, 40)
            params['rsi_overbought'] = trial.suggest_int('rsi_overbought', 60, 85)
            params['use_macd_confirm'] = trial.suggest_categorical('use_macd_confirm', [True, False])
            params['use_bb_filter'] = trial.suggest_categorical('use_bb_filter', [True, False])

        # Select the oscillator values based on the chosen type
        selected_osc_type = params['oscillator_type']
        if selected_osc_type in all_oscillators:
            selected_osc_values = all_oscillators[selected_osc_type]
        else:
            selected_osc_values = osc_values  # Fallback to default

        # Run backtest with selected oscillator
        result = self._run_backtest(params, close_prices, selected_osc_values, rsi_cache,
                                     macd_histogram, bb_upper, bb_lower, use_extra_indicators)

        if result is None:
            return float('-inf')

        # Store result for later retrieval
        trial.set_user_attr('result', result)

        return result[optimize_metric]

    def _run_backtest(self, params, close_prices, osc_values, rsi_cache,
                      macd_histogram, bb_upper, bb_lower, use_extra_indicators):
        """Fast vectorized backtest."""
        import pandas as pd

        # Apply smoothing
        if params['vel_smoothing'] > 1:
            osc_smooth = pd.Series(osc_values).rolling(window=params['vel_smoothing']).mean().bfill().values
        else:
            osc_smooth = osc_values

        velocity = np.diff(osc_smooth, prepend=osc_smooth[0])
        acceleration = np.diff(velocity, prepend=velocity[0])

        # Build conditions
        vel_cross_up = (velocity > 0) & (np.roll(velocity, 1) <= 0)
        vel_cross_down = (velocity < 0) & (np.roll(velocity, 1) >= 0)
        in_oversold = osc_smooth < params['oversold_threshold']
        in_overbought = osc_smooth > params['overbought_threshold']
        extreme_oversold = osc_smooth < (params['oversold_threshold'] * params['extreme_zone_mult'])
        extreme_overbought = osc_smooth > (params['overbought_threshold'] * params['extreme_zone_mult'])

        vel_std = pd.Series(velocity).rolling(10, min_periods=1).std().fillna(np.std(velocity)).values
        strong_momentum_up = velocity > vel_std * 1.5
        strong_momentum_down = velocity < -vel_std * 1.5

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
            vel_cross_up_count = pd.Series(vel_cross_up.astype(int)).rolling(10).sum().values
            buy_cond = (vel_cross_up_count >= 2) & in_oversold
            vel_cross_down_count = pd.Series(vel_cross_down.astype(int)).rolling(10).sum().values
            sell_cond = (vel_cross_down_count >= 2) & in_overbought
        elif sig_type == 'divergence':
            price_series = pd.Series(close_prices)
            osc_series = pd.Series(osc_smooth)
            price_lower_low = (close_prices < price_series.rolling(5).min().shift(1).values)
            osc_higher_low = (osc_smooth > osc_series.rolling(5).min().shift(1).values)
            buy_cond = price_lower_low & osc_higher_low & in_oversold
            price_higher_high = (close_prices > price_series.rolling(5).max().shift(1).values)
            osc_lower_high = (osc_smooth < osc_series.rolling(5).max().shift(1).values)
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

        # Trading simulation
        position = 0
        entry_price = 0.0
        last_trade_bar = -params['min_bars_between']
        trades = []
        exit_on_opposite = params.get('exit_on_opposite_signal', True)
        exit_on_midline = params.get('exit_on_midline_cross', False)

        for i in range(1, len(close_prices)):
            price = close_prices[i]
            bars_since = i - last_trade_bar

            if position == 1:
                if params['stop_loss_pct'] > 0 and price <= entry_price * (1 - params['stop_loss_pct'] / 100):
                    trades.append((price - entry_price) / entry_price * 100)
                    position = 0
                    last_trade_bar = i
                    continue
                if params['take_profit_pct'] > 0 and price >= entry_price * (1 + params['take_profit_pct'] / 100):
                    trades.append((price - entry_price) / entry_price * 100)
                    position = 0
                    last_trade_bar = i
                    continue
                if exit_on_midline and osc_smooth[i] > 0 and osc_smooth[i-1] <= 0:
                    trades.append((price - entry_price) / entry_price * 100)
                    position = 0
                    last_trade_bar = i
                    continue
                if exit_on_opposite and sell_cond[i] and bars_since >= params['min_bars_between']:
                    trades.append((price - entry_price) / entry_price * 100)
                    position = 0
                    last_trade_bar = i
                    continue

            if buy_cond[i] and position == 0 and bars_since >= params['min_bars_between']:
                position = 1
                entry_price = price
                last_trade_bar = i

        if position == 1:
            trades.append((close_prices[-1] - entry_price) / entry_price * 100)

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

        return {
            **params,
            'total_return': total_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'risk_adjusted': risk_adjusted,
            'num_trades': len(trades)
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

    objective = VelocityOptunaObjective(data_path)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(50, n_trials // 5))
    )

    # Progress callback
    log_interval = max(100, n_trials // 20)  # Log every 5% or 100 trials

    def progress_callback(study, trial):
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if n_complete % log_interval == 0 or n_complete == n_trials:
            best_val = study.best_value if study.best_trial else 0
            print(f"[Worker {worker_id}] Trial {n_complete}/{n_trials} | Best {optimize_metric}: {best_val:.2f}", flush=True)

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

    objective = RangeModelObjective(data_path)

    study = optuna.create_study(
        direction='minimize',  # Minimize negative composite score (= maximize score)
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(10, n_trials // 5))
    )

    log_interval = max(10, n_trials // 10)

    def progress_callback(study, trial):
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
