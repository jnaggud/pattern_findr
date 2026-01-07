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
# RANGE MODEL OPTIMIZATION (for price range prediction)
# ============================================================================

_RANGE_DATA_PATH = None
_RANGE_CACHED_DATA = None


def set_range_shared_data(X_train_scaled, y_train, feature_names=None):
    """
    Save range model training data to a temp file for parallel workers.
    Returns the path to the temp file.

    Args:
        X_train_scaled: Scaled training features (numpy array or DataFrame)
        y_train: Training targets
        feature_names: List of feature names (required for feature group selection)
    """
    global _RANGE_DATA_PATH, _RANGE_CACHED_DATA

    _RANGE_CACHED_DATA = None

    fd, path = tempfile.mkstemp(suffix='.joblib', prefix='range_optuna_')
    os.close(fd)

    # Convert to numpy arrays for efficient pickling
    data = {
        'X_train_scaled': np.array(X_train_scaled) if hasattr(X_train_scaled, 'values') else X_train_scaled,
        'y_train': np.array(y_train) if hasattr(y_train, 'values') else y_train,
        'feature_names': list(feature_names) if feature_names is not None else None,
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
    Picklable objective class for range model XGBoost optimization.
    Each worker loads data from file and runs cross-validation.

    Features:
    - Optuna selects which feature GROUPS to include (Hurst, HAR, CARR, etc.)
    - SelectKBest picks top N features from selected groups
    - XGBoost hyperparameters are tuned
    """

    # Define feature groups by name patterns
    FEATURE_GROUPS = {
        'hurst': ['hurst_'],
        'har': ['har_rv_', 'har_weekly_', 'har_monthly_'],
        'range_estimators': ['garman_klass', 'rogers_satchell', 'gk_parkinson', 'rs_parkinson'],
        'carr': ['carr_'],
        'entropy': ['vol_entropy'],
        'iv_rv': ['iv_rv_'],
        'range_efficiency': ['range_efficiency', 'low_efficiency'],
        'novel_indicators': ['arwo', 'dco', 'vcmo', 'ics', 'mji', 'prf', 'ewaf', 'kfif'],
        'composite_osc': ['composite_osc', 'composite_velocity', 'composite_accel'],
        'vix': ['vix'],
        'options': ['pcr_', 'sentiment_', 'iv_weighted', 'iv_call', 'iv_put', 'iv_skew', 'max_pain', 'high_call', 'high_put'],
    }

    # Core features always included (basic price/range features)
    CORE_FEATURES = ['atr_', 'range_', 'volatility_', 'returns', 'roc_', 'rsi_', 'day_of_week', 'month',
                     'is_monday', 'is_friday', 'is_month_end', 'gap', 'sma_', 'price_vs_', 'bb_width',
                     'keltner_', 'hvol_', 'parkinson_vol', 'vol_weighted']

    def __init__(self, data_path):
        self.data_path = data_path
        self._cached_data = None

    def _load_data(self):
        if self._cached_data is None:
            self._cached_data = joblib.load(self.data_path)
        return self._cached_data

    def _get_feature_mask(self, feature_names, selected_groups):
        """Get boolean mask for features to include based on selected groups."""
        mask = []
        for fname in feature_names:
            # Always include core features
            is_core = any(core in fname for core in self.CORE_FEATURES)
            if is_core:
                mask.append(True)
                continue

            # Check if feature belongs to a selected group
            included = False
            for group_name, patterns in self.FEATURE_GROUPS.items():
                if group_name in selected_groups:
                    if any(pattern in fname for pattern in patterns):
                        included = True
                        break
            mask.append(included)
        return np.array(mask)

    def _detect_available_groups(self, feature_names):
        """Detect which feature groups actually have features in the data."""
        available = {}
        for group_name, patterns in self.FEATURE_GROUPS.items():
            # Check if any feature matches this group's patterns
            has_features = any(
                any(pattern in fname for pattern in patterns)
                for fname in feature_names
            )
            available[group_name] = has_features
        return available

    def __call__(self, trial):
        from xgboost import XGBRegressor
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error
        from sklearn.feature_selection import SelectKBest, f_regression

        data = self._load_data()
        X_train_scaled = data['X_train_scaled']
        y_train = data['y_train']
        feature_names = data.get('feature_names')

        # --- SIMPLIFIED: Skip feature group selection, just use SelectKBest ---
        # Feature group selection was causing inconsistent results between workers
        # Instead, let SelectKBest automatically pick the best features
        selected_groups = []  # Empty - not using group selection

        # --- Step 1: SelectKBest (pick top N features from ALL available) ---
        n_features = X_train_scaled.shape[1]
        if n_features > 10:
            # Let Optuna decide how many top features to keep
            k_ratio = trial.suggest_float('selectk_ratio', 0.2, 1.0)
            k = max(5, int(n_features * k_ratio))
            selector = SelectKBest(f_regression, k=k)
            X_selected = selector.fit_transform(X_train_scaled, y_train)
        else:
            X_selected = X_train_scaled

        # --- Step 3: XGBoost Hyperparameters (expanded ranges) ---
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.2, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'n_jobs': 1,
            'objective': 'reg:squarederror',
            'verbosity': 0
        }

        model = XGBRegressor(**params)

        # --- Step 4: Time Series Cross-Validation ---
        tscv = TimeSeriesSplit(n_splits=3)
        mse_scores = []
        r2_scores = []
        for train_idx, val_idx in tscv.split(X_selected):
            model.fit(X_selected[train_idx], y_train[train_idx])
            pred = model.predict(X_selected[val_idx])
            mse = mean_squared_error(y_train[val_idx], pred)
            mse_scores.append(mse)

            # Calculate R² for this fold
            y_val = y_train[val_idx]
            ss_res = np.sum((y_val - pred) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r2_scores.append(r2)

        mse_mean = np.mean(mse_scores)
        r2_mean = np.mean(r2_scores)
        n_features_used = X_selected.shape[1]

        # Store metrics for analysis
        trial.set_user_attr('mse', mse_mean)
        trial.set_user_attr('r2', r2_mean)
        trial.set_user_attr('n_features', n_features_used)
        trial.set_user_attr('selected_groups', selected_groups)

        # --- Primary: R² (higher is better) ---
        # --- Secondary Tie-Breaker: MSE (lower is better) ---
        # Since we MAXIMIZE, higher R² wins
        # When R² is equal, lower MSE wins (subtract small MSE penalty)
        # MSE is typically ~0.0001-0.001, so 1e-3 factor keeps it as tie-breaker
        mse_penalty = mse_mean * 1e-3

        composite_score = r2_mean - mse_penalty

        return composite_score


def create_range_objective(data_path):
    """Create a picklable range model objective function."""
    return RangeModelObjective(data_path)


def run_range_study(data_path, n_trials, seed, worker_id=0):
    """
    Run a single Optuna study for range model optimization.
    Designed to be called from joblib for parallel execution.
    Returns the best trial info.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = RangeModelObjective(data_path)

    study = optuna.create_study(
        direction='maximize',  # Optimizing for R² (higher is better)
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(10, n_trials // 5)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)  # Prune bad trials early
    )

    # Progress callback
    log_interval = max(10, n_trials // 10)  # Log every 10%

    def progress_callback(study, trial):
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if n_complete % log_interval == 0 or n_complete == n_trials:
            best_val = study.best_value if study.best_trial else 0.0
            best_r2 = study.best_trial.user_attrs.get('r2', best_val) if study.best_trial else 0.0
            print(f"[Worker {worker_id}] Trial {n_complete}/{n_trials} | Best R²: {best_r2:.4f}", flush=True)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1, callbacks=[progress_callback])

    best_val = study.best_value if study.best_trial else float('-inf')  # Maximizing, so -inf is worst
    best_params = study.best_params if study.best_trial else {}

    # Extract additional metrics from best trial
    best_mse = float('inf')
    best_r2 = 0.0
    best_n_features = 0
    best_groups = []
    if study.best_trial:
        best_mse = study.best_trial.user_attrs.get('mse', best_val)
        best_r2 = study.best_trial.user_attrs.get('r2', 0)
        best_n_features = study.best_trial.user_attrs.get('n_features', 0)
        best_groups = study.best_trial.user_attrs.get('selected_groups', [])

    print(f"[Worker {worker_id}] Done! Best MSE: {best_mse:.6f}, R²: {best_r2:.4f}, Features: {best_n_features}", flush=True)

    return {
        'best_value': best_val,  # Composite score (for comparison)
        'best_mse': best_mse,    # Pure MSE (for display)
        'best_r2': best_r2,      # R² as tie-breaker
        'best_n_features': best_n_features,
        'best_groups': best_groups,
        'best_params': best_params,
        'n_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    }
