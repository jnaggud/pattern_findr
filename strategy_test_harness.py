#!/usr/bin/env python3
"""
Strategy Test Harness — Additive/Cumulative Testing

Tests each of the 20 NEW_STRATEGIES.md improvements ADDITIVELY:
  Step 0: Baseline (no enhancements)
  Step 1: Baseline + Strategy #1 (Kalman)
  Step 2: Baseline + Strategy #1 + #2 (Wavelet)
  ...etc. Each strategy builds on top of all previous.

Usage:
    python strategy_test_harness.py                    # Run all 20 cumulatively
    python strategy_test_harness.py --trials 5000      # Custom trial count
    python strategy_test_harness.py --ticker SPY       # Different ticker
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import optuna
from datetime import datetime, timedelta
from copy import deepcopy
from collections import OrderedDict

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ['QUIET_WORKERS'] = '1'

# Force unbuffered output so we can see progress in real-time
import builtins
_original_print = builtins.print
def _flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)
builtins.print = _flush_print

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import joblib

from novel_indicators import (
    calculate_arwo, calculate_dco, calculate_vcmo, calculate_ics,
    calculate_mji, calculate_prf, calculate_ewaf, calculate_kfif,
)
from oscillator_indicators import create_composite_oscillator_features
import optuna_worker

# Optional imports
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    from filterpy.kalman import KalmanFilter as KF1D
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from scipy.optimize import minimize_scalar
    from scipy.stats import norm, entropy as scipy_entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from imblearn.over_sampling import BorderlineSMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Data Infrastructure
# =============================================================================

def fetch_data(ticker, interval, days=59):
    """Fetch intraday data from yfinance."""
    end_date = datetime.now() + timedelta(days=1)
    start_date = datetime.now() - timedelta(days=days)
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower()
    if 'volume' in df.columns:
        df = df[df['volume'] > 0]
    return df


def fetch_cross_asset_data(interval, days=59):
    """Fetch VIX and NQ data for cross-asset strategies."""
    end_date = datetime.now() + timedelta(days=1)
    start_date = datetime.now() - timedelta(days=days)
    assets = {}
    for ticker in ['^VIX', 'NQ=F', 'DX-Y.NYB']:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = df.columns.str.lower()
            if len(df) > 50:
                assets[ticker] = df
        except Exception:
            pass
    return assets


def compute_oscillators_standard(train_df):
    """Compute oscillators on training data only, center=False."""
    oscillators = {}
    osc_df = create_composite_oscillator_features(train_df)
    for col in ['osc_composite_smooth', 'composite_oscillator', 'osc_composite']:
        if col in osc_df.columns:
            oscillators['composite_smooth'] = osc_df[col].values
            break

    for name, calc_fn, unpack in [
        ('arwo', calculate_arwo, False), ('dco', calculate_dco, False),
        ('vcmo', calculate_vcmo, False), ('ics', calculate_ics, False),
        ('mji', calculate_mji, False), ('prf', calculate_prf, False),
        ('ewaf', calculate_ewaf, False), ('kfif', calculate_kfif, True),
    ]:
        try:
            result = calc_fn(train_df)
            series = result[0] if unpack else result
            oscillators[name] = np.nan_to_num(series.values, nan=0.0)
        except Exception as e:
            print(f"  WARNING: {name} failed: {e}")

    return oscillators


def precompute_indicators(close_prices):
    """Pre-compute RSI, MACD, Bollinger Bands."""
    close_series = pd.Series(close_prices)

    def calc_rsi_np(prices, period):
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    rsi_cache = {p: calc_rsi_np(close_prices, p) for p in [5, 7, 10, 14, 21, 30]}
    ema12 = close_series.ewm(span=12, adjust=False).mean().values
    ema26 = close_series.ewm(span=26, adjust=False).mean().values
    macd_histogram = (ema12 - ema26) - pd.Series(ema12 - ema26).ewm(span=9, adjust=False).mean().values
    bb_sma = close_series.rolling(20).mean().values
    bb_std = close_series.rolling(20).std().values
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std

    return rsi_cache, macd_histogram, bb_upper, bb_lower


def run_optimization(close_prices, osc_values, all_oscillators, n_trials, n_workers,
                     optimize_metric='risk_adjusted', label="",
                     high_prices=None, low_prices=None, volume=None,
                     use_drawdown_penalty=False, max_drawdown_threshold=15.0,
                     drawdown_penalty_weight=0.3):
    """Run parallel Optuna optimization."""
    rsi_cache, macd_histogram, bb_upper, bb_lower = precompute_indicators(close_prices)

    data_path = optuna_worker.set_velocity_shared_data(
        close_prices, osc_values, rsi_cache, macd_histogram, bb_upper, bb_lower,
        optimize_metric=optimize_metric, use_extra_indicators=True,
        all_oscillators=all_oscillators,
        high_prices=high_prices, low_prices=low_prices, volume=volume,
        use_drawdown_penalty=use_drawdown_penalty,
        max_drawdown_threshold=max_drawdown_threshold,
        drawdown_penalty_weight=drawdown_penalty_weight,
    )

    trials_per_worker = n_trials // n_workers
    start_time = time.time()

    results_lists = joblib.Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
        joblib.delayed(optuna_worker.run_velocity_study)(
            data_path, trials_per_worker, 42 + i,
            optimize_metric=optimize_metric, worker_id=0
        )
        for i in range(n_workers)
    )

    all_results = []
    for rl in results_lists:
        if rl:
            all_results.extend(rl)

    duration = time.time() - start_time
    print(f"  [{label}] {len(all_results):,} valid / {n_trials:,} trials in {duration:.0f}s")

    try:
        os.remove(data_path)
    except Exception:
        pass

    return all_results


def evaluate_on_test(params, test_df, all_oscillators_test):
    """Evaluate a parameter set on test data."""
    close_prices = test_df['close'].values
    high_prices = test_df['high'].values if 'high' in test_df.columns else None
    low_prices = test_df['low'].values if 'low' in test_df.columns else None
    volume = test_df['volume'].values if 'volume' in test_df.columns else None

    osc_type = params.get('oscillator_type', 'composite_smooth')
    if osc_type in all_oscillators_test:
        osc_values = all_oscillators_test[osc_type]
    else:
        osc_df = create_composite_oscillator_features(test_df)
        for col in ['osc_composite_smooth', 'composite_oscillator', 'osc_composite']:
            if col in osc_df.columns:
                osc_values = osc_df[col].values
                break
        else:
            return None

    rsi_cache, macd_histogram, bb_upper, bb_lower = precompute_indicators(close_prices)

    data_path = optuna_worker.set_velocity_shared_data(
        close_prices, osc_values, rsi_cache, macd_histogram, bb_upper, bb_lower,
        optimize_metric='risk_adjusted', use_extra_indicators=True,
        all_oscillators=all_oscillators_test,
        high_prices=high_prices, low_prices=low_prices, volume=volume,
    )
    objective = optuna_worker.VelocityOptunaObjective(data_path)
    result = objective._run_backtest(
        params, close_prices, osc_values, rsi_cache,
        macd_histogram, bb_upper, bb_lower, True, {},
        high_prices=high_prices, low_prices=low_prices, volume=volume,
        all_oscillators=all_oscillators_test,
    )
    try:
        os.remove(data_path)
    except Exception:
        pass
    return result


def select_best(results, top_n=5, metric='risk_adjusted'):
    """Select top N results by metric."""
    valid = [r for r in results if r and metric in r and r[metric] is not None]
    valid.sort(key=lambda x: x[metric], reverse=True)
    return valid[:top_n]


# =============================================================================
# Strategy Implementations (each operates on and returns modified oscillators or masks)
# =============================================================================

def apply_kalman(all_oscillators):
    """#1: Kalman filter smoothing."""
    if not FILTERPY_AVAILABLE:
        return None, "filterpy not installed"
    smoothed = {}
    for name, vals in all_oscillators.items():
        if name == 'kfif':
            smoothed[name] = vals
            continue
        kf = KF1D(dim_x=2, dim_z=1)
        kf.x = np.array([[vals[0]], [0.0]])
        kf.F = np.array([[1, 1], [0, 1]])
        kf.H = np.array([[1, 0]])
        kf.P *= 10.0
        kf.R = np.array([[1.0]])
        kf.Q = np.array([[0.01, 0], [0, 0.01]])
        filtered = np.zeros(len(vals))
        for i in range(len(vals)):
            kf.predict()
            kf.update(vals[i])
            filtered[i] = kf.x[0, 0]
        smoothed[name] = np.clip(filtered, -1, 1)
    return smoothed, None


def apply_wavelet(all_oscillators, wavelet='db4', level=3):
    """#2: Wavelet denoising."""
    if not PYWT_AVAILABLE:
        return None, "PyWavelets not installed"
    denoised = {}
    for name, vals in all_oscillators.items():
        try:
            coeffs = pywt.wavedec(vals, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(vals)))
            new_coeffs = [coeffs[0]]
            for c in coeffs[1:]:
                new_coeffs.append(pywt.threshold(c, threshold, mode='soft'))
            reconstructed = pywt.waverec(new_coeffs, wavelet)[:len(vals)]
            denoised[name] = np.clip(reconstructed, -1, 1)
        except Exception:
            denoised[name] = vals
    return denoised, None


def compute_hmm_mask(close_prices, n_states=3):
    """#3: HMM regime detection — returns boolean mask of tradeable bars."""
    if not HMMLEARN_AVAILABLE:
        return None, "hmmlearn not installed"
    returns = np.diff(np.log(close_prices))
    volatility = pd.Series(returns).rolling(20, min_periods=1).std().values
    features = np.column_stack([returns, volatility])
    features = np.nan_to_num(features, nan=0.0)
    model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=200, random_state=42)
    try:
        model.fit(features)
        states = model.predict(features)
    except Exception as e:
        return None, f"HMM fit failed: {e}"
    state_stats = {}
    for s in range(n_states):
        mask = states == s
        if mask.sum() > 0:
            state_stats[s] = abs(np.mean(returns[mask]))
    best_state = min(state_stats.keys(), key=lambda s: state_stats[s])
    trade_mask = np.zeros(len(close_prices), dtype=bool)
    trade_mask[1:] = (states == best_state)
    trade_mask[0] = trade_mask[1]
    pct = trade_mask.sum() / len(trade_mask) * 100
    return trade_mask, f"Regime {best_state} ({pct:.0f}% of bars)"


def apply_hmm_to_oscillators(all_oscillators, mask):
    """Zero out oscillator values on non-tradeable bars."""
    filtered = {}
    for name, vals in all_oscillators.items():
        modified = vals.copy()
        modified[~mask] = 0.0
        filtered[name] = modified
    return filtered


def compute_ou_thresholds(osc_values, window=300):
    """#4: OU optimal entry/exit thresholds."""
    if not SCIPY_AVAILABLE:
        return None, None, "scipy not installed"
    n = min(len(osc_values), window)
    vals = osc_values[-n:]
    dt = 1.0
    x = vals[:-1]
    dx = np.diff(vals)
    if np.std(x) < 1e-10:
        return None, None, "no variance"
    b = np.sum(dx * (x - np.mean(x))) / np.sum((x - np.mean(x))**2)
    a = np.mean(dx) - b * np.mean(x)
    mu = -b / dt
    if mu <= 0:
        return None, None, "no mean reversion"
    theta = a / (mu * dt)
    residuals = dx - a - b * x
    sigma = np.std(residuals) / np.sqrt(dt)
    spread = sigma / np.sqrt(2 * mu)
    entry_level = theta - 1.0 * spread
    exit_level = theta + 0.5 * spread
    return entry_level, exit_level, None


def compute_gt_score(result, n_subperiods=4):
    """#5: GT-Score composite objective."""
    if result is None:
        return float('-inf')
    total_return = result.get('total_return', 0)
    max_dd = result.get('max_drawdown', 100)
    win_rate = result.get('win_rate', 0)
    n_trades = result.get('num_trades', 0)
    if n_trades < 5:
        return float('-inf')
    ret_score = min(total_return / 20.0, 1.0)
    dd_score = max(0, 1.0 - max_dd / 15.0)
    wr_score = (win_rate - 50) / 30.0
    trade_score = min(n_trades / 100.0, 1.0)
    return 0.30 * ret_score + 0.25 * dd_score + 0.25 * wr_score + 0.20 * trade_score


def apply_instance_selection(close_prices, all_oscillators, forward_bars=10, min_pct=0.15):
    """#6: Label bars by forward return magnitude, mask out neutral."""
    n = len(close_prices)
    fwd_returns = np.zeros(n)
    for i in range(n - forward_bars):
        fwd_returns[i] = (close_prices[i + forward_bars] - close_prices[i]) / close_prices[i] * 100
    strong_mask = np.abs(fwd_returns) > min_pct
    filtered = {}
    for name, vals in all_oscillators.items():
        modified = vals.copy()
        modified[~strong_mask] = 0.0
        filtered[name] = modified
    pct = strong_mask.sum() / n * 100
    return filtered, f"Kept {pct:.0f}% of bars"


def build_xgboost_mask(close_prices, osc_values, all_oscillators, volume, high_prices, low_prices,
                        forward_bars=10, min_win_pct=0.1, conf_threshold=0.5):
    """#7: XGBoost signal classifier — returns confidence mask."""
    if not XGBOOST_AVAILABLE:
        return None, None, "xgboost not installed"
    n = len(close_prices)
    returns_1 = np.zeros(n)
    returns_1[1:] = np.diff(close_prices) / close_prices[:-1]
    vol_20 = pd.Series(returns_1).rolling(20, min_periods=1).std().values
    atr = np.zeros(n)
    if high_prices is not None and low_prices is not None:
        tr = np.maximum(high_prices - low_prices,
                        np.maximum(np.abs(high_prices - np.roll(close_prices, 1)),
                                   np.abs(low_prices - np.roll(close_prices, 1))))
        tr[0] = high_prices[0] - low_prices[0]
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    vol_ratio = np.ones(n)
    if volume is not None:
        vol_sma = pd.Series(volume.astype(float)).rolling(20, min_periods=1).mean().values
        vol_ratio = volume / (vol_sma + 1e-10)
    price_series = pd.Series(close_prices)
    rolling_min = price_series.rolling(20, min_periods=1).min().values
    rolling_max = price_series.rolling(20, min_periods=1).max().values
    price_range = rolling_max - rolling_min
    range_pos = np.where(price_range > 0, (close_prices - rolling_min) / price_range, 0.5)

    features, labels = [], []
    for i in range(50, n - forward_bars):
        vel = osc_values[i] - osc_values[i - 1] if i > 0 else 0
        acc = vel - (osc_values[i - 1] - osc_values[i - 2]) if i > 1 else 0
        features.append([osc_values[i], vel, acc, vol_20[i], atr[i], vol_ratio[i], range_pos[i], returns_1[i]])
        fwd_ret = (close_prices[i + forward_bars] - close_prices[i]) / close_prices[i] * 100
        labels.append(1 if fwd_ret > min_win_pct else 0)

    X, y = np.array(features), np.array(labels)
    if len(y) < 100 or y.sum() < 10:
        return None, None, "insufficient data"

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          use_label_encoder=False, eval_metric='logloss', verbosity=0)
    model.fit(X, y)
    train_acc = (model.predict(X) == y).mean()

    # Score all bars
    confidence = np.zeros(n)
    for i in range(50, n):
        vel = osc_values[i] - osc_values[i - 1] if i > 0 else 0
        acc = vel - (osc_values[i - 1] - osc_values[i - 2]) if i > 1 else 0
        feat = np.array([[osc_values[i], vel, acc, vol_20[i], atr[i], vol_ratio[i], range_pos[i], returns_1[i]]])
        confidence[i] = model.predict_proba(feat)[0][1]

    high_conf_mask = confidence >= conf_threshold
    pct = high_conf_mask.sum() / len(high_conf_mask) * 100
    return high_conf_mask, model, f"Train acc: {train_acc:.1%}, high-conf: {pct:.0f}%"


def compute_transfer_entropy(source_returns, target_returns, lag=1, n_bins=10):
    """#8: Transfer entropy from source to target."""
    n = min(len(source_returns), len(target_returns))
    src = source_returns[-n:]
    tgt = target_returns[-n:]
    # Discretize
    src_binned = np.digitize(src, np.linspace(src.min(), src.max(), n_bins))
    tgt_binned = np.digitize(tgt, np.linspace(tgt.min(), tgt.max(), n_bins))
    # Simple TE approximation: I(tgt_t+1; src_t | tgt_t)
    # Use histogram-based estimation
    te = 0.0
    tgt_past = tgt_binned[:-lag]
    tgt_future = tgt_binned[lag:]
    src_past = src_binned[:-lag]
    n_samples = len(tgt_past)
    # Joint and marginal counts
    from collections import Counter
    joint_3 = Counter(zip(tgt_future, tgt_past, src_past))
    joint_2_ty = Counter(zip(tgt_future, tgt_past))
    marginal_y = Counter(tgt_past)
    joint_2_ys = Counter(zip(tgt_past, src_past))
    for (tf, tp, sp), count in joint_3.items():
        p_tftp_sp = count / n_samples
        p_tf_tp = joint_2_ty[(tf, tp)] / n_samples
        p_tp = marginal_y[tp] / n_samples
        p_tp_sp = joint_2_ys[(tp, sp)] / n_samples
        if p_tp > 0 and p_tp_sp > 0 and p_tf_tp > 0:
            te += p_tftp_sp * np.log2((p_tftp_sp * p_tp) / (p_tf_tp * p_tp_sp) + 1e-10)
    return max(te, 0.0)


def compute_adaptive_stoch_vol(close_prices, window=50):
    """#9: Adaptive stochastic volatility — simplified Bayesian-inspired smoothing."""
    returns = np.zeros(len(close_prices))
    returns[1:] = np.diff(np.log(close_prices))
    # Exponentially weighted variance with adaptive decay
    n = len(returns)
    log_vol = np.zeros(n)
    log_vol[0] = np.log(max(abs(returns[1]) if len(returns) > 1 else 0.01, 1e-6))
    # Random walk SV: log(sigma_t^2) = log(sigma_{t-1}^2) + eta_t
    # Update with Kalman-like: combine prior (persistence) with observation (|return|)
    alpha = 0.94  # Decay
    for i in range(1, n):
        prior = alpha * log_vol[i - 1]
        obs = np.log(max(abs(returns[i]), 1e-6))
        # Adaptive weighting based on how far observation is from prior
        innovation = obs - prior
        adapt_weight = min(0.3, 0.05 + 0.1 * abs(innovation))
        log_vol[i] = (1 - adapt_weight) * prior + adapt_weight * obs
    vol_estimate = np.exp(log_vol / 2)
    return vol_estimate


def apply_nolaw_filter(all_oscillators, order=3, noise_var=0.1):
    """#10: Non-Linear Adaptive Wiener filter."""
    smoothed = {}
    for name, vals in all_oscillators.items():
        n = len(vals)
        filtered = np.zeros(n)
        filtered[0] = vals[0]
        local_window = max(order * 2, 10)
        for i in range(1, n):
            start = max(0, i - local_window)
            local_vals = vals[start:i + 1]
            local_var = np.var(local_vals) if len(local_vals) > 1 else noise_var
            signal_var = max(local_var - noise_var, 0.001)
            gain = signal_var / (signal_var + noise_var)
            predicted = filtered[i - 1]
            filtered[i] = predicted + gain * (vals[i] - predicted)
        smoothed[name] = np.clip(filtered, -1, 1)
    return smoothed, None


def train_drl_stop_loss(close_prices, osc_values, high_prices, low_prices, n_episodes=200):
    """#11: Simple Q-learning dynamic stop-loss (no stable-baselines3 needed)."""
    n = len(close_prices)
    # State bins: (osc_quartile, vol_quartile, pnl_sign) -> 4*4*3 = 48 states
    # Actions: stop_loss % in [0.5, 1.0, 1.5, 2.0, 3.0] -> 5 actions
    returns = np.zeros(n)
    returns[1:] = np.diff(close_prices) / close_prices[:-1]
    vol_20 = pd.Series(np.abs(returns)).rolling(20, min_periods=1).mean().values
    osc_quartiles = np.digitize(osc_values, np.percentile(osc_values[osc_values != 0], [25, 50, 75]) if np.any(osc_values != 0) else [-.25, 0, .25])
    vol_quartiles = np.digitize(vol_20, np.percentile(vol_20[vol_20 > 0], [25, 50, 75]) if np.any(vol_20 > 0) else [0.001, 0.005, 0.01])
    stop_levels = [0.5, 1.0, 1.5, 2.0, 3.0]
    n_states = 48
    n_actions = len(stop_levels)
    Q = np.zeros((n_states, n_actions))
    lr = 0.1
    gamma = 0.95
    epsilon = 0.3

    for ep in range(n_episodes):
        for i in range(50, n - 20):
            oq = min(osc_quartiles[i], 3)
            vq = min(vol_quartiles[i], 3)
            pnl_sign = 1 if returns[i] > 0 else (2 if returns[i] < 0 else 0)
            state = oq * 12 + vq * 3 + pnl_sign
            state = min(state, n_states - 1)
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(Q[state])
            sl_pct = stop_levels[action]
            # Simulate: check if stop was hit in next 10 bars
            entry_price = close_prices[i]
            reward = 0
            for j in range(1, min(11, n - i)):
                ret = (close_prices[i + j] - entry_price) / entry_price * 100
                if ret < -sl_pct:
                    reward = -sl_pct
                    break
                if ret > sl_pct * 1.5:
                    reward = ret
                    break
            if reward == 0:
                reward = (close_prices[min(i + 10, n - 1)] - entry_price) / entry_price * 100
            # Update Q
            next_oq = min(osc_quartiles[min(i + 1, n - 1)], 3)
            next_vq = min(vol_quartiles[min(i + 1, n - 1)], 3)
            next_state = min(next_oq * 12 + next_vq * 3 + pnl_sign, n_states - 1)
            Q[state, action] += lr * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        epsilon *= 0.97  # Decay exploration

    # Extract optimal stop-loss policy
    optimal_stops = np.zeros(n)
    for i in range(n):
        oq = min(osc_quartiles[i], 3)
        vq = min(vol_quartiles[i], 3)
        pnl_sign = 1 if returns[i] > 0 else (2 if returns[i] < 0 else 0)
        state = min(oq * 12 + vq * 3 + pnl_sign, n_states - 1)
        optimal_stops[i] = stop_levels[np.argmax(Q[state])]
    return optimal_stops, Q


def apply_smote_to_xgb_mask(close_prices, osc_values, all_oscillators, volume, high_prices, low_prices,
                             forward_bars=10, min_win_pct=0.1, conf_threshold=0.5):
    """#12: XGBoost + SMOTE — train with balanced classes."""
    if not XGBOOST_AVAILABLE or not IMBLEARN_AVAILABLE:
        return None, None, "xgboost or imblearn not installed"
    n = len(close_prices)
    returns_1 = np.zeros(n)
    returns_1[1:] = np.diff(close_prices) / close_prices[:-1]
    vol_20 = pd.Series(returns_1).rolling(20, min_periods=1).std().values
    atr = np.zeros(n)
    if high_prices is not None and low_prices is not None:
        tr = np.maximum(high_prices - low_prices,
                        np.maximum(np.abs(high_prices - np.roll(close_prices, 1)),
                                   np.abs(low_prices - np.roll(close_prices, 1))))
        tr[0] = high_prices[0] - low_prices[0]
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    vol_ratio = np.ones(n)
    if volume is not None:
        vol_sma = pd.Series(volume.astype(float)).rolling(20, min_periods=1).mean().values
        vol_ratio = volume / (vol_sma + 1e-10)
    price_series = pd.Series(close_prices)
    rolling_min = price_series.rolling(20, min_periods=1).min().values
    rolling_max = price_series.rolling(20, min_periods=1).max().values
    price_range = rolling_max - rolling_min
    range_pos = np.where(price_range > 0, (close_prices - rolling_min) / price_range, 0.5)

    features, labels = [], []
    for i in range(50, n - forward_bars):
        vel = osc_values[i] - osc_values[i - 1] if i > 0 else 0
        acc = vel - (osc_values[i - 1] - osc_values[i - 2]) if i > 1 else 0
        features.append([osc_values[i], vel, acc, vol_20[i], atr[i], vol_ratio[i], range_pos[i], returns_1[i]])
        fwd_ret = (close_prices[i + forward_bars] - close_prices[i]) / close_prices[i] * 100
        labels.append(1 if fwd_ret > min_win_pct else 0)

    X, y = np.array(features), np.array(labels)
    if len(y) < 100 or y.sum() < 10:
        return None, None, "insufficient data"

    # Apply SMOTE
    try:
        smote = BorderlineSMOTE(random_state=42, k_neighbors=min(5, y.sum() - 1))
        X_resampled, y_resampled = smote.fit_resample(X, y)
    except Exception:
        X_resampled, y_resampled = X, y

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          use_label_encoder=False, eval_metric='logloss', verbosity=0)
    model.fit(X_resampled, y_resampled)
    train_acc = (model.predict(X) == y).mean()

    confidence = np.zeros(n)
    for i in range(50, n):
        vel = osc_values[i] - osc_values[i - 1] if i > 0 else 0
        acc = vel - (osc_values[i - 1] - osc_values[i - 2]) if i > 1 else 0
        feat = np.array([[osc_values[i], vel, acc, vol_20[i], atr[i], vol_ratio[i], range_pos[i], returns_1[i]]])
        confidence[i] = model.predict_proba(feat)[0][1]

    high_conf_mask = confidence >= conf_threshold
    pct = high_conf_mask.sum() / len(high_conf_mask) * 100
    return high_conf_mask, model, f"SMOTE train acc: {train_acc:.1%}, high-conf: {pct:.0f}%"


def meta_learn_adaptation(close_prices, osc_values, context_window=50, n_experts=5):
    """#13: Simplified meta-learning — pool of expert smoothing configs, adapt weights by recent perf."""
    n = len(close_prices)
    returns = np.zeros(n)
    returns[1:] = np.diff(close_prices) / close_prices[:-1]

    # Expert configs: different smoothing windows
    expert_windows = [3, 5, 10, 15, 20]
    expert_oscs = []
    for w in expert_windows:
        smoothed = pd.Series(osc_values).rolling(w, min_periods=1).mean().values
        expert_oscs.append(smoothed)

    # Adaptive weights using exponential performance tracking
    weights = np.ones(n_experts) / n_experts
    adapted_osc = np.zeros(n)
    eta = 0.1  # Learning rate

    for i in range(n):
        # Weighted combination
        expert_vals = [expert_oscs[j][i] for j in range(n_experts)]
        adapted_osc[i] = np.dot(weights, expert_vals)

        # Update weights based on how well each expert predicted direction
        if i >= context_window:
            for j in range(n_experts):
                # Reward: did expert's signal direction match actual return?
                recent_start = max(0, i - context_window)
                expert_signals = np.sign(np.diff(expert_oscs[j][recent_start:i + 1]))
                actual_returns = returns[recent_start + 1:i + 1]
                if len(expert_signals) > 0 and len(actual_returns) > 0:
                    min_len = min(len(expert_signals), len(actual_returns))
                    agreement = np.mean(np.sign(expert_signals[:min_len]) == np.sign(actual_returns[:min_len]))
                    weights[j] *= np.exp(eta * (agreement - 0.5))
            weights /= weights.sum()  # Normalize

    return np.clip(adapted_osc, -1, 1), weights


def build_transformer_mask(close_prices, osc_values, volume, high_prices, low_prices,
                            seq_len=20, forward_bars=10, min_win_pct=0.1, conf_threshold=0.5):
    """#14: Sequence-aware signal scoring using GradientBoosting on flattened sequences.
    Uses sklearn instead of PyTorch to avoid deadlocks with joblib/loky workers."""
    from sklearn.ensemble import GradientBoostingClassifier
    n = len(close_prices)
    returns_1 = np.zeros(n)
    returns_1[1:] = np.diff(close_prices) / close_prices[:-1]
    vol_20 = pd.Series(returns_1).rolling(20, min_periods=1).std().values
    velocity = np.zeros(n)
    velocity[1:] = np.diff(osc_values)
    accel = np.zeros(n)
    accel[1:] = np.diff(velocity)

    features_all = np.column_stack([osc_values, velocity, accel, returns_1, vol_20])
    features_all = np.nan_to_num(features_all, nan=0.0)

    # Build flattened sequence features + temporal statistics
    X_seqs, y_labels = [], []
    for i in range(seq_len + 50, n - forward_bars):
        seq = features_all[i - seq_len:i]
        # Temporal features: mean, std, last, trend slope for each feature
        feat = np.concatenate([seq.mean(axis=0), seq.std(axis=0), seq[-1],
                               np.polyfit(range(seq_len), seq[:, 0], 1)])
        X_seqs.append(feat)
        fwd_ret = (close_prices[i + forward_bars] - close_prices[i]) / close_prices[i] * 100
        y_labels.append(1 if fwd_ret > min_win_pct else 0)

    if len(X_seqs) < 50:
        return None, "insufficient sequences"

    X, y = np.array(X_seqs), np.array(y_labels)
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                        subsample=0.8, random_state=42)
    model.fit(X, y)

    confidence = np.zeros(n)
    for i in range(seq_len + 50, n):
        seq = features_all[i - seq_len:i]
        feat = np.concatenate([seq.mean(axis=0), seq.std(axis=0), seq[-1],
                               np.polyfit(range(seq_len), seq[:, 0], 1)])
        confidence[i] = model.predict_proba(feat.reshape(1, -1))[0][1]

    high_conf_mask = confidence >= conf_threshold
    pct = high_conf_mask.sum() / max(1, (np.arange(n) >= seq_len + 50).sum()) * 100
    return high_conf_mask, f"Sequence GBM: {pct:.0f}% high-conf"


def compute_cross_asset_correlation_filter(primary_returns, cross_asset_data, window=50, corr_threshold=0.7):
    """#15: Simplified GNN — correlation-based cross-asset filter."""
    n = len(primary_returns)
    filter_mask = np.ones(n, dtype=bool)
    for ticker, df in cross_asset_data.items():
        if 'close' not in df.columns or len(df) < 50:
            continue
        asset_returns = np.zeros(len(df))
        asset_close = df['close'].values
        asset_returns[1:] = np.diff(asset_close) / asset_close[:-1]
        # Align lengths
        min_len = min(len(primary_returns), len(asset_returns))
        prim = primary_returns[-min_len:]
        asset = asset_returns[-min_len:]
        # Rolling correlation
        for i in range(window, min_len):
            corr = np.corrcoef(prim[i - window:i], asset[i - window:i])[0, 1]
            if abs(corr) > corr_threshold:
                # High correlation = external driving force, suppress mean reversion
                idx = n - min_len + i
                if 0 <= idx < n:
                    filter_mask[idx] = False
    pct = filter_mask.sum() / n * 100
    return filter_mask, f"Cross-asset filter: {pct:.0f}% tradeable"


def compute_copula_tail_filter(primary_returns, vix_data, window=50, tail_threshold=0.4):
    """#16: Simplified copula tail-risk filter using rank correlation."""
    n = len(primary_returns)
    filter_mask = np.ones(n, dtype=bool)
    if vix_data is None or 'close' not in vix_data.columns:
        return filter_mask, "No VIX data"
    vix_returns = np.zeros(len(vix_data))
    vix_close = vix_data['close'].values
    vix_returns[1:] = np.diff(vix_close) / vix_close[:-1]
    min_len = min(n, len(vix_returns))
    prim = primary_returns[-min_len:]
    vix = vix_returns[-min_len:]
    for i in range(window, min_len):
        p_window = prim[i - window:i]
        v_window = vix[i - window:i]
        # Lower-tail dependence: fraction of joint extreme negatives
        p_thresh = np.percentile(p_window, 10)
        v_thresh = np.percentile(v_window, 90)  # VIX spikes when market drops
        joint_stress = np.mean((p_window < p_thresh) & (v_window > v_thresh))
        if joint_stress > tail_threshold:
            idx = n - min_len + i
            if 0 <= idx < n:
                filter_mask[idx] = False
    pct = filter_mask.sum() / n * 100
    return filter_mask, f"Copula filter: {pct:.0f}% tradeable"


def train_vae_anomaly_mask(close_prices, osc_values, volume, seq_len=20, anomaly_pctile=95):
    """#17: Anomaly detection using Isolation Forest (sklearn) — no PyTorch needed.
    Flags regime shifts / unusual market conditions to suppress trading."""
    from sklearn.ensemble import IsolationForest
    n = len(close_prices)
    returns = np.zeros(n)
    returns[1:] = np.diff(close_prices) / close_prices[:-1]
    velocity = np.zeros(n)
    velocity[1:] = np.diff(osc_values)
    vol_ratio = np.ones(n)
    if volume is not None:
        vol_sma = pd.Series(volume.astype(float)).rolling(20, min_periods=1).mean().values
        vol_ratio = volume / (vol_sma + 1e-10)

    features_all = np.column_stack([osc_values, velocity, returns, vol_ratio])
    features_all = np.nan_to_num(features_all, nan=0.0)

    # Build rolling window features
    X_windows = []
    for i in range(seq_len + 10, n):
        window = features_all[i - seq_len:i]
        feat = np.concatenate([window.mean(axis=0), window.std(axis=0),
                               window[-1], window.max(axis=0) - window.min(axis=0)])
        X_windows.append(feat)

    if len(X_windows) < 30:
        return None, "insufficient data"

    X = np.array(X_windows)

    # Isolation Forest: -1 = anomaly, 1 = normal
    contamination = (100 - anomaly_pctile) / 100.0
    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    predictions = iso.fit_predict(X)

    # Map back to full array
    anomaly_mask = np.ones(n, dtype=bool)  # True = normal, tradeable
    for i, pred in enumerate(predictions):
        bar_idx = i + seq_len + 10
        if bar_idx < n and pred == -1:
            anomaly_mask[bar_idx] = False

    pct_normal = anomaly_mask.sum() / n * 100
    return anomaly_mask, f"IsoForest: {pct_normal:.0f}% normal bars"


def online_adaptive_weights(close_prices, osc_values, n_experts=8, eta=0.05):
    """#19: Online learning with multiplicative weights."""
    n = len(close_prices)
    returns = np.zeros(n)
    returns[1:] = np.diff(close_prices) / close_prices[:-1]

    # Experts: different oversold thresholds
    thresholds = np.linspace(-0.8, -0.2, n_experts)
    weights = np.ones(n_experts) / n_experts

    # Track expert signals and compute adapted oscillator
    adapted_osc = np.zeros(n)
    weight_history = np.zeros((n, n_experts))

    for i in range(1, n):
        # Each expert generates signal: 1 if osc < threshold (oversold), -1 if osc > -threshold
        expert_signals = np.zeros(n_experts)
        for j in range(n_experts):
            if osc_values[i] < thresholds[j]:
                expert_signals[j] = 1.0  # Buy signal
            elif osc_values[i] > -thresholds[j]:
                expert_signals[j] = -1.0  # No signal
            else:
                expert_signals[j] = 0.0

        # Weighted consensus
        adapted_osc[i] = np.dot(weights, expert_signals)

        # Update weights based on actual return
        for j in range(n_experts):
            reward = expert_signals[j] * returns[i]
            weights[j] *= np.exp(eta * reward)
        weights /= weights.sum()
        weight_history[i] = weights

    return np.clip(adapted_osc, -1, 1), weight_history


def apply_diffusion_denoise(all_oscillators, n_steps=50, noise_scale=0.3):
    """#20: Simplified denoising diffusion — iterative noise removal."""
    denoised = {}
    for name, vals in all_oscillators.items():
        n = len(vals)
        # Forward: add noise progressively
        # Reverse: denoise using local statistics
        signal = vals.copy()
        # Multi-step denoising: repeatedly estimate and subtract noise
        for step in range(3):  # 3 passes
            # Estimate noise as deviation from local trend
            window = max(5, 10 - step * 2)
            local_mean = pd.Series(signal).rolling(window, min_periods=1, center=False).mean().values
            noise_estimate = signal - local_mean
            # Attenuate noise with adaptive scaling
            noise_var = pd.Series(noise_estimate ** 2).rolling(20, min_periods=1).mean().values
            signal_var = pd.Series(local_mean ** 2).rolling(20, min_periods=1).mean().values
            # Wiener-inspired weighting
            weight = signal_var / (signal_var + noise_var + 1e-10)
            signal = local_mean + weight * noise_estimate
        denoised[name] = np.clip(signal, -1, 1)
    return denoised, None


# =============================================================================
# Cumulative Harness
# =============================================================================

STRATEGY_NAMES = {
    1: "Kalman Filter Smoothing",
    2: "Wavelet Denoising",
    3: "HMM Regime Detection",
    4: "OU Optimal Entry/Exit",
    5: "GT-Score Objective",
    6: "Instance Selection",
    7: "XGBoost Signal Classifier",
    8: "Transfer Entropy Causal",
    9: "Adaptive Stochastic Vol",
    10: "NoLAW Wiener Filter",
    11: "DRL Dynamic Stop-Loss",
    12: "SMOTE Signal Augmentation",
    13: "Meta-Learning Adaptation",
    14: "Transformer Signal Scoring",
    15: "Cross-Asset Correlation",
    16: "Copula Tail Risk Filter",
    17: "VAE Anomaly Detection",
    18: "BO Early Stopping",
    19: "Online Adaptive Regret",
    20: "Diffusion SNR Enhancement",
}


class CumulativeTestHarness:
    """Tests each strategy ADDITIVELY — each builds on top of all previous."""

    def __init__(self, ticker='ES=F', interval='15m', n_trials=10000, n_workers=16,
                 train_ratio=0.8, optimize_metric='risk_adjusted', days=59):
        self.ticker = ticker
        self.interval = interval
        self.n_trials = n_trials
        self.n_workers = n_workers
        self.train_ratio = train_ratio
        self.optimize_metric = optimize_metric
        self.days = days
        self.results = OrderedDict()
        self.baseline = None

        # Data
        self.df = None
        self.train_df = None
        self.test_df = None

        # Running state: modified cumulatively
        self.current_train_osc = None
        self.current_test_osc = None
        self.train_mask = None  # Boolean mask: which bars are tradeable
        self.test_mask = None
        self.use_gt_score = False
        self.use_drawdown_penalty = False
        self.cross_asset_data = None

    def prepare(self):
        print("=" * 80)
        print("CUMULATIVE STRATEGY TEST HARNESS")
        print("=" * 80)
        print(f"Ticker: {self.ticker}, Interval: {self.interval}")
        print(f"Trials: {self.n_trials:,}, Workers: {self.n_workers}")
        print(f"Each strategy is ADDITIVE (builds on all previous)\n")

        print("Fetching data...")
        self.df = fetch_data(self.ticker, self.interval, self.days)
        print(f"  {len(self.df)} bars total")

        split_idx = int(len(self.df) * self.train_ratio)
        self.train_df = self.df.iloc[:split_idx].copy()
        self.test_df = self.df.iloc[split_idx:].copy()
        print(f"  Train: {len(self.train_df)} bars ({self.train_df.index[0].strftime('%Y-%m-%d')} to {self.train_df.index[-1].strftime('%Y-%m-%d')})")
        print(f"  Test:  {len(self.test_df)} bars ({self.test_df.index[0].strftime('%Y-%m-%d')} to {self.test_df.index[-1].strftime('%Y-%m-%d')})")

        bh_train = (self.train_df['close'].iloc[-1] / self.train_df['close'].iloc[0] - 1) * 100
        bh_test = (self.test_df['close'].iloc[-1] / self.test_df['close'].iloc[0] - 1) * 100
        print(f"  Buy & Hold — Train: {bh_train:.2f}%, Test: {bh_test:.2f}%")

        print("\nComputing standard oscillators...")
        self.current_train_osc = compute_oscillators_standard(self.train_df)
        self.current_test_osc = compute_oscillators_standard(self.test_df)
        self.raw_train_osc = deepcopy(self.current_train_osc)  # Keep originals for some strategies
        self.raw_test_osc = deepcopy(self.current_test_osc)
        print(f"  {len(self.current_train_osc)} oscillator types: {list(self.current_train_osc.keys())}")

        # Initialize masks to all-tradeable
        self.train_mask = np.ones(len(self.train_df), dtype=bool)
        self.test_mask = np.ones(len(self.test_df), dtype=bool)
        self.min_mask_pct = 0.10  # Never let mask drop below 10% of bars

        # Fetch cross-asset data for strategies that need it
        print("Fetching cross-asset data (VIX, NQ, DX)...")
        self.cross_asset_data = fetch_cross_asset_data(self.interval, self.days)
        print(f"  Available: {list(self.cross_asset_data.keys())}")
        print()

    def reset_state(self):
        """Reset all cumulative state to fresh baseline (oscillators, masks, flags).
        Call this before running a new ordering/combination of strategies."""
        self.current_train_osc = deepcopy(self.raw_train_osc)
        self.current_test_osc = deepcopy(self.raw_test_osc)
        self.train_mask = np.ones(len(self.train_df), dtype=bool)
        self.test_mask = np.ones(len(self.test_df), dtype=bool)
        self.use_gt_score = False
        self.use_drawdown_penalty = False
        self.results = OrderedDict()

    def _safe_apply_mask(self, current_mask, new_mask, label='train'):
        """AND a new mask with current, but reject if result drops below minimum."""
        candidate = current_mask & new_mask
        pct = candidate.sum() / len(candidate)
        if pct < self.min_mask_pct:
            print(f"  WARNING: {label} mask would drop to {pct:.0%} — keeping previous mask ({current_mask.sum()/len(current_mask):.0%})")
            return current_mask
        return candidate

    def _extract_data(self, which='train'):
        df = self.train_df if which == 'train' else self.test_df
        close = df['close'].values
        high = df['high'].values if 'high' in df.columns else None
        low = df['low'].values if 'low' in df.columns else None
        vol = df['volume'].values if 'volume' in df.columns else None
        return close, high, low, vol

    def _get_masked_oscillators(self, osc_dict, mask):
        """Apply bar mask to oscillators — zero out non-tradeable bars."""
        if mask is None or mask.all():
            return osc_dict
        filtered = {}
        for name, vals in osc_dict.items():
            modified = vals.copy()
            # Only zero out where mask length matches
            mask_len = min(len(mask), len(modified))
            modified[:mask_len][~mask[:mask_len]] = 0.0
            filtered[name] = modified
        return filtered

    def _run_and_evaluate(self, label, top_n=5):
        """Run optimization with current cumulative state."""
        close_train, high_train, low_train, vol_train = self._extract_data('train')

        # Apply current mask to current oscillators
        train_osc = self._get_masked_oscillators(self.current_train_osc, self.train_mask)
        test_osc = self._get_masked_oscillators(self.current_test_osc, self.test_mask)

        default_osc = train_osc.get('composite_smooth', close_train * 0)

        results = run_optimization(
            close_train, default_osc, train_osc,
            n_trials=self.n_trials, n_workers=self.n_workers,
            optimize_metric=self.optimize_metric, label=label,
            high_prices=high_train, low_prices=low_train, volume=vol_train,
            use_drawdown_penalty=self.use_drawdown_penalty,
            max_drawdown_threshold=5.0, drawdown_penalty_weight=0.5,
        )

        if not results:
            return {'error': 'No valid results'}

        # Optionally re-rank by GT-Score
        if self.use_gt_score:
            for r in results:
                r['gt_score'] = compute_gt_score(r)
            top = select_best(results, top_n=top_n, metric='gt_score')
        else:
            top = select_best(results, top_n=top_n, metric=self.optimize_metric)

        if not top:
            return {'error': 'No valid top results'}

        best = top[0]
        test_result = evaluate_on_test(best, self.test_df, test_osc)

        summary = {
            'train_return': best.get('total_return', 0),
            'train_wr': best.get('win_rate', 0),
            'train_trades': best.get('num_trades', 0),
            'train_pf': best.get('profit_factor', 0),
            'train_max_dd': best.get('max_drawdown', 0),
        }
        if test_result:
            summary.update({
                'test_return': test_result.get('total_return', 0),
                'test_wr': test_result.get('win_rate', 0),
                'test_trades': test_result.get('num_trades', 0),
                'test_pf': test_result.get('profit_factor', 0),
                'test_max_dd': test_result.get('max_drawdown', 0),
            })
        else:
            summary.update({'test_return': 0, 'test_wr': 0, 'test_trades': 0, 'test_pf': 0, 'test_max_dd': 0})

        top5_test = []
        for params in top[:5]:
            tr = evaluate_on_test(params, self.test_df, test_osc)
            if tr:
                top5_test.append(tr.get('total_return', 0))
        summary['top5_avg_test'] = np.mean(top5_test) if top5_test else 0
        summary['best_params'] = best
        return summary

    def run_baseline(self):
        print("=" * 80)
        print("STEP 0: BASELINE (standard, no enhancements)")
        print("=" * 80)
        self.baseline = self._run_and_evaluate("BASELINE")
        self._print_result("BASELINE", self.baseline, 0)
        self.results['baseline'] = self.baseline
        return self.baseline

    def run_strategy(self, strategy_id):
        name = STRATEGY_NAMES.get(strategy_id, f"#{strategy_id}")
        print(f"\n{'=' * 80}")
        print(f"STEP {strategy_id}: + {name} (cumulative)")
        print("=" * 80)

        close_train, high_train, low_train, vol_train = self._extract_data('train')
        close_test, high_test, low_test, vol_test = self._extract_data('test')

        applied = False
        info = ""

        if strategy_id == 1:
            # Kalman filter — modify oscillators
            result, err = apply_kalman(self.current_train_osc)
            if err:
                info = f"SKIPPED: {err}"
            else:
                self.current_train_osc = result
                result_test, _ = apply_kalman(self.current_test_osc)
                self.current_test_osc = result_test
                applied = True
                info = f"Applied Kalman to {len(result)} oscillators"

        elif strategy_id == 2:
            result, err = apply_wavelet(self.current_train_osc)
            if err:
                info = f"SKIPPED: {err}"
            else:
                self.current_train_osc = result
                result_test, _ = apply_wavelet(self.current_test_osc)
                self.current_test_osc = result_test
                applied = True
                info = f"Applied wavelet denoising to {len(result)} oscillators"

        elif strategy_id == 3:
            mask, msg = compute_hmm_mask(close_train)
            if mask is None:
                info = f"SKIPPED: {msg}"
            else:
                self.train_mask = self._safe_apply_mask(self.train_mask, mask, 'train')
                # For test, fit HMM on test data (no leakage — HMM is unsupervised)
                test_mask, _ = compute_hmm_mask(close_test)
                if test_mask is not None:
                    self.test_mask = self._safe_apply_mask(self.test_mask, test_mask, 'test')
                applied = True
                info = f"HMM: {msg}, cumulative mask: {self.train_mask.sum()}/{len(self.train_mask)} bars"

        elif strategy_id == 4:
            osc_vals = self.current_train_osc.get('composite_smooth', close_train * 0)
            entry, exit_lvl, err = compute_ou_thresholds(osc_vals)
            if err:
                info = f"SKIPPED: {err}"
            else:
                # Tighten oscillator values: amplify signals near OU entry/exit
                for name, vals in self.current_train_osc.items():
                    near_entry = np.abs(vals - entry) < 0.1
                    near_exit = np.abs(vals - exit_lvl) < 0.1
                    boost = np.ones(len(vals))
                    boost[near_entry] = 1.3  # Amplify entry zones
                    boost[near_exit] = 1.3
                    self.current_train_osc[name] = np.clip(vals * boost, -1, 1)
                for name, vals in self.current_test_osc.items():
                    near_entry = np.abs(vals - entry) < 0.1
                    near_exit = np.abs(vals - exit_lvl) < 0.1
                    boost = np.ones(len(vals))
                    boost[near_entry] = 1.3
                    boost[near_exit] = 1.3
                    self.current_test_osc[name] = np.clip(vals * boost, -1, 1)
                applied = True
                info = f"OU thresholds: entry={entry:.4f}, exit={exit_lvl:.4f}"

        elif strategy_id == 5:
            self.use_gt_score = True
            applied = True
            info = "Enabled GT-Score re-ranking"

        elif strategy_id == 6:
            filtered, msg = apply_instance_selection(close_train, self.current_train_osc)
            self.current_train_osc = filtered
            applied = True
            info = f"Instance selection: {msg}"

        elif strategy_id == 7:
            osc_vals = self.current_train_osc.get('mji',
                       self.current_train_osc.get('composite_smooth', close_train * 0))
            mask, model, msg = build_xgboost_mask(
                close_train, osc_vals, self.current_train_osc, vol_train, high_train, low_train
            )
            if mask is None:
                info = f"SKIPPED: {msg}"
            else:
                self.train_mask = self._safe_apply_mask(self.train_mask, mask, 'train')
                # For test: build features and score
                osc_test = self.current_test_osc.get('mji',
                           self.current_test_osc.get('composite_smooth', close_test * 0))
                test_mask, _, _ = build_xgboost_mask(
                    close_test, osc_test, self.current_test_osc, vol_test, high_test, low_test
                )
                if test_mask is not None:
                    self.test_mask = self._safe_apply_mask(self.test_mask, test_mask, 'test')
                applied = True
                info = f"XGBoost: {msg}"

        elif strategy_id == 8:
            train_returns = np.zeros(len(close_train))
            train_returns[1:] = np.diff(close_train) / close_train[:-1]
            te_total = 0.0
            te_count = 0
            suppressed = 0
            for ticker, asset_df in self.cross_asset_data.items():
                if 'close' not in asset_df.columns or len(asset_df) < 50:
                    continue
                asset_close = asset_df['close'].values
                asset_returns = np.zeros(len(asset_close))
                asset_returns[1:] = np.diff(asset_close) / asset_close[:-1]
                min_len = min(len(train_returns), len(asset_returns))
                te = compute_transfer_entropy(asset_returns[-min_len:], train_returns[-min_len:])
                te_total += te
                te_count += 1
                # High TE = external force driving our asset, suppress mean reversion
                if te > 0.05:
                    # Rolling TE to find high-TE windows (sampled every 10 bars for speed)
                    window = 50
                    for i in range(window, min_len, 10):
                        local_te = compute_transfer_entropy(
                            asset_returns[i - window:i], train_returns[-min_len + i - window:-min_len + i]
                        )
                        if local_te > 0.08:
                            # Suppress the 10-bar window
                            for j in range(10):
                                idx = len(close_train) - min_len + i + j
                                if 0 <= idx < len(close_train):
                                    self.train_mask[idx] = False
                                    suppressed += 1
            applied = True
            avg_te = te_total / max(te_count, 1)
            info = f"Transfer entropy avg={avg_te:.4f}, suppressed {suppressed} bars"

        elif strategy_id == 9:
            asv_train = compute_adaptive_stoch_vol(close_train)
            asv_test = compute_adaptive_stoch_vol(close_test)
            # Scale oscillators by inverse volatility — reduce signals in high-vol
            for name, vals in self.current_train_osc.items():
                vol_scale = 1.0 / (1.0 + asv_train * 10)
                self.current_train_osc[name] = np.clip(vals * vol_scale, -1, 1)
            for name, vals in self.current_test_osc.items():
                vol_scale = 1.0 / (1.0 + asv_test * 10)
                self.current_test_osc[name] = np.clip(vals * vol_scale, -1, 1)
            applied = True
            info = f"ASV: mean vol={np.mean(asv_train):.6f}"

        elif strategy_id == 10:
            result, err = apply_nolaw_filter(self.current_train_osc)
            if err:
                info = f"SKIPPED: {err}"
            else:
                self.current_train_osc = result
                result_test, _ = apply_nolaw_filter(self.current_test_osc)
                self.current_test_osc = result_test
                applied = True
                info = f"Applied NoLAW filter to {len(result)} oscillators"

        elif strategy_id == 11:
            osc_vals = self.current_train_osc.get('composite_smooth', close_train * 0)
            stops, Q = train_drl_stop_loss(close_train, osc_vals, high_train, low_train)
            # Use learned stop-loss distribution to suppress bars where optimal stop is very tight (high risk)
            tight_stop_mask = stops > 0.75  # Only trade when DRL says stop can be > 0.75%
            self.train_mask = self._safe_apply_mask(self.train_mask, tight_stop_mask, 'train')
            # For test
            osc_test = self.current_test_osc.get('composite_smooth', close_test * 0)
            stops_test, _ = train_drl_stop_loss(close_test, osc_test, high_test, low_test, n_episodes=100)
            test_tight = stops_test > 0.75
            self.test_mask = self._safe_apply_mask(self.test_mask, test_tight, 'test')
            applied = True
            info = f"DRL stops: mean={np.mean(stops):.2f}%, mask kept {tight_stop_mask.sum()}/{len(tight_stop_mask)} bars"

        elif strategy_id == 12:
            osc_vals = self.current_train_osc.get('mji',
                       self.current_train_osc.get('composite_smooth', close_train * 0))
            mask, model, msg = apply_smote_to_xgb_mask(
                close_train, osc_vals, self.current_train_osc, vol_train, high_train, low_train
            )
            if mask is None:
                info = f"SKIPPED: {msg}"
            else:
                self.train_mask = self._safe_apply_mask(self.train_mask, mask, 'train')
                osc_test = self.current_test_osc.get('mji',
                           self.current_test_osc.get('composite_smooth', close_test * 0))
                test_mask, _, _ = apply_smote_to_xgb_mask(
                    close_test, osc_test, self.current_test_osc, vol_test, high_test, low_test
                )
                if test_mask is not None:
                    self.test_mask = self._safe_apply_mask(self.test_mask, test_mask, 'test')
                applied = True
                info = f"SMOTE+XGB: {msg}"

        elif strategy_id == 13:
            for name, vals in self.current_train_osc.items():
                adapted, _ = meta_learn_adaptation(close_train, vals)
                self.current_train_osc[name] = adapted
            for name, vals in self.current_test_osc.items():
                adapted, _ = meta_learn_adaptation(close_test, vals)
                self.current_test_osc[name] = adapted
            applied = True
            info = "Applied meta-learned adaptive smoothing"

        elif strategy_id == 14:
            osc_vals = self.current_train_osc.get('composite_smooth', close_train * 0)
            try:
                mask, msg = build_transformer_mask(close_train, osc_vals, vol_train, high_train, low_train)
                if mask is not None:
                    self.train_mask = self._safe_apply_mask(self.train_mask, mask, 'train')
                    # Also build on test
                    osc_test = self.current_test_osc.get('composite_smooth', close_test * 0)
                    test_mask, _ = build_transformer_mask(close_test, osc_test, vol_test, high_test, low_test)
                    if test_mask is not None:
                        self.test_mask = self._safe_apply_mask(self.test_mask, test_mask, 'test')
                    applied = True
                    pct = mask.sum() / len(mask) * 100
                    info = f"Transformer: {pct:.0f}% high-confidence bars"
                else:
                    info = f"SKIPPED: {msg}"
            except Exception as e:
                info = f"SKIPPED: Transformer failed: {e}"

        elif strategy_id == 15:
            train_returns = np.zeros(len(close_train))
            train_returns[1:] = np.diff(close_train) / close_train[:-1]
            mask, msg = compute_cross_asset_correlation_filter(train_returns, self.cross_asset_data)
            self.train_mask = self._safe_apply_mask(self.train_mask, mask, 'train')
            # Test
            test_returns = np.zeros(len(close_test))
            test_returns[1:] = np.diff(close_test) / close_test[:-1]
            test_mask, _ = compute_cross_asset_correlation_filter(test_returns, self.cross_asset_data)
            self.test_mask = self._safe_apply_mask(self.test_mask, test_mask, 'test')
            applied = True
            info = msg

        elif strategy_id == 16:
            train_returns = np.zeros(len(close_train))
            train_returns[1:] = np.diff(close_train) / close_train[:-1]
            vix_data = self.cross_asset_data.get('^VIX', None)
            mask, msg = compute_copula_tail_filter(train_returns, vix_data)
            self.train_mask = self._safe_apply_mask(self.train_mask, mask, 'train')
            test_returns = np.zeros(len(close_test))
            test_returns[1:] = np.diff(close_test) / close_test[:-1]
            test_mask, _ = compute_copula_tail_filter(test_returns, vix_data)
            self.test_mask = self._safe_apply_mask(self.test_mask, test_mask, 'test')
            applied = True
            info = msg

        elif strategy_id == 17:
            osc_for_vae = self.current_train_osc.get('composite_smooth', close_train * 0)
            try:
                mask, msg = train_vae_anomaly_mask(close_train, osc_for_vae, vol_train)
                if mask is not None:
                    self.train_mask = self._safe_apply_mask(self.train_mask, mask, 'train')
                    # Also build on test
                    osc_test_vae = self.current_test_osc.get('composite_smooth', close_test * 0)
                    test_mask, _ = train_vae_anomaly_mask(close_test, osc_test_vae, vol_test)
                    if test_mask is not None:
                        self.test_mask = self._safe_apply_mask(self.test_mask, test_mask, 'test')
                    applied = True
                    info = msg
                else:
                    info = f"SKIPPED: {msg}"
            except Exception as e:
                info = f"SKIPPED: VAE failed: {e}"

        elif strategy_id == 18:
            self.use_drawdown_penalty = True
            applied = True
            info = "Enabled drawdown penalty (DD > 5% penalized)"

        elif strategy_id == 19:
            for name, vals in self.current_train_osc.items():
                adapted, _ = online_adaptive_weights(close_train, vals)
                self.current_train_osc[name] = adapted
            for name, vals in self.current_test_osc.items():
                adapted, _ = online_adaptive_weights(close_test, vals)
                self.current_test_osc[name] = adapted
            applied = True
            info = "Applied online adaptive expert weights"

        elif strategy_id == 20:
            result, err = apply_diffusion_denoise(self.current_train_osc)
            if err:
                info = f"SKIPPED: {err}"
            else:
                self.current_train_osc = result
                result_test, _ = apply_diffusion_denoise(self.current_test_osc)
                self.current_test_osc = result_test
                applied = True
                info = f"Applied diffusion denoising to {len(result)} oscillators"

        print(f"  {info}")
        train_tradeable = self.train_mask.sum() / len(self.train_mask) * 100
        test_tradeable = self.test_mask.sum() / len(self.test_mask) * 100
        print(f"  Cumulative state: train mask {train_tradeable:.0f}%, test mask {test_tradeable:.0f}%")

        if not applied:
            result = {'error': info, 'skipped': True}
        else:
            result = self._run_and_evaluate(f"STEP-{strategy_id}")

        self._print_result(f"#{strategy_id} {name}", result, strategy_id)
        self.results[f'strategy_{strategy_id:02d}'] = result
        return result

    def _print_result(self, label, result, step):
        if result.get('skipped') or result.get('error'):
            print(f"  >> {label}: SKIPPED — {result.get('error', 'unknown')}")
            return
        bl_test = self.baseline.get('test_return', 0) if self.baseline else 0
        delta = result.get('test_return', 0) - bl_test
        arrow = "+" if delta >= 0 else ""
        strats_so_far = f"(strategies 1-{step})" if step > 0 else ""
        print(f"\n  {label} {strats_so_far}:")
        print(f"    Train: Return={result.get('train_return', 0):.2f}%, "
              f"WR={result.get('train_wr', 0):.1f}%, "
              f"Trades={result.get('train_trades', 0)}, "
              f"PF={result.get('train_pf', 0):.2f}, "
              f"MaxDD={result.get('train_max_dd', 0):.2f}%")
        print(f"    Test:  Return={result.get('test_return', 0):.2f}%, "
              f"WR={result.get('test_wr', 0):.1f}%, "
              f"Trades={result.get('test_trades', 0)}, "
              f"PF={result.get('test_pf', 0):.2f}, "
              f"MaxDD={result.get('test_max_dd', 0):.2f}%")
        print(f"    Top5 Avg Test: {result.get('top5_avg_test', 0):.2f}%")
        print(f"    vs Baseline: {arrow}{delta:.2f}% test return")

    def print_summary(self):
        print("\n" + "=" * 100)
        print("CUMULATIVE RESULTS — Each Row Includes ALL Previous Strategies")
        print("=" * 100)
        baseline_test = self.baseline.get('test_return', 0) if self.baseline else 0

        header = f"{'Step':<5} {'Strategies Applied':<35} {'Train Ret':>10} {'Test Ret':>10} {'Delta':>8} {'Test WR':>8} {'Test PF':>8} {'Trades':>7} {'MaxDD':>7}"
        print(header)
        print("-" * len(header))

        if self.baseline and not self.baseline.get('skipped'):
            b = self.baseline
            print(f"{'0':<5} {'BASELINE (none)':<35} "
                  f"{b.get('train_return', 0):>9.2f}% "
                  f"{b.get('test_return', 0):>9.2f}% "
                  f"{'---':>8} "
                  f"{b.get('test_wr', 0):>7.1f}% "
                  f"{b.get('test_pf', 0):>8.2f} "
                  f"{b.get('test_trades', 0):>7} "
                  f"{b.get('test_max_dd', 0):>6.2f}%")

        applied_names = []
        for i in range(1, 21):
            key = f'strategy_{i:02d}'
            if key not in self.results:
                continue
            r = self.results[key]
            name = STRATEGY_NAMES.get(i, f"#{i}")

            if r.get('skipped') or r.get('error'):
                print(f"{i:<5} {'+ ' + name:<35} {'SKIPPED':>10} {r.get('error', '')[:40]}")
                continue

            applied_names.append(f"#{i}")
            strats_label = "+".join(applied_names[-3:])  # Show last 3
            if len(applied_names) > 3:
                strats_label = f"#{applied_names[0]}...{strats_label}"

            test_ret = r.get('test_return', 0)
            delta = test_ret - baseline_test
            arrow = "+" if delta >= 0 else ""

            print(f"{i:<5} {'+ ' + name:<35} "
                  f"{r.get('train_return', 0):>9.2f}% "
                  f"{test_ret:>9.2f}% "
                  f"{arrow}{delta:>6.2f}% "
                  f"{r.get('test_wr', 0):>7.1f}% "
                  f"{r.get('test_pf', 0):>8.2f} "
                  f"{r.get('test_trades', 0):>7} "
                  f"{r.get('test_max_dd', 0):>6.2f}%")

    def save_results(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'strategy_cumulative_results_{self.ticker}_{timestamp}.json'
        save_data = {
            'ticker': self.ticker, 'interval': self.interval,
            'n_trials': self.n_trials, 'n_workers': self.n_workers,
            'mode': 'cumulative_additive',
            'timestamp': datetime.now().isoformat(),
            'results': {},
        }
        for key, val in self.results.items():
            if isinstance(val, dict):
                clean = {k: v for k, v in val.items() if k != 'best_params'}
                if 'best_params' in val and isinstance(val['best_params'], dict):
                    clean['oscillator_type'] = val['best_params'].get('oscillator_type', '')
                    clean['signal_type'] = val['best_params'].get('signal_type', '')
                save_data['results'][key] = clean
            else:
                save_data['results'][key] = val
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nResults saved to: {filename}")
        return filename


def run_combination(harness, strategy_ids, label="", quiet=False):
    """Run a specific combination of strategies in order. Returns the final test result.
    Resets state first so each combination starts clean."""
    harness.reset_state()

    for sid in strategy_ids:
        harness.run_strategy(sid)

    # Get the last result
    last_key = f'strategy_{strategy_ids[-1]:02d}' if strategy_ids else 'baseline'
    result = harness.results.get(last_key, {})

    # If the last strategy was skipped, walk back to find the last valid result
    if result.get('skipped') or result.get('error'):
        for sid in reversed(strategy_ids[:-1]):
            key = f'strategy_{sid:02d}'
            r = harness.results.get(key, {})
            if not r.get('skipped') and not r.get('error'):
                result = r
                break

    return result


def smart_search(harness):
    """Smart combinatorial search:
    Phase 1: Test each strategy solo to rank individual contribution
    Phase 2: Greedy forward selection — build best combination incrementally
    Phase 3: Drop-one refinement from the greedy set
    """
    all_strategies = list(range(1, 21))
    baseline_test = harness.baseline.get('test_return', 0) if harness.baseline else 0

    # =========================================================================
    # PHASE 1: Solo Individual Contribution
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 1: INDIVIDUAL STRATEGY CONTRIBUTION (each tested alone)")
    print("=" * 100)

    solo_results = {}
    for sid in all_strategies:
        name = STRATEGY_NAMES.get(sid, f"#{sid}")
        print(f"\n--- Testing #{sid}: {name} (solo) ---")
        result = run_combination(harness, [sid])
        test_ret = result.get('test_return', 0)
        delta = test_ret - baseline_test
        solo_results[sid] = {
            'test_return': test_ret,
            'delta': delta,
            'test_wr': result.get('test_wr', 0),
            'test_trades': result.get('test_trades', 0),
            'test_pf': result.get('test_pf', 0),
            'test_max_dd': result.get('test_max_dd', 0),
            'skipped': result.get('skipped', False) or result.get('error', False),
        }
        status = "SKIPPED" if solo_results[sid]['skipped'] else f"delta={delta:+.2f}%"
        print(f"  >> #{sid} {name}: Test={test_ret:.2f}%, {status}")

    # Rank by delta
    ranked = sorted(
        [(sid, d) for sid, d in solo_results.items() if not d['skipped']],
        key=lambda x: x[1]['delta'], reverse=True
    )

    print("\n" + "-" * 80)
    print("PHASE 1 RANKINGS (Individual Contribution, sorted by test return delta):")
    print(f"{'Rank':<6} {'#':<4} {'Strategy':<35} {'Test Ret':>10} {'Delta':>10} {'WR':>8} {'Trades':>7} {'MaxDD':>7}")
    print("-" * 90)
    for rank, (sid, d) in enumerate(ranked, 1):
        name = STRATEGY_NAMES.get(sid, f"#{sid}")
        arrow = "+" if d['delta'] >= 0 else ""
        print(f"{rank:<6} {sid:<4} {name:<35} {d['test_return']:>9.2f}% {arrow}{d['delta']:>8.2f}% {d['test_wr']:>7.1f}% {d['test_trades']:>7} {d['test_max_dd']:>6.2f}%")

    # Filter to only strategies that help (positive delta solo)
    helpful = [sid for sid, d in ranked if d['delta'] > 0]
    neutral_or_harmful = [sid for sid, d in ranked if d['delta'] <= 0]
    print(f"\nHelpful strategies (positive solo delta): {helpful}")
    print(f"Neutral/harmful: {neutral_or_harmful}")

    # =========================================================================
    # PHASE 2: Greedy Forward Selection
    # =========================================================================
    print("\n" + "=" * 100)
    print("PHASE 2: GREEDY FORWARD SELECTION")
    print("=" * 100)
    print("Starting from best solo, greedily add the strategy that gives the most marginal gain.\n")

    # Candidates: all non-skipped strategies, prioritize helpful ones first
    candidates = [sid for sid, d in ranked if not d['skipped']]
    selected = []
    best_test_ret = baseline_test
    greedy_history = []

    while candidates:
        best_candidate = None
        best_candidate_ret = best_test_ret
        candidate_scores = []

        for sid in candidates:
            trial_set = selected + [sid]
            print(f"  Trying: {[STRATEGY_NAMES.get(s, f'#{s}') for s in trial_set]} ... ", end="")
            result = run_combination(harness, trial_set)
            test_ret = result.get('test_return', 0)
            marginal = test_ret - best_test_ret
            candidate_scores.append((sid, test_ret, marginal))
            print(f"test={test_ret:.2f}% (marginal={marginal:+.2f}%)")

            if test_ret > best_candidate_ret:
                best_candidate_ret = test_ret
                best_candidate = sid

        # Sort candidates by score for this round
        candidate_scores.sort(key=lambda x: x[2], reverse=True)
        print(f"\n  Round {len(selected)+1} candidate ranking:")
        for sid, ret, marg in candidate_scores[:5]:
            name = STRATEGY_NAMES.get(sid, f"#{sid}")
            print(f"    #{sid} {name}: test={ret:.2f}%, marginal={marg:+.2f}%")

        if best_candidate is None or best_candidate_ret <= best_test_ret:
            print(f"\n  >> STOPPING: No candidate improves over current best ({best_test_ret:.2f}%)")
            break

        selected.append(best_candidate)
        best_test_ret = best_candidate_ret
        candidates.remove(best_candidate)
        greedy_history.append({
            'step': len(selected),
            'added': best_candidate,
            'name': STRATEGY_NAMES.get(best_candidate, f"#{best_candidate}"),
            'selected': list(selected),
            'test_return': best_test_ret,
        })
        print(f"\n  >> SELECTED #{best_candidate} ({STRATEGY_NAMES.get(best_candidate, '')})")
        print(f"     Current set: {selected}, Test return: {best_test_ret:.2f}%\n")

    print("\n" + "-" * 80)
    print("GREEDY SELECTION HISTORY:")
    print(f"{'Step':<6} {'Added':<35} {'Set':>30} {'Test Ret':>10}")
    print("-" * 85)
    for h in greedy_history:
        print(f"{h['step']:<6} {'+ #' + str(h['added']) + ' ' + h['name']:<35} {str(h['selected']):>30} {h['test_return']:>9.2f}%")

    print(f"\nGREEDY BEST: strategies {selected} → test return {best_test_ret:.2f}% (vs baseline {baseline_test:.2f}%, delta={best_test_ret - baseline_test:+.2f}%)")

    # =========================================================================
    # PHASE 3: Drop-One Refinement
    # =========================================================================
    if len(selected) >= 2:
        print("\n" + "=" * 100)
        print("PHASE 3: DROP-ONE REFINEMENT")
        print("=" * 100)
        print(f"Testing removal of each strategy from greedy set {selected}\n")

        drop_results = []
        for sid_to_drop in selected:
            reduced_set = [s for s in selected if s != sid_to_drop]
            name = STRATEGY_NAMES.get(sid_to_drop, f"#{sid_to_drop}")
            print(f"  Without #{sid_to_drop} ({name}): ", end="")
            result = run_combination(harness, reduced_set)
            test_ret = result.get('test_return', 0)
            impact = best_test_ret - test_ret  # Positive = removing hurts (strategy is valuable)
            drop_results.append((sid_to_drop, test_ret, impact))
            print(f"test={test_ret:.2f}% (removing costs {impact:+.2f}%)")

        # Check if dropping any strategy actually helps
        improvements = [(sid, ret, imp) for sid, ret, imp in drop_results if imp < 0]
        if improvements:
            improvements.sort(key=lambda x: x[2])  # Most negative first = biggest improvement from removal
            best_drop = improvements[0]
            print(f"\n  >> IMPROVEMENT FOUND: Dropping #{best_drop[0]} ({STRATEGY_NAMES.get(best_drop[0], '')}) "
                  f"improves test return from {best_test_ret:.2f}% to {best_drop[1]:.2f}%")

            # Apply the best drop
            selected = [s for s in selected if s != best_drop[0]]
            best_test_ret = best_drop[1]

            # Recursively try more drops
            while len(selected) >= 2:
                found_drop = False
                for sid_to_drop in selected:
                    reduced_set = [s for s in selected if s != sid_to_drop]
                    result = run_combination(harness, reduced_set)
                    test_ret = result.get('test_return', 0)
                    if test_ret > best_test_ret:
                        print(f"  >> Further improvement: Dropping #{sid_to_drop} → {test_ret:.2f}%")
                        selected = reduced_set
                        best_test_ret = test_ret
                        found_drop = True
                        break
                if not found_drop:
                    break
        else:
            print(f"\n  >> All strategies in the set are contributing positively. No drops needed.")

    # =========================================================================
    # FINAL RESULT
    # =========================================================================
    print("\n" + "=" * 100)
    print("FINAL OPTIMAL COMBINATION")
    print("=" * 100)
    print(f"\nStrategies (in order): {selected}")
    for sid in selected:
        print(f"  #{sid}: {STRATEGY_NAMES.get(sid, '???')}")
    print(f"\nTest Return: {best_test_ret:.2f}%")
    print(f"Baseline:    {baseline_test:.2f}%")
    print(f"Delta:       {best_test_ret - baseline_test:+.2f}%")

    # Run the final combination one more time to get full stats
    print(f"\n--- Final validation run with optimal set ---")
    final_result = run_combination(harness, selected)
    print(f"\nFinal Stats:")
    print(f"  Train: Return={final_result.get('train_return', 0):.2f}%, WR={final_result.get('train_wr', 0):.1f}%, "
          f"Trades={final_result.get('train_trades', 0)}, PF={final_result.get('train_pf', 0):.2f}, "
          f"MaxDD={final_result.get('train_max_dd', 0):.2f}%")
    print(f"  Test:  Return={final_result.get('test_return', 0):.2f}%, WR={final_result.get('test_wr', 0):.1f}%, "
          f"Trades={final_result.get('test_trades', 0)}, PF={final_result.get('test_pf', 0):.2f}, "
          f"MaxDD={final_result.get('test_max_dd', 0):.2f}%")

    # Save search results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    search_data = {
        'ticker': harness.ticker, 'interval': harness.interval,
        'n_trials': harness.n_trials, 'n_workers': harness.n_workers,
        'mode': 'smart_search',
        'timestamp': datetime.now().isoformat(),
        'baseline_test_return': baseline_test,
        'phase1_solo_results': {str(k): v for k, v in solo_results.items()},
        'phase1_ranking': [{'strategy': sid, 'name': STRATEGY_NAMES.get(sid, ''), 'delta': d['delta']} for sid, d in ranked],
        'phase2_greedy_history': greedy_history,
        'phase3_optimal_set': selected,
        'final_test_return': best_test_ret,
        'final_delta': best_test_ret - baseline_test,
        'final_result': {k: v for k, v in final_result.items() if k != 'best_params'},
    }
    filename = f'strategy_search_results_{harness.ticker}_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(search_data, f, indent=2, default=str)
    print(f"\nSearch results saved to: {filename}")

    return selected, best_test_ret


def main():
    parser = argparse.ArgumentParser(description='Cumulative Strategy Test Harness')
    parser.add_argument('--ticker', default='ES=F', help='Ticker symbol')
    parser.add_argument('--interval', default='15m', help='Bar interval')
    parser.add_argument('--trials', type=int, default=10000, help='Optuna trials per step')
    parser.add_argument('--workers', type=int, default=None, help='Parallel workers')
    parser.add_argument('--metric', default='risk_adjusted', help='Optimization metric')
    parser.add_argument('--days', type=int, default=59, help='Days of data')
    parser.add_argument('--mode', default='cumulative', choices=['cumulative', 'search'],
                        help='cumulative=fixed 1-20 order, search=smart combinatorial search')
    args = parser.parse_args()

    n_workers = args.workers or min(os.cpu_count() or 8, 32)

    harness = CumulativeTestHarness(
        ticker=args.ticker, interval=args.interval,
        n_trials=args.trials, n_workers=n_workers,
        optimize_metric=args.metric, days=args.days,
    )

    harness.prepare()
    harness.run_baseline()

    if args.mode == 'search':
        smart_search(harness)
    else:
        for sid in range(1, 21):
            harness.run_strategy(sid)
        harness.print_summary()
        harness.save_results()


if __name__ == '__main__':
    main()
