#!/usr/bin/env python3
"""
Walk-Forward Validation for Velocity Strategies

This script implements proper train/test split to prevent look-ahead bias:
1. Train period (80%): First ~584 days - optimize parameters here
2. Test period (20%): Last ~146 days - validate parameters here

Usage:
    python velocity_walkforward_validation.py --ticker BTC-USD --interval 1d --period 2y --n-trials 50000 --n-jobs 32
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optuna_worker import VelocityOptunaObjective, run_velocity_study
from oscillator_indicators import create_composite_oscillator_features
from novel_indicators import (
    calculate_arwo, calculate_dco, calculate_vcmo, calculate_ics,
    calculate_mji, calculate_prf, calculate_ewaf, calculate_kfif
)


DATABENTO_SYMBOLS = {
    'ES=F': 'ES.n.0', 'GC=F': 'GC.n.0', 'CL=F': 'CL.n.0',
    'NQ=F': 'NQ.n.0', 'YM=F': 'YM.n.0', 'SI=F': 'SI.n.0',
}


def fetch_data(ticker: str, interval: str, period: str, use_databento: bool = False) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance or Databento."""
    if use_databento:
        import databento as db
        from datetime import timezone

        # Parse period string to days
        period_days = {'60d': 60, '90d': 90, '120d': 120, '180d': 180, '1y': 365, '2y': 730, '5y': 1825}
        days = period_days.get(period)
        if days is None:
            # Try parsing as Xd format
            if period.endswith('d'):
                days = int(period[:-1])
            else:
                raise ValueError(f"Cannot parse period '{period}' for Databento. Use format like '180d'.")

        symbol = DATABENTO_SYMBOLS.get(ticker)
        if not symbol:
            raise ValueError(f"No Databento mapping for {ticker}. Available: {list(DATABENTO_SYMBOLS.keys())}")

        key = os.environ.get('DATABENTO_API_KEY', '')
        client = db.Historical(key)

        # Cap end at midnight UTC today to avoid data_end_after_available_end errors
        now_utc = datetime.now(timezone.utc)
        today_midnight = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc)
        end = today_midnight
        start = end - timedelta(days=days)

        print(f"Databento: requesting {symbol} ohlcv-1m {start.date()} to {end.date()}...")
        data = client.timeseries.get_range(
            dataset='GLBX.MDP3',
            symbols=[symbol],
            stype_in='continuous',
            schema='ohlcv-1m',
            start=start.strftime('%Y-%m-%dT%H:%M:%S'),
            end=end.strftime('%Y-%m-%dT%H:%M:%S'),
        )
        df_1m = data.to_df()
        print(f"  Got {len(df_1m)} 1m bars")

        resample_map = {'15m': '15min', '1h': '1h', '4h': '4h', '1d': '1D'}
        resample_freq = resample_map.get(interval, '15min')
        df = df_1m.resample(resample_freq).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()

        df.columns = [c.lower() for c in df.columns]
        print(f"  Resampled to {len(df)} {interval} bars ({df.index.min()} to {df.index.max()})")
        return df

    print(f"Fetching {ticker} {interval} data for {period}...")

    yf_ticker = yf.Ticker(ticker)
    df = yf_ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    print(f"  Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def prepare_velocity_data(df: pd.DataFrame, use_extra_indicators: bool = True):
    """Prepare data for velocity optimization including all oscillator types.

    Calculates:
    - Composite oscillator (original)
    - ICS: Indicator Convergence Score
    - ARWO: Adaptive Regime-Weighted Oscillator
    - DCO: Divergence Consensus Oscillator
    - VCMO: Volume-Confirmed Momentum Oscillator
    - MJI: Momentum Jerk Indicator
    - PRF: Percentile Rank Fusion
    - EWAF: Entropy-Weighted Adaptive Fusion
    - KFIF: Kalman-Filtered Indicator Fusion
    """

    # Ensure lowercase columns for novel indicators
    df_lower = df.copy()
    df_lower.columns = [c.lower() for c in df_lower.columns]

    close_col = 'close' if 'close' in df.columns else 'Close'
    close_prices = df[close_col].values

    # Calculate composite oscillator
    print("Calculating oscillators...")
    print("  - Composite oscillator...")
    osc_df = create_composite_oscillator_features(df)

    # Find the composite oscillator column
    composite_col = None
    for col in ['osc_composite_smooth', 'composite_oscillator', 'osc_composite']:
        if col in osc_df.columns:
            composite_col = col
            break

    if composite_col is None:
        raise ValueError(f"No composite oscillator column found. Available: {osc_df.columns.tolist()}")

    # Initialize all_oscillators dict with composite
    all_oscillators = {}

    # Align composite oscillator
    aligned_df = osc_df.loc[df.index].dropna()
    valid_idx = aligned_df.index

    all_oscillators['composite'] = aligned_df[composite_col].values

    # Calculate novel indicators (all normalized to -1 to +1)
    print("  - ICS (Indicator Convergence Score)...")
    try:
        ics_values = calculate_ics(df_lower)
        all_oscillators['ics'] = ics_values.loc[valid_idx].values
    except Exception as e:
        print(f"    Warning: ICS calculation failed: {e}")

    print("  - ARWO (Adaptive Regime-Weighted)...")
    try:
        arwo_values = calculate_arwo(df_lower)
        all_oscillators['arwo'] = arwo_values.loc[valid_idx].values
    except Exception as e:
        print(f"    Warning: ARWO calculation failed: {e}")

    print("  - DCO (Divergence Consensus)...")
    try:
        dco_values = calculate_dco(df_lower)
        all_oscillators['dco'] = dco_values.loc[valid_idx].values
    except Exception as e:
        print(f"    Warning: DCO calculation failed: {e}")

    print("  - VCMO (Volume-Confirmed Momentum)...")
    try:
        vcmo_values = calculate_vcmo(df_lower)
        all_oscillators['vcmo'] = vcmo_values.loc[valid_idx].values
    except Exception as e:
        print(f"    Warning: VCMO calculation failed: {e}")

    print("  - MJI (Momentum Jerk)...")
    try:
        mji_values = calculate_mji(df_lower)
        all_oscillators['mji'] = mji_values.loc[valid_idx].values
    except Exception as e:
        print(f"    Warning: MJI calculation failed: {e}")

    print("  - PRF (Percentile Rank Fusion)...")
    try:
        prf_values = calculate_prf(df_lower)
        all_oscillators['prf'] = prf_values.loc[valid_idx].values
    except Exception as e:
        print(f"    Warning: PRF calculation failed: {e}")

    print("  - EWAF (Entropy-Weighted Adaptive Fusion)...")
    try:
        ewaf_values = calculate_ewaf(df_lower)
        all_oscillators['ewaf'] = ewaf_values.loc[valid_idx].values
    except Exception as e:
        print(f"    Warning: EWAF calculation failed: {e}")

    print("  - KFIF (Kalman-Filtered Indicator Fusion)...")
    try:
        kfif_values, _, _ = calculate_kfif(df_lower)
        all_oscillators['kfif'] = kfif_values.loc[valid_idx].values
    except Exception as e:
        print(f"    Warning: KFIF calculation failed: {e}")

    print(f"  Calculated {len(all_oscillators)} oscillator types: {list(all_oscillators.keys())}")

    # Use composite as default osc_values for backward compatibility
    osc_values = all_oscillators.get('composite', all_oscillators[list(all_oscillators.keys())[0]])

    # Align close prices to valid index
    close_prices = df.loc[valid_idx, close_col].values

    # RSI cache (multiple periods)
    rsi_cache = {}
    for period in [5, 14, 21, 30]:
        delta = pd.Series(close_prices).diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_cache[period] = rsi.fillna(50).values

    # MACD histogram
    ema12 = pd.Series(close_prices).ewm(span=12).mean()
    ema26 = pd.Series(close_prices).ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_histogram = (macd_line - signal_line).values

    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close_prices).rolling(bb_period).mean()
    std = pd.Series(close_prices).rolling(bb_period).std()
    bb_upper = (sma + bb_std * std).fillna(method='bfill').values
    bb_lower = (sma - bb_std * std).fillna(method='bfill').values

    # Get low/high prices for MAE calculation
    low_col = 'low' if 'low' in df.columns else 'Low'
    high_col = 'high' if 'high' in df.columns else 'High'
    low_prices = df.loc[valid_idx, low_col].values
    high_prices = df.loc[valid_idx, high_col].values

    # Compute volatility regime (ATR percentile) for vol regime filter
    from velocity_trading.indicators.oscillators import calculate_vol_regime
    try:
        vol_regime = calculate_vol_regime(df_lower, atr_period=14, lookback=50)
        vol_regime_arr = vol_regime.loc[valid_idx].values
    except Exception as e:
        print(f"  Warning: vol_regime calculation failed: {e}")
        vol_regime_arr = np.full(len(close_prices), 0.5)

    # Compute Money Flow Velocity arrays for MFV filter
    mfv_flow_arr = None
    mfv_vel_arr = None
    mfv_acc_arr = None
    try:
        from novel_indicators_v2 import calculate_mfv
        # Determine interval from df index spacing
        _interval = '15m'  # default
        if len(df_lower) >= 2:
            delta = (df_lower.index[1] - df_lower.index[0]).total_seconds()
            if delta >= 86400:
                _interval = '1d'
            elif delta >= 3600:
                _interval = '1h'
            elif delta >= 900:
                _interval = '15m'
            elif delta >= 300:
                _interval = '5m'
        mfv_f, mfv_v, mfv_a = calculate_mfv(df_lower, interval=_interval)
        mfv_flow_arr = mfv_f.loc[valid_idx].values
        mfv_vel_arr = mfv_v.loc[valid_idx].values
        mfv_acc_arr = mfv_a.loc[valid_idx].values
    except Exception as e:
        print(f"  Warning: MFV calculation failed: {e}")

    # Compute Options Influence Zone arrays (from daily snapshots if available)
    options_gamma_zone_arr = None
    options_mp_zone_arr = None
    options_wall_zone_arr = None
    options_combined_zone_arr = None
    try:
        from market_data_db import get_market_db
        _mdb = get_market_db(suppress_init_message=True)
        if _mdb.options_snapshot_days() >= 30:
            start_str = valid_idx[0].strftime('%Y-%m-%d') if hasattr(valid_idx[0], 'strftime') else str(valid_idx[0])[:10]
            end_str = valid_idx[-1].strftime('%Y-%m-%d') if hasattr(valid_idx[-1], 'strftime') else str(valid_idx[-1])[:10]
            snapshots = _mdb.get_options_snapshots_range(start_str, end_str)
            if not snapshots.empty and len(snapshots) >= 10:
                # Forward-fill daily zones to each intraday bar
                gamma_daily = snapshots['gamma_zone']
                mp_daily = snapshots['max_pain_zone']
                wall_daily = snapshots['wall_zone']
                combined_daily = snapshots['combined_zone']

                # Reindex: normalize intraday timestamps to date, align to daily zones
                # Strip timezone from bar dates to match tz-naive DB index
                bar_dates = pd.Series(valid_idx).dt.normalize().dt.tz_localize(None)
                options_gamma_zone_arr = gamma_daily.reindex(bar_dates).ffill().bfill().values
                options_mp_zone_arr = mp_daily.reindex(bar_dates).ffill().bfill().values
                options_wall_zone_arr = wall_daily.reindex(bar_dates).ffill().bfill().values
                options_combined_zone_arr = combined_daily.reindex(bar_dates).ffill().bfill().values
                print(f"  Options zones: {len(snapshots)} daily snapshots loaded")
        else:
            days = _mdb.options_snapshot_days()
            if days > 0:
                print(f"  Options zones: {days} days collected (need 30+ for walkforward)")
    except Exception as e:
        print(f"  Warning: Options zone data not available: {e}")

    # KNN arrays are computed on the FULL dataset in run_walkforward() and injected later.
    # This avoids the test set being too small for MIN_HISTORY_BARS (500).

    return {
        'close_prices': close_prices,
        'low_prices': low_prices,
        'high_prices': high_prices,
        'osc_values': osc_values,
        'rsi_cache': rsi_cache,
        'macd_histogram': macd_histogram,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'dates': valid_idx,
        'bar_hours': valid_idx.hour.values if hasattr(valid_idx, 'hour') else None,
        'use_extra_indicators': use_extra_indicators,
        'all_oscillators': all_oscillators,
        'vol_regime': vol_regime_arr,
        'mfv_flow': mfv_flow_arr,
        'mfv_vel': mfv_vel_arr,
        'mfv_acc': mfv_acc_arr,
        'options_gamma_zone': options_gamma_zone_arr,
        'options_mp_zone': options_mp_zone_arr,
        'options_wall_zone': options_wall_zone_arr,
        'options_combined_zone': options_combined_zone_arr,
        'knn_prob_arrays': None,  # Injected by run_walk_forward_validation() after full-df KNN
        'knn_confidence_arrays': None,
    }


def run_backtest_with_params(params: dict, data: dict) -> dict:
    """Run a single backtest with given parameters."""

    close_prices = data['close_prices']
    low_prices = data.get('low_prices', close_prices)  # Fallback to close if not available
    high_prices = data.get('high_prices', close_prices)

    # Select oscillator type from params (default to composite for backward compatibility)
    osc_type = params.get('oscillator_type', 'composite')
    all_oscillators = data.get('all_oscillators', {})

    if osc_type in all_oscillators:
        osc_values = all_oscillators[osc_type]
    else:
        # Fall back to default osc_values
        osc_values = data['osc_values']

    # Apply smoothing
    if params.get('vel_smoothing', 1) > 1:
        osc_smooth = pd.Series(osc_values).rolling(window=params['vel_smoothing']).mean().bfill().values
    else:
        osc_smooth = osc_values

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

    # Wall zone directional bias: shift oscillator toward wall-predicted direction
    oz_wb = params.get('oz_wall_bias', 0.0)
    options_wall_zone = data.get('options_wall_zone')
    if oz_wb > 0 and options_wall_zone is not None:
        osc_smooth = osc_smooth - options_wall_zone * oz_wb

    velocity = np.diff(osc_smooth, prepend=osc_smooth[0])
    acceleration = np.diff(velocity, prepend=velocity[0])
    jerk = np.diff(acceleration, prepend=acceleration[0])

    # Build conditions
    vel_cross_up = (velocity > 0) & (np.roll(velocity, 1) <= 0)
    vel_cross_down = (velocity < 0) & (np.roll(velocity, 1) >= 0)
    in_oversold = osc_smooth < params.get('oversold_threshold', -0.2)
    in_overbought = osc_smooth > params.get('overbought_threshold', 0.2)
    extreme_mult = params.get('extreme_zone_mult', 2.0)
    extreme_oversold = osc_smooth < (params.get('oversold_threshold', -0.2) * extreme_mult)
    extreme_overbought = osc_smooth > (params.get('overbought_threshold', 0.2) * extreme_mult)

    vel_std_window = params.get('velocity_std_window', 10)
    vel_std = pd.Series(velocity).rolling(vel_std_window, min_periods=1).std().fillna(np.std(velocity)).values
    momentum_mult = params.get('momentum_multiplier', 1.5)
    strong_momentum_up = velocity > vel_std * momentum_mult
    strong_momentum_down = velocity < -vel_std * momentum_mult

    # Signal type conditions
    sig_type = params.get('signal_type', 'any_reversal')
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
    else:
        buy_cond = vel_cross_up | extreme_oversold
        sell_cond = vel_cross_down | extreme_overbought

    # Multi-oscillator OR: catch blind spots where primary oscillator misses a move
    if params.get('use_multi_osc_or', False) and all_oscillators:
        osc_keys = [k for k in sorted(all_oscillators.keys()) if k != params.get('oscillator_type', 'composite')]
        secondary_buy = np.zeros(len(close_prices), dtype=bool)
        vel_smoothing = params.get('vel_smoothing', 1)
        for sec_key in osc_keys[:8]:
            sec_osc = all_oscillators[sec_key]
            if vel_smoothing > 1:
                sec_smooth = pd.Series(sec_osc).rolling(window=vel_smoothing).mean().bfill().values
            else:
                sec_smooth = sec_osc.copy()
            if oz_wb > 0 and options_wall_zone is not None:
                sec_smooth = sec_smooth - options_wall_zone * oz_wb
            sec_vel = np.diff(sec_smooth, prepend=sec_smooth[0])
            sec_cross_up = (sec_vel > 0) & (np.roll(sec_vel, 1) <= 0)
            sec_extreme_os = sec_smooth < (params.get('oversold_threshold', -0.2) * extreme_mult)
            if sig_type in ('velocity_crossover_or_zone', 'zone_only'):
                sec_signal = sec_cross_up | sec_extreme_os
            elif sig_type == 'any_reversal':
                sec_std = pd.Series(sec_vel).rolling(vel_std_window, min_periods=1).std().fillna(np.std(sec_vel)).values
                sec_in_os = sec_smooth < params.get('oversold_threshold', -0.2)
                sec_signal = sec_cross_up | sec_extreme_os | ((sec_vel > sec_std * momentum_mult) & sec_in_os)
            else:
                sec_signal = sec_cross_up | sec_extreme_os
            secondary_buy = secondary_buy | sec_signal
        buy_cond = buy_cond | secondary_buy

    # Entry velocity magnitude filter: reject weak zero-crossings
    if params.get('use_vel_magnitude_filter', False):
        min_vel = params.get('min_entry_velocity', 0.01)
        buy_cond = buy_cond & (np.abs(velocity) > min_vel)

    # Oscillator consensus voting: require N oscillators to agree
    buy_votes = np.zeros(len(close_prices), dtype=int)
    consensus_count = params.get('consensus_count', 2)
    if params.get('use_consensus', False) and all_oscillators and len(all_oscillators) >= 3:
        sell_votes = np.zeros(len(close_prices), dtype=int)
        for osc_name, osc_vals in all_oscillators.items():
            if len(osc_vals) != len(close_prices):
                continue
            o_smooth = osc_vals
            if params.get('vel_smoothing', 1) > 1:
                o_smooth = pd.Series(osc_vals).rolling(window=params['vel_smoothing']).mean().bfill().values
            o_vel = np.diff(o_smooth, prepend=o_smooth[0])
            o_cross_up = (o_vel > 0) & (np.roll(o_vel, 1) <= 0)
            o_cross_down = (o_vel < 0) & (np.roll(o_vel, 1) >= 0)
            o_oversold = o_smooth < params.get('oversold_threshold', -0.2)
            o_overbought = o_smooth > params.get('overbought_threshold', 0.2)
            buy_votes += (o_cross_up | o_oversold).astype(int)
            sell_votes += (o_cross_down | o_overbought).astype(int)
        buy_cond = buy_cond & (buy_votes >= consensus_count)
        sell_cond = sell_cond & (sell_votes >= consensus_count)

    # RTH vs overnight signal strength: require stronger signals overnight
    bar_hours = data.get('bar_hours')
    if params.get('use_rth_filter', False) and bar_hours is not None:
        is_rth = (bar_hours >= 9) & (bar_hours < 16)
        overnight_boost = params.get('overnight_consensus_boost', 2)
        if params.get('use_consensus', False) and all_oscillators and len(all_oscillators) >= 3:
            overnight_ok = buy_votes >= (consensus_count + overnight_boost)
        else:
            min_vel = params.get('min_entry_velocity', 0.01)
            overnight_ok = np.abs(velocity) > min_vel * 2
        buy_cond = buy_cond & (is_rth | overnight_ok)

    # Acceleration requirement
    if params.get('require_accel', False):
        buy_cond = buy_cond & (acceleration > 0)
        sell_cond = sell_cond & (acceleration < 0)

    # RSI filter
    use_extra_indicators = data.get('use_extra_indicators', False)
    rsi_cache = data.get('rsi_cache', {})
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

    # MACD confirmation filter
    macd_histogram = data.get('macd_histogram')
    if use_extra_indicators and params.get('use_macd_confirm', False) and macd_histogram is not None:
        macd_improving = macd_histogram > np.roll(macd_histogram, 1)
        macd_declining = macd_histogram < np.roll(macd_histogram, 1)
        buy_cond = buy_cond & macd_improving
        sell_cond = sell_cond & macd_declining

    # Bollinger Band filter
    bb_lower = data.get('bb_lower')
    bb_upper = data.get('bb_upper')
    if use_extra_indicators and params.get('use_bb_filter', False) and bb_lower is not None:
        buy_cond = buy_cond & (close_prices < bb_lower)
        sell_cond = sell_cond & (close_prices > bb_upper)

    # Volatility regime filter (ATR percentile)
    if params.get('use_vol_regime_filter', False):
        vol_regime = data.get('vol_regime')
        if vol_regime is not None:
            vol_thresh = params.get('vol_regime_percentile_threshold', 0.25)
            buy_cond = buy_cond & (vol_regime > vol_thresh)

    # Money Flow Velocity filter
    if params.get('use_mfv_filter', False):
        mfv_mode = params.get('mfv_mode', 'velocity')
        mfv_thresh = params.get('mfv_threshold', 0.0)
        mfv_flow = data.get('mfv_flow')
        mfv_vel = data.get('mfv_vel')

        if mfv_mode == 'velocity' and mfv_vel is not None:
            buy_cond = buy_cond & (mfv_vel > mfv_thresh)
            sell_cond = sell_cond & (mfv_vel < -mfv_thresh)
        elif mfv_mode == 'flow' and mfv_flow is not None:
            buy_cond = buy_cond & (mfv_flow > mfv_thresh)
            sell_cond = sell_cond & (mfv_flow < -mfv_thresh)
        elif mfv_mode == 'both' and mfv_flow is not None and mfv_vel is not None:
            buy_cond = buy_cond & (mfv_flow > mfv_thresh) & (mfv_vel > mfv_thresh)
            sell_cond = sell_cond & (mfv_flow < -mfv_thresh) & (mfv_vel < -mfv_thresh)

    # Combined zone extreme conviction: ADD signals when zone is strongly directional
    options_combined_zone = data.get('options_combined_zone')
    oz_eb = params.get('oz_extreme_boost', 0.0)
    oz_et = params.get('oz_extreme_thresh', 0.7)
    if oz_eb > 0 and options_combined_zone is not None:
        extreme_bull = options_combined_zone > oz_et
        extreme_bear = options_combined_zone < -oz_et
        relaxed_oversold = osc_smooth < (params['oversold_threshold'] * (1 - oz_eb))
        relaxed_overbought = osc_smooth > (params['overbought_threshold'] * (1 - oz_eb))
        buy_cond = buy_cond | (extreme_bull & relaxed_oversold & (velocity > 0))
        sell_cond = sell_cond | (extreme_bear & relaxed_overbought & (velocity < 0))

    # Options zone entry filter: block longs when combined zone is too bearish
    if params.get('use_oz_entry_filter', False) and options_combined_zone is not None:
        oz_entry_thresh = params.get('oz_entry_threshold', 0.0)
        buy_cond = buy_cond & (options_combined_zone >= oz_entry_thresh)

    # KNN Pattern Matcher filter
    if params.get('use_knn_filter', False):
        knn_prob_arrays = data.get('knn_prob_arrays') or {}
        knn_conf_arrays = data.get('knn_confidence_arrays') or {}
        knn_h = params.get('knn_horizon', 8)
        knn_pt = params.get('knn_prob_threshold', 0.55)
        knn_ct = params.get('knn_confidence_threshold', 0.1)
        prob = knn_prob_arrays.get(knn_h)
        conf = knn_conf_arrays.get(knn_h)
        if prob is not None and conf is not None:
            buy_cond = buy_cond & (prob > knn_pt) & (conf > knn_ct)
            sell_cond = sell_cond & (prob < (1 - knn_pt)) & (conf > knn_ct)

    # Trading simulation
    position = 0  # 0 = flat, 1 = long
    entry_price = 0
    entry_bar = 0
    min_price_during_trade = 0  # For MAE tracking (long positions)
    max_price_during_trade = 0  # For MFE tracking (long positions)
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_drawdown = 0.0

    min_bars_between = params.get('min_bars_between', 1)
    min_hold_bars = params.get('min_hold_bars', 1)
    stop_loss_pct = params.get('stop_loss_pct', 5.0)
    take_profit_pct = params.get('take_profit_pct', 10.0)
    exit_opposite = params.get('exit_on_opposite_signal', True)
    exit_midline = params.get('exit_on_midline_cross', False)

    # Acceleration exit params
    use_accel_exit = params.get('use_accel_exit', False)
    accel_exit_type = params.get('accel_exit_type', 'sign_reversal')
    accel_exit_threshold = params.get('accel_exit_threshold', 0.0)
    accel_exit_min_pnl = params.get('accel_exit_min_pnl', 0.5)
    use_jerk_confirm = params.get('use_jerk_confirm', False)
    jerk_confirm_threshold = params.get('jerk_confirm_threshold', 0.0)

    # Trailing stop params
    use_trailing_stop = params.get('use_trailing_stop', False)
    trailing_stop_pct = params.get('trailing_stop_pct', 1.0)
    trailing_stop_activation_pct = params.get('trailing_stop_activation_pct', 0.3)

    # Break-even stop params
    use_breakeven_stop = params.get('use_breakeven_stop', False)
    breakeven_trigger_pct = params.get('breakeven_trigger_pct', 0.3)
    breakeven_offset_pct = params.get('breakeven_offset_pct', 0.05)

    high_watermark = 0.0
    last_trade_bar = -min_bars_between

    # Adaptive cooldown after losses
    consecutive_losses = 0
    use_adaptive_cooldown = params.get('use_adaptive_cooldown', False)
    cooldown_loss_streak = params.get('cooldown_loss_streak', 2)
    cooldown_extra_bars = params.get('cooldown_extra_bars', 5)

    for i in range(1, len(close_prices)):
        price = close_prices[i]
        low = low_prices[i]

        if position == 0:
            # Check for entry (with adaptive cooldown)
            eff_min_bars = min_bars_between
            if use_adaptive_cooldown and consecutive_losses >= cooldown_loss_streak:
                eff_min_bars += cooldown_extra_bars
            if buy_cond[i] and (i - last_trade_bar) >= eff_min_bars:
                position = 1
                entry_price = price
                entry_bar = i
                # Initialize MAE/MFE tracking at entry price, NOT intrabar low/high
                # We enter at the CLOSE of bar i, so intrabar movement on entry bar
                # already happened BEFORE our entry and shouldn't count
                min_price_during_trade = entry_price
                max_price_during_trade = entry_price
                high_watermark = high_prices[i]
        else:
            # Update MAE/MFE tracking
            min_price_during_trade = min(min_price_during_trade, low)
            max_price_during_trade = max(max_price_during_trade, high_prices[i])
            high_watermark = max(high_watermark, high_prices[i])

            # Check for exit
            current_pnl = (price - entry_price) / entry_price * 100
            bars_held = i - entry_bar

            # Calculate intrabar P&L for stop/take profit (using low/high)
            intrabar_low_pnl = (low - entry_price) / entry_price * 100
            intrabar_high_pnl = (high_prices[i] - entry_price) / entry_price * 100

            exit_reason = None
            exit_price_override = None

            # Gamma vol scaling: adapt SL/TP to gamma-predicted volatility
            options_gamma_zone = data.get('options_gamma_zone')
            oz_gv = params.get('oz_gamma_vol', 0.0)
            if oz_gv > 0 and options_gamma_zone is not None:
                vol_scale = 1 - options_gamma_zone[i] * oz_gv
                eff_sl = max(0.1, stop_loss_pct * vol_scale)
                eff_tp = take_profit_pct * vol_scale
            else:
                eff_sl = stop_loss_pct
                eff_tp = take_profit_pct

            # Priority 1: Stop loss - check if intrabar low hit stop level
            if intrabar_low_pnl <= -eff_sl:
                exit_reason = 'stop_loss'
                exit_price_override = entry_price * (1 - eff_sl / 100)
            # Priority 2: Take profit - check if intrabar high hit take profit level
            elif intrabar_high_pnl >= eff_tp:
                exit_reason = 'take_profit'
                exit_price_override = entry_price * (1 + eff_tp / 100)

            # Priority 3: Trailing stop
            if not exit_reason and use_trailing_stop:
                hwm_pnl = (high_watermark - entry_price) / entry_price * 100
                if hwm_pnl >= trailing_stop_activation_pct:
                    trail_level = high_watermark * (1 - trailing_stop_pct / 100)
                    if low <= trail_level:
                        exit_reason = 'trailing_stop'
                        exit_price_override = trail_level

            # Priority 4: Break-even stop
            if not exit_reason and use_breakeven_stop:
                hwm_pnl = (high_watermark - entry_price) / entry_price * 100
                if hwm_pnl >= breakeven_trigger_pct:
                    be_price = entry_price * (1 + breakeven_offset_pct / 100)
                    # Price must have actually reached be_price before we can exit there
                    if high_watermark >= be_price and low <= be_price:
                        exit_reason = 'breakeven_stop'
                        exit_price_override = be_price

            # Time stop: exit stale trades that haven't moved
            if not exit_reason and params.get('use_time_stop', False):
                ts_bars = params.get('time_stop_bars', 4)
                ts_min_pnl = params.get('time_stop_min_pnl', 0.05)
                if bars_held >= ts_bars and current_pnl < ts_min_pnl:
                    exit_reason = 'time_stop'

            # Priority 5: Acceleration exit (requires min_hold_bars)
            if not exit_reason and use_accel_exit and current_pnl >= accel_exit_min_pnl and bars_held >= min_hold_bars:
                accel_exit_triggered = False
                if accel_exit_type == 'sign_reversal':
                    accel_exit_triggered = acceleration[i] < 0
                elif accel_exit_type == 'magnitude':
                    accel_exit_triggered = acceleration[i] < -accel_exit_threshold
                elif accel_exit_type == 'both':
                    accel_exit_triggered = acceleration[i] < 0 and abs(acceleration[i]) > accel_exit_threshold

                if accel_exit_triggered:
                    if not use_jerk_confirm or jerk[i] < -jerk_confirm_threshold:
                        exit_reason = 'accel_exit'
            # Priority 6: Opposite signal (conditional)
            if not exit_reason and exit_opposite and sell_cond[i] and bars_held >= min_hold_bars:
                opp_mode = params.get('opp_exit_mode', 'always')
                opp_ok = True
                if opp_mode == 'losing_only':
                    opp_ok = current_pnl < 0
                elif opp_mode == 'stale_only':
                    opp_ok = bars_held >= params.get('opp_exit_stale_bars', 5)
                if opp_ok:
                    exit_reason = 'opposite_signal'
            # Priority 7: Midline cross (with consecutive bars + min PnL requirement)
            if not exit_reason and exit_midline and bars_held >= min_hold_bars:
                midline_exit_bars_req = params.get('midline_exit_bars', 1)
                midline_triggered = False
                if midline_exit_bars_req > 1:
                    # Count consecutive bars above midline ending at i
                    consec = 0
                    for j in range(i, max(i - midline_exit_bars_req - 1, -1), -1):
                        if osc_smooth[j] > 0:
                            consec += 1
                        else:
                            break
                    midline_triggered = consec >= midline_exit_bars_req
                else:
                    # Single bar: require actual crossover (was <= 0, now > 0)
                    midline_triggered = osc_smooth[i] > 0 and osc_smooth[i-1] <= 0
                # Only exit on midline if PnL meets threshold
                if midline_triggered:
                    midline_min_pnl = params.get('midline_exit_min_pnl', -999)
                    if current_pnl >= midline_min_pnl:
                        exit_reason = 'midline_cross'

            if exit_reason:
                # Use override price for stop loss / take profit, otherwise use close
                actual_exit_price = exit_price_override if exit_price_override else price
                pnl = (actual_exit_price - entry_price) / entry_price * 100
                equity *= (1 + pnl / 100)
                peak_equity = max(peak_equity, equity)
                drawdown = (peak_equity - equity) / peak_equity * 100
                max_drawdown = max(max_drawdown, drawdown)

                # Calculate MAE (Maximum Adverse Excursion) for this trade
                # MAE = worst unrealized loss during the trade (always positive or zero)
                # For stopped trades, MAE is capped at the stop level
                mae = (entry_price - min_price_during_trade) / entry_price * 100
                mae = max(0, mae)  # Ensure non-negative
                if exit_reason == 'stop_loss':
                    mae = min(mae, stop_loss_pct)  # Cap at stop level for stopped trades

                # Calculate MFE (Maximum Favorable Excursion) for this trade
                # MFE = best unrealized profit during the trade
                mfe = (max_price_during_trade - entry_price) / entry_price * 100
                mfe = max(0, mfe)  # Ensure non-negative
                if exit_reason == 'take_profit':
                    mfe = min(mfe, take_profit_pct)  # Cap at TP level for TP trades

                # Calculate trade efficiency (what % of potential profit was captured)
                # Efficiency = actual P&L / MFE (capped at 100% for winning trades)
                efficiency = (pnl / mfe * 100) if mfe > 0 else 0

                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'entry_price': entry_price,
                    'exit_price': actual_exit_price,
                    'pnl': pnl,
                    'mae': mae,
                    'mfe': mfe,
                    'efficiency': efficiency,
                    'exit_reason': exit_reason
                })

                position = 0
                last_trade_bar = i

                # Track consecutive losses for adaptive cooldown
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

    # Calculate metrics
    if not trades:
        return {
            'total_return': 0,
            'n_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'avg_mae': 0,
            'max_mae': 0
        }

    total_return = equity - 100
    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if t['pnl'] < 0]

    # MAE statistics
    mae_values = [t['mae'] for t in trades]
    avg_mae = np.mean(mae_values) if mae_values else 0
    max_mae = max(mae_values) if mae_values else 0

    # MFE statistics
    mfe_values = [t['mfe'] for t in trades]
    avg_mfe = np.mean(mfe_values) if mfe_values else 0
    max_mfe = max(mfe_values) if mfe_values else 0

    # Efficiency statistics (only for winning trades)
    winning_trades = [t for t in trades if t['pnl'] > 0]
    efficiency_values = [t['efficiency'] for t in winning_trades]
    avg_efficiency = np.mean(efficiency_values) if efficiency_values else 0

    return {
        'total_return': total_return,
        'n_trades': len(trades),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
        'max_drawdown': max_drawdown,
        'profit_factor': sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf'),
        'avg_mae': avg_mae,
        'max_mae': max_mae,
        'avg_mfe': avg_mfe,
        'max_mfe': max_mfe,
        'avg_efficiency': avg_efficiency,
        'trades': trades
    }


def run_walk_forward_validation(
    ticker: str,
    interval: str,
    period: str,
    n_trials: int,
    n_jobs: int,
    train_ratio: float = 0.8,
    optimize_metric: str = 'total_return',
    use_databento: bool = False,
    force_signal_type: str = None
):
    """
    Run walk-forward validation for velocity strategy.

    1. Split data into train (80%) and test (20%)
    2. Optimize on train period
    3. Validate on test period
    """

    print("=" * 80)
    print("WALK-FORWARD VALIDATION FOR VELOCITY STRATEGY")
    print("=" * 80)
    print(f"Ticker: {ticker}")
    print(f"Interval: {interval}")
    print(f"Period: {period}")
    print(f"Train/Test Split: {train_ratio*100:.0f}%/{(1-train_ratio)*100:.0f}%")
    print(f"Optimization Trials: {n_trials:,}")
    print(f"Parallel Workers: {n_jobs}")
    print("=" * 80)

    # Fetch full data
    df = fetch_data(ticker, interval, period, use_databento=use_databento)

    # Calculate split point
    n_bars = len(df)
    train_end_idx = int(n_bars * train_ratio)

    train_df = df.iloc[:train_end_idx]
    test_df = df.iloc[train_end_idx:]

    print(f"\nData Split:")
    print(f"  Total bars: {n_bars}")
    print(f"  Train period: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} bars)")
    print(f"  Test period: {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} bars)")

    # Precompute KNN arrays on FULL dataset (test bars need training-period neighbors)
    full_knn_prob = None
    full_knn_conf = None
    try:
        from knn_pattern_matcher import KNNPatternMatcher
        from novel_indicators_v2 import calculate_all_novel_v2_indicators

        _knn_interval = '15m'
        if len(df) >= 2:
            _delta = (df.index[1] - df.index[0]).total_seconds()
            if _delta >= 86400:
                _knn_interval = '1d'
            elif _delta >= 3600:
                _knn_interval = '1h'

        print("\nKNN: Computing V2 features on full dataset...")
        df_lower_full = df.copy()
        df_lower_full.columns = [c.lower() for c in df_lower_full.columns]
        knn_df = calculate_all_novel_v2_indicators(df_lower_full, interval=_knn_interval)

        # Composite oscillator for osc_smooth/velocity/acceleration features
        _osc_df = create_composite_oscillator_features(df)
        _comp_col = None
        for _c in ['osc_composite_smooth', 'composite_oscillator', 'osc_composite']:
            if _c in _osc_df.columns:
                _comp_col = _c
                break
        if _comp_col:
            _aligned = _osc_df[_comp_col].reindex(knn_df.index).fillna(0).values
            knn_df['osc_smooth'] = _aligned
            _v = np.diff(_aligned, prepend=_aligned[0])
            knn_df['velocity'] = _v
            knn_df['acceleration'] = np.diff(_v, prepend=_v[0])

        # Derived features
        knn_df['close_pct_change'] = np.clip(
            knn_df['close'].pct_change().fillna(0).values * 10, -1, 1
        )
        if 'volume' in knn_df.columns:
            _vol_ma = knn_df['volume'].rolling(20).mean()
            knn_df['volume_ratio'] = np.clip(
                (knn_df['volume'] / _vol_ma.replace(0, np.nan)).fillna(1).values - 1, -1, 1
            )
        else:
            knn_df['volume_ratio'] = 0.0

        _tr = np.maximum(
            knn_df['high'].values - knn_df['low'].values,
            np.maximum(
                np.abs(knn_df['high'].values - np.roll(knn_df['close'].values, 1)),
                np.abs(knn_df['low'].values - np.roll(knn_df['close'].values, 1))
            )
        )
        _atr = pd.Series(_tr).rolling(14).mean().values
        _atr_pctile = pd.Series(_atr).rolling(100).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        ).fillna(0.5).values
        knn_df['atr_percentile'] = np.clip(_atr_pctile * 2 - 1, -1, 1)

        matcher = KNNPatternMatcher(k=50, horizons=[4, 8, 16, 26])
        feat_cols = matcher.get_feature_columns()
        avail = [c for c in feat_cols if c in knn_df.columns]
        print(f"KNN: {len(avail)}/{len(feat_cols)} features, running match_all_bars on {len(knn_df)} bars...")

        import time as _t
        _t0 = _t.time()
        matcher.build_feature_matrix(knn_df, feat_cols)
        knn_results = matcher.match_all_bars()
        print(f"KNN: Completed in {_t.time() - _t0:.1f}s")

        full_knn_prob = {h: knn_results[f'knn_prob_up_{h}'] for h in [4, 8, 16, 26]}
        full_knn_conf = {h: knn_results[f'knn_confidence_{h}'] for h in [4, 8, 16, 26]}
    except Exception as e:
        print(f"Warning: KNN precomputation failed: {e}")
        import traceback
        traceback.print_exc()

    # Prepare train data for optimization
    print("\n" + "=" * 80)
    print("PHASE 1: OPTIMIZATION ON TRAINING DATA")
    print("=" * 80)

    train_data = prepare_velocity_data(train_df)

    # Inject KNN arrays (sliced to train portion) — uses valid_idx alignment
    if full_knn_prob is not None:
        # The full KNN arrays are indexed 0..n_bars-1 (aligned to df.index)
        # train_data covers df.iloc[:train_end_idx] but prepare_velocity_data may drop
        # warmup bars. Use the train data length to slice from the end of the train portion.
        train_n = len(train_data['close_prices'])
        # The valid_idx in prepare_velocity_data drops some initial bars, so the train
        # arrays map to the LAST train_n bars of df.iloc[:train_end_idx].
        # Compute offset: full_knn starts at index 0 of df, train valid starts at (train_end_idx - train_n)
        train_offset = train_end_idx - train_n
        train_data['knn_prob_arrays'] = {h: arr[train_offset:train_end_idx] for h, arr in full_knn_prob.items()}
        train_data['knn_confidence_arrays'] = {h: arr[train_offset:train_end_idx] for h, arr in full_knn_conf.items()}

    # Save data to temp file for parallel workers
    fd, data_path = tempfile.mkstemp(suffix='.joblib', prefix='velocity_wf_')
    os.close(fd)

    # Calculate minimum trade count based on trading days
    # For 15m intraday strategies: expect 3-9 trades/day, floor at 3/day
    train_bars = len(train_data['close_prices'])
    train_days = (train_df.index[-1] - train_df.index[0]).days
    # Estimate trading days (~5/7 of calendar days)
    train_trading_days = max(1, int(train_days * 5 / 7))
    min_trades = max(50, train_trading_days * 3)  # At least 3 trades/day

    print(f"  Minimum trade count filter: {min_trades} trades")
    print(f"  Trade count bonus: enabled (sqrt scaling)")

    train_data_for_optuna = {
        'close_prices': train_data['close_prices'],
        'osc_values': train_data['osc_values'],
        'rsi_cache': train_data['rsi_cache'],
        'macd_histogram': train_data['macd_histogram'],
        'bb_upper': train_data['bb_upper'],
        'bb_lower': train_data['bb_lower'],
        'optimize_metric': optimize_metric,
        'use_extra_indicators': True,
        'all_oscillators': train_data['all_oscillators'],
        'bar_hours': train_data.get('bar_hours'),
        'vol_regime': train_data.get('vol_regime'),
        'mfv_flow': train_data.get('mfv_flow'),
        'mfv_vel': train_data.get('mfv_vel'),
        'mfv_acc': train_data.get('mfv_acc'),
        'options_gamma_zone': train_data.get('options_gamma_zone'),
        'options_mp_zone': train_data.get('options_mp_zone'),
        'options_wall_zone': train_data.get('options_wall_zone'),
        'options_combined_zone': train_data.get('options_combined_zone'),
        'knn_prob_arrays': train_data.get('knn_prob_arrays'),
        'knn_confidence_arrays': train_data.get('knn_confidence_arrays'),
        'force_midline_exit': False,
        'force_opposite_exit': False,
        'sl_range': (0.5, 10.0),        # Broader: search wider SL range
        'tp_range': (1.0, 20.0),        # Broader: search wider TP range
        'use_drawdown_penalty': True,
        'max_drawdown_threshold': 10.0, # Stricter: penalize >10% drawdown
        'drawdown_penalty_weight': 0.5, # Higher penalty weight for drawdown
        'min_trades': min_trades,       # Reject strategies with too few trades
        # Trade count bonus: only for total_return metric (PF is already trade-neutral)
        'trade_count_bonus_weight': 0.3 if optimize_metric == 'total_return' else 0.0,
        # Force signal type if specified
        'force_signal_type': force_signal_type
    }

    if force_signal_type:
        print(f"  Forced signal type: {force_signal_type}")

    joblib.dump(train_data_for_optuna, data_path)

    # Run parallel optimization
    print(f"\nStarting {n_jobs} parallel workers with {n_trials // n_jobs} trials each...")
    start_time = time.time()

    trials_per_worker = n_trials // n_jobs
    seeds = [42 + i for i in range(n_jobs)]

    worker_results = joblib.Parallel(n_jobs=n_jobs, verbose=10)(
        joblib.delayed(run_velocity_study)(
            data_path, trials_per_worker, seed, optimize_metric, worker_id
        )
        for worker_id, seed in enumerate(seeds)
    )

    elapsed = time.time() - start_time
    print(f"\nOptimization completed in {elapsed:.1f}s ({n_trials / elapsed:.1f} trials/sec)")

    # Clean up temp file
    os.remove(data_path)

    # Collect and sort all results
    all_results = []
    for results in worker_results:
        all_results.extend(results)

    # Sort by total return (or optimize metric)
    all_results.sort(key=lambda x: x.get(optimize_metric, 0), reverse=True)

    if not all_results:
        print("ERROR: No valid results from optimization!")
        return None

    best_result = all_results[0]
    # Params are spread into result dict, extract them
    param_keys = [
        'oscillator_type',  # Novel oscillators (ics, arwo, dco, vcmo, mji, prf, ewaf, kfif)
        'signal_type', 'vel_smoothing', 'extreme_zone_mult', 'min_bars_between',
        'require_accel', 'oversold_threshold', 'overbought_threshold',
        'stop_loss_pct', 'take_profit_pct', 'exit_on_opposite_signal', 'exit_on_midline_cross',
        'min_hold_bars', 'velocity_std_window', 'momentum_multiplier',
        'double_bottom_lookback', 'divergence_lookback',
        'use_accel_exit', 'accel_exit_type', 'accel_exit_threshold',
        'accel_exit_min_pnl', 'accel_exit_lookback', 'use_jerk_confirm', 'jerk_confirm_threshold',
        'rsi_filter', 'rsi_period', 'rsi_oversold', 'rsi_overbought',
        'use_macd_confirm', 'use_bb_filter', 'use_regime_filter', 'use_fragility_filter', 'use_entropy_filter',
        'use_mfv_filter', 'mfv_mode', 'mfv_threshold',
        'oz_wall_bias', 'oz_gamma_vol', 'oz_extreme_thresh', 'oz_extreme_boost',
        'use_oz_entry_filter', 'oz_entry_threshold',
        'use_multi_osc_or',
        'use_vel_magnitude_filter', 'min_entry_velocity',
        'use_consensus', 'consensus_count',
        'use_rth_filter', 'overnight_consensus_boost',
        'use_knn_filter', 'knn_horizon', 'knn_prob_threshold', 'knn_confidence_threshold',
        # Exit management params (must match optuna_worker.py)
        'opp_exit_mode', 'opp_exit_stale_bars',
        'use_time_stop', 'time_stop_bars', 'time_stop_min_pnl',
        'midline_exit_bars', 'midline_exit_min_pnl',
        'use_trailing_stop', 'trailing_stop_pct', 'trailing_stop_activation_pct',
        'trailing_stop_type', 'trailing_atr_mult', 'trailing_atr_period',
        'use_breakeven_stop', 'breakeven_trigger_pct', 'breakeven_offset_pct',
        'use_adaptive_cooldown', 'cooldown_loss_streak', 'cooldown_extra_bars',
    ]
    best_params = {k: best_result[k] for k in param_keys if k in best_result}

    print(f"\n{'=' * 80}")
    print("BEST PARAMETERS FROM TRAINING (Top 3)")
    print("=" * 80)

    for i, result in enumerate(all_results[:3]):
        n_trades = result.get('num_trades', result.get('n_trades', 0))
        print(f"\n#{i+1}: Return={result['total_return']:.2f}%, WR={result['win_rate']:.1f}%, "
              f"Trades={n_trades}, DD={result['max_drawdown']:.1f}%")
        # Params are spread directly into result dict
        print(f"    oscillator_type: {result.get('oscillator_type', 'composite')}")
        print(f"    signal_type: {result.get('signal_type')}")
        print(f"    use_accel_exit: {result.get('use_accel_exit')}")
        if result.get('use_accel_exit'):
            print(f"    accel_exit_type: {result.get('accel_exit_type')}")
            print(f"    accel_exit_threshold: {result.get('accel_exit_threshold', 0):.4f}")
            print(f"    accel_exit_min_pnl: {result.get('accel_exit_min_pnl', 0):.2f}%")

    # Prepare test data
    print(f"\n{'=' * 80}")
    print("PHASE 2: OUT-OF-SAMPLE VALIDATION ON TEST DATA")
    print("=" * 80)

    test_data = prepare_velocity_data(test_df)

    # Inject KNN arrays (sliced to test portion)
    if full_knn_prob is not None:
        test_n = len(test_data['close_prices'])
        test_offset = n_bars - test_n  # offset into full array
        test_data['knn_prob_arrays'] = {h: arr[test_offset:] for h, arr in full_knn_prob.items()}
        test_data['knn_confidence_arrays'] = {h: arr[test_offset:] for h, arr in full_knn_conf.items()}

    # Test top N training configs on OOS data and pick the best valid one
    # This prevents overfitting by not just taking the #1 training result
    top_n = min(20, len(all_results))
    print(f"\nTesting top {top_n} training configs on out-of-sample data...")

    oos_candidates = []
    for i, train_result in enumerate(all_results[:top_n]):
        candidate_params = {k: train_result[k] for k in param_keys if k in train_result}
        oos_result = run_backtest_with_params(candidate_params, test_data)
        oos_candidates.append({
            'rank': i + 1,
            'params': candidate_params,
            'train_result': train_result,
            'test_result': oos_result
        })
        n_trades_train = train_result.get('num_trades', train_result.get('n_trades', 0))
        n_trades_test = oos_result.get('n_trades', 0)
        print(f"  #{i+1}: Train={train_result['total_return']:.2f}%/{n_trades_train}t "
              f"-> Test={oos_result['total_return']:.2f}%/{n_trades_test}t "
              f"WR={oos_result['win_rate']:.1f}% PF={oos_result['profit_factor']:.2f}")

    # Pick the best OOS result — require proportional trade count
    # Same trades/day density as training minimum
    test_days = (test_df.index[-1] - test_df.index[0]).days
    test_trading_days = max(1, int(test_days * 5 / 7))
    test_min_trades = max(30, test_trading_days * 3)  # 3 trades/day minimum
    print(f"\n  OOS minimum trade filter: {test_min_trades} trades "
          f"({test_trading_days} trading days × 3 trades/day)")

    valid_oos = [c for c in oos_candidates
                 if c['test_result']['total_return'] > 0
                 and c['test_result']['n_trades'] >= test_min_trades]

    if valid_oos:
        # Score: profit_factor * log2(n_trades) — rewards both edge and frequency
        import math
        def oos_score(candidate):
            tr = candidate['test_result']
            n = tr['n_trades']
            pf = max(tr.get('profit_factor', 1.0), 0.01)
            ret = tr['total_return']
            dd = max(tr['max_drawdown'], 0.1)
            # PF-weighted return with trade confidence
            return ret * pf * math.log2(max(n, 2)) / (1 + dd)

        valid_oos.sort(key=oos_score, reverse=True)

        # Show top 5 scored candidates
        print(f"\n  Top 5 OOS candidates by score:")
        for j, c in enumerate(valid_oos[:5]):
            tr = c['test_result']
            sc = oos_score(c)
            print(f"    #{c['rank']}: score={sc:.2f} | "
                  f"return={tr['total_return']:.2f}% | "
                  f"trades={tr['n_trades']} | "
                  f"WR={tr['win_rate']:.1f}% | "
                  f"PF={tr['profit_factor']:.2f} | "
                  f"DD={tr['max_drawdown']:.1f}%")

        best_candidate = valid_oos[0]
        best_params = best_candidate['params']
        best_result = best_candidate['train_result']
        test_result = best_candidate['test_result']
        score = oos_score(best_candidate)
        print(f"\n  ✓ Selected: Training rank #{best_candidate['rank']} "
              f"(score={score:.2f}, trades={test_result['n_trades']}, "
              f"return={test_result['total_return']:.2f}%, DD={test_result['max_drawdown']:.1f}%)")
    else:
        # Fallback to #1 training result if no valid OOS found
        print(f"\n  ⚠️ No config passed OOS validation (positive return + ≥20 trades)")
        print(f"  Using best training config as fallback")
        test_result = run_backtest_with_params(best_params, test_data)

    print(f"\nTest Period Results ({test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}):")
    print(f"  Total Return: {test_result['total_return']:.2f}%")
    print(f"  Win Rate: {test_result['win_rate']:.1f}%")
    print(f"  Trades: {test_result['n_trades']}")
    print(f"  Max Drawdown: {test_result['max_drawdown']:.1f}%")
    print(f"  Profit Factor: {test_result['profit_factor']:.2f}")
    print(f"  Avg MAE: {test_result['avg_mae']:.2f}%")
    print(f"  Max MAE: {test_result['max_mae']:.2f}%")

    # Compare train vs test
    print(f"\n{'=' * 80}")
    print("TRAIN vs TEST COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<20} {'Train':>15} {'Test':>15} {'Diff':>15}")
    print("-" * 65)

    train_return = best_result['total_return']
    test_return = test_result['total_return']
    print(f"{'Total Return':<20} {train_return:>14.2f}% {test_return:>14.2f}% {test_return - train_return:>+14.2f}%")

    # Calculate average return per day for fair comparison
    train_days = (train_df.index[-1] - train_df.index[0]).days
    test_days = (test_df.index[-1] - test_df.index[0]).days
    train_days = max(train_days, 1)  # Avoid division by zero
    test_days = max(test_days, 1)

    train_daily_return = train_return / train_days
    test_daily_return = test_return / test_days
    print(f"{'Avg Return/Day':<20} {train_daily_return:>14.3f}% {test_daily_return:>14.3f}% {test_daily_return - train_daily_return:>+14.3f}%")
    print(f"{'Days in Period':<20} {train_days:>15} {test_days:>15}")

    train_wr = best_result['win_rate']
    test_wr = test_result['win_rate']
    print(f"{'Win Rate':<20} {train_wr:>14.1f}% {test_wr:>14.1f}% {test_wr - train_wr:>+14.1f}%")

    train_dd = best_result['max_drawdown']
    test_dd = test_result['max_drawdown']
    print(f"{'Max Drawdown':<20} {train_dd:>14.1f}% {test_dd:>14.1f}% {test_dd - train_dd:>+14.1f}%")

    # MAE comparison
    train_avg_mae = best_result.get('avg_mae', 0)
    test_avg_mae = test_result.get('avg_mae', 0)
    print(f"{'Avg MAE':<20} {train_avg_mae:>14.2f}% {test_avg_mae:>14.2f}% {test_avg_mae - train_avg_mae:>+14.2f}%")

    train_max_mae = best_result.get('max_mae', 0)
    test_max_mae = test_result.get('max_mae', 0)
    print(f"{'Max MAE':<20} {train_max_mae:>14.2f}% {test_max_mae:>14.2f}% {test_max_mae - train_max_mae:>+14.2f}%")

    # Validation check
    print(f"\n{'=' * 80}")
    print("VALIDATION ASSESSMENT")
    print("=" * 80)

    # Check for overfitting
    is_valid = True
    warnings = []

    if test_return < 0:
        warnings.append("WARNING: Negative return on test data - possible overfitting")
        is_valid = False
    elif test_return < train_return * 0.3:
        warnings.append("WARNING: Test return is less than 30% of train return - possible overfitting")

    if test_wr < 50 and test_result['n_trades'] >= 5:
        warnings.append("WARNING: Win rate below 50% on test data")

    if test_dd > 30:
        warnings.append("WARNING: Max drawdown exceeds 30% on test data")

    if test_result['n_trades'] < 3:
        warnings.append("WARNING: Less than 3 trades on test data - insufficient validation")

    if warnings:
        for w in warnings:
            print(f"  {w}")
    else:
        print("  ✓ Strategy passed validation checks")
        print("  ✓ Consistent performance between train and test periods")
        print("  ✓ No obvious signs of overfitting")

    # Convert bar-indexed trades to date-indexed trades for bundle import
    test_dates = test_data.get('dates')
    test_trades_dated = []
    if test_dates is not None and 'trades' in test_result:
        for t in test_result['trades']:
            entry_bar = t['entry_bar']
            exit_bar = t['exit_bar']
            if entry_bar < len(test_dates) and exit_bar < len(test_dates):
                test_trades_dated.append({
                    'entry_date': str(test_dates[entry_bar]),
                    'exit_date': str(test_dates[exit_bar]),
                    'entry_price': float(t['entry_price']),
                    'exit_price': float(t['exit_price']),
                    'pnl_pct': float(t['pnl']),
                    'exit_reason': t.get('exit_reason', 'signal')
                })

    # Return results
    return {
        'ticker': ticker,
        'interval': interval,
        'period': period,
        'train_period': f"{train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')}",
        'test_period': f"{test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}",
        'best_params': best_params,
        'train_result': {
            'total_return': best_result['total_return'],
            'win_rate': best_result['win_rate'],
            'n_trades': best_result.get('num_trades', 0),
            'max_drawdown': best_result['max_drawdown'],
            'profit_factor': best_result.get('profit_factor', 0),
            'avg_mae': best_result.get('avg_mae', 0),
            'max_mae': best_result.get('max_mae', 0)
        },
        'test_result': {
            'total_return': test_result['total_return'],
            'win_rate': test_result['win_rate'],
            'n_trades': test_result['n_trades'],
            'max_drawdown': test_result['max_drawdown'],
            'profit_factor': test_result['profit_factor'],
            'avg_mae': test_result.get('avg_mae', 0),
            'max_mae': test_result.get('max_mae', 0)
        },
        'test_trades': test_trades_dated,
        'is_valid': is_valid,
        'warnings': warnings
    }


def update_strategy_config(config_path: str, new_params: dict, validation_results: dict,
                          optimization_info: dict = None):
    """Update strategy config with new optimized parameters.

    Args:
        config_path: Path to the strategy config file
        new_params: Optimized parameters to update
        validation_results: Walk-forward validation results
        optimization_info: Dict with optimization method tracking info:
            - script: The optimization script used
            - n_trials: Number of optimization trials
            - n_jobs: Number of parallel workers
            - metric: Optimization metric used
            - ticker: Ticker symbol
            - interval: Data interval
            - period: Data period
    """

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Update parameters
    params_to_update = [
        'oscillator_type',  # Novel oscillators support
        'signal_type', 'vel_smoothing', 'extreme_zone_mult', 'min_bars_between',
        'require_accel', 'oversold_threshold', 'overbought_threshold',
        'stop_loss_pct', 'take_profit_pct', 'exit_on_opposite_signal', 'exit_on_midline_cross',
        # Acceleration exit params
        'use_accel_exit', 'accel_exit_type', 'accel_exit_threshold',
        'accel_exit_min_pnl', 'accel_exit_lookback', 'use_jerk_confirm', 'jerk_confirm_threshold'
    ]

    for param in params_to_update:
        if param in new_params:
            config[param] = new_params[param]

    # Add validation metadata
    config['wf_validated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config['wf_train_return'] = validation_results['train_result']['total_return']
    config['wf_test_return'] = validation_results['test_result']['total_return']
    config['wf_train_period'] = validation_results['train_period']
    config['wf_test_period'] = validation_results['test_period']

    # Add optimization method tracking
    if optimization_info:
        config['optimization_method'] = {
            'script': optimization_info.get('script', 'velocity_walkforward_validation.py'),
            'n_trials': optimization_info.get('n_trials'),
            'n_jobs': optimization_info.get('n_jobs'),
            'metric': optimization_info.get('metric', 'total_return'),
            'optimized_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    else:
        # Default for walk-forward validation script
        config['optimization_method'] = {
            'script': 'velocity_walkforward_validation.py',
            'optimized_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    # Backup original
    backup_path = config_path + '.before_wf'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            json.dump(json.load(open(config_path)), f, indent=4)

    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\nConfig updated: {config_path}")
    return config


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Validation for Velocity Strategies')
    parser.add_argument('--ticker', type=str, default='BTC-USD', help='Ticker symbol')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval')
    parser.add_argument('--period', type=str, default='2y', help='Data period')
    parser.add_argument('--n-trials', type=int, default=50000, help='Number of optimization trials')
    parser.add_argument('--n-jobs', type=int, default=32, help='Number of parallel workers')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/test split ratio')
    parser.add_argument('--config-path', type=str, help='Path to strategy config to update')
    parser.add_argument('--metric', type=str, default='total_return', help='Optimization metric')
    parser.add_argument('--use-databento', action='store_true',
                        help='Fetch from Databento Historical API (supports >60 days for intraday futures)')
    parser.add_argument('--signal-type', type=str, default=None,
                        help='Force specific signal type (e.g., any_reversal, velocity_crossover_or_zone)')

    args = parser.parse_args()

    # Run walk-forward validation
    results = run_walk_forward_validation(
        ticker=args.ticker,
        interval=args.interval,
        period=args.period,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        train_ratio=args.train_ratio,
        optimize_metric=args.metric,
        use_databento=args.use_databento,
        force_signal_type=args.signal_type
    )

    if results is None:
        print("Walk-forward validation failed!")
        sys.exit(1)

    # Save results with optimization tracking
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"velocity_wf_results_{args.ticker}_{args.period}_{timestamp}.json"

    # Add optimization method tracking to results
    results['optimization_method'] = {
        'script': 'velocity_walkforward_validation.py',
        'n_trials': args.n_trials,
        'n_jobs': args.n_jobs,
        'metric': args.metric,
        'ticker': args.ticker,
        'interval': args.interval,
        'period': args.period,
        'train_ratio': args.train_ratio,
        'optimized_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=str)

    print(f"\nResults saved to: {results_file}")

    # Update config if path provided
    if args.config_path and os.path.exists(args.config_path):
        if results['is_valid']:
            optimization_info = {
                'script': 'velocity_walkforward_validation.py',
                'n_trials': args.n_trials,
                'n_jobs': args.n_jobs,
                'metric': args.metric,
                'ticker': args.ticker,
                'interval': args.interval,
                'period': args.period
            }
            update_strategy_config(args.config_path, results['best_params'], results, optimization_info)
        else:
            print("\nWARNING: Validation failed - config NOT updated")
            print("Review warnings above and consider re-running with different parameters")

    # Print final summary
    print(f"\n{'=' * 80}")
    print("OPTIMIZED PARAMETERS FOR DEPLOYMENT")
    print("=" * 80)

    params = results['best_params']
    print(json.dumps({
        'oscillator_type': params.get('oscillator_type', 'composite'),
        'signal_type': params.get('signal_type'),
        'use_accel_exit': params.get('use_accel_exit'),
        'accel_exit_type': params.get('accel_exit_type'),
        'accel_exit_threshold': params.get('accel_exit_threshold'),
        'accel_exit_min_pnl': params.get('accel_exit_min_pnl'),
        'use_jerk_confirm': params.get('use_jerk_confirm'),
        'stop_loss_pct': params.get('stop_loss_pct'),
        'take_profit_pct': params.get('take_profit_pct'),
        'oversold_threshold': params.get('oversold_threshold'),
        'overbought_threshold': params.get('overbought_threshold')
    }, indent=4))

    return results


if __name__ == '__main__':
    main()
