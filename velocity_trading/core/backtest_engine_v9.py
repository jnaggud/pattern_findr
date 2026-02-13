"""
Regime-Aware Backtest Engine for v9 strategies.

Extends the v8 backtest engine with per-regime parameter switching:
- Each bar is classified into a regime by regime_detector_v9
- Entry decisions use the current regime's signals and params
- Once in a trade, the ENTRY regime's exit params govern until trade closes
- Each regime can set dont_trade=True to skip all entries

Performance: single for-loop over bars (same as v8).
Overhead: one array lookup per bar for regime classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable, Tuple

from velocity_trading.core.backtest_engine import prepare_backtest_arrays, _empty_result
from velocity_trading.core.regime_detector_v9 import (
    classify_regimes, REGIME_NAMES, REGIME_UPTREND, REGIME_DOWNTREND, REGIME_CHOP,
)


def prepare_regime_backtest_arrays(
    df: pd.DataFrame,
    regime_params: dict,
    regime_detector_config: dict,
) -> dict:
    """
    Prepare arrays for regime-aware backtest.

    1. Computes regime classification for every bar
    2. For EACH regime, calls prepare_backtest_arrays() to get signal arrays
    3. Returns combined dict with per-regime signal arrays

    Args:
        df: DataFrame with OHLCV + oscillator columns
        regime_params: Dict mapping regime_id -> config dict.
            E.g. {0: {'signal_type': 'any_reversal', ...}, 1: {...}, 2: {...}}
        regime_detector_config: Parameters for classify_regimes().
            E.g. {'adx_period': 14, 'adx_threshold': 25, ...}

    Returns:
        Dict with:
            'close', 'high', 'low': price arrays (shared)
            'osc_smooth', 'velocity', 'acceleration', 'jerk': shared
            'regimes': int array (0/1/2)
            'is_high_vol': bool array
            'buy_signals_by_regime': {0: array, 1: array, 2: array}
            'sell_signals_by_regime': {0: array, 1: array, 2: array}
            'index': DatetimeIndex
    """
    # Get price columns
    close_col = 'close' if 'close' in df.columns else 'Close'
    high_col = 'high' if 'high' in df.columns else 'High'
    low_col = 'low' if 'low' in df.columns else 'Low'

    high_arr = df[high_col].values.astype(float)
    low_arr = df[low_col].values.astype(float)
    close_arr = df[close_col].values.astype(float)

    # Classify regimes
    regimes, is_high_vol = classify_regimes(
        high_arr, low_arr, close_arr,
        adx_period=regime_detector_config.get('adx_period', 14),
        adx_threshold=regime_detector_config.get('adx_threshold', 25.0),
        atr_period=regime_detector_config.get('atr_period', 14),
        atr_high_vol_percentile=regime_detector_config.get('atr_high_vol_percentile', 90.0),
        atr_high_vol_lookback=regime_detector_config.get('atr_high_vol_lookback', 100),
        regime_min_bars=regime_detector_config.get('regime_min_bars', 4),
    )

    # Prepare signal arrays per regime using v8 engine
    buy_signals_by_regime = {}
    sell_signals_by_regime = {}

    # We need shared oscillator arrays, take them from any regime config
    # (oscillator type is shared across regimes, only signal params differ)
    shared_arrays = None

    for regime_id in [REGIME_UPTREND, REGIME_DOWNTREND, REGIME_CHOP]:
        r_config = regime_params.get(regime_id, regime_params.get(str(regime_id), {}))
        if r_config.get('dont_trade', False):
            n = len(df)
            buy_signals_by_regime[regime_id] = np.zeros(n, dtype=bool)
            sell_signals_by_regime[regime_id] = np.zeros(n, dtype=bool)
            continue

        arrays = prepare_backtest_arrays(df, r_config)
        buy_signals_by_regime[regime_id] = arrays['buy_signals']
        sell_signals_by_regime[regime_id] = arrays['sell_signals']

        if shared_arrays is None:
            shared_arrays = arrays

    # Fallback if all regimes are dont_trade
    if shared_arrays is None:
        # Still need shared arrays for the engine
        dummy_config = list(regime_params.values())[0]
        shared_arrays = prepare_backtest_arrays(df, dummy_config)

    return {
        'close': shared_arrays['close'],
        'high': shared_arrays['high'],
        'low': shared_arrays['low'],
        'osc_smooth': shared_arrays['osc_smooth'],
        'velocity': shared_arrays['velocity'],
        'acceleration': shared_arrays['acceleration'],
        'jerk': shared_arrays['jerk'],
        'regimes': regimes,
        'is_high_vol': is_high_vol,
        'buy_signals_by_regime': buy_signals_by_regime,
        'sell_signals_by_regime': sell_signals_by_regime,
        'index': shared_arrays['index'],
    }


def run_regime_backtest(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    osc_smooth: np.ndarray,
    acceleration: np.ndarray,
    jerk: np.ndarray,
    regimes: np.ndarray,
    is_high_vol: np.ndarray,
    buy_signals_by_regime: dict,
    sell_signals_by_regime: dict,
    regime_params: dict,
    # Optional callables
    entry_filter_fn: Optional[Callable] = None,
    exit_model_fn: Optional[Callable] = None,
    # High-vol behavior
    suppress_entries_high_vol: bool = True,
    # Mode
    return_trades: bool = False,
    # Unused (for **arrays compatibility)
    velocity: np.ndarray = None,
    index: object = None,
) -> dict:
    """
    Regime-aware trade simulation.

    Same core logic as v8 run_backtest() but:
    - Entry: uses current bar's regime to select buy_signals and params
    - Exit: uses entry regime's params for SL/TP/accel/midline/opposite
    - Tracks per-regime trade statistics

    Args:
        close/high/low: Price arrays
        osc_smooth/acceleration/jerk: Indicator arrays
        regimes: int array (0=uptrend, 1=downtrend, 2=chop)
        is_high_vol: bool array for high-vol overlay
        buy_signals_by_regime: {regime_id: bool array}
        sell_signals_by_regime: {regime_id: bool array}
        regime_params: {regime_id: config dict with SL, TP, exit flags, etc.}
        entry_filter_fn: Optional callable(bar_idx, regime_id) -> bool
        exit_model_fn: Optional callable(entry_price, entry_bar, current_bar, hwm) -> (bool, prob, reason)
        suppress_entries_high_vol: If True, skip entries when is_high_vol is True
        return_trades: If True, return full trade list

    Returns:
        Stats dict (same as v8) plus:
        - 'regime_stats': {regime_id: {n_trades, win_rate, total_return, ...}}
        - 'regime_distribution': {regime_id: n_bars}
    """
    n = len(close)
    if n == 0:
        result = _empty_result(return_trades)
        result['regime_stats'] = {}
        result['regime_distribution'] = {}
        return result

    # Helper to get a regime's param with fallback
    def _rp(regime_id, key, default):
        r = regime_params.get(regime_id, regime_params.get(str(regime_id), {}))
        return r.get(key, default)

    # State
    in_position = False
    entry_price = 0.0
    entry_bar = 0
    entry_regime = REGIME_CHOP
    high_watermark = 0.0
    last_exit_bar = -1  # Will use per-regime min_bars_between

    # Tracking
    trades_list = [] if return_trades else None
    pnls = []
    maes = []
    mfes = []
    trade_regimes = []  # Which regime each trade entered under
    min_price_in_trade = 0.0
    max_price_in_trade = 0.0

    # Equity tracking
    equity = 1.0
    peak_equity = 1.0
    max_drawdown = 0.0

    for i in range(n):
        current_regime = regimes[i]

        if not in_position:
            # --- ENTRY ---
            # Check if regime allows trading
            if _rp(current_regime, 'dont_trade', False):
                continue
            if suppress_entries_high_vol and is_high_vol[i]:
                continue

            # Check min_bars_between for current regime
            min_bars_between = _rp(current_regime, 'min_bars_between', 1)
            if (i - last_exit_bar) < min_bars_between:
                continue

            # Check this regime's buy signal
            buy_sigs = buy_signals_by_regime.get(current_regime)
            if buy_sigs is None or not buy_sigs[i]:
                continue

            # Optional novel filter check
            if entry_filter_fn is not None:
                if not entry_filter_fn(i, current_regime):
                    continue

            # ENTER
            in_position = True
            entry_price = close[i]
            entry_bar = i
            entry_regime = current_regime
            high_watermark = close[i]
            min_price_in_trade = low[i]
            max_price_in_trade = high[i]

        else:
            # --- IN POSITION: track MAE/MFE ---
            min_price_in_trade = min(min_price_in_trade, low[i])
            max_price_in_trade = max(max_price_in_trade, high[i])
            high_watermark = max(high_watermark, close[i])

            bars_held = i - entry_bar
            close_pnl = (close[i] - entry_price) / entry_price * 100
            low_pnl = (low[i] - entry_price) / entry_price * 100
            high_pnl = (high[i] - entry_price) / entry_price * 100

            # Use ENTRY regime's exit params
            stop_loss_pct = _rp(entry_regime, 'stop_loss_pct', 5.0)
            take_profit_pct = _rp(entry_regime, 'take_profit_pct', 10.0)
            min_hold_bars = _rp(entry_regime, 'min_hold_bars', 1)
            use_accel_exit = _rp(entry_regime, 'use_accel_exit', False)
            accel_exit_type = _rp(entry_regime, 'accel_exit_type', 'sign_reversal')
            accel_exit_threshold = _rp(entry_regime, 'accel_exit_threshold', 0.0)
            accel_exit_min_pnl = _rp(entry_regime, 'accel_exit_min_pnl', 0.5)
            accel_exit_lookback = _rp(entry_regime, 'accel_exit_lookback', 1)
            use_jerk_confirm = _rp(entry_regime, 'use_jerk_confirm', False)
            jerk_confirm_threshold = _rp(entry_regime, 'jerk_confirm_threshold', 0.0)
            exit_on_midline = _rp(entry_regime, 'exit_on_midline_cross', False)
            exit_on_opposite = _rp(entry_regime, 'exit_on_opposite_signal', True)

            exit_reason = None
            exit_price = close[i]

            # Priority 1: Stop Loss (ignores min_hold_bars)
            if low_pnl <= -stop_loss_pct:
                exit_reason = 'Stop Loss'
                exit_price = entry_price * (1 - stop_loss_pct / 100)

            # Priority 2: Take Profit (ignores min_hold_bars)
            elif high_pnl >= take_profit_pct:
                exit_reason = 'Take Profit'
                exit_price = entry_price * (1 + take_profit_pct / 100)

            # Remaining exits require min_hold_bars
            elif bars_held >= min_hold_bars:

                # Priority 3: Accel Exit
                if use_accel_exit and i >= accel_exit_lookback:
                    pnl_ok = close_pnl >= accel_exit_min_pnl or close_pnl < 0
                    if pnl_ok:
                        accel_cond = False
                        if accel_exit_type == 'sign_reversal':
                            accel_vals = acceleration[i - accel_exit_lookback + 1:i + 1]
                            accel_cond = np.all(accel_vals < 0)
                        elif accel_exit_type == 'magnitude':
                            accel_cond = acceleration[i] < -accel_exit_threshold
                        elif accel_exit_type == 'both':
                            accel_vals = acceleration[i - accel_exit_lookback + 1:i + 1]
                            accel_cond = np.all(accel_vals < 0) and abs(acceleration[i]) > accel_exit_threshold

                        jerk_ok = True
                        if use_jerk_confirm and jerk_confirm_threshold > 0:
                            jerk_ok = jerk[i] < -jerk_confirm_threshold

                        if accel_cond and jerk_ok:
                            exit_reason = 'Accel Exit'

                # Priority 4: ML Exit
                if not exit_reason and exit_model_fn is not None:
                    try:
                        should_exit, prob, reason = exit_model_fn(
                            entry_price, entry_bar, i, high_watermark
                        )
                        if should_exit:
                            exit_reason = f'ML Exit ({close_pnl:.2f}%, p={prob:.2f})'
                    except Exception:
                        pass

                # Priority 5: Midline Cross
                if not exit_reason and exit_on_midline:
                    if osc_smooth[i] > 0:
                        exit_reason = 'Midline Cross'

                # Priority 6: Opposite Signal (from entry regime's sell signals)
                if not exit_reason and exit_on_opposite:
                    sell_sigs = sell_signals_by_regime.get(entry_regime)
                    if sell_sigs is not None and sell_sigs[i]:
                        exit_reason = 'Opposite Signal'

            # --- EXECUTE EXIT ---
            if exit_reason:
                pnl = (exit_price - entry_price) / entry_price * 100
                mae = (entry_price - min_price_in_trade) / entry_price * 100
                mfe = (max_price_in_trade - entry_price) / entry_price * 100

                pnls.append(pnl)
                maes.append(mae)
                mfes.append(mfe)
                trade_regimes.append(entry_regime)

                # Update equity and drawdown
                equity *= (1 + pnl / 100)
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity * 100
                max_drawdown = max(max_drawdown, dd)

                if return_trades:
                    trades_list.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'mae': mae,
                        'mfe': mfe,
                        'bars_held': bars_held,
                        'entry_regime': REGIME_NAMES.get(entry_regime, str(entry_regime)),
                    })

                in_position = False
                last_exit_bar = i

    # --- Build result ---
    n_trades = len(pnls)
    if n_trades == 0:
        result = _empty_result(return_trades)
        result['regime_stats'] = _build_regime_stats([], [], [], [])
        result['regime_distribution'] = _build_regime_distribution(regimes)
        return result

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001

    total_return = (equity - 1) * 100

    result = {
        'total_return': total_return,
        'n_trades': n_trades,
        'win_rate': len(wins) / n_trades * 100,
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
        'max_drawdown': max_drawdown,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else (999.99 if gross_profit > 0 else 0),
        'avg_mae': np.mean(maes) if maes else 0,
        'max_mae': max(maes) if maes else 0,
        'avg_mfe': np.mean(mfes) if mfes else 0,
        'regime_stats': _build_regime_stats(pnls, trade_regimes, maes, mfes),
        'regime_distribution': _build_regime_distribution(regimes),
    }

    if return_trades:
        result['trades'] = trades_list
        if in_position:
            result['open_position'] = {
                'entry_bar': entry_bar,
                'entry_price': entry_price,
                'entry_regime': REGIME_NAMES.get(entry_regime, str(entry_regime)),
            }
        else:
            result['open_position'] = None

    return result


def _build_regime_stats(
    pnls: list,
    trade_regimes: list,
    maes: list,
    mfes: list,
) -> dict:
    """Build per-regime trade statistics."""
    stats = {}
    for regime_id, name in REGIME_NAMES.items():
        indices = [j for j, r in enumerate(trade_regimes) if r == regime_id]
        r_pnls = [pnls[j] for j in indices]
        r_maes = [maes[j] for j in indices]

        n = len(r_pnls)
        if n == 0:
            stats[name] = {
                'n_trades': 0, 'win_rate': 0.0, 'total_return': 0.0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'max_drawdown': 0.0,
                'profit_factor': 0.0, 'avg_mae': 0.0,
            }
            continue

        wins = [p for p in r_pnls if p > 0]
        losses = [p for p in r_pnls if p <= 0]
        gp = sum(wins) if wins else 0
        gl = abs(sum(losses)) if losses else 0.001

        # Compute equity curve for this regime's trades
        eq = 1.0
        peak = 1.0
        mdd = 0.0
        for p in r_pnls:
            eq *= (1 + p / 100)
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100
            mdd = max(mdd, dd)

        stats[name] = {
            'n_trades': n,
            'win_rate': len(wins) / n * 100,
            'total_return': (eq - 1) * 100,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'max_drawdown': mdd,
            'profit_factor': gp / gl if gl > 0 else (999.99 if gp > 0 else 0),
            'avg_mae': np.mean(r_maes) if r_maes else 0,
        }
    return stats


def _build_regime_distribution(regimes: np.ndarray) -> dict:
    """Build bar count distribution for regimes."""
    dist = {}
    n = len(regimes)
    for regime_id, name in REGIME_NAMES.items():
        count = int(np.sum(regimes == regime_id))
        dist[name] = {'count': count, 'pct': round(count / n * 100, 1) if n > 0 else 0.0}
    return dist
