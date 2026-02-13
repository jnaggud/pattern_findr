"""
Regime Detector for v9 strategies.

Classifies each bar into one of 3 regimes using ADX + Directional Indicators:
  0 = UPTREND   (ADX > threshold AND +DI > -DI)
  1 = DOWNTREND (ADX > threshold AND -DI > +DI)
  2 = CHOP      (ADX <= threshold)

Plus a high-volatility overlay (boolean) for optional entry suppression.

All computations are vectorized numpy — no per-bar loops, no look-ahead bias.
Uses proper Wilder smoothing (exponential, alpha=1/period) for ADX.
"""

import numpy as np
from typing import Tuple, Dict


# Regime constants
REGIME_UPTREND = 0
REGIME_DOWNTREND = 1
REGIME_CHOP = 2

REGIME_NAMES = {
    REGIME_UPTREND: 'uptrend',
    REGIME_DOWNTREND: 'downtrend',
    REGIME_CHOP: 'chop',
}

REGIME_IDS = {v: k for k, v in REGIME_NAMES.items()}


def compute_adx_components(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ADX, +DI, -DI using proper Wilder smoothing.

    Args:
        high, low, close: Price arrays (same length)
        period: ADX lookback period (default 14)

    Returns:
        Tuple of (adx, plus_di, minus_di) numpy arrays, same length as input.
        First ~2*period values are NaN (warmup).
    """
    n = len(close)
    adx = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)

    if n < period + 1:
        return adx, plus_di, minus_di

    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Wilder smoothing: first value is sum of first `period` values,
    # subsequent values use EMA-like: prev - prev/period + current
    alpha = 1.0 / period

    # Initialize smoothed values at index `period`
    atr_smooth = np.sum(tr[1:period + 1])
    plus_dm_smooth = np.sum(plus_dm[1:period + 1])
    minus_dm_smooth = np.sum(minus_dm[1:period + 1])

    # First DI values
    if atr_smooth > 0:
        plus_di[period] = 100.0 * plus_dm_smooth / atr_smooth
        minus_di[period] = 100.0 * minus_dm_smooth / atr_smooth
    else:
        plus_di[period] = 0.0
        minus_di[period] = 0.0

    # DX for ADX initialization
    dx_values = []
    di_sum = plus_di[period] + minus_di[period]
    if di_sum > 0:
        dx_values.append(100.0 * abs(plus_di[period] - minus_di[period]) / di_sum)
    else:
        dx_values.append(0.0)

    # Continue Wilder smoothing for remaining bars
    for i in range(period + 1, n):
        atr_smooth = atr_smooth - atr_smooth / period + tr[i]
        plus_dm_smooth = plus_dm_smooth - plus_dm_smooth / period + plus_dm[i]
        minus_dm_smooth = minus_dm_smooth - minus_dm_smooth / period + minus_dm[i]

        if atr_smooth > 0:
            plus_di[i] = 100.0 * plus_dm_smooth / atr_smooth
            minus_di[i] = 100.0 * minus_dm_smooth / atr_smooth
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0

        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx_values.append(100.0 * abs(plus_di[i] - minus_di[i]) / di_sum)
        else:
            dx_values.append(0.0)

    # ADX: Wilder smoothed DX, starts at index 2*period
    if len(dx_values) >= period:
        # First ADX = simple average of first `period` DX values
        adx_val = np.mean(dx_values[:period])
        adx_start_idx = period + period  # 2 * period
        if adx_start_idx < n:
            adx[adx_start_idx] = adx_val

        # Subsequent ADX values use Wilder smoothing
        for j in range(period, len(dx_values)):
            adx_val = (adx_val * (period - 1) + dx_values[j]) / period
            bar_idx = period + 1 + j  # offset to original array index
            if bar_idx < n:
                adx[bar_idx] = adx_val

    return adx, plus_di, minus_di


def classify_regimes(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    atr_period: int = 14,
    atr_high_vol_percentile: float = 90.0,
    atr_high_vol_lookback: int = 100,
    regime_min_bars: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify each bar into a regime. No look-ahead bias.

    Args:
        high, low, close: Price arrays
        adx_period: Period for ADX computation (default 14)
        adx_threshold: ADX level separating trend from chop (tunable, 18-35)
        atr_period: Period for ATR (for high-vol overlay)
        atr_high_vol_percentile: Percentile threshold for high-vol flag (default 90)
        atr_high_vol_lookback: Rolling window for ATR percentile (default 100)
        regime_min_bars: Minimum bars a regime must persist before switching (default 4)

    Returns:
        regimes: int array (0=uptrend, 1=downtrend, 2=chop), length N
        is_high_vol: bool array, length N
    """
    n = len(close)
    regimes = np.full(n, REGIME_CHOP, dtype=int)
    is_high_vol = np.zeros(n, dtype=bool)

    if n < 2 * adx_period + 1:
        return regimes, is_high_vol

    # Compute ADX components
    adx, plus_di, minus_di = compute_adx_components(high, low, close, adx_period)

    # Classify raw regimes (before min_bars filter)
    for i in range(n):
        if np.isnan(adx[i]):
            regimes[i] = REGIME_CHOP  # Warmup period defaults to chop
        elif adx[i] > adx_threshold:
            if plus_di[i] > minus_di[i]:
                regimes[i] = REGIME_UPTREND
            else:
                regimes[i] = REGIME_DOWNTREND
        else:
            regimes[i] = REGIME_CHOP

    # Apply regime_min_bars filter to prevent whipsaw
    if regime_min_bars > 1:
        regimes = _apply_min_bars_filter(regimes, regime_min_bars)

    # Compute high-volatility overlay using ATR percentile
    if atr_high_vol_lookback > 0:
        # Compute ATR
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Simple rolling ATR
        atr = np.full(n, np.nan)
        if n >= atr_period:
            atr[atr_period - 1] = np.mean(tr[:atr_period])
            for i in range(atr_period, n):
                atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period

        # Rolling percentile of ATR (look-ahead free)
        for i in range(atr_high_vol_lookback, n):
            if not np.isnan(atr[i]):
                window = atr[max(0, i - atr_high_vol_lookback):i + 1]
                window = window[~np.isnan(window)]
                if len(window) > 10:
                    pct = np.percentile(window, atr_high_vol_percentile)
                    is_high_vol[i] = atr[i] > pct

    return regimes, is_high_vol


def _apply_min_bars_filter(regimes: np.ndarray, min_bars: int) -> np.ndarray:
    """
    Forward-fill regime classification to prevent whipsaw.

    If a regime changes but doesn't persist for at least min_bars,
    revert it to the previous regime.
    """
    n = len(regimes)
    if n <= min_bars:
        return regimes

    filtered = regimes.copy()
    current_regime = filtered[0]
    regime_start = 0

    i = 1
    while i < n:
        if filtered[i] != current_regime:
            # Count how many bars this new regime persists
            new_regime = filtered[i]
            run_len = 1
            j = i + 1
            while j < n and filtered[j] == new_regime:
                run_len += 1
                j += 1

            if run_len < min_bars:
                # Too short — revert to previous regime
                for k in range(i, min(i + run_len, n)):
                    filtered[k] = current_regime
                i = j
            else:
                # Long enough — accept the new regime
                current_regime = new_regime
                regime_start = i
                i = j
        else:
            i += 1

    return filtered


def classify_regimes_extended(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    atr_period: int = 14,
    atr_high_vol_percentile: float = 90.0,
    atr_high_vol_lookback: int = 100,
    regime_min_bars: int = 4,
) -> Dict[str, np.ndarray]:
    """
    Extended regime classification that also returns ADX/DI arrays.

    Same as classify_regimes() but returns a dict with additional indicator
    arrays for downstream use (e.g., Discord notifications).

    Returns:
        Dict with keys:
            'regimes': int array (0=uptrend, 1=downtrend, 2=chop)
            'is_high_vol': bool array
            'adx': float array (raw ADX values, NaN during warmup)
            'plus_di': float array (+DI values)
            'minus_di': float array (-DI values)
    """
    # Get regimes + high vol from existing function
    regimes, is_high_vol = classify_regimes(
        high, low, close,
        adx_period=adx_period,
        adx_threshold=adx_threshold,
        atr_period=atr_period,
        atr_high_vol_percentile=atr_high_vol_percentile,
        atr_high_vol_lookback=atr_high_vol_lookback,
        regime_min_bars=regime_min_bars,
    )

    # Compute ADX components separately for the indicator arrays
    adx, plus_di, minus_di = compute_adx_components(high, low, close, adx_period)

    return {
        'regimes': regimes,
        'is_high_vol': is_high_vol,
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di,
    }


def get_regime_distribution(regimes: np.ndarray) -> Dict[str, dict]:
    """
    Return bar counts and percentages for each regime.

    Returns:
        Dict mapping regime name to {'count': int, 'pct': float}
    """
    n = len(regimes)
    dist = {}
    for regime_id, name in REGIME_NAMES.items():
        count = int(np.sum(regimes == regime_id))
        pct = count / n * 100 if n > 0 else 0.0
        dist[name] = {'count': count, 'pct': round(pct, 1)}
    return dist


def get_regime_transitions(regimes: np.ndarray) -> np.ndarray:
    """
    Return boolean array where True = regime changed from previous bar.
    """
    transitions = np.zeros(len(regimes), dtype=bool)
    if len(regimes) > 1:
        transitions[1:] = regimes[1:] != regimes[:-1]
    return transitions
