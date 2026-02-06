"""
Novel Indicators and Composite Oscillators
==========================================
Advanced indicator compositions beyond simple averaging.

Composite Oscillators:
- ARWO: Adaptive Regime-Weighted Oscillator
- DCO: Divergence Consensus Oscillator
- VCMO: Volume-Confirmed Momentum Oscillator

Indicator Formulas:
- ICS: Indicator Convergence Score
- MJI: Momentum Jerk Indicator
- PRF: Percentile Rank Fusion
- EWAF: Entropy-Weighted Adaptive Fusion
- KFIF: Kalman-Filtered Indicator Fusion
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import pykalman for KFIF
try:
    from pykalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    # Only warn once (not in every worker process)
    import warnings
    warnings.warn("pykalman not installed. KFIF will use fallback implementation.", ImportWarning, stacklevel=1)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_series(series: pd.Series, method: str = 'minmax', window: int = 100) -> pd.Series:
    """Normalize a series to -1 to +1 range."""
    if method == 'minmax':
        rolling_min = series.rolling(window=window, min_periods=1).min()
        rolling_max = series.rolling(window=window, min_periods=1).max()
        range_val = rolling_max - rolling_min
        range_val = range_val.replace(0, 1)  # Avoid division by zero
        normalized = 2 * (series - rolling_min) / range_val - 1
    elif method == 'zscore':
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std().replace(0, 1)
        normalized = (series - rolling_mean) / rolling_std
        normalized = normalized.clip(-3, 3) / 3  # Clip to 3 std devs, scale to -1,1
    else:
        normalized = series
    return normalized.clip(-1, 1)


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ADX (Average Directional Index) - trend strength 0-100."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    return adx.fillna(0)


def linear_regression_slope(series: pd.Series, period: int) -> pd.Series:
    """Calculate rolling linear regression slope."""
    def calc_slope(x):
        if len(x) < 2 or x.isna().any():
            return 0
        y = np.arange(len(x))
        try:
            slope, _, _, _, _ = stats.linregress(y, x)
            return slope
        except:
            return 0
    return series.rolling(window=period).apply(calc_slope, raw=False)


# ============================================================================
# BASIC INDICATOR CALCULATIONS (for internal use)
# ============================================================================

def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI (0-100)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _calc_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Stochastic %K (0-100)."""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    return 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)


def _calc_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Williams %R (-100 to 0)."""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)


def _calc_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate CCI."""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (typical_price - sma) / (0.015 * mad + 1e-10)


def _calc_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Calculate MFI (0-100)."""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    money_flow_positive = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    money_flow_negative = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_sum = money_flow_positive.rolling(window=period).sum()
    negative_sum = money_flow_negative.rolling(window=period).sum()

    return 100 - (100 / (1 + positive_sum / (negative_sum + 1e-10)))


def _calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD line and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, histogram


def _calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume."""
    direction = np.sign(close.diff())
    return (volume * direction).cumsum()


def _calc_roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Calculate Rate of Change."""
    return ((close - close.shift(period)) / close.shift(period)) * 100


def _calc_aroon_oscillator(high: pd.Series, low: pd.Series, period: int = 25) -> pd.Series:
    """Calculate Aroon Oscillator (-100 to +100)."""
    aroon_up = 100 * high.rolling(window=period + 1).apply(lambda x: x.argmax()) / period
    aroon_down = 100 * low.rolling(window=period + 1).apply(lambda x: x.argmin()) / period
    return aroon_up - aroon_down


# ============================================================================
# COMPOSITE OSCILLATOR 1: ARWO (Adaptive Regime-Weighted Oscillator)
# ============================================================================

def calculate_arwo(df: pd.DataFrame, adx_period: int = 14, smooth: int = 3) -> pd.Series:
    """
    Adaptive Regime-Weighted Oscillator (ARWO)

    Dynamically weights momentum vs mean-reversion indicators based on market regime.
    - High ADX (trending): Favor momentum indicators (ROC, MACD, Aroon)
    - Low ADX (ranging): Favor mean-reversion indicators (RSI, Stochastic, Williams %R)

    Returns: Series normalized to -1 to +1
    """
    high, low, close = df['high'], df['low'], df['close']

    # Calculate regime indicator (ADX)
    adx = calculate_adx(high, low, close, adx_period)
    regime_score = (adx / 100).clip(0, 1)  # 0 = ranging, 1 = trending

    # Momentum indicators (good in trends)
    roc_10 = _calc_roc(close, 10)
    roc_norm = normalize_series(roc_10, 'minmax', 50)

    macd_line, macd_hist = _calc_macd(close)
    macd_norm = normalize_series(macd_hist, 'minmax', 50)

    aroon_osc = _calc_aroon_oscillator(high, low, 25)
    aroon_norm = aroon_osc / 100  # Already -1 to +1

    trend_component = (roc_norm + macd_norm + aroon_norm) / 3

    # Mean-reversion indicators (good in ranges)
    rsi = _calc_rsi(close, 14)
    rsi_norm = (rsi - 50) / 50

    stoch = _calc_stochastic(high, low, close, 14)
    stoch_norm = (stoch - 50) / 50

    willr = _calc_williams_r(high, low, close, 14)
    willr_norm = (willr + 50) / 50

    reversion_component = (rsi_norm + stoch_norm + willr_norm) / 3

    # Adaptive weighting
    arwo = (regime_score * trend_component) + ((1 - regime_score) * reversion_component)

    # Smooth
    if smooth > 1:
        arwo = arwo.rolling(window=smooth, center=False).mean().fillna(arwo)

    return arwo.clip(-1, 1)


# ============================================================================
# COMPOSITE OSCILLATOR 2: DCO (Divergence Consensus Oscillator)
# ============================================================================

def calculate_dco(df: pd.DataFrame, lookback: int = 14, smooth: int = 3) -> pd.Series:
    """
    Divergence Consensus Oscillator (DCO)

    Measures how many indicators are diverging from price simultaneously.
    Positive values = bearish divergence (price up, indicators down)
    Negative values = bullish divergence (price down, indicators up)

    Returns: Series normalized to -1 to +1
    """
    high, low, close = df['high'], df['low'], df['close']
    volume = df.get('volume', pd.Series(1, index=df.index))

    # Calculate price slope
    price_slope = linear_regression_slope(close, lookback)
    price_slope_norm = normalize_series(price_slope, 'zscore', 50)

    # Indicators to check for divergence
    indicators = {}
    indicators['rsi'] = _calc_rsi(close, 14)
    indicators['stoch'] = _calc_stochastic(high, low, close, 14)
    indicators['cci'] = _calc_cci(high, low, close, 20)
    indicators['mfi'] = _calc_mfi(high, low, close, volume, 14)

    macd_line, macd_hist = _calc_macd(close)
    indicators['macd'] = macd_line

    obv = _calc_obv(close, volume)
    indicators['obv'] = obv

    # Calculate divergence for each indicator
    divergence_scores = []

    for name, ind in indicators.items():
        ind_slope = linear_regression_slope(ind, lookback)
        ind_slope_norm = normalize_series(ind_slope, 'zscore', 50)

        # Divergence score: negative when indicator diverges bearishly from price
        # Price going up + indicator going down = positive (bearish divergence)
        # Price going down + indicator going up = negative (bullish divergence)
        div_score = price_slope_norm - ind_slope_norm

        # Weight by the strength of the divergence
        div_strength = abs(div_score)

        divergence_scores.append(div_score * (1 + div_strength))

    # Combine divergences
    dco = pd.concat(divergence_scores, axis=1).mean(axis=1)

    # Normalize
    dco = normalize_series(dco, 'minmax', 50)

    # Smooth
    if smooth > 1:
        dco = dco.rolling(window=smooth, center=False).mean().fillna(dco)

    return dco.clip(-1, 1)


# ============================================================================
# COMPOSITE OSCILLATOR 3: VCMO (Volume-Confirmed Momentum Oscillator)
# ============================================================================

def calculate_vcmo(df: pd.DataFrame, mom_period: int = 10, vol_period: int = 20, smooth: int = 3) -> pd.Series:
    """
    Volume-Confirmed Momentum Oscillator (VCMO)

    Only trusts momentum signals when volume confirms the move.
    Uses volume conviction as a multiplier for momentum signals.

    Returns: Series normalized to -1 to +1
    """
    close = df['close']
    volume = df.get('volume', pd.Series(1, index=df.index))

    # Raw momentum composite
    roc_10 = _calc_roc(close, mom_period)
    roc_20 = _calc_roc(close, mom_period * 2)
    momentum = close - close.shift(mom_period)

    roc_norm = normalize_series(roc_10, 'minmax', 50)
    roc_20_norm = normalize_series(roc_20, 'minmax', 50)
    mom_norm = normalize_series(momentum, 'minmax', 50)

    raw_momentum = (roc_norm + roc_20_norm + mom_norm) / 3

    # Volume conviction (z-score of volume)
    vol_mean = volume.rolling(window=vol_period).mean()
    vol_std = volume.rolling(window=vol_period).std().replace(0, 1)
    volume_zscore = (volume - vol_mean) / vol_std

    # OBV trend direction
    obv = _calc_obv(close, volume)
    obv_slope = linear_regression_slope(obv, vol_period)
    obv_direction = np.sign(obv_slope)

    # Accumulation/Distribution Line slope
    clv = ((close - df['low']) - (df['high'] - close)) / (df['high'] - df['low'] + 1e-10)
    ad_line = (clv * volume).cumsum()
    ad_slope = linear_regression_slope(ad_line, vol_period)
    ad_direction = np.sign(ad_slope)

    # Volume trend agreement (do OBV and A/D agree?)
    volume_trend = obv_direction * ad_direction  # +1 if agree, -1 if disagree

    # Volume conviction: high volume + agreement = high conviction
    volume_conviction = volume_zscore * volume_trend

    # Amplify momentum when volume confirms, dampen when it doesn't
    # tanh keeps the multiplier bounded
    vcmo = raw_momentum * (1 + np.tanh(volume_conviction * 0.5))

    # Normalize to -1 to +1
    vcmo = normalize_series(vcmo, 'minmax', 50)

    # Smooth
    if smooth > 1:
        vcmo = vcmo.rolling(window=smooth, center=False).mean().fillna(vcmo)

    return vcmo.clip(-1, 1)


# ============================================================================
# INDICATOR FORMULA 1: ICS (Indicator Convergence Score)
# ============================================================================

def calculate_ics(df: pd.DataFrame, smooth: int = 3) -> pd.Series:
    """
    Indicator Convergence Score (ICS)

    Measures statistical dispersion of normalized indicators.
    Strong signals only when indicators converge AND agree on direction.

    Returns: Series normalized to -1 to +1
    """
    high, low, close = df['high'], df['low'], df['close']
    volume = df.get('volume', pd.Series(1, index=df.index))

    # Normalize all indicators to -1 to +1
    indicators = pd.DataFrame(index=df.index)

    rsi = _calc_rsi(close, 14)
    indicators['rsi'] = (rsi - 50) / 50

    stoch = _calc_stochastic(high, low, close, 14)
    indicators['stoch'] = (stoch - 50) / 50

    willr = _calc_williams_r(high, low, close, 14)
    indicators['willr'] = (willr + 50) / 50

    cci = _calc_cci(high, low, close, 20)
    indicators['cci'] = (cci / 200).clip(-1, 1)

    mfi = _calc_mfi(high, low, close, volume, 14)
    indicators['mfi'] = (mfi - 50) / 50

    # Calculate dispersion (how much do they disagree?)
    dispersion = indicators.std(axis=1)

    # Calculate direction (what direction do they point?)
    direction = indicators.mean(axis=1)

    # Calculate agreement (what fraction agree with the mean direction?)
    direction_sign = np.sign(direction)
    # For each row, count how many indicators have the same sign as the mean
    indicator_signs = np.sign(indicators)
    agreement = (indicator_signs.T == direction_sign.values).T.mean(axis=1)

    # ICS: Strong direction * high agreement * low dispersion
    ics = direction * agreement * (1 / (1 + dispersion * 2))

    # Smooth
    if smooth > 1:
        ics = ics.rolling(window=smooth, center=False).mean().fillna(ics)

    return ics.clip(-1, 1)


# ============================================================================
# INDICATOR FORMULA 2: MJI (Momentum Jerk Indicator)
# ============================================================================

def calculate_mji(df: pd.DataFrame, mom_period: int = 10, smooth_period: int = 5) -> pd.Series:
    """
    Momentum Jerk Indicator (MJI)

    Second derivative of momentum - detects when momentum is about to change direction.
    - Positive jerk while momentum is negative = potential bottom
    - Negative jerk while momentum is positive = potential top

    Returns: Series normalized to -1 to +1
    """
    close = df['close']
    atr = calculate_atr(df['high'], df['low'], close, 14)

    # Calculate momentum
    momentum = close - close.shift(mom_period)

    # First derivative (velocity/acceleration)
    velocity = momentum.diff()

    # Second derivative (jerk)
    jerk = velocity.diff()

    # Smooth the jerk
    jerk_smooth = jerk.ewm(span=smooth_period, adjust=False).mean()

    # Normalize by ATR to make it comparable across different price levels
    mji = jerk_smooth / (atr + 1e-10)

    # Normalize to -1 to +1
    mji = normalize_series(mji, 'minmax', 50)

    return mji.clip(-1, 1)


# ============================================================================
# INDICATOR FORMULA 3: PRF (Percentile Rank Fusion)
# ============================================================================

def calculate_prf(df: pd.DataFrame, lookback: int = 100, smooth: int = 3) -> pd.Series:
    """
    Percentile Rank Fusion (PRF)

    Converts each indicator to its historical percentile rank, then combines.
    Captures "how extreme is this reading historically?"

    Returns: Series normalized to -1 to +1
    """
    high, low, close = df['high'], df['low'], df['close']
    volume = df.get('volume', pd.Series(1, index=df.index))

    # Adaptive lookback - scale based on available data
    # Use 40% of data length, capped at requested lookback
    effective_lookback = min(lookback, int(len(df) * 0.4))
    effective_lookback = max(30, effective_lookback)  # Minimum 30 bars

    def percentile_rank(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling percentile rank (0-100)."""
        # Fill NaNs with median for calculation purposes
        series_filled = series.fillna(series.median())

        def calc_pct(x):
            x_arr = np.array(x)
            if len(x_arr) < 2:
                return 50.0
            # Use all but last value to rank the last value
            historical = x_arr[:-1]
            current = x_arr[-1]
            return stats.percentileofscore(historical, current, kind='rank')

        # Use min_periods that's 50% of window for earlier results
        return series_filled.rolling(window=window, min_periods=max(15, window // 2)).apply(calc_pct, raw=True)

    # Calculate percentile ranks for each indicator
    ranks = pd.DataFrame(index=df.index)

    ranks['rsi'] = percentile_rank(_calc_rsi(close, 14), effective_lookback)
    ranks['stoch'] = percentile_rank(_calc_stochastic(high, low, close, 14), effective_lookback)
    ranks['cci'] = percentile_rank(_calc_cci(high, low, close, 20), effective_lookback)
    ranks['willr'] = percentile_rank(_calc_williams_r(high, low, close, 14), effective_lookback)
    ranks['mfi'] = percentile_rank(_calc_mfi(high, low, close, volume, 14), effective_lookback)

    # Average percentile rank (only need 3+ valid to compute)
    prf = ranks.apply(lambda row: row.dropna().mean() if row.notna().sum() >= 3 else np.nan, axis=1)

    # Transform to oscillator scale (-1 to +1)
    # 0 percentile -> -1, 50 percentile -> 0, 100 percentile -> +1
    prf_oscillator = (prf - 50) / 50

    # Smooth
    if smooth > 1:
        prf_oscillator = prf_oscillator.rolling(window=smooth, center=False, min_periods=1).mean()

    return prf_oscillator.clip(-1, 1)


# ============================================================================
# INDICATOR FORMULA 4: EWAF (Entropy-Weighted Adaptive Fusion)
# ============================================================================

def calculate_ewaf(df: pd.DataFrame, entropy_window: int = 20, n_bins: int = 10, smooth: int = 3) -> pd.Series:
    """
    Entropy-Weighted Adaptive Fusion (EWAF)

    Weights indicators by their recent information content (Shannon entropy).
    Low entropy = indicator at extremes = higher weight (making a statement).
    High entropy = indicator wandering = lower weight (noise).

    Returns: Series normalized to -1 to +1
    """
    high, low, close = df['high'], df['low'], df['close']
    volume = df.get('volume', pd.Series(1, index=df.index))

    def rolling_entropy(series: pd.Series, window: int, bins: int) -> pd.Series:
        """Calculate rolling Shannon entropy. Lower = more concentrated."""
        def calc_entropy(x):
            if len(x) < bins:
                return 1.0  # High entropy (uncertain) for short windows
            try:
                hist, _ = np.histogram(x, bins=bins, density=True)
                hist = hist[hist > 0]
                if len(hist) == 0:
                    return 1.0
                # Normalize entropy to 0-1 range (divide by max possible entropy)
                max_entropy = np.log(bins)
                return -np.sum(hist * np.log(hist + 1e-10)) / max_entropy
            except:
                return 1.0
        return series.rolling(window=window, min_periods=min(10, window)).apply(calc_entropy, raw=False)

    # Calculate normalized indicators
    indicators = {}
    indicators['rsi'] = (_calc_rsi(close, 14) - 50) / 50
    indicators['stoch'] = (_calc_stochastic(high, low, close, 14) - 50) / 50
    indicators['willr'] = (_calc_williams_r(high, low, close, 14) + 50) / 50
    indicators['cci'] = (_calc_cci(high, low, close, 20) / 200).clip(-1, 1)
    indicators['mfi'] = (_calc_mfi(high, low, close, volume, 14) - 50) / 50

    # Calculate entropies
    entropies = {}
    for name, ind in indicators.items():
        entropies[name] = rolling_entropy(ind, entropy_window, n_bins)

    # Inverse entropy weighting (low entropy = high weight)
    ewaf = pd.Series(0.0, index=df.index)
    weight_sum = pd.Series(0.0, index=df.index)

    for name in indicators.keys():
        weight = 1 / (entropies[name] + 0.1)  # Add small constant to avoid division by zero
        ewaf += indicators[name] * weight
        weight_sum += weight

    ewaf = ewaf / weight_sum

    # Smooth
    if smooth > 1:
        ewaf = ewaf.rolling(window=smooth, center=False).mean().fillna(ewaf)

    return ewaf.clip(-1, 1)


# ============================================================================
# INDICATOR FORMULA 5: KFIF (Kalman-Filtered Indicator Fusion)
# ============================================================================

def calculate_kfif(df: pd.DataFrame, process_noise: float = 0.01,
                   observation_noise: float = 0.5, smooth: int = 1) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Kalman-Filtered Indicator Fusion (KFIF)

    Uses a Kalman filter to optimally combine noisy indicator readings.
    Provides smoothed estimate with uncertainty bounds.

    Returns: Tuple of (KFIF value, upper_band, lower_band) all normalized to -1 to +1
    """
    high, low, close = df['high'], df['low'], df['close']
    volume = df.get('volume', pd.Series(1, index=df.index))

    # Prepare normalized indicator observations
    observations = pd.DataFrame(index=df.index)
    observations['rsi'] = (_calc_rsi(close, 14) - 50) / 50
    observations['stoch'] = (_calc_stochastic(high, low, close, 14) - 50) / 50
    observations['willr'] = (_calc_williams_r(high, low, close, 14) + 50) / 50
    observations['cci'] = (_calc_cci(high, low, close, 20) / 200).clip(-1, 1)
    observations['mfi'] = (_calc_mfi(high, low, close, volume, 14) - 50) / 50

    # Fill NaN values for Kalman filter
    observations = observations.fillna(0)
    obs_array = observations.values

    n_obs = obs_array.shape[1]

    if KALMAN_AVAILABLE:
        try:
            # Use pykalman for optimal filtering
            kf = KalmanFilter(
                n_dim_state=1,
                n_dim_obs=n_obs,
                initial_state_mean=0,
                initial_state_covariance=1,
                transition_matrices=np.array([[1]]),
                observation_matrices=np.ones((n_obs, 1)),
                transition_covariance=np.array([[process_noise]]),
                observation_covariance=np.eye(n_obs) * observation_noise
            )

            state_means, state_covs = kf.filter(obs_array)

            kfif = pd.Series(state_means.flatten(), index=df.index)
            kfif_std = pd.Series(np.sqrt(state_covs.flatten()), index=df.index)

        except Exception as e:
            print(f"Kalman filter error: {e}, using fallback")
            kfif = observations.mean(axis=1)
            kfif_std = observations.std(axis=1)
    else:
        # Fallback: exponentially weighted mean with rolling std
        kfif = observations.ewm(span=10, adjust=False).mean().mean(axis=1)
        kfif_std = observations.rolling(window=20).std().mean(axis=1).fillna(0.1)

    # Calculate bands
    upper_band = (kfif + 2 * kfif_std).clip(-1, 1)
    lower_band = (kfif - 2 * kfif_std).clip(-1, 1)
    kfif = kfif.clip(-1, 1)

    return kfif, upper_band, lower_band


# ============================================================================
# MASTER FUNCTION: Calculate All Novel Indicators
# ============================================================================

def calculate_all_novel_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all novel indicators and add them to the dataframe.

    Adds columns:
    - arwo: Adaptive Regime-Weighted Oscillator
    - dco: Divergence Consensus Oscillator
    - vcmo: Volume-Confirmed Momentum Oscillator
    - ics: Indicator Convergence Score
    - mji: Momentum Jerk Indicator
    - prf: Percentile Rank Fusion
    - ewaf: Entropy-Weighted Adaptive Fusion
    - kfif: Kalman-Filtered Indicator Fusion
    - kfif_upper: KFIF upper confidence band
    - kfif_lower: KFIF lower confidence band
    """
    result = df.copy()

    # Ensure lowercase columns
    result.columns = result.columns.str.lower()

    print("Calculating novel indicators...")

    # Composite Oscillators
    print("  - ARWO (Adaptive Regime-Weighted)...")
    result['arwo'] = calculate_arwo(result)

    print("  - DCO (Divergence Consensus)...")
    result['dco'] = calculate_dco(result)

    print("  - VCMO (Volume-Confirmed Momentum)...")
    result['vcmo'] = calculate_vcmo(result)

    # Indicator Formulas
    print("  - ICS (Indicator Convergence Score)...")
    result['ics'] = calculate_ics(result)

    print("  - MJI (Momentum Jerk)...")
    result['mji'] = calculate_mji(result)

    print("  - PRF (Percentile Rank Fusion)...")
    result['prf'] = calculate_prf(result)

    print("  - EWAF (Entropy-Weighted Adaptive Fusion)...")
    result['ewaf'] = calculate_ewaf(result)

    print("  - KFIF (Kalman-Filtered Indicator Fusion)...")
    kfif, kfif_upper, kfif_lower = calculate_kfif(result)
    result['kfif'] = kfif
    result['kfif_upper'] = kfif_upper
    result['kfif_lower'] = kfif_lower

    print("Done calculating novel indicators.")

    return result


# ============================================================================
# UTILITY: Get indicator by name
# ============================================================================

NOVEL_INDICATORS = {
    'arwo': calculate_arwo,
    'dco': calculate_dco,
    'vcmo': calculate_vcmo,
    'ics': calculate_ics,
    'mji': calculate_mji,
    'prf': calculate_prf,
    'ewaf': calculate_ewaf,
}

def get_novel_indicator(df: pd.DataFrame, name: str) -> pd.Series:
    """Get a specific novel indicator by name."""
    name = name.lower()
    if name == 'kfif':
        kfif, _, _ = calculate_kfif(df)
        return kfif
    elif name in NOVEL_INDICATORS:
        return NOVEL_INDICATORS[name](df)
    else:
        raise ValueError(f"Unknown indicator: {name}. Available: {list(NOVEL_INDICATORS.keys()) + ['kfif']}")


if __name__ == "__main__":
    # Test with sample data
    print("Testing novel indicators...")

    # Create sample data
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2023-01-01', periods=n, freq='D')

    # Generate realistic OHLCV data
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n)) * 2
    low = close - np.abs(np.random.randn(n)) * 2
    open_price = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000000, 10000000, n)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    # Calculate all indicators
    result = calculate_all_novel_indicators(df)

    # Print summary
    print("\nIndicator Statistics:")
    for col in ['arwo', 'dco', 'vcmo', 'ics', 'mji', 'prf', 'ewaf', 'kfif']:
        if col in result.columns:
            series = result[col].dropna()
            print(f"  {col.upper()}: mean={series.mean():.4f}, std={series.std():.4f}, "
                  f"min={series.min():.4f}, max={series.max():.4f}")

    print("\nAll tests passed!")
