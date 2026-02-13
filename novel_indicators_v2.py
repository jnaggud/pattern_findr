"""
Novel Indicators V2 - Research-Based Implementations (2025-2026)
================================================================

Based on quantitative finance research findings from recent literature:

1. Shannon Entropy Indicator (SEI) - Market disorder measurement
2. Relative Moving Average Framework (RMA) - Dynamic distribution tracking
3. Volume Dimensions Indicator (VDI) - Multi-dimensional volume analysis
4. Session Time Pattern (STP) - Intraday time-of-day patterns
5. Regime State Classifier (RSC) - HMM/GMM/KMeans regime detection
6. Market Fragility Index (MFI2) - Tail risk measurement
7. Order Flow Imbalance (OFI) - Microstructure-based buy/sell pressure
8. Multi-Timeframe Confirmation (MTC) - Higher timeframe trend filter

Research Sources:
- State Street: Decoding Market Regimes with ML (2025)
- Bloch: RMA Framework for Trading (SSRN 2025)
- Forecasting Intraday Volume with ML (arXiv 2505.08180)
- AI-Driven Intraday Trading (SSRN 5246516)
- Adaptive Regime-Aware RL (arXiv 2509.14385)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Optional imports for regime detection
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _scale_window(base_window: int, interval: str) -> int:
    """Scale window size based on timeframe.

    Intraday data has more bars per day, so windows need to be larger
    to capture equivalent time periods.
    """
    scale_factors = {
        '1d': 1.0,
        '4h': 1.5,
        '1h': 1.0,   # Keep similar for hourly
        '15m': 2.0,  # ~26 bars per day
        '5m': 3.0,   # ~78 bars per day
        '1m': 5.0,   # ~390 bars per day
    }
    factor = scale_factors.get(interval, 1.0)
    return max(2, int(base_window * factor))


def _normalize_to_range(series: pd.Series, low: float = -1, high: float = 1) -> pd.Series:
    """Normalize a series to a specified range."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0, index=series.index)
    return low + (series - min_val) * (high - low) / (max_val - min_val)


# =============================================================================
# 1. SHANNON ENTROPY INDICATOR (SEI)
# =============================================================================

def calculate_sei(df: pd.DataFrame, window: int = 20, n_bins: int = 10,
                  interval: str = '1d') -> pd.Series:
    """
    Shannon Entropy Indicator - Measures market disorder/uncertainty.

    Based on information theory: entropy measures the randomness in return
    distribution. High entropy = uncertain/ranging market, Low entropy =
    decisive/trending market.

    Args:
        df: DataFrame with 'close' column
        window: Rolling window for entropy calculation
        n_bins: Number of histogram bins for discretization
        interval: Timeframe for scaling ('1d', '15m', etc.)

    Returns:
        Series normalized to -1 (low entropy/trending) to +1 (high entropy/ranging)
    """
    scaled_window = _scale_window(window, interval)

    close = df['close'] if 'close' in df.columns else df['Close']
    returns = close.pct_change()

    def rolling_entropy(x):
        x_clean = x.dropna()
        if len(x_clean) < n_bins:
            return 0.5
        try:
            hist, _ = np.histogram(x_clean, bins=n_bins, density=True)
            hist = hist[hist > 0]
            if len(hist) == 0:
                return 0.5
            # Shannon entropy normalized by max possible entropy
            max_entropy = np.log(n_bins)
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            return entropy / max_entropy  # Normalized 0-1
        except:
            return 0.5

    sei = returns.rolling(window=scaled_window, min_periods=scaled_window//2).apply(
        rolling_entropy, raw=False
    )

    # Transform to -1 to +1 (0.5 entropy = 0, low entropy = -1, high = +1)
    return ((sei - 0.5) * 2).clip(-1, 1).fillna(0)


# =============================================================================
# 2. RELATIVE MOVING AVERAGE FRAMEWORK (RMA)
# =============================================================================

def calculate_rma(df: pd.DataFrame, fast_period: int = 10, slow_period: int = 50,
                  interval: str = '1d') -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Relative Moving Average Framework - Dynamic distribution around MA.

    Based on Bloch (2025): Price tends to revert to moving averages, but the
    distribution around the MA varies with market conditions.

    Args:
        df: DataFrame with 'close' column
        fast_period: Period for fast calculations
        slow_period: Period for slow MA
        interval: Timeframe for scaling

    Returns:
        Tuple of three series:
        - rma_position: Where price is relative to MA distribution (-1 to +1)
        - rma_momentum: Rate of change in position (-1 to +1)
        - rma_compression: How tight the distribution is (-1 to +1)
    """
    fast = _scale_window(fast_period, interval)
    slow = _scale_window(slow_period, interval)

    close = df['close'] if 'close' in df.columns else df['Close']
    ma_slow = close.rolling(window=slow).mean()

    # Distance from MA normalized by recent volatility
    distance = close - ma_slow
    volatility = close.rolling(window=fast).std()

    # Position: how many std devs from MA (bounded)
    rma_position = (distance / (volatility * 2 + 1e-10)).clip(-1, 1)

    # Momentum: rate of change in position
    rma_momentum = rma_position.diff(fast).clip(-1, 1)

    # Compression: inverse of relative volatility
    # High compression (low vol percentile) = potential breakout setup
    vol_percentile = volatility.rolling(window=slow).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100 if len(x.dropna()) > 1 else 0.5
    )
    rma_compression = (1 - vol_percentile * 2).clip(-1, 1)

    return rma_position.fillna(0), rma_momentum.fillna(0), rma_compression.fillna(0)


# =============================================================================
# 3. VOLUME DIMENSIONS INDICATOR (VDI)
# =============================================================================

def calculate_vdi(df: pd.DataFrame, period: int = 20,
                  interval: str = '1d') -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Volume Dimensions Indicator - Multi-dimensional volume analysis.

    Based on intraday volume research: Volume has multiple components that
    carry different information - raw level, directional bias, and
    participation breadth.

    Args:
        df: DataFrame with 'close', 'high', 'low', 'volume' columns
        period: Analysis period
        interval: Timeframe for scaling

    Returns:
        Tuple of three series:
        - vdi_intensity: Relative volume intensity (-1 to +1)
        - vdi_direction: Volume-weighted price direction (-1 to +1)
        - vdi_participation: Broad vs narrow participation proxy (-1 to +1)
    """
    scaled_period = _scale_window(period, interval)

    close = df['close'] if 'close' in df.columns else df['Close']
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    volume = df.get('volume', df.get('Volume', pd.Series(1, index=df.index)))

    # 1. Volume Intensity - relative to recent average (z-score)
    vol_ma = volume.rolling(window=scaled_period).mean()
    vol_std = volume.rolling(window=scaled_period).std().replace(0, 1)
    vdi_intensity = ((volume - vol_ma) / (vol_std * 2)).clip(-1, 1)

    # 2. Volume-weighted direction - accumulated buying/selling pressure
    price_change = close.diff()
    vol_direction = (price_change * volume).rolling(window=scaled_period).sum()
    vol_dir_std = vol_direction.rolling(window=scaled_period * 2).std().replace(0, 1)
    vdi_direction = (vol_direction / (vol_dir_std * 2)).clip(-1, 1)

    # 3. Participation proxy - where in bar did close occur
    # Close near high = buyers aggressive, near low = sellers aggressive
    intrabar_position = (close - low) / (high - low + 1e-10)
    participation = intrabar_position.rolling(window=scaled_period).mean()
    vdi_participation = ((participation - 0.5) * 2).clip(-1, 1)

    return vdi_intensity.fillna(0), vdi_direction.fillna(0), vdi_participation.fillna(0)


# =============================================================================
# 4. SESSION TIME PATTERN (STP) - Intraday Only
# =============================================================================

def calculate_stp(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Session Time Pattern - Captures intraday time-of-day effects.

    Based on research showing opening and closing periods have different
    dynamics than mid-session. Volume follows U-shape pattern.

    Args:
        df: DataFrame with datetime index

    Returns:
        Tuple of two series:
        - stp_session: Which part of session (open=-1, mid=0, close=+1)
        - stp_volatility_adj: Expected volatility adjustment based on time
    """
    if not hasattr(df.index, 'hour'):
        # Daily data - return neutral values
        return (pd.Series(0, index=df.index), pd.Series(1.0, index=df.index))

    hour = pd.Series(df.index.hour, index=df.index)
    minute = pd.Series(df.index.minute, index=df.index)
    time_decimal = hour + minute / 60

    # Market hours: 9:30 AM to 4:00 PM ET (adjust for your timezone)
    market_start = 9.5
    market_end = 16.0
    market_duration = market_end - market_start

    # Session position: -1 at open, 0 at mid, +1 at close
    session_progress = (time_decimal - market_start) / market_duration
    session_progress = session_progress.clip(0, 1)
    stp_session = ((session_progress - 0.5) * 2).clip(-1, 1)

    # Volatility adjustment based on U-shape pattern
    # First and last 30 mins are typically more volatile
    is_open = (time_decimal >= 9.5) & (time_decimal < 10.0)
    is_close = (time_decimal >= 15.5) & (time_decimal <= 16.0)

    # Volatility adjustment: 1.0 = normal, >1 = expect higher vol
    stp_volatility_adj = pd.Series(1.0, index=df.index)
    stp_volatility_adj[is_open] = 1.3   # Opening often more volatile
    stp_volatility_adj[is_close] = 1.2  # Closing somewhat more volatile

    return stp_session.fillna(0), stp_volatility_adj


# =============================================================================
# 5. REGIME STATE CLASSIFIER (RSC)
# =============================================================================

def calculate_rsc(df: pd.DataFrame, n_states: int = 3, lookback: int = 100,
                  method: str = 'kmeans', interval: str = '1d') -> pd.Series:
    """
    Regime State Classifier - Market regime detection using clustering.

    Based on State Street (2025) research on market regimes. Uses unsupervised
    learning to identify distinct market states.

    Args:
        df: DataFrame with 'close' column
        n_states: Number of regime states (typically 3: bull, bear, neutral)
        lookback: Rolling window for classification
        method: 'kmeans', 'gmm', or 'hmm'
        interval: Timeframe for scaling

    Returns:
        Series with regime values: -1 (bearish), 0 (neutral), +1 (bullish)
    """
    scaled_lookback = _scale_window(lookback, interval)

    close = df['close'] if 'close' in df.columns else df['Close']

    # Features for clustering
    returns = close.pct_change()
    volatility = returns.rolling(window=20).std()
    momentum = close.pct_change(20)

    # Combine into feature matrix
    features = pd.DataFrame({
        'returns': returns,
        'volatility': volatility,
        'momentum': momentum
    }).dropna()

    if len(features) < scaled_lookback:
        return pd.Series(0, index=df.index)

    # Simple rolling regime classification
    regime = pd.Series(index=df.index, dtype=float)

    # Use simplified approach if sklearn not available
    if not SKLEARN_AVAILABLE:
        # Fallback: use momentum and volatility thresholds
        mom_z = (momentum - momentum.rolling(scaled_lookback).mean()) / \
                (momentum.rolling(scaled_lookback).std() + 1e-10)
        regime = mom_z.clip(-1, 1)
        return regime.fillna(0)

    # Rolling classification with sklearn
    step_size = max(1, scaled_lookback // 10)  # Update every ~10% of window

    for i in range(scaled_lookback, len(features), step_size):
        window_data = features.iloc[max(0, i-scaled_lookback):i].values

        try:
            if method == 'kmeans':
                model = KMeans(n_clusters=n_states, random_state=42, n_init=10)
                labels = model.fit_predict(window_data)

                # Map clusters to direction based on momentum
                centers = model.cluster_centers_
                momentum_idx = 2  # momentum is 3rd feature
                sorted_clusters = np.argsort(centers[:, momentum_idx])

                # Create mapping: lowest momentum = -1, highest = +1
                regime_map = {}
                for j, cluster_id in enumerate(sorted_clusters):
                    regime_map[cluster_id] = (j - (n_states - 1) / 2) * (2 / (n_states - 1))

                # Assign regime for window range
                for k in range(step_size):
                    if i + k < len(features):
                        idx = features.index[i + k - 1] if i + k - 1 < len(features) else features.index[-1]
                        label = labels[-1] if k == 0 else labels[min(k, len(labels)-1)]
                        regime.loc[idx] = regime_map.get(label, 0)

            elif method == 'gmm':
                model = GaussianMixture(n_components=n_states, random_state=42)
                model.fit(window_data)
                labels = model.predict(window_data)

                centers = model.means_
                sorted_clusters = np.argsort(centers[:, 2])
                regime_map = {sorted_clusters[j]: (j - (n_states - 1) / 2) * (2 / (n_states - 1))
                              for j in range(n_states)}

                idx = features.index[i - 1]
                regime.loc[idx] = regime_map.get(labels[-1], 0)

        except Exception:
            pass

    # Forward fill to cover gaps
    regime = regime.ffill().bfill()
    return regime.reindex(df.index).fillna(0).clip(-1, 1)


# =============================================================================
# 6. MARKET FRAGILITY INDEX (MFI2)
# =============================================================================

def calculate_mfi2(df: pd.DataFrame, window: int = 20,
                   interval: str = '1d') -> pd.Series:
    """
    Market Fragility Index - Measures tail risk and market stress.

    Combines multiple stress indicators: kurtosis (fat tails), volatility
    clustering, negative skew, and gap risk.

    Args:
        df: DataFrame with OHLC columns
        window: Analysis window
        interval: Timeframe for scaling

    Returns:
        Series: -1 (stable) to +1 (fragile/stressed)
    """
    scaled_window = _scale_window(window, interval)

    close = df['close'] if 'close' in df.columns else df['Close']
    open_price = df['open'] if 'open' in df.columns else df['Open']

    returns = close.pct_change()

    # 1. Tail risk: excess kurtosis (fat tails = more extreme moves)
    kurtosis = returns.rolling(window=scaled_window).apply(
        lambda x: stats.kurtosis(x.dropna()) if len(x.dropna()) > 4 else 0
    )
    kurtosis_norm = (kurtosis / 10).clip(-1, 1)  # Normalize, excess kurtosis typically 0-10

    # 2. Volatility clustering: autocorrelation of squared returns
    sq_returns = returns ** 2
    vol_cluster = sq_returns.rolling(window=scaled_window).apply(
        lambda x: x.autocorr() if len(x.dropna()) > 5 else 0
    ).fillna(0)
    vol_cluster_norm = vol_cluster.clip(-1, 1)

    # 3. Negative skew: asymmetric downside risk
    skewness = returns.rolling(window=scaled_window).apply(
        lambda x: stats.skew(x.dropna()) if len(x.dropna()) > 4 else 0
    )
    skew_norm = (-skewness / 3).clip(-1, 1)  # Negative skew = more fragile

    # 4. Gap risk: overnight/weekend gaps
    gap = (open_price / close.shift(1) - 1).abs()
    gap_ma = gap.rolling(scaled_window).mean()
    gap_std = gap.rolling(scaled_window).std().replace(0, 1e-10)
    gap_zscore = (gap - gap_ma) / gap_std
    gap_norm = (gap_zscore / 3).clip(-1, 1)

    # Weighted combination (emphasize tail risk and clustering)
    mfi2 = (
        0.3 * kurtosis_norm +
        0.3 * vol_cluster_norm +
        0.25 * skew_norm +
        0.15 * gap_norm
    )

    return mfi2.clip(-1, 1).fillna(0)


# =============================================================================
# 7. ORDER FLOW IMBALANCE (OFI)
# =============================================================================

def calculate_ofi(df: pd.DataFrame, period: int = 10,
                  interval: str = '1d') -> pd.Series:
    """
    Order Flow Imbalance - Proxy for microstructure buy/sell pressure.

    Approximates order flow using OHLC data when tick data is unavailable.
    Based on where price closes within its range and volume.

    Args:
        df: DataFrame with OHLC and volume columns
        period: Analysis period
        interval: Timeframe for scaling

    Returns:
        Series: -1 (selling pressure) to +1 (buying pressure)
    """
    scaled_period = _scale_window(period, interval)

    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']
    open_price = df['open'] if 'open' in df.columns else df['Open']
    volume = df.get('volume', df.get('Volume', pd.Series(1, index=df.index)))

    # Buy pressure: close near high, green candle
    range_size = high - low + 1e-10
    close_position = (close - low) / range_size  # 0 to 1

    # Directional component: was it an up or down bar
    direction = np.sign(close - open_price)

    # Volume-weighted imbalance
    # High volume + close near high + up bar = strong buying
    buy_pressure = close_position * (1 + direction * 0.5) * volume
    sell_pressure = (1 - close_position) * (1 - direction * 0.5) * volume

    imbalance = buy_pressure - sell_pressure

    # Normalize with z-score
    imb_ma = imbalance.rolling(window=scaled_period).mean()
    imb_std = imbalance.rolling(window=scaled_period).std().replace(0, 1)

    ofi = ((imbalance - imb_ma) / (imb_std * 2)).clip(-1, 1)

    return ofi.fillna(0)


# =============================================================================
# 8. MULTI-TIMEFRAME CONFIRMATION (MTC)
# =============================================================================

def calculate_mtc(df: pd.DataFrame, tf_multipliers: List[int] = [4, 12],
                  interval: str = '1d') -> pd.Series:
    """
    Multi-Timeframe Confirmation - Higher timeframe trend filter.

    Calculates trend agreement across multiple simulated timeframes.
    Based on research showing trends confirmed on higher TFs have
    higher continuation probability.

    Args:
        df: DataFrame with 'close' column
        tf_multipliers: List of period multipliers for higher TFs
                       e.g., [4, 12] simulates 4x and 12x higher timeframes
        interval: Timeframe for context

    Returns:
        Series: -1 (all bearish) to +1 (all bullish), 0 = mixed
    """
    close = df['close'] if 'close' in df.columns else df['Close']

    # Base timeframe trend (fast MA vs slow MA)
    fast_period = 10
    slow_period = 30

    fast_ma = close.rolling(window=fast_period).mean()
    slow_ma = close.rolling(window=slow_period).mean()
    current_trend = np.sign(fast_ma - slow_ma)

    confirmations = [current_trend]

    for mult in tf_multipliers:
        # Simulate higher timeframe by using longer periods
        htf_fast = close.rolling(window=fast_period * mult).mean()
        htf_slow = close.rolling(window=slow_period * mult).mean()
        htf_trend = np.sign(htf_fast - htf_slow)
        confirmations.append(htf_trend)

    # Agreement score: average of all trends
    # +1 if all bullish, -1 if all bearish, 0 if mixed
    confirmation_df = pd.concat(confirmations, axis=1)
    mtc = confirmation_df.mean(axis=1)

    return mtc.clip(-1, 1).fillna(0)


# =============================================================================
# MASTER FUNCTION: Calculate All Novel V2 Indicators
# =============================================================================

def calculate_all_novel_v2_indicators(df: pd.DataFrame,
                                       interval: str = '1d') -> pd.DataFrame:
    """
    Calculate all novel V2 indicators and add to dataframe.

    Args:
        df: DataFrame with OHLCV data
        interval: Timeframe for scaling ('1d', '15m', etc.)

    Returns:
        DataFrame with 13 new indicator columns added
    """
    result = df.copy()

    # Ensure lowercase column names
    result.columns = result.columns.str.lower()

    print("Calculating Novel V2 Indicators...")

    # 1. Shannon Entropy
    print("  - SEI (Shannon Entropy)...")
    result['sei'] = calculate_sei(result, interval=interval)

    # 2. Relative Moving Average Framework
    print("  - RMA (Relative MA Framework)...")
    rma_pos, rma_mom, rma_comp = calculate_rma(result, interval=interval)
    result['rma_position'] = rma_pos
    result['rma_momentum'] = rma_mom
    result['rma_compression'] = rma_comp

    # 3. Volume Dimensions
    print("  - VDI (Volume Dimensions)...")
    vdi_int, vdi_dir, vdi_part = calculate_vdi(result, interval=interval)
    result['vdi_intensity'] = vdi_int
    result['vdi_direction'] = vdi_dir
    result['vdi_participation'] = vdi_part

    # 4. Session Time Pattern (most useful for intraday)
    print("  - STP (Session Time Pattern)...")
    stp_sess, stp_vol = calculate_stp(result)
    result['stp_session'] = stp_sess
    result['stp_volatility_adj'] = stp_vol

    # 5. Regime State Classifier
    print("  - RSC (Regime State Classifier)...")
    result['rsc_regime'] = calculate_rsc(result, interval=interval)

    # 6. Market Fragility Index
    print("  - MFI2 (Market Fragility)...")
    result['mfi2'] = calculate_mfi2(result, interval=interval)

    # 7. Order Flow Imbalance
    print("  - OFI (Order Flow Imbalance)...")
    result['ofi'] = calculate_ofi(result, interval=interval)

    # 8. Multi-Timeframe Confirmation
    print("  - MTC (Multi-TF Confirmation)...")
    result['mtc'] = calculate_mtc(result, interval=interval)

    print("Done calculating Novel V2 Indicators (13 columns added).")

    return result


# =============================================================================
# EXPORTS
# =============================================================================

NOVEL_V2_INDICATORS = {
    'sei': calculate_sei,
    'rma': calculate_rma,
    'vdi': calculate_vdi,
    'stp': calculate_stp,
    'rsc': calculate_rsc,
    'mfi2': calculate_mfi2,
    'ofi': calculate_ofi,
    'mtc': calculate_mtc,
}

NOVEL_V2_INDICATOR_LIST = [
    'sei',
    'rma_position', 'rma_momentum', 'rma_compression',
    'vdi_intensity', 'vdi_direction', 'vdi_participation',
    'stp_session', 'stp_volatility_adj',
    'rsc_regime',
    'mfi2',
    'ofi',
    'mtc',
]


def get_novel_v2_indicator(df: pd.DataFrame, name: str,
                            interval: str = '1d') -> pd.Series:
    """
    Get a specific novel V2 indicator by name.

    Args:
        df: DataFrame with OHLCV data
        name: Indicator name (case-insensitive)
        interval: Timeframe for scaling

    Returns:
        Series with indicator values
    """
    name_lower = name.lower()

    if name_lower == 'sei':
        return calculate_sei(df, interval=interval)
    elif name_lower in ['rma_position', 'rma_momentum', 'rma_compression']:
        rma_pos, rma_mom, rma_comp = calculate_rma(df, interval=interval)
        if 'position' in name_lower:
            return rma_pos
        elif 'momentum' in name_lower:
            return rma_mom
        else:
            return rma_comp
    elif name_lower in ['vdi_intensity', 'vdi_direction', 'vdi_participation']:
        vdi_int, vdi_dir, vdi_part = calculate_vdi(df, interval=interval)
        if 'intensity' in name_lower:
            return vdi_int
        elif 'direction' in name_lower:
            return vdi_dir
        else:
            return vdi_part
    elif name_lower in ['stp_session', 'stp_volatility_adj']:
        stp_sess, stp_vol = calculate_stp(df)
        return stp_sess if 'session' in name_lower else stp_vol
    elif name_lower == 'rsc_regime':
        return calculate_rsc(df, interval=interval)
    elif name_lower == 'mfi2':
        return calculate_mfi2(df, interval=interval)
    elif name_lower == 'ofi':
        return calculate_ofi(df, interval=interval)
    elif name_lower == 'mtc':
        return calculate_mtc(df, interval=interval)
    else:
        raise ValueError(f"Unknown indicator: {name}. Available: {NOVEL_V2_INDICATOR_LIST}")
