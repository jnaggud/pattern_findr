"""
Timeframe Configuration Module
==============================
Defines timeframe-specific parameter scaling for indicators and optimization.

Daily (1d):    Base periods (default)
4-hour (4h):   ~1.5x scaling factor
1-hour (1h):   ~6.5x scaling factor
15-min (15m):  ~26x scaling factor
5-min (5m):    ~78x scaling factor
"""

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

TimeframeType = Literal['1d', '4h', '1h', '15m', '5m', '1m']


@dataclass
class TimeframeParams:
    """Parameters scaled for a specific timeframe."""
    name: str
    bars_per_day: float

    # Novel indicator periods
    entropy_window: int
    percentile_lookback: int
    kalman_process_noise: float
    regime_lookback: int

    # Oscillator component periods
    rsi_period: int
    stoch_period: int
    cci_period: int
    adx_period: int

    # Velocity optimization ranges (min, max)
    vel_smoothing_range: Tuple[int, int]
    double_bottom_lookback_range: Tuple[int, int]
    divergence_lookback_range: Tuple[int, int]
    velocity_std_window_range: Tuple[int, int]
    momentum_multiplier_range: Tuple[float, float]


TIMEFRAME_CONFIGS: Dict[str, TimeframeParams] = {
    '1d': TimeframeParams(
        name='daily',
        bars_per_day=1,
        entropy_window=20,
        percentile_lookback=100,
        kalman_process_noise=0.01,
        regime_lookback=100,
        rsi_period=14,
        stoch_period=14,
        cci_period=20,
        adx_period=14,
        vel_smoothing_range=(1, 15),
        double_bottom_lookback_range=(5, 20),
        divergence_lookback_range=(3, 10),
        velocity_std_window_range=(5, 20),
        momentum_multiplier_range=(1.0, 3.0),
    ),
    '4h': TimeframeParams(
        name='4hour',
        bars_per_day=6.5,
        entropy_window=30,
        percentile_lookback=150,
        kalman_process_noise=0.015,
        regime_lookback=150,
        rsi_period=21,
        stoch_period=21,
        cci_period=30,
        adx_period=21,
        vel_smoothing_range=(1, 25),
        double_bottom_lookback_range=(8, 35),
        divergence_lookback_range=(5, 15),
        velocity_std_window_range=(8, 30),
        momentum_multiplier_range=(1.0, 3.0),
    ),
    '1h': TimeframeParams(
        name='hourly',
        bars_per_day=6.5,
        entropy_window=50,
        percentile_lookback=200,
        kalman_process_noise=0.02,
        regime_lookback=200,
        rsi_period=14,
        stoch_period=14,
        cci_period=20,
        adx_period=14,
        vel_smoothing_range=(1, 30),
        double_bottom_lookback_range=(10, 50),
        divergence_lookback_range=(5, 20),
        velocity_std_window_range=(10, 40),
        momentum_multiplier_range=(1.0, 3.0),
    ),
    '15m': TimeframeParams(
        name='15min',
        bars_per_day=26,
        entropy_window=80,
        percentile_lookback=300,
        kalman_process_noise=0.03,
        regime_lookback=300,
        rsi_period=14,
        stoch_period=14,
        cci_period=20,
        adx_period=14,
        vel_smoothing_range=(1, 50),
        double_bottom_lookback_range=(15, 80),
        divergence_lookback_range=(8, 30),
        velocity_std_window_range=(15, 60),
        momentum_multiplier_range=(1.0, 3.5),
    ),
    '5m': TimeframeParams(
        name='5min',
        bars_per_day=78,
        entropy_window=100,
        percentile_lookback=400,
        kalman_process_noise=0.04,
        regime_lookback=400,
        rsi_period=14,
        stoch_period=14,
        cci_period=20,
        adx_period=14,
        vel_smoothing_range=(1, 80),
        double_bottom_lookback_range=(20, 120),
        divergence_lookback_range=(10, 50),
        velocity_std_window_range=(20, 100),
        momentum_multiplier_range=(1.0, 4.0),
    ),
    '1m': TimeframeParams(
        name='1min',
        bars_per_day=390,
        entropy_window=150,
        percentile_lookback=500,
        kalman_process_noise=0.05,
        regime_lookback=500,
        rsi_period=14,
        stoch_period=14,
        cci_period=20,
        adx_period=14,
        vel_smoothing_range=(1, 100),
        double_bottom_lookback_range=(30, 150),
        divergence_lookback_range=(15, 60),
        velocity_std_window_range=(30, 120),
        momentum_multiplier_range=(1.0, 4.0),
    ),
}


def get_timeframe_config(interval: str) -> TimeframeParams:
    """Get configuration for a specific timeframe.

    Args:
        interval: Timeframe string ('1d', '4h', '1h', '15m', '5m', '1m')

    Returns:
        TimeframeParams dataclass with scaled parameters
    """
    return TIMEFRAME_CONFIGS.get(interval, TIMEFRAME_CONFIGS['1d'])


def scale_period(base_period: int, from_interval: str, to_interval: str) -> int:
    """Scale a period from one timeframe to another.

    Args:
        base_period: The period value in the source timeframe
        from_interval: Source timeframe ('1d', '15m', etc.)
        to_interval: Target timeframe

    Returns:
        Scaled period for the target timeframe
    """
    from_config = TIMEFRAME_CONFIGS.get(from_interval, TIMEFRAME_CONFIGS['1d'])
    to_config = TIMEFRAME_CONFIGS.get(to_interval, TIMEFRAME_CONFIGS['1d'])
    scale_factor = to_config.bars_per_day / from_config.bars_per_day
    return max(2, int(base_period * scale_factor))


def get_velocity_param_ranges(interval: str = '1d') -> dict:
    """Get velocity optimization parameter ranges scaled for the given timeframe.

    Args:
        interval: Timeframe string

    Returns:
        Dictionary with parameter name -> (min, max) tuples
    """
    config = get_timeframe_config(interval)
    return {
        'velocity_std_window': config.velocity_std_window_range,
        'double_bottom_lookback': config.double_bottom_lookback_range,
        'divergence_lookback': config.divergence_lookback_range,
        'vel_smoothing': config.vel_smoothing_range,
        'momentum_multiplier': config.momentum_multiplier_range,
    }


def get_indicator_params(interval: str = '1d') -> dict:
    """Get indicator calculation parameters for the given timeframe.

    Args:
        interval: Timeframe string

    Returns:
        Dictionary with indicator parameters
    """
    config = get_timeframe_config(interval)
    return {
        'entropy_window': config.entropy_window,
        'percentile_lookback': config.percentile_lookback,
        'kalman_process_noise': config.kalman_process_noise,
        'regime_lookback': config.regime_lookback,
        'rsi_period': config.rsi_period,
        'stoch_period': config.stoch_period,
        'cci_period': config.cci_period,
        'adx_period': config.adx_period,
    }


def is_intraday(interval: str) -> bool:
    """Check if the interval is intraday (not daily).

    Args:
        interval: Timeframe string

    Returns:
        True if intraday, False if daily
    """
    return interval != '1d'


def get_export_base_dir(interval: str) -> str:
    """Get the base directory for exports based on timeframe.

    Args:
        interval: Timeframe string

    Returns:
        Base directory path ('intraday' or empty string for daily)
    """
    return 'intraday' if is_intraday(interval) else ''
