"""Technical indicators for signal generation."""

from .oscillators import (
    calculate_composite_oscillator,
    get_oscillator_zone,
    detect_crossings
)

from .velocity import (
    calculate_velocity_signals,
    get_recent_signals,
    find_most_recent_signal,
    check_exit_conditions,
    SignalType,
    Signal
)

__all__ = [
    'calculate_composite_oscillator',
    'get_oscillator_zone',
    'detect_crossings',
    'calculate_velocity_signals',
    'get_recent_signals',
    'find_most_recent_signal',
    'check_exit_conditions',
    'SignalType',
    'Signal'
]
