#!/usr/bin/env python3
"""
CUSTOM CRASH DETECTION INDICATORS
=================================

Traditional indicators fail during extreme market crashes.
We need purpose-built crash detection indicators.

"""

import pandas as pd
import numpy as np

def crash_velocity_indicator(data, window=3, threshold=-8.0):
    """
    Detects rapid price declines (crash velocity)
    
    Args:
        data: DataFrame with 'close' column
        window: Days to calculate velocity over
        threshold: % decline threshold (e.g., -8% in 3 days)
    
    Returns:
        Series: True when crash velocity detected
    """
    returns = data['close'].pct_change()
    rolling_return = returns.rolling(window).sum() * 100
    
    # Trigger when decline exceeds threshold
    return rolling_return < threshold

def volume_panic_indicator(data, volume_multiplier=2.5, decline_threshold=-3.0):
    """
    Detects panic selling: High volume + price decline
    
    Args:
        data: DataFrame with 'close', 'volume' columns
        volume_multiplier: Volume vs 20-day average
        decline_threshold: Daily decline threshold
    
    Returns:
        Series: True when volume panic detected
    """
    daily_return = data['close'].pct_change() * 100
    volume_avg = data['volume'].rolling(20).mean()
    volume_ratio = data['volume'] / volume_avg
    
    # Panic = High volume + significant decline
    panic = (volume_ratio > volume_multiplier) & (daily_return < decline_threshold)
    return panic

def drawdown_capitulation_indicator(data, drawdown_threshold=-15.0, window=5):
    """
    Detects capitulation based on severe drawdown
    
    Args:
        data: DataFrame with 'close' column
        drawdown_threshold: % drawdown from recent high
        window: Days to look back for high
    
    Returns:
        Series: True when capitulation detected
    """
    # Calculate rolling high
    rolling_high = data['close'].rolling(window=20).max()
    current_drawdown = (data['close'] / rolling_high - 1) * 100
    
    # Severe drawdown = potential capitulation
    return current_drawdown < drawdown_threshold

def gap_down_indicator(data, gap_threshold=-4.0):
    """
    Detects significant gap downs (market open panic)
    
    Args:
        data: DataFrame with 'open', 'close' columns
        gap_threshold: % gap down threshold
    
    Returns:
        Series: True when gap down detected
    """
    prev_close = data['close'].shift(1)
    gap_return = (data['open'] / prev_close - 1) * 100
    
    return gap_return < gap_threshold

def multi_day_decline_indicator(data, days=5, threshold=-12.0):
    """
    Detects sustained multi-day declines
    
    Args:
        data: DataFrame with 'close' column
        days: Number of consecutive days to check
        threshold: Total % decline threshold
    
    Returns:
        Series: True when multi-day decline detected
    """
    # Calculate N-day return
    n_day_return = (data['close'] / data['close'].shift(days) - 1) * 100
    
    return n_day_return < threshold

def fear_index_indicator(data, rsi_threshold=20, willr_threshold=-90, volume_spike=2.0):
    """
    Composite fear indicator combining multiple signals
    
    Args:
        data: DataFrame with 'close', 'volume', 'RSI_14', 'WILLR_14'
        rsi_threshold: RSI oversold level
        willr_threshold: Williams %R oversold level  
        volume_spike: Volume multiplier
    
    Returns:
        Series: Fear intensity score (0-10)
    """
    fear_score = pd.Series(0, index=data.index)
    
    # RSI component
    if 'RSI_14' in data.columns:
        rsi_fear = (data['RSI_14'] < rsi_threshold).astype(int) * 2
        fear_score += rsi_fear
    
    # Williams %R component  
    if 'WILLR_14' in data.columns:
        willr_fear = (data['WILLR_14'] < willr_threshold).astype(int) * 2
        fear_score += willr_fear
    
    # Volume component
    volume_avg = data['volume'].rolling(20).mean()
    volume_fear = (data['volume'] > volume_avg * volume_spike).astype(int) * 2
    fear_score += volume_fear
    
    # Price decline component
    daily_return = data['close'].pct_change() * 100
    decline_fear = (daily_return < -3).astype(int) * 2
    fear_score += decline_fear
    
    # Drawdown component  
    rolling_high = data['close'].rolling(10).max()
    drawdown = (data['close'] / rolling_high - 1) * 100
    drawdown_fear = (drawdown < -8).astype(int) * 2
    fear_score += drawdown_fear
    
    return fear_score

def crash_bottom_reversal_indicator(data, velocity_change=5.0, volume_confirmation=1.5):
    """
    Detects potential crash bottoms/reversals
    
    Args:
        data: DataFrame with 'close', 'volume'
        velocity_change: Change in decline velocity
        volume_confirmation: Volume confirmation multiplier
    
    Returns:
        Series: True when potential bottom detected
    """
    # Calculate velocity (rate of decline change)
    returns = data['close'].pct_change() * 100
    velocity = returns.diff()  # Change in daily returns
    
    # Volume confirmation
    volume_avg = data['volume'].rolling(5).mean()  
    volume_ratio = data['volume'] / volume_avg
    
    # Bottom signal: Velocity improving + volume confirmation + still oversold
    velocity_improving = velocity > velocity_change
    volume_confirming = volume_ratio > volume_confirmation
    still_declining = returns < -1  # Still down but less severe
    
    bottom_signal = velocity_improving & volume_confirming & still_declining
    return bottom_signal

def implement_crash_indicators(data):
    """
    Add all crash indicators to the data
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        DataFrame: Data with crash indicators added
    """
    # Add crash detection indicators
    data['crash_velocity'] = crash_velocity_indicator(data)
    data['volume_panic'] = volume_panic_indicator(data)
    data['drawdown_capitulation'] = drawdown_capitulation_indicator(data)
    data['gap_down'] = gap_down_indicator(data)
    data['multi_day_decline'] = multi_day_decline_indicator(data)
    data['fear_index'] = fear_index_indicator(data)
    data['crash_bottom_reversal'] = crash_bottom_reversal_indicator(data)
    
    # Composite crash score (0-7 based on how many indicators trigger)
    crash_indicators = [
        'crash_velocity', 'volume_panic', 'drawdown_capitulation', 
        'gap_down', 'multi_day_decline'
    ]
    data['crash_composite_score'] = data[crash_indicators].sum(axis=1)
    
    # High crash score = strong buy signal
    data['crash_buy_signal'] = data['crash_composite_score'] >= 2
    
    return data

if __name__ == "__main__":
    print("🚨 CRASH DETECTION INDICATORS")
    print("=" * 40)
    print()
    print("💡 CONCEPT: Purpose-built indicators for extreme market conditions")
    print()
    print("📊 PROPOSED INDICATORS:")
    print("   1. Crash Velocity: Rapid decline detection (-8% in 3 days)")  
    print("   2. Volume Panic: High volume + price decline")
    print("   3. Drawdown Capitulation: Severe drawdown from highs")
    print("   4. Gap Down: Market open panic gaps") 
    print("   5. Multi-day Decline: Sustained selling pressure")
    print("   6. Fear Index: Composite fear/panic measure")
    print("   7. Crash Bottom Reversal: Potential bottom detection")
    print()
    print("🎯 ADVANTAGE: These will trigger when traditional indicators fail")
    print("   - RSI might only hit 30 during crash → Crash indicators trigger")
    print("   - Williams %R might stay -70 → Volume panic triggers") 
    print("   - MACD too slow → Velocity indicator catches rapid decline")
    print()
    print("⚡ INTEGRATION: Add these to universal_strategy as priority indicators")
    print("   - If crash_composite_score >= 2: FORCE BUY regardless of other indicators")
    print("   - Bypass trend filter during crash conditions")
    print("   - Use crash-specific position sizing (bigger positions during panic)")
