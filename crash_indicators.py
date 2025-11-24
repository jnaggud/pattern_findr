#!/usr/bin/env python3
"""
CRASH DETECTION INDICATORS for Pattern_FindR
===========================================

Purpose-built indicators that trigger during extreme market crashes
when traditional indicators fail.
"""

import pandas as pd
import numpy as np

def add_crash_indicators(data):
    """
    Add crash detection indicators to market data
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        DataFrame: Data with crash indicators added
    """
    
    # 1. CRASH VELOCITY - Rapid price declines
    returns_3d = data['close'].pct_change(3) * 100
    data['crash_velocity_3d'] = returns_3d < -8.0  # 8% decline in 3 days
    
    # 2. VOLUME PANIC - High volume + decline
    volume_avg_20d = data['volume'].rolling(20).mean()
    volume_ratio = data['volume'] / volume_avg_20d
    daily_return = data['close'].pct_change() * 100
    data['volume_panic'] = (volume_ratio > 2.5) & (daily_return < -3.0)
    
    # 3. DRAWDOWN CAPITULATION - Severe drawdown from highs
    rolling_high_20d = data['close'].rolling(20).max()
    drawdown = (data['close'] / rolling_high_20d - 1) * 100
    data['drawdown_capitulation'] = drawdown < -15.0  # 15% drawdown
    
    # 4. GAP DOWN PANIC - Market open gaps
    prev_close = data['close'].shift(1)
    gap_return = (data['open'] / prev_close - 1) * 100
    data['gap_down_panic'] = gap_return < -4.0  # 4% gap down
    
    # 5. MULTI-DAY CARNAGE - Sustained decline
    returns_5d = data['close'].pct_change(5) * 100
    data['multi_day_carnage'] = returns_5d < -12.0  # 12% decline in 5 days
    
    # 6. FEAR COMPOSITE SCORE - Multiple fear signals
    fear_score = pd.Series(0, index=data.index)
    
    # RSI extreme fear (if available)
    if 'RSI_14' in data.columns:
        fear_score += (data['RSI_14'] < 18).astype(int)  # Extreme oversold
    
    # Williams %R extreme fear (if available)  
    if 'WILLR_14' in data.columns:
        fear_score += (data['WILLR_14'] < -95).astype(int)  # Extreme oversold
        
    # Volume spike fear
    fear_score += (volume_ratio > 3.0).astype(int)
    
    # Sharp decline fear
    fear_score += (daily_return < -5.0).astype(int)
    
    # Severe drawdown fear
    fear_score += (drawdown < -20.0).astype(int)
    
    data['fear_composite_score'] = fear_score
    
    # 7. CRASH COMPOSITE - Ultimate crash detector
    crash_signals = [
        'crash_velocity_3d', 'volume_panic', 'drawdown_capitulation',
        'gap_down_panic', 'multi_day_carnage'
    ]
    data['crash_composite_score'] = data[crash_signals].sum(axis=1)
    
    # 8. CRASH BUY SIGNAL - When to buy the crash
    # Trigger when 2+ crash indicators fire OR fear score >= 3
    data['crash_buy_signal'] = (data['crash_composite_score'] >= 2) | (data['fear_composite_score'] >= 3)
    
    # 9. CRASH BOTTOM REVERSAL - Potential bottom detection
    velocity = daily_return.diff()  # Change in decline rate
    volume_confirm = data['volume'] > volume_avg_20d * 1.8
    velocity_improving = velocity > 3.0  # Declining slower
    still_down = daily_return < -1  # But still declining
    
    data['crash_bottom_signal'] = velocity_improving & volume_confirm & still_down
    
    return data

def get_crash_buy_score(data, index):
    """
    Calculate crash-specific buy score for a given index
    
    Args:
        data: DataFrame with crash indicators
        index: Current data index
        
    Returns:
        int: Crash buy score (0-10)
    """
    if index >= len(data):
        return 0
        
    score = 0
    row = data.iloc[index]
    
    # Each crash indicator adds to buy score
    if row.get('crash_velocity_3d', False):
        score += 2  # High weight for velocity
    
    if row.get('volume_panic', False):
        score += 2  # High weight for panic
        
    if row.get('drawdown_capitulation', False):
        score += 2  # High weight for capitulation
        
    if row.get('gap_down_panic', False):
        score += 1
        
    if row.get('multi_day_carnage', False):
        score += 1
        
    # Fear score adds directly
    fear_score = row.get('fear_composite_score', 0)
    score += min(fear_score, 2)  # Cap fear contribution at 2
    
    return score

if __name__ == "__main__":
    print("🚨 CRASH INDICATORS MODULE")
    print("=" * 30)
    print()
    print("📊 CRASH DETECTION INDICATORS:")
    print("   1. crash_velocity_3d: 8% decline in 3 days")
    print("   2. volume_panic: 2.5x volume + 3% decline")  
    print("   3. drawdown_capitulation: 15% drawdown from high")
    print("   4. gap_down_panic: 4% gap down at open")
    print("   5. multi_day_carnage: 12% decline in 5 days")
    print("   6. fear_composite_score: Multiple fear signals (0-5)")
    print("   7. crash_composite_score: Total crash signals (0-5)")
    print("   8. crash_buy_signal: When to buy (composite >= 2 OR fear >= 3)")
    print()
    print("🎯 INTEGRATION POINTS:")
    print("   - Add to indicators.py")
    print("   - Modify universal_strategy to prioritize crash signals")
    print("   - Bypass trend filter during crash conditions")
    print("   - Use crash-specific position sizing")
