#!/usr/bin/env python3
"""
ENHANCED CRASH INDICATORS - More aggressive bottom catching
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
import pandas as pd
import numpy as np

def enhanced_crash_indicators():
    print("🚨 ENHANCED CRASH INDICATORS")
    print("=" * 40)
    
    # Get SPY data for testing
    print("Fetching data...")
    data = yf.Ticker('SPY').history(period='1y', interval='1d')
    data.reset_index(inplace=True)
    data.columns = [col.lower() for col in data.columns]
    
    print("Creating ENHANCED crash indicators...")
    
    # === ORIGINAL INDICATORS (for comparison) ===
    # 1. Original crash velocity (8% in 3 days)
    returns_3d = data['close'].pct_change(3) * 100
    data['crash_velocity_original'] = returns_3d < -8.0
    
    # === ENHANCED INDICATORS (more sensitive) ===
    
    # 1. ENHANCED VELOCITY - Multiple timeframes
    returns_1d = data['close'].pct_change(1) * 100
    returns_2d = data['close'].pct_change(2) * 100
    returns_5d = data['close'].pct_change(5) * 100
    
    # More sensitive velocity triggers
    data['velocity_1d_severe'] = returns_1d < -4.0    # 4% in 1 day
    data['velocity_2d_severe'] = returns_2d < -6.0    # 6% in 2 days  
    data['velocity_3d_enhanced'] = returns_3d < -6.0  # 6% in 3 days (vs 8%)
    data['velocity_5d_carnage'] = returns_5d < -10.0  # 10% in 5 days (vs 12%)
    
    # 2. ENHANCED VOLUME PANIC - Lower thresholds
    volume_avg = data['volume'].rolling(20).mean()
    volume_ratio = data['volume'] / volume_avg
    
    # Multiple volume panic levels
    data['volume_panic_mild'] = (volume_ratio > 1.8) & (returns_1d < -2.0)    # Mild panic
    data['volume_panic_severe'] = (volume_ratio > 2.5) & (returns_1d < -3.0)   # Original 
    data['volume_panic_extreme'] = (volume_ratio > 3.5) & (returns_1d < -1.0)  # Extreme volume
    
    # 3. ENHANCED DRAWDOWN - Progressive levels
    rolling_high_10d = data['close'].rolling(10).max()
    rolling_high_20d = data['close'].rolling(20).max() 
    rolling_high_50d = data['close'].rolling(50).max()
    
    drawdown_10d = (data['close'] / rolling_high_10d - 1) * 100
    drawdown_20d = (data['close'] / rolling_high_20d - 1) * 100
    drawdown_50d = (data['close'] / rolling_high_50d - 1) * 100
    
    data['drawdown_warning'] = drawdown_10d < -8.0   # 8% from 10-day high
    data['drawdown_alert'] = drawdown_20d < -12.0    # 12% from 20-day high
    data['drawdown_panic'] = drawdown_20d < -15.0    # Original 15%
    data['drawdown_catastrophe'] = drawdown_50d < -20.0  # 20% from 50-day high
    
    # 4. MOMENTUM CRASH - Rate of decline acceleration
    velocity_change = returns_1d.diff()  # Change in daily returns
    data['momentum_crash'] = velocity_change < -3.0  # Acceleration of decline
    
    # 5. SUPPORT LEVEL BREAKS - Technical breakdown
    sma_50 = data['close'].rolling(50).mean()
    sma_200 = data['close'].rolling(200).mean()
    
    # Price breaking key support levels
    data['break_sma50'] = (data['close'] < sma_50 * 0.95)  # 5% below SMA50
    data['break_sma200'] = (data['close'] < sma_200 * 0.98) # 2% below SMA200
    
    # 6. CAPITULATION COMBO - Multiple signals together
    # RSI extreme + Volume spike + Price decline
    rsi_extreme = returns_1d < -3.0  # Proxy for RSI extreme
    vol_spike = volume_ratio > 2.0
    data['capitulation_combo'] = rsi_extreme & vol_spike
    
    # 7. ENHANCED FEAR INDEX - More sensitive scoring
    fear_score = pd.Series(0, index=data.index)
    
    # Velocity fears (0-4 points)
    fear_score += data['velocity_1d_severe'].astype(int)
    fear_score += data['velocity_2d_severe'].astype(int) 
    fear_score += data['velocity_3d_enhanced'].astype(int)
    fear_score += data['velocity_5d_carnage'].astype(int)
    
    # Volume fears (0-3 points)
    fear_score += data['volume_panic_mild'].astype(int)
    fear_score += data['volume_panic_severe'].astype(int)
    fear_score += data['volume_panic_extreme'].astype(int)
    
    # Drawdown fears (0-4 points)  
    fear_score += data['drawdown_warning'].astype(int)
    fear_score += data['drawdown_alert'].astype(int)
    fear_score += data['drawdown_panic'].astype(int)
    fear_score += data['drawdown_catastrophe'].astype(int)
    
    # Technical fears (0-3 points)
    fear_score += data['momentum_crash'].astype(int)
    fear_score += data['break_sma50'].astype(int)
    fear_score += data['capitulation_combo'].astype(int)
    
    data['enhanced_fear_score'] = fear_score  # Max possible: 14
    
    # 8. ENHANCED CRASH COMPOSITE - More aggressive
    crash_indicators = [
        'velocity_1d_severe', 'velocity_2d_severe', 'velocity_3d_enhanced',
        'volume_panic_mild', 'volume_panic_severe', 
        'drawdown_warning', 'drawdown_alert', 'drawdown_panic',
        'momentum_crash', 'break_sma50', 'capitulation_combo'
    ]
    
    data['enhanced_crash_score'] = data[crash_indicators].sum(axis=1)  # Max: 11
    
    # 9. ENHANCED BUY SIGNALS - Multiple trigger levels
    # Level 1: Early warning (1+ indicators OR fear >= 2)
    data['crash_buy_early'] = (data['enhanced_crash_score'] >= 1) | (data['enhanced_fear_score'] >= 2)
    
    # Level 2: Moderate alert (2+ indicators OR fear >= 4)  
    data['crash_buy_moderate'] = (data['enhanced_crash_score'] >= 2) | (data['enhanced_fear_score'] >= 4)
    
    # Level 3: High confidence (3+ indicators OR fear >= 6)
    data['crash_buy_high'] = (data['enhanced_crash_score'] >= 3) | (data['enhanced_fear_score'] >= 6)
    
    # Level 4: Maximum aggression (1+ severe indicators)
    severe_indicators = ['velocity_1d_severe', 'volume_panic_severe', 'drawdown_panic', 'capitulation_combo']
    data['crash_buy_severe'] = data[severe_indicators].any(axis=1)
    
    # ULTIMATE CRASH BUY: Any of the above
    data['crash_buy_ultimate'] = (
        data['crash_buy_early'] | data['crash_buy_moderate'] | 
        data['crash_buy_high'] | data['crash_buy_severe']
    )
    
    # === TEST ON APRIL 2025 CRASH ===
    print("\n🔍 TESTING ON APRIL 2025 CRASH:")
    print("-" * 40)
    
    april_data = data[data['date'].dt.month == 4]
    crash_period = april_data[(april_data['date'].dt.day >= 1) & (april_data['date'].dt.day <= 20)]
    
    for _, row in crash_period.iterrows():
        date = row['date'].strftime('%m-%d')
        price = row['close']
        
        # Original vs Enhanced
        orig_buy = row.get('crash_velocity_original', False)
        ultimate_buy = row['crash_buy_ultimate']
        fear_score = row['enhanced_fear_score']
        crash_score = row['enhanced_crash_score']
        
        # Show signal strength
        if ultimate_buy:
            signal = f"🚨BUY (F:{fear_score} C:{crash_score})"
        elif crash_score > 0 or fear_score > 0:
            signal = f"⚠️WATCH (F:{fear_score} C:{crash_score})"
        else:
            signal = "HOLD"
            
        orig_symbol = "📈" if orig_buy else "  "
        
        print(f"{date}: ${price:6.2f} | {signal:25} | Orig:{orig_symbol}")
    
    # Summary comparison
    orig_signals = crash_period['crash_velocity_original'].sum() 
    ultimate_signals = crash_period['crash_buy_ultimate'].sum()
    
    print(f"\n📊 ENHANCEMENT RESULTS:")
    print(f"   Original signals: {orig_signals}")
    print(f"   Enhanced signals: {ultimate_signals}")
    print(f"   Improvement: +{ultimate_signals - orig_signals} days")
    
    if ultimate_signals > orig_signals:
        print(f"   ✅ Enhanced indicators catch more opportunities!")
        
        # Show the new signal days
        new_signals = crash_period[crash_period['crash_buy_ultimate'] & ~crash_period['crash_velocity_original']]
        print(f"\n📈 NEW CRASH BUY DAYS:")
        for _, row in new_signals.iterrows():
            print(f"      {row['date'].strftime('%Y-%m-%d')}: ${row['close']:.2f}")
            
    return ultimate_signals > orig_signals

if __name__ == "__main__":
    success = enhanced_crash_indicators()
    
    if success:
        print(f"\n🎉 ENHANCED CRASH INDICATORS READY!")
        print(f"   More sensitive bottom detection")
        print(f"   Ready for integration into indicators.py")
    else:
        print(f"\n⚠️  Enhancement may need further tuning")
