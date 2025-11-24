#!/usr/bin/env python3
"""
Simple crash indicator test - no verbose output
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import yfinance as yf
import pandas as pd

# Manual crash indicator calculation (avoid indicators.py complexity)
def test_crash_indicators():
    print("🚨 CRASH INDICATOR TEST")
    print("=" * 30)
    
    # Get SPY data
    print("Fetching SPY data...")
    ticker = yf.Ticker('SPY')
    data = ticker.history(period='1y', interval='1d')
    data.reset_index(inplace=True)
    data.columns = [col.lower() for col in data.columns]
    
    # Manual crash indicator calculations
    print("Calculating crash indicators...")
    
    # 1. Crash velocity (8% decline in 3 days)
    returns_3d = data['close'].pct_change(3) * 100
    data['crash_velocity'] = returns_3d < -8.0
    
    # 2. Volume panic (2.5x volume + 3% decline)
    volume_avg = data['volume'].rolling(20).mean()
    volume_ratio = data['volume'] / volume_avg
    daily_return = data['close'].pct_change() * 100
    data['volume_panic'] = (volume_ratio > 2.5) & (daily_return < -3.0)
    
    # 3. Drawdown capitulation (15% from 20-day high)
    rolling_high = data['close'].rolling(20).max()
    drawdown = (data['close'] / rolling_high - 1) * 100
    data['drawdown_cap'] = drawdown < -15.0
    
    # 4. Multi-day carnage (12% decline in 5 days)
    returns_5d = data['close'].pct_change(5) * 100
    data['carnage'] = returns_5d < -12.0
    
    # Composite crash score
    data['crash_score'] = (
        data['crash_velocity'].astype(int) +
        data['volume_panic'].astype(int) + 
        data['drawdown_cap'].astype(int) +
        data['carnage'].astype(int)
    )
    
    # Crash buy signal (2+ indicators)
    data['crash_buy'] = data['crash_score'] >= 2
    
    # Test April 2025 crash period
    april_data = data[data['date'].dt.month == 4]
    crash_period = april_data[(april_data['date'].dt.day >= 3) & (april_data['date'].dt.day <= 15)]
    
    print(f"\n📅 April 3-15, 2025 ({len(crash_period)} days)")
    print("-" * 40)
    
    for _, row in crash_period.iterrows():
        date = row['date'].strftime('%m-%d')
        price = row['close']
        score = row['crash_score']
        buy_signal = "🚨BUY" if row['crash_buy'] else "HOLD"
        
        # Show which indicators triggered
        indicators = []
        if row['crash_velocity']: indicators.append('V')
        if row['volume_panic']: indicators.append('P') 
        if row['drawdown_cap']: indicators.append('D')
        if row['carnage']: indicators.append('C')
        
        trigger_text = ''.join(indicators) if indicators else '-'
        
        print(f"{date}: ${price:6.2f} | Score:{score} | {buy_signal:4} | [{trigger_text:4}]")
    
    # Summary
    crash_buys = crash_period['crash_buy'].sum()
    print(f"\n📊 RESULTS:")
    print(f"   Crash buy signals: {crash_buys}/{len(crash_period)} days")
    
    if crash_buys > 0:
        print(f"   ✅ SUCCESS! Crash indicators triggered!")
        buy_days = crash_period[crash_period['crash_buy']]['date']
        for date in buy_days:
            print(f"      📈 {date.strftime('%Y-%m-%d')}")
    else:
        print(f"   ❌ No crash signals - need to tune thresholds")
        
        # Show why no triggers
        print(f"\n🔍 DIAGNOSTIC:")
        sample = crash_period.iloc[2]  # April 5th area
        print(f"   3-day return: {((sample['close']/crash_period.iloc[0]['close'])-1)*100:.1f}%")
        print(f"   Drawdown: {((sample['close']/rolling_high.iloc[sample.name])-1)*100:.1f}%")
    
    return crash_buys > 0

if __name__ == "__main__":
    success = test_crash_indicators()
    
    if success:
        print(f"\n🎉 CRASH INDICATORS WORKING!")
        print(f"   Ready for integration with optimization system")
    else:
        print(f"\n⚠️  Need to adjust crash indicator thresholds")
