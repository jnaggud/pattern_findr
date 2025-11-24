#!/usr/bin/env python3

import yfinance as yf
from simple_regime_detector import detect_market_regime
import pandas as pd

print("🔍 CRITICAL TEST: April 2025 Regime Detection")
print()

# Get data including April 2025 
ticker = 'SPY'
data = yf.Ticker(ticker).history(period='1y', interval='1d')
data.reset_index(inplace=True) 
data.columns = [col.lower() for col in data.columns]

print(f"📊 Data range: {data['date'].min()} to {data['date'].max()}")

# Check what data we actually have
print("\n📅 Data availability check:")
print(f"   Total days: {len(data)}")
print(f"   Date range: {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")

# Find the crash period (should be around April 2025)
april_data = data[data['date'].dt.month == 4]
if len(april_data) == 0:
    print("❌ No April data found!")
    print("Available months:", sorted(data['date'].dt.month.unique()))
    
    # Try to find the lowest point in available data
    min_price_idx = data['close'].idxmin()
    min_date = data.iloc[min_price_idx]['date']
    min_price = data.iloc[min_price_idx]['close']
    
    print(f"\n💥 ACTUAL MARKET BOTTOM IN DATA:")
    print(f"   Date: {min_date.strftime('%Y-%m-%d')}")
    print(f"   Price: ${min_price:.2f}")
    
    try:
        bottom_regime = detect_market_regime(data, min_price_idx)
        print(f"   Detected regime: {bottom_regime}")
        
        if bottom_regime in ['crash', 'bear']:
            print(f"   ✅ GOOD: Market bottom correctly detected as {bottom_regime}")
        else:
            print(f"   ❌ CRITICAL BUG: Market bottom detected as {bottom_regime} instead of crash/bear!")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    # Also test a few points around the bottom
    print(f"\n🔍 REGIME DETECTION AROUND MARKET BOTTOM:")
    start_idx = max(0, min_price_idx - 5)
    end_idx = min(len(data), min_price_idx + 6)
    
    for i in range(start_idx, end_idx):
        date = data.iloc[i]['date']
        price = data.iloc[i]['close']
        
        try:
            regime = detect_market_regime(data, i)
            marker = "💥" if i == min_price_idx else "   "
            print(f"{marker} {date.strftime('%Y-%m-%d')}: Price=${price:.2f} | Regime={regime}")
        except Exception as e:
            print(f"   ❌ ERROR detecting regime for {date}: {e}")
            
else:
    print(f"📅 April data: {len(april_data)} days")
    print()
    
    # Test regime detection for April 2025 crash period
    print("🔍 REGIME DETECTION TEST for April:")
    
    for i in range(len(april_data)):
        data_idx = april_data.index[i]
        date = data.iloc[data_idx]['date']
        price = data.iloc[data_idx]['close']
        
        try:
            regime = detect_market_regime(data, data_idx)
            print(f"   {date.strftime('%Y-%m-%d')}: Price=${price:.2f} | Regime={regime}")
            
            # Focus on the crash period (around April 8-10, 2025)
            if date.day >= 7 and date.day <= 12:
                if regime not in ['crash', 'bear']:
                    print(f"   ⚠️  PROBLEM: April crash not detected as crash/bear!")
                else:
                    print(f"   ✅ Correctly detected as {regime}")
                    
        except Exception as e:
            print(f"   ❌ ERROR detecting regime for {date}: {e}")
            
    print()
    
    # Test the lowest point specifically
    min_price_idx = april_data['close'].idxmin()
    min_date = data.iloc[min_price_idx]['date']
    min_price = data.iloc[min_price_idx]['close']
    
    print(f"💥 APRIL 2025 BOTTOM TEST:")
    print(f"   Date: {min_date.strftime('%Y-%m-%d')}")
    print(f"   Price: ${min_price:.2f}")
    
    try:
        bottom_regime = detect_market_regime(data, min_price_idx)
        print(f"   Detected regime: {bottom_regime}")
        
        if bottom_regime in ['crash', 'bear']:
            print(f"   ✅ GOOD: Market bottom correctly detected as {bottom_regime}")
        else:
            print(f"   ❌ CRITICAL BUG: Market bottom detected as {bottom_regime} instead of crash/bear!")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
