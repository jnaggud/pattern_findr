#!/usr/bin/env python3
"""
Debug if crash detection indicators are integrated properly
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
from indicators import get_all_indicators
import pandas as pd

def debug_crash_integration():
    print("🔍 DEBUGGING CRASH INDICATOR INTEGRATION")
    print("=" * 50)
    
    # Get SPY data
    data = yf.Ticker('SPY').history(period='1y', interval='1d')
    data.reset_index(inplace=True)
    data.columns = [col.lower() for col in data.columns]
    
    print("📊 Raw data columns before indicators:")
    print(f"   {list(data.columns)}")
    
    # Add indicators
    enriched_data = get_all_indicators(data)
    
    print(f"\n📈 Total columns after indicators: {len(enriched_data.columns)}")
    
    # Check for crash/bottom detection columns
    crash_columns = []
    for col in enriched_data.columns:
        if any(keyword in col.lower() for keyword in ['crash', 'bottom', 'panic', 'hammer', 'exhaustion', 'capitulation']):
            crash_columns.append(col)
    
    print(f"\n🚨 CRASH/BOTTOM DETECTION COLUMNS FOUND:")
    if crash_columns:
        for col in crash_columns:
            # Check if column has any True values
            if col in enriched_data.columns:
                true_count = enriched_data[col].sum() if enriched_data[col].dtype == bool else 'N/A'
                print(f"   ✅ {col}: {true_count} True values")
            else:
                print(f"   ❌ {col}: Column missing")
    else:
        print("   ❌ NO CRASH/BOTTOM DETECTION COLUMNS FOUND!")
    
    # Test April 2025 specifically
    april_data = enriched_data[enriched_data['date'].dt.month == 4]
    if len(april_data) > 0:
        print(f"\n📅 APRIL 2025 TEST ({len(april_data)} days):")
        
        # Check crash_buy_signal specifically
        if 'crash_buy_signal' in enriched_data.columns:
            april_crash_signals = april_data['crash_buy_signal'].sum()
            print(f"   crash_buy_signal: {april_crash_signals} days triggered")
            
            if april_crash_signals > 0:
                print("   ✅ Crash buy signals ARE triggering in April!")
                
                # Show specific days
                crash_days = april_data[april_data['crash_buy_signal'] == True]
                for _, row in crash_days.head(3).iterrows():
                    print(f"      {row['date'].strftime('%Y-%m-%d')}: ${row['close']:.2f}")
            else:
                print("   ❌ NO crash buy signals in April")
                
                # Debug why not triggering
                print("\n   🔍 Checking component indicators:")
                component_indicators = ['selling_exhaustion', 'panic_recovery', 'volume_confirmation', 
                                      'hammer_pattern', 'crash_composite_score']
                
                sample_day = april_data.iloc[5]  # Mid-April
                for indicator in component_indicators:
                    if indicator in enriched_data.columns:
                        value = sample_day[indicator]
                        print(f"      {indicator}: {value}")
                    else:
                        print(f"      {indicator}: MISSING")
        else:
            print("   ❌ crash_buy_signal column NOT FOUND!")
    
    # Test the crash override logic manually
    if 'crash_buy_signal' in enriched_data.columns:
        print(f"\n⚡ TESTING CRASH OVERRIDE LOGIC:")
        
        # Simulate universal_strategy crash override
        crash_override_signals = enriched_data['crash_buy_signal']
        crash_boost = crash_override_signals.astype(float) * 5
        
        print(f"   Total crash override days: {crash_override_signals.sum()}")
        print(f"   Max crash boost: {crash_boost.max()}")
        
        # Check April specifically
        april_crash_boost = crash_boost[enriched_data['date'].dt.month == 4]
        print(f"   April crash boost sum: {april_crash_boost.sum()}")
        
        if april_crash_boost.sum() > 0:
            print("   ✅ Crash boost should work in April!")
        else:
            print("   ❌ No crash boost in April - this is the problem!")

if __name__ == "__main__":
    debug_crash_integration()
