#!/usr/bin/env python3
"""
Debug what happens in March before April to understand trade pattern
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
from indicators import get_all_indicators
from optimization import universal_strategy
import json
import pandas as pd

def debug_march_signals():
    print("🔍 DEBUGGING MARCH → APRIL TRANSITION")
    print("=" * 60)
    
    # Get data and add indicators
    data = yf.Ticker('SPY').history(period='1y', interval='1d')
    data.reset_index(inplace=True)
    data.columns = [col.lower() for col in data.columns]
    enriched_data = get_all_indicators(data)
    
    # Load parameters
    with open('optimizations/SPY_20251124_180031_trial77_ret46.4pct_params.json', 'r') as f:
        params = json.load(f)
    
    # Generate signals
    strategy_result = universal_strategy(enriched_data, params)
    signals = strategy_result['signals'] if isinstance(strategy_result, dict) else strategy_result
    
    # Focus on March-April transition
    march_april = enriched_data[(enriched_data.index.to_series().dt.month >= 3) & 
                               (enriched_data.index.to_series().dt.month <= 4)]
    
    if len(march_april) == 0:
        print("❌ No March-April data found")
        return
    
    march_april_signals = signals.loc[march_april.index]
    
    print(f"📅 MARCH → APRIL SIGNAL TRANSITION:")
    print("-" * 60)
    print(f"{'Date':<12} {'Price':<8} {'Signal':<8} {'Transition':<12}")
    print("-" * 60)
    
    signal_changes = []
    for i, (idx, signal) in enumerate(march_april_signals.items()):
        date = idx.strftime('%m-%d')
        price = march_april.loc[idx, 'close']
        
        # Check for signal transitions
        prev_signal = march_april_signals.iloc[i-1] if i > 0 else 0
        transition = ""
        
        if prev_signal != signal:
            if signal == 1:
                transition = "→ BUY ENTRY"
                signal_changes.append((idx, signal, 'BUY_ENTRY'))
            elif signal == -1:
                transition = "→ SELL ENTRY"  
                signal_changes.append((idx, signal, 'SELL_ENTRY'))
            elif signal == 0:
                if prev_signal == 1:
                    transition = "→ EXIT LONG"
                elif prev_signal == -1:
                    transition = "→ EXIT SHORT"
                signal_changes.append((idx, signal, 'EXIT'))
        
        signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
        print(f"{date:<12} ${price:<7.2f} {signal_text:<8} {transition:<12}")
    
    print(f"\n📊 SIGNAL TRANSITION SUMMARY:")
    print(f"   Total signal changes: {len(signal_changes)}")
    
    # Show key transitions
    print(f"\n🔄 KEY TRANSITIONS:")
    for date, signal, change_type in signal_changes[-10:]:  # Last 10 changes
        price = march_april.loc[date, 'close']
        print(f"   {date.strftime('%Y-%m-%d')}: ${price:.2f} → {change_type}")
    
    # Check what happens right before April crash
    april_1_idx = None
    for idx in march_april.index:
        if idx.month == 4 and idx.day == 1:
            april_1_idx = idx
            break
    
    if april_1_idx:
        april_1_pos = march_april.index.get_loc(april_1_idx)
        if april_1_pos > 0:
            pre_april_signal = march_april_signals.iloc[april_1_pos - 1]
            april_1_signal = march_april_signals.iloc[april_1_pos]
            
            print(f"\n🎯 CRITICAL TRANSITION (Mar 31 → Apr 1):")
            print(f"   Mar 31 signal: {pre_april_signal}")
            print(f"   Apr 1 signal: {april_1_signal}")
            print(f"   Transition type: {'NEW BUY' if pre_april_signal != 1 and april_1_signal == 1 else 'CONTINUED BUY' if pre_april_signal == 1 and april_1_signal == 1 else 'OTHER'}")

if __name__ == "__main__":
    debug_march_signals()
