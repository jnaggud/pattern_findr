#!/usr/bin/env python3
"""
Debug the exact same strategy execution to prove determinism
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from optimization import universal_strategy
from backtester import Backtester
import yfinance as yf
from indicators import get_all_indicators

def test_exact_determinism():
    """Test if the same parameters produce the same results every time"""
    
    print("🔍 TESTING EXACT DETERMINISM")
    print("="*50)
    
    # Load the same data as the app
    data = yf.download("SPY", period="1y", interval="1d")
    data = data.reset_index()
    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    enriched_data = get_all_indicators(data)
    
    print(f"📊 Data shape: {enriched_data.shape}")
    
    # Use the EXACT parameters from your best strategy
    params = {
        'buy_score_threshold': 8,
        'sell_score_threshold': 9,
        'min_hold_days': 1,
        'require_confirmation': False,
        'use_trend_filter': True,
        'adx_threshold': 18,
        # Add the first few indicators that should exist
        'use_RSI_14': True,
        'RSI_14_buy': 30.0,
        'RSI_14_sell': 70.0,
        'use_MACD_12_26_9': True,
        'MACD_12_26_9_buy': -0.5,
        'MACD_12_26_9_sell': 0.5,
        'use_MACDs_12_26_9': True,
        'MACDs_12_26_9_buy': -0.2,
        'MACDs_12_26_9_sell': 0.2,
    }
    
    print(f"🎯 Testing with thresholds: buy={params['buy_score_threshold']}, sell={params['sell_score_threshold']}")
    
    # Reset the call counter to ensure consistent state
    if hasattr(universal_strategy, 'call_count'):
        delattr(universal_strategy, 'call_count')
    
    results = []
    for run in range(3):
        print(f"\n{'='*30}")
        print(f"RUN {run + 1}")
        print(f"{'='*30}")
        
        # Reset counter for each run
        if hasattr(universal_strategy, 'call_count'):
            delattr(universal_strategy, 'call_count')
            
        # Method 1: Direct strategy call
        signals = universal_strategy(enriched_data, params)
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        
        # Method 2: Backtester execution  
        if hasattr(universal_strategy, 'call_count'):
            delattr(universal_strategy, 'call_count')
            
        backtester = Backtester(enriched_data, f"Test Run {run+1}", universal_strategy, params)
        backtester.run()
        trades, summary = backtester.get_results()
        
        result = {
            'run': run + 1,
            'signals_buy': buy_count,
            'signals_sell': sell_count,
            'trades': summary['total_trades'],
            'return_pct': summary['total_return_pct'],
            'ending_capital': summary['ending_capital']
        }
        results.append(result)
        
        print(f"📈 Signals: {buy_count} buy, {sell_count} sell")
        print(f"💰 Trades: {summary['total_trades']}")
        print(f"📊 Return: {summary['total_return_pct']:.2f}%")
        print(f"💵 Ending: ${summary['ending_capital']:,.2f}")
    
    print(f"\n{'='*50}")  
    print("🔍 DETERMINISM ANALYSIS")
    print(f"{'='*50}")
    
    # Check if all results are identical
    first_result = results[0]
    all_identical = all(
        r['signals_buy'] == first_result['signals_buy'] and
        r['signals_sell'] == first_result['signals_sell'] and  
        r['trades'] == first_result['trades'] and
        abs(r['return_pct'] - first_result['return_pct']) < 0.01 and
        abs(r['ending_capital'] - first_result['ending_capital']) < 0.01
        for r in results
    )
    
    if all_identical:
        print("✅ PERFECT DETERMINISM: All runs produced identical results!")
        print("   The discrepancy must be in parameter transfer or data differences")
    else:
        print("❌ NON-DETERMINISTIC BEHAVIOR DETECTED:")
        for i, result in enumerate(results):
            print(f"   Run {i+1}: {result['return_pct']:.2f}% (${result['ending_capital']:,.2f})")
        print("   This explains the console vs UI discrepancy!")
    
    return results

if __name__ == "__main__":
    test_exact_determinism()
