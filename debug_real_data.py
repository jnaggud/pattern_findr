#!/usr/bin/env python3
"""
Debug using the ACTUAL data that optimization uses (with indicators)
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from indicators import get_all_indicators
from optimization import universal_strategy
from backtester import Backtester
import yfinance as yf

def debug_with_real_data():
    """Debug using the exact same data pipeline as the app"""
    
    print("🔍 DEBUGGING WITH REAL OPTIMIZATION DATA")
    print("="*60)
    
    try:
        # Use the exact same data pipeline as the app
        print("📊 Loading data like the app does...")
        data = yf.download("SPY", period="1y", interval="1d")
        data = data.reset_index()
        data.columns = ['date', 'open', 'high', 'low', 'close', 'volume'] 
        print(f"   Raw data shape: {data.shape}")
        print(f"   Raw columns: {list(data.columns)}")
        
        # Calculate indicators like the app does
        print("🧮 Calculating indicators like the app does...")
        enriched_data = get_all_indicators(data)
        print(f"   Enriched data shape: {enriched_data.shape}")
        print(f"   Total columns: {len(enriched_data.columns)}")
        
        # Check what indicators are available
        indicator_columns = [col for col in enriched_data.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume']]
        print(f"   Available indicators: {len(indicator_columns)}")
        print(f"   First 10 indicators: {indicator_columns[:10]}")
        
        # Use more realistic parameters that match available indicators
        available_indicators = indicator_columns[:10]  # Use first 10 available indicators
        params_realistic = {
            'buy_score_threshold': 3,  # Much lower threshold
            'sell_score_threshold': 3,  # Much lower threshold  
            'min_hold_days': 1,
            'require_confirmation': False,
            'use_trend_filter': True,
            'adx_threshold': 18,
        }
        
        # Add available indicators to params
        for i, indicator in enumerate(available_indicators):
            params_realistic[f'use_{indicator}'] = True
            # Set realistic buy/sell thresholds based on data
            if indicator in enriched_data.columns:
                min_val = enriched_data[indicator].min()
                max_val = enriched_data[indicator].max()
                mid_val = (min_val + max_val) / 2
                params_realistic[f'{indicator}_buy'] = min_val + (max_val - min_val) * 0.3
                params_realistic[f'{indicator}_sell'] = min_val + (max_val - min_val) * 0.7
        
        print(f"\n🎯 Using realistic parameters with {len(available_indicators)} indicators")
        print(f"   Buy/sell thresholds: {params_realistic['buy_score_threshold']}/{params_realistic['sell_score_threshold']}")
        
        # Test the strategy
        print("\n" + "="*50)
        print("TESTING WITH REAL ENRICHED DATA")
        print("="*50)
        
        signals = universal_strategy(enriched_data, params_realistic)
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        
        print(f"📈 Signals generated: {buy_signals} buy, {sell_signals} sell")
        print(f"📊 Signal distribution: {signals.value_counts().to_dict()}")
        
        # Test with backtester
        backtester = Backtester(enriched_data, "Real Test", universal_strategy, params_realistic)
        backtester.run()
        trades, summary = backtester.get_results()
        
        print(f"\n💰 Backtester results:")
        print(f"   Total Return: {summary['total_return_pct']:.2f}%")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Win Rate: {summary['win_rate']:.1f}%")
        
        if summary['total_trades'] > 0:
            print("✅ SUCCESS: Real data with indicators produces trades!")
        else:
            print("❌ STILL FAILING: Even with real data, no trades generated")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_with_real_data()
