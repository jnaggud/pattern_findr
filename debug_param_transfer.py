#!/usr/bin/env python3
"""
Debug the exact parameter transfer issue between optimization and UI
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from optimization import universal_strategy
from backtester import Backtester

def debug_exact_discrepancy():
    """Debug why the same strategy gives different results in optimization vs UI"""
    
    print("🔍 DEBUGGING EXACT PARAMETER TRANSFER DISCREPANCY")
    print("="*70)
    
    # Load the actual SPY data that the optimization uses
    try:
        # Try to load the same data the optimization is using
        data_files = [
            "data_cache/SPY_1y_1d.csv",
            "SPY_data.csv",
            "data/SPY_1y_1d.csv"
        ]
        
        data = None
        for file_path in data_files:
            try:
                data = pd.read_csv(file_path)
                print(f"✅ Loaded data from {file_path}")
                print(f"   Data shape: {data.shape}")
                print(f"   Columns: {list(data.columns)[:10]}...")
                break
            except FileNotFoundError:
                continue
                
        if data is None:
            print("❌ Could not load actual data files, using synthetic data")
            # Create synthetic data as fallback
            np.random.seed(42)
            data = pd.DataFrame({
                'close': np.random.randn(250).cumsum() + 100,
                'high': np.random.randn(250).cumsum() + 102,  
                'low': np.random.randn(250).cumsum() + 98,
                'open': np.random.randn(250).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 250),
            })
            # Add many indicators to match real scenario
            for i in range(50):
                data[f'indicator_{i}'] = np.random.uniform(-1, 1, 250)
            for i in range(20):  
                data[f'pattern_{i}'] = np.random.choice([True, False], 250)
        
        # Simulate the EXACT parameters from your best strategy
        params_from_console = {
            'buy_score_threshold': 18,
            'sell_score_threshold': 15,
            'min_hold_days': 1,
            'require_confirmation': False,
            'use_trend_filter': True,
            'adx_threshold': 18,
            # Add some indicators that should exist
            'use_RSI_14': True,
            'RSI_14_buy': 30.0,
            'RSI_14_sell': 70.0,
            'use_MACD_12_26_9': True, 
            'MACD_12_26_9_buy': -0.5,
            'MACD_12_26_9_sell': 0.5,
            'use_PPO_12_26_9': True,
            'PPO_12_26_9_buy': -2.0,
            'PPO_12_26_9_sell': 2.0,
        }
        
        print(f"\n📊 Testing with data shape: {data.shape}")
        print(f"🎯 Using console parameters: buy_threshold={params_from_console['buy_score_threshold']}, sell_threshold={params_from_console['sell_score_threshold']}")
        
        # Method 1: Direct strategy execution (optimization style)
        print("\n" + "="*50)
        print("METHOD 1: Optimization-Style Execution")  
        print("="*50)
        
        signals_method1 = universal_strategy(data, params_from_console)
        buy_signals_1 = (signals_method1 == 1).sum()
        sell_signals_1 = (signals_method1 == -1).sum()
        
        print(f"📈 Method 1 signals: {buy_signals_1} buy, {sell_signals_1} sell")
        
        # Method 2: Backtester execution (UI style)  
        print("\n" + "="*50)
        print("METHOD 2: UI-Style Backtester Execution")
        print("="*50)
        
        # Reset the call counter to see if that affects anything
        if hasattr(universal_strategy, 'call_count'):
            delattr(universal_strategy, 'call_count')
        
        backtester = Backtester(data, "Debug Strategy", universal_strategy, params_from_console)
        backtester.run()
        trades, summary = backtester.get_results()
        
        print(f"💰 Method 2 results:")
        print(f"   Total Return: {summary['total_return_pct']:.2f}%")
        print(f"   Total Trades: {summary['total_trades']}")  
        print(f"   Win Rate: {summary['win_rate']:.1f}%")
        
        # Method 3: Check signal generation in backtester
        print("\n" + "="*50)
        print("METHOD 3: Direct Signal Check in Backtester Context")
        print("="*50)
        
        # Reset counter again
        if hasattr(universal_strategy, 'call_count'):
            delattr(universal_strategy, 'call_count')
            
        signals_method3 = universal_strategy(data, params_from_console)
        buy_signals_3 = (signals_method3 == 1).sum()
        sell_signals_3 = (signals_method3 == -1).sum()
        
        print(f"📈 Method 3 signals: {buy_signals_3} buy, {sell_signals_3} sell")
        
        # Analysis
        print("\n" + "="*50)
        print("🔍 DISCREPANCY ANALYSIS")
        print("="*50)
        
        print(f"Method 1 vs Method 3 signals match: {buy_signals_1 == buy_signals_3 and sell_signals_1 == sell_signals_3}")
        print(f"Signal consistency: {buy_signals_1 == buy_signals_3 == buy_signals_3}")
        
        if summary['total_trades'] == 0:
            print("❌ CRITICAL: Backtester produced 0 trades despite signals!")
            print("   This suggests a bug in signal-to-trade conversion")
        
        if buy_signals_1 != buy_signals_3:
            print("❌ CRITICAL: Same parameters produced different signals!")
            print("   This suggests state is being modified between calls")
            
        print(f"\n📊 Expected vs Actual:")
        print(f"   Console claimed: 26.61% return")
        print(f"   Backtester actual: {summary['total_return_pct']:.2f}% return")
        print(f"   Discrepancy: {26.61 - summary['total_return_pct']:.2f}%")
        
    except Exception as e:
        print(f"❌ ERROR during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_exact_discrepancy()
