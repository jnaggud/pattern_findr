#!/usr/bin/env python3
"""
Debug script to find the discrepancy between optimization scores and actual backtesting results.
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from backtester import Backtester
from optimization import universal_strategy

def debug_strategy_execution():
    """Test a strategy both ways to see where the discrepancy comes from"""
    
    print("🔍 DEBUGGING STRATEGY EXECUTION DISCREPANCY")
    print("="*60)
    
    # Create sample data (simplified)
    np.random.seed(42)
    data = pd.DataFrame({
        'close': np.random.randn(250).cumsum() + 100,
        'high': np.random.randn(250).cumsum() + 102,  
        'low': np.random.randn(250).cumsum() + 98,
        'open': np.random.randn(250).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 250),
        'RSI_14': np.random.uniform(20, 80, 250),
        'MACD_12_26_9': np.random.uniform(-2, 2, 250),
        'MACDh_12_26_9': np.random.uniform(-1, 1, 250),
    })
    
    # Simulate the exact parameters from your best strategy
    params = {
        'buy_score_threshold': 6,
        'sell_score_threshold': 20,
        'min_hold_days': 1,
        'require_confirmation': False,
        'use_trend_filter': True,
        'adx_threshold': 18,
        'use_RSI_14': True,
        'RSI_14_buy': 30.0,  # Example optimized values
        'RSI_14_sell': 70.0,
        'use_MACD_12_26_9': True,
        'MACD_12_26_9_buy': -0.5,
        'MACD_12_26_9_sell': 0.5,
        'use_MACDh_12_26_9': True,
        'MACDh_12_26_9_buy': -0.2,
        'MACDh_12_26_9_sell': 0.2,
    }
    
    print(f"📊 Data shape: {data.shape}")
    print(f"🎯 Strategy params: buy_threshold={params['buy_score_threshold']}, sell_threshold={params['sell_score_threshold']}")
    
    try:
        # Method 1: Direct strategy call (like optimization does)
        print("\n" + "="*40)
        print("METHOD 1: Direct Strategy Call")
        print("="*40)
        
        signals = universal_strategy(data, params)
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        
        print(f"📈 Signals generated: {buy_signals} buy, {sell_signals} sell")
        print(f"📊 Signal distribution: {signals.value_counts().to_dict()}")
        
        # Method 2: Backtester execution (like UI does)
        print("\n" + "="*40)
        print("METHOD 2: Backtester Execution")
        print("="*40)
        
        strategy_name = "Debug Strategy"
        backtester = Backtester(data, strategy_name, universal_strategy, params)
        backtester.run()
        trades, summary = backtester.get_results()
        
        print(f"💰 Backtester results:")
        print(f"   Total Return: {summary['total_return_pct']:.2f}%")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Win Rate: {summary['win_rate']:.1f}%")
        print(f"   Starting Capital: ${summary['starting_capital']:,.2f}")
        print(f"   Ending Capital: ${summary['ending_capital']:,.2f}")
        
        # Method 3: Check if signals match trades
        print("\n" + "="*40)
        print("METHOD 3: Signal vs Trade Analysis")
        print("="*40)
        
        if len(trades) > 0:
            print(f"🔍 First 5 trades:")
            for i, trade in enumerate(trades.head(5).itertuples()):
                print(f"   Trade {i+1}: {trade.action} on {trade.date} at ${trade.price:.2f}")
        else:
            print("❌ No trades found despite signals!")
            
        # Check if there's a mismatch
        expected_trades = min(buy_signals, sell_signals) * 2  # Rough estimate
        actual_trades = summary['total_trades']
        
        print(f"\n📊 DISCREPANCY ANALYSIS:")
        print(f"   Buy signals: {buy_signals}")
        print(f"   Sell signals: {sell_signals}")
        print(f"   Expected trades (rough): {expected_trades}")
        print(f"   Actual trades: {actual_trades}")
        print(f"   Signal-to-trade conversion: {actual_trades/max(1, min(buy_signals, sell_signals))*100:.1f}%")
        
        if actual_trades < expected_trades * 0.1:
            print("🚨 MAJOR ISSUE: Very few signals are converting to actual trades!")
        
    except Exception as e:
        print(f"❌ ERROR during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_strategy_execution()
