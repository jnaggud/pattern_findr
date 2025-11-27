#!/usr/bin/env python3
"""
Debug why trades aren't being executed during April crash despite buy signals
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
from indicators import get_all_indicators
from optimization import universal_strategy
from backtester import Backtester
import json
import pandas as pd

def debug_trade_execution():
    print("🔍 DEBUGGING TRADE EXECUTION vs SIGNALS")
    print("=" * 60)
    
    # Get data and add indicators
    data = yf.Ticker('SPY').history(period='1y', interval='1d')
    data.reset_index(inplace=True)
    data.columns = [col.lower() for col in data.columns]
    enriched_data = get_all_indicators(data)
    
    # Load the latest optimization parameters
    with open('optimizations/SPY_20251124_180031_trial77_ret46.4pct_params.json', 'r') as f:
        params = json.load(f)
    
    print(f"📊 Using parameters: buy_threshold={params.get('buy_score_threshold')}, sell_threshold={params.get('sell_score_threshold')}")
    
    # Generate signals using universal_strategy
    strategy_result = universal_strategy(enriched_data, params)
    signals = strategy_result['signals'] if isinstance(strategy_result, dict) else strategy_result
    buy_scores = strategy_result.get('buy_score') if isinstance(strategy_result, dict) else None
    
    # Focus on April 2025
    april_mask = enriched_data.index[enriched_data.index.to_series().dt.month == 4]
    if len(april_mask) == 0:
        print("❌ No April 2025 data found")
        return
    
    print(f"\n📅 APRIL 2025 SIGNAL ANALYSIS:")
    print("-" * 60)
    print(f"{'Date':<12} {'Price':<8} {'Signal':<8} {'BuyScore':<10} {'Executed?':<10}")
    print("-" * 60)
    
    april_signals = signals.loc[april_mask]
    april_scores = buy_scores.loc[april_mask] if buy_scores is not None else None
    april_data = enriched_data.loc[april_mask]
    
    buy_signal_dates = []
    for i, (idx, signal) in enumerate(april_signals.items()):
        date = idx.strftime('%m-%d')
        price = april_data.loc[idx, 'close']
        score = april_scores.loc[idx] if april_scores is not None else 'N/A'
        
        if signal == 1:  # Buy signal
            buy_signal_dates.append(idx)
            print(f"{date:<12} ${price:<7.2f} {'BUY':<8} {score:<10.1f} {'?':<10}")
        elif signal == -1:  # Sell signal  
            print(f"{date:<12} ${price:<7.2f} {'SELL':<8} {score if isinstance(score, str) else 'N/A':<10} {'?':<10}")
    
    print(f"\n📊 SIGNALS SUMMARY:")
    print(f"   Buy signals in April: {(april_signals == 1).sum()}")
    print(f"   Sell signals in April: {(april_signals == -1).sum()}")
    
    # Now test the backtester
    print(f"\n🔄 TESTING BACKTESTER EXECUTION:")
    print("-" * 60)
    
    backtester = Backtester(enriched_data, "Test Strategy", universal_strategy, params, 100000)
    backtester.run()
    trade_log, summary = backtester.get_results()
    
    print(f"📈 BACKTESTER RESULTS:")
    print(f"   Total trades executed: {len(trade_log)}")
    print(f"   Total return: {summary['total_return_pct']:.2f}%")
    
    if len(trade_log) > 0:
        print(f"\n📋 EXECUTED TRADES:")
        print(f"{'Entry Date':<12} {'Entry Price':<12} {'Exit Date':<12} {'Exit Price':<12} {'Profit':<8} {'Type':<12}")
        print("-" * 80)
        
        april_trades = []
        for _, trade in trade_log.iterrows():
            # Handle both string and datetime entry_date
            if isinstance(trade['entry_date'], str):
                entry_date = pd.to_datetime(trade['entry_date'])
            else:
                entry_date = trade['entry_date']
                
            exit_date = None
            if pd.notna(trade['exit_date']):
                if isinstance(trade['exit_date'], str):
                    exit_date = pd.to_datetime(trade['exit_date'])
                else:
                    exit_date = trade['exit_date']
            
            # Check if trade entry is in April 2025
            if entry_date.month == 4 and entry_date.year == 2025:
                april_trades.append(trade)
                exit_str = exit_date.strftime('%m-%d') if exit_date else 'Open'
                exit_price_str = f"${trade['exit_price']:.2f}" if pd.notna(trade['exit_price']) else 'N/A'
                profit_str = f"${trade['profit']:.0f}" if pd.notna(trade['profit']) else 'Open'
                trade_type = trade.get('trade_type', 'STANDARD')
                
                print(f"{entry_date.strftime('%m-%d'):<12} ${trade['entry_price']:<11.2f} {exit_str:<12} {exit_price_str:<12} {profit_str:<8} {trade_type:<12}")
        
        print(f"\n📊 APRIL TRADE SUMMARY:")
        print(f"   Trades entered in April: {len(april_trades)}")
        
        if len(april_trades) == 0:
            print(f"   ❌ NO TRADES EXECUTED IN APRIL despite {(april_signals == 1).sum()} buy signals!")
            
            # Debug why no trades
            print(f"\n🔍 DEBUGGING WHY NO APRIL TRADES:")
            
            # Check if signals align with trade execution logic
            for signal_date in buy_signal_dates[:3]:  # Check first 3 buy signals
                signal_idx = enriched_data.index.get_loc(signal_date)
                if signal_idx > 0:  # Make sure we're not at first row
                    prev_signal = signals.iloc[signal_idx - 1]
                    curr_signal = signals.iloc[signal_idx]
                    
                    print(f"   Signal {signal_date.strftime('%Y-%m-%d')}:")
                    print(f"     Previous signal: {prev_signal}")
                    print(f"     Current signal: {curr_signal}")
                    print(f"     Should trigger trade: {curr_signal == 1 and prev_signal != 1}")
        else:
            print(f"   ✅ April trades executed successfully!")
    else:
        print(f"   ❌ NO TRADES EXECUTED AT ALL!")
        print(f"   This suggests a fundamental backtester issue")

if __name__ == "__main__":
    debug_trade_execution()
