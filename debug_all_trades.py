#!/usr/bin/env python3
"""
Debug ALL trades to see where they're happening and what's in the data
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

def debug_all_trades():
    print("🔍 DEBUGGING ALL TRADES - COMPLETE ANALYSIS")
    print("=" * 80)
    
    # Get data and add indicators
    data = yf.Ticker('SPY').history(period='1y', interval='1d')
    data.reset_index(inplace=True)
    data.columns = [col.lower() for col in data.columns]
    enriched_data = get_all_indicators(data)
    
    # Load parameters
    with open('optimizations/SPY_20251124_180031_trial77_ret46.4pct_params.json', 'r') as f:
        params = json.load(f)
    
    # Run backtester
    backtester = Backtester(enriched_data, "Debug Strategy", universal_strategy, params, 100000)
    backtester.run()
    trade_log, summary = backtester.get_results()
    
    print(f"📊 BACKTESTER SUMMARY:")
    print(f"   Total trades: {len(trade_log)}")
    print(f"   Total return: {summary['total_return_pct']:.2f}%")
    
    if len(trade_log) > 0:
        print(f"\n📋 ALL TRADES BREAKDOWN:")
        print(f"{'#':<3} {'Entry Date':<12} {'Entry Price':<12} {'Exit Date':<12} {'Type':<15} {'Profit':<10}")
        print("-" * 85)
        
        march_trades = 0
        april_trades = 0
        dca_trades = 0
        
        for i, (_, trade) in enumerate(trade_log.iterrows()):
            # Handle entry_date
            if isinstance(trade['entry_date'], str):
                entry_date = pd.to_datetime(trade['entry_date'])
            else:
                entry_date = trade['entry_date']
                
            # Handle exit_date
            exit_date = None
            if pd.notna(trade['exit_date']):
                if isinstance(trade['exit_date'], str):
                    exit_date = pd.to_datetime(trade['exit_date'])
                else:
                    exit_date = trade['exit_date']
            
            # Get trade type
            trade_type = trade.get('trade_type', 'STANDARD')
            if trade_type == 'DCA_ADD':
                dca_trades += 1
            
            # Count by month
            if entry_date.month == 3 and entry_date.year == 2025:
                march_trades += 1
            elif entry_date.month == 4 and entry_date.year == 2025:
                april_trades += 1
            
            # Format output
            exit_str = exit_date.strftime('%Y-%m-%d') if exit_date else 'Open'
            profit_str = f"${trade['profit']:.0f}" if pd.notna(trade['profit']) else 'Open'
            
            print(f"{i+1:<3} {entry_date.strftime('%Y-%m-%d'):<12} ${trade['entry_price']:<11.2f} {exit_str:<12} {trade_type:<15} {profit_str:<10}")
        
        print(f"\n📊 TRADE DISTRIBUTION:")
        print(f"   March 2025 trades: {march_trades}")
        print(f"   April 2025 trades: {april_trades}")  
        print(f"   DCA additions: {dca_trades}")
        print(f"   Total trades: {len(trade_log)}")
        
        # Check a few specific April dates
        april_dates_to_check = ['2025-04-01', '2025-04-03', '2025-04-08', '2025-04-09']
        print(f"\n🎯 CHECKING SPECIFIC APRIL DATES:")
        
        for check_date in april_dates_to_check:
            check_dt = pd.to_datetime(check_date)
            found_trades = []
            
            for _, trade in trade_log.iterrows():
                if isinstance(trade['entry_date'], str):
                    entry_date = pd.to_datetime(trade['entry_date'])
                else:
                    entry_date = trade['entry_date']
                
                if entry_date.date() == check_dt.date():
                    found_trades.append(trade)
            
            if found_trades:
                print(f"   {check_date}: {len(found_trades)} trades found")
                for trade in found_trades:
                    trade_type = trade.get('trade_type', 'STANDARD')
                    print(f"      → ${trade['entry_price']:.2f} ({trade_type})")
            else:
                print(f"   {check_date}: No trades")
    
    else:
        print(f"❌ NO TRADES FOUND AT ALL!")

if __name__ == "__main__":
    debug_all_trades()
