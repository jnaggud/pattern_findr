#!/usr/bin/env python3
"""
Debug why there was no sell signal around March 26th
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

def debug_march_26_sell():
    print("🔍 DEBUGGING MARCH 26 SELL SIGNAL MISSING")
    print("=" * 60)
    
    # Get data and add indicators
    data = yf.Ticker('SPY').history(period='1y', interval='1d')
    data.reset_index(inplace=True)
    data.columns = [col.lower() for col in data.columns]
    enriched_data = get_all_indicators(data)
    
    # Load parameters
    with open('optimizations/SPY_20251124_180031_trial77_ret46.4pct_params.json', 'r') as f:
        params = json.load(f)
    
    print(f"📊 Parameters: buy_threshold={params.get('buy_score_threshold')}, sell_threshold={params.get('sell_score_threshold')}")
    
    # Generate signals and scores
    strategy_result = universal_strategy(enriched_data, params)
    signals = strategy_result['signals'] if isinstance(strategy_result, dict) else strategy_result
    buy_scores = strategy_result.get('buy_score') if isinstance(strategy_result, dict) else None
    sell_scores = strategy_result.get('sell_score') if isinstance(strategy_result, dict) else None
    
    # Focus on March period around entry and expected exit
    march_data = enriched_data[(enriched_data.index.to_series().dt.month == 3) & 
                              (enriched_data.index.to_series().dt.year == 2025)]
    
    if len(march_data) == 0:
        print("❌ No March 2025 data found")
        return
    
    march_signals = signals.loc[march_data.index]
    march_buy_scores = buy_scores.loc[march_data.index] if buy_scores is not None else None
    march_sell_scores = sell_scores.loc[march_data.index] if sell_scores is not None else None
    
    print(f"\n📅 MARCH 2025 DETAILED ANALYSIS:")
    print("-" * 80)
    print(f"{'Date':<12} {'Price':<8} {'Signal':<8} {'BuyScore':<10} {'SellScore':<10} {'Analysis':<20}")
    print("-" * 80)
    
    entry_date = None
    entry_price = None
    
    for i, (idx, signal) in enumerate(march_signals.items()):
        date = idx.strftime('%m-%d')
        price = march_data.loc[idx, 'close']
        buy_score = march_buy_scores.loc[idx] if march_buy_scores is not None else 'N/A'
        sell_score = march_sell_scores.loc[idx] if march_sell_scores is not None else 'N/A'
        
        # Track entry point
        if date == '03-10':
            entry_date = idx
            entry_price = price
            analysis = "🎯 ENTRY POINT"
        elif date == '03-26':
            if entry_price:
                profit_pct = (price / entry_price - 1) * 100
                analysis = f"💰 +{profit_pct:.1f}% vs entry"
            else:
                analysis = "Expected EXIT"
        else:
            analysis = ""
        
        signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
        buy_str = f"{buy_score:.1f}" if isinstance(buy_score, (int, float)) else str(buy_score)
        sell_str = f"{sell_score:.1f}" if isinstance(sell_score, (int, float)) else str(sell_score)
        
        print(f"{date:<12} ${price:<7.2f} {signal_text:<8} {buy_str:<10} {sell_str:<10} {analysis:<20}")
    
    # Focus on March 26th specifically
    march_26 = None
    for idx in march_data.index:
        if idx.day == 26:
            march_26 = idx
            break
    
    if march_26:
        print(f"\n🎯 MARCH 26TH ANALYSIS:")
        print("-" * 40)
        
        march_26_price = march_data.loc[march_26, 'close']
        march_26_signal = march_signals.loc[march_26]
        march_26_buy_score = march_buy_scores.loc[march_26] if march_buy_scores is not None else 0
        march_26_sell_score = march_sell_scores.loc[march_26] if march_sell_scores is not None else 0
        
        sell_threshold = params.get('sell_score_threshold', 5)
        buy_threshold = params.get('buy_score_threshold', 1)
        
        print(f"   Price: ${march_26_price:.2f}")
        print(f"   Signal: {march_26_signal} ({'BUY' if march_26_signal == 1 else 'SELL' if march_26_signal == -1 else 'HOLD'})")
        print(f"   Buy score: {march_26_buy_score:.1f} (threshold: {buy_threshold})")
        print(f"   Sell score: {march_26_sell_score:.1f} (threshold: {sell_threshold})")
        
        if entry_price:
            profit_pct = (march_26_price / entry_price - 1) * 100
            print(f"   Profit vs Mar 10: +{profit_pct:.1f}%")
        
        print(f"\n🔍 WHY NO SELL SIGNAL?")
        if march_26_sell_score < sell_threshold:
            print(f"   ❌ Sell score ({march_26_sell_score:.1f}) < threshold ({sell_threshold})")
            print(f"   Need {sell_threshold - march_26_sell_score:.1f} more points to trigger sell")
        else:
            print(f"   ✅ Sell score ({march_26_sell_score:.1f}) >= threshold ({sell_threshold})")
            print(f"   🤔 Sell should have triggered - check conflict resolution!")
        
        # Check if there was a conflict that day
        if march_26_buy_score >= buy_threshold and march_26_sell_score >= sell_threshold:
            print(f"   ⚠️  CONFLICT: Both buy and sell triggered!")
            print(f"   Buy excess: {(march_26_buy_score / buy_threshold - 1) * 100:.1f}%")
            print(f"   Sell excess: {(march_26_sell_score / sell_threshold - 1) * 100:.1f}%")
            
            if (march_26_sell_score / sell_threshold - 1) > (march_26_buy_score / buy_threshold - 1):
                print(f"   → Sell should have won conflict resolution")
            else:
                print(f"   → Buy won conflict resolution")
    
    # Check what indicators were overbought around March 26
    print(f"\n📈 MARKET CONDITION INDICATORS ON MARCH 26:")
    if march_26:
        rsi = march_data.loc[march_26].get('RSI_14', 'N/A')
        willr = march_data.loc[march_26].get('WILLR_14', 'N/A')
        macd = march_data.loc[march_26].get('MACD_12_26_9', 'N/A')
        
        print(f"   RSI(14): {rsi}")
        print(f"   Williams %R: {willr}")
        print(f"   MACD: {macd}")
        
        if isinstance(rsi, (int, float)) and rsi > 70:
            print(f"   🔴 RSI overbought (>{70}) - should favor selling")
        if isinstance(willr, (int, float)) and willr > -20:
            print(f"   🔴 Williams %R overbought (>-20) - should favor selling")

if __name__ == "__main__":
    debug_march_26_sell()
