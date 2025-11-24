#!/usr/bin/env python3
"""
Detailed analysis of why the strategy missed April 2025 bottom using exact optimized parameters.
This will show day-by-day indicator values and signal generation.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# Import the actual strategy
from optimization import universal_strategy
from indicators import get_all_indicators

def analyze_strategy_miss():
    """Analyze why the optimized strategy missed April 2025 bottom"""
    print("🔍 DETAILED APRIL 2025 STRATEGY MISS ANALYSIS")
    print("="*60)
    
    # Load the same data period as shown in the chart
    spy = yf.Ticker("SPY")
    data = spy.history(period="1y", interval="1d")  # Last year of data
    data.reset_index(inplace=True)
    data.columns = [col.lower() for col in data.columns]
    
    print(f"📊 Loaded {len(data)} days of SPY data")
    print(f"📅 Date range: {data['date'].min()} to {data['date'].max()}")
    
    # Add all indicators (same as optimization)
    enriched_data = get_all_indicators(data)
    print(f"📈 Added indicators, now have {len(enriched_data.columns)} columns")
    
    # Use the EXACT parameters from your optimized strategy
    strategy_params = {
        # Core voting thresholds (from your results)
        'buy_score_threshold': 3,  # Needs 3 indicators to agree
        'sell_score_threshold': 2,  # Needs 2 indicators to agree
        'signal_persistence_days': 2,
        
        # Active indicators (from your strategy)
        'use_RSI_14': True,
        'use_MACD_12_26_9': True,
        'use_MACDh_12_26_9': True,
        'use_MACDs_12_26_9': True,
        
        # Add more key indicators that might be active
        'use_WILLR_14': True,
        'use_STOCH_14_3_3': True,
        'use_CCI_14': True,
        'use_MFI_14': True,
        'use_pattern_bullish_engulfing': True,
        'use_pattern_bearish_engulfing': True,
        'use_pattern_hammer': True,
        'use_pattern_doji': True,
        
        # Indicator thresholds (reasonable defaults - will optimize these)
        'RSI_14_buy': 30,
        'RSI_14_sell': 70,
        'MACD_12_26_9_buy': -1.0,
        'MACD_12_26_9_sell': 1.0,
        'MACDh_12_26_9_buy': -0.5,
        'MACDh_12_26_9_sell': 0.5,
        'MACDs_12_26_9_buy': -0.5,
        'MACDs_12_26_9_sell': 0.5,
        'WILLR_14_buy': -80,
        'WILLR_14_sell': -20,
        'STOCH_14_3_3_buy': 20,
        'STOCH_14_3_3_sell': 80,
        'CCI_14_buy': -100,
        'CCI_14_sell': 100,
        'MFI_14_buy': 20,
        'MFI_14_sell': 80,
        
        # Trend filter (may be blocking signals)
        'use_trend_filter': True,
        'trend_rsi_threshold': 35,
        'trend_adx_threshold': 20,
        'trend_willr_threshold': -80,
        'trend_stoch_threshold': 15,
        'trend_sma_period': 50
    }
    
    print(f"\n🎯 STRATEGY PARAMETERS:")
    print(f"   Buy Threshold: {strategy_params['buy_score_threshold']} indicators must agree")
    print(f"   Sell Threshold: {strategy_params['sell_score_threshold']} indicators must agree")
    print(f"   Trend Filter: {'ENABLED' if strategy_params['use_trend_filter'] else 'DISABLED'}")
    
    # Generate signals using the exact strategy
    signals = universal_strategy(enriched_data, strategy_params)
    
    # Focus on April 2025
    enriched_data['date'] = pd.to_datetime(enriched_data['date'])
    april_mask = (enriched_data['date'] >= '2025-04-01') & (enriched_data['date'] <= '2025-04-15')
    april_data = enriched_data[april_mask].copy()
    april_signals = signals[april_mask]
    
    if len(april_data) == 0:
        print("❌ No April 2025 data found!")
        return
    
    print(f"\n📅 APRIL 2025 DAY-BY-DAY ANALYSIS:")
    print("-" * 60)
    
    # Analyze each day in detail
    for i, (_, row) in enumerate(april_data.iterrows()):
        if i >= len(april_signals):
            break
            
        date_str = row['date'].strftime('%Y-%m-%d (%a)')
        signal = april_signals.iloc[i]
        signal_text = "🔴 SELL" if signal == -1 else "🟢 BUY" if signal == 1 else "⚪ HOLD"
        price_change = ((row['close'] / row['open'] - 1) * 100)
        
        print(f"\n{date_str}: {signal_text}")
        print(f"   💰 Price: ${row['close']:.2f} (Change: {price_change:+.2f}%)")
        
        # Check each indicator's buy signal
        buy_signals = 0
        indicator_details = []
        
        # RSI
        if 'RSI_14' in row and not pd.isna(row['RSI_14']):
            rsi_val = row['RSI_14']
            rsi_buy = rsi_val < strategy_params['RSI_14_buy']
            if rsi_buy:
                buy_signals += 1
                indicator_details.append(f"RSI(14): {rsi_val:.1f} < {strategy_params['RSI_14_buy']} ✅")
            else:
                indicator_details.append(f"RSI(14): {rsi_val:.1f} ≥ {strategy_params['RSI_14_buy']} ❌")
        
        # Williams %R
        if 'WILLR_14' in row and not pd.isna(row['WILLR_14']):
            willr_val = row['WILLR_14']
            willr_buy = willr_val < strategy_params['WILLR_14_buy']
            if willr_buy:
                buy_signals += 1
                indicator_details.append(f"WillR(14): {willr_val:.1f} < {strategy_params['WILLR_14_buy']} ✅")
            else:
                indicator_details.append(f"WillR(14): {willr_val:.1f} ≥ {strategy_params['WILLR_14_buy']} ❌")
        
        # MACD
        if 'MACD_12_26_9' in row and not pd.isna(row['MACD_12_26_9']):
            macd_val = row['MACD_12_26_9']
            macd_buy = macd_val > strategy_params['MACD_12_26_9_buy']
            if macd_buy:
                buy_signals += 1
                indicator_details.append(f"MACD: {macd_val:.3f} > {strategy_params['MACD_12_26_9_buy']} ✅")
            else:
                indicator_details.append(f"MACD: {macd_val:.3f} ≤ {strategy_params['MACD_12_26_9_buy']} ❌")
        
        # Stochastic
        if 'STOCH_14_3_3' in row and not pd.isna(row['STOCH_14_3_3']):
            stoch_val = row['STOCH_14_3_3']
            stoch_buy = stoch_val < strategy_params['STOCH_14_3_3_buy']
            if stoch_buy:
                buy_signals += 1
                indicator_details.append(f"Stoch: {stoch_val:.1f} < {strategy_params['STOCH_14_3_3_buy']} ✅")
            else:
                indicator_details.append(f"Stoch: {stoch_val:.1f} ≥ {strategy_params['STOCH_14_3_3_buy']} ❌")
        
        # CCI
        if 'CCI_14' in row and not pd.isna(row['CCI_14']):
            cci_val = row['CCI_14']
            cci_buy = cci_val < strategy_params['CCI_14_buy']
            if cci_buy:
                buy_signals += 1
                indicator_details.append(f"CCI: {cci_val:.1f} < {strategy_params['CCI_14_buy']} ✅")
            else:
                indicator_details.append(f"CCI: {cci_val:.1f} ≥ {strategy_params['CCI_14_buy']} ❌")
        
        # Check patterns
        patterns_triggered = []
        if 'pattern_hammer' in row and row['pattern_hammer']:
            buy_signals += 1
            patterns_triggered.append("Hammer ✅")
        if 'pattern_doji' in row and row['pattern_doji']:
            buy_signals += 1
            patterns_triggered.append("Doji ✅")
        if 'pattern_bullish_engulfing' in row and row['pattern_bullish_engulfing']:
            buy_signals += 1
            patterns_triggered.append("Bullish Engulfing ✅")
        
        # Show indicator details
        for detail in indicator_details[:4]:  # Show first 4 to avoid clutter
            print(f"     {detail}")
        
        if patterns_triggered:
            print(f"     Patterns: {', '.join(patterns_triggered)}")
        
        # Check trend filter
        trend_blocked = False
        if strategy_params['use_trend_filter']:
            # Check if trend filter is blocking
            if 'RSI_14' in row and row['RSI_14'] > strategy_params['trend_rsi_threshold']:
                trend_blocked = True
                print(f"     🚫 TREND FILTER: RSI {row['RSI_14']:.1f} > {strategy_params['trend_rsi_threshold']} (blocks buy)")
        
        # Final verdict
        needed = strategy_params['buy_score_threshold']
        print(f"   📊 Buy Signals: {buy_signals}/{needed} {'✅ ENOUGH' if buy_signals >= needed and not trend_blocked else '❌ NOT ENOUGH'}")
        
        if buy_signals >= needed and trend_blocked:
            print(f"   🚫 BLOCKED BY TREND FILTER")
        elif buy_signals < needed:
            shortage = needed - buy_signals
            print(f"   ⚠️  NEED {shortage} MORE SIGNAL(S)")
    
    # Find the actual bottom and analyze it specifically
    min_idx = april_data['close'].idxmin()
    bottom_row = april_data.loc[min_idx]
    bottom_date = bottom_row['date'].strftime('%Y-%m-%d')
    
    print(f"\n🎯 APRIL 2025 BOTTOM ANALYSIS:")
    print(f"   📅 Actual bottom: {bottom_date} at ${bottom_row['close']:.2f}")
    
    # Count how many indicators would have signaled at the bottom
    bottom_buy_signals = 0
    if bottom_row['RSI_14'] < strategy_params['RSI_14_buy']:
        bottom_buy_signals += 1
        print(f"   ✅ RSI oversold: {bottom_row['RSI_14']:.1f} < {strategy_params['RSI_14_buy']}")
    else:
        print(f"   ❌ RSI not oversold: {bottom_row['RSI_14']:.1f} ≥ {strategy_params['RSI_14_buy']}")
    
    if bottom_row['WILLR_14'] < strategy_params['WILLR_14_buy']:
        bottom_buy_signals += 1
        print(f"   ✅ WillR oversold: {bottom_row['WILLR_14']:.1f} < {strategy_params['WILLR_14_buy']}")
    else:
        print(f"   ❌ WillR not oversold: {bottom_row['WILLR_14']:.1f} ≥ {strategy_params['WILLR_14_buy']}")
    
    print(f"\n💡 CONCLUSION:")
    print(f"   Signals at bottom: {bottom_buy_signals}")
    print(f"   Required signals: {strategy_params['buy_score_threshold']}")
    print(f"   Result: {'✅ WOULD HAVE BOUGHT' if bottom_buy_signals >= strategy_params['buy_score_threshold'] else '❌ MISSED BOTTOM'}")
    
    if bottom_buy_signals < strategy_params['buy_score_threshold']:
        shortage = strategy_params['buy_score_threshold'] - bottom_buy_signals
        print(f"\n🔧 FIXES TO CATCH BOTTOM:")
        print(f"   1. Lower buy threshold to {bottom_buy_signals}")
        print(f"   2. Relax indicator thresholds (RSI < {bottom_row['RSI_14']:.0f}, WillR < {bottom_row['WILLR_14']:.0f})")
        print(f"   3. Disable trend filter if it's blocking signals")

if __name__ == "__main__":
    analyze_strategy_miss()
