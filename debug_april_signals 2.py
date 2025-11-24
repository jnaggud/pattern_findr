#!/usr/bin/env python3
"""
Debug script to analyze indicator values and signal generation for April 3-8 period.
This will help understand why buy signals aren't being generated during the market bottom.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Import the strategy components
from optimization import universal_strategy
from indicators import get_all_indicators

def load_spy_data():
    """Load SPY data covering April 2024"""
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="2y", interval="1d")
    data.reset_index(inplace=True)
    data.columns = [col.lower() for col in data.columns]
    return data

def analyze_april_period():
    """Analyze the April 3-8, 2024 period in detail"""
    print("🔍 DEBUGGING APRIL 3-8, 2024 SIGNAL GENERATION")
    print("="*60)
    
    # Load data
    data = load_spy_data()
    print(f"📊 Loaded {len(data)} days of SPY data")
    print(f"📅 Date range: {data['date'].min()} to {data['date'].max()}")
    
    # Convert date column before adding indicators
    data['date'] = pd.to_datetime(data['date'])
    
    # Add all indicators
    enriched_data = get_all_indicators(data)
    print(f"📈 Added indicators, now have {len(enriched_data.columns)} columns")
    print(f"📋 First 10 columns: {list(enriched_data.columns[:10])}")
    
    # Check if date column exists
    if 'date' not in enriched_data.columns:
        print("⚠️  'date' column missing! Checking for similar columns...")
        date_like_cols = [col for col in enriched_data.columns if 'date' in col.lower() or 'time' in col.lower()]
        print(f"📅 Date-like columns found: {date_like_cols}")
        if date_like_cols:
            enriched_data['date'] = enriched_data[date_like_cols[0]]
        else:
            print("❌ No date column found! Using index instead.")
            enriched_data.reset_index(inplace=True)
            if 'Date' in enriched_data.columns:
                enriched_data['date'] = enriched_data['Date']
            else:
                print("❌ Cannot proceed without date column!")
                return
    
    # Focus on April 2024 data
    april_mask = (enriched_data['date'] >= '2024-04-01') & (enriched_data['date'] <= '2024-04-15')
    april_data = enriched_data[april_mask].copy()
    
    if len(april_data) == 0:
        print("❌ No April 2024 data found!")
        return
    
    print(f"\n📅 APRIL 2024 DATA ({len(april_data)} days)")
    print("-" * 40)
    
    # Show basic price action
    for _, row in april_data.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        print(f"{date_str}: Close=${row['close']:.2f}, Change={((row['close']/row['open']-1)*100):+.2f}%")
    
    # Test with typical strategy parameters from recent optimizations
    test_params = {
        'buy_score_threshold': 4,
        'sell_score_threshold': 14,
        'signal_persistence_days': 2,
        
        # Enable key indicators that should catch bottoms
        'use_RSI_14': True,
        'use_MACD_12_26_9': True,
        'use_WILLR_14': True,
        'use_STOCH_14_3_3': True,
        'use_CCI_14': True,
        'use_MFI_14': True,
        'use_pattern_hammer': True,
        'use_pattern_doji': True,
        
        # RSI parameters (should be oversold)
        'RSI_14_buy': 35,
        'RSI_14_sell': 70,
        
        # MACD parameters
        'MACD_12_26_9_buy': -0.5,
        'MACD_12_26_9_sell': 0.5,
        
        # Williams %R (should be oversold)
        'WILLR_14_buy': -80,
        'WILLR_14_sell': -20,
        
        # Stochastic (should be oversold)
        'STOCH_14_3_3_buy': 20,
        'STOCH_14_3_3_sell': 80,
        
        # Trend filter (relaxed for bottoms)
        'use_trend_filter': True,
        'trend_rsi_threshold': 40,
        'trend_adx_threshold': 15,
        'trend_willr_threshold': -70,
        'trend_stoch_threshold': 25,
        'trend_sma_period': 50
    }
    
    print(f"\n🎯 TESTING STRATEGY WITH PARAMETERS:")
    print(f"   Buy Threshold: {test_params['buy_score_threshold']}")
    print(f"   Sell Threshold: {test_params['sell_score_threshold']}")
    print(f"   RSI Buy: {test_params['RSI_14_buy']}")
    print(f"   Williams %R Buy: {test_params['WILLR_14_buy']}")
    
    # Generate signals
    signals = universal_strategy(enriched_data, test_params)
    april_signals = signals[april_mask]
    
    print(f"\n📊 SIGNAL ANALYSIS FOR APRIL 2024:")
    print("-" * 40)
    
    # Show detailed analysis for each day
    for i, (_, row) in enumerate(april_data.iterrows()):
        date_str = row['date'].strftime('%Y-%m-%d (%a)')
        signal = april_signals.iloc[i] if i < len(april_signals) else 0
        signal_text = "🔴 SELL" if signal == -1 else "🟢 BUY" if signal == 1 else "⚪ HOLD"
        
        print(f"\n{date_str}: {signal_text}")
        print(f"   Price: ${row['close']:.2f} (Change: {((row['close']/row['open']-1)*100):+.2f}%)")
        
        # Check key oversold indicators
        if 'RSI_14' in row:
            rsi_val = row['RSI_14']
            rsi_signal = "OVERSOLD" if rsi_val < test_params['RSI_14_buy'] else "NEUTRAL"
            print(f"   RSI(14): {rsi_val:.1f} ({rsi_signal})")
        
        if 'WILLR_14' in row:
            willr_val = row['WILLR_14']
            willr_signal = "OVERSOLD" if willr_val < test_params['WILLR_14_buy'] else "NEUTRAL"
            print(f"   Williams %R: {willr_val:.1f} ({willr_signal})")
        
        if 'STOCH_14_3_3' in row:
            stoch_val = row['STOCH_14_3_3']
            stoch_signal = "OVERSOLD" if stoch_val < test_params['STOCH_14_3_3_buy'] else "NEUTRAL"
            print(f"   Stochastic: {stoch_val:.1f} ({stoch_signal})")
        
        if 'MACD_12_26_9' in row:
            macd_val = row['MACD_12_26_9']
            macd_signal = "BULLISH" if macd_val > test_params['MACD_12_26_9_buy'] else "BEARISH"
            print(f"   MACD: {macd_val:.3f} ({macd_signal})")
        
        # Check pattern signals
        if 'pattern_hammer' in row and row['pattern_hammer']:
            print(f"   📈 Hammer pattern detected!")
        if 'pattern_doji' in row and row['pattern_doji']:
            print(f"   📈 Doji pattern detected!")
    
    # Count signals in April
    buy_signals = (april_signals == 1).sum()
    sell_signals = (april_signals == -1).sum()
    
    print(f"\n📈 APRIL 2024 SIGNAL SUMMARY:")
    print(f"   Buy signals: {buy_signals}")
    print(f"   Sell signals: {sell_signals}")
    print(f"   Hold days: {len(april_signals) - buy_signals - sell_signals}")
    
    if buy_signals == 0:
        print(f"\n❌ NO BUY SIGNALS DETECTED IN APRIL!")
        print(f"   This suggests the strategy parameters are too restrictive")
        print(f"   or the trend filter is blocking oversold signals.")
        
        # Test with more aggressive parameters
        print(f"\n🧪 TESTING WITH MORE AGGRESSIVE PARAMETERS:")
        aggressive_params = test_params.copy()
        aggressive_params.update({
            'buy_score_threshold': 1,  # Much lower
            'RSI_14_buy': 40,         # Less oversold
            'WILLR_14_buy': -70,      # Less oversold
            'STOCH_14_3_3_buy': 30,   # Less oversold
            'trend_rsi_threshold': 45, # More permissive trend filter
            'use_trend_filter': False  # Disable trend filter entirely
        })
        
        aggressive_signals = universal_strategy(enriched_data, aggressive_params)
        april_aggressive = aggressive_signals[april_mask]
        aggressive_buys = (april_aggressive == 1).sum()
        
        print(f"   Aggressive buy signals: {aggressive_buys}")
        
        if aggressive_buys > 0:
            print(f"   ✅ Aggressive parameters generate buy signals!")
            print(f"   💡 Consider lowering buy thresholds or disabling trend filter")
        else:
            print(f"   ❌ Even aggressive parameters don't generate signals")
            print(f"   🔍 Need to investigate indicator calculations")

if __name__ == "__main__":
    analyze_april_period()
