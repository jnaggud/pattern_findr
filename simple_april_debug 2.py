#!/usr/bin/env python3
"""
Simple debug script to check specific indicator values for April 3-8, 2024
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

def analyze_april_bottom():
    """Check indicator values during April 2025"""
    print("🔍 APRIL 2025 ANALYSIS")
    print("="*50)
    
    # Load SPY data for April 2025 (recent data)
    spy = yf.Ticker("SPY")
    data = spy.history(start="2025-03-01", end="2025-05-01", interval="1d")
    data.reset_index(inplace=True)
    
    print(f"📊 Loaded {len(data)} days of SPY data")
    print(f"📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
    
    # Add basic indicators manually
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_williams_r(high, low, close, window=14):
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate key oversold indicators
    data['RSI_14'] = calculate_rsi(data['Close'])
    data['WillR_14'] = calculate_williams_r(data['High'], data['Low'], data['Close'])
    
    # MACD calculation
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    
    # Focus on April 2025 (adjust date range based on current date)
    april_data = data[(data['Date'] >= '2025-04-01') & (data['Date'] <= '2025-04-15')].copy()
    
    if len(april_data) == 0:
        print("⚠️  April 2025 data not available yet, checking recent data...")
        # If April 2025 isn't available, look at most recent data
        recent_data = data.tail(10)  # Last 10 days
        april_data = recent_data.copy()
        print(f"📊 Using last {len(april_data)} days of available data instead")
    
    print(f"\n📈 APRIL 2025 MARKET ANALYSIS:")
    print("-" * 40)
    
    # Show each day's key metrics
    for _, row in april_data.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d (%a)')
        price_change = ((row['Close'] / row['Open'] - 1) * 100)
        
        print(f"\n{date_str}:")
        print(f"   💰 Price: ${row['Close']:.2f} (Change: {price_change:+.2f}%)")
        
        # Check oversold conditions
        rsi_val = row['RSI_14']
        willr_val = row['WillR_14']
        macd_val = row['MACD']
        
        # Determine if indicators show oversold
        rsi_oversold = rsi_val < 30 if not pd.isna(rsi_val) else False
        willr_oversold = willr_val < -80 if not pd.isna(willr_val) else False
        macd_bullish = macd_val > row['MACD_Signal'] if not pd.isna(macd_val) else False
        
        print(f"   📊 RSI(14): {rsi_val:.1f} {'🔴 OVERSOLD' if rsi_oversold else '⚪ NORMAL'}")
        print(f"   📊 WillR(14): {willr_val:.1f} {'🔴 OVERSOLD' if willr_oversold else '⚪ NORMAL'}")
        print(f"   📊 MACD: {macd_val:.3f} {'🟢 BULLISH' if macd_bullish else '🔴 BEARISH'}")
        
        # Overall signal assessment
        oversold_count = sum([rsi_oversold, willr_oversold])
        if oversold_count >= 2:
            print(f"   🎯 STRONG BUY SIGNAL ({oversold_count} oversold indicators)")
        elif oversold_count >= 1:
            print(f"   💡 Moderate buy signal ({oversold_count} oversold indicator)")
        else:
            print(f"   ❌ No clear buy signal")
    
    # Find the lowest close in the period
    min_close_idx = april_data['Close'].idxmin()
    bottom_row = april_data.loc[min_close_idx]
    bottom_date = bottom_row['Date'].strftime('%Y-%m-%d')
    
    period_name = "APRIL 2025" if len(data[(data['Date'] >= '2025-04-01')]) > 0 else "RECENT PERIOD"
    
    print(f"\n🎯 {period_name} LOWEST POINT ANALYSIS:")
    print(f"   📅 Lowest close: {bottom_date} at ${bottom_row['Close']:.2f}")
    print(f"   📊 RSI at lowest: {bottom_row['RSI_14']:.1f}")
    print(f"   📊 WillR at lowest: {bottom_row['WillR_14']:.1f}")
    print(f"   📊 MACD at lowest: {bottom_row['MACD']:.3f}")
    
    # Check if typical strategy thresholds would catch this
    print(f"\n🔍 STRATEGY THRESHOLD ANALYSIS:")
    print(f"   Conservative RSI < 30: {'✅ YES' if bottom_row['RSI_14'] < 30 else '❌ NO'}")
    print(f"   Moderate RSI < 35: {'✅ YES' if bottom_row['RSI_14'] < 35 else '❌ NO'}")
    print(f"   Aggressive RSI < 40: {'✅ YES' if bottom_row['RSI_14'] < 40 else '❌ NO'}")
    print(f"   WillR < -80: {'✅ YES' if bottom_row['WillR_14'] < -80 else '❌ NO'}")
    print(f"   WillR < -70: {'✅ YES' if bottom_row['WillR_14'] < -70 else '❌ NO'}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if bottom_row['RSI_14'] > 30:
        print(f"   📈 RSI threshold should be > {bottom_row['RSI_14']:.0f} to catch this low")
    if bottom_row['WillR_14'] > -80:
        print(f"   📈 Williams %R threshold should be > {bottom_row['WillR_14']:.0f} to catch this low")
    
    # Check recent strategy parameters that missed this
    current_strategy_buy_rsi = 4   # From your recent results (buy threshold = 4)
    current_strategy_willr = -80   # Typical threshold
    
    print(f"\n🔧 CURRENT STRATEGY ANALYSIS:")
    print(f"   Your buy threshold: {current_strategy_buy_rsi} indicators must agree")
    print(f"   Period low RSI: {bottom_row['RSI_14']:.1f}")
    print(f"   RSI oversold (< 30): {'✅ YES' if bottom_row['RSI_14'] < 30 else '❌ NO'}")
    print(f"   WillR oversold (< -80): {'✅ YES' if bottom_row['WillR_14'] < -80 else '❌ NO'}")
    
    # Count how many indicators would signal oversold
    oversold_signals = 0
    if bottom_row['RSI_14'] < 30:
        oversold_signals += 1
    if bottom_row['WillR_14'] < -80:
        oversold_signals += 1
    
    print(f"   Oversold signals at low: {oversold_signals}")
    print(f"   Would generate buy signal: {'✅ YES' if oversold_signals >= current_strategy_buy_rsi else '❌ NO - NEED MORE SIGNALS'}")

if __name__ == "__main__":
    analyze_april_bottom()
