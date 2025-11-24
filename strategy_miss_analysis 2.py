#!/usr/bin/env python3
"""
Analyze why the strategy with Buy Threshold = 3 missed April 2025 bottom
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

def analyze_exact_miss():
    """Analyze the exact failure using simple indicator calculations"""
    print("🔍 STRATEGY MISS ANALYSIS - BUY THRESHOLD = 3")
    print("="*60)
    
    # Load SPY data
    spy = yf.Ticker("SPY")
    data = spy.history(start="2025-03-01", end="2025-05-01", interval="1d")
    data.reset_index(inplace=True)
    
    print(f"📊 Loaded {len(data)} days of SPY data")
    print(f"📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
    
    # Calculate key indicators manually
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
    
    def calculate_stochastic(high, low, close, k_window=14, d_window=3):
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    # Add indicators
    data['RSI_14'] = calculate_rsi(data['Close'])
    data['WillR_14'] = calculate_williams_r(data['High'], data['Low'], data['Close'])
    
    # MACD
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # Stochastic
    data['Stoch_K'], data['Stoch_D'] = calculate_stochastic(data['High'], data['Low'], data['Close'])
    
    # CCI
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    sma_tp = tp.rolling(window=14).mean()
    mad = tp.rolling(window=14).apply(lambda x: np.mean(np.abs(x - x.mean())))
    data['CCI'] = (tp - sma_tp) / (0.015 * mad)
    
    # Focus on April 2025
    april_data = data[(data['Date'] >= '2025-04-01') & (data['Date'] <= '2025-04-15')].copy()
    
    if len(april_data) == 0:
        print("❌ No April 2025 data found!")
        return
    
    print(f"\n📅 APRIL 2025 DETAILED ANALYSIS (Buy Threshold = 3):")
    print("-" * 60)
    
    # Define typical buy thresholds from optimization
    buy_thresholds = {
        'RSI_14': 30,      # RSI < 30 = oversold
        'WillR_14': -80,   # WillR < -80 = oversold  
        'MACD': 0,         # MACD > 0 = bullish
        'MACD_Hist': 0,    # MACD Hist > 0 = momentum up
        'Stoch_K': 20,     # Stoch < 20 = oversold
        'CCI': -100        # CCI < -100 = oversold
    }
    
    # Analyze each day
    for _, row in april_data.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d (%a)')
        price_change = ((row['Close'] / row['Open'] - 1) * 100)
        
        print(f"\n{date_str}:")
        print(f"   💰 Price: ${row['Close']:.2f} (Change: {price_change:+.2f}%)")
        
        # Count buy signals
        buy_signals = 0
        signal_details = []
        
        # RSI
        if not pd.isna(row['RSI_14']):
            rsi_buy = row['RSI_14'] < buy_thresholds['RSI_14']
            if rsi_buy:
                buy_signals += 1
                signal_details.append(f"✅ RSI: {row['RSI_14']:.1f} < {buy_thresholds['RSI_14']}")
            else:
                signal_details.append(f"❌ RSI: {row['RSI_14']:.1f} ≥ {buy_thresholds['RSI_14']}")
        
        # Williams %R
        if not pd.isna(row['WillR_14']):
            willr_buy = row['WillR_14'] < buy_thresholds['WillR_14']
            if willr_buy:
                buy_signals += 1
                signal_details.append(f"✅ WillR: {row['WillR_14']:.1f} < {buy_thresholds['WillR_14']}")
            else:
                signal_details.append(f"❌ WillR: {row['WillR_14']:.1f} ≥ {buy_thresholds['WillR_14']}")
        
        # MACD
        if not pd.isna(row['MACD']):
            macd_buy = row['MACD'] > buy_thresholds['MACD']
            if macd_buy:
                buy_signals += 1
                signal_details.append(f"✅ MACD: {row['MACD']:.3f} > {buy_thresholds['MACD']}")
            else:
                signal_details.append(f"❌ MACD: {row['MACD']:.3f} ≤ {buy_thresholds['MACD']}")
        
        # MACD Histogram  
        if not pd.isna(row['MACD_Hist']):
            macd_hist_buy = row['MACD_Hist'] > buy_thresholds['MACD_Hist']
            if macd_hist_buy:
                buy_signals += 1
                signal_details.append(f"✅ MACD-H: {row['MACD_Hist']:.3f} > {buy_thresholds['MACD_Hist']}")
            else:
                signal_details.append(f"❌ MACD-H: {row['MACD_Hist']:.3f} ≤ {buy_thresholds['MACD_Hist']}")
        
        # Stochastic
        if not pd.isna(row['Stoch_K']):
            stoch_buy = row['Stoch_K'] < buy_thresholds['Stoch_K']
            if stoch_buy:
                buy_signals += 1
                signal_details.append(f"✅ Stoch: {row['Stoch_K']:.1f} < {buy_thresholds['Stoch_K']}")
            else:
                signal_details.append(f"❌ Stoch: {row['Stoch_K']:.1f} ≥ {buy_thresholds['Stoch_K']}")
        
        # CCI
        if not pd.isna(row['CCI']):
            cci_buy = row['CCI'] < buy_thresholds['CCI']
            if cci_buy:
                buy_signals += 1
                signal_details.append(f"✅ CCI: {row['CCI']:.1f} < {buy_thresholds['CCI']}")
            else:
                signal_details.append(f"❌ CCI: {row['CCI']:.1f} ≥ {buy_thresholds['CCI']}")
        
        # Show details
        for detail in signal_details:
            print(f"     {detail}")
        
        # Final verdict
        required = 3
        verdict = "🟢 BUY SIGNAL" if buy_signals >= required else "❌ NO BUY"
        print(f"   📊 Total Buy Signals: {buy_signals}/{required} → {verdict}")
        
        if buy_signals < required:
            shortage = required - buy_signals
            print(f"   ⚠️  Need {shortage} more signal(s) to trigger buy")
    
    # Find and analyze the bottom specifically
    min_idx = april_data['Close'].idxmin()
    bottom_row = april_data.loc[min_idx]
    bottom_date = bottom_row['Date'].strftime('%Y-%m-%d')
    
    print(f"\n🎯 BOTTOM DAY ANALYSIS ({bottom_date}):")
    print(f"   💰 Lowest price: ${bottom_row['Close']:.2f}")
    
    bottom_signals = 0
    bottom_details = []
    
    # Check each indicator at the bottom
    if bottom_row['RSI_14'] < buy_thresholds['RSI_14']:
        bottom_signals += 1
        bottom_details.append(f"✅ RSI oversold: {bottom_row['RSI_14']:.1f}")
    else:
        bottom_details.append(f"❌ RSI not oversold: {bottom_row['RSI_14']:.1f}")
    
    if bottom_row['WillR_14'] < buy_thresholds['WillR_14']:
        bottom_signals += 1
        bottom_details.append(f"✅ WillR oversold: {bottom_row['WillR_14']:.1f}")
    else:
        bottom_details.append(f"❌ WillR not oversold: {bottom_row['WillR_14']:.1f}")
    
    if bottom_row['MACD'] > buy_thresholds['MACD']:
        bottom_signals += 1
        bottom_details.append(f"✅ MACD bullish: {bottom_row['MACD']:.3f}")
    else:
        bottom_details.append(f"❌ MACD bearish: {bottom_row['MACD']:.3f}")
    
    if bottom_row['Stoch_K'] < buy_thresholds['Stoch_K']:
        bottom_signals += 1
        bottom_details.append(f"✅ Stoch oversold: {bottom_row['Stoch_K']:.1f}")
    else:
        bottom_details.append(f"❌ Stoch not oversold: {bottom_row['Stoch_K']:.1f}")
    
    for detail in bottom_details:
        print(f"     {detail}")
    
    print(f"\n💡 BOTTOM ANALYSIS RESULT:")
    print(f"   Signals at bottom: {bottom_signals}/3")
    print(f"   Result: {'✅ WOULD BUY' if bottom_signals >= 3 else '❌ MISSED BOTTOM'}")
    
    if bottom_signals < 3:
        print(f"\n🔧 WHY THE STRATEGY MISSED:")
        print(f"   • Required 3 indicators to agree simultaneously")
        print(f"   • Only {bottom_signals} indicators were signaling at the bottom")
        print(f"   • Need to either:")
        print(f"     1. Lower buy threshold to {bottom_signals}")
        print(f"     2. Relax indicator thresholds to get more signals")
        print(f"     3. Use different indicators that are more sensitive")
        
        # Suggest fixes
        print(f"\n🎯 SUGGESTED FIXES:")
        if bottom_row['RSI_14'] >= 30:
            print(f"   • Increase RSI threshold to {bottom_row['RSI_14']:.0f} (from 30)")
        if bottom_row['WillR_14'] >= -80:
            print(f"   • Increase WillR threshold to {bottom_row['WillR_14']:.0f} (from -80)")
        if bottom_row['Stoch_K'] >= 20:
            print(f"   • Increase Stoch threshold to {bottom_row['Stoch_K']:.0f} (from 20)")

if __name__ == "__main__":
    analyze_exact_miss()
