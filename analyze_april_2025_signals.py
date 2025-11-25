#!/usr/bin/env python3
"""
Analyze April 2025 buy signals in the optimization CSV
"""

import pandas as pd
from pathlib import Path

def analyze_april_signals():
    print("🔍 ANALYZING APRIL 2025 BUY SIGNALS")
    print("=" * 50)
    
    # Find the most recent optimization CSV
    opt_dir = Path('optimizations')
    csv_files = sorted(opt_dir.glob('*ret*.csv'), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not csv_files:
        print("❌ No optimization CSV files found")
        return
    
    latest_csv = csv_files[0]
    print(f"📊 Analyzing: {latest_csv.name}")
    
    # Load CSV
    df = pd.read_csv(latest_csv)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for April 2025
    april_2025 = df[(df['date'].dt.year == 2025) & (df['date'].dt.month == 4)]
    
    if len(april_2025) == 0:
        print("❌ No April 2025 data found in CSV")
        return
    
    print(f"📅 Found {len(april_2025)} days of April 2025 data")
    print()
    
    # Analyze each day
    print("📈 APRIL 2025 SIGNAL ANALYSIS:")
    print("-" * 60)
    print(f"{'Date':<12} {'Price':<8} {'Signal':<6} {'Buy':<5} {'Sell':<5} {'BuyTrig':<8} {'SellTrig':<9} {'CrashScore':<10}")
    print("-" * 60)
    
    buy_signal_days = []
    
    for _, row in april_2025.iterrows():
        date = row['date'].strftime('%m-%d')
        price = row['close']
        signal = row['signal']
        buy_score = row.get('buy_score', 'N/A')
        sell_score = row.get('sell_score', 'N/A')
        buy_triggered = row.get('buy_signal_triggered', False)
        sell_triggered = row.get('sell_signal_triggered', False)
        
        # Look for crash composite score
        crash_score = 'N/A'
        for col in df.columns:
            if 'crash_composite_score' in col or 'crash_buy_signal' in col:
                crash_score = row.get(col, 'N/A')
                break
        
        signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
        buy_trig = "✅" if buy_triggered else "❌"
        sell_trig = "✅" if sell_triggered else "❌"
        
        print(f"{date:<12} ${price:<7.2f} {signal_text:<6} {buy_score:<5} {sell_score:<5} {buy_trig:<8} {sell_trig:<9} {crash_score:<10}")
        
        if signal == 1 or buy_triggered:
            buy_signal_days.append(row['date'])
    
    print("-" * 60)
    
    # Summary
    total_buys = len(buy_signal_days)
    print(f"\n📊 SUMMARY:")
    print(f"   Total buy signals in April 2025: {total_buys}")
    print(f"   Total days analyzed: {len(april_2025)}")
    print(f"   Buy signal percentage: {total_buys/len(april_2025)*100:.1f}%")
    
    if total_buys > 0:
        print(f"\n📈 BUY SIGNAL DATES:")
        for buy_date in buy_signal_days:
            corresponding_row = april_2025[april_2025['date'] == buy_date].iloc[0]
            print(f"   {buy_date.strftime('%Y-%m-%d')}: ${corresponding_row['close']:.2f}")
    else:
        print(f"\n❌ NO BUY SIGNALS DETECTED IN APRIL 2025!")
        print(f"\n🔍 DEBUGGING INFO:")
        
        # Check why no buy signals
        sample_day = april_2025.iloc[5]  # Mid-April
        print(f"\n   Sample day: {sample_day['date'].strftime('%Y-%m-%d')}")
        print(f"   Price: ${sample_day['close']:.2f}")
        print(f"   Buy score: {sample_day.get('buy_score', 'N/A')}")
        print(f"   Buy threshold: {sample_day.get('buy_threshold', 'N/A')}")
        print(f"   Signal: {sample_day['signal']}")
        
        # Check for bottom detection indicators
        bottom_indicators = ['crash_composite_score', 'crash_buy_signal', 'selling_exhaustion', 
                           'panic_recovery', 'volume_confirmation', 'hammer_pattern']
        
        print(f"\n   Bottom detection indicators:")
        for indicator in bottom_indicators:
            for col in df.columns:
                if indicator in col:
                    value = sample_day.get(col, 'N/A')
                    print(f"     {col}: {value}")
    
    # Check the actual bottom day
    if len(april_2025) > 0:
        bottom_idx = april_2025['close'].idxmin()
        bottom_row = april_2025.loc[bottom_idx]
        bottom_price = bottom_row['close']
        bottom_date = bottom_row['date']
        
        print(f"\n🎯 ACTUAL MARKET BOTTOM:")
        print(f"   Date: {bottom_date.strftime('%Y-%m-%d')}")
        print(f"   Price: ${bottom_price:.2f}")
        print(f"   Signal on bottom day: {'BUY' if bottom_row['signal'] == 1 else 'SELL' if bottom_row['signal'] == -1 else 'HOLD'}")
        print(f"   Buy score: {bottom_row.get('buy_score', 'N/A')}")
        print(f"   Buy triggered: {'Yes' if bottom_row.get('buy_signal_triggered', False) else 'No'}")

if __name__ == "__main__":
    analyze_april_signals()
