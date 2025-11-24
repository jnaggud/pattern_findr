#!/usr/bin/env python3
"""
Analyze the CSV logs to debug why strategies missed the April 2025 bottom
"""

import pandas as pd
import numpy as np
import json
import os

def analyze_april_miss():
    """Analyze CSV logs to understand why April 2025 bottom was missed"""
    print("🔍 ANALYZING APRIL 2025 BOTTOM MISS FROM CSV LOGS")
    print("="*60)
    
    # Find the best performing strategy
    opt_dir = "/Users/jeffersonduggan/Documents/Pattern_FindR/optimizations"
    csv_files = [f for f in os.listdir(opt_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV files found in optimizations directory!")
        return
    
    # Sort by return percentage (extract from filename)
    csv_files_with_returns = []
    for f in csv_files:
        try:
            # Extract return from filename like "SPY_20251124_060723_trial103_ret53.4pct.csv"
            ret_part = f.split('_ret')[1].split('pct.csv')[0]
            return_pct = float(ret_part)
            csv_files_with_returns.append((f, return_pct))
        except:
            continue
    
    if not csv_files_with_returns:
        print("❌ No valid CSV files with return data found!")
        return
    
    # Sort by return (highest first)
    csv_files_with_returns.sort(key=lambda x: x[1], reverse=True)
    
    print(f"📊 Found {len(csv_files_with_returns)} strategy logs:")
    for f, ret in csv_files_with_returns:
        print(f"   {f} → {ret:.1f}% return")
    
    # Analyze the best strategy
    best_file, best_return = csv_files_with_returns[0]
    print(f"\n🎯 ANALYZING BEST STRATEGY: {best_file} ({best_return:.1f}% return)")
    
    # Load the CSV
    csv_path = os.path.join(opt_dir, best_file)
    df = pd.read_csv(csv_path)
    
    print(f"📈 Loaded {len(df)} days of data")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Load parameters
    params_file = best_file.replace('.csv', '_params.json')
    params_path = os.path.join(opt_dir, params_file)
    
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        print(f"\n⚙️  STRATEGY PARAMETERS:")
        print(f"   Buy threshold: {params.get('buy_score_threshold', 'N/A')}")
        print(f"   Sell threshold: {params.get('sell_score_threshold', 'N/A')}")
        print(f"   Signal persistence: {params.get('signal_persistence_days', 'N/A')} days")
        print(f"   Trend filter: {params.get('use_trend_filter', 'N/A')}")
        print(f"   Position sizing: {params.get('enable_position_sizing', 'N/A')}")
        
        # Show key indicator thresholds
        key_indicators = ['RSI_14', 'WILLR_14', 'MACD_12_26_9', 'STOCH_14_3_3']
        print(f"\n📊 KEY INDICATOR THRESHOLDS:")
        for ind in key_indicators:
            buy_key = f'{ind}_buy'
            sell_key = f'{ind}_sell'
            use_key = f'use_{ind}'
            if params.get(use_key, False):
                buy_thresh = params.get(buy_key, 'N/A')
                sell_thresh = params.get(sell_key, 'N/A')
                print(f"   ✅ {ind}: BUY < {buy_thresh}, SELL > {sell_thresh}")
    else:
        print("⚠️  Parameters file not found")
        params = {}
    
    # Focus on April 2025
    df['date'] = pd.to_datetime(df['date'])
    april_data = df[(df['date'] >= '2025-04-01') & (df['date'] <= '2025-04-15')].copy()
    
    if len(april_data) == 0:
        print("❌ No April 2025 data found in CSV!")
        return
    
    print(f"\n📅 APRIL 2025 ANALYSIS ({len(april_data)} days):")
    print("-" * 50)
    
    # Find the bottom day
    min_close_idx = april_data['close'].idxmin()
    bottom_day = april_data.loc[min_close_idx]
    bottom_date = bottom_day['date'].strftime('%Y-%m-%d')
    
    print(f"🎯 MARKET BOTTOM: {bottom_date} at ${bottom_day['close']:.2f}")
    print(f"   Signal: {bottom_day['signal_text']}")
    print(f"   Buy Score: {bottom_day['buy_score']}/{bottom_day['buy_threshold']}")
    print(f"   Buy Signal Triggered: {bottom_day['buy_signal_triggered']}")
    
    # Show indicator values at bottom
    print(f"\n📊 INDICATOR VALUES AT BOTTOM ({bottom_date}):")
    
    # Key indicators to check
    indicator_cols = [col for col in april_data.columns if col.startswith('ind_')]
    key_indicators_found = []
    
    for col in indicator_cols:
        if 'RSI_14' in col or 'WILLR_14' in col or 'MACD_12_26_9' in col or 'STOCH' in col:
            key_indicators_found.append(col)
    
    for col in key_indicators_found[:8]:  # Show top 8
        indicator_name = col.replace('ind_', '')
        value = bottom_day[col]
        
        # Get thresholds if available
        buy_thresh = params.get(f'{indicator_name}_buy', 'N/A')
        sell_thresh = params.get(f'{indicator_name}_sell', 'N/A')
        
        # Determine if this would trigger a buy signal
        triggered = "?"
        if buy_thresh != 'N/A':
            try:
                if indicator_name in ['RSI_14', 'WILLR_14', 'STOCH_14_3_3']:
                    # Lower is better for these (oversold)
                    triggered = "✅" if value < buy_thresh else "❌"
                elif 'MACD' in indicator_name:
                    # Higher is better for MACD (bullish)
                    triggered = "✅" if value > buy_thresh else "❌"
                else:
                    triggered = "?"
            except:
                triggered = "?"
        
        print(f"   {triggered} {indicator_name}: {value:.3f} (buy threshold: {buy_thresh})")
    
    # Analyze each day in April
    print(f"\n📅 DAY-BY-DAY APRIL 2025 SIGNALS:")
    print("-" * 50)
    
    for _, row in april_data.iterrows():
        date_str = row['date'].strftime('%m-%d')
        price_change = ((row['close'] / row['open'] - 1) * 100) if row['open'] > 0 else 0
        
        signal_text = "🟢 BUY" if row['signal'] == 1 else "🔴 SELL" if row['signal'] == -1 else "⚪ HOLD"
        
        print(f"   {date_str}: {signal_text} @ ${row['close']:.2f} ({price_change:+.1f}%)")
        print(f"          Buy: {row['buy_score']}/{row['buy_threshold']} {'✅' if row['buy_signal_triggered'] else '❌'}")
        
        # Show top contributing indicators on signal days
        if row['signal'] != 0:
            rsi_val = row.get('ind_RSI_14', 'N/A')
            willr_val = row.get('ind_WILLR_14', 'N/A') 
            macd_val = row.get('ind_MACD_12_26_9', 'N/A')
            print(f"          RSI: {rsi_val:.1f}, WillR: {willr_val:.1f}, MACD: {macd_val:.3f}")
    
    # Count signals in April
    buy_signals = (april_data['signal'] == 1).sum()
    sell_signals = (april_data['signal'] == -1).sum()
    
    print(f"\n📊 APRIL 2025 SIGNAL SUMMARY:")
    print(f"   Buy signals: {buy_signals}")
    print(f"   Sell signals: {sell_signals}")
    print(f"   Hold days: {len(april_data) - buy_signals - sell_signals}")
    
    # Final analysis
    print(f"\n💡 APRIL BOTTOM MISS ANALYSIS:")
    if bottom_day['buy_signal_triggered']:
        print(f"   ✅ Strategy DID generate buy signal at bottom!")
        print(f"   🤔 Issue might be in position sizing or trade execution")
    else:
        shortage = bottom_day['buy_threshold'] - bottom_day['buy_score']
        print(f"   ❌ Strategy missed bottom - needed {shortage} more buy signals")
        print(f"   🔧 Suggested fixes:")
        print(f"      1. Lower buy threshold from {bottom_day['buy_threshold']} to {bottom_day['buy_score']}")
        print(f"      2. Relax indicator thresholds to catch extreme oversold conditions")
        
        # Check if trend filter blocked signals
        if params.get('use_trend_filter', False):
            print(f"      3. Consider disabling trend filter during crash conditions")
    
    # Check position sizing
    if bottom_day.get('position_shares', 0) == 0:
        print(f"   📊 No position held at bottom - strategy was in cash")
    else:
        print(f"   📊 Position at bottom: {bottom_day.get('position_shares', 0)} shares")

if __name__ == "__main__":
    analyze_april_miss()
