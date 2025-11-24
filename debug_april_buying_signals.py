#!/usr/bin/env python3

import yfinance as yf
import pandas as pd
from indicators import get_all_indicators
from simple_regime_detector import detect_market_regime
from optimization import universal_strategy
import json
from pathlib import Path

print("🔍 DEBUGGING APRIL 3-21 2025: Why No Buy Signals During Crash?")
print("="*70)

# Get data and enrich with indicators
ticker = 'SPY'
data = yf.Ticker(ticker).history(period='1y', interval='1d')
data.reset_index(inplace=True) 
data.columns = [col.lower() for col in data.columns]
enriched_data = get_all_indicators(data)

# Load the most recent strategy parameters
opt_dir = Path('optimizations')
param_files = sorted(opt_dir.glob('*_params.json'), key=lambda x: x.stat().st_mtime, reverse=True)
latest_file = param_files[0]

with open(latest_file, 'r') as f:
    params = json.load(f)

print(f"📊 Using parameters from: {latest_file.name}")
print(f"🎯 Regime-aware: {params.get('enable_regime_aware', False)}")
print(f"💥 Crash buy threshold: {params.get('crash_buy_score_threshold', 'MISSING')}")
print(f"📉 Bear buy threshold: {params.get('bear_buy_score_threshold', 'MISSING')}")
print(f"📈 Bull buy threshold: {params.get('bull_buy_score_threshold', 'MISSING')}")
print()

# Focus on April 2025 data
april_data = enriched_data[enriched_data['date'].dt.month == 4].copy()
print(f"📅 April 2025 data: {len(april_data)} days")

# Test the strategy on this data with current parameters
print("\n🧪 RUNNING STRATEGY ON APRIL DATA...")
try:
    # Run the universal_strategy function with our parameters
    signals = universal_strategy(april_data, params)
    
    # Analyze the results for April 3-21 specifically
    target_dates = april_data[(april_data['date'].dt.day >= 3) & (april_data['date'].dt.day <= 21)]
    
    print(f"\n🎯 APRIL 3-21 ANALYSIS ({len(target_dates)} days):")
    print(f"   Period: {target_dates['date'].min().strftime('%Y-%m-%d')} to {target_dates['date'].max().strftime('%Y-%m-%d')}")
    
    buy_signals = signals.get('buy_signals', pd.Series())
    sell_signals = signals.get('sell_signals', pd.Series())
    
    if len(buy_signals) > 0:
        # Check buy signals during the crash period
        target_buy_signals = buy_signals[target_dates.index]
        target_sell_signals = sell_signals[target_dates.index]
        
        buy_count = target_buy_signals.sum()
        sell_count = target_sell_signals.sum()
        
        print(f"   📈 Buy signals: {buy_count}")
        print(f"   📉 Sell signals: {sell_count}")
        
        if buy_count == 0:
            print(f"\n❌ PROBLEM CONFIRMED: No buy signals during crash period!")
            
            # Analyze WHY no buy signals
            print(f"\n🔍 DIAGNOSTIC ANALYSIS:")
            
            # Check indicator values during crash
            key_indicators = ['RSI_14', 'WILLR_14', 'MACD_12_26_9']
            
            for date_idx in target_dates.index[:5]:  # First 5 days
                date = enriched_data.iloc[date_idx]['date']
                price = enriched_data.iloc[date_idx]['close']
                regime = detect_market_regime(enriched_data, date_idx)
                
                print(f"\n   📅 {date.strftime('%Y-%m-%d')}: Price=${price:.2f}, Regime={regime}")
                
                # Check key indicators
                for indicator in key_indicators:
                    if indicator in enriched_data.columns:
                        value = enriched_data.iloc[date_idx][indicator]
                        
                        # Get the threshold that should apply
                        if params.get('enable_regime_aware', False):
                            threshold_key = f'{regime}_{indicator}_buy'
                            threshold = params.get(threshold_key, params.get(f'{indicator}_buy', 'MISSING'))
                        else:
                            threshold = params.get(f'{indicator}_buy', 'MISSING')
                        
                        triggered = value < threshold if threshold != 'MISSING' else False
                        
                        print(f"      {indicator}: {value:.2f} < {threshold} = {triggered}")
                
                # Check if trend filter blocked signals
                trend_filter = signals.get('trend_filter', pd.Series())
                if len(trend_filter) > 0:
                    trend_active = trend_filter.iloc[date_idx] if date_idx < len(trend_filter) else 'UNKNOWN'
                    print(f"      🛡️  Trend filter: {trend_active}")
                    if trend_active == False:
                        print(f"         ⚠️  TREND FILTER BLOCKING BUY SIGNALS!")
        else:
            print(f"   ✅ Buy signals detected: {buy_count} days")
            
            # Show which days had buy signals
            signal_dates = target_dates[target_buy_signals]['date']
            for date in signal_dates:
                print(f"      📈 Buy signal: {date.strftime('%Y-%m-%d')}")
    else:
        print("❌ No signals returned from strategy")
        
except Exception as e:
    print(f"❌ Error running strategy: {e}")
    import traceback
    traceback.print_exc()

# Also check the trend filter parameters
print(f"\n🛡️  TREND FILTER ANALYSIS:")
print(f"   Enabled: {params.get('use_trend_filter', 'MISSING')}")
print(f"   ADX threshold: {params.get('trend_adx_threshold', 'MISSING')}")
print(f"   RSI oversold: {params.get('trend_rsi_oversold', 'MISSING')}")
print(f"   WILLR threshold: {params.get('trend_willr_threshold', 'MISSING')}")

# Check if trend filter is too restrictive during crashes
if params.get('use_trend_filter', False):
    print(f"\n⚠️  POTENTIAL ISSUE: Trend filter might be blocking crash buy signals")
    print(f"   During crashes, ADX is high and RSI is low - trend filter might block buys")
    print(f"   Consider: Make trend filter less restrictive during crash regimes")
