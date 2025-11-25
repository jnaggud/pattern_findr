#!/usr/bin/env python3
"""
Test universal_strategy directly to see why crash override fails
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
from indicators import get_all_indicators
from optimization import universal_strategy
import json

def test_strategy_directly():
    print("🧪 TESTING UNIVERSAL_STRATEGY DIRECTLY")
    print("=" * 50)
    
    # Get data and add indicators
    data = yf.Ticker('SPY').history(period='1y', interval='1d')
    data.reset_index(inplace=True)
    data.columns = [col.lower() for col in data.columns]
    enriched_data = get_all_indicators(data)
    
    print(f"📊 Data shape: {enriched_data.shape}")
    print(f"📅 Date range: {enriched_data.index[0]} to {enriched_data.index[-1]}")
    
    # Load the latest optimization parameters
    with open('optimizations/SPY_20251124_180031_trial77_ret46.4pct_params.json', 'r') as f:
        params = json.load(f)
    
    print(f"\n⚙️  Using parameters from best trial (46.4% return)")
    print(f"   crash_buy_score_threshold: {params.get('crash_buy_score_threshold', 'MISSING')}")
    print(f"   buy_score_threshold: {params.get('buy_score_threshold', 'MISSING')}")
    print(f"   use_crash_composite_score: {params.get('use_crash_composite_score', 'MISSING')}")
    
    # Check if crash_buy_signal exists in data
    if 'crash_buy_signal' in enriched_data.columns:
        crash_signals = enriched_data['crash_buy_signal'].sum()
        print(f"   crash_buy_signal in data: ✅ ({crash_signals} triggers)")
    else:
        print(f"   crash_buy_signal in data: ❌ MISSING")
        return
    
    # Run strategy on subset (April area)
    april_start = 85  # Approximate April start
    april_end = 110   # Approximate April end
    april_data = enriched_data.iloc[april_start:april_end].copy()
    
    print(f"\n📅 Testing on April subset: {len(april_data)} days")
    
    try:
        # Test strategy
        result = universal_strategy(april_data, params)
        
        print(f"\n📈 STRATEGY RESULTS:")
        print(f"   Buy signals: {result['buy_signals'].sum()}")
        print(f"   Sell signals: {result['sell_signals'].sum()}")
        
        # Check buy scores
        if 'buy_score' in result:
            max_buy_score = result['buy_score'].max()
            print(f"   Max buy score: {max_buy_score}")
        
        # Check for crash boost
        crash_days_in_april = april_data['crash_buy_signal'].sum()
        print(f"   Crash signals in April subset: {crash_days_in_april}")
        
        if crash_days_in_april > 0:
            print(f"\n🚨 CRASH OVERRIDE TEST:")
            crash_mask = april_data['crash_buy_signal'] == True
            
            for idx in april_data[crash_mask].index:
                relative_idx = idx - april_data.index[0]
                if relative_idx < len(result['buy_signals']):
                    buy_signal = result['buy_signals'].iloc[relative_idx]
                    buy_score = result.get('buy_score', pd.Series([0]*len(result['buy_signals']))).iloc[relative_idx]
                    print(f"      Day {relative_idx}: crash_buy_signal=True → buy_signal={buy_signal}, buy_score={buy_score}")
        
    except Exception as e:
        print(f"❌ Strategy execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_strategy_directly()
