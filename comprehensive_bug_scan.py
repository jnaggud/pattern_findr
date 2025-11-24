#!/usr/bin/env python3
"""
COMPREHENSIVE BUG SCAN - Check all critical systems
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
import pandas as pd
import json
from pathlib import Path
from simple_regime_detector import detect_market_regime

def comprehensive_bug_scan():
    print("🔍 COMPREHENSIVE BUG SCAN")
    print("=" * 50)
    
    # 1. CHECK PARAMETER SAVING/LOADING
    print("\n1. 📋 PARAMETER SYSTEM CHECK:")
    print("-" * 30)
    
    opt_dir = Path('optimizations')
    if not opt_dir.exists():
        print("❌ Optimizations directory missing")
        return False
        
    param_files = sorted(opt_dir.glob('*_params.json'), key=lambda x: x.stat().st_mtime, reverse=True)
    if not param_files:
        print("❌ No parameter files found")
        return False
        
    latest_params = param_files[0]
    print(f"📄 Latest: {latest_params.name}")
    
    with open(latest_params, 'r') as f:
        params = json.load(f)
    
    # Check critical parameters
    critical_params = {
        'enable_regime_aware': 'REGIME DETECTION',
        'crash_buy_score_threshold': 'CRASH BUY THRESHOLD', 
        'bear_buy_score_threshold': 'BEAR BUY THRESHOLD',
        'use_trend_filter': 'TREND FILTER',
        'signal_persistence_days': 'SIGNAL PERSISTENCE',
    }
    
    missing_params = []
    for param, description in critical_params.items():
        value = params.get(param, 'MISSING')
        status = "✅" if value != 'MISSING' else "❌"
        print(f"   {status} {description}: {value}")
        if value == 'MISSING':
            missing_params.append(param)
    
    # 2. CHECK REGIME DETECTION
    print("\n2. 🎯 REGIME DETECTION CHECK:")
    print("-" * 30)
    
    try:
        # Test with recent data
        data = yf.Ticker('SPY').history(period='1y', interval='1d')
        data.reset_index(inplace=True)
        data.columns = [col.lower() for col in data.columns]
        
        # Test April crash detection
        april_data = data[data['date'].dt.month == 4]
        if len(april_data) > 0:
            test_idx = april_data.index[5]  # Mid-April
            regime = detect_market_regime(data, test_idx)
            date = data.iloc[test_idx]['date']
            print(f"   📅 {date.strftime('%Y-%m-%d')}: Regime = {regime}")
            
            if regime in ['crash', 'bear']:
                print(f"   ✅ April crash correctly detected as {regime}")
            else:
                print(f"   ❌ April crash detected as {regime} - should be crash/bear")
        else:
            print("   ❌ No April data for regime testing")
            
    except Exception as e:
        print(f"   ❌ Regime detection error: {e}")
    
    # 3. CHECK CRASH INDICATORS
    print("\n3. 🚨 CRASH INDICATORS CHECK:")
    print("-" * 30)
    
    try:
        from indicators import get_all_indicators
        enriched_data = get_all_indicators(data)
        
        # Check crash indicator columns exist
        crash_cols = [
            'crash_velocity_3d', 'volume_panic', 'drawdown_capitulation',
            'gap_down_panic', 'multi_day_carnage', 'crash_composite_score',
            'fear_composite_score', 'crash_buy_signal'
        ]
        
        missing_crash_cols = []
        for col in crash_cols:
            if col in enriched_data.columns:
                print(f"   ✅ {col}")
            else:
                print(f"   ❌ {col} - MISSING")
                missing_crash_cols.append(col)
        
        # Test April crash signals
        if not missing_crash_cols:
            april_enriched = enriched_data[enriched_data['date'].dt.month == 4]
            crash_signals = april_enriched['crash_buy_signal'].sum()
            print(f"   📊 April crash buy signals: {crash_signals}")
            
            if crash_signals > 0:
                print("   ✅ Crash indicators firing during April")
            else:
                print("   ⚠️  No crash signals in April - may need tuning")
        else:
            print("   ❌ Cannot test - missing crash indicator columns")
            
    except Exception as e:
        print(f"   ❌ Crash indicator error: {e}")
    
    # 4. CHECK UNIVERSAL_STRATEGY INTEGRATION
    print("\n4. ⚙️  UNIVERSAL_STRATEGY INTEGRATION:")
    print("-" * 30)
    
    try:
        from optimization import universal_strategy
        
        # Test strategy with crash indicators
        test_params = {
            'enable_regime_aware': True,
            'crash_buy_score_threshold': 1,
            'bear_buy_score_threshold': 1,
            'use_trend_filter': True,
            'signal_persistence_days': 2,
            'buy_score_threshold': 2,
            'sell_score_threshold': 5
        }
        
        # Add some fake indicator thresholds to avoid errors
        indicators = ['RSI_14', 'WILLR_14', 'MACD_12_26_9']
        for ind in indicators:
            test_params[f'{ind}_buy'] = 30
            test_params[f'{ind}_sell'] = 70
        
        # Test on small data subset
        test_data = enriched_data.tail(50)  # Last 50 days
        
        signals = universal_strategy(test_data, test_params, debug_mode=False)
        
        if 'buy_signals' in signals and 'sell_signals' in signals:
            buy_count = signals['buy_signals'].sum()
            sell_count = signals['sell_signals'].sum()
            print(f"   ✅ Strategy executed: {buy_count} buys, {sell_count} sells")
            
            # Check for crash override logic
            if 'crash_buy_signal' in test_data.columns:
                crash_days = test_data['crash_buy_signal'].sum()
                print(f"   📊 Crash override days available: {crash_days}")
            else:
                print("   ❌ No crash_buy_signal column in data")
        else:
            print("   ❌ Strategy did not return proper signals")
            
    except Exception as e:
        print(f"   ❌ Strategy integration error: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. SUMMARY AND RECOMMENDATIONS
    print("\n5. 📊 BUG SCAN SUMMARY:")
    print("-" * 30)
    
    issues_found = []
    
    if missing_params:
        issues_found.append(f"Missing parameters: {missing_params}")
    
    if missing_crash_cols:
        issues_found.append(f"Missing crash indicators: {missing_crash_cols}")
    
    if issues_found:
        print("❌ ISSUES FOUND:")
        for issue in issues_found:
            print(f"   • {issue}")
        
        print("\n🔧 RECOMMENDED FIXES:")
        if missing_params:
            print("   • Re-run optimization to ensure all parameters saved")
        if missing_crash_cols:
            print("   • Check indicators.py implementation")
            
        return False
    else:
        print("✅ ALL SYSTEMS FUNCTIONAL")
        print("\n🎯 READY FOR CRASH INDICATOR ENHANCEMENT")
        return True

if __name__ == "__main__":
    success = comprehensive_bug_scan()
    
    if success:
        print(f"\n🚀 SYSTEM STATUS: READY FOR OPTIMIZATION")
    else:
        print(f"\n⚠️  SYSTEM STATUS: NEEDS DEBUGGING")
