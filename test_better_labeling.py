#!/usr/bin/env python3
"""
Test better labeling strategies for ML training
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
import pandas as pd
import numpy as np
from peak_valley_detector import PeakValleyDetector

def test_labeling_strategies():
    print("🔍 TESTING DIFFERENT LABELING STRATEGIES")
    print("=" * 60)
    
    # Get test data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="1y", interval="1d")
    data.columns = [col.lower() for col in data.columns]
    
    detector = PeakValleyDetector()
    
    # Test different methods
    methods = [
        {
            'name': 'Conservative (Current)',
            'method': 'scipy_peaks',
            'params': {'prominence_pct': 2.0, 'distance': 10}
        },
        {
            'name': 'Moderate',
            'method': 'scipy_peaks', 
            'params': {'prominence_pct': 1.0, 'distance': 5}
        },
        {
            'name': 'Aggressive',
            'method': 'scipy_peaks',
            'params': {'prominence_pct': 0.8, 'distance': 3}
        },
        {
            'name': 'Percentage-Based',
            'method': 'percentage_swing',
            'params': {'swing_pct': 2.0}
        },
        {
            'name': 'Rolling Window',
            'method': 'rolling_window',
            'params': {'window_short': 3, 'window_long': 10, 'min_change_pct': 1.5}
        }
    ]
    
    results = []
    
    for method_config in methods:
        print(f"\n🎯 TESTING {method_config['name']}:")
        print("-" * 40)
        
        try:
            _, labels = detector.create_labeled_dataset(
                data, 
                method=method_config['method'],
                **method_config['params']
            )
            
            # Analyze label distribution
            label_counts = labels.value_counts().sort_index()
            total = len(labels)
            
            buy_count = label_counts.get(1, 0)
            sell_count = label_counts.get(-1, 0)
            hold_count = label_counts.get(0, 0)
            
            signal_ratio = (buy_count + sell_count) / total * 100
            
            print(f"   BUY: {buy_count} ({buy_count/total*100:.1f}%)")
            print(f"   SELL: {sell_count} ({sell_count/total*100:.1f}%)")  
            print(f"   HOLD: {hold_count} ({hold_count/total*100:.1f}%)")
            print(f"   Signal Ratio: {signal_ratio:.1f}%")
            
            # Check balance
            if signal_ratio < 5:
                status = "❌ Too Conservative"
            elif signal_ratio > 20:
                status = "⚠️ Too Aggressive"
            else:
                status = "✅ Good Balance"
            
            print(f"   Status: {status}")
            
            results.append({
                'method': method_config['name'],
                'buy_count': buy_count,
                'sell_count': sell_count,
                'signal_ratio': signal_ratio,
                'status': status
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            
    print(f"\n📊 SUMMARY:")
    print("-" * 20)
    for result in results:
        print(f"{result['method']:<20} | Signals: {result['buy_count']+result['sell_count']:2d} | Ratio: {result['signal_ratio']:4.1f}% | {result['status']}")
    
    # Recommend best method
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    
    good_methods = [r for r in results if r['signal_ratio'] >= 8 and r['signal_ratio'] <= 15]
    
    if good_methods:
        best = max(good_methods, key=lambda x: x['signal_ratio'])
        print(f"✅ Recommended: {best['method']} ({best['signal_ratio']:.1f}% signals)")
        print(f"   This provides {best['buy_count']} BUY + {best['sell_count']} SELL signals")
    else:
        print("⚠️ All methods are either too conservative or too aggressive")
        print("   Consider using percentage-based labeling or custom thresholds")

if __name__ == "__main__":
    test_labeling_strategies()
