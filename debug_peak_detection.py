#!/usr/bin/env python3
"""
Debug why peak/valley detection is finding so few signals
"""

import yfinance as yf
import pandas as pd
import numpy as np
from peak_valley_detector import PeakValleyDetector

def debug_peak_detection():
    print("🔍 DEBUGGING PEAK/VALLEY DETECTION")
    print("=" * 50)
    
    # Get SPY data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="1y", interval="1d")
    data.columns = [col.lower() for col in data.columns]
    
    print(f"📊 Data Range: {data.index.min()} to {data.index.max()}")
    print(f"📊 Total Days: {len(data)}")
    print(f"📊 Price Range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
    print(f"📊 Total Price Change: {(data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100:.1f}%")
    
    detector = PeakValleyDetector()
    
    # Test different parameters
    test_configs = [
        {"method": "rolling_window", "params": {"window_short": 5, "window_long": 10, "min_change_pct": 1.0}},
        {"method": "rolling_window", "params": {"window_short": 10, "window_long": 20, "min_change_pct": 2.0}},
        {"method": "rolling_window", "params": {"window_short": 10, "window_long": 20, "min_change_pct": 3.0}},
        {"method": "scipy_peaks", "params": {"prominence_pct": 1.0, "distance": 5}},
        {"method": "scipy_peaks", "params": {"prominence_pct": 2.0, "distance": 10}},
        {"method": "percentage_swing", "params": {"swing_pct": 3.0, "lookback": 10}},
        {"method": "percentage_swing", "params": {"swing_pct": 5.0, "lookback": 15}},
    ]
    
    print(f"\n🧪 TESTING DIFFERENT DETECTION PARAMETERS:")
    print("-" * 80)
    print(f"{'Method':<20} {'Parameters':<35} {'Peaks':<8} {'Valleys':<8} {'Total':<8}")
    print("-" * 80)
    
    for config in test_configs:
        method = config["method"]
        params = config["params"]
        
        try:
            if method == "rolling_window":
                labels = detector.rolling_window_method(data, **params)
            elif method == "scipy_peaks":
                labels = detector.scipy_peaks_method(data, **params)
            elif method == "percentage_swing":
                labels = detector.percentage_swing_method(data, **params)
            
            peaks = (labels == -1).sum()
            valleys = (labels == 1).sum()
            total = peaks + valleys
            
            param_str = f"{list(params.values())}"[:35]
            print(f"{method:<20} {param_str:<35} {peaks:<8} {valleys:<8} {total:<8}")
            
        except Exception as e:
            print(f"{method:<20} ERROR: {str(e)[:50]}")
    
    # Let's manually check some obvious peaks and valleys
    print(f"\n📈 MANUAL PEAK/VALLEY ANALYSIS:")
    print("-" * 50)
    
    # Find the highest and lowest points manually
    max_idx = data['high'].idxmax()
    min_idx = data['low'].idxmin()
    
    print(f"Highest Point: {max_idx.strftime('%Y-%m-%d')} - ${data.loc[max_idx, 'high']:.2f}")
    print(f"Lowest Point: {min_idx.strftime('%Y-%m-%d')} - ${data.loc[min_idx, 'low']:.2f}")
    
    # Check 20-day rolling highs/lows
    rolling_high = data['high'].rolling(20, center=True).max()
    rolling_low = data['low'].rolling(20, center=True).min()
    
    # Find where current high equals rolling high (potential peaks)
    potential_peaks = data['high'] == rolling_high
    # Find where current low equals rolling low (potential valleys)
    potential_valleys = data['low'] == rolling_low
    
    print(f"\n20-day Rolling Analysis:")
    print(f"  Potential peaks (high = rolling_max): {potential_peaks.sum()}")
    print(f"  Potential valleys (low = rolling_min): {potential_valleys.sum()}")
    
    # Show some examples
    peak_dates = data[potential_peaks].index[:5]
    valley_dates = data[potential_valleys].index[:5]
    
    print(f"\nFirst 5 potential peaks:")
    for date in peak_dates:
        price = data.loc[date, 'high']
        print(f"  {date.strftime('%Y-%m-%d')}: ${price:.2f}")
    
    print(f"\nFirst 5 potential valleys:")
    for date in valley_dates:
        price = data.loc[date, 'low']
        print(f"  {date.strftime('%Y-%m-%d')}: ${price:.2f}")
    
    # Check why the restrictive conditions are failing
    print(f"\n🔍 ANALYZING WHY CONDITIONS FAIL:")
    print("-" * 40)
    
    # Test the rolling window conditions step by step
    high = data['high']
    low = data['low']
    
    rolling_max_short = high.rolling(10, center=True).max()
    rolling_min_short = low.rolling(10, center=True).min()
    rolling_max_long = high.rolling(20, center=True).max()
    rolling_min_long = low.rolling(20, center=True).min()
    
    # Peak conditions
    cond1 = (high == rolling_max_short)
    cond2 = (high == rolling_max_long)
    cond3 = (high.shift(5) < high * (1 - 0.03))  # 5 days ago was 3% lower
    cond4 = (high.shift(-5) < high * (1 - 0.03))  # 5 days later is 3% lower
    
    print(f"Peak condition breakdown:")
    print(f"  high == rolling_max_short: {cond1.sum()}")
    print(f"  high == rolling_max_long: {cond2.sum()}")
    print(f"  5 days ago was 3%+ lower: {cond3.sum()}")
    print(f"  5 days later is 3%+ lower: {cond4.sum()}")
    print(f"  ALL conditions met: {(cond1 & cond2 & cond3 & cond4).sum()}")
    
    # Valley conditions
    cond1_v = (low == rolling_min_short)
    cond2_v = (low == rolling_min_long)
    cond3_v = (low.shift(5) > low * (1 + 0.03))  # 5 days ago was 3% higher
    cond4_v = (low.shift(-5) > low * (1 + 0.03))  # 5 days later is 3% higher
    
    print(f"\nValley condition breakdown:")
    print(f"  low == rolling_min_short: {cond1_v.sum()}")
    print(f"  low == rolling_min_long: {cond2_v.sum()}")
    print(f"  5 days ago was 3%+ higher: {cond3_v.sum()}")
    print(f"  5 days later is 3%+ higher: {cond4_v.sum()}")
    print(f"  ALL conditions met: {(cond1_v & cond2_v & cond3_v & cond4_v).sum()}")

if __name__ == "__main__":
    debug_peak_detection()
