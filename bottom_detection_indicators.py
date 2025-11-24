#!/usr/bin/env python3
"""
BOTTOM DETECTION INDICATORS - Catch the actual bottom, not the crash start
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
import pandas as pd
import numpy as np

def bottom_detection_indicators():
    print("💎 BOTTOM DETECTION INDICATORS")
    print("=" * 40)
    
    # Get SPY data
    print("Fetching data...")
    data = yf.Ticker('SPY').history(period='1y', interval='1d')
    data.reset_index(inplace=True)
    data.columns = [col.lower() for col in data.columns]
    
    print("Creating BOTTOM detection indicators...")
    
    # === BOTTOM DETECTION LOGIC ===
    # Look for signs that selling is exhausted and reversal is starting
    
    # 1. SELLING EXHAUSTION - Volume spikes but price stops falling
    volume_avg = data['volume'].rolling(20).mean()
    volume_ratio = data['volume'] / volume_avg
    daily_return = data['close'].pct_change() * 100
    
    # High volume but smaller declines = selling exhaustion
    data['selling_exhaustion'] = (volume_ratio > 2.5) & (daily_return > -2.0) & (daily_return < 0.5)
    
    # 2. HAMMER CANDLES - Long lower wicks (rejection of lower prices)
    candle_range = data['high'] - data['low']
    lower_wick = data['close'] - data['low']  # Assume close > low for simplicity
    upper_wick = data['high'] - data['close']
    
    # Hammer: Long lower wick, small upper wick, small body
    data['hammer_pattern'] = (lower_wick > candle_range * 0.6) & (upper_wick < candle_range * 0.2)
    
    # 3. REVERSAL DIVERGENCE - Price makes new low but momentum doesn't
    # Use rolling minimums as proxy for momentum divergence
    rolling_low_5d = data['low'].rolling(5).min()
    rolling_low_10d = data['low'].rolling(10).min()
    
    # New 5-day low but not 10-day low = potential divergence
    data['momentum_divergence'] = (data['low'] == rolling_low_5d) & (data['low'] > rolling_low_10d)
    
    # 4. PANIC SPIKE RECOVERY - Sharp decline followed by recovery
    returns_1d = data['close'].pct_change(1) * 100
    returns_2d = data['close'].pct_change(2) * 100
    
    # Yesterday was panic (-4%+), today is recovering (+1%+)
    data['panic_recovery'] = (returns_1d.shift(1) < -4.0) & (returns_1d > 1.0)
    
    # 5. OVERSOLD BOUNCE - Extreme oversold followed by bounce
    # Use 3-day RSI proxy (percentage of up days)
    up_days = (returns_1d > 0).rolling(3).sum()
    oversold_proxy = up_days == 0  # 0 up days in last 3 = oversold
    
    # Oversold condition ending with bounce
    data['oversold_bounce'] = oversold_proxy.shift(1) & (returns_1d > 0.5)
    
    # 6. VOLUME CONFIRMATION - High volume on bounce (smart money buying)
    # Previous day high volume + today bounce + today high volume
    high_volume = volume_ratio > 2.0
    data['volume_confirmation'] = (
        high_volume.shift(1) &  # Yesterday high volume
        (returns_1d > 0.5) &    # Today bounce
        high_volume             # Today also high volume
    )
    
    # 7. MULTI-DAY BOTTOM - Price stabilizes after decline
    # Look for 2+ days of small moves after big decline
    big_decline_window = returns_2d < -5.0  # 5% decline in 2 days
    small_moves = abs(returns_1d) < 1.5     # Small daily moves
    
    # Big decline followed by 2 days of stability
    stability = small_moves & small_moves.shift(1)
    data['stabilization_bottom'] = big_decline_window.shift(2) & stability
    
    # 8. GAP FILL RECOVERY - Gap down followed by gap fill
    prev_close = data['close'].shift(1)
    gap_down = (data['open'] / prev_close - 1) * 100 < -2.0
    gap_fill = data['close'] > prev_close  # Close above previous close
    
    data['gap_fill_recovery'] = gap_down & gap_fill
    
    # 9. FEAR CAPITULATION - Extreme fear followed by relief
    # Multiple panic indicators yesterday, calmer today
    fear_indicators = [
        returns_1d < -3.0,      # Big decline
        volume_ratio > 3.0,     # Huge volume
        candle_range / data['close'] > 0.04  # Big intraday range (4%+)
    ]
    
    fear_score = pd.Series(0, index=data.index)
    for indicator in fear_indicators:
        fear_score += indicator.astype(int)
    
    # High fear yesterday, lower fear today
    data['fear_capitulation'] = (fear_score.shift(1) >= 2) & (fear_score <= 1)
    
    # 10. COMPOSITE BOTTOM SCORE
    bottom_indicators = [
        'selling_exhaustion', 'hammer_pattern', 'momentum_divergence',
        'panic_recovery', 'oversold_bounce', 'volume_confirmation',
        'stabilization_bottom', 'gap_fill_recovery', 'fear_capitulation'
    ]
    
    data['bottom_score'] = data[bottom_indicators].sum(axis=1)  # Max: 9
    
    # 11. BOTTOM BUY SIGNALS - Different confidence levels
    # Conservative: 2+ bottom indicators
    data['bottom_buy_conservative'] = data['bottom_score'] >= 2
    
    # Moderate: 1+ bottom indicators + high volume
    data['bottom_buy_moderate'] = (data['bottom_score'] >= 1) & (volume_ratio > 2.0)
    
    # Aggressive: Any bottom indicator
    data['bottom_buy_aggressive'] = data['bottom_score'] >= 1
    
    # Ultimate: Best combination
    data['bottom_buy_ultimate'] = (
        data['bottom_buy_conservative'] |  # Conservative signal OR
        data['panic_recovery'] |           # Panic recovery OR  
        data['volume_confirmation'] |      # Volume confirmation OR
        (data['hammer_pattern'] & (volume_ratio > 2.0))  # Hammer + volume
    )
    
    # === TEST ON APRIL 2025 BOTTOM ===
    print("\n💎 TESTING ON APRIL 2025 BOTTOM:")
    print("-" * 40)
    
    april_data = data[data['date'].dt.month == 4]
    
    # Find the actual bottom (lowest close)
    bottom_idx = april_data['close'].idxmin()
    bottom_date = april_data.loc[bottom_idx, 'date']
    bottom_price = april_data.loc[bottom_idx, 'close']
    
    print(f"📍 ACTUAL BOTTOM: {bottom_date.strftime('%Y-%m-%d')} at ${bottom_price:.2f}")
    print()
    
    # Test bottom detection around the actual bottom
    test_period = april_data[(april_data['date'].dt.day >= 5) & (april_data['date'].dt.day <= 15)]
    
    for _, row in test_period.iterrows():
        date = row['date'].strftime('%m-%d')
        price = row['close']
        
        bottom_score = row['bottom_score']
        ultimate_buy = row['bottom_buy_ultimate']
        
        # Show which indicators triggered
        triggered = []
        for ind in bottom_indicators:
            if row.get(ind, False):
                triggered.append(ind.replace('_', '')[:4])  # Short names
        
        trigger_text = '+'.join(triggered) if triggered else 'None'
        
        # Distance from actual bottom
        days_from_bottom = (row['date'] - bottom_date).days
        if days_from_bottom == 0:
            timing = " 🎯EXACT"
        elif abs(days_from_bottom) <= 2:
            timing = f" 📍±{abs(days_from_bottom)}d"
        else:
            timing = f" ({days_from_bottom:+d}d)"
        
        signal = f"🚨BUY (Score:{bottom_score})" if ultimate_buy else "HOLD"
        
        print(f"{date}: ${price:6.2f} | {signal:15} | [{trigger_text:20}]{timing}")
    
    # Summary
    bottom_signals = test_period['bottom_buy_ultimate'].sum()
    
    # Check timing relative to actual bottom
    actual_bottom_row = test_period[test_period['date'] == bottom_date]
    if len(actual_bottom_row) > 0:
        bottom_triggered = actual_bottom_row.iloc[0]['bottom_buy_ultimate']
        print(f"\n📊 BOTTOM DETECTION RESULTS:")
        print(f"   Total signals in test period: {bottom_signals}")
        print(f"   Signal on actual bottom day: {'✅ YES' if bottom_triggered else '❌ NO'}")
        
        # Find closest signal to bottom
        signal_days = test_period[test_period['bottom_buy_ultimate']]
        if len(signal_days) > 0:
            distances = abs((signal_days['date'] - bottom_date).dt.days)
            closest_distance = distances.min()
            print(f"   Closest signal to bottom: {closest_distance} days")
            
            if closest_distance <= 1:
                print(f"   ✅ EXCELLENT: Caught bottom within 1 day!")
                return True
            elif closest_distance <= 3:
                print(f"   ✅ GOOD: Caught bottom within 3 days")
                return True
            else:
                print(f"   ⚠️  Timing needs improvement")
                return False
        else:
            print(f"   ❌ No bottom signals detected")
            return False
    
    return False

if __name__ == "__main__":
    success = bottom_detection_indicators()
    
    if success:
        print(f"\n🎉 BOTTOM DETECTION WORKING!")
        print(f"   Ready to catch actual market bottoms")
        print(f"   Integration will improve crash catching timing")
    else:
        print(f"\n⚠️  Bottom detection needs refinement")
