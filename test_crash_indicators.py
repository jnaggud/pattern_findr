#!/usr/bin/env python3

import yfinance as yf
from indicators import get_all_indicators

print("🚨 TESTING CRASH INDICATORS on April 2025")
print("=" * 50)

# Get data and add crash indicators
ticker = 'SPY'
data = yf.Ticker(ticker).history(period='1y', interval='1d')
data.reset_index(inplace=True)
data.columns = [col.lower() for col in data.columns]
enriched_data = get_all_indicators(data)

# Focus on April crash
april_data = enriched_data[enriched_data['date'].dt.month == 4]
crash_period = april_data[(april_data['date'].dt.day >= 3) & (april_data['date'].dt.day <= 15)]

print(f"📅 April 3-15 crash period: {len(crash_period)} days")
print()

# Check crash indicators
for _, row in crash_period.iterrows():
    date = row['date'].strftime('%Y-%m-%d')
    price = row['close']
    
    # Check individual crash indicators  
    velocity = row.get('crash_velocity_3d', False)
    volume_panic = row.get('volume_panic', False)
    drawdown = row.get('drawdown_capitulation', False)
    gap_down = row.get('gap_down_panic', False)
    carnage = row.get('multi_day_carnage', False)
    
    # Check composite scores
    crash_score = row.get('crash_composite_score', 0)
    fear_score = row.get('fear_composite_score', 0)
    crash_buy = row.get('crash_buy_signal', False)
    
    indicators_triggered = []
    if velocity: indicators_triggered.append('Velocity')
    if volume_panic: indicators_triggered.append('VolPanic') 
    if drawdown: indicators_triggered.append('Drawdown')
    if gap_down: indicators_triggered.append('Gap')
    if carnage: indicators_triggered.append('Carnage')
    
    trigger_text = '+'.join(indicators_triggered) if indicators_triggered else 'None'
    
    print(f"{date}: ${price:.2f} | Crash:{crash_score} Fear:{fear_score} | BUY:{crash_buy} | [{trigger_text}]")

# Summary
total_crash_buys = crash_period['crash_buy_signal'].sum()
print(f"\n📊 CRASH INDICATORS SUMMARY:")
print(f"   Total crash buy signals: {total_crash_buys}/{len(crash_period)} days")

if total_crash_buys > 0:
    print(f"\n🎉 SUCCESS! Crash indicators would have triggered buy signals!")
    print(f"   This solves the problem of buy_score = 0 during crashes")
    
    # Show which days triggered
    crash_buy_days = crash_period[crash_period['crash_buy_signal'] == True]
    print(f"\n📈 CRASH BUY SIGNAL DAYS:")
    for _, row in crash_buy_days.iterrows():
        print(f"   {row['date'].strftime('%Y-%m-%d')}: ${row['close']:.2f}")
else:
    print(f"\n⚠️  Crash indicators not sensitive enough - need to tune thresholds")
    
    # Show the values to understand why they didn't trigger
    print(f"\n🔍 DIAGNOSTIC - Why no crash signals?")
    for _, row in crash_period.head(3).iterrows():
        date = row['date'].strftime('%Y-%m-%d')
        print(f"\n   {date}:")
        print(f"      3-day return: {((row['close']/crash_period.iloc[max(0, row.name-3)]['close'])-1)*100:.1f}%")
        if 'volume' in row:
            volume_ratio = row['volume'] / april_data['volume'].rolling(20).mean().iloc[row.name] if not pd.isna(april_data['volume'].rolling(20).mean().iloc[row.name]) else 1
            print(f"      Volume ratio: {volume_ratio:.1f}x")
        rolling_high = april_data['close'].rolling(20).max().iloc[row.name]
        if not pd.isna(rolling_high):
            drawdown_pct = (row['close'] / rolling_high - 1) * 100
            print(f"      Drawdown: {drawdown_pct:.1f}%")
