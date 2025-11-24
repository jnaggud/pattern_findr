#!/usr/bin/env python3
"""
Simple test of Enhanced Backtester without heavy indicator calculation
"""
import yfinance as yf
import pandas as pd
import numpy as np
from backtester_enhanced import EnhancedBacktesterV2

print("📊 Loading SPY data for Feb-April 2025...")
spy = yf.Ticker('SPY')
data = spy.history(start='2025-02-01', end='2025-05-01', interval='1d')
data.reset_index(inplace=True)
data.columns = [col.lower() for col in data.columns]

print(f"✅ Loaded {len(data)} days of data")

# Create a simple strategy function that just returns signals
def simple_crash_strategy(data, params):
    """
    Simple strategy: BUY when price drops 5%+ from recent high, SELL when up 10%
    """
    signals = pd.Series(0, index=data.index)
    
    for i in range(20, len(data)):
        # Look back 20 days
        recent_high = data['high'].iloc[i-20:i].max()
        current_price = data['close'].iloc[i]
        
        # Buy signal: down 5%+ from recent high
        if current_price < recent_high * 0.95:
            signals.iloc[i] = 1
        
        # Sell signal: up 10% from 5 days ago
        if i >= 5:
            price_5d_ago = data['close'].iloc[i-5]
            if current_price > price_5d_ago * 1.10:
                signals.iloc[i] = -1
    
    return signals

print("\n" + "=" * 60)
print("TEST 1: WITHOUT Stop-Loss/Take-Profit")
print("=" * 60)

# Test without enhancements
from backtester import Backtester
params_std = {}
bt_std = Backtester(data, "Simple", simple_crash_strategy, params_std, 100000)
bt_std.run()
trades_std, summary_std = bt_std.get_results()

print(f"\n📈 Results:")
print(f"  Trades: {summary_std['total_trades']}")
print(f"  Return: {summary_std['total_return_pct']:.2f}%")
print(f"  Win Rate: {summary_std['win_rate']:.1f}%")

if len(trades_std) > 0:
    print(f"\n📋 Trade Log:")
    for _, t in trades_std.iterrows():
        entry_dt = pd.to_datetime(t['entry_date'])
        exit_dt = pd.to_datetime(t['exit_date'])
        days = (exit_dt - entry_dt).days
        print(f"  {entry_dt.date()} → {exit_dt.date()} ({days:3d} days) | "
              f"${t['entry_price']:.2f} → ${t['exit_price']:.2f} | "
              f"${t['profit']:+,.2f}")

print("\n" + "=" * 60)
print("TEST 2: WITH Stop-Loss 8% / Take-Profit 15%")
print("=" * 60)

# Test WITH enhancements
params_enh = {
    'stop_loss_pct': 8.0,
    'take_profit_pct': 15.0,
    'use_stop_loss': True,
    'use_take_profit': True
}

bt_enh = EnhancedBacktesterV2(data, "Enhanced", simple_crash_strategy, params_enh, 100000)
bt_enh.run()
trades_enh, summary_enh = bt_enh.get_results()

print(f"\n📈 Results:")
print(f"  Trades: {summary_enh['total_trades']}")
print(f"  Return: {summary_enh['total_return_pct']:.2f}%")
print(f"  Win Rate: {summary_enh['win_rate']:.1f}%")
print(f"  Avg Hold: {summary_enh['avg_hold_days']:.1f} days")
print(f"  Stop-Loss Exits: {summary_enh['stop_loss_exits']}")
print(f"  Take-Profit Exits: {summary_enh['take_profit_exits']}")

if len(trades_enh) > 0:
    print(f"\n📋 Trade Log:")
    for _, t in trades_enh.iterrows():
        entry_dt = pd.to_datetime(t['entry_date'])
        exit_dt = pd.to_datetime(t['exit_date'])
        exit_reason = t.get('exit_reason', 'SIGNAL')
        print(f"  {entry_dt.date()} → {exit_dt.date()} ({t['hold_days']:3d} days) | "
              f"${t['entry_price']:.2f} → ${t['exit_price']:.2f} | "
              f"${t['profit']:+,.2f} ({t['return_pct']:+.1f}%) | {exit_reason}")

print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)
print(f"Standard:  {summary_std['total_trades']} trades, {summary_std['total_return_pct']:+.2f}% return")
print(f"Enhanced:  {summary_enh['total_trades']} trades, {summary_enh['total_return_pct']:+.2f}% return")
print(f"Difference: {summary_enh['total_trades'] - summary_std['total_trades']:+d} trades, "
      f"{summary_enh['total_return_pct'] - summary_std['total_return_pct']:+.2f}% return")

print("\n✅ Enhanced backtester allows controlled exits!")
print("   - Stops prevent big losses")
print("   - Take-profit locks in gains")
print("   - Frees capital for new opportunities (like April crash)")
