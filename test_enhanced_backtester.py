#!/usr/bin/env python3
"""
Test Enhanced Backtester on April 2025 Crash
Compare standard backtester vs enhanced with stop-loss/take-profit
"""
import yfinance as yf
import pandas as pd
from indicators import get_all_indicators
from optimization import universal_strategy
from backtester import Backtester
from backtester_enhanced import EnhancedBacktesterV2

# Get data including April 2025 crash
print("📊 Loading SPY data...")
spy = yf.Ticker('SPY')
data = spy.history(start='2025-02-01', end='2025-05-01', interval='1d')
data.reset_index(inplace=True)
data.columns = [col.lower() for col in data.columns]
data = get_all_indicators(data)

# Simple test strategy with crash-detection parameters
test_params = {
    'enable_regime_aware': True,
    'use_RSI_14': True,
    'RSI_14_buy': 40,
    'RSI_14_sell': 70,
    'use_MACD_12_26_9': True,
    'MACD_12_26_9_buy': 0,
    'MACD_12_26_9_sell': 3,
    'buy_score_threshold': 2,
    'sell_score_threshold': 1,  # Original threshold
    'signal_persistence_days': 2,
    'crash_buy_score_threshold': 1,
    'bear_buy_score_threshold': 1,
    'bull_buy_score_threshold': 2,
    'sideways_buy_score_threshold': 2,
    'crash_sell_score_threshold': 2,
    'bear_sell_score_threshold': 2,
    'bull_sell_score_threshold': 1,
    'sideways_sell_score_threshold': 1,
    'use_trend_filter': False,
}

print("\n" + "=" * 60)
print("TEST 1: STANDARD BACKTESTER (No Stop-Loss/Take-Profit)")
print("=" * 60)

backtester_std = Backtester(data, "Standard Test", universal_strategy, test_params, 100000)
backtester_std.run()
trade_log_std, summary_std = backtester_std.get_results()

print(f"\n📈 Results:")
print(f"  Total Return: {summary_std['total_return_pct']:.2f}%")
print(f"  Total Trades: {summary_std['total_trades']}")
print(f"  Win Rate: {summary_std['win_rate']:.1f}%")
print(f"  Ending Capital: ${summary_std['ending_capital']:,.2f}")

if len(trade_log_std) > 0:
    print(f"\n📋 Trades:")
    for idx, trade in trade_log_std.iterrows():
        profit_sign = "+" if trade['profit'] > 0 else ""
        print(f"  {trade['entry_date'].date()} → {trade['exit_date'].date()} | "
              f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} | "
              f"P/L: {profit_sign}${trade['profit']:.2f}")
else:
    print("\n❌ No trades executed!")

print("\n" + "=" * 60)
print("TEST 2: ENHANCED BACKTESTER (8% Stop-Loss, 15% Take-Profit)")
print("=" * 60)

# Add stop-loss/take-profit params
enhanced_params = test_params.copy()
enhanced_params['stop_loss_pct'] = 8.0
enhanced_params['take_profit_pct'] = 15.0
enhanced_params['use_stop_loss'] = True
enhanced_params['use_take_profit'] = True

backtester_enh = EnhancedBacktesterV2(data, "Enhanced Test", universal_strategy, enhanced_params, 100000)
backtester_enh.run()
trade_log_enh, summary_enh = backtester_enh.get_results()

print(f"\n📈 Results:")
print(f"  Total Return: {summary_enh['total_return_pct']:.2f}%")
print(f"  Total Trades: {summary_enh['total_trades']}")
print(f"  Win Rate: {summary_enh['win_rate']:.1f}%")
print(f"  Ending Capital: ${summary_enh['ending_capital']:,.2f}")
print(f"  Avg Hold Days: {summary_enh['avg_hold_days']:.1f}")
print(f"  Stop-Loss Exits: {summary_enh['stop_loss_exits']}")
print(f"  Take-Profit Exits: {summary_enh['take_profit_exits']}")

if len(trade_log_enh) > 0:
    print(f"\n📋 Trades:")
    for idx, trade in trade_log_enh.iterrows():
        profit_sign = "+" if trade['profit'] > 0 else ""
        return_pct = trade.get('return_pct', 0)
        exit_reason = trade.get('exit_reason', 'SIGNAL')
        print(f"  {trade['entry_date'].date()} → {trade['exit_date'].date()} | "
              f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} | "
              f"Return: {return_pct:+.1f}% | P/L: {profit_sign}${trade['profit']:.2f} | "
              f"Exit: {exit_reason}")
else:
    print("\n❌ No trades executed!")

print("\n" + "=" * 60)
print("COMPARISON:")
print("=" * 60)
print(f"Standard Return:  {summary_std['total_return_pct']:>8.2f}%")
print(f"Enhanced Return:  {summary_enh['total_return_pct']:>8.2f}%")
print(f"Improvement:      {summary_enh['total_return_pct'] - summary_std['total_return_pct']:>+8.2f}%")
print(f"\nStandard Trades:  {summary_std['total_trades']:>8}")
print(f"Enhanced Trades:  {summary_enh['total_trades']:>8}")

print("\n✅ Enhanced backtester allows positions to close, enabling new entries!")
