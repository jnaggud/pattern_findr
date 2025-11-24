#!/usr/bin/env python3
"""
Test if regime-aware crash detection will buy during April 2025
"""
import yfinance as yf
import pandas as pd
from indicators import get_all_indicators
from optimization import universal_strategy
from simple_regime_detector import detect_market_regime

# Get data
spy = yf.Ticker('SPY')
data = spy.history(start='2025-03-01', end='2025-05-01', interval='1d')
data.reset_index(inplace=True)
data.columns = [col.lower() for col in data.columns]

# Add indicators
data = get_all_indicators(data)

# Create test strategy with regime-aware mode
params = {
    'enable_regime_aware': True,
    'use_RSI_14': True,
    'RSI_14_buy': 40,  # Higher threshold
    'RSI_14_sell': 70,
    'use_MACD_12_26_9': True,
    'MACD_12_26_9_buy': 0,
    'MACD_12_26_9_sell': 3,
    'buy_score_threshold': 2,  # Default: need 2 signals
    'sell_score_threshold': 1,
    'signal_persistence_days': 2,
    # Regime-specific thresholds
    'crash_buy_score_threshold': 1,  # Crash: only need 1 signal
    'bear_buy_score_threshold': 1,
    'bull_buy_score_threshold': 2,
    'sideways_buy_score_threshold': 2,
    'crash_sell_score_threshold': 2,
    'bear_sell_score_threshold': 2,
    'bull_sell_score_threshold': 1,
    'sideways_sell_score_threshold': 1,
    'use_trend_filter': False,
}

# Run strategy
signals = universal_strategy(data, params)

# Check April signals
print('\n📊 April 2025 Trading Signals (with Regime-Aware Fix):\n')
april_mask = (data['date'] >= '2025-04-01') & (data['date'] <= '2025-04-30')
april_data = data[april_mask].copy()
april_signals = signals[april_mask]

for idx, (_, row) in enumerate(april_data.iterrows()):
    date_str = row['date'].strftime('%Y-%m-%d')
    signal = april_signals.iloc[idx]
    price = row['close']
    rsi = row.get('RSI_14', 0)
    macd = row.get('MACD_12_26_9', 0)
    
    # Get regime
    data_idx = data[data['date'] == row['date']].index[0]
    regime = detect_market_regime(data, data_idx)
    
    signal_str = '🟢 BUY ' if signal == 1 else ('🔴 SELL' if signal == -1 else '     ')
    print(f'{signal_str} {date_str} | {regime:8s} | Price: ${price:6.2f} | RSI: {rsi:5.1f} | MACD: {macd:6.2f}')

print('\n✅ If you see BUY signals around April 7-9 ($493 bottom), the fix is working!')
print('🔄 Now re-run optimization in Streamlit to generate new strategies with this fix.')
