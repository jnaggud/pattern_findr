#!/usr/bin/env python3

import pandas as pd

# Read the cached data
try:
    data = pd.read_csv("data_cache/SPY_1y_1d.csv")
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    
    # Calculate simple RSI for testing
    import pandas_ta as ta
    data['RSI'] = ta.rsi(data['close'], length=14)
    
    print(f"Data shape: {data.shape}")
    print(f"RSI range: [{data['RSI'].min():.2f}, {data['RSI'].max():.2f}]")
    print(f"RSI mean: {data['RSI'].mean():.2f}")
    
    # Test thresholds
    print(f"\nRSI < 30: {(data['RSI'] < 30).sum()} days")
    print(f"RSI > 70: {(data['RSI'] > 70).sum()} days")
    print(f"RSI < 50: {(data['RSI'] < 50).sum()} days")
    print(f"RSI > 50: {(data['RSI'] > 50).sum()} days")
    
except Exception as e:
    print(f"Error: {e}")
