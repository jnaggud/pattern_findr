#!/usr/bin/env python3

# Simple test to understand the parameter structure issue

# Test the universal_strategy function directly
import pandas as pd
import numpy as np

# Create mock data
data = pd.DataFrame({
    'RSI_14': np.random.uniform(20, 80, 100),
    'close': np.random.uniform(500, 600, 100)
})
data.index = pd.date_range('2023-01-01', periods=100, freq='D')

# Test params structure that should work
params = {
    'use_RSI_14': True,
    'RSI_14_buy': 30.0,
    'RSI_14_sell': 70.0,
    'buy_score_threshold': 1,
    'sell_score_threshold': 1
}

print("Test params:", params)
print("RSI_14 range:", data['RSI_14'].min(), "to", data['RSI_14'].max())

# Test active indicators extraction
active_indicators = [param.replace('use_', '') for param in params.keys() 
                    if param.startswith('use_') and params[param] == True]

print("Active indicators:", active_indicators)

# Test parameter access
for indicator in active_indicators:
    print(f"  {indicator}_buy: {params.get(f'{indicator}_buy')}")
    print(f"  {indicator}_sell: {params.get(f'{indicator}_sell')}")
    
    # Test condition
    if indicator == 'RSI_14':
        buy_condition = data[indicator] < params.get(f'{indicator}_buy')
        sell_condition = data[indicator] > params.get(f'{indicator}_sell')
        print(f"  Buy conditions: {buy_condition.sum()}")
        print(f"  Sell conditions: {sell_condition.sum()}")
