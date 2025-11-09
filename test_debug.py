#!/usr/bin/env python3

import pandas as pd
import numpy as np
from indicators import get_all_indicators
from data_loader import load_data

# Load some test data
print("Loading test data...")
data = load_data("SPY", "1y", "1d")
enriched_data = get_all_indicators(data)

print(f"Data shape: {enriched_data.shape}")
print(f"Columns: {list(enriched_data.columns)[:10]}...")

# Test a simple RSI condition
print("\nTesting RSI_14:")
print(f"RSI_14 range: [{enriched_data['RSI_14'].min():.2f}, {enriched_data['RSI_14'].max():.2f}]")

# Test some thresholds
buy_threshold = 30.0
sell_threshold = 70.0
buy_condition = enriched_data['RSI_14'] < buy_threshold
sell_condition = enriched_data['RSI_14'] > sell_threshold

print(f"RSI < {buy_threshold}: {buy_condition.sum()} conditions")
print(f"RSI > {sell_threshold}: {sell_condition.sum()} conditions")

# Test boolean patterns
print(f"\nTesting patterns:")
for col in enriched_data.columns:
    if col.startswith('pattern_'):
        true_count = enriched_data[col].sum()
        print(f"{col}: {true_count} True values")
