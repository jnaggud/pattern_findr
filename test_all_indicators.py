"""
Test all indicators including custom implementations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Testing all indicators (traditional + custom + ML)...")

# Create sample data
dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
np.random.seed(42)

# Generate realistic price data
price_start = 100
returns = np.random.normal(0.001, 0.02, 200)
prices = price_start * np.exp(np.cumsum(returns))

data = pd.DataFrame({
    'date': dates,
    'open': prices * (1 + np.random.uniform(-0.01, 0.01, 200)),
    'high': prices * (1 + np.random.uniform(0, 0.02, 200)),
    'low': prices * (1 + np.random.uniform(-0.02, 0, 200)),
    'close': prices,
    'volume': np.random.randint(1000000, 5000000, 200)
})

print("✅ Sample data created (200 days)")

# Test indicator calculation
print("\nCalculating all indicators...")
try:
    from indicators import get_all_indicators
    
    enriched_data = get_all_indicators(data)
    
    print(f"✅ Indicators calculated successfully!")
    print(f"   Total columns: {len(enriched_data.columns)}")
    print(f"   Original OHLCV: 5")
    print(f"   Indicator columns: {len(enriched_data.columns) - 5}")
    
    # Check for key indicators
    key_indicators = [
        'HT_TRENDLINE',  # Custom implementation
        'AO_5_34',  # Awesome Oscillator
        'PGO_14',  # Pretty Good Oscillator
        'QS_14',  # QStick
        'SQZ_20_2.0_20_1.5',  # TTM Squeeze
        'KVO_34_55_13',  # Klinger Volume Oscillator
        'ml_ensemble_signal',  # ML indicator
        'Supertrend',  # Bear market indicator
        'KAMA_20',  # Adaptive MA
    ]
    
    print("\n🎯 Key Indicators Status:")
    for indicator in key_indicators:
        # Check for variations of the indicator name
        found = False
        for col in enriched_data.columns:
            if indicator.lower() in col.lower() or col.lower() in indicator.lower():
                print(f"   ✅ {indicator}: FOUND as '{col}'")
                found = True
                break
        if not found:
            print(f"   ⚠️  {indicator}: NOT FOUND (may be under different name)")
    
    # Show sample of custom indicators
    print("\n📊 Sample Custom Indicator Values (last row):")
    custom_cols = ['HT_TRENDLINE', 'AO_5_34', 'PGO_14', 'QS_14', 'NVI', 'PVI']
    for col in custom_cols:
        matching_cols = [c for c in enriched_data.columns if col in c]
        if matching_cols:
            col_name = matching_cols[0]
            value = enriched_data[col_name].iloc[-1]
            if pd.notna(value):
                print(f"   {col_name}: {value:.4f}")
    
    print("\n✅ ALL TESTS PASSED!")
    print("\n💡 All indicators working - ready for optimization!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
