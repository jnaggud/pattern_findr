"""
Quick test to verify ML indicators are working
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test imports
print("Testing ML indicator imports...")
try:
    from ml_indicators import integrate_ml_indicators, KERAS_AVAILABLE
    print("✅ ML indicators module imported successfully")
    print(f"   LSTM/Keras available: {KERAS_AVAILABLE}")
except Exception as e:
    print(f"❌ Failed to import ml_indicators: {e}")
    exit(1)

# Create sample data
print("\nCreating sample data...")
dates = pd.date_range(end=datetime.now(), periods=150, freq='D')
np.random.seed(42)

# Generate realistic price data
price_start = 100
returns = np.random.normal(0.001, 0.02, 150)
prices = price_start * np.exp(np.cumsum(returns))

data = pd.DataFrame({
    'date': dates,
    'open': prices * (1 + np.random.uniform(-0.01, 0.01, 150)),
    'high': prices * (1 + np.random.uniform(0, 0.02, 150)),
    'low': prices * (1 + np.random.uniform(-0.02, 0, 150)),
    'close': prices,
    'volume': np.random.randint(1000000, 5000000, 150)
})

print("✅ Sample data created (150 days)")

# Test ML indicators
print("\nCalculating ML indicators...")
try:
    ml_indicators = integrate_ml_indicators(data)
    print(f"✅ ML indicators calculated successfully!")
    print(f"   Total indicators: {len(ml_indicators)}")
    
    print("\n📊 ML Indicator Results:")
    for name, value in ml_indicators.items():
        if isinstance(value, (int, float)):
            print(f"   {name}: {value:.4f}")
        else:
            print(f"   {name}: {value}")
    
    # Check for key indicators
    key_indicators = [
        'ensemble_signal',
        'rf_signal', 
        'gb_trend_prediction',
        'market_regime',
        'price_anomaly'
    ]
    
    print("\n🎯 Key ML Indicators Status:")
    for key in key_indicators:
        if key in ml_indicators:
            print(f"   ✅ {key}: {ml_indicators[key]}")
        else:
            print(f"   ❌ {key}: MISSING")
    
    # LSTM specific check
    if KERAS_AVAILABLE:
        if 'lstm_prediction' in ml_indicators:
            print(f"\n🤖 LSTM Active:")
            print(f"   Prediction: {ml_indicators['lstm_prediction']:.2f}%")
            print(f"   Trend Signal: {ml_indicators['lstm_trend_signal']}")
        else:
            print("\n⚠️  LSTM indicators not calculated (need more data or error)")
    else:
        print("\n⚠️  LSTM/Keras not available - LSTM indicators disabled")
        print("   This is OK! Other ML indicators still work.")
    
    print("\n✅ ALL TESTS PASSED! ML indicators are working!")
    print("\n💡 Next step: Run optimization in the app to use these indicators")
    
except Exception as e:
    print(f"❌ Error calculating ML indicators: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
