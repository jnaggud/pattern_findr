#!/usr/bin/env python3
"""
Simple integration test for new indicators.
"""

import pandas as pd
import numpy as np
import sys

def create_test_data():
    """Create synthetic OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
    n = len(dates)
    
    # Generate synthetic price data
    np.random.seed(42)
    base_price = 100
    price_changes = np.random.normal(0, 0.02, n)
    
    prices = [base_price]
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    close = np.array(prices)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    volume = np.random.randint(10000, 100000, n)
    
    return pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

def test_main_integration():
    """Test main indicators integration."""
    print("🔄 Testing Main Integration...")
    
    try:
        # Test data
        test_data = create_test_data()
        print(f"   - Test data: {len(test_data)} rows")
        
        # Test indicators
        from indicators import get_all_indicators
        enriched_data = get_all_indicators(test_data)
        
        print(f"   - Input columns: {len(test_data.columns)}")
        print(f"   - Output columns: {len(enriched_data.columns)}")
        
        # Check for advanced indicators
        adv_cols = [col for col in enriched_data.columns if col.startswith('adv_')]
        enh_cols = [col for col in enriched_data.columns if col.startswith('enh_')]
        
        print(f"   - Advanced indicators: {len(adv_cols)}")
        print(f"   - Enhanced patterns: {len(enh_cols)}")
        
        if len(adv_cols) >= 5 and len(enh_cols) >= 5:
            print("✅ Integration successful!")
            return True
        else:
            print("❌ Missing indicators")
            return False
            
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return False

def test_pattern_expansion():
    """Test CNN pattern expansion."""
    print("\n🔄 Testing Pattern Expansion...")
    
    try:
        from dl_models import CHART_PATTERNS
        
        print(f"   - Total patterns: {len(CHART_PATTERNS)}")
        
        if len(CHART_PATTERNS) >= 30:
            print("✅ Pattern expansion successful!")
            return True
        else:
            print(f"❌ Expected 30+ patterns, got {len(CHART_PATTERNS)}")
            return False
            
    except Exception as e:
        print(f"❌ Pattern expansion failed: {e}")
        return False

def test_simplified_indicators():
    """Test simplified indicators directly.""" 
    print("\n🔄 Testing Simplified Indicators...")
    
    try:
        from advanced_indicators_simple import calculate_advanced_technical_signals
        
        test_data = create_test_data()
        signals = calculate_advanced_technical_signals(test_data)
        
        print(f"   - Generated {len(signals)} signals")
        
        expected_signals = ['strong_uptrend', 'oversold_reversal', 'volume_confirmation']
        for signal in expected_signals:
            if signal in signals:
                print(f"   ✅ {signal}: {signals[signal]}")
            
        if len(signals) >= 5:
            print("✅ Simplified indicators working!")
            return True
        else:
            print("❌ Insufficient signals")
            return False
            
    except Exception as e:
        print(f"❌ Simplified indicators failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Running Integration Tests\n")
    
    results = []
    results.append(test_main_integration())
    results.append(test_pattern_expansion())
    results.append(test_simplified_indicators())
    
    print("\n" + "="*40)
    print("📊 RESULTS")
    print("="*40)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("🎉 Ready for deployment!")
        sys.exit(0)
    else:
        print(f"❌ {total - passed} TESTS FAILED ({passed}/{total})")
        sys.exit(1)
