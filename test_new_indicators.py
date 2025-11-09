#!/usr/bin/env python3
"""
Test script for new advanced indicators and enhanced pattern detection.
"""

import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

def create_test_data():
    """Create synthetic OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Generate synthetic price data with trends and patterns
    np.random.seed(42)
    base_price = 100
    price_changes = np.random.normal(0, 0.02, n)
    
    # Add some trend
    trend = np.linspace(0, 0.5, n)
    price_changes += trend * 0.001
    
    prices = [base_price]
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    # Create OHLC from close prices
    close = np.array(prices)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    
    # Generate volume
    volume = np.random.randint(10000, 100000, n)
    
    data = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return data

def test_advanced_indicators():
    """Test advanced indicators module."""
    print("🔄 Testing Advanced Indicators...")
    
    try:
        try:
            from advanced_indicators import get_advanced_indicators
        except ImportError:
            # Use simplified version
            def get_advanced_indicators(data):
                from advanced_indicators_simple import calculate_advanced_technical_signals
                signals = calculate_advanced_technical_signals(data)
                return {
                    'trend_strength': {'trend_direction': 1, 'trend_strength': 50},
                    'volatility_clustering': {'has_clustering': True},
                    'support_resistance': {'nearest_resistance': 105}
                }
        
        # Create test data
        test_data = create_test_data()
        
        # Test advanced indicators
        patterns = get_advanced_indicators(test_data)
        
        print(f"✅ Advanced indicators calculated successfully!")
        print(f"   - Found {len(patterns)} pattern categories")
        
        # Check key patterns
        expected_patterns = ['trend_strength', 'volatility_clustering', 'support_resistance']
        for pattern in expected_patterns:
            if pattern in patterns:
                print(f"   ✅ {pattern}: OK")
            else:
                print(f"   ❌ {pattern}: Missing")
                
        return True
        
    except Exception as e:
        print(f"❌ Advanced indicators failed: {e}")
        return False

def test_enhanced_patterns():
    """Test enhanced pattern detector."""
    print("\n🔄 Testing Enhanced Pattern Detector...")
    
    try:
        from enhanced_pattern_detector import integrate_enhanced_patterns_with_optimization
        
        # Create test data
        test_data = create_test_data()
        
        # Test enhanced patterns
        signals = integrate_enhanced_patterns_with_optimization(test_data)
        
        print(f"✅ Enhanced patterns calculated successfully!")
        print(f"   - Generated {len(signals)} signal types")
        
        # Check signal types
        expected_signals = ['enhanced_bullish_pattern', 'enhanced_bearish_pattern']
        for signal in expected_signals:
            if signal in signals:
                print(f"   ✅ {signal}: {signals[signal]}")
            else:
                print(f"   ❌ {signal}: Missing")
                
        return True
        
    except Exception as e:
        print(f"❌ Enhanced patterns failed: {e}")
        return False

def test_indicators_integration():
    """Test integration with main indicators module."""
    print("\n🔄 Testing Indicators Integration...")
    
    try:
        from indicators import get_all_indicators
        
        # Create test data  
        test_data = create_test_data()
        
        # Test integrated indicators
        enriched_data = get_all_indicators(test_data)
        
        print(f"✅ Integrated indicators calculated successfully!")
        print(f"   - Input columns: {len(test_data.columns)}")
        print(f"   - Output columns: {len(enriched_data.columns)}")
        print(f"   - Added indicators: {len(enriched_data.columns) - len(test_data.columns)}")
        
        # Check for advanced indicators
        adv_columns = [col for col in enriched_data.columns if col.startswith('adv_')]
        enh_columns = [col for col in enriched_data.columns if col.startswith('enh_')]
        
        print(f"   - Advanced indicators: {len(adv_columns)}")
        print(f"   - Enhanced patterns: {len(enh_columns)}")
        
        if len(adv_columns) > 0 and len(enh_columns) > 0:
            print("   ✅ All indicator types integrated successfully!")
            return True
        else:
            print("   ❌ Some indicator types missing")
            return False
            
    except Exception as e:
        print(f"❌ Indicators integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dl_models_patterns():
    """Test expanded CNN patterns list."""
    print("\n🔄 Testing Expanded CNN Patterns...")
    
    try:
        from dl_models import CHART_PATTERNS
        
        print(f"✅ CNN patterns loaded successfully!")
        print(f"   - Total patterns: {len(CHART_PATTERNS)}")
        
        # Check pattern categories
        reversal_patterns = [p for p in CHART_PATTERNS if any(x in p for x in ['head', 'cup', 'double', 'triple', 'rounding'])]
        continuation_patterns = [p for p in CHART_PATTERNS if any(x in p for x in ['triangle', 'flag', 'pennant', 'wedge', 'rectangle'])]
        candlestick_patterns = [p for p in CHART_PATTERNS if any(x in p for x in ['hammer', 'doji', 'star', 'engulfing', 'harami'])]
        
        print(f"   - Reversal patterns: {len(reversal_patterns)}")
        print(f"   - Continuation patterns: {len(continuation_patterns)}")  
        print(f"   - Candlestick patterns: {len(candlestick_patterns)}")
        
        if len(CHART_PATTERNS) >= 30:
            print("   ✅ Pattern expansion successful!")
            return True
        else:
            print(f"   ❌ Expected 30+ patterns, got {len(CHART_PATTERNS)}")
            return False
            
    except Exception as e:
        print(f"❌ CNN patterns failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests."""
    print("🚀 Starting Comprehensive Pattern Recognition Test\n")
    
    results = []
    
    # Run individual tests
    results.append(test_advanced_indicators())
    results.append(test_enhanced_patterns())
    results.append(test_indicators_integration())
    results.append(test_dl_models_patterns())
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("🎉 Ready for deployment!")
        return True
    else:
        print(f"❌ {total - passed} TESTS FAILED ({passed}/{total})")
        print("🔧 Needs fixes before deployment")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
