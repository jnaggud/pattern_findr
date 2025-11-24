"""
Test Enhancement #1: Signal Optimization
Compare original vs enhanced strategy performance
"""

import pandas as pd
import numpy as np
from optimization import universal_strategy, enhanced_universal_strategy, SIGNAL_OPTIMIZATION_AVAILABLE
from signal_optimization import optimize_signal_parameters
import time

def test_signal_optimization():
    """Test Enhancement #1 signal optimization improvements"""
    
    print("🔍 TESTING ENHANCEMENT #1: SIGNAL OPTIMIZATION")
    print("=" * 60)
    
    if not SIGNAL_OPTIMIZATION_AVAILABLE:
        print("❌ Signal optimization module not available!")
        return
    
    # Load sample data (simulate if needed)
    print("📊 Loading test data...")
    try:
        # Try to load actual data
        data = pd.read_csv('enriched_data.csv')
        print(f"   ✅ Loaded {len(data)} rows of actual data")
    except:
        # Create synthetic test data
        print("   ⚠️  Creating synthetic test data...")
        data = create_synthetic_data()
    
    # Create test parameters based on successful configuration
    test_params = {
        # Basic thresholds (current successful configuration)
        'buy_score_threshold': 1,
        'sell_score_threshold': 9,
        'signal_persistence_days': 2,
        'use_trend_filter': True,
        
        # Trend filter parameters
        'trend_adx_threshold': 25,
        'trend_rsi_oversold': 30,
        'trend_willr_threshold': -80,
        'trend_stoch_threshold': 20,
        'trend_sma_period': 50,
        
        # Sample indicator parameters (subset for testing)
        'use_RSI_14': True,
        'RSI_14_buy': 30,
        'RSI_14_sell': 70,
        
        'use_MACD_12_26_9': True,
        'MACD_12_26_9_buy': -1,
        'MACD_12_26_9_sell': 1,
        
        'use_STOCHk_14_3_3': True,
        'STOCHk_14_3_3_buy': 20,
        'STOCHk_14_3_3_sell': 80,
        
        'use_pattern_bullish_engulfing': True,
        'pattern_bullish_engulfing_buy': True,
        'pattern_bullish_engulfing_sell': False,
        
        # Add more indicators if they exist in data
    }
    
    # Add all available indicators from data
    for col in data.columns:
        if col.startswith(('RSI', 'MACD', 'STOCH', 'pattern_', 'ADX', 'WILLR')):
            if f'use_{col}' not in test_params:
                test_params[f'use_{col}'] = True
                # Set reasonable default thresholds
                if col.startswith('pattern_'):
                    test_params[f'{col}_buy'] = True
                    test_params[f'{col}_sell'] = False
                else:
                    mid_val = data[col].median() if pd.api.types.is_numeric_dtype(data[col]) else 50
                    test_params[f'{col}_buy'] = mid_val * 0.8  # 20% below median
                    test_params[f'{col}_sell'] = mid_val * 1.2  # 20% above median
    
    print(f"📈 Testing with {len([k for k in test_params.keys() if k.startswith('use_') and test_params[k]])} indicators")
    
    # Test 1: Original Strategy
    print("\n🔧 TESTING ORIGINAL STRATEGY...")
    start_time = time.time()
    original_signals = universal_strategy(data, test_params)
    original_time = time.time() - start_time
    
    original_trades = (original_signals != 0).sum()
    original_buys = (original_signals == 1).sum()
    original_sells = (original_signals == -1).sum()
    
    print(f"   📊 Original Results:")
    print(f"      Total trades: {original_trades}")
    print(f"      Buy signals: {original_buys}")
    print(f"      Sell signals: {original_sells}")
    print(f"      Execution time: {original_time:.2f}s")
    
    # Test 2: Enhanced Strategy
    print("\n🚀 TESTING ENHANCED STRATEGY...")
    start_time = time.time()
    enhanced_signals = enhanced_universal_strategy(data, test_params, use_signal_optimization=True)
    enhanced_time = time.time() - start_time
    
    enhanced_trades = (enhanced_signals != 0).sum()
    enhanced_buys = (enhanced_signals == 1).sum()
    enhanced_sells = (enhanced_signals == -1).sum()
    
    print(f"   📊 Enhanced Results:")
    print(f"      Total trades: {enhanced_trades}")
    print(f"      Buy signals: {enhanced_buys}")
    print(f"      Sell signals: {enhanced_sells}")
    print(f"      Execution time: {enhanced_time:.2f}s")
    
    # Test 3: Signal Analysis
    print("\n🔍 SIGNAL OPTIMIZATION ANALYSIS...")
    optimization_results = optimize_signal_parameters(data, test_params, target_trades=40)
    
    print(f"   📈 Analysis Results:")
    print(f"      Current trades: {optimization_results['analysis']['current_trades']}")
    print(f"      Target trades: 40")
    
    if optimization_results['optimization_results']['recommended_thresholds']:
        best_config = optimization_results['optimization_results']['recommended_thresholds'][0]
        print(f"      Best threshold config: {best_config['config']}")
        print(f"      Predicted trades: {best_config['trade_count']}")
        print(f"      Buy/sell ratio: {best_config['buy_sell_ratio']}")
    
    # Comparison Summary
    print("\n📊 ENHANCEMENT #1 COMPARISON SUMMARY")
    print("=" * 50)
    print(f"Trade Count Improvement: {enhanced_trades - original_trades:+d} ({((enhanced_trades/max(1, original_trades) - 1) * 100):+.1f}%)")
    print(f"Buy Signal Improvement: {enhanced_buys - original_buys:+d} ({((enhanced_buys/max(1, original_buys) - 1) * 100):+.1f}%)")
    print(f"Sell Signal Improvement: {enhanced_sells - original_sells:+d} ({((enhanced_sells/max(1, original_sells) - 1) * 100):+.1f}%)")
    print(f"Performance Impact: {enhanced_time - original_time:+.2f}s")
    
    if enhanced_trades > original_trades * 1.1:  # 10% improvement
        print("✅ Enhancement #1: SIGNIFICANT IMPROVEMENT in signal frequency!")
    elif enhanced_trades > original_trades:
        print("✅ Enhancement #1: Moderate improvement in signal frequency")
    else:
        print("⚠️  Enhancement #1: No significant improvement - may need tuning")
    
    return {
        'original': {'trades': original_trades, 'buys': original_buys, 'sells': original_sells, 'time': original_time},
        'enhanced': {'trades': enhanced_trades, 'buys': enhanced_buys, 'sells': enhanced_sells, 'time': enhanced_time},
        'improvement': enhanced_trades - original_trades,
        'optimization_results': optimization_results
    }

def create_synthetic_data():
    """Create synthetic market data for testing"""
    print("   🔧 Creating 250-day synthetic market data...")
    
    dates = pd.date_range('2024-01-01', periods=250, freq='D')
    
    # Create realistic price data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 250)  # Daily returns
    price = 100 * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'date': dates,
        'close': price,
        'open': price * (1 + np.random.normal(0, 0.005, 250)),
        'high': price * (1 + np.abs(np.random.normal(0, 0.01, 250))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.01, 250))),
        'volume': np.random.lognormal(15, 1, 250),
    })
    
    # Add technical indicators
    data['RSI_14'] = 50 + 30 * np.sin(np.arange(250) * 0.1) + np.random.normal(0, 5, 250)
    data['RSI_14'] = data['RSI_14'].clip(0, 100)
    
    data['MACD_12_26_9'] = np.random.normal(0, 2, 250) + np.sin(np.arange(250) * 0.05)
    data['STOCHk_14_3_3'] = 50 + 40 * np.sin(np.arange(250) * 0.08) + np.random.normal(0, 10, 250)
    data['STOCHk_14_3_3'] = data['STOCHk_14_3_3'].clip(0, 100)
    
    data['ADX_14'] = 20 + 10 * np.abs(np.sin(np.arange(250) * 0.03)) + np.random.normal(0, 3, 250)
    data['WILLR_14'] = -50 + 40 * np.sin(np.arange(250) * 0.12) + np.random.normal(0, 10, 250)
    data['WILLR_14'] = data['WILLR_14'].clip(-100, 0)
    
    # Add boolean pattern
    data['pattern_bullish_engulfing'] = np.random.choice([True, False], 250, p=[0.05, 0.95])
    
    return data

if __name__ == "__main__":
    results = test_signal_optimization()
    print("\n🎯 Test completed!")
