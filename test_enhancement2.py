"""
Test Enhancement #2: Position Sizing & Risk Management
Compare standard vs enhanced backtesting with dynamic position sizing
"""

import pandas as pd
import numpy as np
from optimization import universal_strategy, POSITION_SIZING_AVAILABLE
from enhanced_backtester import EnhancedBacktester
from backtester import Backtester  
from position_sizing import PositionSizer, optimize_position_sizing_parameters
import time

def test_position_sizing():
    """Test Enhancement #2 position sizing improvements"""
    
    print("🎯 TESTING ENHANCEMENT #2: POSITION SIZING & RISK MANAGEMENT")
    print("=" * 65)
    
    if not POSITION_SIZING_AVAILABLE:
        print("❌ Position sizing module not available!")
        return
    
    # Load or create test data
    print("📊 Loading test data...")
    try:
        data = pd.read_csv('enriched_data.csv')
        print(f"   ✅ Loaded {len(data)} rows of actual data")
    except:
        print("   ⚠️  Creating synthetic test data...")
        data = create_enhanced_synthetic_data()
    
    # Create test parameters (based on successful configuration)
    test_params = {
        # Basic strategy parameters
        'buy_score_threshold': 2,
        'sell_score_threshold': 6,
        'signal_persistence_days': 2,
        'use_trend_filter': True,
        
        # Trend filter parameters
        'trend_adx_threshold': 25,
        'trend_rsi_oversold': 30,
        'trend_willr_threshold': -80,
        'trend_stoch_threshold': 20,
        'trend_sma_period': 50,
        
        # Sample indicators
        'use_RSI_14': True,
        'RSI_14_buy': 25,
        'RSI_14_sell': 75,
        
        'use_MACD_12_26_9': True,
        'MACD_12_26_9_buy': -1,
        'MACD_12_26_9_sell': 1,
        
        'use_STOCHk_14_3_3': True,
        'STOCHk_14_3_3_buy': 20,
        'STOCHk_14_3_3_sell': 80,
        
        # Add more indicators if available
    }
    
    # Add available indicators from data
    for col in data.columns:
        if any(col.startswith(prefix) for prefix in ['RSI', 'MACD', 'STOCH', 'ADX', 'WILLR', 'pattern_']):
            if f'use_{col}' not in test_params:
                test_params[f'use_{col}'] = True
                if col.startswith('pattern_'):
                    test_params[f'{col}_buy'] = True
                    test_params[f'{col}_sell'] = False
                else:
                    mid_val = data[col].median() if pd.api.types.is_numeric_dtype(data[col]) else 50
                    test_params[f'{col}_buy'] = mid_val * 0.9
                    test_params[f'{col}_sell'] = mid_val * 1.1
    
    print(f"📈 Testing with {len([k for k in test_params.keys() if k.startswith('use_') and test_params[k]])} indicators")
    
    # Get baseline signals
    signals = universal_strategy(data, test_params)
    signal_count = (signals != 0).sum()
    print(f"   📊 Generated {signal_count} trading signals for testing")
    
    if signal_count == 0:
        print("❌ No trading signals generated - cannot test position sizing")
        return
    
    # Test 1: Standard Backtester (100% position sizing)
    print("\n📊 TESTING STANDARD BACKTESTER (100% Position Sizing)...")
    start_time = time.time()
    
    standard_backtester = Backtester(
        data=data,
        strategy_name="Standard Test",
        strategy_func=universal_strategy,
        params=test_params,
        starting_capital=100000
    )
    
    standard_backtester.run()
    _, standard_summary = standard_backtester.get_results()
    standard_time = time.time() - start_time
    
    print(f"   📊 Standard Results:")
    print(f"      Total Return: {standard_summary.get('total_return_pct', 0):.2f}%")
    print(f"      Total Trades: {standard_summary.get('total_trades', 0)}")
    print(f"      Win Rate: {standard_summary.get('win_rate', 0):.1f}%")
    print(f"      Max Drawdown: {standard_summary.get('max_drawdown_pct', 0):.2f}%")
    print(f"      Execution Time: {standard_time:.2f}s")
    
    # Test 2: Enhanced Backtester (Dynamic Position Sizing - Conservative)
    print("\n🚀 TESTING ENHANCED BACKTESTER (15% Max Position Size)...")
    start_time = time.time()
    
    enhanced_params = test_params.copy()
    enhanced_params.update({
        'enable_position_sizing': True,
        'max_position_pct': 0.15,
        'kelly_optimization': True,
        'volatility_adjustment': True,
        'max_total_exposure': 0.60
    })
    
    enhanced_backtester = EnhancedBacktester(
        data=data,
        strategy_name="Enhanced Conservative Test",
        strategy_func=universal_strategy,
        params=enhanced_params,
        starting_capital=100000,
        enable_position_sizing=True,
        max_position_pct=0.15
    )
    
    enhanced_backtester.run()
    _, enhanced_summary = enhanced_backtester.get_results()
    enhanced_time = time.time() - start_time
    
    print(f"   📊 Enhanced Conservative Results:")
    print(f"      Total Return: {enhanced_summary.get('total_return_pct', 0):.2f}%")
    print(f"      Total Trades: {enhanced_summary.get('total_trades', 0)}")
    print(f"      Win Rate: {enhanced_summary.get('win_rate', 0):.1f}%")
    print(f"      Max Drawdown: {enhanced_summary.get('max_drawdown_pct', 0):.2f}%")
    print(f"      Avg Position Size: {enhanced_summary.get('avg_position_size_pct', 0):.1f}%")
    print(f"      Risk Level: {enhanced_summary.get('risk_level', 'Unknown')}")
    print(f"      Execution Time: {enhanced_time:.2f}s")
    
    # Test 3: Enhanced Backtester (Dynamic Position Sizing - Aggressive)
    print("\n🔥 TESTING ENHANCED BACKTESTER (25% Max Position Size)...")
    start_time = time.time()
    
    aggressive_params = test_params.copy()
    aggressive_params.update({
        'enable_position_sizing': True,
        'max_position_pct': 0.25,
        'kelly_optimization': True,
        'volatility_adjustment': True,
        'max_total_exposure': 0.75
    })
    
    aggressive_backtester = EnhancedBacktester(
        data=data,
        strategy_name="Enhanced Aggressive Test",
        strategy_func=universal_strategy,
        params=aggressive_params,
        starting_capital=100000,
        enable_position_sizing=True,
        max_position_pct=0.25
    )
    
    aggressive_backtester.run()
    _, aggressive_summary = aggressive_backtester.get_results()
    aggressive_time = time.time() - start_time
    
    print(f"   📊 Enhanced Aggressive Results:")
    print(f"      Total Return: {aggressive_summary.get('total_return_pct', 0):.2f}%")
    print(f"      Total Trades: {aggressive_summary.get('total_trades', 0)}")
    print(f"      Win Rate: {aggressive_summary.get('win_rate', 0):.1f}%")
    print(f"      Max Drawdown: {aggressive_summary.get('max_drawdown_pct', 0):.2f}%")
    print(f"      Avg Position Size: {aggressive_summary.get('avg_position_size_pct', 0):.1f}%")
    print(f"      Risk Level: {aggressive_summary.get('risk_level', 'Unknown')}")
    print(f"      Execution Time: {aggressive_time:.2f}s")
    
    # Test 4: Position Sizing Optimization
    print("\n🔬 TESTING POSITION SIZING OPTIMIZATION...")
    optimization_results = optimize_position_sizing_parameters(data, signals, test_params)
    
    print(f"   📊 Optimization Results:")
    print(f"      Optimal Config: {optimization_results.get('optimal_config', 'Unknown')}")
    print(f"      Recommended Max Position: {optimization_results.get('recommended_max_position', 0.15):.1%}")
    
    # Comparison Summary
    print("\n📊 ENHANCEMENT #2 COMPARISON SUMMARY")
    print("=" * 55)
    
    standard_return = standard_summary.get('total_return_pct', 0)
    enhanced_return = enhanced_summary.get('total_return_pct', 0)
    aggressive_return = aggressive_summary.get('total_return_pct', 0)
    
    print(f"Return Comparison:")
    print(f"   Standard (100% positions): {standard_return:.2f}%")
    print(f"   Enhanced Conservative (15%): {enhanced_return:.2f}% ({enhanced_return - standard_return:+.2f}%)")
    print(f"   Enhanced Aggressive (25%): {aggressive_return:.2f}% ({aggressive_return - standard_return:+.2f}%)")
    
    standard_drawdown = standard_summary.get('max_drawdown_pct', 0)
    enhanced_drawdown = enhanced_summary.get('max_drawdown_pct', 0)
    aggressive_drawdown = aggressive_summary.get('max_drawdown_pct', 0)
    
    print(f"\nRisk Comparison:")
    print(f"   Standard Max Drawdown: {standard_drawdown:.2f}%")
    print(f"   Enhanced Conservative: {enhanced_drawdown:.2f}% ({enhanced_drawdown - standard_drawdown:+.2f}%)")
    print(f"   Enhanced Aggressive: {aggressive_drawdown:.2f}% ({aggressive_drawdown - standard_drawdown:+.2f}%)")
    
    # Risk-Adjusted Performance
    def calculate_sharpe_ratio(returns, drawdown):
        if drawdown == 0:
            return float('inf') if returns > 0 else 0
        return returns / drawdown
    
    standard_sharpe = calculate_sharpe_ratio(standard_return, standard_drawdown)
    enhanced_sharpe = calculate_sharpe_ratio(enhanced_return, enhanced_drawdown)
    aggressive_sharpe = calculate_sharpe_ratio(aggressive_return, aggressive_drawdown)
    
    print(f"\nRisk-Adjusted Performance (Return/Drawdown Ratio):")
    print(f"   Standard: {standard_sharpe:.2f}")
    print(f"   Enhanced Conservative: {enhanced_sharpe:.2f}")
    print(f"   Enhanced Aggressive: {aggressive_sharpe:.2f}")
    
    # Determine best strategy
    if enhanced_sharpe > standard_sharpe and enhanced_return > standard_return:
        print("✅ Enhancement #2: SIGNIFICANT IMPROVEMENT in risk-adjusted returns!")
    elif enhanced_return > standard_return * 1.05:  # 5% improvement
        print("✅ Enhancement #2: Notable improvement in returns")
    else:
        print("⚠️  Enhancement #2: Mixed results - may need parameter tuning")
    
    return {
        'standard': standard_summary,
        'enhanced_conservative': enhanced_summary,
        'enhanced_aggressive': aggressive_summary,
        'optimization_results': optimization_results,
        'best_strategy': 'enhanced' if enhanced_sharpe > standard_sharpe else 'standard'
    }

def create_enhanced_synthetic_data():
    """Create more sophisticated synthetic data for position sizing tests"""
    print("   🔧 Creating 500-day synthetic market data with volatility regimes...")
    
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    
    np.random.seed(42)
    
    # Create realistic market with different volatility regimes
    base_vol = 0.015
    vol_regimes = []
    
    # Create volatility regimes: low (0.01), medium (0.02), high (0.04)
    for i in range(500):
        if i < 150:  # Low vol period
            vol_regimes.append(0.01)
        elif i < 300:  # Medium vol period  
            vol_regimes.append(0.02)
        elif i < 400:  # High vol period
            vol_regimes.append(0.04)
        else:  # Return to medium vol
            vol_regimes.append(0.02)
    
    returns = []
    for vol in vol_regimes:
        returns.append(np.random.normal(0.001, vol))
    
    price = 100 * np.cumprod(1 + np.array(returns))
    
    data = pd.DataFrame({
        'date': dates,
        'close': price,
        'open': price * (1 + np.random.normal(0, 0.005, 500)),
        'high': price * (1 + np.abs(np.random.normal(0, 0.015, 500))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.015, 500))),
        'volume': np.random.lognormal(15, 1, 500),
    })
    
    # Add realistic technical indicators
    data['RSI_14'] = 50 + 30 * np.sin(np.arange(500) * 0.1) + np.random.normal(0, 8, 500)
    data['RSI_14'] = data['RSI_14'].clip(0, 100)
    
    data['MACD_12_26_9'] = np.random.normal(0, 3, 500) + np.sin(np.arange(500) * 0.05)
    data['STOCHk_14_3_3'] = 50 + 40 * np.sin(np.arange(500) * 0.08) + np.random.normal(0, 12, 500)
    data['STOCHk_14_3_3'] = data['STOCHk_14_3_3'].clip(0, 100)
    
    data['ADX_14'] = 20 + 15 * np.abs(np.sin(np.arange(500) * 0.03)) + np.random.normal(0, 4, 500)
    data['WILLR_14'] = -50 + 40 * np.sin(np.arange(500) * 0.12) + np.random.normal(0, 12, 500)
    data['WILLR_14'] = data['WILLR_14'].clip(-100, 0)
    
    # Add patterns with varying frequency based on volatility
    pattern_prob = []
    for vol in vol_regimes:
        if vol > 0.03:  # High volatility = more patterns
            pattern_prob.append(0.08)
        elif vol < 0.015:  # Low volatility = fewer patterns
            pattern_prob.append(0.02)
        else:
            pattern_prob.append(0.04)
    
    data['pattern_bullish_engulfing'] = np.random.binomial(1, pattern_prob, 500).astype(bool)
    
    return data

if __name__ == "__main__":
    results = test_position_sizing()
    print("\n🎯 Enhancement #2 test completed!")
