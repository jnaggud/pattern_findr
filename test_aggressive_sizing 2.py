"""
Test Aggressive Position Sizing for Enhancement #2
Compare conservative vs aggressive position sizing settings
"""

import pandas as pd
import numpy as np
from optimization import universal_strategy
from enhanced_backtester import EnhancedBacktester
from backtester import Backtester
import time

def test_aggressive_position_sizing():
    """Test aggressive position sizing to get closer to 180% standard returns"""
    
    print("🔥 TESTING AGGRESSIVE POSITION SIZING")
    print("=" * 50)
    
    # Create enhanced synthetic data for testing
    data = create_test_data()
    
    # Base test parameters
    test_params = {
        'buy_score_threshold': 2,
        'sell_score_threshold': 6, 
        'signal_persistence_days': 2,
        'use_trend_filter': True,
        'trend_adx_threshold': 25,
        'trend_rsi_oversold': 30,
        'trend_willr_threshold': -80,
        
        # Add indicators
        'use_RSI_14': True,
        'RSI_14_buy': 25,
        'RSI_14_sell': 75,
        
        'use_MACD_12_26_9': True,
        'MACD_12_26_9_buy': -1,
        'MACD_12_26_9_sell': 1,
        
        'use_STOCHk_14_3_3': True,
        'STOCHk_14_3_3_buy': 20,
        'STOCHk_14_3_3_sell': 80,
        
        'use_pattern_bullish_engulfing': True,
        'pattern_bullish_engulfing_buy': True,
        'pattern_bullish_engulfing_sell': False,
    }
    
    print(f"📊 Testing with {len([k for k in test_params.keys() if k.startswith('use_')])} indicators")
    
    # Test 1: Standard Backtester (100% baseline)
    print("\n📊 BASELINE: Standard Backtester (100% Position Sizing)...")
    standard_backtester = Backtester(
        data=data,
        strategy_name="Standard Baseline",
        strategy_func=universal_strategy,
        params=test_params,
        starting_capital=100000
    )
    standard_backtester.run()
    _, standard_summary = standard_backtester.get_results()
    
    print(f"   📈 Standard Results: {standard_summary.get('total_return_pct', 0):.2f}% returns")
    print(f"   📉 Max Drawdown: {standard_summary.get('max_drawdown_pct', 0):.2f}%")
    
    # Test 2: Conservative Enhanced (25% max position)
    print("\n🛡️  CONSERVATIVE: Enhanced Backtester (25% Max Position)...")
    conservative_params = test_params.copy()
    conservative_params.update({
        'enable_position_sizing': True,
        'max_position_pct': 0.25,
        'volatility_adjustment': True,
        'max_total_exposure': 0.70,
        'aggressive_vol_scaling': False,
        'confidence_risk_multiplier': 1.2
    })
    
    conservative_backtester = EnhancedBacktester(
        data=data,
        strategy_name="Conservative Enhanced",
        strategy_func=universal_strategy,
        params=conservative_params,
        starting_capital=100000,
        enable_position_sizing=True,
        max_position_pct=0.25
    )
    conservative_backtester.run()
    _, conservative_summary = conservative_backtester.get_results()
    
    print(f"   📈 Conservative Results: {conservative_summary.get('total_return_pct', 0):.2f}% returns")
    print(f"   📉 Max Drawdown: {conservative_summary.get('max_drawdown_pct', 0):.2f}%")
    print(f"   📊 Avg Position Size: {conservative_summary.get('avg_position_size_pct', 0):.1f}%")
    
    # Test 3: Aggressive Enhanced (50% max position)
    print("\n🚀 AGGRESSIVE: Enhanced Backtester (50% Max Position)...")
    aggressive_params = test_params.copy()
    aggressive_params.update({
        'enable_position_sizing': True,
        'max_position_pct': 0.50,
        'volatility_adjustment': True,
        'max_total_exposure': 0.85,
        'aggressive_vol_scaling': True,
        'confidence_risk_multiplier': 1.8
    })
    
    aggressive_backtester = EnhancedBacktester(
        data=data,
        strategy_name="Aggressive Enhanced",
        strategy_func=universal_strategy,
        params=aggressive_params,
        starting_capital=100000,
        enable_position_sizing=True,
        max_position_pct=0.50
    )
    aggressive_backtester.run()
    _, aggressive_summary = aggressive_backtester.get_results()
    
    print(f"   📈 Aggressive Results: {aggressive_summary.get('total_return_pct', 0):.2f}% returns")
    print(f"   📉 Max Drawdown: {aggressive_summary.get('max_drawdown_pct', 0):.2f}%")
    print(f"   📊 Avg Position Size: {aggressive_summary.get('avg_position_size_pct', 0):.1f}%")
    
    # Test 4: ULTRA-Aggressive (75% max position)
    print("\n💥 ULTRA-AGGRESSIVE: Enhanced Backtester (75% Max Position)...")
    ultra_params = test_params.copy()
    ultra_params.update({
        'enable_position_sizing': True,
        'max_position_pct': 0.75,
        'volatility_adjustment': True,
        'max_total_exposure': 0.95,
        'aggressive_vol_scaling': True,
        'confidence_risk_multiplier': 2.0
    })
    
    ultra_backtester = EnhancedBacktester(
        data=data,
        strategy_name="Ultra-Aggressive Enhanced",
        strategy_func=universal_strategy,
        params=ultra_params,
        starting_capital=100000,
        enable_position_sizing=True,
        max_position_pct=0.75
    )
    ultra_backtester.run()
    _, ultra_summary = ultra_backtester.get_results()
    
    print(f"   📈 Ultra Results: {ultra_summary.get('total_return_pct', 0):.2f}% returns")
    print(f"   📉 Max Drawdown: {ultra_summary.get('max_drawdown_pct', 0):.2f}%")
    print(f"   📊 Avg Position Size: {ultra_summary.get('avg_position_size_pct', 0):.1f}%")
    
    # Comparison Analysis
    print("\n📊 AGGRESSIVE POSITION SIZING ANALYSIS")
    print("=" * 48)
    
    standard_return = standard_summary.get('total_return_pct', 0)
    conservative_return = conservative_summary.get('total_return_pct', 0)
    aggressive_return = aggressive_summary.get('total_return_pct', 0)
    ultra_return = ultra_summary.get('total_return_pct', 0)
    
    print(f"Return Progression:")
    print(f"   Standard (100%): {standard_return:.2f}%")
    print(f"   Conservative (25%): {conservative_return:.2f}% ({conservative_return/standard_return*100:.1f}% of standard)")
    print(f"   Aggressive (50%): {aggressive_return:.2f}% ({aggressive_return/standard_return*100:.1f}% of standard)")
    print(f"   Ultra-Aggressive (75%): {ultra_return:.2f}% ({ultra_return/standard_return*100:.1f}% of standard)")
    
    standard_dd = standard_summary.get('max_drawdown_pct', 0)
    conservative_dd = conservative_summary.get('max_drawdown_pct', 0)
    aggressive_dd = aggressive_summary.get('max_drawdown_pct', 0)
    ultra_dd = ultra_summary.get('max_drawdown_pct', 0)
    
    print(f"\nDrawdown Progression:")
    print(f"   Standard: {standard_dd:.2f}%")
    print(f"   Conservative: {conservative_dd:.2f}%")
    print(f"   Aggressive: {aggressive_dd:.2f}%")
    print(f"   Ultra-Aggressive: {ultra_dd:.2f}%")
    
    # Risk-adjusted returns
    def calc_sharpe(ret, dd):
        return ret / max(dd, 0.1) if dd > 0 else ret * 10
    
    print(f"\nRisk-Adjusted Performance (Return/Drawdown):")
    print(f"   Standard: {calc_sharpe(standard_return, standard_dd):.2f}")
    print(f"   Conservative: {calc_sharpe(conservative_return, conservative_dd):.2f}")
    print(f"   Aggressive: {calc_sharpe(aggressive_return, aggressive_dd):.2f}")
    print(f"   Ultra-Aggressive: {calc_sharpe(ultra_return, ultra_dd):.2f}")
    
    # Find best balance
    best_balance = max(
        [('Conservative', calc_sharpe(conservative_return, conservative_dd)),
         ('Aggressive', calc_sharpe(aggressive_return, aggressive_dd)),
         ('Ultra-Aggressive', calc_sharpe(ultra_return, ultra_dd))],
        key=lambda x: x[1]
    )
    
    print(f"\n🏆 Best Risk-Adjusted Strategy: {best_balance[0]} (Ratio: {best_balance[1]:.2f})")
    
    # Success assessment
    if ultra_return > standard_return * 0.7:  # 70%+ of standard returns
        print("✅ AGGRESSIVE SIZING SUCCESS: Achieved 70%+ of standard returns with risk management!")
    elif aggressive_return > standard_return * 0.5:  # 50%+ of standard returns
        print("✅ AGGRESSIVE SIZING GOOD: Achieved 50%+ of standard returns with risk management")
    else:
        print("⚠️  AGGRESSIVE SIZING: May need further tuning for higher returns")
    
    return {
        'standard': standard_summary,
        'conservative': conservative_summary,
        'aggressive': aggressive_summary,
        'ultra': ultra_summary,
        'best_strategy': best_balance[0].lower()
    }

def create_test_data():
    """Create synthetic test data with various market conditions"""
    print("   🔧 Creating 400-day test data with bull/bear cycles...")
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=400, freq='D')
    
    # Create market cycles: bull (high returns), bear (negative), sideways (flat)
    returns = []
    for i in range(400):
        if i < 120:  # Bull market
            returns.append(np.random.normal(0.002, 0.015))  # +0.2% daily avg
        elif i < 200:  # Bear market
            returns.append(np.random.normal(-0.001, 0.025))  # -0.1% daily avg
        elif i < 280:  # Sideways
            returns.append(np.random.normal(0.0005, 0.010))  # +0.05% daily avg
        else:  # Recovery bull
            returns.append(np.random.normal(0.0015, 0.020))  # +0.15% daily avg
    
    price = 100 * np.cumprod(1 + np.array(returns))
    
    data = pd.DataFrame({
        'date': dates,
        'close': price,
        'open': price * (1 + np.random.normal(0, 0.003, 400)),
        'high': price * (1 + np.abs(np.random.normal(0, 0.01, 400))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.01, 400))),
        'volume': np.random.lognormal(15, 1, 400),
    })
    
    # Add realistic technical indicators
    data['RSI_14'] = 50 + 30 * np.sin(np.arange(400) * 0.1) + np.random.normal(0, 8, 400)
    data['RSI_14'] = data['RSI_14'].clip(0, 100)
    
    data['MACD_12_26_9'] = np.random.normal(0, 3, 400) + np.sin(np.arange(400) * 0.05) 
    data['STOCHk_14_3_3'] = 50 + 40 * np.sin(np.arange(400) * 0.08) + np.random.normal(0, 12, 400)
    data['STOCHk_14_3_3'] = data['STOCHk_14_3_3'].clip(0, 100)
    
    # Pattern with market-dependent frequency
    pattern_prob = []
    for i in range(400):
        if i < 120:  # Bull - fewer patterns
            pattern_prob.append(0.03)
        elif i < 200:  # Bear - more patterns (volatility)
            pattern_prob.append(0.08)
        elif i < 280:  # Sideways - normal
            pattern_prob.append(0.04)
        else:  # Recovery - more patterns
            pattern_prob.append(0.06)
    
    data['pattern_bullish_engulfing'] = np.random.binomial(1, pattern_prob, 400).astype(bool)
    
    return data

if __name__ == "__main__":
    results = test_aggressive_position_sizing()
    print("\n🔥 Aggressive position sizing test completed!")
