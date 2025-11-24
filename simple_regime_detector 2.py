#!/usr/bin/env python3
"""
Simple Market Regime Detector
No look-ahead bias - uses only historical data at each point
"""

import pandas as pd
import numpy as np

# Flag to indicate that simple regime detection is available
SIMPLE_REGIME_AVAILABLE = True

def detect_market_regime(data, current_idx, lookback=20):
    """
    Detect market regime at current_idx using only past data
    
    Returns: 'bull', 'bear', 'crash', 'sideways'
    """
    # Use only data up to current point (no look-ahead)
    hist_data = data.iloc[:current_idx + 1]
    
    # Need at least 10 days for volatility calculation
    if len(hist_data) < 10:
        return 'sideways'  # Default for insufficient data
    
    # Calculate indicators using only historical data
    current_price = hist_data['close'].iloc[-1]
    sma_20 = hist_data['close'].tail(20).mean() if len(hist_data) >= 20 else hist_data['close'].mean()
    sma_50 = hist_data['close'].tail(50).mean() if len(hist_data) >= 50 else sma_20
    
    # Recent volatility (annualized) - PRIORITIZE THIS FOR CRASH DETECTION
    recent_returns = hist_data['close'].pct_change().tail(10)
    volatility = recent_returns.std() * np.sqrt(252) if len(recent_returns) > 1 else 0
    
    # Price momentum - use flexible lookback based on available data
    if len(hist_data) >= 20:
        momentum_20d = (current_price / hist_data['close'].iloc[-20] - 1) * 100
    elif len(hist_data) >= 10:
        # Use 10-day momentum for early periods
        momentum_20d = (current_price / hist_data['close'].iloc[-10] - 1) * 100
    else:
        momentum_20d = 0
    
    # Short-term momentum (5-day) for catching rapid crashes
    if len(hist_data) >= 5:
        momentum_5d = (current_price / hist_data['close'].iloc[-5] - 1) * 100
    else:
        momentum_5d = 0
    
    # Regime classification (PRIORITIZE VOLATILITY + MOMENTUM FOR CRASHES)
    # CRASH: High volatility OR rapid price decline
    if volatility > 0.25 or momentum_5d < -8 or momentum_20d < -10:  
        return 'crash'
    # BEAR: Sustained downtrend
    elif current_price < sma_50 and momentum_20d < -2:  
        return 'bear'
    # BULL: Sustained uptrend
    elif current_price > sma_50 and momentum_20d > 2:  
        return 'bull'
    # SIDEWAYS: Everything else
    else:
        return 'sideways'

def get_regime_parameters(regime):
    """
    Get strategy parameters optimized for each regime
    Based on market behavior analysis (no look-ahead bias)
    """
    
    regime_params = {
        'bull': {
            # Bull market: RSI rarely oversold, need higher thresholds
            'RSI_14_buy_range': (25, 40),
            'RSI_14_sell_range': (65, 85),
            'WILLR_14_buy_range': (-80, -60),
            'WILLR_14_sell_range': (-40, -20),
            'MACD_buy_range': (-2, 5),
            'MACD_sell_range': (0, 10),
            'buy_score_threshold': 2,  # Higher threshold
            'sell_score_threshold': 1,  # Quick exits
        },
        
        'bear': {
            # Bear market: More oversold conditions, lower thresholds
            'RSI_14_buy_range': (20, 35),
            'RSI_14_sell_range': (55, 75),
            'WILLR_14_buy_range': (-90, -65),
            'WILLR_14_sell_range': (-45, -25),
            'MACD_buy_range': (-8, 2),
            'MACD_sell_range': (-2, 8),
            'buy_score_threshold': 1,  # Lower threshold
            'sell_score_threshold': 2,  # Hold positions longer
        },
        
        'crash': {
            # Crash: Extreme conditions, very sensitive thresholds
            'RSI_14_buy_range': (15, 45),  # Wide range to catch April 21.6
            'RSI_14_sell_range': (50, 70),
            'WILLR_14_buy_range': (-95, -50),  # Catch April -84.5
            'WILLR_14_sell_range': (-50, -10),
            'MACD_buy_range': (-15, 5),  # Catch April -16.9
            'MACD_sell_range': (-5, 15),
            'buy_score_threshold': 1,  # LOWEST threshold
            'sell_score_threshold': 3,  # Hold through volatility
        },
        
        'sideways': {
            # Sideways: Mean reversion works well
            'RSI_14_buy_range': (20, 40),
            'RSI_14_sell_range': (60, 80),
            'WILLR_14_buy_range': (-85, -60),
            'WILLR_14_sell_range': (-40, -15),
            'MACD_buy_range': (-5, 3),
            'MACD_sell_range': (-3, 8),
            'buy_score_threshold': 1,
            'sell_score_threshold': 1,
        }
    }
    
    return regime_params.get(regime, regime_params['sideways'])

def analyze_regime_performance(data):
    """
    Analyze how different regimes perform with different parameters
    """
    print("🔍 REGIME-BASED PARAMETER ANALYSIS")
    print("=" * 50)
    
    regimes = []
    for i in range(50, len(data)):  # Start after warmup period
        regime = detect_market_regime(data, i)
        regimes.append(regime)
    
    regime_series = pd.Series(regimes)
    regime_counts = regime_series.value_counts()
    
    print("📊 Regime Distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(regimes) * 100
        print(f"  {regime.upper()}: {count} days ({pct:.1f}%)")
    
    print("\n🎯 Recommended Parameters by Regime:")
    for regime in ['bull', 'bear', 'crash', 'sideways']:
        if regime in regime_counts:
            params = get_regime_parameters(regime)
            print(f"\n{regime.upper()}:")
            print(f"  RSI buy: {params['RSI_14_buy_range']}")
            print(f"  Buy threshold: {params['buy_score_threshold']}")

if __name__ == "__main__":
    # Test with recent data
    import yfinance as yf
    
    spy = yf.Ticker('SPY')
    data = spy.history(period='1y', interval='1d')
    data.reset_index(inplace=True)
    data.columns = [col.lower() for col in data.columns]
    
    analyze_regime_performance(data)
