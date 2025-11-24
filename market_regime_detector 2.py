#!/usr/bin/env python3
"""
Enhancement #3: Market Regime Detection
Dynamically adjust strategy thresholds based on market conditions
"""

import pandas as pd
import numpy as np

class MarketRegimeDetector:
    """
    Detects market regimes and adjusts strategy parameters accordingly
    """
    
    def __init__(self, lookback_period=20):
        self.lookback_period = lookback_period
        
    def detect_regime(self, data, current_idx):
        """
        Detect current market regime based on recent price action and volatility
        
        Returns:
            regime: 'bull', 'bear', 'crash', 'sideways'
            confidence: 0.0 to 1.0
        """
        if current_idx < self.lookback_period:
            return 'sideways', 0.5  # Default for insufficient data
        
        # Get recent data
        recent_data = data.iloc[current_idx - self.lookback_period:current_idx + 1]
        
        # Calculate key metrics
        price_change = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1) * 100
        volatility = recent_data['close'].pct_change().std() * np.sqrt(252) * 100  # Annualized
        recent_volatility = recent_data['close'].pct_change().tail(5).std() * np.sqrt(252) * 100
        
        # RSI for momentum
        rsi = self._calculate_rsi(recent_data['close'])
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # Moving average trend
        sma_short = recent_data['close'].tail(5).mean()
        sma_long = recent_data['close'].tail(self.lookback_period).mean()
        trend_direction = 1 if sma_short > sma_long else -1
        
        # Regime detection logic
        regime, confidence = self._classify_regime(
            price_change, volatility, recent_volatility, current_rsi, trend_direction
        )
        
        return regime, confidence
    
    def _classify_regime(self, price_change, volatility, recent_volatility, rsi, trend_direction):
        """Classify market regime based on metrics"""
        
        # CRASH: Extreme negative movement with high volatility
        if price_change < -15 and recent_volatility > 40:
            return 'crash', 0.9
        
        if price_change < -10 and volatility > 30:
            return 'crash', 0.8
            
        if price_change < -8 and recent_volatility > 35:
            return 'crash', 0.7
        
        # BEAR: Sustained downtrend
        if price_change < -5 and trend_direction < 0 and rsi < 40:
            return 'bear', 0.8
            
        if price_change < -3 and trend_direction < 0 and volatility > 25:
            return 'bear', 0.7
        
        # BULL: Sustained uptrend
        if price_change > 8 and trend_direction > 0 and rsi > 60:
            return 'bull', 0.8
            
        if price_change > 5 and trend_direction > 0 and volatility < 20:
            return 'bull', 0.7
        
        # SIDEWAYS: Low volatility, small price changes
        if abs(price_change) < 3 and volatility < 15:
            return 'sideways', 0.7
        
        # Default to current trend direction
        if trend_direction > 0:
            return 'bull', 0.5
        else:
            return 'bear', 0.5
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def get_regime_parameters(self, regime, confidence):
        """
        Get strategy parameters optimized for the detected market regime
        
        Based on April 2025 analysis, we need different thresholds for different conditions
        """
        
        base_params = {
            'bull': {
                'buy_score_threshold': 3,           # Higher threshold in bull markets
                'sell_score_threshold': 2,          # Quick exits
                'RSI_buy_range': (25, 35),         # Less oversold needed
                'MACD_buy_range': (-2, 5),         # More bullish MACD needed
                'WILLR_buy_range': (-80, -60),     # Less oversold needed
                'signal_persistence': 2,            # Wait for confirmation
                'use_trend_filter': True            # Use trend filter in bull markets
            },
            
            'bear': {
                'buy_score_threshold': 2,           # Lower threshold in bear markets
                'sell_score_threshold': 3,          # Higher threshold to avoid false sells
                'RSI_buy_range': (20, 40),         # More oversold range
                'MACD_buy_range': (-8, 2),         # Allow more bearish MACD
                'WILLR_buy_range': (-90, -50),     # Deeper oversold
                'signal_persistence': 1,            # Faster response
                'use_trend_filter': False           # Don't use trend filter in bear markets
            },
            
            'crash': {
                'buy_score_threshold': 1,           # LOWEST threshold for crash buying
                'sell_score_threshold': 4,          # High threshold to avoid panic selling
                'RSI_buy_range': (15, 45),         # WIDE range to catch April 21.6 RSI
                'MACD_buy_range': (-20, 5),        # VERY bearish MACD allowed (-16.975 April)
                'WILLR_buy_range': (-95, -40),     # VERY oversold range (-84.5 April)
                'signal_persistence': 1,            # Immediate response to crash signals
                'use_trend_filter': False           # NO trend filter during crashes
            },
            
            'sideways': {
                'buy_score_threshold': 2,           # Moderate threshold
                'sell_score_threshold': 2,          # Balanced
                'RSI_buy_range': (25, 40),         # Standard oversold
                'MACD_buy_range': (-5, 3),         # Balanced MACD
                'WILLR_buy_range': (-85, -55),     # Standard oversold
                'signal_persistence': 2,            # Wait for confirmation
                'use_trend_filter': True            # Use trend filter in sideways markets
            }
        }
        
        regime_params = base_params.get(regime, base_params['sideways'])
        
        # Adjust confidence - lower confidence means use more conservative (sideways) settings
        if confidence < 0.6:
            # Blend with sideways parameters
            sideways_params = base_params['sideways']
            for key in regime_params:
                if key in sideways_params:
                    if isinstance(regime_params[key], tuple):
                        # Blend ranges
                        regime_range = regime_params[key]
                        sideways_range = sideways_params[key]
                        blend_factor = confidence
                        blended_range = (
                            regime_range[0] * blend_factor + sideways_range[0] * (1 - blend_factor),
                            regime_range[1] * blend_factor + sideways_range[1] * (1 - blend_factor)
                        )
                        regime_params[key] = blended_range
                    elif isinstance(regime_params[key], (int, float)):
                        # Blend numeric values
                        regime_params[key] = (
                            regime_params[key] * confidence + 
                            sideways_params[key] * (1 - confidence)
                        )
        
        return regime_params

def create_regime_aware_strategy_params(data, current_idx, base_params):
    """
    Create strategy parameters that adapt to current market regime
    
    This replaces the fixed parameters with regime-aware ones
    """
    
    detector = MarketRegimeDetector()
    regime, confidence = detector.detect_regime(data, current_idx)
    regime_params = detector.get_regime_parameters(regime, confidence)
    
    # Update base parameters with regime-aware settings
    adapted_params = base_params.copy()
    
    # Core voting thresholds
    adapted_params['buy_score_threshold'] = int(regime_params['buy_score_threshold'])
    adapted_params['sell_score_threshold'] = int(regime_params['sell_score_threshold'])
    adapted_params['signal_persistence_days'] = int(regime_params['signal_persistence'])
    adapted_params['use_trend_filter'] = regime_params['use_trend_filter']
    
    # Update indicator ranges (use middle of range for now, could be randomized)
    rsi_range = regime_params['RSI_buy_range']
    adapted_params['RSI_14_buy'] = (rsi_range[0] + rsi_range[1]) / 2
    
    macd_range = regime_params['MACD_buy_range'] 
    adapted_params['MACD_12_26_9_buy'] = (macd_range[0] + macd_range[1]) / 2
    
    willr_range = regime_params['WILLR_buy_range']
    adapted_params['WILLR_14_buy'] = (willr_range[0] + willr_range[1]) / 2
    
    return adapted_params, regime, confidence

if __name__ == "__main__":
    print("📊 Market Regime Detector - Enhancement #3")
    print("🎯 Dynamically adapts strategy parameters based on market conditions")
    print("🚨 Optimized to catch crashes like April 2025!")
