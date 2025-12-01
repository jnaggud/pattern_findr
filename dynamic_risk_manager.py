"""
Dynamic Risk Management for ML Trading System
Calculates dynamic stop loss and take profit levels based on:
1. ML model confidence levels
2. Composite technical indicator strength
3. Market volatility (ATR)
4. Volume-weighted support/resistance levels
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

class DynamicRiskManager:
    def __init__(self):
        self.default_sl_pct = 8.0  # Base stop loss %
        self.default_tp_pct = 15.0  # Base take profit %
        
    def calculate_dynamic_levels(self, 
                               current_price: float,
                               ml_confidence: float,
                               composite_tech: float,
                               signal_direction: int,
                               price_data: pd.DataFrame,
                               volume_data: pd.Series = None) -> Dict[str, float]:
        """
        Calculate dynamic stop loss and take profit levels
        
        Args:
            current_price: Current asset price
            ml_confidence: ML model confidence (-1 to 1)
            composite_tech: Composite technical indicator (-1 to 1)
            signal_direction: 1 for long, -1 for short
            price_data: Historical price data (high, low, close)
            volume_data: Volume data for VWAP calculations
            
        Returns:
            Dict with stop_loss, take_profit, and risk_reward_ratio
        """
        
        # 1. Calculate base volatility (ATR)
        atr_periods = 14
        atr = self._calculate_atr(price_data, atr_periods)
        current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.02
        
        # 2. Calculate ML confidence multiplier
        # High confidence = tighter stops, wider targets
        abs_confidence = abs(ml_confidence)
        confidence_multiplier = 0.5 + (abs_confidence * 1.5)  # 0.5x to 2.0x
        
        # 3. Calculate technical strength multiplier
        # Strong technical = similar to confidence
        tech_strength = abs(composite_tech)
        tech_multiplier = 0.7 + (tech_strength * 0.8)  # 0.7x to 1.5x
        
        # 4. Calculate convergence bonus (when ML and tech agree)
        convergence_score = self._calculate_convergence(ml_confidence, composite_tech, signal_direction)
        convergence_multiplier = 1.0 + (convergence_score * 0.5)  # Up to 1.5x bonus
        
        # 5. Volume-weighted levels (if volume data available)
        vwap_support, vwap_resistance = self._calculate_vwap_levels(price_data, volume_data)
        
        # 6. Calculate dynamic stop loss
        base_sl_atr = current_atr * 2.0  # 2 ATR base
        
        # Adjust based on confidence (high confidence = tighter stop)
        sl_atr_adjusted = base_sl_atr * (1 / confidence_multiplier)
        
        # Convert to percentage
        sl_percent = (sl_atr_adjusted / current_price) * 100
        
        # Apply bounds (minimum 3%, maximum 15%)
        sl_percent = np.clip(sl_percent, 3.0, 15.0)
        
        # 7. Calculate dynamic take profit
        # Base: Risk-Reward ratio from 1.5:1 to 3:1 based on confidence
        base_rr_ratio = 1.5 + (confidence_multiplier * convergence_multiplier * 0.5)
        
        # Use VWAP levels if available and reasonable
        if signal_direction == 1:  # Long
            if vwap_resistance and vwap_resistance > current_price:
                vwap_tp_percent = ((vwap_resistance - current_price) / current_price) * 100
                if 10 <= vwap_tp_percent <= 40:  # Reasonable range
                    tp_percent = vwap_tp_percent
                else:
                    tp_percent = sl_percent * base_rr_ratio
            else:
                tp_percent = sl_percent * base_rr_ratio
        else:  # Short
            if vwap_support and vwap_support < current_price:
                vwap_tp_percent = ((current_price - vwap_support) / current_price) * 100
                if 10 <= vwap_tp_percent <= 40:
                    tp_percent = vwap_tp_percent
                else:
                    tp_percent = sl_percent * base_rr_ratio
            else:
                tp_percent = sl_percent * base_rr_ratio
        
        # Apply bounds for TP (minimum 8%, maximum 50%)
        tp_percent = np.clip(tp_percent, 8.0, 50.0)
        
        # 8. Calculate actual price levels
        if signal_direction == 1:  # Long position
            stop_loss = current_price * (1 - sl_percent / 100)
            take_profit = current_price * (1 + tp_percent / 100)
        else:  # Short position
            stop_loss = current_price * (1 + sl_percent / 100)
            take_profit = current_price * (1 - tp_percent / 100)
        
        return {
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'stop_loss_pct': round(sl_percent, 2),
            'take_profit_pct': round(tp_percent, 2),
            'risk_reward_ratio': round(tp_percent / sl_percent, 2),
            'confidence_multiplier': round(confidence_multiplier, 2),
            'tech_multiplier': round(tech_multiplier, 2),
            'convergence_score': round(convergence_score, 2),
            'atr_value': round(current_atr, 2),
            'vwap_support': vwap_support,
            'vwap_resistance': vwap_resistance
        }
    
    def _calculate_atr(self, price_data: pd.DataFrame, periods: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        if len(price_data) < periods:
            return pd.Series([price_data['high'].iloc[-1] - price_data['low'].iloc[-1]], 
                           index=[price_data.index[-1]])
        
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=periods, min_periods=1).mean()
        
        return atr
    
    def _calculate_convergence(self, ml_confidence: float, composite_tech: float, signal_direction: int) -> float:
        """
        Calculate convergence score between ML and technical indicators
        Returns 0-1 where 1 is perfect agreement
        """
        # Normalize both to same scale
        ml_normalized = ml_confidence
        tech_normalized = composite_tech
        
        # For the signal direction, we want both indicators pointing same way
        if signal_direction == 1:  # Long signal
            ml_bullish = max(0, ml_normalized)  # Positive part
            tech_bullish = max(0, tech_normalized)
            agreement = min(ml_bullish, tech_bullish)
        else:  # Short signal
            ml_bearish = abs(min(0, ml_normalized))  # Negative part made positive
            tech_bearish = abs(min(0, tech_normalized))
            agreement = min(ml_bearish, tech_bearish)
        
        return agreement
    
    def _calculate_vwap_levels(self, price_data: pd.DataFrame, volume_data: pd.Series = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate Volume Weighted Average Price support and resistance levels
        """
        if volume_data is None or len(price_data) < 20:
            return None, None
        
        try:
            # Use last 20 days for VWAP calculation
            recent_data = price_data.tail(20)
            recent_volume = volume_data.tail(20)
            
            # Calculate VWAP
            typical_price = (recent_data['high'] + recent_data['low'] + recent_data['close']) / 3
            vwap = (typical_price * recent_volume).sum() / recent_volume.sum()
            
            # Calculate standard deviation bands
            price_vol_product = typical_price * recent_volume
            squared_diff = ((typical_price - vwap) ** 2) * recent_volume
            variance = squared_diff.sum() / recent_volume.sum()
            std_dev = np.sqrt(variance)
            
            # Support and resistance levels (VWAP ± 1 std dev)
            vwap_support = vwap - std_dev
            vwap_resistance = vwap + std_dev
            
            return round(vwap_support, 2), round(vwap_resistance, 2)
            
        except Exception:
            return None, None


class EnhancedSignalFilter:
    """
    Enhanced signal filtering for better entry timing
    """
    def __init__(self):
        self.min_confidence = 0.3  # Minimum ML confidence for trade
        self.min_tech_strength = 0.2  # Minimum technical strength
        self.min_convergence = 0.1  # Minimum convergence score
        
    def should_enter_trade(self, 
                          signal: int,
                          ml_confidence: float, 
                          composite_tech: float,
                          previous_ml_confidence: pd.Series,
                          previous_composite_tech: pd.Series,
                          lookback_periods: int = 5) -> Dict[str, any]:
        """
        Advanced signal filtering logic
        
        Returns:
            Dict with 'enter_trade' boolean and reasoning
        """
        if signal == 0:
            return {'enter_trade': False, 'reason': 'No signal generated'}
        
        abs_ml_conf = abs(ml_confidence)
        abs_tech_strength = abs(composite_tech)
        
        # 1. Minimum confidence check
        if abs_ml_conf < self.min_confidence:
            return {'enter_trade': False, 'reason': f'ML confidence too low: {ml_confidence:.3f}'}
        
        # 2. Technical strength check
        if abs_tech_strength < self.min_tech_strength:
            return {'enter_trade': False, 'reason': f'Technical strength too low: {composite_tech:.3f}'}
        
        # 3. Convergence check (both should point same direction)
        if signal == 1:  # Long signal
            if ml_confidence < 0 or composite_tech < -0.3:  # Tech very bearish
                return {'enter_trade': False, 'reason': 'ML and technical divergence for long signal'}
        elif signal == -1:  # Short signal
            if ml_confidence > 0 or composite_tech > 0.3:  # Tech very bullish
                return {'enter_trade': False, 'reason': 'ML and technical divergence for short signal'}
        
        # 4. Momentum check - avoid entries at local extremes
        if len(previous_ml_confidence) >= lookback_periods:
            recent_ml = previous_ml_confidence.tail(lookback_periods)
            recent_tech = previous_composite_tech.tail(lookback_periods)
            
            # Check if we're at a local extreme (potential reversal point)
            current_ml_rank = (recent_ml <= ml_confidence).sum()
            current_tech_rank = (recent_tech <= composite_tech).sum()
            
            # If current reading is at extreme (top 20% or bottom 20%), be cautious
            extreme_threshold = lookback_periods * 0.8
            if signal == 1:  # Long signal
                if current_ml_rank >= extreme_threshold and current_tech_rank >= extreme_threshold:
                    return {'enter_trade': False, 'reason': 'Indicators at local extremes - potential reversal'}
            elif signal == -1:  # Short signal  
                if current_ml_rank <= lookback_periods * 0.2 and current_tech_rank <= lookback_periods * 0.2:
                    return {'enter_trade': False, 'reason': 'Indicators at local extremes - potential reversal'}
        
        # 5. All checks passed
        convergence_score = min(abs_ml_conf, abs_tech_strength)
        return {
            'enter_trade': True, 
            'reason': f'Strong signal: ML={ml_confidence:.3f}, Tech={composite_tech:.3f}, Conv={convergence_score:.3f}'
        }
