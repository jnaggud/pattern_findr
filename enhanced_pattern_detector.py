import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from advanced_indicators import get_advanced_indicators

@dataclass
class PatternSignal:
    """Data class for pattern detection signals."""
    pattern_name: str
    confidence: float
    signal_type: str  # 'bullish', 'bearish', 'neutral'
    signal_strength: float  # 0-100
    timestamp: pd.Timestamp
    additional_data: Dict = None

class EnhancedPatternDetector:
    """
    Enhanced pattern detector combining CNN models with time-series analysis.
    """
    
    def __init__(self, cnn_models=None):
        self.cnn_models = cnn_models or {}
        self.pattern_cache = {}
        self.confidence_threshold = 0.6
        
    def detect_comprehensive_patterns(self, data: pd.DataFrame) -> Dict[str, PatternSignal]:
        """
        Detect all patterns using both CNN and time-series methods.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary of pattern signals
        """
        signals = {}
        
        # Get advanced time-series patterns
        try:
            ts_patterns = get_advanced_indicators(data)
            ts_signals = self._convert_ts_to_signals(ts_patterns, data.index[-1])
            signals.update(ts_signals)
        except Exception as e:
            logging.warning(f"Time-series pattern detection failed: {e}")
        
        # Get CNN pattern predictions (if models are available)
        try:
            cnn_signals = self._get_cnn_pattern_signals(data)
            signals.update(cnn_signals)
        except Exception as e:
            logging.warning(f"CNN pattern detection failed: {e}")
        
        # Combine and validate signals
        validated_signals = self._validate_and_combine_signals(signals)
        
        return validated_signals
    
    def _convert_ts_to_signals(self, patterns: Dict, timestamp: pd.Timestamp) -> Dict[str, PatternSignal]:
        """Convert time-series pattern results to PatternSignal objects."""
        signals = {}
        
        # Trend signals
        trend = patterns.get('trend_strength', {})
        if trend.get('trend_strength', 0) > 30:
            signal_type = 'bullish' if trend.get('trend_direction', 0) > 0 else 'bearish'
            signals['strong_trend'] = PatternSignal(
                pattern_name='strong_trend',
                confidence=min(trend.get('trend_strength', 0) / 100, 1.0),
                signal_type=signal_type,
                signal_strength=trend.get('trend_strength', 0),
                timestamp=timestamp,
                additional_data={'adx': trend.get('adx', 0)}
            )
        
        # Mean reversion signals
        mean_rev = patterns.get('mean_reversion_signal', {})
        if mean_rev.get('bb_mean_reversion', False):
            signal_type = 'bullish' if mean_rev.get('oversold', False) else 'bearish'
            signals['mean_reversion'] = PatternSignal(
                pattern_name='mean_reversion',
                confidence=min(mean_rev.get('mean_reversion_strength', 0) / 100, 1.0),
                signal_type=signal_type,
                signal_strength=mean_rev.get('mean_reversion_strength', 0),
                timestamp=timestamp
            )
        
        # Volatility breakout signals
        vol_breakout = patterns.get('volatility_breakout', {})
        if vol_breakout.get('atr_breakout', False):
            signals['volatility_breakout'] = PatternSignal(
                pattern_name='volatility_breakout',
                confidence=0.7,
                signal_type='neutral',  # Direction depends on price movement
                signal_strength=min(vol_breakout.get('volatility_expansion', 100), 100),
                timestamp=timestamp
            )
        
        # Momentum divergence
        momentum_div = patterns.get('momentum_divergence', {})
        if momentum_div.get('bullish_divergence', False):
            signals['bullish_divergence'] = PatternSignal(
                pattern_name='bullish_divergence',
                confidence=0.8,
                signal_type='bullish',
                signal_strength=momentum_div.get('divergence_strength', 50),
                timestamp=timestamp
            )
        elif momentum_div.get('bearish_divergence', False):
            signals['bearish_divergence'] = PatternSignal(
                pattern_name='bearish_divergence',
                confidence=0.8,
                signal_type='bearish',
                signal_strength=momentum_div.get('divergence_strength', 50),
                timestamp=timestamp
            )
        
        # Support/Resistance levels
        support_res = patterns.get('support_resistance', {})
        if support_res:
            # Near resistance
            if support_res.get('distance_to_resistance', 100) < 2:  # Within 2%
                signals['near_resistance'] = PatternSignal(
                    pattern_name='near_resistance',
                    confidence=0.6,
                    signal_type='bearish',
                    signal_strength=support_res.get('resistance_strength', 1) * 20,
                    timestamp=timestamp,
                    additional_data={'level': support_res.get('nearest_resistance')}
                )
            
            # Near support
            if support_res.get('distance_to_support', 100) < 2:  # Within 2%
                signals['near_support'] = PatternSignal(
                    pattern_name='near_support',
                    confidence=0.6,
                    signal_type='bullish',
                    signal_strength=support_res.get('support_strength', 1) * 20,
                    timestamp=timestamp,
                    additional_data={'level': support_res.get('nearest_support')}
                )
        
        # Volume confirmation
        volume_price = patterns.get('volume_price_correlation', {})
        if volume_price.get('volume_confirms_price', False):
            signals['volume_confirmation'] = PatternSignal(
                pattern_name='volume_confirmation',
                confidence=min(volume_price.get('correlation_strength', 0) / 100, 1.0),
                signal_type='neutral',
                signal_strength=volume_price.get('correlation_strength', 0),
                timestamp=timestamp
            )
        
        # Cycle detection
        cycles = patterns.get('cycle_detection', {})
        if cycles.get('dominant_cycle', 0) > 0:
            signals['cycle_pattern'] = PatternSignal(
                pattern_name='cycle_pattern',
                confidence=min(cycles.get('cycle_strength', 0) / 100, 1.0),
                signal_type='neutral',
                signal_strength=cycles.get('cycle_strength', 0),
                timestamp=timestamp,
                additional_data={'cycle_length': cycles.get('dominant_cycle')}
            )
        
        return signals
    
    def _get_cnn_pattern_signals(self, data: pd.DataFrame) -> Dict[str, PatternSignal]:
        """Get CNN-based pattern signals."""
        signals = {}
        
        if not self.cnn_models:
            return signals
        
        # Generate chart image from recent data
        try:
            chart_image = self._create_chart_image(data.tail(50))  # Last 50 periods
            
            for pattern_name, model in self.cnn_models.items():
                try:
                    prediction, probability = model.predict(chart_image)
                    
                    if probability > self.confidence_threshold:
                        # Determine signal type based on pattern
                        signal_type = self._get_pattern_signal_type(pattern_name)
                        
                        signals[f'cnn_{pattern_name}'] = PatternSignal(
                            pattern_name=f'cnn_{pattern_name}',
                            confidence=probability,
                            signal_type=signal_type,
                            signal_strength=probability * 100,
                            timestamp=data.index[-1],
                            additional_data={'model_type': 'CNN'}
                        )
                except Exception as e:
                    logging.warning(f"CNN prediction failed for {pattern_name}: {e}")
        
        except Exception as e:
            logging.warning(f"Chart image creation failed: {e}")
        
        return signals
    
    def _get_pattern_signal_type(self, pattern_name: str) -> str:
        """Determine signal type for CNN patterns."""
        bullish_patterns = [
            'cup_and_handle', 'double_bottom', 'triple_bottom', 'inverse_head_and_shoulders',
            'rounding_bottom', 'ascending_triangle', 'hammer', 'engulfing_bullish',
            'harami_bullish', 'morning_star'
        ]
        
        bearish_patterns = [
            'head_and_shoulders', 'double_top', 'triple_top', 'rounding_top',
            'descending_triangle', 'shooting_star', 'engulfing_bearish',
            'harami_bearish', 'evening_star'
        ]
        
        if pattern_name in bullish_patterns:
            return 'bullish'
        elif pattern_name in bearish_patterns:
            return 'bearish'
        else:
            return 'neutral'
    
    def _create_chart_image(self, data: pd.DataFrame) -> np.ndarray:
        """Create a chart image from OHLC data for CNN analysis."""
        # This would integrate with your existing chart generation
        # For now, return a placeholder
        return np.random.rand(224, 224, 3)  # Placeholder image
    
    def _validate_and_combine_signals(self, signals: Dict[str, PatternSignal]) -> Dict[str, PatternSignal]:
        """Validate and combine overlapping signals."""
        validated = {}
        
        for name, signal in signals.items():
            # Filter by confidence threshold
            if signal.confidence >= self.confidence_threshold:
                validated[name] = signal
        
        # Combine conflicting signals (e.g., multiple trend signals)
        validated = self._resolve_signal_conflicts(validated)
        
        return validated
    
    def _resolve_signal_conflicts(self, signals: Dict[str, PatternSignal]) -> Dict[str, PatternSignal]:
        """Resolve conflicts between similar signals."""
        # Group signals by category
        trend_signals = [s for s in signals.values() if 'trend' in s.pattern_name]
        reversal_signals = [s for s in signals.values() if any(x in s.pattern_name for x in ['reversion', 'divergence'])]
        
        # Keep highest confidence signal in each category
        if trend_signals:
            best_trend = max(trend_signals, key=lambda x: x.confidence)
            # Remove other trend signals
            signals = {k: v for k, v in signals.items() if 'trend' not in v.pattern_name or v == best_trend}
        
        return signals
    
    def get_trading_signals(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Get simplified buy/sell signals for trading strategy.
        
        Returns:
            Tuple of (buy_signals, sell_signals)
        """
        patterns = self.detect_comprehensive_patterns(data)
        
        buy_signals = []
        sell_signals = []
        
        for name, signal in patterns.items():
            if signal.signal_type == 'bullish' and signal.confidence > 0.7:
                buy_signals.append(name)
            elif signal.signal_type == 'bearish' and signal.confidence > 0.7:
                sell_signals.append(name)
        
        return buy_signals, sell_signals
    
    def generate_pattern_score(self, data: pd.DataFrame) -> float:
        """
        Generate an overall pattern score for the current market state.
        
        Returns:
            Score between -100 (very bearish) and +100 (very bullish)
        """
        patterns = self.detect_comprehensive_patterns(data)
        
        total_score = 0
        total_weight = 0
        
        for signal in patterns.values():
            weight = signal.confidence * signal.signal_strength / 100
            
            if signal.signal_type == 'bullish':
                score = signal.signal_strength
            elif signal.signal_type == 'bearish':
                score = -signal.signal_strength
            else:
                score = 0
            
            total_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0
        
        return max(-100, min(100, total_score / total_weight))


def integrate_enhanced_patterns_with_optimization(data: pd.DataFrame) -> Dict[str, bool]:
    """
    Integration function for the main optimization system.
    
    Args:
        data: OHLCV DataFrame
        
    Returns:
        Dictionary of boolean signals for optimization
    """
    detector = EnhancedPatternDetector()
    
    try:
        buy_signals, sell_signals = detector.get_trading_signals(data)
        pattern_score = detector.generate_pattern_score(data)
        
        return {
            'enhanced_bullish_pattern': len(buy_signals) > 0,
            'enhanced_bearish_pattern': len(sell_signals) > 0,
            'enhanced_strong_bullish': pattern_score > 50,
            'enhanced_strong_bearish': pattern_score < -50,
            'enhanced_pattern_confluence': len(buy_signals) > 2 or len(sell_signals) > 2,
            'enhanced_neutral_market': abs(pattern_score) < 20
        }
    
    except Exception as e:
        logging.warning(f"Enhanced pattern detection failed: {e}")
        return {
            'enhanced_bullish_pattern': False,
            'enhanced_bearish_pattern': False,
            'enhanced_strong_bullish': False,
            'enhanced_strong_bearish': False,
            'enhanced_pattern_confluence': False,
            'enhanced_neutral_market': True
        }


# Add these new indicators to the main system
ENHANCED_PATTERN_INDICATORS = [
    'enhanced_bullish_pattern',
    'enhanced_bearish_pattern', 
    'enhanced_strong_bullish',
    'enhanced_strong_bearish',
    'enhanced_pattern_confluence',
    'enhanced_neutral_market'
]
