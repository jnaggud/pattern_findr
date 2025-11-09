import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import tempfile
import shutil
from dl_pattern_detector import create_chart_image
from dl_models import PatternDetectionEnsemble, CHART_PATTERNS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RealTimePatternDetector:
    """
    Real-time pattern detection for financial data using pre-trained models.
    """
    
    def __init__(self, load_models=True):
        self.ensemble = PatternDetectionEnsemble(CHART_PATTERNS)
        self.models_loaded = False
        
        if load_models:
            self.load_models()
    
    def load_models(self):
        """Load all pre-trained models."""
        try:
            self.ensemble.load_all_models()
            self.models_loaded = True
            logging.info("Pattern detection models loaded successfully")
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            logging.info("Models not found. Please train models first.")
            self.models_loaded = False
    
    def detect_patterns_in_data(self, data, window_size=60, confidence_threshold=0.7):
        """
        Detect patterns in financial data.
        
        Args:
            data (pd.DataFrame): OHLCV data with datetime index
            window_size (int): Number of days to analyze for pattern detection
            confidence_threshold (float): Minimum confidence for pattern detection
            
        Returns:
            dict: Detected patterns with their probabilities and locations
        """
        if not self.models_loaded:
            logging.warning("Models not loaded. Cannot detect patterns.")
            return {}
        
        if len(data) < window_size:
            logging.warning(f"Data too short ({len(data)} days) for pattern detection (need {window_size})")
            return {}
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            logging.error("Data missing required OHLC columns")
            return {}
        
        # Prepare data for mplfinance (capitalize column names)
        chart_data = data.copy()
        chart_data.columns = [col.capitalize() for col in chart_data.columns]
        
        # Create temporary directory for chart images
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Analyze the most recent window
            end_idx = len(chart_data) - 1
            start_idx = max(0, end_idx - window_size + 1)
            
            # Extract window data
            window_data = chart_data.iloc[start_idx:end_idx + 1].copy()
            
            # Create chart image
            chart_path = os.path.join(temp_dir, "pattern_analysis.png")
            chart_image_path = create_chart_image(
                window_data, 
                len(window_data) - 1, 
                window_size=len(window_data),
                save_path=chart_path
            )
            
            if chart_image_path is None:
                logging.error("Failed to create chart image")
                return {}
            
            # Detect patterns using ensemble
            pattern_results = self.ensemble.predict_all_patterns(
                chart_image_path, 
                threshold=confidence_threshold
            )
            
            # Process results
            detected_patterns = {}
            for pattern, result in pattern_results.items():
                if result['prediction'] and result['probability'] >= confidence_threshold:
                    detected_patterns[pattern] = {
                        'probability': result['probability'],
                        'confidence': result['probability'],
                        'pattern_type': self._get_pattern_type(pattern),
                        'signal_strength': self._calculate_signal_strength(result['probability']),
                        'detected_at': data.index[-1],
                        'analysis_window': f"{data.index[start_idx]} to {data.index[end_idx]}"
                    }
            
            logging.info(f"Pattern detection complete. Found {len(detected_patterns)} patterns above threshold.")
            
            return detected_patterns
            
        except Exception as e:
            logging.error(f"Error during pattern detection: {e}")
            return {}
        
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _get_pattern_type(self, pattern_name):
        """Get the type of pattern (bullish/bearish/continuation)."""
        bullish_patterns = [
            'inverse_head_and_shoulders',
            'cup_and_handle', 
            'double_bottom'
        ]
        
        bearish_patterns = [
            'head_and_shoulders',
            'double_top'
        ]
        
        continuation_patterns = [
            'triangle',
            'flag',
            'pennant'
        ]
        
        if pattern_name in bullish_patterns:
            return 'bullish'
        elif pattern_name in bearish_patterns:
            return 'bearish'
        elif pattern_name in continuation_patterns:
            return 'continuation'
        else:
            return 'neutral'
    
    def _calculate_signal_strength(self, probability):
        """Calculate signal strength based on probability."""
        if probability >= 0.9:
            return 'very_strong'
        elif probability >= 0.8:
            return 'strong'
        elif probability >= 0.7:
            return 'moderate'
        elif probability >= 0.6:
            return 'weak'
        else:
            return 'very_weak'
    
    def generate_trading_signals(self, detected_patterns, current_price=None):
        """
        Generate buy/sell signals based on detected patterns.
        
        Args:
            detected_patterns (dict): Output from detect_patterns_in_data
            current_price (float): Current price for context
            
        Returns:
            dict: Trading signals with buy/sell recommendations
        """
        signals = {
            'buy_signals': [],
            'sell_signals': [],
            'hold_signals': [],
            'overall_sentiment': 'neutral',
            'confidence_score': 0.0
        }
        
        if not detected_patterns:
            return signals
        
        bullish_score = 0
        bearish_score = 0
        total_confidence = 0
        
        for pattern_name, pattern_data in detected_patterns.items():
            pattern_type = pattern_data['pattern_type']
            probability = pattern_data['probability']
            signal_strength = pattern_data['signal_strength']
            
            # Weight the signal
            weight = probability
            
            if pattern_type == 'bullish':
                bullish_score += weight
                signals['buy_signals'].append({
                    'pattern': pattern_name,
                    'reason': f"{pattern_name.replace('_', ' ').title()} pattern detected",
                    'confidence': probability,
                    'strength': signal_strength
                })
            
            elif pattern_type == 'bearish':
                bearish_score += weight
                signals['sell_signals'].append({
                    'pattern': pattern_name,
                    'reason': f"{pattern_name.replace('_', ' ').title()} pattern detected",
                    'confidence': probability,
                    'strength': signal_strength
                })
            
            else:  # continuation or neutral
                signals['hold_signals'].append({
                    'pattern': pattern_name,
                    'reason': f"{pattern_name.replace('_', ' ').title()} pattern suggests continuation",
                    'confidence': probability,
                    'strength': signal_strength
                })
            
            total_confidence += probability
        
        # Determine overall sentiment
        if bullish_score > bearish_score * 1.2:  # Require 20% higher confidence for bullish
            signals['overall_sentiment'] = 'bullish'
        elif bearish_score > bullish_score * 1.2:
            signals['overall_sentiment'] = 'bearish'
        else:
            signals['overall_sentiment'] = 'neutral'
        
        # Calculate overall confidence
        signals['confidence_score'] = total_confidence / len(detected_patterns) if detected_patterns else 0.0
        
        return signals
    
    def get_pattern_based_signals(self, data, confidence_threshold=0.7):
        """
        Complete pipeline: detect patterns and generate signals.
        
        Args:
            data (pd.DataFrame): OHLCV data
            confidence_threshold (float): Minimum confidence for pattern detection
            
        Returns:
            tuple: (buy_signals, sell_signals) as boolean Series
        """
        # Detect patterns
        patterns = self.detect_patterns_in_data(data, confidence_threshold=confidence_threshold)
        
        # Generate signals
        trading_signals = self.generate_trading_signals(patterns)
        
        # Create boolean series for buy/sell signals
        buy_signals = pd.Series(False, index=data.index)
        sell_signals = pd.Series(False, index=data.index)
        
        # Mark the most recent date with signals if patterns detected
        if patterns and len(data) > 0:
            latest_date = data.index[-1]
            
            # Buy signals
            if trading_signals['overall_sentiment'] == 'bullish' and trading_signals['confidence_score'] > confidence_threshold:
                buy_signals.iloc[-1] = True
            
            # Sell signals  
            elif trading_signals['overall_sentiment'] == 'bearish' and trading_signals['confidence_score'] > confidence_threshold:
                sell_signals.iloc[-1] = True
        
        return buy_signals, sell_signals


def train_pattern_models():
    """
    Train all pattern detection models.
    This should be run once to create the models.
    """
    logging.info("Starting pattern model training...")
    
    from dl_models import get_pattern_ensemble
    
    # Create ensemble
    ensemble = get_pattern_ensemble()
    
    # Train all models
    ensemble.train_all_models(num_samples=200, epochs=25)
    
    logging.info("Pattern model training completed!")


def check_model_availability():
    """Check if trained models are available."""
    model_dirs = [f"models/{pattern}" for pattern in CHART_PATTERNS]
    available_models = []
    
    for pattern, model_dir in zip(CHART_PATTERNS, model_dirs):
        model_path = f"{model_dir}/model.h5"
        if os.path.exists(model_path):
            available_models.append(pattern)
    
    return available_models


if __name__ == "__main__":
    # Check if models exist
    available = check_model_availability()
    
    if len(available) == 0:
        print("No trained models found. Training models...")
        train_pattern_models()
    else:
        print(f"Found {len(available)} trained models: {available}")
        
        # Test the detector
        detector = RealTimePatternDetector()
        print("Pattern detector ready for use!")
