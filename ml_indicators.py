"""
Deep Learning & Machine Learning Indicators for Pattern_FindR
Uses pre-trained models for fast inference during optimization
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy.signal import find_peaks
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow for loading pre-trained models
try:
    import tensorflow as tf
    from keras.models import load_model
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

class PretrainedModelLoader:
    """Loads and manages pre-trained ML models"""
    
    def __init__(self, model_dir='trained_models'):
        self.model_dir = model_dir
        self.models = {}
        self.metadata = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available pre-trained models"""
        if not os.path.exists(self.model_dir):
            print(f"⚠️  Model directory {self.model_dir} not found. Run model_trainer.py first.")
            return
        
        # Load metadata
        metadata_path = os.path.join(self.model_dir, 'model_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        
        # Load LSTM model
        if KERAS_AVAILABLE:
            lstm_path = os.path.join(self.model_dir, 'lstm_model.h5')
            scaler_path = os.path.join(self.model_dir, 'lstm_scaler.pkl')
            
            if os.path.exists(lstm_path) and os.path.exists(scaler_path):
                try:
                    self.models['lstm'] = load_model(lstm_path)
                    with open(scaler_path, 'rb') as f:
                        self.models['lstm_scaler'] = pickle.load(f)
                except:
                    pass
        
        # Load Random Forest models
        rf_path = os.path.join(self.model_dir, 'random_forest_models.pkl')
        if os.path.exists(rf_path):
            try:
                with open(rf_path, 'rb') as f:
                    self.models['random_forest'] = pickle.load(f)
            except:
                pass
        
        # Load Clustering models
        clustering_path = os.path.join(self.model_dir, 'clustering_models.pkl')
        if os.path.exists(clustering_path):
            try:
                with open(clustering_path, 'rb') as f:
                    self.models['clustering'] = pickle.load(f)
            except:
                pass
    
    def is_model_fresh(self, max_age_days=7):
        """Check if models are fresh enough"""
        if not self.metadata:
            return False
        
        training_date = datetime.fromisoformat(self.metadata.get('training_date', '2020-01-01'))
        age = datetime.now() - training_date
        return age.days <= max_age_days
    
    def get_model(self, model_type, model_name=None):
        """Get a specific model"""
        if model_type not in self.models:
            return None
        
        if model_name:
            return self.models[model_type].get(model_name)
        return self.models[model_type]

# Global model loader instance
MODEL_LOADER = PretrainedModelLoader()

class MLIndicators:
    """Machine Learning based technical indicators"""
    
    def __init__(self, data):
        """
        Initialize with OHLCV data
        
        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.data = data.copy()
        self.lookback = 20
        self.scaler = MinMaxScaler()
        
    def calculate_all_ml_indicators(self):
        """Calculate all ML-based indicators"""
        indicators = {}
        
        # 1. LSTM Price Prediction (if available)
        if KERAS_AVAILABLE and len(self.data) >= 100:
            try:
                indicators['lstm_prediction'] = self.lstm_price_prediction()
                indicators['lstm_trend_signal'] = self.lstm_trend_signal()
            except Exception as e:
                print(f"LSTM calculation failed: {e}")
                indicators['lstm_prediction'] = 0
                indicators['lstm_trend_signal'] = 0
        
        # 2. Anomaly Detection
        indicators['price_anomaly'] = self.detect_price_anomalies()
        indicators['volume_anomaly'] = self.detect_volume_anomalies()
        
        # 3. Market Regime Detection
        indicators['market_regime'] = self.detect_market_regime()
        indicators['regime_confidence'] = self.regime_confidence()
        
        # 4. Pattern Recognition with Random Forest
        indicators['rf_signal'] = self.random_forest_signal()
        indicators['rf_confidence'] = self.random_forest_confidence()
        
        # 5. Volatility Prediction
        indicators['predicted_volatility'] = self.predict_volatility()
        indicators['volatility_regime'] = self.volatility_regime()
        
        # 6. Support/Resistance ML Detection
        indicators['ml_support'] = self.ml_support_level()
        indicators['ml_resistance'] = self.ml_resistance_level()
        indicators['distance_to_support'] = self.distance_to_support()
        indicators['distance_to_resistance'] = self.distance_to_resistance()
        
        # 7. Momentum Clustering
        indicators['momentum_cluster'] = self.momentum_clustering()
        
        # 8. Gradient Boosting Trend Prediction
        indicators['gb_trend_prediction'] = self.gradient_boosting_trend()
        
        # 9. PCA-based Market State
        indicators['pca_state'] = self.pca_market_state()
        
        # 10. Ensemble Signal (combines multiple ML models)
        indicators['ensemble_signal'] = self.ensemble_ml_signal(indicators)
        
        return indicators
    
    def lstm_price_prediction(self):
        """Use pre-trained LSTM to predict next price movement (FAST)"""
        # Get pre-trained LSTM model
        lstm_model = MODEL_LOADER.get_model('lstm')
        lstm_scaler = MODEL_LOADER.get_model('lstm_scaler')
        
        if not lstm_model or not lstm_scaler or len(self.data) < 10:
            return 0.0
        
        try:
            # Use the same scaler that was used during training
            close_prices = self.data['close'].values
            scaled_data = lstm_scaler.transform(close_prices.reshape(-1, 1))
            
            # Get last sequence (model was trained with lookback=10)
            lookback = 10
            if len(scaled_data) < lookback:
                return 0.0
                
            last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
            
            # Fast inference - no training!
            prediction_scaled = lstm_model.predict(last_sequence, verbose=0)[0][0]
            
            # Convert to percentage change
            current_price = self.data['close'].iloc[-1]
            predicted_price = lstm_scaler.inverse_transform([[prediction_scaled]])[0][0]
            
            return (predicted_price - current_price) / current_price
            
        except Exception as e:
            return 0.0
    
    def lstm_trend_signal(self):
        """Convert LSTM prediction to binary signal"""
        prediction = self.lstm_price_prediction()
        if prediction > 1.0:  # Predicts >1% increase
            return 1  # Bullish
        elif prediction < -1.0:  # Predicts >1% decrease
            return -1  # Bearish
        else:
            return 0  # Neutral
    
    def detect_price_anomalies(self):
        """
        Detect anomalous price movements using Isolation Forest
        Returns: 1 if current price is anomalous, 0 otherwise
        """
        if len(self.data) < 50:
            return 0
        
        try:
            # Features: returns, volume, volatility
            features = pd.DataFrame({
                'returns': self.data['close'].pct_change(),
                'volume_change': self.data['volume'].pct_change(),
                'high_low_range': (self.data['high'] - self.data['low']) / self.data['close']
            }).fillna(0)
            
            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(features)
            
            # Return if current point is anomaly (-1 = anomaly, 1 = normal)
            return 1 if anomalies[-1] == -1 else 0
            
        except Exception as e:
            return 0
    
    def detect_volume_anomalies(self):
        """Detect anomalous volume using Z-score"""
        if len(self.data) < 20:
            return 0
        
        try:
            volume_zscore = zscore(self.data['volume'].tail(50))
            current_zscore = volume_zscore.iloc[-1]
            
            # Volume is anomalous if Z-score > 2
            return 1 if abs(current_zscore) > 2 else 0
        except:
            return 0
    
    def detect_market_regime(self):
        """
        Detect market regime using K-Means clustering
        Returns: 0 = Low volatility/range, 1 = Trending, 2 = High volatility
        """
        if len(self.data) < 50:
            return 0
        
        try:
            # Features for regime detection
            features = pd.DataFrame({
                'volatility': self.data['close'].pct_change().rolling(10).std(),
                'trend': self.data['close'].rolling(20).mean() - self.data['close'].rolling(50).mean(),
                'volume': self.data['volume'] / self.data['volume'].rolling(20).mean()
            }).fillna(0)
            
            # K-Means clustering
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features)
            
            return int(clusters[-1])
            
        except Exception as e:
            return 0
    
    def regime_confidence(self):
        """Confidence level for regime detection (0-1)"""
        if len(self.data) < 50:
            return 0.5
        
        try:
            features = pd.DataFrame({
                'volatility': self.data['close'].pct_change().rolling(10).std(),
                'trend': self.data['close'].rolling(20).mean() - self.data['close'].rolling(50).mean(),
                'volume': self.data['volume'] / self.data['volume'].rolling(20).mean()
            }).fillna(0)
            
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(features)
            
            # Distance to cluster center (smaller = more confident)
            distances = kmeans.transform(features.iloc[[-1]])
            min_distance = np.min(distances)
            
            # Convert to confidence (0-1)
            confidence = 1 / (1 + min_distance)
            return confidence
            
        except:
            return 0.5
    
    def random_forest_signal(self):
        """
        Use pre-trained Random Forest to predict next day's direction (FAST)
        Returns: 1 = up, -1 = down, 0 = uncertain
        """
        # Get pre-trained Random Forest model
        rf_models = MODEL_LOADER.get_model('random_forest')
        
        if not rf_models or 'direction' not in rf_models:
            return 0
        
        try:
            # Create same features as training
            self.data['returns'] = self.data['close'].pct_change()
            self.data['volatility'] = self.data['returns'].rolling(10).std()
            self.data['momentum'] = self.data['close'] - self.data['close'].shift(10)
            self.data['volume_change'] = self.data['volume'].pct_change()
            
            # Prepare current features
            features = ['returns', 'volatility', 'momentum', 'volume_change']
            current_features = self.data[features].fillna(0).iloc[[-1]]
            
            # Fast inference - no training!
            rf_model = rf_models['direction']
            prediction_proba = rf_model.predict_proba(current_features)[0]
            
            # Convert to signal with confidence threshold
            if prediction_proba[1] > 0.6:  # 60% confidence for up
                return 1
            elif prediction_proba[0] > 0.6:  # 60% confidence for down
                return -1
            else:
                return 0
                
        except:
            return 0
    
    def random_forest_confidence(self):
        """Return confidence level of Random Forest prediction"""
        if len(self.data) < 100:
            return 0.5
        
        try:
            # Same process as random_forest_signal but return probability
            self.data['returns'] = self.data['close'].pct_change()
            self.data['volatility'] = self.data['returns'].rolling(10).std()
            self.data['momentum'] = self.data['close'] - self.data['close'].shift(10)
            self.data['volume_change'] = self.data['volume'].pct_change()
            self.data['target'] = (self.data['returns'].shift(-1) > 0).astype(int)
            
            features = ['returns', 'volatility', 'momentum', 'volume_change']
            X = self.data[features].fillna(0)[:-1]
            y = self.data['target'].fillna(0)[:-1]
            
            if len(X) < 50:
                return 0.5
            
            rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)  # Optimized for speed
            rf.fit(X, y)
            
            current_features = self.data[features].fillna(0).iloc[[-1]]
            prediction_proba = rf.predict_proba(current_features)[0]
            
            return max(prediction_proba)
            
        except:
            return 0.5
    
    def predict_volatility(self):
        """Predict next period's volatility using ensemble methods"""
        if len(self.data) < 50:
            return 0
        
        try:
            # Historical volatility
            returns = self.data['close'].pct_change()
            vol_series = returns.rolling(10).std()
            
            # Simple autoregressive prediction
            recent_vols = vol_series.tail(10).values
            predicted_vol = np.mean(recent_vols) * 1.1  # Slight trending
            
            return predicted_vol * 100  # As percentage
            
        except:
            return 0
    
    def volatility_regime(self):
        """Classify current volatility regime: 0=low, 1=medium, 2=high"""
        if len(self.data) < 50:
            return 1
        
        try:
            current_vol = self.data['close'].pct_change().tail(10).std()
            historical_vol = self.data['close'].pct_change().std()
            
            if current_vol < historical_vol * 0.7:
                return 0  # Low vol
            elif current_vol > historical_vol * 1.3:
                return 2  # High vol
            else:
                return 1  # Normal vol
        except:
            return 1
    
    def ml_support_level(self):
        """Detect support level using peak detection and clustering"""
        if len(self.data) < 50:
            return self.data['close'].iloc[-1]
        
        try:
            # Find local minima
            lows = self.data['low'].values
            peaks, _ = find_peaks(-lows, distance=5)
            
            if len(peaks) < 3:
                return self.data['low'].tail(20).min()
            
            # Cluster the minima to find support zones
            peak_prices = lows[peaks].reshape(-1, 1)
            if len(peak_prices) >= 3:
                kmeans = KMeans(n_clusters=min(3, len(peak_prices)), random_state=42, n_init=10)
                kmeans.fit(peak_prices)
                
                # Find closest cluster below current price
                current_price = self.data['close'].iloc[-1]
                support_levels = sorted(kmeans.cluster_centers_.flatten())
                
                for level in reversed(support_levels):
                    if level < current_price:
                        return level
            
            return self.data['low'].tail(20).min()
            
        except:
            return self.data['low'].tail(20).min()
    
    def ml_resistance_level(self):
        """Detect resistance level using peak detection and clustering"""
        if len(self.data) < 50:
            return self.data['close'].iloc[-1]
        
        try:
            # Find local maxima
            highs = self.data['high'].values
            peaks, _ = find_peaks(highs, distance=5)
            
            if len(peaks) < 3:
                return self.data['high'].tail(20).max()
            
            # Cluster the maxima to find resistance zones
            peak_prices = highs[peaks].reshape(-1, 1)
            if len(peak_prices) >= 3:
                kmeans = KMeans(n_clusters=min(3, len(peak_prices)), random_state=42, n_init=10)
                kmeans.fit(peak_prices)
                
                # Find closest cluster above current price
                current_price = self.data['close'].iloc[-1]
                resistance_levels = sorted(kmeans.cluster_centers_.flatten())
                
                for level in resistance_levels:
                    if level > current_price:
                        return level
            
            return self.data['high'].tail(20).max()
            
        except:
            return self.data['high'].tail(20).max()
    
    def distance_to_support(self):
        """Percentage distance to support level"""
        try:
            current = self.data['close'].iloc[-1]
            support = self.ml_support_level()
            return ((current - support) / current) * 100
        except:
            return 0
    
    def distance_to_resistance(self):
        """Percentage distance to resistance level"""
        try:
            current = self.data['close'].iloc[-1]
            resistance = self.ml_resistance_level()
            return ((resistance - current) / current) * 100
        except:
            return 0
    
    def momentum_clustering(self):
        """Cluster current momentum state: 0=weak, 1=moderate, 2=strong"""
        if len(self.data) < 50:
            return 1
        
        try:
            # Calculate multiple momentum metrics
            roc_5 = ((self.data['close'].iloc[-1] - self.data['close'].iloc[-6]) / 
                     self.data['close'].iloc[-6]) * 100
            roc_10 = ((self.data['close'].iloc[-1] - self.data['close'].iloc[-11]) / 
                      self.data['close'].iloc[-11]) * 100
            volume_trend = self.data['volume'].tail(10).mean() / self.data['volume'].tail(20).mean()
            
            # Simple rule-based clustering
            momentum_score = abs(roc_5) + abs(roc_10) + (volume_trend - 1) * 10
            
            if momentum_score < 5:
                return 0  # Weak
            elif momentum_score > 15:
                return 2  # Strong
            else:
                return 1  # Moderate
        except:
            return 1
    
    def gradient_boosting_trend(self):
        """Use Gradient Boosting to predict trend: 1=up, -1=down, 0=sideways"""
        if len(self.data) < 100:
            return 0
        
        try:
            # Create features
            self.data['returns'] = self.data['close'].pct_change()
            self.data['sma_20'] = self.data['close'].rolling(20).mean()
            self.data['sma_50'] = self.data['close'].rolling(50).mean()
            self.data['volume_ma'] = self.data['volume'].rolling(20).mean()
            
            # Create labels
            future_return = self.data['close'].shift(-5) / self.data['close'] - 1
            self.data['trend'] = 0
            self.data.loc[future_return > 0.02, 'trend'] = 1  # Up trend
            self.data.loc[future_return < -0.02, 'trend'] = -1  # Down trend
            
            features = ['returns', 'sma_20', 'sma_50', 'volume_ma']
            X = self.data[features].fillna(method='ffill').fillna(0)[:-5]
            y = self.data['trend'].fillna(0)[:-5]
            
            if len(X) < 50:
                return 0
            
            # Convert to classification problem (3 classes: -1, 0, 1)
            y = y + 1  # Now 0, 1, 2
            
            gb = GradientBoostingClassifier(n_estimators=20, max_depth=2, random_state=42)  # Optimized for speed
            gb.fit(X, y)
            
            current_features = self.data[features].fillna(method='ffill').fillna(0).iloc[[-1]]
            prediction = gb.predict(current_features)[0]
            
            return prediction - 1  # Convert back to -1, 0, 1
            
        except Exception as e:
            return 0
    
    def pca_market_state(self):
        """Use PCA to reduce market state to single dimension"""
        if len(self.data) < 50:
            return 0
        
        try:
            # Multiple market features
            features = pd.DataFrame({
                'returns': self.data['close'].pct_change(),
                'volatility': self.data['close'].pct_change().rolling(10).std(),
                'volume': self.data['volume'] / self.data['volume'].rolling(20).mean(),
                'high_low': (self.data['high'] - self.data['low']) / self.data['close'],
                'trend': self.data['close'].rolling(20).mean() / self.data['close'].rolling(50).mean() - 1
            }).fillna(0)
            
            # PCA to 1 component
            pca = PCA(n_components=1)
            pca_values = pca.fit_transform(features)
            
            return pca_values[-1][0]
            
        except:
            return 0
    
    def ensemble_ml_signal(self, indicators):
        """
        Combine multiple ML signals into one ensemble signal
        Returns: 1=strong buy, 0.5=weak buy, 0=neutral, -0.5=weak sell, -1=strong sell
        """
        try:
            signals = []
            
            # Weight each signal
            if 'lstm_trend_signal' in indicators and indicators['lstm_trend_signal'] != 0:
                signals.append(indicators['lstm_trend_signal'] * 0.3)  # 30% weight
            
            if 'rf_signal' in indicators and indicators['rf_signal'] != 0:
                signals.append(indicators['rf_signal'] * 0.25)  # 25% weight
            
            if 'gb_trend_prediction' in indicators and indicators['gb_trend_prediction'] != 0:
                signals.append(indicators['gb_trend_prediction'] * 0.25)  # 25% weight
            
            # Regime and anomaly adjustments
            if 'market_regime' in indicators and indicators['market_regime'] == 2:
                # High volatility regime - reduce signal strength
                signals = [s * 0.7 for s in signals]
            
            if 'price_anomaly' in indicators and indicators['price_anomaly'] == 1:
                # Price anomaly detected - be cautious
                signals = [s * 0.5 for s in signals]
            
            if len(signals) == 0:
                return 0
            
            # Average and normalize
            ensemble = np.mean(signals)
            ensemble = np.clip(ensemble, -1, 1)
            
            return ensemble
            
        except:
            return 0


def integrate_ml_indicators(data):
    """
    Main function to integrate ML indicators into the data pipeline
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        Dictionary of ML indicator values
    """
    if len(data) < 50:
        return {}
    
    try:
        ml = MLIndicators(data)
        indicators = ml.calculate_all_ml_indicators()
        return indicators
    except Exception as e:
        print(f"ML indicators calculation failed: {e}")
        return {}


# List of all ML indicators for reference
ML_INDICATOR_LIST = [
    'lstm_prediction',
    'lstm_trend_signal',
    'price_anomaly',
    'volume_anomaly',
    'market_regime',
    'regime_confidence',
    'rf_signal',
    'rf_confidence',
    'predicted_volatility',
    'volatility_regime',
    'ml_support',
    'ml_resistance',
    'distance_to_support',
    'distance_to_resistance',
    'momentum_cluster',
    'gb_trend_prediction',
    'pca_market_state',
    'ensemble_ml_signal'
]
