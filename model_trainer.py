"""
Model Training Pipeline for Pattern_FindR
Trains and saves all ML/DL models for fast inference during optimization
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow for LSTM
try:
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

class ModelTrainer:
    """Trains and saves all ML models for Pattern_FindR"""
    
    def __init__(self, model_dir='trained_models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def load_portfolio_tickers(self):
        """Load portfolio tickers from app configuration - completely dynamic"""
        tickers = []
        
        # Method 1: Try portfolio_config.json (if saved from app)
        try:
            if os.path.exists('portfolio_config.json'):
                with open('portfolio_config.json', 'r') as f:
                    portfolio_data = json.load(f)
                    tickers = portfolio_data.get('tickers', [])
                    if tickers:
                        print(f"📁 Loaded portfolio tickers from config: {tickers}")
                        return tickers
        except Exception as e:
            print(f"⚠️  Could not load portfolio_config.json: {e}")
        
        # Method 2: Check if app has any saved strategies (extract tickers from those)
        try:
            if os.path.exists('saved_strategies'):
                strategy_files = [f for f in os.listdir('saved_strategies') if f.endswith('.json')]
                if strategy_files:
                    # Get tickers from recent strategies
                    recent_tickers = set()
                    for filename in strategy_files[-10:]:  # Last 10 strategies
                        try:
                            with open(os.path.join('saved_strategies', filename), 'r') as f:
                                strategy = json.load(f)
                                if 'ticker' in strategy:
                                    recent_tickers.add(strategy['ticker'])
                        except:
                            continue
                    
                    if recent_tickers:
                        tickers = list(recent_tickers)
                        print(f"📊 Extracted tickers from saved strategies: {tickers}")
                        return tickers
        except Exception as e:
            print(f"⚠️  Could not extract from saved strategies: {e}")
        
        # Method 3: Interactive input if no configuration found
        print("\n🤔 No portfolio configuration found!")
        print("Please enter the tickers you want to train models on.")
        print("These should match the tickers you use in your app portfolio.")
        print("\nExamples:")
        print("  - SPY MSTY MSTR")
        print("  - AAPL TSLA NVDA MSFT")
        print("  - QQQ VTI IWM")
        
        while not tickers:
            ticker_input = input("\nEnter tickers (space-separated): ").strip().upper()
            if ticker_input:
                tickers = ticker_input.split()
                print(f"✅ Will train on: {tickers}")
                
                # Save this configuration for future use
                try:
                    config = {'tickers': tickers, 'created_by': 'model_trainer', 'timestamp': datetime.now().isoformat()}
                    with open('portfolio_config.json', 'w') as f:
                        json.dump(config, f, indent=2)
                    print(f"💾 Saved configuration to portfolio_config.json")
                except Exception as e:
                    print(f"⚠️  Could not save config: {e}")
                
                return tickers
            else:
                print("Please enter at least one ticker.")
        
        return tickers
        
    def get_training_data(self, symbols=None, period='2y'):
        """Get comprehensive training data from portfolio tickers"""
        # Load portfolio tickers if not specified
        if symbols is None:
            symbols = self.load_portfolio_tickers()
        
        print(f"📈 Fetching training data for {len(symbols)} symbols: {', '.join(symbols)}")
        
        all_data = []
        for symbol in symbols:
            try:
                data = yf.download(symbol, period=period, interval='1d')
                data = data.reset_index()
                data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                data['symbol'] = symbol
                all_data.append(data)
                print(f"   ✅ {symbol}: {len(data)} days")
            except Exception as e:
                print(f"   ❌ {symbol}: {e}")
                
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"📊 Combined dataset: {len(combined_data)} total rows")
        return combined_data
    
    def prepare_features(self, data):
        """Create features for ML training"""
        # Technical features
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(10).std()
        data['momentum'] = data['close'] - data['close'].shift(10)
        data['volume_change'] = data['volume'].pct_change()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['volume_ma'] = data['volume'].rolling(20).mean()
        
        # Labels for classification
        data['target_direction'] = (data['returns'].shift(-1) > 0).astype(int)
        data['target_trend'] = pd.cut(data['returns'].shift(-5).rolling(5).mean(), 
                                     bins=3, labels=[-1, 0, 1]).astype(float)
        
        return data.dropna()
    
    def train_lstm_model(self, data, lookback=10):
        """Train LSTM price prediction model"""
        if not KERAS_AVAILABLE:
            print("❌ TensorFlow not available - skipping LSTM")
            return None
            
        print("🧠 Training LSTM model...")
        
        # Prepare sequences for each symbol separately
        all_X, all_y = [], []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            close_prices = symbol_data['close'].values
            
            if len(close_prices) < lookback + 10:
                continue
                
            # Scale data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))
            
            # Create sequences
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i, 0])
                y.append(scaled_data[i, 0])
            
            all_X.extend(X)
            all_y.extend(y)
        
        if len(all_X) == 0:
            print("❌ No valid sequences for LSTM")
            return None
            
        X = np.array(all_X)
        y = np.array(all_y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Build model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train with proper validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model.fit(X_train, y_train, 
                 validation_data=(X_val, y_val),
                 epochs=20, batch_size=64, verbose=1)
        
        # Save model and scaler
        model.save(os.path.join(self.model_dir, 'lstm_model.h5'))
        with open(os.path.join(self.model_dir, 'lstm_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
            
        print("✅ LSTM model trained and saved")
        return model
    
    def train_random_forest_models(self, data):
        """Train Random Forest models for different tasks"""
        print("🌳 Training Random Forest models...")
        
        features = ['returns', 'volatility', 'momentum', 'volume_change']
        X = data[features].fillna(0)
        
        models = {}
        
        # Direction prediction model
        y_direction = data['target_direction'].fillna(0)
        valid_mask = ~(X.isna().any(axis=1) | y_direction.isna())
        
        if valid_mask.sum() > 100:
            rf_direction = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rf_direction.fit(X[valid_mask], y_direction[valid_mask])
            models['direction'] = rf_direction
            print("   ✅ Direction prediction model")
        
        # Trend prediction model  
        y_trend = data['target_trend'].fillna(0)
        valid_mask = ~(X.isna().any(axis=1) | y_trend.isna())
        
        if valid_mask.sum() > 100:
            rf_trend = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rf_trend.fit(X[valid_mask], y_trend[valid_mask])
            models['trend'] = rf_trend
            print("   ✅ Trend prediction model")
        
        # Save models
        with open(os.path.join(self.model_dir, 'random_forest_models.pkl'), 'wb') as f:
            pickle.dump(models, f)
            
        return models
    
    def train_clustering_models(self, data):
        """Train clustering models for market regime detection"""
        print("🎯 Training clustering models...")
        
        features = pd.DataFrame({
            'volatility': data['volatility'],
            'trend': data['sma_20'] - data['sma_50'],
            'volume': data['volume'] / data['volume_ma']
        }).fillna(0)
        
        models = {}
        
        # Market regime clustering
        if len(features) > 100:
            kmeans_regime = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans_regime.fit(features)
            models['regime'] = kmeans_regime
            print("   ✅ Market regime clustering")
        
        # Support/Resistance clustering
        price_features = data[['high', 'low']].fillna(method='ffill').fillna(0)
        if len(price_features) > 100:
            kmeans_sr = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans_sr.fit(price_features)
            models['support_resistance'] = kmeans_sr  
            print("   ✅ Support/Resistance clustering")
        
        # Save models
        with open(os.path.join(self.model_dir, 'clustering_models.pkl'), 'wb') as f:
            pickle.dump(models, f)
            
        return models
    
    def train_all_models(self):
        """Train all ML models and save them"""
        print("🚀 Starting complete model training pipeline...")
        start_time = datetime.now()
        
        # Get training data
        data = self.get_training_data()
        data = self.prepare_features(data)
        
        print(f"\n📊 Training on {len(data)} samples across {data['symbol'].nunique()} symbols")
        
        # Train all models
        lstm_model = self.train_lstm_model(data)
        rf_models = self.train_random_forest_models(data)
        clustering_models = self.train_clustering_models(data)
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'data_samples': len(data),
            'symbols': list(data['symbol'].unique()),
            'models_trained': []
        }
        
        if lstm_model:
            metadata['models_trained'].append('LSTM')
        if rf_models:
            metadata['models_trained'].extend([f'RandomForest_{k}' for k in rf_models.keys()])
        if clustering_models:
            metadata['models_trained'].extend([f'Clustering_{k}' for k in clustering_models.keys()])
        
        with open(os.path.join(self.model_dir, 'model_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        elapsed = datetime.now() - start_time
        print(f"\n🎉 Model training complete!")
        print(f"⏱️  Total time: {elapsed}")
        print(f"📁 Models saved to: {self.model_dir}")
        print(f"🧠 Models trained: {', '.join(metadata['models_trained'])}")
        
        return metadata

def main():
    """Run the complete model training pipeline"""
    trainer = ModelTrainer()
    metadata = trainer.train_all_models()
    
    print(f"\n✅ All models ready for fast inference!")
    print(f"💡 Use these models in ml_indicators.py for real-time optimization")

if __name__ == "__main__":
    main()
