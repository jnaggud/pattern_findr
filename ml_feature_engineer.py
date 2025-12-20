#!/usr/bin/env python3
"""
ML Feature Engineering Module

Prepares features from technical indicators for machine learning models.
Uses existing indicators.py but adds ML-specific feature transformations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import logging

logger = logging.getLogger(__name__)

# Import existing indicator system (no modifications needed)
from indicators import get_all_indicators

# Import Deep Learning Extractor
try:
    from dl_feature_extractor import DLFeatureExtractor
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    logger.info("Deep Learning modules not found. DL features will be skipped.")

class MLFeatureEngineer:
    """
    Creates ML-ready features from technical indicators and price data
    """
    
    def __init__(self):
        self.feature_names = []
        self.target_columns = []
        self.correlation_matrix = None
        self.dl_extractor = None # Store the DL extractor instance

    def save_dl_model(self, filepath):
        """Save the DL extractor if it exists"""
        if self.dl_extractor:
            self.dl_extractor.save_model(filepath)
            return True
        return False

    def load_dl_model(self, filepath):
        """Load a DL extractor from file"""
        try:
            # Force reload the module to pick up safe_mode=False fix
            import sys
            import importlib
            if 'dl_feature_extractor' in sys.modules:
                importlib.reload(sys.modules['dl_feature_extractor'])
            
            # Re-initialize extractor
            self.dl_extractor = DLFeatureExtractor(sequence_length=180, encoding_dim=8)
            success = self.dl_extractor.load_model(filepath)
            if success:
                print("✅ MLFeatureEngineer: Loaded DL model successfully")
                return True
        except Exception as e:
            print(f"❌ MLFeatureEngineer: Failed to load DL model: {e}")
        return False

    def create_base_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create base features using existing indicator system
        
        Args:
            data: Raw OHLC price data
            
        Returns:
            DataFrame with all technical indicators
        """
        logger.info("Creating base features using existing indicators...")
        
        # Prepare data format for existing indicator system
        data_for_indicators = data.copy()
        if 'date' not in data_for_indicators.columns:
            data_for_indicators.reset_index(inplace=True)
            if 'Date' in data_for_indicators.columns:
                data_for_indicators.rename(columns={'Date': 'date'}, inplace=True)
            else:
                data_for_indicators.rename(columns={data_for_indicators.columns[0]: 'date'}, inplace=True)
        
        # Use existing indicator system - no changes needed!
        enriched_data = get_all_indicators(data_for_indicators)
        
        logger.info(f"Generated {len(enriched_data.columns)} base indicators")
        return enriched_data
    
    def create_ml_specific_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add ML-specific features that complement existing indicators
        
        Args:
            data: DataFrame with OHLC and existing indicators
            
        Returns:
            DataFrame with additional ML features
        """
        logger.info("Adding ML-specific features...")
        
        ml_data = data.copy()
        
        # Price-based features
        ml_data['price_vs_sma20'] = ml_data['close'] / ml_data.get('SMA_20', ml_data['close'])
        ml_data['price_vs_sma50'] = ml_data['close'] / ml_data.get('SMA_50', ml_data['close'])
        ml_data['price_vs_ema20'] = ml_data['close'] / ml_data.get('EMA_20', ml_data['close'])
        
        # Volatility features
        ml_data['rolling_volatility_5'] = ml_data['close'].pct_change().rolling(5).std() * np.sqrt(252)
        ml_data['rolling_volatility_20'] = ml_data['close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        # Price momentum features
        ml_data['price_momentum_3'] = ml_data['close'].pct_change(3)
        ml_data['price_momentum_7'] = ml_data['close'].pct_change(7) 
        ml_data['price_momentum_14'] = ml_data['close'].pct_change(14)
        
        # Volume-price relationship
        if 'volume' in ml_data.columns:
            ml_data['volume_sma_ratio'] = ml_data['volume'] / ml_data['volume'].rolling(20).mean()
            ml_data['price_volume_trend'] = (ml_data['close'].pct_change() * ml_data['volume']).rolling(5).mean()
        
        # Cross-indicator features (if available)
        if 'RSI_14' in ml_data.columns and 'RSI_21' in ml_data.columns:
            ml_data['rsi_divergence'] = ml_data['RSI_14'] - ml_data['RSI_21']
        
        if 'MACD_12_26_9' in ml_data.columns and 'MACD_signal_12_26_9' in ml_data.columns:
            ml_data['macd_signal_diff'] = ml_data['MACD_12_26_9'] - ml_data['MACD_signal_12_26_9']
        
        # Bollinger Band position (if available)
        if all(col in ml_data.columns for col in ['BBL_20_2.0', 'BBU_20_2.0']):
            bb_width = ml_data['BBU_20_2.0'] - ml_data['BBL_20_2.0']
            ml_data['bb_position'] = (ml_data['close'] - ml_data['BBL_20_2.0']) / bb_width
            ml_data['bb_width_normalized'] = bb_width / ml_data['close']
        
        # Time-based features
        ml_data['day_of_week'] = ml_data.index.dayofweek
        ml_data['month'] = ml_data.index.month
        ml_data['is_month_end'] = (ml_data.index.day > 25).astype(int)
        
        logger.info(f"Added {len(ml_data.columns) - len(data.columns)} ML-specific features")
        return ml_data
    
    def create_dl_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Deep Learning embeddings using LSTM Autoencoder
        """
        if not DL_AVAILABLE:
            return data
            
        try:
            # If we don't have an extractor yet, create one and train it
            if self.dl_extractor is None:
                # Updated to 180 to support Multi-Scale CNN (30, 60, 90, 180 days)
                self.dl_extractor = DLFeatureExtractor(sequence_length=180, encoding_dim=8)
                # Train and generate
                embeddings = self.dl_extractor.generate_embeddings(data, epochs=15, train=True)
            else:
                # Use existing (loaded) extractor in inference mode
                embeddings = self.dl_extractor.generate_embeddings(data, epochs=15, train=False)
            
            # Join embeddings
            result = pd.concat([data, embeddings], axis=1)
            return result
        except Exception as e:
            import traceback
            logger.warning(f"DL Feature Generation Failed: {e}")
            logger.debug("Full traceback:")
            logger.debug(traceback.format_exc())
            return data

    def create_lagged_features(self, data: pd.DataFrame, 
                             columns: List[str] = None,
                             lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """
        Create lagged versions of important indicators
        
        Args:
            data: DataFrame with indicators
            columns: Specific columns to lag (if None, use top indicators)
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        logger.info(f"Creating lagged features for {lags} periods...")
        
        lagged_data = data.copy()
        
        # Default important columns if not specified
        if columns is None:
            important_cols = []
            # Look for key indicators that exist
            key_indicators = ['RSI_14', 'MACD_12_26_9', 'WILLR_14', 'close', 'volume']
            for col in key_indicators:
                if col in data.columns:
                    important_cols.append(col)
            columns = important_cols[:10]  # Limit to top 10 to avoid explosion
        
        # Create lagged features
        lag_count = 0
        for col in columns:
            if col in data.columns:
                for lag in lags:
                    lag_name = f"{col}_lag_{lag}"
                    lagged_data[lag_name] = data[col].shift(lag)
                    lag_count += 1
        
        logger.info(f"Created {lag_count} lagged features")
        return lagged_data
    
    def create_rolling_features(self, data: pd.DataFrame,
                              columns: List[str] = None,
                              windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create rolling statistics for key indicators
        
        Args:
            data: DataFrame with indicators
            columns: Columns to create rolling features for
            windows: Rolling window sizes
            
        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Creating rolling statistics for windows {windows}...")
        
        rolling_data = data.copy()
        
        # Default columns if not specified
        if columns is None:
            columns = []
            key_indicators = ['RSI_14', 'MACD_12_26_9', 'close']
            for col in key_indicators:
                if col in data.columns:
                    columns.append(col)
        
        rolling_count = 0
        for col in columns:
            if col in data.columns:
                for window in windows:
                    # Rolling mean
                    rolling_data[f"{col}_rolling_mean_{window}"] = data[col].rolling(window).mean()
                    # Rolling std
                    rolling_data[f"{col}_rolling_std_{window}"] = data[col].rolling(window).std()
                    # Rolling min/max for normalized position
                    rolling_min = data[col].rolling(window).min()
                    rolling_max = data[col].rolling(window).max()
                    rolling_data[f"{col}_rolling_position_{window}"] = (
                        (data[col] - rolling_min) / (rolling_max - rolling_min + 1e-8)
                    )
                    rolling_count += 3
        
        logger.info(f"Created {rolling_count} rolling features")
        return rolling_data
    
    def select_features(self, data: pd.DataFrame, 
                       correlation_threshold: float = 0.90,  # More aggressive
                       remove_low_variance: bool = True) -> pd.DataFrame:
        """
        AGGRESSIVE feature selection to remove constant, identical, and highly correlated features
        
        Args:
            data: DataFrame with all features
            correlation_threshold: Remove features with correlation above this
            remove_low_variance: Whether to remove low-variance features
            
        Returns:
            DataFrame with selected features
        """
        logger.info("Aggressive feature selection...")
        
        # Start with numeric columns only
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_data = data[numeric_cols].copy()
        initial_count = len(feature_data.columns)
        
        logger.info(f"Starting with {initial_count} numeric features")
        
        # Step 1: Remove columns with too many NaN values
        nan_threshold = 0.3  # More strict - remove if >30% NaN
        valid_cols = []
        for col in feature_data.columns:
            nan_pct = feature_data[col].isnull().sum() / len(feature_data)
            if nan_pct <= nan_threshold:
                valid_cols.append(col)
        
        feature_data = feature_data[valid_cols]
        removed_nan = initial_count - len(valid_cols)
        logger.info(f"Removed {removed_nan} features with >{nan_threshold*100}% missing data")
        
        # Step 2: Handle indicator warm-up period and remove truly problematic features
        if remove_low_variance:
            constant_features = []
            low_variance_features = []
            
            # Skip first 30 rows for variance calculation (indicator warm-up period)
            warmup_period = min(30, len(feature_data) // 4)  # At most 25% of data
            
            for col in feature_data.columns:
                values = feature_data[col].dropna()
                if len(values) == 0:
                    constant_features.append(col)
                    continue
                
                # Check values after warm-up period
                if len(values) > warmup_period:
                    values_after_warmup = values.iloc[warmup_period:]
                else:
                    values_after_warmup = values
                    
                # Check for truly constant features (after warm-up)
                if values_after_warmup.nunique() <= 1:
                    constant_features.append(col)
                    continue
                
                # Check for very low variance (but be less aggressive now)
                std_val = values_after_warmup.std()
                unique_pct = values_after_warmup.nunique() / len(values_after_warmup)
                
                # Only remove if truly problematic (constant after warm-up)
                if std_val < 1e-10 or unique_pct < 0.05:  # Much more lenient
                    low_variance_features.append(col)
            
            features_to_remove = set(constant_features + low_variance_features)
            feature_data = feature_data.drop(columns=features_to_remove)
            
            logger.info(f"Removed {len(constant_features)} truly constant features")
            logger.info(f"Removed {len(low_variance_features)} problematic low-variance features")
            logger.info(f"Features kept: {feature_data.shape[1]} after intelligent filtering")
        
        # Step 3: Remove identical columns (same values)
        if len(feature_data.columns) > 1:
            filled_data = feature_data.fillna(feature_data.median())
            
            # Find groups of identical columns
            unique_cols = []
            removed_identical = []
            
            for col in filled_data.columns:
                if col not in removed_identical:
                    # Check if this column is identical to any already kept column
                    is_duplicate = False
                    for kept_col in unique_cols:
                        if filled_data[col].equals(filled_data[kept_col]):
                            removed_identical.append(col)
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        unique_cols.append(col)
            
            feature_data = feature_data[unique_cols]
            logger.info(f"Removed {len(removed_identical)} identical features")
        
        # Step 4: Remove highly correlated features (more aggressive)
        if len(feature_data.columns) > 1:
            filled_data = feature_data.fillna(feature_data.median())
            correlation_matrix = filled_data.corr().abs()
            self.correlation_matrix = correlation_matrix
            
            # Use a more sophisticated correlation removal
            # Remove features iteratively, keeping those with highest variance
            to_remove = set()
            
            for i in range(len(correlation_matrix.columns)):
                col_i = correlation_matrix.columns[i]
                if col_i in to_remove:
                    continue
                    
                for j in range(i + 1, len(correlation_matrix.columns)):
                    col_j = correlation_matrix.columns[j]
                    if col_j in to_remove:
                        continue
                    
                    # If correlation is high, remove the one with lower variance
                    if correlation_matrix.iloc[i, j] > correlation_threshold:
                        var_i = filled_data[col_i].var()
                        var_j = filled_data[col_j].var()
                        
                        # Remove the one with lower variance
                        if var_i > var_j:
                            to_remove.add(col_j)
                        else:
                            to_remove.add(col_i)
            
            feature_data = feature_data.drop(columns=list(to_remove))
            logger.info(f"Removed {len(to_remove)} highly correlated features (>{correlation_threshold})")
        
        # Step 5: Final sanity check - ensure meaningful variance spread
        if len(feature_data.columns) > 0:
            filled_data = feature_data.fillna(feature_data.median())
            
            # Remove features where all values are too similar (coefficient of variation < 0.01)
            final_cols = []
            for col in feature_data.columns:
                mean_val = filled_data[col].mean()
                std_val = filled_data[col].std()
                
                # Coefficient of variation (std/mean) should be reasonable
                if mean_val != 0:
                    cv = abs(std_val / mean_val)
                    if cv > 0.001:  # At least 0.1% variation
                        final_cols.append(col)
                elif std_val > 1e-6:  # For zero-mean features, just check std
                    final_cols.append(col)
            
            removed_low_cv = len(feature_data.columns) - len(final_cols)
            feature_data = feature_data[final_cols]
            logger.info(f"Removed {removed_low_cv} features with insufficient variation")
        
        # Store final feature names
        self.feature_names = feature_data.columns.tolist()
        
        final_count = len(self.feature_names)
        total_removed = initial_count - final_count
        
        logger.info("Feature Selection Summary")
        logger.info(f"Initial features: {initial_count}")
        logger.info(f"Final features: {final_count}")
        logger.info(f"Total removed: {total_removed} ({total_removed/initial_count*100:.1f}%)")
        
        return feature_data
    
    def prepare_ml_dataset(self, data: pd.DataFrame,
                          include_lagged: bool = True,
                          include_rolling: bool = True,
                          feature_selection: bool = True,
                          use_dl_features: bool = True) -> pd.DataFrame:
        """
        Complete ML dataset preparation pipeline
        
        Args:
            data: Raw OHLC price data
            include_lagged: Whether to include lagged features
            include_rolling: Whether to include rolling features  
            feature_selection: Whether to perform feature selection
            use_dl_features: Whether to generate Deep Learning embeddings
            
        Returns:
            ML-ready feature DataFrame
        """
        logger.info("Preparing ML dataset")
        
        # Step 1: Create base features using existing indicators
        ml_data = self.create_base_features(data)
        
        # Step 2: Add ML-specific features
        ml_data = self.create_ml_specific_features(ml_data)
        
        # Step 2.5: Add Deep Learning features (Embeddings)
        if use_dl_features:
            ml_data = self.create_dl_features(ml_data)
        
        # Step 3: Add lagged features (optional)
        if include_lagged:
            ml_data = self.create_lagged_features(ml_data)
        
        # Step 4: Add rolling features (optional)
        if include_rolling:
            ml_data = self.create_rolling_features(ml_data)
        
        # Step 5: Feature selection (optional)
        if feature_selection:
            ml_data = self.select_features(ml_data)
        
        logger.info("ML dataset ready")
        logger.info(f"Shape: {ml_data.shape}")
        logger.info(f"Features: {len(self.feature_names) if self.feature_names else len(ml_data.columns)}")
        logger.info(f"Date range: {ml_data.index.min()} to {ml_data.index.max()}")
        
        return ml_data
    
    def get_feature_importance_candidates(self) -> List[str]:
        """Get list of features that are good candidates for importance analysis"""
        if not self.feature_names:
            return []
        
        # Prioritize certain types of features
        priority_patterns = [
            'RSI', 'MACD', 'WILLR', 'SMA', 'EMA', 'BB', 
            'price_vs_', 'momentum', 'volatility', 'volume'
        ]
        
        important_features = []
        for feature in self.feature_names:
            for pattern in priority_patterns:
                if pattern in feature:
                    important_features.append(feature)
                    break
        
        return important_features[:20]  # Top 20 most likely important features

if __name__ == "__main__":
    # Test the feature engineer
    import yfinance as yf
    
    print("🧪 Testing ML Feature Engineer")
    
    # Get test data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="6mo", interval="1d")  
    data.columns = [col.lower() for col in data.columns]
    
    # Test feature engineering
    engineer = MLFeatureEngineer()
    ml_features = engineer.prepare_ml_dataset(data)
    
    print(f"\n📊 Feature Engineering Results:")
    print(f"   Original columns: {len(data.columns)}")
    print(f"   Final ML features: {len(ml_features.columns)}")
    print(f"   Sample features: {ml_features.columns[:10].tolist()}")
    
    # Check for NaN values
    nan_counts = ml_features.isnull().sum()
    print(f"   Features with NaN: {(nan_counts > 0).sum()}")
    print(f"   Max NaN per feature: {nan_counts.max()}")
