#!/usr/bin/env python3
"""
Deep Learning Feature Extractor

Uses LSTM Autoencoders to generate compressed embeddings (features) 
from sequential price/indicator data. These embeddings capture 
complex temporal patterns that classical models (RF, XGB) might miss.
"""

import os
import logging
import json

# FORCE CPU to avoid "stream cannot wait for itself" Metal error on Mac
# This must be set BEFORE importing tensorflow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
import random

# Set Global Seeds for Reproducibility
# This is CRITICAL to ensure 'DL_pred_0' in Training == 'DL_pred_0' in Production
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout, Conv1D, MaxPooling1D, Flatten, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

class DLFeatureExtractor:
    def __init__(self, sequence_length=10, encoding_dim=8):
        """
        Args:
            sequence_length (int): How many past days to look at (window size)
            encoding_dim (int): Number of features to generate (latent space size)
        """
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.scaler = MinMaxScaler()
        self.model = None
        self.encoder = None
        self.feature_cols = None
        
    def _prepare_sequences(self, data):
        """
        Convert 2D DataFrame/Array into 3D sequences AND generate MULTI-TASK targets
        Target 1: Regression (Next Day Return)
        Target 2: Classification (Direction: 1=Up, 0=Down)
        """
        X, y_reg, y_class = [], [], []
        
        # Ensure we are working with numpy array
        if hasattr(data, 'values'):
            data_values = data.values
        else:
            data_values = data
            
        # Loop
        total_len = len(data_values)
        
        for i in range(total_len):
            if i < self.sequence_length:
                # Pad with zeros (Not enough history)
                X.append(np.zeros((self.sequence_length, data_values.shape[1])))
                y_reg.append(0)
                y_class.append(0)
            else:
                # Valid Sequence: t-30 to t
                seq = data_values[i-self.sequence_length:i, :]
                X.append(seq)
                
                # Targets (Only if future exists)
                if i < total_len - 1:
                    # Target 1: Regression (Next Step Return)
                    next_return = np.mean(data_values[i+1])
                    y_reg.append(next_return)
                    
                    # Target 2: Classification (Direction)
                    y_class.append(1 if next_return > 0 else 0)
                else:
                    # Last row (Today) -> No target for tomorrow yet
                    # Fill with 0 (doesn't matter for inference)
                    y_reg.append(0)
                    y_class.append(0)
                
        return np.array(X), np.array(y_reg), np.array(y_class)

    def build_forecasting_model(self, input_dim):
        """
        Builds a Multi-Scale CNN + LSTM Model
        Inputs:
            - Window 30 (Short Term)
            - Window 60 (Medium Term)
            - Window 90 (Long Term)
            - Window 180 (Trend)
        
        The model slices the input sequence (max length) into these windows 
        and processes them with parallel CNNs.
        """
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Concatenate, GlobalAveragePooling1D
        
        # Input is the maximum sequence length (e.g. 180)
        inputs = Input(shape=(self.sequence_length, input_dim))
        
        # --- Branch 1: Short Term (Last 30 days) ---
        # Slice: Take the last 30 steps
        # Lambda layer to slice: inputs[:, -30:, :]
        slice_30 = tf.keras.layers.Lambda(lambda x: x[:, -30:, :])(inputs)
        x30 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(slice_30)
        x30 = Dropout(0.2)(x30)
        x30 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x30)
        x30 = GlobalAveragePooling1D()(x30) # Shape: (batch, 32)
        
        # --- Branch 2: Medium Term (Last 60 days) ---
        # Only create if sequence length allows
        branches = [x30]
        
        if self.sequence_length >= 60:
            slice_60 = tf.keras.layers.Lambda(lambda x: x[:, -60:, :])(inputs)
            x60 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(slice_60)
            x60 = MaxPooling1D(pool_size=2)(x60)
            x60 = Dropout(0.2)(x60)
            x60 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(x60)
            x60 = GlobalAveragePooling1D()(x60)
            branches.append(x60)
            
        if self.sequence_length >= 90:
            slice_90 = tf.keras.layers.Lambda(lambda x: x[:, -90:, :])(inputs)
            x90 = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(slice_90)
            x90 = MaxPooling1D(pool_size=2)(x90)
            x90 = Dropout(0.2)(x90)
            x90 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(x90)
            x90 = GlobalAveragePooling1D()(x90)
            branches.append(x90)

        if self.sequence_length >= 180:
            slice_180 = tf.keras.layers.Lambda(lambda x: x[:, -180:, :])(inputs)
            x180 = Conv1D(filters=32, kernel_size=9, activation='relu', padding='same',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))(slice_180)
            x180 = MaxPooling1D(pool_size=4)(x180) # More aggressive pooling
            x180 = Dropout(0.2)(x180)
            x180 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))(x180)
            x180 = GlobalAveragePooling1D()(x180)
            branches.append(x180)

        # --- Concatenate All Scales ---
        if len(branches) > 1:
            merged = Concatenate()(branches)
        else:
            merged = branches[0]

        # Dense Fusion with more regularization
        merged = Dense(64, activation='relu',
                       kernel_regularizer=tf.keras.regularizers.l2(0.01))(merged)
        merged = Dropout(0.4)(merged)  # Increased from 0.3
        
        # Bottleneck (The Feature Embedding)
        # Use 'linear' activation to preserve full embedding range (not clipped by ReLU)
        bottleneck = Dense(self.encoding_dim, activation='linear', name='embedding_layer',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(merged)
        
        # Head 1: Regression (Price Change)
        out_reg = Dense(1, activation='linear', name='regression_output')(bottleneck)
        
        # Head 2: Classification (Direction)
        out_class = Dense(1, activation='sigmoid', name='class_output')(bottleneck)
        
        # Full Model
        model = Model(inputs, [out_reg, out_class])
        
        model.compile(
            optimizer='adam', 
            loss={'regression_output': 'mse', 'class_output': 'binary_crossentropy'},
            loss_weights={'regression_output': 1.0, 'class_output': 0.5}
        )
        
        # Encoder
        encoder = Model(inputs, bottleneck)
        
        return model, encoder

    def generate_embeddings(self, df, feature_cols=None, epochs=15, verbose=0, train=True):
        """
        Main method to train multi-task model and generate predictive features
        
        Args:
            train (bool): If True, trains the model. If False, uses existing encoder (Inference Mode).
        """
        
        # 1. Select Features
        if feature_cols is None:
            if self.feature_cols is not None:
                feature_cols = self.feature_cols
            else:
                feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Persist the feature list so training/inference stay compatible
        if train:
            self.feature_cols = list(feature_cols)
        elif self.feature_cols is None:
            self.feature_cols = list(feature_cols)

        # Reindex to ensure stable dimensionality across runs
        data = df.reindex(columns=self.feature_cols, fill_value=0).copy()
            
        # 2. STATIONARITY: Use Returns
        data_pct = data.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        
        # 3. Scale Data
        # Note: In strict production, we should load the scaler too. 
        # For now, fitting on 5y history is acceptable approximation.
        scaled_data = self.scaler.fit_transform(data_pct)
        
        # 4. Create Sequences
        # We only need X for inference
        if train:
            X, y_reg, y_class = self._prepare_sequences(scaled_data)
        else:
            # In inference, we might not have targets (future), so just generate X
            # Reuse the same function but ignore Y
            X, _, _ = self._prepare_sequences(scaled_data)
        
        # 5. Build & Train Model (Only if training)
        input_dim = X.shape[2]
        
        if train:
            logging.getLogger(__name__).info(f"Training Multi-Task DL Model on {len(df)} samples...")
            self.model, self.encoder = self.build_forecasting_model(input_dim)
            
            es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
            
            self.model.fit(
                X, {'regression_output': y_reg, 'class_output': y_class},
                epochs=epochs,
                batch_size=32,
                shuffle=True,
                callbacks=[es],
                verbose=verbose
            )
        else:
            logging.getLogger(__name__).info(f"Generating DL Features (Inference Mode) on {len(df)} samples...")
            if self.encoder is None:
                raise ValueError("Model not trained or loaded! Call load_model() first.")
        
        # 6. Generate Embeddings
        embeddings = self.encoder.predict(X, verbose=verbose)
        
        # 7. Format Output
        embed_df = pd.DataFrame(
            embeddings,
            columns=[f"DL_pred_{i}" for i in range(self.encoding_dim)],
            index=df.index
        )
        
        # Set first 'sequence_length' rows to 0
        embed_df.iloc[:self.sequence_length] = 0
        
        logging.getLogger(__name__).info(f"Generated {self.encoding_dim} Predictive DL features")
        return embed_df

    def save_model(self, filepath):
        """Save the trained encoder model to disk"""
        if self.encoder:
            self.encoder.save(filepath)
            try:
                meta_path = filepath.replace('.h5', '_feature_cols.json')
                with open(meta_path, 'w') as f:
                    json.dump({'feature_cols': self.feature_cols or []}, f)
            except Exception:
                pass
            logging.getLogger(__name__).info(f"DL Extractor saved to {filepath}")
        else:
            logging.getLogger(__name__).warning("No encoder to save!")

    def load_model(self, filepath):
        """Load a trained encoder model from disk"""
        try:
            # Allow loading models with lambda functions (safe for our own models)
            self.encoder = tf.keras.models.load_model(filepath, safe_mode=False)
            self.model = None # We don't need the full training model for inference
            try:
                meta_path = filepath.replace('.h5', '_feature_cols.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f) or {}
                    cols = meta.get('feature_cols')
                    if isinstance(cols, list) and cols:
                        self.feature_cols = cols
            except Exception:
                pass
            logging.getLogger(__name__).info(f"DL Extractor loaded from {filepath}")
            return True
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to load DL Extractor: {e}")
            return False

if __name__ == "__main__":
    # Simple Test
    import yfinance as yf
    
    print("Testing DL Feature Extractor...")
    df = yf.Ticker("SPY").history(period="5y") # 5 years
    
    extractor = DLFeatureExtractor(sequence_length=30, encoding_dim=8)
    embeddings = extractor.generate_embeddings(df, epochs=5, verbose=1)
    
    print(embeddings.tail())
    print("Shape:", embeddings.shape)
