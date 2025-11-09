import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
import pickle
from datetime import datetime
import glob
from sklearn.model_selection import train_test_split
from dl_pattern_detector import load_and_preprocess_image, use_synthetic_data_for_training, build_cnn_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PatternDetectionModel:
    """
    Deep learning model for detecting chart patterns in financial data.
    """
    
    def __init__(self, pattern_name, model_type='cnn', input_shape=(224, 224, 3)):
        self.pattern_name = pattern_name
        self.model_type = model_type
        self.input_shape = input_shape
        self.model = None
        self.model_dir = f"models/{pattern_name}"
        self.is_trained = False
        
        # Create models directory
        os.makedirs(self.model_dir, exist_ok=True)
        
    def build_model(self):
        """Build the neural network model."""
        if self.model_type == 'cnn':
            self.model = self._build_cnn()
        elif self.model_type == 'transfer':
            self.model = self._build_transfer_learning_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        logging.info(f"Built {self.model_type} model for {self.pattern_name}")
        
    def _build_cnn(self):
        """Build a custom CNN model."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def _build_transfer_learning_model(self):
        """Build a transfer learning model using MobileNetV2."""
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def prepare_data(self, image_paths, labels, test_size=0.2, validation_size=0.2):
        """
        Prepare training data by loading and preprocessing images.
        
        Args:
            image_paths (list): List of image file paths
            labels (list): List of corresponding labels
            test_size (float): Fraction of data for testing
            validation_size (float): Fraction of training data for validation
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logging.info(f"Loading {len(image_paths)} images for {self.pattern_name}...")
        
        # Load and preprocess images
        X = []
        y = []
        
        for img_path, label in zip(image_paths, labels):
            img = load_and_preprocess_image(img_path, target_size=self.input_shape[:2])
            if img is not None:
                X.append(img)
                y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        logging.info(f"Successfully loaded {len(X)} images")
        logging.info(f"Positive examples: {np.sum(y)}, Negative examples: {len(y) - np.sum(y)}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size, random_state=42, stratify=y_temp
        )
        
        logging.info(f"Training set: {len(X_train)} samples")
        logging.info(f"Validation set: {len(X_val)} samples")
        logging.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if self.model is None:
            self.build_model()
        
        # Create data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,  # Don't flip financial charts
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                filepath=f"{self.model_dir}/best_model.h5",
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        logging.info(f"Starting training for {self.pattern_name}...")
        
        # Train the model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.is_trained = True
        
        # Save training history
        with open(f"{self.model_dir}/training_history.pkl", 'wb') as f:
            pickle.dump(history.history, f)
        
        logging.info(f"Training completed for {self.pattern_name}")
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or trained")
        
        # Evaluate the model
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate additional metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        metrics = {
            'test_loss': results[0],
            'test_accuracy': results[1],
            'test_precision': results[2],
            'test_recall': results[3],
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logging.info(f"Test accuracy for {self.pattern_name}: {metrics['test_accuracy']:.4f}")
        
        return metrics
    
    def predict(self, image_path_or_array, threshold=0.5):
        """
        Predict whether a pattern is present in an image.
        
        Args:
            image_path_or_array: Path to image file or numpy array
            threshold: Probability threshold for positive prediction
            
        Returns:
            tuple: (prediction, probability)
        """
        if self.model is None:
            self.load_model()
        
        # Load and preprocess image
        if isinstance(image_path_or_array, str):
            img = load_and_preprocess_image(image_path_or_array, target_size=self.input_shape[:2])
        else:
            img = image_path_or_array
        
        if img is None:
            return False, 0.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        
        # Make prediction
        prob = self.model.predict(img_batch, verbose=0)[0][0]
        prediction = prob > threshold
        
        return prediction, prob
    
    def save_model(self):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_path = f"{self.model_dir}/model.h5"
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'pattern_name': self.pattern_name,
            'model_type': self.model_type,
            'input_shape': self.input_shape,
            'is_trained': self.is_trained,
            'created_at': datetime.now().isoformat()
        }
        
        with open(f"{self.model_dir}/metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logging.info(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load a trained model."""
        model_path = f"{self.model_dir}/model.h5"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained model found at {model_path}")
        
        self.model = tf.keras.models.load_model(model_path)
        self.is_trained = True
        
        # Load metadata if available
        metadata_path = f"{self.model_dir}/metadata.pkl"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                logging.info(f"Loaded model for {metadata['pattern_name']} created at {metadata['created_at']}")
        
        logging.info(f"Model loaded from {model_path}")


class PatternDetectionEnsemble:
    """
    Ensemble of multiple pattern detection models.
    """
    
    def __init__(self, patterns):
        self.patterns = patterns
        self.models = {}
        
        for pattern in patterns:
            self.models[pattern] = PatternDetectionModel(pattern)
    
    def train_all_models(self, num_samples=200, epochs=30):
        """
        Train all models in the ensemble.
        
        Args:
            num_samples: Number of training samples per pattern
            epochs: Number of training epochs
        """
        logging.info(f"Training ensemble for {len(self.patterns)} patterns...")
        
        for pattern in self.patterns:
            logging.info(f"\n{'='*50}")
            logging.info(f"Training model for: {pattern}")
            logging.info(f"{'='*50}")
            
            # Generate or load training data
            X_paths, y_labels = use_synthetic_data_for_training(pattern, num_samples)
            
            # Prepare data
            model = self.models[pattern]
            X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(X_paths, y_labels)
            
            # Train model
            history = model.train(X_train, y_train, X_val, y_val, epochs=epochs)
            
            # Evaluate model
            metrics = model.evaluate(X_test, y_test)
            
            # Save model
            model.save_model()
            
            logging.info(f"Completed training for {pattern}")
            logging.info(f"Final test accuracy: {metrics['test_accuracy']:.4f}")
    
    def predict_all_patterns(self, image_path_or_array, threshold=0.5):
        """
        Predict all patterns for a given image.
        
        Args:
            image_path_or_array: Path to image or numpy array
            threshold: Probability threshold
            
        Returns:
            dict: Pattern predictions and probabilities
        """
        results = {}
        
        for pattern, model in self.models.items():
            try:
                prediction, probability = model.predict(image_path_or_array, threshold)
                results[pattern] = {
                    'prediction': prediction,
                    'probability': probability
                }
            except Exception as e:
                logging.error(f"Error predicting {pattern}: {e}")
                results[pattern] = {
                    'prediction': False,
                    'probability': 0.0
                }
        
        return results
    
    def load_all_models(self):
        """Load all trained models."""
        for pattern, model in self.models.items():
            try:
                model.load_model()
                logging.info(f"Loaded model for {pattern}")
            except FileNotFoundError:
                logging.warning(f"No trained model found for {pattern}")


# Main patterns to detect
CHART_PATTERNS = [
    # Classic reversal patterns
    'head_and_shoulders',
    'inverse_head_and_shoulders',
    'cup_and_handle',
    'double_top',
    'double_bottom',
    'triple_top',
    'triple_bottom',
    'rounding_top',
    'rounding_bottom',
    
    # Continuation patterns  
    'triangle',
    'ascending_triangle',
    'descending_triangle',
    'symmetrical_triangle',
    'flag',
    'pennant',
    'wedge_rising',
    'wedge_falling',
    'rectangle',
    
    # Gap patterns
    'breakaway_gap',
    'runaway_gap',
    'exhaustion_gap',
    
    # Candlestick patterns
    'hammer',
    'doji',
    'shooting_star',
    'engulfing_bullish',
    'engulfing_bearish',
    'harami_bullish',
    'harami_bearish',
    'morning_star',
    'evening_star',
    
    # Volume-price patterns
    'volume_spike_breakout',
    'volume_divergence',
    'accumulation_distribution'
]

def get_pattern_ensemble():
    """Get the pattern detection ensemble."""
    return PatternDetectionEnsemble(CHART_PATTERNS)

if __name__ == "__main__":
    # Create and train ensemble
    ensemble = get_pattern_ensemble()
    
    # Train all models (this will take a while!)
    ensemble.train_all_models(num_samples=200, epochs=30)
    
    logging.info("All models trained successfully!")
