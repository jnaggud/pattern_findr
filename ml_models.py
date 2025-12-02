#!/usr/bin/env python3
"""
ML Models Module for Trading Signal Prediction

Handles training, evaluation, and prediction of various machine learning models
for trading signal generation based on technical indicators.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample
from typing import Dict, Tuple, List, Any
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available. Install with: pip install lightgbm")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna not available. Install with: pip install optuna")

class TradingMLModels:
    """
    Machine Learning models for trading signal prediction
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.training_history = {}
        self.best_model = None
        self.best_score = 0.0
        
    def prepare_data(self, features: pd.DataFrame, labels: pd.Series,
                    test_size: float = 0.2,
                    use_time_split: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training with proper time-series splitting
        
        Args:
            features: Feature DataFrame
            labels: Target labels (-1, 0, 1)
            test_size: Fraction for test set
            use_time_split: Use time-aware splitting vs random
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"📊 Preparing data for ML training...")
        
        # Remove rows with NaN in features or labels
        valid_mask = ~(features.isnull().any(axis=1) | labels.isnull())
        clean_features = features[valid_mask]
        clean_labels = labels[valid_mask]
        
        # Clean data
        clean_features = features.dropna()
        clean_labels = labels.loc[clean_features.index]
        
        # Store feature names for future reference/saving
        self.feature_names = list(clean_features.columns)
        
        print(f"   Data shape after cleaning: {clean_features.shape}")
        
        # Split data
        if use_time_split:
            # Time-series split (no shuffling)
            split_idx = int(len(clean_features) * (1 - test_size))
            X_train = clean_features.iloc[:split_idx]
            X_test = clean_features.iloc[split_idx:]
            y_train = clean_labels.iloc[:split_idx]
            y_test = clean_labels.iloc[split_idx:]
            print(f"   Time split: train={len(X_train)}, test={len(X_test)}")
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                clean_features, clean_labels, test_size=test_size, 
                random_state=42, stratify=clean_labels
            )
            print(f"   Random split: train={len(X_train)}, test={len(X_test)}")
        
        # Label distribution
        train_dist = y_train.value_counts().sort_index()
        test_dist = y_test.value_counts().sort_index()
        
        print(f"   Train labels: {dict(train_dist)}")
        print(f"   Test labels: {dict(test_dist)}")
        
        return X_train, X_test, y_train, y_test
    
    def create_random_forest(self, **kwargs) -> RandomForestClassifier:
        """Create Random Forest model with optimized parameters"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'class_weight': 'balanced',  # Handle imbalanced classes
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        return RandomForestClassifier(**default_params)
    
    def create_xgboost(self, **kwargs) -> Any:
        """Create XGBoost model if available"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
        default_params.update(kwargs)
        
        return xgb.XGBClassifier(**default_params)
    
    def create_lightgbm(self, **kwargs) -> Any:
        """Create LightGBM model if available"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
            'class_weight': 'balanced'
        }
        default_params.update(kwargs)
        
        return lgb.LGBMClassifier(**default_params)
    
    def create_svm(self, **kwargs) -> SVC:
        """Create SVM model with optimized parameters"""
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'class_weight': 'balanced',
            'probability': True,  # Enable probability predictions
            'random_state': 42
        }
        default_params.update(kwargs)
        
        return SVC(**default_params)
    
    def balance_training_data(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Upsample minority classes to match majority class count
        """
        print("⚖️  Balancing training data...")
        
        # Combine for resampling
        train_data = X_train.copy()
        train_data['target'] = y_train
        
        # Get class counts
        class_counts = train_data['target'].value_counts()
        majority_class = class_counts.idxmax()
        majority_count = class_counts.max()
        
        print(f"   Original distribution: {dict(class_counts)}")
        
        balanced_dfs = []
        
        for label in class_counts.index:
            class_subset = train_data[train_data['target'] == label]
            
            if label == majority_class:
                balanced_dfs.append(class_subset)
            else:
                # Upsample minority class
                upsampled = resample(
                    class_subset,
                    replace=True,     # Sample with replacement
                    n_samples=majority_count,  # Match majority class
                    random_state=42
                )
                balanced_dfs.append(upsampled)
        
        # Combine and shuffle
        balanced_data = pd.concat(balanced_dfs)
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split back
        y_balanced = balanced_data['target']
        X_balanced = balanced_data.drop('target', axis=1)
        
        print(f"   Balanced distribution: {dict(y_balanced.value_counts())}")
        
        return X_balanced, y_balanced

    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   model_params: Dict = None, use_scaling: bool = True) -> Dict:
        """
        Train a specific model
        
        Args:
            model_name: Name of model ('random_forest', 'xgboost', 'lightgbm', 'svm')
            X_train: Training features
            y_train: Training labels
            model_params: Custom model parameters
            use_scaling: Whether to scale features
            
        Returns:
            Dictionary with training results
        """
        print(f"🎯 Training {model_name} model...")
        
        # Balance classes before training
        X_train_balanced, y_train_balanced = self.balance_training_data(X_train, y_train)
        
        model_params = model_params or {}
        
        # Create model
        if model_name == 'random_forest':
            model = self.create_random_forest(**model_params)
        elif model_name == 'xgboost':
            model = self.create_xgboost(**model_params)
        elif model_name == 'lightgbm':
            model = self.create_lightgbm(**model_params)
        elif model_name == 'svm':
            model = self.create_svm(**model_params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Scale features if requested
        scaler = None
        # Fit scaler on UNBALANCED training data to get true population stats
        # (Using balanced data for scaling can skew mean/std if synthetic points are clustered)
        if use_scaling:
            scaler = StandardScaler()
            scaler.fit(X_train)  # Fit on original distribution
            self.scalers[model_name] = scaler
            
            # Transform balanced data
            X_train_scaled = pd.DataFrame(
                scaler.transform(X_train_balanced),
                columns=X_train_balanced.columns
            )
        else:
            X_train_scaled = X_train_balanced
        
        # Handle label encoding for XGBoost/LightGBM (need 0-indexed labels)
        y_train_encoded = y_train_balanced
        label_encoder = None
        
        if model_name in ['xgboost', 'lightgbm']:
            label_encoder = LabelEncoder()
            y_train_encoded = pd.Series(
                label_encoder.fit_transform(y_train_balanced), 
                name='target'
            )
            print(f"   Encoded labels: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
        
        # Train model
        try:
            model.fit(X_train_scaled, y_train_encoded)
            
            # Cross-validation score (use encoded labels for consistency)
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train_encoded,
                cv=TimeSeriesSplit(n_splits=3),  # Time-aware CV
                scoring='accuracy'
            )
            
            # Store model and label encoder
            self.models[model_name] = model
            if label_encoder is not None:
                self.scalers[f"{model_name}_label_encoder"] = label_encoder
            
            # Calculate training accuracy
            y_train_pred = model.predict(X_train_scaled)
            train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
            
            # Training results
            results = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores,
                'train_accuracy': train_accuracy,
                'feature_count': len(X_train.columns),
                'training_samples': len(X_train),
                'label_encoder': label_encoder
            }
            
            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                results['feature_importance'] = importance_df
            
            self.training_history[model_name] = results
            
            print(f"   ✅ CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Training failed: {str(e)}")
            return {'error': str(e)}
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate trained model on test set
        
        Args:
            model_name: Name of trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_name not in self.models:
            return {'error': f'Model {model_name} not trained yet'}
        
        print(f"📊 Evaluating {model_name} model...")
        
        model = self.models[model_name]
        
        # Scale test features if scaler exists
        X_test_scaled = X_test
        if model_name in self.scalers:
            scaler = self.scalers[model_name]
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        # Predictions
        y_pred_raw = model.predict(X_test_scaled)
        
        # Handle label decoding for XGBoost/LightGBM
        label_encoder = self.scalers.get(f"{model_name}_label_encoder")
        if label_encoder is not None:
            y_pred = label_encoder.inverse_transform(y_pred_raw)
        else:
            y_pred = y_pred_raw
        
        # Probabilities (if available)
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle multi-class metrics
        avg_method = 'weighted'  # Good for imbalanced classes
        precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_proba,
            'test_samples': len(y_test)
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Update best model if this one is better
        if f1 > self.best_score:
            self.best_score = f1
            self.best_model = model_name
            print(f"   🏆 New best model: {model_name}")
        
        return results
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Train and evaluate all available models
        
        Returns:
            Dictionary with results for all models
        """
        print(f"\n🚀 TRAINING ALL AVAILABLE MODELS")
        print("=" * 50)
        
        all_results = {}
        
        # Models to try
        models_to_train = ['random_forest']
        
        if XGBOOST_AVAILABLE:
            models_to_train.append('xgboost')
        
        if LIGHTGBM_AVAILABLE:
            models_to_train.append('lightgbm')
            
        models_to_train.append('svm')  # Add SVM (sklearn always available)
        
        # Train each model
        for model_name in models_to_train:
            try:
                print(f"\n--- {model_name.upper()} ---")
                
                # Train model
                train_results = self.train_model(model_name, X_train, y_train)
                
                if 'error' not in train_results:
                    # Evaluate model
                    eval_results = self.evaluate_model(model_name, X_test, y_test)
                    
                    # Combine results
                    all_results[model_name] = {
                        **train_results,
                        **eval_results
                    }
                else:
                    all_results[model_name] = train_results
                    
            except Exception as e:
                print(f"   ❌ Failed to train {model_name}: {str(e)}")
                all_results[model_name] = {'error': str(e)}
        
        print(f"\n🏆 BEST MODEL: {self.best_model} (F1: {self.best_score:.4f})")
        
        return all_results
    
    def predict_signals(self, features: pd.DataFrame, model_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trading signals from features
        
        Args:
            features: Feature DataFrame
            model_name: Specific model to use (if None, use best model)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        model_name = model_name or self.best_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        # Align features to training set
        if self.feature_names:
            # Add missing columns with 0
            missing_cols = [col for col in self.feature_names if col not in features.columns]
            if missing_cols:
                for col in missing_cols:
                    features[col] = 0
            
            # Drop extra columns and reorder
            features = features[self.feature_names]
        
        model = self.models[model_name]
        
        # Scale features if needed
        features_scaled = features
        if model_name in self.scalers:
            scaler = self.scalers[model_name]
            features_scaled = pd.DataFrame(
                scaler.transform(features),
                columns=features.columns,
                index=features.index
            )
        
        # Predictions
        predictions_raw = model.predict(features_scaled)
        
        # Handle label decoding for XGBoost/LightGBM
        label_encoder = self.scalers.get(f"{model_name}_label_encoder")
        if label_encoder is not None:
            predictions = label_encoder.inverse_transform(predictions_raw)
        else:
            predictions = predictions_raw
        
        # Probabilities
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)
        
        return predictions, probabilities
    
    def optimize_hyperparameters(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                                 n_trials: int = 100, cv_folds: int = 3) -> Dict:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            model_name: Name of model to optimize
            X_train: Training features
            y_train: Training labels
            n_trials: Number of optimization trials
            cv_folds: Number of CV folds for evaluation
            
        Returns:
            Dictionary with best parameters and score
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        
        print(f"🔍 Optimizing {model_name} hyperparameters with {n_trials} trials...")
        
        # Balance training data first
        X_train_balanced, y_train_balanced = self.balance_training_data(X_train, y_train)
        
        # Define objective function for Optuna
        def objective(trial):
            params = self._suggest_parameters(trial, model_name)
            
            try:
                # Create model with trial parameters
                if model_name == 'random_forest':
                    model = self.create_random_forest(**params)
                elif model_name == 'xgboost':
                    model = self.create_xgboost(**params)
                elif model_name == 'lightgbm':
                    model = self.create_lightgbm(**params)
                elif model_name == 'svm':
                    model = self.create_svm(**params)
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                # Prepare data (scaling if needed)
                X_scaled = X_train_balanced
                y_encoded = y_train_balanced
                
                if model_name in ['xgboost', 'lightgbm', 'svm']:
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = pd.DataFrame(
                        scaler.fit_transform(X_train_balanced),
                        columns=X_train_balanced.columns
                    )
                
                if model_name in ['xgboost', 'lightgbm']:
                    # Encode labels
                    label_encoder = LabelEncoder()
                    y_encoded = pd.Series(label_encoder.fit_transform(y_train_balanced))
                
                # Cross-validation with time series split
                cv_scores = cross_val_score(
                    model, X_scaled, y_encoded,
                    cv=TimeSeriesSplit(n_splits=cv_folds),
                    scoring='f1_weighted',
                    n_jobs=-1
                )
                
                return cv_scores.mean()
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return 0.0
        
        # Create and run study
        study = optuna.create_study(direction='maximize', 
                                   sampler=optuna.samplers.TPESampler(seed=42))
        
        # Suppress optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Results
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"✅ Optimization complete!")
        print(f"   Best Score: {best_score:.4f}")
        print(f"   Best Params: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study,
            'n_trials': n_trials
        }
    
    def _suggest_parameters(self, trial, model_name: str) -> Dict:
        """Suggest hyperparameters for Optuna trial"""
        
        if model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            }
        
        elif model_name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            }
        
        elif model_name == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'num_leaves': trial.suggest_int('num_leaves', 10, 200),
            }
        
        elif model_name == 'svm':
            return {
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
            }
        
        else:
            return {}
    
    def train_with_optimization(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                               n_trials: int = 100) -> Dict:
        """
        Train model with hyperparameter optimization
        
        Args:
            model_name: Name of model to train
            X_train: Training features
            y_train: Training labels
            n_trials: Number of optimization trials
            
        Returns:
            Training results with optimized parameters
        """
        print(f"🚀 Training {model_name} with hyperparameter optimization...")
        
        # Step 1: Optimize hyperparameters
        optimization_result = self.optimize_hyperparameters(
            model_name, X_train, y_train, n_trials=n_trials
        )
        
        best_params = optimization_result['best_params']
        
        # Step 2: Train final model with best parameters
        final_result = self.train_model(model_name, X_train, y_train, 
                                       model_params=best_params, use_scaling=True)
        
        # Add optimization info to results
        final_result['optimization'] = {
            'best_params': best_params,
            'best_cv_score': optimization_result['best_score'],
            'n_trials': n_trials,
            'optimized': True
        }
        
        return final_result
    
    def save_model_version(self, model_name: str, directory: str, metadata: Dict = None) -> str:
        """
        Save a specific model version with metadata
        
        Args:
            model_name: Name of the model to save
            directory: Directory to save to
            metadata: Additional metadata (metrics, date, etc.)
        
        Returns:
            Path to saved file
        """
        import os
        import json
        from datetime import datetime
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.joblib"
        filepath = os.path.join(directory, filename)
        
        # Prepare data package
        save_data = {
            'model_name': model_name,
            'model': self.models[model_name],
            'scaler': self.scalers.get(model_name),
            'label_encoder': self.scalers.get(f"{model_name}_label_encoder"),
            'feature_names': self.feature_names,
            'metadata': metadata or {},
            'timestamp': timestamp,
            'python_version': '3.x',
            'library_versions': {'sklearn': '1.x'} # Placeholder
        }
        
        joblib.dump(save_data, filepath)
        print(f"💾 Model version saved to {filepath}")
        return filepath

    def load_model_version(self, filepath: str) -> Dict:
        """
        Load a specific model version
        
        Returns:
            Dictionary containing model and metadata
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        data = joblib.load(filepath)
        
        # Restore to current instance if needed
        model_name = data['model_name']
        self.models[model_name] = data['model']
        
        if data.get('scaler'):
            self.scalers[model_name] = data['scaler']
            
        if data.get('label_encoder'):
            self.scalers[f"{model_name}_label_encoder"] = data['label_encoder']
            
        # Only update feature names if empty (to avoid overwriting current session if loading for reference)
        if not self.feature_names:
            self.feature_names = data['feature_names']
            
        # Removed noisy print statement
        return data

    def save_models(self, filepath: str):
        """Save all trained models to file"""
        save_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'best_model': self.best_model,
            'best_score': self.best_score
        }
        joblib.dump(save_data, filepath)
        # Removed noisy print statement
    
    def load_models(self, filepath: str):
        """Load trained models from file"""
        save_data = joblib.load(filepath)
        self.models = save_data['models']
        self.scalers = save_data['scalers'] 
        self.feature_names = save_data['feature_names']
        self.training_history = save_data['training_history']
        self.best_model = save_data['best_model']
        self.best_score = save_data['best_score']
        # Removed noisy print statement

if __name__ == "__main__":
    # Test the ML models
    from ml_feature_engineer import MLFeatureEngineer
    from peak_valley_detector import PeakValleyDetector
    import yfinance as yf
    
    print("🧪 Testing ML Models")
    
    # Get test data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="1y", interval="1d")
    data.columns = [col.lower() for col in data.columns]
    
    # Create features and labels
    engineer = MLFeatureEngineer()
    features = engineer.prepare_ml_dataset(data)
    
    detector = PeakValleyDetector()
    _, labels = detector.create_labeled_dataset(data, method="rolling_window")
    
    # Test models
    ml_models = TradingMLModels()
    X_train, X_test, y_train, y_test = ml_models.prepare_data(features, labels)
    
    # Train all models
    results = ml_models.train_all_models(X_train, y_train, X_test, y_test)
    
    print(f"\n📊 Model Comparison:")
    for model_name, result in results.items():
        if 'error' not in result:
            print(f"   {model_name}: F1={result['f1_score']:.4f}, Acc={result['accuracy']:.4f}")
        else:
            print(f"   {model_name}: ERROR - {result['error']}")
