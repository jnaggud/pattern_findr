#!/usr/bin/env python3
"""
Debug why different ML models are producing identical results
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
import pandas as pd
import numpy as np
from ml_feature_engineer import MLFeatureEngineer
from peak_valley_detector import PeakValleyDetector
from ml_models import TradingMLModels

def debug_identical_models():
    print("🔍 DEBUGGING IDENTICAL MODEL PERFORMANCE")
    print("=" * 60)
    
    # Get test data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="1y", interval="1d")
    data.columns = [col.lower() for col in data.columns]
    
    print(f"📊 Raw Data: {data.shape}")
    
    # Create features and labels
    engineer = MLFeatureEngineer()
    features = engineer.prepare_ml_dataset(data, 
                                         include_lagged=False, 
                                         include_rolling=False, 
                                         feature_selection=True)
    
    detector = PeakValleyDetector()
    _, labels = detector.create_labeled_dataset(data, method="scipy_peaks")
    
    print(f"📊 Features: {features.shape}")
    print(f"📊 Labels: {labels.value_counts().to_dict()}")
    
    # Align features and labels
    common_index = features.index.intersection(labels.index)
    features_aligned = features.loc[common_index]
    labels_aligned = labels.loc[common_index]
    
    print(f"📊 Aligned: Features {features_aligned.shape}, Labels {len(labels_aligned)}")
    
    # Create ML models instance
    ml_models = TradingMLModels()
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test = ml_models.prepare_data(features_aligned, labels_aligned)
    
    print(f"\n🔍 TRAIN/TEST SPLIT ANALYSIS:")
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Train labels: {y_train.value_counts().to_dict()}")
    print(f"   Test labels: {y_test.value_counts().to_dict()}")
    
    # Check if test sets are identical
    print(f"\n🔍 DATA CONSISTENCY CHECK:")
    print(f"   X_train hash: {hash(str(X_train.values.tobytes()))}")
    print(f"   X_test hash: {hash(str(X_test.values.tobytes()))}")
    print(f"   y_train hash: {hash(str(y_train.values.tobytes()))}")
    print(f"   y_test hash: {hash(str(y_test.values.tobytes()))}")
    
    # Train models individually and check intermediate results
    models_to_test = ['random_forest', 'xgboost']
    
    for model_name in models_to_test:
        print(f"\n🎯 DETAILED {model_name.upper()} ANALYSIS:")
        print("-" * 40)
        
        # Train model
        train_result = ml_models.train_model(model_name, X_train, y_train)
        
        if 'error' not in train_result:
            print(f"   ✅ Training successful")
            print(f"   CV mean: {train_result['cv_mean']:.6f}")
            print(f"   CV std: {train_result['cv_std']:.6f}")
            print(f"   CV scores: {train_result['cv_scores']}")
            
            # Get the actual model
            model = ml_models.models[model_name]
            
            # Make predictions manually
            predictions = model.predict(X_test)
            print(f"   Raw predictions: {np.unique(predictions, return_counts=True)}")
            print(f"   Prediction shape: {predictions.shape}")
            print(f"   Prediction type: {type(predictions)}")
            
            # Check if models are actually different
            if model_name == 'random_forest':
                print(f"   Random Forest trees: {model.n_estimators}")
                print(f"   Random Forest max_depth: {model.max_depth}")
            elif model_name == 'xgboost':
                print(f"   XGBoost n_estimators: {model.n_estimators}")
                print(f"   XGBoost max_depth: {model.max_depth}")
            
            # Evaluate manually
            eval_result = ml_models.evaluate_model(model_name, X_test, y_test)
            
            print(f"   Test accuracy: {eval_result['accuracy']:.6f}")
            print(f"   Test F1: {eval_result['f1_score']:.6f}")
            print(f"   Test samples: {eval_result['test_samples']}")
            
        else:
            print(f"   ❌ Training failed: {train_result['error']}")
    
    # Check if the models are making identical predictions
    if len(ml_models.models) >= 2:
        print(f"\n🔍 PREDICTION COMPARISON:")
        print("-" * 30)
        
        model_names = list(ml_models.models.keys())
        predictions = {}
        
        for model_name in model_names:
            model = ml_models.models[model_name]
            pred = model.predict(X_test)
            predictions[model_name] = pred
            print(f"   {model_name} predictions: {pred[:10]}...")  # First 10
        
        # Compare predictions
        if len(model_names) >= 2:
            pred1 = predictions[model_names[0]]
            pred2 = predictions[model_names[1]]
            
            identical_predictions = np.array_equal(pred1, pred2)
            print(f"   Predictions identical: {identical_predictions}")
            
            if not identical_predictions:
                diff_count = np.sum(pred1 != pred2)
                print(f"   Different predictions: {diff_count}/{len(pred1)}")
    
    # Check feature quality
    print(f"\n🔍 FEATURE QUALITY CHECK:")
    print("-" * 25)
    
    # Check for constant features
    constant_features = []
    for col in X_train.columns:
        if X_train[col].nunique() <= 1:
            constant_features.append(col)
    
    print(f"   Constant features: {len(constant_features)}")
    if constant_features:
        print(f"   Constant feature names: {constant_features[:5]}")
    
    # Check feature variance
    variances = X_train.var()
    low_var_features = variances[variances < 1e-6].index.tolist()
    print(f"   Very low variance features: {len(low_var_features)}")
    
    # Check if any features are identical
    identical_pairs = []
    for i, col1 in enumerate(X_train.columns):
        for j, col2 in enumerate(X_train.columns[i+1:], i+1):
            if X_train[col1].equals(X_train[col2]):
                identical_pairs.append((col1, col2))
    
    print(f"   Identical feature pairs: {len(identical_pairs)}")
    if identical_pairs:
        print(f"   Sample identical pairs: {identical_pairs[:3]}")
    
    # Check label distribution in detail
    print(f"\n🔍 LABEL ANALYSIS:")
    print("-" * 15)
    print(f"   Total unique labels: {y_test.nunique()}")
    print(f"   Label distribution: {y_test.value_counts().to_dict()}")
    
    # If labels are too imbalanced, that could cause identical performance
    if y_test.nunique() <= 2:
        print(f"   ⚠️ Only {y_test.nunique()} unique labels - very imbalanced!")
        majority_class = y_test.value_counts().iloc[0]
        total = len(y_test)
        print(f"   Majority class accuracy: {majority_class/total:.4f}")

if __name__ == "__main__":
    debug_identical_models()
