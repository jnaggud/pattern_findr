#!/usr/bin/env python3
"""
COMPREHENSIVE DIAGNOSIS - Find ALL critical issues in ML pipeline
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
from indicators import get_all_indicators
from ml_models import TradingMLModels

def comprehensive_diagnosis():
    print("🚨 COMPREHENSIVE ML PIPELINE DIAGNOSIS")
    print("=" * 70)
    
    # Get test data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="1y", interval="1d")
    data.columns = [col.lower() for col in data.columns]
    
    print(f"📊 Raw Data: {data.shape}")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")
    print(f"   Sample prices: {data['close'].head(3).tolist()}")
    
    # === ISSUE #1: TECHNICAL INDICATORS DIAGNOSIS ===
    print(f"\n🔍 ISSUE #1: TECHNICAL INDICATORS ANALYSIS")
    print("-" * 60)
    
    # Test indicators directly
    print("Testing indicators directly...")
    try:
        indicators_df = get_all_indicators(data.copy())
        print(f"✅ Raw indicators shape: {indicators_df.shape}")
        
        # Check for constant indicators
        constant_indicators = []
        low_variance_indicators = []
        
        for col in indicators_df.select_dtypes(include=[np.number]).columns:
            values = indicators_df[col].dropna()
            if len(values) > 0:
                unique_vals = values.nunique()
                std_val = values.std()
                
                if unique_vals <= 1:
                    constant_indicators.append(col)
                elif std_val < 1e-10:  # Very low variance
                    low_variance_indicators.append((col, std_val, unique_vals))
        
        print(f"❌ Constant indicators: {len(constant_indicators)}")
        if constant_indicators:
            print(f"   Examples: {constant_indicators[:5]}")
        
        print(f"⚠️ Low variance indicators: {len(low_variance_indicators)}")
        if low_variance_indicators:
            for col, std, unique in low_variance_indicators[:5]:
                print(f"   {col}: std={std:.2e}, unique={unique}")
        
        # Check specific problematic indicators
        problem_indicators = ['MACDh_12_26_9', 'PPOh_12_26_9', 'ROC_10']
        for ind in problem_indicators:
            if ind in indicators_df.columns:
                values = indicators_df[ind].dropna()
                print(f"   {ind}: unique={values.nunique()}, std={values.std():.6f}")
                print(f"     First 5 values: {values.head().tolist()}")
        
    except Exception as e:
        print(f"❌ Error with indicators: {e}")
    
    # === ISSUE #2: FEATURE ENGINEERING DIAGNOSIS ===
    print(f"\n🔍 ISSUE #2: FEATURE ENGINEERING PIPELINE")
    print("-" * 50)
    
    engineer = MLFeatureEngineer()
    
    # Step by step analysis
    base_features = engineer.create_base_features(data)
    print(f"Base features: {base_features.shape}")
    
    # Check base features quality
    numeric_cols = base_features.select_dtypes(include=[np.number]).columns
    print(f"Numeric columns: {len(numeric_cols)}")
    
    # Variance analysis
    variance_analysis = []
    for col in numeric_cols:
        values = base_features[col].dropna()
        if len(values) > 1:
            variance_analysis.append({
                'column': col,
                'unique_count': values.nunique(), 
                'unique_pct': values.nunique() / len(values) * 100,
                'std': values.std(),
                'range': values.max() - values.min() if values.nunique() > 1 else 0
            })
    
    # Sort by diversity (ascending = worst first)
    variance_analysis.sort(key=lambda x: x['unique_pct'])
    
    print(f"\n📊 WORST FEATURES (lowest diversity):")
    for i, feat in enumerate(variance_analysis[:10]):
        print(f"   {i+1:2d}. {feat['column']:<25} | {feat['unique_count']:3d} unique ({feat['unique_pct']:4.1f}%) | std: {feat['std']:.6f}")
    
    # === ISSUE #3: FEATURE SELECTION EFFECTIVENESS ===
    print(f"\n🔍 ISSUE #3: FEATURE SELECTION ANALYSIS")
    print("-" * 45)
    
    # Test current selection method
    ml_enhanced = engineer.create_ml_specific_features(base_features)
    with_lagged = engineer.create_lagged_features(ml_enhanced)
    with_rolling = engineer.create_rolling_features(with_lagged)
    
    print(f"Before selection: {with_rolling.shape[1]} features")
    
    # Manual variance check before selection
    numeric_features = with_rolling.select_dtypes(include=[np.number])
    
    # Check what should be removed
    should_remove_constant = []
    should_remove_low_var = []
    
    for col in numeric_features.columns:
        values = numeric_features[col].dropna()
        if len(values) > 0:
            if values.nunique() <= 1:
                should_remove_constant.append(col)
            elif values.std() < 1e-6:  # Very low variance threshold
                should_remove_low_var.append(col)
    
    print(f"Should remove constant: {len(should_remove_constant)}")
    print(f"Should remove low variance: {len(should_remove_low_var)}")
    
    # Run actual selection
    selected = engineer.select_features(with_rolling.copy())
    print(f"After selection: {selected.shape[1]} features")
    
    # Check if problematic features survived
    survived_problems = []
    for col in selected.columns:
        if col in numeric_features.columns:
            values = selected[col].dropna()
            if len(values) > 0 and values.std() < 1e-4:
                survived_problems.append((col, values.std(), values.nunique()))
    
    print(f"❌ Problematic features that survived: {len(survived_problems)}")
    for col, std, unique in survived_problems[:5]:
        print(f"   {col}: std={std:.2e}, unique={unique}")
    
    # === ISSUE #4: PEAK/VALLEY DETECTION ===
    print(f"\n🔍 ISSUE #4: PEAK/VALLEY DETECTION")
    print("-" * 40)
    
    detector = PeakValleyDetector()
    
    # Test different detection methods
    methods_to_test = [
        {'method': 'scipy_peaks', 'params': {'prominence_pct': 2.0, 'distance': 10}},  # Current
        {'method': 'scipy_peaks', 'params': {'prominence_pct': 1.0, 'distance': 5}},   # Moderate  
        {'method': 'scipy_peaks', 'params': {'prominence_pct': 0.5, 'distance': 3}},   # Aggressive
    ]
    
    for config in methods_to_test:
        try:
            _, labels = detector.create_labeled_dataset(data, config['method'], **config['params'])
            label_counts = labels.value_counts()
            total = len(labels)
            
            buy_pct = label_counts.get(1, 0) / total * 100
            sell_pct = label_counts.get(-1, 0) / total * 100
            hold_pct = label_counts.get(0, 0) / total * 100
            
            print(f"   {config['method']} ({config['params']}): BUY {buy_pct:.1f}%, SELL {sell_pct:.1f}%, HOLD {hold_pct:.1f}%")
            
        except Exception as e:
            print(f"   ❌ {config['method']}: {e}")
    
    # === ISSUE #5: MODEL TRAINING DIAGNOSIS ===
    print(f"\n🔍 ISSUE #5: MODEL TRAINING ANALYSIS")
    print("-" * 40)
    
    # Use a simple dataset for testing
    try:
        detector = PeakValleyDetector()
        _, labels = detector.create_labeled_dataset(data, 'scipy_peaks', prominence_pct=1.0, distance=5)
        
        # Use simplified features for testing
        simple_features = selected.iloc[:, :20]  # Just first 20 features
        
        # Align data
        common_index = simple_features.index.intersection(labels.index)
        features_aligned = simple_features.loc[common_index]
        labels_aligned = labels.loc[common_index]
        
        print(f"Training data: {features_aligned.shape}")
        print(f"Label distribution: {labels_aligned.value_counts().to_dict()}")
        
        # Quick model test
        ml_models = TradingMLModels()
        X_train, X_test, y_train, y_test = ml_models.prepare_data(features_aligned, labels_aligned)
        
        print(f"Train/test split: {X_train.shape}, {X_test.shape}")
        print(f"Feature variance in training set:")
        
        # Check feature quality in training set
        feature_quality = []
        for col in X_train.columns:
            std_val = X_train[col].std()
            unique_val = X_train[col].nunique()
            feature_quality.append((col, std_val, unique_val))
        
        # Sort by std (ascending = worst first)
        feature_quality.sort(key=lambda x: x[1])
        
        print("   Worst features (lowest std):")
        for col, std, unique in feature_quality[:5]:
            print(f"     {col}: std={std:.6f}, unique={unique}")
        
    except Exception as e:
        print(f"❌ Model training test failed: {e}")
    
    # === FINAL RECOMMENDATIONS ===
    print(f"\n💡 COMPREHENSIVE RECOMMENDATIONS:")
    print("=" * 40)
    print("1. 🔧 FIX TECHNICAL INDICATORS:")
    print("   - Many indicators returning constant values")
    print("   - Check indicator calculation parameters")
    print("   - Verify input data alignment")
    print()
    print("2. 🎯 IMPROVE FEATURE SELECTION:")
    print("   - Current method misses low-variance features")
    print("   - Need stricter variance threshold (e.g., std < 1e-4)")
    print("   - Remove features with <95% unique values")
    print()
    print("3. 📊 FIX PEAK/VALLEY DETECTION:")
    print("   - Use more aggressive parameters (0.5% prominence)")
    print("   - Target 15-25% signal ratio, not 5%")
    print()
    print("4. 🤖 MODEL TRAINING IMPROVEMENTS:")
    print("   - Ensure feature diversity before training")
    print("   - Use stratified sampling for imbalanced data")
    print("   - Add model-specific debugging")
    print()
    print("5. 🔍 DATA LEAKAGE CHECK:")
    print("   - Verify no look-ahead bias in indicators")
    print("   - Check backtesting logic for time alignment")

if __name__ == "__main__":
    comprehensive_diagnosis()
