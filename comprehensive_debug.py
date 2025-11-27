#!/usr/bin/env python3
"""
COMPREHENSIVE DEBUG - Find ALL bugs in the ML pipeline
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

def comprehensive_debug():
    print("🔍 COMPREHENSIVE ML PIPELINE DEBUG")
    print("=" * 70)
    
    # Get test data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="1y", interval="1d")
    data.columns = [col.lower() for col in data.columns]
    
    print(f"📊 Raw Data Shape: {data.shape}")
    
    # === STEP 1: BASE FEATURE GENERATION ===
    print(f"\n🔧 STEP 1: BASE FEATURE ANALYSIS")
    print("-" * 50)
    
    engineer = MLFeatureEngineer()
    base_features = engineer.create_base_features(data)
    
    print(f"Base features shape: {base_features.shape}")
    
    # Check for immediate duplicates in base features
    base_duplicates = []
    for i in range(len(base_features.columns)):
        for j in range(i+1, len(base_features.columns)):
            col1, col2 = base_features.columns[i], base_features.columns[j]
            if base_features[col1].equals(base_features[col2]):
                base_duplicates.append((col1, col2))
    
    print(f"❌ Base duplicate pairs: {len(base_duplicates)}")
    if base_duplicates:
        print(f"   Sample duplicates: {base_duplicates[:3]}")
    
    # === STEP 2: ML-SPECIFIC FEATURES ===
    print(f"\n🧠 STEP 2: ML-SPECIFIC FEATURES")
    print("-" * 40)
    
    ml_enhanced = engineer.create_ml_specific_features(base_features)
    
    print(f"ML enhanced shape: {ml_enhanced.shape}")
    
    # Check what ML features actually added
    new_cols = set(ml_enhanced.columns) - set(base_features.columns)
    print(f"New ML columns: {len(new_cols)}")
    if new_cols:
        print(f"Sample new: {list(new_cols)[:5]}")
    
    # === STEP 3: LAGGED FEATURES ===  
    print(f"\n📅 STEP 3: LAGGED FEATURES")
    print("-" * 30)
    
    with_lagged = engineer.create_lagged_features(ml_enhanced)
    
    lagged_cols = set(with_lagged.columns) - set(ml_enhanced.columns)
    print(f"Lagged features shape: {with_lagged.shape}")
    print(f"New lagged columns: {len(lagged_cols)}")
    
    # === STEP 4: ROLLING FEATURES ===
    print(f"\n📊 STEP 4: ROLLING FEATURES")
    print("-" * 32)
    
    with_rolling = engineer.create_rolling_features(with_lagged)
    
    rolling_cols = set(with_rolling.columns) - set(with_lagged.columns)
    print(f"Rolling features shape: {with_rolling.shape}")
    print(f"New rolling columns: {len(rolling_cols)}")
    
    # === STEP 5: FEATURE SELECTION ANALYSIS ===
    print(f"\n🎯 STEP 5: FEATURE SELECTION DEEP DIVE")
    print("-" * 45)
    
    print(f"Pre-selection shape: {with_rolling.shape}")
    
    # Manual analysis before selection
    numeric_cols = with_rolling.select_dtypes(include=[np.number]).columns
    print(f"Numeric columns: {len(numeric_cols)}")
    
    # Check for constant columns manually
    constant_cols = []
    for col in numeric_cols:
        if with_rolling[col].nunique() <= 1:
            constant_cols.append(col)
    
    print(f"Manual constant detection: {len(constant_cols)}")
    
    # Check for identical columns manually
    manual_identical = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            if with_rolling[col1].equals(with_rolling[col2]):
                manual_identical.append((col1, col2))
    
    print(f"Manual identical pairs: {len(manual_identical)}")
    if manual_identical:
        print(f"Sample identical: {manual_identical[:3]}")
    
    # Now run the actual feature selection
    selected_features = engineer.select_features(with_rolling)
    
    print(f"Post-selection shape: {selected_features.shape}")
    
    # === STEP 6: FINAL DUPLICATION CHECK ===
    print(f"\n🔍 STEP 6: FINAL DUPLICATE VERIFICATION")
    print("-" * 42)
    
    final_duplicates = []
    for i in range(len(selected_features.columns)):
        for j in range(i+1, len(selected_features.columns)):
            col1, col2 = selected_features.columns[i], selected_features.columns[j]
            if selected_features[col1].equals(selected_features[col2]):
                final_duplicates.append((col1, col2))
    
    print(f"Final duplicate pairs: {len(final_duplicates)}")
    if final_duplicates:
        print(f"❌ REMAINING DUPLICATES: {final_duplicates}")
    
    # === STEP 7: VALUE ANALYSIS ===
    print(f"\n📋 STEP 7: VALUE DIVERSITY ANALYSIS")
    print("-" * 38)
    
    # Check first few features for value diversity
    sample_features = selected_features.iloc[:, :10]
    
    for i, col in enumerate(sample_features.columns):
        unique_vals = sample_features[col].nunique()
        unique_pct = unique_vals / len(sample_features) * 100
        std_val = sample_features[col].std()
        
        print(f"   {i+1:2d}. {col:<25} | Unique: {unique_vals:3d} ({unique_pct:4.1f}%) | Std: {std_val:.4f}")
    
    # === STEP 8: CROSS-CORRELATION MATRIX ===
    print(f"\n🔗 STEP 8: CORRELATION ANALYSIS")
    print("-" * 35)
    
    # Check correlation of first 20 features
    sample_corr = selected_features.iloc[:, :20].corr().abs()
    
    high_corr_pairs = []
    for i in range(len(sample_corr.columns)):
        for j in range(i+1, len(sample_corr.columns)):
            corr_val = sample_corr.iloc[i, j]
            if corr_val > 0.95:
                col1, col2 = sample_corr.columns[i], sample_corr.columns[j]
                high_corr_pairs.append((col1, col2, corr_val))
    
    print(f"High correlation pairs (>95%): {len(high_corr_pairs)}")
    if high_corr_pairs:
        print(f"Sample high correlations: {high_corr_pairs[:3]}")
    
    # === STEP 9: FEATURE PREVIEW SIMULATION ===
    print(f"\n🔬 STEP 9: FEATURE PREVIEW SIMULATION")
    print("-" * 42)
    
    # Simulate what would be shown in feature preview
    preview_sample = selected_features.head(10)
    
    print(f"Preview columns (first 5): {list(preview_sample.columns[:5])}")
    
    # Check if rows have identical values across columns
    row_issues = []
    for idx in preview_sample.index:
        row_values = preview_sample.loc[idx].values
        unique_in_row = len(set(row_values[~pd.isna(row_values)]))
        total_values = len(row_values[~pd.isna(row_values)])
        
        if unique_in_row < total_values * 0.7:  # Less than 70% unique values
            row_issues.append((idx, unique_in_row, total_values))
    
    print(f"Rows with repetitive values: {len(row_issues)}")
    if row_issues:
        for idx, unique, total in row_issues[:3]:
            print(f"   Row {idx}: {unique}/{total} unique values")
            print(f"   Values: {preview_sample.loc[idx].values[:10]}")
    
    # === SUMMARY ===
    print(f"\n📋 COMPREHENSIVE SUMMARY:")
    print("=" * 30)
    print(f"✅ Pipeline ran successfully")
    print(f"📊 Final features: {selected_features.shape[1]}")
    print(f"❌ Critical issues found:")
    print(f"   - Base duplicates: {len(base_duplicates)}")
    print(f"   - Manual identical pairs: {len(manual_identical)}")  
    print(f"   - Final duplicates: {len(final_duplicates)}")
    print(f"   - Rows with repetitive values: {len(row_issues)}")
    
    if len(final_duplicates) > 0 or len(row_issues) > 0:
        print(f"\n🚨 MAJOR ISSUES DETECTED - PIPELINE NEEDS FIXES")
    else:
        print(f"\n✅ Pipeline appears to be working correctly")

if __name__ == "__main__":
    comprehensive_debug()
