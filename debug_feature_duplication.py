#!/usr/bin/env python3
"""
Debug feature duplication issues in ML feature engineering
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
import pandas as pd
import numpy as np
from ml_feature_engineer import MLFeatureEngineer

def debug_feature_duplication():
    print("🔍 DEBUGGING FEATURE DUPLICATION ISSUES")
    print("=" * 60)
    
    # Get test data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="6mo", interval="1d")
    data.columns = [col.lower() for col in data.columns]
    
    print(f"📊 Raw Data Shape: {data.shape}")
    print(f"📊 Raw Data Columns: {list(data.columns)}")
    
    # Create feature engineer
    engineer = MLFeatureEngineer()
    
    # Step-by-step feature creation
    print(f"\n🔧 STEP-BY-STEP FEATURE ANALYSIS:")
    print("-" * 50)
    
    # Step 1: Base features
    base_features = engineer.create_base_features(data)
    print(f"1️⃣ Base Features Shape: {base_features.shape}")
    
    # Check for duplicates in base features
    duplicate_cols = []
    for i, col1 in enumerate(base_features.columns):
        for j, col2 in enumerate(base_features.columns[i+1:], i+1):
            if base_features[col1].equals(base_features[col2]):
                duplicate_cols.append((col1, col2))
    
    print(f"   Duplicate base feature pairs: {len(duplicate_cols)}")
    if duplicate_cols:
        print("   First 5 duplicates:")
        for pair in duplicate_cols[:5]:
            print(f"     {pair[0]} == {pair[1]}")
    
    # Step 2: ML-specific features
    ml_features = engineer.create_ml_specific_features(base_features)
    print(f"\n2️⃣ ML Features Shape: {ml_features.shape}")
    
    # Check what was added
    new_cols = set(ml_features.columns) - set(base_features.columns)
    print(f"   New ML columns added: {len(new_cols)}")
    print(f"   Sample new columns: {list(new_cols)[:5]}")
    
    # Check for constant columns
    constant_cols = []
    for col in ml_features.columns:
        if ml_features[col].nunique() <= 1:
            constant_cols.append(col)
    
    print(f"   Constant value columns: {len(constant_cols)}")
    if constant_cols:
        print(f"   Constant columns: {constant_cols[:10]}")
    
    # Step 3: Check correlation matrix
    print(f"\n3️⃣ CORRELATION ANALYSIS:")
    numeric_cols = ml_features.select_dtypes(include=[np.number]).columns
    clean_data = ml_features[numeric_cols].fillna(ml_features[numeric_cols].median())
    
    if len(clean_data.columns) > 1:
        corr_matrix = clean_data.corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.99:  # Very high correlation
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
        
        print(f"   Pairs with >99% correlation: {len(high_corr_pairs)}")
        if high_corr_pairs:
            print("   Top 10 highly correlated pairs:")
            for pair in high_corr_pairs[:10]:
                print(f"     {pair[0]} <-> {pair[1]} (r={pair[2]:.4f})")
    
    # Step 4: Detailed analysis of specific problematic features
    print(f"\n4️⃣ DETAILED FEATURE ANALYSIS:")
    print("-" * 40)
    
    # Check some specific features that might be problematic
    sample_features = list(ml_features.columns)[:20]
    
    print(f"📋 Sample of first 20 features:")
    for i, col in enumerate(sample_features):
        unique_vals = ml_features[col].nunique()
        mean_val = ml_features[col].mean() if pd.api.types.is_numeric_dtype(ml_features[col]) else "N/A"
        print(f"   {i+1:2d}. {col:<25} | Unique: {unique_vals:4d} | Mean: {mean_val}")
    
    # Step 5: Check for identical columns by values
    print(f"\n5️⃣ IDENTICAL COLUMNS CHECK:")
    print("-" * 35)
    
    identical_groups = {}
    processed_cols = set()
    
    for col1 in ml_features.columns:
        if col1 in processed_cols:
            continue
            
        identical_to_col1 = [col1]
        
        for col2 in ml_features.columns:
            if col2 != col1 and col2 not in processed_cols:
                if ml_features[col1].equals(ml_features[col2]):
                    identical_to_col1.append(col2)
                    processed_cols.add(col2)
        
        if len(identical_to_col1) > 1:
            identical_groups[col1] = identical_to_col1
        
        processed_cols.add(col1)
    
    print(f"Groups of identical columns: {len(identical_groups)}")
    for i, (leader, group) in enumerate(identical_groups.items()):
        if i < 5:  # Show first 5 groups
            print(f"   Group {i+1}: {len(group)} identical columns")
            print(f"     Leader: {leader}")
            print(f"     Duplicates: {group[1:3]}")  # Show first 2 duplicates
    
    # Step 6: Show actual values for problematic features
    print(f"\n6️⃣ SAMPLE VALUES FOR PROBLEMATIC FEATURES:")
    print("-" * 45)
    
    if identical_groups:
        first_group = list(identical_groups.values())[0]
        leader_col = first_group[0]
        
        print(f"📊 Values for problematic group (leader: {leader_col}):")
        sample_vals = ml_features[leader_col].head(10)
        for i, val in enumerate(sample_vals):
            print(f"   Row {i}: {val}")
        
        print(f"\n   Stats: min={ml_features[leader_col].min()}, max={ml_features[leader_col].max()}, std={ml_features[leader_col].std():.6f}")

if __name__ == "__main__":
    debug_feature_duplication()
