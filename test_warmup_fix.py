#!/usr/bin/env python3
"""
Test the warm-up period fix for technical indicators
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
import pandas as pd
import numpy as np
from ml_feature_engineer import MLFeatureEngineer

def test_warmup_fix():
    print("🧪 TESTING WARM-UP PERIOD FIX")
    print("=" * 50)
    
    # Get test data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="1y", interval="1d")
    data.columns = [col.lower() for col in data.columns]
    
    # Create feature engineer
    engineer = MLFeatureEngineer()
    
    # Generate features with new warm-up handling
    print("🚀 Testing feature generation with warm-up handling...")
    ml_features = engineer.prepare_ml_dataset(
        data,
        include_lagged=True,
        include_rolling=True,
        feature_selection=True
    )
    
    print(f"✅ Features generated: {ml_features.shape}")
    
    # Test the preview simulation (after warm-up)
    print(f"\n📊 PREVIEW ANALYSIS:")
    warmup_period = min(30, len(ml_features) // 4)
    preview_start = max(warmup_period, 0)
    preview_df = ml_features.iloc[preview_start:preview_start+5]
    
    print(f"   Warm-up period: {warmup_period} rows")
    print(f"   Preview starts at row: {preview_start}")
    
    # Check sample indicators
    sample_indicators = ['MACDh_12_26_9', 'PPOh_12_26_9', 'ROC_10', 'RSI_14', 'volume']
    
    for indicator in sample_indicators:
        if indicator in ml_features.columns:
            # Compare first 5 values vs preview values
            first_5 = ml_features[indicator].dropna().head(5)
            preview_5 = preview_df[indicator].dropna()
            
            print(f"\n   📈 {indicator}:")
            print(f"      First 5 values: {first_5.tolist()}")
            print(f"      Preview values: {preview_5.tolist()}")
            
            # Check if preview shows more variation
            first_unique = first_5.nunique()
            preview_unique = preview_5.nunique()
            
            print(f"      First 5 unique: {first_unique}")
            print(f"      Preview unique: {preview_unique}")
            
            if preview_unique > first_unique:
                print(f"      ✅ Preview shows MORE variation!")
            elif preview_unique == first_unique and first_unique > 1:
                print(f"      ✅ Both show good variation")
            else:
                print(f"      ⚠️ Preview doesn't improve variation")
    
    # Check overall feature quality
    print(f"\n📋 OVERALL FEATURE QUALITY:")
    
    # Analyze features after warm-up
    features_after_warmup = ml_features.iloc[warmup_period:]
    
    good_features = 0
    poor_features = []
    
    for col in features_after_warmup.select_dtypes(include=[np.number]).columns:
        values = features_after_warmup[col].dropna()
        if len(values) > 0:
            unique_pct = values.nunique() / len(values)
            std_val = values.std()
            
            if unique_pct > 0.1 and std_val > 1e-6:  # Good feature
                good_features += 1
            else:
                poor_features.append((col, unique_pct, std_val))
    
    print(f"   ✅ Good features: {good_features}")
    print(f"   ❌ Poor features: {len(poor_features)}")
    
    if poor_features:
        print("   Top 5 poor features:")
        for col, unique_pct, std_val in poor_features[:5]:
            print(f"      {col}: {unique_pct:.3f} unique, std={std_val:.2e}")
    
    # Success metrics
    total_features = len(ml_features.columns)
    good_feature_pct = good_features / total_features * 100
    
    print(f"\n🎯 SUCCESS METRICS:")
    print(f"   Total features: {total_features}")
    print(f"   Good features: {good_features} ({good_feature_pct:.1f}%)")
    
    if good_feature_pct > 80:
        print(f"   ✅ EXCELLENT feature quality!")
    elif good_feature_pct > 60:
        print(f"   ✅ Good feature quality")
    else:
        print(f"   ⚠️ Feature quality needs improvement")

if __name__ == "__main__":
    test_warmup_fix()
