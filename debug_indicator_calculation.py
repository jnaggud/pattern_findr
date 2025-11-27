#!/usr/bin/env python3
"""
Debug why technical indicators are returning constant values
"""

import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd
import numpy as np
from indicators import get_all_indicators

def debug_indicator_calculation():
    print("🔍 DEBUGGING INDICATOR CALCULATION PIPELINE")
    print("=" * 60)
    
    # Get test data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="1y", interval="1d")
    data.columns = [col.lower() for col in data.columns]
    
    print(f"📊 Raw Data Check:")
    print(f"   Shape: {data.shape}")
    print(f"   Columns: {data.columns.tolist()}")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")
    print(f"   Price variation:")
    print(f"     Close min: ${data['close'].min():.2f}")
    print(f"     Close max: ${data['close'].max():.2f}")
    print(f"     Close std: ${data['close'].std():.2f}")
    print()
    
    # Show sample of raw data
    print("📋 Sample Raw Data (first 5 rows):")
    print(data[['open', 'high', 'low', 'close', 'volume']].head())
    print()
    
    # Test indicator calculation step by step
    print("🔧 Testing Indicator Calculation...")
    
    # Prepare data for indicators (this is where the bug might be)
    data_for_indicators = data.copy()
    
    print(f"   Before date column preparation:")
    print(f"     Index type: {type(data_for_indicators.index)}")
    print(f"     Columns: {data_for_indicators.columns.tolist()}")
    
    # Add date column (this might be the issue!)
    if 'date' not in data_for_indicators.columns:
        data_for_indicators.reset_index(inplace=True)
        if 'Date' in data_for_indicators.columns:
            data_for_indicators.rename(columns={'Date': 'date'}, inplace=True)
        else:
            data_for_indicators.rename(columns={data_for_indicators.columns[0]: 'date'}, inplace=True)
    
    print(f"   After date column preparation:")
    print(f"     Shape: {data_for_indicators.shape}")
    print(f"     Columns: {data_for_indicators.columns.tolist()}")
    print(f"     Date column type: {type(data_for_indicators['date'].iloc[0])}")
    print()
    
    # Show sample prepared data
    print("📋 Sample Prepared Data (first 5 rows):")
    print(data_for_indicators[['date', 'open', 'high', 'low', 'close', 'volume']].head())
    print()
    
    # Test simple indicator manually first
    print("🧮 Manual MACD Calculation Test:")
    try:
        import pandas_ta as ta
        
        # Calculate MACD manually to see if it works
        manual_macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
        
        if manual_macd is not None and not manual_macd.empty:
            macd_col = [col for col in manual_macd.columns if 'MACD' in col and 'h' not in col][0]
            macd_values = manual_macd[macd_col].dropna()
            
            print(f"   ✅ Manual MACD calculation successful!")
            print(f"   Values count: {len(macd_values)}")
            print(f"   Unique values: {macd_values.nunique()}")
            print(f"   Std deviation: {macd_values.std():.6f}")
            print(f"   Sample values: {macd_values.head().tolist()}")
        else:
            print(f"   ❌ Manual MACD calculation failed")
            
    except Exception as e:
        print(f"   ❌ Manual MACD error: {e}")
    print()
    
    # Now test the full indicator pipeline
    print("🏭 Testing Full Indicator Pipeline:")
    try:
        # This is the exact call that's failing
        indicators_df = get_all_indicators(data_for_indicators)
        
        print(f"   ✅ Indicators calculated successfully!")
        print(f"   Shape: {indicators_df.shape}")
        print(f"   Columns count: {len(indicators_df.columns)}")
        
        # Check specific problematic indicators
        problem_indicators = ['MACDh_12_26_9', 'PPOh_12_26_9', 'ROC_10', 'RSI_14']
        
        for indicator in problem_indicators:
            if indicator in indicators_df.columns:
                values = indicators_df[indicator].dropna()
                print(f"   📊 {indicator}:")
                print(f"      Unique values: {values.nunique()}")
                print(f"      Std deviation: {values.std():.8f}")
                print(f"      Min/Max: {values.min():.6f} / {values.max():.6f}")
                print(f"      First 5: {values.head().tolist()}")
                print(f"      Last 5: {values.tail().tolist()}")
                
                # Check if it's truly constant
                if values.nunique() <= 1:
                    print(f"      🚨 CONSTANT VALUE DETECTED!")
                elif values.std() < 1e-6:
                    print(f"      ⚠️ EXTREMELY LOW VARIANCE!")
                else:
                    print(f"      ✅ Normal variation")
                print()
            else:
                print(f"   ❌ {indicator} not found in indicators")
    
    except Exception as e:
        print(f"   ❌ Full indicator pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test data alignment issues
    print("🔄 Testing Data Alignment:")
    print(f"   Original data index: {data.index[:3].tolist()}")
    print(f"   Prepared data dates: {data_for_indicators['date'][:3].tolist()}")
    
    # Check if there are any NaN or infinite values that could cause issues
    print("\n🔍 Data Quality Check:")
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in data.columns:
            col_data = data[col]
            print(f"   {col}:")
            print(f"     NaN count: {col_data.isnull().sum()}")
            print(f"     Infinite count: {np.isinf(col_data).sum()}")
            print(f"     Zero count: {(col_data == 0).sum()}")
            print(f"     Negative count: {(col_data < 0).sum()}")

if __name__ == "__main__":
    debug_indicator_calculation()
