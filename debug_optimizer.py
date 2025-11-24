#!/usr/bin/env python3
"""
Debug Optimizer - Identifies and fixes common optimization issues
Run this to diagnose skipping and errors during optimization
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_model_loading():
    """Check if ML models are loading correctly"""
    print("🔍 CHECKING ML MODEL LOADING...")
    try:
        from ml_indicators import MODEL_LOADER
        
        # Check available models
        print(f"📁 Model directory: {MODEL_LOADER.model_dir}")
        print(f"🧠 Models loaded: {list(MODEL_LOADER.models.keys())}")
        print(f"📊 Metadata: {MODEL_LOADER.metadata}")
        
        # Test each model type
        lstm_model = MODEL_LOADER.get_model('lstm')
        lstm_scaler = MODEL_LOADER.get_model('lstm_scaler')
        rf_models = MODEL_LOADER.get_model('random_forest')
        clustering_models = MODEL_LOADER.get_model('clustering')
        
        print(f"✅ LSTM Model: {'✓' if lstm_model else '✗'}")
        print(f"✅ LSTM Scaler: {'✓' if lstm_scaler else '✗'}")
        print(f"✅ Random Forest: {'✓' if rf_models else '✗'}")
        print(f"✅ Clustering: {'✓' if clustering_models else '✗'}")
        
        if rf_models:
            print(f"   RF subtypes: {list(rf_models.keys()) if isinstance(rf_models, dict) else 'Not a dict'}")
        
        print("✅ ML model loading check complete\n")
        return True
        
    except Exception as e:
        print(f"❌ ML model loading FAILED: {e}\n")
        return False

def check_indicators():
    """Check if indicators are being calculated correctly"""
    print("🔍 CHECKING INDICATOR CALCULATION...")
    try:
        from indicators import get_all_indicators
        import yfinance as yf
        
        # Test with a small dataset
        print("📈 Downloading test data (SPY, 30 days)...")
        test_data = yf.download('SPY', period='30d', interval='1d')
        if test_data.empty:
            print("❌ No test data downloaded")
            return False
        
        test_data = test_data.reset_index()
        test_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        print(f"📊 Test data shape: {test_data.shape}")
        
        # Calculate indicators
        print("🧮 Calculating indicators...")
        enriched_data = get_all_indicators(test_data)
        
        print(f"📈 Enriched data shape: {enriched_data.shape}")
        print(f"📋 Total columns: {len(enriched_data.columns)}")
        
        # Check for missing required columns
        required_cols = ['close', 'open', 'high', 'low', 'volume', 'trend_filter']
        missing_cols = [col for col in required_cols if col not in enriched_data.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        # Check for NaN issues
        last_row = enriched_data.iloc[-1]
        nan_cols = [col for col in enriched_data.columns if pd.isna(last_row[col])]
        if nan_cols:
            print(f"⚠️  NaN in last row: {len(nan_cols)} columns")
            print(f"   First 10: {nan_cols[:10]}")
        
        print("✅ Indicator calculation check complete\n")
        return True
        
    except Exception as e:
        print(f"❌ Indicator calculation FAILED: {e}\n")
        return False

def check_optimization_logic():
    """Check optimization logic for common issues"""
    print("🔍 CHECKING OPTIMIZATION LOGIC...")
    try:
        from optimization import objective
        import yfinance as yf
        from indicators import get_all_indicators
        import optuna
        
        # Create a test study
        study = optuna.create_study(direction='maximize')
        
        # Get test data
        print("📈 Preparing test data...")
        test_data = yf.download('SPY', period='60d', interval='1d')
        test_data = test_data.reset_index()
        test_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        enriched_data = get_all_indicators(test_data)
        
        print(f"📊 Test data ready: {enriched_data.shape}")
        
        # Test different trade preferences
        trade_preferences = [0.2, 0.5, 0.8]
        
        for trade_pref in trade_preferences:
            print(f"\n🎯 Testing trade_preference = {trade_pref}")
            
            try:
                # Create a trial
                trial = study.ask()
                
                # Test objective function
                result = objective(trial, enriched_data, trade_preference=trade_pref)
                
                if result is None or pd.isna(result):
                    print(f"❌ Objective returned invalid result: {result}")
                else:
                    print(f"✅ Objective result: {result:.4f}")
                    
            except Exception as e:
                print(f"❌ Objective function failed: {e}")
                import traceback
                traceback.print_exc()
        
        print("✅ Optimization logic check complete\n")
        return True
        
    except Exception as e:
        print(f"❌ Optimization logic FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def check_backtester():
    """Check backtester for common issues"""
    print("🔍 CHECKING BACKTESTER...")
    try:
        from backtester import Backtester
        from optimization import universal_strategy
        import yfinance as yf
        from indicators import get_all_indicators
        
        # Get test data
        test_data = yf.download('SPY', period='60d', interval='1d')
        test_data = test_data.reset_index()
        test_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        enriched_data = get_all_indicators(test_data)
        
        # Test parameters
        test_params = {
            'buy_score_threshold': 2,
            'sell_score_threshold': 2,
            'min_hold_days': 1,
            'require_confirmation': False,
            'use_trend_filter': False,
            'adx_threshold': 20,
            'use_RSI_14': True,
            'RSI_14_buy_threshold': 30,
            'RSI_14_sell_threshold': 70
        }
        
        print("🔄 Testing backtester...")
        backtester = Backtester(
            enriched_data,
            "Test Strategy",
            universal_strategy,
            test_params,
            10000
        )
        
        backtester.run()
        trade_log, summary = backtester.get_results()
        
        print(f"✅ Backtest completed:")
        print(f"   Trades: {len(trade_log)}")
        print(f"   Return: {summary.get('total_return_pct', 'N/A')}%")
        print(f"   Win Rate: {summary.get('win_rate', 'N/A')}%")
        
        print("✅ Backtester check complete\n")
        return True
        
    except Exception as e:
        print(f"❌ Backtester FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic checks"""
    print("🚀 PATTERN_FINDR DIAGNOSTIC TOOL")
    print("=" * 50)
    print("This will identify common optimization issues\n")
    
    checks = [
        ("ML Model Loading", check_model_loading),
        ("Indicator Calculation", check_indicators),
        ("Optimization Logic", check_optimization_logic),
        ("Backtester", check_backtester),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        print(f"🔍 Running {check_name} check...")
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} check crashed: {e}")
            results[check_name] = False
    
    print("=" * 50)
    print("🎯 DIAGNOSTIC SUMMARY:")
    print("=" * 50)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("If you're still seeing issues, they might be related to:")
        print("- Network connectivity (data download)")
        print("- Specific ticker data quality")
        print("- Optimization parameter combinations")
    else:
        print("⚠️  ISSUES FOUND!")
        print("Fix the failed checks above before running optimization.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
