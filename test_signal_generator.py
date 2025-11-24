#!/usr/bin/env python3
"""
Test script for the daily signal generator
Runs a quick test to verify everything is configured correctly
"""

import os
import sys
import json

def test_setup():
    """Test if all required files and configurations are in place"""
    
    print("🧪 Testing Daily Signal Generator Setup")
    print("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Check if portfolio config exists
    print("\n1. Checking portfolio configuration...")
    if os.path.exists('portfolio_config.json'):
        print("   ✅ portfolio_config.json found")
        try:
            with open('portfolio_config.json', 'r') as f:
                config = json.load(f)
                print(f"   ✅ Portfolio has {len(config['tickers'])} tickers: {', '.join(config['tickers'])}")
                tests_passed += 1
        except Exception as e:
            print(f"   ❌ Error reading portfolio_config.json: {e}")
            tests_failed += 1
    else:
        print("   ❌ portfolio_config.json not found")
        tests_failed += 1
    
    # Test 2: Check if saved strategies directory exists
    print("\n2. Checking saved strategies...")
    if os.path.exists('saved_strategies'):
        strategies = [f for f in os.listdir('saved_strategies') if f.endswith('.json')]
        if strategies:
            print(f"   ✅ Found {len(strategies)} saved strategies")
            tests_passed += 1
        else:
            print("   ⚠️  No saved strategies found")
            print("   💡 Run optimization in Streamlit app and save strategies first")
            tests_failed += 1
    else:
        print("   ❌ saved_strategies directory not found")
        tests_failed += 1
    
    # Test 3: Check Python dependencies
    print("\n3. Checking Python dependencies...")
    try:
        import pandas
        import yfinance
        print("   ✅ pandas installed")
        print("   ✅ yfinance installed")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ Missing dependency: {e}")
        tests_failed += 1
    
    # Test 4: Check indicator module
    print("\n4. Checking indicator module...")
    try:
        from indicators import get_all_indicators
        print("   ✅ indicators.py module accessible")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ Cannot import indicators: {e}")
        tests_failed += 1
    
    # Test 5: Check optimization module
    print("\n5. Checking optimization module...")
    try:
        from optimization import universal_strategy
        print("   ✅ optimization.py module accessible")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ Cannot import optimization: {e}")
        tests_failed += 1
    
    # Test 6: Check output directories
    print("\n6. Checking output directories...")
    if not os.path.exists('daily_signals'):
        os.makedirs('daily_signals')
        print("   ✅ Created daily_signals directory")
    else:
        print("   ✅ daily_signals directory exists")
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
        print("   ✅ Created logs directory")
    else:
        print("   ✅ logs directory exists")
    tests_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    print(f"✅ Passed: {tests_passed}")
    print(f"❌ Failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n🎉 All tests passed! System is ready.")
        print("\n📋 Next steps:")
        print("   1. Edit portfolio_config.json to add your tickers")
        print("   2. Run: python daily_signal_generator.py")
        print("   3. Check output in daily_signals/ directory")
        print("   4. Set up cron job for automation (see DAILY_TRADING_SETUP.md)")
        return True
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        if tests_failed == 1 and "saved strategies" in str(tests_failed):
            print("\n💡 Quick fix: Run optimization in Streamlit app and save strategies")
        return False


def test_single_ticker():
    """Test signal generation for a single ticker"""
    print("\n" + "="*60)
    print("🧪 Testing signal generation for SPY")
    print("="*60)
    
    try:
        from daily_signal_generator import DailySignalGenerator
        
        # Create generator
        generator = DailySignalGenerator()
        
        # Test single ticker
        signal = generator.generate_signals_for_ticker('SPY')
        
        if signal:
            print("\n✅ Signal generation test PASSED")
            print(f"   Generated signal: {signal['signal']}")
            print(f"   Price: ${signal['price']}")
            return True
        else:
            print("\n❌ Signal generation test FAILED")
            print("   No signal was generated")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run setup tests
    setup_ok = test_setup()
    
    if setup_ok:
        # Ask if user wants to test signal generation
        print("\n" + "="*60)
        response = input("Would you like to test signal generation for SPY? (y/n): ")
        if response.lower() == 'y':
            test_single_ticker()
    
    print("\n✅ Testing complete!")
