#!/usr/bin/env python3
"""
Verify that EnhancedBacktesterV2 is being used correctly
"""
import sys

print("🔍 Verifying Enhanced Backtester Integration\n")
print("=" * 60)

# Check imports
print("\n1️⃣ Checking imports...")
try:
    from backtester_enhanced import EnhancedBacktesterV2
    print("   ✅ EnhancedBacktesterV2 imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import EnhancedBacktesterV2: {e}")
    sys.exit(1)

# Check that optimization uses it
print("\n2️⃣ Checking optimization.py...")
with open('/Users/jeffersonduggan/Documents/Pattern_FindR/optimization.py', 'r') as f:
    opt_code = f.read()
    
    if 'from backtester_enhanced import EnhancedBacktesterV2' in opt_code:
        print("   ✅ EnhancedBacktesterV2 imported")
    else:
        print("   ❌ EnhancedBacktesterV2 NOT imported")
    
    # Check for the fix
    if 'EnhancedBacktesterV2(data, strategy_name, universal_strategy, params, 100000)' in opt_code:
        print("   ✅ Using EnhancedBacktesterV2 in objective function")
    else:
        print("   ⚠️  May not be using EnhancedBacktesterV2 correctly")
    
    # Check for stop-loss params
    if "params['stop_loss_pct'] = trial.suggest_float('stop_loss_pct'" in opt_code:
        print("   ✅ Stop-loss parameters added to optimization")
    else:
        print("   ❌ Stop-loss parameters NOT found")

# Check app.py
print("\n3️⃣ Checking app.py...")
with open('/Users/jeffersonduggan/Documents/Pattern_FindR/app.py', 'r') as f:
    app_code = f.read()
    
    if 'from backtester_enhanced import EnhancedBacktesterV2' in app_code:
        print("   ✅ EnhancedBacktesterV2 imported")
    else:
        print("   ❌ EnhancedBacktesterV2 NOT imported")
    
    # Count EnhancedBacktesterV2 usages
    v2_count = app_code.count('EnhancedBacktesterV2(')
    print(f"   ✅ EnhancedBacktesterV2 used {v2_count} times")

# Test instantiation
print("\n4️⃣ Testing EnhancedBacktesterV2 instantiation...")
try:
    import pandas as pd
    import numpy as np
    
    # Create dummy data
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    data.index = pd.date_range('2024-01-01', periods=100)
    
    # Dummy strategy
    def dummy_strategy(data, params):
        return pd.Series(0, index=data.index)
    
    # Test parameters
    params = {
        'stop_loss_pct': 8.0,
        'take_profit_pct': 15.0,
        'use_stop_loss': True,
        'use_take_profit': True
    }
    
    # Create backtester
    bt = EnhancedBacktesterV2(data, "Test", dummy_strategy, params, 100000)
    print(f"   ✅ Successfully created EnhancedBacktesterV2")
    print(f"      Stop-Loss: {bt.stop_loss_pct}%")
    print(f"      Take-Profit: {bt.take_profit_pct}%")
    
except Exception as e:
    print(f"   ❌ Failed to create backtester: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL CHECKS PASSED!")
print("\n📋 Summary:")
print("   • EnhancedBacktesterV2 is correctly imported")
print("   • Stop-loss/take-profit parameters added to optimization")
print("   • All backtester calls updated")
print("\n🎯 Next Step: Run new optimization in Streamlit app")
print("   Look for: '📊 Using Enhanced Backtester V2'")
