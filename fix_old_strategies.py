#!/usr/bin/env python3
"""
Fix Old Saved Strategies - Add Ticker Information

This script updates old strategy JSON files that don't have ticker/period/interval info.
It will ask you which ticker to assign to each strategy.
"""

import os
import json
from datetime import datetime

def fix_old_strategies():
    """Update old strategy files to include ticker information"""
    storage_dir = "saved_strategies"
    
    if not os.path.exists(storage_dir):
        print(f"❌ Directory {storage_dir} not found!")
        return
    
    # Find all strategy files
    strategy_files = [f for f in os.listdir(storage_dir) if f.endswith('.json')]
    
    if not strategy_files:
        print("No strategy files found!")
        return
    
    print(f"Found {len(strategy_files)} strategy files\n")
    
    # Check each file
    files_to_fix = []
    for filename in strategy_files:
        filepath = os.path.join(storage_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Check if ticker is missing or Unknown
            if 'ticker' not in data or data.get('ticker') == 'Unknown':
                files_to_fix.append((filepath, filename, data))
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
    
    if not files_to_fix:
        print("✅ All strategies already have ticker information!")
        return
    
    print(f"Found {len(files_to_fix)} strategies that need updating:\n")
    
    # Show strategies that need fixing
    for i, (filepath, filename, data) in enumerate(files_to_fix, 1):
        strategy_name = data.get('name', 'Unknown')
        return_pct = data.get('performance', {}).get('total_return_pct', 0)
        print(f"{i}. {strategy_name} - {return_pct:.2f}% return")
        print(f"   File: {filename}")
        print()
    
    # Ask user what ticker to assign
    print("\n" + "="*60)
    print("WHICH TICKER WERE THESE STRATEGIES OPTIMIZED ON?")
    print("="*60)
    print("\nOptions:")
    print("1. All strategies → Same ticker (e.g., MSTY)")
    print("2. Individually specify ticker for each")
    print("3. Cancel")
    
    choice = input("\nYour choice (1/2/3): ").strip()
    
    if choice == '1':
        # All same ticker
        ticker = input("\nEnter ticker symbol (e.g., MSTY): ").strip().upper()
        period = input("Enter period (default: 1y): ").strip() or "1y"
        interval = input("Enter interval (default: 1d): ").strip() or "1d"
        
        # Update all files
        updated_count = 0
        for filepath, filename, data in files_to_fix:
            try:
                data['ticker'] = ticker
                data['period'] = period
                data['interval'] = interval
                
                # Save updated file
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                updated_count += 1
                print(f"✅ Updated: {filename}")
            except Exception as e:
                print(f"❌ Error updating {filename}: {e}")
        
        print(f"\n🎉 Updated {updated_count}/{len(files_to_fix)} strategies!")
        
    elif choice == '2':
        # Individual ticker for each
        updated_count = 0
        for filepath, filename, data in files_to_fix:
            strategy_name = data.get('name', 'Unknown')
            return_pct = data.get('performance', {}).get('total_return_pct', 0)
            
            print(f"\n{'='*60}")
            print(f"Strategy: {strategy_name}")
            print(f"Return: {return_pct:.2f}%")
            print(f"File: {filename}")
            print(f"{'='*60}")
            
            ticker = input("Enter ticker (or 'skip' to skip this one): ").strip().upper()
            
            if ticker == 'SKIP':
                print("⏭️ Skipped")
                continue
            
            period = input("Enter period (default: 1y): ").strip() or "1y"
            interval = input("Enter interval (default: 1d): ").strip() or "1d"
            
            try:
                data['ticker'] = ticker
                data['period'] = period
                data['interval'] = interval
                
                # Save updated file
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                updated_count += 1
                print(f"✅ Updated!")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n🎉 Updated {updated_count}/{len(files_to_fix)} strategies!")
        
    else:
        print("\n❌ Cancelled")
        return
    
    print("\n✅ Done! Refresh your Streamlit app to see the changes.")
    print("\n💡 Tip: Clear Streamlit cache by pressing 'C' in the app menu")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔧 Fix Old Saved Strategies")
    print("="*60 + "\n")
    
    try:
        fix_old_strategies()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
