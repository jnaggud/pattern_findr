#!/usr/bin/env python3
"""
Remove duplicate strategy files based on timestamp
Keeps the first occurrence of each unique strategy
"""

import os
import json
from pathlib import Path

def remove_duplicate_strategies():
    """Remove duplicate strategy files, keeping only unique ones by timestamp"""
    storage_dir = "saved_strategies"
    
    if not os.path.exists(storage_dir):
        print(f"❌ Directory {storage_dir} not found")
        return
    
    print(f"🔍 Scanning {storage_dir} for duplicates...\n")
    
    seen_timestamps = {}  # timestamp -> filepath
    duplicates = []
    total_files = 0
    
    # Scan all files
    for filename in os.listdir(storage_dir):
        if filename.endswith('.json'):
            total_files += 1
            filepath = os.path.join(storage_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    strategy_data = json.load(f)
                    timestamp = strategy_data.get('timestamp')
                    
                    if timestamp in seen_timestamps:
                        # This is a duplicate
                        duplicates.append({
                            'file': filepath,
                            'timestamp': timestamp,
                            'original': seen_timestamps[timestamp]
                        })
                    else:
                        # First occurrence, keep it
                        seen_timestamps[timestamp] = filepath
                        
            except Exception as e:
                print(f"⚠️  Error reading {filename}: {e}")
    
    # Report findings
    print(f"📊 Total files scanned: {total_files}")
    print(f"✅ Unique strategies: {len(seen_timestamps)}")
    print(f"🔄 Duplicates found: {len(duplicates)}\n")
    
    if not duplicates:
        print("✨ No duplicates found! Your strategies are clean.")
        return
    
    # Show duplicates
    print("📋 Duplicate files to be removed:")
    for dup in duplicates:
        print(f"  ❌ {dup['file']}")
        print(f"     (same as {dup['original']})")
    
    # Confirm deletion
    print(f"\n⚠️  This will DELETE {len(duplicates)} duplicate file(s).")
    response = input("Continue? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ Cancelled. No files deleted.")
        return
    
    # Delete duplicates
    deleted = 0
    for dup in duplicates:
        try:
            os.remove(dup['file'])
            deleted += 1
            print(f"✅ Deleted: {dup['file']}")
        except Exception as e:
            print(f"❌ Error deleting {dup['file']}: {e}")
    
    print(f"\n🎉 Done! Deleted {deleted} duplicate file(s).")
    print(f"📊 Remaining strategies: {len(seen_timestamps)}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧹 Strategy Duplicate Remover")
    print("=" * 60)
    print()
    
    remove_duplicate_strategies()
    
    print("\n💡 Tip: Restart Streamlit to clear cache and see changes.")
