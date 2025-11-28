#!/usr/bin/env python3
"""
Generate requirements-lock.txt using pip freeze.
This is the standard way to create lock files with exact versions.
"""

import subprocess
import datetime
import os

def generate_lock_file():
    """Generate requirements-lock.txt using pip freeze."""
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    
    print("🔄 Generating lock file using pip freeze...")
    
    try:
        # Run pip freeze to get all installed packages
        result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True, check=True)
        packages = result.stdout.strip()
        
        # Create header
        header = [
            "# Pattern_FindR - Complete Lock File (pip freeze)",
            f"# Generated on {timestamp}",
            "# Use this for exact environment reproduction",
            "# Install with: pip install -r requirements-lock.txt",
            "",
        ]
        
        # Combine header and packages
        content = '\n'.join(header) + packages + '\n'
        
        # Write to file
        with open('requirements-lock.txt', 'w') as f:
            f.write(content)
        
        # Count packages
        package_count = len([line for line in packages.split('\n') if line.strip() and not line.startswith('#')])
        
        print(f"✅ Generated requirements-lock.txt with {package_count} packages")
        print(f"� File size: {len(content)} characters")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running pip freeze: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    generate_lock_file()
