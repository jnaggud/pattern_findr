#!/usr/bin/env python3
"""
Setup script for Pattern_FindR Deep Learning Models

This script helps set up the deep learning pattern detection models.
Run this once to train all the models for pattern detection.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_training.log')
    ]
)

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'tensorflow',
        'scikit-learn',
        'matplotlib',
        'mplfinance',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed")
    return True

def setup_directories():
    """Create necessary directories for models and data."""
    directories = [
        'models',
        'chart_images',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def train_models():
    """Train all deep learning models."""
    try:
        from real_time_pattern_detector import train_pattern_models
        
        print("\n🤖 Starting deep learning model training...")
        print("This will take approximately 30-60 minutes depending on your hardware.")
        print("You can monitor progress in the model_training.log file.")
        
        # Train all models
        train_pattern_models()
        
        print("\n✅ All models trained successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logging.error(f"Model training failed: {e}", exc_info=True)
        return False

def verify_models():
    """Verify that all models were created successfully."""
    try:
        from real_time_pattern_detector import check_model_availability, CHART_PATTERNS
        
        available_models = check_model_availability()
        total_models = len(CHART_PATTERNS)
        
        print(f"\n📊 Model Status: {len(available_models)}/{total_models} models available")
        
        for pattern in CHART_PATTERNS:
            status = "✅" if pattern in available_models else "❌"
            print(f"{status} {pattern.replace('_', ' ').title()}")
        
        if len(available_models) == total_models:
            print("\n🎉 All models are ready for use!")
            return True
        else:
            print(f"\n⚠️ Only {len(available_models)} out of {total_models} models available")
            return False
            
    except Exception as e:
        print(f"\n❌ Model verification failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Pattern_FindR Deep Learning Setup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Setup directories
    setup_directories()
    
    # Ask user if they want to train models
    print("\n🤖 Deep Learning Model Training")
    print("This will train neural networks to detect 8 different chart patterns:")
    print("- Head and Shoulders")
    print("- Inverse Head and Shoulders") 
    print("- Cup and Handle")
    print("- Double Top")
    print("- Double Bottom")
    print("- Triangle")
    print("- Flag") 
    print("- Pennant")
    
    response = input("\nDo you want to train the models now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        # Train models
        if train_models():
            # Verify models
            if verify_models():
                print("\n🎯 Setup completed successfully!")
                print("You can now use deep learning pattern detection in the main app.")
                return 0
            else:
                print("\n⚠️ Setup completed with warnings. Some models may not be available.")
                return 1
        else:
            print("\n❌ Setup failed during model training.")
            return 1
    else:
        print("\n📝 Skipping model training.")
        print("You can train models later using the button in the app sidebar.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
