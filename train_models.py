#!/usr/bin/env python3
"""
Quick Model Training Script for Pattern_FindR
Run this to train and save all ML models before using the app
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_trainer import ModelTrainer

def main():
    print("🚀 Pattern_FindR Model Training")
    print("=" * 50)
    print("This will train ML models on historical data for fast inference during optimization.")
    print("Training time: ~5-10 minutes depending on your hardware.")
    print("")
    
    # Ask user for confirmation
    confirm = input("Start training? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    print("\n🧠 Starting model training pipeline...")
    
    # Initialize trainer and run
    trainer = ModelTrainer()
    metadata = trainer.train_all_models()
    
    print("\n" + "=" * 50)
    print("✅ MODEL TRAINING COMPLETE!")
    print(f"📁 Models saved to: {trainer.model_dir}")
    print(f"🧠 Models trained: {', '.join(metadata['models_trained'])}")
    print("\n💡 You can now run the Streamlit app with fast ML inference:")
    print("   streamlit run app.py")
    print("\n🔄 Retraining schedule:")
    print("   - Recommended: Weekly (models auto-check freshness)")
    print("   - Run this script again when you want fresh models")
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
