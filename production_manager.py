#!/usr/bin/env python3
"""
Production Manager for Trading System
Handles deployment, configuration, logging, and health checks for the production environment.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import shutil

PRODUCTION_DIR = "production_env"
CONFIG_FILE = os.path.join(PRODUCTION_DIR, "production_config.json")
SIGNALS_LOG = os.path.join(PRODUCTION_DIR, "production_signals.csv")
MODELS_DIR = "saved_models"

class ProductionManager:
    def __init__(self):
        self._ensure_env()
        
    def _ensure_env(self):
        """Ensure production environment exists"""
        if not os.path.exists(PRODUCTION_DIR):
            os.makedirs(PRODUCTION_DIR)
            print(f"📁 Created production environment at {PRODUCTION_DIR}")
            
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            
        # Create empty config if missing
        if not os.path.exists(CONFIG_FILE):
            default_config = {
                "active_model_path": None,
                "deployed_at": None,
                "model_name": None,
                "retraining_interval_days": 30,
                "last_training_date": None,
                "risk_settings": {
                    "stop_loss_atr": 2.0,
                    "take_profit_atr": 3.0,
                    "max_position_size": 0.1
                }
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(default_config, f, indent=4)

    def get_config(self):
        """Load current configuration"""
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error reading config: {e}")
            return {}

    def update_config(self, new_config):
        """Update configuration"""
        current = self.get_config()
        current.update(new_config)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(current, f, indent=4)
        return current

    def list_saved_models(self):
        """List all available saved models"""
        if not os.path.exists(MODELS_DIR):
            return []
            
        models = []
        import joblib
        
        for f in os.listdir(MODELS_DIR):
            if f.endswith('.joblib'):
                path = os.path.join(MODELS_DIR, f)
                try:
                    # We don't load the whole model, just stat the file
                    stat = os.stat(path)
                    created = datetime.fromtimestamp(stat.st_ctime)
                    
                    # Try to parse filename for cleaner info
                    # Format: modelname_YYYYMMDD_HHMMSS.joblib
                    parts = f.replace('.joblib', '').split('_')
                    model_name = parts[0]
                    
                    # PEEK into the file to get real metadata
                    # This is crucial for the UI to show correct feature counts/scaler status
                    try:
                        data = joblib.load(path)
                        meta = data.get('metadata', {})
                        feature_names = data.get('feature_names', [])
                        has_scaler = data.get('scaler') is not None
                    except:
                        meta = {}
                        feature_names = []
                        has_scaler = False
                    
                    models.append({
                        'filename': f,
                        'path': path,
                        'created': created,
                        'name': model_name,
                        'size_kb': stat.st_size / 1024,
                        'metadata': meta,
                        'feature_count': len(feature_names),
                        'has_scaler': has_scaler,
                        'feature_names': feature_names # Include for UI
                    })
                except Exception as e:
                    print(f"⚠️ Error reading {f}: {e}")
        
        return sorted(models, key=lambda x: x['created'], reverse=True)

    def deploy_model(self, model_path):
        """Deploy a specific model to production"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        # Update config
        config_update = {
            "active_model_path": model_path,
            "deployed_at": datetime.now().isoformat(),
            "model_name": os.path.basename(model_path).split('_')[0]
        }
        self.update_config(config_update)
        print(f"🚀 Deployed model from {model_path}")
        return True

    def log_signal(self, date, ticker, signal, confidence, price, metadata=None):
        """Log a generated signal to the audit trail"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'signal_date': date,
            'ticker': ticker,
            'signal': signal,
            'signal_text': 'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD',
            'confidence': confidence,
            'price': price,
            'metadata': json.dumps(metadata or {})
        }
        
        df = pd.DataFrame([entry])
        
        # Append to CSV
        if not os.path.exists(SIGNALS_LOG):
            df.to_csv(SIGNALS_LOG, index=False)
        else:
            df.to_csv(SIGNALS_LOG, mode='a', header=False, index=False)
            
    def get_signal_history(self):
        """Get full signal history"""
        if not os.path.exists(SIGNALS_LOG):
            return pd.DataFrame()
        return pd.read_csv(SIGNALS_LOG)

    def check_system_health(self):
        """Check system health metrics"""
        config = self.get_config()
        active_model = config.get('active_model_path')
        
        health = {
            'status': 'OK',
            'issues': [],
            'active_model': 'None',
            'model_age_days': 0
        }
        
        if not active_model or not os.path.exists(active_model):
            health['status'] = 'WARNING'
            health['issues'].append("No active model deployed")
        else:
            health['active_model'] = os.path.basename(active_model)
            
            # Check model age based on file creation
            try:
                stat = os.stat(active_model)
                created = datetime.fromtimestamp(stat.st_ctime)
                age = (datetime.now() - created).days
                health['model_age_days'] = age
                
                limit = config.get('retraining_interval_days', 30)
                if age > limit:
                    health['status'] = 'WARNING'
                    health['issues'].append(f"Model is {age} days old (Limit: {limit}). Retraining recommended.")
            except:
                pass
                
        return health

if __name__ == "__main__":
    # Simple test
    pm = ProductionManager()
    print("Production Manager initialized")
    print("Config:", pm.get_config())
