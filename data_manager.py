import pandas as pd
import yfinance as yf
import os
import json
import shutil
from datetime import datetime, timedelta

class DataManager:
    def __init__(self, data_dir: str = "market_data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def _get_file_path(self, ticker: str, interval: str = '1d') -> str:
        # Sanitize ticker
        safe_ticker = ticker.replace("/", "-").replace("^", "")
        return os.path.join(self.data_dir, f"{safe_ticker}_{interval}.parquet")
    
    def update_data(self, ticker: str, interval: str = '1d') -> pd.DataFrame:
        """
        Update data. 
        Daily: Incremental. 
        Intraday: Fresh fetch (due to YF limits).
        """
        file_path = self._get_file_path(ticker, interval)
        
        # --- INTRADAY HANDLING (Fresh Fetch) ---
        if interval != '1d':
            # Limit periods based on YF constraints
            period = "max"
            if interval in ['1m', '2m', '5m', '15m', '30m', '90m']:
                period = "59d"
            elif interval in ['60m', '1h']:
                period = "729d"
                
            print(f"📥 Fetching fresh intraday data ({interval}) for {ticker}...")
            try:
                new_data = yf.Ticker(ticker).history(period=period, interval=interval)
                if not new_data.empty:
                    new_data.index = pd.to_datetime(new_data.index).tz_localize(None)
                    new_data.columns = [c.lower() for c in new_data.columns]
                    new_data = new_data[['open', 'high', 'low', 'close', 'volume']]
                    new_data.to_parquet(file_path)
                    print(f"✅ Updated {ticker} ({interval}). Rows: {len(new_data)}")
                    return new_data
                return pd.DataFrame()
            except Exception as e:
                print(f"❌ Error fetching intraday data: {e}")
                return pd.DataFrame()

        # --- DAILY HANDLING (Incremental) ---
        # 1. Load existing data
        if os.path.exists(file_path):
            try:
                existing_data = pd.read_parquet(file_path)
                last_date = existing_data.index[-1]
                # Start download from next day
                start_date = last_date + timedelta(days=1)
                
                # Check if start_date is in the future
                if start_date > datetime.now():
                    print(f"✅ Data for {ticker} is up to date (Last: {last_date}).")
                    return existing_data
            except Exception as e:
                print(f"Error reading {file_path}, re-downloading full history: {e}")
                existing_data = None
                start_date = None
        else:
            existing_data = None
            start_date = None
            
        # 2. Download new data
        print(f"📥 Fetching data for {ticker} starting from {start_date if start_date else '10y ago'}...")
        
        try:
            if start_date:
                # Format for yfinance
                new_data = yf.Ticker(ticker).history(start=start_date, interval=interval)
            else:
                new_data = yf.Ticker(ticker).history(period="10y", interval=interval)
                
            # Clean new data
            if not new_data.empty:
                new_data.index = pd.to_datetime(new_data.index).tz_localize(None)
                new_data.columns = [c.lower() for c in new_data.columns]
                new_data = new_data[['open', 'high', 'low', 'close', 'volume']]
                
                # 3. Merge
                if existing_data is not None and not existing_data.empty:
                    # Combine
                    combined = pd.concat([existing_data, new_data])
                    # Deduplicate by index
                    combined = combined[~combined.index.duplicated(keep='last')]
                    combined.sort_index(inplace=True)
                    final_data = combined
                else:
                    final_data = new_data
                    
                # 4. Save
                final_data.to_parquet(file_path)
                print(f"✅ Updated {ticker}. Total rows: {len(final_data)}. Last: {final_data.index[-1]}")
                return final_data
            
            else:
                print(f"No new data found for {ticker}.")
                return existing_data if existing_data is not None else pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error updating data: {e}")
            return existing_data if existing_data is not None else pd.DataFrame()
    
    def get_data(self, ticker: str, limit: int = 500, interval: str = '1d') -> pd.DataFrame:
        """Get the last N rows of data from local storage"""
        file_path = self._get_file_path(ticker, interval)
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            return df.tail(limit)
        else:
            # Fallback: force update if no file
            return self.update_data(ticker, interval).tail(limit)

class TradeStateManager:
    def __init__(self, state_file: str = "trade_state.json"):
        self.state_file = state_file
        
    def load_state(self) -> dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                return self._get_default_state()
        return self._get_default_state()
    
    def save_state(self, state: dict):
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=4, default=str)
            
    def _get_default_state(self):
        return {
            "status": "FLAT", # or LONG
            "ticker": None,
            "entry_price": 0.0,
            "entry_date": None,
            "highest_price": 0.0, # For trailing stops
            "strategy_id": None,
            "pnl": 0.0
        }

class StrategyFileManager:
    """Manages saving and loading of full strategy bundles."""
    def __init__(self, base_dir: str = "strategies"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
    def save_strategy_bundle(self, name: str, config: dict, active_models: dict) -> str:
        """
        Saves a full strategy bundle.
        active_models: dict of {'base': path, 'meta': path, 'alpha': path, 'dl': path}
        """
        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = name.replace(" ", "_")
        folder_name = f"{safe_name}_{timestamp}"
        bundle_path = os.path.join(self.base_dir, folder_name)
        os.makedirs(bundle_path, exist_ok=True)
        
        # 1. Save Config
        # Update model paths in config to be relative to the bundle (just filenames)
        bundle_config = config.copy()
        bundle_config['strategy_name'] = name
        bundle_config['created_at'] = timestamp
        
        # 2. Copy Models
        for model_type, src_path in active_models.items():
            if src_path and os.path.exists(src_path):
                filename = os.path.basename(src_path)
                dst_path = os.path.join(bundle_path, filename)
                shutil.copy2(src_path, dst_path)
                
                # Update config keys to point to BUNDLED file (filename only)
                if model_type == 'base': bundle_config['model_path'] = filename
                elif model_type == 'meta': bundle_config['meta_model_path'] = filename
                elif model_type == 'alpha': bundle_config['alpha_model_path'] = filename
                elif model_type == 'dl': bundle_config['dl_model_path'] = filename
        
        # Save updated config
        config_path = os.path.join(bundle_path, "strategy_config.json")
        with open(config_path, 'w') as f:
            json.dump(bundle_config, f, indent=4)
            
        return bundle_path

    def list_strategies(self):
        """Returns list of available strategy bundles."""
        strategies = []
        if not os.path.exists(self.base_dir):
            return []
            
        for d in sorted(os.listdir(self.base_dir), reverse=True):
            path = os.path.join(self.base_dir, d)
            if os.path.isdir(path):
                # Check for config
                if os.path.exists(os.path.join(path, "strategy_config.json")):
                    strategies.append(d)
        return strategies
