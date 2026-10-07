import time
import os
# Suppress TensorFlow/ABSL warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import sys

# Import our custom modules
from data_manager import DataManager, TradeStateManager
from alert_system import AlertManager
from polygon_manager import PolygonManager
from ml_utils import generate_features, generate_signals, calculate_metrics_from_signals, apply_trend_filter, apply_meta_filter

# Optional Imports
try:
    from conformal_utils import ConformalPredictionWrapper
except ImportError:
    ConformalPredictionWrapper = None

try:
    from regime_utils import MarketRegimeDetector
except ImportError:
    MarketRegimeDetector = None

# Configuration
CHECK_INTERVAL_SECONDS = 60 * 15 # 15 Minutes
MODEL_PATH = "saved_models_v2/xgboost_BTC-USD_latest.joblib" # Placeholder
STRATEGY_CONFIG_FILE = "strategy_config.json"

# Alert Config (Replace with your actual URLs/Credentials or load from env)
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")

def load_strategy_config():
    if os.path.exists(STRATEGY_CONFIG_FILE):
        with open(STRATEGY_CONFIG_FILE, 'r') as f:
            return json.load(f)
    return None

def select_model_interactive(default_path=None):
    """
    Allow user to select a Bundle or Model.
    If Bundle selected -> updates root strategy_config.json and returns base model path.
    """
    model_dir = "saved_models_v2"
    strategy_dir = "strategies"
    
    # Collect Bundles
    bundles = []
    if os.path.exists(strategy_dir):
        bundles = sorted([d for d in os.listdir(strategy_dir) 
                         if os.path.isdir(os.path.join(strategy_dir, d)) 
                         and os.path.exists(os.path.join(strategy_dir, d, "strategy_config.json"))], reverse=True)

    # Collect Loose Models
    models = []
    if os.path.exists(model_dir):
        models = sorted([f for f in os.listdir(model_dir) if f.endswith('.joblib')], reverse=True)
    
    if not bundles and not models:
        return default_path
        
    print("\n📦 Available Strategy Bundles:")
    for i, b in enumerate(bundles):
        print(f"  {i+1}. 📦 {b}")
        
    print("\n💾 Individual Models:")
    offset = len(bundles)
    for i, m in enumerate(models):
        print(f"  {offset+i+1}. {m}")
        
    print(f"\n  0. Use Configured/Default: {os.path.basename(default_path) if default_path else 'None'}")
    
    try:
        choice = input("\nSelect # (or Press Enter): ").strip()
        if not choice or choice == '0':
            return default_path
        
        idx = int(choice) - 1
        
        # Bundle Selected
        if 0 <= idx < len(bundles):
            bundle_name = bundles[idx]
            bundle_path = os.path.join(strategy_dir, bundle_name)
            print(f"✅ Selected Bundle: {bundle_name}")
            
            # Load Bundle Config
            bundle_conf_path = os.path.join(bundle_path, "strategy_config.json")
            with open(bundle_conf_path, 'r') as f:
                b_conf = json.load(f)
            
            # Update Root Config with Bundle Paths
            # Base Model
            base_name = os.path.basename(b_conf.get('model_path'))
            b_conf['model_path'] = f"{bundle_path}/{base_name}"
            
            # DL
            if b_conf.get('dl_model_path'):
                b_conf['dl_model_path'] = f"{bundle_path}/{os.path.basename(b_conf['dl_model_path'])}"
                
            # Meta
            if b_conf.get('meta_model_path'):
                b_conf['meta_model_path'] = f"{bundle_path}/{os.path.basename(b_conf['meta_model_path'])}"
                
            # Alpha
            if b_conf.get('alpha_model_path'):
                b_conf['alpha_model_path'] = f"{bundle_path}/{os.path.basename(b_conf['alpha_model_path'])}"
            
            # Save to Root Config (Overwriting current!)
            with open(STRATEGY_CONFIG_FILE, 'w') as f:
                json.dump(b_conf, f, indent=4)
                
            print(f"🔄 Updated {STRATEGY_CONFIG_FILE} to point to bundle.")
            return b_conf['model_path']

        # Loose Model Selected
        elif len(bundles) <= idx < len(bundles) + len(models):
            m_idx = idx - len(bundles)
            selected = os.path.join(model_dir, models[m_idx])
            print(f"✅ Selected Model: {selected}")
            return selected
            
    except Exception as e:
        print(f"Selection Error: {e}")
        pass
        
    return default_path

def send_status_update(ticker, df_window, features_window, signals_window, probs_window, alert_mgr, title="📊 Status Update", meta_signals=None, meta_equity=None, extra_fields=None):
    """Helper to send rich status dashboard"""
    # Calculate Strategy Metrics
    metrics = calculate_metrics_from_signals(df_window, signals_window)
    eq_curve = metrics['equity_curve']
    
    # Format Fields
    fields = [
        {'name': 'Price', 'value': f"${df_window['close'].iloc[-1]:,.2f}", 'inline': True},
        {'name': 'Window Return', 'value': f"{metrics['total_return']:+.1f}%", 'inline': True},
        {'name': 'Win Rate', 'value': f"{metrics['win_rate']:.0f}% ({metrics['num_trades']} trds)", 'inline': True},
        {'name': 'Max Drawdown', 'value': f"{metrics['max_drawdown']:.1f}%", 'inline': True},
        {'name': 'Confidence', 'value': f"{probs_window['prob_buy'].iloc[-1]:.1%}", 'inline': True}
    ]
    
    if extra_fields:
        fields.extend(extra_fields)
    
    # Send
    alert_mgr.send_alert(
        title=f"{title}: {ticker}",
        message=f"System Running | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        data=df_window,
        ticker=ticker,
        color=0x3498db, # Blue
        fields=fields,
        chart_lookback=len(df_window),
        equity_curve=eq_curve,
        features=features_window,
        probs=probs_window,
        signals=signals_window,
        meta_signals=meta_signals,
        meta_equity_curve=meta_equity
    )

def main():
    print(" Starting Live Trader...")
    
    # 1. Initialize Managers
    data_mgr = DataManager()
    state_mgr = TradeStateManager()
    alert_mgr = AlertManager(discord_webhook_url=DISCORD_WEBHOOK)
    poly_mgr = PolygonManager(api_key=POLYGON_API_KEY)
    
    # 2. Load Configuration
    config = load_strategy_config()
    if not config:
        print(f" {STRATEGY_CONFIG_FILE} not found. Using defaults.")
        config = {'ticker': 'BTC-USD'}
        
    ticker = config.get('ticker', 'BTC-USD')
    interval = config.get('interval', '1d')
    
    # Determine Sleep Interval
    # We set this to 60s to ensure we hit scheduled alert windows (e.g. 11:00-11:15)
    # The data manager handles caching so we don't spam requests.
    sleep_seconds = 60 
    
    model_path = config.get('model_path')
    
    # Interactive Selection (Override config)
    model_path = select_model_interactive(model_path)
    
    if not model_path or not os.path.exists(model_path):
        print(f" Model not found at {model_path}. Please check configuration.")
        return

    # 3. Load Model
    print(f" Loading model from {model_path}...")
    model_data = joblib.load(model_path)
    
    print(f" Watching {ticker} ({interval}). Interval: {sleep_seconds}s")
    print(f"   Strategy: Buy > {config.get('buy_threshold', 0.5)}, Sell > {config.get('sell_threshold', 0.5)}")
    
    # Load Meta-Model if configured
    meta_model_data = None
    if config.get('use_meta_filter') and config.get('meta_model_path'):
        m_path = config.get('meta_model_path')
        if os.path.exists(m_path):
            print(f" Loading Meta-Model from {m_path}...")
            try:
                meta_model_data = joblib.load(m_path)
            except Exception as e:
                print(f" Failed to load Meta-Model: {e}")
                
    # Load Alpha Regressors
    alpha_path = config.get('alpha_model_path', f"saved_models_v2/alpha_regressors_{ticker}.joblib")
    alpha_models = None
    last_alpha_mtime = 0
    
    if os.path.exists(alpha_path):
        last_alpha_mtime = os.path.getmtime(alpha_path)
        try:
            print(f" Loading Alpha Models from {alpha_path}...")
            alpha_models = joblib.load(alpha_path)
        except Exception as e:
            print(f" Failed to load Alpha Models: {e}")
    else:
        print(f" ⚠️ Alpha Models not found: {alpha_path}")
        print(f"    (Sniper/Range Strategy Disabled. Train 'Range Predictor' in Alpha Lab tab to enable)")
    
    # Track config file changes
    last_config_mtime = 0
    if os.path.exists(STRATEGY_CONFIG_FILE):
        last_config_mtime = os.path.getmtime(STRATEGY_CONFIG_FILE)
    
    startup_sent = False
    daily_alerts = set()
    
    # 4. Main Loop
    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{current_time}] Update Cycle...")
            
            # Reload Config ONLY if file changed
            if os.path.exists(STRATEGY_CONFIG_FILE):
                current_mtime = os.path.getmtime(STRATEGY_CONFIG_FILE)
                if current_mtime > last_config_mtime:
                    print("📂 Strategy Config changed on disk. Reloading...")
                    new_config = load_strategy_config()
                    last_config_mtime = current_mtime
                    
                    if new_config:
                        # Update thresholds dynamically
                        config = new_config
                        
                        # Update Interval & Sleep
                        interval = config.get('interval', '1d')
                        # Force 60s sleep for accurate scheduling
                        sleep_seconds = 60
                        
                        # Check if model path changed in config AND user wants to switch?
                        # Since user manually selected a model at start, strictly following config now 
                        # might override manual choice. But "Config Changed" implies user explicitly exported.
                        # So we SHOULD switch if config changed.
                        
                        if config.get('model_path') != model_path:
                            new_path = config.get('model_path')
                            print(f"🔄 Switching Model to: {new_path}")
                            try:
                                if os.path.exists(new_path):
                                    model_data = joblib.load(new_path)
                                    model_path = new_path
                                    print("✅ Model reloaded successfully.")
                                    alert_mgr.send_alert("Model Switched", f"Now running: {os.path.basename(model_path)}", color=0x9b59b6)
                                else:
                                    print(f"❌ New model path does not exist: {new_path}")
                            except Exception as e:
                                print(f"❌ Failed to reload model: {e}")
                        
                        # Check Polygon Key Update
                        if config.get('polygon_api_key') and config.get('polygon_api_key') != poly_mgr.api_key:
                            print("🔑 Polygon Key updated. Re-initializing manager...")
                            poly_mgr = PolygonManager(api_key=config.get('polygon_api_key'))

                        # Check Meta Model Update
                        use_meta = config.get('use_meta_filter', False)
                        meta_path = config.get('meta_model_path')
                        
                        if use_meta:
                            if meta_path and os.path.exists(meta_path):
                                try:
                                    print(f"🧠 Reloading Meta-Model from {meta_path}...")
                                    meta_model_data = joblib.load(meta_path)
                                except Exception as e:
                                    print(f"⚠️ Failed to reload Meta-Model: {e}")
                        else:
                            if meta_model_data is not None:
                                print("🧠 Meta-Model Disabled.")
                                meta_model_data = None

            # Check for Alpha Model Updates (Hot Reload)
            alpha_path = config.get('alpha_model_path', f"saved_models_v2/alpha_regressors_{ticker}.joblib")
            if os.path.exists(alpha_path):
                current_alpha_mtime = os.path.getmtime(alpha_path)
                if current_alpha_mtime > last_alpha_mtime:
                     print(f"🎯 Alpha Models changed on disk. Reloading...")
                     try:
                         alpha_models = joblib.load(alpha_path)
                         last_alpha_mtime = current_alpha_mtime
                         print("✅ Alpha Models reloaded successfully.")
                     except Exception as e:
                         print(f"⚠️ Failed to reload Alpha Models: {e}")
            
            # A. Update Data
            df = data_mgr.update_data(ticker, interval=interval)
            if df.empty:
                print("⚠️ No data received. Retrying next cycle.")
                time.sleep(60)
                continue
                
            # B. Generate Features (Use last 500 rows for speed, but enough for lags)
            # Ensure we have enough data for feature gen (DL needs 30, SMA needs 50)
            df_window = df.tail(500).copy()
            
            # Check for DL
            use_dl = model_data.get('use_dl', False)
            # Fallback inference
            if not use_dl and 'feature_names' in model_data:
                if any('DL_pred_' in f for f in model_data['feature_names']):
                    use_dl = True
            
            dl_config = None
            if use_dl:
                # Prioritize config path, else default
                dl_path = config.get('dl_model_path', f"saved_models_v2/dl_{ticker}.h5")
                dl_config = {'train': False, 'model_path': dl_path}
            
            features = generate_features(df_window, dl_config=dl_config, use_regime=config.get('use_regime_detection', False))
            
            if features.empty:
                print("⚠️ Feature generation returned empty. Skipping.")
                time.sleep(60)
                continue
                
            # C. Generate Signals
            # We only care about the LAST row (the most recent candle)
            # BUT generate_signals needs context for some logic (slope), so we pass full dataframe
            signals, prob_df = generate_signals(
                model_data, features, 
                threshold=0.5, # Base threshold, overridden by config
                use_slope_signals=config.get('use_slope_signals', False),
                slope_buy_thresh=config.get('slope_buy_thresh', 0.0),
                slope_sell_thresh=config.get('slope_sell_thresh', 0.0),
                use_ml_confirm=config.get('use_ml_confirm', False),
                ml_confirm_thresh=config.get('ml_confirm_thresh', 0.3),
                slope_roc_thresh=config.get('slope_roc_thresh', 0.0),
                use_mom_zone=config.get('use_mom_zone', False),
                buy_threshold=config.get('buy_threshold', 0.6),
                sell_threshold=config.get('sell_threshold', 0.6)
            )
            
            # Apply Trend Filter (Base + Trend)
            s_raw = signals.copy()
            if config.get('use_trend_filter', True):
                s_raw = apply_trend_filter(s_raw, df_window, features)
            
            # Apply Conformal Prediction Filter (New)
            if config.get('use_conformal', False) and ConformalPredictionWrapper:
                cp_alpha = config.get('cp_alpha', 0.1)
                print(f"🔬 Applying Conformal Filter (Alpha={cp_alpha})...")
                
                try:
                    # We need the base estimator from model_data
                    # model_data is a dict loaded from joblib
                    base_model = model_data.get('model')
                    if base_model:
                        # Initialize Wrapper
                        # We use cv=5 to calibrate on the historical window we have loaded
                        cp = ConformalPredictionWrapper(base_model, method="score", cv=5)
                        
                        # Generate labels for calibration (peaks/valleys on window)
                        from ml_utils import detect_peaks_valleys
                        cal_labels, _ = detect_peaks_valleys(df_window, order=5) # Default order
                        
                        # Align
                        common_idx = features.index.intersection(cal_labels.index)
                        X_cal = features.loc[common_idx]
                        y_cal = cal_labels.loc[common_idx]
                        
                        # Clean
                        valid_mask = X_cal.notna().all(axis=1) & y_cal.notna()
                        X_cal = X_cal[valid_mask]
                        y_cal = y_cal[valid_mask]
                        
                        if len(X_cal) > 50: # Minimum data for calibration
                            cp.fit(X_cal, y_cal)
                            
                            # Filter the signals
                            # We only really care about filtering the *last* signal, but we filter all for consistency
                            # Align X for prediction
                            X_pred = features.loc[s_raw.index].fillna(0)
                            s_raw = cp.filter_signals(s_raw, X_pred, alpha=cp_alpha)
                            print(f"   ✅ CP Filter applied.")
                        else:
                            print("   ⚠️ Not enough data to calibrate CP. Skipping filter.")
                    else:
                        print("   ⚠️ Base model not found in model_data. Skipping CP.")
                except Exception as e:
                    print(f"   ❌ CP Filter Failed: {e}")

            # Apply Meta-Filter (Meta)
            s_meta = s_raw.copy()
            has_meta = config.get('use_meta_filter', False) and meta_model_data is not None
            meta_thresh = config.get('meta_threshold', 0.5)
            if has_meta:
                s_meta = apply_meta_filter(meta_model_data, s_meta, prob_df, features, threshold=meta_thresh)
            
            # Slice for Analysis (Window)
            window_size = 180
            if len(features) > window_size:
                f_window = features.iloc[-window_size:]
                p_window = prob_df.iloc[-window_size:]
                d_window = df.iloc[-window_size:]
                s_raw_win = s_raw.iloc[-window_size:]
                s_meta_win = s_meta.iloc[-window_size:]
            else:
                f_window = features
                p_window = prob_df
                d_window = df
                s_raw_win = s_raw
                s_meta_win = s_meta
            
            # Calculate Comparison Metrics
            metrics_raw = calculate_metrics_from_signals(d_window, s_raw_win)
            metrics_meta = calculate_metrics_from_signals(d_window, s_meta_win)
            
            # Decide active signal
            active_metrics = metrics_raw
            meta_equity = None
            active_signal_source = s_raw_win
            meta_status_msg = ""
            
            if has_meta:
                meta_ret = metrics_meta['total_return']
                raw_ret = metrics_raw['total_return']
                meta_equity = metrics_meta['equity_curve']
                
                if meta_ret > raw_ret:
                    active_signal_source = s_meta_win
                    active_metrics = metrics_meta
                    meta_status_msg = f"✨ Meta-Model Active (Return: {meta_ret:+.1f}% vs Raw: {raw_ret:+.1f}%)"
                else:
                    active_signal_source = s_raw_win
                    active_metrics = metrics_raw
                    meta_status_msg = f"⚠️ Raw Model Active (Return: {raw_ret:+.1f}% vs Meta: {meta_ret:+.1f}%)"
            else:
                meta_status_msg = "⚪ Meta-Model Disabled"
            
            # Extract latest signal from ACTIVE source
            last_idx = features.index[-1]
            last_signal = active_signal_source.iloc[-1]
            last_prob_buy = prob_df.iloc[-1]['prob_buy']
            last_prob_sell = prob_df.iloc[-1]['prob_sell']
            last_close = df_window.loc[last_idx, 'close']
            
            print(f"   Last Close: ${last_close:.2f} | Signal: {last_signal} | {meta_status_msg}")
            
            # Alpha Sniper Logic
            sniper_fields = []
            if alpha_models:
                try:
                    # Fetch Daily Data for prediction (Regressors trained on Daily)
                    # We use update_data to ensure we have the latest daily candle
                    df_daily = data_mgr.update_data(ticker, interval='1d')
                    if not df_daily.empty:
                        # Generate features
                        f_daily = generate_features(df_daily)
                        # We use YESTERDAY's features (last completed day) to predict TODAY's range
                        if len(f_daily) >= 2:
                            feat_row = f_daily.iloc[[-2]]
                            
                            # Predict
                            pred_h = alpha_models['high']['model'].predict(feat_row)[0]
                            pred_l = alpha_models['low']['model'].predict(feat_row)[0]
                            
                            # Base Price: Today's Open (Last row of daily data)
                            today_open = df_daily.iloc[-1]['open']
                            
                            buy_lim = today_open * (1 + pred_l)
                            sell_lim = today_open * (1 + pred_h)
                            
                            sniper_fields = [
                                {'name': '🎯 Sniper Buy', 'value': f"${buy_lim:,.2f}", 'inline': True},
                                {'name': '🎯 Sniper Sell', 'value': f"${sell_lim:,.2f}", 'inline': True}
                            ]
                            print(f"   🎯 Sniper Limits: Buy ${buy_lim:.2f} | Sell ${sell_lim:.2f}")
                except Exception as e:
                    print(f"   ⚠️ Alpha Prediction Error: {e}")

            # D. State Logic
            state = state_mgr.load_state()
            current_status = state.get('status', 'FLAT')
            
            # Fetch Sentiment & Stats if Signal Active
            sent_data = {}
            pct_chg = 0.0
            rsi_val = "N/A"
            
            eq_curve = active_metrics['equity_curve']
            
            # Format Strategy Stats
            strat_fields = [
                {'name': 'Window Return', 'value': f"{active_metrics['total_return']:+.1f}%", 'inline': True},
                {'name': 'Win Rate', 'value': f"{active_metrics['win_rate']:.0f}% ({active_metrics['num_trades']} trds)", 'inline': True},
                {'name': 'Max Drawdown', 'value': f"{active_metrics['max_drawdown']:.1f}%", 'inline': True}
            ]
            
            if has_meta:
                strat_fields.append({'name': 'Raw vs Meta', 'value': f"{metrics_raw['total_return']:+.1f}% / {metrics_meta['total_return']:+.1f}%", 'inline': True})
            
            if sniper_fields:
                strat_fields.extend(sniper_fields)

            # --- STATUS UPDATES (Startup + Daily Schedule) ---
            
            # 1. Startup Alert
            if not startup_sent:
                print("🔔 Sending Startup Dashboard...")
                send_status_update(ticker, d_window, f_window, active_signal_source, p_window, alert_mgr, 
                                 "🚀 Startup Status", meta_signals=s_meta_win, meta_equity=meta_equity, extra_fields=sniper_fields)
                startup_sent = True
                
            # 2. Scheduled Alerts
            now = datetime.now()
            day_str = now.strftime("%Y-%m-%d")
            hm = now.strftime("%H:%M")
            
            # Schedule (Local Time)
            if "08:30" <= hm <= "08:45":
                key = f"{day_str}_OPEN"
                if key not in daily_alerts:
                    send_status_update(ticker, d_window, f_window, active_signal_source, p_window, alert_mgr, 
                                     "🔔 Market Open Update", meta_signals=s_meta_win, meta_equity=meta_equity, extra_fields=sniper_fields)
                    daily_alerts.add(key)
                    
            if "11:00" <= hm <= "11:15":
                key = f"{day_str}_MID"
                if key not in daily_alerts:
                    send_status_update(ticker, d_window, f_window, active_signal_source, p_window, alert_mgr, 
                                     "☀️ Mid-Day Update", meta_signals=s_meta_win, meta_equity=meta_equity, extra_fields=sniper_fields)
                    daily_alerts.add(key)
                    
            if "15:00" <= hm <= "15:15":
                key = f"{day_str}_CLOSE"
                if key not in daily_alerts:
                    send_status_update(ticker, d_window, f_window, active_signal_source, p_window, alert_mgr, 
                                     "🏁 Market Close Update", meta_signals=s_meta_win, meta_equity=meta_equity, extra_fields=sniper_fields)
                    daily_alerts.add(key)

            # Hourly Updates (09:00 - 16:00)
            # Send at top of hour, excluding overlap with Mid (11) and Close (15)
            hour = now.hour
            minute = now.minute
            if 9 <= hour <= 16 and minute < 5:
                if hour not in [11, 15]: # Skip 11:00 and 15:00 as they have special alerts
                    key = f"{day_str}_HOUR_{hour}"
                    if key not in daily_alerts:
                        send_status_update(ticker, d_window, f_window, active_signal_source, p_window, alert_mgr, 
                                         f"⏱️ {hour}:00 Market Update", meta_signals=s_meta_win, meta_equity=meta_equity, extra_fields=sniper_fields)
                        daily_alerts.add(key)
            
            # Midnight Update
            if hour == 0 and minute < 5:
                key = f"{day_str}_MIDNIGHT"
                if key not in daily_alerts:
                    send_status_update(ticker, d_window, f_window, active_signal_source, p_window, alert_mgr, 
                                     "🌙 Midnight Update", meta_signals=s_meta_win, meta_equity=meta_equity, extra_fields=sniper_fields)
                    daily_alerts.add(key)

            if last_signal != 0:
                # Calc 24h Change
                if len(df_window) > 1:
                    prev_close = df_window['close'].iloc[-2]
                    pct_chg = (last_close - prev_close) / prev_close * 100
                
                # Get RSI
                if 'rsi_14' in features.columns:
                    rsi_val = f"{features['rsi_14'].iloc[-1]:.1f}"
                    
                try:
                    print("   Checking Polygon Sentiment...")
                    sent_data = poly_mgr.get_options_sentiment(ticker)
                except Exception as e:
                    print(f"   Polygon Error: {e}")
            
            alert_sent = False
            
            # BUY LOGIC
            if last_signal == 1 and current_status == 'FLAT':
                print(" BUY SIGNAL DETECTED!")
                if sent_data:
                    print(f"   Options Sentiment: {sent_data.get('sentiment')} (PCR: {sent_data.get('pcr_volume')})")
                
                # Update State
                state['status'] = 'LONG'
                state['entry_price'] = float(last_close)
                state['entry_date'] = str(last_idx)
                state['ticker'] = ticker
                state_mgr.save_state(state)
                
                # Prepare Alert
                warn_msg = ""
                if sent_data.get('sentiment') == 'BEARISH':
                    warn_msg = "\n WARNING: High Put/Call Ratio (Bearish Flows)"
                    
                fields = [
                    {'name': 'Price', 'value': f"${last_close:,.2f}", 'inline': True},
                    {'name': '24h Change', 'value': f"{pct_chg:+.2f}%", 'inline': True},
                    {'name': 'RSI (14)', 'value': str(rsi_val), 'inline': True},
                    {'name': 'Confidence', 'value': f"{last_prob_buy:.1%}", 'inline': True}
                ]
                if 'pcr_volume' in sent_data:
                    fields.append({'name': 'Put/Call Ratio', 'value': f"{sent_data['pcr_volume']} ({sent_data['sentiment']})", 'inline': True})
                
                # Add Strategy Stats
                fields.extend(strat_fields)
                
                if has_meta:
                    fields.append({'name': 'Strategy Status', 'value': meta_status_msg, 'inline': False})

                # Send Alert
                alert_mgr.send_alert(
                    title=f"🚀 [ML] BUY {ticker}",
                    message=f"Entry Signal Detected @ ${last_close:,.2f}\nConfidence: {last_prob_buy:.1%}{warn_msg}",
                    data=df_window,
                    ticker=ticker,
                    color=0x00ff00, # Green
                    fields=fields,
                    chart_lookback=180,
                    equity_curve=eq_curve,
                    features=f_window,
                    probs=p_window,
                    signals=active_signal_source,
                    meta_signals=s_meta_win,
                    meta_equity_curve=meta_equity
                )
                alert_sent = True
                
            # SELL LOGIC
            elif last_signal == -1 and current_status == 'LONG':
                print("🔴 SELL SIGNAL DETECTED!")
                
                entry_price = state.get('entry_price', last_close)
                pnl_pct = (last_close - entry_price) / entry_price * 100
                
                # Update State
                state['status'] = 'FLAT'
                state['exit_price'] = float(last_close)
                state['last_pnl'] = pnl_pct
                state_mgr.save_state(state)
                
                fields = [
                    {'name': 'Exit Price', 'value': f"${last_close:,.2f}", 'inline': True},
                    {'name': 'Entry Price', 'value': f"${entry_price:,.2f}", 'inline': True},
                    {'name': 'PnL', 'value': f"{pnl_pct:+.2f}%", 'inline': True},
                    {'name': '24h Change', 'value': f"{pct_chg:+.2f}%", 'inline': True},
                    {'name': 'RSI', 'value': str(rsi_val), 'inline': True}
                ]
                
                if 'pcr_volume' in sent_data:
                    fields.append({'name': 'PCR', 'value': f"{sent_data['pcr_volume']}", 'inline': True})

                # Add Strategy Stats
                fields.extend(strat_fields)
                
                if has_meta:
                    fields.append({'name': 'Strategy Status', 'value': meta_status_msg, 'inline': False})

                alert_mgr.send_alert(
                    title=f"📉 [ML] SELL {ticker}",
                    message=f"Exit Signal Detected @ ${last_close:,.2f}\nProfit: {pnl_pct:+.2f}%",
                    data=df_window,
                    ticker=ticker,
                    color=0xff0000, # Red
                    fields=fields,
                    chart_lookback=180,
                    equity_curve=eq_curve,
                    features=f_window,
                    probs=p_window,
                    signals=active_signal_source,
                    meta_signals=s_meta_win,
                    meta_equity_curve=meta_equity
                )
                alert_sent = True
                
            # STOP LOSS / TAKE PROFIT CHECK (Optional)
            # If enabled in config
            if current_status == 'LONG':
                stop_loss_pct = config.get('stop_loss_pct', 0.0)
                take_profit_pct = config.get('take_profit_pct', 0.0)
                entry_price = state.get('entry_price', last_close)
                current_pnl = (last_close - entry_price) / entry_price
                
                exit_triggered = False
                exit_reason = ""
                
                if stop_loss_pct > 0 and current_pnl < -stop_loss_pct:
                    exit_triggered = True
                    exit_reason = "Stop Loss Hit"
                elif take_profit_pct > 0 and current_pnl > take_profit_pct:
                    exit_triggered = True
                    exit_reason = "Take Profit Hit"
                    
                if exit_triggered:
                    print(f"⚠️ {exit_reason}!")
                    state['status'] = 'FLAT'
                    state_mgr.save_state(state)
                    
                    alert_mgr.send_alert(
                        title=f"🛡️ [ML] {exit_reason}: {ticker}",
                        message=f"Exiting @ ${last_close:.2f}\nPnL: {current_pnl*100:+.2f}%",
                        data=df_window,
                        ticker=ticker,
                        color=0xffaa00, # Orange
                         fields=[
                            {'name': 'Exit Price', 'value': f"${last_close:.2f}", 'inline': True},
                            {'name': 'Reason', 'value': exit_reason, 'inline': True}
                        ]
                    )

            if not alert_sent:
                print("   No action taken.")
                
            # Wait for next cycle
            time.sleep(sleep_seconds)
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping Trader...")
            break
        except Exception as e:
            print(f"\n❌ Error in Main Loop: {e}")
            print("Retrying in 60s...")
            time.sleep(60)

if __name__ == "__main__":
    main()
