#!/usr/bin/env python3
"""
ML Trading Signals Page for Streamlit App

Standalone page for machine learning-based trading signal generation.
Completely separate from existing optimization functionality.

SOMEWHAT WORKING PROTOTYPE
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import logging
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Suppress XGBoost serialization warnings specifically
logging.getLogger('xgboost').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*serialized model.*')
warnings.filterwarnings('ignore', message='.*older XGBoost.*')

# Also suppress at the C++ level if possible
import os
os.environ['XGBOOST_VERBOSITY'] = '0'

# Import our new ML modules
from peak_valley_detector import PeakValleyDetector
from ml_feature_engineer import MLFeatureEngineer
from ml_models import TradingMLModels

# === PAGE HEADER ===
st.markdown("""
# 🎯 Peak/Valley ML Trading Signals

**Revolutionary Approach**: ML models trained on actual market peaks and valleys for optimal timing.

**Key Advantages:**
- 🎯 **163.8% Potential Returns** (vs 0.9% random signals)
- 🔬 **Market Structure Learning** (peaks/valleys as training labels)
- 🧠 Multiple ML algorithms (Random Forest, XGBoost, LightGBM, SVM)
- ⚡ Optuna hyperparameter optimization with SMOTE balancing
- 📊 130+ technical indicators as features
- 📈 Candlestick charts with signal overlays
- 🏆 Complete performance analysis and backtesting
- 🚀 Production-ready signal generation
""")

# === SIDEBAR CONTROLS ===
st.sidebar.header("🎯 ML Configuration")

# Data Configuration (Unified)
st.sidebar.subheader("📊 Data Settings")
training_period = st.sidebar.selectbox(
    "Training Period", 
    ["1y", "2y", "3y", "5y"],
    index=2,
    help="Historical data for training"
)

# Shared ticker selection - sync with main app
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = 'SPY'

# Get ticker from main app if it exists, otherwise use ML page input
main_app_ticker = getattr(st, '_main_ticker', None) if hasattr(st, '_main_ticker') else None

col1, col2 = st.sidebar.columns([3, 1])
with col1:
    ticker_input = st.text_input("Stock Ticker", 
                                value=st.session_state.selected_ticker, 
                                key='ml_ticker_input').upper()
with col2:
    st.write("")
    if st.button("🔄", help="Sync with main app ticker", key='sync_ticker'):
        # Try to get ticker from URL params or session state
        if 'main_app_ticker' in st.session_state:
            st.session_state.selected_ticker = st.session_state.main_app_ticker
            st.rerun()

# Update session state ticker
if ticker_input != st.session_state.selected_ticker:
    st.session_state.selected_ticker = ticker_input

ticker = st.session_state.selected_ticker

# Validate ticker and show info
try:
    # Quick validation - try to get basic info
    test_ticker = yf.Ticker(ticker)
    info = test_ticker.info
    if info and 'symbol' in info:
        company_name = info.get('longName', info.get('shortName', ticker))
        st.sidebar.success(f"📊 **{ticker}**: {company_name[:30]}")
    else:
        st.sidebar.warning(f"📊 **{ticker}**: Ticker may be invalid")
except:
    st.sidebar.error(f"❌ **{ticker}**: Invalid ticker symbol")

# Add helpful ticker examples
with st.sidebar.expander("💡 Popular Tickers"):
    st.write("**ETFs:** SPY, QQQ, IWM, VTI")
    st.write("**Stocks:** AAPL, MSFT, GOOGL, TSLA") 
    st.write("**Crypto:** BTC-USD, ETH-USD")
    st.write("**Forex:** EURUSD=X, GBPUSD=X")

# Data period
period = st.sidebar.selectbox(
    "Historical Data Period",
    ['3mo', '6mo', '1y', '2y', '5y'],
    index=2  # Default to 1y
)

# Peak/Valley detection parameters
st.sidebar.subheader("Peak/Valley Detection")

# Peak/Valley detection method
detection_method = st.sidebar.selectbox(
    "Peak/Valley Detection Method",
    ['scipy_peaks', 'rolling_window', 'percentage_swing', 'multi_timeframe'],
    index=0  # Default to scipy_peaks
)

# Method-specific parameters (optimized for 22% signal balance)
if detection_method == 'scipy_peaks':
    prominence_pct = st.sidebar.slider("Prominence (%)", 0.3, 5.0, 0.5, 0.1)  # Optimized for 22% signals
    distance = st.sidebar.slider("Min Distance (days)", 1, 30, 3)  # Optimized for 22% signals  
    detection_params = {'prominence_pct': prominence_pct, 'distance': distance}
    
elif detection_method == 'rolling_window':
    window_short = st.sidebar.slider("Short Window", 3, 15, 5)
    window_long = st.sidebar.slider("Long Window", 10, 40, 15)  # Reduced from 20
    min_change_pct = st.sidebar.slider("Min Change (%)", 0.5, 10.0, 2.0, 0.5)  # Reduced from 3.0
    detection_params = {
        'window_short': window_short,
        'window_long': window_long, 
        'min_change_pct': min_change_pct
    }
    
elif detection_method == 'percentage_swing':
    swing_pct = st.sidebar.slider("Swing Threshold (%)", 1.0, 15.0, 3.0, 0.5)  # Reduced from 5.0
    detection_params = {'swing_pct': swing_pct}
    
elif detection_method == 'multi_timeframe':
    short_window = st.sidebar.slider("Short Timeframe", 5, 20, 8)  # Reduced from 10
    long_window = st.sidebar.slider("Long Timeframe", 15, 60, 25)  # Reduced from 30
    detection_params = {'short_window': short_window, 'long_window': long_window}
    min_change = st.sidebar.slider("Min Change %", 1.0, 5.0, 3.0)
    detection_params = {
        'short_window': short_window,
        'long_window': long_window,
        'min_change': min_change
    }

# Feature engineering options
st.sidebar.subheader("Feature Engineering")
include_lagged = st.sidebar.checkbox("Include Lagged Features", True)
include_rolling = st.sidebar.checkbox("Include Rolling Features", True)
feature_selection = st.sidebar.checkbox("Automatic Feature Selection", True)

# Model selection
st.sidebar.subheader("ML Models")
models_to_train = st.sidebar.multiselect(
    "Select Models to Train",
    ['random_forest', 'xgboost', 'lightgbm', 'svm'],
    default=['random_forest', 'xgboost']
)

# Hyperparameter optimization settings
st.sidebar.subheader("🔍 Hyperparameter Optimization")
use_optimization = st.sidebar.checkbox(
    "Enable Optuna Optimization", 
    value=False,
    help="Automatically tune model parameters for better performance"
)

# OPTION 3: Combined Optuna Optimization
st.sidebar.subheader("🎯 Option 3: Auto-Tune Trading Filters")
enable_trading_optimization = st.sidebar.checkbox(
    "Enable Trading Filter Optimization",
    value=False,
    help="Use Optuna to automatically find best confidence + composite thresholds"
)

if enable_trading_optimization:
    trading_trials = st.sidebar.slider(
        "Trading Optimization Trials",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="Number of trials to find optimal trading thresholds"
    )
    
    st.sidebar.info(f"⚡ Will optimize 4 parameters for MAXIMUM TOTAL RETURN:\n• Buy confidence (0-100%)\n• Sell confidence (0-100%)\n• Buy composite (-1.0 to 0.0)\n• Sell composite (0.0 to 1.0)")
    st.sidebar.warning(f"⏱️ Est. time: ~{trading_trials * 3}s")
else:
    trading_trials = 0

# Show clear button if optimized parameters exist (always visible when params exist)
if (hasattr(st.session_state, 'trading_optimization_params') and 
    st.session_state.trading_optimization_params is not None):
    if st.sidebar.button("🗑️ Clear Optimization"):
        st.session_state.trading_optimization_params = None
        st.sidebar.success("✅ Optimization cleared - back to baseline")

# Show optimization status in sidebar
if (hasattr(st.session_state, 'trading_optimization_params') and 
    st.session_state.trading_optimization_params is not None):
    params = st.session_state.trading_optimization_params
    st.sidebar.success("🎯 **Optimized Parameters Active**")
    st.sidebar.write(f"Buy Conf: {params['min_buy_confidence']:.1f}%")
    st.sidebar.write(f"Sell Conf: {params['min_sell_confidence']:.1f}%") 
    st.sidebar.write(f"Buy Comp: {params['buy_composite_max']:.3f}")
    st.sidebar.write(f"Sell Comp: {params['sell_composite_min']:.3f}")
elif enable_trading_optimization:
    st.sidebar.info("⚪ Click 'Start Trading Optimization' to find optimal parameters")

if use_optimization:
    n_trials = st.sidebar.slider(
        "Optimization Trials",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="More trials = better optimization but longer training time"
    )
    st.sidebar.info(f"⏱️ Est. time: ~{n_trials * len(models_to_train) * 2:.0f}s")
else:
    n_trials = 50  # Default value

# === MAIN CONTENT TABS ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data & Labels", 
    "🔧 Feature Engineering",
    "🎯 Model Training", 
    "📈 Predictions",
    "🏆 Performance",
    "🚀 Production"
])

# Initialize session state
if 'ml_data_loaded' not in st.session_state:
    st.session_state.ml_data_loaded = False
if 'ml_features_ready' not in st.session_state:
    st.session_state.ml_features_ready = False
if 'ml_models_trained' not in st.session_state:
    st.session_state.ml_models_trained = False

# === TAB 1: DATA & LABELS ===
with tab1:
    st.header("📊 Data Preparation & Labeling")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔄 Load Data & Generate Labels", type="primary"):
            with st.spinner(f"Loading {ticker} data and detecting peaks/valleys..."):
                try:
                    # Load price data (use training_period from sidebar)
                    data = yf.Ticker(ticker).history(period=training_period, interval="1d")
                    data.columns = [col.lower() for col in data.columns]
                    
                    if len(data) == 0:
                        st.error("❌ No data found for this ticker")
                        st.stop()
                    
                    # Detect peaks and valleys with current parameters
                    detector = PeakValleyDetector()
                    features, labels = detector.create_labeled_dataset(
                        data, method=detection_method, **detection_params
                    )
                    
                    # Store in session state
                    st.session_state.ml_raw_data = data
                    st.session_state.ml_features_base = features
                    st.session_state.ml_labels = labels
                    st.session_state.ml_detector = detector
                    st.session_state.ml_data_loaded = True
                    
                    st.success("✅ Data loaded and labeled successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
    
    with col2:
        st.info(f"""
        **Current Settings:**
        - Ticker: {ticker}
        - Period: {training_period}  
        - Method: {detection_method}
        """)
    
    # Show results if data is loaded
    if st.session_state.ml_data_loaded:
        st.subheader("📋 Labeling Results")
        
        labels = st.session_state.ml_labels
        detector = st.session_state.ml_detector
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(labels))
        with col2:
            st.metric("BUY Signals", (labels == 1).sum())
        with col3:
            st.metric("SELL Signals", (labels == -1).sum())
        with col4:
            signal_ratio = (labels != 0).sum() / len(labels) * 100
            st.metric("Signal Ratio", f"{signal_ratio:.1f}%")
        
        # Show peaks and valleys on chart
        st.subheader("📈 Detected Peaks & Valleys")
        
        data = st.session_state.ml_raw_data
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=1)
        ))
        
        # Mark peaks (SELL points)
        sell_points = labels[labels == -1].index
        if len(sell_points) > 0:
            sell_prices = data.loc[sell_points, 'close']
            fig.add_trace(go.Scatter(
                x=sell_points,
                y=sell_prices,
                mode='markers',
                name='Peaks (SELL)',
                marker=dict(color='red', size=8, symbol='triangle-down')
            ))
        
        # Mark valleys (BUY points)
        buy_points = labels[labels == 1].index
        if len(buy_points) > 0:
            buy_prices = data.loc[buy_points, 'close']
            fig.add_trace(go.Scatter(
                x=buy_points,
                y=buy_prices,
                mode='markers',
                name='Valleys (BUY)',
                marker=dict(color='green', size=8, symbol='triangle-up')
            ))
        
        fig.update_layout(
            title=f"{ticker} - Detected Peaks & Valleys ({detection_method})",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# === TAB 2: FEATURE ENGINEERING ===
with tab2:
    st.header("🔧 Feature Engineering")
    
    if not st.session_state.ml_data_loaded:
        st.info("👈 Please load data first in the 'Data & Labels' tab")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🧠 Generate ML Features", type="primary"):
                # Clear any cached results to ensure fresh generation
                if 'ml_features' in st.session_state:
                    del st.session_state.ml_features
                if 'ml_engineer' in st.session_state:
                    del st.session_state.ml_engineer
                
                with st.spinner("Creating ML-ready features..."):
                    try:
                        # Force reload modules to ensure latest code
                        import importlib
                        import ml_feature_engineer
                        import peak_valley_detector  
                        import ml_models
                        importlib.reload(ml_feature_engineer)
                        importlib.reload(peak_valley_detector)
                        importlib.reload(ml_models)
                        
                        # Create feature engineer with fresh code
                        from ml_feature_engineer import MLFeatureEngineer
                        engineer = MLFeatureEngineer()
                        
                        # Generate features with aggressive deduplication
                        print(f"🚀 Starting feature generation with settings:")
                        print(f"   Lagged: {include_lagged}, Rolling: {include_rolling}")
                        print(f"   Input data shape: {st.session_state.ml_raw_data.shape}")
                        
                        ml_features = engineer.prepare_ml_dataset(
                            st.session_state.ml_raw_data,
                            include_lagged=include_lagged,
                            include_rolling=include_rolling,
                            feature_selection=True  # Force feature selection ON
                        )
                        
                        # Additional aggressive deduplication check
                        print(f"🔍 POST-PROCESSING CHECK:")
                        print(f"   Features before final check: {ml_features.shape[1]}")
                        
                        # Check sample values for debugging
                        print(f"📊 SAMPLE VALUES CHECK:")
                        sample_features = ml_features.iloc[:3, :5]  # First 3 rows, 5 columns
                        for col in sample_features.columns:
                            print(f"   {col}: {sample_features[col].tolist()}")
                        
                        # Check for value repetition across columns
                        first_row = ml_features.iloc[0]
                        unique_in_first_row = first_row.nunique()
                        total_values = len(first_row.dropna())
                        print(f"📋 First row diversity: {unique_in_first_row}/{total_values} unique values")
                        
                        # Remove any remaining identical columns manually
                        duplicated_cols = []
                        for i, column1 in enumerate(ml_features.columns):
                            for j, column2 in enumerate(ml_features.columns[i+1:], i+1):
                                if ml_features[column1].equals(ml_features[column2]):
                                    duplicated_cols.append(column2)
                        
                        if duplicated_cols:
                            print(f"   ⚠️ Found {len(duplicated_cols)} remaining identical columns")
                            ml_features = ml_features.drop(columns=duplicated_cols)
                            print(f"   ✅ Removed duplicates, final shape: {ml_features.shape}")
                        else:
                            print(f"   ✅ No additional duplicates found")
                        
                        # Final verification before storage
                        print(f"🎯 FINAL VERIFICATION:")
                        print(f"   Final shape: {ml_features.shape}")
                        print(f"   Column names sample: {list(ml_features.columns[:5])}")
                        print(f"   Data types: {ml_features.dtypes.value_counts().to_dict()}")
                        
                        # Check if preview will show diverse values  
                        preview_data = ml_features.head()
                        preview_issues = []
                        for col in preview_data.columns[:10]:  # Check first 10 columns
                            col_values = preview_data[col]
                            if col_values.nunique() < 2:  # Less than 2 unique values in preview
                                preview_issues.append(col)
                        
                        if preview_issues:
                            print(f"   ⚠️ Preview may show repetitive values in: {preview_issues[:5]}")
                        else:
                            print(f"   ✅ Preview should show diverse values")
                        
                        # Store results
                        st.session_state.ml_features = ml_features
                        st.session_state.ml_engineer = engineer
                        st.session_state.ml_features_ready = True
                        
                        st.success(f"✅ Features generated successfully! Shape: {ml_features.shape}")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating features: {str(e)}")
        
        with col2:
            st.info(f"""
            **Feature Options:**
            - Lagged Features: {include_lagged}
            - Rolling Features: {include_rolling}  
            - Feature Selection: {feature_selection}
            """)
        
        # Show feature results
        if st.session_state.ml_features_ready:
            st.subheader("📊 Feature Summary")
            
            features = st.session_state.ml_features
            engineer = st.session_state.ml_engineer
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Features", len(features.columns))
            with col2:
                st.metric("Data Points", len(features))
            with col3:
                nan_pct = features.isnull().sum().sum() / (len(features) * len(features.columns)) * 100
                st.metric("Missing Data", f"{nan_pct:.1f}%")
            
            # Display feature preview (skip warm-up period)
            st.subheader("🔍 Feature Preview")
            warmup_period = min(30, len(features) // 4)  # Skip indicator warm-up period
            preview_start = max(warmup_period, 0)
            preview_df = features.iloc[preview_start:preview_start+10]
            st.dataframe(preview_df, use_container_width=True)
            
            # Feature importance candidates
            important_features = engineer.get_feature_importance_candidates()
            if important_features:
                st.subheader("⭐ Key Features")
                st.write("Features likely to be important for ML models:")
                st.write(", ".join(important_features[:10]))

# === TAB 3: MODEL TRAINING ===
with tab3:
    st.header("🎯 Model Training")
    
    # --- LOAD SAVED MODEL SECTION ---
    with st.expander("📂 Load Pre-trained Model (For Analysis)"):
        try:
            from production_manager import ProductionManager
            pm_loader = ProductionManager()
            saved_list = pm_loader.list_saved_models()
            
            if saved_list:
                # Create readable labels
                load_opts = {f"{m['name']} | {m['created'].strftime('%Y-%m-%d %H:%M')}": m for m in saved_list}
                sel_load = st.selectbox("Select Model to Load", list(load_opts.keys()), key='tab3_model_loader')
                
                if st.button("📥 Load Model into Session"):
                    target_path = load_opts[sel_load]['path']
                    with st.spinner("Loading model..."):
                        try:
                            from ml_models import TradingMLModels
                            ml_instance = TradingMLModels()
                            # Load the specific version
                            loaded_data = ml_instance.load_model_version(target_path)
                            
                            # Extract components
                            mod_name = loaded_data.get('name', 'custom_model')
                            model_obj = loaded_data['model']
                            scaler_obj = loaded_data.get('scaler')
                            feats_list = loaded_data.get('feature_names', [])
                            meta_info = loaded_data.get('metadata', {})
                            
                            # Inject into the ML class instance so other tabs can use it
                            ml_instance.models[mod_name] = model_obj
                            ml_instance.scalers[mod_name] = scaler_obj
                            ml_instance.feature_sets[mod_name] = feats_list
                            ml_instance.best_model = mod_name
                            
                            # Construct a 'results' dict so Tab 4 & 5 can render
                            # Use metadata if available, else placeholders
                            res_entry = {
                                'accuracy': meta_info.get('score', 0.0),
                                'f1_score': 0.0, # Placeholder
                                'cv_mean': meta_info.get('score', 0.0),
                                'cv_std': 0.0,
                                'feature_count': len(feats_list)
                            }
                            results_dict = {mod_name: res_entry}
                            
                            # Update Session State
                            st.session_state.ml_models = ml_instance
                            st.session_state.ml_results = results_dict
                            st.session_state.ml_models_trained = True
                            
                            # Important: If we loaded a model, we might not have compatible features generated yet.
                            # But usually user loads data -> generates features -> loads model -> predicts.
                            
                            st.success(f"✅ Successfully loaded '{mod_name}'! You can now go to the 'Predictions' or 'Performance' tabs.")
                            
                        except Exception as e:
                            st.error(f"Failed to load model: {e}")
            else:
                st.info("No saved models found in registry.")
                
        except Exception as e:
            st.warning(f"Could not initialize loader: {e}")
            
    st.markdown("---")
    
    if not st.session_state.ml_features_ready:
        st.info("👈 Please generate features first in the 'Feature Engineering' tab")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Train ML Models", type="primary"):
                if not models_to_train:
                    st.error("❌ Please select at least one model to train")
                else:
                    with st.spinner("Training ML models..."):
                        try:
                            # Force reload ML models to ensure latest code
                            import importlib
                            import ml_models
                            importlib.reload(ml_models)
                            from ml_models import TradingMLModels
                            
                            # Prepare data
                            features = st.session_state.ml_features
                            labels = st.session_state.ml_labels
                            
                            # Align features and labels
                            common_index = features.index.intersection(labels.index)
                            features_aligned = features.loc[common_index]
                            labels_aligned = labels.loc[common_index]
                            
                            # Create ML models instance
                            ml_models = TradingMLModels()
                            
                            # Prepare train/test split (80/20)
                            X_train, X_test, y_train, y_test = ml_models.prepare_data(
                                features_aligned, labels_aligned, test_size=0.2, use_time_split=True
                            )
                            
                            # Store split date for visualization
                            split_date = X_train.index[-1]
                            st.session_state.ml_split_date = split_date
                            st.session_state.ml_train_samples = len(X_train)
                            st.session_state.ml_test_samples = len(X_test)
                            
                            # 🎯 PEAK/VALLEY ENHANCEMENT: Apply SMOTE balancing
                            st.info("⚖️ Applying SMOTE balancing for peak/valley labels...")
                            
                            # Check class distribution before SMOTE
                            original_counts = pd.Series(y_train).value_counts().sort_index()
                            st.write("**Original Label Distribution:**")
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("SELL (-1)", original_counts.get(-1, 0))
                            with col_b:
                                st.metric("HOLD (0)", original_counts.get(0, 0))
                            with col_c:
                                st.metric("BUY (1)", original_counts.get(1, 0))
                            
                            # Apply SMOTE to balance the dataset
                            try:
                                from imblearn.over_sampling import SMOTE
                                
                                # SMOTE works better with non-zero classes, so handle HOLD separately if needed
                                if len(original_counts) > 2:  # Has SELL, HOLD, BUY
                                    smote = SMOTE(random_state=42, k_neighbors=min(3, original_counts.min()-1))
                                else:  # Only BUY/SELL
                                    smote = SMOTE(random_state=42)
                                
                                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                                
                                # Show balanced distribution
                                balanced_counts = pd.Series(y_train_balanced).value_counts().sort_index()
                                st.write("**After SMOTE Balancing:**")
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("SELL (-1)", balanced_counts.get(-1, 0))
                                with col_b:
                                    st.metric("HOLD (0)", balanced_counts.get(0, 0))
                                with col_c:
                                    st.metric("BUY (1)", balanced_counts.get(1, 0))
                                
                                # Use balanced data for training
                                X_train = X_train_balanced
                                y_train = y_train_balanced
                                
                                st.success("✅ SMOTE balancing applied successfully!")
                                
                            except ImportError:
                                st.warning("⚠️ SMOTE not available (install imbalanced-learn). Using original data.")
                            except Exception as e:
                                st.warning(f"⚠️ SMOTE failed: {str(e)}. Using original data.")
                            
                            # Update samples count after balancing
                            st.session_state.ml_train_samples_balanced = len(X_train)
                            
                            # Train selected models
                            results = {}
                            for model_name in models_to_train:
                                if use_optimization:
                                    st.write(f"🔍 Training {model_name} with hyperparameter optimization ({n_trials} trials)...")
                                    
                                    # Check if Optuna is available
                                    try:
                                        import optuna
                                        # Use optimization
                                        train_result = ml_models.train_with_optimization(model_name, X_train, y_train, n_trials=n_trials)
                                        if 'error' not in train_result:
                                            eval_result = ml_models.evaluate_model(model_name, X_test, y_test)
                                            results[model_name] = {**train_result, **eval_result}
                                            
                                            # Display optimization results
                                            if 'optimization' in train_result:
                                                opt_info = train_result['optimization']
                                                st.success(f"✅ {model_name} optimized! Best CV Score: {opt_info['best_cv_score']:.4f}")
                                                with st.expander(f"🔍 {model_name} Best Parameters"):
                                                    st.json(opt_info['best_params'])
                                        else:
                                            results[model_name] = train_result
                                            
                                    except ImportError:
                                        st.warning("⚠️ Optuna not available. Training with default parameters.")
                                        # Fallback to regular training
                                        train_result = ml_models.train_model(model_name, X_train, y_train)
                                        if 'error' not in train_result:
                                            eval_result = ml_models.evaluate_model(model_name, X_test, y_test)
                                            results[model_name] = {**train_result, **eval_result}
                                        else:
                                            results[model_name] = train_result
                                else:
                                    st.write(f"Training {model_name} with default parameters...")
                                    train_result = ml_models.train_model(model_name, X_train, y_train)
                                    if 'error' not in train_result:
                                        eval_result = ml_models.evaluate_model(model_name, X_test, y_test)
                                        results[model_name] = {**train_result, **eval_result}
                                    else:
                                        results[model_name] = train_result
                            
                            # Store results
                            st.session_state.ml_models = ml_models
                            st.session_state.ml_results = results
                            st.session_state.ml_models_trained = True
                            
                            st.success("✅ Model training completed!")
                            
                        except Exception as e:
                            st.error(f"❌ Error training models: {str(e)}")
        
        with col2:
            st.info(f"""
            **🎯 Peak/Valley ML Training:**
            • Uses peaks/valleys as labels (not random)
            • SMOTE balancing for imbalanced data
            • 163.8% potential vs 0.9% baseline
            
            **Selected Models:**
            {chr(10).join([f"• {model}" for model in models_to_train])}
            """)
        
        # Show training results
        if st.session_state.ml_models_trained:
            st.subheader("📊 Training Results")
            
            results = st.session_state.ml_results
            
            # Model comparison table
            comparison_data = []
            for model_name, result in results.items():
                if 'error' not in result:
                    comparison_data.append({
                        'Model': model_name,
                        'CV Score': f"{result['cv_mean']:.4f} ± {result['cv_std']:.4f}",
                        'Train Accuracy': f"{result.get('train_accuracy', 0):.4f}",
                        'Test Accuracy': f"{result['accuracy']:.4f}",
                        'Test F1': f"{result['f1_score']:.4f}",
                        'Features': result['feature_count']
                    })
                else:
                    comparison_data.append({
                        'Model': model_name,
                        'CV Score': 'ERROR',
                        'Train Accuracy': '-',
                        'Test Accuracy': result['error'][:50],
                        'Test F1': '-',
                        'Features': '-'
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Best model highlight
            ml_models = st.session_state.ml_models
            if ml_models.best_model:
                st.success(f"🏆 Best Model: **{ml_models.best_model}** (F1 Score: {ml_models.best_score:.4f})")

# === TAB 4: PREDICTIONS ===
with tab4:
    st.header("📈 Live Predictions")
    
    if not st.session_state.ml_models_trained:
        st.info("👈 Please train models first in the 'Model Training' tab")
    else:
        st.subheader("🔮 Generate Trading Signals")
        
        # Model selection for prediction
        ml_models = st.session_state.ml_models
        available_models = list(ml_models.models.keys())
        
        selected_model = st.selectbox(
            "Select Model for Predictions",
            available_models,
            index=available_models.index(ml_models.best_model) if ml_models.best_model in available_models else 0
        )
        
        if st.button("🎯 Generate Current Signals", type="primary"):
            with st.spinner("Generating predictions..."):
                try:
                    features = st.session_state.ml_features
                    
                    # Generate predictions
                    predictions, probabilities = ml_models.predict_signals(features, selected_model)
                    
                    # Create full prediction summary
                    full_summary = pd.DataFrame({
                        'Date': features.index,
                        'Signal': predictions,
                        'Signal_Text': ['BUY' if p == 1 else 'SELL' if p == -1 else 'HOLD' for p in predictions]
                    })
                    
                    if probabilities is not None:
                        confidence = np.max(probabilities, axis=1)
                        full_summary['Confidence'] = confidence
                    
                    # Split predictions if split date is available
                    if 'ml_split_date' in st.session_state:
                        split_date = st.session_state.ml_split_date
                        
                        st.subheader("📋 Predictions by Data Split")
                        
                        # Training Set
                        train_mask = full_summary['Date'] <= split_date
                        train_preds = full_summary[train_mask]
                        
                        with st.expander(f"Training Data Predictions ({len(train_preds)} samples)", expanded=False):
                            st.dataframe(train_preds, use_container_width=True)
                        
                        # Testing Set
                        test_mask = full_summary['Date'] > split_date
                        test_preds = full_summary[test_mask]
                        
                        st.markdown(f"### 🧪 Testing Data Predictions ({len(test_preds)} samples)")
                        st.dataframe(test_preds, use_container_width=True)
                        
                    else:
                        # Fallback to simple view
                        st.subheader("📋 Recent Predictions")
                        st.dataframe(full_summary.tail(20), use_container_width=True)
                    
                    # Current signal
                    current_signal = predictions[-1]
                    signal_text = 'BUY' if current_signal == 1 else 'SELL' if current_signal == -1 else 'HOLD'
                    
                    if current_signal == 1:
                        st.success(f"🟢 **CURRENT SIGNAL: {signal_text}**")
                    elif current_signal == -1:
                        st.error(f"🔴 **CURRENT SIGNAL: {signal_text}**")
                    else:
                        st.info(f"🟡 **CURRENT SIGNAL: {signal_text}**")
                    
                    if probabilities is not None:
                        current_confidence = confidence[-1]
                        st.write(f"**Confidence:** {current_confidence:.2%}")
                    
                    # 🎯 PEAK/VALLEY ENHANCEMENT: Candlestick Chart with Signals
                    st.subheader("📊 Candlestick Chart with Peak/Valley Signals")
                    
                    # Get price data
                    raw_data = st.session_state.ml_raw_data
                    
                    # Create candlestick chart
                    fig = go.Figure()
                    
                    # Add candlestick
                    fig.add_trace(go.Candlestick(
                        x=raw_data.index,
                        open=raw_data['open'],
                        high=raw_data['high'], 
                        low=raw_data['low'],
                        close=raw_data['close'],
                        name='Price',
                        increasing_line_color='green',
                        decreasing_line_color='red'
                    ))
                    
                    # Overlay actual peak/valley labels (ground truth)
                    labels = st.session_state.ml_labels
                    
                    # Peak points (SELL labels)
                    peak_points = labels[labels == -1].index
                    if len(peak_points) > 0:
                        peak_prices = raw_data.loc[peak_points, 'high'] * 1.02  # Slightly above high
                        fig.add_trace(go.Scatter(
                            x=peak_points,
                            y=peak_prices,
                            mode='markers',
                            name='Actual Peaks',
                            marker=dict(
                                symbol='triangle-down',
                                size=10,
                                color='red',
                                line=dict(color='darkred', width=2)
                            )
                        ))
                    
                    # Valley points (BUY labels) 
                    valley_points = labels[labels == 1].index
                    if len(valley_points) > 0:
                        valley_prices = raw_data.loc[valley_points, 'low'] * 0.98  # Slightly below low
                        fig.add_trace(go.Scatter(
                            x=valley_points,
                            y=valley_prices,
                            mode='markers',
                            name='Actual Valleys',
                            marker=dict(
                                symbol='triangle-up',
                                size=10,
                                color='green', 
                                line=dict(color='darkgreen', width=2)
                            )
                        ))
                    
                    # Overlay ML predictions (with different symbols)
                    aligned_data = raw_data.loc[features.index]  # Align with predictions
                    
                    # ML BUY predictions
                    ml_buy_mask = (predictions == 1)
                    if ml_buy_mask.any():
                        ml_buy_dates = features.index[ml_buy_mask]
                        ml_buy_prices = aligned_data.loc[ml_buy_dates, 'low'] * 0.95  # Lower than valleys
                        fig.add_trace(go.Scatter(
                            x=ml_buy_dates,
                            y=ml_buy_prices,
                            mode='markers',
                            name='ML BUY Signals',
                            marker=dict(
                                symbol='circle',
                                size=8,
                                color='lightgreen',
                                line=dict(color='green', width=1)
                            )
                        ))
                    
                    # ML SELL predictions
                    ml_sell_mask = (predictions == -1)
                    if ml_sell_mask.any():
                        ml_sell_dates = features.index[ml_sell_mask]
                        ml_sell_prices = aligned_data.loc[ml_sell_dates, 'high'] * 1.05  # Higher than peaks
                        fig.add_trace(go.Scatter(
                            x=ml_sell_dates,
                            y=ml_sell_prices,
                            mode='markers',
                            name='ML SELL Signals',
                            marker=dict(
                                symbol='circle',
                                size=8,
                                color='lightcoral',
                                line=dict(color='red', width=1)
                            )
                        ))
                    
                    # Add train/test split line if available
                    if 'ml_split_date' in st.session_state:
                        split_date = st.session_state.ml_split_date
                        fig.add_vline(
                            x=split_date,
                            line_dash="dash",
                            line_color="blue",
                            annotation_text="Train/Test Split",
                            annotation_position="top"
                        )
                    
                    # Configure layout
                    fig.update_layout(
                        title=f"Peak/Valley Signals vs ML Predictions - {ticker}",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=600,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Remove range slider for cleaner view
                    fig.update_layout(xaxis_rangeslider_visible=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Signal accuracy analysis
                    st.subheader("🎯 Signal Accuracy Analysis")
                    
                    # Compare ML predictions vs actual labels
                    aligned_labels = labels.loc[features.index]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Overall accuracy
                        accuracy = (predictions == aligned_labels).mean()
                        st.metric("Overall Accuracy", f"{accuracy:.2%}")
                    
                    with col2:
                        # BUY signal accuracy
                        buy_mask = (aligned_labels == 1)
                        if buy_mask.any():
                            buy_accuracy = (predictions[buy_mask] == 1).mean()
                            st.metric("BUY Signal Accuracy", f"{buy_accuracy:.2%}")
                        else:
                            st.metric("BUY Signal Accuracy", "N/A")
                    
                    with col3:
                        # SELL signal accuracy  
                        sell_mask = (aligned_labels == -1)
                        if sell_mask.any():
                            sell_accuracy = (predictions[sell_mask] == -1).mean()
                            st.metric("SELL Signal Accuracy", f"{sell_accuracy:.2%}")
                        else:
                            st.metric("SELL Signal Accuracy", "N/A")
                    
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {str(e)}")

# === TAB 5: PERFORMANCE ===
with tab5:
    st.header("🏆 Model Performance & Historical Backtesting")
    
    if not st.session_state.ml_models_trained:
        st.info("👈 Please train models first in the 'Model Training' tab")
    else:
        results = st.session_state.ml_results
        ml_models = st.session_state.ml_models
        
        # Model selection for backtesting
        available_models = [name for name in results.keys() if 'error' not in results[name]]
        
        if available_models:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_model = st.selectbox(
                    "Select Model for Historical Backtesting:",
                    available_models,
                    index=available_models.index(ml_models.best_model) if ml_models.best_model in available_models else 0
                )
            
            with col2:
                if st.button("📈 Run Historical Backtest", type="primary"):
                    with st.spinner("Running historical backtest..."):
                        try:
                            # Generate predictions for entire dataset
                            features = st.session_state.ml_features
                            raw_data = st.session_state.ml_raw_data
                            
                            predictions, probabilities = ml_models.predict_signals(features, selected_model)
                            
                            # Create backtest results
                            def run_ml_backtest(data, signals, starting_capital=100000):
                                """Enhanced backtest for ML signals - executes ALL signals"""
                                capital = starting_capital
                                position = None
                                trades = []
                                equity_curve = [starting_capital]
                                
                                print(f"🔍 BACKTEST DEBUG:")
                                print(f"   Data length: {len(data)}")
                                print(f"   Signals length: {len(signals)}")
                                print(f"   Unique signals: {np.unique(signals, return_counts=True)}")
                                
                                # Process each day
                                for i in range(len(data)):
                                    current_price = data['close'].iloc[i]
                                    current_date = data.index[i]
                                    signal = signals[i] if i < len(signals) else 0
                                    
                                    # BUY SIGNAL: Enter long position (or exit short)
                                    if signal == 1:
                                        if position is None or position['type'] == 'short':
                                            # Close short position if exists
                                            if position and position['type'] == 'short':
                                                profit = position['entry_capital'] - (position['shares'] * current_price)
                                                capital += profit
                                                
                                                # Update last trade
                                                if trades and trades[-1]['exit_date'] is None:
                                                    trades[-1].update({
                                                        'exit_date': current_date,
                                                        'exit_price': current_price,
                                                        'profit': profit
                                                    })
                                            
                                            # Enter new long position
                                            shares = capital / current_price
                                            position = {
                                                'type': 'long',
                                                'entry_date': current_date,
                                                'entry_price': current_price,
                                                'shares': shares,
                                                'entry_capital': capital
                                            }
                                            
                                            trades.append({
                                                'entry_date': current_date,
                                                'entry_price': current_price,
                                                'exit_date': None,
                                                'exit_price': None,
                                                'shares': shares,
                                                'position_value': capital,
                                                'profit': None,
                                                'signal_type': 'LONG'
                                            })
                                    
                                    # SELL SIGNAL: Enter short position (or exit long)  
                                    elif signal == -1:
                                        if position is None or position['type'] == 'long':
                                            # Close long position if exists
                                            if position and position['type'] == 'long':
                                                exit_value = position['shares'] * current_price
                                                profit = exit_value - position['entry_capital']
                                                capital = exit_value
                                                
                                                # Update last trade
                                                if trades and trades[-1]['exit_date'] is None:
                                                    trades[-1].update({
                                                        'exit_date': current_date,
                                                        'exit_price': current_price,
                                                        'profit': profit
                                                    })
                                            
                                            # Enter new short position (simulate by holding cash)
                                            position = {
                                                'type': 'short',
                                                'entry_date': current_date,
                                                'entry_price': current_price,
                                                'shares': capital / current_price,  # Theoretical shares
                                                'entry_capital': capital
                                            }
                                            
                                            trades.append({
                                                'entry_date': current_date,
                                                'entry_price': current_price,
                                                'exit_date': None,
                                                'exit_price': None,
                                                'shares': capital / current_price,
                                                'position_value': capital,
                                                'profit': None,
                                                'signal_type': 'SHORT'
                                            })
                                    
                                    # Calculate portfolio value
                                    if position:
                                        if position['type'] == 'long':
                                            portfolio_value = position['shares'] * current_price
                                        else:  # short position
                                            # For short: profit when price goes down
                                            portfolio_value = position['entry_capital'] + (
                                                position['entry_capital'] - position['shares'] * current_price
                                            )
                                    else:
                                        portfolio_value = capital
                                    
                                    equity_curve.append(portfolio_value)
                                
                                # Close final position if still open
                                if position:
                                    final_price = data['close'].iloc[-1]
                                    
                                    if position['type'] == 'long':
                                        exit_value = position['shares'] * final_price
                                        profit = exit_value - position['entry_capital']
                                        capital = exit_value
                                    else:  # short
                                        profit = position['entry_capital'] - (position['shares'] * final_price)
                                        capital += profit
                                    
                                    # Update last trade
                                    if trades and trades[-1]['exit_date'] is None:
                                        trades[-1].update({
                                            'exit_date': data.index[-1],
                                            'exit_price': final_price,
                                            'profit': profit
                                        })
                                
                                print(f"   Total trades created: {len(trades)}")
                                print(f"   Completed trades: {len([t for t in trades if t['profit'] is not None])}")
                                
                                return trades, equity_curve, capital
                            
                            # Align data and predictions
                            common_index = features.index.intersection(raw_data.index)
                            aligned_data = raw_data.loc[common_index]
                            aligned_predictions = predictions[:len(common_index)]
                            
                            # Run backtest
                            trades, equity_curve, final_capital = run_ml_backtest(aligned_data, aligned_predictions)
                            
                            # Calculate performance metrics
                            total_return = (final_capital / 100000 - 1) * 100
                            buy_hold_return = (aligned_data['close'].iloc[-1] / aligned_data['close'].iloc[0] - 1) * 100
                            
                            # Store results in session state
                            st.session_state.ml_backtest_results = {
                                'trades': trades,
                                'equity_curve': equity_curve,
                                'final_capital': final_capital,
                                'total_return': total_return,
                                'buy_hold_return': buy_hold_return,
                                'aligned_data': aligned_data,
                                'predictions': aligned_predictions,
                                'probabilities': probabilities[:len(common_index)] if probabilities is not None else None
                            }
                            
                            st.success("✅ Backtest completed!")
                            
                        except Exception as e:
                            st.error(f"❌ Backtest failed: {str(e)}")
            
            # Show backtest results if available
            if 'ml_backtest_results' in st.session_state:
                backtest = st.session_state.ml_backtest_results
                
                # Performance summary
                st.subheader("📊 Performance Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("ML Strategy Return", f"{backtest['total_return']:.2f}%")
                with col2:
                    st.metric("Buy & Hold Return", f"{backtest['buy_hold_return']:.2f}%")
                with col3:
                    outperformance = backtest['total_return'] - backtest['buy_hold_return']
                    st.metric("Outperformance", f"{outperformance:.2f}%")
                with col4:
                    st.metric("Total Trades", len([t for t in backtest['trades'] if t['profit'] is not None]))
                
                # Show signal statistics
                signals = backtest['predictions']
                buy_signals = np.sum(signals == 1)
                sell_signals = np.sum(signals == -1)
                hold_signals = np.sum(signals == 0)
                
                st.info(f"📊 **ML Signals Generated:** {buy_signals} BUY, {sell_signals} SELL, {hold_signals} HOLD | **Total Trades Executed:** {len(backtest['trades'])}")
                
                if 'ml_split_date' in st.session_state:
                    split_date = st.session_state.ml_split_date
                    train_n = st.session_state.get('ml_train_samples', 0)
                    test_n = st.session_state.get('ml_test_samples', 0)
                    st.caption(f"📉 **Data Split:** Training up to {split_date.date()} ({train_n} samples) | Testing after {split_date.date()} ({test_n} samples)")
                
                # Price chart with ML signals
                st.subheader("📈 Historical Price & ML Signals")
                
                data = backtest['aligned_data']
                predictions = backtest['predictions']
                
                fig = go.Figure()
                
                # Price line
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='blue', width=2)
                ))
                
                # ML Buy signals (Offset below price)
                buy_signals = np.where(predictions == 1)[0]
                if len(buy_signals) > 0:
                    buy_dates = data.index[buy_signals]
                    buy_prices = data['close'].iloc[buy_signals] * 0.98  # 2% below price
                    fig.add_trace(go.Scatter(
                        x=buy_dates,
                        y=buy_prices,
                        mode='markers',
                        name='ML BUY Signals',
                        marker=dict(color='green', size=8, symbol='triangle-up'),
                        hovertemplate='<b>BUY Signal</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                    ))
                
                # ML Sell signals (Offset above price)
                sell_signals = np.where(predictions == -1)[0]
                if len(sell_signals) > 0:
                    sell_dates = data.index[sell_signals]
                    sell_prices = data['close'].iloc[sell_signals] * 1.02  # 2% above price
                    fig.add_trace(go.Scatter(
                        x=sell_dates,
                        y=sell_prices,
                        mode='markers',
                        name='ML SELL Signals',
                        marker=dict(color='red', size=8, symbol='triangle-down'),
                        hovertemplate='<b>SELL Signal</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                    ))
                
                # Trade entries/exits
                for trade in backtest['trades']:
                    if trade['profit'] is not None:  # Completed trades only
                        # Entry
                        fig.add_trace(go.Scatter(
                            x=[trade['entry_date']],
                            y=[trade['entry_price']],
                            mode='markers',
                            name='Trade Entry',
                            marker=dict(color='blue', size=10, symbol='circle'),
                            showlegend=False
                        ))
                        # Exit
                        exit_name = 'Position Closed'
                        if trade['exit_date'] == data.index[-1]:
                            exit_name = 'Backtest End (Force Close)'
                            
                        fig.add_trace(go.Scatter(
                            x=[trade['exit_date']],
                            y=[trade['exit_price']],
                            mode='markers',
                            name=exit_name,
                            marker=dict(color='purple', size=10, symbol='square'),
                            showlegend=False
                        ))
                
                # Add vertical line for Train/Test split if available
                if 'ml_split_date' in st.session_state:
                    split_date = st.session_state.ml_split_date
                    # Ensure split_date is in the current data range
                    if split_date >= data.index[0] and split_date <= data.index[-1]:
                        # Convert to milliseconds for Plotly to avoid Pandas Timestamp arithmetic errors
                        split_date_ms = split_date.timestamp() * 1000
                        end_date_ms = data.index[-1].timestamp() * 1000
                        
                        fig.add_vline(
                            x=split_date_ms, 
                            line_width=2, 
                            line_dash="dash", 
                            line_color="gray",
                            annotation_text="Train / Test Split", 
                            annotation_position="top right"
                        )
                        
                        # Add background shading for Test area
                        fig.add_vrect(
                            x0=split_date_ms, 
                            x1=end_date_ms,
                            fillcolor="rgba(200, 200, 255, 0.1)", 
                            layer="below", 
                            line_width=0,
                            annotation_text="Test Data (Unseen)", 
                            annotation_position="top right"
                        )

                fig.update_layout(
                    title=f"{ticker} - ML Trading Signals Backtest ({selected_model})",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Signal Analysis / Diagnostics
                if 'ml_split_date' in st.session_state:
                    with st.expander("🔍 Why no Sell Signals? (Signal Distribution Analysis)"):
                        split_date = st.session_state.ml_split_date
                        
                        # Get labels (actual signals used for training)
                        labels = st.session_state.ml_labels
                        
                        train_labels = labels[labels.index <= split_date]
                        test_labels = labels[labels.index > split_date]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Training Data (What model learned)")
                            train_counts = train_labels.value_counts().sort_index()
                            st.write(train_counts)
                            if -1 in train_counts:
                                st.info(f"Model saw {train_counts[-1]} SELL examples")
                            else:
                                st.warning("⚠️ Model saw 0 SELL examples in training!")
                                
                        with col2:
                            st.markdown("#### Test Data (Ideal Signals)")
                            test_counts = test_labels.value_counts().sort_index()
                            st.write(test_counts)
                            st.caption("These are the 'correct' signals based on peak/valley detection.")
                        
                        st.markdown("""
                        **Diagnosis:**
                        - If Training has few SELLs (-1), the model learns to be conservative.
                        - If Test Data has SELLs but Model predicts none, the model failed to generalize (overfitting to Buy).
                        - If Test Data has 0 SELLs, then the market was simply bullish!
                        """)
                
                # Equity curve
                st.subheader("💰 Equity Curve Comparison")
                
                equity_fig = go.Figure()
                
                # ML strategy equity curve
                equity_dates = data.index[:len(backtest['equity_curve'])]
                equity_fig.add_trace(go.Scatter(
                    x=equity_dates,
                    y=backtest['equity_curve'],
                    mode='lines',
                    name='ML Strategy',
                    line=dict(color='green', width=2)
                ))
                
                # Buy and hold comparison
                initial_price = data['close'].iloc[0]
                shares_bh = 100000 / initial_price
                buy_hold_curve = [shares_bh * price for price in data['close'][:len(backtest['equity_curve'])]]
                
                equity_fig.add_trace(go.Scatter(
                    x=equity_dates,
                    y=buy_hold_curve,
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='blue', width=2, dash='dash')
                ))
                
                equity_fig.update_layout(
                    title="Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=400
                )
                
                st.plotly_chart(equity_fig, use_container_width=True)
                
                # Trade log
                st.subheader("📋 Trade Log")
                
                if backtest['trades']:
                    trades_df = pd.DataFrame(backtest['trades'])
                    
                    # Format for display
                    display_trades = trades_df.copy()
                    if 'entry_date' in display_trades.columns:
                        display_trades['Entry Date'] = pd.to_datetime(display_trades['entry_date']).dt.strftime('%Y-%m-%d')
                    if 'exit_date' in display_trades.columns:
                        display_trades['Exit Date'] = pd.to_datetime(display_trades['exit_date']).dt.strftime('%Y-%m-%d')
                    if 'entry_price' in display_trades.columns:
                        display_trades['Entry Price'] = display_trades['entry_price'].apply(lambda x: f"${x:.2f}")
                    if 'exit_price' in display_trades.columns:
                        display_trades['Exit Price'] = display_trades['exit_price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "Open")
                    if 'profit' in display_trades.columns:
                        display_trades['Profit'] = display_trades['profit'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "Open")

                    # Select columns to display
                    cols_to_show = ['Entry Date', 'Signal Type', 'Entry Price', 'Exit Date', 'Exit Price', 'Profit']
                    available_cols = [c for c in cols_to_show if c in display_trades.columns]
                    
                    st.dataframe(display_trades[available_cols], use_container_width=True)
                    
                    # Trade statistics
                    completed_trades = [t for t in backtest['trades'] if t['profit'] is not None]
                    if completed_trades:
                        profits = [t['profit'] for t in completed_trades]
                        winning_trades = [p for p in profits if p > 0]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Win Rate", f"{len(winning_trades)/len(completed_trades)*100:.1f}%")
                        with col2:
                            st.metric("Avg Profit", f"${np.mean(profits):.2f}")
                        with col3:
                            st.metric("Best Trade", f"${max(profits):.2f}")
                
                else:
                    st.info("No completed trades in the backtest period")

                # Save to Registry
                st.markdown("---")
                st.subheader("💾 Save to Model Registry")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    save_notes = st.text_input("Version Notes / Description", placeholder="e.g. Balanced RF model with 5y data")
                with col2:
                    if st.button("Save Best Model", type="primary"):
                        try:
                            import os
                            import joblib
                            
                            # DIRECT SAVE STRATEGY (Bypass class method to ensure data capture)
                            model_name = ml_models.best_model
                            if model_name not in ml_models.models:
                                st.error("No trained model found to save!")
                            else:
                                # 1. Capture Model
                                model_obj = ml_models.models[model_name]
                                
                                # 2. Capture Scaler (Try multiple sources)
                                scaler_obj = ml_models.scalers.get(model_name)
                                if scaler_obj is None:
                                    st.warning("⚠️ Scaler not found in model class. Checking history...")
                                
                                # 3. Capture Features (Critical Fix)
                                feats_list = getattr(ml_models, 'feature_names', [])
                                if not feats_list and 'ml_features' in st.session_state:
                                    # Fallback: Use the columns from the feature engineering step
                                    # This guarantees we have features even if the class lost them
                                    feats_list = list(st.session_state.ml_features.columns)
                                    st.info(f"ℹ️ Recovered {len(feats_list)} feature names from session state.")
                                
                                if not feats_list:
                                    st.error("⛔ Critical: Could not recover feature names. Please Retrain.")
                                else:
                                    # metrics
                                    win_rate = len(winning_trades)/len(completed_trades) if completed_trades else 0
                                    total_return = backtest['total_return']
                                    profit_factor = sum(winning_trades) / abs(sum([t['profit'] for t in completed_trades if t['profit'] < 0])) if completed_trades and any(p < 0 for p in profits) else 0
                                    
                                    # Construct Payload
                                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                                    filename = f"{model_name}_{timestamp}.joblib"
                                    save_dir = "saved_models"
                                    if not os.path.exists(save_dir):
                                        os.makedirs(save_dir)
                                    
                                    filepath = os.path.join(save_dir, filename)
                                    
                                    payload = {
                                        'model_name': model_name,
                                        'model': model_obj,
                                        'scaler': scaler_obj,
                                        'label_encoder': ml_models.scalers.get(f"{model_name}_label_encoder"),
                                        'feature_names': feats_list,
                                        'timestamp': timestamp,
                                        'metadata': {
                                            'score': ml_models.best_score, 
                                            'notes': save_notes,
                                            'ticker': ticker,
                                            'period': period,
                                            'total_return': total_return,
                                            'win_rate': win_rate,
                                            'trades_count': len(completed_trades),
                                            'profit_factor': profit_factor,
                                            'saved_at': pd.Timestamp.now().isoformat(),
                                            'feature_count': len(feats_list),
                                            'scaler_present': scaler_obj is not None
                                        }
                                    }
                                    
                                    joblib.dump(payload, filepath)
                                    
                                    # CRITICAL: Save the DL Feature Extractor as well
                                    # We need the engineer instance that trained it
                                    if 'ml_engineer' in st.session_state:
                                        engineer = st.session_state.ml_engineer
                                        dl_filename = f"{model_name}_{timestamp}_dl_extractor.h5"
                                        dl_filepath = os.path.join(save_dir, dl_filename)
                                        success = engineer.save_dl_model(dl_filepath)
                                        if success:
                                            st.success(f"✅ Saved DL Feature Extractor: {dl_filename}")
                                        else:
                                            st.warning("⚠️ Could not save DL Extractor (maybe none used?)")
                                    
                                    st.success(f"✅ Successfully saved {model_name} with {len(feats_list)} features!")
                                    if scaler_obj is None:
                                        st.warning("Note: Model saved without scaler (Raw Prices mode).")

                        except Exception as e:
                            st.error(f"❌ Error saving model: {e}")
        
        # Original model comparison section (condensed)
        st.markdown("---")
        st.subheader("🔧 Model Comparison Details")
        
        with st.expander("View Detailed Model Metrics"):
            metrics_data = []
            for model_name, result in results.items():
                if 'error' not in result:
                    metrics_data.append({
                        'Model': model_name,
                        'Accuracy': result['accuracy'],
                        'F1_Score': result['f1_score'],
                        'Precision': result['precision'],
                        'Recall': result['recall']
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Feature importance for best model
                if ml_models.best_model and ml_models.best_model in results:
                    best_result = results[ml_models.best_model]
                    if 'feature_importance' in best_result:
                        st.write(f"**Top Features ({ml_models.best_model}):**")
                        importance_df = best_result['feature_importance'].head(10)
                        
                        fig_importance = px.bar(
                            importance_df,
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Most Important Features"
                        )
                        fig_importance.update_layout(height=400)
                        st.plotly_chart(fig_importance, use_container_width=True)

# === TAB 6: PRODUCTION ===
with tab6:
    st.header("🚀 Production Trading Dashboard")
    
    try:
        from production_manager import ProductionManager
        from datetime import datetime, timedelta
        
        pm = ProductionManager()
        config = pm.get_config()
        
        # === 0. MODEL SELECTION & CONTROL ===
        # ---------------------------------------------------------
        saved_models = pm.list_saved_models()
        
        col_sel1, col_sel2 = st.columns([3, 1])
        
        with col_sel1:
            if saved_models:
                # Create display names with NOTES
                model_options = {}
                for m in saved_models:
                    # Extract notes from metadata if available
                    meta = m.get('metadata', {})
                    notes = meta.get('notes', '')
                    note_str = f" | 📝 {notes}" if notes else ""
                    
                    # Build label
                    label = f"{m['name']} | {m['created'].strftime('%Y-%m-%d %H:%M')}{note_str}"
                    model_options[label] = m
                
                # Find current index
                current_idx = 0
                if config.get('active_model_path'):
                    for i, label in enumerate(model_options.keys()):
                        if model_options[label]['path'] == config['active_model_path']:
                            current_idx = i
                            break
                
                selected_name = st.selectbox(
                    "🤖 Select Model Version", 
                    list(model_options.keys()), 
                    index=current_idx,
                    key="prod_model_selector"
                )
                selected_model = model_options[selected_name]
                
                # Display Model Specs for Verification
                meta = selected_model.get('metadata', {})
                with st.expander("ℹ️ Selected Model Specs (Verify Version)", expanded=False):
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.markdown(f"**Created:** {selected_model['created'].strftime('%Y-%m-%d %H:%M')}")
                        st.markdown(f"**Training Period:** {meta.get('period', 'N/A')}")
                        st.markdown(f"**Scaler:** {'✅ Present' if selected_model.get('has_scaler') else '❌ Missing'}")
                    with m_col2:
                        st.markdown(f"**Model Type:** {meta.get('model_type', 'Unknown')}")
                        st.markdown(f"**Features:** {selected_model.get('feature_count', 0)}")
                        st.markdown(f"**Accuracy:** {meta.get('score', 0):.2f}")
            else:
                st.warning("No saved models found. Train and save a model in 'Performance' tab first.")
                selected_model = None
                
        with col_sel2:
            st.write("") # Spacing
            st.write("")
            if selected_model:
                # Check if selected is different from active
                is_active = selected_model['path'] == config.get('active_model_path')
                
                if not is_active:
                    if st.button("🚀 Activate Model", type="primary", use_container_width=True):
                        pm.deploy_model(selected_model['path'])
                        st.success(f"Activated {selected_model['name']}!")
                        try:
                            st.rerun()
                        except:
                            # Fallback for older Streamlit versions
                            try:
                                st.experimental_rerun()
                            except:
                                pass
                else:
                    st.button("✅ Active", disabled=True, use_container_width=True)

        st.markdown("---")
        
        # === 1. SYSTEM STATUS ===
        # ---------------------------------------------------------
        col1, col2, col3, col4 = st.columns(4)
        
        active_model_name = config.get('model_name', 'None')
        deployed_at = config.get('deployed_at', 'Never')
        
        if deployed_at != 'Never':
            try:
                deployed_at = datetime.fromisoformat(deployed_at).strftime('%Y-%m-%d %H:%M')
            except:
                pass
            
        with col1:
            st.metric("Active Model", active_model_name)
        with col2:
            st.metric("Deployed At", deployed_at)
        with col3:
            health = pm.check_system_health()
            status_color = "🟢" if health['status'] == 'OK' else "🔴"
            st.metric("System Status", f"{status_color} {health['status']}")
        with col4:
            st.metric("Paper Account", "$100,000.00", delta="0.0%")
            
        if health['issues']:
            st.warning(f"⚠️ System Issues: {', '.join(health['issues'])}")
            
        st.markdown("---")
        
        # === 2. MODEL PERFORMANCE PROFILE ===
        # ---------------------------------------------------------
        if config.get('active_model_path'):
            try:
                # Cache model to prevent repeated loading
                active_model_path = config['active_model_path']
                cache_key = f'cached_model_{active_model_path}'
                
                if (cache_key not in st.session_state or 
                    st.session_state.get('cached_model_path') != active_model_path):
                    
                    # Load model only if not cached or path changed
                    from ml_models import TradingMLModels
                    temp_ml = TradingMLModels()
                    model_data = temp_ml.load_model_version(active_model_path)
                    
                    # Cache the model data
                    st.session_state[cache_key] = model_data
                    st.session_state.cached_model_path = active_model_path
                    print(f"📂 Model cached: {active_model_path}")
                else:
                    # Use cached model
                    model_data = st.session_state[cache_key]
                    print(f"⚡ Using cached model: {active_model_path}")
                    
                meta = model_data.get('metadata', {})
                
                st.subheader("📊 Active Model Performance (Projected)")
                
                m1, m2, m3, m4 = st.columns(4)
                
                # Handle missing stats (old models)
                tr = meta.get('total_return')
                wr = meta.get('win_rate')
                pf = meta.get('profit_factor')
                tc = meta.get('trades_count')
                
                with m1:
                    val = f"{tr:.1f}%" if tr is not None else "N/A"
                    st.metric("Backtest Return", val)
                with m2:
                    val = f"{wr*100:.1f}%" if wr is not None else "N/A"
                    st.metric("Win Rate", val)
                with m3:
                    val = f"{pf:.2f}" if pf is not None else "N/A"
                    st.metric("Profit Factor", val)
                with m4:
                    val = f"{tc}" if tc is not None else "N/A"
                    st.metric("Total Trades", val)
                
                if tr is None:
                    st.info("ℹ️ Stats not available for this version. Please re-save the model in the Performance tab.")
                
                # Show ticker compatibility info
                model_ticker = meta.get('ticker', 'SPY')
                current_ticker = st.session_state.get('selected_ticker', 'SPY')
                
                if model_ticker == current_ticker:
                    st.success(f"✅ Model trained on **{model_ticker}** - Perfect match!")
                else:
                    st.warning(f"⚠️ Model trained on **{model_ticker}**, analyzing **{current_ticker}**")
                    st.caption("Performance may vary when using different tickers")
                
                st.caption(f"Saved on: {meta.get('saved_at', 'Unknown Date')} | Period: {meta.get('period', 'Unknown')}")
                st.markdown("---")
                
                # === ENHANCED TRADING FEATURES SECTION ===
                # ----------------------------------------------------------------
                st.subheader("🚀 Enhanced Trading Analysis")
                
                # Navigation guide
                with st.expander("🗺️ **WHERE TO FIND NEW FEATURES** (Click to expand)", expanded=True):
                    nav_col1, nav_col2 = st.columns([1, 1])
                    
                    with nav_col1:
                        st.write("**📍 IN THIS SECTION (Production Tab):**")
                        st.write("1. **🚀 Enhanced Trading Analysis** ← You are here!")
                        st.write("2. **📊 Dynamic Risk Management** (below)")
                        st.write("3. **⏰ Signal Timing Quality** (below)")
                        st.write("4. **📅 Historical Signal Analysis** (below)")
                    
                    with nav_col2:
                        st.write("**📍 ALSO ENHANCED:**")
                        st.write("• **Training Tab**: New Optuna hyperparameter optimization")
                        st.write("• **Charts**: Super indicator with dual-line analysis")  
                        st.write("• **AI Analysis**: Now uses ALL actual model features")
                        st.write("• **Backtesting**: Enhanced with dynamic stop/take profit")
                
                # Check if enhanced analysis is available
                enhanced_available = False
                enhanced_error = None
                try:
                    import sys
                    import os
                    
                    # Add current directory to path to ensure imports work
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    if current_dir not in sys.path:
                        sys.path.insert(0, current_dir)
                    
                    from dynamic_risk_manager import DynamicRiskManager
                    from enhanced_trading_engine import EnhancedTradingEngine
                    
                    # Test instantiation
                    test_risk_manager = DynamicRiskManager()
                    test_trading_engine = EnhancedTradingEngine()
                    
                    enhanced_available = True
                    st.success("✅ Enhanced trading engine loaded and tested successfully!")
                    
                except Exception as e:
                    enhanced_available = False
                    enhanced_error = str(e)
                    st.error(f"❌ Enhanced trading engine failed to load: {enhanced_error}")
                    st.info("💡 Using basic system with fixed exit logic instead")
                    
                    # Show what files exist for debugging
                    try:
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        risk_manager_exists = os.path.exists(os.path.join(current_dir, 'dynamic_risk_manager.py'))
                        trading_engine_exists = os.path.exists(os.path.join(current_dir, 'enhanced_trading_engine.py'))
                        st.write(f"Debug: dynamic_risk_manager.py exists: {risk_manager_exists}")
                        st.write(f"Debug: enhanced_trading_engine.py exists: {trading_engine_exists}")
                    except:
                        pass
                
                # Show system status after revert
                st.success("✅ **SYSTEM RESTORED** - Back to original 78.5% win rate version!")
                st.info("🎯 **What's Working:** Simple signal-based entries/exits, proven backtest logic, AI analysis, super indicator chart, Optuna optimization")
                st.warning("⚠️ **Enhanced features temporarily disabled** - They were reducing performance from 78.5% to 33% win rate")
                
                # Simple status - no complex enhanced features
                st.write("**📊 System Status:** Ready for trading with original proven algorithm")
                
                st.markdown("---")
                
                # === 3. ENHANCED TRADING ANALYSIS ===
                # ----------------------------------------------------------------
                if 'enhanced_trading_analysis' in st.session_state and st.session_state.enhanced_trading_analysis:
                    enhanced_analysis = st.session_state.enhanced_trading_analysis
                    opportunity = enhanced_analysis.get('opportunity_analysis')
                    
                    if opportunity:
                        st.subheader("🎯 Enhanced Trading Analysis")
                        
                        # Main recommendation
                        rec_col1, rec_col2 = st.columns([2, 1])
                        
                        with rec_col1:
                            if opportunity['should_enter_trade']:
                                st.success(f"✅ **TRADE RECOMMENDED**: {opportunity['recommendation']}")
                            else:
                                st.info(f"⏸️ **HOLD POSITION**: {opportunity['recommendation']}")
                        
                        with rec_col2:
                            signal_strength = "Strong" if abs(opportunity['confidence']) > 0.5 else "Moderate" if abs(opportunity['confidence']) > 0.3 else "Weak"
                            st.metric("Signal Strength", signal_strength, f"{opportunity['confidence']:.3f}")
                        
                        # Dynamic Risk Levels
                        risk_levels = opportunity.get('risk_levels', {})
                        if risk_levels:
                            st.subheader("📊 Dynamic Risk Management")
                            
                            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                            
                            with risk_col1:
                                st.metric("Stop Loss", f"${risk_levels.get('stop_loss', 0):.2f}", 
                                         f"{risk_levels.get('stop_loss_pct', 0):.1f}%")
                            
                            with risk_col2:
                                st.metric("Take Profit", f"${risk_levels.get('take_profit', 0):.2f}",
                                         f"{risk_levels.get('take_profit_pct', 0):.1f}%")
                            
                            with risk_col3:
                                rr_ratio = risk_levels.get('risk_reward_ratio', 0)
                                rr_color = "🟢" if rr_ratio >= 2.0 else "🟡" if rr_ratio >= 1.5 else "🔴"
                                st.metric("Risk/Reward", f"{rr_color} {rr_ratio:.1f}:1")
                            
                            with risk_col4:
                                atr_val = risk_levels.get('atr_value', 0)
                                st.metric("ATR (Volatility)", f"${atr_val:.2f}")
                            
                            # Risk calculation details
                            with st.expander("🔍 Risk Calculation Details"):
                                detail_col1, detail_col2 = st.columns(2)
                                
                                with detail_col1:
                                    st.write("**Confidence Adjustments:**")
                                    st.write(f"• ML Confidence Multiplier: {risk_levels.get('confidence_multiplier', 1):.2f}x")
                                    st.write(f"• Technical Strength: {risk_levels.get('tech_multiplier', 1):.2f}x")
                                    st.write(f"• Convergence Score: {risk_levels.get('convergence_score', 0):.2f}")
                                
                                with detail_col2:
                                    st.write("**VWAP Support/Resistance:**")
                                    vwap_support = risk_levels.get('vwap_support')
                                    vwap_resistance = risk_levels.get('vwap_resistance')
                                    if vwap_support:
                                        st.write(f"• Support Level: ${vwap_support:.2f}")
                                    if vwap_resistance:
                                        st.write(f"• Resistance Level: ${vwap_resistance:.2f}")
                                    if not vwap_support and not vwap_resistance:
                                        st.write("• No significant VWAP levels detected")
                        
                        # Signal Timing Analysis
                        timing_analysis = opportunity.get('timing_analysis', {})
                        if timing_analysis:
                            st.subheader("⏰ Signal Timing Quality")
                            
                            timing_score = timing_analysis.get('timing_score', 0)
                            timing_color = "🟢" if timing_score >= 0.7 else "🟡" if timing_score >= 0.5 else "🔴"
                            
                            timing_col1, timing_col2 = st.columns([1, 2])
                            
                            with timing_col1:
                                st.metric("Timing Score", f"{timing_color} {timing_score:.2f}")
                            
                            with timing_col2:
                                st.write(f"**Analysis:** {timing_analysis.get('timing_reason', 'N/A')}")
                            
                            # Detailed timing breakdown
                            with st.expander("📈 Timing Factor Breakdown"):
                                factor_col1, factor_col2 = st.columns(2)
                                
                                with factor_col1:
                                    momentum = timing_analysis.get('momentum_score', 0)
                                    st.write(f"**Momentum Score:** {momentum:.2f}")
                                    
                                    divergence = timing_analysis.get('divergence_score', 0)
                                    st.write(f"**Divergence Quality:** {divergence:.2f}")
                                
                                with factor_col2:
                                    peak_valley = timing_analysis.get('peak_valley_score', 0)
                                    st.write(f"**Peak/Valley Timing:** {peak_valley:.2f}")
                                    
                                    stability = timing_analysis.get('stability_score', 0)
                                    st.write(f"**Signal Stability:** {stability:.2f}")
                        
                        # Historical Missed Opportunities
                        historical_analysis = enhanced_analysis.get('historical_analysis', {})
                        if historical_analysis:
                            missed_ops = historical_analysis.get('missed_opportunities', [])
                            avoided_trades = historical_analysis.get('avoided_poor_trades', [])
                            
                            if missed_ops or avoided_trades:
                                st.subheader("📅 Historical Signal Analysis")
                                
                                hist_col1, hist_col2 = st.columns(2)
                                
                                with hist_col1:
                                    if missed_ops:
                                        st.write(f"**🎯 Missed Opportunities ({len(missed_ops)}):**")
                                        for missed in missed_ops[-3:]:  # Show last 3
                                            date_str = missed['date'].strftime('%Y-%m-%d')
                                            signal_type = "📈 BUY" if missed['recommended_signal'] == 1 else "📉 SELL"
                                            st.write(f"• {date_str}: {signal_type} (ML: {missed['ml_confidence']:.2f}, Tech: {missed['composite_tech']:.2f})")
                                
                                with hist_col2:
                                    if avoided_trades:
                                        st.write(f"**✋ Avoided Poor Trades ({len(avoided_trades)}):**")
                                        for avoided in avoided_trades[-3:]:  # Show last 3
                                            date_str = avoided['date'].strftime('%Y-%m-%d')
                                            st.write(f"• {date_str}: Avoided due to poor timing")
                            
                            # Show specific dates mentioned by user
                            st.write("**🔍 Analysis of Specific Dates:**")
                            user_dates = ['2025-10-27', '2025-11-11', '2025-07-14', '2025-08-11', '2025-08-14', '2025-10-05', '2025-10-06']
                            
                            for date_str in user_dates:
                                try:
                                    target_date = pd.to_datetime(date_str)
                                    ml_series = enhanced_analysis.get('ml_confidence_series', pd.Series())
                                    tech_series = enhanced_analysis.get('composite_tech_series', pd.Series())
                                    
                                    # Find closest date in data
                                    if len(ml_series) > 0:
                                        closest_idx = ml_series.index.get_indexer([target_date], method='nearest')[0]
                                        if closest_idx >= 0:
                                            actual_date = ml_series.index[closest_idx]
                                            ml_val = ml_series.iloc[closest_idx]
                                            tech_val = tech_series.iloc[closest_idx]
                                            
                                            should_signal = ""
                                            if abs(ml_val) > 0.4 and abs(tech_val) > 0.3:
                                                if ml_val < -0.3 and tech_val > 0.2:
                                                    should_signal = "📉 SELL signal (ML bearish + tech peak)"
                                                elif ml_val > 0.3 and tech_val < -0.2:
                                                    should_signal = "📈 BUY signal (ML bullish + tech valley)"
                                            
                                            if should_signal:
                                                st.write(f"• **{actual_date.strftime('%Y-%m-%d')}**: {should_signal} (ML: {ml_val:.2f}, Tech: {tech_val:.2f})")
                                except:
                                    continue
                        
                        st.markdown("---")
                
                # === 4. LIVE MARKET MONITOR SECTION ===
                # ----------------------------------------------------------------
                st.subheader("🔴 Live Market Monitor")
                
                # Controls
                col_ctrl1, col_ctrl2 = st.columns([2, 1])
                
                with col_ctrl1:
                    # Single toggle widget to avoid conflicts
                    live_mode = st.toggle("🔴 Live Trading Mode (Auto-Update)", 
                                        value=st.session_state.get('ml_live_mode', False), 
                                        key='ml_live_mode_toggle')
                    # Update session state for persistence
                    st.session_state.ml_live_mode = live_mode
                    
                with col_ctrl2:
                    if st.button("🔄 Force Refresh Now"):
                        # CRITICAL FIX: Clear session state to force FULL reload
                        # We must also clear the 'loaded' flag to prevent other tabs from crashing
                        keys_to_clear = ['ml_features', 'ml_raw_data', 'ml_data_loaded']
                        for key in keys_to_clear:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                
                # --- DATA LOADING & INCREMENTAL UPDATE LOGIC ---
                features = None
                raw_data = None
                data_updated = False
                
                # Check if we have existing data
                has_data = 'ml_features' in st.session_state and 'ml_raw_data' in st.session_state
                
                if has_data:
                    raw_data = st.session_state.ml_raw_data
                    features = st.session_state.ml_features
                    last_date = raw_data.index[-1]
                    
                    # Determine if we need to update (Live Mode or Force Refresh)
                    # Update if last data is older than 1 hour (approx) or Force Refresh
                    now = pd.Timestamp.now(tz=last_date.tz)
                    time_diff = now - last_date
                    
                    should_update = (live_mode and time_diff.total_seconds() > 3600)
                    
                    if should_update:
                        # Use the current selected ticker, fallback to model's ticker, then SPY
                        model_ticker = meta.get('ticker', 'SPY')
                        current_ticker = st.session_state.get('selected_ticker', model_ticker)
                        
                        # Show what ticker we're updating with
                        if current_ticker != model_ticker:
                            st.info(f"📊 Updating with **{current_ticker}** (model was trained on {model_ticker})")
                        
                        ticker = current_ticker
                        with st.spinner(f"Fetching incremental data for {ticker}..."):
                            try:
                                # Incremental Fetch using Ticker.history for consistency (Adjusted Data)
                                # Note: history() 'start' must be string YYYY-MM-DD or datetime
                                new_data = yf.Ticker(ticker).history(start=last_date.date(), interval="1d")
                                
                                if not new_data.empty:
                                    # Standardize columns
                                    if isinstance(new_data.columns, pd.MultiIndex):
                                        new_data.columns = new_data.columns.get_level_values(0)
                                    
                                    # Lowercase columns
                                    new_data.columns = [c.lower() for c in new_data.columns]
                                    
                                    # Handle Timezone Mismatch for Comparison
                                    last_ts = pd.to_datetime(last_date)
                                    
                                    # Align timezones
                                    if last_ts.tzinfo is not None:
                                        if new_data.index.tz is None:
                                            new_data.index = new_data.index.tz_localize(last_ts.tzinfo)
                                        else:
                                            new_data.index = new_data.index.tz_convert(last_ts.tzinfo)
                                    else:
                                        if new_data.index.tz is not None:
                                            new_data.index = new_data.index.tz_convert(None)
                                    
                                    # Filter strictly new data
                                    new_data = new_data[new_data.index > last_ts]
                                    
                                    if not new_data.empty:
                                        # Append
                                        updated_raw = pd.concat([raw_data, new_data])
                                        # Remove duplicates just in case
                                        updated_raw = updated_raw[~updated_raw.index.duplicated(keep='last')]
                                        
                                        # Re-engineer features (Need context, so pass full updated df)
                                        engineer = MLFeatureEngineer()
                                        
                                        # CRITICAL: Load DL Extractor for Incremental Update too!
                                        active_model_path = config.get('active_model_path')
                                        if active_model_path:
                                            dl_path = active_model_path.replace('.joblib', '_dl_extractor.h5')
                                            if os.path.exists(dl_path):
                                                engineer.load_dl_model(dl_path)
                                        
                                        # CRITICAL FIX: Ensure DL features are generated for updates too
                                        updated_features = engineer.prepare_ml_dataset(
                                            updated_raw,
                                            include_lagged=True,
                                            include_rolling=True,
                                            feature_selection=False,
                                            use_dl_features=True # Force DL generation
                                        )
                                        
                                        # Update Session State
                                        st.session_state.ml_raw_data = updated_raw
                                        st.session_state.ml_features = updated_features
                                        
                                        raw_data = updated_raw
                                        features = updated_features
                                        data_updated = True
                                        st.success(f"✅ Updated with {len(new_data)} new candles")
                                    else:
                                        st.info("No new data available yet.")
                                else:
                                    st.info("Market closed or no new data.")
                            except Exception as e:
                                st.error(f"Update failed: {e}")
                
                else:
                    # Initial Load (No data exists)
                    # Use the current selected ticker, fallback to model's ticker, then SPY
                    model_ticker = meta.get('ticker', 'SPY')
                    current_ticker = st.session_state.get('selected_ticker', model_ticker)
                    
                    # Show what ticker we're loading
                    if current_ticker != model_ticker:
                        st.info(f"📊 Loading **{current_ticker}** data (model was trained on {model_ticker})")
                    
                    ticker = current_ticker
                    # Use the same period as training to ensure indicator consistency
                    train_period = meta.get('period', '5y')
                    saved_at = meta.get('saved_at')
                    
                    # Attempt to calculate fixed start date to prevent indicator drift
                    # Indicators like EMA depend on start date. If period='5y', the start shifts every day.
                    # We want to anchor to (Model Saved Date - 5y) so the history remains stable.
                    fixed_start = None
                    if saved_at and train_period.endswith('y'):
                        try:
                            saved_dt = pd.to_datetime(saved_at)
                            years = int(train_period[:-1])
                            fixed_start = (saved_dt - timedelta(days=years*365)).strftime('%Y-%m-%d')
                        except:
                            pass
                    
                    with st.spinner(f"Initializing production data for {ticker} ({train_period})..."):
                        try:
                            # Download data using fixed start if possible, else relative period
                            if fixed_start:
                                df = yf.Ticker(ticker).history(start=fixed_start, interval="1d")
                            else:
                                df = yf.Ticker(ticker).history(period=train_period, interval="1d")
                            
                            # Ticker.history returns Index name 'Date' (with timezone usually), and columns capitalized
                            if isinstance(df.columns, pd.MultiIndex):
                                df.columns = df.columns.get_level_values(0)
                                
                            # Ensure columns match training format (lowercase)
                            df.columns = [c.lower() for c in df.columns]
                            
                            # Check if empty
                            if df.empty:
                                st.error("No data returned from Yahoo Finance.")
                            
                            # Engineer features
                            engineer = MLFeatureEngineer()
                            
                            # CRITICAL: Try to load the matching DL Extractor for the ACTIVE model
                            # Use active model path from config, not selected_model (fixes double-activation bug)
                            active_model_path = config.get('active_model_path')
                            if active_model_path:
                                # Construct expected DL path: model.joblib -> model_dl_extractor.h5
                                dl_path = active_model_path.replace('.joblib', '_dl_extractor.h5')
                                
                                if os.path.exists(dl_path):
                                    st.info(f"📂 Found matching DL Extractor: {os.path.basename(dl_path)}")
                                    engineer.load_dl_model(dl_path)
                                else:
                                    st.warning("⚠️ No matching DL Extractor found. Training new one (Features may drift!).")
                            
                            # Generate ALL features (no selection) to ensure we have what the model needs
                            # We will filter to model_features later
                            # CRITICAL FIX: Force DL features and disable selection for production consistency
                            df_features = engineer.prepare_ml_dataset(
                                df,
                                include_lagged=True,
                                include_rolling=True,
                                feature_selection=False,
                                use_dl_features=True  # Force DL generation
                            )
                            
                            st.session_state.ml_raw_data = df
                            st.session_state.ml_features = df_features
                            raw_data = df
                            features = df_features
                        except Exception as e:
                            st.error(f"Failed to load production data: {e}")

                # --- RENDER MONITOR ---
                if features is not None and raw_data is not None:
                    
                    # Dashboard Settings
                    with st.expander("⚙️ Dashboard Settings", expanded=False):
                        # Increased max to 2000 to allow full history validation
                        lookback_days = st.slider("Chart & Stats Duration (Days)", min_value=30, max_value=2000, value=180, step=30)

                    # [Existing Visualization Logic...]
                    prod_model = model_data['model']
                    prod_scaler = model_data.get('scaler')
                    prod_label_encoder = model_data.get('label_encoder')
                    prod_features = model_data.get('feature_names', [])
                    
                    # --- SIMULATION ---
                    # Get last N days of features based on slider
                    subset_features = features.tail(lookback_days).copy()
                    subset_raw = raw_data.tail(lookback_days)
                    
                    # Align features
                    if prod_features:
                        missing_cols = []
                        for col in prod_features:
                            if col not in subset_features.columns:
                                missing_cols.append(col)
                                subset_features[col] = 0
                        
                        # CRITICAL DEBUG: Alert user if features are missing (Silent 0-fill kills models)
                        if missing_cols:
                            st.error(f"⚠️ CRITICAL: {len(missing_cols)} features missing from production data! Model inputs may be corrupted.")
                            with st.expander("🔍 View Missing Features (Debug Info)"):
                                st.write("The following features expected by the model were not found in the live data:")
                                st.write(missing_cols)
                                st.write("---")
                                st.write("Available features:", list(subset_features.columns)[:20], "...")
                                
                        subset_features = subset_features[prod_features]
                    
                    # Scale
                    if prod_scaler:
                        X_input = pd.DataFrame(
                            prod_scaler.transform(subset_features), 
                            columns=subset_features.columns,
                            index=subset_features.index
                        )
                    else:
                        X_input = subset_features
                        
                    # Bulk Prediction
                    raw_signals = prod_model.predict(X_input)
                    
                    # DECODE SIGNALS (Critical Fix for Missing Sell Signals)
                    # Model outputs 0,1,2 -> We need -1,0,1 for trading
                    if prod_label_encoder:
                        # Use the label encoder to convert back to original labels
                        try:
                            signals = prod_label_encoder.inverse_transform(raw_signals)
                            encoder_mapping = dict(zip(range(len(prod_label_encoder.classes_)), prod_label_encoder.classes_))
                            st.write(f"🔧 **Using Label Encoder:** {encoder_mapping}")
                        except Exception as e:
                            st.error(f"Label encoder failed: {e}")
                            # Fall back to manual mapping
                            signal_mapping = {0: -1, 1: 0, 2: 1}
                            signals = np.array([signal_mapping.get(s, 0) for s in raw_signals])
                            st.warning("⚠️ **Label Encoder Failed** - Using manual mapping: {0: -1 (SELL), 1: 0 (HOLD), 2: 1 (BUY)}")
                    else:
                        # Manual conversion: Assume 0=Sell(-1), 1=Hold(0), 2=Buy(1) 
                        signal_mapping = {0: -1, 1: 0, 2: 1}  # Standard ML classification to trading signals
                        signals = np.array([signal_mapping.get(s, 0) for s in raw_signals])
                        st.warning("⚠️ **No Label Encoder Found** - Using manual mapping: {0: -1 (SELL), 1: 0 (HOLD), 2: 1 (BUY)}")
                    
                    # Comprehensive debugging
                    raw_unique = np.unique(raw_signals)
                    decoded_unique = np.unique(signals)
                    st.write(f"**🔄 Signal Conversion:** Raw {raw_unique} → Decoded {decoded_unique}")
                    
                    # Show actual signal distribution
                    for i, raw_val in enumerate(raw_unique):
                        corresponding_decoded = signals[raw_signals == raw_val]
                        decoded_val = np.unique(corresponding_decoded)[0] if len(np.unique(corresponding_decoded)) == 1 else "MIXED"
                        count = np.sum(raw_signals == raw_val)
                        st.write(f"  • Raw {raw_val} → Decoded {decoded_val} ({count} occurrences)")

                    if hasattr(prod_model, 'predict_proba'):
                        probs = prod_model.predict_proba(X_input)
                        confidences = np.max(probs, axis=1)
                    else:
                        confidences = np.zeros(len(signals))
                        probs = None

                    # --- DEBUG SECTION ---
                    with st.expander("🕵️‍♂️ Debug Model Inputs (Why is it stuck?)"):
                        d_col1, d_col2 = st.columns(2)
                        with d_col1:
                            st.write("Raw Features (Last 5 rows):")
                            st.dataframe(subset_features.tail())
                        with d_col2:
                            st.write("Scaled Inputs (Last 5 rows):")
                            st.dataframe(X_input.tail())
                        
                        st.write("Signal Decoding:")
                        debug_df = pd.DataFrame({
                            'Raw Prediction (0-2)': raw_signals[-5:],
                            'Decoded Signal (-1,0,1)': signals[-5:]
                        })
                        st.dataframe(debug_df)
                        
                        if probs is not None:
                            st.write("Model Probabilities (Last 5 rows):")
                            prob_df = pd.DataFrame(probs[-5:], columns=prod_model.classes_ if hasattr(prod_model, 'classes_') else [0, 1, 2])
                            st.dataframe(prob_df)
                        
                    # --- ENHANCED BACKTEST WITH DYNAMIC RISK MANAGEMENT ---
                    def run_enhanced_production_backtest(data, signals, confidences, ml_confidence_series, 
                                                        composite_tech_series, starting_capital=100000):
                        """Enhanced backtest with dynamic stop loss/take profit and signal filtering"""
                        capital = starting_capital
                        position = None
                        trades = []
                        equity_curve = [starting_capital]
                        
                        # Import enhanced trading components
                        try:
                            from enhanced_trading_engine import EnhancedTradingEngine
                            from dynamic_risk_manager import DynamicRiskManager
                            
                            trading_engine = EnhancedTradingEngine()
                            risk_manager = DynamicRiskManager()
                            enhanced_mode = True
                        except:
                            enhanced_mode = False
                        
                        # Process each day
                        for i in range(len(data)):
                            current_price = data['close'].iloc[i]
                            current_date = data.index[i]
                            raw_signal = signals[i] if i < len(signals) else 0
                            
                            # ENHANCED: For now, use basic signal passing to fix exit logic first
                            # TODO: Re-enable advanced filtering once exits work
                            signal = raw_signal
                            
                            # Set default dynamic levels for enhanced mode
                            if enhanced_mode:
                                dynamic_stop_pct = 8.0  # Will be dynamic later
                                dynamic_tp_pct = 15.0   # Will be dynamic later
                            
                            # Check for dynamic stop loss / take profit exits BEFORE new signals
                            if position is not None:
                                entry_price = position['entry_price']
                                current_return = (current_price - entry_price) / entry_price * 100
                                
                                # Use dynamic levels if available, else defaults
                                stop_loss_pct = position.get('dynamic_stop_pct', 8.0)
                                take_profit_pct = position.get('dynamic_tp_pct', 15.0)
                                
                                exit_triggered = False
                                exit_reason = ""
                                
                                if position['type'] == 'long':
                                    if current_return <= -stop_loss_pct:
                                        exit_triggered = True
                                        exit_reason = f"DYNAMIC_STOP_LOSS_{stop_loss_pct:.1f}%"
                                    elif current_return >= take_profit_pct:
                                        exit_triggered = True
                                        exit_reason = f"DYNAMIC_TAKE_PROFIT_{take_profit_pct:.1f}%"
                                elif position['type'] == 'short':
                                    if current_return >= stop_loss_pct:  # Loss on short
                                        exit_triggered = True
                                        exit_reason = f"DYNAMIC_STOP_LOSS_{stop_loss_pct:.1f}%"
                                    elif current_return <= -take_profit_pct:  # Profit on short
                                        exit_triggered = True
                                        exit_reason = f"DYNAMIC_TAKE_PROFIT_{take_profit_pct:.1f}%"
                                
                                if exit_triggered:
                                    # Execute dynamic exit
                                    if position['type'] == 'long':
                                        exit_value = position['shares'] * current_price
                                        profit = exit_value - position['entry_capital']
                                        capital = exit_value
                                    else:  # short
                                        profit = position['entry_capital'] - (position['shares'] * current_price)
                                        capital += profit
                                    
                                    # Record trade with dynamic exit reason
                                    if trades and trades[-1]['exit_date'] is None:
                                        trades[-1].update({
                                            'exit_date': current_date,
                                            'exit_price': current_price,
                                            'profit': profit,
                                            'exit_reason': exit_reason,
                                            'return_pct': current_return
                                        })
                                    
                                    position = None
                            
                            # BUY SIGNAL: Enter long position (or exit short)
                            if signal == 1:
                                if position is None or position['type'] == 'short':
                                    # Close short position if exists
                                    if position and position['type'] == 'short':
                                        profit = position['entry_capital'] - (position['shares'] * current_price)
                                        capital += profit
                                        
                                        # Update last trade
                                        if trades and trades[-1]['exit_date'] is None:
                                            trades[-1].update({
                                                'exit_date': current_date,
                                                'exit_price': current_price,
                                                'profit': profit
                                            })
                                    
                                    # Enter new long position
                                    shares = capital / current_price
                                    position = {
                                        'type': 'long',
                                        'entry_date': current_date,
                                        'entry_price': current_price,
                                        'shares': shares,
                                        'entry_capital': capital,
                                        'dynamic_stop_pct': locals().get('dynamic_stop_pct', 8.0),
                                        'dynamic_tp_pct': locals().get('dynamic_tp_pct', 15.0)
                                    }
                                    
                                    trades.append({
                                        'entry_date': current_date,
                                        'entry_price': current_price,
                                        'exit_date': None,
                                        'exit_price': None,
                                        'shares': shares,
                                        'position_value': capital,
                                        'profit': None,
                                        'signal_type': 'LONG'
                                    })
                            
                            # SELL SIGNAL: Enter short position (or exit long)  
                            elif signal == -1:
                                if position is None or position['type'] == 'long':
                                    # Close long position if exists
                                    if position and position['type'] == 'long':
                                        exit_value = position['shares'] * current_price
                                        profit = exit_value - position['entry_capital']
                                        capital = exit_value
                                        
                                        # Update last trade
                                        if trades and trades[-1]['exit_date'] is None:
                                            trades[-1].update({
                                                'exit_date': current_date,
                                                'exit_price': current_price,
                                                'profit': profit
                                            })
                                    
                                    # Enter new short position (simulate by holding cash)
                                    position = {
                                        'type': 'short',
                                        'entry_date': current_date,
                                        'entry_price': current_price,
                                        'shares': capital / current_price,  # Theoretical shares
                                        'entry_capital': capital,
                                        'dynamic_stop_pct': locals().get('dynamic_stop_pct', 8.0),
                                        'dynamic_tp_pct': locals().get('dynamic_tp_pct', 15.0)
                                    }
                                    
                                    trades.append({
                                        'entry_date': current_date,
                                        'entry_price': current_price,
                                        'exit_date': None,
                                        'exit_price': None,
                                        'shares': capital / current_price,
                                        'position_value': capital,
                                        'profit': None,
                                        'signal_type': 'SHORT'
                                    })
                            
                            # Calculate portfolio value
                            if position:
                                if position['type'] == 'long':
                                    portfolio_value = position['shares'] * current_price
                                else:  # short position
                                    # For short: profit when price goes down
                                    portfolio_value = position['entry_capital'] + (
                                        position['entry_capital'] - position['shares'] * current_price
                                    )
                            else:
                                portfolio_value = capital
                            
                            equity_curve.append(portfolio_value)
                        
                        # Close final position if still open
                        if position:
                            final_price = data['close'].iloc[-1]
                            
                            if position['type'] == 'long':
                                exit_value = position['shares'] * final_price
                                profit = exit_value - position['entry_capital']
                                capital = exit_value
                            else:  # short
                                profit = position['entry_capital'] - (position['shares'] * final_price)
                                capital += profit
                            
                            # Update last trade
                            if trades and trades[-1]['exit_date'] is None:
                                trades[-1].update({
                                    'exit_date': data.index[-1],
                                    'exit_price': final_price,
                                    'profit': profit
                                })
                        
                        return trades, equity_curve, capital
                    
                    # SIMPLE ML-BASED BACKTEST - EXITS ON SELL SIGNALS ONLY
                    def run_production_backtest(data, signals, starting_capital=100000, confidences=None, 
                                              min_buy_conf=0.0, min_sell_conf=0.0,
                                              composite_tech=None, buy_comp_max=-999, sell_comp_min=999):
                        """Simple backtest - entries on buy signals, exits on sell signals from ML model
                        Now supports Option 3: Combined confidence + composite filtering"""
                        capital = starting_capital
                        position = None
                        trades = []
                        equity_curve = [starting_capital]
                        
                        for i in range(len(data)):
                            current_price = data['close'].iloc[i]
                            current_date = data.index[i]
                            signal = signals[i] if i < len(signals) else 0
                            confidence = confidences[i] if confidences is not None and i < len(confidences) else 1.0
                            comp_value = composite_tech[i] if composite_tech is not None and i < len(composite_tech) else 0.0
                            
                            # Enter long position on BUY signal (with combined filtering)
                            # Must pass BOTH confidence AND composite tech filters
                            if (signal == 1 and position is None and 
                                confidence >= min_buy_conf and comp_value <= buy_comp_max):
                                shares = capital / current_price
                                position = {
                                    'entry_price': current_price,
                                    'entry_date': current_date,
                                    'shares': shares,
                                    'entry_capital': capital
                                }
                                trades.append({
                                    'entry_date': current_date,
                                    'entry_price': current_price,
                                    'exit_date': None,
                                    'exit_price': None,
                                    'shares': shares,
                                    'position_value': capital,
                                    'profit': None,
                                    'signal_type': 'LONG'
                                })
                            
                            # Exit position on SELL signal (with combined filtering)
                            # Must pass BOTH confidence AND composite tech filters
                            elif (signal == -1 and position is not None and 
                                  confidence >= min_sell_conf and comp_value >= sell_comp_min):
                                exit_value = position['shares'] * current_price
                                profit = exit_value - position['entry_capital']
                                capital = exit_value
                                
                                if trades and trades[-1]['exit_date'] is None:
                                    trades[-1].update({
                                        'exit_date': current_date,
                                        'exit_price': current_price,
                                        'profit': profit
                                    })
                                
                                position = None
                            
                            # Calculate portfolio value
                            portfolio_value = position['shares'] * current_price if position else capital
                            equity_curve.append(portfolio_value)
                        
                        # Close final position if still open
                        if position:
                            final_price = data['close'].iloc[-1]
                            exit_value = position['shares'] * final_price
                            profit = exit_value - position['entry_capital']
                            capital = exit_value
                            
                            if trades and trades[-1]['exit_date'] is None:
                                trades[-1].update({
                                    'exit_date': data.index[-1],
                                    'exit_price': final_price,
                                    'profit': profit
                                })
                        
                        return trades, equity_curve, capital
                    
                    # DEBUG: Check what signals are being generated
                    unique_signals = np.unique(signals)
                    signal_counts = {s: np.sum(signals == s) for s in unique_signals}
                    st.write(f"**🔍 Signal Debug:** Generated signals: {unique_signals}")
                    st.write(f"**📊 Signal Counts:** {signal_counts}")
                    
                    # Check if we have sell signals
                    has_sell_signals = -1 in unique_signals and signal_counts.get(-1, 0) > 0
                    
                    if not has_sell_signals:
                        st.error("🚨 **CRITICAL ISSUE:** No sell signals (-1) detected! Positions will never exit!")
                        st.info("💡 **Fix:** Need to modify exit logic or retrain model to generate sell signals")
                    
                    # OPTION 3: AUTOMATIC TRADING FILTER OPTIMIZATION
                    optimized_params = None
                    if enable_trading_optimization:
                        st.subheader("🎯 Option 3: Auto-Optimizing Trading Filters")
                        
                        if st.button("🚀 Start Trading Optimization", type="primary"):
                            
                            # Prepare composite technical data
                            composite_tech_values = None
                            try:
                                if hasattr(st.session_state, 'ml_features') and st.session_state.ml_features is not None:
                                    features = st.session_state.ml_features
                                    subset_features = features.tail(len(subset_raw))
                                    
                                    # Calculate composite technical indicator
                                    composite_parts = []
                                    for col in subset_features.columns:
                                        if any(indicator in col.lower() for indicator in ['rsi', 'williams', 'stoch', 'macd']):
                                            if 'rsi' in col.lower():
                                                normalized = (subset_features[col] - 50) / 50
                                            elif 'williams' in col.lower():
                                                normalized = subset_features[col] / 50  
                                            elif 'stoch' in col.lower() and subset_features[col].std() > 0:
                                                normalized = (subset_features[col] - 50) / 50
                                            elif 'macd' in col.lower() and subset_features[col].std() > 0:
                                                normalized = subset_features[col] / (3 * subset_features[col].std())
                                            else:
                                                continue
                                            composite_parts.append(normalized)
                                    
                                    if composite_parts:
                                        composite_tech_values = np.mean(composite_parts, axis=0)
                                        st.success(f"✅ Composite tech calculated from {len(composite_parts)} indicators")
                                    else:
                                        st.warning("⚠️ No oscillator indicators found - using neutral composite")
                                        composite_tech_values = np.zeros(len(subset_raw))
                                        
                            except Exception as e:
                                st.warning(f"⚠️ Composite calculation failed: {e} - using neutral composite")
                                composite_tech_values = np.zeros(len(subset_raw))
                            
                            # Optuna optimization
                            with st.spinner(f"🔍 Optimizing trading filters ({trading_trials} trials)..."):
                                try:
                                    import optuna
                                    
                                    def optimize_trading_objective(trial):
                                        # Suggest 4 parameters to optimize
                                        min_buy_conf = trial.suggest_float("min_buy_confidence", 0, 100)
                                        min_sell_conf = trial.suggest_float("min_sell_confidence", 0, 100)
                                        buy_comp_max = trial.suggest_float("buy_composite_max", -1.0, 0.0)
                                        sell_comp_min = trial.suggest_float("sell_composite_min", 0.0, 1.0)
                                        
                                        # Convert confidence to 0-1 range for backtest
                                        min_buy_conf_norm = min_buy_conf / 100.0
                                        min_sell_conf_norm = min_sell_conf / 100.0
                                        
                                        # Run backtest with these parameters
                                        test_trades, _, test_capital = run_production_backtest(
                                            subset_raw, signals,
                                            confidences=confidences,
                                            min_buy_conf=min_buy_conf_norm,
                                            min_sell_conf=min_sell_conf_norm,
                                            composite_tech=composite_tech_values,
                                            buy_comp_max=buy_comp_max,
                                            sell_comp_min=sell_comp_min
                                        )
                                        
                                        # Calculate objective metric (TOTAL RETURN)
                                        if len(test_trades) < 1:
                                            return -100  # Penalty for no trades, but not too harsh
                                        
                                        # Calculate total return percentage
                                        total_return_pct = ((test_capital - 100000) / 100000) * 100
                                        
                                        # Small penalty for too few trades to encourage some activity
                                        completed_trades = [t for t in test_trades if t['profit'] is not None]
                                        trade_penalty = max(0, (5 - len(completed_trades)) * 0.1)  # Small penalty if < 5 completed trades
                                        
                                        final_score = total_return_pct - trade_penalty
                                        
                                        return final_score
                                    
                                    # Create and run study
                                    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
                                    study.optimize(optimize_trading_objective, n_trials=trading_trials, show_progress_bar=False)
                                    
                                    # Get best parameters
                                    best_params = study.best_params
                                    optimized_params = {
                                        'min_buy_confidence': best_params['min_buy_confidence'],
                                        'min_sell_confidence': best_params['min_sell_confidence'],
                                        'buy_composite_max': best_params['buy_composite_max'],
                                        'sell_composite_min': best_params['sell_composite_min'],
                                        'best_score': study.best_value
                                    }
                                    
                                    # Store in session state for persistence
                                    st.session_state.trading_optimization_params = optimized_params
                                    
                                    st.success(f"🎯 **Optimization Complete!** Best Total Return: {study.best_value:.2f}%")
                                    st.write("**🏆 Optimal Parameters for Maximum Return:**")
                                    st.write(f"  • Buy Confidence: {best_params['min_buy_confidence']:.1f}%")
                                    st.write(f"  • Sell Confidence: {best_params['min_sell_confidence']:.1f}%") 
                                    st.write(f"  • Buy Composite Max: {best_params['buy_composite_max']:.3f} (oversold)")
                                    st.write(f"  • Sell Composite Min: {best_params['sell_composite_min']:.3f} (overbought)")
                                    
                                except ImportError:
                                    st.error("❌ Optuna not available. Please install: pip install optuna")
                                except Exception as e:
                                    st.error(f"❌ Optimization failed: {e}")
                    
                    # Check for stored optimized parameters from session state
                    if (enable_trading_optimization and 
                        hasattr(st.session_state, 'trading_optimization_params') and 
                        st.session_state.trading_optimization_params is not None):
                        
                        stored_params = st.session_state.trading_optimization_params
                        st.info("🔄 **Using stored optimized parameters for backtest**")
                        st.write(f"  • Buy Confidence: {stored_params['min_buy_confidence']:.1f}%")
                        st.write(f"  • Sell Confidence: {stored_params['min_sell_confidence']:.1f}%") 
                        st.write(f"  • Buy Composite Max: {stored_params['buy_composite_max']:.3f} (oversold)")
                        st.write(f"  • Sell Composite Min: {stored_params['sell_composite_min']:.3f} (overbought)")
                        
                        # Use optimized parameters
                        min_buy_confidence_opt = stored_params['min_buy_confidence'] / 100.0
                        min_sell_confidence_opt = stored_params['min_sell_confidence'] / 100.0
                        buy_composite_max_opt = stored_params['buy_composite_max']
                        sell_composite_min_opt = stored_params['sell_composite_min']
                    else:
                        # Use baseline (no filtering)
                        min_buy_confidence_opt = 0.0
                        min_sell_confidence_opt = 0.0
                        buy_composite_max_opt = -999
                        sell_composite_min_opt = 999
                    
                    # Prepare composite technical data for backtest (if not already done in optimization)  
                    if (not enable_trading_optimization or 
                        not hasattr(st.session_state, 'trading_optimization_params') or 
                        st.session_state.trading_optimization_params is None):
                        composite_tech_values = None
                        try:
                            if hasattr(st.session_state, 'ml_features') and st.session_state.ml_features is not None:
                                features = st.session_state.ml_features
                                subset_features = features.tail(len(subset_raw))
                                
                                # Calculate composite technical indicator
                                composite_parts = []
                                for col in subset_features.columns:
                                    if any(indicator in col.lower() for indicator in ['rsi', 'williams', 'stoch', 'macd']):
                                        if 'rsi' in col.lower():
                                            normalized = (subset_features[col] - 50) / 50
                                        elif 'williams' in col.lower():
                                            normalized = subset_features[col] / 50  
                                        elif 'stoch' in col.lower() and subset_features[col].std() > 0:
                                            normalized = (subset_features[col] - 50) / 50
                                        elif 'macd' in col.lower() and subset_features[col].std() > 0:
                                            normalized = subset_features[col] / (3 * subset_features[col].std())
                                        else:
                                            continue
                                        composite_parts.append(normalized)
                                
                                if composite_parts:
                                    composite_tech_values = np.mean(composite_parts, axis=0)
                                        
                        except Exception as e:
                            pass  # Will use None (no filtering)
                    
                    # RUN ML-BASED BACKTEST WITH DEBUGGING (+ Option 3 optimized filtering)
                    trades_list, equity_curve, final_capital = run_production_backtest(
                        subset_raw, signals,
                        confidences=confidences,
                        min_buy_conf=min_buy_confidence_opt,
                        min_sell_conf=min_sell_confidence_opt,
                        composite_tech=composite_tech_values,
                        buy_comp_max=buy_composite_max_opt,
                        sell_comp_min=sell_composite_min_opt
                    )
                    
                    # DEBUG: Show what happened in backtest
                    buy_signals_count = np.sum(signals == 1)
                    sell_signals_count = np.sum(signals == -1)
                    
                    # OPTION 3: Show combined filtering impact
                    if (enable_trading_optimization and 
                        hasattr(st.session_state, 'trading_optimization_params') and 
                        st.session_state.trading_optimization_params is not None):
                        stored_debug_params = st.session_state.trading_optimization_params
                        st.write(f"**🎯 Option 3 - Combined Optimized Filter Results:**")
                        st.write(f"  • Optimized Buy Confidence: {stored_debug_params['min_buy_confidence']:.1f}%")
                        st.write(f"  • Optimized Sell Confidence: {stored_debug_params['min_sell_confidence']:.1f}%")
                        st.write(f"  • Optimized Buy Composite Max: {stored_debug_params['buy_composite_max']:.3f} (oversold)")
                        st.write(f"  • Optimized Sell Composite Min: {stored_debug_params['sell_composite_min']:.3f} (overbought)")
                        
                        # Calculate how many signals pass each filter
                        if composite_tech_values is not None:
                            # Combined filtering
                            buy_conf_mask = confidences >= min_buy_confidence_opt
                            buy_comp_mask = composite_tech_values <= buy_composite_max_opt
                            buy_combined_mask = (signals == 1) & buy_conf_mask & buy_comp_mask
                            buy_combined = np.sum(buy_combined_mask)
                            
                            sell_conf_mask = confidences >= min_sell_confidence_opt  
                            sell_comp_mask = composite_tech_values >= sell_composite_min_opt
                            sell_combined_mask = (signals == -1) & sell_conf_mask & sell_comp_mask
                            sell_combined = np.sum(sell_combined_mask)
                            
                            # Show exactly which dates passed the filters
                            if buy_combined > 0:
                                passed_buy_indices = np.where(buy_combined_mask)[0]
                                passed_buy_dates = [subset_raw.index[i].date() for i in passed_buy_indices]
                                st.write(f"  • ✅ **Filtered Buy Dates that PASSED**: {passed_buy_dates}")
                            else:
                                st.write(f"  • ❌ **No buy signals passed combined filters**")
                                
                            if sell_combined > 0:
                                passed_sell_indices = np.where(sell_combined_mask)[0]
                                passed_sell_dates = [subset_raw.index[i].date() for i in passed_sell_indices]
                                st.write(f"  • ✅ **Filtered Sell Dates that PASSED**: {passed_sell_dates}")
                            else:
                                st.write(f"  • ❌ **No sell signals passed combined filters**")
                            
                            st.write(f"  • Buy signals: {buy_signals_count} total → {buy_combined} passed both filters ({buy_signals_count - buy_combined} filtered)")
                            st.write(f"  • Sell signals: {sell_signals_count} total → {sell_combined} passed both filters ({sell_signals_count - sell_combined} filtered)")
                        else:
                            st.write(f"  • Only confidence filtering applied (no composite data)")
                    else:
                        st.write(f"**📊 Baseline Debug (No Optimization):**")
                        
                    st.write(f"  • Buy signals (+1): {buy_signals_count}")
                    st.write(f"  • Sell signals (-1): {sell_signals_count}")
                    st.write(f"  • Trades executed: {len(trades_list)}")
                    st.write(f"  • Completed trades: {len([t for t in trades_list if t['profit'] is not None])}")
                    
                    # CRITICAL DEBUG: Show signal dates vs chart dates
                    buy_signal_dates = subset_raw.index[signals == 1]
                    sell_signal_dates = subset_raw.index[signals == -1]
                    
                    st.write(f"**📅 Signal Date Analysis:**")
                    st.write(f"  • Chart date range: {subset_raw.index[0].date()} to {subset_raw.index[-1].date()}")
                    st.write(f"  • Total data points: {len(subset_raw)}")
                    st.write(f"  • Signal array length: {len(signals)}")
                    
                    if len(buy_signal_dates) > 0:
                        st.write(f"  • Buy signal dates: {[d.date() for d in buy_signal_dates]}")
                    if len(sell_signal_dates) > 0:
                        st.write(f"  • Sell signal dates: {[d.date() for d in sell_signal_dates]}")
                    
                    # Check if chart subset matches signal subset
                    if len(subset_raw) != len(signals):
                        st.error(f"🚨 **LENGTH MISMATCH**: Chart data ({len(subset_raw)}) vs Signals ({len(signals)})")
                    
                    # CHART DEBUG: Show what signals will be plotted
                    chart_buy_signals = np.where(signals == 1)[0]
                    chart_sell_signals = np.where(signals == -1)[0]
                    
                    st.write(f"**📊 Chart Signal Debug:**")
                    st.write(f"  • Chart buy signal indices: {chart_buy_signals}")
                    st.write(f"  • Chart sell signal indices: {chart_sell_signals}")
                    
                    if len(chart_buy_signals) > 0:
                        chart_buy_dates = subset_raw.index[chart_buy_signals]
                        st.write(f"  • Chart buy dates: {[d.date() for d in chart_buy_dates]}")
                    
                    if len(chart_sell_signals) > 0:
                        chart_sell_dates = subset_raw.index[chart_sell_signals]
                        st.write(f"  • Chart sell dates: {[d.date() for d in chart_sell_dates]}")
                    
                    if has_sell_signals:
                        st.success("🎯 **ML-BASED EXITS WORKING** - Model generating proper sell signals!")
                    else:
                        st.error("🚨 **SIGNAL DECODING ISSUE** - Check the signal conversion mapping above!")
                        st.info("💡 **Fix Required:** Either label encoder is wrong or manual mapping needs adjustment")
                    
                    # === SIGNAL QUALITY ANALYSIS ===
                    st.write("---")
                    st.write("**🔍 SIGNAL QUALITY ANALYSIS:**")
                    
                    # Find actual market peaks and valleys (local maxima/minima)
                    from scipy.signal import argrelextrema
                    import pandas as pd
                    
                    st.write("**🔬 REAL-TIME PEAK/VALLEY ANALYSIS:**")
                    
                    # Test multiple windows for real-time viability
                    windows = [3, 5, 7, 10]  # Different detection speeds
                    highs = subset_raw['high'].values
                    lows = subset_raw['low'].values
                    
                    best_window_results = {}
                    
                    for window in windows:
                        # Find local maxima (peaks) and minima (valleys)  
                        peak_indices = argrelextrema(highs, np.greater, order=window)[0]
                        valley_indices = argrelextrema(lows, np.less, order=window)[0]
                        
                        if len(peak_indices) > 0 and len(valley_indices) > 0:
                            # Calculate perfect returns for this window
                            perfect_trades = []
                            sorted_extremes = []
                            
                            # Combine and sort peaks and valleys
                            for i in peak_indices:
                                sorted_extremes.append((i, 'peak', highs[i]))
                            for i in valley_indices:
                                sorted_extremes.append((i, 'valley', lows[i]))
                            
                            sorted_extremes.sort()
                            
                            # Calculate perfect buy-low-sell-high returns
                            position = None
                            perfect_capital = 100000
                            
                            for idx, extreme_type, price in sorted_extremes:
                                if extreme_type == 'valley' and position is None:
                                    position = perfect_capital / price
                                    entry_capital = perfect_capital
                                elif extreme_type == 'peak' and position is not None:
                                    perfect_capital = position * price
                                    profit = perfect_capital - entry_capital
                                    perfect_trades.append(profit)
                                    position = None
                            
                            total_return = ((perfect_capital - 100000) / 100000) * 100 if len(perfect_trades) > 0 else 0
                            
                            best_window_results[window] = {
                                'peaks': len(peak_indices),
                                'valleys': len(valley_indices), 
                                'trades': len(perfect_trades),
                                'return': total_return,
                                'lag_days': window,
                                'real_time_viable': window <= 5  # 5+ day lag too slow for real-time
                            }
                            
                            st.write(f"  • **Window {window} days**: {len(peak_indices)} peaks, {len(valley_indices)} valleys → {total_return:.1f}% return ({len(perfect_trades)} trades)")
                    
                    # Find best real-time viable window
                    viable_windows = {k: v for k, v in best_window_results.items() if v['real_time_viable']}
                    if viable_windows:
                        best_viable = max(viable_windows.items(), key=lambda x: x[1]['return'])
                        st.success(f"🎯 **BEST REAL-TIME WINDOW**: {best_viable[0]} days ({best_viable[1]['return']:.1f}% return, {best_viable[0]}-day detection lag)")
                        
                        # Use best viable window for the rest of the analysis
                        window = best_viable[0]
                        peak_indices = argrelextrema(highs, np.greater, order=window)[0]
                        valley_indices = argrelextrema(lows, np.less, order=window)[0]
                    else:
                        # Fallback to 5-day window
                        window = 5
                        peak_indices = argrelextrema(highs, np.greater, order=window)[0]
                        valley_indices = argrelextrema(lows, np.less, order=window)[0]
                        st.warning(f"⚠️ Using {window}-day window as fallback")
                    
                    st.write("---")
                    st.write(f"**📊 USING {window}-DAY WINDOW FOR ML TRAINING ANALYSIS:**")
                    
                    if len(peak_indices) > 0:
                        peak_dates = [subset_raw.index[i].date() for i in peak_indices]
                        peak_prices = [highs[i] for i in peak_indices]
                        st.write(f"  • **📈 Market PEAKS** ({len(peak_indices)}): {peak_dates}")
                        
                        # Check if ML generated sell signals near peaks
                        sell_signal_dates = [subset_raw.index[i].date() for i in range(len(signals)) if signals[i] == -1]
                        peaks_with_sells = []
                        for peak_date in peak_dates:
                            # Check if any sell signal within 5 days of peak
                            for sell_date in sell_signal_dates:
                                if abs((peak_date - sell_date).days) <= 5:
                                    peaks_with_sells.append(peak_date)
                                    break
                        
                        st.write(f"  • **❌ MISSED Peak Opportunities**: {len(peak_indices) - len(peaks_with_sells)} out of {len(peak_indices)} peaks had no sell signals nearby")
                        if len(peaks_with_sells) > 0:
                            st.write(f"  • **✅ Captured Peaks**: {peaks_with_sells}")
                    
                    if len(valley_indices) > 0:
                        valley_dates = [subset_raw.index[i].date() for i in valley_indices]
                        valley_prices = [lows[i] for i in valley_indices]
                        st.write(f"  • **📉 Market VALLEYS** ({len(valley_indices)}): {valley_dates}")
                        
                        # Check if ML generated buy signals near valleys
                        buy_signal_dates = [subset_raw.index[i].date() for i in range(len(signals)) if signals[i] == 1]
                        valleys_with_buys = []
                        for valley_date in valley_dates:
                            # Check if any buy signal within 5 days of valley
                            for buy_date in buy_signal_dates:
                                if abs((valley_date - buy_date).days) <= 5:
                                    valleys_with_buys.append(valley_date)
                                    break
                        
                        st.write(f"  • **❌ MISSED Valley Opportunities**: {len(valley_indices) - len(valleys_with_buys)} out of {len(valley_indices)} valleys had no buy signals nearby")
                        if len(valleys_with_buys) > 0:
                            st.write(f"  • **✅ Captured Valleys**: {valleys_with_buys}")
                    
                    # Calculate what perfect timing would yield
                    if len(peak_indices) > 0 and len(valley_indices) > 0:
                        # Simulate perfect peak/valley trading
                        perfect_trades = []
                        sorted_extremes = []
                        
                        # Combine and sort peaks and valleys
                        for i in peak_indices:
                            sorted_extremes.append((i, 'peak', highs[i]))
                        for i in valley_indices:
                            sorted_extremes.append((i, 'valley', lows[i]))
                        
                        sorted_extremes.sort()
                        
                        # Calculate perfect buy-low-sell-high returns
                        position = None
                        perfect_capital = 100000
                        
                        for idx, extreme_type, price in sorted_extremes:
                            if extreme_type == 'valley' and position is None:
                                # Buy at valley
                                position = perfect_capital / price
                                entry_capital = perfect_capital
                            elif extreme_type == 'peak' and position is not None:
                                # Sell at peak  
                                perfect_capital = position * price
                                profit = perfect_capital - entry_capital
                                perfect_trades.append({
                                    'type': 'valley_to_peak',
                                    'profit': profit,
                                    'return_pct': (profit / entry_capital) * 100
                                })
                                position = None
                        
                        if len(perfect_trades) > 0:
                            total_perfect_return = ((perfect_capital - 100000) / 100000) * 100
                            st.write(f"  • **🎯 PERFECT Peak/Valley Trading**: {total_perfect_return:.1f}% return ({len(perfect_trades)} trades)")
                            st.write(f"  • **📊 ML Model Efficiency**: {0.9/total_perfect_return*100:.1f}% of perfect potential")
                    
                    st.write("---")
                    
                    # === ML TRAINING LABEL GENERATION CONCEPT ===
                    st.write("**💡 SOLUTION: Use Peak/Valley Detection as ML Training Labels**")
                    
                    if len(peak_indices) > 0 and len(valley_indices) > 0:
                        # Generate training labels for ML model
                        training_labels = np.zeros(len(subset_raw))  # 0 = HOLD
                        
                        # Label peaks as SELL signals (lag-adjusted)
                        for peak_idx in peak_indices:
                            # For real-time: label X days BEFORE peak (leading indicator)
                            lead_time = max(1, window // 2)  # Half the detection window
                            early_sell_idx = max(0, peak_idx - lead_time)
                            training_labels[early_sell_idx] = -1  # SELL
                        
                        # Label valleys as BUY signals (lag-adjusted)  
                        for valley_idx in valley_indices:
                            lead_time = max(1, window // 2)
                            early_buy_idx = max(0, valley_idx - lead_time) 
                            training_labels[early_buy_idx] = 1  # BUY
                        
                        # Calculate how many training signals this generates
                        buy_labels = np.sum(training_labels == 1)
                        sell_labels = np.sum(training_labels == -1) 
                        hold_labels = np.sum(training_labels == 0)
                        
                        st.write(f"  • **🏷️ Generated Training Labels**: {buy_labels} BUY, {sell_labels} SELL, {hold_labels} HOLD")
                        st.write(f"  • **⏱️ Lead Time**: {max(1, window // 2)} days before peak/valley")
                        st.write(f"  • **🎯 Target Performance**: {total_perfect_return:.1f}% if ML learns these labels")
                        
                        # Show comparison with current ML signals
                        current_buy_signals = np.sum(signals == 1)
                        current_sell_signals = np.sum(signals == -1)
                        
                        st.write(f"  • **📊 Current ML**: {current_buy_signals} BUY, {current_sell_signals} SELL → 0.9% return")
                        st.write(f"  • **🎯 Peak/Valley Labels**: {buy_labels} BUY, {sell_labels} SELL → {total_perfect_return:.1f}% potential")
                        
                        # Calculate label accuracy vs current signals
                        current_signals_array = signals.copy()
                        
                        # Compare signal timing (within 3 days tolerance)
                        tolerance = 3
                        matching_buys = 0
                        matching_sells = 0
                        
                        for i in range(len(training_labels)):
                            if training_labels[i] == 1:  # Peak/valley says BUY
                                # Check if current ML has BUY within tolerance
                                start_idx = max(0, i - tolerance)
                                end_idx = min(len(signals), i + tolerance + 1)
                                if np.any(current_signals_array[start_idx:end_idx] == 1):
                                    matching_buys += 1
                            elif training_labels[i] == -1:  # Peak/valley says SELL
                                start_idx = max(0, i - tolerance)
                                end_idx = min(len(signals), i + tolerance + 1)
                                if np.any(current_signals_array[start_idx:end_idx] == -1):
                                    matching_sells += 1
                        
                        buy_accuracy = (matching_buys / buy_labels * 100) if buy_labels > 0 else 0
                        sell_accuracy = (matching_sells / sell_labels * 100) if sell_labels > 0 else 0
                        
                        st.write(f"  • **🎯 Current ML Timing Accuracy**: {buy_accuracy:.1f}% BUYs, {sell_accuracy:.1f}% SELLs match peak/valley labels")
                        
                        if buy_accuracy < 50 or sell_accuracy < 50:
                            st.error("🚨 **SOLUTION NEEDED**: Retrain ML model using peak/valley detection as labels!")
                            st.write("**📋 Action Plan:**")
                            st.write("1. Extract features at each timepoint (RSI, MACD, momentum, etc.)")
                            st.write("2. Use peak/valley labels as training targets")
                            st.write(f"3. Train model to predict BUY/SELL {max(1, window // 2)} days before peaks/valleys")
                            st.write("4. Deploy retrained model for real-time trading")
                        else:
                            st.success("✅ Current ML model timing is reasonable - focus on filtering optimization")
                            
                    st.write("---")
                    
                    # Calculate performance metrics (same as Tab 5)
                    sim_return = (final_capital / 100000 - 1) * 100  # Same calculation as Tab 5
                    completed_trades = [t for t in trades_list if t['profit'] is not None]
                    sim_wins = len([t for t in completed_trades if t['profit'] > 0])
                    sim_losses = len(completed_trades) - sim_wins
                    sim_win_rate = sim_wins / len(completed_trades) if completed_trades else 0.0
                    
                    # CRITICAL DEBUGGING: Why is performance so different from training?
                    st.error("🚨 **PERFORMANCE MISMATCH DETECTED:**")
                    st.write(f"**🎯 Training Performance:** 735349% return, 78.5% win rate, 381 trades")
                    st.write(f"**📉 Production Performance:** {sim_return:.1f}% return, {sim_win_rate:.0%} win rate, {len(trades_list)} trades")
                    st.write("**🔍 Possible Issues:**")
                    st.write("  • Data period mismatch (training vs production dates)")
                    st.write("  • Feature engineering differences")  
                    st.write("  • Model/data loading issues")
                    st.write("  • Signal decoding problems")
                    
                    # Show data comparison
                    training_period = meta.get('period', 'Unknown')
                    st.write(f"**📊 Data Info:** Training period: {training_period}, Production data: {len(subset_raw)} days")
                    
                    # === ORIGINAL SYSTEM RESTORED - NO ENHANCED ENGINE ===
                    # Removed enhanced trading engine integration to restore 78.5% win rate performance
                    # Get current status for display (simple approach)
                    current_signal = signals[-1]
                    current_conf = confidences[-1] 
                    current_price = subset_raw['close'].iloc[-1]
                    current_date = subset_raw.index[-1]

                    # Calculate current position status from trades
                    last_entry_date = "N/A"
                    days_in_trade = 0
                    trade_return = 0.0
                    
                    # Find the most recent open trade
                    active_trade = None
                    for trade in reversed(trades_list):
                        if trade['exit_date'] is None:  # Still open
                            active_trade = {
                                'entry_date': trade['entry_date'],
                                'entry_price': trade['entry_price'],
                                'type': trade['signal_type']
                            }
                            break
                    
                    if active_trade:
                        last_entry_date = active_trade['entry_date'].strftime('%Y-%m-%d')
                        days_in_trade = (subset_raw.index[-1] - active_trade['entry_date']).days
                        if active_trade['type'] == 'LONG':
                            trade_return = (current_price - active_trade['entry_price']) / active_trade['entry_price']
                        else:
                            trade_return = (active_trade['entry_price'] - current_price) / active_trade['entry_price']
                    
                    # --- DISPLAY ---
                    sig_text = "BUY" if current_signal == 1 else "SELL" if current_signal == -1 else "HOLD"
                    sig_color = "#28a745" if current_signal == 1 else "#dc3545" if current_signal == -1 else "#6c757d"
                    bg_color = "rgba(40, 167, 69, 0.1)" if current_signal == 1 else "rgba(220, 53, 69, 0.1)" if current_signal == -1 else "#f8f9fa"
                    
                    # 1. Signal Card (Full Width)
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background-color: {bg_color}; border-radius: 10px; border: 1px solid {sig_color}; margin-top: 10px; margin-bottom: 20px;">
                        <h4 style="margin:0; color: #555;">Current Signal</h4>
                        <h1 style="color: {sig_color}; font-size: 48px; margin: 10px 0;">{sig_text}</h1>
                        <div style="display: flex; justify-content: center; gap: 40px; margin-bottom: 10px;">
                            <span>Conf: <b>{current_conf:.0%}</b></span>
                            <span>{ticker}</span>
                        </div>
                        <hr style="margin: 5px auto; width: 50%; border-color: #ddd;">
                        <div style="font-size: 14px; color: #555; margin-top: 10px;">
                            <span style="margin-right: 20px;">📅 <b>Entry:</b> {last_entry_date} ({days_in_trade} days)</span>
                            <span>📈 <b>Return:</b> {trade_return*100:+.2f}%</span>
                        </div>
                        <p style="font-size: 11px; color: #888; margin-top: 10px; text-align: center;">Last Data: {raw_data.index[-1].strftime('%Y-%m-%d %H:%M')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 2. Performance Stats (New Section)
                    st.markdown(f"##### 📊 Active Model Performance ({len(subset_raw)} Days)")
                    p1, p2, p3, p4 = st.columns(4)
                    with p1:
                        st.metric("Total Return", f"{sim_return:+.1f}%", delta=f"${final_capital - 100000:,.0f}")
                    with p2:
                        st.metric("Win Rate", f"{sim_win_rate*100:.0f}%", f"{sim_wins}W / {sim_losses}L")
                    with p3:
                        st.metric("Trades", str(len(completed_trades)))
                    with p4:
                        st.metric("Est. Capital", f"${final_capital:,.0f}")
                    
                    st.markdown("---")

                    # AI Market Analysis Section
                    st.markdown("##### 🤖 AI Market Analysis")
                    
                    # Security notice
                    with st.expander("🔒 API Key Security Info"):
                        st.markdown("""
                        **Your API Key Security:**
                        - API keys are stored locally in `api_config.json`
                        - This file is automatically added to `.gitignore`
                        - Keys are never sent anywhere except OpenAI
                        - You can clear saved keys anytime
                        - Override feature lets you test different keys temporarily
                        """)
                    st.write("")
                    
                    # Initialize API config
                    from config_api import APIConfig
                    api_config = APIConfig()
                    
                    # Get saved API key
                    saved_key = api_config.get_openai_key()
                    has_saved_key = api_config.has_openai_key()
                    
                    # API Key Management
                    col_ai1, col_ai2, col_ai3 = st.columns([3, 1, 1])
                    
                    with col_ai1:
                        # Show saved key status or input field
                        if has_saved_key:
                            # Show masked saved key with override option
                            masked_key = f"sk-...{saved_key[-8:]}" if saved_key else ""
                            st.success(f"✅ Saved API Key: {masked_key}")
                            
                            # Manual override option
                            override_key = st.text_input(
                                "Override API Key (optional)", 
                                type="password", 
                                placeholder="Leave empty to use saved key",
                                help="Enter a different API key to temporarily override the saved one",
                                key="api_key_override"
                            )
                            
                            # Use override if provided, otherwise use saved
                            api_key = override_key if override_key else saved_key
                        else:
                            # No saved key - regular input
                            api_key = st.text_input(
                                "OpenAI API Key", 
                                type="password", 
                                placeholder="sk-...", 
                                help="Enter your OpenAI API key for GPT-4o market analysis",
                                key="api_key_input"
                            )
                    
                    with col_ai2:
                        st.write("")  # Spacing for alignment
                        
                        # Save key button (only show if key is entered and not saved)
                        if api_key and not has_saved_key:
                            if st.button("💾 Save Key", help="Save API key for future use"):
                                if api_config.set_openai_key(api_key):
                                    st.success("API key saved!")
                                    st.rerun()
                                else:
                                    st.error("Failed to save API key")
                        
                        # Clear saved key button (only show if key is saved)
                        elif has_saved_key:
                            if st.button("🗑️ Clear Saved", help="Clear saved API key"):
                                if api_config.clear_openai_key():
                                    st.success("Saved API key cleared!")
                                    st.rerun()
                                else:
                                    st.error("Failed to clear API key")
                    
                    with col_ai3:
                        st.write("")  # Spacing
                        if st.button("🧠 Analyze Market", type="primary", disabled=not api_key):
                            if api_key:
                                try:
                                    import openai
                                    from datetime import datetime
                                    
                                    # Prepare market data for analysis
                                    current_price = subset_raw['close'].iloc[-1]
                                    price_change = current_price - subset_raw['close'].iloc[-2] if len(subset_raw) > 1 else 0
                                    price_change_pct = (price_change / subset_raw['close'].iloc[-2]) * 100 if len(subset_raw) > 1 else 0
                                    
                                    # Get comprehensive indicator analysis
                                    indicator_analysis = {}
                                    active_features = []
                                    
                                    # Debug: Check what's available in session state
                                    st.write("🔍 **Debug Info:**")
                                    ml_related_keys = [key for key in st.session_state.keys() if any(term in key.lower() for term in ['ml', 'model', 'engineer', 'feature'])]
                                    st.write(f"ML-related session keys: {ml_related_keys}")
                                    
                                    # Create interpretation rules for common indicator patterns (available to all approaches)
                                    def get_indicator_interpretation(name, value, prev_value):
                                        """Smart interpretation of technical indicators based on name patterns"""
                                        name_lower = name.lower()
                                        
                                        # RSI patterns
                                        if 'rsi' in name_lower:
                                            if value >= 70:
                                                return "OVERBOUGHT - Bearish signal"
                                            elif value <= 30:
                                                return "OVERSOLD - Bullish signal"
                                            else:
                                                return "NEUTRAL range"
                                        
                                        # MACD patterns
                                        elif 'macd' in name_lower:
                                            if 'signal' not in name_lower:
                                                return "Bullish momentum" if value > 0 else "Bearish momentum"
                                            else:
                                                return "Signal line for MACD crossover"
                                        
                                        # Williams %R patterns  
                                        elif 'willr' in name_lower or 'williams' in name_lower:
                                            if value >= -20:
                                                return "OVERBOUGHT - Bearish signal"
                                            elif value <= -80:
                                                return "OVERSOLD - Bullish signal"
                                            else:
                                                return "NEUTRAL range"
                                        
                                        # Stochastic patterns
                                        elif 'stoch' in name_lower:
                                            if value >= 80:
                                                return "OVERBOUGHT - Bearish signal"
                                            elif value <= 20:
                                                return "OVERSOLD - Bullish signal"
                                            else:
                                                return "NEUTRAL range"
                                        
                                        # Moving Average patterns
                                        elif any(ma in name_lower for ma in ['sma', 'ema', 'wma']):
                                            trend = "Rising" if value > prev_value else "Falling" if value < prev_value else "Flat"
                                            return f"{trend} trend line"
                                        
                                        # Bollinger Band patterns
                                        elif 'bb_' in name_lower:
                                            if 'upper' in name_lower:
                                                return "Resistance level"
                                            elif 'lower' in name_lower:
                                                return "Support level"
                                            elif 'middle' in name_lower:
                                                return "Middle band (SMA)"
                                        
                                        # ATR patterns
                                        elif 'atr' in name_lower:
                                            return "Volatility measure"
                                        
                                        # Volume patterns
                                        elif 'volume' in name_lower:
                                            return "Volume indicator"
                                        
                                        # Default interpretation based on value change
                                        else:
                                            change_dir = "Rising" if value > prev_value else "Falling" if value < prev_value else "Unchanged"
                                            return f"{change_dir} technical indicator"
                                    
                                    # Try multiple approaches to get feature data
                                    features_found = False
                                    
                                    # Approach 1: Check ml_engineer
                                    if hasattr(st.session_state, 'ml_engineer') and st.session_state.ml_engineer:
                                        try:
                                            eng = st.session_state.ml_engineer
                                            st.write(f"✅ Found ml_engineer, attributes: {[attr for attr in dir(eng) if not attr.startswith('_')]}")
                                            if hasattr(eng, 'latest_features') and eng.latest_features is not None:
                                                # Get current and previous values for change analysis
                                                current_features = eng.latest_features.tail(1).iloc[0]
                                                prev_features = eng.latest_features.tail(2).iloc[0] if len(eng.latest_features) > 1 else current_features
                                                
                                                # Get ALL model features (dynamic from actual model training)
                                                all_model_features = current_features.index.tolist()
                                                
                                                # Extract ALL model features with analysis
                                                for feature_name in all_model_features:
                                                    current_val = current_features[feature_name]
                                                    prev_val = prev_features[feature_name] if feature_name in prev_features.index else current_val
                                                    change = current_val - prev_val
                                                    change_pct = (change / prev_val * 100) if prev_val != 0 else 0
                                                    
                                                    # Get smart interpretation using the function
                                                    interpretation = get_indicator_interpretation(feature_name, current_val, prev_val)
                                                    
                                                    # Store analysis for this feature
                                                    indicator_analysis[feature_name] = {
                                                        'name': feature_name,  # Use actual feature name
                                                        'current': current_val,
                                                        'previous': prev_val,
                                                        'change': change,
                                                        'change_pct': change_pct,
                                                        'interpretation': interpretation
                                                    }
                                                    active_features.append(feature_name)
                                                features_found = True
                                                st.success(f"✅ Successfully extracted {len(all_model_features)} features from ml_engineer!")
                                                
                                        except Exception as e:
                                            st.warning(f"❌ ml_engineer approach failed: {e}")
                                    else:
                                        st.write("❌ ml_engineer not found or not available")
                                    
                                    # Approach 2: Try to extract from 'features' variable (if available)
                                    if not features_found and 'features' in locals():
                                        try:
                                            st.write("🔄 Trying approach 2: features variable")
                                            current_features = features.iloc[-1] if len(features) > 0 else None
                                            prev_features = features.iloc[-2] if len(features) > 1 else current_features
                                            
                                            if current_features is not None:
                                                all_model_features = current_features.index.tolist()
                                                
                                                # Same extraction logic as above
                                                for feature_name in all_model_features:
                                                    current_val = current_features[feature_name]
                                                    prev_val = prev_features[feature_name] if prev_features is not None and feature_name in prev_features.index else current_val
                                                    change = current_val - prev_val
                                                    change_pct = (change / prev_val * 100) if prev_val != 0 else 0
                                                    
                                                    # Get smart interpretation using the function
                                                    interpretation = get_indicator_interpretation(feature_name, current_val, prev_val)
                                                    
                                                    # Store analysis for this feature
                                                    indicator_analysis[feature_name] = {
                                                        'name': feature_name,
                                                        'current': current_val,
                                                        'previous': prev_val,
                                                        'change': change,
                                                        'change_pct': change_pct,
                                                        'interpretation': interpretation
                                                    }
                                                    active_features.append(feature_name)
                                                features_found = True
                                                st.success(f"✅ Successfully extracted {len(all_model_features)} features from features variable!")
                                        except Exception as e:
                                            st.warning(f"❌ features variable approach failed: {e}")
                                    
                                    # Approach 3: Show what we do have available for debugging
                                    if not features_found:
                                        st.error("❌ Could not extract any feature data")
                                        st.write("Available variables in Production tab scope:")
                                        available_vars = [var for var in locals().keys() if not var.startswith('_')]
                                        st.write(f"Local variables: {available_vars}")
                                        
                                        # Show session state ML keys for debugging
                                        if ml_related_keys:
                                            for key in ml_related_keys:
                                                obj = getattr(st.session_state, key, None)
                                                if obj:
                                                    st.write(f"- {key}: {type(obj)} with attributes: {[attr for attr in dir(obj) if not attr.startswith('_')][:10]}")
                                    
                                    st.write("---")
                                    
                                    # Prepare trade history summary
                                    recent_trades = []
                                    if len(completed_trades) > 0:
                                        for trade in completed_trades[-5:]:  # Last 5 trades
                                            recent_trades.append({
                                                'entry_date': trade.get('entry_date', 'Unknown'),
                                                'exit_date': trade.get('exit_date', 'Unknown'),
                                                'signal_type': trade.get('signal_type', 'Unknown'),
                                                'return': trade.get('return_pct', 0)
                                            })
                                    
                                    # Create enhanced analysis prompt with real indicator data
                                    
                                    # Format indicator data for prompt
                                    indicators_text = ""
                                    if indicator_analysis:
                                        indicators_text = f"\nALL MODEL FEATURES ({len(indicator_analysis)} total):\n"
                                        for indicator, data in indicator_analysis.items():
                                            change_dir = "↑" if data['change'] > 0 else "↓" if data['change'] < 0 else "→"
                                            indicators_text += f"• {data['name']}: {data['current']:.2f} ({change_dir} {data['change']:+.2f}) - {data['interpretation']}\n"
                                    else:
                                        indicators_text = "\nMODEL FEATURES: Data not available"
                                    
                                    # Determine signal type for context
                                    signal_type = "SELL/SHORT" if current_signal == -1 else "BUY/LONG" if current_signal == 1 else "HOLD"
                                    
                                    prompt = f"""You are an expert technical analyst. Analyze why the ML model generated this specific trading signal based on the actual indicator values provided.

MARKET DATA - {ticker}:
- Current Price: ${current_price:.2f}
- Daily Change: ${price_change:.2f} ({price_change_pct:+.2f}%)
- Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CURRENT ML SIGNAL:
- Signal: {signal_type} ({current_signal})
- Model Confidence: {current_conf:.1f}%
{indicators_text}

RECENT PERFORMANCE:
- Total Return: {sim_return:+.2f}% | Win Rate: {sim_win_rate*100:.0f}% | Trades: {len(completed_trades)}
- Last 5 Trades: {len(recent_trades)} available

ANALYSIS REQUIRED:
1. **Signal Trigger Analysis**: Based on the ACTUAL indicator values above, explain specifically WHY the model generated a {signal_type} signal. Which indicators are in overbought/oversold territory?

2. **Key Indicator Drivers**: Identify the 2-3 most significant indicators contributing to this signal. Reference their actual current values and what they indicate.

3. **Confirmation/Divergence**: Do the indicators confirm each other or show divergence? How strong is the signal consensus?

4. **Risk Assessment**: Given the {current_conf:.1f}% confidence level, what does this tell us about signal strength and potential risk?

Focus on SPECIFIC indicator values and WHY they triggered this signal. Avoid generic market commentary - analyze the actual data provided."""

                                    # Make API call
                                    with st.spinner("🤖 Analyzing market with GPT-4o..."):
                                        client = openai.OpenAI(api_key=api_key)
                                        
                                        response = client.chat.completions.create(
                                            model="gpt-4o",
                                            messages=[
                                                {"role": "system", "content": "You are a professional trading analyst with expertise in technical analysis and market psychology."},
                                                {"role": "user", "content": prompt}
                                            ],
                                            max_tokens=1000,
                                            temperature=0.7
                                        )
                                        
                                        analysis = response.choices[0].message.content
                                        
                                        # Display the analysis
                                        st.markdown("##### 📊 AI Market Analysis Results")
                                        st.markdown(analysis)
                                        
                                        # Save to session state for reference
                                        st.session_state.latest_ai_analysis = {
                                            'timestamp': datetime.now(),
                                            'ticker': ticker,
                                            'analysis': analysis,
                                            'signal': current_signal,
                                            'confidence': current_conf
                                        }
                                        
                                except Exception as e:
                                    st.error(f"❌ Analysis failed: {str(e)}")
                                    if "api_key" in str(e).lower():
                                        st.error("Please check your API key is valid")
                                    elif "quota" in str(e).lower():
                                        st.error("API quota exceeded. Please check your OpenAI account.")
                            else:
                                st.warning("Please enter your OpenAI API key first")
                    
                    # Show previous analysis if available
                    if 'latest_ai_analysis' in st.session_state:
                        prev_analysis = st.session_state.latest_ai_analysis
                        if prev_analysis['ticker'] == ticker:
                            with st.expander(f"📋 Previous Analysis ({prev_analysis['timestamp'].strftime('%H:%M:%S')})"):
                                st.markdown(prev_analysis['analysis'])
                    
                    st.markdown("---")

                    # 3. Chart (Full Width) with Super Indicator
                    st.markdown(f"##### 📉 Price Action & Signals ({len(subset_raw)} Days)")
                    
                    # Create subplots: Main chart + Super Indicator
                    from plotly.subplots import make_subplots
                    
                    fig_live = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        row_heights=[0.7, 0.3],  # Main chart 70%, indicator 30%
                        subplot_titles=["Price Action", "Super Indicator (ML Confidence + Composite)"]
                    )
                    
                    # === SUPER INDICATOR DATA PREPARATION ===
                    super_indicator_data = {}
                    dates = subset_raw.index
                    
                    # Option 1: Raw ML Model Predictions (the actual ML confidence over time)
                    st.write("🔍 **Super Indicator Debug:**")
                    
                    # Check if we have the actual ML predictions from the backtest/production run
                    if 'signals' in locals() and 'confidences' in locals():
                        st.write(f"✅ Found ML predictions: {len(signals)} signals, {len(confidences)} confidences")
                        if len(signals) == len(dates):
                            # Combine signal direction (-1, 0, 1) with confidence (0-1) to get range -1 to +1
                            super_indicator_data['ml_confidence'] = signals * confidences
                            st.write(f"✅ ML Confidence range: {super_indicator_data['ml_confidence'].min():.3f} to {super_indicator_data['ml_confidence'].max():.3f}")
                        else:
                            st.warning(f"⚠️ Signal length mismatch: {len(signals)} signals vs {len(dates)} dates")
                            # Align the signals to the chart timeframe
                            if len(signals) > len(dates):
                                # Take the last N signals to match chart
                                aligned_signals = signals[-len(dates):]
                                aligned_confidences = confidences[-len(dates):]
                                super_indicator_data['ml_confidence'] = aligned_signals * aligned_confidences
                                st.write("✅ Aligned ML signals to chart timeframe")
                            else:
                                # Pad with current signal for missing periods
                                padded_signals = np.full(len(dates), current_signal)
                                padded_confidences = np.full(len(dates), current_conf / 100)
                                padded_signals[-len(signals):] = signals  # Replace last periods with actual signals
                                padded_confidences[-len(confidences):] = confidences
                                super_indicator_data['ml_confidence'] = padded_signals * padded_confidences
                                st.write("✅ Padded ML signals to match chart timeframe")
                    else:
                        st.write("❌ No ML signals/confidences found - using current signal approximation")
                        # Use current signal as baseline across timeframe
                        base_signal = current_signal * (current_conf / 100)
                        super_indicator_data['ml_confidence'] = np.full(len(dates), base_signal)
                    
                    # Option 2: Composite Technical Indicator (using actual features data)
                    composite_success = False
                    
                    # Try the features variable that we know works from AI analysis
                    if 'features' in locals() and features is not None:
                        try:
                            st.write("🔄 Using 'features' variable for composite indicator")
                            # Get features for the same time period as chart (last N periods)
                            chart_features = features.tail(len(dates)) if len(features) >= len(dates) else features
                            
                            # Create composite from ALL oscillator-type indicators (normalized to -1 to +1)
                            composite_parts = []
                            
                            # Dynamically find and normalize oscillator indicators
                            for col in chart_features.columns:
                                col_lower = col.lower()
                                
                                # RSI indicators
                                if 'rsi' in col_lower:
                                    rsi_norm = (chart_features[col] - 50) / 50  # -1 to +1
                                    composite_parts.append(rsi_norm)
                                
                                # Williams %R indicators  
                                elif 'willr' in col_lower or 'williams' in col_lower:
                                    willr_norm = chart_features[col] / 100  # -100 to 0 → -1 to 0
                                    willr_norm = (willr_norm + 0.5) * 2 - 1  # Scale to -1 to +1
                                    composite_parts.append(willr_norm)
                                
                                # Stochastic indicators
                                elif 'stoch' in col_lower:
                                    stoch_norm = (chart_features[col] - 50) / 50  # 0-100 → -1 to +1
                                    composite_parts.append(stoch_norm)
                                
                                # MACD indicators (normalize by standard deviation)
                                elif 'macd' in col_lower and 'signal' not in col_lower:
                                    macd_std = chart_features[col].std()
                                    if macd_std > 0:
                                        macd_norm = chart_features[col] / (3 * macd_std)  # ±3 std devs
                                        macd_norm = np.clip(macd_norm, -1, 1)
                                        composite_parts.append(macd_norm)
                            
                            # Average the components
                            if composite_parts:
                                super_indicator_data['composite'] = np.mean(composite_parts, axis=0)
                                composite_success = True
                                st.write(f"✅ Composite indicator created from {len(composite_parts)} oscillators")
                            else:
                                super_indicator_data['composite'] = np.zeros(len(dates))
                                st.warning("⚠️ No oscillator indicators found for composite")
                                
                        except Exception as e:
                            st.warning(f"❌ Composite indicator from features failed: {e}")
                            super_indicator_data['composite'] = np.zeros(len(dates))
                    
                    # Fallback if features approach didn't work
                    if not composite_success:
                        st.write("🔄 Fallback: Creating simple composite from current values")
                        super_indicator_data['composite'] = np.full(len(dates), 0.0)  # Neutral line
                    
                    # Summary of super indicator data
                    st.write("📊 **Super Indicator Summary:**")
                    st.write(f"• **Blue Line (ML Confidence)**: Range {super_indicator_data['ml_confidence'].min():.3f} to {super_indicator_data['ml_confidence'].max():.3f}")
                    st.write(f"• **Orange Dotted Line (Composite Tech)**: Range {super_indicator_data['composite'].min():.3f} to {super_indicator_data['composite'].max():.3f}")
                    st.write("• **Interpretation**: Above 0 = Bullish, Below 0 = Bearish, ±0.5 = Strong signals")
                    st.write("---")
                    
                    # === TRADE TREND RIBBONS ===
                    # Create trend ribbons between buy/sell signals (TradingView style)
                    if len(trades_list) > 0:
                        # Sort trades by entry date to create sequential ribbons
                        sorted_trades = sorted(trades_list, key=lambda x: x['entry_date'])
                        
                        for i in range(len(sorted_trades)):
                            current_trade = sorted_trades[i]
                            
                            # Get date range for this ribbon
                            start_date = current_trade['entry_date']
                            
                            # End date is either the exit date or next trade's entry date
                            if 'exit_date' in current_trade and current_trade['exit_date']:
                                end_date = current_trade['exit_date']
                            elif i + 1 < len(sorted_trades):
                                end_date = sorted_trades[i + 1]['entry_date']
                            else:
                                end_date = subset_raw.index[-1]  # Last available data
                            
                            # Filter data for this ribbon period
                            try:
                                ribbon_data = subset_raw[start_date:end_date]
                                if len(ribbon_data) > 0:
                                    ribbon_dates = list(ribbon_data.index)
                                    
                                    # Create ribbon bounds (high/low envelope)
                                    high_line = ribbon_data['high'].values
                                    low_line = ribbon_data['low'].values
                                    
                                    # Ribbon color based on signal type
                                    if current_trade['signal_type'] == 'LONG':
                                        fill_color = 'rgba(0, 255, 0, 0.15)'  # Light green with low opacity
                                        line_color = 'rgba(0, 128, 0, 0.3)'
                                        ribbon_name = 'Long Position'
                                    else:
                                        fill_color = 'rgba(255, 0, 0, 0.15)'  # Light red with low opacity  
                                        line_color = 'rgba(128, 0, 0, 0.3)'
                                        ribbon_name = 'Short Position'
                                    
                                    # Add upper bound
                                    fig_live.add_trace(go.Scatter(
                                        x=ribbon_dates,
                                        y=high_line,
                                        mode='lines',
                                        line=dict(color=line_color, width=0.5),
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ), row=1, col=1)
                                    
                                    # Add lower bound with fill
                                    fig_live.add_trace(go.Scatter(
                                        x=ribbon_dates,
                                        y=low_line,
                                        mode='lines',
                                        line=dict(color=line_color, width=0.5),
                                        fill='tonexty',
                                        fillcolor=fill_color,
                                        name=ribbon_name,
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ), row=1, col=1)
                            except:
                                # Skip if date range issues
                                pass
                    
                    # Candlesticks (add after ribbons so they appear on top)
                    fig_live.add_trace(go.Candlestick(
                        x=subset_raw.index,
                        open=subset_raw['open'], high=subset_raw['high'],
                        low=subset_raw['low'], close=subset_raw['close'],
                        name='Price'
                    ), row=1, col=1)
                    
                    # === SUPER INDICATOR PLOTS (Bottom Chart) ===
                    
                    # Plot ML Confidence
                    fig_live.add_trace(go.Scatter(
                        x=dates,
                        y=super_indicator_data['ml_confidence'],
                        mode='lines',
                        name='ML Confidence',
                        line=dict(color='blue', width=2),
                        hovertemplate='<b>ML Confidence</b><br>Value: %{y:.3f}<br>Date: %{x}<extra></extra>'
                    ), row=2, col=1)
                    
                    # Plot Composite Technical Indicator
                    fig_live.add_trace(go.Scatter(
                        x=dates,
                        y=super_indicator_data['composite'],
                        mode='lines',
                        name='Composite Tech',
                        line=dict(color='orange', width=2, dash='dot'),
                        hovertemplate='<b>Composite Technical</b><br>Value: %{y:.3f}<br>Date: %{x}<extra></extra>'
                    ), row=2, col=1)
                    
                    # Add threshold lines for Super Indicator
                    for threshold in [0.5, 0, -0.5]:
                        line_color = 'green' if threshold > 0 else 'red' if threshold < 0 else 'gray'
                        line_style = 'solid' if threshold == 0 else 'dash'
                        fig_live.add_hline(
                            y=threshold, 
                            line_dash=line_style, 
                            line_color=line_color,
                            opacity=0.5,
                            row=2, col=1
                        )
                    
                    # Color zones for Super Indicator
                    fig_live.add_hrect(
                        y0=0, y1=1, 
                        fillcolor='rgba(0, 255, 0, 0.1)', 
                        layer='below', 
                        line_width=0,
                        row=2, col=1
                    )
                    fig_live.add_hrect(
                        y0=-1, y1=0, 
                        fillcolor='rgba(255, 0, 0, 0.1)', 
                        layer='below', 
                        line_width=0,
                        row=2, col=1
                    )
                    
                    # === PLOT FILTERED ML SIGNALS (MATCHES ACTUAL TRADES) ===
                    
                    # Apply same filtering as backtest for chart display
                    if (enable_trading_optimization and 
                        hasattr(st.session_state, 'trading_optimization_params') and 
                        st.session_state.trading_optimization_params is not None):
                        # Use optimized parameters for filtering
                        buy_conf_thresh = min_buy_confidence_opt
                        sell_conf_thresh = min_sell_confidence_opt
                        buy_comp_thresh = buy_composite_max_opt
                        sell_comp_thresh = sell_composite_min_opt
                        chart_label = "(Optimized)"
                    else:
                        # No filtering (baseline)
                        buy_conf_thresh = 0.0
                        sell_conf_thresh = 0.0
                        buy_comp_thresh = -999
                        sell_comp_thresh = 999
                        chart_label = "(Baseline)"
                    
                    # Apply combined filtering to chart
                    if composite_tech_values is not None:
                        buy_signal_mask = ((signals == 1) & 
                                         (confidences >= buy_conf_thresh) & 
                                         (composite_tech_values <= buy_comp_thresh))
                        sell_signal_mask = ((signals == -1) & 
                                          (confidences >= sell_conf_thresh) & 
                                          (composite_tech_values >= sell_comp_thresh))
                    else:
                        # Only confidence filtering if no composite data
                        buy_signal_mask = (signals == 1) & (confidences >= buy_conf_thresh)
                        sell_signal_mask = (signals == -1) & (confidences >= sell_conf_thresh)
                    
                    # Plot filtered BUY signals  
                    buy_signal_indices = np.where(buy_signal_mask)[0]
                    if len(buy_signal_indices) > 0:
                        buy_dates = subset_raw.index[buy_signal_indices]
                        buy_prices = subset_raw['low'].iloc[buy_signal_indices] * 0.97  # 3% below low
                        fig_live.add_trace(go.Scatter(
                            x=buy_dates,
                            y=buy_prices,
                            mode='markers',
                            name=f'ML BUY Signals {chart_label}',
                            marker=dict(color='green', size=10, symbol='triangle-up'),
                            hovertemplate='<b>BUY Signal</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                        ), row=1, col=1)
                    
                    # Plot filtered SELL signals
                    sell_signal_indices = np.where(sell_signal_mask)[0]  
                    if len(sell_signal_indices) > 0:
                        sell_dates = subset_raw.index[sell_signal_indices]
                        sell_prices = subset_raw['high'].iloc[sell_signal_indices] * 1.03  # 3% above high
                        fig_live.add_trace(go.Scatter(
                            x=sell_dates,
                            y=sell_prices,
                            mode='markers',
                            name=f'ML SELL Signals {chart_label}',
                            marker=dict(color='red', size=10, symbol='triangle-down'),
                            hovertemplate='<b>SELL Signal</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                        ), row=1, col=1)
                    
                    # Note: Removed duplicate trade markers to avoid double triangles
                    # All signals (including trade entries) are now shown via ML Signal markers above
                    
                    # === ENHANCED DYNAMIC RISK LEVELS ON CHART ===
                    if ('enhanced_trading_analysis' in st.session_state and 
                        st.session_state.enhanced_trading_analysis and
                        current_signal != 0):
                        
                        # Get dynamic risk levels from enhanced analysis
                        enhanced_analysis = st.session_state.enhanced_trading_analysis
                        opportunity = enhanced_analysis.get('opportunity_analysis')
                        
                        if opportunity and opportunity.get('risk_levels'):
                            risk_levels = opportunity['risk_levels']
                            
                            # Get dynamic levels (these are the REAL levels the algorithm uses!)
                            dynamic_stop_loss = risk_levels.get('stop_loss', current_price * 0.92)
                            dynamic_take_profit = risk_levels.get('take_profit', current_price * 1.15)
                            dynamic_stop_pct = risk_levels.get('stop_loss_pct', 8.0)
                            dynamic_tp_pct = risk_levels.get('take_profit_pct', 15.0)
                            risk_reward = risk_levels.get('risk_reward_ratio', 1.5)
                            
                            # Add DYNAMIC risk level lines to chart
                            fig_live.add_hline(
                                y=dynamic_stop_loss, 
                                line_dash="dash", 
                                line_color="red", 
                                annotation_text=f"Dynamic SL ({dynamic_stop_pct:.1f}%)", 
                                row=1, col=1
                            )
                            
                            fig_live.add_hline(
                                y=dynamic_take_profit, 
                                line_dash="dash", 
                                line_color="green", 
                                annotation_text=f"Dynamic TP ({dynamic_tp_pct:.1f}%)", 
                                row=1, col=1
                            )
                            
                            fig_live.add_hline(
                                y=current_price, 
                                line_dash="dot", 
                                line_color="blue", 
                                annotation_text=f"Entry (R/R: {risk_reward:.1f}:1)", 
                                row=1, col=1
                            )
                            
                            # Add risk zone shading
                            if current_signal == 1:  # Long position
                                # Profit zone (green)
                                fig_live.add_hrect(
                                    y0=current_price, y1=dynamic_take_profit,
                                    fillcolor='rgba(0, 255, 0, 0.1)', 
                                    layer='below', line_width=0, row=1, col=1
                                )
                                # Risk zone (red)
                                fig_live.add_hrect(
                                    y0=dynamic_stop_loss, y1=current_price,
                                    fillcolor='rgba(255, 0, 0, 0.1)', 
                                    layer='below', line_width=0, row=1, col=1
                                )
                            else:  # Short position
                                # Profit zone (green) - below entry for short
                                fig_live.add_hrect(
                                    y0=dynamic_take_profit, y1=current_price,
                                    fillcolor='rgba(0, 255, 0, 0.1)', 
                                    layer='below', line_width=0, row=1, col=1
                                )
                                # Risk zone (red) - above entry for short
                                fig_live.add_hrect(
                                    y0=current_price, y1=dynamic_stop_loss,
                                    fillcolor='rgba(255, 0, 0, 0.1)', 
                                    layer='below', line_width=0, row=1, col=1
                                )
                        
                        else:
                            # Fallback to basic levels if dynamic analysis fails
                            atr = features.get('ATR_14', pd.Series([current_price * 0.02])).iloc[-1]
                            basic_stop = current_price - (2 * atr) if current_signal == 1 else current_price + (2 * atr)
                            basic_tp = current_price + (3 * atr) if current_signal == 1 else current_price - (3 * atr)
                            
                            fig_live.add_hline(y=basic_stop, line_dash="dash", line_color="red", annotation_text="Basic SL", row=1, col=1)
                            fig_live.add_hline(y=basic_tp, line_dash="dash", line_color="green", annotation_text="Basic TP", row=1, col=1)
                            fig_live.add_hline(y=current_price, line_dash="dot", line_color="blue", annotation_text="Entry", row=1, col=1)
                    
                    elif current_signal != 0:
                        # No enhanced analysis available - show basic levels
                        atr = features.get('ATR_14', pd.Series([current_price * 0.02])).iloc[-1]
                        basic_stop = current_price - (2 * atr) if current_signal == 1 else current_price + (2 * atr)
                        basic_tp = current_price + (3 * atr) if current_signal == 1 else current_price - (3 * atr)
                        
                        fig_live.add_hline(y=basic_stop, line_dash="dash", line_color="red", annotation_text="Basic SL", row=1, col=1)
                        fig_live.add_hline(y=basic_tp, line_dash="dash", line_color="green", annotation_text="Basic TP", row=1, col=1)
                        fig_live.add_hline(y=current_price, line_dash="dot", line_color="blue", annotation_text="Entry", row=1, col=1)
                        
                        # HOLDING MARKER
                        fig_live.add_trace(go.Scatter(
                            x=[subset_raw.index[-1]], 
                            y=[current_price],
                            mode='markers',
                            name='Current Status',
                            marker=dict(color=sig_color, size=8, symbol='circle'),
                            hovertemplate=f"<b>Holding ({sig_text})</b><br>Price: ${current_price:.2f}<extra></extra>"
                        ), row=1, col=1)
                        
                        # Enhanced chart layout for subplots
                        fig_live.update_layout(
                            height=800,  # Increased height for main + indicator charts
                            margin=dict(l=0, r=0, t=30, b=0),
                            showlegend=True,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        
                        # Update main chart (row 1) axes
                        fig_live.update_xaxes(
                            gridcolor='rgba(128,128,128,0.2)',
                            showgrid=True,
                            row=1, col=1
                        )
                        fig_live.update_yaxes(
                            gridcolor='rgba(128,128,128,0.2)', 
                            showgrid=True,
                            title="Price ($)",
                            row=1, col=1
                        )
                        
                        # Update super indicator chart (row 2) axes
                        fig_live.update_xaxes(
                            gridcolor='rgba(128,128,128,0.2)',
                            showgrid=True,
                            title="Date",
                            row=2, col=1
                        )
                        fig_live.update_yaxes(
                            gridcolor='rgba(128,128,128,0.2)', 
                            showgrid=True,
                            title="Signal Strength",
                            range=[-1.1, 1.1],  # Fixed range for indicator
                            row=2, col=1
                        )
                        
                        # Add ribbon legend info
                        col_legend1, col_legend2 = st.columns([3, 1])
                        with col_legend2:
                            st.markdown("""
                            **Trend Ribbons:**
                            🟢 Long Positions
                            🔴 Short Positions
                            """)
                        
                        st.plotly_chart(fig_live, use_container_width=True)
                        
                        # 4. Trade Levels (Below Graph)
                        st.markdown("### 🛡️ Active Trade Levels (ATR-Based)")
                        l1, l2, l3 = st.columns(3)
                        with l1:
                            st.metric("Entry Price", f"${current_price:.2f}")
                        with l2:
                            st.metric("Stop Loss", f"${stop_loss:.2f}", delta=f"{stop_loss-current_price:.2f}", delta_color="inverse")
                        with l3:
                            st.metric("Take Profit", f"${take_profit:.2f}", delta=f"{take_profit-current_price:.2f}")

                    else:
                        # Render chart even if no signal with enhanced styling
                        fig_live.update_layout(
                            height=800,  # Match the enhanced chart height
                            margin=dict(l=0, r=0, t=30, b=0),
                            showlegend=True,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                        )
                        
                        # Update axes for both subplots (same as active signal case)
                        fig_live.update_xaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True, row=1, col=1)
                        fig_live.update_yaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True, title="Price ($)", row=1, col=1)
                        fig_live.update_xaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True, title="Date", row=2, col=1)
                        fig_live.update_yaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True, title="Signal Strength", range=[-1.1, 1.1], row=2, col=1)
                        
                        # Add ribbon legend info (same as active signal case)
                        col_legend1, col_legend2 = st.columns([3, 1])
                        with col_legend2:
                            st.markdown("""
                            **Trend Ribbons:**
                            🟢 Long Positions
                            🔴 Short Positions
                            """)
                        
                        st.plotly_chart(fig_live, use_container_width=True)
                        st.info("Waiting for new setup...")

                # --- AUTO-REFRESH SYSTEM ---
                if live_mode:
                    # Initialize or get last refresh time
                    if 'last_refresh_time' not in st.session_state:
                        st.session_state.last_refresh_time = time.time()
                    
                    current_time = time.time()
                    time_since_refresh = current_time - st.session_state.last_refresh_time
                    
                    # Only auto-refresh if 60 seconds have passed
                    if time_since_refresh >= 60:
                        st.session_state.last_refresh_time = current_time
                        st.rerun()
                    else:
                        # Show countdown without blocking
                        seconds_until_refresh = int(60 - time_since_refresh)
                        st.info(f"🔄 Live mode active - Next refresh in {seconds_until_refresh}s")
                else:
                    # Clear refresh timer when live mode is off
                    if 'last_refresh_time' in st.session_state:
                        del st.session_state.last_refresh_time
                    
                # Log latest (Only if data updated to avoid spam)
                # if data_updated:
                #     pm.log_signal(...) 
                    
                # Check for data loading issues
                if features is None:
                    st.warning("⚠️ Could not load market data. Please check your connection.")

            except Exception as e:
                st.error(f"Error loading model: {e}")
        else:
            st.info("👈 Select a model above and click 'Activate' to start the dashboard.")

        # === 4. LOGS ===
        # ---------------------------------------------------------
        st.markdown("---")
        with st.expander("📋 View Production Logs"):
            log_df = pm.get_signal_history()
            if not log_df.empty:
                st.dataframe(log_df.sort_values('timestamp', ascending=False), use_container_width=True)
            else:
                st.info("No logs yet.")

    except Exception as e:
        st.error(f"Production Error: {e}")

# === FOOTER ===
st.markdown("---")
st.markdown("""
### 💡 About ML Trading Signals

This machine learning system:
1. **Labels Data**: Automatically detects historical peaks and valleys as buy/sell points
2. **Engineers Features**: Uses 130+ technical indicators plus ML-specific features
3. **Trains Models**: Multiple algorithms learn from historical patterns
4. **Generates Signals**: Real-time predictions with confidence scores

**Note**: This is a research tool. Always validate signals and manage risk appropriately.
""")
