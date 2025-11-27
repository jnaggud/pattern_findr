#!/usr/bin/env python3
"""
ML Trading Signals Page for Streamlit App

Standalone page for machine learning-based trading signal generation.
Completely separate from existing optimization functionality.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our new ML modules
from peak_valley_detector import PeakValleyDetector
from ml_feature_engineer import MLFeatureEngineer
from ml_models import TradingMLModels

# === PAGE HEADER ===
st.markdown("""
# 🤖 Machine Learning Trading Signals

Generate trading signals using machine learning models trained on historical price patterns and technical indicators.

**Features:**
- 🎯 Automatic peak/valley detection for labeling
- 🧠 Multiple ML algorithms (Random Forest, XGBoost, SVM)  
- 📊 130+ technical indicators as features
- 📈 Real-time signal generation
- 🏆 Model performance comparison
""")

# === SIDEBAR CONTROLS ===
st.sidebar.header("🎯 ML Configuration")

# Ticker selection
ticker = st.sidebar.text_input("Stock Ticker", "SPY").upper()

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
                    # Load price data
                    data = yf.Ticker(ticker).history(period=period, interval="1d")
                    data.columns = [col.lower() for col in data.columns]
                    
                    if len(data) == 0:
                        st.error("❌ No data found for this ticker")
                        st.stop()
                    
                    # Detect peaks and valleys
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
        - Period: {period}  
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
                            
                            # Train selected models
                            results = {}
                            for model_name in models_to_train:
                                st.write(f"Training {model_name}...")
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
                from ml_models import TradingMLModels
                temp_ml = TradingMLModels()
                model_data = temp_ml.load_model_version(config['active_model_path'])
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
                
                st.caption(f"Saved on: {meta.get('saved_at', 'Unknown Date')} | Period: {meta.get('period', 'Unknown')}")
                st.markdown("---")
                
                # === 3. LIVE MARKET MONITOR (AUTO-LOAD) ===
                # ---------------------------------------------------------
                st.subheader("📡 Live Market Monitor")
                
                import time
                
                # Controls
                col_ctrl1, col_ctrl2 = st.columns([2, 1])
                with col_ctrl1:
                    live_mode = st.toggle("🔴 Live Trading Mode (Auto-Update)", value=st.session_state.get('ml_live_mode', False), key='toggle_live_mode')
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
                        ticker = meta.get('ticker', 'SPY')
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
                    ticker = meta.get('ticker', 'SPY')
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
                    
                    # DECODE SIGNALS (Critical Fix)
                    # Model outputs 0,1,2 -> We need -1,0,1
                    if prod_label_encoder:
                        signals = prod_label_encoder.inverse_transform(raw_signals)
                    else:
                        # Fallback (Assume standard mapping if encoder missing, though risky)
                        # If model outputs 0,1,2 and we don't have encoder, we might be in trouble.
                        # But typically 0=Sell, 1=Hold, 2=Buy. Map manually if needed.
                        # For now, pass raw, but warn in debug.
                        signals = raw_signals

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
                        
                    # --- UNIFIED BACKTEST (Same as Tab 5) ---
                    # Use the exact same run_ml_backtest function to ensure consistency
                    def run_production_backtest(data, signals, starting_capital=100000):
                        """Same backtest logic as Tab 5 - ensures consistency"""
                        capital = starting_capital
                        position = None
                        trades = []
                        equity_curve = [starting_capital]
                        
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
                        
                        return trades, equity_curve, capital
                    
                    # Run the unified backtest
                    trades_list, equity_curve, final_capital = run_production_backtest(subset_raw, signals)
                    
                    # Calculate performance metrics (same as Tab 5)
                    sim_return = (final_capital / 100000 - 1) * 100  # Same calculation as Tab 5
                    completed_trades = [t for t in trades_list if t['profit'] is not None]
                    sim_wins = len([t for t in completed_trades if t['profit'] > 0])
                    sim_losses = len(completed_trades) - sim_wins
                    sim_win_rate = sim_wins / len(completed_trades) if completed_trades else 0.0
                    
                    # Current Status (Last point) - needed for display
                    current_signal = signals[-1]
                    current_conf = confidences[-1]
                    current_price = subset_raw['close'].iloc[-1]

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

                    # 3. Chart (Full Width)
                    st.markdown(f"##### 📉 Price Action & Signals ({len(subset_raw)} Days)")
                    
                    fig_live = go.Figure()
                    
                    # Candlesticks
                    fig_live.add_trace(go.Candlestick(
                        x=subset_raw.index,
                        open=subset_raw['open'], high=subset_raw['high'],
                        low=subset_raw['low'], close=subset_raw['close'],
                        name='Price'
                    ))
                    
                    # Plot trades from the unified backtest
                    for trade in trades_list:
                        if trade['signal_type'] == 'LONG':
                            marker_color = 'green'
                            marker_symbol = 'triangle-up'
                            marker_y = subset_raw.loc[trade['entry_date'], 'low'] * 0.98  # Below price
                        else:  # SHORT
                            marker_color = 'red' 
                            marker_symbol = 'triangle-down'
                            marker_y = subset_raw.loc[trade['entry_date'], 'high'] * 1.02  # Above price
                        
                        fig_live.add_trace(go.Scatter(
                            x=[trade['entry_date']], y=[marker_y],
                            mode='markers',
                            name=f"{trade['signal_type']} Entry",
                            marker=dict(color=marker_color, size=10, symbol=marker_symbol),
                            hovertemplate=f"<b>{trade['signal_type']} Entry</b><br>Price: ${trade['entry_price']:.2f}<extra></extra>",
                            showlegend=False
                        ))
                    
                    # Add lines and markers if ACTIVE signal
                    if current_signal != 0:
                        # Risk Calculation
                        if 'ATR_14' in features.columns:
                            atr = features['ATR_14'].iloc[-1]
                        else:
                            atr = current_price * 0.02 
                            
                        stop_loss = current_price - (2 * atr) if current_signal == 1 else current_price + (2 * atr)
                        take_profit = current_price + (3 * atr) if current_signal == 1 else current_price - (3 * atr)

                        # Lines on Chart
                        fig_live.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text="SL")
                        fig_live.add_hline(y=take_profit, line_dash="dash", line_color="green", annotation_text="TP")
                        fig_live.add_hline(y=current_price, line_dash="dot", line_color="blue", annotation_text="Entry")
                        
                        # HOLDING MARKER
                        fig_live.add_trace(go.Scatter(
                            x=[subset_raw.index[-1]], 
                            y=[current_price],
                            mode='markers',
                            name='Current Status',
                            marker=dict(color=sig_color, size=8, symbol='circle'),
                            hovertemplate=f"<b>Holding ({sig_text})</b><br>Price: ${current_price:.2f}<extra></extra>"
                        ))
                        
                        fig_live.update_layout(
                            height=500, 
                            margin=dict(l=0, r=0, t=0, b=0),
                            xaxis_rangeslider_visible=False,
                            showlegend=False
                        )
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
                        # Render chart even if no signal
                        fig_live.update_layout(
                            height=500, 
                            margin=dict(l=0, r=0, t=0, b=0),
                            xaxis_rangeslider_visible=False,
                            showlegend=False
                        )
                        st.plotly_chart(fig_live, use_container_width=True)
                        st.info("Waiting for new setup...")

                # --- AUTO-REFRESH LOOP ---
                if live_mode:
                    # Auto-refresh every 60 seconds to check for new data
                    time.sleep(60)
                    st.rerun()
                    
                # Log latest (Only if data updated to avoid spam)
                # if data_updated:
                #     pm.log_signal(...) 
                    
                else:
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
