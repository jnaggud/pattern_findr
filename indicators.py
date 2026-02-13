import os
import warnings
import pandas as pd
import numpy as np
import pandas_ta as ta
from advanced_indicators import calculate_advanced_technical_signals, ADVANCED_INDICATORS

try:
    from enhanced_pattern_detector import integrate_enhanced_patterns_with_optimization, ENHANCED_PATTERN_INDICATORS
    # NO PRINT STATEMENTS - causes spam during indicator calculation
except Exception as e:
    # NO PRINT STATEMENTS - causes spam during indicator calculation
    ENHANCED_PATTERN_INDICATORS = []
    def integrate_enhanced_patterns_with_optimization(data):
        return {}
from ml_indicators import integrate_ml_indicators, ML_INDICATOR_LIST

# Import oscillator-based indicators (composite oscillator, derivatives, novel indicators)
try:
    from oscillator_indicators import integrate_oscillator_indicators, OSCILLATOR_INDICATOR_LIST
    OSCILLATOR_INDICATORS_AVAILABLE = True
except ImportError as e:
    OSCILLATOR_INDICATORS_AVAILABLE = False
    OSCILLATOR_INDICATOR_LIST = []
    def integrate_oscillator_indicators(data):
        return data

# Suppress specific FutureWarning from pandas_ta
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas_ta.candles.ha")

def get_all_indicators(data, optuna_params=None):
    """
    Adds a comprehensive set of technical indicators to the data.
    optuna_params: Optional dictionary with optimized trend filter parameters
    """
    # Ensure data is sorted by date and set it as the index
    data = data.sort_values(by='date').set_index('date')

    # Create a custom strategy with a curated list of reliable indicators
    # Some pandas_ta versions (like the lightweight ones on certain platforms)
    # do not expose ta.Strategy at all. In that case we skip the strategy
    # application and rely on our custom/fallback indicators below.
    custom_strategy = None
    try:
        custom_strategy = ta.Strategy(
            name="Comprehensive Strategy",
            description="A collection of reliable, non-TA-Lib indicators with bear market enhancements",
            ta=[
                # Momentum
                {"kind": "rsi"}, {"kind": "macd"}, {"kind": "ppo"}, {"kind": "roc"}, 
                {"kind": "stoch"}, {"kind": "bop"}, {"kind": "cmo"}, {"kind": "willr"},
                {"kind": "cci"}, {"kind": "tsi"}, {"kind": "uo"},  # Bear market momentum indicators
                {"kind": "kdj"}, {"kind": "pgo"}, {"kind": "squeeze"},  # Additional momentum
                {"kind": "ao"}, {"kind": "bias"},  # Awesome Oscillator, Bias

                # Trend
                {"kind": "adx"}, {"kind": "aroon"}, {"kind": "psar"}, {"kind": "vwap"}, 
                {"kind": "ichimoku"}, {"kind": "sma", "length": 50}, {"kind": "ema", "length": 50},
                {"kind": "supertrend"}, {"kind": "vortex"},  # Bear market trend indicators
                {"kind": "dema", "length": 20}, {"kind": "tema", "length": 20},  # Double/Triple EMA
                {"kind": "hma", "length": 20}, {"kind": "wma", "length": 20},  # Hull/Weighted MA
                {"kind": "kama", "length": 20}, {"kind": "t3", "length": 20},  # Adaptive MAs
                {"kind": "qstick"},  # Quantitative Stick (ht_trendline added custom below)

                # Volatility
                {"kind": "bbands"}, {"kind": "atr"}, {"kind": "donchian", "lower_length": 20, "upper_length": 20},
                {"kind": "kc"}, {"kind": "ui"},  # ui = Ulcer Index (downside volatility)
                {"kind": "massi"}, {"kind": "natr"}, {"kind": "thermo"},  # Additional volatility
                {"kind": "hwc"},  # Holt-Winter Channel
                {"kind": "true_range"},  # True Range

                # Volume
                {"kind": "obv"}, {"kind": "cmf"}, {"kind": "mfi"}, {"kind": "eom"}, {"kind": "ad"},
                {"kind": "pvt"},  # Price Volume Trend
                {"kind": "nvi"}, {"kind": "pvi"},  # Negative/Positive Volume Index
                {"kind": "vwma", "length": 20},  # Volume Weighted MA
                {"kind": "kvo"},  # Klinger Volume Oscillator
                {"kind": "aobv"}  # Archer On-Balance Volume
            ]
        )
    except AttributeError:
        # Older / minimal pandas_ta builds without Strategy support
        custom_strategy = None

    # Apply the custom strategy (with error handling for missing indicators)
    if custom_strategy is not None:
        try:
            data.ta.strategy(custom_strategy)
        except AttributeError as e:
            # Some indicators might not be available in this pandas_ta version
            # Continue with available indicators
            print(f"Note: Some indicators not available in pandas_ta, will compute custom versions: {e}")
    
    # --- Add Custom Implementations for Missing Indicators ---
    
    # CRITICAL: Ensure RSI_14 exists (required by ML features)
    if 'RSI_14' not in data.columns and 'close' in data.columns:
        try:
            data['RSI_14'] = ta.rsi(data['close'], length=14)
        except Exception as e:
            print(f"Failed to calculate RSI_14: {e}")
            # Manual RSI calculation as fallback
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI_14'] = 100 - (100 / (1 + rs))
    
    # CRITICAL: Ensure MACD exists (required by ML features)
    if 'MACD_12_26_9' not in data.columns and 'close' in data.columns:
        try:
            macd_result = ta.macd(data['close'], fast=12, slow=26, signal=9)
            if macd_result is not None:
                data = pd.concat([data, macd_result], axis=1)
        except Exception as e:
            print(f"Failed to calculate MACD: {e}")
            # Manual MACD calculation as fallback
            ema_12 = data['close'].ewm(span=12, adjust=False).mean()
            ema_26 = data['close'].ewm(span=26, adjust=False).mean()
            data['MACD_12_26_9'] = ema_12 - ema_26
            data['MACDs_12_26_9'] = data['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
            data['MACDh_12_26_9'] = data['MACD_12_26_9'] - data['MACDs_12_26_9']
    
    # CRITICAL: Ensure WILLR exists (required by ML features)
    if 'WILLR_14' not in data.columns and all(col in data.columns for col in ['high', 'low', 'close']):
        try:
            data['WILLR_14'] = ta.willr(data['high'], data['low'], data['close'], length=14)
        except Exception as e:
            print(f"Failed to calculate WILLR_14: {e}")
            # Manual Williams %R calculation as fallback
            highest_high = data['high'].rolling(window=14).max()
            lowest_low = data['low'].rolling(window=14).min()
            data['WILLR_14'] = -100 * ((highest_high - data['close']) / (highest_high - lowest_low))
    
    # Hilbert Transform Trendline (if not available from pandas_ta)
    if 'HT_TRENDLINE' not in data.columns and 'close' in data.columns:
        # Simple approximation using weighted moving average
        data['HT_TRENDLINE'] = data['close'].ewm(span=7, adjust=False).mean()
    
    # Pretty Good Oscillator (PGO) - if missing
    if 'PGO_14' not in data.columns and 'close' in data.columns:
        sma_14 = data['close'].rolling(window=14).mean()
        atr_14 = data['high'].rolling(14).max() - data['low'].rolling(14).min()
        data['PGO_14'] = ((data['close'] - sma_14) / atr_14) * 100
    
    # Awesome Oscillator (AO) - if missing
    if 'AO_5_34' not in data.columns and 'high' in data.columns and 'low' in data.columns:
        median_price = (data['high'] + data['low']) / 2
        ao_fast = median_price.rolling(window=5).mean()
        ao_slow = median_price.rolling(window=34).mean()
        data['AO_5_34'] = ao_fast - ao_slow
    
    # Bias Indicator - if missing
    if 'BIAS_SMA_26' not in data.columns and 'close' in data.columns:
        sma_26 = data['close'].rolling(window=26).mean()
        data['BIAS_SMA_26'] = ((data['close'] - sma_26) / sma_26) * 100
    
    # QStick - if missing
    if 'QS_14' not in data.columns and 'open' in data.columns and 'close' in data.columns:
        data['QS_14'] = (data['close'] - data['open']).rolling(window=14).mean()
    
    # TTM Squeeze - if missing (simplified version)
    if 'SQZ_20_2.0_20_1.5' not in data.columns:
        # Bollinger Bands
        if 'BBL_20_2.0' in data.columns and 'BBU_20_2.0' in data.columns:
            bb_width = data['BBU_20_2.0'] - data['BBL_20_2.0']
        else:
            sma_20 = data['close'].rolling(window=20).mean()
            std_20 = data['close'].rolling(window=20).std()
            bb_width = 4 * std_20
        
        # Keltner Channels
        if 'KCLe_20_2' in data.columns and 'KCUe_20_2' in data.columns:
            kc_width = data['KCUe_20_2'] - data['KCLe_20_2']
        else:
            ema_20 = data['close'].ewm(span=20, adjust=False).mean()
            atr = (data['high'] - data['low']).rolling(window=20).mean()
            kc_width = 4 * atr
        
        # Squeeze: 1 when BB inside KC, 0 otherwise
        data['SQZ_20_2.0_20_1.5'] = (bb_width < kc_width).astype(int)
    
    # KDJ - if missing (Stochastic + J line)
    if 'K_14_3' not in data.columns or 'D_3' not in data.columns:
        # Calculate Stochastic if needed
        low_min = data['low'].rolling(window=14).min()
        high_max = data['high'].rolling(window=14).max()
        
        k_value = 100 * ((data['close'] - low_min) / (high_max - low_min))
        data['K_14_3'] = k_value.rolling(window=3).mean()
        data['D_3'] = data['K_14_3'].rolling(window=3).mean()
        data['J_14_3'] = 3 * data['K_14_3'] - 2 * data['D_3']  # J line
    
    # Holt-Winter Channel (HWC) - if missing (simplified)
    if 'HWC_20' not in data.columns and 'close' in data.columns:
        hwc_ma = data['close'].ewm(span=20, adjust=False).mean()
        hwc_std = data['close'].rolling(window=20).std()
        data['HWC_20'] = hwc_ma
        data['HWCu_20'] = hwc_ma + 2 * hwc_std
        data['HWCl_20'] = hwc_ma - 2 * hwc_std
    
    # MASSI (Mass Index) - if missing
    if 'MASSI_9_25' not in data.columns and 'high' in data.columns and 'low' in data.columns:
        range_hl = data['high'] - data['low']
        ema9 = range_hl.ewm(span=9, adjust=False).mean()
        ema9_ema9 = ema9.ewm(span=9, adjust=False).mean()
        mass_ratio = ema9 / ema9_ema9
        data['MASSI_9_25'] = mass_ratio.rolling(window=25).sum()
    
    # Thermo (Thermometer Indicator) - if missing
    if 'THERMO_20_2_0.5' not in data.columns:
        data['THERMO_20_2_0.5'] = (data['high'] - data['low']).rolling(window=20).mean()
    
    # Negative/Positive Volume Index - if missing
    if 'NVI' not in data.columns and 'volume' in data.columns:
        nvi = pd.Series(1000, index=data.index)
        pvi = pd.Series(1000, index=data.index)
        
        for i in range(1, len(data)):
            price_change = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
            
            if data['volume'].iloc[i] < data['volume'].iloc[i-1]:
                # Volume decreased - update NVI
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + price_change)
                pvi.iloc[i] = pvi.iloc[i-1]
            else:
                # Volume increased - update PVI
                pvi.iloc[i] = pvi.iloc[i-1] * (1 + price_change)
                nvi.iloc[i] = nvi.iloc[i-1]
        
        data['NVI'] = nvi
        data['PVI'] = pvi
    
    # Klinger Volume Oscillator - if missing
    if 'KVO_34_55_13' not in data.columns and 'volume' in data.columns:
        # Simplified KVO
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        trend = (typical_price > typical_price.shift(1)).astype(int) * 2 - 1  # 1 or -1
        volume_force = data['volume'] * trend * 100
        
        kvo_fast = volume_force.ewm(span=34, adjust=False).mean()
        kvo_slow = volume_force.ewm(span=55, adjust=False).mean()
        data['KVO_34_55_13'] = kvo_fast - kvo_slow
        data['KVOs_13'] = data['KVO_34_55_13'].ewm(span=13, adjust=False).mean()
    
    # Archer On-Balance Volume (AOBV) - if missing
    if 'AOBV' not in data.columns and 'volume' in data.columns:
        obv = (data['volume'] * ((data['close'] > data['close'].shift(1)).astype(int) * 2 - 1)).cumsum()
        data['AOBV'] = obv.ewm(span=20, adjust=False).mean()

    # --- Add Novel/Custom Indicators ---

    # 1. Volatility-Adjusted MACD
    if 'MACD_12_26_9' in data.columns and 'ATRr_14' in data.columns:
        data['macd_adj'] = data['MACD_12_26_9'] / data['ATRr_14']

    # 2. Z-Score of a Moving Average
    if 'SMA_20' in data.columns:
        sma_20 = data['SMA_20']
        rolling_std = sma_20.rolling(window=20).std()
        data['sma_20_zscore'] = (sma_20 - sma_20.rolling(window=20).mean()) / rolling_std

    # 3. RSI of VWAP
    if 'VWAP_D' in data.columns:
        data['rsi_vwap'] = ta.rsi(close=data['VWAP_D'], length=14)
    
    # 4. Bear Market Strength Indicator (custom)
    # Measures sustained downward pressure
    if 'close' in data.columns:
        # Calculate percentage below 20-day high
        rolling_max = data['close'].rolling(window=20).max()
        data['drawdown_pct'] = ((data['close'] - rolling_max) / rolling_max) * 100
        
        # Downtrend consistency (how many recent days were down)
        data['down_days_ratio'] = (data['close'] < data['close'].shift(1)).rolling(window=10).mean()
    
    # 5. Volume-Weighted Downtrend Indicator
    # Heavy volume on down days = stronger bear trend
    if 'close' in data.columns and 'volume' in data.columns:
        # Calculate if it's a down day
        down_day = (data['close'] < data['close'].shift(1))
        # Volume on down days
        down_volume = data['volume'].where(down_day, 0)
        total_volume = data['volume']
        # Ratio of down volume to total volume (10-day rolling)
        data['bear_volume_ratio'] = (down_volume.rolling(window=10).sum() / 
                                      total_volume.rolling(window=10).sum())
    
    # 6. Trend Strength Score (combines multiple trend indicators)
    # Higher negative score = stronger bear trend
    trend_score = 0
    score_count = 0
    if 'ADX_14' in data.columns and 'DMP_14' in data.columns and 'DMN_14' in data.columns:
        # ADX shows trend strength, DMN > DMP shows bearish
        trend_score += ((data['DMN_14'] - data['DMP_14']) / 100) * (data['ADX_14'] / 100)
        score_count += 1
    if 'AROOND_14' in data.columns and 'AROONU_14' in data.columns:
        # Aroon Down > Aroon Up = bearish
        trend_score += (data['AROOND_14'] - data['AROONU_14']) / 100
        score_count += 1
    if score_count > 0:
        data['bear_trend_strength'] = trend_score / score_count

    # --- Add Advanced Time-Series Pattern Indicators ---
    try:
        # Reset index temporarily for advanced indicators processing
        temp_data = data.reset_index()
        temp_data.rename(columns={'date': 'date'}, inplace=True)
        
        # Get advanced technical signals
        advanced_signals = calculate_advanced_technical_signals(temp_data)
        
        # Add advanced signals as boolean indicators
        for signal_name, signal_value in advanced_signals.items():
            data[f'adv_{signal_name}'] = signal_value
            
    except Exception as e:
        print(f"Warning: Could not calculate advanced indicators: {e}")
        # Add default values for advanced indicators if calculation fails
        for indicator in ADVANCED_INDICATORS:
            data[f'adv_{indicator}'] = False

    # --- Add Enhanced Pattern Detection Indicators ---
    try:
        # Reset index temporarily for enhanced pattern processing
        temp_data = data.reset_index()
        
        # Get enhanced pattern signals
        enhanced_signals = integrate_enhanced_patterns_with_optimization(temp_data)
        
        # Add enhanced pattern signals as boolean indicators
        for signal_name, signal_value in enhanced_signals.items():
            data[f'enh_{signal_name}'] = signal_value
            
    except Exception as e:
        print(f"Warning: Could not calculate enhanced pattern indicators: {e}")
        # Add default values for enhanced pattern indicators if calculation fails
        for indicator in ENHANCED_PATTERN_INDICATORS:
            data[f'enh_{indicator}'] = False

    # --- Add Machine Learning Indicators ---
    try:
        # Reset index temporarily for ML indicator processing
        temp_data = data.reset_index()
        
        # Get ML indicator signals
        ml_signals = integrate_ml_indicators(temp_data)
        
        # Add ML signals as indicators
        for signal_name, signal_value in ml_signals.items():
            data[f'ml_{signal_name}'] = signal_value
            
    except Exception as e:
        print(f"Warning: Could not calculate ML indicators: {e}")
        # Add default values for ML indicators if calculation fails
        for indicator in ML_INDICATOR_LIST:
            data[f'ml_{indicator}'] = 0

    # Clean up columns with too many NaNs (drop columns that are >80% NaN)
    data.dropna(axis=1, thresh=len(data) - 50, inplace=True)
    
    # CRITICAL FOR LIVE TRADING: Check if indicators work on the LAST (most recent) row
    # This simulates "Can we trade TODAY?"
    last_row_nan_cols = []
    if len(data) > 0:
        last_row = data.iloc[-1]
        for col in data.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:  # Skip OHLCV
                if pd.isna(last_row[col]):
                    last_row_nan_cols.append(col)
    
    # Remove indicators that can't produce signals for the current day
    if last_row_nan_cols:
        print(f"\n⚠️  WARNING: Removing {len(last_row_nan_cols)} indicators that cannot produce real-time signals:")
        print(f"   These indicators have NaN values on the most recent date (live trading requirement)")
        for col in last_row_nan_cols[:10]:  # Show first 10
            print(f"   - {col}")
        if len(last_row_nan_cols) > 10:
            print(f"   ... and {len(last_row_nan_cols) - 10} more")
        
        # Drop these unreliable indicators
        data.drop(columns=last_row_nan_cols, inplace=True)
    
    # Fill remaining NaN values with neutral/default values (for warmup period only)
    # This preserves the full date range for visualization
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:
            # For numeric columns, use forward fill then backward fill
            data[col].fillna(method='ffill', inplace=True)
            data[col].fillna(method='bfill', inplace=True)
            # If still NaN (empty column), fill with 0
            data[col].fillna(0, inplace=True)
        elif data[col].dtype == 'bool':
            # For boolean columns, fill with False (no signal)
            data[col].fillna(False, inplace=True)
    
    # PERFORMANCE ENHANCEMENT 1: Multi-timeframe RSI for better crash detection
    if 'RSI_14' in data.columns:
        # 3-day average RSI - catches sustained oversold conditions
        data['RSI_14_3day'] = data['RSI_14'].rolling(window=3, min_periods=1).mean()
        # 7-day average RSI - catches longer-term oversold trends  
        data['RSI_14_7day'] = data['RSI_14'].rolling(window=7, min_periods=1).mean()
    
    # PERFORMANCE ENHANCEMENT 2: Enhanced multi-period RSI ensemble
    for period in [7, 21, 28]:
        try:
            rsi_col = f'RSI_{period}'
            if rsi_col not in data.columns:
                data[rsi_col] = ta.rsi(data['close'], length=period)
        except Exception as e:
            print(f"Failed to add {rsi_col}: {e}")
    
    # PERFORMANCE ENHANCEMENT 3: Volatility-adjusted indicators
    if 'close' in data.columns:
        # 20-day volatility (annualized)
        returns = data['close'].pct_change()
        data['volatility_20d'] = returns.rolling(20).std() * (252**0.5)
        
        # Crash detection: 5-day return < -10% AND RSI oversold
        data['return_5d'] = data['close'].pct_change(5)
        data['crash_detected'] = (data['return_5d'] < -0.10) & (data['RSI_14'] < 25)
        
        # Volume surge detection (if volume data available)
        if 'volume' in data.columns:
            data['volume_avg_20d'] = data['volume'].rolling(20).mean()
            data['volume_surge'] = data['volume'] > (1.5 * data['volume_avg_20d'])

    # Add trend_filter indicator (used by optimization.py for trend filtering)
    # This combines ADX trend strength with price/MA trend direction
    # === APPROACH A: USE OPTIMIZED PARAMETERS FROM OPTUNA ===
    trend_filter = pd.Series(True, index=data.index)  # Default to True (allow trading)
    
    # Get optimized parameters or use defaults - USE NEW PARAMETER NAMES
    adx_threshold = optuna_params.get('trend_adx_threshold', 20) if optuna_params else 20
    rsi_oversold_threshold = optuna_params.get('trend_rsi_oversold', 35) if optuna_params else 35
    willr_threshold = optuna_params.get('trend_willr_threshold', -80) if optuna_params else -80
    stoch_threshold = optuna_params.get('trend_stoch_threshold', 15) if optuna_params else 15
    sma_period = optuna_params.get('trend_sma_period', 50) if optuna_params else 50
    
    if 'ADX_14' in data.columns:
        # Strong trend = ADX > optimized threshold (was static 20)
        adx_trend = data['ADX_14'] > adx_threshold
        trend_filter = trend_filter & adx_trend
    
    # Use optimized SMA period for trend detection
    sma_column = f'SMA_{sma_period}'
    if sma_column in data.columns and 'close' in data.columns:
        # Price trend = close above optimized SMA period
        price_trend = data['close'] > data[sma_column]
        trend_filter = trend_filter | price_trend  # OR logic: either ADX strong OR price trending
    elif 'SMA_50' in data.columns and 'close' in data.columns:
        # Fallback to SMA_50 if optimized period not available
        price_trend = data['close'] > data['SMA_50']
        trend_filter = trend_filter | price_trend

    if 'RSI_14' in data.columns:
        # More aggressive oversold condition - optimized threshold
        oversold_bailout = data['RSI_14'] < rsi_oversold_threshold
        trend_filter = trend_filter | oversold_bailout
    
    # Add additional oversold conditions with optimized thresholds
    if 'WILLR_14' in data.columns:
        # Williams %R below optimized threshold (was static -80)
        willr_oversold = data['WILLR_14'] < willr_threshold
        trend_filter = trend_filter | willr_oversold
        
    if 'STOCHk_14_3_3' in data.columns:
        # Stochastic %K below optimized threshold (was static 15)
        stoch_oversold = data['STOCHk_14_3_3'] < stoch_threshold
        trend_filter = trend_filter | stoch_oversold
    
    data['trend_filter'] = trend_filter
    
    # === BOTTOM DETECTION INDICATORS ===
    # These trigger at actual market bottoms, not crash starts - better timing!
    
    # Prepare common variables
    if 'volume' in data.columns:
        volume_avg_20d = data['volume'].rolling(20).mean()
        volume_ratio = data['volume'] / volume_avg_20d
    else:
        volume_ratio = pd.Series(1.0, index=data.index)
        
    daily_return = data['close'].pct_change() * 100
    returns_2d = data['close'].pct_change(2) * 100
    
    # 1. SELLING EXHAUSTION - High volume but price stops falling
    data['selling_exhaustion'] = (volume_ratio > 2.5) & (daily_return > -2.0) & (daily_return < 0.5)
    
    # 2. HAMMER PATTERN - Rejection of lower prices
    if 'high' in data.columns and 'low' in data.columns:
        candle_range = data['high'] - data['low']
        lower_wick = data['close'] - data['low']
        upper_wick = data['high'] - data['close']
        # Hammer: Long lower wick, small upper wick
        data['hammer_pattern'] = (lower_wick > candle_range * 0.6) & (upper_wick < candle_range * 0.2)
    else:
        data['hammer_pattern'] = False
    
    # 3. PANIC RECOVERY - Sharp decline followed by recovery
    data['panic_recovery'] = (daily_return.shift(1) < -4.0) & (daily_return > 1.0)
    
    # 4. OVERSOLD BOUNCE - End of oversold condition with bounce
    up_days = (daily_return > 0).rolling(3).sum()
    oversold_proxy = up_days == 0  # 0 up days in last 3 = oversold
    data['oversold_bounce'] = oversold_proxy.shift(1) & (daily_return > 0.5)
    
    # 5. VOLUME CONFIRMATION - High volume on bounce (smart money)
    high_volume = volume_ratio > 2.0
    data['volume_confirmation'] = (
        high_volume.shift(1) &  # Yesterday high volume
        (daily_return > 0.5) &  # Today bounce
        high_volume             # Today also high volume
    )
    
    # 6. STABILIZATION BOTTOM - Price stabilizes after decline
    big_decline = returns_2d < -5.0
    small_moves = abs(daily_return) < 1.5
    stability = small_moves & small_moves.shift(1)
    data['stabilization_bottom'] = big_decline.shift(2) & stability
    
    # 7. GAP FILL RECOVERY - Gap down followed by gap fill
    if 'open' in data.columns:
        prev_close = data['close'].shift(1)
        gap_down = (data['open'] / prev_close - 1) * 100 < -2.0
        gap_fill = data['close'] > prev_close
        data['gap_fill_recovery'] = gap_down & gap_fill
    else:
        data['gap_fill_recovery'] = False
    
    # 8. FEAR CAPITULATION - Extreme fear followed by relief
    fear_indicators = [
        daily_return < -3.0,    # Big decline
        volume_ratio > 3.0,     # Huge volume
    ]
    
    fear_score = pd.Series(0, index=data.index)
    for indicator in fear_indicators:
        fear_score += indicator.astype(int)
    
    # High fear yesterday, lower fear today
    data['fear_capitulation'] = (fear_score.shift(1) >= 2) & (fear_score <= 1)
    
    # 9. BOTTOM COMPOSITE SCORE
    bottom_indicators = [
        'selling_exhaustion', 'hammer_pattern', 'panic_recovery',
        'oversold_bounce', 'volume_confirmation', 'stabilization_bottom',
        'gap_fill_recovery', 'fear_capitulation'
    ]
    
    data['crash_composite_score'] = data[bottom_indicators].sum(axis=1)  # Renamed for compatibility
    
    # 10. BOTTOM BUY SIGNAL - Ultimate bottom detection
    data['crash_buy_signal'] = (
        (data['crash_composite_score'] >= 2) |      # 2+ bottom indicators OR
        data['panic_recovery'] |                    # Panic recovery OR
        data['volume_confirmation'] |               # Volume confirmation OR
        (data['hammer_pattern'] & (volume_ratio > 2.0))  # Hammer + high volume
    )

    # Mark rows where indicators aren't ready (first 50 rows as warmup period)
    data['indicators_ready'] = True
    data.iloc[:50, data.columns.get_loc('indicators_ready')] = False

    # --- Add Oscillator-Based Indicators ---
    # This adds: composite oscillator, derivatives (velocity, acceleration, jerk),
    # rolling statistics, novel oscillators (ARWO, DCO, VCMO, ICS, MJI, PRF, EWAF, KFIF),
    # and consensus/dispersion features
    try:
        # Reset index temporarily for oscillator indicator processing
        temp_data = data.reset_index()

        # Integrate all oscillator indicators (pass interval for V2 indicator scaling)
        # Try to detect interval from data frequency
        detected_interval = '1d'  # default
        if len(data) >= 2:
            try:
                time_diff = (data.index[1] - data.index[0]).total_seconds()
                if time_diff <= 300:  # 5 min or less
                    detected_interval = '5m'
                elif time_diff <= 900:  # 15 min
                    detected_interval = '15m'
                elif time_diff <= 3600:  # 1 hour
                    detected_interval = '1h'
                elif time_diff <= 14400:  # 4 hours
                    detected_interval = '4h'
            except:
                pass
        temp_data = integrate_oscillator_indicators(temp_data, interval=detected_interval)

        # Get new columns added by oscillator indicators
        original_cols = set(data.columns)
        new_cols = [c for c in temp_data.columns if c not in original_cols and c != 'date']

        # Add new oscillator columns back to data
        for col in new_cols:
            if col in temp_data.columns:
                data[col] = temp_data[col].values

    except Exception as e:
        print(f"Warning: Could not calculate oscillator indicators: {e}")

    return data
