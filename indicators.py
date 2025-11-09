import pandas as pd
import pandas_ta as ta
import warnings

# Suppress specific FutureWarning from pandas_ta
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas_ta.candles.ha")

def get_all_indicators(data):
    """
    Adds a comprehensive set of technical indicators to the data.
    """
    # Ensure data is sorted by date and set it as the index
    data = data.sort_values(by='date').set_index('date')

    # Create a custom strategy with a curated list of reliable indicators
    custom_strategy = ta.Strategy(
        name="Comprehensive Strategy",
        description="A collection of reliable, non-TA-Lib indicators",
        ta=[
            # Momentum
            {"kind": "rsi"}, {"kind": "macd"}, {"kind": "ppo"}, {"kind": "roc"}, 
            {"kind": "stoch"}, {"kind": "bop"}, {"kind": "cmo"}, {"kind": "willr"},

            # Trend
            {"kind": "adx"}, {"kind": "aroon"}, {"kind": "psar"}, {"kind": "vwap"}, 
            {"kind": "ichimoku"}, {"kind": "sma", "length": 50}, {"kind": "ema", "length": 50},

            # Volatility
            {"kind": "bbands"}, {"kind": "atr"}, {"kind": "donchian", "lower_length": 20, "upper_length": 20},
            {"kind": "kc"},

            # Volume
            {"kind": "obv"}, {"kind": "cmf"}, {"kind": "mfi"}, {"kind": "eom"}, {"kind": "ad"}
        ]
    )

    # Apply the custom strategy
    data.ta.strategy(custom_strategy)

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

    # Clean up columns with too many NaNs and drop rows with any remaining NaNs
    data.dropna(axis=1, thresh=len(data) - 50, inplace=True)
    data.dropna(inplace=True)

    return data
