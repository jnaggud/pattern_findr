import pandas as pd
import numpy as np

# --- Strategy Logic Functions ---

def rsi_strategy(data, params, pattern_functions):
    """RSI strategy with standardized column names"""
    if 'rsi' not in data.columns:
        raise ValueError("Missing RSI values in data")
    buy_signals = data['rsi'] < params.get('rsi_oversold', 30)
    sell_signals = data['rsi'] > params.get('rsi_overbought', 70)
    signals = np.where(buy_signals, 1, np.where(sell_signals, -1, 0))
    return pd.Series(signals, index=data.index)

def sma_crossover_strategy(data, params, pattern_functions):
    """SMA Crossover with standardized column names"""
    fast_sma_col = f"sma_{params.get('sma_fast', 20)}"
    slow_sma_col = f"sma_{params.get('sma_slow', 50)}"
    if fast_sma_col not in data.columns or slow_sma_col not in data.columns:
        raise ValueError("Missing SMA columns in data")
    buy_signals = (data[fast_sma_col] > data[slow_sma_col]) & (data[fast_sma_col].shift(1) <= data[slow_sma_col].shift(1))
    sell_signals = (data[fast_sma_col] < data[slow_sma_col]) & (data[fast_sma_col].shift(1) >= data[slow_sma_col].shift(1))
    signals = np.where(buy_signals, 1, np.where(sell_signals, -1, 0))
    return pd.Series(signals, index=data.index)

def macd_crossover_strategy(data, params, pattern_functions):
    """MACD Crossover with standardized column names"""
    if 'macd' not in data.columns or 'macd_signal' not in data.columns:
        raise ValueError("Missing MACD columns in data")
    buy_signals = (data['macd'] > data['macd_signal']) & (data['macd'].shift(1) <= data['macd_signal'].shift(1))
    sell_signals = (data['macd'] < data['macd_signal']) & (data['macd'].shift(1) >= data['macd_signal'].shift(1))
    signals = np.where(buy_signals, 1, np.where(sell_signals, -1, 0))
    return pd.Series(signals, index=data.index)

def engulfing_rsi_strategy(data, params, pattern_functions):
    """Engulfing with RSI strategy with standardized column names"""
    if 'rsi' not in data.columns:
        raise ValueError("Missing RSI values in data")
    find_bullish_engulfing = pattern_functions['Bullish Engulfing']
    find_bearish_engulfing = pattern_functions['Bearish Engulfing']
    buy_signals = find_bullish_engulfing(data) & (data['rsi'] < params.get('rsi_oversold', 30))
    sell_signals = find_bearish_engulfing(data) & (data['rsi'] > params.get('rsi_overbought', 70))
    signals = np.where(buy_signals, 1, np.where(sell_signals, -1, 0))
    return pd.Series(signals, index=data.index)

def bollinger_band_strategy(data, params, pattern_functions):
    """Bollinger Band strategy with standardized column names"""
    if 'close' not in data.columns or 'bb_lower' not in data.columns or 'bb_upper' not in data.columns:
        raise ValueError("Missing required columns for Bollinger Band strategy")
    buy_signals = data['close'] < data['bb_lower']
    sell_signals = data['close'] > data['bb_upper']
    signals = np.where(buy_signals, 1, np.where(sell_signals, -1, 0))
    return pd.Series(signals, index=data.index)

def dl_pattern_rsi_strategy(data, params, pattern_functions):
    """DL Pattern with RSI strategy with standardized column names"""
    if 'rsi' not in data.columns:
        raise ValueError("Missing RSI values in data")
    # This is a placeholder for where you would integrate your DL model's signals
    # For now, we'll simulate it with a simple pattern
    dl_buy_signal = pattern_functions['Bullish Engulfing'](data) # Replace with actual DL signal
    buy_signals = dl_buy_signal & (data['rsi'] < params.get('rsi_oversold', 30))
    signals = np.where(buy_signals, 1, 0)
    return pd.Series(signals, index=data.index)

# --- Strategy Search Space ---

def get_strategy_search_space():
    """
    Defines the library of strategy logic functions to be optimized.
    """
    strategy_library = {
        'RSI Oversold/Overbought': rsi_strategy,
        'SMA Crossover': sma_crossover_strategy,
        'MACD Crossover': macd_crossover_strategy,
        'Engulfing with RSI': engulfing_rsi_strategy,
        'Bollinger Band Mean Reversion': bollinger_band_strategy,
        'DL Pattern with RSI Filter': dl_pattern_rsi_strategy
    }
    return strategy_library
