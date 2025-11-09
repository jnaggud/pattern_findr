import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def simple_rsi(close, periods=14):
    """Calculate RSI using simple pandas operations."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def simple_atr(high, low, close, periods=14):
    """Calculate ATR using simple pandas operations."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=periods).mean()
    return atr

def calculate_advanced_technical_signals(data):
    """Calculate simplified advanced technical signals."""
    
    if len(data) < 50:
        # Return default values for insufficient data
        return {
            'strong_uptrend': False,
            'strong_downtrend': False,
            'oversold_reversal': False,
            'overbought_reversal': False,
            'volatility_expansion': False,
            'bollinger_squeeze': False,
            'bullish_divergence': False,
            'bearish_divergence': False,
            'volume_confirmation': False
        }
    
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data.get('volume', pd.Series([1] * len(close)))
    
    signals = {}
    
    # 1. Trend Detection
    try:
        # Linear regression for trend
        x = np.arange(len(close))
        slope, _, r_value, _, _ = stats.linregress(x, close)
        
        trend_strength = min(abs(slope) * 1000, 100)
        signals['strong_uptrend'] = (slope > 0 and trend_strength > 30)
        signals['strong_downtrend'] = (slope < 0 and trend_strength > 30)
    except:
        signals['strong_uptrend'] = False
        signals['strong_downtrend'] = False
    
    # 2. RSI Signals
    try:
        rsi = simple_rsi(close)
        signals['oversold_reversal'] = rsi.iloc[-1] < 30 if not pd.isna(rsi.iloc[-1]) else False
        signals['overbought_reversal'] = rsi.iloc[-1] > 70 if not pd.isna(rsi.iloc[-1]) else False
    except:
        signals['oversold_reversal'] = False
        signals['overbought_reversal'] = False
    
    # 3. Volatility Signals
    try:
        atr = simple_atr(high, low, close)
        atr_avg = atr.rolling(window=20).mean()
        current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
        avg_atr = atr_avg.iloc[-1] if not pd.isna(atr_avg.iloc[-1]) else 0
        
        signals['volatility_expansion'] = current_atr > avg_atr * 1.5 if avg_atr > 0 else False
        
        # Simple Bollinger Bands
        sma = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        bb_width = (std * 2) / sma
        bb_squeeze = bb_width.iloc[-1] < bb_width.rolling(window=20).quantile(0.1).iloc[-1] if not pd.isna(bb_width.iloc[-1]) else False
        signals['bollinger_squeeze'] = bb_squeeze
        
    except:
        signals['volatility_expansion'] = False
        signals['bollinger_squeeze'] = False
    
    # 4. Momentum Divergence (simplified)
    try:
        # Simple momentum
        momentum = close - close.shift(10)
        price_trend = close.iloc[-1] > close.iloc[-20]
        momentum_trend = momentum.iloc[-1] > momentum.iloc[-20] if not pd.isna(momentum.iloc[-1]) else False
        
        signals['bullish_divergence'] = not price_trend and momentum_trend
        signals['bearish_divergence'] = price_trend and not momentum_trend
    except:
        signals['bullish_divergence'] = False
        signals['bearish_divergence'] = False
    
    # 5. Volume Confirmation
    try:
        avg_volume = volume.rolling(window=20).mean()
        price_change = abs(close.pct_change().iloc[-1])
        volume_ratio = volume.iloc[-1] / avg_volume.iloc[-1] if not pd.isna(avg_volume.iloc[-1]) else 1
        
        signals['volume_confirmation'] = volume_ratio > 1.2 and price_change > 0.01
    except:
        signals['volume_confirmation'] = False
    
    return signals

# For compatibility
ADVANCED_INDICATORS = [
    'strong_uptrend',
    'strong_downtrend', 
    'oversold_reversal',
    'overbought_reversal',
    'volatility_expansion',
    'bollinger_squeeze',
    'bullish_divergence',
    'bearish_divergence',
    'volume_confirmation'
]
