import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesPatternDetector:
    """
    Advanced time-series pattern detection using statistical and ML methods.
    """
    
    def __init__(self, data):
        self.data = data.copy()
        self.features = {}
        self.patterns = {}
    
    def detect_all_patterns(self):
        """Detect all available time-series patterns."""
        patterns = {
            # Trend patterns
            'trend_strength': self.detect_trend_strength(),
            'trend_acceleration': self.detect_trend_acceleration(),
            'trend_exhaustion': self.detect_trend_exhaustion(),
            
            # Cycle patterns  
            'cycle_detection': self.detect_cycles(),
            'seasonal_patterns': self.detect_seasonality(),
            
            # Volatility patterns
            'volatility_clustering': self.detect_volatility_clustering(),
            'volatility_breakout': self.detect_volatility_breakout(),
            
            # Support/Resistance
            'support_resistance': self.detect_support_resistance(),
            'breakout_confirmation': self.detect_breakout_confirmation(),
            
            # Mean reversion
            'mean_reversion_signal': self.detect_mean_reversion(),
            'oversold_oversupply': self.detect_oversold_oversupply(),
            
            # Advanced patterns
            'fractal_patterns': self.detect_fractal_patterns(),
            'chaos_patterns': self.detect_chaos_patterns(),
            'anomaly_detection': self.detect_anomalies(),
            
            # Momentum patterns
            'momentum_divergence': self.detect_momentum_divergence(),
            'momentum_exhaustion': self.detect_momentum_exhaustion(),
            
            # Volume patterns
            'volume_price_correlation': self.detect_volume_price_correlation(),
            'smart_money_flow': self.detect_smart_money_flow()
        }
        
        self.patterns = patterns
        return patterns
    
    def detect_trend_strength(self):
        """Detect trend strength using multiple methods."""
        close = self.data['close']
        
        # Linear regression slope
        x = np.arange(len(close))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, close)
        
        # ADX-based trend strength
        adx_data = ta.adx(self.data['high'], self.data['low'], self.data['close'], length=14)
        adx = adx_data['ADX_14'] if 'ADX_14' in adx_data.columns else pd.Series([0] * len(self.data))
        plus_di = adx_data['DMP_14'] if 'DMP_14' in adx_data.columns else pd.Series([0] * len(self.data))
        minus_di = adx_data['DMN_14'] if 'DMN_14' in adx_data.columns else pd.Series([0] * len(self.data))
        
        return {
            'slope': slope,
            'r_squared': r_value**2,
            'adx': adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0,
            'trend_direction': 1 if slope > 0 else -1,
            'trend_strength': min(abs(slope) * 1000, 100),  # Normalized strength
            'directional_strength': abs(plus_di.iloc[-1] - minus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0
        }
    
    def detect_trend_acceleration(self):
        """Detect if trend is accelerating or decelerating."""
        close = self.data['close']
        
        # Calculate velocity (first derivative)
        velocity = np.diff(close)
        
        # Calculate acceleration (second derivative)
        acceleration = np.diff(velocity)
        
        # Recent acceleration
        recent_accel = np.mean(acceleration[-5:]) if len(acceleration) >= 5 else 0
        
        return {
            'acceleration': recent_accel,
            'is_accelerating': recent_accel > 0,
            'acceleration_strength': abs(recent_accel)
        }
    
    def detect_trend_exhaustion(self):
        """Detect trend exhaustion signals."""
        close = self.data['close']
        volume = self.data.get('volume', pd.Series([1] * len(close)))
        
        # RSI divergence
        rsi = ta.rsi(close, length=14)
        
        # Volume analysis
        avg_volume = volume.rolling(window=20).mean()
        volume_ratio = volume.iloc[-1] / avg_volume.iloc[-1] if not pd.isna(avg_volume.iloc[-1]) else 1
        
        # Price exhaustion (smaller moves on high volume)
        price_change = abs(close.pct_change().iloc[-1])
        
        return {
            'rsi_extreme': rsi.iloc[-1] > 70 or rsi.iloc[-1] < 30,
            'rsi_value': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            'volume_exhaustion': volume_ratio > 2 and price_change < 0.01,
            'exhaustion_score': min((abs(rsi.iloc[-1] - 50) / 50) * 100, 100) if not pd.isna(rsi.iloc[-1]) else 0
        }
    
    def detect_cycles(self):
        """Detect cyclical patterns using FFT."""
        close = self.data['close']
        
        # Remove trend
        detrended = close - close.rolling(window=20).mean()
        detrended = detrended.dropna()
        
        if len(detrended) < 20:
            return {'dominant_cycle': 0, 'cycle_strength': 0}
        
        # FFT analysis
        fft = np.fft.fft(detrended.values)
        frequencies = np.fft.fftfreq(len(detrended))
        
        # Find dominant frequency
        power_spectrum = np.abs(fft)**2
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_period = 1 / abs(frequencies[dominant_freq_idx]) if frequencies[dominant_freq_idx] != 0 else 0
        
        return {
            'dominant_cycle': int(dominant_period) if dominant_period > 2 else 0,
            'cycle_strength': power_spectrum[dominant_freq_idx] / np.sum(power_spectrum) * 100
        }
    
    def detect_seasonality(self):
        """Detect seasonal patterns."""
        close = self.data['close']
        
        # Simple seasonality detection
        if len(close) < 60:  # Need at least 60 periods
            return {'has_seasonality': False, 'seasonal_strength': 0}
        
        # Check for weekly patterns (5-day cycle)
        weekly_corr = self.calculate_autocorrelation(close, lag=5)
        
        # Check for monthly patterns (20-day cycle)  
        monthly_corr = self.calculate_autocorrelation(close, lag=20)
        
        return {
            'has_seasonality': max(weekly_corr, monthly_corr) > 0.3,
            'weekly_correlation': weekly_corr,
            'monthly_correlation': monthly_corr,
            'seasonal_strength': max(weekly_corr, monthly_corr) * 100
        }
    
    def calculate_autocorrelation(self, series, lag):
        """Calculate autocorrelation at given lag."""
        if len(series) <= lag:
            return 0
        
        n = len(series) - lag
        series1 = series.iloc[:-lag]
        series2 = series.iloc[lag:]
        
        correlation = np.corrcoef(series1, series2)[0, 1]
        return correlation if not np.isnan(correlation) else 0
    
    def detect_volatility_clustering(self):
        """Detect GARCH-like volatility clustering."""
        returns = self.data['close'].pct_change().dropna()
        
        if len(returns) < 20:
            return {'has_clustering': False, 'clustering_strength': 0}
        
        # Calculate squared returns (volatility proxy)
        squared_returns = returns**2
        
        # Test for autocorrelation in squared returns
        autocorr_1 = self.calculate_autocorrelation(squared_returns, 1)
        autocorr_5 = self.calculate_autocorrelation(squared_returns, 5)
        
        clustering_strength = max(autocorr_1, autocorr_5)
        
        return {
            'has_clustering': clustering_strength > 0.1,
            'clustering_strength': clustering_strength * 100,
            'volatility_persistence': autocorr_1 * 100
        }
    
    def detect_volatility_breakout(self):
        """Detect volatility breakout patterns."""
        close = self.data['close']
        
        # Calculate ATR
        atr = ta.atr(self.data['high'], self.data['low'], self.data['close'], length=14)
        atr_avg = atr.rolling(window=20).mean()
        
        current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
        avg_atr = atr_avg.iloc[-1] if not pd.isna(atr_avg.iloc[-1]) else 0
        
        # Bollinger Bands squeeze
        bb_bands = ta.bbands(close, length=20, std=2)
        bb_upper = bb_bands['BBU_20_2.0'] if 'BBU_20_2.0' in bb_bands.columns else close
        bb_middle = bb_bands['BBM_20_2.0'] if 'BBM_20_2.0' in bb_bands.columns else close
        bb_lower = bb_bands['BBL_20_2.0'] if 'BBL_20_2.0' in bb_bands.columns else close
        bb_width = (bb_upper - bb_lower) / bb_middle
        bb_squeeze = bb_width.iloc[-1] < bb_width.rolling(window=20).quantile(0.1).iloc[-1]
        
        return {
            'atr_breakout': current_atr > avg_atr * 1.5 if avg_atr > 0 else False,
            'bb_squeeze': bb_squeeze if not pd.isna(bb_squeeze) else False,
            'volatility_expansion': (current_atr / avg_atr * 100) if avg_atr > 0 else 100
        }
    
    def detect_support_resistance(self):
        """Detect support and resistance levels."""
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        
        # Find local maxima and minima
        max_indices = argrelextrema(high.values, np.greater, order=5)[0]
        min_indices = argrelextrema(low.values, np.less, order=5)[0]
        
        # Get resistance levels (recent highs)
        resistance_levels = high.iloc[max_indices].tail(5).values if len(max_indices) > 0 else []
        
        # Get support levels (recent lows)
        support_levels = low.iloc[min_indices].tail(5).values if len(min_indices) > 0 else []
        
        current_price = close.iloc[-1]
        
        # Find nearest levels
        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price)) if len(resistance_levels) > 0 else current_price * 1.05
        nearest_support = min(support_levels, key=lambda x: abs(x - current_price)) if len(support_levels) > 0 else current_price * 0.95
        
        return {
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'resistance_strength': len([r for r in resistance_levels if abs(r - nearest_resistance) < nearest_resistance * 0.002]),
            'support_strength': len([s for s in support_levels if abs(s - nearest_support) < nearest_support * 0.002]),
            'distance_to_resistance': (nearest_resistance - current_price) / current_price * 100,
            'distance_to_support': (current_price - nearest_support) / current_price * 100
        }
    
    def detect_breakout_confirmation(self):
        """Detect breakout confirmation signals."""
        close = self.data['close']
        volume = self.data.get('volume', pd.Series([1] * len(close)))
        
        # Volume confirmation
        avg_volume = volume.rolling(window=20).mean()
        volume_spike = volume.iloc[-1] > avg_volume.iloc[-1] * 1.5 if not pd.isna(avg_volume.iloc[-1]) else False
        
        # Price momentum
        momentum = ta.mom(close, length=10)
        strong_momentum = abs(momentum.iloc[-1]) > momentum.rolling(window=20).std().iloc[-1] * 2 if not pd.isna(momentum.iloc[-1]) else False
        
        return {
            'volume_confirmation': volume_spike,
            'momentum_confirmation': strong_momentum,
            'breakout_strength': (volume.iloc[-1] / avg_volume.iloc[-1]) * abs(momentum.iloc[-1]) if not pd.isna(avg_volume.iloc[-1]) and not pd.isna(momentum.iloc[-1]) else 0
        }
    
    def detect_mean_reversion(self):
        """Detect mean reversion signals."""
        close = self.data['close']
        
        # Bollinger Bands
        bb_bands = ta.bbands(close, length=20, std=2)
        bb_upper = bb_bands['BBU_20_2.0'] if 'BBU_20_2.0' in bb_bands.columns else close
        bb_middle = bb_bands['BBM_20_2.0'] if 'BBM_20_2.0' in bb_bands.columns else close  
        bb_lower = bb_bands['BBL_20_2.0'] if 'BBL_20_2.0' in bb_bands.columns else close
        
        # Current position relative to bands
        current_price = close.iloc[-1]
        bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else 0.5
        
        # RSI mean reversion
        rsi = ta.rsi(close, length=14)
        rsi_mean_reversion = (rsi.iloc[-1] < 30 and rsi.iloc[-2] > rsi.iloc[-1]) or (rsi.iloc[-1] > 70 and rsi.iloc[-2] < rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else False
        
        return {
            'bb_mean_reversion': bb_position < 0.1 or bb_position > 0.9,
            'rsi_mean_reversion': rsi_mean_reversion,
            'mean_reversion_strength': abs(bb_position - 0.5) * 200,  # Distance from middle
            'oversold': bb_position < 0.1,
            'overbought': bb_position > 0.9
        }
    
    def detect_oversold_oversupply(self):
        """Detect oversold/overbought conditions using multiple indicators."""
        close = self.data['close']
        
        # Multiple oscillators
        rsi = ta.rsi(close, length=14)
        stoch_data = ta.stoch(self.data['high'], self.data['low'], self.data['close'])
        stoch_k = stoch_data['STOCHk_14_3_3'] if 'STOCHk_14_3_3' in stoch_data.columns else pd.Series([50] * len(close))
        stoch_d = stoch_data['STOCHd_14_3_3'] if 'STOCHd_14_3_3' in stoch_data.columns else pd.Series([50] * len(close))
        williams_r = ta.willr(self.data['high'], self.data['low'], self.data['close'], length=14)
        
        # Consensus scoring
        oversold_signals = 0
        overbought_signals = 0
        
        if not pd.isna(rsi.iloc[-1]):
            if rsi.iloc[-1] < 30:
                oversold_signals += 1
            elif rsi.iloc[-1] > 70:
                overbought_signals += 1
                
        if not pd.isna(stoch_d.iloc[-1]):
            if stoch_d.iloc[-1] < 20:
                oversold_signals += 1
            elif stoch_d.iloc[-1] > 80:
                overbought_signals += 1
                
        if not pd.isna(williams_r.iloc[-1]):
            if williams_r.iloc[-1] < -80:
                oversold_signals += 1
            elif williams_r.iloc[-1] > -20:
                overbought_signals += 1
        
        return {
            'oversold_consensus': oversold_signals,
            'overbought_consensus': overbought_signals,
            'extreme_condition': oversold_signals >= 2 or overbought_signals >= 2,
            'condition_strength': max(oversold_signals, overbought_signals) / 3 * 100
        }
    
    def detect_fractal_patterns(self):
        """Detect fractal patterns using highs and lows."""
        high = self.data['high']
        low = self.data['low']
        
        def is_fractal_high(data, index, n=2):
            """Check if index is a fractal high."""
            if index < n or index >= len(data) - n:
                return False
            
            for i in range(index - n, index + n + 1):
                if i != index and data.iloc[i] >= data.iloc[index]:
                    return False
            return True
        
        def is_fractal_low(data, index, n=2):
            """Check if index is a fractal low.""" 
            if index < n or index >= len(data) - n:
                return False
                
            for i in range(index - n, index + n + 1):
                if i != index and data.iloc[i] <= data.iloc[index]:
                    return False
            return True
        
        # Find recent fractals
        fractal_highs = []
        fractal_lows = []
        
        for i in range(2, len(high) - 2):
            if is_fractal_high(high, i):
                fractal_highs.append((i, high.iloc[i]))
            if is_fractal_low(low, i):
                fractal_lows.append((i, low.iloc[i]))
        
        return {
            'recent_fractal_highs': len([fh for fh in fractal_highs if fh[0] > len(high) - 20]),
            'recent_fractal_lows': len([fl for fl in fractal_lows if fl[0] > len(low) - 20]),
            'fractal_pattern_strength': len(fractal_highs) + len(fractal_lows)
        }
    
    def detect_chaos_patterns(self):
        """Detect chaos and non-linear patterns."""
        close = self.data['close']
        returns = close.pct_change().dropna()
        
        if len(returns) < 50:
            return {'hurst_exponent': 0.5, 'is_chaotic': False}
        
        # Simplified Hurst exponent calculation
        lags = range(2, min(20, len(returns)//4))
        variability = []
        
        for lag in lags:
            # Calculate the variability at this lag
            lagged_returns = returns.rolling(window=lag).std().dropna()
            if len(lagged_returns) > 0:
                variability.append(np.mean(lagged_returns))
        
        if len(variability) < 2:
            return {'hurst_exponent': 0.5, 'is_chaotic': False}
        
        # Linear regression on log-log plot
        log_lags = np.log(list(lags)[:len(variability)])
        log_variability = np.log(variability)
        
        hurst_exponent, _, _, _, _ = stats.linregress(log_lags, log_variability)
        
        return {
            'hurst_exponent': hurst_exponent,
            'is_persistent': hurst_exponent > 0.5,
            'is_anti_persistent': hurst_exponent < 0.5,
            'chaos_strength': abs(hurst_exponent - 0.5) * 200
        }
    
    def detect_anomalies(self):
        """Detect anomalous patterns using isolation forest."""
        close = self.data['close']
        
        if len(close) < 20:
            return {'anomaly_score': 0, 'is_anomaly': False}
        
        # Prepare features
        returns = close.pct_change().fillna(0)
        volatility = returns.rolling(window=5).std().fillna(0)
        
        features = np.column_stack([returns.values, volatility.values])
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = iso_forest.fit_predict(features)
        
        # Current anomaly status
        current_anomaly = anomaly_scores[-1] == -1
        
        return {
            'is_anomaly': current_anomaly,
            'anomaly_score': abs(anomaly_scores[-1]) * 100,
            'recent_anomalies': np.sum(anomaly_scores[-10:] == -1)
        }
    
    def detect_momentum_divergence(self):
        """Detect momentum divergence patterns."""
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        
        # MACD
        macd_data = ta.macd(close)
        macd = macd_data['MACD_12_26_9'] if 'MACD_12_26_9' in macd_data.columns else pd.Series([0] * len(close))
        macd_signal = macd_data['MACDs_12_26_9'] if 'MACDs_12_26_9' in macd_data.columns else pd.Series([0] * len(close))
        
        # RSI
        rsi = ta.rsi(close, length=14)
        
        # Find recent highs and lows
        recent_high_price_idx = np.argmax(high.iloc[-20:].values) + len(high) - 20
        recent_low_price_idx = np.argmin(low.iloc[-20:].values) + len(low) - 20
        
        # Check for divergence
        bullish_divergence = False
        bearish_divergence = False
        
        if recent_low_price_idx > 10 and not pd.isna(rsi.iloc[recent_low_price_idx]):
            # Price made lower low, but RSI made higher low
            if (low.iloc[-1] < low.iloc[recent_low_price_idx] and 
                rsi.iloc[-1] > rsi.iloc[recent_low_price_idx]):
                bullish_divergence = True
        
        if recent_high_price_idx > 10 and not pd.isna(rsi.iloc[recent_high_price_idx]):
            # Price made higher high, but RSI made lower high
            if (high.iloc[-1] > high.iloc[recent_high_price_idx] and 
                rsi.iloc[-1] < rsi.iloc[recent_high_price_idx]):
                bearish_divergence = True
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'macd_trend': 1 if macd.iloc[-1] > macd_signal.iloc[-1] else -1 if not pd.isna(macd.iloc[-1]) else 0,
            'divergence_strength': 50 if bullish_divergence or bearish_divergence else 0
        }
    
    def detect_momentum_exhaustion(self):
        """Detect momentum exhaustion signals."""
        close = self.data['close']
        
        # Rate of Change
        roc = ta.roc(close, length=10)
        
        # Momentum
        momentum = ta.mom(close, length=10)
        
        # Stochastic
        stoch_data = ta.stoch(self.data['high'], self.data['low'], self.data['close'])
        stoch_k = stoch_data['STOCHk_14_3_3'] if 'STOCHk_14_3_3' in stoch_data.columns else pd.Series([50] * len(close))
        stoch_d = stoch_data['STOCHd_14_3_3'] if 'STOCHd_14_3_3' in stoch_data.columns else pd.Series([50] * len(close))
        
        # Check for slowing momentum
        roc_slowing = roc.iloc[-1] < roc.iloc[-2] < roc.iloc[-3] if not pd.isna(roc.iloc[-1]) else False
        momentum_slowing = momentum.iloc[-1] < momentum.iloc[-2] < momentum.iloc[-3] if not pd.isna(momentum.iloc[-1]) else False
        stoch_extreme = stoch_k.iloc[-1] > 80 or stoch_k.iloc[-1] < 20 if not pd.isna(stoch_k.iloc[-1]) else False
        
        return {
            'roc_slowing': roc_slowing,
            'momentum_slowing': momentum_slowing,
            'stoch_extreme': stoch_extreme,
            'exhaustion_score': sum([roc_slowing, momentum_slowing, stoch_extreme]) / 3 * 100
        }
    
    def detect_volume_price_correlation(self):
        """Analyze volume-price relationships.""" 
        close = self.data['close']
        volume = self.data.get('volume', pd.Series([1] * len(close)))
        
        # Price-volume correlation
        returns = close.pct_change().fillna(0)
        volume_change = volume.pct_change().fillna(0)
        
        correlation = np.corrcoef(returns.iloc[-20:], volume_change.iloc[-20:])[0, 1] if len(returns) >= 20 else 0
        
        # On-Balance Volume
        obv = ta.obv(close, volume)
        obv_trend = 1 if obv.iloc[-1] > obv.iloc[-5] else -1 if not pd.isna(obv.iloc[-1]) else 0
        
        return {
            'price_volume_correlation': correlation if not np.isnan(correlation) else 0,
            'obv_trend': obv_trend,
            'volume_confirms_price': correlation > 0.3,
            'correlation_strength': abs(correlation) * 100 if not np.isnan(correlation) else 0
        }
    
    def detect_smart_money_flow(self):
        """Detect smart money flow patterns."""
        high = self.data['high']
        low = self.data['low'] 
        close = self.data['close']
        volume = self.data.get('volume', pd.Series([1] * len(close)))
        
        # Money Flow Index
        mfi = ta.mfi(high, low, close, volume, length=14)
        
        # Accumulation/Distribution Line
        ad_line = ta.ad(high, low, close, volume)
        
        # Volume Price Trend
        vpt = volume.iloc[0]  # Initialize
        vpt_values = [vpt]
        
        for i in range(1, len(close)):
            price_change = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
            vpt += volume.iloc[i] * price_change
            vpt_values.append(vpt)
        
        vpt_series = pd.Series(vpt_values, index=close.index)
        vpt_trend = 1 if vpt_series.iloc[-1] > vpt_series.iloc[-5] else -1
        
        return {
            'mfi_value': mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else 50,
            'ad_line_trend': 1 if ad_line.iloc[-1] > ad_line.iloc[-5] else -1 if not pd.isna(ad_line.iloc[-1]) else 0,
            'vpt_trend': vpt_trend,
            'smart_money_bullish': mfi.iloc[-1] > 50 and ad_line.iloc[-1] > ad_line.iloc[-5] if not pd.isna(mfi.iloc[-1]) and not pd.isna(ad_line.iloc[-1]) else False,
            'money_flow_strength': abs(mfi.iloc[-1] - 50) * 2 if not pd.isna(mfi.iloc[-1]) else 0
        }


def get_advanced_indicators(data):
    """
    Get all advanced time-series indicators for the given data.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        dict: Dictionary of all pattern detection results
    """
    detector = TimeSeriesPatternDetector(data)
    return detector.detect_all_patterns()


# Integration functions for the main app
def calculate_advanced_technical_signals(data):
    """Calculate advanced technical signals that can be used in optimization."""
    patterns = get_advanced_indicators(data)
    
    # Convert pattern results to binary signals
    signals = {}
    
    # Trend signals
    trend = patterns['trend_strength']
    signals['strong_uptrend'] = (trend['trend_direction'] > 0 and trend['trend_strength'] > 30)
    signals['strong_downtrend'] = (trend['trend_direction'] < 0 and trend['trend_strength'] > 30)
    
    # Reversal signals
    mean_reversion = patterns['mean_reversion_signal']
    signals['oversold_reversal'] = mean_reversion['oversold']
    signals['overbought_reversal'] = mean_reversion['overbought']
    
    # Breakout signals
    volatility = patterns['volatility_breakout']
    signals['volatility_expansion'] = volatility['atr_breakout']
    signals['bollinger_squeeze'] = volatility['bb_squeeze']
    
    # Momentum signals
    momentum_div = patterns['momentum_divergence']
    signals['bullish_divergence'] = momentum_div['bullish_divergence']
    signals['bearish_divergence'] = momentum_div['bearish_divergence']
    
    # Volume signals
    volume_corr = patterns['volume_price_correlation']
    signals['volume_confirmation'] = volume_corr['volume_confirms_price']
    
    return signals


# Add to indicators.py integration
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
