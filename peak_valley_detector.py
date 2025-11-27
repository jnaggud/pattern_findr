#!/usr/bin/env python3
"""
Peak and Valley Detection Module for ML Trading Signals

This module provides multiple algorithms to detect peaks and valleys in price data
for creating labeled datasets for machine learning models.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, argrelextrema
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class PeakValleyDetector:
    """
    Detects peaks and valleys in price data using multiple algorithms
    """
    
    def __init__(self):
        self.peaks = None
        self.valleys = None
        self.labels = None
        
    def rolling_window_method(self, data: pd.DataFrame, 
                            window_short: int = 10, 
                            window_long: int = 20,
                            min_change_pct: float = 3.0) -> pd.Series:
        """
        Detect peaks/valleys using rolling window maxima/minima
        
        Args:
            data: DataFrame with OHLC data
            window_short: Short-term window for peak detection
            window_long: Long-term window for confirmation  
            min_change_pct: Minimum percentage change required
            
        Returns:
            Series with labels: 1=BUY(valley), -1=SELL(peak), 0=HOLD
        """
        print(f"🔍 Rolling Window Detection: {window_short}/{window_long} day windows, {min_change_pct}% min change")
        
        high = data['high'] if 'high' in data.columns else data['close']
        low = data['low'] if 'low' in data.columns else data['close']
        close = data['close']
        
        # Rolling maxima and minima
        rolling_max_short = high.rolling(window_short, center=True).max()
        rolling_min_short = low.rolling(window_short, center=True).min()
        rolling_max_long = high.rolling(window_long, center=True).max()
        rolling_min_long = low.rolling(window_long, center=True).min()
        
        # Initialize labels
        labels = pd.Series(0, index=data.index)
        
        # Detect peaks (sell points) - less restrictive conditions
        peak_condition = (
            (high == rolling_max_long) &  # Must be long-term high
            (
                (high.shift(3) < high * (1 - min_change_pct/100)) |  # 3 days ago was lower OR
                (high.shift(-3) < high * (1 - min_change_pct/100))   # 3 days later is lower
            )
        )
        
        # Detect valleys (buy points) - less restrictive conditions
        valley_condition = (
            (low == rolling_min_long) &  # Must be long-term low
            (
                (low.shift(3) > low * (1 + min_change_pct/100)) |   # 3 days ago was higher OR
                (low.shift(-3) > low * (1 + min_change_pct/100))    # 3 days later is higher  
            )
        )
        
        labels.loc[peak_condition] = -1  # SELL at peaks
        labels.loc[valley_condition] = 1  # BUY at valleys
        
        # Store results
        self.peaks = labels[labels == -1].index
        self.valleys = labels[labels == 1].index
        
        print(f"   ✅ Found {len(self.peaks)} peaks and {len(self.valleys)} valleys")
        return labels
    
    def scipy_peaks_method(self, data: pd.DataFrame,
                          prominence_pct: float = 2.0,
                          distance: int = 10) -> pd.Series:
        """
        Use scipy's find_peaks with prominence for detection
        
        Args:
            data: DataFrame with OHLC data
            prominence_pct: Minimum prominence as percentage of price
            distance: Minimum distance between peaks
            
        Returns:
            Series with labels: 1=BUY(valley), -1=SELL(peak), 0=HOLD
        """
        print(f"🔍 SciPy Peaks Detection: {prominence_pct}% prominence, {distance} day distance")
        
        high = data['high'] if 'high' in data.columns else data['close']
        low = data['low'] if 'low' in data.columns else data['close']
        close = data['close']
        
        # Calculate prominence as percentage of average price
        avg_price = close.mean()
        prominence_threshold = avg_price * (prominence_pct / 100)
        
        # Find peaks in highs
        peak_indices, peak_properties = find_peaks(
            high.values, 
            prominence=prominence_threshold,
            distance=distance
        )
        
        # Find valleys by inverting the low series
        valley_indices, valley_properties = find_peaks(
            -low.values,
            prominence=prominence_threshold, 
            distance=distance
        )
        
        # Create labels
        labels = pd.Series(0, index=data.index)
        
        # Mark peaks and valleys
        if len(peak_indices) > 0:
            labels.iloc[peak_indices] = -1  # SELL at peaks
            
        if len(valley_indices) > 0:
            labels.iloc[valley_indices] = 1  # BUY at valleys
        
        # Store results
        self.peaks = labels[labels == -1].index
        self.valleys = labels[labels == 1].index
        
        print(f"   ✅ Found {len(self.peaks)} peaks and {len(self.valleys)} valleys")
        return labels
    
    def percentage_swing_method(self, data: pd.DataFrame,
                               swing_pct: float = 5.0,
                               lookback: int = 20) -> pd.Series:
        """
        Detect swings based on percentage moves from recent highs/lows
        
        Args:
            data: DataFrame with OHLC data  
            swing_pct: Minimum percentage swing to qualify
            lookback: Days to look back for high/low reference
            
        Returns:
            Series with labels: 1=BUY(valley), -1=SELL(peak), 0=HOLD
        """
        print(f"🔍 Percentage Swing Detection: {swing_pct}% swings, {lookback} day lookback")
        
        high = data['high'] if 'high' in data.columns else data['close']
        low = data['low'] if 'low' in data.columns else data['close'] 
        close = data['close']
        
        # Rolling highs and lows
        rolling_high = high.rolling(lookback).max()
        rolling_low = low.rolling(lookback).min()
        
        labels = pd.Series(0, index=data.index)
        
        for i in range(lookback, len(data)):
            current_high = high.iloc[i]
            current_low = low.iloc[i]
            recent_high = rolling_high.iloc[i-1]  # Previous period high
            recent_low = rolling_low.iloc[i-1]    # Previous period low
            
            # Peak: Current high is at/near rolling high, then drops significantly
            if (current_high >= recent_high * 0.99 and  # Within 1% of recent high
                i + 5 < len(data)):  # Ensure we can look ahead
                
                future_low = low.iloc[i:i+10].min()  # Lowest in next 10 days
                drop_pct = (current_high - future_low) / current_high * 100
                
                if drop_pct >= swing_pct:
                    labels.iloc[i] = -1  # SELL signal
            
            # Valley: Current low is at/near rolling low, then rises significantly  
            if (current_low <= recent_low * 1.01 and  # Within 1% of recent low
                i + 5 < len(data)):  # Ensure we can look ahead
                
                future_high = high.iloc[i:i+10].max()  # Highest in next 10 days
                rise_pct = (future_high - current_low) / current_low * 100
                
                if rise_pct >= swing_pct:
                    labels.iloc[i] = 1  # BUY signal
        
        # Store results
        self.peaks = labels[labels == -1].index
        self.valleys = labels[labels == 1].index
        
        print(f"   ✅ Found {len(self.peaks)} peaks and {len(self.valleys)} valleys")
        return labels
    
    def multi_timeframe_method(self, data: pd.DataFrame,
                             short_window: int = 10,
                             long_window: int = 30,
                             min_change: float = 3.0) -> pd.Series:
        """
        Combine multiple timeframe analysis for robust detection
        
        Args:
            data: DataFrame with OHLC data
            short_window: Short-term detection window
            long_window: Long-term confirmation window
            min_change: Minimum percentage change required
            
        Returns:
            Series with labels: 1=BUY(valley), -1=SELL(peak), 0=HOLD
        """
        print(f"🔍 Multi-Timeframe Detection: {short_window}/{long_window} windows, {min_change}% change")
        
        # Get short and long term signals
        short_labels = self.rolling_window_method(data, short_window, short_window*2, min_change)
        long_labels = self.rolling_window_method(data, long_window, long_window*2, min_change*1.5)
        
        # Combine: require agreement between timeframes
        combined_labels = pd.Series(0, index=data.index)
        
        # Peaks: both timeframes agree on sell
        peak_agreement = (short_labels == -1) & (long_labels == -1)
        combined_labels.loc[peak_agreement] = -1
        
        # Valleys: both timeframes agree on buy
        valley_agreement = (short_labels == 1) & (long_labels == 1)
        combined_labels.loc[valley_agreement] = 1
        
        # Store results
        self.peaks = combined_labels[combined_labels == -1].index
        self.valleys = combined_labels[combined_labels == 1].index
        
        print(f"   ✅ Multi-timeframe consensus: {len(self.peaks)} peaks and {len(self.valleys)} valleys")
        return combined_labels
    
    def get_detection_summary(self) -> Dict:
        """Get summary of detected peaks and valleys"""
        if self.peaks is None or self.valleys is None:
            return {"error": "No detection has been run yet"}
        
        return {
            "peaks_count": len(self.peaks),
            "valleys_count": len(self.valleys), 
            "total_signals": len(self.peaks) + len(self.valleys),
            "peaks_dates": self.peaks.tolist()[:5] if len(self.peaks) > 0 else [],
            "valleys_dates": self.valleys.tolist()[:5] if len(self.valleys) > 0 else []
        }
    
    def create_labeled_dataset(self, data: pd.DataFrame, 
                             method: str = "scipy_peaks",
                             **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create complete labeled dataset for ML training
        
        Args:
            data: Price data with indicators
            method: Detection method to use
            **kwargs: Parameters for detection method
            
        Returns:
            Tuple of (features_dataframe, labels_series)
        """
        print(f"\n🎯 Creating labeled dataset using {method} method")
        
        # Select detection method
        if method == "rolling_window":
            labels = self.rolling_window_method(data, **kwargs)
        elif method == "scipy_peaks":
            labels = self.scipy_peaks_method(data, **kwargs)
        elif method == "percentage_swing":
            labels = self.percentage_swing_method(data, **kwargs) 
        elif method == "multi_timeframe":
            labels = self.multi_timeframe_method(data, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Remove rows with NaN values (common at start/end due to rolling windows)
        clean_data = data.dropna()
        clean_labels = labels.loc[clean_data.index]
        
        print(f"📊 Dataset Summary:")
        print(f"   Total rows: {len(clean_labels)}")
        print(f"   BUY signals (1): {(clean_labels == 1).sum()}")
        print(f"   SELL signals (-1): {(clean_labels == -1).sum()}")
        print(f"   HOLD signals (0): {(clean_labels == 0).sum()}")
        print(f"   Signal ratio: {((clean_labels != 0).sum() / len(clean_labels) * 100):.1f}%")
        
        return clean_data, clean_labels

if __name__ == "__main__":
    # Simple test
    import yfinance as yf
    
    print("🧪 Testing Peak Valley Detector")
    
    # Get test data
    ticker = yf.Ticker("SPY")
    data = ticker.history(period="1y", interval="1d")
    data.columns = [col.lower() for col in data.columns]
    
    # Test detector
    detector = PeakValleyDetector()
    
    # Test scipy peaks method (new default)
    labels = detector.scipy_peaks_method(data, prominence_pct=2.0, distance=10)
    summary = detector.get_detection_summary()
    
    print(f"\n📋 Results Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
