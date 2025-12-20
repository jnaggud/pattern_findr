import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas_ta as ta

class MarketRegimeDetector:
    def __init__(self, n_components=3, lookback=20, vol_lookback=20, trend_lookback=50, random_state=42):
        """
        Enhanced Market Regime Detector using multiple signals.
        
        Regimes:
        - Bull Trend: Strong upward momentum, low/moderate volatility
        - Bear/Stress: High volatility, negative momentum
        - Sideways/Chop: Low trend strength, range-bound
        
        Parameters:
        - n_components: Number of regimes (default 3)
        - lookback: General lookback for momentum (default 20)
        - vol_lookback: Lookback for volatility calculation (default 20)
        - trend_lookback: Lookback for trend detection (default 50)
        """
        self.n_components = n_components
        self.lookback = lookback
        self.vol_lookback = vol_lookback
        self.trend_lookback = trend_lookback
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', 
                                    n_init=10, random_state=random_state)
        self.scaler = StandardScaler()
        self.regime_map = {}
        
    def prepare_features(self, data: pd.DataFrame):
        """Generates enhanced features for regime detection with improved bear detection."""
        df = data.copy()
        
        # 1. Volatility (Multiple Measures)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['volatility'] = df['atr'] / df['close']
        
        # Rolling Std of Returns (Realized Vol)
        df['returns_daily'] = df['close'].pct_change()
        df['realized_vol'] = df['returns_daily'].rolling(self.vol_lookback).std() * np.sqrt(252)
        
        # Volatility Regime (High vs Low relative to recent history)
        df['vol_zscore'] = (df['realized_vol'] - df['realized_vol'].rolling(60).mean()) / df['realized_vol'].rolling(60).std()
        df['vol_zscore'] = df['vol_zscore'].fillna(0)
        
        # 2. Trend Strength (ADX) with Direction
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            adx_col = [c for c in adx.columns if c.startswith('ADX')][0]
            dmp_col = [c for c in adx.columns if c.startswith('DMP')][0]
            dmn_col = [c for c in adx.columns if c.startswith('DMN')][0]
            df['trend_strength'] = adx[adx_col]
            df['dmp'] = adx[dmp_col]
            df['dmn'] = adx[dmn_col]
            # Trend Direction: DMP > DMN = Bullish, DMN > DMP = Bearish
            df['trend_direction'] = (df['dmp'] - df['dmn']) / (df['dmp'] + df['dmn'] + 1e-8)
            # Bear trend indicator: DMN dominance
            df['bear_trend'] = (df['dmn'] > df['dmp']).astype(float) * (df['trend_strength'] / 50)
        else:
            df['trend_strength'] = 0
            df['trend_direction'] = 0
            df['bear_trend'] = 0
            
        # 3. Price Position relative to Moving Averages (Multiple timeframes)
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        
        # Price vs SMAs (normalized)
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        df['price_vs_sma200'] = (df['close'] - df['sma_200']) / df['sma_200']
        
        # SMA Slopes (trend direction)
        df['sma50_slope'] = df['sma_50'].pct_change(10) * 100
        df['sma200_slope'] = df['sma_200'].pct_change(20) * 100
        
        # Death Cross / Golden Cross indicator
        df['sma_cross'] = (df['sma_50'] - df['sma_200']) / df['sma_200']
        
        # 4. Momentum (Multi-timeframe) - KEY FOR BEAR DETECTION
        df['mom_5'] = df['close'].pct_change(5)
        df['mom_20'] = df['close'].pct_change(self.lookback)
        df['mom_50'] = df['close'].pct_change(50)
        
        # Drawdown from recent high (bear market indicator)
        df['rolling_high'] = df['close'].rolling(60).max()
        df['drawdown'] = (df['close'] - df['rolling_high']) / df['rolling_high']
        
        # Consecutive down days
        df['down_day'] = (df['returns_daily'] < 0).astype(float)
        df['consec_down'] = df['down_day'].rolling(10).sum() / 10  # Fraction of down days
        
        # 5. Range-bound detection (Bollinger Band Width)
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
            bbu = [c for c in bb.columns if 'BBU' in c][0]
            bbl = [c for c in bb.columns if 'BBL' in c][0]
            df['bb_width'] = (bb[bbu] - bb[bbl]) / df['close']
        else:
            df['bb_width'] = 0
            
        # Select features for clustering - now includes bear-specific features
        feature_cols = ['volatility', 'vol_zscore', 'trend_strength', 'trend_direction', 
                        'bear_trend', 'price_vs_sma50', 'price_vs_sma200', 'sma_cross',
                        'sma50_slope', 'mom_20', 'mom_50', 'drawdown', 'consec_down', 'bb_width']
        
        # Drop NaNs
        features = df[feature_cols].dropna()
        return features

    def fit_predict(self, data: pd.DataFrame):
        """
        Hybrid approach: Uses direct rule-based classification for clear regimes,
        GMM for ambiguous cases. This ensures bear markets are properly detected.
        """
        features = self.prepare_features(data)
        
        if features.empty:
            return pd.Series(0, index=data.index), {'0': 'Unknown'}
        
        # Create regime series with direct rule-based classification
        regime_series = pd.Series(1, index=features.index)  # Default to Chop (1)
        
        # Get raw (unscaled) features for rule-based classification
        raw_features = features.copy()
        
        # =================================================================
        # RULE-BASED CLASSIFICATION (Primary - catches clear regimes)
        # =================================================================
        
        # BEAR CONDITIONS (most important to get right)
        # Any of these strong bear signals triggers Bear/Stress
        bear_mask = (
            # Strong negative momentum
            (raw_features['mom_50'] < -0.10) |  # Down >10% in 50 days
            (raw_features['mom_20'] < -0.08) |  # Down >8% in 20 days
            # Significant drawdown from high
            (raw_features['drawdown'] < -0.15) |  # Down >15% from 60-day high
            # Death cross with negative momentum
            ((raw_features['sma_cross'] < -0.02) & (raw_features['mom_20'] < 0)) |
            # Price well below SMA200 with falling slope
            ((raw_features['price_vs_sma200'] < -0.05) & (raw_features['sma50_slope'] < 0)) |
            # Strong bearish trend (DMN dominance with high ADX)
            ((raw_features['bear_trend'] > 0.4) & (raw_features['trend_strength'] > 25))
        )
        
        # BULL CONDITIONS
        bull_mask = (
            # Strong positive momentum
            ((raw_features['mom_50'] > 0.10) & (raw_features['mom_20'] > 0.03)) |
            # Golden cross with positive momentum
            ((raw_features['sma_cross'] > 0.02) & (raw_features['mom_20'] > 0)) |
            # Price above SMA200 with rising slope and positive momentum
            ((raw_features['price_vs_sma200'] > 0.05) & 
             (raw_features['sma50_slope'] > 0) & 
             (raw_features['trend_direction'] > 0.2)) |
            # Strong bullish trend
            ((raw_features['trend_direction'] > 0.3) & 
             (raw_features['trend_strength'] > 25) & 
             (raw_features['mom_20'] > 0))
        )
        
        # Apply classifications (Bear takes priority over Bull for safety)
        regime_series[bull_mask & ~bear_mask] = 0  # Bull = 0
        regime_series[bear_mask] = 2  # Bear = 2
        # Remaining = 1 (Chop/Sideways)
        
        # =================================================================
        # GMM REFINEMENT (Secondary - for ambiguous periods)
        # =================================================================
        # Use GMM to refine the Chop regions only
        chop_mask = regime_series == 1
        if chop_mask.sum() > 50:  # Only if enough ambiguous points
            X_chop = self.scaler.fit_transform(features.loc[chop_mask])
            
            # Fit GMM on chop regions to find sub-regimes
            try:
                self.gmm.fit(X_chop)
                gmm_labels = self.gmm.predict(X_chop)
                
                # Analyze GMM clusters to see if any are actually Bull/Bear
                feature_names = features.columns.tolist()
                df_temp = pd.DataFrame(X_chop, columns=feature_names)
                df_temp['label'] = gmm_labels
                summary = df_temp.groupby('label').mean()
                
                # Check each cluster
                chop_indices = features.index[chop_mask]
                for i, (idx, label) in enumerate(zip(chop_indices, gmm_labels)):
                    row = summary.loc[label]
                    
                    # If cluster has clear bearish characteristics, reclassify
                    if (row['mom_50'] < -0.3 or row['drawdown'] < -0.3 or 
                        (row['price_vs_sma200'] < -0.3 and row['sma50_slope'] < -0.2)):
                        regime_series.loc[idx] = 2  # Bear
                    # If cluster has clear bullish characteristics, reclassify
                    elif (row['mom_50'] > 0.3 and row['price_vs_sma200'] > 0.2 and 
                          row['trend_direction'] > 0.2):
                        regime_series.loc[idx] = 0  # Bull
            except Exception:
                pass  # Keep rule-based classification if GMM fails
        
        # Create regime map
        self.regime_map = {0: 'Bull Trend', 1: 'Sideways/Chop', 2: 'Bear/Stress'}
        
        # Align to original data index
        full_regime_series = pd.Series(np.nan, index=data.index)
        full_regime_series.loc[features.index] = regime_series
        
        # Forward fill initial NaNs
        full_regime_series = full_regime_series.ffill().fillna(1)  # Default to Chop
        
        return full_regime_series, self.regime_map
