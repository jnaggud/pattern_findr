"""
Strategy Discovery Engine
=========================
Systematically discovers winning trading strategies by:
1. Generating massive indicator library using pandas_ta
2. Creating creative conditions (slopes, crossovers, divergences, combinations)
3. Grid searching through strategy combinations
4. Using ML for feature importance and predictions
5. Surfacing top-performing strategies with interpretable rules

Outputs all data to /analysis2/ for further analysis.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from itertools import combinations, product
import warnings
from datetime import datetime
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
import json

warnings.filterwarnings('ignore')

# Try to import novel indicators
try:
    from novel_indicators import (
        calculate_arwo, calculate_dco, calculate_vcmo, calculate_ics,
        calculate_mji, calculate_prf, calculate_ewaf, calculate_kfif
    )
    NOVEL_INDICATORS_AVAILABLE = True
except ImportError:
    NOVEL_INDICATORS_AVAILABLE = False
    print("Warning: novel_indicators not available for Strategy Discovery")

# Configuration
ANALYSIS_DIR = "analysis2"
N_JOBS = max(1, (os.cpu_count() or 4) - 1)


@dataclass
class StrategyRule:
    """Represents a single trading rule/condition"""
    name: str
    condition_func: Callable
    description: str
    category: str  # 'entry_long', 'entry_short', 'exit_long', 'exit_short', 'filter'


@dataclass
class Strategy:
    """Represents a complete trading strategy"""
    name: str
    entry_long_rules: List[str]
    entry_short_rules: List[str]
    exit_long_rules: List[str]
    exit_short_rules: List[str]
    filter_rules: List[str] = field(default_factory=list)


@dataclass
class StrategyResult:
    """Results from backtesting a strategy"""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    win_rate: float
    num_trades: int
    profit_factor: float
    max_drawdown: float
    avg_trade: float
    rules_description: str
    strategy_config: Dict
    buy_hold_return: float = 0.0  # Buy & hold baseline for comparison
    outperformance: float = 0.0  # Strategy return - buy & hold return


class IndicatorGenerator:
    """Generate all possible technical indicators using pandas_ta"""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
        """
        self.df = df.copy()
        self.indicators = pd.DataFrame(index=df.index)

    def generate_all(self, verbose: bool = True) -> pd.DataFrame:
        """Generate all indicators"""
        if verbose:
            print("📊 Generating comprehensive indicator library...")

        # Store original columns
        self.indicators['close'] = self.df['close']
        self.indicators['open'] = self.df['open']
        self.indicators['high'] = self.df['high']
        self.indicators['low'] = self.df['low']
        if 'volume' in self.df.columns:
            self.indicators['volume'] = self.df['volume']

        # Generate each category
        self._generate_momentum_indicators(verbose)
        self._generate_trend_indicators(verbose)
        self._generate_volatility_indicators(verbose)
        self._generate_volume_indicators(verbose)
        self._generate_moving_averages(verbose)
        self._generate_custom_indicators(verbose)
        self._generate_novel_indicators(verbose)
        self._generate_slopes_and_derivatives(verbose)

        # Clean up
        self.indicators = self.indicators.replace([np.inf, -np.inf], np.nan)
        self.indicators = self.indicators.ffill().bfill()

        if verbose:
            print(f"✅ Generated {len(self.indicators.columns)} total indicators")

        return self.indicators

    def _generate_momentum_indicators(self, verbose: bool):
        """Generate momentum indicators"""
        if verbose:
            print("  → Momentum indicators...")

        close = self.df['close']
        high = self.df['high']
        low = self.df['low']

        # RSI at multiple periods
        for period in [7, 14, 21, 28]:
            try:
                rsi = ta.rsi(close, length=period)
                if rsi is not None:
                    self.indicators[f'rsi_{period}'] = rsi
            except:
                pass

        # Stochastic
        for k_period in [9, 14, 21]:
            try:
                stoch = ta.stoch(high, low, close, k=k_period)
                if stoch is not None and len(stoch.columns) >= 2:
                    self.indicators[f'stoch_k_{k_period}'] = stoch.iloc[:, 0]
                    self.indicators[f'stoch_d_{k_period}'] = stoch.iloc[:, 1]
            except:
                pass

        # MACD
        for fast, slow, signal in [(12, 26, 9), (8, 21, 5), (5, 13, 3)]:
            try:
                macd = ta.macd(close, fast=fast, slow=slow, signal=signal)
                if macd is not None:
                    self.indicators[f'macd_{fast}_{slow}'] = macd.iloc[:, 0]
                    self.indicators[f'macd_signal_{fast}_{slow}'] = macd.iloc[:, 1]
                    self.indicators[f'macd_hist_{fast}_{slow}'] = macd.iloc[:, 2]
            except:
                pass

        # Williams %R
        for period in [10, 14, 21]:
            try:
                willr = ta.willr(high, low, close, length=period)
                if willr is not None:
                    self.indicators[f'willr_{period}'] = willr
            except:
                pass

        # CCI
        for period in [14, 20, 40]:
            try:
                cci = ta.cci(high, low, close, length=period)
                if cci is not None:
                    self.indicators[f'cci_{period}'] = cci
            except:
                pass

        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            try:
                roc = ta.roc(close, length=period)
                if roc is not None:
                    self.indicators[f'roc_{period}'] = roc
            except:
                pass

        # MOM (Momentum)
        for period in [10, 14, 20]:
            try:
                mom = ta.mom(close, length=period)
                if mom is not None:
                    self.indicators[f'mom_{period}'] = mom
            except:
                pass

        # Ultimate Oscillator
        try:
            uo = ta.uo(high, low, close)
            if uo is not None:
                self.indicators['uo'] = uo
        except:
            pass

        # Awesome Oscillator
        try:
            ao = ta.ao(high, low)
            if ao is not None:
                self.indicators['ao'] = ao
        except:
            pass

        # PPO (Percentage Price Oscillator)
        try:
            ppo = ta.ppo(close)
            if ppo is not None:
                self.indicators['ppo'] = ppo.iloc[:, 0] if hasattr(ppo, 'iloc') else ppo
        except:
            pass

    def _generate_trend_indicators(self, verbose: bool):
        """Generate trend indicators"""
        if verbose:
            print("  → Trend indicators...")

        close = self.df['close']
        high = self.df['high']
        low = self.df['low']

        # ADX
        for period in [14, 20, 28]:
            try:
                adx = ta.adx(high, low, close, length=period)
                if adx is not None and len(adx.columns) >= 3:
                    self.indicators[f'adx_{period}'] = adx.iloc[:, 0]
                    self.indicators[f'dmp_{period}'] = adx.iloc[:, 1]  # +DI
                    self.indicators[f'dmn_{period}'] = adx.iloc[:, 2]  # -DI
            except:
                pass

        # Aroon
        for period in [14, 25]:
            try:
                aroon = ta.aroon(high, low, length=period)
                if aroon is not None and len(aroon.columns) >= 2:
                    self.indicators[f'aroon_up_{period}'] = aroon.iloc[:, 0]
                    self.indicators[f'aroon_down_{period}'] = aroon.iloc[:, 1]
                    self.indicators[f'aroon_osc_{period}'] = aroon.iloc[:, 0] - aroon.iloc[:, 1]
            except:
                pass

        # Parabolic SAR
        try:
            psar = ta.psar(high, low, close)
            if psar is not None:
                # PSAR returns multiple columns, we want the combined
                self.indicators['psar'] = psar.iloc[:, 0].combine_first(psar.iloc[:, 1])
                self.indicators['psar_trend'] = (close > self.indicators['psar']).astype(int)
        except:
            pass

        # Supertrend
        for period, mult in [(10, 3), (14, 2), (20, 2.5)]:
            try:
                st = ta.supertrend(high, low, close, length=period, multiplier=mult)
                if st is not None:
                    self.indicators[f'supertrend_{period}_{mult}'] = st.iloc[:, 0]
                    self.indicators[f'supertrend_dir_{period}_{mult}'] = st.iloc[:, 1]
            except:
                pass

        # Ichimoku
        try:
            ichi = ta.ichimoku(high, low, close)
            if ichi is not None and len(ichi) > 0:
                ichi_df = ichi[0]  # Main dataframe
                for col in ichi_df.columns:
                    self.indicators[f'ichi_{col}'] = ichi_df[col]
        except:
            pass

        # Vortex Indicator
        try:
            vortex = ta.vortex(high, low, close)
            if vortex is not None:
                self.indicators['vortex_pos'] = vortex.iloc[:, 0]
                self.indicators['vortex_neg'] = vortex.iloc[:, 1]
        except:
            pass

    def _generate_volatility_indicators(self, verbose: bool):
        """Generate volatility indicators"""
        if verbose:
            print("  → Volatility indicators...")

        close = self.df['close']
        high = self.df['high']
        low = self.df['low']

        # ATR
        for period in [7, 14, 21]:
            try:
                atr = ta.atr(high, low, close, length=period)
                if atr is not None:
                    self.indicators[f'atr_{period}'] = atr
                    self.indicators[f'atr_pct_{period}'] = atr / close * 100
            except:
                pass

        # Bollinger Bands
        for period in [10, 20, 30]:
            for std in [1.5, 2, 2.5]:
                try:
                    bb = ta.bbands(close, length=period, std=std)
                    if bb is not None and len(bb.columns) >= 3:
                        self.indicators[f'bb_upper_{period}_{std}'] = bb.iloc[:, 0]
                        self.indicators[f'bb_mid_{period}_{std}'] = bb.iloc[:, 1]
                        self.indicators[f'bb_lower_{period}_{std}'] = bb.iloc[:, 2]
                        # BB %B (where price is within bands)
                        bb_width = bb.iloc[:, 0] - bb.iloc[:, 2]
                        self.indicators[f'bb_pctb_{period}_{std}'] = (close - bb.iloc[:, 2]) / bb_width
                        self.indicators[f'bb_width_{period}_{std}'] = bb_width / bb.iloc[:, 1] * 100
                except:
                    pass

        # Keltner Channels
        for period in [10, 20]:
            try:
                kc = ta.kc(high, low, close, length=period)
                if kc is not None and len(kc.columns) >= 3:
                    self.indicators[f'kc_upper_{period}'] = kc.iloc[:, 0]
                    self.indicators[f'kc_mid_{period}'] = kc.iloc[:, 1]
                    self.indicators[f'kc_lower_{period}'] = kc.iloc[:, 2]
            except:
                pass

        # Donchian Channels
        for period in [10, 20, 55]:
            try:
                dc = ta.donchian(high, low, lower_length=period, upper_length=period)
                if dc is not None and len(dc.columns) >= 3:
                    self.indicators[f'dc_upper_{period}'] = dc.iloc[:, 0]
                    self.indicators[f'dc_mid_{period}'] = dc.iloc[:, 1]
                    self.indicators[f'dc_lower_{period}'] = dc.iloc[:, 2]
            except:
                pass

        # Historical Volatility
        for period in [10, 20, 30]:
            try:
                returns = close.pct_change()
                hvol = returns.rolling(period).std() * np.sqrt(252) * 100
                self.indicators[f'hvol_{period}'] = hvol
            except:
                pass

        # True Range
        try:
            tr = ta.true_range(high, low, close)
            if tr is not None:
                self.indicators['true_range'] = tr
        except:
            pass

    def _generate_volume_indicators(self, verbose: bool):
        """Generate volume indicators"""
        if verbose:
            print("  → Volume indicators...")

        if 'volume' not in self.df.columns:
            return

        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        volume = self.df['volume']

        # OBV
        try:
            obv = ta.obv(close, volume)
            if obv is not None:
                self.indicators['obv'] = obv
                self.indicators['obv_sma_20'] = obv.rolling(20).mean()
        except:
            pass

        # AD (Accumulation/Distribution)
        try:
            ad = ta.ad(high, low, close, volume)
            if ad is not None:
                self.indicators['ad'] = ad
        except:
            pass

        # CMF (Chaikin Money Flow)
        for period in [10, 20, 21]:
            try:
                cmf = ta.cmf(high, low, close, volume, length=period)
                if cmf is not None:
                    self.indicators[f'cmf_{period}'] = cmf
            except:
                pass

        # MFI (Money Flow Index)
        for period in [10, 14, 20]:
            try:
                mfi = ta.mfi(high, low, close, volume, length=period)
                if mfi is not None:
                    self.indicators[f'mfi_{period}'] = mfi
            except:
                pass

        # VWAP
        try:
            vwap = ta.vwap(high, low, close, volume)
            if vwap is not None:
                self.indicators['vwap'] = vwap
                self.indicators['vwap_dist'] = (close - vwap) / vwap * 100
        except:
            pass

        # Volume SMA ratios
        for period in [10, 20, 50]:
            try:
                vol_sma = volume.rolling(period).mean()
                self.indicators[f'vol_sma_{period}'] = vol_sma
                self.indicators[f'vol_ratio_{period}'] = volume / vol_sma
            except:
                pass

        # Force Index
        try:
            fi = ta.efi(close, volume)
            if fi is not None:
                self.indicators['force_index'] = fi
        except:
            pass

        # Elder Ray
        try:
            elder = ta.eri(high, low, close)
            if elder is not None:
                self.indicators['bull_power'] = elder.iloc[:, 0]
                self.indicators['bear_power'] = elder.iloc[:, 1]
        except:
            pass

    def _generate_moving_averages(self, verbose: bool):
        """Generate multiple moving average types and periods"""
        if verbose:
            print("  → Moving averages...")

        close = self.df['close']

        # SMA
        for period in [5, 10, 20, 50, 100, 200]:
            try:
                sma = ta.sma(close, length=period)
                if sma is not None:
                    self.indicators[f'sma_{period}'] = sma
                    self.indicators[f'sma_{period}_dist'] = (close - sma) / sma * 100
            except:
                pass

        # EMA
        for period in [5, 10, 12, 20, 26, 50, 100, 200]:
            try:
                ema = ta.ema(close, length=period)
                if ema is not None:
                    self.indicators[f'ema_{period}'] = ema
                    self.indicators[f'ema_{period}_dist'] = (close - ema) / ema * 100
            except:
                pass

        # WMA
        for period in [10, 20, 50]:
            try:
                wma = ta.wma(close, length=period)
                if wma is not None:
                    self.indicators[f'wma_{period}'] = wma
            except:
                pass

        # HMA (Hull Moving Average)
        for period in [9, 16, 25]:
            try:
                hma = ta.hma(close, length=period)
                if hma is not None:
                    self.indicators[f'hma_{period}'] = hma
            except:
                pass

        # TEMA (Triple Exponential Moving Average)
        for period in [10, 20]:
            try:
                tema = ta.tema(close, length=period)
                if tema is not None:
                    self.indicators[f'tema_{period}'] = tema
            except:
                pass

        # DEMA (Double Exponential Moving Average)
        for period in [10, 20]:
            try:
                dema = ta.dema(close, length=period)
                if dema is not None:
                    self.indicators[f'dema_{period}'] = dema
            except:
                pass

        # ZLEMA (Zero Lag EMA)
        for period in [10, 20]:
            try:
                zlema = ta.zlma(close, length=period)
                if zlema is not None:
                    self.indicators[f'zlema_{period}'] = zlema
            except:
                pass

        # MA Crossover signals
        ma_pairs = [(5, 20), (10, 50), (20, 50), (50, 200)]
        for fast, slow in ma_pairs:
            if f'ema_{fast}' in self.indicators.columns and f'ema_{slow}' in self.indicators.columns:
                self.indicators[f'ema_cross_{fast}_{slow}'] = (
                    self.indicators[f'ema_{fast}'] - self.indicators[f'ema_{slow}']
                )

    def _generate_custom_indicators(self, verbose: bool):
        """Generate custom composite indicators"""
        if verbose:
            print("  → Custom composite indicators...")

        close = self.df['close']
        high = self.df['high']
        low = self.df['low']

        # Price position within range
        for period in [10, 20, 50]:
            try:
                highest = high.rolling(period).max()
                lowest = low.rolling(period).min()
                self.indicators[f'price_position_{period}'] = (close - lowest) / (highest - lowest + 1e-10)
            except:
                pass

        # Normalized price momentum
        for period in [5, 10, 20]:
            try:
                returns = close.pct_change(period)
                self.indicators[f'norm_mom_{period}'] = returns / returns.rolling(50).std()
            except:
                pass

        # Z-score of price
        for period in [20, 50, 100]:
            try:
                mean = close.rolling(period).mean()
                std = close.rolling(period).std()
                self.indicators[f'zscore_{period}'] = (close - mean) / (std + 1e-10)
            except:
                pass

        # Efficiency Ratio (Kaufman)
        for period in [10, 20]:
            try:
                change = abs(close - close.shift(period))
                volatility = abs(close.diff()).rolling(period).sum()
                self.indicators[f'efficiency_ratio_{period}'] = change / (volatility + 1e-10)
            except:
                pass

        # Trend Strength Index
        try:
            if 'adx_14' in self.indicators.columns and 'rsi_14' in self.indicators.columns:
                self.indicators['trend_strength'] = (
                    self.indicators['adx_14'] *
                    abs(self.indicators['rsi_14'] - 50) / 50
                )
        except:
            pass

        # Composite Momentum
        try:
            mom_cols = [col for col in self.indicators.columns if 'rsi' in col or 'cci' in col or 'mfi' in col]
            if mom_cols:
                # Normalize each to 0-100 scale and average
                normalized = pd.DataFrame()
                for col in mom_cols[:5]:  # Limit to 5
                    min_val = self.indicators[col].min()
                    max_val = self.indicators[col].max()
                    normalized[col] = (self.indicators[col] - min_val) / (max_val - min_val + 1e-10) * 100
                self.indicators['composite_momentum'] = normalized.mean(axis=1)
        except:
            pass

    def _generate_novel_indicators(self, verbose: bool):
        """Generate novel composite oscillators from novel_indicators module"""
        if not NOVEL_INDICATORS_AVAILABLE:
            if verbose:
                print("  → Novel indicators (skipped - module not available)")
            return

        if verbose:
            print("  → Novel composite oscillators...")

        # Prepare DataFrame for novel indicators
        df_for_novel = pd.DataFrame({
            'open': self.df['open'],
            'high': self.df['high'],
            'low': self.df['low'],
            'close': self.df['close'],
            'volume': self.df.get('volume', pd.Series(1, index=self.df.index))
        }, index=self.df.index)

        try:
            # ARWO - Adaptive Regime-Weighted Oscillator
            self.indicators['novel_arwo'] = calculate_arwo(df_for_novel)
        except Exception as e:
            if verbose:
                print(f"    Warning: ARWO failed: {e}")

        try:
            # DCO - Divergence Consensus Oscillator
            self.indicators['novel_dco'] = calculate_dco(df_for_novel)
        except Exception as e:
            if verbose:
                print(f"    Warning: DCO failed: {e}")

        try:
            # VCMO - Volume-Confirmed Momentum Oscillator
            self.indicators['novel_vcmo'] = calculate_vcmo(df_for_novel)
        except Exception as e:
            if verbose:
                print(f"    Warning: VCMO failed: {e}")

        try:
            # ICS - Indicator Convergence Score
            self.indicators['novel_ics'] = calculate_ics(df_for_novel)
        except Exception as e:
            if verbose:
                print(f"    Warning: ICS failed: {e}")

        try:
            # MJI - Momentum Jerk Indicator
            self.indicators['novel_mji'] = calculate_mji(df_for_novel)
        except Exception as e:
            if verbose:
                print(f"    Warning: MJI failed: {e}")

        try:
            # PRF - Percentile Rank Fusion
            self.indicators['novel_prf'] = calculate_prf(df_for_novel)
        except Exception as e:
            if verbose:
                print(f"    Warning: PRF failed: {e}")

        try:
            # EWAF - Entropy-Weighted Adaptive Fusion
            self.indicators['novel_ewaf'] = calculate_ewaf(df_for_novel)
        except Exception as e:
            if verbose:
                print(f"    Warning: EWAF failed: {e}")

        try:
            # KFIF - Kalman-Filtered Indicator Fusion
            kfif_val, kfif_upper, kfif_lower = calculate_kfif(df_for_novel)
            self.indicators['novel_kfif'] = kfif_val
            self.indicators['novel_kfif_width'] = kfif_upper - kfif_lower
        except Exception as e:
            if verbose:
                print(f"    Warning: KFIF failed: {e}")

        # Add novel consensus features
        novel_cols = [c for c in self.indicators.columns if c.startswith('novel_') and not c.endswith('_width')]
        if len(novel_cols) >= 3:
            try:
                self.indicators['novel_consensus'] = self.indicators[novel_cols].mean(axis=1)
                self.indicators['novel_dispersion'] = self.indicators[novel_cols].std(axis=1)
            except:
                pass

    def _generate_slopes_and_derivatives(self, verbose: bool):
        """Generate slopes (1st derivative) and acceleration (2nd derivative) for key indicators"""
        if verbose:
            print("  → Slopes and derivatives...")

        # Key indicators to compute derivatives for
        key_indicators = [
            'rsi_14', 'macd_12_26', 'cci_20', 'adx_14', 'mfi_14',
            'bb_pctb_20_2', 'stoch_k_14', 'willr_14', 'ao', 'uo',
            'composite_momentum', 'zscore_20', 'obv',
            # Novel indicators
            'novel_arwo', 'novel_dco', 'novel_vcmo', 'novel_ics',
            'novel_mji', 'novel_prf', 'novel_ewaf', 'novel_kfif', 'novel_consensus'
        ]

        for ind in key_indicators:
            if ind in self.indicators.columns:
                try:
                    # Slope (1st derivative) - various smoothing
                    for smooth in [1, 3, 5]:
                        if smooth == 1:
                            slope = self.indicators[ind].diff()
                        else:
                            smoothed = self.indicators[ind].rolling(smooth).mean()
                            slope = smoothed.diff()
                        self.indicators[f'{ind}_slope_{smooth}'] = slope

                    # Acceleration (2nd derivative)
                    slope_3 = self.indicators.get(f'{ind}_slope_3')
                    if slope_3 is not None:
                        self.indicators[f'{ind}_accel'] = slope_3.diff()

                    # Slope of slope (another way to measure acceleration)
                    self.indicators[f'{ind}_slope_slope'] = self.indicators[f'{ind}_slope_1'].diff()

                    # Normalized slope (slope as % of value)
                    self.indicators[f'{ind}_slope_norm'] = (
                        self.indicators[f'{ind}_slope_1'] /
                        (abs(self.indicators[ind]) + 1e-10) * 100
                    )
                except:
                    pass

        # Price slopes
        close = self.indicators['close']
        for period in [1, 3, 5, 10]:
            try:
                self.indicators[f'price_slope_{period}'] = close.diff(period) / period
                self.indicators[f'price_slope_pct_{period}'] = close.pct_change(period) * 100
            except:
                pass

        # Price acceleration
        try:
            self.indicators['price_accel'] = self.indicators['price_slope_1'].diff()
        except:
            pass


class ConditionGenerator:
    """Generate trading conditions from indicators"""

    def __init__(self, indicators_df: pd.DataFrame):
        self.indicators = indicators_df
        self.conditions = {}

    def generate_all_conditions(self, verbose: bool = True) -> Dict[str, pd.Series]:
        """Generate all possible conditions"""
        if verbose:
            print("\n🔧 Generating trading conditions...")

        self._generate_threshold_conditions(verbose)
        self._generate_crossover_conditions(verbose)
        self._generate_slope_conditions(verbose)
        self._generate_divergence_conditions(verbose)
        self._generate_combination_conditions(verbose)
        self._generate_pattern_conditions(verbose)

        if verbose:
            print(f"✅ Generated {len(self.conditions)} total conditions")

        return self.conditions

    def _generate_threshold_conditions(self, verbose: bool):
        """Generate threshold-based conditions"""
        if verbose:
            print("  → Threshold conditions...")

        # RSI thresholds
        for period in [7, 14, 21]:
            col = f'rsi_{period}'
            if col in self.indicators.columns:
                for thresh_low, thresh_high in [(20, 80), (25, 75), (30, 70)]:
                    self.conditions[f'{col}_oversold_{thresh_low}'] = self.indicators[col] < thresh_low
                    self.conditions[f'{col}_overbought_{thresh_high}'] = self.indicators[col] > thresh_high
                    self.conditions[f'{col}_neutral'] = (self.indicators[col] >= thresh_low) & (self.indicators[col] <= thresh_high)

        # Stochastic thresholds
        for period in [9, 14]:
            col = f'stoch_k_{period}'
            if col in self.indicators.columns:
                self.conditions[f'{col}_oversold'] = self.indicators[col] < 20
                self.conditions[f'{col}_overbought'] = self.indicators[col] > 80

        # CCI thresholds
        for period in [14, 20]:
            col = f'cci_{period}'
            if col in self.indicators.columns:
                for thresh in [100, 150, 200]:
                    self.conditions[f'{col}_oversold_{thresh}'] = self.indicators[col] < -thresh
                    self.conditions[f'{col}_overbought_{thresh}'] = self.indicators[col] > thresh

        # Williams %R thresholds
        for period in [10, 14]:
            col = f'willr_{period}'
            if col in self.indicators.columns:
                self.conditions[f'{col}_oversold'] = self.indicators[col] < -80
                self.conditions[f'{col}_overbought'] = self.indicators[col] > -20

        # MFI thresholds
        for period in [10, 14]:
            col = f'mfi_{period}'
            if col in self.indicators.columns:
                self.conditions[f'{col}_oversold'] = self.indicators[col] < 20
                self.conditions[f'{col}_overbought'] = self.indicators[col] > 80

        # ADX trend strength
        for period in [14, 20]:
            col = f'adx_{period}'
            if col in self.indicators.columns:
                self.conditions[f'{col}_strong_trend'] = self.indicators[col] > 25
                self.conditions[f'{col}_very_strong_trend'] = self.indicators[col] > 40
                self.conditions[f'{col}_weak_trend'] = self.indicators[col] < 20

        # BB %B thresholds
        for period in [20]:
            for std in [2]:
                col = f'bb_pctb_{period}_{std}'
                if col in self.indicators.columns:
                    self.conditions[f'{col}_below_lower'] = self.indicators[col] < 0
                    self.conditions[f'{col}_above_upper'] = self.indicators[col] > 1
                    self.conditions[f'{col}_lower_zone'] = self.indicators[col] < 0.2
                    self.conditions[f'{col}_upper_zone'] = self.indicators[col] > 0.8

        # Z-score extremes
        for period in [20, 50]:
            col = f'zscore_{period}'
            if col in self.indicators.columns:
                for thresh in [1.5, 2, 2.5]:
                    self.conditions[f'{col}_extreme_low_{thresh}'] = self.indicators[col] < -thresh
                    self.conditions[f'{col}_extreme_high_{thresh}'] = self.indicators[col] > thresh

    def _generate_crossover_conditions(self, verbose: bool):
        """Generate crossover conditions"""
        if verbose:
            print("  → Crossover conditions...")

        # MA crossovers
        ma_pairs = [
            ('ema_5', 'ema_20'),
            ('ema_10', 'ema_50'),
            ('ema_20', 'ema_50'),
            ('ema_50', 'ema_200'),
            ('sma_10', 'sma_50'),
            ('sma_50', 'sma_200'),
            ('hma_9', 'hma_25'),
        ]

        for fast, slow in ma_pairs:
            if fast in self.indicators.columns and slow in self.indicators.columns:
                fast_above = self.indicators[fast] > self.indicators[slow]
                self.conditions[f'{fast}_above_{slow}'] = fast_above
                self.conditions[f'{fast}_cross_above_{slow}'] = fast_above & ~fast_above.shift(1).fillna(False)
                self.conditions[f'{fast}_cross_below_{slow}'] = ~fast_above & fast_above.shift(1).fillna(False)

        # MACD crossovers
        if 'macd_12_26' in self.indicators.columns and 'macd_signal_12_26' in self.indicators.columns:
            macd_above = self.indicators['macd_12_26'] > self.indicators['macd_signal_12_26']
            self.conditions['macd_above_signal'] = macd_above
            self.conditions['macd_cross_above_signal'] = macd_above & ~macd_above.shift(1).fillna(False)
            self.conditions['macd_cross_below_signal'] = ~macd_above & macd_above.shift(1).fillna(False)
            self.conditions['macd_above_zero'] = self.indicators['macd_12_26'] > 0
            self.conditions['macd_hist_positive'] = self.indicators['macd_hist_12_26'] > 0
            self.conditions['macd_hist_increasing'] = self.indicators['macd_hist_12_26'] > self.indicators['macd_hist_12_26'].shift(1)

        # Stochastic crossovers
        for period in [14]:
            k_col = f'stoch_k_{period}'
            d_col = f'stoch_d_{period}'
            if k_col in self.indicators.columns and d_col in self.indicators.columns:
                k_above_d = self.indicators[k_col] > self.indicators[d_col]
                self.conditions[f'stoch_{period}_k_above_d'] = k_above_d
                self.conditions[f'stoch_{period}_k_cross_above_d'] = k_above_d & ~k_above_d.shift(1).fillna(False)
                self.conditions[f'stoch_{period}_k_cross_below_d'] = ~k_above_d & k_above_d.shift(1).fillna(False)

        # Price vs MA
        close = self.indicators['close']
        for ma in ['sma_20', 'sma_50', 'sma_200', 'ema_20', 'ema_50', 'ema_200', 'vwap']:
            if ma in self.indicators.columns:
                above = close > self.indicators[ma]
                self.conditions[f'price_above_{ma}'] = above
                self.conditions[f'price_cross_above_{ma}'] = above & ~above.shift(1).fillna(False)
                self.conditions[f'price_cross_below_{ma}'] = ~above & above.shift(1).fillna(False)

        # DI crossovers
        if 'dmp_14' in self.indicators.columns and 'dmn_14' in self.indicators.columns:
            dmp_above = self.indicators['dmp_14'] > self.indicators['dmn_14']
            self.conditions['dmp_above_dmn'] = dmp_above
            self.conditions['dmp_cross_above_dmn'] = dmp_above & ~dmp_above.shift(1).fillna(False)
            self.conditions['dmp_cross_below_dmn'] = ~dmp_above & dmp_above.shift(1).fillna(False)

    def _generate_slope_conditions(self, verbose: bool):
        """Generate slope-based conditions (key for velocity/acceleration strategy)"""
        if verbose:
            print("  → Slope conditions...")

        # Find all slope columns
        slope_cols = [col for col in self.indicators.columns if '_slope_' in col and '_slope_slope' not in col]
        accel_cols = [col for col in self.indicators.columns if '_accel' in col or '_slope_slope' in col]

        for col in slope_cols:
            if col in self.indicators.columns:
                slope = self.indicators[col]

                # Basic slope direction
                self.conditions[f'{col}_positive'] = slope > 0
                self.conditions[f'{col}_negative'] = slope < 0

                # Slope turning points (zero crossings)
                self.conditions[f'{col}_cross_up'] = (slope > 0) & (slope.shift(1) <= 0)
                self.conditions[f'{col}_cross_down'] = (slope < 0) & (slope.shift(1) >= 0)

                # Strong slope
                slope_std = slope.rolling(50).std()
                self.conditions[f'{col}_strong_positive'] = slope > slope_std
                self.conditions[f'{col}_strong_negative'] = slope < -slope_std

        for col in accel_cols:
            if col in self.indicators.columns:
                accel = self.indicators[col]

                # Acceleration direction
                self.conditions[f'{col}_positive'] = accel > 0
                self.conditions[f'{col}_negative'] = accel < 0

                # Acceleration turning points
                self.conditions[f'{col}_cross_up'] = (accel > 0) & (accel.shift(1) <= 0)
                self.conditions[f'{col}_cross_down'] = (accel < 0) & (accel.shift(1) >= 0)

        # Combined velocity + acceleration conditions (THE KEY INSIGHT)
        key_indicators = ['rsi_14', 'macd_12_26', 'cci_20', 'composite_momentum']
        for ind in key_indicators:
            slope_col = f'{ind}_slope_3'
            accel_col = f'{ind}_accel'

            if slope_col in self.indicators.columns and accel_col in self.indicators.columns:
                slope = self.indicators[slope_col]
                accel = self.indicators[accel_col]

                # Both positive = accelerating upward (SELL signal - nearing top)
                self.conditions[f'{ind}_both_positive'] = (slope > 0) & (accel > 0)

                # Both negative = accelerating downward (BUY signal - nearing bottom)
                self.conditions[f'{ind}_both_negative'] = (slope < 0) & (accel < 0)

                # Positive slope, negative accel = decelerating upward (potential top)
                self.conditions[f'{ind}_decelerating_up'] = (slope > 0) & (accel < 0)

                # Negative slope, positive accel = decelerating downward (potential bottom)
                self.conditions[f'{ind}_decelerating_down'] = (slope < 0) & (accel > 0)

                # Slope turning while acceleration confirms
                self.conditions[f'{ind}_slope_turn_up_accel_pos'] = (
                    (slope > 0) & (slope.shift(1) <= 0) & (accel > 0)
                )
                self.conditions[f'{ind}_slope_turn_down_accel_neg'] = (
                    (slope < 0) & (slope.shift(1) >= 0) & (accel < 0)
                )

    def _generate_divergence_conditions(self, verbose: bool):
        """Generate divergence conditions"""
        if verbose:
            print("  → Divergence conditions...")

        close = self.indicators['close']
        lookback = 14

        # Calculate price highs and lows
        price_high = close.rolling(lookback).max() == close
        price_low = close.rolling(lookback).min() == close

        # Divergence indicators
        div_indicators = ['rsi_14', 'macd_hist_12_26', 'cci_20', 'mfi_14', 'obv']

        for ind in div_indicators:
            if ind in self.indicators.columns:
                indicator = self.indicators[ind]
                ind_high = indicator.rolling(lookback).max() == indicator
                ind_low = indicator.rolling(lookback).min() == indicator

                # Bullish divergence: price makes lower low, indicator makes higher low
                price_lower_low = close < close.shift(lookback)
                ind_higher_low = indicator > indicator.shift(lookback)
                self.conditions[f'{ind}_bullish_div'] = price_lower_low & ind_higher_low

                # Bearish divergence: price makes higher high, indicator makes lower high
                price_higher_high = close > close.shift(lookback)
                ind_lower_high = indicator < indicator.shift(lookback)
                self.conditions[f'{ind}_bearish_div'] = price_higher_high & ind_lower_high

    def _generate_combination_conditions(self, verbose: bool):
        """Generate multi-indicator combination conditions"""
        if verbose:
            print("  → Combination conditions...")

        # Oversold combo: multiple indicators agree
        oversold_conditions = [
            self.conditions.get('rsi_14_oversold_30'),
            self.conditions.get('stoch_k_14_oversold'),
            self.conditions.get('cci_20_oversold_100'),
            self.conditions.get('mfi_14_oversold'),
            self.conditions.get('bb_pctb_20_2_below_lower'),
        ]
        oversold_conditions = [c for c in oversold_conditions if c is not None]

        if len(oversold_conditions) >= 2:
            oversold_sum = sum([c.astype(int) for c in oversold_conditions])
            self.conditions['multi_oversold_2'] = oversold_sum >= 2
            self.conditions['multi_oversold_3'] = oversold_sum >= 3

        # Overbought combo
        overbought_conditions = [
            self.conditions.get('rsi_14_overbought_70'),
            self.conditions.get('stoch_k_14_overbought'),
            self.conditions.get('cci_20_overbought_100'),
            self.conditions.get('mfi_14_overbought'),
            self.conditions.get('bb_pctb_20_2_above_upper'),
        ]
        overbought_conditions = [c for c in overbought_conditions if c is not None]

        if len(overbought_conditions) >= 2:
            overbought_sum = sum([c.astype(int) for c in overbought_conditions])
            self.conditions['multi_overbought_2'] = overbought_sum >= 2
            self.conditions['multi_overbought_3'] = overbought_sum >= 3

        # Trend alignment
        trend_up_conditions = [
            self.conditions.get('price_above_ema_20'),
            self.conditions.get('price_above_ema_50'),
            self.conditions.get('ema_20_above_ema_50'),
            self.conditions.get('macd_above_zero'),
            self.conditions.get('adx_14_strong_trend'),
        ]
        trend_up_conditions = [c for c in trend_up_conditions if c is not None]

        if len(trend_up_conditions) >= 2:
            trend_sum = sum([c.astype(int) for c in trend_up_conditions])
            self.conditions['trend_aligned_bullish'] = trend_sum >= 3
            self.conditions['trend_aligned_bearish'] = trend_sum <= 1

        # Momentum + Trend combo
        if 'rsi_14_oversold_30' in self.conditions and 'price_above_ema_50' in self.conditions:
            self.conditions['oversold_in_uptrend'] = (
                self.conditions['rsi_14_oversold_30'] &
                self.conditions['price_above_ema_50']
            )

        if 'rsi_14_overbought_70' in self.conditions and 'price_above_ema_50' in self.conditions:
            self.conditions['overbought_in_downtrend'] = (
                self.conditions['rsi_14_overbought_70'] &
                ~self.conditions['price_above_ema_50']
            )

        # Velocity consensus
        vel_conditions_buy = []
        vel_conditions_sell = []
        for ind in ['rsi_14', 'macd_12_26', 'cci_20']:
            buy_cond = self.conditions.get(f'{ind}_decelerating_down')
            sell_cond = self.conditions.get(f'{ind}_decelerating_up')
            if buy_cond is not None:
                vel_conditions_buy.append(buy_cond)
            if sell_cond is not None:
                vel_conditions_sell.append(sell_cond)

        if len(vel_conditions_buy) >= 2:
            vel_buy_sum = sum([c.astype(int) for c in vel_conditions_buy])
            self.conditions['velocity_consensus_buy'] = vel_buy_sum >= 2

        if len(vel_conditions_sell) >= 2:
            vel_sell_sum = sum([c.astype(int) for c in vel_conditions_sell])
            self.conditions['velocity_consensus_sell'] = vel_sell_sum >= 2

    def _generate_pattern_conditions(self, verbose: bool):
        """Generate pattern-based conditions"""
        if verbose:
            print("  → Pattern conditions...")

        close = self.indicators['close']
        high = self.indicators['high']
        low = self.indicators['low']

        # Higher highs / lower lows
        for lookback in [3, 5]:
            self.conditions[f'higher_high_{lookback}'] = high > high.shift(lookback)
            self.conditions[f'lower_low_{lookback}'] = low < low.shift(lookback)
            self.conditions[f'higher_low_{lookback}'] = low > low.shift(lookback)
            self.conditions[f'lower_high_{lookback}'] = high < high.shift(lookback)

        # Consecutive up/down days
        up_day = close > close.shift(1)
        down_day = close < close.shift(1)

        for n in [2, 3, 4]:
            self.conditions[f'consecutive_up_{n}'] = up_day.rolling(n).sum() == n
            self.conditions[f'consecutive_down_{n}'] = down_day.rolling(n).sum() == n

        # Range expansion/contraction
        tr = high - low
        tr_sma = tr.rolling(14).mean()
        self.conditions['range_expansion'] = tr > tr_sma * 1.5
        self.conditions['range_contraction'] = tr < tr_sma * 0.5

        # Inside bar
        self.conditions['inside_bar'] = (high < high.shift(1)) & (low > low.shift(1))

        # Outside bar
        self.conditions['outside_bar'] = (high > high.shift(1)) & (low < low.shift(1))


class StrategyBacktester:
    """Backtest trading strategies"""

    def __init__(self, data: pd.DataFrame, indicators: pd.DataFrame, conditions: Dict[str, pd.Series]):
        self.data = data
        self.indicators = indicators
        self.conditions = conditions
        self.conditions_df = pd.DataFrame(conditions)

    def backtest_strategy(self,
                         entry_long_conditions: List[str],
                         exit_long_conditions: List[str],
                         entry_short_conditions: List[str] = None,
                         exit_short_conditions: List[str] = None,
                         filter_conditions: List[str] = None,
                         initial_capital: float = 100000,
                         position_size: float = 1.0) -> StrategyResult:
        """Backtest a strategy defined by condition combinations"""

        # Build entry/exit signals
        entry_long = self._combine_conditions(entry_long_conditions, 'AND')
        exit_long = self._combine_conditions(exit_long_conditions, 'OR')

        if entry_short_conditions:
            entry_short = self._combine_conditions(entry_short_conditions, 'AND')
        else:
            entry_short = pd.Series(False, index=self.data.index)

        if exit_short_conditions:
            exit_short = self._combine_conditions(exit_short_conditions, 'OR')
        else:
            exit_short = pd.Series(False, index=self.data.index)

        # Apply filters
        if filter_conditions:
            filter_signal = self._combine_conditions(filter_conditions, 'AND')
            entry_long = entry_long & filter_signal
            entry_short = entry_short & filter_signal

        # Run simulation
        capital = initial_capital
        position = None  # None, 'long', or 'short'
        entry_price = 0
        trades = []
        equity_curve = [capital]

        close = self.data['close'].values

        for i in range(1, len(self.data)):
            current_price = close[i]

            # Check exits first
            if position == 'long' and exit_long.iloc[i]:
                pnl = (current_price - entry_price) / entry_price
                capital *= (1 + pnl * position_size)
                trades.append({
                    'type': 'long',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'entry_idx': entry_idx,
                    'exit_idx': i
                })
                position = None

            elif position == 'short' and exit_short.iloc[i]:
                pnl = (entry_price - current_price) / entry_price
                capital *= (1 + pnl * position_size)
                trades.append({
                    'type': 'short',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'entry_idx': entry_idx,
                    'exit_idx': i
                })
                position = None

            # Check entries
            if position is None:
                if entry_long.iloc[i]:
                    position = 'long'
                    entry_price = current_price
                    entry_idx = i
                elif entry_short.iloc[i]:
                    position = 'short'
                    entry_price = current_price
                    entry_idx = i

            equity_curve.append(capital)

        # Close any open position
        if position == 'long':
            pnl = (close[-1] - entry_price) / entry_price
            capital *= (1 + pnl * position_size)
            trades.append({
                'type': 'long',
                'entry_price': entry_price,
                'exit_price': close[-1],
                'pnl': pnl,
                'entry_idx': entry_idx,
                'exit_idx': len(self.data) - 1
            })
        elif position == 'short':
            pnl = (entry_price - close[-1]) / entry_price
            capital *= (1 + pnl * position_size)
            trades.append({
                'type': 'short',
                'entry_price': entry_price,
                'exit_price': close[-1],
                'pnl': pnl,
                'entry_idx': entry_idx,
                'exit_idx': len(self.data) - 1
            })

        # Calculate buy & hold return (baseline)
        buy_hold_return = (close[-1] - close[0]) / close[0]

        # Calculate metrics
        return self._calculate_metrics(
            trades, equity_curve, initial_capital,
            entry_long_conditions, exit_long_conditions,
            entry_short_conditions, exit_short_conditions,
            filter_conditions,
            buy_hold_return
        )

    def _combine_conditions(self, condition_names: List[str], logic: str = 'AND') -> pd.Series:
        """Combine multiple conditions with AND/OR logic"""
        if not condition_names:
            return pd.Series(True, index=self.data.index)

        valid_conditions = []
        for name in condition_names:
            if name in self.conditions:
                valid_conditions.append(self.conditions[name])

        if not valid_conditions:
            return pd.Series(False, index=self.data.index)

        if logic == 'AND':
            result = valid_conditions[0]
            for c in valid_conditions[1:]:
                result = result & c
        else:  # OR
            result = valid_conditions[0]
            for c in valid_conditions[1:]:
                result = result | c

        return result.fillna(False)

    def _calculate_metrics(self, trades, equity_curve, initial_capital,
                          entry_long, exit_long, entry_short, exit_short, filters,
                          buy_hold_return: float = 0.0) -> StrategyResult:
        """Calculate strategy performance metrics"""

        if len(trades) == 0:
            return StrategyResult(
                strategy_name="",
                total_return=0,
                sharpe_ratio=0,
                win_rate=0,
                num_trades=0,
                profit_factor=0,
                max_drawdown=0,
                avg_trade=0,
                rules_description="No trades",
                strategy_config={},
                buy_hold_return=buy_hold_return,
                outperformance=-buy_hold_return  # 0 return vs buy & hold
            )

        # Returns
        pnls = [t['pnl'] for t in trades]
        total_return = (equity_curve[-1] - initial_capital) / initial_capital

        # Win rate
        winners = sum(1 for p in pnls if p > 0)
        win_rate = winners / len(pnls)

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

        # Sharpe ratio
        if len(pnls) > 1:
            sharpe = np.sqrt(252 / len(pnls)) * (np.mean(pnls) / (np.std(pnls) + 1e-10))
        else:
            sharpe = 0

        # Max drawdown
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)

        # Average trade
        avg_trade = np.mean(pnls)

        # Build description
        rules_desc = f"LONG: {entry_long} | EXIT: {exit_long}"
        if entry_short:
            rules_desc += f" | SHORT: {entry_short} | EXIT: {exit_short}"
        if filters:
            rules_desc += f" | FILTER: {filters}"

        return StrategyResult(
            strategy_name=f"Strategy_{len(trades)}trades",
            total_return=total_return,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            num_trades=len(trades),
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            avg_trade=avg_trade,
            rules_description=rules_desc,
            strategy_config={
                'entry_long': entry_long,
                'exit_long': exit_long,
                'entry_short': entry_short,
                'exit_short': exit_short,
                'filters': filters
            },
            buy_hold_return=buy_hold_return,
            outperformance=total_return - buy_hold_return
        )


class StrategyDiscoveryEngine:
    """Main engine for discovering winning strategies"""

    def __init__(self, data: pd.DataFrame, output_dir: str = ANALYSIS_DIR):
        self.data = data
        self.output_dir = output_dir
        self.indicators = None
        self.conditions = None
        self.results = []
        self.progress_callback = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def _update_progress(self, step: int, total_steps: int, message: str, detail: str = ""):
        """Update progress via callback if set"""
        if self.progress_callback:
            self.progress_callback(step, total_steps, message, detail)
        print(message)
        if detail:
            print(f"  {detail}")

    def run_discovery(self,
                     max_strategies: int = 1000,
                     n_jobs: int = N_JOBS,
                     verbose: bool = True,
                     progress_callback: callable = None,
                     sort_by: str = 'total_return') -> List[StrategyResult]:
        """Run the full discovery process

        Args:
            max_strategies: Maximum number of strategy combinations to test
            n_jobs: Number of parallel workers
            verbose: Print progress to console
            progress_callback: Optional callback function(step, total_steps, message, detail)
            sort_by: How to sort results ('total_return', 'sharpe_ratio', 'win_rate', 'profit_factor', 'composite')
        """
        self.sort_by = sort_by
        self.progress_callback = progress_callback
        total_steps = 6

        self._update_progress(0, total_steps, "🚀 STRATEGY DISCOVERY ENGINE", "Initializing...")

        # Step 1: Generate indicators
        self._update_progress(1, total_steps, "📊 STEP 1/6: Generating Indicators",
                            "Creating 100+ technical indicators using pandas_ta...")
        indicator_gen = IndicatorGenerator(self.data)
        self.indicators = indicator_gen.generate_all(verbose)
        self._update_progress(1, total_steps, "📊 STEP 1/6: Generating Indicators",
                            f"✅ Generated {len(self.indicators.columns)} indicators")

        # Save indicators
        self._save_indicators()

        # Step 2: Generate conditions
        self._update_progress(2, total_steps, "🔧 STEP 2/6: Generating Conditions",
                            "Creating trading conditions (slopes, crossovers, divergences)...")
        condition_gen = ConditionGenerator(self.indicators)
        self.conditions = condition_gen.generate_all_conditions(verbose)
        self._update_progress(2, total_steps, "🔧 STEP 2/6: Generating Conditions",
                            f"✅ Generated {len(self.conditions)} conditions")

        # Save conditions
        self._save_conditions()

        # Step 3: Generate strategy combinations
        self._update_progress(3, total_steps, "🎯 STEP 3/6: Generating Strategy Combinations",
                            f"Creating up to {max_strategies} strategy combinations...")
        strategies = self._generate_strategy_combinations(max_strategies, verbose)
        self._update_progress(3, total_steps, "🎯 STEP 3/6: Generating Strategy Combinations",
                            f"✅ Generated {len(strategies)} strategies to test")

        # Step 4: Backtest all strategies
        self._update_progress(4, total_steps, f"⚡ STEP 4/6: Backtesting {len(strategies)} Strategies",
                            f"Running parallel backtests with {n_jobs} workers. This may take a few minutes...")
        backtester = StrategyBacktester(self.data, self.indicators, self.conditions)

        # For Streamlit, use sequential with progress updates if callback is set
        # Otherwise use parallel for console usage
        if self.progress_callback:
            # Sequential execution with progress updates for Streamlit
            self.results = []
            best_return = 0
            for i, strategy in enumerate(strategies):
                result = self._backtest_single(backtester, strategy)
                self.results.append(result)

                if result and result.total_return > best_return:
                    best_return = result.total_return

                # Update progress every 25 strategies
                if (i + 1) % 25 == 0 or (i + 1) == len(strategies):
                    pct = (i + 1) / len(strategies) * 100
                    self._update_progress(4, total_steps,
                        f"⚡ STEP 4/6: Backtesting Strategies",
                        f"Progress: {i+1}/{len(strategies)} ({pct:.0f}%) | Best return: {best_return:.1%}")
        else:
            # Parallel execution for console usage (faster but no progress)
            self.results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(self._backtest_single)(backtester, s) for s in strategies
            )

        # Filter out None results
        self.results = [r for r in self.results if r is not None and r.num_trades > 0]
        self._update_progress(4, total_steps, "⚡ STEP 4/6: Backtesting Complete",
                            f"✅ {len(self.results)} strategies produced valid results")

        # Step 5: Rank and save results
        self._update_progress(5, total_steps, "🏆 STEP 5/6: Ranking Strategies",
                            f"Sorting by {self.sort_by} and saving results...")
        self._rank_and_save_results(sort_by=self.sort_by)

        if self.results:
            top_return = self.results[0].total_return if self.results else 0
            self._update_progress(5, total_steps, "🏆 STEP 5/6: Ranking Complete",
                                f"✅ Top strategy: {top_return:.1%} return")

        # Step 6: ML feature importance
        self._update_progress(6, total_steps, "🤖 STEP 6/6: ML Feature Importance Analysis",
                            "Training RandomForest to identify predictive features...")
        self._run_ml_analysis()
        self._update_progress(6, total_steps, "🤖 STEP 6/6: ML Analysis Complete",
                            "✅ Feature importance saved to CSV")

        self._update_progress(6, total_steps, "🎉 DISCOVERY COMPLETE!",
                            f"Found {len(self.results)} valid strategies. Top return: {self.results[0].total_return:.1%}" if self.results else "No valid strategies found")

        return self.results

    def _generate_strategy_combinations(self, max_strategies: int, verbose: bool) -> List[Dict]:
        """Generate strategy combinations to test"""
        strategies = []

        # Get condition names by category
        entry_long_candidates = []
        entry_short_candidates = []
        exit_candidates = []
        filter_candidates = []

        for name in self.conditions.keys():
            # Classify conditions
            if any(x in name for x in ['oversold', 'cross_up', 'cross_above', 'bullish',
                                        'decelerating_down', 'both_negative', 'slope_turn_up',
                                        'lower_low', 'velocity_consensus_buy']):
                entry_long_candidates.append(name)
                exit_candidates.append(name)  # Can also be short exit

            if any(x in name for x in ['overbought', 'cross_down', 'cross_below', 'bearish',
                                        'decelerating_up', 'both_positive', 'slope_turn_down',
                                        'higher_high', 'velocity_consensus_sell']):
                entry_short_candidates.append(name)
                exit_candidates.append(name)  # Can also be long exit

            if any(x in name for x in ['strong_trend', 'trend_aligned', 'above_zero']):
                filter_candidates.append(name)

        if verbose:
            print(f"  Entry Long candidates: {len(entry_long_candidates)}")
            print(f"  Entry Short candidates: {len(entry_short_candidates)}")
            print(f"  Exit candidates: {len(exit_candidates)}")
            print(f"  Filter candidates: {len(filter_candidates)}")

        # Generate combinations
        # Single entry + single exit
        for entry in entry_long_candidates[:50]:  # Limit to prevent explosion
            for exit_cond in exit_candidates[:50]:
                if entry != exit_cond:
                    strategies.append({
                        'entry_long': [entry],
                        'exit_long': [exit_cond],
                        'entry_short': None,
                        'exit_short': None,
                        'filters': None
                    })

        # Double entry conditions (more selective)
        for entry1, entry2 in combinations(entry_long_candidates[:30], 2):
            for exit_cond in exit_candidates[:20]:
                strategies.append({
                    'entry_long': [entry1, entry2],
                    'exit_long': [exit_cond],
                    'entry_short': None,
                    'exit_short': None,
                    'filters': None
                })

        # With filters
        for entry in entry_long_candidates[:30]:
            for exit_cond in exit_candidates[:20]:
                for filt in filter_candidates[:10]:
                    strategies.append({
                        'entry_long': [entry],
                        'exit_long': [exit_cond],
                        'entry_short': None,
                        'exit_short': None,
                        'filters': [filt]
                    })

        # Long + Short strategies
        for entry_l in entry_long_candidates[:20]:
            for exit_l in entry_short_candidates[:10]:  # Use short entry as long exit
                for entry_s in entry_short_candidates[:20]:
                    for exit_s in entry_long_candidates[:10]:  # Use long entry as short exit
                        strategies.append({
                            'entry_long': [entry_l],
                            'exit_long': [exit_l],
                            'entry_short': [entry_s],
                            'exit_short': [exit_s],
                            'filters': None
                        })

        # Velocity-focused strategies (THE KEY ONES)
        velocity_entries = [c for c in entry_long_candidates if 'velocity' in c or 'decel' in c or 'slope' in c]
        velocity_exits = [c for c in exit_candidates if 'velocity' in c or 'decel' in c or 'slope' in c]

        for entry in velocity_entries:
            for exit_cond in velocity_exits:
                if entry != exit_cond:
                    strategies.append({
                        'entry_long': [entry],
                        'exit_long': [exit_cond],
                        'entry_short': None,
                        'exit_short': None,
                        'filters': None
                    })

        # Velocity + threshold combo
        for vel_entry in velocity_entries[:20]:
            for thresh in [c for c in entry_long_candidates if 'oversold' in c][:10]:
                for vel_exit in velocity_exits[:10]:
                    strategies.append({
                        'entry_long': [vel_entry, thresh],
                        'exit_long': [vel_exit],
                        'entry_short': None,
                        'exit_short': None,
                        'filters': None
                    })

        # Shuffle and limit
        np.random.shuffle(strategies)
        strategies = strategies[:max_strategies]

        if verbose:
            print(f"  Generated {len(strategies)} strategy combinations to test")

        return strategies

    def _backtest_single(self, backtester: StrategyBacktester, strategy: Dict) -> Optional[StrategyResult]:
        """Backtest a single strategy (for parallel execution)"""
        try:
            return backtester.backtest_strategy(
                entry_long_conditions=strategy['entry_long'],
                exit_long_conditions=strategy['exit_long'],
                entry_short_conditions=strategy.get('entry_short'),
                exit_short_conditions=strategy.get('exit_short'),
                filter_conditions=strategy.get('filters')
            )
        except Exception as e:
            return None

    def _rank_and_save_results(self, sort_by: str = 'total_return'):
        """Rank strategies and save to CSV

        Args:
            sort_by: How to sort results. Options:
                - 'total_return': Sort by total return (highest first)
                - 'sharpe_ratio': Sort by Sharpe ratio (highest first)
                - 'win_rate': Sort by win rate (highest first)
                - 'composite': Sort by composite score (balanced)
        """

        # Define sorting functions
        def composite_score(r: StrategyResult) -> float:
            if r.num_trades < 5:  # Minimum trades
                return -1000
            score = 0
            score += r.total_return * 100  # Return
            score += r.sharpe_ratio * 20  # Risk-adjusted
            score += r.win_rate * 30  # Consistency
            score += r.profit_factor * 10  # Edge
            score -= r.max_drawdown * 50  # Risk
            return score

        # Sort based on selected method
        if sort_by == 'total_return':
            self.results.sort(key=lambda r: r.total_return if r.num_trades >= 5 else -1000, reverse=True)
        elif sort_by == 'sharpe_ratio':
            self.results.sort(key=lambda r: r.sharpe_ratio if r.num_trades >= 5 else -1000, reverse=True)
        elif sort_by == 'win_rate':
            self.results.sort(key=lambda r: r.win_rate if r.num_trades >= 5 else -1000, reverse=True)
        elif sort_by == 'profit_factor':
            self.results.sort(key=lambda r: r.profit_factor if r.num_trades >= 5 else -1000, reverse=True)
        else:  # composite
            self.results.sort(key=composite_score, reverse=True)

        # Save top strategies
        top_results = self.results[:100]

        results_df = pd.DataFrame([
            {
                'rank': i + 1,
                'total_return': r.total_return,
                'sharpe_ratio': r.sharpe_ratio,
                'win_rate': r.win_rate,
                'num_trades': r.num_trades,
                'profit_factor': r.profit_factor,
                'max_drawdown': r.max_drawdown,
                'avg_trade': r.avg_trade,
                'composite_score': composite_score(r),
                'entry_long': str(r.strategy_config.get('entry_long', [])),
                'exit_long': str(r.strategy_config.get('exit_long', [])),
                'entry_short': str(r.strategy_config.get('entry_short', [])),
                'exit_short': str(r.strategy_config.get('exit_short', [])),
                'filters': str(r.strategy_config.get('filters', []))
            }
            for i, r in enumerate(top_results)
        ])

        results_df.to_csv(f"{self.output_dir}/top_strategies.csv", index=False)

        # Also save as JSON for detailed analysis
        with open(f"{self.output_dir}/top_strategies.json", 'w') as f:
            json.dump([
                {
                    'rank': i + 1,
                    'total_return': r.total_return,
                    'sharpe_ratio': r.sharpe_ratio,
                    'win_rate': r.win_rate,
                    'num_trades': r.num_trades,
                    'profit_factor': r.profit_factor,
                    'max_drawdown': r.max_drawdown,
                    'avg_trade': r.avg_trade,
                    'rules': r.rules_description,
                    'config': r.strategy_config
                }
                for i, r in enumerate(top_results[:20])
            ], f, indent=2, default=str)

        print(f"\n📁 Saved top {len(top_results)} strategies to {self.output_dir}/")

        # Print top 10
        print("\n" + "="*60)
        print("🏆 TOP 10 STRATEGIES")
        print("="*60)

        for i, r in enumerate(top_results[:10]):
            print(f"\n#{i+1}: Return={r.total_return:.1%} | Sharpe={r.sharpe_ratio:.2f} | "
                  f"WinRate={r.win_rate:.0%} | Trades={r.num_trades}")
            print(f"    Entry: {r.strategy_config.get('entry_long', [])}")
            print(f"    Exit:  {r.strategy_config.get('exit_long', [])}")

    def _save_indicators(self):
        """Save indicators to CSV"""
        self.indicators.to_csv(f"{self.output_dir}/indicators_raw.csv")
        print(f"  Saved indicators to {self.output_dir}/indicators_raw.csv")

    def _save_conditions(self):
        """Save conditions to CSV"""
        conditions_df = pd.DataFrame(self.conditions)
        conditions_df.to_csv(f"{self.output_dir}/conditions_signals.csv")
        print(f"  Saved conditions to {self.output_dir}/conditions_signals.csv")

        # Also save condition summary
        condition_summary = pd.DataFrame({
            'condition': list(self.conditions.keys()),
            'true_count': [c.sum() for c in self.conditions.values()],
            'true_pct': [c.mean() * 100 for c in self.conditions.values()]
        })
        condition_summary = condition_summary.sort_values('true_pct', ascending=False)
        condition_summary.to_csv(f"{self.output_dir}/conditions_summary.csv", index=False)

    def _run_ml_analysis(self):
        """Run ML feature importance analysis"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler

            # Create target: 1 if price goes up next N bars
            future_return = self.data['close'].pct_change(5).shift(-5)
            target = (future_return > 0).astype(int)

            # Prepare features
            features = self.indicators.copy()
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(0)

            # Align
            common_idx = features.index.intersection(target.dropna().index)
            X = features.loc[common_idx]
            y = target.loc[common_idx]

            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train random forest
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
            rf.fit(X_scaled, y)

            # Feature importance
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)

            importance_df.to_csv(f"{self.output_dir}/feature_importance.csv", index=False)

            print(f"\n📊 Top 20 Most Important Features:")
            for i, row in importance_df.head(20).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

        except Exception as e:
            print(f"  ML analysis failed: {e}")


def run_full_discovery(data: pd.DataFrame,
                      max_strategies: int = 1000,
                      output_dir: str = ANALYSIS_DIR) -> List[StrategyResult]:
    """Convenience function to run full discovery"""
    engine = StrategyDiscoveryEngine(data, output_dir)
    return engine.run_discovery(max_strategies=max_strategies)


if __name__ == "__main__":
    print("Strategy Discovery Engine loaded.")
    print(f"Using {N_JOBS} parallel workers.")
    print(f"Output directory: {ANALYSIS_DIR}")
