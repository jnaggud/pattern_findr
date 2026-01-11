"""
Price Prediction Module for Velocity Trading System

Provides three core prediction capabilities:
1. PriceRangePredictor - Predicts daily high/low range
2. PriceTargetCalculator - Calculates optimal take-profit levels
3. ExitTimingPredictor - Predicts optimal exit timing

Uses: XGBoost regression, options data (Polygon), technical indicators, VIX data
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("Warning: LightGBM not available, ensemble will use XGBoost only")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# Try to import novel indicators
try:
    from novel_indicators import (
        calculate_arwo, calculate_dco, calculate_vcmo,
        calculate_ics, calculate_mji, calculate_prf, calculate_ewaf, calculate_kfif
    )
    NOVEL_INDICATORS_AVAILABLE = True
except ImportError:
    NOVEL_INDICATORS_AVAILABLE = False
    print("Warning: novel_indicators not available")

# Try to import MAPIE for conformal prediction
# Note: MAPIE 1.2.0+ renamed MapieRegressor to SplitConformalRegressor
try:
    from mapie.regression import SplitConformalRegressor as MapieRegressor
    MAPIE_AVAILABLE = True
except ImportError:
    try:
        # Fallback for older versions
        from mapie.regression import MapieRegressor
        MAPIE_AVAILABLE = True
    except ImportError:
        MAPIE_AVAILABLE = False
        print("Warning: mapie not available, conformal prediction disabled")


# Import optuna_worker for proper multiprocessing (file-based data sharing)
try:
    import optuna_worker
    OPTUNA_WORKER_AVAILABLE = True
except ImportError:
    OPTUNA_WORKER_AVAILABLE = False
    print("Warning: optuna_worker not available, parallel optimization may be slow")

# Import the main composite oscillator (source of truth - 15 components)
try:
    from oscillator_predictor_page import create_composite_oscillator
    MAIN_COMPOSITE_AVAILABLE = True
except ImportError:
    MAIN_COMPOSITE_AVAILABLE = False
    print("Warning: create_composite_oscillator not available, using simplified version")

# Import Deep Learning Feature Extractor
try:
    from dl_feature_extractor import DLFeatureExtractor
    DL_FEATURE_AVAILABLE = True
except ImportError:
    DL_FEATURE_AVAILABLE = False
    print("Warning: DLFeatureExtractor not available, deep learning features disabled")


# =============================================================================
# FEATURE GROUP CONFIGURATION
# Toggle feature groups on/off to optimize performance and reduce noise
# Based on feature importance analysis, these groups contribute <5% combined
# =============================================================================
FEATURE_CONFIG = {
    # Core features (HIGH importance - always enabled)
    'atr_features': True,           # ATR 7/14/21, ratios - TOP predictors
    'range_features': True,         # Daily range, percentiles - core target
    'volatility_features': True,    # Realized vol, BB width - HIGH importance
    'vix_features': True,           # VIX and derivatives - HIGH importance

    # Medium importance features
    'volume_features': True,        # Volume velocity, momentum - MEDIUM importance
    'trend_features': True,         # Trend direction, strength - MEDIUM importance
    'composite_osc': True,          # Main 15-component oscillator - MEDIUM importance

    # Low importance features (can disable to improve speed)
    'temporal_features': True,      # Day of week, month - LOW importance (<2%)
    'options_features': True,       # PCR, IV, max pain - LOW when no real data
    'lag_features': False,          # ATR/range lags - LOW importance, adds ~20 features
    'interaction_features': False,  # uptrend_atr, downtrend_volume - LOW importance
    'scientific_features': False,   # Hurst exponent - VERY LOW, slow to compute
    'novel_indicators': False,      # ARWO, DCO, etc - disabled by default

    # Deep Learning Features (CNN+LSTM embeddings)
    'dl_features': True,            # DL embeddings - captures temporal patterns XGBoost may miss

    # Feature Pruning - remove features with zero importance from analysis
    'prune_zero_importance': True,  # Remove features that never split in any tree

    # NEW: Cross-Market Features (for enhanced range prediction)
    # Set USE_NEW_CROSS_MARKET_FEATURES = False for baseline, True for test
    'vvix_features': True,          # VVIX (VIX of VIX) - volatility of volatility
    'sector_breadth_features': True, # Sector ETFs breadth - market internal health
    'credit_spread_features': True,  # HYG/LQD credit spreads - risk sentiment
    'treasury_features': True,       # Treasury yields 10Y/30Y - macro context
    'dollar_features': True,         # Dollar index (DXY) - currency context
    'correlation_features': True,    # SPY/QQQ correlation - market structure
    'earnings_features': True,       # Earnings calendar for top SPY holdings
}

# TOGGLE: Set to False for baseline (old features only), True for test (with new features)
USE_NEW_CROSS_MARKET_FEATURES = True

# =============================================================================
# ZERO-IMPORTANCE FEATURES TO PRUNE
# These features had exactly 0.0 importance in 60 walk-forward iterations
# Most are options-related (no real data) or regime-based (constant values)
# =============================================================================
ZERO_IMPORTANCE_FEATURES = {
    # Options features with no real data (all defaults)
    'pcr_volume', 'pcr_oi', 'sentiment_bullish', 'sentiment_bearish',
    'iv_skew', 'max_pain_distance', 'high_call_oi_distance', 'high_put_oi_distance',
    'atm_skew', 'atm_iv_rv_spread', 'gamma_long', 'gamma_short',
    'net_gamma_normalized', 'net_delta_normalized',
    'unusual_activity_bullish', 'unusual_activity_bearish', 'unusual_activity_count',
    'term_structure_slope', 'term_structure_backwardation', 'term_structure_contango',
    'near_term_iv', 'far_term_iv', 'iv_term_spread',

    # Regime features that don't provide good split points
    'is_downtrend', 'is_uptrend', 'vix_regime_high', 'vix_regime_low',
    'range_momentum_high_vol', 'range_momentum_double_high_vol',
    'range_momentum_high_vol_lag1', 'range_mean_3d_high_vol', 'range_accel_high_vol',
    'upside_potential_high_vol', 'upside_potential_double_high_vol',
    'upside_potential_x_sma50_high_vol', 'momentum_trend_divergence_high_vol',
    'downside_potential_high_vol', 'downside_momentum_high_vol', 'vix_accel_high_vol',

    # Very low importance binary/temporal
    'volume_spike', 'earnings_week',
}

def set_cross_market_features(enabled: bool):
    """Toggle cross-market features on/off for A/B testing."""
    global USE_NEW_CROSS_MARKET_FEATURES
    USE_NEW_CROSS_MARKET_FEATURES = enabled
    FEATURE_CONFIG['vvix_features'] = enabled
    FEATURE_CONFIG['sector_breadth_features'] = enabled
    FEATURE_CONFIG['credit_spread_features'] = enabled
    FEATURE_CONFIG['treasury_features'] = enabled
    FEATURE_CONFIG['dollar_features'] = enabled
    FEATURE_CONFIG['correlation_features'] = enabled
    FEATURE_CONFIG['earnings_features'] = enabled
    print(f"Cross-market features {'ENABLED' if enabled else 'DISABLED'}")


# =============================================================================
# TRAINING SUMMARY - Collects key metrics for compact end-of-run output
# =============================================================================
class TrainingSummary:
    """Collects training metrics and prints a compact summary at the end."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all collected data for a new run."""
        self.start_time = datetime.now()
        self.data = {
            # Data info
            'total_rows': 0,
            'training_samples': 0,
            'test_samples': 0,
            'features_created': 0,
            'features_pruned_zero_imp': 0,
            'features_pruned_corr': 0,
            'features_used': 0,
            'feature_sample_ratio': 0.0,

            # Feature health checks
            'iv_fallback_pct': 0.0,  # % of IV features that are fallback values
            'dxy_value': 0.0,  # DXY value (should be ~100-110)
            'vix_value': 0.0,
            'dl_samples': 0,  # Samples used for DL training

            # Optimization results
            'range_best_score': 0.0,
            'range_trials': 0,
            'range_all_scores': [],  # All worker scores
            'high_best_score': 0.0,
            'high_all_scores': [],
            'low_best_score': 0.0,
            'low_all_scores': [],
            'low_negative_count': 0,  # Workers with negative R²

            # Final metrics
            'test_r2': 0.0,
            'test_rmse': 0.0,
            'ensemble_size': 0,
            'low_ensemble_size': 0,

            # Model details
            'best_params': {},
            'quantile_levels': [],

            # Top features
            'top_features_corr': [],
            'top_features_importance': [],
            'nonzero_importance_count': 0,

            # Warnings
            'warnings': [],

            # Timing
            'range_opt_time': 0.0,
            'highlow_opt_time': 0.0,
            'total_time': 0.0,
        }

    def add_warning(self, msg: str):
        """Add a warning to track."""
        if msg not in self.data['warnings']:
            self.data['warnings'].append(msg)

    def check_issues(self):
        """Check for common issues and add warnings automatically."""
        d = self.data

        # Feature/sample ratio check
        if d['training_samples'] > 0 and d['features_used'] > 0:
            ratio = d['features_used'] / d['training_samples']
            d['feature_sample_ratio'] = ratio
            if ratio > 0.5:
                self.add_warning(f"HIGH OVERFIT RISK: {ratio:.1f} features/sample (want <0.1)")
            elif ratio > 0.2:
                self.add_warning(f"Moderate overfit risk: {ratio:.2f} features/sample")

        # Negative LOW scores
        if d['low_all_scores']:
            d['low_negative_count'] = sum(1 for s in d['low_all_scores'] if s < 0)
            if d['low_negative_count'] > 0:
                self.add_warning(f"LOW model unstable: {d['low_negative_count']}/{len(d['low_all_scores'])} workers had negative R²")

        # Large optimization vs test gap
        if d['range_best_score'] > 0 and d['test_r2'] > 0:
            gap = (d['range_best_score'] - d['test_r2']) / d['range_best_score']
            if gap > 0.3:
                self.add_warning(f"Overfit detected: {gap*100:.0f}% drop from optimization to test")

        # Very low training samples
        if d['training_samples'] < 100:
            self.add_warning(f"Low training samples: {d['training_samples']} (recommend 200+)")

        # IV fallback check
        if d['iv_fallback_pct'] > 50:
            self.add_warning(f"IV features are {d['iv_fallback_pct']:.0f}% fallback (no real options data)")

        # DXY sanity check (should be ~100-110)
        if d['dxy_value'] > 0 and (d['dxy_value'] < 80 or d['dxy_value'] > 130):
            self.add_warning(f"DXY value suspicious: {d['dxy_value']:.2f} (expected ~100-110)")

        # DL samples check
        if d['dl_samples'] > 0 and d['dl_samples'] < 500:
            self.add_warning(f"DL trained on only {d['dl_samples']} samples (need 1000+ for reliability)")

        # HIGH model variance check
        if d['high_all_scores']:
            high_range = max(d['high_all_scores']) - min(d['high_all_scores'])
            if high_range > 0.15:
                self.add_warning(f"HIGH model high variance: scores range {min(d['high_all_scores']):.3f} to {max(d['high_all_scores']):.3f}")

    def print_summary(self):
        """Print compact summary at end of training."""
        self.data['total_time'] = (datetime.now() - self.start_time).total_seconds()
        self.check_issues()
        d = self.data

        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)

        # Data
        print(f"\n[DATA]")
        print(f"  Samples: {d['training_samples']} train / {d['test_samples']} test")
        pruned_total = d['features_pruned_zero_imp'] + d['features_pruned_corr']
        print(f"  Features: {d['features_created']} created -> {pruned_total} pruned ({d['features_pruned_zero_imp']} zero-imp, {d['features_pruned_corr']} corr) -> {d['features_used']} used")
        print(f"  Feature/Sample Ratio: {d['feature_sample_ratio']:.2f} (want <0.1)")

        # Feature Health
        print(f"\n[FEATURE HEALTH]")
        print(f"  VIX: {d['vix_value']:.2f} | DXY: {d['dxy_value']:.2f} | IV Fallback: {d['iv_fallback_pct']:.0f}%")
        if d['dl_samples'] > 0:
            print(f"  DL Training: {d['dl_samples']} samples")
        print(f"  Non-zero Importance: {d['nonzero_importance_count']}/{d['features_used']} features")

        # Optimization scores with distribution
        print(f"\n[OPTIMIZATION]")
        print(f"  RANGE: best={d['range_best_score']:.4f} ({d['range_trials']} trials)")
        if d['range_all_scores']:
            print(f"         scores: min={min(d['range_all_scores']):.3f} med={sorted(d['range_all_scores'])[len(d['range_all_scores'])//2]:.3f} max={max(d['range_all_scores']):.3f}")

        print(f"  HIGH:  best={d['high_best_score']:.4f}")
        if d['high_all_scores']:
            print(f"         scores: min={min(d['high_all_scores']):.3f} med={sorted(d['high_all_scores'])[len(d['high_all_scores'])//2]:.3f} max={max(d['high_all_scores']):.3f}")

        print(f"  LOW:   best={d['low_best_score']:.4f}")
        if d['low_all_scores']:
            neg_count = sum(1 for s in d['low_all_scores'] if s < 0)
            print(f"         scores: min={min(d['low_all_scores']):.3f} med={sorted(d['low_all_scores'])[len(d['low_all_scores'])//2]:.3f} max={max(d['low_all_scores']):.3f} (neg:{neg_count})")

        # Final Metrics
        print(f"\n[TEST METRICS]")
        print(f"  R²: {d['test_r2']:.4f} | RMSE: {d['test_rmse']:.3f}%")
        print(f"  Ensemble: {d['ensemble_size']} RANGE + {d['low_ensemble_size']} LOW models")
        if d['quantile_levels']:
            print(f"  Quantile CIs: {d['quantile_levels']}")

        # Top Features
        if d['top_features_corr']:
            print(f"\n[TOP 5 FEATURES - Correlation]")
            for i, (feat, corr) in enumerate(d['top_features_corr'][:5], 1):
                print(f"  {i}. {feat}: {corr:.3f}")

        if d['top_features_importance']:
            print(f"\n[TOP 5 FEATURES - Importance]")
            for i, feat in enumerate(d['top_features_importance'][:5], 1):
                print(f"  {i}. {feat}")

        # Best params (condensed)
        if d['best_params']:
            p = d['best_params']
            print(f"\n[BEST PARAMS]")
            print(f"  n_est={p.get('n_estimators', '?')} depth={p.get('max_depth', '?')} lr={p.get('learning_rate', 0):.4f}")
            print(f"  subsample={p.get('subsample', 0):.2f} colsample={p.get('colsample_bytree', 0):.2f}")
            print(f"  reg_alpha={p.get('reg_alpha', 0):.4f} reg_lambda={p.get('reg_lambda', 0):.4f}")

        # Warnings
        if d['warnings']:
            print(f"\n[WARNINGS] ({len(d['warnings'])})")
            for w in d['warnings']:
                print(f"  ⚠ {w}")
        else:
            print(f"\n[WARNINGS] None")

        # Timing
        print(f"\n[TIMING]")
        print(f"  Range Opt: {d['range_opt_time']:.1f}s | HighLow Opt: {d['highlow_opt_time']:.1f}s | Total: {d['total_time']:.1f}s")

        print("="*70 + "\n")

        return d  # Return data for programmatic access


# Global summary instance (reset at start of each train_range_model call)
_training_summary = TrainingSummary()


class PriceRangePredictor:
    """
    Predicts daily high/low range using technical indicators and options data.

    Uses XGBRegressor to predict:
    - Next day's range (high - low) as percentage of close
    - Next day's high and low prices

    Features include:
    - ATR (multiple periods)
    - Historical range percentiles
    - Implied Volatility (from options or VIX proxy)
    - Put/Call Ratio
    - Day of week / month effects
    - Market regime indicators
    """

    # Class-level caches (shared across all instances for efficiency)
    _class_vix_cache = None
    _class_vix_cache_date = None
    _class_vvix_cache = None
    _class_vvix_cache_date = None
    _class_sector_cache = {}
    _class_sector_cache_date = None
    _class_credit_cache = None
    _class_credit_cache_date = None
    _class_treasury_cache = None
    _class_treasury_cache_date = None
    _class_dollar_cache = None
    _class_dollar_cache_date = None
    _class_qqq_cache = None
    _class_qqq_cache_date = None
    _class_earnings_cache = {}
    _class_earnings_cache_date = None

    # Top SPY holdings for earnings calendar (top 10 = ~35% of SPY)
    SPY_TOP_HOLDINGS = [
        'AAPL',   # ~7%
        'MSFT',   # ~7%
        'NVDA',   # ~6%
        'AMZN',   # ~4%
        'META',   # ~2.5%
        'GOOGL',  # ~2%
        'GOOG',   # ~2%
        'BRK-B',  # ~2%
        'TSLA',   # ~2%
        'UNH',    # ~1.5%
    ]

    # Sector ETFs for breadth analysis
    SECTOR_ETFS = [
        'XLK',   # Technology
        'XLF',   # Financials
        'XLV',   # Healthcare
        'XLE',   # Energy
        'XLI',   # Industrials
        'XLP',   # Consumer Staples
        'XLY',   # Consumer Discretionary
        'XLU',   # Utilities
        'XLB',   # Materials
        'XLRE',  # Real Estate
        'XLC',   # Communication Services
    ]

    def __init__(self, polygon_manager=None):
        """
        Initialize the range predictor.

        Args:
            polygon_manager: Optional PolygonManager instance for options data
        """
        self.polygon = polygon_manager
        self.range_model = None
        self.ensemble_models = []  # For ensemble of top N models
        self.selected_features = []  # For feature selection
        self.selected_indices = []  # Indices of selected features
        self.high_model = None
        self.low_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_names = []
        self.model_metrics = {}
        # Instance cache references class cache for backwards compatibility
        self._vix_cache = None
        self._vix_cache_date = None

    def fetch_vix_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch VIX data from Yahoo Finance.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with VIX OHLC data indexed by date
        """
        if not YF_AVAILABLE:
            print("Warning: yfinance not available, skipping VIX data")
            return pd.DataFrame()

        try:
            # Use CLASS-LEVEL cache if recent (shared across all instances)
            today = datetime.now().date()
            if PriceRangePredictor._class_vix_cache is not None and PriceRangePredictor._class_vix_cache_date == today:
                # Return cached data without printing (avoid log spam in walk-forward)
                return PriceRangePredictor._class_vix_cache

            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

            print(f"   Fetching VIX data from {start_date} to {end_date}...")
            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)

            if vix.empty:
                print("   Warning: No VIX data returned")
                return pd.DataFrame()

            # Handle multi-level columns
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)

            vix.columns = vix.columns.str.lower()

            # Cache at CLASS level (shared across all instances)
            PriceRangePredictor._class_vix_cache = vix
            PriceRangePredictor._class_vix_cache_date = today

            print(f"   VIX data loaded: {len(vix)} bars, current VIX={vix['close'].iloc[-1]:.2f}")
            return vix

        except Exception as e:
            print(f"   VIX fetch error: {e}")
            return pd.DataFrame()

    def fetch_vvix_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch VVIX (VIX of VIX) data from Yahoo Finance.
        VVIX measures the expected volatility of VIX - high VVIX = expecting VIX moves.
        """
        if not YF_AVAILABLE:
            return pd.DataFrame()

        try:
            today = datetime.now().date()
            if PriceRangePredictor._class_vvix_cache is not None and PriceRangePredictor._class_vvix_cache_date == today:
                return PriceRangePredictor._class_vvix_cache

            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

            vvix = yf.download('^VVIX', start=start_date, end=end_date, progress=False)

            if vvix.empty:
                return pd.DataFrame()

            if isinstance(vvix.columns, pd.MultiIndex):
                vvix.columns = vvix.columns.get_level_values(0)
            vvix.columns = vvix.columns.str.lower()

            PriceRangePredictor._class_vvix_cache = vvix
            PriceRangePredictor._class_vvix_cache_date = today

            return vvix

        except Exception as e:
            print(f"   VVIX fetch error: {e}")
            return pd.DataFrame()

    def fetch_sector_data(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch sector ETF data for breadth analysis.
        """
        if not YF_AVAILABLE:
            return {}

        try:
            today = datetime.now().date()
            if PriceRangePredictor._class_sector_cache and PriceRangePredictor._class_sector_cache_date == today:
                return PriceRangePredictor._class_sector_cache

            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

            sector_data = {}
            # Download all sector ETFs in one call for efficiency
            tickers = ' '.join(self.SECTOR_ETFS)
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker')

            for etf in self.SECTOR_ETFS:
                try:
                    if len(self.SECTOR_ETFS) == 1:
                        df = data.copy()
                    else:
                        df = data[etf].copy()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = df.columns.str.lower()
                    if not df.empty:
                        sector_data[etf] = df
                except:
                    continue

            PriceRangePredictor._class_sector_cache = sector_data
            PriceRangePredictor._class_sector_cache_date = today

            return sector_data

        except Exception as e:
            print(f"   Sector data fetch error: {e}")
            return {}

    def fetch_credit_data(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch HYG (high yield) and LQD (investment grade) for credit spread analysis.
        """
        if not YF_AVAILABLE:
            return {}

        try:
            today = datetime.now().date()
            if PriceRangePredictor._class_credit_cache is not None and PriceRangePredictor._class_credit_cache_date == today:
                return PriceRangePredictor._class_credit_cache

            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

            credit_data = {}
            for ticker in ['HYG', 'LQD']:
                try:
                    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = df.columns.str.lower()
                    if not df.empty:
                        credit_data[ticker] = df
                except:
                    continue

            PriceRangePredictor._class_credit_cache = credit_data
            PriceRangePredictor._class_credit_cache_date = today

            return credit_data

        except Exception as e:
            print(f"   Credit data fetch error: {e}")
            return {}

    def fetch_treasury_data(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch Treasury yield data (10Y and 30Y).
        """
        if not YF_AVAILABLE:
            return {}

        try:
            today = datetime.now().date()
            if PriceRangePredictor._class_treasury_cache is not None and PriceRangePredictor._class_treasury_cache_date == today:
                return PriceRangePredictor._class_treasury_cache

            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

            treasury_data = {}
            for ticker, name in [('^TNX', '10Y'), ('^TYX', '30Y')]:
                try:
                    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = df.columns.str.lower()
                    if not df.empty:
                        treasury_data[name] = df
                except:
                    continue

            PriceRangePredictor._class_treasury_cache = treasury_data
            PriceRangePredictor._class_treasury_cache_date = today

            return treasury_data

        except Exception as e:
            print(f"   Treasury data fetch error: {e}")
            return {}

    def fetch_dollar_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch Dollar Index (DXY) data. Uses UUP ETF as more reliable proxy.
        """
        if not YF_AVAILABLE:
            return pd.DataFrame()

        try:
            today = datetime.now().date()
            if PriceRangePredictor._class_dollar_cache is not None and PriceRangePredictor._class_dollar_cache_date == today:
                return PriceRangePredictor._class_dollar_cache

            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

            # Try actual DXY tickers first, UUP as last resort (UUP trades ~$27 not ~$100)
            for ticker in ['DX=F', 'DX-Y.NYB', 'UUP']:
                try:
                    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = df.columns.str.lower()
                    if not df.empty:
                        PriceRangePredictor._class_dollar_cache = df
                        PriceRangePredictor._class_dollar_cache_date = today
                        return df
                except:
                    continue

            return pd.DataFrame()

        except Exception as e:
            print(f"   Dollar data fetch error: {e}")
            return pd.DataFrame()

    def fetch_qqq_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch QQQ data for correlation analysis with SPY.
        """
        if not YF_AVAILABLE:
            return pd.DataFrame()

        try:
            today = datetime.now().date()
            if PriceRangePredictor._class_qqq_cache is not None and PriceRangePredictor._class_qqq_cache_date == today:
                return PriceRangePredictor._class_qqq_cache

            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

            df = yf.download('QQQ', start=start_date, end=end_date, progress=False)

            if df.empty:
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = df.columns.str.lower()

            PriceRangePredictor._class_qqq_cache = df
            PriceRangePredictor._class_qqq_cache_date = today

            return df

        except Exception as e:
            print(f"   QQQ data fetch error: {e}")
            return pd.DataFrame()

    def fetch_earnings_calendar(self) -> Dict[str, List[datetime]]:
        """
        Fetch upcoming earnings dates for top SPY holdings.
        Returns dict of ticker -> list of earnings dates.
        """
        if not YF_AVAILABLE:
            return {}

        try:
            today = datetime.now().date()
            if PriceRangePredictor._class_earnings_cache and PriceRangePredictor._class_earnings_cache_date == today:
                return PriceRangePredictor._class_earnings_cache

            earnings_data = {}
            for ticker in self.SPY_TOP_HOLDINGS:
                try:
                    stock = yf.Ticker(ticker)
                    cal = stock.calendar
                    if cal is not None and not cal.empty:
                        # Try to get earnings date(s)
                        if 'Earnings Date' in cal.index:
                            dates = cal.loc['Earnings Date']
                            if isinstance(dates, pd.Series):
                                earnings_data[ticker] = [d for d in dates if pd.notna(d)]
                            elif pd.notna(dates):
                                earnings_data[ticker] = [dates]
                except:
                    continue

            PriceRangePredictor._class_earnings_cache = earnings_data
            PriceRangePredictor._class_earnings_cache_date = today

            return earnings_data

        except Exception as e:
            print(f"   Earnings calendar fetch error: {e}")
            return {}

    def get_options_features(self, ticker: str, current_price: float = None) -> Dict:
        """
        Extract comprehensive options-based features for range prediction.

        ENHANCED: Now uses get_full_options_analysis for maximum data extraction
        including ATM IV, term structure, Greeks, and unusual activity.

        Returns:
            dict with pcr_volume, pcr_oi, sentiment, iv_weighted, max_pain, high_oi_strikes,
            plus new fields: atm_iv, term_structure_slope, net_gamma, activity_bias
        """
        default_features = {
            'pcr_volume': 1.0,
            'pcr_oi': 1.0,
            'sentiment': 'NEUTRAL',
            'total_volume': 0,
            'iv_weighted': None,
            'iv_call': None,
            'iv_put': None,
            'iv_skew': None,
            'max_pain': None,
            'high_call_strike': None,
            'high_put_strike': None,
            # NEW enhanced fields
            'atm_iv': None,
            'atm_skew': None,
            'term_structure_slope': None,
            'term_structure_type': None,
            'near_term_iv': None,
            'far_term_iv': None,
            'net_delta': None,
            'net_gamma': None,
            'gamma_interpretation': None,
            'activity_bias': None,
            'unusual_contracts_count': None
        }

        if self.polygon is None:
            return default_features

        features = default_features.copy()

        try:
            # Use comprehensive analysis to get ALL available data
            full_analysis = self.polygon.get_full_options_analysis(ticker)

            if full_analysis.get('available'):
                # Core sentiment
                features['pcr_volume'] = full_analysis.get('pcr_volume', 1.0)
                features['pcr_oi'] = full_analysis.get('pcr_oi', 1.0)
                features['sentiment'] = full_analysis.get('sentiment', 'NEUTRAL')

                # Implied Volatility (multiple measures)
                features['iv_weighted'] = full_analysis.get('iv_weighted')
                features['iv_call'] = full_analysis.get('iv_call')
                features['iv_put'] = full_analysis.get('iv_put')
                features['iv_skew'] = full_analysis.get('iv_skew')

                # ATM IV (most relevant for range prediction!)
                features['atm_iv'] = full_analysis.get('atm_iv')
                features['atm_skew'] = full_analysis.get('atm_skew')

                # Term Structure (contango = normal, backwardation = fear)
                features['term_structure_slope'] = full_analysis.get('term_structure_slope')
                features['term_structure_type'] = full_analysis.get('term_structure_type')
                features['near_term_iv'] = full_analysis.get('near_term_iv')
                features['far_term_iv'] = full_analysis.get('far_term_iv')

                # Key Levels
                features['max_pain'] = full_analysis.get('max_pain')
                features['high_call_strike'] = full_analysis.get('highest_call_oi_strike')
                features['high_put_strike'] = full_analysis.get('highest_put_oi_strike')

                # Greeks (market maker positioning)
                features['net_delta'] = full_analysis.get('net_delta')
                features['net_gamma'] = full_analysis.get('net_gamma')
                features['gamma_interpretation'] = full_analysis.get('gamma_interpretation')

                # Unusual Activity
                features['activity_bias'] = full_analysis.get('activity_bias')
                features['unusual_contracts_count'] = full_analysis.get('unusual_contracts_count')

                print(f"   Options features loaded (ENHANCED):")
                print(f"     PCR={features['pcr_volume']:.2f}, Sentiment={features['sentiment']}")
                print(f"     IV Weighted={features.get('iv_weighted', 'N/A')}, ATM IV={features.get('atm_iv', 'N/A')}")
                print(f"     Term Structure: {features.get('term_structure_type', 'N/A')} (slope={features.get('term_structure_slope', 'N/A')}%)")
                print(f"     Max Pain=${features.get('max_pain', 'N/A')}")
                print(f"     Net Gamma={features.get('net_gamma', 'N/A')} ({features.get('gamma_interpretation', 'N/A')})")
                print(f"     Activity Bias={features.get('activity_bias', 'N/A')}, Unusual={features.get('unusual_contracts_count', 0)} contracts")
            else:
                # Fallback to individual calls if full analysis fails
                print(f"   Full analysis unavailable, using fallback methods...")

                sentiment_data = self.polygon.get_options_sentiment(ticker)
                if sentiment_data.get('status') == 'ok':
                    features['pcr_volume'] = sentiment_data.get('pcr_volume', 1.0)
                    features['pcr_oi'] = sentiment_data.get('pcr_oi', 1.0)
                    features['sentiment'] = sentiment_data.get('sentiment', 'NEUTRAL')

                iv_data = self.polygon.calculate_aggregate_iv(ticker, current_price)
                if iv_data.get('available'):
                    features['iv_weighted'] = iv_data.get('iv_weighted')
                    features['iv_call'] = iv_data.get('iv_call')
                    features['iv_put'] = iv_data.get('iv_put')
                    features['iv_skew'] = iv_data.get('iv_skew')

                max_pain_data = self.polygon.get_max_pain(ticker)
                if max_pain_data.get('available'):
                    features['max_pain'] = max_pain_data.get('max_pain')

                high_oi_data = self.polygon.get_high_oi_strikes(ticker)
                if high_oi_data.get('available'):
                    features['high_call_strike'] = high_oi_data.get('highest_call_strike')
                    features['high_put_strike'] = high_oi_data.get('highest_put_strike')

        except Exception as e:
            print(f"   Options data error: {e}")

        return features

    def create_range_features(self, df: pd.DataFrame, options_features: Dict = None) -> pd.DataFrame:
        """
        Create features for range prediction.

        Args:
            df: OHLCV DataFrame
            options_features: Optional dict with PCR, IV data

        Returns:
            DataFrame with features for prediction
        """
        features = pd.DataFrame(index=df.index)

        # --- Price-based features ---
        features['close'] = df['close']
        features['daily_range'] = (df['high'] - df['low']) / df['close']
        features['daily_range_pct'] = features['daily_range'] * 100

        # Range relative to close
        features['high_from_open'] = (df['high'] - df['open']) / df['open']
        features['low_from_open'] = (df['open'] - df['low']) / df['open']

        # --- ATR features (multiple periods) ---
        if TA_AVAILABLE:
            features['atr_7'] = ta.atr(df['high'], df['low'], df['close'], length=7)
            features['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            features['atr_21'] = ta.atr(df['high'], df['low'], df['close'], length=21)
        else:
            # Manual ATR calculation
            for period in [7, 14, 21]:
                tr = pd.concat([
                    df['high'] - df['low'],
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                ], axis=1).max(axis=1)
                features[f'atr_{period}'] = tr.rolling(period).mean()

        # ATR as percentage of close
        features['atr_14_pct'] = features['atr_14'] / df['close'] * 100
        features['atr_ratio_7_21'] = features['atr_7'] / features['atr_21']

        # --- Historical range percentiles ---
        features['range_percentile_5d'] = features['daily_range'].rolling(5).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )
        features['range_percentile_10d'] = features['daily_range'].rolling(10).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )
        features['range_percentile_20d'] = features['daily_range'].rolling(20).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )

        # Rolling range statistics
        features['range_mean_3d'] = features['daily_range'].rolling(3).mean()  # Added early for RSI-normalized features
        features['range_mean_5d'] = features['daily_range'].rolling(5).mean()
        features['range_mean_7d'] = features['daily_range'].rolling(7).mean()  # Added early for RSI-normalized features
        features['range_std_5d'] = features['daily_range'].rolling(5).std()
        features['range_mean_20d'] = features['daily_range'].rolling(20).mean()
        features['range_std_20d'] = features['daily_range'].rolling(20).std()
        features['range_max_3d'] = features['daily_range'].rolling(3).max()   # For super-features
        features['range_max_5d'] = features['daily_range'].rolling(5).max()   # For super-features

        # Range z-score
        features['range_zscore'] = (features['daily_range'] - features['range_mean_20d']) / features['range_std_20d']

        # --- Volatility features ---
        features['returns'] = df['close'].pct_change()
        features['volatility_5d'] = features['returns'].rolling(5).std() * np.sqrt(252) * 100
        features['volatility_20d'] = features['returns'].rolling(20).std() * np.sqrt(252) * 100
        features['vol_ratio'] = features['volatility_5d'] / features['volatility_20d']

        # --- Volume features (with derivatives) ---
        volume = df['volume']
        volume_sma = volume.rolling(20).mean()
        features['volume_rel'] = volume / volume_sma  # Relative volume

        # Volume velocity (1st derivative)
        features['volume_velocity'] = volume.diff()
        features['volume_velocity_5d'] = volume.rolling(5).mean().diff()
        features['volume_rel_velocity'] = features['volume_rel'].diff()

        # Volume acceleration (2nd derivative)
        features['volume_accel'] = features['volume_velocity'].diff()
        features['volume_accel_5d'] = features['volume_velocity_5d'].diff()

        # Normalized volume velocity
        vol_std = volume.rolling(20).std()
        features['volume_velocity_norm'] = features['volume_velocity'] / (vol_std + 1)

        # =================================================================
        # ENHANCED VOLUME FEATURES (Based on analysis findings)
        # Volume showed high correlation but underused importance
        # =================================================================
        # Volume Z-score (identify abnormal volume days)
        features['volume_zscore'] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1)

        # Volume trend (is volume increasing or decreasing over time?)
        features['volume_trend_5d'] = volume.rolling(5).mean() / volume.rolling(20).mean()
        features['volume_trend_10d'] = volume.rolling(10).mean() / volume.rolling(20).mean()

        # Volume-price divergence (high volume but small price move = important)
        price_change = abs(df['close'].pct_change())
        features['vol_price_divergence'] = features['volume_rel'] / (price_change * 100 + 0.1)

        # Volume spike detection
        features['volume_spike'] = (features['volume_zscore'] > 2.0).astype(int)

        # Volume percentile (where is today's volume in recent history?)
        features['volume_percentile'] = volume.rolling(60).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )

        # Volume momentum (rate of change in volume)
        features['volume_momentum'] = volume.pct_change(5)

        # =================================================================
        # INTERACTION TERMS (Based on correlation analysis findings)
        # These capture non-linear relationships between top features
        # =================================================================

        # 1. Range-Momentum Interaction (range * ROC)
        # High ROC with expanding range = strong directional move
        features['range_momentum'] = features['daily_range'] * features.get('roc_5', df['close'].pct_change(5) * 100).abs() / 100

        # 2. Volume-Regime Interaction (volume * volatility regime)
        # High volume during high vol regime = significant
        features['vol_regime_interaction'] = features['volume_rel'] * features['volatility_20d'] / 20

        # 3. ATR Breakout Interaction (current range vs ATR - identifies breakout days)
        features['atr_breakout'] = features['daily_range'] / (features['atr_14'] / df['close'] + 0.001)
        features['atr_breakout_zscore'] = (features['atr_breakout'] - features['atr_breakout'].rolling(20).mean()) / (features['atr_breakout'].rolling(20).std() + 0.001)

        # 4. Trend Dislocation (price vs trend * volatility - identifies reversal setups)
        trend_deviation = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).mean()
        features['trend_dislocation'] = trend_deviation * features['volatility_20d'] / 20

        # 5. Volatility Compression (ATR ratio near 1 = low vol, breakout coming)
        features['vol_compression'] = 1 - abs(1 - features['atr_ratio_7_21'])
        features['vol_compression_signal'] = (features['vol_compression'] > 0.9).astype(int)

        # 6. Range Mean Reversion Signal (extreme ranges tend to revert)
        features['range_reversion_signal'] = -features['range_zscore'] * features['range_percentile_20d']

        # =================================================================
        # RSI-NORMALIZED FEATURES (Part 1 - features that don't need upside_potential)
        # Dividing by RSI captures momentum-adjusted volatility
        # These had 0.718 correlation vs 0.644 for best existing features
        # =================================================================

        # Ensure RSI exists before creating RSI-based features
        if 'rsi_14' in features.columns:
            rsi_safe = features['rsi_14'].clip(lower=1)  # Avoid division by zero
            rsi_inverted = 100 - features['rsi_14']  # Inverted RSI (high when oversold)

            # Top 4 discovered features - all involve dividing by RSI
            # 1. vol_regime_interaction / rsi_14 → 0.718 correlation (BEST)
            features['vol_regime_over_rsi'] = features['vol_regime_interaction'] / rsi_safe * 100

            # 2. range_mean_5d / rsi_14 → 0.714 correlation
            features['range_5d_over_rsi'] = features['range_mean_5d'] / rsi_safe * 100

            # 3. daily_range / rsi_14 → 0.704 correlation
            features['range_over_rsi'] = features['daily_range'] / rsi_safe * 100

            # 4. daily_range_pct / rsi_14 → 0.704 correlation
            features['range_pct_over_rsi'] = features['daily_range_pct'] / rsi_safe

            # These don't need upside_potential
            features['price_sma50_over_rsi'] = features['price_vs_sma50'] / rsi_safe * 100
            features['atr_pct_over_rsi'] = features['atr_14_pct'] / rsi_safe

            # === EXPANDED RSI-NORMALIZED FEATURES ===
            # ATR variants normalized by RSI
            features['atr_7_over_rsi'] = features['atr_7'] / rsi_safe
            features['atr_14_over_rsi'] = features['atr_14'] / rsi_safe
            features['atr_21_over_rsi'] = features['atr_21'] / rsi_safe

            # Volatility metrics normalized by RSI
            features['vol_5d_over_rsi'] = features['volatility_5d'] / rsi_safe * 100
            features['vol_20d_over_rsi'] = features['volatility_20d'] / rsi_safe * 100

            # Range statistics normalized by RSI
            features['range_mean_3d_over_rsi'] = features['range_mean_3d'] / rsi_safe * 100
            features['range_mean_7d_over_rsi'] = features['range_mean_7d'] / rsi_safe * 100
            features['range_mean_20d_over_rsi'] = features['range_mean_20d'] / rsi_safe * 100
            features['range_std_5d_over_rsi'] = features['range_std_5d'] / rsi_safe * 100

            # Price position features normalized by RSI
            features['price_sma20_over_rsi'] = features['price_vs_sma20'] / rsi_safe * 100
            features['drawdown_over_rsi'] = features['drawdown_from_high_20d'] / rsi_safe

            # Momentum features multiplied by inverted RSI (amplify when oversold)
            features['range_x_inv_rsi'] = features['daily_range'] * rsi_inverted / 100
            features['atr_x_inv_rsi'] = features['atr_14_pct'] * rsi_inverted / 100
            features['vol_x_inv_rsi'] = features['volatility_20d'] * rsi_inverted / 100

            # RSI-based regime indicators
            features['rsi_oversold_intensity'] = np.maximum(0, 30 - features['rsi_14']) / 30
            features['rsi_overbought_intensity'] = np.maximum(0, features['rsi_14'] - 70) / 30

            # RSI velocity and acceleration
            features['rsi_velocity'] = features['rsi_14'].diff()
            features['rsi_accel'] = features['rsi_velocity'].diff()

            # RSI divergence from price (price up but RSI down = bearish divergence)
            price_direction = np.sign(df['close'].diff(5))
            rsi_direction = np.sign(features['rsi_14'].diff(5))
            features['rsi_price_divergence'] = (price_direction != rsi_direction).astype(int)

            print(f"   RSI-normalized features (Part 1) added: 24 new features")

        # =================================================================
        # TREND-AWARE FEATURES (To fix downtrend prediction bias)
        # Analysis showed model predicts highs better than lows
        # =================================================================

        # Trend direction and strength
        features['trend_direction'] = np.sign(df['close'] - df['close'].rolling(20).mean())
        features['trend_strength'] = abs(df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()

        # Downtrend indicator (more weight on low prediction)
        features['is_downtrend'] = (features['trend_direction'] < 0).astype(int)
        features['is_uptrend'] = (features['trend_direction'] > 0).astype(int)

        # Trend-adjusted range components (helps with asymmetric prediction)
        features['upside_potential'] = (df['high'].rolling(5).max() - df['close']) / df['close']
        features['downside_potential'] = (df['close'] - df['low'].rolling(5).min()) / df['close']
        features['range_asymmetry'] = features['upside_potential'] - features['downside_potential']

        # Trend momentum (acceleration of trend)
        sma_20_change = df['close'].rolling(20).mean().pct_change(5)
        features['trend_momentum'] = sma_20_change * 100

        # Price position in recent range (where is close relative to high-low?)
        features['price_position'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min() + 0.001)

        # Consecutive down/up days
        features['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(int).groupby((df['close'] <= df['close'].shift(1)).cumsum()).cumsum()
        features['consecutive_down'] = (df['close'] < df['close'].shift(1)).astype(int).groupby((df['close'] >= df['close'].shift(1)).cumsum()).cumsum()

        # Recent low/high relative to ATR (identifies stretched moves)
        features['low_stretch'] = (df['close'] - df['low'].rolling(10).min()) / features['atr_14']
        features['high_stretch'] = (df['high'].rolling(10).max() - df['close']) / features['atr_14']

        # =================================================================
        # RSI-NORMALIZED FEATURES (Part 2 - features that need upside_potential)
        # =================================================================
        if 'rsi_14' in features.columns:
            rsi_safe = features['rsi_14'].clip(lower=1)
            features['upside_over_rsi'] = features['upside_potential'] / rsi_safe * 100
            print(f"   RSI-normalized features (Part 2) added: 1 new feature")

        # =================================================================
        # TOP DISCOVERED INTERACTION FEATURES (from comprehensive auto-discover)
        # These had 0.65-0.72 correlation - significantly better than baseline
        # =================================================================

        # 5. vol_regime_interaction - price_position → 0.682 correlation
        features['vol_regime_minus_position'] = features['vol_regime_interaction'] - features['price_position']

        # 6. range_mean_5d + upside_potential → 0.681 correlation
        features['range_plus_upside'] = features['range_mean_5d'] + features['upside_potential']

        # 7. range_mean_5d * upside_potential → 0.679 correlation
        features['range_times_upside'] = features['range_mean_5d'] * features['upside_potential']

        # 8. daily_range + upside_potential → 0.678 correlation
        features['daily_range_plus_upside'] = features['daily_range'] + features['upside_potential']

        # 9. upside_potential * atr_7 → 0.673 correlation
        features['upside_times_atr7'] = features['upside_potential'] * features['atr_7']

        # 10. upside_potential * atr_14 → 0.668 correlation
        features['upside_times_atr14'] = features['upside_potential'] * features['atr_14']

        # Additional high-correlation interactions discovered
        features['upside_plus_range_mean20'] = features['upside_potential'] + features['range_mean_20d']
        features['range_times_high_stretch'] = features['range_mean_5d'] * features['high_stretch']
        features['vol_regime_plus_range'] = features['vol_regime_interaction'] + features['daily_range']
        features['vol_regime_minus_trend'] = features['vol_regime_interaction'] - features['trend_dislocation']

        print(f"   Top discovered interactions added: 10 new features")

        # NOTE: Super-features moved to end of feature creation (after range_mean_3d is defined)

        # =================================================================
        # DOWNTREND INTERACTION TERMS (configurable - low importance)
        # Analysis showed features flip in downtrends - need specific interactions
        # =================================================================
        if FEATURE_CONFIG.get('interaction_features', False):
            # Downtrend-specific volume (volume matters more in downtrends)
            features['downtrend_volume'] = features['is_downtrend'] * features['volume_rel']

            # Downtrend ATR (ATR more predictive in downtrends)
            features['downtrend_atr'] = features['is_downtrend'] * features['atr_7']
            features['downtrend_atr_14'] = features['is_downtrend'] * features['atr_14']

            # Downtrend momentum (ROC stronger signal in downtrends)
            features['downtrend_roc_5'] = features['is_downtrend'] * abs(df['close'].pct_change(5) * 100)
            features['downtrend_roc_10'] = features['is_downtrend'] * abs(df['close'].pct_change(10) * 100)

            # Uptrend-specific features (for symmetric treatment)
            features['uptrend_volume'] = features['is_uptrend'] * features['volume_rel']
            features['uptrend_atr'] = features['is_uptrend'] * features['atr_14']
            features['uptrend_momentum'] = features['is_uptrend'] * abs(df['close'].pct_change(5) * 100)

            # Range lagged interactions (yesterday's range predicts today's)
            features['range_lag_x_mean'] = features.get('range_lag_1', features['daily_range'].shift(1)) * features['range_mean_5d']

            # ATR lagged interaction
            features['atr_lag_x_current'] = features.get('atr_14_lag_1', features['atr_14'].shift(1)) * features['atr_14']

        # --- Momentum features ---
        features['roc_5'] = df['close'].pct_change(5) * 100
        features['roc_10'] = df['close'].pct_change(10) * 100

        # RSI
        if TA_AVAILABLE:
            features['rsi_14'] = ta.rsi(df['close'], length=14)
        else:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi_14'] = 100 - (100 / (1 + rs))

        # --- Calendar features ---
        features['day_of_week'] = df.index.dayofweek
        features['month'] = df.index.month
        features['is_monday'] = (df.index.dayofweek == 0).astype(int)
        features['is_friday'] = (df.index.dayofweek == 4).astype(int)
        features['is_month_end'] = df.index.is_month_end.astype(int)

        # --- Gap features ---
        features['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        features['gap_abs'] = abs(features['gap'])

        # --- Trend features ---
        features['sma_20'] = df['close'].rolling(20).mean()
        features['sma_50'] = df['close'].rolling(50).mean()
        features['price_vs_sma20'] = (df['close'] - features['sma_20']) / features['sma_20']
        features['price_vs_sma50'] = (df['close'] - features['sma_50']) / features['sma_50']
        features['sma_20_slope'] = features['sma_20'].pct_change(5)

        # --- Options features (if available) ---
        if options_features:
            # Basic PCR features
            features['pcr_volume'] = options_features.get('pcr_volume', 1.0)
            features['pcr_oi'] = options_features.get('pcr_oi', 1.0)

            # Encode sentiment
            sentiment = options_features.get('sentiment', 'NEUTRAL')
            features['sentiment_bullish'] = 1 if sentiment == 'BULLISH' else 0
            features['sentiment_bearish'] = 1 if sentiment == 'BEARISH' else 0

            # Implied Volatility features (very important for range prediction!)
            iv_weighted = options_features.get('iv_weighted')
            if iv_weighted is not None:
                features['iv_weighted'] = iv_weighted
                features['iv_call'] = options_features.get('iv_call', iv_weighted)
                features['iv_put'] = options_features.get('iv_put', iv_weighted)
                features['iv_skew'] = options_features.get('iv_skew', 0)
            else:
                # Use historical volatility as proxy if IV not available
                features['iv_weighted'] = features['volatility_20d']
                features['iv_call'] = features['volatility_20d']
                features['iv_put'] = features['volatility_20d']
                features['iv_skew'] = 0

            # Max pain distance from current price
            max_pain = options_features.get('max_pain')
            current_close = df['close'].iloc[-1]
            if max_pain is not None and current_close > 0:
                features['max_pain_distance'] = (max_pain - current_close) / current_close
            else:
                features['max_pain_distance'] = 0

            # High OI strike distances
            high_call = options_features.get('high_call_strike')
            high_put = options_features.get('high_put_strike')
            if high_call is not None and current_close > 0:
                features['high_call_oi_distance'] = (high_call - current_close) / current_close
            else:
                features['high_call_oi_distance'] = 0
            if high_put is not None and current_close > 0:
                features['high_put_oi_distance'] = (current_close - high_put) / current_close
            else:
                features['high_put_oi_distance'] = 0

            # =================================================================
            # NEW ENHANCED OPTIONS FEATURES (from get_full_options_analysis)
            # =================================================================

            # ATM IV (most relevant for daily range prediction!)
            atm_iv = options_features.get('atm_iv')
            if atm_iv is not None:
                features['atm_iv'] = atm_iv
                features['atm_skew'] = options_features.get('atm_skew', 0)
                # ATM IV vs historical volatility spread
                features['atm_iv_rv_spread'] = atm_iv - features['volatility_20d']
            else:
                features['atm_iv'] = features['volatility_20d']
                features['atm_skew'] = 0
                features['atm_iv_rv_spread'] = 0

            # Term Structure (contango/backwardation)
            term_slope = options_features.get('term_structure_slope')
            if term_slope is not None:
                features['term_structure_slope'] = term_slope
                features['term_structure_backwardation'] = 1 if term_slope < -5 else 0
                features['term_structure_contango'] = 1 if term_slope > 5 else 0
                # Near vs far term IV spread
                near_iv = options_features.get('near_term_iv', 0)
                far_iv = options_features.get('far_term_iv', 0)
                features['near_term_iv'] = near_iv
                features['far_term_iv'] = far_iv
                features['iv_term_spread'] = near_iv - far_iv  # Positive = backwardation
            else:
                features['term_structure_slope'] = 0
                features['term_structure_backwardation'] = 0
                features['term_structure_contango'] = 0
                features['near_term_iv'] = 0
                features['far_term_iv'] = 0
                features['iv_term_spread'] = 0

            # Greeks (market maker positioning)
            net_gamma = options_features.get('net_gamma')
            if net_gamma is not None:
                # Normalize gamma by price for comparability
                features['net_gamma_normalized'] = net_gamma / (current_close * 1000000) if current_close > 0 else 0
                features['gamma_long'] = 1 if options_features.get('gamma_interpretation') == 'long gamma' else 0
                features['gamma_short'] = 1 if options_features.get('gamma_interpretation') == 'short gamma' else 0
            else:
                features['net_gamma_normalized'] = 0
                features['gamma_long'] = 0
                features['gamma_short'] = 0

            net_delta = options_features.get('net_delta')
            if net_delta is not None:
                # Normalize delta
                features['net_delta_normalized'] = net_delta / 1000000 if net_delta else 0
            else:
                features['net_delta_normalized'] = 0

            # Unusual Activity
            activity_bias = options_features.get('activity_bias', 'NEUTRAL')
            features['unusual_activity_bullish'] = 1 if activity_bias == 'BULLISH' else 0
            features['unusual_activity_bearish'] = 1 if activity_bias == 'BEARISH' else 0
            unusual_count = options_features.get('unusual_contracts_count', 0)
            features['unusual_activity_count'] = unusual_count if unusual_count else 0

        else:
            features['pcr_volume'] = 1.0
            features['pcr_oi'] = 1.0
            features['sentiment_bullish'] = 0
            features['sentiment_bearish'] = 0
            features['iv_weighted'] = features['volatility_20d']
            features['iv_call'] = features['volatility_20d']
            features['iv_put'] = features['volatility_20d']
            features['iv_skew'] = 0
            features['max_pain_distance'] = 0
            features['high_call_oi_distance'] = 0
            features['high_put_oi_distance'] = 0
            # Enhanced options features defaults
            features['atm_iv'] = features['volatility_20d']
            features['atm_skew'] = 0
            features['atm_iv_rv_spread'] = 0
            features['term_structure_slope'] = 0
            features['term_structure_backwardation'] = 0
            features['term_structure_contango'] = 0
            features['near_term_iv'] = 0
            features['far_term_iv'] = 0
            features['iv_term_spread'] = 0
            features['net_gamma_normalized'] = 0
            features['gamma_long'] = 0
            features['gamma_short'] = 0
            features['net_delta_normalized'] = 0
            features['unusual_activity_bullish'] = 0
            features['unusual_activity_bearish'] = 0
            features['unusual_activity_count'] = 0

        # =================================================================
        # IV FEATURE DEBUG LOGGING (To diagnose 0-importance IV features)
        # Analysis showed IV features have high correlation but 0 importance
        # This is likely because they're just copies of volatility_20d
        # =================================================================
        iv_unique = features['iv_weighted'].nunique() if 'iv_weighted' in features else 0
        vol20_unique = features['volatility_20d'].nunique() if 'volatility_20d' in features else 0
        iv_vol_match = 0.0
        if 'iv_weighted' in features and 'volatility_20d' in features:
            try:
                iv_vol_match = (features['iv_weighted'] == features['volatility_20d']).mean() * 100
            except:
                pass
        print(f"   IV Features Debug:")
        print(f"     iv_weighted unique values: {iv_unique}")
        print(f"     volatility_20d unique values: {vol20_unique}")
        print(f"     iv_weighted == volatility_20d: {iv_vol_match:.1f}%")
        if iv_vol_match > 95:
            print(f"     WARNING: IV features appear to be fallback values (same as volatility_20d)")
            print(f"     This explains why IV has high correlation but 0 model importance!")

        # --- VIX Features (CRITICAL for volatility prediction!) ---
        vix_df = self.fetch_vix_data()
        if not vix_df.empty:
            try:
                # Align VIX with price data by date
                # Convert both to date-only index for matching
                vix_daily = vix_df.copy()
                vix_daily.index = pd.to_datetime(vix_daily.index).date

                # Create VIX features
                vix_close = vix_daily['close']
                features['vix'] = df.index.map(lambda x: vix_close.get(x.date() if hasattr(x, 'date') else x, np.nan))

                # Fill forward for any missing dates
                features['vix'] = features['vix'].ffill().bfill()

                # VIX-derived features
                features['vix_change_1d'] = features['vix'].pct_change() * 100
                features['vix_change_5d'] = features['vix'].pct_change(5) * 100
                features['vix_sma_10'] = features['vix'].rolling(10).mean()
                features['vix_vs_sma'] = (features['vix'] - features['vix_sma_10']) / features['vix_sma_10']
                features['vix_percentile'] = features['vix'].rolling(60).apply(
                    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
                )

                # VIX term structure proxy (compare to historical)
                features['vix_zscore'] = (features['vix'] - features['vix'].rolling(20).mean()) / features['vix'].rolling(20).std()

                # VIX regime (high/normal/low volatility environment)
                vix_20_pct = features['vix'].rolling(252).quantile(0.2)
                vix_80_pct = features['vix'].rolling(252).quantile(0.8)
                features['vix_regime_low'] = (features['vix'] < vix_20_pct).astype(int)
                features['vix_regime_high'] = (features['vix'] > vix_80_pct).astype(int)

                # VIX Velocity (1st derivative) - rate of change in VIX
                features['vix_velocity'] = features['vix'].diff()
                features['vix_velocity_3d'] = features['vix'].rolling(3).mean().diff()
                features['vix_velocity_5d'] = features['vix'].rolling(5).mean().diff()

                # VIX Acceleration (2nd derivative) - change in rate of change
                features['vix_accel'] = features['vix_velocity'].diff()
                features['vix_accel_3d'] = features['vix_velocity_3d'].diff()
                features['vix_accel_5d'] = features['vix_velocity_5d'].diff()

                # Normalized VIX velocity (relative to recent volatility of VIX itself)
                vix_std = features['vix'].rolling(20).std()
                features['vix_velocity_norm'] = features['vix_velocity'] / (vix_std + 0.01)

                print(f"   VIX features added: current VIX={features['vix'].iloc[-1]:.2f}, velocity={features['vix_velocity'].iloc[-1]:.2f}")
            except Exception as e:
                print(f"   VIX feature error: {e}")
                # Add placeholder VIX features
                for col in ['vix', 'vix_change_1d', 'vix_change_5d', 'vix_sma_10', 'vix_vs_sma',
                           'vix_percentile', 'vix_zscore', 'vix_regime_low', 'vix_regime_high',
                           'vix_velocity', 'vix_velocity_3d', 'vix_velocity_5d',
                           'vix_accel', 'vix_accel_3d', 'vix_accel_5d', 'vix_velocity_norm']:
                    features[col] = 0
        else:
            # Add placeholder VIX features if not available
            for col in ['vix', 'vix_change_1d', 'vix_change_5d', 'vix_sma_10', 'vix_vs_sma',
                       'vix_percentile', 'vix_zscore', 'vix_regime_low', 'vix_regime_high',
                       'vix_velocity', 'vix_velocity_3d', 'vix_velocity_5d',
                       'vix_accel', 'vix_accel_3d', 'vix_accel_5d', 'vix_velocity_norm']:
                features[col] = 0

        # --- Novel Indicators (Advanced composite indicators) ---
        # Novel indicators controlled by FEATURE_CONFIG
        if FEATURE_CONFIG.get('novel_indicators', False) and NOVEL_INDICATORS_AVAILABLE:
            try:
                print("   Calculating novel indicators...")

                # ARWO - Adaptive Regime-Weighted Oscillator
                features['arwo'] = calculate_arwo(df)

                # DCO - Divergence Consensus Oscillator
                features['dco'] = calculate_dco(df)

                # VCMO - Volume-Confirmed Momentum Oscillator
                features['vcmo'] = calculate_vcmo(df)

                # ICS - Indicator Convergence Score
                features['ics'] = calculate_ics(df)

                # MJI - Momentum Jerk Indicator (2nd derivative of momentum)
                features['mji'] = calculate_mji(df)

                # PRF - Percentile Rank Fusion
                features['prf'] = calculate_prf(df)

                # EWAF - Entropy-Weighted Adaptive Fusion
                features['ewaf'] = calculate_ewaf(df)

                # KFIF - Kalman-Filtered Indicator Fusion (if available)
                try:
                    features['kfif'] = calculate_kfif(df)
                    print(f"   KFIF added")
                except Exception as kfif_err:
                    features['kfif'] = 0
                    print(f"   KFIF skipped: {kfif_err}")

                print(f"   Novel indicators added: ARWO={features['arwo'].iloc[-1]:.3f}, MJI={features['mji'].iloc[-1]:.3f}")
            except Exception as e:
                print(f"   Novel indicator error: {e}")
                # Don't add placeholder columns - they cause issues with feature selection
        else:
            # Novel indicators disabled - don't add placeholder columns
            print("   Novel indicators DISABLED (FEATURE_CONFIG['novel_indicators'] = False)")

        # --- Composite Oscillator Features ---
        # Use the MAIN composite oscillator (15 components) for consistency with velocity trading
        try:
            if MAIN_COMPOSITE_AVAILABLE:
                # Use the full 15-component composite from oscillator_predictor_page.py
                df_with_osc = create_composite_oscillator(df)
                features['composite_osc'] = df_with_osc['composite_oscillator']
                features['composite_osc_smooth'] = df_with_osc['composite_smooth']

                # Add individual normalized components for feature importance analysis
                norm_cols = [c for c in df_with_osc.columns if c.endswith('_norm') or c in ['bb_position', 'adx_trend']]
                for col in norm_cols:
                    if col in df_with_osc.columns:
                        features[col] = df_with_osc[col]

                print(f"   Main composite oscillator (15 components): {features['composite_osc_smooth'].iloc[-1]:.3f}")
            else:
                # Fallback to simplified 4-component version
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                rsi_norm = (rsi - 50) / 50

                low_14 = df['low'].rolling(14).min()
                high_14 = df['high'].rolling(14).max()
                stoch_k = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
                stoch_norm = (stoch_k - 50) / 50

                williams_r = -100 * (high_14 - df['close']) / (high_14 - low_14 + 1e-10)
                williams_norm = (williams_r + 50) / 50

                typical_price = (df['high'] + df['low'] + df['close']) / 3
                cci = (typical_price - typical_price.rolling(20).mean()) / (0.015 * typical_price.rolling(20).std())
                cci_norm = (cci / 100).clip(-1, 1)

                features['composite_osc'] = (rsi_norm * 0.3 + stoch_norm * 0.25 + williams_norm * 0.25 + cci_norm * 0.2)
                features['composite_osc_smooth'] = features['composite_osc'].rolling(3).mean().fillna(features['composite_osc'])
                print(f"   Fallback composite oscillator (4 components): {features['composite_osc'].iloc[-1]:.3f}")

            # Composite velocity and acceleration (always calculated)
            features['composite_velocity'] = features['composite_osc_smooth'].diff()
            features['composite_accel'] = features['composite_velocity'].diff()

        except Exception as comp_err:
            print(f"   Composite oscillator error: {comp_err}")
            features['composite_osc'] = 0
            features['composite_osc_smooth'] = 0
            features['composite_velocity'] = 0
            features['composite_accel'] = 0

        # --- Additional Volatility Predictors ---
        # Bollinger Band Width (volatility indicator)
        bb_sma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        features['bb_width'] = (2 * bb_std / bb_sma) * 100  # As percentage

        # Keltner Channel Width
        atr_10 = features.get('atr_14', df['high'] - df['low']).rolling(14).mean()
        features['keltner_width'] = (2 * 1.5 * atr_10 / df['close']) * 100

        # Historical Volatility Ratio (short vs long)
        features['hvol_ratio'] = features['volatility_5d'] / features['volatility_20d']

        # Parkinson volatility estimator (uses high-low)
        features['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * ((np.log(df['high'] / df['low'])) ** 2).rolling(20).mean()
        ) * np.sqrt(252) * 100

        # Range expansion/contraction indicator
        features['range_expansion'] = features['daily_range'] / features['daily_range'].rolling(10).mean()

        # Overnight gap volatility
        features['gap_volatility_5d'] = features['gap_abs'].rolling(5).std() * 100

        # Volume-weighted volatility
        if 'volume' in df.columns:
            vol_weight = df['volume'] / df['volume'].rolling(20).mean()
            features['vol_weighted_range'] = features['daily_range'] * vol_weight

        # =================================================================
        # RESEARCH-BACKED SCIENTIFIC INDICATORS
        # Based on academic papers for volatility/range prediction
        # Controlled by FEATURE_CONFIG['scientific_features'] - slow, low importance
        # =================================================================
        if FEATURE_CONFIG.get('scientific_features', False):
            print("   Calculating research-backed scientific indicators...")

            # --- 1. HURST EXPONENT (Regime Detection) ---
            # Source: Qian & Rasheed, "Hurst Exponent and Financial Market Predictability"
            # H > 0.5: trending (persistent), H < 0.5: mean-reverting, H = 0.5: random
            try:
                def calculate_hurst(series, max_lag=20):
                    """Calculate Hurst exponent using R/S method."""
                    if len(series) < max_lag * 2:
                        return 0.5
                    lags = range(2, min(max_lag, len(series) // 2))
                    tau = []
                    for lag in lags:
                        # Standard deviation of lagged differences
                        tau.append(np.std(np.subtract(series[lag:], series[:-lag])))
                    if len(tau) < 2 or any(t <= 0 for t in tau):
                        return 0.5
                    # Hurst exponent from log-log regression
                    try:
                        reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
                        return reg[0]
                    except:
                        return 0.5

                # Rolling Hurst on log returns
                log_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
                features['hurst_20d'] = log_returns.rolling(40).apply(
                    lambda x: calculate_hurst(x.values, 20), raw=False
                )
                features['hurst_60d'] = log_returns.rolling(100).apply(
                    lambda x: calculate_hurst(x.values, 30), raw=False
                )

                # Hurst regime indicators
                features['hurst_trending'] = (features['hurst_20d'] > 0.55).astype(int)
                features['hurst_mean_reverting'] = (features['hurst_20d'] < 0.45).astype(int)

                print(f"   Hurst exponent: {features['hurst_20d'].iloc[-1]:.3f}")
            except Exception as e:
                print(f"   Hurst calculation error: {e}")
                features['hurst_20d'] = 0.5
                features['hurst_60d'] = 0.5
                features['hurst_trending'] = 0
                features['hurst_mean_reverting'] = 0

            # --- 2. HAR COMPONENTS (Heterogeneous Autoregressive) ---
            # Source: Corsi (2009), "A Simple Approximate Long-Memory Model of Realized Volatility"
            # Uses daily, weekly, monthly realized volatility components
            try:
                # Realized volatility (using range as proxy, more efficient than close-to-close)
                rv_daily = features['daily_range']  # Today's range
                features['har_rv_daily'] = rv_daily
                features['har_rv_weekly'] = rv_daily.rolling(5).mean()  # 5-day (weekly) average
                features['har_rv_monthly'] = rv_daily.rolling(22).mean()  # 22-day (monthly) average

                # HAR component ratios (detect regime changes)
                features['har_weekly_daily_ratio'] = features['har_rv_weekly'] / (features['har_rv_daily'] + 1e-10)
                features['har_monthly_weekly_ratio'] = features['har_rv_monthly'] / (features['har_rv_weekly'] + 1e-10)

                print(f"   HAR components: D={features['har_rv_daily'].iloc[-1]:.4f}, W={features['har_rv_weekly'].iloc[-1]:.4f}, M={features['har_rv_monthly'].iloc[-1]:.4f}")
            except Exception as e:
                print(f"   HAR calculation error: {e}")
                features['har_rv_daily'] = features['daily_range']
                features['har_rv_weekly'] = features['daily_range']
                features['har_rv_monthly'] = features['daily_range']
                features['har_weekly_daily_ratio'] = 1
                features['har_monthly_weekly_ratio'] = 1

            # --- 3. RANGE-BASED VOLATILITY ESTIMATORS ---
            # Source: Parkinson (1980), Garman-Klass (1980), Rogers-Satchell (1991)
            # More efficient than close-to-close volatility (already have Parkinson above)
            try:
                log_hl = np.log(df['high'] / df['low'])
                log_co = np.log(df['close'] / df['open'])
                log_ho = np.log(df['high'] / df['open'])
                log_lo = np.log(df['low'] / df['open'])
                log_hc = np.log(df['high'] / df['close'])
                log_lc = np.log(df['low'] / df['close'])

                # Garman-Klass estimator (uses O,H,L,C) - better for small drift
                gk_daily = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
                features['garman_klass_vol'] = np.sqrt(gk_daily.rolling(20).mean() * 252) * 100

                # Rogers-Satchell estimator (handles drift better)
                rs_daily = log_ho * log_hc + log_lo * log_lc
                features['rogers_satchell_vol'] = np.sqrt(rs_daily.rolling(20).mean() * 252) * 100

                # Estimator ratios (can indicate market behavior differences)
                features['gk_parkinson_ratio'] = features['garman_klass_vol'] / (features['parkinson_vol'] + 1e-10)
                features['rs_parkinson_ratio'] = features['rogers_satchell_vol'] / (features['parkinson_vol'] + 1e-10)

                print(f"   Range estimators: GK={features['garman_klass_vol'].iloc[-1]:.2f}%, RS={features['rogers_satchell_vol'].iloc[-1]:.2f}%")
            except Exception as e:
                print(f"   Range estimator calculation error: {e}")
                features['garman_klass_vol'] = features['parkinson_vol']
                features['rogers_satchell_vol'] = features['parkinson_vol']
                features['gk_parkinson_ratio'] = 1
                features['rs_parkinson_ratio'] = 1

            # --- 4. CARR FEATURES (Conditional Autoregressive Range) ---
            # Source: Chou (2005), "The Conditional Autoregressive Range Model"
            # Direct range modeling with autoregressive structure
            try:
                # More lagged range features (CARR-style)
                features['carr_range_ma5'] = features['daily_range'].rolling(5).mean()
                features['carr_range_ma10'] = features['daily_range'].rolling(10).mean()

                # Range momentum (is range expanding or contracting?)
                features['carr_range_momentum'] = features['daily_range'] - features['carr_range_ma5']

                # Return-range interaction (returns often predict range)
                features['carr_abs_return'] = abs(features['returns'])
                features['carr_return_range_corr'] = features['carr_abs_return'].rolling(20).corr(features['daily_range'])

                # Volume-range interaction (high volume often = high range)
                if 'volume' in df.columns:
                    vol_norm = df['volume'] / df['volume'].rolling(20).mean()
                    features['carr_volume_range_interaction'] = vol_norm * features['daily_range']

                print(f"   CARR features: range_momentum={features['carr_range_momentum'].iloc[-1]:.4f}")
            except Exception as e:
                print(f"   CARR calculation error: {e}")
                features['carr_range_ma5'] = features['daily_range']
                features['carr_range_ma10'] = features['daily_range']
                features['carr_range_momentum'] = 0
                features['carr_abs_return'] = 0
                features['carr_return_range_corr'] = 0
                features['carr_volume_range_interaction'] = 0

            # --- 5. VOLATILITY ENTROPY ---
            # Source: "Entropy of Volatility Changes" (2025)
            # Low entropy = predictable vol patterns, high entropy = chaotic
            try:
                from scipy.stats import entropy as scipy_entropy

                def calculate_vol_entropy(returns, bins=10):
                    """Calculate entropy of return distribution."""
                    if len(returns) < bins:
                        return 0.5
                    hist, _ = np.histogram(returns, bins=bins, density=True)
                    hist = hist + 1e-10  # Avoid log(0)
                    return scipy_entropy(hist) / np.log(bins)  # Normalize to 0-1

                # Rolling entropy on returns
                features['vol_entropy_20d'] = log_returns.rolling(20).apply(
                    lambda x: calculate_vol_entropy(x.values, 10), raw=False
                )
                features['vol_entropy_60d'] = log_returns.rolling(60).apply(
                    lambda x: calculate_vol_entropy(x.values, 15), raw=False
                )

                # Entropy change (rising entropy = increasing unpredictability)
                features['vol_entropy_change'] = features['vol_entropy_20d'].diff(5)

                print(f"   Volatility entropy: {features['vol_entropy_20d'].iloc[-1]:.3f}")
            except ImportError:
                print("   scipy.stats not available for entropy calculation")
                features['vol_entropy_20d'] = 0.5
                features['vol_entropy_60d'] = 0.5
                features['vol_entropy_change'] = 0
            except Exception as e:
                print(f"   Entropy calculation error: {e}")
                features['vol_entropy_20d'] = 0.5
                features['vol_entropy_60d'] = 0.5
                features['vol_entropy_change'] = 0

            # --- 6. IV-RV SPREAD (Implied vs Realized Volatility) ---
            # Source: Chen & Li (2023), "Why does IV forecast RV?"
            # IV > RV suggests market expects higher volatility
            try:
                # Use 20-day realized vol for comparison
                rv_20d = features['volatility_20d']  # Already annualized

                # If we have IV from options
                if 'iv_weighted' in features.columns:
                    features['iv_rv_spread'] = features['iv_weighted'] - rv_20d
                    features['iv_rv_ratio'] = features['iv_weighted'] / (rv_20d + 1e-10)

                    # IV premium (IV typically > RV, deviation is informative)
                    iv_rv_hist = features['iv_rv_spread'].rolling(60)
                    features['iv_rv_zscore'] = (features['iv_rv_spread'] - iv_rv_hist.mean()) / (iv_rv_hist.std() + 1e-10)

                    print(f"   IV-RV spread: {features['iv_rv_spread'].iloc[-1]:.2f}%")
                else:
                    features['iv_rv_spread'] = 0
                    features['iv_rv_ratio'] = 1
                    features['iv_rv_zscore'] = 0
            except Exception as e:
                print(f"   IV-RV calculation error: {e}")
                features['iv_rv_spread'] = 0
                features['iv_rv_ratio'] = 1
                features['iv_rv_zscore'] = 0

            # --- 7. RANGE EFFICIENCY RATIO ---
            # How much of the range was "used" by price movement (close-open vs high-low)
            try:
                features['range_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
                features['range_efficiency_ma5'] = features['range_efficiency'].rolling(5).mean()

                # Low efficiency = indecision (doji), often precedes breakout
                features['low_efficiency_signal'] = (features['range_efficiency'] < 0.3).astype(int)

                print(f"   Range efficiency: {features['range_efficiency'].iloc[-1]:.3f}")
            except Exception as e:
                print(f"   Range efficiency error: {e}")
                features['range_efficiency'] = 0.5
                features['range_efficiency_ma5'] = 0.5
                features['low_efficiency_signal'] = 0

            print("   Research-backed indicators complete.")
        else:
            print("   Scientific indicators DISABLED (FEATURE_CONFIG['scientific_features'] = False)")

        # =================================================================
        # CROSS-MARKET FEATURES (NEW - for enhanced range prediction)
        # These capture broader market dynamics that affect SPY volatility
        # =================================================================

        # --- 1. VVIX Features (VIX of VIX) ---
        # VVIX measures expected volatility of VIX itself
        # High VVIX = market expects VIX to move significantly
        if FEATURE_CONFIG.get('vvix_features', True):
            vvix_df = self.fetch_vvix_data()
            if not vvix_df.empty:
                try:
                    vvix_daily = vvix_df.copy()
                    vvix_daily.index = pd.to_datetime(vvix_daily.index).date
                    vvix_close = vvix_daily['close']

                    features['vvix'] = df.index.map(lambda x: vvix_close.get(x.date() if hasattr(x, 'date') else x, np.nan))
                    features['vvix'] = features['vvix'].ffill().bfill()

                    # VVIX derivatives
                    features['vvix_change_1d'] = features['vvix'].pct_change() * 100
                    features['vvix_change_5d'] = features['vvix'].pct_change(5) * 100
                    features['vvix_sma_10'] = features['vvix'].rolling(10).mean()
                    features['vvix_vs_sma'] = (features['vvix'] - features['vvix_sma_10']) / features['vvix_sma_10']
                    features['vvix_zscore'] = (features['vvix'] - features['vvix'].rolling(20).mean()) / features['vvix'].rolling(20).std()

                    # VVIX velocity and acceleration
                    features['vvix_velocity'] = features['vvix'].diff()
                    features['vvix_accel'] = features['vvix_velocity'].diff()

                    # VIX/VVIX ratio (fear of fear)
                    if 'vix' in features.columns and features['vix'].notna().any():
                        features['vix_vvix_ratio'] = features['vix'] / features['vvix']
                        features['vix_vvix_ratio_zscore'] = (
                            features['vix_vvix_ratio'] - features['vix_vvix_ratio'].rolling(20).mean()
                        ) / features['vix_vvix_ratio'].rolling(20).std()

                    print(f"   VVIX features added: current VVIX={features['vvix'].iloc[-1]:.2f}")
                except Exception as e:
                    print(f"   VVIX feature error: {e}")
                    for col in ['vvix', 'vvix_change_1d', 'vvix_change_5d', 'vvix_sma_10', 'vvix_vs_sma',
                               'vvix_zscore', 'vvix_velocity', 'vvix_accel', 'vix_vvix_ratio', 'vix_vvix_ratio_zscore']:
                        features[col] = 0
            else:
                for col in ['vvix', 'vvix_change_1d', 'vvix_change_5d', 'vvix_sma_10', 'vvix_vs_sma',
                           'vvix_zscore', 'vvix_velocity', 'vvix_accel', 'vix_vvix_ratio', 'vix_vvix_ratio_zscore']:
                    features[col] = 0

        # --- 2. Sector Breadth Features ---
        # Market breadth: how many sectors are participating in the move
        if FEATURE_CONFIG.get('sector_breadth_features', True):
            sector_data = self.fetch_sector_data()
            if sector_data:
                try:
                    # Calculate returns for each sector and count how many are positive/above SMA
                    sector_returns_1d = {}
                    sector_above_sma = {}

                    for etf, etf_df in sector_data.items():
                        try:
                            etf_daily = etf_df.copy()
                            etf_daily.index = pd.to_datetime(etf_daily.index).date
                            etf_close = etf_daily['close']

                            # Map to main df index
                            mapped = df.index.map(lambda x: etf_close.get(x.date() if hasattr(x, 'date') else x, np.nan))
                            mapped = pd.Series(mapped, index=df.index).ffill().bfill()

                            sector_returns_1d[etf] = mapped.pct_change() * 100
                            sector_above_sma[etf] = (mapped > mapped.rolling(20).mean()).astype(int)
                        except:
                            continue

                    if sector_returns_1d:
                        # Breadth: count of sectors with positive returns
                        returns_df = pd.DataFrame(sector_returns_1d)
                        features['sector_breadth_positive'] = (returns_df > 0).sum(axis=1)
                        features['sector_breadth_pct'] = features['sector_breadth_positive'] / len(sector_returns_1d)

                        # Average sector return
                        features['sector_avg_return'] = returns_df.mean(axis=1)

                        # Sector dispersion (std of returns - high = divergence)
                        features['sector_dispersion'] = returns_df.std(axis=1)

                        # Sectors above SMA20
                        sma_df = pd.DataFrame(sector_above_sma)
                        features['sector_above_sma_count'] = sma_df.sum(axis=1)
                        features['sector_above_sma_pct'] = features['sector_above_sma_count'] / len(sector_above_sma)

                        # Breadth momentum
                        features['breadth_momentum'] = features['sector_breadth_pct'].diff(5)

                        print(f"   Sector breadth features added: {len(sector_data)} sectors, breadth={features['sector_breadth_pct'].iloc[-1]:.2f}")
                    else:
                        for col in ['sector_breadth_positive', 'sector_breadth_pct', 'sector_avg_return',
                                   'sector_dispersion', 'sector_above_sma_count', 'sector_above_sma_pct', 'breadth_momentum']:
                            features[col] = 0
                except Exception as e:
                    print(f"   Sector breadth error: {e}")
                    for col in ['sector_breadth_positive', 'sector_breadth_pct', 'sector_avg_return',
                               'sector_dispersion', 'sector_above_sma_count', 'sector_above_sma_pct', 'breadth_momentum']:
                        features[col] = 0
            else:
                for col in ['sector_breadth_positive', 'sector_breadth_pct', 'sector_avg_return',
                           'sector_dispersion', 'sector_above_sma_count', 'sector_above_sma_pct', 'breadth_momentum']:
                    features[col] = 0

        # --- 3. Credit Spread Features (HYG/LQD) ---
        # Credit spreads widen when risk appetite decreases
        if FEATURE_CONFIG.get('credit_spread_features', True):
            credit_data = self.fetch_credit_data()
            if 'HYG' in credit_data and 'LQD' in credit_data:
                try:
                    hyg_df = credit_data['HYG'].copy()
                    lqd_df = credit_data['LQD'].copy()

                    hyg_df.index = pd.to_datetime(hyg_df.index).date
                    lqd_df.index = pd.to_datetime(lqd_df.index).date

                    hyg_close = hyg_df['close']
                    lqd_close = lqd_df['close']

                    # Map to main df index
                    hyg_mapped = df.index.map(lambda x: hyg_close.get(x.date() if hasattr(x, 'date') else x, np.nan))
                    lqd_mapped = df.index.map(lambda x: lqd_close.get(x.date() if hasattr(x, 'date') else x, np.nan))

                    hyg_series = pd.Series(hyg_mapped, index=df.index).ffill().bfill()
                    lqd_series = pd.Series(lqd_mapped, index=df.index).ffill().bfill()

                    # HYG/LQD ratio (high yield vs investment grade)
                    # Rising ratio = risk-on, falling = risk-off
                    features['hyg_lqd_ratio'] = hyg_series / lqd_series
                    features['hyg_lqd_change_1d'] = features['hyg_lqd_ratio'].pct_change() * 100
                    features['hyg_lqd_change_5d'] = features['hyg_lqd_ratio'].pct_change(5) * 100
                    features['hyg_lqd_zscore'] = (
                        features['hyg_lqd_ratio'] - features['hyg_lqd_ratio'].rolling(20).mean()
                    ) / features['hyg_lqd_ratio'].rolling(20).std()

                    # Credit spread velocity
                    features['credit_spread_velocity'] = features['hyg_lqd_ratio'].diff()
                    features['credit_spread_accel'] = features['credit_spread_velocity'].diff()

                    print(f"   Credit spread features added: HYG/LQD={features['hyg_lqd_ratio'].iloc[-1]:.4f}")
                except Exception as e:
                    print(f"   Credit spread error: {e}")
                    for col in ['hyg_lqd_ratio', 'hyg_lqd_change_1d', 'hyg_lqd_change_5d', 'hyg_lqd_zscore',
                               'credit_spread_velocity', 'credit_spread_accel']:
                        features[col] = 0
            else:
                for col in ['hyg_lqd_ratio', 'hyg_lqd_change_1d', 'hyg_lqd_change_5d', 'hyg_lqd_zscore',
                           'credit_spread_velocity', 'credit_spread_accel']:
                    features[col] = 0

        # --- 4. Treasury Yield Features ---
        # Yield curve dynamics affect equity volatility
        if FEATURE_CONFIG.get('treasury_features', True):
            treasury_data = self.fetch_treasury_data()
            if '10Y' in treasury_data:
                try:
                    y10_df = treasury_data['10Y'].copy()
                    y10_df.index = pd.to_datetime(y10_df.index).date
                    y10_close = y10_df['close']

                    # Map to main df index
                    y10_mapped = df.index.map(lambda x: y10_close.get(x.date() if hasattr(x, 'date') else x, np.nan))
                    y10_series = pd.Series(y10_mapped, index=df.index).ffill().bfill()

                    features['treasury_10y'] = y10_series
                    features['treasury_10y_change_1d'] = y10_series.diff()
                    features['treasury_10y_change_5d'] = y10_series.diff(5)
                    features['treasury_10y_zscore'] = (y10_series - y10_series.rolling(20).mean()) / y10_series.rolling(20).std()
                    features['treasury_10y_velocity'] = y10_series.diff()

                    # Yield curve slope (if 30Y available)
                    if '30Y' in treasury_data:
                        y30_df = treasury_data['30Y'].copy()
                        y30_df.index = pd.to_datetime(y30_df.index).date
                        y30_close = y30_df['close']
                        y30_mapped = df.index.map(lambda x: y30_close.get(x.date() if hasattr(x, 'date') else x, np.nan))
                        y30_series = pd.Series(y30_mapped, index=df.index).ffill().bfill()

                        features['treasury_30y'] = y30_series
                        features['yield_curve_slope'] = y30_series - y10_series  # 30Y - 10Y spread
                        features['yield_curve_slope_change'] = features['yield_curve_slope'].diff(5)
                    else:
                        features['treasury_30y'] = 0
                        features['yield_curve_slope'] = 0
                        features['yield_curve_slope_change'] = 0

                    print(f"   Treasury features added: 10Y={features['treasury_10y'].iloc[-1]:.2f}%")
                except Exception as e:
                    print(f"   Treasury feature error: {e}")
                    for col in ['treasury_10y', 'treasury_10y_change_1d', 'treasury_10y_change_5d',
                               'treasury_10y_zscore', 'treasury_10y_velocity', 'treasury_30y',
                               'yield_curve_slope', 'yield_curve_slope_change']:
                        features[col] = 0
            else:
                for col in ['treasury_10y', 'treasury_10y_change_1d', 'treasury_10y_change_5d',
                           'treasury_10y_zscore', 'treasury_10y_velocity', 'treasury_30y',
                           'yield_curve_slope', 'yield_curve_slope_change']:
                    features[col] = 0

        # --- 5. Dollar Index (DXY) Features ---
        # Strong dollar typically weighs on equities
        if FEATURE_CONFIG.get('dollar_features', True):
            dollar_df = self.fetch_dollar_data()
            if not dollar_df.empty:
                try:
                    dxy_daily = dollar_df.copy()
                    dxy_daily.index = pd.to_datetime(dxy_daily.index).date
                    dxy_close = dxy_daily['close']

                    # Map to main df index
                    dxy_mapped = df.index.map(lambda x: dxy_close.get(x.date() if hasattr(x, 'date') else x, np.nan))
                    dxy_series = pd.Series(dxy_mapped, index=df.index).ffill().bfill()

                    features['dxy'] = dxy_series
                    features['dxy_change_1d'] = dxy_series.pct_change() * 100
                    features['dxy_change_5d'] = dxy_series.pct_change(5) * 100
                    features['dxy_sma_20'] = dxy_series.rolling(20).mean()
                    features['dxy_vs_sma'] = (dxy_series - features['dxy_sma_20']) / features['dxy_sma_20']
                    features['dxy_zscore'] = (dxy_series - dxy_series.rolling(20).mean()) / dxy_series.rolling(20).std()

                    # Dollar velocity and acceleration
                    features['dxy_velocity'] = dxy_series.diff()
                    features['dxy_accel'] = features['dxy_velocity'].diff()

                    print(f"   Dollar features added: DXY={features['dxy'].iloc[-1]:.2f}")
                except Exception as e:
                    print(f"   Dollar feature error: {e}")
                    for col in ['dxy', 'dxy_change_1d', 'dxy_change_5d', 'dxy_sma_20', 'dxy_vs_sma',
                               'dxy_zscore', 'dxy_velocity', 'dxy_accel']:
                        features[col] = 0
            else:
                for col in ['dxy', 'dxy_change_1d', 'dxy_change_5d', 'dxy_sma_20', 'dxy_vs_sma',
                           'dxy_zscore', 'dxy_velocity', 'dxy_accel']:
                    features[col] = 0

        # --- 6. SPY/QQQ Correlation Features ---
        # High correlation = normal market, low correlation = divergence/rotation
        if FEATURE_CONFIG.get('correlation_features', True):
            qqq_df = self.fetch_qqq_data()
            if not qqq_df.empty:
                try:
                    qqq_daily = qqq_df.copy()
                    qqq_daily.index = pd.to_datetime(qqq_daily.index).date
                    qqq_close = qqq_daily['close']

                    # Map to main df index
                    qqq_mapped = df.index.map(lambda x: qqq_close.get(x.date() if hasattr(x, 'date') else x, np.nan))
                    qqq_series = pd.Series(qqq_mapped, index=df.index).ffill().bfill()

                    # QQQ returns
                    qqq_returns = qqq_series.pct_change()
                    spy_returns = df['close'].pct_change()

                    # Rolling correlation
                    features['spy_qqq_corr_10d'] = spy_returns.rolling(10).corr(qqq_returns)
                    features['spy_qqq_corr_20d'] = spy_returns.rolling(20).corr(qqq_returns)

                    # Correlation change (sudden drops often precede volatility)
                    features['spy_qqq_corr_change'] = features['spy_qqq_corr_20d'].diff(5)

                    # SPY/QQQ relative strength
                    features['spy_qqq_ratio'] = df['close'] / qqq_series
                    features['spy_qqq_ratio_zscore'] = (
                        features['spy_qqq_ratio'] - features['spy_qqq_ratio'].rolling(20).mean()
                    ) / features['spy_qqq_ratio'].rolling(20).std()

                    # Return divergence (SPY - QQQ return)
                    features['spy_qqq_return_diff'] = (spy_returns - qqq_returns) * 100

                    print(f"   Correlation features added: SPY/QQQ corr={features['spy_qqq_corr_20d'].iloc[-1]:.3f}")
                except Exception as e:
                    print(f"   Correlation feature error: {e}")
                    for col in ['spy_qqq_corr_10d', 'spy_qqq_corr_20d', 'spy_qqq_corr_change',
                               'spy_qqq_ratio', 'spy_qqq_ratio_zscore', 'spy_qqq_return_diff']:
                        features[col] = 0
            else:
                for col in ['spy_qqq_corr_10d', 'spy_qqq_corr_20d', 'spy_qqq_corr_change',
                           'spy_qqq_ratio', 'spy_qqq_ratio_zscore', 'spy_qqq_return_diff']:
                    features[col] = 0

        # --- 7. Earnings Calendar Features ---
        # Big tech earnings drive SPY volatility
        if FEATURE_CONFIG.get('earnings_features', True):
            try:
                # Calculate days until next earnings for top holdings
                # This is more useful for live prediction - for training, we use a simple proxy
                # High earnings activity period = higher expected volatility

                # For now, use month/quarter-end as earnings proxy
                # Most earnings are in Jan/Apr/Jul/Oct
                earnings_months = [1, 4, 7, 10]
                features['earnings_month'] = df.index.month.isin(earnings_months).astype(int)

                # Week of month (earnings usually weeks 2-4 of earnings months)
                features['week_of_month'] = (df.index.day - 1) // 7 + 1
                features['earnings_week'] = (
                    features['earnings_month'] &
                    (features['week_of_month'].isin([2, 3, 4]))
                ).astype(int)

                # Quarterly expiration weeks (triple witching) - high volatility
                # Third Friday of Mar/Jun/Sep/Dec
                features['opex_month'] = df.index.month.isin([3, 6, 9, 12]).astype(int)
                features['opex_week'] = (
                    features['opex_month'] &
                    (features['week_of_month'] == 3)
                ).astype(int)

                print(f"   Earnings features added (calendar-based)")
            except Exception as e:
                print(f"   Earnings feature error: {e}")
                for col in ['earnings_month', 'week_of_month', 'earnings_week', 'opex_month', 'opex_week']:
                    features[col] = 0

        # =================================================================
        # AUTO-DISCOVERED HIGH-CORRELATION FEATURES
        # From automated feature discovery testing 140+ combinations
        # These features showed significantly higher correlation with targets
        # =================================================================
        print("   Adding auto-discovered high-correlation features...")

        # 1. range_mean_3d: ALREADY DEFINED EARLY (line 766) for RSI-normalized features
        # Short-term range momentum - very predictive of next day's range

        # 2. range_momentum_high_vol: Range momentum in high volatility regime (+0.775 correlation)
        # Range * ROC interaction, but only during high VIX periods
        # High vol regime makes range momentum much more predictive
        if 'vix_regime_high' in features.columns:
            features['range_momentum_high_vol'] = features['range_momentum'] * features['vix_regime_high']
        else:
            # Fallback: use volatility_20d > 80th percentile
            vol_high = (features['volatility_20d'] > features['volatility_20d'].rolling(60).quantile(0.8)).astype(int)
            features['range_momentum_high_vol'] = features['range_momentum'] * vol_high

        # 3. upside_potential_high_vol: Upside potential in high volatility regime (+0.746 correlation)
        # Room to run higher, but only matters in high vol environment
        if 'vix_regime_high' in features.columns:
            features['upside_potential_high_vol'] = features['upside_potential'] * features['vix_regime_high']
        else:
            vol_high = (features['volatility_20d'] > features['volatility_20d'].rolling(60).quantile(0.8)).astype(int)
            features['upside_potential_high_vol'] = features['upside_potential'] * vol_high

        # 4. upside_potential_x_price_vs_sma50: Interaction term (+0.656 correlation)
        # Upside potential is more meaningful when price is above/below 50 SMA
        # Above SMA50 + high upside potential = strong bullish, wider high range expected
        features['upside_potential_x_price_vs_sma50'] = features['upside_potential'] * features['price_vs_sma50']

        # Additional discovered interactions from correlation analysis
        # 5. downside_potential_high_vol: Symmetric to upside for bearish setups
        if 'vix_regime_high' in features.columns:
            features['downside_potential_high_vol'] = features['downside_potential'] * features['vix_regime_high']
        else:
            vol_high = (features['volatility_20d'] > features['volatility_20d'].rolling(60).quantile(0.8)).astype(int)
            features['downside_potential_high_vol'] = features['downside_potential'] * vol_high

        # 6. range_mean_3d_high_vol: Short-term range in high vol regime
        if 'vix_regime_high' in features.columns:
            features['range_mean_3d_high_vol'] = features['range_mean_3d'] * features['vix_regime_high']
        else:
            vol_high = (features['volatility_20d'] > features['volatility_20d'].rolling(60).quantile(0.8)).astype(int)
            features['range_mean_3d_high_vol'] = features['range_mean_3d'] * vol_high

        # =================================================================
        # ADDITIONAL AUTO-DISCOVERED FEATURES (Round 2)
        # From comprehensive feature discovery testing 140+ combinations
        # =================================================================

        # Get vol_high for regime features (reuse if already computed)
        if 'vix_regime_high' in features.columns:
            vol_high = features['vix_regime_high']
        else:
            vol_high = (features['volatility_20d'] > features['volatility_20d'].rolling(60).quantile(0.8)).astype(int)

        # 7. Triple interaction: upside_potential × price_vs_sma50 × high_vol (-0.796 correlation!)
        # Strongest predictor found - captures regime-dependent trend exhaustion
        features['upside_potential_x_sma50_high_vol'] = (
            features['upside_potential'] * features['price_vs_sma50'] * vol_high
        )

        # 8. Double regime: range_momentum_high_vol × high_vol again (+0.786)
        # Amplifies signal during extremely high vol (VIX spikes)
        features['range_momentum_double_high_vol'] = features['range_momentum_high_vol'] * vol_high

        # 9. Double regime: upside_potential_high_vol × high_vol (+0.733)
        features['upside_potential_double_high_vol'] = features['upside_potential_high_vol'] * vol_high

        # 10-11. Rolling max and mean features - ALREADY DEFINED EARLY (lines 768-773)
        # range_max_3d, range_max_5d, range_mean_7d moved to line 768 for RSI-normalized features

        # 12. Interaction difference: momentum vs trend signal (+0.643)
        # Captures divergence between range momentum and trend position
        features['momentum_trend_divergence'] = (
            features['range_momentum_high_vol'] - features['upside_potential_x_price_vs_sma50']
        )

        # 12b. NEW TOP FEATURE: momentum_trend_divergence × high_vol (+0.811 correlation)
        # This was discovered as the #1 new feature in auto-discovery
        # Captures divergence amplified during high volatility regimes
        features['momentum_trend_divergence_high_vol'] = features['momentum_trend_divergence'] * vol_high

        # 13. Volume level features - 3-week volume regime (+0.503)
        if 'volume' in df.columns:
            vol_3w_avg = df['volume'].rolling(15).mean()
            vol_3w_std = df['volume'].rolling(15).std()
            features['vol_level_3w'] = (df['volume'] - vol_3w_avg) / (vol_3w_std + 1)
            features['vol_level_3w_zscore'] = features['vol_level_3w'].rolling(5).mean()

        # 14. Range acceleration - 2nd derivative of range (rate of change of range momentum)
        features['range_accel'] = features['range_mean_3d'].diff().diff()
        features['range_accel_high_vol'] = features['range_accel'] * vol_high

        # 15. Lag features for top performers (high correlation with lags)
        features['upside_potential_x_sma50_lag1'] = features['upside_potential_x_price_vs_sma50'].shift(1)
        features['range_momentum_high_vol_lag1'] = features['range_momentum_high_vol'].shift(1)

        # =================================================================
        # DOWNSIDE-SPECIFIC FEATURES
        # These are designed to improve low predictions which typically have
        # worse R² than high predictions due to market asymmetry
        # =================================================================

        # 16. Drawdown from recent high - how far has price fallen
        # Larger drawdowns often precede larger downside range
        rolling_high_20d = df['high'].rolling(20).max()
        features['drawdown_from_high_20d'] = (df['close'] - rolling_high_20d) / rolling_high_20d * 100

        # 17. Down days ratio - what % of recent days were down
        # Higher ratio suggests continued selling pressure
        is_down_day = (df['close'] < df['open']).astype(int)
        features['down_days_ratio_5d'] = is_down_day.rolling(5).mean()
        features['down_days_ratio_10d'] = is_down_day.rolling(10).mean()

        # 18. Gap indicators - gaps often indicate fear/panic
        features['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
        features['gap_down'] = (features['gap_pct'] < 0).astype(int) * features['gap_pct'].abs()

        # 19. Downside momentum - rate of decline
        features['downside_momentum_3d'] = features['downside_potential'].diff(3)
        features['downside_momentum_high_vol'] = features['downside_momentum_3d'] * vol_high

        # 20. Lower low indicator - consecutive lower lows
        is_lower_low = (df['low'] < df['low'].shift(1)).astype(int)
        features['lower_low_streak'] = is_lower_low.rolling(5).sum()

        # 21. Close relative to day's range - near low suggests fear
        day_range = df['high'] - df['low']
        features['close_position_in_range'] = (df['close'] - df['low']) / (day_range + 0.001)

        # 22. VIX acceleration - sudden VIX spikes precede large lows
        if 'vix' in features.columns and features['vix'].notna().any():
            features['vix_accel'] = features['vix'].diff().diff()
            features['vix_accel_high_vol'] = features['vix_accel'] * vol_high

        # =================================================================
        # FEAR/PANIC FEATURES (To specifically improve LOW predictions)
        # Analysis showed LOW model has R² = -0.17 vs HIGH R² = 0.23
        # These features capture fear dynamics missing from the model
        # =================================================================

        # 23. Fear Index - combination of VIX spike and volume spike
        # Panic selling shows both high VIX velocity and abnormal volume
        if 'vix' in features.columns and features['vix'].notna().any():
            vix_velocity = features['vix'].diff().fillna(0)
            volume_spike_indicator = (features['volume_zscore'] > 1.5).astype(float)
            features['fear_index'] = vix_velocity * volume_spike_indicator
            features['fear_index_smooth'] = features['fear_index'].rolling(3).mean()

        # 24. Panic Signal - 3+ consecutive down days triggers panic mode
        features['panic_signal'] = (features['consecutive_down'] >= 3).astype(int)
        features['panic_intensity'] = features['consecutive_down'] * features['down_days_ratio_5d']

        # 25. Gap Down Magnitude - large gap downs signal fear
        features['gap_down_magnitude'] = features['gap_pct'].clip(upper=0).abs()
        features['gap_down_cumulative_3d'] = features['gap_down_magnitude'].rolling(3).sum()

        # 26. Capitulation Signal - extreme volume + down day + VIX spike
        extreme_volume = (features['volume_zscore'] > 2.0).astype(float)
        down_day = (df['close'] < df['open']).astype(float)
        if 'vix' in features.columns and features['vix'].notna().any():
            vix_spike = (features['vix_zscore'] > 1.5).astype(float)
            features['capitulation_signal'] = extreme_volume * down_day * vix_spike
        else:
            features['capitulation_signal'] = extreme_volume * down_day

        # 27. Sell Pressure - combination of indicators suggesting selling
        features['sell_pressure'] = (
            features['down_days_ratio_5d'] * 0.3 +
            features['lower_low_streak'] / 5 * 0.3 +
            (1 - features['close_position_in_range']) * 0.4
        )

        # 28. RSI-adjusted downside (low RSI + high range = more downside coming)
        if 'rsi_14' in features.columns:
            rsi_low_indicator = (features['rsi_14'] < 35).astype(float)
            features['rsi_adjusted_downside'] = features['downside_potential'] * (1 + rsi_low_indicator)
            features['oversold_range'] = features['daily_range'] * rsi_low_indicator

        # 29. Breakdown Signal - price below multiple support levels
        sma_20_below = (df['close'] < features['sma_20']).astype(float)
        sma_50_below = (df['close'] < features['sma_50']).astype(float) if 'sma_50' in features.columns else 0
        features['breakdown_signal'] = sma_20_below + sma_50_below

        # 30. Range Expansion in Downtrend - range expands more in fear
        features['downtrend_range_expansion'] = features['is_downtrend'] * features['range_zscore']

        # =================================================================
        # MULTI-TIMEFRAME FEAR FEATURES (Different lookback periods)
        # Short-term (1-3d): Immediate panic detection
        # Medium-term (5-10d): Sustained selling pressure
        # Longer-term (20d): Regime change detection
        # =================================================================

        # Gap down cumulative at different horizons
        features['gap_down_cumulative_5d'] = features['gap_down_magnitude'].rolling(5).sum()
        features['gap_down_cumulative_10d'] = features['gap_down_magnitude'].rolling(10).sum()

        # Down days ratio at multiple lookbacks
        is_down_day = (df['close'] < df['open']).astype(int)
        features['down_days_ratio_3d'] = is_down_day.rolling(3).mean()
        features['down_days_ratio_10d'] = is_down_day.rolling(10).mean()
        features['down_days_ratio_20d'] = is_down_day.rolling(20).mean()

        # Fear index at different smoothing levels
        if 'vix' in features.columns and features['vix'].notna().any():
            features['fear_index_smooth_5d'] = features['fear_index'].rolling(5).mean()
            features['fear_index_smooth_10d'] = features['fear_index'].rolling(10).mean()
            features['fear_index_max_5d'] = features['fear_index'].rolling(5).max()

        # Panic signal variants (different thresholds)
        features['panic_signal_2d'] = (features['consecutive_down'] >= 2).astype(int)
        features['panic_signal_4d'] = (features['consecutive_down'] >= 4).astype(int)
        features['panic_signal_5d'] = (features['consecutive_down'] >= 5).astype(int)

        # Sell pressure at different horizons
        features['sell_pressure_3d'] = (
            features['down_days_ratio_3d'] * 0.3 +
            features['lower_low_streak'] / 3 * 0.3 +
            (1 - features['close_position_in_range']) * 0.4
        )
        features['sell_pressure_10d'] = (
            features['down_days_ratio_10d'] * 0.3 +
            features['lower_low_streak'] / 5 * 0.3 +
            (1 - features['close_position_in_range']) * 0.4
        )

        # VIX velocity at different lookbacks
        if 'vix' in features.columns and features['vix'].notna().any():
            features['vix_velocity_1d'] = features['vix'].diff(1)
            features['vix_velocity_3d'] = features['vix'].diff(3)
            features['vix_velocity_10d'] = features['vix'].diff(10)
            features['vix_acceleration_3d'] = features['vix_velocity_3d'].diff()

        # Drawdown velocity (how fast is price falling from high)
        features['drawdown_velocity_3d'] = features['drawdown_from_high_20d'].diff(3)
        features['drawdown_velocity_5d'] = features['drawdown_from_high_20d'].diff(5)
        features['drawdown_accel'] = features['drawdown_velocity_3d'].diff()

        # Volume-weighted fear (fear matters more with high volume)
        features['vol_weighted_panic'] = features['panic_intensity'] * features['volume_rel']
        features['vol_weighted_sell_pressure'] = features['sell_pressure'] * features['volume_rel']

        # Lower low streak at different windows
        is_lower_low = (df['low'] < df['low'].shift(1)).astype(int)
        features['lower_low_streak_3d'] = is_lower_low.rolling(3).sum()
        features['lower_low_streak_10d'] = is_lower_low.rolling(10).sum()

        # Close position in range - rolling versions
        features['close_pos_min_5d'] = features['close_position_in_range'].rolling(5).min()
        features['close_pos_mean_5d'] = features['close_position_in_range'].rolling(5).mean()

        print(f"   Fear/panic features added: 40+ new features for LOW prediction")
        print(f"   Downside-specific features added: 10 new features for low prediction")
        print(f"   Auto-discovered features added (Round 2): {len([c for c in features.columns if 'double_high_vol' in c or 'range_max' in c or 'range_mean_7d' in c])} new features")

        # --- Lagged features (configurable - adds ~12 features, low importance) ---
        if FEATURE_CONFIG.get('lag_features', False):
            for lag in [1, 2, 3]:
                features[f'range_lag_{lag}'] = features['daily_range'].shift(lag)
                features[f'atr_14_lag_{lag}'] = features['atr_14'].shift(lag)

            # VIX lagged features (if available)
            if 'vix' in features.columns and features['vix'].notna().any():
                for lag in [1, 2]:
                    features[f'vix_lag_{lag}'] = features['vix'].shift(lag)

        # --- Deep Learning Features (CNN+LSTM embeddings) ---
        if FEATURE_CONFIG.get('dl_features', False) and DL_FEATURE_AVAILABLE:
            try:
                print("   Generating deep learning embeddings...")

                # Initialize DL extractor with 60-day lookback (enables 2 CNN branches)
                # and 12 embedding dims for richer representation
                dl_extractor = DLFeatureExtractor(sequence_length=60, encoding_dim=12)

                # EXPANDED input features for DL model - include more informative signals
                dl_input_cols = [
                    # Core volatility & range features
                    'atr_14_pct', 'volatility_5d', 'volatility_20d', 'vol_ratio',
                    'daily_range_pct', 'range_zscore', 'range_mean_5d', 'range_std_5d',
                    # Momentum indicators
                    'rsi_14', 'roc_5', 'roc_10', 'roc_20',
                    # Price position features
                    'volume_rel', 'price_vs_sma20', 'price_vs_sma50',
                    'price_position', 'upside_potential', 'drawdown_from_high_20d',
                    # Fear/sentiment features
                    'consecutive_down', 'sell_pressure', 'panic_intensity'
                ]
                dl_input_cols = [c for c in dl_input_cols if c in features.columns]

                if len(dl_input_cols) >= 8:  # Need at least 8 features
                    # Create DL embeddings with more epochs for better training
                    dl_embeddings = dl_extractor.generate_embeddings(
                        features[dl_input_cols],
                        epochs=20,  # Increased from 10 for better learning
                        verbose=0,
                        train=True
                    )

                    # Fix NaN/Inf issues in DL embeddings
                    dl_embeddings = dl_embeddings.replace([np.inf, -np.inf], np.nan)
                    dl_embeddings = dl_embeddings.fillna(0)
                    # Clip extreme values to prevent numerical issues
                    for col in dl_embeddings.columns:
                        dl_embeddings[col] = dl_embeddings[col].clip(-10, 10)

                    # Add to features (DL_pred_0 through DL_pred_11 with 12 dims)
                    for col in dl_embeddings.columns:
                        features[col] = dl_embeddings[col]

                    print(f"   DL embeddings added: {len(dl_embeddings.columns)} features "
                          f"(seq_len=60, input_features={len(dl_input_cols)})")

                    # Collect for summary
                    _training_summary.data['dl_samples'] = len(dl_embeddings)

                    # Store the extractor for later inference
                    self._dl_extractor = dl_extractor
                else:
                    print(f"   DL features skipped: insufficient input features ({len(dl_input_cols)}/8 required)")
            except Exception as dl_err:
                print(f"   DL feature error: {dl_err}")
                # Don't fail - just skip DL features

        # =================================================================
        # COMPOSITE SUPER-FEATURES (Fix for XGBoost collinearity issue)
        # XGBoost arbitrarily picks ONE from correlated feature groups.
        # These super-features COMBINE the top correlated features so
        # the model CANNOT ignore their collective predictive signal.
        # Placed here (at end) to ensure all dependencies are defined.
        # =================================================================

        # Super-feature 1: Range-Upside Composite (top 3 range features)
        # Combines: range_plus_upside (0.68), range_times_upside (0.68), daily_range_plus_upside (0.68)
        features['super_range_upside'] = (
            features['range_plus_upside'] +
            features['range_times_upside'] +
            features['daily_range_plus_upside']
        ) / 3

        # Super-feature 2: Vol-Regime Composite (vol regime features)
        # Combines: vol_regime_minus_position (0.68), vol_regime_minus_trend (0.65), vol_regime_interaction (0.64)
        features['super_vol_regime'] = (
            features['vol_regime_minus_position'] +
            features['vol_regime_minus_trend'] +
            features['vol_regime_interaction']
        ) / 3

        # Super-feature 3: Upside-ATR Composite (upside * volatility features)
        # Combines: upside_times_atr7 (0.67), upside_times_atr14 (0.67), upside_plus_range_mean20 (0.67)
        features['super_upside_vol'] = (
            features['upside_times_atr7'] +
            features['upside_times_atr14'] +
            features['upside_plus_range_mean20']
        ) / 3

        # Super-feature 4: Master Composite (weighted average of all top features)
        # This is the "nuclear option" - combines ALL top correlated features
        features['super_master'] = (
            features['vol_regime_minus_position'] * 0.15 +
            features['range_plus_upside'] * 0.15 +
            features['range_times_upside'] * 0.10 +
            features['upside_times_atr7'] * 0.10 +
            features['upside_times_atr14'] * 0.10 +
            features['range_mean_3d'] * 0.10 +
            features['momentum_trend_divergence'] * 0.10 +
            features['vol_regime_interaction'] * 0.10 +
            features['upside_potential'] * 0.10
        )

        # Standardize super-features for better model performance
        for sf in ['super_range_upside', 'super_vol_regime', 'super_upside_vol', 'super_master']:
            sf_mean = features[sf].rolling(20).mean()
            sf_std = features[sf].rolling(20).std()
            features[f'{sf}_zscore'] = (features[sf] - sf_mean) / (sf_std + 1e-8)

        print(f"   Super-features added: 8 composite features (4 raw + 4 z-scored)")

        # --- Feature Pruning (remove zero-importance features) ---
        if FEATURE_CONFIG.get('prune_zero_importance', True):
            pruned_count = 0
            for feat_to_remove in ZERO_IMPORTANCE_FEATURES:
                if feat_to_remove in features.columns:
                    features.drop(columns=[feat_to_remove], inplace=True)
                    pruned_count += 1
            if pruned_count > 0:
                print(f"   Pruned {pruned_count} zero-importance features")
                _training_summary.data['features_pruned_zero_imp'] = pruned_count

        # Store feature names (excluding target-related columns)
        self.feature_names = [col for col in features.columns
                             if col not in ['close', 'daily_range', 'daily_range_pct']]

        print(f"   Total features created: {len(self.feature_names)}")

        return features

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for training.

        Returns DataFrame with:
        - next_range: Next day's (high-low)/close as FRACTION (0.01 = 1%)
        - next_range_pct: Next day's range as PERCENTAGE (1.0 = 1%) - USE THIS FOR TRAINING
        - next_high_pct: Next day's high as % above current close
        - next_low_pct: Next day's low as % below current close

        IMPORTANT: XGBoost struggles with tiny target values (0.01-0.03).
        Use next_range_pct for training and divide by 100 for prediction.
        """
        targets = pd.DataFrame(index=df.index)

        # Next day's range as fraction
        targets['next_range'] = ((df['high'].shift(-1) - df['low'].shift(-1)) / df['close']).shift(0)

        # Next day's range as PERCENTAGE (scaled up for better XGBoost learning)
        targets['next_range_pct'] = targets['next_range'] * 100

        # Next day's high/low relative to current close
        targets['next_high_pct'] = ((df['high'].shift(-1) - df['close']) / df['close']) * 100
        targets['next_low_pct'] = ((df['close'] - df['low'].shift(-1)) / df['close']) * 100

        return targets

    def train_range_model(self, df: pd.DataFrame, options_features: Dict = None,
                          n_trials: int = 30, n_workers: int = None, progress_callback=None,
                          feature_selection: bool = True, optimize_highlow: bool = True) -> Dict:
        """
        Train XGBRegressor to predict next day's range.

        Args:
            df: OHLCV DataFrame with sufficient history
            options_features: Optional options data
            n_trials: Optuna trials for hyperparameter tuning
            n_workers: Number of parallel workers for Optuna (default: CPU cores - 1)
            progress_callback: Optional callback for progress updates
            feature_selection: Whether to apply feature selection (default True, disable for walk-forward stability)
            optimize_highlow: Whether to run separate optimization for HIGH/LOW models (default True).
                              Set to False for faster training (~40% speedup) at slight accuracy cost.

        Returns:
            dict with model, metrics, feature_importances
        """
        import os
        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 4) - 1)
        if not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
            raise ImportError("XGBoost and sklearn required for training")

        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Reset training summary for this run
        _training_summary.reset()
        _training_summary.data['total_rows'] = len(df)

        # Create features and targets
        features = self.create_range_features(df, options_features)
        targets = self.create_targets(df)

        # Collect feature health data for summary
        _training_summary.data['features_created'] = len(self.feature_names)
        if 'vix' in features.columns:
            _training_summary.data['vix_value'] = features['vix'].iloc[-1] if not features['vix'].isna().all() else 0
        if 'dxy' in features.columns:
            _training_summary.data['dxy_value'] = features['dxy'].iloc[-1] if not features['dxy'].isna().all() else 0
        # Check IV fallback (if iv_weighted == volatility_20d, it's fallback)
        if 'iv_weighted' in features.columns and 'volatility_20d' in features.columns:
            iv_match = (features['iv_weighted'] == features['volatility_20d']).mean() * 100
            _training_summary.data['iv_fallback_pct'] = iv_match

        # Align and clean
        # IMPORTANT: Use next_range_pct (percentage) for training - XGBoost struggles with tiny values
        X = features[self.feature_names].copy()
        y = targets['next_range_pct'].copy()  # Use PERCENTAGE target (1.0 = 1%), not fraction (0.01)

        # Remove NaN
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        # Time series split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Collect sample counts for summary
        _training_summary.data['training_samples'] = len(X_train)
        _training_summary.data['test_samples'] = len(X_test)

        # FEATURE SELECTION - DISABLED BY DEFAULT
        # Testing showed ALL FEATURES performs BEST:
        #   OLD (33→30 features):  R² = 0.256
        #   MEDIUM (167→109):      R² = 0.368
        #   ALL (214→214):         R² = 0.423  (+65% vs OLD!)
        #
        # XGBoost has built-in regularization (reg_alpha, reg_lambda, max_depth)
        # that handles high-dimensional data well. Feature selection was causing underfitting.
        n_train = len(X_train)
        n_features_total = len(self.feature_names)

        # Compute correlations for reference/debugging only
        correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
        self.feature_correlations = correlations

        # =================================================================
        # CORRELATION-BASED FEATURE PRUNING
        # Remove redundant features that are >0.95 correlated with each other
        # This prevents XGBoost from arbitrarily picking one and ignoring others
        # =================================================================
        print(f"\n[FEATURE PRUNING] Removing redundant highly-correlated features...")

        # Compute feature-feature correlation matrix
        feature_corr_matrix = X_train.corr().abs()

        # Find features to drop (keep higher target-correlated one from each pair)
        features_to_drop = set()
        feature_list = list(X_train.columns)

        for i, feat_i in enumerate(feature_list):
            if feat_i in features_to_drop:
                continue
            for j, feat_j in enumerate(feature_list[i+1:], i+1):
                if feat_j in features_to_drop:
                    continue
                # Check if features are highly correlated with each other
                if feature_corr_matrix.loc[feat_i, feat_j] > 0.95:
                    # Keep the one with higher target correlation
                    corr_i = correlations.get(feat_i, 0)
                    corr_j = correlations.get(feat_j, 0)
                    if corr_i >= corr_j:
                        features_to_drop.add(feat_j)
                    else:
                        features_to_drop.add(feat_i)

        # Apply pruning
        if len(features_to_drop) > 0:
            print(f"   Dropping {len(features_to_drop)} redundant features (>0.95 inter-correlation)")
            top_features = [f for f in self.feature_names if f not in features_to_drop]
        else:
            top_features = self.feature_names

        self.correlation_selected_features = top_features
        self.training_feature_names = top_features
        self.dropped_redundant_features = list(features_to_drop)

        print(f"[FEATURES] Using {len(top_features)} features (pruned {len(features_to_drop)} redundant)")
        print(f"   Training samples: {n_train}")
        print(f"   Top 5 by correlation: {correlations.head(5).index.tolist()}")
        print(f"   Correlations: {[f'{correlations[f]:.3f}' for f in correlations.head(5).index]}")

        # Collect for summary
        _training_summary.data['features_pruned_corr'] = len(features_to_drop)
        _training_summary.data['features_used'] = len(top_features)
        _training_summary.data['top_features_corr'] = [(f, correlations[f]) for f in correlations.head(10).index]

        # Apply feature pruning to training/test data
        X_train = X_train[top_features]
        X_test = X_test[top_features]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert to numpy for efficient pickling (critical for multiprocessing!)
        y_train_np = y_train.values if hasattr(y_train, 'values') else np.array(y_train)

        # Use file-based data sharing for proper multiprocessing (like working Optuna code)
        if not OPTUNA_WORKER_AVAILABLE:
            raise RuntimeError("optuna_worker module required for parallel optimization")

        from joblib import Parallel, delayed
        import time as time_module

        print(f"   Using joblib Parallel for true multiprocessing...")
        # Pass feature names for Optuna feature group selection
        feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else self.feature_names
        data_path = optuna_worker.set_range_shared_data(X_train_scaled, y_train_np, feature_names=feature_names)
        print(f"   Feature names passed to optimizer: {len(feature_names)} features")

        # Divide trials among workers (like velocity optimization)
        trials_per_worker = max(1, n_trials // n_workers)
        actual_trials = trials_per_worker * n_workers

        print(f"Starting Optuna optimization: {actual_trials} trials across {n_workers} parallel workers...")
        print(f"   Data shape: X={X_train_scaled.shape}, y={len(y_train_np)}")
        print(f"   Trials per worker: {trials_per_worker}")

        start_opt = time_module.time()

        try:
            # Run multiple independent studies in parallel using joblib (like velocity optimization)
            results_list = Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
                delayed(optuna_worker.run_range_study)(
                    data_path, trials_per_worker, seed=42 + i, worker_id=i
                )
                for i in range(n_workers)
            )
        finally:
            # Clean up shared data file
            optuna_worker.clear_range_shared_data()

        total_time = time_module.time() - start_opt

        # Find best result across all workers (MAXIMIZE composite score)
        # best_value is negative, so min() finds the best (most negative = highest positive score)
        best_result = min(results_list, key=lambda x: x['best_value'])
        total_completed = sum(r['n_trials'] for r in results_list)

        best_composite_score = best_result.get('best_score', -best_result['best_value'])
        print(f"Optimization complete!")
        print(f"   Best Composite Score: {best_composite_score:.4f} (0.6*R² + 0.4*MAPE_component)")
        print(f"   Features used: {X_train_scaled.shape[1]} (all)")
        print(f"   Total time: {total_time:.1f}s ({total_completed/total_time:.1f} trials/sec)")

        # Collect for summary
        _training_summary.data['range_best_score'] = best_composite_score
        _training_summary.data['range_trials'] = total_completed
        _training_summary.data['range_opt_time'] = total_time
        _training_summary.data['range_all_scores'] = [r.get('best_score', -r['best_value']) for r in results_list]

        # Collect all top trials from all workers for ensemble
        all_top_trials = []
        for result in results_list:
            if 'top_trials' in result:
                all_top_trials.extend(result['top_trials'])

        # Sort by score (descending) and take top 3 globally
        all_top_trials = sorted(all_top_trials, key=lambda x: x['score'], reverse=True)
        ensemble_size = min(3, len(all_top_trials))
        top_3_trials = all_top_trials[:ensemble_size]

        print(f"\n[ENSEMBLE] Creating ensemble from top {ensemble_size} trials:")
        for i, trial in enumerate(top_3_trials):
            print(f"   Model {i+1}: Score={trial['score']:.4f}")

        # Train ensemble of models
        self.ensemble_models = []
        for i, trial in enumerate(top_3_trials):
            params = trial['params'].copy()
            params['n_jobs'] = 1
            params['objective'] = 'reg:squarederror'
            params['verbosity'] = 0

            model = xgb.XGBRegressor(**params)
            model.fit(X_train_scaled, y_train)
            self.ensemble_models.append(model)
            print(f"   Model {i+1} trained successfully")

        # Use the best model as the primary (for feature importance, etc.)
        best_params = top_3_trials[0]['params'].copy()
        best_params['n_jobs'] = 1
        best_params['objective'] = 'reg:squarederror'
        best_params['verbosity'] = 0
        self.range_model = self.ensemble_models[0]  # Best model is primary
        print(f"[DEBUG] Primary model params: {best_params}")

        # Feature Selection: DISABLED
        # Testing showed ALL features performs best (R² = 0.423 vs 0.256 with selection)
        # XGBoost's built-in regularization handles high-dimensional data well.
        # Keeping this code structure but always using all features.

        # Log feature importance for reference
        # Use training_feature_names (after pruning) to match model's feature count
        feature_importance = self.range_model.feature_importances_
        training_features = getattr(self, 'training_feature_names', None) or self.feature_names
        importance_df = pd.DataFrame({
            'feature': training_features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        nonzero_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()

        print(f"\n[FEATURES] Using ALL {len(self.feature_names)} features (importance-based selection DISABLED)")
        print(f"   Non-zero importance: {len(nonzero_features)}/{len(self.feature_names)}")
        print(f"   Top 5 by importance: {importance_df.head(5)['feature'].tolist()}")

        # No feature filtering - use all
        self.selected_features = []
        self.selected_indices = []

        # Get high/low targets aligned with training data (needed for both optimization and model training)
        y_high_full = targets['next_high_pct'].copy()
        y_low_full = targets['next_low_pct'].copy()
        y_high_valid = y_high_full.loc[valid_idx]
        y_low_valid = y_low_full.loc[valid_idx]
        y_high_train = y_high_valid.iloc[:split_idx]
        y_low_train = y_low_valid.iloc[:split_idx]

        # Handle any length mismatches
        if len(y_high_train) != X_train_scaled.shape[0]:
            y_high_train = targets['next_high_pct'].loc[y.iloc[:split_idx].index]
            y_low_train = targets['next_low_pct'].loc[y.iloc[:split_idx].index]

        # Convert to numpy arrays
        y_high_train_arr = y_high_train.values if hasattr(y_high_train, 'values') else y_high_train
        y_low_train_arr = y_low_train.values if hasattr(y_low_train, 'values') else y_low_train

        # =================================================================
        # HIGH/LOW SPECIFIC OPTIMIZATION (optional)
        # Optimize hyperparameters directly for predicting high/low
        # instead of reusing range-optimized params (which was causing lower R²)
        # =================================================================
        if optimize_highlow:
            print("\n[HIGH/LOW OPTIMIZATION] Training dedicated models for high and low prediction...")

            # Save data for parallel workers
            # Use training_feature_names (after pruning) to match X_train_scaled columns
            training_features = getattr(self, 'training_feature_names', None) or self.feature_names
            highlow_data_path = optuna_worker.set_highlow_shared_data(
                X_train_scaled, y_high_train_arr, y_low_train_arr, training_features
            )

            # Use same parallelization pattern as RANGE optimization:
            # Spawn multiple joblib workers, each running their own Optuna study
            total_hl_trials = max(n_trials // 2, 50)  # Total trials for each target
            trials_per_worker = max(3, total_hl_trials // n_workers)
            actual_trials = trials_per_worker * n_workers

            print(f"   HIGH: {actual_trials} trials across {n_workers} workers ({trials_per_worker}/worker)")

            try:
                # HIGH optimization - parallel workers
                from joblib import Parallel, delayed
                high_results = Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
                    delayed(optuna_worker.run_highlow_study)(
                        highlow_data_path, trials_per_worker, seed=42 + i, target='high', worker_id=i
                    )
                    for i in range(n_workers)
                )

                # Find best HIGH result across all workers
                best_high = max(high_results, key=lambda x: x['best_score'])
                if best_high and best_high['best_params']:
                    self.high_best_params = best_high['best_params'].copy()
                    self.high_best_params['n_jobs'] = 1
                    self.high_best_params['objective'] = 'reg:squarederror'
                    self.high_best_params['verbosity'] = 0
                    print(f"   HIGH optimization: Score={best_high['best_score']:.4f}")
                    # Collect for summary
                    _training_summary.data['high_best_score'] = best_high['best_score']
                    _training_summary.data['high_all_scores'] = [r['best_score'] for r in high_results]
                else:
                    self.high_best_params = best_params.copy()
                    print(f"   HIGH optimization: Using range params (fallback)")

                print(f"   LOW: {actual_trials} trials across {n_workers} workers ({trials_per_worker}/worker)")

                # LOW optimization - parallel workers
                low_results = Parallel(n_jobs=n_workers, backend='loky', verbose=0)(
                    delayed(optuna_worker.run_highlow_study)(
                        highlow_data_path, trials_per_worker, seed=142 + i, target='low', worker_id=i
                    )
                    for i in range(n_workers)
                )

                # Find best LOW result across all workers
                best_low = max(low_results, key=lambda x: x['best_score'])
                if best_low and best_low['best_params']:
                    self.low_best_params = best_low['best_params'].copy()
                    self.low_best_params['n_jobs'] = 1
                    self.low_best_params['objective'] = 'reg:squarederror'
                    self.low_best_params['verbosity'] = 0
                    print(f"   LOW optimization: Score={best_low['best_score']:.4f}")
                    # Collect for summary
                    _training_summary.data['low_best_score'] = best_low['best_score']
                    _training_summary.data['low_all_scores'] = [r['best_score'] for r in low_results]
                else:
                    self.low_best_params = best_params.copy()
                    print(f"   LOW optimization: Using range params (fallback)")

            except Exception as hl_err:
                print(f"   HIGH/LOW optimization failed: {hl_err}")
                print(f"   Using range-optimized params as fallback")
                self.high_best_params = best_params.copy()
                self.low_best_params = best_params.copy()
        else:
            # Skip HIGH/LOW optimization - use range-optimized params
            print("\n[HIGH/LOW MODELS] Using range-optimized params (optimize_highlow=False)")
            self.high_best_params = best_params.copy()
            self.low_best_params = best_params.copy()

        # =================================================================
        # ASYMMETRIC LOW MODEL ADJUSTMENTS
        # Analysis showed LOW predictions have R² = -0.17 vs HIGH R² = 0.23
        # LOW model needs more regularization to prevent overfitting to noise
        # =================================================================
        print("\n[ASYMMETRIC ADJUSTMENT] Applying LOW-specific hyperparameter tuning...")

        # Increase regularization for LOW model (more prone to overfitting on volatile target)
        self.low_best_params['reg_alpha'] = self.low_best_params.get('reg_alpha', 0.0) + 0.3  # L1 regularization
        self.low_best_params['reg_lambda'] = max(self.low_best_params.get('reg_lambda', 1.0), 1.5)  # L2 regularization

        # Reduce tree depth for LOW (simpler model = less overfitting)
        if self.low_best_params.get('max_depth', 6) > 4:
            self.low_best_params['max_depth'] = 4

        # Slower learning rate for LOW (more careful fitting)
        self.low_best_params['learning_rate'] = min(self.low_best_params.get('learning_rate', 0.1), 0.05)

        # Increase min_child_weight for LOW (require more samples per leaf)
        self.low_best_params['min_child_weight'] = max(self.low_best_params.get('min_child_weight', 1), 3)

        # More subsampling for LOW (bagging helps with noisy targets)
        self.low_best_params['subsample'] = min(self.low_best_params.get('subsample', 1.0), 0.7)
        self.low_best_params['colsample_bytree'] = min(self.low_best_params.get('colsample_bytree', 1.0), 0.7)

        print(f"   LOW adjusted: max_depth={self.low_best_params.get('max_depth')}, "
              f"reg_alpha={self.low_best_params.get('reg_alpha'):.2f}, "
              f"learning_rate={self.low_best_params.get('learning_rate'):.3f}")

        # Train dedicated HIGH model with optimized params
        print("\n[HIGH MODEL] Training with optimized hyperparameters...")
        self.high_model = xgb.XGBRegressor(**self.high_best_params)
        self.high_model.fit(X_train_scaled, y_high_train_arr)

        # Train dedicated LOW model with optimized params
        # =================================================================
        # LOW MODEL ENSEMBLE (Multiple models for better LOW predictions)
        # LOW predictions are harder - use ensemble to reduce variance
        # =================================================================
        print("[LOW MODEL ENSEMBLE] Training multiple models for robust LOW prediction...")

        self.low_ensemble_models = []
        ensemble_configs = [
            # Config 1: Original optimized params
            {'name': 'base', 'params': self.low_best_params.copy()},

            # Config 2: Very conservative (more regularization)
            {'name': 'conservative', 'params': {
                **self.low_best_params,
                'max_depth': 3,
                'reg_alpha': self.low_best_params.get('reg_alpha', 0.3) + 0.5,
                'reg_lambda': 2.0,
                'learning_rate': 0.03,
                'min_child_weight': 5,
            }},

            # Config 3: Different tree structure
            {'name': 'wide_shallow', 'params': {
                **self.low_best_params,
                'max_depth': 2,
                'n_estimators': self.low_best_params.get('n_estimators', 100) + 50,
                'colsample_bytree': 0.5,
                'subsample': 0.6,
            }},

            # Config 4: Focus on recent data (higher learning rate, fewer trees)
            {'name': 'adaptive', 'params': {
                **self.low_best_params,
                'max_depth': 4,
                'learning_rate': 0.08,
                'n_estimators': 80,
                'subsample': 0.8,
            }},
        ]

        for config in ensemble_configs:
            try:
                model = xgb.XGBRegressor(**config['params'])
                model.fit(X_train_scaled, y_low_train_arr)
                self.low_ensemble_models.append({'name': config['name'], 'model': model, 'type': 'xgb'})
                print(f"   LOW ensemble member '{config['name']}' (XGB) trained")
            except Exception as e:
                print(f"   LOW ensemble member '{config['name']}' failed: {e}")

        # Add LightGBM to ensemble (handles collinearity differently than XGBoost)
        if LGBM_AVAILABLE:
            try:
                lgb_params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'max_depth': 4,
                    'learning_rate': 0.05,
                    'n_estimators': 100,
                    'min_child_samples': 10,
                    'reg_alpha': 0.5,
                    'reg_lambda': 1.0,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'verbosity': -1,
                    'force_col_wise': True,  # Better for small datasets
                }
                lgb_model = lgb.LGBMRegressor(**lgb_params)
                lgb_model.fit(X_train_scaled, y_low_train_arr)
                self.low_ensemble_models.append({'name': 'lightgbm', 'model': lgb_model, 'type': 'lgb'})
                print(f"   LOW ensemble member 'lightgbm' (LGB) trained")
            except Exception as e:
                print(f"   LightGBM ensemble member failed: {e}")

        # Primary LOW model is the first (base) for compatibility
        if self.low_ensemble_models:
            self.low_model = self.low_ensemble_models[0]['model']
        else:
            # Fallback: train single model
            self.low_model = xgb.XGBRegressor(**self.low_best_params)
            self.low_model.fit(X_train_scaled, y_low_train_arr)

        print(f"   LOW ensemble: {len(self.low_ensemble_models)} models trained (XGB + LightGBM)")

        # =================================================================
        # REGIME DETECTION - DISABLED FOR HIGH MODEL
        # Analysis showed regime detection hurt HIGH model (R² dropped 40%)
        # but helped LOW model (R² improved 25%). Keep regime for LOW only.
        # =================================================================
        print("\n[REGIME DETECTION] Disabled for HIGH model (was hurting performance)")
        print("   HIGH model will use base model only (R² was 0.30 before regime)")

        self.vol_regime_thresholds = None
        self.regime_high_models = {}  # Empty - HIGH won't use regime
        self.regime_low_models = {}   # Could enable for LOW later if needed

        # =================================================================
        # QUANTILE REGRESSION FOR LEARNED CONFIDENCE BANDS
        # Instead of heuristic z-scores, learn the prediction intervals
        # directly from data using XGBoost's quantile objective
        # =================================================================
        print("\n[QUANTILE REGRESSION] Training models for learned confidence bands...")

        # Train quantile models for different confidence levels
        self.quantile_models = {}

        # y_high_train_arr and y_low_train_arr were already prepared above
        # Only proceed if we have valid targets
        if y_high_train_arr is not None and y_low_train_arr is not None and len(y_high_train_arr) == X_train_scaled.shape[0]:

            # Quantile levels for different confidence intervals
            # For TRADING, tighter bands (50-68%) are more actionable
            # For RISK MANAGEMENT, wider bands (80-95%) are safer
            quantile_levels = {
                0.50: (0.25, 0.75),  # 50% confidence - TIGHT, most actionable
                0.68: (0.16, 0.84),  # 68% confidence - ~1 std dev, good balance
                0.80: (0.10, 0.90),  # 80% confidence
                0.90: (0.05, 0.95),  # 90% confidence
                0.95: (0.025, 0.975) # 95% confidence - widest
            }

            # Use HIGH-optimized params for HIGH quantile models
            quantile_high_params = self.high_best_params.copy()
            quantile_high_params['objective'] = 'reg:quantileerror'
            quantile_high_params['n_jobs'] = 1
            quantile_high_params['verbosity'] = 0
            # Reduce complexity slightly for quantile models
            quantile_high_params['max_depth'] = min(quantile_high_params.get('max_depth', 6), 6)
            quantile_high_params['n_estimators'] = min(quantile_high_params.get('n_estimators', 100), 100)

            # Use LOW-optimized params for LOW quantile models
            quantile_low_params = self.low_best_params.copy()
            quantile_low_params['objective'] = 'reg:quantileerror'
            quantile_low_params['n_jobs'] = 1
            quantile_low_params['verbosity'] = 0
            quantile_low_params['max_depth'] = min(quantile_low_params.get('max_depth', 6), 6)
            quantile_low_params['n_estimators'] = min(quantile_low_params.get('n_estimators', 100), 100)

            for conf_level, (q_low, q_high) in quantile_levels.items():
                try:
                    # Train lower quantile model for HIGH predictions (using HIGH-optimized params)
                    params_high_lower = quantile_high_params.copy()
                    params_high_lower['quantile_alpha'] = q_low
                    model_high_lower = xgb.XGBRegressor(**params_high_lower)
                    model_high_lower.fit(X_train_scaled, y_high_train_arr)

                    # Train upper quantile model for HIGH predictions
                    params_high_upper = quantile_high_params.copy()
                    params_high_upper['quantile_alpha'] = q_high
                    model_high_upper = xgb.XGBRegressor(**params_high_upper)
                    model_high_upper.fit(X_train_scaled, y_high_train_arr)

                    # Train lower quantile model for LOW predictions (using LOW-optimized params)
                    params_low_lower = quantile_low_params.copy()
                    params_low_lower['quantile_alpha'] = q_low
                    model_low_lower = xgb.XGBRegressor(**params_low_lower)
                    model_low_lower.fit(X_train_scaled, y_low_train_arr)

                    # Train upper quantile model for LOW predictions
                    params_low_upper = quantile_low_params.copy()
                    params_low_upper['quantile_alpha'] = q_high
                    model_low_upper = xgb.XGBRegressor(**params_low_upper)
                    model_low_upper.fit(X_train_scaled, y_low_train_arr)

                    self.quantile_models[conf_level] = {
                        'high_lower': model_high_lower,  # 5th percentile of high
                        'high_upper': model_high_upper,  # 95th percentile of high
                        'low_lower': model_low_lower,    # 5th percentile of low
                        'low_upper': model_low_upper,    # 95th percentile of low
                        'quantiles': (q_low, q_high)
                    }
                    print(f"   {int(conf_level*100)}% CI: Trained 4 quantile models (q={q_low}, {q_high})")

                except Exception as q_err:
                    print(f"   {int(conf_level*100)}% CI: Failed - {q_err}")
                    import traceback
                    traceback.print_exc()

            # Note: high_model and low_model are already trained above with
            # dedicated hyperparameter optimization (not median quantile)

        else:
            print(f"   SKIPPED quantile training due to data issues")

        print(f"[QUANTILE REGRESSION] Complete - {len(self.quantile_models)} confidence levels available")
        print(f"[QUANTILE REGRESSION] Keys: {list(self.quantile_models.keys())}")
        print(f"[QUANTILE REGRESSION] high_model is None: {self.high_model is None}")
        print(f"[QUANTILE REGRESSION] low_model is None: {self.low_model is None}")

        # Evaluate on test set using ensemble average
        print("[DEBUG] Predicting on test set with ensemble...")
        if hasattr(self, 'ensemble_models') and len(self.ensemble_models) > 1:
            # Ensemble prediction: average of all models
            predictions = np.array([m.predict(X_test_scaled) for m in self.ensemble_models])
            y_pred = np.mean(predictions, axis=0)
            print(f"[DEBUG] Ensemble predictions: {len(self.ensemble_models)} models averaged")
        else:
            y_pred = self.range_model.predict(X_test_scaled)
        print(f"[DEBUG] Predictions shape: {y_pred.shape}")

        print("[DEBUG] Calculating metrics...")
        self.model_metrics = {
            # NOTE: RMSE/MAE are in PERCENTAGE since target was scaled. Convert back to fraction for downstream use.
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)) / 100.0,  # Convert % to fraction
            'rmse_pct': np.sqrt(mean_squared_error(y_test, y_pred)),  # Keep % version for display
            'mae': mean_absolute_error(y_test, y_pred) / 100.0,  # Convert % to fraction
            'mae_pct': mean_absolute_error(y_test, y_pred),  # Keep % version for display
            'r2': r2_score(y_test, y_pred),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train_scaled.shape[1],
            'best_params': best_params
        }
        print(f"[DEBUG] Metrics: R2={self.model_metrics['r2']:.4f}, RMSE={self.model_metrics['rmse_pct']:.3f}%")

        # Feature importance (selected features)
        print("[DEBUG] Creating feature importance...")
        # Use training_feature_names (after pruning) to match model's feature count
        feature_names_for_importance = getattr(self, 'training_feature_names', None) or self.feature_names
        importance = pd.DataFrame({
            'feature': feature_names_for_importance,
            'importance': self.range_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"[DEBUG] Top 3 features: {importance.head(3)['feature'].tolist()}")

        # Collect final metrics for summary
        _training_summary.data['test_r2'] = self.model_metrics['r2']
        _training_summary.data['test_rmse'] = self.model_metrics['rmse_pct']
        _training_summary.data['ensemble_size'] = len(self.ensemble_models)
        _training_summary.data['low_ensemble_size'] = len(getattr(self, 'low_ensemble_models', []))
        _training_summary.data['best_params'] = best_params
        _training_summary.data['top_features_importance'] = importance.head(10)['feature'].tolist()
        _training_summary.data['nonzero_importance_count'] = (importance['importance'] > 0).sum()
        _training_summary.data['quantile_levels'] = list(getattr(self, 'quantile_models', {}).keys())

        # Print the compact training summary
        _training_summary.print_summary()

        print("[DEBUG] Returning result dict...")
        return {
            'model': self.range_model,
            'scaler': self.scaler,
            'metrics': self.model_metrics,
            'feature_importance': importance,
            'feature_names': self.feature_names
        }

    def predict_daily_range(self, df: pd.DataFrame, options_features: Dict = None,
                           confidence_level: float = 0.9) -> Dict:
        """
        Predict next day's range, high, and low.

        Args:
            df: Recent OHLCV data (at least 50 bars)
            options_features: Optional current options data
            confidence_level: For prediction intervals (0.8, 0.9, 0.95)

        Returns:
            dict with predicted_range, predicted_high, predicted_low, confidence_interval
        """
        print("[DEBUG predict] Starting predict_daily_range...")
        if self.range_model is None:
            raise ValueError("Model not trained. Call train_range_model() first.")

        # Create features for latest bar
        print("[DEBUG predict] Creating features...")
        features = self.create_range_features(df, options_features)

        # Use training_feature_names (won't be overwritten by create_range_features)
        # Fall back to feature_names for backwards compatibility
        feature_cols = getattr(self, 'training_feature_names', None) or self.feature_names
        X = features[feature_cols].iloc[[-1]]  # Last row only
        print(f"[DEBUG predict] Features shape: {X.shape}, NaN count: {X.isna().sum().sum()}")

        if X.isna().any().any():
            # Fill NaN with column means if needed
            print("[DEBUG predict] Filling NaN values...")
            X = X.fillna(X.mean())

        print("[DEBUG predict] Scaling features...")
        X_scaled = self.scaler.transform(X)

        # Apply feature selection if available
        if hasattr(self, 'selected_indices') and self.selected_indices:
            X_scaled = X_scaled[:, self.selected_indices]
            print(f"[DEBUG predict] Using {len(self.selected_indices)} selected features")
        else:
            print(f"[DEBUG predict] Using all {X_scaled.shape[1]} features")

        # Get current close
        current_close = df['close'].iloc[-1]
        print(f"[DEBUG predict] Current close: ${current_close:.2f}")

        # Calculate predicted high/low ratios
        high_ratio = 0.55
        low_ratio = 0.45

        # Make point prediction for range (used as fallback if quantile models unavailable)
        # IMPORTANT: Model was trained on PERCENTAGE target (1.0 = 1%), must convert back to fraction
        print("[DEBUG predict] Making point prediction...")
        # Use ensemble average if available
        if hasattr(self, 'ensemble_models') and len(self.ensemble_models) > 1:
            predictions = np.array([m.predict(X_scaled)[0] for m in self.ensemble_models])
            predicted_range_pct = np.mean(predictions)
            print(f"[DEBUG predict] Ensemble of {len(self.ensemble_models)} models averaged")
        else:
            predicted_range_pct = self.range_model.predict(X_scaled)[0]
        predicted_range = predicted_range_pct / 100.0  # Convert from percentage to fraction

        print(f"[DEBUG predict] Predicted range: {predicted_range_pct:.3f}% ({predicted_range:.6f} fraction)")

        # Calculate predicted high/low
        predicted_high = current_close * (1 + predicted_range * high_ratio)
        predicted_low = current_close * (1 - predicted_range * low_ratio)
        print(f"[DEBUG predict] High: ${predicted_high:.2f}, Low: ${predicted_low:.2f}")

        # ==========================================================================
        # LEARNED CONFIDENCE BANDS (Quantile Regression)
        # Band widths are learned from data, not heuristic multipliers
        # ==========================================================================
        #
        # Quantile regression models predict actual percentiles of the distribution:
        # - 90% CI: 5th and 95th percentiles
        # - 80% CI: 10th and 90th percentiles
        # - 95% CI: 2.5th and 97.5th percentiles
        #
        # This approach learns heteroscedastic uncertainty (wider bands when
        # the model should be less confident, tighter when more confident)

        # Debug: Check each condition for quantile bands
        has_quantile_attr = hasattr(self, 'quantile_models')
        has_conf_level = has_quantile_attr and confidence_level in self.quantile_models
        has_high_model = hasattr(self, 'high_model') and self.high_model is not None

        print(f"[DEBUG predict] Quantile check: has_attr={has_quantile_attr}, "
              f"conf_level_exists={has_conf_level} (requested={confidence_level}), "
              f"high_model_exists={has_high_model}")
        if has_quantile_attr:
            print(f"[DEBUG predict] Available confidence levels: {list(self.quantile_models.keys())}")

        use_quantile_bands = has_quantile_attr and has_conf_level and has_high_model

        if use_quantile_bands:
            print(f"[DEBUG predict] Using LEARNED confidence bands (quantile regression)")

            q_models = self.quantile_models[confidence_level]

            # Get quantile predictions (in percentage)
            high_lower_pct = q_models['high_lower'].predict(X_scaled)[0]
            high_upper_pct = q_models['high_upper'].predict(X_scaled)[0]
            low_lower_pct = q_models['low_lower'].predict(X_scaled)[0]
            low_upper_pct = q_models['low_upper'].predict(X_scaled)[0]

            # =================================================================
            # REGIME-AWARE PREDICTION
            # Detect current volatility regime and blend predictions
            # =================================================================
            current_regime = 'mid'  # Default
            regime_weight = 0.0

            if hasattr(self, 'vol_regime_thresholds') and self.vol_regime_thresholds is not None:
                # Get current volatility from features
                current_vol = None
                if 'volatility_20d' in features.columns:
                    current_vol = features['volatility_20d'].iloc[-1]
                elif 'atr_14_pct' in features.columns:
                    current_vol = features['atr_14_pct'].iloc[-1]

                if current_vol is not None and not np.isnan(current_vol):
                    thresholds = self.vol_regime_thresholds
                    if current_vol >= thresholds['high']:
                        current_regime = 'high_vol'
                        # Weight based on how far into high-vol territory
                        regime_weight = min(1.0, (current_vol - thresholds['high']) / (thresholds['high'] - thresholds['median']) * 0.5 + 0.5)
                    elif current_vol <= thresholds['low']:
                        current_regime = 'low_vol'
                        regime_weight = min(1.0, (thresholds['low'] - current_vol) / (thresholds['median'] - thresholds['low']) * 0.5 + 0.5)
                    print(f"   Regime detected: {current_regime} (vol={current_vol:.2f}%, weight={regime_weight:.2f})")

            # Get median predictions for high/low with regime blending
            base_high_pct = self.high_model.predict(X_scaled)[0]

            # Blend with regime-specific model if available (conservative 30% max blend)
            if (hasattr(self, 'regime_high_models') and
                current_regime in self.regime_high_models and
                regime_weight > 0.4):  # Only use regime model if weight > 40%

                regime_high_pct = self.regime_high_models[current_regime].predict(X_scaled)[0]
                # Blend: base * (1 - weight) + regime * weight (max 30% regime contribution)
                blend_factor = regime_weight * 0.3  # Conservative: max 30% regime influence
                pred_high_pct = base_high_pct * (1 - blend_factor) + regime_high_pct * blend_factor
                print(f"   HIGH blended: base={base_high_pct:.3f}%, regime={regime_high_pct:.3f}% -> {pred_high_pct:.3f}%")
            else:
                pred_high_pct = base_high_pct

            # LOW prediction uses ensemble if available
            if hasattr(self, 'low_ensemble_models') and len(self.low_ensemble_models) > 1:
                # Get predictions from all ensemble members
                low_predictions = []
                for member in self.low_ensemble_models:
                    pred = member['model'].predict(X_scaled)[0]
                    low_predictions.append(pred)

                # Use median of ensemble (more robust than mean to outliers)
                base_low_pct = np.median(low_predictions)

                # Also store ensemble stats for debugging
                low_ensemble_std = np.std(low_predictions)
                low_ensemble_range = max(low_predictions) - min(low_predictions)
                print(f"   LOW ensemble: median={base_low_pct:.3f}%, std={low_ensemble_std:.3f}%, range={low_ensemble_range:.3f}%")
            else:
                base_low_pct = self.low_model.predict(X_scaled)[0]

            # Blend LOW with regime-specific model if available (conservative 30% max blend)
            if (hasattr(self, 'regime_low_models') and
                current_regime in self.regime_low_models and
                regime_weight > 0.4):

                regime_low_pct = self.regime_low_models[current_regime].predict(X_scaled)[0]
                # Blend: base * (1 - weight) + regime * weight (max 30% regime contribution)
                blend_factor = regime_weight * 0.3  # Conservative: max 30% regime influence
                pred_low_pct = base_low_pct * (1 - blend_factor) + regime_low_pct * blend_factor
                print(f"   LOW blended: base={base_low_pct:.3f}%, regime={regime_low_pct:.3f}% -> {pred_low_pct:.3f}%")
            else:
                pred_low_pct = base_low_pct

            # Convert percentages to prices
            # next_high_pct = (next_high - close) / close * 100
            # So: next_high = close * (1 + next_high_pct / 100)
            predicted_high = current_close * (1 + pred_high_pct / 100)
            predicted_low = current_close * (1 - pred_low_pct / 100)  # Note: low is subtracted

            high_lower = current_close * (1 + high_lower_pct / 100)
            high_upper = current_close * (1 + high_upper_pct / 100)
            low_lower = current_close * (1 - low_upper_pct / 100)  # Swap: upper quantile of low% = lower price
            low_upper = current_close * (1 - low_lower_pct / 100)  # Swap: lower quantile of low% = upper price

            # Calculate uncertainties for display
            high_uncertainty = (high_upper - high_lower) / 2
            low_uncertainty = (low_upper - low_lower) / 2

            conformal_bounds = {
                'method': 'quantile_regression',
                'confidence_level': confidence_level,
                'quantiles': q_models['quantiles'],
                'high_pct_bounds': (high_lower_pct, high_upper_pct),
                'low_pct_bounds': (low_lower_pct, low_upper_pct)
            }

            print(f"[DEBUG predict] Quantile bands: High=[${high_lower:.2f}, ${high_upper:.2f}], "
                  f"Low=[${low_lower:.2f}, ${low_upper:.2f}]")

        else:
            # Fallback to RMSE-based bands if quantile models not available
            print(f"[DEBUG predict] Using FALLBACK confidence bands (RMSE-based)")

            rmse = self.model_metrics.get('rmse', 0.005)

            # Simple RMSE-based bands with minimal adjustments
            base_z = {0.8: 1.28, 0.9: 1.645, 0.95: 1.96}.get(confidence_level, 1.645)

            high_uncertainty = rmse * base_z * current_close
            low_uncertainty = rmse * base_z * current_close * 1.15  # Slightly wider for lows

            high_lower = predicted_high - high_uncertainty
            high_upper = predicted_high + high_uncertainty
            low_lower = predicted_low - low_uncertainty
            low_upper = predicted_low + low_uncertainty

            conformal_bounds = {
                'method': 'rmse_fallback',
                'rmse': rmse,
                'base_z': base_z
            }

            print(f"[DEBUG predict] Fallback bands: RMSE={rmse:.4f}, z={base_z:.2f}")

        print(f"[DEBUG predict] Returning prediction dict...")
        # Convert all values to Python floats to avoid numpy array issues
        result = {
            'current_close': float(current_close),
            'predicted_range': float(predicted_range),
            'predicted_range_dollars': float(current_close * predicted_range),
            'predicted_high': float(predicted_high),
            'predicted_low': float(predicted_low),
            'high_lower': float(high_lower),
            'high_upper': float(high_upper),
            'low_lower': float(low_lower),
            'low_upper': float(low_upper),
            'high_uncertainty': float(high_uncertainty),
            'low_uncertainty': float(low_uncertainty),
            'confidence_level': float(confidence_level),
            'confidence_method': conformal_bounds.get('method', 'rmse'),
            'prediction_date': df.index[-1] + timedelta(days=1) if hasattr(df.index[-1], 'date') else None,
            'model_r2': float(self.model_metrics.get('r2', 0)),
            'atr_14': float(features['atr_14'].iloc[-1]) if ('atr_14' in features and pd.notna(features['atr_14'].iloc[-1])) else None
        }
        print(f"[DEBUG predict] DONE! Predicted High: ${result['predicted_high']:.2f}, Low: ${result['predicted_low']:.2f}")
        # Convert to float in case they're numpy arrays
        hl, hu = float(high_lower), float(high_upper)
        ll, lu = float(low_lower), float(low_upper)
        print(f"[DEBUG predict] Confidence bounds ({confidence_level*100:.0f}%): High=[${hl:.2f}, ${hu:.2f}], Low=[${ll:.2f}, ${lu:.2f}]")
        return result

    def save_predictions(self, predictions: dict, ticker: str, path: str = None) -> str:
        """
        Save walk-forward predictions to JSON for use in Options Builder.

        Args:
            predictions: Dictionary of prediction results from predict_daily_range()
            ticker: Stock ticker symbol
            path: Optional custom path for saving

        Returns:
            Path where predictions were saved
        """
        import json
        import os

        if path is None:
            # Create predictions directory if needed
            os.makedirs('predictions', exist_ok=True)
            path = f"predictions/{ticker}_range_predictions.json"

        save_data = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'model_r2': predictions.get('model_r2', self.model_metrics.get('r2', 0)),
            'model_metrics': self.model_metrics,
            'predictions': {
                'current_close': predictions.get('current_close'),
                'predicted_high': predictions.get('predicted_high'),
                'predicted_low': predictions.get('predicted_low'),
                'predicted_range': predictions.get('predicted_range'),
                'predicted_range_dollars': predictions.get('predicted_range_dollars'),
                'confidence_level': predictions.get('confidence_level', 0.9),
                'ci_high_upper': predictions.get('high_upper'),
                'ci_high_lower': predictions.get('high_lower'),
                'ci_low_upper': predictions.get('low_upper'),
                'ci_low_lower': predictions.get('low_lower'),
                'high_uncertainty': predictions.get('high_uncertainty'),
                'low_uncertainty': predictions.get('low_uncertainty'),
                'prediction_date': str(predictions.get('prediction_date', '')),
                'atr_14': predictions.get('atr_14'),
            },
            'feature_names': self.feature_names[:20] if self.feature_names else [],  # Top 20 features
        }

        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"   Predictions saved to {path}")
        return path

    @staticmethod
    def load_predictions(path: str) -> Optional[dict]:
        """
        Load saved predictions from JSON.

        Args:
            path: Path to the saved predictions file

        Returns:
            Dictionary of saved predictions or None if file doesn't exist
        """
        import json
        import os

        if not os.path.exists(path):
            print(f"   Warning: Predictions file not found: {path}")
            return None

        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"   Error loading predictions: {e}")
            return None

    @staticmethod
    def list_saved_predictions(directory: str = 'predictions') -> List[str]:
        """
        List all saved prediction files.

        Args:
            directory: Directory to search for prediction files

        Returns:
            List of prediction file paths
        """
        import os
        import glob

        if not os.path.exists(directory):
            return []

        return sorted(glob.glob(os.path.join(directory, '*_range_predictions.json')))


# =============================================================================
# REGIME-SWITCHING RANGE PREDICTOR
# Based on analysis finding that feature importance varies dramatically by regime
# =============================================================================

class RegimeSwitchingRangePredictor:
    """
    Use different models based on current volatility regime.

    Analysis showed:
    - In HIGH volatility: range_std_5d, volatility_5d, atr_14_pct are 50-64% more predictive
    - In LOW volatility: roc_5, atr_ratio_7_21 work better
    - Feature importance has high variance because predictive power is regime-dependent

    This class trains separate models for each regime and routes predictions accordingly.
    """

    def __init__(self, polygon_manager=None):
        self.polygon = polygon_manager
        self.high_vol_predictor = None
        self.low_vol_predictor = None
        self.vol_threshold = None
        self.is_trained = False

        # Regime-specific feature weights (boost important features for each regime)
        self.high_vol_feature_boost = {
            'range_std_5d': 2.0,
            'volatility_5d': 2.0,
            'atr_14_pct': 1.5,
            'parkinson_vol': 1.5,
            'range_lag_1': 1.3,
            'atr_14': 1.3
        }

        self.low_vol_feature_boost = {
            'roc_5': 1.5,
            'roc_10': 1.3,
            'atr_ratio_7_21': 1.5,
            'price_vs_sma50': 1.3,
            'price_vs_sma20': 1.3,
            'vol_compression': 1.5
        }

    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate volatility regime for each row.
        Returns Series with 'high', 'low', or 'normal' for each row.
        """
        # Annualized volatility (20-day rolling)
        returns = df['close'].pct_change()
        vol_20d = returns.rolling(20).std() * np.sqrt(252) * 100  # As percentage

        # Calculate regime thresholds based on historical distribution
        vol_25_pct = vol_20d.quantile(0.25)
        vol_75_pct = vol_20d.quantile(0.75)

        regime = pd.Series(index=df.index, data='normal')
        regime[vol_20d < vol_25_pct] = 'low'
        regime[vol_20d > vol_75_pct] = 'high'

        return regime, vol_20d

    def _calculate_trend_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate trend regime for each row.
        Returns Series with 'uptrend', 'downtrend', or 'sideways'.
        """
        # 20-day trend
        trend_20d = df['close'].pct_change(20)

        regime = pd.Series(index=df.index, data='sideways')
        regime[trend_20d > 0.02] = 'uptrend'  # >2% over 20 days
        regime[trend_20d < -0.02] = 'downtrend'

        return regime

    def train(self, df: pd.DataFrame, options_features: Dict = None,
              n_trials: int = 50, n_jobs: int = 1) -> Dict:
        """
        Train separate models for high-vol and low-vol regimes.

        Returns:
            Dict with training metrics for both models
        """
        print("=" * 60)
        print("REGIME-SWITCHING MODEL TRAINING")
        print("=" * 60)

        # Calculate regimes
        vol_regime, vol_20d = self._calculate_volatility_regime(df)

        # Store threshold for prediction time
        self.vol_threshold = vol_20d.median()
        print(f"Volatility threshold: {self.vol_threshold:.2f}%")

        # Count samples in each regime
        high_vol_count = (vol_regime == 'high').sum()
        low_vol_count = (vol_regime == 'low').sum()
        normal_count = (vol_regime == 'normal').sum()

        print(f"Regime distribution: High={high_vol_count}, Normal={normal_count}, Low={low_vol_count}")

        # Split data by regime
        high_vol_mask = vol_regime == 'high'
        low_vol_mask = vol_regime == 'low'

        # For training, combine 'normal' with whichever has fewer samples to balance
        if high_vol_count < low_vol_count:
            # Add some normal samples to high vol training
            high_vol_mask = (vol_regime == 'high') | (vol_regime == 'normal')
        else:
            # Add some normal samples to low vol training
            low_vol_mask = (vol_regime == 'low') | (vol_regime == 'normal')

        # Create predictors
        self.high_vol_predictor = PriceRangePredictor(self.polygon)
        self.low_vol_predictor = PriceRangePredictor(self.polygon)

        # Train high-vol model
        print("\n" + "-" * 40)
        print("Training HIGH VOLATILITY model...")
        print("-" * 40)
        df_high_vol = df[high_vol_mask].copy()
        if len(df_high_vol) >= 60:  # Need minimum samples
            high_vol_results = self.high_vol_predictor.train_range_model(
                df_high_vol,
                options_features=options_features,
                n_trials=n_trials,
                n_jobs=n_jobs
            )
            print(f"High-vol model R²: {high_vol_results.get('r2', 0):.4f}")
        else:
            print(f"WARNING: Not enough high-vol samples ({len(df_high_vol)}), using full model")
            high_vol_results = self.high_vol_predictor.train_range_model(
                df, options_features=options_features, n_trials=n_trials, n_jobs=n_jobs
            )

        # Train low-vol model
        print("\n" + "-" * 40)
        print("Training LOW VOLATILITY model...")
        print("-" * 40)
        df_low_vol = df[low_vol_mask].copy()
        if len(df_low_vol) >= 60:
            low_vol_results = self.low_vol_predictor.train_range_model(
                df_low_vol,
                options_features=options_features,
                n_trials=n_trials,
                n_jobs=n_jobs
            )
            print(f"Low-vol model R²: {low_vol_results.get('r2', 0):.4f}")
        else:
            print(f"WARNING: Not enough low-vol samples ({len(df_low_vol)}), using full model")
            low_vol_results = self.low_vol_predictor.train_range_model(
                df, options_features=options_features, n_trials=n_trials, n_jobs=n_jobs
            )

        self.is_trained = True

        print("\n" + "=" * 60)
        print("REGIME-SWITCHING TRAINING COMPLETE")
        print("=" * 60)

        return {
            'high_vol_r2': high_vol_results.get('r2', 0),
            'low_vol_r2': low_vol_results.get('r2', 0),
            'high_vol_samples': len(df_high_vol),
            'low_vol_samples': len(df_low_vol),
            'vol_threshold': self.vol_threshold
        }

    def predict_daily_range(self, df: pd.DataFrame, options_features: Dict = None,
                            confidence_level: float = 0.9) -> Dict:
        """
        Predict range using the appropriate regime model.

        Automatically detects current volatility regime and routes to the right model.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Determine current regime
        vol_regime, vol_20d = self._calculate_volatility_regime(df)
        current_vol = vol_20d.iloc[-1]
        current_regime = vol_regime.iloc[-1]

        print(f"Current volatility: {current_vol:.2f}% (Regime: {current_regime.upper()})")

        # Route to appropriate model
        if current_regime == 'high' or current_vol > self.vol_threshold:
            print("Using HIGH VOLATILITY model")
            result = self.high_vol_predictor.predict_daily_range(
                df, options_features=options_features, confidence_level=confidence_level
            )
            result['regime_used'] = 'high_volatility'
        else:
            print("Using LOW VOLATILITY model")
            result = self.low_vol_predictor.predict_daily_range(
                df, options_features=options_features, confidence_level=confidence_level
            )
            result['regime_used'] = 'low_volatility'

        result['current_volatility'] = current_vol
        result['volatility_threshold'] = self.vol_threshold

        return result

    def get_regime_info(self, df: pd.DataFrame) -> Dict:
        """Get current regime information without making a prediction."""
        vol_regime, vol_20d = self._calculate_volatility_regime(df)
        trend_regime = self._calculate_trend_regime(df)

        return {
            'volatility_regime': vol_regime.iloc[-1],
            'trend_regime': trend_regime.iloc[-1],
            'current_volatility': vol_20d.iloc[-1],
            'volatility_threshold': self.vol_threshold,
            'volatility_percentile': (vol_20d.iloc[-1] - vol_20d.min()) / (vol_20d.max() - vol_20d.min()) if vol_20d.max() != vol_20d.min() else 0.5
        }


class PriceTargetCalculator:
    """
    Calculates optimal take-profit levels using multiple methods:
    1. ATR-based targets (1x, 1.5x, 2x ATR)
    2. Fibonacci extensions
    3. Support/Resistance levels
    4. Options-based levels (max pain, high OI strikes)
    """

    def __init__(self, polygon_manager=None):
        self.polygon = polygon_manager

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for the given period."""
        if TA_AVAILABLE:
            atr = ta.atr(df['high'], df['low'], df['close'], length=period)
            return atr.iloc[-1] if not atr.empty else 0
        else:
            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            ], axis=1).max(axis=1)
            return tr.rolling(period).mean().iloc[-1]

    def calculate_atr_targets(self, entry_price: float, atr: float,
                             direction: str = 'long') -> Dict:
        """
        Calculate ATR-based price targets.

        Args:
            entry_price: Position entry price
            atr: Current ATR value
            direction: 'long' or 'short'

        Returns:
            dict with conservative, moderate, aggressive targets
        """
        multipliers = {'conservative': 1.0, 'moderate': 1.5, 'aggressive': 2.0, 'extended': 3.0}

        targets = {}
        for level, mult in multipliers.items():
            if direction == 'long':
                targets[level] = {
                    'price': entry_price + (atr * mult),
                    'pct_gain': (atr * mult / entry_price) * 100,
                    'method': f'{mult}x ATR'
                }
            else:
                targets[level] = {
                    'price': entry_price - (atr * mult),
                    'pct_gain': (atr * mult / entry_price) * 100,
                    'method': f'{mult}x ATR'
                }

        return targets

    def find_swing_points(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        """Find recent swing high and swing low."""
        recent = df.tail(lookback)
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        swing_high_date = recent['high'].idxmax()
        swing_low_date = recent['low'].idxmin()

        return {
            'swing_high': swing_high,
            'swing_low': swing_low,
            'swing_high_date': swing_high_date,
            'swing_low_date': swing_low_date,
            'swing_range': swing_high - swing_low
        }

    def calculate_fibonacci_targets(self, entry_price: float, swing_high: float,
                                   swing_low: float, direction: str = 'long') -> Dict:
        """
        Calculate Fibonacci extension targets.

        Uses standard Fibonacci ratios: 1.0, 1.272, 1.618, 2.0, 2.618
        """
        swing_range = swing_high - swing_low
        fib_levels = [1.0, 1.272, 1.618, 2.0, 2.618]

        targets = {}
        for i, level in enumerate(fib_levels):
            level_name = f'fib_{level}'
            if direction == 'long':
                # For long: extend above swing high
                target_price = swing_low + (swing_range * level)
            else:
                # For short: extend below swing low
                target_price = swing_high - (swing_range * level)

            targets[level_name] = {
                'price': target_price,
                'pct_gain': ((target_price - entry_price) / entry_price) * 100 if direction == 'long'
                           else ((entry_price - target_price) / entry_price) * 100,
                'method': f'Fib {level}'
            }

        return targets

    def calculate_support_resistance(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
        """
        Find key support and resistance levels using price action.

        Uses:
        - Recent highs/lows
        - Volume-weighted price levels
        - Round numbers
        """
        recent = df.tail(lookback)
        current_price = df['close'].iloc[-1]

        # Find local maxima and minima
        highs = recent['high'].values
        lows = recent['low'].values

        # Resistance levels (prices above current)
        resistance_levels = []
        for h in sorted(set(highs), reverse=True):
            if h > current_price:
                resistance_levels.append(h)
            if len(resistance_levels) >= 3:
                break

        # Support levels (prices below current)
        support_levels = []
        for l in sorted(set(lows)):
            if l < current_price:
                support_levels.append(l)
            if len(support_levels) >= 3:
                break
        support_levels = sorted(support_levels, reverse=True)

        # Round number levels
        magnitude = 10 ** (len(str(int(current_price))) - 2)
        round_above = ((current_price // magnitude) + 1) * magnitude
        round_below = (current_price // magnitude) * magnitude

        return {
            'resistance_1': resistance_levels[0] if len(resistance_levels) > 0 else None,
            'resistance_2': resistance_levels[1] if len(resistance_levels) > 1 else None,
            'resistance_3': resistance_levels[2] if len(resistance_levels) > 2 else None,
            'support_1': support_levels[0] if len(support_levels) > 0 else None,
            'support_2': support_levels[1] if len(support_levels) > 1 else None,
            'support_3': support_levels[2] if len(support_levels) > 2 else None,
            'round_above': round_above,
            'round_below': round_below
        }

    def get_options_targets(self, ticker: str) -> Dict:
        """
        Get price levels from options data (max pain, high OI strikes).

        Requires Polygon API access.
        """
        if self.polygon is None:
            return {'available': False}

        try:
            result = {
                'available': False,
                'pcr_volume': 1.0,
                'pcr_oi': 1.0,
                'sentiment': 'NEUTRAL',
                'max_pain': None,
                'high_call_oi_strike': None,
                'high_put_oi_strike': None
            }

            # 1. Get PCR and sentiment
            sentiment = self.polygon.get_options_sentiment(ticker)
            if sentiment.get('status') == 'ok':
                result['available'] = True
                result['pcr_volume'] = sentiment.get('pcr_volume', 1.0)
                result['pcr_oi'] = sentiment.get('pcr_oi', 1.0)
                result['sentiment'] = sentiment.get('sentiment', 'NEUTRAL')

            # 2. Get max pain
            max_pain_data = self.polygon.get_max_pain(ticker)
            if max_pain_data.get('available'):
                result['available'] = True
                result['max_pain'] = max_pain_data.get('max_pain')

            # 3. Get high OI strikes
            high_oi_data = self.polygon.get_high_oi_strikes(ticker)
            if high_oi_data.get('available'):
                result['available'] = True
                result['high_call_oi_strike'] = high_oi_data.get('highest_call_strike')
                result['high_put_oi_strike'] = high_oi_data.get('highest_put_strike')
                result['high_oi_calls'] = high_oi_data.get('high_oi_calls', [])
                result['high_oi_puts'] = high_oi_data.get('high_oi_puts', [])

            return result

        except Exception as e:
            return {'available': False, 'error': str(e)}

    def get_optimal_targets(self, df: pd.DataFrame, entry_price: float,
                           direction: str = 'long', ticker: str = None) -> Dict:
        """
        Combine all methods to suggest optimal take-profit levels.

        Args:
            df: OHLCV DataFrame
            entry_price: Position entry price
            direction: 'long' or 'short'
            ticker: Optional ticker for options data

        Returns:
            dict with combined targets and recommendations
        """
        current_price = df['close'].iloc[-1]
        atr = self.calculate_atr(df)
        swings = self.find_swing_points(df)

        # Get all target types
        atr_targets = self.calculate_atr_targets(entry_price, atr, direction)
        fib_targets = self.calculate_fibonacci_targets(
            entry_price, swings['swing_high'], swings['swing_low'], direction
        )
        sr_levels = self.calculate_support_resistance(df)
        options_data = self.get_options_targets(ticker) if ticker else {'available': False}

        # Combine into recommended targets
        all_targets = []

        # Add ATR targets
        for level, data in atr_targets.items():
            all_targets.append({
                'level': level,
                'price': data['price'],
                'pct_gain': data['pct_gain'],
                'method': data['method'],
                'category': 'ATR'
            })

        # Add Fibonacci targets
        for level, data in fib_targets.items():
            if direction == 'long' and data['price'] > entry_price:
                all_targets.append({
                    'level': level,
                    'price': data['price'],
                    'pct_gain': data['pct_gain'],
                    'method': data['method'],
                    'category': 'Fibonacci'
                })
            elif direction == 'short' and data['price'] < entry_price:
                all_targets.append({
                    'level': level,
                    'price': data['price'],
                    'pct_gain': data['pct_gain'],
                    'method': data['method'],
                    'category': 'Fibonacci'
                })

        # Sort by price (ascending for long, descending for short)
        all_targets.sort(key=lambda x: x['price'], reverse=(direction == 'short'))

        # Select recommended targets
        recommendations = {
            'conservative': atr_targets.get('conservative'),
            'moderate': atr_targets.get('moderate'),
            'aggressive': atr_targets.get('aggressive')
        }

        return {
            'entry_price': entry_price,
            'current_price': current_price,
            'direction': direction,
            'atr': atr,
            'swing_high': swings['swing_high'],
            'swing_low': swings['swing_low'],
            'recommendations': recommendations,
            'all_targets': all_targets,
            'support_resistance': sr_levels,
            'options_data': options_data
        }


class ExitTimingPredictor:
    """
    Predicts optimal exit timing using oscillator momentum analysis.

    Features:
    - Oscillator velocity and acceleration
    - Distance from overbought/oversold
    - Momentum exhaustion indicators
    - Historical holding period analysis
    """

    def __init__(self):
        self.timing_model = None
        self.metrics = {}

    def calculate_momentum_features(self, df: pd.DataFrame, osc_column: str = 'osc_smooth') -> pd.DataFrame:
        """
        Calculate momentum-based features for exit timing.

        Args:
            df: DataFrame with oscillator data
            osc_column: Column name for oscillator values

        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=df.index)

        # Check if oscillator column exists
        if osc_column not in df.columns:
            # Calculate a simple oscillator if not provided
            if 'close' in df.columns:
                # Use RSI as default oscillator
                if TA_AVAILABLE:
                    df[osc_column] = (ta.rsi(df['close'], length=14) - 50) / 50  # Normalize to -1 to 1
                else:
                    delta = df['close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    df[osc_column] = (rsi - 50) / 50

        osc = df[osc_column]

        # --- Oscillator derivatives ---
        features['osc_value'] = osc
        features['velocity'] = osc.diff()
        features['acceleration'] = features['velocity'].diff()

        # Smoothed versions
        features['velocity_smooth'] = features['velocity'].rolling(3).mean()
        features['acceleration_smooth'] = features['acceleration'].rolling(3).mean()

        # --- Distance from extremes ---
        features['dist_from_zero'] = abs(osc)
        features['dist_from_overbought'] = 1.0 - osc  # Assuming osc normalized to -1 to 1
        features['dist_from_oversold'] = osc - (-1.0)

        # --- Zone indicators ---
        features['in_overbought'] = (osc > 0.6).astype(int)
        features['in_oversold'] = (osc < -0.6).astype(int)
        features['in_extreme'] = ((osc > 0.8) | (osc < -0.8)).astype(int)

        # --- Momentum exhaustion ---
        # Divergence between price momentum and oscillator
        if 'close' in df.columns:
            features['price_momentum'] = df['close'].pct_change(5)
            features['osc_momentum'] = osc.diff(5)
            features['divergence'] = features['price_momentum'] - features['osc_momentum']

        # --- Reversal probability indicators ---
        # Velocity sign change approaching
        features['vel_approaching_zero'] = abs(features['velocity_smooth']) < 0.01
        features['vel_sign'] = np.sign(features['velocity_smooth'])
        features['vel_sign_change'] = (features['vel_sign'] != features['vel_sign'].shift(1)).astype(int)

        # --- Time in zone ---
        features['bars_in_positive'] = (osc > 0).groupby((osc <= 0).cumsum()).cumsum()
        features['bars_in_negative'] = (osc < 0).groupby((osc >= 0).cumsum()).cumsum()

        # --- Peak/trough detection ---
        features['is_local_max'] = ((osc > osc.shift(1)) & (osc > osc.shift(-1))).astype(int)
        features['is_local_min'] = ((osc < osc.shift(1)) & (osc < osc.shift(-1))).astype(int)
        features['bars_since_peak'] = features['is_local_max'].groupby(features['is_local_max'].cumsum()).cumcount()
        features['bars_since_trough'] = features['is_local_min'].groupby(features['is_local_min'].cumsum()).cumcount()

        return features

    def calculate_exit_urgency(self, momentum_features: pd.DataFrame,
                               position: str = 'long') -> Dict:
        """
        Calculate exit urgency based on momentum indicators.

        Args:
            momentum_features: DataFrame from calculate_momentum_features
            position: 'long' or 'short'

        Returns:
            dict with urgency score and components
        """
        latest = momentum_features.iloc[-1]

        urgency_score = 0
        reasons = []

        if position == 'long':
            # For LONG positions, urgency increases when:
            # 1. Oscillator is high (overbought)
            if latest.get('osc_value', 0) > 0.6:
                urgency_score += 20
                reasons.append("Oscillator overbought")
            if latest.get('osc_value', 0) > 0.8:
                urgency_score += 15
                reasons.append("Oscillator extremely overbought")

            # 2. Velocity turning negative
            if latest.get('velocity_smooth', 0) < 0:
                urgency_score += 25
                reasons.append("Velocity negative (momentum fading)")

            # 3. Acceleration negative (momentum decelerating)
            if latest.get('acceleration_smooth', 0) < 0:
                urgency_score += 15
                reasons.append("Acceleration negative")

            # 4. Extended time in overbought
            bars_positive = latest.get('bars_in_positive', 0)
            if bars_positive > 10:
                urgency_score += 10
                reasons.append(f"Extended run ({int(bars_positive)} bars positive)")

            # 5. Near local peak
            bars_since_peak = latest.get('bars_since_peak', 100)
            if bars_since_peak < 3:
                urgency_score += 15
                reasons.append("Near recent peak")

        else:  # SHORT position
            # Inverse logic for shorts
            if latest.get('osc_value', 0) < -0.6:
                urgency_score += 20
                reasons.append("Oscillator oversold")
            if latest.get('osc_value', 0) < -0.8:
                urgency_score += 15
                reasons.append("Oscillator extremely oversold")
            if latest.get('velocity_smooth', 0) > 0:
                urgency_score += 25
                reasons.append("Velocity positive (momentum returning)")
            if latest.get('acceleration_smooth', 0) > 0:
                urgency_score += 15
                reasons.append("Acceleration positive")

        # Clamp to 0-100
        urgency_score = min(100, max(0, urgency_score))

        # Determine urgency level
        if urgency_score >= 70:
            urgency_level = 'HIGH'
            recommendation = 'EXIT NOW'
        elif urgency_score >= 40:
            urgency_level = 'MEDIUM'
            recommendation = 'PREPARE EXIT'
        else:
            urgency_level = 'LOW'
            recommendation = 'HOLD'

        return {
            'urgency_score': urgency_score,
            'urgency_level': urgency_level,
            'recommendation': recommendation,
            'reasons': reasons,
            'osc_value': latest.get('osc_value', 0),
            'velocity': latest.get('velocity_smooth', 0),
            'acceleration': latest.get('acceleration_smooth', 0)
        }

    def estimate_bars_to_exit(self, momentum_features: pd.DataFrame,
                              position: str = 'long') -> Dict:
        """
        Estimate bars until optimal exit based on momentum patterns.

        Uses historical pattern matching to estimate timing.
        """
        latest = momentum_features.iloc[-1]

        # Simple heuristic based on momentum state
        osc = latest.get('osc_value', 0)
        velocity = latest.get('velocity_smooth', 0)

        if position == 'long':
            if velocity > 0 and osc < 0.5:
                # Still accelerating upward, more room
                est_bars = 7
                confidence = 60
                note = "Momentum still building"
            elif velocity > 0 and osc >= 0.5:
                # Overbought but still moving up
                est_bars = 3
                confidence = 70
                note = "Approaching exhaustion"
            elif velocity <= 0 and osc > 0.3:
                # Momentum fading in overbought
                est_bars = 1
                confidence = 80
                note = "Momentum fading, exit soon"
            else:
                # Momentum already reversed
                est_bars = 0
                confidence = 85
                note = "Exit signal active"
        else:  # SHORT
            if velocity < 0 and osc > -0.5:
                est_bars = 7
                confidence = 60
                note = "Downward momentum building"
            elif velocity < 0 and osc <= -0.5:
                est_bars = 3
                confidence = 70
                note = "Approaching oversold exhaustion"
            elif velocity >= 0 and osc < -0.3:
                est_bars = 1
                confidence = 80
                note = "Momentum reversing, exit soon"
            else:
                est_bars = 0
                confidence = 85
                note = "Exit signal active"

        return {
            'estimated_bars': est_bars,
            'bars_range': (max(0, est_bars - 2), est_bars + 3),
            'confidence': confidence,
            'note': note
        }

    def predict_exit_timing(self, df: pd.DataFrame, position: str = 'long',
                           osc_column: str = 'osc_smooth') -> Dict:
        """
        Full exit timing prediction.

        Args:
            df: DataFrame with OHLCV and oscillator data
            position: Current position direction
            osc_column: Column name for oscillator

        Returns:
            dict with urgency, timing estimate, and detailed indicators
        """
        # Calculate momentum features
        momentum_features = self.calculate_momentum_features(df, osc_column)

        # Calculate exit urgency
        urgency = self.calculate_exit_urgency(momentum_features, position)

        # Estimate bars to exit
        timing = self.estimate_bars_to_exit(momentum_features, position)

        # Get latest indicator values
        latest = momentum_features.iloc[-1]

        return {
            'position': position,
            'urgency': urgency,
            'timing': timing,
            'indicators': {
                'oscillator': latest.get('osc_value', 0),
                'velocity': latest.get('velocity_smooth', 0),
                'acceleration': latest.get('acceleration_smooth', 0),
                'bars_in_zone': latest.get('bars_in_positive', 0) if position == 'long' else latest.get('bars_in_negative', 0),
                'momentum_exhaustion': latest.get('dist_from_overbought', 1) if position == 'long' else latest.get('dist_from_oversold', 1)
            },
            'recommendation': urgency['recommendation'],
            'confidence': timing['confidence']
        }


# Convenience function to create all predictors
def create_prediction_suite(polygon_api_key: str = None):
    """
    Create all three prediction components.

    Args:
        polygon_api_key: Optional API key for Polygon.io

    Returns:
        tuple of (PriceRangePredictor, PriceTargetCalculator, ExitTimingPredictor)
    """
    from polygon_manager import PolygonManager

    polygon = PolygonManager(polygon_api_key) if polygon_api_key else None

    range_predictor = PriceRangePredictor(polygon)
    target_calculator = PriceTargetCalculator(polygon)
    exit_predictor = ExitTimingPredictor()

    return range_predictor, target_calculator, exit_predictor
