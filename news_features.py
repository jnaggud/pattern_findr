"""
News-Based Features and Filters

This module provides:
1. Big Move Detector - Identify potential high-volatility days based on news
2. Lagged News Features - T-1, T-2, T-3 news features for prediction
3. Trade Filter - Filter trade entries based on news uncertainty

Usage:
    from news_features import BigMoveDetector, add_lagged_news_features, NewsTradeFilter

    # Big move detection
    detector = BigMoveDetector()
    alert = detector.check_big_move_risk('SPY')

    # Lagged features for prediction
    features = add_lagged_news_features(features_df, ticker='SPY')

    # Trade filtering
    filter = NewsTradeFilter()
    should_trade = filter.allow_entry('SPY', direction='long')
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Database path
NEWS_DB = os.path.join(os.path.dirname(__file__), 'news_sentiment_cache.db')


# =============================================================================
# 1. BIG MOVE DETECTOR
# =============================================================================

class BigMoveDetector:
    """
    Detect potential big move days based on news buzz and sentiment extremes.

    Big moves often occur when:
    - Buzz is unusually high (lots of news activity)
    - Sentiment is extreme (very bullish OR very bearish)
    - Sentiment changed dramatically from prior days

    Returns risk scores and alerts that can be used to:
    - Widen stop losses / take profits
    - Reduce position sizes
    - Use wider prediction intervals
    """

    def __init__(self, lookback_days: int = 60):
        """
        Args:
            lookback_days: Days of history to calculate percentiles
        """
        self.lookback_days = lookback_days

    def get_historical_stats(self, ticker: str) -> Dict:
        """Get historical buzz and sentiment statistics for percentile calculation."""
        if not os.path.exists(NEWS_DB):
            return {}

        try:
            conn = sqlite3.connect(NEWS_DB)

            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')

            df = pd.read_sql_query(
                """SELECT date, sentiment_score, sentiment_bullish, sentiment_bearish,
                          buzz_score, buzz_articles_week
                   FROM daily_sentiment
                   WHERE symbol = ? AND date >= ? AND date <= ?
                   ORDER BY date""",
                conn,
                params=(ticker, start_date, end_date)
            )
            conn.close()

            if df.empty or len(df) < 10:
                return {}

            return {
                'buzz_mean': df['buzz_score'].mean(),
                'buzz_std': df['buzz_score'].std(),
                'buzz_p90': df['buzz_score'].quantile(0.9),
                'sentiment_mean': df['sentiment_score'].mean(),
                'sentiment_std': df['sentiment_score'].std(),
                'sentiment_abs_p90': df['sentiment_score'].abs().quantile(0.9),
                'n_days': len(df)
            }

        except Exception as e:
            print(f"Error getting historical stats: {e}")
            return {}

    def get_today_sentiment(self, ticker: str) -> Dict:
        """Get today's (or most recent) sentiment data."""
        if not os.path.exists(NEWS_DB):
            return {}

        try:
            conn = sqlite3.connect(NEWS_DB)

            df = pd.read_sql_query(
                """SELECT date, sentiment_score, sentiment_bullish, sentiment_bearish,
                          buzz_score, buzz_articles_week
                   FROM daily_sentiment
                   WHERE symbol = ?
                   ORDER BY date DESC
                   LIMIT 3""",
                conn,
                params=(ticker,)
            )
            conn.close()

            if df.empty:
                return {}

            today = df.iloc[0].to_dict()

            # Calculate sentiment change if we have prior days
            if len(df) >= 2:
                today['sentiment_change_1d'] = today['sentiment_score'] - df.iloc[1]['sentiment_score']
            if len(df) >= 3:
                today['sentiment_change_2d'] = today['sentiment_score'] - df.iloc[2]['sentiment_score']

            return today

        except Exception as e:
            print(f"Error getting today's sentiment: {e}")
            return {}

    def check_big_move_risk(self, ticker: str) -> Dict:
        """
        Check if conditions suggest a potential big move day.

        Returns:
            Dict with:
            - risk_score: 0-100 (higher = more likely big move)
            - risk_level: 'low', 'medium', 'high', 'extreme'
            - signals: List of triggered conditions
            - recommendation: Suggested action
        """
        stats = self.get_historical_stats(ticker)
        today = self.get_today_sentiment(ticker)

        if not stats or not today:
            return {
                'risk_score': 0,
                'risk_level': 'unknown',
                'signals': ['Insufficient data'],
                'recommendation': 'Use normal parameters'
            }

        signals = []
        risk_score = 0

        # 1. High buzz (unusual news activity)
        buzz = today.get('buzz_score', 0)
        buzz_z = (buzz - stats['buzz_mean']) / (stats['buzz_std'] + 0.01)

        if buzz_z > 2.0:
            signals.append(f"🔥 Extreme buzz: {buzz:.1f} (z={buzz_z:.1f})")
            risk_score += 30
        elif buzz_z > 1.5:
            signals.append(f"📢 High buzz: {buzz:.1f} (z={buzz_z:.1f})")
            risk_score += 20
        elif buzz_z > 1.0:
            signals.append(f"📰 Elevated buzz: {buzz:.1f}")
            risk_score += 10

        # 2. Extreme sentiment (very bullish or bearish)
        sentiment = today.get('sentiment_score', 0)
        sentiment_z = abs(sentiment - stats['sentiment_mean']) / (stats['sentiment_std'] + 0.01)

        if sentiment_z > 2.0:
            direction = "bullish" if sentiment > 0 else "bearish"
            signals.append(f"🎯 Extreme {direction}: {sentiment:+.2f} (z={sentiment_z:.1f})")
            risk_score += 25
        elif sentiment_z > 1.5:
            direction = "bullish" if sentiment > 0 else "bearish"
            signals.append(f"📊 Strong {direction}: {sentiment:+.2f}")
            risk_score += 15

        # 3. Sentiment reversal (dramatic change from prior days)
        change_1d = today.get('sentiment_change_1d', 0)
        if abs(change_1d) > 0.3:
            signals.append(f"🔄 Sentiment flip: {change_1d:+.2f} in 1 day")
            risk_score += 20
        elif abs(change_1d) > 0.2:
            signals.append(f"↔️ Sentiment shift: {change_1d:+.2f}")
            risk_score += 10

        # 4. Bearish extreme (historically leads to bigger moves)
        if sentiment < -0.3:
            signals.append(f"⚠️ Bearish extreme often = volatility")
            risk_score += 15

        # 5. High buzz + extreme sentiment combo
        if buzz_z > 1.0 and sentiment_z > 1.0:
            signals.append(f"💥 Buzz + sentiment combo")
            risk_score += 15

        # Determine risk level
        if risk_score >= 70:
            risk_level = 'extreme'
            recommendation = 'Consider: wider stops, smaller size, skip entry'
        elif risk_score >= 50:
            risk_level = 'high'
            recommendation = 'Consider: wider stops, reduced position size'
        elif risk_score >= 30:
            risk_level = 'medium'
            recommendation = 'Normal trading with awareness'
        else:
            risk_level = 'low'
            recommendation = 'Normal trading conditions'

        return {
            'risk_score': min(risk_score, 100),
            'risk_level': risk_level,
            'signals': signals if signals else ['No unusual activity'],
            'recommendation': recommendation,
            'buzz': buzz,
            'sentiment': sentiment,
            'buzz_z': buzz_z,
            'sentiment_z': sentiment_z
        }

    def get_big_move_probability(self, ticker: str) -> float:
        """
        Estimate probability of a big move (>2% daily range).

        Returns: 0.0-1.0 probability estimate
        """
        result = self.check_big_move_risk(ticker)

        # Convert risk score to probability (rough heuristic)
        # Base probability ~15%, scales up with risk
        base_prob = 0.15
        risk_multiplier = 1.0 + (result['risk_score'] / 100) * 2.0

        return min(base_prob * risk_multiplier, 0.8)


# =============================================================================
# 2. LAGGED NEWS FEATURES
# =============================================================================

def add_lagged_news_features(features: pd.DataFrame, ticker: str,
                              max_lag: int = 3) -> pd.DataFrame:
    """
    Add lagged news features (t-1, t-2, t-3) for better prediction.

    News from prior days may predict today's volatility better than same-day news
    (since markets need time to react).

    Args:
        features: DataFrame with DatetimeIndex
        ticker: Stock ticker
        max_lag: Maximum lag days (default 3)

    Returns:
        Features DataFrame with added lagged news columns
    """
    if not os.path.exists(NEWS_DB):
        print("   [Lagged News] No database found")
        return features

    try:
        conn = sqlite3.connect(NEWS_DB)

        # Get date range
        start_date = features.index.min() - timedelta(days=max_lag + 5)
        end_date = features.index.max()

        # Load sentiment data
        df = pd.read_sql_query(
            """SELECT date, sentiment_score, sentiment_bullish, sentiment_bearish,
                      buzz_score, company_news_score
               FROM daily_sentiment
               WHERE symbol = ? AND date >= ? AND date <= ?
               ORDER BY date""",
            conn,
            params=(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        )
        conn.close()

        if df.empty:
            print("   [Lagged News] No sentiment data found")
            return features

        # Convert to datetime index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Create lagged features
        new_features = {}

        base_cols = ['sentiment_score', 'buzz_score', 'sentiment_bullish', 'sentiment_bearish']

        for lag in range(1, max_lag + 1):
            for col in base_cols:
                if col in df.columns:
                    lagged = df[col].shift(lag)
                    new_features[f'news_{col}_lag{lag}'] = lagged

        # Create sentiment change features
        if 'sentiment_score' in df.columns:
            for lag in range(1, max_lag):
                change = df['sentiment_score'].shift(lag) - df['sentiment_score'].shift(lag + 1)
                new_features[f'news_sentiment_change_lag{lag}'] = change

        # Create cumulative buzz (sum of last N days)
        if 'buzz_score' in df.columns:
            new_features['news_buzz_cumulative_3d'] = df['buzz_score'].rolling(3).sum()
            new_features['news_buzz_trend'] = df['buzz_score'] - df['buzz_score'].shift(3)

        # Merge with features
        if not new_features:
            return features

        lagged_df = pd.DataFrame(new_features, index=df.index)

        # Normalize indices to date only for matching
        lagged_df.index = pd.to_datetime(lagged_df.index).normalize()

        # Create date-only index for features
        features_dates = pd.to_datetime(features.index).normalize()

        # Match and assign
        matched = 0
        for col in lagged_df.columns:
            if col not in features.columns:
                features[col] = np.nan

        for i, feat_date in enumerate(features_dates):
            if feat_date in lagged_df.index:
                for col in lagged_df.columns:
                    features.iloc[i, features.columns.get_loc(col)] = lagged_df.loc[feat_date, col]
                matched += 1

        print(f"   [Lagged News] Added {len(lagged_df.columns)} lagged features for {matched}/{len(features)} days")

        return features

    except Exception as e:
        print(f"   [Lagged News] Error: {e}")
        return features


def get_lagged_news_correlations(ticker: str, target_col: str = 'next_range_pct') -> pd.DataFrame:
    """
    Calculate correlations between lagged news features and target.

    Use this to find optimal lag for news features.
    """
    from price_prediction import PriceRangePredictor
    from market_data_db import get_market_db

    db = get_market_db()
    df = db.get_data(ticker, '2023-01-01', '2025-12-31')

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    predictor = PriceRangePredictor()

    # Create basic features
    import sys, io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    features = predictor.create_range_features(df, ticker=ticker)
    sys.stdout = old_stdout

    # Add lagged news features
    features = add_lagged_news_features(features, ticker, max_lag=5)

    # Create targets
    targets = predictor.create_targets(df)

    # Calculate correlations
    news_cols = [c for c in features.columns if 'news_' in c.lower() and 'lag' in c.lower()]

    correlations = []
    for col in news_cols:
        valid = ~(features[col].isna() | targets[target_col].isna())
        if valid.sum() > 50:
            corr = np.corrcoef(features.loc[valid, col], targets.loc[valid, target_col])[0, 1]
            correlations.append({'feature': col, 'correlation': corr, 'n': valid.sum()})

    if not correlations:
        return pd.DataFrame(columns=['feature', 'correlation', 'n'])
    return pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)


# =============================================================================
# 3. NEWS TRADE FILTER
# =============================================================================

class NewsTradeFilter:
    """
    Filter trade entries based on news conditions.

    Use this to avoid entering trades during high-uncertainty periods
    or to adjust position sizing based on news risk.

    Usage:
        filter = NewsTradeFilter()

        # Check if trade should proceed
        result = filter.check_entry('SPY', direction='long')
        if result['allow_entry']:
            # Execute trade with result['position_size_multiplier']
        else:
            # Skip trade due to result['reason']
    """

    def __init__(self,
                 max_risk_score: int = 70,
                 min_sentiment_for_long: float = -0.3,
                 max_sentiment_for_short: float = 0.3,
                 max_buzz_z: float = 2.5):
        """
        Args:
            max_risk_score: Block entries above this risk score
            min_sentiment_for_long: Block longs if sentiment below this
            max_sentiment_for_short: Block shorts if sentiment above this
            max_buzz_z: Block all entries if buzz z-score above this
        """
        self.max_risk_score = max_risk_score
        self.min_sentiment_for_long = min_sentiment_for_long
        self.max_sentiment_for_short = max_sentiment_for_short
        self.max_buzz_z = max_buzz_z
        self.detector = BigMoveDetector()

    def check_entry(self, ticker: str, direction: str = 'long') -> Dict:
        """
        Check if a trade entry should be allowed.

        Args:
            ticker: Stock ticker
            direction: 'long' or 'short'

        Returns:
            Dict with:
            - allow_entry: bool
            - reason: str (why blocked, if applicable)
            - position_size_multiplier: 0.0-1.0
            - risk_details: Dict with full risk analysis
        """
        risk = self.detector.check_big_move_risk(ticker)

        result = {
            'allow_entry': True,
            'reason': None,
            'position_size_multiplier': 1.0,
            'risk_details': risk
        }

        # Check 1: Overall risk score
        if risk['risk_score'] >= self.max_risk_score:
            result['allow_entry'] = False
            result['reason'] = f"Risk score too high: {risk['risk_score']}"
            result['position_size_multiplier'] = 0.0
            return result

        # Check 2: Direction-specific sentiment
        sentiment = risk.get('sentiment', 0)

        if direction == 'long' and sentiment < self.min_sentiment_for_long:
            result['allow_entry'] = False
            result['reason'] = f"Sentiment too bearish for long: {sentiment:.2f}"
            result['position_size_multiplier'] = 0.0
            return result

        if direction == 'short' and sentiment > self.max_sentiment_for_short:
            result['allow_entry'] = False
            result['reason'] = f"Sentiment too bullish for short: {sentiment:.2f}"
            result['position_size_multiplier'] = 0.0
            return result

        # Check 3: Extreme buzz (uncertainty)
        buzz_z = risk.get('buzz_z', 0)
        if buzz_z > self.max_buzz_z:
            result['allow_entry'] = False
            result['reason'] = f"Buzz too extreme: z={buzz_z:.1f}"
            result['position_size_multiplier'] = 0.0
            return result

        # Adjust position size based on risk
        if risk['risk_score'] >= 50:
            result['position_size_multiplier'] = 0.5
            result['reason'] = "Reduced size due to elevated risk"
        elif risk['risk_score'] >= 30:
            result['position_size_multiplier'] = 0.75
            result['reason'] = "Slightly reduced size"

        return result

    def get_trade_recommendation(self, ticker: str) -> Dict:
        """
        Get comprehensive trade recommendation based on news.

        Returns entry allowance for both long and short directions.
        """
        long_check = self.check_entry(ticker, 'long')
        short_check = self.check_entry(ticker, 'short')

        risk = self.detector.check_big_move_risk(ticker)

        return {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'risk_level': risk['risk_level'],
            'risk_score': risk['risk_score'],
            'signals': risk['signals'],
            'long': {
                'allow': long_check['allow_entry'],
                'reason': long_check['reason'],
                'size_mult': long_check['position_size_multiplier']
            },
            'short': {
                'allow': short_check['allow_entry'],
                'reason': short_check['reason'],
                'size_mult': short_check['position_size_multiplier']
            },
            'recommendation': risk['recommendation']
        }


# =============================================================================
# INTEGRATION WITH PRICE PREDICTION
# =============================================================================

def integrate_news_features_with_predictor():
    """
    Patch PriceRangePredictor to include lagged news features.

    Call this once at startup to enable lagged news features in all predictions.
    """
    from price_prediction import PriceRangePredictor

    # Save original method
    original_create_features = PriceRangePredictor.create_range_features

    def enhanced_create_features(self, df, options_features=None, ticker=None):
        # Call original
        features = original_create_features(self, df, options_features, ticker)

        # Add lagged news if ticker provided
        if ticker:
            features = add_lagged_news_features(features, ticker, max_lag=3)

        return features

    # Patch
    PriceRangePredictor.create_range_features = enhanced_create_features
    print("[News Integration] Lagged news features enabled in PriceRangePredictor")


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NEWS FEATURES TEST")
    print("=" * 70)

    # Test big move detector
    print("\n[1] Big Move Detector")
    print("-" * 40)
    detector = BigMoveDetector()

    for ticker in ['SPY', 'AAPL', 'NVDA']:
        result = detector.check_big_move_risk(ticker)
        print(f"\n{ticker}:")
        print(f"  Risk Score: {result['risk_score']}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Signals: {', '.join(result['signals'][:3])}")
        print(f"  Recommendation: {result['recommendation']}")

    # Test trade filter
    print("\n[2] Trade Filter")
    print("-" * 40)
    filter = NewsTradeFilter()

    for ticker in ['SPY', 'AAPL']:
        rec = filter.get_trade_recommendation(ticker)
        print(f"\n{ticker}:")
        print(f"  Risk: {rec['risk_level']} ({rec['risk_score']})")
        print(f"  Long: {'✅' if rec['long']['allow'] else '❌'} {rec['long']['reason'] or 'OK'}")
        print(f"  Short: {'✅' if rec['short']['allow'] else '❌'} {rec['short']['reason'] or 'OK'}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
