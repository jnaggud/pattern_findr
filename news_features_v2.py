"""
News Features V2 - Engineered Features with Actual Predictive Power

Based on correlation analysis:
- buzz_x_volatility: 0.289 avg correlation (BEST)
- sentiment_cumsum_5d: 0.228 avg correlation
- sentiment_cumsum_3d: 0.187 avg correlation

Key insights:
1. Raw sentiment has low correlation (~0.09)
2. Cumulative sentiment (multi-day) is more predictive
3. Buzz × Volatility interaction captures "news during volatile periods"
4. Negative sentiment correlation = high sentiment predicts SMALLER ranges (complacency)
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from typing import Dict, Optional
from datetime import datetime, timedelta


def load_news_data(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load news sentiment data from cache database."""
    news_db = os.path.join(os.path.dirname(__file__), 'news_sentiment_cache.db')

    if not os.path.exists(news_db):
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(news_db)

        query = f'''
            SELECT date, sentiment_score, buzz_score, sentiment_bullish, sentiment_bearish
            FROM daily_sentiment
            WHERE symbol = ?
            ORDER BY date
        '''

        df = pd.read_sql_query(query, conn, params=(ticker,))
        conn.close()

        if len(df) == 0:
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Filter by date range if provided
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df

    except Exception as e:
        print(f"[News V2] Error loading data: {e}")
        return pd.DataFrame()


def create_predictive_news_features(price_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Create engineered news features with proven predictive power.

    Args:
        price_df: DataFrame with OHLCV data (must have 'high', 'low', 'close' columns)
        ticker: Stock ticker symbol

    Returns:
        DataFrame with new features added
    """
    # Load news data
    news_df = load_news_data(ticker)

    if news_df.empty:
        print(f"[News V2] No news data for {ticker}")
        return price_df

    # Ensure price_df has datetime index
    if not isinstance(price_df.index, pd.DatetimeIndex):
        if 'date' in price_df.columns:
            price_df = price_df.copy()
            price_df['date'] = pd.to_datetime(price_df['date'])
            price_df.set_index('date', inplace=True)

    # Calculate daily volatility for interaction features
    price_df = price_df.copy()
    price_df['_daily_range_pct'] = (price_df['high'] - price_df['low']) / price_df['close'] * 100

    # Merge news with price data
    merged = price_df.join(news_df[['sentiment_score', 'buzz_score']], how='left')

    # Forward fill missing news (use last available)
    merged['sentiment_score'] = merged['sentiment_score'].fillna(method='ffill')
    merged['buzz_score'] = merged['buzz_score'].fillna(method='ffill')

    # Fill any remaining NaN with neutral values
    merged['sentiment_score'] = merged['sentiment_score'].fillna(0)
    merged['buzz_score'] = merged['buzz_score'].fillna(1)

    # ========================================
    # TOP TIER FEATURES (Correlation > 0.15)
    # ========================================

    # 1. Buzz × Volatility Interaction (0.289 correlation) - BEST
    merged['news_buzz_x_volatility'] = merged['buzz_score'] * merged['_daily_range_pct']

    # 2. Cumulative Sentiment 5d (0.228 correlation)
    merged['news_sentiment_cumsum_5d'] = merged['sentiment_score'].rolling(5, min_periods=1).sum()

    # 3. Cumulative Sentiment 3d (0.187 correlation)
    merged['news_sentiment_cumsum_3d'] = merged['sentiment_score'].rolling(3, min_periods=1).sum()

    # ========================================
    # MID TIER FEATURES (Correlation 0.08-0.15)
    # ========================================

    # 4. Lagged Sentiment (t-1) - 0.12 correlation
    merged['news_sentiment_lag1'] = merged['sentiment_score'].shift(1)

    # 5. Lagged Sentiment (t-3) - 0.10 correlation
    merged['news_sentiment_lag3'] = merged['sentiment_score'].shift(3)

    # 6. Sentiment Flip (sign change) - 0.07 correlation
    merged['news_sentiment_flip'] = (
        merged['sentiment_score'] * merged['sentiment_score'].shift(1) < 0
    ).astype(int)

    # 7. Cumulative Buzz 3d
    merged['news_buzz_cumsum_3d'] = merged['buzz_score'].rolling(3, min_periods=1).sum()

    # 8. Buzz Z-Score (unusual activity)
    buzz_mean = merged['buzz_score'].rolling(20, min_periods=5).mean()
    buzz_std = merged['buzz_score'].rolling(20, min_periods=5).std()
    merged['news_buzz_zscore'] = (merged['buzz_score'] - buzz_mean) / (buzz_std + 1e-6)

    # 9. Sentiment Extreme (abs > 0.5)
    merged['news_sentiment_extreme'] = (merged['sentiment_score'].abs() > 0.5).astype(int)

    # ========================================
    # COMPOSITE FEATURES
    # ========================================

    # 10. News Risk Score (combination of extremes + buzz)
    merged['news_risk_score'] = (
        merged['news_sentiment_extreme'] * 30 +
        merged['news_buzz_zscore'].clip(-2, 2) * 15 +
        merged['news_sentiment_flip'] * 20
    ).clip(0, 100)

    # 11. Sentiment Momentum (rate of change)
    merged['news_sentiment_momentum'] = merged['sentiment_score'].diff(3)

    # 12. Sentiment × Volatility Interaction
    merged['news_sentiment_x_volatility'] = merged['sentiment_score'] * merged['_daily_range_pct']

    # 13. Sentiment Trend (is sentiment trending up or down)
    merged['news_sentiment_trend_5d'] = merged['sentiment_score'].rolling(5).mean() - merged['sentiment_score'].rolling(10).mean()

    # 14. News Surprise (sentiment vs recent average)
    sent_mean_10d = merged['sentiment_score'].rolling(10, min_periods=3).mean()
    merged['news_sentiment_surprise'] = merged['sentiment_score'] - sent_mean_10d

    # ========================================
    # VALIDATED SENTIMENT × TECHNICAL INTERACTIONS
    # (From walk-forward testing Jan 2026)
    # Best for: LOW_VOL (+1.2% R²), RANGE_BOUND (+50% R²)
    # ========================================

    # 15. Sentiment Cumsum × Trend (best for LOW_VOL regime)
    # Captures cumulative sentiment direction aligned with price trend
    if 'close' in price_df.columns:
        sma_20 = price_df['close'].rolling(20, min_periods=5).mean()
        price_vs_sma20 = (price_df['close'] - sma_20) / sma_20 * 100
        sent_cumsum_5d = merged['sentiment_score'].rolling(5, min_periods=1).sum()
        merged['news_sent_cumsum_x_trend'] = sent_cumsum_5d * price_vs_sma20.values

    # 16. Sentiment × Range Momentum (best for RANGE_BOUND regime)
    # Captures sentiment direction aligned with recent range expansion/contraction
    range_momentum = merged['_daily_range_pct'].diff(3) if '_daily_range_pct' in merged.columns else (
        (price_df['high'] - price_df['low']) / price_df['close'] * 100
    ).diff(3)
    merged['news_sent_x_range_momentum'] = merged['sentiment_score'] * range_momentum

    # 17. Sentiment × Volatility 20d (general sentiment-vol interaction)
    # Uses rolling volatility instead of daily range
    vol_20d = price_df['close'].pct_change().rolling(20, min_periods=5).std() * np.sqrt(252) * 100
    merged['news_sent_x_volatility_20d'] = merged['sentiment_score'] * vol_20d.values

    # Clean up temp column
    if '_daily_range_pct' in merged.columns:
        merged.drop('_daily_range_pct', axis=1, inplace=True)

    # Count how many features were created
    news_cols = [c for c in merged.columns if c.startswith('news_')]
    merged_days = merged[news_cols[0]].notna().sum() if news_cols else 0

    print(f"[News V2] Added {len(news_cols)} predictive news features for {merged_days}/{len(merged)} days")

    return merged


def get_news_feature_correlations(price_df: pd.DataFrame, ticker: str) -> Dict[str, Dict[str, float]]:
    """
    Calculate correlations of news features with next-day high/low/range.

    Returns dict of {feature_name: {high_corr, low_corr, range_corr, avg_corr}}
    """
    # Add features
    df = create_predictive_news_features(price_df.copy(), ticker)

    # Calculate targets
    df['next_high_pct'] = (df['high'].shift(-1) - df['close']) / df['close'] * 100
    df['next_low_pct'] = (df['close'] - df['low'].shift(-1)) / df['close'] * 100
    df['next_range_pct'] = (df['high'].shift(-1) - df['low'].shift(-1)) / df['close'] * 100

    df = df.dropna()

    # Get news feature columns
    news_cols = [c for c in df.columns if c.startswith('news_')]

    results = {}
    for col in news_cols:
        high_corr = df[col].corr(df['next_high_pct'])
        low_corr = df[col].corr(df['next_low_pct'])
        range_corr = df[col].corr(df['next_range_pct'])
        avg_corr = (abs(high_corr) + abs(low_corr) + abs(range_corr)) / 3

        results[col] = {
            'high_corr': high_corr,
            'low_corr': low_corr,
            'range_corr': range_corr,
            'avg_corr': avg_corr
        }

    return results


def print_feature_report(price_df: pd.DataFrame, ticker: str):
    """Print a formatted report of news feature correlations."""
    correlations = get_news_feature_correlations(price_df, ticker)

    # Sort by average correlation
    sorted_features = sorted(correlations.items(), key=lambda x: x[1]['avg_corr'], reverse=True)

    print("\n" + "=" * 80)
    print("NEWS FEATURES V2 - CORRELATION REPORT")
    print("=" * 80)
    print(f"{'Feature':<35} {'High':>10} {'Low':>10} {'Range':>10} {'Avg|Corr|':>10}")
    print("-" * 80)

    for name, corrs in sorted_features:
        print(f"{name:<35} {corrs['high_corr']:>+10.4f} {corrs['low_corr']:>+10.4f} "
              f"{corrs['range_corr']:>+10.4f} {corrs['avg_corr']:>10.4f}")

    print("=" * 80)


if __name__ == "__main__":
    # Test the module
    from market_data_db import MarketDataDB
    from datetime import datetime, timedelta

    db = MarketDataDB()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=500)

    price_df = db.get_data('SPY', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if 'date' in price_df.columns:
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df.set_index('date', inplace=True)

    print_feature_report(price_df, 'SPY')
